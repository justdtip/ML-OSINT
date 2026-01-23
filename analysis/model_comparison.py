"""
Comparison of Cumulative vs Delta Unified Models

Tests the hypothesis that:
- Cumulative model correlations were artifacts of trending data
- Delta model should perform BETTER on reconstruction tasks

Metrics:
1. Cross-source reconstruction error (mask one source, predict from others)
2. Within-source reconstruction error
3. Latent space quality (clustering, separation)
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR,
    MODEL_COMPARISON_OUTPUT_DIR,
)

import torch
import torch.nn.functional as F

# Import both model types
from unified_interpolation_delta import (
    SOURCE_CONFIGS as DELTA_CONFIGS,
    UnifiedInterpolationModelDelta,
    extract_equipment_delta_features,
    MODEL_DIR
)
from unified_interpolation import (
    SOURCE_CONFIGS as CUMULATIVE_CONFIGS,
    UnifiedInterpolationModel
)
from interpolation_data_loaders import (
    SentinelDataLoader,
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader
)

ANALYSIS_DIR_LOCAL = Path(__file__).parent
RESULTS_DIR = MODEL_COMPARISON_OUTPUT_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models(device):
    """Load both cumulative and delta models."""
    models = {}

    # Delta model
    delta_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'
    if delta_path.exists():
        delta_model = UnifiedInterpolationModelDelta(
            source_configs=DELTA_CONFIGS,
            d_embed=64, nhead=4, num_fusion_layers=2
        )
        state = torch.load(delta_path, map_location='cpu', weights_only=False)
        delta_model.load_state_dict(state)
        delta_model.to(device)
        delta_model.eval()
        models['delta'] = delta_model
        print("Loaded delta model")

    # Cumulative model
    cumulative_path = MODEL_DIR / 'unified_interpolation_best.pt'
    if cumulative_path.exists():
        # Detect n_features from saved weights
        state = torch.load(cumulative_path, map_location='cpu', weights_only=False)

        # Update configs based on saved model dimensions
        from copy import deepcopy
        cumulative_configs = deepcopy(CUMULATIVE_CONFIGS)
        for name in cumulative_configs:
            encoder_key = f'encoders.{name}.feature_proj.0.weight'
            if encoder_key in state:
                n_features = state[encoder_key].shape[1]
                cumulative_configs[name].n_features = n_features

        cumulative_model = UnifiedInterpolationModel(
            source_configs=cumulative_configs,
            d_embed=64, nhead=4, num_fusion_layers=2
        )
        cumulative_model.load_state_dict(state)
        cumulative_model.to(device)
        cumulative_model.eval()
        models['cumulative'] = cumulative_model
        print("Loaded cumulative model")

    return models


def load_data(use_delta_equipment=True, exclude_sentinel=True, normalize=True):
    """Load all source data."""
    source_data = {}
    feature_names = {}

    loaders = {
        'sentinel': SentinelDataLoader,
        'deepstate': DeepStateDataLoader,
        'equipment': EquipmentDataLoader,
        'firms': FIRMSDataLoader,
        'ucdp': UCDPDataLoader
    }

    if exclude_sentinel:
        del loaders['sentinel']

    min_samples = float('inf')

    for name, loader_class in loaders.items():
        loader = loader_class().load().process()
        data = loader.processed_data

        if name == 'equipment' and use_delta_equipment:
            if hasattr(loader, 'feature_names'):
                data, feat_names = extract_equipment_delta_features(data, loader.feature_names)
                feature_names[name] = feat_names
            else:
                feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]
        else:
            if hasattr(loader, 'feature_names'):
                feature_names[name] = loader.feature_names
            else:
                feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]

        source_data[name] = torch.tensor(data, dtype=torch.float32)
        min_samples = min(min_samples, len(data))
        print(f"  {name}: {data.shape}")

    # Align to common length
    n_samples = int(min_samples)
    for name in source_data:
        source_data[name] = source_data[name][:n_samples]

    # Normalize
    if normalize:
        for name in source_data:
            mean = source_data[name].mean(dim=0, keepdim=True)
            std = source_data[name].std(dim=0, keepdim=True) + 1e-8
            source_data[name] = (source_data[name] - mean) / std

    print(f"  Aligned to {n_samples} samples (normalized: {normalize})")
    return source_data, feature_names, n_samples


def compute_reconstruction_error(model, data, device, source_names):
    """Compute reconstruction MSE for each source."""
    model.eval()
    errors = {}

    with torch.no_grad():
        # Move data to device
        features = {name: data[name].to(device) for name in source_names}

        # Full reconstruction (no masking)
        outputs = model(features, return_reconstructions=True)
        reconstructions = outputs['reconstructions']

        for name in source_names:
            if name in reconstructions:
                pred = reconstructions[name].cpu()
                target = data[name]
                mse = F.mse_loss(pred, target).item()
                # Correlation across all features
                pred_flat = pred.numpy().flatten()
                target_flat = target.numpy().flatten()
                corr = np.corrcoef(pred_flat, target_flat)[0, 1]
                errors[name] = {'mse': mse, 'correlation': corr}

    return errors


def compute_masked_reconstruction(model, data, device, source_names, mask_source):
    """Compute reconstruction when one source is masked."""
    model.eval()

    with torch.no_grad():
        features = {name: data[name].to(device) for name in source_names}

        # Create mask (zeros out the masked source)
        batch_size = data[mask_source].shape[0]
        mask = {
            name: torch.zeros(batch_size, device=device) if name == mask_source
            else torch.ones(batch_size, device=device)
            for name in source_names
        }

        outputs = model(features, mask=mask, return_reconstructions=True)
        reconstructions = outputs['reconstructions']

        if mask_source in reconstructions:
            pred = reconstructions[mask_source].cpu()
            target = data[mask_source]
            mse = F.mse_loss(pred, target).item()

            # Per-feature correlation
            pred_np = pred.numpy()
            target_np = target.numpy()
            corrs = []
            for i in range(pred_np.shape[1]):
                if np.std(pred_np[:, i]) > 1e-8 and np.std(target_np[:, i]) > 1e-8:
                    c = np.corrcoef(pred_np[:, i], target_np[:, i])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)

            return {
                'mse': mse,
                'mean_correlation': np.mean(corrs) if corrs else 0.0,
                'max_correlation': np.max(corrs) if corrs else 0.0,
                'n_positive': sum(1 for c in corrs if c > 0)
            }

    return None


def compare_latent_spaces(models, data, device):
    """Compare latent space properties between models."""
    results = {}

    for model_name, model in models.items():
        model.eval()

        with torch.no_grad():
            source_names = list(model.source_names)
            features = {name: data[name].to(device) for name in source_names if name in data}

            # Get embeddings
            embeddings = model.encode_sources(features)

            # Stack embeddings
            emb_list = [embeddings[name].cpu().numpy() for name in source_names if name in embeddings]
            all_emb = np.concatenate(emb_list, axis=0)

            # Compute properties
            # 1. Mean pairwise distance
            from scipy.spatial.distance import pdist
            distances = pdist(all_emb)

            # 2. Variance explained by first few PCs
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, all_emb.shape[1]))
            pca.fit(all_emb)

            # 3. Silhouette score (do sources form distinct clusters?)
            from sklearn.metrics import silhouette_score
            # Create labels for each source
            labels = []
            for i, name in enumerate(source_names):
                if name in embeddings:
                    labels.extend([i] * len(embeddings[name]))

            if len(set(labels)) > 1:
                sil_score = silhouette_score(all_emb, labels)
            else:
                sil_score = 0.0

            results[model_name] = {
                'mean_pairwise_distance': float(np.mean(distances)),
                'std_pairwise_distance': float(np.std(distances)),
                'variance_explained_5pc': float(sum(pca.explained_variance_ratio_[:5])),
                'silhouette_score': float(sil_score)
            }

    return results


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    models = load_models(device)

    if len(models) < 2:
        print("Need both cumulative and delta models for comparison")
        # Still run what we can

    results = {'timestamp': datetime.now().isoformat()}

    # For each model, load appropriate data and evaluate
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} model")
        print('='*60)

        use_delta = (model_name == 'delta')
        print(f"\nLoading data (delta equipment: {use_delta})...")
        data, feature_names, n_samples = load_data(
            use_delta_equipment=use_delta,
            exclude_sentinel=True,  # Sentinel has only 32 monthly samples
            normalize=True
        )

        # Filter to sources present in data
        source_names = [s for s in model.source_names if s in data]

        # 1. Full reconstruction error
        print("\n1. Full reconstruction error...")
        recon_errors = compute_reconstruction_error(model, data, device, source_names)
        print("   Source        MSE      Correlation")
        for name, metrics in recon_errors.items():
            print(f"   {name:12} {metrics['mse']:8.4f}  {metrics['correlation']:.4f}")

        # 2. Masked reconstruction (cross-source prediction)
        print("\n2. Cross-source reconstruction (masked)...")
        masked_results = {}
        print("   Masked Source   MSE      Mean Corr   Max Corr")
        for mask_source in source_names:
            if mask_source in data:
                result = compute_masked_reconstruction(model, data, device, source_names, mask_source)
                if result:
                    masked_results[mask_source] = result
                    print(f"   {mask_source:14} {result['mse']:8.4f}  {result['mean_correlation']:.4f}      {result['max_correlation']:.4f}")

        results[model_name] = {
            'full_reconstruction': recon_errors,
            'masked_reconstruction': masked_results,
            'n_samples': n_samples
        }

    # 3. Compare latent spaces
    if len(models) == 2:
        print("\n3. Latent space comparison...")

        # Need to use delta data for delta model, cumulative for cumulative
        latent_results = {}
        for model_name, model in models.items():
            use_delta = (model_name == 'delta')
            data, _, _ = load_data(use_delta_equipment=use_delta, exclude_sentinel=True, normalize=True)

            model.eval()
            source_names = list(model.source_names)
            with torch.no_grad():
                features = {name: data[name].to(device) for name in source_names if name in data}
                embeddings = model.encode_sources(features)

                # Stack all embeddings
                emb_list = [embeddings[name].cpu().numpy() for name in source_names if name in embeddings]
                all_emb = np.concatenate(emb_list, axis=0)

                # Create labels
                labels = []
                for i, name in enumerate(source_names):
                    if name in embeddings:
                        labels.extend([i] * len(embeddings[name]))

                # Metrics
                from scipy.spatial.distance import pdist
                from sklearn.decomposition import PCA
                from sklearn.metrics import silhouette_score

                distances = pdist(all_emb)
                pca = PCA(n_components=min(10, all_emb.shape[1]))
                pca.fit(all_emb)
                sil_score = silhouette_score(all_emb, labels) if len(set(labels)) > 1 else 0.0

                latent_results[model_name] = {
                    'mean_distance': float(np.mean(distances)),
                    'variance_5pc': float(sum(pca.explained_variance_ratio_[:5])),
                    'silhouette': float(sil_score)
                }

        print("\n   Model        Mean Dist   Var(5PC)   Silhouette")
        for model_name, metrics in latent_results.items():
            print(f"   {model_name:12} {metrics['mean_distance']:8.4f}   {metrics['variance_5pc']:.4f}     {metrics['silhouette']:.4f}")

        results['latent_comparison'] = latent_results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if 'cumulative' in results and 'delta' in results:
        print("\nFull Reconstruction MSE (lower is better):")
        print("   Source       Cumulative    Delta      Winner")
        for source in results['cumulative']['full_reconstruction'].keys():
            cum_mse = results['cumulative']['full_reconstruction'].get(source, {}).get('mse', float('inf'))
            delta_mse = results['delta']['full_reconstruction'].get(source, {}).get('mse', float('inf'))
            winner = "DELTA" if delta_mse < cum_mse else "CUMULATIVE"
            print(f"   {source:12} {cum_mse:10.4f}  {delta_mse:10.4f}  {winner}")

        print("\nMasked Reconstruction Mean Correlation (higher is better):")
        print("   Source       Cumulative    Delta      Winner")
        for source in results['cumulative']['masked_reconstruction'].keys():
            cum_corr = results['cumulative']['masked_reconstruction'].get(source, {}).get('mean_correlation', 0)
            delta_corr = results['delta']['masked_reconstruction'].get(source, {}).get('mean_correlation', 0)
            winner = "DELTA" if delta_corr > cum_corr else "CUMULATIVE"
            print(f"   {source:12} {cum_corr:10.4f}  {delta_corr:10.4f}  {winner}")

    # Save results
    output_path = RESULTS_DIR / 'model_comparison.json'

    # Convert to JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
