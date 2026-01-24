"""
Comprehensive Model Comparison Report

Analyzes and compares unified model variants:
1. Cumulative Model - Uses raw cumulative equipment losses
2. Delta Model - Uses only daily change (delta) equipment features

Generates detailed report on:
- Architecture and parameter counts
- Feature engineering differences
- Training performance
- Cross-source reconstruction ability
- Latent space properties
- Feature importance rankings
- Temporal lag relationships
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR, ANALYSIS_REPORTS_DIR,
)

import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Import model classes
from unified_interpolation import (
    SOURCE_CONFIGS as CUMULATIVE_CONFIGS,
    UnifiedInterpolationModel
)
from unified_interpolation_delta import (
    SOURCE_CONFIGS as DELTA_CONFIGS,
    UnifiedInterpolationModelDelta,
    extract_equipment_delta_features
)
from interpolation_data_loaders import (
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader
)

REPORT_DIR = ANALYSIS_REPORTS_DIR
REPORT_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_all_models(device):
    """Load all three model variants."""
    models = {}

    # Cumulative model
    cumulative_path = MODEL_DIR / 'unified_interpolation_best.pt'
    if cumulative_path.exists():
        state = torch.load(cumulative_path, map_location='cpu', weights_only=False)
        from copy import deepcopy
        configs = deepcopy(CUMULATIVE_CONFIGS)
        for name in configs:
            key = f'encoders.{name}.feature_proj.0.weight'
            if key in state:
                configs[name].n_features = state[key].shape[1]
        model = UnifiedInterpolationModel(configs, d_embed=64, nhead=4, num_fusion_layers=2)
        model.load_state_dict(state)
        model.to(device).eval()
        models['cumulative'] = {'model': model, 'configs': configs}
        print("Loaded cumulative model")

    # Delta model
    delta_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'
    if delta_path.exists():
        state = torch.load(delta_path, map_location='cpu', weights_only=False)
        from copy import deepcopy
        configs = deepcopy(DELTA_CONFIGS)
        for name in configs:
            key = f'encoders.{name}.feature_proj.0.weight'
            if key in state:
                configs[name].n_features = state[key].shape[1]
        model = UnifiedInterpolationModelDelta(configs, d_embed=64, nhead=4, num_fusion_layers=2)
        model.load_state_dict(state)
        model.to(device).eval()
        models['delta'] = {'model': model, 'configs': configs}
        print("Loaded delta model")

    return models


def load_data_for_model(model_type):
    """Load data appropriate for each model type."""
    source_data = {}
    feature_names = {}

    loaders = {
        'deepstate': DeepStateDataLoader,
        'equipment': EquipmentDataLoader,
        'firms': FIRMSDataLoader,
        'ucdp': UCDPDataLoader
    }

    min_samples = float('inf')

    for name, loader_class in loaders.items():
        loader = loader_class().load().process()
        data = loader.processed_data

        if name == 'equipment':
            if model_type == 'delta':
                if hasattr(loader, 'feature_names'):
                    data, feat_names = extract_equipment_delta_features(data, loader.feature_names)
                    feature_names[name] = feat_names
                else:
                    feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]
            else:  # cumulative
                if hasattr(loader, 'feature_names'):
                    feature_names[name] = loader.feature_names
                else:
                    feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]
        else:
            if hasattr(loader, 'feature_names'):
                feature_names[name] = loader.feature_names
            else:
                feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]

        source_data[name] = torch.tensor(data, dtype=torch.float32)
        min_samples = min(min_samples, len(data))

    # Align and normalize
    n_samples = int(min_samples)
    for name in source_data:
        source_data[name] = source_data[name][:n_samples]
        mean = source_data[name].mean(dim=0, keepdim=True)
        std = source_data[name].std(dim=0, keepdim=True) + 1e-8
        source_data[name] = (source_data[name] - mean) / std

    return source_data, feature_names, n_samples


def analyze_architecture(models):
    """Analyze architectural differences between models."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        configs = model_info['configs']

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Feature counts per source
        source_features = {}
        for name, cfg in configs.items():
            source_features[name] = cfg.n_features

        # Layer structure
        encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n)
        fusion_params = sum(p.numel() for n, p in model.named_parameters() if 'fusion' in n)
        decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n)

        results[model_name] = {
            'total_params': total_params,
            'source_features': source_features,
            'total_input_features': sum(source_features.values()),
            'encoder_params': encoder_params,
            'fusion_params': fusion_params,
            'decoder_params': decoder_params,
            'd_embed': model.d_embed,
            'n_sources': len(configs)
        }

    return results


def compute_reconstruction_metrics(models, device):
    """Compute reconstruction error and correlation for each model."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        data, feature_names, n_samples = load_data_for_model(model_name)
        source_names = list(data.keys())

        model.eval()
        with torch.no_grad():
            features = {name: data[name].to(device) for name in source_names}
            outputs = model(features, return_reconstructions=True)
            reconstructions = outputs['reconstructions']

            metrics = {}
            for name in source_names:
                if name in reconstructions:
                    pred = reconstructions[name].cpu().numpy()
                    target = data[name].numpy()

                    mse = np.mean((pred - target) ** 2)
                    mae = np.mean(np.abs(pred - target))

                    # Per-feature correlations
                    feature_corrs = []
                    for i in range(pred.shape[1]):
                        if np.std(pred[:, i]) > 1e-8 and np.std(target[:, i]) > 1e-8:
                            c, _ = pearsonr(pred[:, i], target[:, i])
                            if not np.isnan(c):
                                feature_corrs.append(c)

                    metrics[name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'mean_corr': float(np.mean(feature_corrs)) if feature_corrs else 0.0,
                        'max_corr': float(np.max(feature_corrs)) if feature_corrs else 0.0,
                        'min_corr': float(np.min(feature_corrs)) if feature_corrs else 0.0,
                        'n_features': pred.shape[1]
                    }

            results[model_name] = {
                'metrics': metrics,
                'n_samples': n_samples
            }

    return results


def compute_masked_reconstruction(models, device):
    """Compute cross-source prediction ability (mask one source, predict from others)."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        data, feature_names, n_samples = load_data_for_model(model_name)
        source_names = list(data.keys())

        model.eval()
        masked_results = {}

        for mask_source in source_names:
            with torch.no_grad():
                features = {name: data[name].to(device) for name in source_names}
                batch_size = data[mask_source].shape[0]

                mask = {
                    name: torch.zeros(batch_size, device=device) if name == mask_source
                    else torch.ones(batch_size, device=device)
                    for name in source_names
                }

                outputs = model(features, mask=mask, return_reconstructions=True)
                reconstructions = outputs['reconstructions']

                if mask_source in reconstructions:
                    pred = reconstructions[mask_source].cpu().numpy()
                    target = data[mask_source].numpy()

                    mse = np.mean((pred - target) ** 2)

                    feature_corrs = []
                    for i in range(pred.shape[1]):
                        if np.std(pred[:, i]) > 1e-8 and np.std(target[:, i]) > 1e-8:
                            c, _ = pearsonr(pred[:, i], target[:, i])
                            if not np.isnan(c):
                                feature_corrs.append(c)

                    masked_results[mask_source] = {
                        'mse': float(mse),
                        'mean_corr': float(np.mean(feature_corrs)) if feature_corrs else 0.0,
                        'max_corr': float(np.max(feature_corrs)) if feature_corrs else 0.0,
                        'n_positive': sum(1 for c in feature_corrs if c > 0)
                    }

        results[model_name] = masked_results

    return results


def analyze_latent_space(models, device):
    """Analyze latent space properties for each model."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        data, feature_names, n_samples = load_data_for_model(model_name)
        source_names = list(data.keys())

        model.eval()
        with torch.no_grad():
            features = {name: data[name].to(device) for name in source_names}
            embeddings = model.encode_sources(features)

            # Collect all embeddings
            all_embeddings = []
            labels = []
            for i, name in enumerate(source_names):
                if name in embeddings:
                    emb = embeddings[name].cpu().numpy()
                    all_embeddings.append(emb)
                    labels.extend([i] * len(emb))

            all_emb = np.concatenate(all_embeddings, axis=0)

            # Pairwise distances
            distances = pdist(all_emb)

            # PCA analysis
            pca = PCA(n_components=min(10, all_emb.shape[1]))
            pca.fit(all_emb)

            # Silhouette score (how well do sources cluster?)
            sil_score = silhouette_score(all_emb, labels) if len(set(labels)) > 1 else 0.0

            # Cross-source embedding correlations
            source_emb_means = {}
            for i, name in enumerate(source_names):
                if name in embeddings:
                    source_emb_means[name] = embeddings[name].cpu().numpy().mean(axis=0)

            cross_source_corrs = {}
            for i, name1 in enumerate(source_names):
                for name2 in source_names[i+1:]:
                    if name1 in source_emb_means and name2 in source_emb_means:
                        c, _ = pearsonr(source_emb_means[name1], source_emb_means[name2])
                        cross_source_corrs[f"{name1}_vs_{name2}"] = float(c)

            results[model_name] = {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'variance_explained_5pc': float(sum(pca.explained_variance_ratio_[:5])),
                'variance_explained_3pc': float(sum(pca.explained_variance_ratio_[:3])),
                'silhouette_score': float(sil_score),
                'cross_source_correlations': cross_source_corrs,
                'embedding_dim': all_emb.shape[1]
            }

    return results


def compute_feature_importance(models, device):
    """Compute feature importance via gradient-based attribution."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        data, feature_names, n_samples = load_data_for_model(model_name)
        source_names = list(data.keys())

        # Enable gradients for input
        model.eval()
        importance = {name: np.zeros(data[name].shape[1]) for name in source_names}

        # Sample a subset for efficiency
        sample_size = min(100, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)

        for idx in indices:
            features = {name: data[name][idx:idx+1].to(device).requires_grad_(True)
                       for name in source_names}

            outputs = model(features, return_reconstructions=True)

            # Use reconstruction loss as target
            loss = 0
            for name in source_names:
                if name in outputs['reconstructions']:
                    loss += F.mse_loss(outputs['reconstructions'][name], features[name])

            loss.backward()

            for name in source_names:
                if features[name].grad is not None:
                    importance[name] += np.abs(features[name].grad.cpu().numpy()[0])

        # Normalize
        for name in importance:
            importance[name] /= sample_size

        # Get top features per source
        model_importance = {}
        for name in source_names:
            imp = importance[name]
            feat_names = feature_names.get(name, [f"feat_{i}" for i in range(len(imp))])
            sorted_idx = np.argsort(imp)[::-1]
            top_features = [(int(i), feat_names[i] if i < len(feat_names) else f"feat_{i}",
                           float(imp[i])) for i in sorted_idx[:10]]
            model_importance[name] = top_features

        results[model_name] = model_importance

    return results


def compute_temporal_relationships(models, device, max_lag=30):
    """Compute cross-correlation at various lags between sources."""
    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        data, feature_names, n_samples = load_data_for_model(model_name)
        source_names = list(data.keys())

        model.eval()
        with torch.no_grad():
            features = {name: data[name].to(device) for name in source_names}
            embeddings = model.encode_sources(features)

            # Get mean embedding per timestep
            emb_timeseries = {}
            for name in source_names:
                if name in embeddings:
                    emb_timeseries[name] = embeddings[name].cpu().numpy().mean(axis=1)

        # Cross-correlations at different lags
        temporal_results = {}
        pairs = [
            ('equipment', 'firms'),
            ('equipment', 'deepstate'),
            ('equipment', 'ucdp'),
            ('firms', 'deepstate'),
            ('firms', 'ucdp')
        ]

        for src1, src2 in pairs:
            if src1 in emb_timeseries and src2 in emb_timeseries:
                ts1 = emb_timeseries[src1]
                ts2 = emb_timeseries[src2]

                correlations = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        c, _ = pearsonr(ts1[:lag], ts2[-lag:])
                    elif lag > 0:
                        c, _ = pearsonr(ts1[lag:], ts2[:-lag])
                    else:
                        c, _ = pearsonr(ts1, ts2)
                    correlations.append((lag, float(c) if not np.isnan(c) else 0.0))

                best_lag = max(correlations, key=lambda x: abs(x[1]))
                zero_lag = [c for l, c in correlations if l == 0][0]

                temporal_results[f"{src1}_vs_{src2}"] = {
                    'peak_lag': best_lag[0],
                    'peak_correlation': best_lag[1],
                    'zero_lag_correlation': zero_lag
                }

        results[model_name] = temporal_results

    return results


def generate_report(models, device):
    """Generate comprehensive comparison report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().isoformat()}")

    report = {
        'timestamp': datetime.now().isoformat(),
        'models_analyzed': list(models.keys())
    }

    # 1. Architecture Analysis
    print("\n" + "-"*80)
    print("1. ARCHITECTURE COMPARISON")
    print("-"*80)
    arch = analyze_architecture(models)
    report['architecture'] = arch

    print(f"\n{'Model':<12} {'Params':>10} {'Features':>10} {'d_embed':>8} {'Sources':>8}")
    print("-"*50)
    for name, info in arch.items():
        print(f"{name:<12} {info['total_params']:>10,} {info['total_input_features']:>10} "
              f"{info['d_embed']:>8} {info['n_sources']:>8}")

    print("\nFeature counts per source:")
    for name, info in arch.items():
        print(f"  {name}: {info['source_features']}")

    # 2. Reconstruction Performance
    print("\n" + "-"*80)
    print("2. RECONSTRUCTION PERFORMANCE")
    print("-"*80)
    recon = compute_reconstruction_metrics(models, device)
    report['reconstruction'] = recon

    print("\nFull Reconstruction (self-encoding):")
    print(f"{'Model':<12} {'Source':<12} {'MSE':>8} {'MAE':>8} {'Mean Corr':>10}")
    print("-"*55)
    for model_name, model_results in recon.items():
        for source, metrics in model_results['metrics'].items():
            print(f"{model_name:<12} {source:<12} {metrics['mse']:>8.4f} "
                  f"{metrics['mae']:>8.4f} {metrics['mean_corr']:>10.4f}")

    # 3. Cross-Source Prediction
    print("\n" + "-"*80)
    print("3. CROSS-SOURCE PREDICTION (Masked Reconstruction)")
    print("-"*80)
    masked = compute_masked_reconstruction(models, device)
    report['masked_reconstruction'] = masked

    print("\nPredicting masked source from other sources:")
    print(f"{'Model':<12} {'Masked':<12} {'MSE':>8} {'Mean Corr':>10} {'Max Corr':>10}")
    print("-"*55)
    for model_name, model_results in masked.items():
        for source, metrics in model_results.items():
            print(f"{model_name:<12} {source:<12} {metrics['mse']:>8.4f} "
                  f"{metrics['mean_corr']:>10.4f} {metrics['max_corr']:>10.4f}")

    # 4. Latent Space Analysis
    print("\n" + "-"*80)
    print("4. LATENT SPACE ANALYSIS")
    print("-"*80)
    latent = analyze_latent_space(models, device)
    report['latent_space'] = latent

    print(f"\n{'Model':<12} {'Mean Dist':>10} {'Var(5PC)':>10} {'Silhouette':>10}")
    print("-"*45)
    for model_name, metrics in latent.items():
        print(f"{model_name:<12} {metrics['mean_distance']:>10.4f} "
              f"{metrics['variance_explained_5pc']:>10.4f} {metrics['silhouette_score']:>10.4f}")

    print("\nCross-source embedding correlations:")
    for model_name, metrics in latent.items():
        print(f"  {model_name}:")
        for pair, corr in metrics['cross_source_correlations'].items():
            print(f"    {pair}: {corr:.4f}")

    # 5. Feature Importance
    print("\n" + "-"*80)
    print("5. FEATURE IMPORTANCE (Top 5 per source)")
    print("-"*80)
    importance = compute_feature_importance(models, device)
    report['feature_importance'] = importance

    for model_name, model_imp in importance.items():
        print(f"\n{model_name.upper()} Model:")
        for source, features in model_imp.items():
            print(f"  {source}:")
            for idx, name, imp in features[:5]:
                print(f"    {name}: {imp:.4f}")

    # 6. Temporal Relationships
    print("\n" + "-"*80)
    print("6. TEMPORAL LAG RELATIONSHIPS")
    print("-"*80)
    temporal = compute_temporal_relationships(models, device)
    report['temporal'] = temporal

    print("\nPeak cross-correlation lags (positive = first leads second):")
    for model_name, model_temporal in temporal.items():
        print(f"\n{model_name.upper()} Model:")
        for pair, metrics in model_temporal.items():
            print(f"  {pair}: lag={metrics['peak_lag']:+d}, r={metrics['peak_correlation']:.4f} "
                  f"(zero-lag: {metrics['zero_lag_correlation']:.4f})")

    # 7. Summary Comparison
    print("\n" + "-"*80)
    print("7. SUMMARY COMPARISON")
    print("-"*80)

    # Best model per metric
    print("\nBest model per metric:")

    # Reconstruction MSE (lower is better)
    avg_mse = {}
    for model_name, model_results in recon.items():
        avg_mse[model_name] = np.mean([m['mse'] for m in model_results['metrics'].values()])
    best_recon = min(avg_mse, key=avg_mse.get)
    print(f"  Reconstruction MSE: {best_recon} ({avg_mse[best_recon]:.4f})")

    # Cross-source prediction (higher correlation is better)
    avg_masked_corr = {}
    for model_name, model_results in masked.items():
        avg_masked_corr[model_name] = np.mean([m['mean_corr'] for m in model_results.values()])
    best_masked = max(avg_masked_corr, key=avg_masked_corr.get)
    print(f"  Cross-source prediction: {best_masked} ({avg_masked_corr[best_masked]:.4f})")

    # Latent space compactness (higher silhouette is better)
    best_latent = max(latent, key=lambda x: latent[x]['silhouette_score'])
    print(f"  Latent space clustering: {best_latent} ({latent[best_latent]['silhouette_score']:.4f})")

    report['summary'] = {
        'best_reconstruction': best_recon,
        'best_cross_source': best_masked,
        'best_latent_clustering': best_latent,
        'avg_mse': avg_mse,
        'avg_masked_corr': avg_masked_corr
    }

    # Save report
    report_path = REPORT_DIR / 'comprehensive_model_comparison.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nReport saved to: {report_path}")

    return report


def main():
    device = get_device()
    print(f"Using device: {device}")

    print("\nLoading models...")
    models = load_all_models(device)

    if len(models) < 3:
        print(f"Warning: Only {len(models)} models loaded. Expected 3.")

    report = generate_report(models, device)

    print("\n" + "="*80)
    print("REPORT COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
