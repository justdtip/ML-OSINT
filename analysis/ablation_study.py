#!/usr/bin/env python3
"""
Ablation Study: Testing the contribution of *_per_day features

This script:
1. Trains the hybrid model WITH *_per_day features (fixed calendar-day normalization)
2. Trains the hybrid model WITHOUT *_per_day features (ablation)
3. Compares reconstruction and cross-source prediction performance

Expected result:
- If the previous hybrid performance was due to leakage, the ablated model should
  perform similarly to the delta model
- If *_per_day features genuinely help (using calendar days), the full hybrid
  should outperform the ablated version
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from copy import deepcopy

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.paths import MODEL_COMPARISON_OUTPUT_DIR

from unified_interpolation_hybrid import (
    SOURCE_CONFIGS,
    CrossSourceDatasetHybrid,
    UnifiedInterpolationModelHybrid,
    UnifiedTrainerHybrid,
    MODEL_DIR
)

REPORT_DIR = MODEL_COMPARISON_OUTPUT_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate_model(model, val_loader, device, source_names):
    """Evaluate reconstruction and cross-source prediction."""
    model.eval()
    results = {}

    with torch.no_grad():
        # Full reconstruction
        full_recon_mse = {name: [] for name in source_names}
        full_recon_corr = {name: [] for name in source_names}

        # Masked reconstruction (cross-source prediction)
        masked_recon_mse = {name: [] for name in source_names}
        masked_recon_corr = {name: [] for name in source_names}

        for batch in val_loader:
            features = {k: v.to(device) for k, v in batch.items()}
            batch_size = next(iter(features.values())).size(0)

            # Full reconstruction (no masking)
            outputs = model(features, return_reconstructions=True)
            for name in source_names:
                pred = outputs['reconstructions'][name].cpu().numpy()
                target = features[name].cpu().numpy()
                mse = np.mean((pred - target) ** 2)
                full_recon_mse[name].append(mse)

                # Mean correlation across features
                corrs = []
                for i in range(pred.shape[1]):
                    if np.std(pred[:, i]) > 1e-8 and np.std(target[:, i]) > 1e-8:
                        c = np.corrcoef(pred[:, i], target[:, i])[0, 1]
                        if not np.isnan(c):
                            corrs.append(c)
                if corrs:
                    full_recon_corr[name].append(np.mean(corrs))

            # Masked reconstruction for each source
            for masked_source in source_names:
                mask = {
                    name: torch.zeros(batch_size, device=device) if name == masked_source
                    else torch.ones(batch_size, device=device)
                    for name in source_names
                }

                outputs = model(features, mask=mask, return_reconstructions=True)
                pred = outputs['reconstructions'][masked_source].cpu().numpy()
                target = features[masked_source].cpu().numpy()

                mse = np.mean((pred - target) ** 2)
                masked_recon_mse[masked_source].append(mse)

                corrs = []
                for i in range(pred.shape[1]):
                    if np.std(pred[:, i]) > 1e-8 and np.std(target[:, i]) > 1e-8:
                        c = np.corrcoef(pred[:, i], target[:, i])[0, 1]
                        if not np.isnan(c):
                            corrs.append(c)
                if corrs:
                    masked_recon_corr[masked_source].append(np.mean(corrs))

    # Aggregate results
    results['full_reconstruction'] = {
        name: {
            'mse': float(np.mean(full_recon_mse[name])),
            'mean_corr': float(np.mean(full_recon_corr[name])) if full_recon_corr[name] else 0.0
        }
        for name in source_names
    }

    results['masked_reconstruction'] = {
        name: {
            'mse': float(np.mean(masked_recon_mse[name])),
            'mean_corr': float(np.mean(masked_recon_corr[name])) if masked_recon_corr[name] else 0.0
        }
        for name in source_names
    }

    return results


def train_and_evaluate(ablate_per_day: bool, epochs: int = 100, temporal_gap: int = 7):
    """Train a model variant and evaluate it."""
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training {'ABLATED (no *_per_day)' if ablate_per_day else 'FULL HYBRID'} model")
    print(f"{'='*60}")

    # Create fresh configs to avoid contamination
    configs = deepcopy(SOURCE_CONFIGS)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CrossSourceDatasetHybrid(
        configs,
        train=True,
        exclude_clown_units=True,
        temporal_gap=temporal_gap,
        ablate_per_day=ablate_per_day
    )

    val_dataset = CrossSourceDatasetHybrid(
        configs,
        train=False,
        exclude_clown_units=True,
        temporal_gap=temporal_gap,
        norm_stats=train_dataset.norm_stats,
        ablate_per_day=ablate_per_day
    )

    # Update configs with actual feature counts
    for name, config in configs.items():
        if name in train_dataset.feature_names:
            config.n_features = len(train_dataset.feature_names[name])

    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Create model
    print("\nCreating model...")
    model = UnifiedInterpolationModelHybrid(
        source_configs=configs,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # List equipment features
    print(f"\nEquipment features ({len(train_dataset.feature_names['equipment'])}):")
    for i, name in enumerate(train_dataset.feature_names['equipment']):
        print(f"  {i:2d}: {name}")

    # Train
    trainer = UnifiedTrainerHybrid(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device=device
    )

    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train(epochs=epochs, patience=20)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    # Evaluate
    print("\nEvaluating...")
    source_names = list(configs.keys())
    results = evaluate_model(model, val_loader, device, source_names)
    results['training'] = {
        'best_val_loss': trainer.best_val_loss,
        'epochs_trained': len(history['train_loss']),
        'total_params': total_params,
        'n_train_samples': len(train_dataset),
        'n_val_samples': len(val_dataset),
        'equipment_features': train_dataset.feature_names['equipment'],
        'temporal_gap': temporal_gap
    }

    return results, model


def main():
    print("=" * 70)
    print("ABLATION STUDY: *_per_day Feature Contribution")
    print("=" * 70)
    print("\nThis study tests whether the hybrid model's superior performance")
    print("comes from genuine cross-source signal or from temporal leakage.")
    print("\nFixes applied:")
    print("  1. Calendar-day normalization (not sample indices)")
    print("  2. Training-only normalization stats")
    print("  3. Temporal gap between train/val sets")

    # Train full hybrid model (with *_per_day features)
    full_results, _ = train_and_evaluate(ablate_per_day=False, epochs=100, temporal_gap=7)

    # Train ablated model (without *_per_day features)
    ablated_results, _ = train_and_evaluate(ablate_per_day=True, epochs=100, temporal_gap=7)

    # Compare results
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    print("\n--- Full Reconstruction (Lower MSE = Better) ---")
    print(f"{'Source':<15} {'Full Hybrid':>15} {'Ablated':>15} {'Delta':>10}")
    print("-" * 55)

    avg_full_mse = []
    avg_ablated_mse = []

    for source in full_results['full_reconstruction'].keys():
        full_mse = full_results['full_reconstruction'][source]['mse']
        ablated_mse = ablated_results['full_reconstruction'][source]['mse']
        avg_full_mse.append(full_mse)
        avg_ablated_mse.append(ablated_mse)
        delta = full_mse - ablated_mse
        print(f"{source:<15} {full_mse:>15.4f} {ablated_mse:>15.4f} {delta:>+10.4f}")

    print("-" * 55)
    print(f"{'AVERAGE':<15} {np.mean(avg_full_mse):>15.4f} {np.mean(avg_ablated_mse):>15.4f}")

    print("\n--- Cross-Source Prediction (Higher Correlation = Better) ---")
    print(f"{'Masked Source':<15} {'Full Hybrid':>15} {'Ablated':>15} {'Delta':>10}")
    print("-" * 55)

    avg_full_corr = []
    avg_ablated_corr = []

    for source in full_results['masked_reconstruction'].keys():
        full_corr = full_results['masked_reconstruction'][source]['mean_corr']
        ablated_corr = ablated_results['masked_reconstruction'][source]['mean_corr']
        avg_full_corr.append(full_corr)
        avg_ablated_corr.append(ablated_corr)
        delta = full_corr - ablated_corr
        print(f"{source:<15} {full_corr:>15.4f} {ablated_corr:>15.4f} {delta:>+10.4f}")

    print("-" * 55)
    print(f"{'AVERAGE':<15} {np.mean(avg_full_corr):>15.4f} {np.mean(avg_ablated_corr):>15.4f}")

    # Interpretation
    print("\n--- INTERPRETATION ---")
    full_avg_corr = np.mean(avg_full_corr)
    ablated_avg_corr = np.mean(avg_ablated_corr)
    delta_corr = full_avg_corr - ablated_avg_corr

    if delta_corr > 0.1:
        print(f"  *_per_day features contribute +{delta_corr:.3f} to cross-source correlation")
        print("  The calendar-day normalized features provide genuine predictive signal")
    elif delta_corr > 0:
        print(f"  *_per_day features provide marginal benefit (+{delta_corr:.3f})")
        print("  Consider whether the added complexity is worth it")
    else:
        print(f"  *_per_day features provide no benefit (delta={delta_corr:.3f})")
        print("  Previous performance was likely due to temporal leakage")
        print("  Recommend using delta-only model")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'full_hybrid': full_results,
        'ablated': ablated_results,
        'comparison': {
            'avg_full_mse': float(np.mean(avg_full_mse)),
            'avg_ablated_mse': float(np.mean(avg_ablated_mse)),
            'avg_full_corr': float(np.mean(avg_full_corr)),
            'avg_ablated_corr': float(np.mean(avg_ablated_corr)),
            'per_day_contribution': float(delta_corr)
        }
    }

    output_path = REPORT_DIR / 'ablation_study_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
