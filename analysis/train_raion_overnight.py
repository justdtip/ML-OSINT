"""
Overnight training script for raion-level geographic learning.

Saves checkpoints every 10 steps and logs training progress.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader

from analysis.multi_resolution_data import (
    MultiResolutionConfig,
    MultiResolutionDataset,
    multi_resolution_collate_fn,
)
from analysis.multi_resolution_han import MultiResolutionHAN, SourceConfig
from analysis.geographic_source_encoder import SpatialSourceConfig


# Configuration
CHECKPOINT_DIR = Path("analysis/checkpoints/raion_training")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-4,  # Lower LR for stability
    'max_steps': 5000,      # ~5000 steps overnight
    'save_every': 10,       # Save checkpoint every 10 steps
    'log_every': 10,        # Log metrics every 10 steps
    'grad_clip': 1.0,
    'd_model': 64,
    'nhead': 4,
    'dropout': 0.1,
}


def clean_missing_values(features_dict):
    """Replace -999.0 (MISSING_VALUE) sentinel values with 0.0."""
    MISSING_VALUE = -999.0
    cleaned = {}
    for name, features in features_dict.items():
        features_clean = features.clone()
        features_clean = torch.where(
            features_clean == MISSING_VALUE,
            torch.zeros_like(features_clean),
            features_clean
        )
        cleaned[name] = features_clean
    return cleaned


def save_checkpoint(model, optimizer, step, metrics, path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['metrics']


def run_training():
    torch.manual_seed(42)
    np.random.seed(42)

    config = TRAINING_CONFIG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CHECKPOINT_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RAION-LEVEL GEOGRAPHIC LEARNING - OVERNIGHT TRAINING")
    print("=" * 70)
    print(f"\nRun directory: {run_dir}")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # =========================================================================
    # Dataset Setup
    # =========================================================================
    print("\n[1/3] Loading dataset...")

    data_config = MultiResolutionConfig(
        daily_sources=['geoconfirmed_raion', 'personnel'],
        monthly_sources=['sentinel'],
        start_date='2023-06-01',
        end_date='2024-01-31',
        daily_seq_len=30,
        monthly_seq_len=3,
    )

    train_dataset = MultiResolutionDataset(config=data_config, split='train')
    # Pass normalization stats from train to val to prevent data leakage
    val_dataset = MultiResolutionDataset(
        config=data_config,
        split='val',
        norm_stats=train_dataset.norm_stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=multi_resolution_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=multi_resolution_collate_fn,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Get feature dimensions from first batch
    sample_batch = next(iter(train_loader))
    geoconfirmed_features = sample_batch['daily_features']['geoconfirmed_raion'].shape[-1]
    personnel_features = sample_batch['daily_features']['personnel'].shape[-1]
    sentinel_features = sample_batch['monthly_features']['sentinel'].shape[-1]

    n_raions = sample_batch['raion_masks']['geoconfirmed_raion'].shape[-1]
    features_per_raion = geoconfirmed_features // n_raions

    print(f"  Geoconfirmed: {geoconfirmed_features} = {n_raions} raions × {features_per_raion} features")
    print(f"  Personnel: {personnel_features} features")
    print(f"  Sentinel: {sentinel_features} features")

    # =========================================================================
    # Model Setup
    # =========================================================================
    print("\n[2/3] Creating model...")

    daily_configs = {
        'geoconfirmed_raion': SourceConfig(
            name='geoconfirmed_raion',
            n_features=geoconfirmed_features,
            resolution='daily',
            description='Geoconfirmed raion-level features',
        ),
        'personnel': SourceConfig(
            name='personnel',
            n_features=personnel_features,
            resolution='daily',
            description='Personnel losses',
        ),
    }

    monthly_configs = {
        'sentinel': SourceConfig(
            name='sentinel',
            n_features=sentinel_features,
            resolution='monthly',
            description='Sentinel satellite data',
        ),
    }

    geoconfirmed_spatial_config = SpatialSourceConfig(
        name='geoconfirmed_raion',
        n_raions=n_raions,
        features_per_raion=features_per_raion,
        use_geographic_prior=False,
    )

    model = MultiResolutionHAN(
        daily_source_configs=daily_configs,
        monthly_source_configs=monthly_configs,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_daily_layers=1,
        num_monthly_layers=1,
        num_fusion_layers=1,
        dropout=config['dropout'],
        use_geographic_prior=True,
        custom_spatial_configs={'geoconfirmed_raion': geoconfirmed_spatial_config},
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    if hasattr(model.daily_fusion, 'geographic_encoders'):
        print(f"  Geographic encoders: {list(model.daily_fusion.geographic_encoders.keys())}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    print("\n[3/3] Starting training...")
    print("-" * 70)

    model.train()

    # Metrics tracking
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'grad_norms': [],
        'steps': [],
    }

    running_loss = 0.0
    running_grad_norm = 0.0
    best_val_loss = float('inf')

    step = 0
    epoch = 0

    # Create infinite data iterator
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    train_iter = infinite_loader(train_loader)

    try:
        while step < config['max_steps']:
            batch = next(train_iter)

            # Clean missing values
            daily_features_clean = clean_missing_values(batch['daily_features'])
            monthly_features_clean = clean_missing_values(batch['monthly_features'])

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                daily_features=daily_features_clean,
                daily_masks=batch['daily_masks'],
                monthly_features=monthly_features_clean,
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch['month_boundary_indices'],
                raion_masks=batch.get('raion_masks'),
            )

            # Loss: MSE on casualty prediction with dummy target
            # TODO: Replace with actual targets when available
            casualty_pred = outputs['casualty_pred']
            dummy_target = torch.randn_like(casualty_pred) * 0.5
            loss = nn.functional.mse_loss(casualty_pred, dummy_target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['grad_clip']
            )

            # Check for NaN
            if torch.isnan(loss) or torch.isnan(grad_norm):
                print(f"  WARNING: NaN detected at step {step}, skipping...")
                optimizer.zero_grad()
                continue

            optimizer.step()

            # Update running metrics
            running_loss += loss.item()
            running_grad_norm += grad_norm.item()
            step += 1

            # Log and checkpoint every N steps
            if step % config['log_every'] == 0:
                avg_loss = running_loss / config['log_every']
                avg_grad = running_grad_norm / config['log_every']

                # Validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_daily = clean_missing_values(val_batch['daily_features'])
                        val_monthly = clean_missing_values(val_batch['monthly_features'])

                        val_outputs = model(
                            daily_features=val_daily,
                            daily_masks=val_batch['daily_masks'],
                            monthly_features=val_monthly,
                            monthly_masks=val_batch['monthly_masks'],
                            month_boundaries=val_batch['month_boundary_indices'],
                            raion_masks=val_batch.get('raion_masks'),
                        )

                        val_pred = val_outputs['casualty_pred']
                        val_target = torch.randn_like(val_pred) * 0.5
                        val_loss = nn.functional.mse_loss(val_pred, val_target)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
                model.train()

                # Log metrics
                metrics_history['train_loss'].append(avg_loss)
                metrics_history['val_loss'].append(avg_val_loss)
                metrics_history['grad_norms'].append(avg_grad)
                metrics_history['steps'].append(step)

                # Print progress
                print(f"  Step {step:5d} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Grad: {avg_grad:.4f}")

                # Save checkpoint
                if step % config['save_every'] == 0:
                    checkpoint_path = run_dir / f"checkpoint_step_{step:05d}.pt"
                    save_checkpoint(model, optimizer, step, metrics_history, checkpoint_path)

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = run_dir / "best_model.pt"
                        save_checkpoint(model, optimizer, step, metrics_history, best_path)
                        print(f"  New best model! Val Loss: {avg_val_loss:.4f}")

                # Reset running metrics
                running_loss = 0.0
                running_grad_norm = 0.0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final checkpoint
    final_path = run_dir / "final_model.pt"
    save_checkpoint(model, optimizer, step, metrics_history, final_path)

    # Save metrics history
    metrics_path = run_dir / "metrics_history.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps: {step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to: {run_dir}")
    print(f"  Metrics saved to: {metrics_path}")

    return metrics_history


if __name__ == '__main__':
    run_training()
