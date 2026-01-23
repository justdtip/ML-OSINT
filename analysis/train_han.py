#!/usr/bin/env python3
"""
Training Script for Hierarchical Attention Network

Usage:
    python analysis/train_han.py --epochs 100 --batch_size 4 --lr 1e-4

This trains the hierarchical multi-head attention network on real OSINT data.
"""

import argparse
import os
import sys
from pathlib import Path

# Enable MPS fallback for unsupported ops (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from hierarchical_attention_network import (
    HierarchicalAttentionNetwork,
    DOMAIN_CONFIGS,
    TOTAL_FEATURES
)
from conflict_data_loader import RealConflictDataset, create_data_loaders
from training_utils import (
    WarmupCosineScheduler,
    UncertaintyWeightedLoss,
    TimeSeriesAugmentation,
    GradientAccumulator
)
from feature_selection import filter_features, get_reduced_feature_names
from missing_data_imputation import impute_domain_data
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL,
)

# Backward compatibility aliases (now using centralized config)
# ANALYSIS_DIR and MODEL_DIR are imported from config.paths
MODEL_DIR.mkdir(exist_ok=True)


class HierarchicalAttentionTrainer:
    """Trainer for the Hierarchical Attention Network."""

    def __init__(
        self,
        model: HierarchicalAttentionNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        noise_std: float = 0.05,
        device: str = 'cpu',
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        accumulation_steps: int = 1,
        use_uncertainty_loss: bool = True,
        augmentation_prob: float = 0.5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.noise_std = noise_std  # Gaussian noise std for data augmentation
        self.accumulation_steps = accumulation_steps
        self.use_uncertainty_loss = use_uncertainty_loss

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Use WarmupCosineScheduler instead of plain CosineAnnealingLR
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            warmup_start_lr=lr * 0.01,  # Start warmup at 1% of target LR
            min_lr=1e-6
        )

        # Initialize GradientAccumulator for effective larger batch sizes
        self.grad_accumulator = GradientAccumulator(
            self.optimizer,
            accumulation_steps=accumulation_steps,
            max_grad_norm=1.0
        )

        # Initialize TimeSeriesAugmentation
        self.augmenter = TimeSeriesAugmentation(
            shift_range=1,  # Conservative shift for monthly data
            warp_magnitude=0.1,
            jitter_std=noise_std,
            feature_dropout_prob=0.1,
            augmentation_prob=augmentation_prob
        )

        # Loss weights for multi-task learning
        if use_uncertainty_loss:
            # Use UncertaintyWeightedLoss for learnable task weights
            self.uncertainty_loss = UncertaintyWeightedLoss(
                task_names=['forecast', 'regime'],
                init_log_var=0.0
            ).to(device)
            # Add uncertainty loss parameters to optimizer
            self.optimizer.add_param_group({
                'params': self.uncertainty_loss.parameters(),
                'lr': lr * 10  # Higher LR for task weights to adapt quickly
            })
        else:
            self.uncertainty_loss = None
            # Fixed loss weights for multi-task learning
            self.loss_weights = {
                'forecast': 1.0,
                'regime': 0.5,
                'anomaly': 0.3
            }

    def get_current_lr(self) -> float:
        """Get the current learning rate from the scheduler."""
        return self.scheduler.get_last_lr()[0]

    def get_task_weights(self) -> dict:
        """Get current task weights (from uncertainty loss or fixed)."""
        if self.uncertainty_loss is not None:
            return self.uncertainty_loss.get_task_weights()
        else:
            return self.loss_weights.copy()

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'forecast': 0, 'regime': 0, 'total': 0}
        n_batches = 0

        # Reset gradient accumulator at start of epoch
        self.grad_accumulator.zero_grad()

        for features, masks, targets in self.train_loader:
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            masks = {k: v.to(self.device) for k, v in masks.items()}

            # Apply TimeSeriesAugmentation
            for domain_name in features:
                # Convert to numpy for augmentation
                feat_np = features[domain_name].cpu().numpy()
                # Apply augmentation (training=True)
                augmented_np = self.augmenter(feat_np, training=True)
                features[domain_name] = torch.tensor(
                    augmented_np, dtype=torch.float32, device=self.device
                )

            # Forward pass
            outputs = self.model(features, masks, return_attention=False)

            # Calculate individual task losses
            task_losses = {}

            # Forecast loss (predict next timestep features)
            if 'forecast' in outputs:
                target_features = torch.cat([
                    targets['next_features'][d][:, 0, :].to(self.device)
                    for d in self.model.domain_names
                ], dim=-1)
                forecast_loss = F.mse_loss(outputs['forecast'][:, -1, :], target_features)
                task_losses['forecast'] = forecast_loss
                total_losses['forecast'] += forecast_loss.item()

            # Regime classification loss (with label smoothing for regularization)
            if 'regime_logits' in outputs and 'regime' in targets:
                regime_loss = F.cross_entropy(
                    outputs['regime_logits'][:, -1, :],
                    targets['regime'].to(self.device),
                    label_smoothing=0.1
                )
                task_losses['regime'] = regime_loss
                total_losses['regime'] += regime_loss.item()

            # Combine losses using uncertainty weighting or fixed weights
            if self.use_uncertainty_loss and self.uncertainty_loss is not None:
                loss = self.uncertainty_loss(task_losses)
            else:
                loss = 0
                for task_name, task_loss in task_losses.items():
                    weight = self.loss_weights.get(task_name, 1.0)
                    loss = loss + weight * task_loss

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            # Step with gradient accumulator (handles clipping and optimizer step)
            if self.grad_accumulator.step(loss):
                # Optimizer step was performed
                pass

            total_losses['total'] += loss.item()
            n_batches += 1

        # Ensure any remaining gradients are applied at end of epoch
        if not self.grad_accumulator.should_step():
            # Manually apply remaining gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def validate(self) -> dict:
        """Validate model."""
        self.model.eval()
        total_losses = {'forecast': 0, 'regime': 0, 'total': 0}
        correct_regime = 0
        total_regime = 0
        n_batches = 0

        with torch.no_grad():
            for features, masks, targets in self.val_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}

                outputs = self.model(features, masks)

                # Forecast loss
                if 'forecast' in outputs:
                    target_features = torch.cat([
                        targets['next_features'][d][:, 0, :].to(self.device)
                        for d in self.model.domain_names
                    ], dim=-1)
                    forecast_loss = F.mse_loss(outputs['forecast'][:, -1, :], target_features)
                    total_losses['forecast'] += forecast_loss.item()

                # Regime accuracy
                if 'regime_logits' in outputs and 'regime' in targets:
                    pred_regime = outputs['regime_logits'][:, -1, :].argmax(dim=-1)
                    true_regime = targets['regime'].to(self.device)
                    correct_regime += (pred_regime == true_regime).sum().item()
                    total_regime += true_regime.size(0)

                    regime_loss = F.cross_entropy(
                        outputs['regime_logits'][:, -1, :],
                        true_regime
                    )
                    total_losses['regime'] += regime_loss.item()

                n_batches += 1

        results = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        if total_regime > 0:
            results['regime_accuracy'] = correct_regime / total_regime
        return results

    def train(self, epochs: int = 100, patience: int = 50, verbose: bool = True) -> dict:
        """Full training loop with early stopping."""
        history = {
            'train_loss': [], 'val_loss': [],
            'train_forecast': [], 'val_forecast': [],
            'val_regime_acc': [],
            'learning_rate': [],
            'task_weights': []
        }
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        # Calculate and log effective batch size
        actual_batch_size = self.train_loader.batch_size
        effective_batch_size = self.grad_accumulator.get_effective_batch_size(actual_batch_size)
        print(f"\nEffective batch size: {effective_batch_size} "
              f"(actual: {actual_batch_size} x accumulation: {self.accumulation_steps})")

        print(f"\nTraining for up to {epochs} epochs (early stopping patience={patience})...")
        print("-" * 70)

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Get current learning rate and task weights
            current_lr = self.get_current_lr()
            task_weights = self.get_task_weights()

            # Record history
            history['train_loss'].append(train_metrics['total'])
            history['val_loss'].append(val_metrics.get('total', val_metrics['forecast']))
            history['train_forecast'].append(train_metrics['forecast'])
            history['val_forecast'].append(val_metrics['forecast'])
            history['learning_rate'].append(current_lr)
            history['task_weights'].append(task_weights.copy())
            if 'regime_accuracy' in val_metrics:
                history['val_regime_acc'].append(val_metrics['regime_accuracy'])

            # Save best model and check early stopping
            if val_metrics['forecast'] < best_val_loss:
                best_val_loss = val_metrics['forecast']
                best_epoch = epoch
                patience_counter = 0
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss
                }
                if self.uncertainty_loss is not None:
                    save_dict['uncertainty_loss_state_dict'] = self.uncertainty_loss.state_dict()
                torch.save(save_dict, HAN_BEST_MODEL)
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                regime_acc = val_metrics.get('regime_accuracy', 0)
                weight_str = ", ".join([f"{k}:{v:.2f}" for k, v in task_weights.items()])
                print(f"Epoch {epoch:3d}: "
                      f"train_loss={train_metrics['total']:.4f}, "
                      f"val_forecast={val_metrics['forecast']:.4f}, "
                      f"regime_acc={regime_acc:.2%}, "
                      f"lr={current_lr:.2e}, "
                      f"weights=[{weight_str}]"
                      f" {'*' if epoch == best_epoch else ''}")

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Save final model
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history
        }
        if self.uncertainty_loss is not None:
            save_dict['uncertainty_loss_state_dict'] = self.uncertainty_loss.state_dict()
        torch.save(save_dict, HAN_FINAL_MODEL)

        print("-" * 70)
        print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Models saved to {MODEL_DIR}")

        return history


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Attention Network')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=60, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=4, help='Sequence length (months)')
    parser.add_argument('--d_model', type=int, default=32, help='Model dimension')
    parser.add_argument('--dropout', type=float, default=0.35, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--noise_std', type=float, default=0.15, help='Noise std for data augmentation')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage (recommended for transformer operations on Apple Silicon)')

    # New parameters for utilities integration
    parser.add_argument('--temporal_gap', type=int, default=14,
                        help='Gap in days between splits to prevent data leakage')
    parser.add_argument('--resolution', type=str, default='monthly',
                        choices=['weekly', 'monthly'],
                        help='Temporal resolution for data aggregation')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs for learning rate scheduler')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps for effective larger batch size')
    parser.add_argument('--use_reduced_features', action='store_true', default=True,
                        help='Use reduced feature set to avoid redundancy')
    parser.add_argument('--no_reduced_features', action='store_false', dest='use_reduced_features',
                        help='Disable reduced features (use all features)')
    parser.add_argument('--use_uncertainty_loss', action='store_true', default=True,
                        help='Use uncertainty-weighted multi-task loss')
    parser.add_argument('--no_uncertainty_loss', action='store_false', dest='use_uncertainty_loss',
                        help='Disable uncertainty loss (use fixed task weights)')
    parser.add_argument('--augmentation_prob', type=float, default=0.5,
                        help='Probability of applying time series augmentation')

    args = parser.parse_args()

    print("=" * 70)
    print("HIERARCHICAL ATTENTION NETWORK TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Sequence length: {args.seq_len} months")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Device: {args.device}")
    print(f"\n  New parameters:")
    print(f"  Temporal gap: {args.temporal_gap} days")
    print(f"  Resolution: {args.resolution}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Use reduced features: {args.use_reduced_features}")
    print(f"  Use uncertainty loss: {args.use_uncertainty_loss}")
    print(f"  Augmentation prob: {args.augmentation_prob}")

    # Detect device
    if args.force_cpu:
        device = 'cpu'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"  Using device: {device}")

    # Log reduced feature information if enabled
    if args.use_reduced_features:
        print("\n  Reduced feature sets:")
        reduced_names = get_reduced_feature_names()
        for domain, names in reduced_names.items():
            print(f"    {domain}: {len(names)} features")

    # Create data loaders with new parameters
    print("\nLoading data...")
    train_loader, val_loader, test_loader, norm_stats = create_data_loaders(
        DOMAIN_CONFIGS,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        temporal_gap_days=args.temporal_gap,
        resolution=args.resolution
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Normalization stats computed from training data")

    # Create model
    print("\nCreating model...")
    model = HierarchicalAttentionNetwork(
        domain_configs=DOMAIN_CONFIGS,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_temporal_layers=args.num_layers,
        dropout=args.dropout
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Create trainer with new parameters
    trainer = HierarchicalAttentionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        noise_std=args.noise_std,
        device=device,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        use_uncertainty_loss=args.use_uncertainty_loss,
        augmentation_prob=args.augmentation_prob
    )

    # Train
    history = trainer.train(epochs=args.epochs, patience=args.patience, verbose=True)

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if history['val_regime_acc']:
        print(f"Final regime accuracy: {history['val_regime_acc'][-1]:.2%}")
    print(f"Final learning rate: {history['learning_rate'][-1]:.2e}")
    if history['task_weights']:
        final_weights = history['task_weights'][-1]
        print(f"Final task weights: {final_weights}")


if __name__ == "__main__":
    main()
