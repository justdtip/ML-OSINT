#!/usr/bin/env python3
"""
Modular Training Experiment for Multi-Resolution HAN

This script trains the Multi-Resolution HAN model with different data source
configurations to compare performance and identify optimal source combinations.

Experimental Configurations:
- baseline: Current behavior (all sources, aggregated spatial)
- spatial_rich: All spatial features enabled (VIIRS/FIRMS/DeepState tiled)
- ablation_*: Individual source ablations for feature importance analysis

Usage:
    # Run single experiment
    python analysis/experiments/modular_training_experiment.py --config baseline

    # Run comparison across configurations
    python analysis/experiments/modular_training_experiment.py --compare baseline spatial_rich

    # Run all ablations
    python analysis/experiments/modular_training_experiment.py --run-ablations

Author: ML Engineering Team
Date: 2026-01-27
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enable MPS fallback for unsupported ops
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from config.paths import (
    PROJECT_ROOT, OUTPUT_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, ensure_dir,
)
from analysis.modular_data_config import (
    ModularDataConfig,
    get_data_source_config,
    list_presets,
    create_ablation_config,
)
from analysis.modular_multi_resolution_data import (
    ModularMultiResolutionDataset,
    create_modular_dataloaders,
)
from analysis.multi_resolution_data import (
    MultiResolutionConfig,
    multi_resolution_collate_fn,
)
from analysis.multi_resolution_han import (
    MultiResolutionHAN,
    create_multi_resolution_han,
    SourceConfig,
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a modular training experiment."""
    # Data configuration
    data_config_name: str = 'baseline'
    multi_res_config: Optional[MultiResolutionConfig] = None

    # Training parameters
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    patience: int = 20

    # Model parameters
    d_model: int = 128
    nhead: int = 8
    num_daily_layers: int = 3
    num_monthly_layers: int = 2
    num_fusion_layers: int = 2
    dropout: float = 0.1

    # Output settings
    output_dir: Optional[Path] = None
    checkpoint_freq: int = 10
    seed: int = 42

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = OUTPUT_DIR / "experiments" / "modular_training"
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self) if hasattr(self, '__dataclass_fields__') else {}
        result['data_config_name'] = self.data_config_name
        result['batch_size'] = self.batch_size
        result['epochs'] = self.epochs
        result['learning_rate'] = self.learning_rate
        result['weight_decay'] = self.weight_decay
        result['warmup_epochs'] = self.warmup_epochs
        result['patience'] = self.patience
        result['d_model'] = self.d_model
        result['nhead'] = self.nhead
        result['num_daily_layers'] = self.num_daily_layers
        result['num_monthly_layers'] = self.num_monthly_layers
        result['num_fusion_layers'] = self.num_fusion_layers
        result['dropout'] = self.dropout
        result['output_dir'] = str(self.output_dir)
        result['checkpoint_freq'] = self.checkpoint_freq
        result['seed'] = self.seed
        return result


@dataclass
class ExperimentResults:
    """Results from a training experiment."""
    config_name: str
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    total_epochs: int = 0
    training_time_seconds: float = 0.0
    n_parameters: int = 0
    feature_info: Dict[str, Any] = field(default_factory=dict)

    # Per-task metrics (if available)
    task_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config_name': self.config_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': self.total_epochs,
            'training_time_seconds': self.training_time_seconds,
            'n_parameters': self.n_parameters,
            'feature_info': self.feature_info,
            'task_metrics': self.task_metrics,
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute combined loss for multi-task prediction.

    This is a simplified loss function. The full training script uses
    more sophisticated multi-task loss weighting.
    """
    total_loss = torch.tensor(0.0, device=device)
    n_tasks = 0

    # Forecast loss (MSE on predicted vs target features)
    if 'forecast_pred' in outputs and 'forecast_targets' in batch:
        forecast_pred = outputs['forecast_pred']  # [B, T, total_features]
        forecast_targets = batch['forecast_targets']
        forecast_masks = batch.get('forecast_masks', {})

        # Simplified approach: compute MSE between full forecast and concatenated targets
        # Stack all target sources into a single tensor
        target_list = []
        mask_list = []

        for source_name in sorted(forecast_targets.keys()):
            target = forecast_targets[source_name].to(device)  # [B, T, source_features]
            target_list.append(target)

            mask = forecast_masks.get(source_name)
            if mask is not None:
                mask_list.append(mask.to(device))

        if target_list:
            # Concatenate targets along feature dimension
            concat_target = torch.cat(target_list, dim=-1)  # [B, T, total_target_features]

            # Align sequence lengths
            min_t = min(forecast_pred.size(1), concat_target.size(1))
            pred_slice = forecast_pred[:, :min_t, :]
            target_slice = concat_target[:, :min_t, :]

            # Align feature dimensions (take min)
            min_f = min(pred_slice.size(-1), target_slice.size(-1))
            pred_slice = pred_slice[:, :, :min_f]
            target_slice = target_slice[:, :, :min_f]

            # Create combined mask if available
            if mask_list:
                concat_mask = torch.cat(mask_list, dim=-1)[:, :min_t, :min_f]
                # Compute masked MSE
                if concat_mask.any():
                    mse = nn.functional.mse_loss(
                        pred_slice[concat_mask],
                        target_slice[concat_mask],
                        reduction='mean'
                    )
                else:
                    mse = nn.functional.mse_loss(pred_slice, target_slice)
            else:
                mse = nn.functional.mse_loss(pred_slice, target_slice)

            total_loss = total_loss + mse
            n_tasks += 1

    # Casualty prediction loss (if available)
    if 'casualty_pred' in outputs:
        casualty_pred = outputs['casualty_pred']
        # In a full implementation, we'd have casualty targets
        # For now, use a placeholder regularization loss
        reg_loss = 0.01 * casualty_pred.pow(2).mean()
        total_loss = total_loss + reg_loss
        n_tasks += 1

    # Temporal encoded representation regularization
    if 'temporal_output' in outputs:
        temporal = outputs['temporal_output']
        # Encourage smooth temporal evolution
        if temporal.size(1) > 1:
            temporal_smoothness = (temporal[:, 1:] - temporal[:, :-1]).pow(2).mean()
            total_loss = total_loss + 0.001 * temporal_smoothness
            n_tasks += 1

    # Normalize by number of tasks
    if n_tasks > 0:
        total_loss = total_loss / n_tasks

    # Ensure we have a valid loss even if no tasks
    if n_tasks == 0:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move batch to device
        daily_features = {k: v.to(device) for k, v in batch['daily_features'].items()}
        daily_masks = {k: v.to(device) for k, v in batch['daily_masks'].items()}
        monthly_features = {k: v.to(device) for k, v in batch['monthly_features'].items()}
        monthly_masks = {k: v.to(device) for k, v in batch['monthly_masks'].items()}
        month_boundaries = batch['month_boundary_indices'].to(device)

        # Forward pass
        try:
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )
        except Exception as e:
            print(f"  Warning: Forward pass failed on batch {batch_idx}: {e}")
            continue

        # Compute loss
        loss = compute_loss(outputs, batch, device)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: Invalid loss on batch {batch_idx}, skipping")
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        # Move batch to device
        daily_features = {k: v.to(device) for k, v in batch['daily_features'].items()}
        daily_masks = {k: v.to(device) for k, v in batch['daily_masks'].items()}
        monthly_features = {k: v.to(device) for k, v in batch['monthly_features'].items()}
        monthly_masks = {k: v.to(device) for k, v in batch['monthly_masks'].items()}
        month_boundaries = batch['month_boundary_indices'].to(device)

        try:
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )
        except Exception as e:
            print(f"  Warning: Validation forward pass failed: {e}")
            continue

        loss = compute_loss(outputs, batch, device)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

class ModularTrainingExperiment:
    """
    Runner for modular training experiments.

    This class handles:
    - Data loading with modular configuration
    - Model creation adapted to data sources
    - Training loop with checkpointing
    - Results logging and comparison
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = get_device()
        self.results: Dict[str, ExperimentResults] = {}

        # Create output directories
        self.output_dir = ensure_dir(config.output_dir)
        self.checkpoint_dir = ensure_dir(config.output_dir / "checkpoints")
        self.logs_dir = ensure_dir(config.output_dir / "logs")

        print(f"Experiment output directory: {self.output_dir}")
        print(f"Device: {self.device}")

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def run_single_experiment(
        self,
        data_config_name: str,
        epochs: Optional[int] = None,
    ) -> ExperimentResults:
        """
        Run a single training experiment with specified data configuration.

        Args:
            data_config_name: Name of the data configuration preset
            epochs: Override number of epochs (optional)

        Returns:
            ExperimentResults with training history and metrics
        """
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {data_config_name}")
        print("=" * 70)

        epochs = epochs or self.config.epochs
        start_time = time.time()

        # Load data configuration
        try:
            data_config = get_data_source_config(data_config_name)
        except ValueError as e:
            print(f"Error: {e}")
            return ExperimentResults(config_name=data_config_name)

        print(f"\nData Configuration:")
        print(data_config)

        # Create data loaders
        print("\n--- Creating Data Loaders ---")
        try:
            train_loader, val_loader, test_loader, norm_stats, _ = create_modular_dataloaders(
                multi_res_config=self.config.multi_res_config,
                data_source_config=data_config,
                batch_size=self.config.batch_size,
                num_workers=0,
                seed=self.config.seed,
            )
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            import traceback
            traceback.print_exc()
            return ExperimentResults(config_name=data_config_name)

        # Get feature info from dataset
        feature_info = train_loader.dataset.get_feature_info()

        # Create model
        print("\n--- Creating Model ---")
        model = self._create_model(feature_info)
        model = model.to(self.device)

        n_params = count_parameters(model)
        print(f"Model created with {n_params:,} trainable parameters")

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.warmup_epochs,
            T_mult=2,
        )

        # Training loop
        print("\n--- Training ---")
        results = ExperimentResults(
            config_name=data_config_name,
            n_parameters=n_params,
            feature_info=feature_info,
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, self.device, epoch
            )
            results.train_losses.append(train_loss)

            # Validate
            val_loss = validate(model, val_loader, self.device)
            results.val_losses.append(val_loss)

            # Update scheduler
            scheduler.step()

            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                results.best_val_loss = val_loss
                results.best_epoch = epoch + 1
                patience_counter = 0

                # Save best checkpoint
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    data_config_name, is_best=True
                )
                print(f"  New best validation loss! Saved checkpoint.")
            else:
                patience_counter += 1

            # Periodic checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    data_config_name, is_best=False
                )

            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (patience={self.config.patience})")
                break

        results.total_epochs = epoch + 1
        results.training_time_seconds = time.time() - start_time

        print(f"\n--- Training Complete ---")
        print(f"Best Val Loss: {results.best_val_loss:.4f} at epoch {results.best_epoch}")
        print(f"Training Time: {results.training_time_seconds / 60:.1f} minutes")

        # Save results
        self._save_results(data_config_name, results, data_config)

        self.results[data_config_name] = results
        return results

    def _create_model(self, feature_info: Dict[str, Any]) -> MultiResolutionHAN:
        """Create model adapted to the loaded data sources."""
        # Build source configs from feature info
        daily_source_configs = {}
        monthly_source_configs = {}

        for source_name, info in feature_info.items():
            config = SourceConfig(
                name=source_name,
                n_features=info['n_features'],
                resolution=info['resolution'],
            )

            if info['resolution'] == 'daily':
                daily_source_configs[source_name] = config
            else:
                monthly_source_configs[source_name] = config

        print(f"  Daily sources: {list(daily_source_configs.keys())}")
        print(f"  Monthly sources: {list(monthly_source_configs.keys())}")

        # Create model
        model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_daily_layers=self.config.num_daily_layers,
            num_monthly_layers=self.config.num_monthly_layers,
            num_fusion_layers=self.config.num_fusion_layers,
            dropout=self.config.dropout,
        )

        return model

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
        val_loss: float,
        config_name: str,
        is_best: bool,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'config_name': config_name,
            'experiment_config': self.config.to_dict(),
        }

        if is_best:
            path = self.checkpoint_dir / f"{config_name}_best.pt"
        else:
            path = self.checkpoint_dir / f"{config_name}_epoch_{epoch + 1}.pt"

        torch.save(checkpoint, path)

    def _save_results(
        self,
        config_name: str,
        results: ExperimentResults,
        data_config: ModularDataConfig,
    ) -> None:
        """Save experiment results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'config_name': config_name,
            'data_config': data_config.to_dict(),
            'experiment_config': self.config.to_dict(),
            'results': results.to_dict(),
        }

        path = self.logs_dir / f"{config_name}_results.json"

        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to: {path}")

    def run_comparison(
        self,
        config_names: List[str],
        epochs: Optional[int] = None,
    ) -> Dict[str, ExperimentResults]:
        """
        Run experiments with multiple configurations and compare results.

        Args:
            config_names: List of data configuration preset names to compare
            epochs: Override number of epochs for all experiments

        Returns:
            Dictionary mapping config names to their results
        """
        print("\n" + "=" * 70)
        print("COMPARISON EXPERIMENT")
        print(f"Configurations: {config_names}")
        print("=" * 70)

        for config_name in config_names:
            self.run_single_experiment(config_name, epochs=epochs)

        # Generate comparison report
        self._generate_comparison_report(config_names)

        return self.results

    def run_ablations(self, epochs: Optional[int] = None) -> Dict[str, ExperimentResults]:
        """
        Run ablation experiments for all major data sources.

        This runs the baseline and then experiments with each major source disabled.

        Args:
            epochs: Override number of epochs

        Returns:
            Dictionary mapping config names to their results
        """
        ablation_configs = [
            'baseline',
            'ablation_no_viirs',
            'ablation_equipment_only',
            'ablation_daily_only',
            'ablation_monthly_only',
        ]

        return self.run_comparison(ablation_configs, epochs=epochs)

    def _generate_comparison_report(self, config_names: List[str]) -> None:
        """Generate a comparison report for the experiments."""
        print("\n" + "=" * 70)
        print("COMPARISON REPORT")
        print("=" * 70)

        # Table header
        print(f"\n{'Configuration':<25} {'Best Val Loss':<15} {'Best Epoch':<12} {'Params':<12} {'Time (min)':<10}")
        print("-" * 74)

        # Sort by best validation loss
        sorted_configs = sorted(
            config_names,
            key=lambda c: self.results.get(c, ExperimentResults(c)).best_val_loss
        )

        for config_name in sorted_configs:
            results = self.results.get(config_name)
            if results is None:
                print(f"{config_name:<25} {'FAILED':<15}")
                continue

            print(
                f"{config_name:<25} "
                f"{results.best_val_loss:<15.4f} "
                f"{results.best_epoch:<12d} "
                f"{results.n_parameters:<12,d} "
                f"{results.training_time_seconds / 60:<10.1f}"
            )

        # Save comparison report
        report_path = self.logs_dir / "comparison_report.json"
        report = {
            'timestamp': datetime.now().isoformat(),
            'configurations': config_names,
            'results': {
                name: self.results[name].to_dict()
                for name in config_names if name in self.results
            },
            'ranking': sorted_configs,
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\nComparison report saved to: {report_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modular Training Experiment for Multi-Resolution HAN"
    )

    # Configuration
    parser.add_argument(
        '--config', type=str, default='baseline',
        help='Data configuration preset name (default: baseline)'
    )
    parser.add_argument(
        '--compare', type=str, nargs='+',
        help='Run comparison across multiple configurations'
    )
    parser.add_argument(
        '--run-ablations', action='store_true',
        help='Run ablation experiments for all major sources'
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    # Model parameters
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')

    # Other
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--list-presets', action='store_true', help='List available presets')

    args = parser.parse_args()

    # List presets and exit
    if args.list_presets:
        print("Available Data Source Presets:")
        print("-" * 40)
        for name, desc in list_presets().items():
            print(f"  {name}: {desc}")
        return

    # Create experiment config
    exp_config = ExperimentConfig(
        data_config_name=args.config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        d_model=args.d_model,
        nhead=args.nhead,
        seed=args.seed,
    )

    if args.output_dir:
        exp_config.output_dir = Path(args.output_dir)

    # Create and run experiment
    experiment = ModularTrainingExperiment(exp_config)

    if args.run_ablations:
        experiment.run_ablations(epochs=args.epochs)
    elif args.compare:
        experiment.run_comparison(args.compare, epochs=args.epochs)
    else:
        experiment.run_single_experiment(args.config, epochs=args.epochs)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {exp_config.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
