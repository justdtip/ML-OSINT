#!/usr/bin/env python3
"""
Training Pipeline for Multi-Resolution Hierarchical Attention Network

This module provides a complete training pipeline for the MultiResolutionHAN model,
handling multi-resolution time series data (daily + monthly) with:
- Uncertainty-weighted multi-task learning
- Proper normalization (shared stats from train set)
- Observation-rate-stratified metrics
- Gradient accumulation for larger effective batch sizes
- Warmup cosine learning rate schedule
- Comprehensive checkpointing and logging

Usage:
    python analysis/train_multi_resolution.py --batch_size 8 --epochs 200

Author: ML Engineering Team
Date: 2026-01-21
"""

import os
import sys
import argparse
import json
import math
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Enable MPS fallback for unsupported ops (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Suppress PyTorch UserWarning about mismatched key_padding_mask and attn_mask types
# This is a harmless warning from nn.MultiheadAttention when using mixed dtypes
warnings.filterwarnings(
    "ignore",
    message=".*Support for mismatched key_padding_mask and attn_mask.*",
    category=UserWarning,
)

# Suppress pin_memory warning on MPS (known limitation)
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*argument is set as true but not supported on MPS.*",
    category=UserWarning,
)

# Suppress extreme values warning (handled by model clipping)
warnings.filterwarnings(
    "ignore",
    message=".*Extreme values.*possibly incomplete MISSING_VALUE handling.*",
    category=UserWarning,
)

# Suppress autocast deprecation warning (still works, just new API syntax)
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)

# Suppress GradScaler deprecation warning
warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.GradScaler.*is deprecated.*",
    category=FutureWarning,
)

# Suppress var() degrees of freedom warning (happens with small batches)
warnings.filterwarnings(
    "ignore",
    message=".*var\\(\\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast

# Local imports
from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    MultiResolutionSample,
    multi_resolution_collate_fn,
    create_multi_resolution_dataloaders,
    MISSING_VALUE,
)

from training_targets import (
    TargetLoader,
    targets_to_tensors,
    compute_phase_loss,
    compute_transition_loss,
    compute_casualty_loss,
    compute_anomaly_loss,
    N_PHASES,
    PHASE_TO_INDEX,
    INDEX_TO_PHASE,
)

from multi_resolution_modules import (
    CrossResolutionFusion,
    FusionOutput,
    PredictionOutput,
    TaskType,
    MultiResolutionPredictor,
    create_month_boundaries,
)

# Import the main model from multi_resolution_han
from multi_resolution_han import (
    MultiResolutionHAN,
    create_multi_resolution_han,
    SourceConfig,
)

# Geographic prior imports
from analysis.geographic_source_encoder import SpatialSourceConfig
from analysis.loaders.raion_adapter import get_per_raion_mask, RAION_ADAPTER_REGISTRY
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL, ensure_dir,
)

# Training run management (for probe integration)
from training_output_manager import (
    TrainingRunManager,
    TrainingRunMetadata,
    TRAINING_RUNS_DIR,
    STAGE_NAMES,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Backward compatibility aliases (now using centralized config)
# ANALYSIS_DIR is imported from config.paths
# Use MULTI_RES_CHECKPOINT_DIR for multi-resolution specific checkpoints
CHECKPOINT_DIR = MULTI_RES_CHECKPOINT_DIR
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CHECKPOINT SELECTION UTILITIES
# =============================================================================
# Based on Probe 2.1.4 findings, cross-modal fusion quality varies significantly
# across training epochs:
#   - Epoch 10:  RSA = 0.77 (excellent fusion - domains well-aligned)
#   - Epoch 50:  RSA = 0.37 (moderate fusion - some alignment)
#   - Epoch 100: RSA = 0.34 (degraded fusion - overfit to task-specific patterns)
#
# For cross-modal fusion experiments (transfer learning, multi-domain analysis,
# interpretability studies), use epoch 10 checkpoint for best alignment.
# For pure prediction tasks, use best validation loss checkpoint.
# =============================================================================

# Default epoch for optimal cross-modal fusion (from RSA probe analysis)
OPTIMAL_FUSION_EPOCH = 10


def load_checkpoint_by_epoch(
    epoch: int,
    checkpoint_dir: Optional[Path] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load a specific epoch checkpoint for fusion experiments.

    Based on Probe 2.1.4 (RSA analysis) findings, epoch 10 has the best
    cross-modal fusion quality (RSA = 0.77), while later epochs show degraded
    fusion due to overfitting to task-specific patterns:
      - Epoch 10:  RSA = 0.77 (excellent)
      - Epoch 50:  RSA = 0.37 (moderate)
      - Epoch 100: RSA = 0.34 (degraded)

    Use this function when:
      - Performing cross-modal fusion experiments
      - Transfer learning to new domains
      - Interpretability/attention analysis studies
      - Probing internal representations

    For pure prediction tasks, use load_best_checkpoint() instead.

    Args:
        epoch: Epoch number to load (e.g., 10 for optimal fusion)
        checkpoint_dir: Directory containing checkpoints. Defaults to
            MULTI_RES_CHECKPOINT_DIR from config.paths
        device: Device to load checkpoint to ('cpu', 'cuda', 'mps')

    Returns:
        Checkpoint dictionary containing:
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state
            - scheduler_state_dict: LR scheduler state
            - multi_task_loss_state_dict: Task weight state
            - epoch: Training epoch
            - best_val_loss: Best validation loss at time of save
            - train_history: Training metrics history
            - val_history: Validation metrics history
            - config: Training configuration
            - fusion_metrics (optional): RSA and alignment scores if saved

    Raises:
        FileNotFoundError: If checkpoint for specified epoch not found

    Example:
        >>> # Load optimal fusion checkpoint for interpretability analysis
        >>> checkpoint = load_checkpoint_by_epoch(10)
        >>> model.load_state_dict(checkpoint['model_state_dict'])
        >>> print(f"Loaded epoch {checkpoint['epoch']} checkpoint")
        >>> if 'fusion_metrics' in checkpoint:
        ...     print(f"RSA score: {checkpoint['fusion_metrics']['rsa']:.3f}")
    """
    if checkpoint_dir is None:
        checkpoint_dir = MULTI_RES_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)

    # Try standard epoch checkpoint naming convention
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'

    if not checkpoint_path.exists():
        # Try alternative naming patterns
        alt_patterns = [
            f'epoch_{epoch}_checkpoint.pt',
            f'checkpoint_{epoch}.pt',
            f'multi_res_epoch_{epoch}.pt',
        ]
        for pattern in alt_patterns:
            alt_path = checkpoint_dir / pattern
            if alt_path.exists():
                checkpoint_path = alt_path
                break

    if not checkpoint_path.exists():
        available = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        available_epochs = sorted([
            int(p.stem.split('_')[-1]) for p in available
            if p.stem.split('_')[-1].isdigit()
        ])
        raise FileNotFoundError(
            f"Checkpoint for epoch {epoch} not found at {checkpoint_path}. "
            f"Available epochs: {available_epochs or 'none'}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    return checkpoint


def load_best_checkpoint(
    checkpoint_dir: Optional[Path] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load the best validation loss checkpoint.

    Use this for prediction tasks where minimizing loss is the primary goal.
    For fusion/interpretability experiments, use load_checkpoint_by_epoch(10).

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to load checkpoint to

    Returns:
        Checkpoint dictionary (see load_checkpoint_by_epoch for contents)

    Raises:
        FileNotFoundError: If best checkpoint not found
    """
    if checkpoint_dir is None:
        checkpoint_dir = MULTI_RES_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / 'best_checkpoint.pt'

    if not best_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_path}")

    return torch.load(best_path, map_location=device, weights_only=False)


def load_fusion_checkpoint(
    checkpoint_dir: Optional[Path] = None,
    device: str = 'cpu',
    prefer_rsa_tracked: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint with best cross-modal fusion quality (highest RSA).

    This is a convenience wrapper that loads the optimal fusion checkpoint.
    Based on probe analysis, this is typically epoch 10 where RSA = 0.77.

    Priority order for checkpoint selection:
    1. best_rsa_checkpoint.pt - If RSA was tracked during training (prefer_rsa_tracked=True)
    2. early_fusion_checkpoint.pt - Saved at optimal fusion epoch (10)
    3. checkpoint_epoch_10.pt - Fallback to epoch 10 periodic checkpoint

    Args:
        checkpoint_dir: Directory containing checkpoints
        device: Device to load checkpoint to
        prefer_rsa_tracked: If True, prefer best_rsa_checkpoint.pt when available.
            Set False to always use epoch-10 based checkpoint.

    Returns:
        Checkpoint dictionary with best fusion quality

    Raises:
        FileNotFoundError: If no fusion or epoch 10 checkpoint found
    """
    if checkpoint_dir is None:
        checkpoint_dir = MULTI_RES_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)

    # Try best RSA checkpoint first (if RSA was tracked during training)
    if prefer_rsa_tracked:
        rsa_path = checkpoint_dir / 'best_rsa_checkpoint.pt'
        if rsa_path.exists():
            return torch.load(rsa_path, map_location=device, weights_only=False)

    # Try dedicated early fusion checkpoint (saved at epoch 10)
    fusion_path = checkpoint_dir / 'early_fusion_checkpoint.pt'
    if fusion_path.exists():
        return torch.load(fusion_path, map_location=device, weights_only=False)

    # Fall back to optimal fusion epoch
    return load_checkpoint_by_epoch(
        epoch=OPTIMAL_FUSION_EPOCH,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )


def list_available_checkpoints(checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    List all available checkpoints with their metadata.

    Args:
        checkpoint_dir: Directory to scan for checkpoints

    Returns:
        Dictionary with checkpoint information:
            - epochs: List of available epoch checkpoints
            - has_best: Whether best_checkpoint.pt exists (best val loss)
            - has_fusion: Whether early_fusion_checkpoint.pt exists (epoch 10)
            - has_best_rsa: Whether best_rsa_checkpoint.pt exists (tracked RSA)
            - has_latest: Whether latest_checkpoint.pt exists
            - recommended_fusion_epoch: Optimal epoch for fusion (10)
            - checkpoint_dir: Path to checkpoint directory
    """
    if checkpoint_dir is None:
        checkpoint_dir = MULTI_RES_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)

    # Find epoch checkpoints
    epoch_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    epochs = sorted([
        int(p.stem.split('_')[-1]) for p in epoch_files
        if p.stem.split('_')[-1].isdigit()
    ])

    return {
        'epochs': epochs,
        'has_best': (checkpoint_dir / 'best_checkpoint.pt').exists(),
        'has_fusion': (checkpoint_dir / 'early_fusion_checkpoint.pt').exists(),
        'has_best_rsa': (checkpoint_dir / 'best_rsa_checkpoint.pt').exists(),
        'has_latest': (checkpoint_dir / 'latest_checkpoint.pt').exists(),
        'recommended_fusion_epoch': OPTIMAL_FUSION_EPOCH,
        'checkpoint_dir': str(checkpoint_dir),
    }


# =============================================================================
# GEOGRAPHIC PRIOR UTILITIES
# =============================================================================

# All available raion sources for --all-raion-sources flag
ALL_RAION_SOURCES = [
    'geoconfirmed_raion',
    'air_raid_sirens_raion',
    'ucdp_raion',
    'warspotting_raion',
    'deepstate_raion',
    'firms_expanded_raion',
]


def build_spatial_configs_from_dataset(
    source_names: List[str],
) -> Dict[str, SpatialSourceConfig]:
    """
    Build SpatialSourceConfig for each raion source that has per-raion mask data.

    This function retrieves per-raion mask information from the raion adapter registry
    (populated when raion data is loaded) and creates SpatialSourceConfig objects
    for use with the GeographicSourceEncoder in MultiResolutionHAN.

    Args:
        source_names: List of raion source names to build configs for
            (e.g., ['geoconfirmed_raion', 'ucdp_raion'])

    Returns:
        Dict mapping source_name to SpatialSourceConfig for sources that have
        per-raion mask data available in the registry.

    Example:
        >>> spatial_configs = build_spatial_configs_from_dataset(
        ...     ['geoconfirmed_raion', 'ucdp_raion']
        ... )
        >>> for name, config in spatial_configs.items():
        ...     print(f"{name}: {config.n_raions} raions, {config.features_per_raion} features")
    """
    spatial_configs = {}

    for source_name in source_names:
        # Check if this source has per-raion mask info in the registry
        mask_info = get_per_raion_mask(source_name)

        if mask_info is None:
            print(f"    Warning: No per-raion mask info for {source_name} - skipping geographic prior")
            continue

        # Build SpatialSourceConfig from mask info
        config = SpatialSourceConfig(
            name=source_name,
            n_raions=mask_info.n_raions,
            features_per_raion=mask_info.n_features_per_raion,
            raion_keys=mask_info.raion_keys,
            use_geographic_prior=True,
        )

        spatial_configs[source_name] = config
        print(f"    Built spatial config for {source_name}: "
              f"{config.n_raions} raions, {config.features_per_raion} features/raion")

    return spatial_configs


@dataclass
class TrainerConfig:
    """Configuration for the MultiResolutionTrainer."""
    # Model configuration
    d_model: int = 128
    nhead: int = 8
    num_daily_layers: int = 3
    num_monthly_layers: int = 2
    num_fusion_layers: int = 2
    dropout: float = 0.1

    # Training configuration
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 200
    patience: int = 30
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Multi-task configuration
    task_names: List[str] = field(default_factory=lambda: [
        'casualty', 'regime', 'anomaly', 'forecast'
    ])

    # Data configuration
    daily_seq_len: int = 365
    monthly_seq_len: int = 12
    prediction_horizon: int = 1

    # Checkpointing
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    save_every: int = 10

    # Fusion checkpoint configuration
    # Based on Probe 2.1.4 findings, track RSA for early fusion checkpoint
    track_fusion_quality: bool = True
    fusion_checkpoint_epoch: int = OPTIMAL_FUSION_EPOCH

    # Device
    device: str = 'auto'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# =============================================================================
# TEMPORAL REGULARIZATION (Temporal Deconfounding)
# =============================================================================

class TemporalRegularizer(nn.Module):
    """
    Regularizer to prevent the model from learning temporal shortcuts.

    Implements two penalties:
    1. Correlation Penalty: Penalizes predictions that correlate with time position
    2. Smoothness Penalty: Penalizes overly smooth predictions (indicates position-based learning)

    Based on temporal-deconfounding-plan.md - addresses 71% spurious correlations.
    """

    def __init__(
        self,
        correlation_weight: float = 0.01,
        smoothness_weight: float = 0.001,
        target_roughness: float = 0.1,
    ):
        """
        Initialize temporal regularizer.

        Args:
            correlation_weight: Weight for correlation penalty (default: 0.01)
            smoothness_weight: Weight for smoothness penalty (default: 0.001)
            target_roughness: Target variance in prediction deltas (default: 0.1)
        """
        super().__init__()
        self.correlation_weight = correlation_weight
        self.smoothness_weight = smoothness_weight
        self.target_roughness = target_roughness

    def compute_correlation_penalty(
        self,
        predictions: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute penalty for predictions that correlate with temporal position.

        Args:
            predictions: Tensor of shape [batch, seq, features] or [batch, seq]
            seq_len: Sequence length
            device: Device for computation

        Returns:
            Scalar penalty tensor
        """
        # Create normalized position vector: [0, 1, 2, ...] -> mean=0, std=1
        positions = torch.arange(seq_len, dtype=predictions.dtype, device=device)
        pos_normalized = (positions - positions.mean()) / (positions.std() + 1e-8)

        # Flatten predictions to [batch, seq] if needed
        if predictions.dim() == 3:
            # Average across feature dimension
            pred_flat = predictions.mean(dim=-1)  # [batch, seq]
        else:
            pred_flat = predictions

        # Normalize predictions per-sample
        pred_mean = pred_flat.mean(dim=-1, keepdim=True)
        pred_std = pred_flat.std(dim=-1, keepdim=True) + 1e-8
        pred_normalized = (pred_flat - pred_mean) / pred_std

        # Compute correlation for each sample: (pred * pos).mean()
        # pos_normalized is [seq], pred_normalized is [batch, seq]
        correlations = (pred_normalized * pos_normalized.unsqueeze(0)).mean(dim=-1)  # [batch]

        # Only penalize positive correlations (predictions increasing with time)
        penalty = torch.relu(correlations).mean()

        return penalty

    def compute_smoothness_penalty(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize overly smooth predictions (indicates position-based learning).

        If predictions change too slowly over time, it suggests the model is
        using position rather than features.

        Args:
            predictions: Tensor of shape [batch, seq, features] or [batch, seq]

        Returns:
            Scalar penalty tensor
        """
        if predictions.dim() == 2:
            predictions = predictions.unsqueeze(-1)  # [batch, seq, 1]

        # Compute temporal deltas
        deltas = predictions[:, 1:, :] - predictions[:, :-1, :]  # [batch, seq-1, features]

        # Compute variance of deltas (roughness)
        roughness = deltas.pow(2).mean(dim=1)  # [batch, features]

        # Penalize if roughness is below target
        penalty = torch.relu(self.target_roughness - roughness).mean()

        return penalty

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total temporal regularization penalty.

        Args:
            outputs: Dict with model outputs (casualty_pred, regime_logits, etc.)
            seq_len: Sequence length
            device: Computation device

        Returns:
            Tuple of (total_penalty, penalty_breakdown_dict)
        """
        total_penalty = torch.tensor(0.0, device=device)
        penalty_breakdown = {}

        # Apply to casualty predictions
        if 'casualty_pred' in outputs:
            pred = outputs['casualty_pred']
            actual_seq_len = pred.shape[1]  # Get seq_len from tensor, not parameter
            corr_penalty = self.compute_correlation_penalty(
                pred, actual_seq_len, device
            )
            total_penalty = total_penalty + self.correlation_weight * corr_penalty
            penalty_breakdown['casualty_corr'] = corr_penalty.item()

            smooth_penalty = self.compute_smoothness_penalty(pred)
            total_penalty = total_penalty + self.smoothness_weight * smooth_penalty
            penalty_breakdown['casualty_smooth'] = smooth_penalty.item()

        # Apply to regime logits (use softmax to get probabilities)
        if 'regime_logits' in outputs:
            regime_probs = torch.softmax(outputs['regime_logits'], dim=-1)
            actual_seq_len = regime_probs.shape[1]  # Get seq_len from tensor
            corr_penalty = self.compute_correlation_penalty(
                regime_probs, actual_seq_len, device
            )
            total_penalty = total_penalty + self.correlation_weight * corr_penalty
            penalty_breakdown['regime_corr'] = corr_penalty.item()

        return total_penalty, penalty_breakdown


# =============================================================================
# MULTI-TASK LOSS WITH UNCERTAINTY WEIGHTING
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Learns task weights automatically using homoscedastic uncertainty.

    Based on Kendall et al. (2018) "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics".

    Loss = sum_i(0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i)

    This allows tasks to adaptively weight themselves based on uncertainty.
    Higher uncertainty (larger log_var) -> lower weight on that task's loss.

    IMPORTANT: log_var is clamped to [-2, 2] to prevent runaway negative losses.
    This keeps task weights in range [0.14, 7.4], which is reasonable.
    """

    # Default task weight priors based on probe analysis
    # Casualty head dominates at 62.6% of loss (Probe findings).
    # These priors rebalance to ~25% casualty, ~35% regime, ~25% anomaly, ~15% forecast.
    # exp(-log_var) gives the precision (weight), so:
    #   log_var=1.4 -> exp(-1.4) ≈ 0.25 weight
    #   log_var=1.05 -> exp(-1.05) ≈ 0.35 weight
    #   log_var=1.9 -> exp(-1.9) ≈ 0.15 weight
    DEFAULT_TASK_PRIORS = {
        'casualty': 1.4,     # exp(-1.4) ≈ 0.25 weight (reduced from dominant)
        'regime': 1.05,      # exp(-1.05) ≈ 0.35 weight (increased - best performing)
        'transition': 1.4,   # exp(-1.4) ≈ 0.25 weight (regime transition auxiliary)
        'anomaly': 1.4,      # exp(-1.4) ≈ 0.25 weight
        'forecast': 1.9,     # exp(-1.9) ≈ 0.15 weight (lowest - suspicious loss)
        'daily_forecast': 1.4,  # exp(-1.4) ≈ 0.25 weight (daily-resolution predictions)
        'isw_alignment': 2.3,  # exp(-2.3) ≈ 0.10 weight (auxiliary alignment task)
    }

    def __init__(
        self,
        task_names: List[str],
        init_log_var: Union[float, Dict[str, float]] = 0.0,
        log_var_min: float = -2.0,
        log_var_max: float = 2.0,
        log_var_reg: float = 0.01,
        use_task_priors: bool = False,
    ):
        """
        Initialize multi-task loss.

        Args:
            task_names: List of task identifiers
            init_log_var: Initial value for log variance parameters. Can be:
                - float: Same initial value for all tasks
                - dict: Per-task initial values (missing tasks use 0.0)
            log_var_min: Minimum value for log_var (prevents negative loss explosion)
            log_var_max: Maximum value for log_var (prevents vanishing gradients)
            log_var_reg: Regularization strength to keep log_var near 0
            use_task_priors: If True, use DEFAULT_TASK_PRIORS for initialization
                (overrides init_log_var). Based on probe analysis showing casualty
                head dominates at 62.6% of total loss.
        """
        super().__init__()

        if not task_names:
            raise ValueError("task_names must be a non-empty list")

        self.task_names = list(task_names)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.log_var_reg = log_var_reg

        # Determine initial values for each task
        if use_task_priors:
            init_values = {name: self.DEFAULT_TASK_PRIORS.get(name, 0.0) for name in self.task_names}
        elif isinstance(init_log_var, dict):
            init_values = {name: init_log_var.get(name, 0.0) for name in self.task_names}
        else:
            init_values = {name: float(init_log_var) for name in self.task_names}

        # Learnable log-variance for each task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_values[name], dtype=torch.float32))
            for name in self.task_names
        })

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted combined loss.

        Args:
            losses: Dictionary mapping task names to their unweighted losses
            masks: Optional per-task masks (for computing effective losses)

        Returns:
            total_loss: Combined weighted loss (always positive due to clamping)
            task_weights: Dictionary of current task weights for logging
        """
        device = self._get_device()
        total_loss = torch.tensor(0.0, device=device)
        task_weights = {}
        log_var_reg_loss = torch.tensor(0.0, device=device)

        n_valid_losses = 0
        for task_name in self.task_names:
            if task_name not in losses:
                continue

            task_loss = losses[task_name]

            # Skip if loss is nan
            if torch.isnan(task_loss):
                warnings.warn(f"Task {task_name} has NaN loss, skipping")
                continue

            # Skip if loss is exactly zero (no contribution)
            if task_loss.item() == 0:
                continue

            # Clamp log_var to reasonable range
            # [-2, 2] gives weights in [0.14, 7.4] - reasonable range
            log_var = self.log_vars[task_name].clamp(self.log_var_min, self.log_var_max)

            # Clamp task_loss to reasonable range
            if task_loss.item() > 1e6:
                warnings.warn(f"Task {task_name} loss exceeds 1e6, clamping")
                task_loss = task_loss.clamp(max=1e6)

            # Kendall et al. formulation:
            # L = 0.5 * exp(-log_var) * L_i + 0.5 * log_var
            # Note: This can produce negative total loss when log_var < 0, but that's fine -
            # gradient descent only cares about the direction of gradients, not the sign of loss.
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
            total_loss = total_loss + weighted_loss
            n_valid_losses += 1

            task_weights[task_name] = precision.item()

            # Regularization to keep log_var near 0 (prevents drift)
            log_var_reg_loss = log_var_reg_loss + self.log_vars[task_name].pow(2)

        # Add regularization on log_vars
        if n_valid_losses > 0:
            total_loss = total_loss + self.log_var_reg * log_var_reg_loss

        # If no valid losses, return a small loss connected to learnable params
        if n_valid_losses == 0:
            total_loss = sum(lv.pow(2) for lv in self.log_vars.values()) * 1e-6
            warnings.warn("No valid task losses, using log_var regularization")

        return total_loss, task_weights

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (precision = exp(-log_var))."""
        weights = {}
        with torch.no_grad():
            for task_name in self.task_names:
                log_var = self.log_vars[task_name].clamp(self.log_var_min, self.log_var_max)
                weights[task_name] = torch.exp(-log_var).item()
        return weights

    def get_uncertainties(self) -> Dict[str, float]:
        """Get current task uncertainties (variance = exp(log_var))."""
        uncertainties = {}
        with torch.no_grad():
            for task_name in self.task_names:
                log_var = self.log_vars[task_name]
                uncertainties[task_name] = torch.exp(log_var).item()
        return uncertainties

    def _get_device(self) -> torch.device:
        """Get device of the module parameters."""
        return next(self.parameters()).device


# =============================================================================
# ZERO-INFLATED NEGATIVE BINOMIAL (ZINB) LOSS
# =============================================================================

def zinb_nll_loss(
    y_true: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    max_loss: float = 20.0,
) -> torch.Tensor:
    """
    Compute Zero-Inflated Negative Binomial negative log-likelihood.

    ZINB models count data with:
    - Excess zeros (zero-inflation component)
    - Overdispersion (variance > mean)

    The distribution is a mixture:
        P(Y=0) = pi + (1-pi) * NB(0|mu, theta)
        P(Y=k) = (1-pi) * NB(k|mu, theta) for k > 0

    Args:
        y_true: Target counts [batch] - non-negative integers
        pi: Zero-inflation probability [batch] - in (0, 1)
        mu: Conditional mean [batch] - positive
        theta: Dispersion parameter [batch] - positive (smaller = more overdispersion)
        eps: Small constant for numerical stability
        max_loss: Maximum loss value to prevent outliers dominating training

    Returns:
        Scalar loss (mean NLL across batch)

    References:
        - DynAttn (arXiv:2512.21435) uses ZINB for conflict fatality forecasting
        - VIEWS project methodology
    """
    # Input validation
    if y_true.min() < 0:
        warnings.warn(f"ZINB received negative counts (min={y_true.min().item()}), clamping to 0")

    # Ensure numerical stability with proper clamping
    pi = pi.clamp(eps, 1 - eps)
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    # Negative Binomial log probability
    # NB(k|mu, theta) uses the "mu, theta" parameterization where:
    #   mean = mu
    #   variance = mu + mu^2/theta
    # theta is sometimes called "r" or "size" in other parameterizations

    # Log of NB(0|mu, theta) = theta * log(theta / (theta + mu))
    # Use log-difference form for stability: theta * (log(theta) - log(theta + mu))
    # Note: theta and mu are already clamped to >= eps, so log is safe
    log_nb_zero = theta * (torch.log(theta) - torch.log(theta + mu))

    # Separate zero and non-zero cases
    is_zero = (y_true == 0).float()
    is_nonzero = 1 - is_zero

    # Zero case: log P(Y=0) = log(pi + (1-pi) * NB(0))
    # Use proper log-sum-exp for numerical stability
    # Note: pi is clamped to (eps, 1-eps), so log is safe for both pi and 1-pi
    log_pi = torch.log(pi)
    log_one_minus_pi = torch.log(1 - pi)

    # log(pi + (1-pi)*exp(log_nb_zero)) via logsumexp
    # Stack [log(pi), log(1-pi) + log_nb_zero] and logsumexp
    log_terms_zero = torch.stack([log_pi, log_one_minus_pi + log_nb_zero], dim=-1)
    log_prob_zero = torch.logsumexp(log_terms_zero, dim=-1)

    # Non-zero case: log P(Y=k) = log(1-pi) + log(NB(k))
    # log NB(k) = lgamma(k+theta) - lgamma(theta) - lgamma(k+1)
    #           + theta*log(theta/(theta+mu)) + k*log(mu/(theta+mu))
    k = y_true.clamp(min=0).float()  # Ensure non-negative, cast to float for lgamma

    # Compute log NB for non-zero k
    # lgamma is safe for k >= 0 (k+1 >= 1) and theta >= eps (already clamped)
    log_nb_nonzero = (
        torch.lgamma(k + theta)
        - torch.lgamma(theta)
        - torch.lgamma(k + 1)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + k * (torch.log(mu) - torch.log(theta + mu))
    )
    log_prob_nonzero = log_one_minus_pi + log_nb_nonzero

    # Combine: total log probability
    log_prob = is_zero * log_prob_zero + is_nonzero * log_prob_nonzero

    # Return negative log-likelihood (we want to minimize)
    nll = -log_prob

    # Clamp to prevent extreme outliers from dominating
    # Also handle any numerical issues
    nll = nll.clamp(min=0.0, max=max_loss)

    # Check for NaN and warn (don't silently replace)
    if torch.isnan(nll).any():
        n_nan = torch.isnan(nll).sum().item()
        warnings.warn(f"ZINB loss has {n_nan} NaN values, replacing with max_loss")
        nll = torch.nan_to_num(nll, nan=max_loss)

    return nll.mean()


def interpret_casualty_outputs_as_zinb(
    casualty_pred: torch.Tensor,
    casualty_var: torch.Tensor,
    mu_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reinterpret existing casualty head outputs as ZINB parameters.

    The CasualtyPredictionHead outputs:
        - casualty_pred: [batch, seq, 3] - originally (deaths_best, low, high)
        - casualty_var: [batch, seq, 3] - originally variance estimates

    We reinterpret the last timestep as ZINB parameters:
        - pred[:, 0] -> pi (zero-inflation) via sigmoid
        - pred[:, 1] -> mu (mean) via exp (log-link for natural magnitude scaling)
        - var[:, 0] -> theta (dispersion) via softplus, bounded

    Args:
        casualty_pred: Model prediction output [batch, 3]
        casualty_var: Model variance output [batch, 3]
        mu_scale: Optional scale factor for mu (applied after exp)

    Returns:
        Tuple of (pi, mu, theta) each [batch]
    """
    # Zero-inflation probability from first pred channel
    # Clamp input to prevent extreme sigmoid saturation
    pi = torch.sigmoid(casualty_pred[:, 0].clamp(-10, 10))

    # Conditional mean from second pred channel
    # Use exp (log-link) for natural scaling across orders of magnitude
    # Clamp input to prevent overflow: exp(10) ~ 22000, exp(15) ~ 3.3M
    log_mu = casualty_pred[:, 1].clamp(-5, 12)  # mu range: ~0.007 to ~160,000
    mu = torch.exp(log_mu) * mu_scale

    # Dispersion from first variance channel
    # Bound theta to reasonable range for conflict data
    # Lower theta = more overdispersion, typical range 0.1-100
    theta_raw = F.softplus(casualty_var[:, 0])
    theta = theta_raw.clamp(min=0.1, max=100.0)

    return pi, mu, theta


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing decay.

    The scheduler performs:
    1. Linear warmup from warmup_start_lr to base_lr over warmup_epochs
    2. Cosine annealing decay to min_lr over remaining epochs
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 0.0,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr

        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if total_epochs <= warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be > warmup_epochs ({warmup_epochs})"
            )

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        # Use first base_lr as reference (all groups get same schedule)
        base_lr = self.base_lrs[0] if self.base_lrs else 1e-4

        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            lr = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay

        # Return same LR for all param groups (PyTorch 2.10+ requires matching length)
        return [lr] * len(self.optimizer.param_groups)


# =============================================================================
# GRADIENT ACCUMULATOR
# =============================================================================

class GradientAccumulator:
    """
    Helper class for gradient accumulation to simulate larger batch sizes.

    Accumulates gradients over multiple forward passes before performing
    an optimizer step with gradient clipping.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.optimizer = optimizer
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.current_step = 0
        self._accumulated_loss = 0.0

    def step(self, loss: torch.Tensor, model: nn.Module, scaler: Optional[GradScaler] = None) -> bool:
        """
        Accumulate gradients and perform optimizer step if ready.

        Args:
            loss: The loss tensor (gradients should already be computed)
            model: The model (for gradient clipping)
            scaler: Optional GradScaler for mixed precision training

        Returns:
            True if optimizer step was performed
        """
        self.current_step += 1
        self._accumulated_loss += loss.detach().item()

        if self.should_step():
            if scaler is not None:
                # Unscale gradients before clipping
                scaler.unscale_(self.optimizer)

            # Apply gradient clipping
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm
                )

            if scaler is not None:
                # Use scaler for optimizer step
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard optimizer step
                self.optimizer.step()

            self.optimizer.zero_grad()
            self._accumulated_loss = 0.0
            return True

        return False

    def zero_grad(self) -> None:
        """Zero gradients and reset counter."""
        self.optimizer.zero_grad()
        self.current_step = 0
        self._accumulated_loss = 0.0

    def should_step(self) -> bool:
        """Check if optimizer step should be performed."""
        return self.current_step % self.accumulation_steps == 0

    def get_effective_batch_size(self, batch_size: int) -> int:
        """Calculate effective batch size with accumulation."""
        return batch_size * self.accumulation_steps


# =============================================================================
# MULTI-RESOLUTION COLLATE FUNCTION (ENHANCED)
# =============================================================================

def enhanced_multi_resolution_collate_fn(
    batch: List[MultiResolutionSample]
) -> Dict[str, Any]:
    """
    Enhanced collate function for multi-resolution data.

    Handles:
    - Padding daily sequences to uniform length
    - Padding monthly sequences to uniform length
    - Stacking observation masks
    - Aligning month boundaries
    - Computing observation rates per batch item
    """
    if not batch:
        raise ValueError("Empty batch")

    batch_size = len(batch)

    # Get source names from first sample
    daily_sources = list(batch[0].daily_features.keys())
    monthly_sources = list(batch[0].monthly_features.keys())

    # Find max sequence lengths in batch
    max_daily_len = max(
        sample.daily_features[daily_sources[0]].shape[0]
        for sample in batch
    ) if daily_sources else 0

    max_monthly_len = max(
        sample.monthly_features[monthly_sources[0]].shape[0]
        for sample in batch
    ) if monthly_sources else 0

    # Initialize output dictionaries
    batched_daily_features = {}
    batched_daily_masks = {}
    batched_monthly_features = {}
    batched_monthly_masks = {}

    # Track observation rates for stratified metrics
    daily_obs_rates = []
    monthly_obs_rates = []

    # Batch daily data
    for source_name in daily_sources:
        n_features = batch[0].daily_features[source_name].shape[1]

        features_batch = torch.full(
            (batch_size, max_daily_len, n_features),
            fill_value=MISSING_VALUE,
            dtype=torch.float32
        )
        masks_batch = torch.zeros(
            (batch_size, max_daily_len, n_features),
            dtype=torch.bool
        )

        for i, sample in enumerate(batch):
            seq_len = sample.daily_features[source_name].shape[0]
            features_batch[i, :seq_len] = sample.daily_features[source_name]
            masks_batch[i, :seq_len] = sample.daily_masks[source_name]

            # Compute observation rate for this sample
            if source_name == daily_sources[0]:
                obs_rate = sample.daily_masks[source_name].float().mean().item()
                daily_obs_rates.append(obs_rate)

        batched_daily_features[source_name] = features_batch
        batched_daily_masks[source_name] = masks_batch

    # Batch monthly data
    for source_name in monthly_sources:
        n_features = batch[0].monthly_features[source_name].shape[1]

        features_batch = torch.full(
            (batch_size, max_monthly_len, n_features),
            fill_value=MISSING_VALUE,
            dtype=torch.float32
        )
        masks_batch = torch.zeros(
            (batch_size, max_monthly_len, n_features),
            dtype=torch.bool
        )

        for i, sample in enumerate(batch):
            seq_len = sample.monthly_features[source_name].shape[0]
            features_batch[i, :seq_len] = sample.monthly_features[source_name]
            masks_batch[i, :seq_len] = sample.monthly_masks[source_name]

            if source_name == monthly_sources[0]:
                obs_rate = sample.monthly_masks[source_name].float().mean().item()
                monthly_obs_rates.append(obs_rate)

        batched_monthly_features[source_name] = features_batch
        batched_monthly_masks[source_name] = masks_batch

    # Batch month boundary indices
    max_n_months = max(sample.month_boundary_indices.shape[0] for sample in batch)
    month_boundaries_batch = torch.zeros(
        (batch_size, max_n_months, 2),
        dtype=torch.long
    )

    for i, sample in enumerate(batch):
        n_months = sample.month_boundary_indices.shape[0]
        month_boundaries_batch[i, :n_months] = sample.month_boundary_indices

    # Sequence lengths for masking
    daily_seq_lens = torch.tensor([
        sample.daily_features[daily_sources[0]].shape[0] if daily_sources else 0
        for sample in batch
    ], dtype=torch.long)

    monthly_seq_lens = torch.tensor([
        sample.monthly_features[monthly_sources[0]].shape[0] if monthly_sources else 0
        for sample in batch
    ], dtype=torch.long)

    # Extract dates for target lookup
    # Daily dates: convert numpy datetime64 to string format YYYY-MM-DD
    batch_daily_dates = []
    for sample in batch:
        dates = []
        for d in sample.daily_dates:
            try:
                # Convert numpy datetime64 to string
                dt = pd.Timestamp(d)
                dates.append(dt.strftime('%Y-%m-%d'))
            except Exception:
                dates.append('')
        batch_daily_dates.append(dates)

    # Monthly dates: extract (year, month) tuples
    batch_monthly_year_months = []
    for sample in batch:
        year_months = []
        for d in sample.monthly_dates:
            try:
                dt = pd.Timestamp(d)
                year_months.append((dt.year, dt.month))
            except Exception:
                year_months.append((0, 0))
        batch_monthly_year_months.append(year_months)

    # =========================================================================
    # BATCH FORECAST TARGETS (for autoregressive training)
    # =========================================================================
    batched_forecast_targets = {}
    batched_forecast_masks = {}

    # Check if forecast targets are available
    if batch[0].forecast_targets is not None and batch[0].forecast_targets:
        forecast_sources = list(batch[0].forecast_targets.keys())

        for source_name in forecast_sources:
            n_features = batch[0].forecast_targets[source_name].shape[1]

            targets_batch = torch.full(
                (batch_size, max_monthly_len, n_features),
                fill_value=MISSING_VALUE,
                dtype=torch.float32
            )
            masks_batch = torch.zeros(
                (batch_size, max_monthly_len, n_features),
                dtype=torch.bool
            )

            for i, sample in enumerate(batch):
                if sample.forecast_targets is not None and source_name in sample.forecast_targets:
                    seq_len = sample.forecast_targets[source_name].shape[0]
                    targets_batch[i, :seq_len] = sample.forecast_targets[source_name]
                    if sample.forecast_masks is not None and source_name in sample.forecast_masks:
                        masks_batch[i, :seq_len] = sample.forecast_masks[source_name]

            batched_forecast_targets[source_name] = targets_batch
            batched_forecast_masks[source_name] = masks_batch

    # =========================================================================
    # BATCH ISW EMBEDDINGS (for narrative alignment)
    # =========================================================================
    batched_isw_embedding = None
    batched_isw_mask = None

    if batch[0].isw_embedding is not None:
        isw_embedding_dim = batch[0].isw_embedding.shape[1]

        batched_isw_embedding = torch.zeros(
            (batch_size, max_daily_len, isw_embedding_dim),
            dtype=torch.float32
        )
        batched_isw_mask = torch.zeros(
            (batch_size, max_daily_len),
            dtype=torch.bool
        )

        for i, sample in enumerate(batch):
            if sample.isw_embedding is not None:
                seq_len = sample.isw_embedding.shape[0]
                batched_isw_embedding[i, :seq_len] = sample.isw_embedding
                if sample.isw_mask is not None:
                    batched_isw_mask[i, :seq_len] = sample.isw_mask

    return {
        'daily_features': batched_daily_features,
        'daily_masks': batched_daily_masks,
        'monthly_features': batched_monthly_features,
        'monthly_masks': batched_monthly_masks,
        'month_boundary_indices': month_boundaries_batch,
        'daily_seq_lens': daily_seq_lens,
        'monthly_seq_lens': monthly_seq_lens,
        'batch_size': batch_size,
        'sample_indices': [sample.sample_idx for sample in batch],
        'daily_obs_rates': torch.tensor(daily_obs_rates, dtype=torch.float32),
        'monthly_obs_rates': torch.tensor(monthly_obs_rates, dtype=torch.float32),
        'daily_dates': batch_daily_dates,
        'monthly_year_months': batch_monthly_year_months,
        'forecast_targets': batched_forecast_targets,
        'forecast_masks': batched_forecast_masks,
        'isw_embedding': batched_isw_embedding,
        'isw_mask': batched_isw_mask,
    }


# =============================================================================
# MULTI-RESOLUTION HAN MODEL (imported from multi_resolution_han.py)
# =============================================================================
# MultiResolutionHAN and create_multi_resolution_han are imported at the top
# from multi_resolution_han module. The model is defined in that module to
# avoid duplication and ensure consistency.
#
# NOTE: The duplicate class definition that was here has been removed.
# See multi_resolution_han.py for the model implementation.


# MULTI-RESOLUTION TRAINER
# =============================================================================

class MultiResolutionTrainer:
    """
    Trainer for MultiResolutionHAN with multi-task learning.

    Features:
    - Multi-resolution data loading (daily + monthly)
    - Uncertainty-weighted multi-task loss
    - Gradient accumulation for larger effective batch sizes
    - Warmup cosine learning rate schedule
    - Early stopping with patience
    - Comprehensive logging and checkpointing
    - Observation-rate-stratified metrics
    """

    def __init__(
        self,
        model: MultiResolutionHAN,
        train_dataset: MultiResolutionDataset,
        val_dataset: MultiResolutionDataset,
        test_dataset: Optional[MultiResolutionDataset] = None,
        batch_size: int = 8,
        accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 200,
        patience: int = 30,
        warmup_epochs: int = 10,
        checkpoint_dir: str = str(CHECKPOINT_DIR),
        device: str = 'auto',
        run_manager: Optional[TrainingRunManager] = None,
        use_amp: bool = True,
        gradient_checkpointing: bool = True,
        use_temporal_reg: bool = False,
        temporal_corr_weight: float = 0.01,
        temporal_smooth_weight: float = 0.001,
    ):
        """
        Initialize trainer.

        Args:
            model: MultiResolutionHAN model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
            batch_size: Batch size per accumulation step
            accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            warmup_epochs: Number of warmup epochs
            checkpoint_dir: Directory for saving checkpoints
            device: Device ('auto', 'cpu', 'cuda', 'mps')
            run_manager: Optional TrainingRunManager for organized output (probe integration)
            use_amp: Use automatic mixed precision (bf16/fp16) for faster training
            gradient_checkpointing: Trade compute for memory by recomputing activations
            use_temporal_reg: Enable temporal regularization to prevent temporal shortcuts
            temporal_corr_weight: Weight for temporal correlation penalty
            temporal_smooth_weight: Weight for temporal smoothness penalty
        """
        # Detect device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)

        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx, 50xx, A100, etc.)
        if device == 'cuda':
            torch.set_float32_matmul_precision('high')  # Uses TF32 for float32 matmuls
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = model.to(self.device)

        # Mixed precision training setup
        # Only use AMP on CUDA (not supported on MPS/CPU)
        self.use_amp = use_amp and (device == 'cuda')
        if self.use_amp:
            # Use bfloat16 if available (better for training), else float16
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                print("  Mixed precision: bfloat16 (optimal for training)")
            else:
                self.amp_dtype = torch.float16
                print("  Mixed precision: float16")
            self.scaler = GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            if use_amp and device != 'cuda':
                print("  Mixed precision: disabled (requires CUDA)")

        # Gradient checkpointing (trade compute for memory)
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
            print("  Gradient checkpointing: enabled")

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=enhanced_multi_resolution_collate_fn,
            pin_memory=(device != 'cpu'),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=enhanced_multi_resolution_collate_fn,
            pin_memory=(device != 'cpu'),
        )

        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=enhanced_multi_resolution_collate_fn,
                pin_memory=(device != 'cpu'),
            )
        else:
            self.test_loader = None

        # Training config
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Run manager for probe integration
        self.run_manager = run_manager
        if run_manager is not None:
            # Setup run directory structure
            run_manager.setup()
            # Get stage 3 (HAN) directory for this run
            self.run_checkpoint_dir = run_manager.get_stage_dir(3)
            self.run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Training run ID: {run_manager.run_id}")
            print(f"Run checkpoints: {self.run_checkpoint_dir}")
        else:
            self.run_checkpoint_dir = None

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        # Ensure warmup_epochs is less than total_epochs
        effective_warmup = min(warmup_epochs, max(1, num_epochs - 1))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=effective_warmup,
            total_epochs=num_epochs,
            warmup_start_lr=learning_rate * 0.01,
            min_lr=1e-6,
        )

        # ISW alignment weight (applied separately, not through multi-task uncertainty)
        # Lower weight (0.1) because ISW alignment is auxiliary to main prediction tasks
        self.isw_alignment_weight = 0.1

        # Multi-task loss (updated task names to match real targets)
        # Use task priors to rebalance from dominant casualty head (62.6% -> 25%)
        task_names = ['casualty', 'regime', 'transition', 'anomaly', 'forecast']
        # Add daily forecast if model has the head
        if hasattr(model, 'daily_forecast_head'):
            task_names.append('daily_forecast')
        # Add ISW alignment if model has it enabled
        if hasattr(model, 'use_isw_alignment') and model.use_isw_alignment:
            task_names.append('isw_alignment')
        self.multi_task_loss = MultiTaskLoss(
            task_names=task_names,
            use_task_priors=True,  # Probe-based weight initialization
        ).to(self.device)

        # Temporal regularizer (temporal-deconfounding-plan.md)
        # Prevents model from learning temporal shortcuts that cause early overfitting
        self.use_temporal_reg = use_temporal_reg
        if use_temporal_reg:
            self.temporal_regularizer = TemporalRegularizer(
                correlation_weight=temporal_corr_weight,
                smoothness_weight=temporal_smooth_weight,
            ).to(self.device)
            print(f"Temporal regularization enabled: corr_weight={temporal_corr_weight}, smooth_weight={temporal_smooth_weight}")
        else:
            self.temporal_regularizer = None

        # Add loss parameters to optimizer
        self.optimizer.add_param_group({
            'params': self.multi_task_loss.parameters(),
            'lr': learning_rate * 10,  # Higher LR for task weights
        })

        # Gradient accumulator
        self.grad_accumulator = GradientAccumulator(
            self.optimizer,
            accumulation_steps=accumulation_steps,
            max_grad_norm=1.0,
        )

        # Initialize target loader for real training targets
        self.target_loader = TargetLoader(str(DATA_DIR))
        if not self.target_loader.load():
            warnings.warn(
                "Failed to load all training targets. "
                "Some targets may fall back to synthetic values."
            )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

        # History tracking
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

        # Config for checkpointing
        self.config = {
            'batch_size': batch_size,
            'accumulation_steps': accumulation_steps,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'patience': patience,
            'warmup_epochs': warmup_epochs,
            'device': str(self.device),
        }

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on transformer layers to reduce memory.

        This trades ~30% more compute for ~70% less activation memory,
        allowing larger batch sizes or model dimensions.
        """
        from torch.utils.checkpoint import checkpoint

        # Enable checkpointing on daily encoders
        for name, encoder in self.model.daily_encoders.items():
            if hasattr(encoder, 'transformer'):
                # PyTorch 2.0+ native checkpointing
                if hasattr(encoder.transformer, 'gradient_checkpointing_enable'):
                    encoder.transformer.gradient_checkpointing_enable()
                else:
                    # Fallback: wrap transformer forward
                    encoder._orig_forward = encoder.forward
                    def make_checkpointed_forward(enc):
                        def checkpointed_forward(*args, **kwargs):
                            return checkpoint(enc._orig_forward, *args, use_reentrant=False, **kwargs)
                        return checkpointed_forward
                    encoder.forward = make_checkpointed_forward(encoder)

        # Enable on monthly encoder if available
        if hasattr(self.model, 'monthly_encoder'):
            for name, enc in self.model.monthly_encoder.source_encoders.items():
                if hasattr(enc, 'encoder') and hasattr(enc.encoder, 'layers'):
                    for layer in enc.encoder.layers:
                        layer.use_checkpoint = True

        # Enable on temporal encoder
        if hasattr(self.model, 'temporal_encoder'):
            if hasattr(self.model.temporal_encoder, 'layers'):
                for layer in self.model.temporal_encoder.layers:
                    if hasattr(layer, 'use_checkpoint'):
                        layer.use_checkpoint = True

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device and preprocess MISSING_VALUE.

        This method also replaces MISSING_VALUE (-999.0) sentinel with 0.0 in feature
        tensors as a safety net, ensuring no extreme values flow through the model
        even if model-level handling misses something.
        """
        moved = {}

        # FIRST PASS: Move all tensors to device (including masks)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        moved[key][k] = v.to(self.device)
                    else:
                        moved[key][k] = v
            else:
                moved[key] = value

        # SECOND PASS: Apply masking using already-moved tensors
        # This fixes the device mismatch bug where masks were accessed from original batch
        for key in ('daily_features', 'monthly_features', 'forecast_targets'):
            if key in moved:
                mask_key = key.replace('features', 'masks').replace('targets', 'masks')
                if mask_key in moved:
                    for k in moved[key]:
                        if k in moved[mask_key]:
                            # Both feature tensor and mask are now on the same device
                            moved[key][k] = moved[key][k].masked_fill(~moved[mask_key][k], 0.0)

        # Verify no extreme values remain in feature tensors
        for key in ('daily_features', 'monthly_features', 'forecast_targets'):
            if key in moved and isinstance(moved[key], dict):
                for source_name, tensor in moved[key].items():
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any():
                            warnings.warn(f"NaN values found in {key}[{source_name}]")
                        if (tensor.abs() > 100).any():
                            # Log warning but don't fail - model should handle this
                            extreme_count = (tensor.abs() > 100).sum().item()
                            warnings.warn(
                                f"Extreme values ({extreme_count}) in {key}[{source_name}], "
                                "possibly incomplete MISSING_VALUE handling"
                            )

        return moved

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual task losses using real training targets.

        Uses TargetLoader to get real targets from:
        - Phase labels (from curated timeline data)
        - Casualty data (from HDX/UCDP conflict data)
        - VIIRS anomaly scores (from nightlight data)
        """
        # Check for NaN in model outputs and replace to prevent propagation
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor) and torch.isnan(val).any():
                warnings.warn(f"NaN detected in model output: {key}")
                # Replace NaN with zeros to prevent propagation
                outputs[key] = torch.nan_to_num(val, nan=0.0)

        losses = {}
        batch_size = batch['batch_size']

        # Get dates from batch for target lookup
        daily_dates = batch.get('daily_dates', None)
        monthly_year_months = batch.get('monthly_year_months', None)

        # =====================================================================
        # Get real targets from TargetLoader
        # =====================================================================

        # For each sample in batch, get the last daily date (prediction target)
        # and the monthly targets
        batch_daily_dates_last = []
        batch_monthly_ym_last = []

        if daily_dates is not None:
            for dates_list in daily_dates:
                # Get the last non-empty date
                last_date = ''
                for d in reversed(dates_list):
                    if d:
                        last_date = d
                        break
                batch_daily_dates_last.append(last_date)

        if monthly_year_months is not None:
            for ym_list in monthly_year_months:
                # Get the last valid (year, month)
                last_ym = (0, 0)
                for ym in reversed(ym_list):
                    if ym[0] > 0:
                        last_ym = ym
                        break
                batch_monthly_ym_last.append(last_ym)

        # Get combined targets from TargetLoader
        if batch_daily_dates_last and batch_monthly_ym_last:
            try:
                # Get targets - use last date for each sample for simplicity
                combined_targets = self.target_loader.get_combined_targets(
                    daily_dates=batch_daily_dates_last,
                    monthly_year_months=batch_monthly_ym_last,
                )

                # Convert to tensors
                target_tensors = targets_to_tensors(combined_targets, self.device)
            except Exception as e:
                # If target loading fails, fall back to synthetic targets
                warnings.warn(f"Target loading failed, using synthetic targets: {e}")
                target_tensors = None
        else:
            target_tensors = None

        # =====================================================================
        # Phase/Regime Classification Loss
        # NOTE: Current model outputs 4 classes, but we have 11 real phases.
        # TODO: Update MultiResolutionHAN to use num_regime_classes=N_PHASES (11)
        # For now, we map the 11 phases to 4 coarse categories when needed.
        # =====================================================================
        regime_logits = outputs['regime_logits']  # [batch, seq, n_classes]
        seq_len = regime_logits.size(1)
        model_n_classes = regime_logits.size(-1)

        if target_tensors is not None and 'daily_phase_index' in target_tensors:
            # Get phase targets - shape [batch]
            phase_targets = target_tensors['daily_phase_index']
            phase_valid = target_tensors['daily_phase_valid']

            # Expand to match sequence length for masked loss
            # Use last timestep's logits vs the target phase
            regime_logits_last = regime_logits[:, -1, :]  # [batch, n_classes]

            # Handle mismatch between model output classes and target classes
            if model_n_classes != N_PHASES:
                # Map 11 phases to coarse categories (model has 4 classes)
                # 0-2: offensive phases -> class 0
                # 3-4: donbas/counteroffensive -> class 1
                # 5-7: bakhmut/2023/avdiivka -> class 2
                # 8-10: 2024 offensive/kursk/baseline -> class 3
                phase_to_coarse = torch.tensor(
                    [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3],
                    device=self.device, dtype=torch.long
                )
                # Clamp phase targets to valid range
                clamped_targets = phase_targets.clamp(0, len(phase_to_coarse) - 1)
                phase_targets_mapped = phase_to_coarse[clamped_targets]
            else:
                phase_targets_mapped = phase_targets

            # Compute masked cross-entropy loss with anti-collapse measures
            if phase_valid.any():
                valid_idx = phase_valid.nonzero(as_tuple=True)[0]
                valid_targets = phase_targets_mapped[valid_idx]

                # === ANTI-COLLAPSE: Class-balanced weights ===
                # Compute class frequencies and use inverse as weights
                # This prevents the model from only predicting the majority class
                class_counts = torch.bincount(valid_targets, minlength=model_n_classes).float()
                class_counts = class_counts.clamp(min=1)  # Avoid division by zero
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum() * model_n_classes  # Normalize

                # Clamp logits to prevent extreme values
                clamped_logits = regime_logits_last[valid_idx].clamp(-50, 50)

                # === ANTI-COLLAPSE: Label smoothing ===
                # Prevents overconfident predictions that lead to collapse
                regime_loss = F.cross_entropy(
                    clamped_logits,
                    valid_targets,
                    weight=class_weights,
                    label_smoothing=0.1,  # 10% smoothing to prevent overconfidence
                )

                # === ANTI-COLLAPSE: Entropy regularization (STRONGER) ===
                # Encourage the model to make confident but diverse predictions
                # Penalize when prediction entropy is too low (constant outputs)
                regime_probs = F.softmax(clamped_logits, dim=-1)
                # Per-sample entropy
                sample_entropy = -(regime_probs * (regime_probs + 1e-8).log()).sum(dim=-1)
                mean_entropy = sample_entropy.mean()
                # Target entropy: 50% of max (was 40%)
                target_entropy = math.log(model_n_classes) * 0.5
                entropy_penalty = F.relu(target_entropy - mean_entropy)

                # === ANTI-COLLAPSE: Prediction diversity across batch (STRONGER) ===
                # Penalize when all samples predict the same class
                mean_probs = regime_probs.mean(dim=0)  # Average prediction distribution
                batch_diversity = -(mean_probs * (mean_probs + 1e-8).log()).sum()
                # Should be high if predictions are diverse across batch
                diversity_target = math.log(model_n_classes) * 0.7  # was 0.6
                diversity_penalty = F.relu(diversity_target - batch_diversity)

                # === ANTI-COLLAPSE: Loss floor ===
                # Don't let regime loss go below 0.05 to prevent collapse
                regime_loss = torch.maximum(regime_loss, torch.tensor(0.05, device=regime_loss.device))

                # Combine losses (STRONGER weights: 0.3 instead of 0.1)
                losses['regime'] = torch.nan_to_num(regime_loss, nan=0.0)
                losses['regime'] += entropy_penalty * 0.3 + diversity_penalty * 0.3
            else:
                # No valid targets - use small regularization connected to model output
                losses['regime'] = regime_logits_last.pow(2).mean() * 1e-6
        else:
            # No real targets available - use autoregressive regime prediction
            # Instead of synthetic random targets, use temporal consistency loss:
            # consecutive timesteps should have similar regime predictions
            regime_logits_last = regime_logits[:, -1, :]

            # Temporal consistency loss: encourage smooth regime transitions
            # Compare predictions at t vs t-1 (should be similar most of the time)
            if seq_len >= 2:
                regime_t = regime_logits[:, 1:, :]  # [batch, seq-1, n_classes]
                regime_t_minus_1 = regime_logits[:, :-1, :]  # [batch, seq-1, n_classes]
                # KL divergence between consecutive predictions (should be small)
                regime_probs_t = F.softmax(regime_t, dim=-1)
                regime_log_probs_t_minus_1 = F.log_softmax(regime_t_minus_1, dim=-1)
                # Symmetric KL for stability
                kl_forward = F.kl_div(regime_log_probs_t_minus_1, regime_probs_t, reduction='batchmean')
                kl_backward = F.kl_div(
                    F.log_softmax(regime_t, dim=-1),
                    F.softmax(regime_t_minus_1, dim=-1),
                    reduction='batchmean'
                )
                temporal_consistency = (kl_forward + kl_backward) / 2
                # Also add entropy regularization to prevent collapse to single class
                entropy = -(regime_probs_t * regime_probs_t.clamp(min=1e-8).log()).sum(dim=-1).mean()
                target_entropy = math.log(model_n_classes) * 0.5  # Encourage moderate entropy
                entropy_penalty = (target_entropy - entropy).abs()
                losses['regime'] = temporal_consistency * 0.1 + entropy_penalty * 0.1
            else:
                # Single timestep - just regularize logits
                losses['regime'] = regime_logits_last.pow(2).mean() * 1e-6

        # =====================================================================
        # Phase Transition Detection Loss (binary)
        # =====================================================================
        # Use anomaly_score as a proxy for transition detection
        # (model should learn to detect phase transitions)
        anomaly_score = outputs['anomaly_score']  # [batch, seq, 1]

        if target_tensors is not None and 'daily_is_transition' in target_tensors:
            transition_targets = target_tensors['daily_is_transition'].float()
            phase_valid = target_tensors['daily_phase_valid']

            # Use last timestep
            anomaly_score_last = anomaly_score[:, -1, 0]  # [batch]

            if phase_valid.any():
                valid_idx = phase_valid.nonzero(as_tuple=True)[0]
                # Clamp logits to prevent extreme values
                clamped_scores = anomaly_score_last[valid_idx].clamp(-50, 50)
                transition_loss = F.binary_cross_entropy_with_logits(
                    clamped_scores,
                    transition_targets[valid_idx],
                )
                losses['transition'] = torch.nan_to_num(transition_loss, nan=0.0)
            else:
                # No valid targets - small regularization
                losses['transition'] = anomaly_score_last.pow(2).mean() * 1e-6
        else:
            # Small regularization connected to model output
            anomaly_score_last = anomaly_score[:, -1, 0]
            losses['transition'] = anomaly_score_last.pow(2).mean() * 1e-6

        # =====================================================================
        # Casualty Prediction Loss (ZINB - Zero-Inflated Negative Binomial)
        # =====================================================================
        # ZINB properly models count data with:
        # - Zero-inflation: many location-months have zero casualties
        # - Overdispersion: variance > mean in conflict data
        # - No gaming possible: proper distribution, not variance-MSE tradeoff
        #
        # The existing head outputs are reinterpreted as ZINB parameters:
        #   casualty_pred[:, 0] -> pi (zero-inflation prob) via sigmoid
        #   casualty_pred[:, 1] -> mu (conditional mean) via softplus
        #   casualty_var[:, 0] -> theta (dispersion) via softplus
        # =====================================================================
        casualty_pred = outputs['casualty_pred']  # [batch, seq, 3]
        casualty_var = outputs['casualty_var']  # [batch, seq, 3]

        # Use the last timestep for prediction
        casualty_pred_last = casualty_pred[:, -1, :]  # [batch, 3]
        casualty_var_last = casualty_var[:, -1, :]  # [batch, 3]

        if target_tensors is not None and 'monthly_fatalities' in target_tensors:
            fatality_targets = target_tensors['monthly_fatalities']  # [batch]
            casualty_valid = target_tensors['monthly_casualty_valid']

            if casualty_valid.any():
                valid_idx = casualty_valid.nonzero(as_tuple=True)[0]

                # Get valid predictions and targets
                pred_valid = casualty_pred_last[valid_idx]  # [n_valid, 3]
                var_valid = casualty_var_last[valid_idx]  # [n_valid, 3]
                target_counts_raw = fatality_targets[valid_idx]  # [n_valid] - raw counts

                # Scale targets to manageable range for ZINB
                # Monthly fatalities are O(1000-10000), scale down to O(1-10)
                # This allows the network's exp() output (initially ~1) to match
                # Network learns to predict scaled counts, interpretation scales back up
                TARGET_SCALE = 1000.0
                target_counts = target_counts_raw / TARGET_SCALE

                # Interpret outputs as ZINB parameters
                # mu_scale=1.0 because we now use exp() which naturally scales
                # The network learns log(mu) directly, so it can span orders of magnitude
                pi, mu, theta = interpret_casualty_outputs_as_zinb(pred_valid, var_valid)

                # Compute ZINB NLL loss on scaled targets
                casualty_loss = zinb_nll_loss(
                    y_true=target_counts,
                    pi=pi,
                    mu=mu,
                    theta=theta,
                )

                losses['casualty'] = casualty_loss
            else:
                # No valid targets - regularize towards reasonable ZINB params
                # Encourage: pi near 0.5, log_mu near 0 (mu~1), theta near 1
                pi_reg = (torch.sigmoid(casualty_pred_last[:, 0]) - 0.5).pow(2).mean()
                # log_mu regularization: encourage log_mu (pred[:, 1]) near 0
                log_mu_reg = casualty_pred_last[:, 1].pow(2).mean() * 0.01
                theta_reg = (F.softplus(casualty_var_last[:, 0]) - 1.0).pow(2).mean()
                losses['casualty'] = (pi_reg + log_mu_reg + theta_reg) * 0.01
        else:
            # Fallback: regularize predictions
            pi_reg = (torch.sigmoid(casualty_pred_last[:, 0]) - 0.5).pow(2).mean()
            log_mu_reg = casualty_pred_last[:, 1].pow(2).mean() * 0.01
            losses['casualty'] = (pi_reg + log_mu_reg) * 0.01

        # =====================================================================
        # Anomaly Detection Loss (VIIRS-based)
        # =====================================================================
        if target_tensors is not None and 'daily_anomaly' in target_tensors:
            viirs_anomaly_targets = target_tensors['daily_anomaly']  # [batch]
            viirs_valid = target_tensors['daily_viirs_valid']

            # Use anomaly score output
            anomaly_score_last = anomaly_score[:, -1, 0]  # [batch]

            if viirs_valid.any():
                valid_idx = viirs_valid.nonzero(as_tuple=True)[0]
                # Clamp predictions to prevent extreme values
                clamped_anomaly = anomaly_score_last[valid_idx].clamp(-10, 10)
                valid_targets = viirs_anomaly_targets[valid_idx]

                # MSE loss for anomaly score prediction
                anomaly_loss = F.mse_loss(
                    clamped_anomaly,
                    valid_targets,
                )

                # === ANTI-COLLAPSE: Variance penalty ===
                # Penalize when predictions have much lower variance than targets
                # This prevents the model from just predicting the mean
                pred_var = clamped_anomaly.var()
                target_var = valid_targets.var()
                if target_var > 1e-6:  # Only if targets have meaningful variance
                    variance_ratio = pred_var / (target_var + 1e-6)
                    # Penalize if prediction variance is less than 50% of target variance
                    variance_penalty = F.relu(0.5 - variance_ratio)
                else:
                    variance_penalty = torch.tensor(0.0, device=self.device)

                losses['anomaly'] = torch.nan_to_num(anomaly_loss, nan=0.0)
                losses['anomaly'] += variance_penalty * 0.1
            else:
                # No valid VIIRS - small regularization
                losses['anomaly'] = anomaly_score_last.pow(2).mean() * 1e-6
        else:
            # Fallback: regularize to zero
            anomaly_score_last = anomaly_score[:, -1, :]
            losses['anomaly'] = anomaly_score_last.pow(2).mean() * 0.01

        # =====================================================================
        # Autoregressive Forecast Loss
        # Predict next month's features from current representation.
        # This forces the model to learn temporal dynamics for prediction.
        # =====================================================================
        forecast_pred = outputs['forecast_pred']  # [batch, seq, n_features]

        # Check if we have real forecast targets from the batch
        forecast_targets = batch.get('forecast_targets', {})
        forecast_masks = batch.get('forecast_masks', {})

        if forecast_targets:
            # Compute autoregressive loss using real next-timestep features
            forecast_loss = self._compute_autoregressive_forecast_loss(
                forecast_pred=forecast_pred,
                forecast_targets=forecast_targets,
                forecast_masks=forecast_masks,
            )
            losses['forecast'] = forecast_loss
        else:
            # Fallback: use temporal smoothness regularization
            # Encourage predictions to vary over time (prevent collapse)
            # but also be temporally smooth (not erratic)
            if forecast_pred.size(1) >= 2:
                # Temporal difference penalty (smoothness)
                diff = forecast_pred[:, 1:, :] - forecast_pred[:, :-1, :]
                smoothness = diff.pow(2).mean()
                # Variance encouragement (prevent collapse)
                variance = forecast_pred.var(dim=1).mean()
                target_variance = 0.5  # Encourage non-zero variance
                variance_penalty = F.relu(target_variance - variance)
                losses['forecast'] = smoothness * 0.01 + variance_penalty * 0.1
            else:
                losses['forecast'] = forecast_pred.pow(2).mean() * 1e-6

        # =====================================================================
        # Daily Forecast Loss (DailyForecastingHead)
        # =====================================================================
        daily_forecast_pred = outputs.get('daily_forecast_pred')
        daily_forecast_targets = batch.get('daily_forecast_targets')
        daily_forecast_masks = batch.get('daily_forecast_masks')

        if daily_forecast_pred is not None and daily_forecast_targets is not None:
            # daily_forecast_pred: [batch, horizon, n_daily_features]
            # daily_forecast_targets: [batch, horizon, n_daily_features]
            daily_forecast_targets = daily_forecast_targets.to(daily_forecast_pred.device)

            if daily_forecast_masks is not None:
                daily_forecast_masks = daily_forecast_masks.to(daily_forecast_pred.device)
                # Mask out missing values
                valid_mask = daily_forecast_masks & (daily_forecast_targets > -900)
            else:
                valid_mask = daily_forecast_targets > -900

            if valid_mask.any():
                pred_valid = daily_forecast_pred[valid_mask]
                target_valid = daily_forecast_targets[valid_mask]
                daily_forecast_loss = F.mse_loss(pred_valid, target_valid)
                losses['daily_forecast'] = daily_forecast_loss
            else:
                losses['daily_forecast'] = daily_forecast_pred.pow(2).mean() * 1e-6

        # =====================================================================
        # ISW Narrative Alignment Loss (optional)
        # =====================================================================
        # Compute contrastive alignment loss between model temporal representations
        # and ISW narrative embeddings if:
        # 1. Model has ISW alignment module enabled
        # 2. Batch contains ISW embeddings
        if (hasattr(self.model, 'isw_alignment') and
            self.model.isw_alignment is not None and
            'isw_embedding' in batch):

            isw_embeddings = batch['isw_embedding']  # [batch, seq, 1024]
            isw_mask = batch.get('isw_mask', None)    # [batch, seq] True = valid

            # Get temporal output from model outputs
            temporal_output = outputs.get('temporal_output')  # [batch, seq, d_model]

            if temporal_output is not None and isw_embeddings is not None:
                # Align sequence lengths
                min_seq = min(temporal_output.shape[1], isw_embeddings.shape[1])
                temporal_aligned = temporal_output[:, :min_seq, :]
                isw_aligned = isw_embeddings[:, :min_seq, :]

                if isw_mask is not None:
                    isw_mask_aligned = isw_mask[:, :min_seq]
                else:
                    isw_mask_aligned = None

                # Project to shared space
                model_proj, isw_proj = self.model.isw_alignment(
                    temporal_aligned,
                    isw_aligned,
                    isw_mask_aligned,
                )

                # Compute contrastive alignment loss
                isw_alignment_loss = self.model.isw_alignment.compute_alignment_loss(
                    model_proj,
                    isw_proj,
                    isw_mask_aligned,
                )

                # Add to losses with weight 0.1 (lower than primary tasks)
                losses['isw_alignment'] = isw_alignment_loss * self.isw_alignment_weight

        return losses

    def _compute_autoregressive_forecast_loss(
        self,
        forecast_pred: torch.Tensor,
        forecast_targets: Dict[str, torch.Tensor],
        forecast_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute autoregressive loss for next-timestep feature prediction.

        This forces the model to PREDICT future states rather than just classify
        current state, addressing the C5 finding of output collapse.

        The forecast_pred output at time t should predict features at time t+1.
        We compare predictions with actual future features from forecast_targets.

        Args:
            forecast_pred: Model forecast predictions [batch, seq, n_features]
            forecast_targets: Dict of actual future features per source
                              Each tensor: [batch, seq, source_features]
            forecast_masks: Dict of validity masks per source
                           Each tensor: [batch, seq, source_features]

        Returns:
            Scalar loss tensor
        """
        batch_size, seq_len, n_pred_features = forecast_pred.shape
        device = forecast_pred.device

        total_loss = torch.tensor(0.0, device=device)
        n_valid_sources = 0

        # Concatenate all source targets for comparison with forecast_pred
        all_targets = []
        all_masks = []
        source_dims = {}
        current_dim = 0

        for source_name, target in forecast_targets.items():
            if source_name not in forecast_masks:
                continue

            # target shape: [batch, seq, source_features]
            mask = forecast_masks[source_name]
            n_source_features = target.shape[-1]

            # Track which dimensions correspond to which source
            source_dims[source_name] = (current_dim, current_dim + n_source_features)
            current_dim += n_source_features

            all_targets.append(target)
            all_masks.append(mask)

        if not all_targets:
            # No valid targets, return small regularization
            return forecast_pred.pow(2).mean() * 1e-6

        # Concatenate all sources
        # Shape: [batch, seq, total_features]
        concat_targets = torch.cat(all_targets, dim=-1)
        concat_masks = torch.cat(all_masks, dim=-1)

        total_target_features = concat_targets.shape[-1]

        # Handle dimension mismatch between forecast_pred and targets
        if n_pred_features != total_target_features:
            # Use a linear projection or just use the overlapping dimensions
            # For simplicity, compute loss on min(pred, target) features
            min_features = min(n_pred_features, total_target_features)
            pred_for_loss = forecast_pred[..., :min_features]
            targets_for_loss = concat_targets[..., :min_features]
            masks_for_loss = concat_masks[..., :min_features]
        else:
            pred_for_loss = forecast_pred
            targets_for_loss = concat_targets
            masks_for_loss = concat_masks

        # Autoregressive loss: prediction at t should match target at t
        # (targets are already shifted by prediction_horizon in the dataset)
        # Use MSE loss on valid positions
        valid_mask = masks_for_loss  # [batch, seq, features]

        if valid_mask.any():
            # Flatten for masked selection (use reshape for non-contiguous tensors)
            pred_flat = pred_for_loss.reshape(-1)
            target_flat = targets_for_loss.reshape(-1)
            mask_flat = valid_mask.reshape(-1)

            # Select only valid positions
            pred_valid = pred_flat[mask_flat]
            target_valid = target_flat[mask_flat]

            # Compute MSE loss with anti-collapse measures
            if pred_valid.numel() > 0:
                mse_loss = F.mse_loss(pred_valid, target_valid)

                # === ANTI-COLLAPSE: Variance preservation ===
                # Penalize when predictions have collapsed to near-constant values
                pred_var = pred_valid.var()
                target_var = target_valid.var()
                if target_var > 1e-6:
                    variance_ratio = pred_var / (target_var + 1e-6)
                    # Penalize if prediction variance is less than 30% of target variance
                    variance_penalty = F.relu(0.3 - variance_ratio) * 0.5
                else:
                    variance_penalty = torch.tensor(0.0, device=device)

                # Also add a temporal consistency term
                # Predictions should be smooth over time (but not constant!)
                if seq_len >= 2:
                    diff = pred_for_loss[:, 1:, :] - pred_for_loss[:, :-1, :]
                    smoothness_loss = diff.pow(2).mean() * 0.01

                    # === ANTI-COLLAPSE: Temporal variation requirement ===
                    # Penalize if predictions don't vary enough over time
                    temporal_var = pred_for_loss.var(dim=1).mean()
                    temporal_penalty = F.relu(0.1 - temporal_var) * 0.1
                else:
                    smoothness_loss = torch.tensor(0.0, device=device)
                    temporal_penalty = torch.tensor(0.0, device=device)

                total_loss = mse_loss + smoothness_loss + variance_penalty + temporal_penalty

                # === ANTI-COLLAPSE: Loss floor ===
                # Don't let forecast loss collapse below 0.01
                # This forces the model to keep learning instead of finding trivial solutions
                total_loss = torch.maximum(total_loss, torch.tensor(0.01, device=device))
            else:
                total_loss = forecast_pred.pow(2).mean() * 1e-6
        else:
            # No valid targets - small regularization
            total_loss = forecast_pred.pow(2).mean() * 1e-6

        return total_loss

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.grad_accumulator.zero_grad()

        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)
        n_batches = 0
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # Log ~10 times per epoch

        import time as _time
        _batch_start = _time.time()
        _iter_start = _time.time()
        print(f"  Entering training loop ({total_batches} batches)...", flush=True)
        for batch_idx, batch in enumerate(self.train_loader):
            _load_time = _time.time() - _iter_start
            if batch_idx < 20 or batch_idx % 16 == 0:
                print(f"  [B{batch_idx}] Loaded ({_load_time:.2f}s)", flush=True)
            batch = self._move_batch_to_device(batch)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                # Forward pass (include raion_masks if available for geographic sources)
                outputs = self.model(
                    daily_features=batch['daily_features'],
                    daily_masks=batch['daily_masks'],
                    monthly_features=batch['monthly_features'],
                    monthly_masks=batch['monthly_masks'],
                    month_boundaries=batch['month_boundary_indices'],
                    raion_masks=batch.get('raion_masks'),
                )

                # Compute individual task losses
                task_losses = self._compute_losses(outputs, batch)

                # Combine with uncertainty weighting
                total_loss, task_weights = self.multi_task_loss(task_losses)

                # Add temporal regularization if enabled
                if self.use_temporal_reg and self.temporal_regularizer is not None:
                    seq_len = batch['daily_features'][list(batch['daily_features'].keys())[0]].shape[1]
                    temp_penalty, temp_breakdown = self.temporal_regularizer(
                        outputs, seq_len, self.device
                    )
                    total_loss = total_loss + temp_penalty

            # Scale very large losses to prevent gradient explosion
            if total_loss.item() > 1e6:
                warnings.warn(f"Total loss {total_loss.item():.2e} exceeds 1e6, scaling down")
                total_loss = total_loss / (total_loss.item() / 1000)

            # Scale for gradient accumulation
            scaled_loss = total_loss / self.accumulation_steps

            # Backward pass with gradient scaling for mixed precision
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Step with gradient accumulation (handles scaler if AMP enabled)
            # Note: NaN gradient check removed - too slow for 73M params.
            # GradScaler handles inf/nan for mixed precision.
            self.grad_accumulator.step(total_loss, self.model, scaler=self.scaler)

            # Track losses
            epoch_losses['total'] += total_loss.item()
            for task_name, loss_val in task_losses.items():
                epoch_losses[task_name] += loss_val.item()

            # Track observation rates
            epoch_metrics['daily_obs_rate'] += batch['daily_obs_rates'].mean().item()
            epoch_metrics['monthly_obs_rate'] += batch['monthly_obs_rates'].mean().item()

            n_batches += 1
            _batch_time = _time.time() - _batch_start

            # Progress logging within epoch
            if n_batches % log_interval == 0 or n_batches == total_batches:
                avg_loss = epoch_losses['total'] / n_batches
                avg_time = _batch_time / n_batches
                print(f"  Batch {n_batches}/{total_batches} - loss: {avg_loss:.4f} ({avg_time:.2f}s/batch)", flush=True)

            # Reset timer for next iteration's load time measurement
            _iter_start = _time.time()

        # Average losses
        results = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        results.update({k: v / max(n_batches, 1) for k, v in epoch_metrics.items()})
        results['task_weights'] = self.multi_task_loss.get_task_weights()
        results['learning_rate'] = self.scheduler.get_last_lr()[0]

        # === COLLAPSE DETECTION ===
        # Warn if any loss component has collapsed to near-zero
        collapse_threshold = 0.001
        collapsed_tasks = []
        for task in ['regime', 'anomaly', 'forecast']:
            if task in results and results[task] < collapse_threshold:
                collapsed_tasks.append(f"{task}={results[task]:.6f}")
        if collapsed_tasks:
            warnings.warn(
                f"COLLAPSE DETECTED: Tasks with near-zero loss: {', '.join(collapsed_tasks)}. "
                f"Model may be outputting constant predictions. "
                f"Consider increasing anti-collapse regularization weights."
            )

        # Step scheduler
        self.scheduler.step()

        return results

    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary of validation metrics including observation-rate-stratified performance
        """
        self.model.eval()

        epoch_losses = defaultdict(float)

        # Track metrics stratified by observation rate
        high_obs_losses = defaultdict(list)  # >80% observed
        low_obs_losses = defaultdict(list)   # <50% observed

        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)

                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    # Forward pass (include raion_masks if available)
                    outputs = self.model(
                        daily_features=batch['daily_features'],
                        daily_masks=batch['daily_masks'],
                        monthly_features=batch['monthly_features'],
                        monthly_masks=batch['monthly_masks'],
                        month_boundaries=batch['month_boundary_indices'],
                        raion_masks=batch.get('raion_masks'),
                    )

                    # Compute losses
                    task_losses = self._compute_losses(outputs, batch)
                    total_loss, _ = self.multi_task_loss(task_losses)

                # Track losses
                epoch_losses['total'] += total_loss.item()
                for task_name, loss_val in task_losses.items():
                    epoch_losses[task_name] += loss_val.item()

                # Stratify by observation rate
                obs_rates = batch['monthly_obs_rates']
                for i, obs_rate in enumerate(obs_rates):
                    rate = obs_rate.item()
                    if rate > 0.8:
                        high_obs_losses['total'].append(total_loss.item())
                    elif rate < 0.5:
                        low_obs_losses['total'].append(total_loss.item())

                n_batches += 1

        # Average losses
        results = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Add stratified metrics
        if high_obs_losses['total']:
            results['high_obs_loss'] = np.mean(high_obs_losses['total'])
        if low_obs_losses['total']:
            results['low_obs_loss'] = np.mean(low_obs_losses['total'])

        # Uncertainty calibration (correlation between predicted uncertainty and error)
        results['uncertainty_stats'] = self.multi_task_loss.get_uncertainties()

        return results

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        fusion_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Save model checkpoint with training state and optional fusion metrics.

        This method saves multiple checkpoint types:
        1. latest_checkpoint.pt - Most recent epoch (overwritten each epoch)
        2. best_checkpoint.pt - Best validation loss (for prediction tasks)
        3. checkpoint_epoch_{N}.pt - Periodic snapshots (every 10 epochs)
        4. early_fusion_checkpoint.pt - Best fusion quality (for transfer/interpretability)

        The early fusion checkpoint is critical for cross-modal experiments.
        Based on Probe 2.1.4 findings, fusion quality peaks at epoch 10 (RSA=0.77)
        and degrades with further training as the model overfits to task patterns.

        Args:
            epoch: Current training epoch
            is_best: Whether this is the best validation loss checkpoint
            fusion_metrics: Optional dictionary with fusion quality metrics:
                - rsa: Representational Similarity Analysis score (0-1, higher=better)
                - cross_domain_alignment: Cross-domain representation alignment
                - attention_entropy: Attention distribution entropy

        Note:
            For fusion experiments (transfer learning, interpretability, probing),
            use load_fusion_checkpoint() or load_checkpoint_by_epoch(10).
            For pure prediction, use load_best_checkpoint().
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'multi_task_loss_state_dict': self.multi_task_loss.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'config': self.config,
        }

        # Add fusion metrics if provided
        if fusion_metrics is not None:
            checkpoint['fusion_metrics'] = fusion_metrics

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint (based on validation loss)
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (val_loss={self.best_val_loss:.4f})")

        # Save periodic checkpoint (every 10 epochs)
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)

        # Save early fusion checkpoint at optimal fusion epoch
        # Based on Probe 2.1.4: epoch 10 has RSA=0.77 (best fusion quality)
        if epoch == OPTIMAL_FUSION_EPOCH:
            fusion_path = self.checkpoint_dir / 'early_fusion_checkpoint.pt'
            # Add note about why this checkpoint is special
            checkpoint['fusion_checkpoint_note'] = (
                f"Saved at epoch {epoch} for optimal cross-modal fusion quality. "
                f"Based on Probe 2.1.4 RSA analysis: epoch 10 RSA=0.77, "
                f"epoch 50 RSA=0.37, epoch 100 RSA=0.34. "
                f"Use this checkpoint for transfer learning, interpretability, "
                f"and cross-domain analysis experiments."
            )
            torch.save(checkpoint, fusion_path)
            print(f"  Saved early fusion checkpoint at epoch {epoch} "
                  f"(optimal cross-modal fusion)")

        # === PROBE INTEGRATION: Save to run directory ===
        # This enables linking probe runs to specific training runs
        if self.run_checkpoint_dir is not None:
            # Add run metadata to checkpoint
            checkpoint['training_run_id'] = self.run_manager.run_id

            # Save latest to run directory
            run_latest = self.run_checkpoint_dir / 'latest_checkpoint.pt'
            torch.save(checkpoint, run_latest)

            # Save best to run directory
            if is_best:
                run_best = self.run_checkpoint_dir / 'best_checkpoint.pt'
                torch.save(checkpoint, run_best)

            # Save periodic checkpoints to run directory
            if epoch % 10 == 0:
                run_epoch = self.run_checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save(checkpoint, run_epoch)

    def save_fusion_checkpoint_with_rsa(
        self,
        epoch: int,
        rsa_score: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Save checkpoint if RSA score exceeds current best fusion quality.

        This method tracks fusion quality independently of validation loss.
        Call this after computing RSA scores during training to maintain
        the checkpoint with best cross-modal alignment.

        Based on Probe 2.1.4 findings:
          - Early epochs (10) have best fusion: RSA = 0.77
          - Later epochs overfit to tasks: RSA degrades to 0.34

        Args:
            epoch: Current training epoch
            rsa_score: RSA similarity score (0-1, higher = better fusion)
            additional_metrics: Optional dict with extra fusion metrics
                (e.g., attention_entropy, domain_alignment)

        Example:
            >>> # During training loop, compute RSA periodically
            >>> if epoch % 5 == 0:
            ...     rsa = compute_cross_domain_rsa(model, val_loader)
            ...     trainer.save_fusion_checkpoint_with_rsa(epoch, rsa)
        """
        # Initialize best RSA tracking if not exists
        if not hasattr(self, '_best_rsa_score'):
            self._best_rsa_score = 0.0
            self._best_rsa_epoch = 0

        # Check if this is better fusion quality
        if rsa_score > self._best_rsa_score:
            self._best_rsa_score = rsa_score
            self._best_rsa_epoch = epoch

            fusion_metrics = {
                'rsa': rsa_score,
                'epoch': epoch,
            }
            if additional_metrics:
                fusion_metrics.update(additional_metrics)

            # Build checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'multi_task_loss_state_dict': self.multi_task_loss.state_dict(),
                'best_val_loss': self.best_val_loss,
                'train_history': dict(self.train_history),
                'val_history': dict(self.val_history),
                'config': self.config,
                'fusion_metrics': fusion_metrics,
                'fusion_checkpoint_note': (
                    f"Best RSA checkpoint. RSA={rsa_score:.4f} at epoch {epoch}. "
                    f"Use for cross-modal fusion, transfer learning, and "
                    f"interpretability experiments."
                ),
            }

            fusion_path = self.checkpoint_dir / 'best_rsa_checkpoint.pt'
            torch.save(checkpoint, fusion_path)
            print(f"  Saved best RSA checkpoint: RSA={rsa_score:.4f} at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.multi_task_loss.load_state_dict(checkpoint['multi_task_loss_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = defaultdict(list, checkpoint['train_history'])
        self.val_history = defaultdict(list, checkpoint['val_history'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            verbose: Whether to print progress

        Returns:
            Dictionary with training history and final metrics
        """
        effective_batch = self.grad_accumulator.get_effective_batch_size(self.batch_size)

        if verbose:
            print("=" * 80)
            print("MULTI-RESOLUTION HAN TRAINING")
            print("=" * 80)
            print(f"Device: {self.device}")
            print(f"Effective batch size: {effective_batch} "
                  f"({self.batch_size} x {self.accumulation_steps})")
            print(f"Train samples: {len(self.train_dataset)}")
            print(f"Val samples: {len(self.val_dataset)}")
            print(f"Max epochs: {self.num_epochs}")
            print(f"Patience: {self.patience}")
            print("-" * 80, flush=True)
            print(f"Starting training loop...", flush=True)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train
            print(f"\n[Epoch {epoch}] Starting training...", flush=True)
            train_metrics = self.train_epoch()
            print(f"[Epoch {epoch}] Training complete. Starting validation...", flush=True)

            # Validate
            val_metrics = self.validate()

            # Track history
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.train_history[key].append(value)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.val_history[key].append(value)

            # Check for best model
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Print progress (every epoch for visibility)
            if verbose:
                task_weights = train_metrics.get('task_weights', {})
                weight_str = ", ".join([f"{k}:{v:.2f}" for k, v in task_weights.items()])

                # Build raw task loss string for debugging
                raw_task_losses = []
                for task in self.multi_task_loss.task_names:
                    if task in train_metrics:
                        raw_task_losses.append(f"{task}:{train_metrics[task]:.3f}")
                raw_loss_str = ", ".join(raw_task_losses) if raw_task_losses else "none"

                print(f"Epoch {epoch:3d}: "
                      f"train_loss={train_metrics['total']:.4f}, "
                      f"val_loss={val_metrics['total']:.4f}, "
                      f"lr={train_metrics['learning_rate']:.2e}", flush=True)
                print(f"          raw_losses=[{raw_loss_str}]", flush=True)
                print(f"          weights=[{weight_str}]"
                      f" {'*' if is_best else ''}", flush=True)

            # Early stopping
            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                break

        if verbose:
            print("-" * 80)
            print(f"Training complete. Best val loss: {self.best_val_loss:.4f} "
                  f"at epoch {self.best_epoch}")

        return {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }

    def evaluate(self, dataset: MultiResolutionDataset) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=enhanced_multi_resolution_collate_fn,
        )

        self.model.eval()

        all_losses = defaultdict(list)
        all_predictions = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                batch = self._move_batch_to_device(batch)

                outputs = self.model(
                    daily_features=batch['daily_features'],
                    daily_masks=batch['daily_masks'],
                    monthly_features=batch['monthly_features'],
                    monthly_masks=batch['monthly_masks'],
                    month_boundaries=batch['month_boundary_indices'],
                    raion_masks=batch.get('raion_masks'),
                )

                task_losses = self._compute_losses(outputs, batch)

                for task_name, loss_val in task_losses.items():
                    all_losses[task_name].append(loss_val.item())

                # Store predictions (last timestep only for simplicity)
                all_predictions['casualty'].append(outputs['casualty_pred'][:, -1, :].cpu())
                all_predictions['casualty_var'].append(outputs['casualty_var'][:, -1, :].cpu())
                all_predictions['regime'].append(
                    F.softmax(outputs['regime_logits'][:, -1, :], dim=-1).cpu()
                )
                all_predictions['anomaly'].append(
                    torch.sigmoid(outputs['anomaly_score'][:, -1, :]).cpu()
                )

        # Aggregate metrics
        metrics = {
            f'{task}_loss': np.mean(losses)
            for task, losses in all_losses.items()
        }
        metrics['total_loss'] = sum(metrics.values()) / len(metrics)

        return metrics


# =============================================================================
# TESTS
# =============================================================================

def test_dataloader_batch_shapes():
    """Test that DataLoader produces correct batch shapes."""
    print("\n[TEST] DataLoader batch shapes...")

    config = MultiResolutionConfig(
        daily_seq_len=180,
        monthly_seq_len=6,
        prediction_horizon=1,
    )

    try:
        train_loader, val_loader, _, norm_stats = create_multi_resolution_dataloaders(
            config=config,
            batch_size=2,
            num_workers=0,
        )

        for batch in train_loader:
            # Check daily features
            for source_name, tensor in batch['daily_features'].items():
                assert tensor.dim() == 3, f"Expected 3D tensor for daily {source_name}"
                assert tensor.size(0) == batch['batch_size'], "Batch size mismatch"
                print(f"  Daily {source_name}: {tensor.shape}")

            # Check monthly features
            for source_name, tensor in batch['monthly_features'].items():
                assert tensor.dim() == 3, f"Expected 3D tensor for monthly {source_name}"
                print(f"  Monthly {source_name}: {tensor.shape}")

            # Check month boundaries
            assert batch['month_boundary_indices'].dim() == 3
            assert batch['month_boundary_indices'].size(-1) == 2
            print(f"  Month boundaries: {batch['month_boundary_indices'].shape}")

            print("[PASS] DataLoader batch shapes correct")
            break

    except Exception as e:
        print(f"[FAIL] DataLoader test failed: {e}")
        raise


def test_collate_variable_lengths():
    """Test that collate function handles variable-length sequences."""
    print("\n[TEST] Collate function with variable lengths...")

    # Create mock samples with different lengths
    class MockSample:
        def __init__(self, daily_len, monthly_len):
            self.daily_features = {
                'test': torch.randn(daily_len, 10)
            }
            self.daily_masks = {
                'test': torch.ones(daily_len, 10, dtype=torch.bool)
            }
            self.monthly_features = {
                'test': torch.randn(monthly_len, 5)
            }
            self.monthly_masks = {
                'test': torch.ones(monthly_len, 5, dtype=torch.bool)
            }
            self.month_boundary_indices = torch.zeros(monthly_len, 2, dtype=torch.long)
            self.daily_dates = np.array([])
            self.monthly_dates = np.array([])
            self.sample_idx = 0
            self.forecast_targets = None
            self.forecast_masks = None
            self.isw_embedding = None

    try:
        batch = [
            MockSample(100, 5),
            MockSample(80, 6),
            MockSample(120, 4),
        ]

        collated = enhanced_multi_resolution_collate_fn(batch)

        # Check padding
        max_daily = max(100, 80, 120)
        max_monthly = max(5, 6, 4)

        assert collated['daily_features']['test'].size(1) == max_daily
        assert collated['monthly_features']['test'].size(1) == max_monthly

        print(f"  Max daily length: {max_daily}")
        print(f"  Max monthly length: {max_monthly}")
        print("[PASS] Variable length handling correct")

    except Exception as e:
        print(f"[FAIL] Collate test failed: {e}")
        raise


def test_multi_task_loss():
    """Test that multi-task loss computes correctly."""
    print("\n[TEST] Multi-task loss computation...")

    try:
        loss_fn = MultiTaskLoss(task_names=['task_a', 'task_b', 'task_c'])

        losses = {
            'task_a': torch.tensor(1.0),
            'task_b': torch.tensor(2.0),
            'task_c': torch.tensor(0.5),
        }

        total_loss, weights = loss_fn(losses)

        assert not torch.isnan(total_loss), "Loss is NaN"
        assert total_loss.item() > 0, "Loss should be positive"
        assert len(weights) == 3, "Should have 3 task weights"

        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Task weights: {weights}")

        # Test gradients flow
        total_loss.backward()
        for name, param in loss_fn.log_vars.items():
            assert param.grad is not None, f"No gradient for {name}"

        print("[PASS] Multi-task loss computation correct")

    except Exception as e:
        print(f"[FAIL] Multi-task loss test failed: {e}")
        raise


def test_gradient_accumulation():
    """Test that gradient accumulation works correctly."""
    print("\n[TEST] Gradient accumulation...")

    try:
        model = nn.Linear(10, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        accumulator = GradientAccumulator(optimizer, accumulation_steps=4)

        # Simulate 4 steps
        for i in range(4):
            x = torch.randn(5, 10)
            y = model(x).sum()
            y.backward()
            stepped = accumulator.step(y, model)

            if i < 3:
                assert not stepped, f"Should not step at iteration {i}"
            else:
                assert stepped, "Should step at iteration 3"

        print(f"  Accumulation steps: {accumulator.accumulation_steps}")
        print(f"  Effective batch size: {accumulator.get_effective_batch_size(8)}")
        print("[PASS] Gradient accumulation correct")

    except Exception as e:
        print(f"[FAIL] Gradient accumulation test failed: {e}")
        raise


def test_checkpoint_save_load():
    """Test that checkpointing saves and loads correctly."""
    print("\n[TEST] Checkpoint save/load...")

    import tempfile

    try:
        # Create minimal model
        model = nn.Linear(10, 2)
        optimizer = optim.Adam(model.parameters())

        # Save
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)

            # Load
            loaded = torch.load(f.name)

            assert loaded['epoch'] == 5
            assert 'model_state_dict' in loaded
            assert 'optimizer_state_dict' in loaded

        print("[PASS] Checkpoint save/load correct")

    except Exception as e:
        print(f"[FAIL] Checkpoint test failed: {e}")
        raise


def test_checkpoint_selection_utilities():
    """Test checkpoint selection utilities for fusion experiments."""
    print("\n[TEST] Checkpoint selection utilities...")

    import tempfile
    import shutil

    try:
        # Create temporary checkpoint directory
        temp_dir = Path(tempfile.mkdtemp())

        # Create mock checkpoints
        mock_checkpoint = {
            'epoch': 10,
            'model_state_dict': {'weight': torch.randn(10, 10)},
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'multi_task_loss_state_dict': {},
            'best_val_loss': 0.5,
            'train_history': {},
            'val_history': {},
            'config': {},
        }

        # Save epoch 10 checkpoint
        torch.save(mock_checkpoint, temp_dir / 'checkpoint_epoch_10.pt')

        # Save epoch 20 checkpoint
        mock_checkpoint['epoch'] = 20
        torch.save(mock_checkpoint, temp_dir / 'checkpoint_epoch_20.pt')

        # Save early fusion checkpoint
        mock_checkpoint['epoch'] = 10
        mock_checkpoint['fusion_checkpoint_note'] = 'Test fusion checkpoint'
        torch.save(mock_checkpoint, temp_dir / 'early_fusion_checkpoint.pt')

        # Save best checkpoint
        mock_checkpoint['epoch'] = 50
        torch.save(mock_checkpoint, temp_dir / 'best_checkpoint.pt')

        # Test load_checkpoint_by_epoch
        loaded = load_checkpoint_by_epoch(10, checkpoint_dir=temp_dir)
        assert loaded['epoch'] == 10, f"Expected epoch 10, got {loaded['epoch']}"
        print("  load_checkpoint_by_epoch(10): OK")

        # Test load_checkpoint_by_epoch with non-existent epoch
        try:
            load_checkpoint_by_epoch(99, checkpoint_dir=temp_dir)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "99" in str(e)
            print("  load_checkpoint_by_epoch(99) raises FileNotFoundError: OK")

        # Test load_best_checkpoint
        loaded = load_best_checkpoint(checkpoint_dir=temp_dir)
        assert loaded['epoch'] == 50, f"Expected epoch 50, got {loaded['epoch']}"
        print("  load_best_checkpoint(): OK")

        # Test load_fusion_checkpoint (should prefer early_fusion_checkpoint.pt)
        loaded = load_fusion_checkpoint(checkpoint_dir=temp_dir)
        assert 'fusion_checkpoint_note' in loaded, "Should load fusion checkpoint"
        print("  load_fusion_checkpoint(): OK")

        # Test list_available_checkpoints
        info = list_available_checkpoints(checkpoint_dir=temp_dir)
        assert 10 in info['epochs'], "Should list epoch 10"
        assert 20 in info['epochs'], "Should list epoch 20"
        assert info['has_best'], "Should have best checkpoint"
        assert info['has_fusion'], "Should have fusion checkpoint"
        assert info['recommended_fusion_epoch'] == OPTIMAL_FUSION_EPOCH
        print(f"  list_available_checkpoints(): {info}")

        # Cleanup
        shutil.rmtree(temp_dir)

        print("[PASS] Checkpoint selection utilities correct")

    except Exception as e:
        print(f"[FAIL] Checkpoint selection test failed: {e}")
        # Cleanup on failure
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("RUNNING TESTS FOR train_multi_resolution.py")
    print("=" * 80)

    test_collate_variable_lengths()
    test_multi_task_loss()
    test_gradient_accumulation()
    test_checkpoint_save_load()
    test_checkpoint_selection_utilities()

    # DataLoader test requires actual data
    try:
        test_dataloader_batch_shapes()
    except Exception as e:
        print(f"[SKIP] DataLoader test skipped (data not available): {e}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main training script for MultiResolutionHAN.

    Usage:
        python train_multi_resolution.py --batch_size 8 --epochs 200
    """
    parser = argparse.ArgumentParser(
        description='Train Multi-Resolution Hierarchical Attention Network'
    )

    # Data arguments
    parser.add_argument('--daily_seq_len', type=int, default=365,
                        help='Daily sequence length (default: 365)')
    parser.add_argument('--monthly_seq_len', type=int, default=12,
                        help='Monthly sequence length (default: 12)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Prediction horizon in months (default: 1)')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model hidden dimension (default: 128)')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num_daily_layers', type=int, default=3,
                        help='Daily encoder layers (default: 3)')
    parser.add_argument('--num_monthly_layers', type=int, default=2,
                        help='Monthly encoder layers (default: 2)')
    parser.add_argument('--num_fusion_layers', type=int, default=2,
                        help='Fusion layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs (default: 10)')

    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device (default: auto)')

    # Memory optimization arguments
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision (bf16/fp16) for faster training (default: True)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing to reduce memory usage (default: True)')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                        help='Disable gradient checkpointing')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(CHECKPOINT_DIR),
                        help='Checkpoint directory')

    # Data configuration arguments (optimization-implementation-plan.md)
    parser.add_argument('--use-disaggregated-equipment', action='store_true', default=True,
                        help='Use disaggregated equipment sources (drones/armor/artillery) instead of aggregated (default: True)')
    parser.add_argument('--no-disaggregated-equipment', action='store_true',
                        help='Use aggregated equipment source instead of disaggregated')
    parser.add_argument('--detrend-viirs', action='store_true', default=True,
                        help='Apply first-order differencing to VIIRS to remove trend (default: True)')
    parser.add_argument('--no-detrend-viirs', action='store_true',
                        help='Disable VIIRS detrending')

    # Detrending configuration (temporal-deconfounding-plan.md)
    parser.add_argument('--apply-detrending', action='store_true', default=False,
                        help='Apply detrending to remove slow trends while preserving daily fluctuations (default: False)')
    parser.add_argument('--no-apply-detrending', action='store_true',
                        help='Disable detrending')
    parser.add_argument('--detrending-window', type=int, default=14,
                        help='Rolling window size for detrending in days (default: 14)')

    # Temporal regularization (temporal-deconfounding-plan.md)
    parser.add_argument('--use-temporal-reg', action='store_true', default=False,
                        help='Enable temporal regularization to prevent learning time shortcuts (default: False)')
    parser.add_argument('--no-temporal-reg', action='store_true',
                        help='Disable temporal regularization')
    parser.add_argument('--temporal-corr-weight', type=float, default=0.01,
                        help='Weight for correlation penalty in temporal regularization (default: 0.01)')
    parser.add_argument('--temporal-smooth-weight', type=float, default=0.001,
                        help='Weight for smoothness penalty in temporal regularization (default: 0.001)')

    # ISW alignment
    parser.add_argument('--use-isw-alignment', action='store_true', default=False,
                        help='Enable ISW narrative alignment via contrastive learning (default: False)')
    parser.add_argument('--isw-weight', type=float, default=0.1,
                        help='Weight for ISW alignment loss (default: 0.1)')

    # Spatial features (unit positions, frontlines, fire hotspots per region)
    parser.add_argument('--include-spatial', action='store_true', default=False,
                        help='Include spatial features from DeepState (units, frontlines) and FIRMS (fire hotspots)')
    parser.add_argument('--start-date', type=str, default='2022-02-24',
                        help='Start date for training data (default: 2022-02-24, use 2022-05-15 for spatial to avoid missing data)')

    # Geographic prior arguments for raion sources
    parser.add_argument('--use-geographic-prior', action='store_true', default=False,
                        help='Enable geographic attention priors for spatial/raion sources')
    parser.add_argument('--raion-sources', type=str, nargs='*', default=None,
                        help='List of raion sources to use (e.g., geoconfirmed_raion ucdp_raion). '
                             'Available: geoconfirmed_raion, air_raid_sirens_raion, ucdp_raion, '
                             'warspotting_raion, deepstate_raion, firms_expanded_raion, combined_raion')
    parser.add_argument('--all-raion-sources', action='store_true', default=False,
                        help='Use all available raion sources (convenience flag)')
    parser.add_argument('--raion-only', action='store_true', default=False,
                        help='Use ONLY raion sources (exclude default non-raion sources like equipment, viina, etc.)')

    # Other arguments
    parser.add_argument('--test', action='store_true',
                        help='Run tests instead of training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Run management (probe integration)
    parser.add_argument('--run-id', type=str, default=None,
                        help='Custom run ID for tracking (auto-generated if not set)')
    parser.add_argument('--no-run-manager', action='store_true',
                        help='Disable run manager (skip organized output directory)')

    args = parser.parse_args()

    # Run tests if requested
    if args.test:
        run_all_tests()
        return

    # Resolve data configuration flags (handle --no-* overrides)
    use_disaggregated_equipment = args.use_disaggregated_equipment and not args.no_disaggregated_equipment
    detrend_viirs = args.detrend_viirs and not args.no_detrend_viirs
    apply_detrending = args.apply_detrending and not args.no_apply_detrending
    use_temporal_reg = args.use_temporal_reg and not args.no_temporal_reg

    # Resolve memory optimization flags
    use_amp = args.use_amp and not args.no_amp
    gradient_checkpointing = args.gradient_checkpointing and not args.no_gradient_checkpointing

    print("=" * 80)
    print("MULTI-RESOLUTION HAN TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Daily sequence length: {args.daily_seq_len}")
    print(f"  Monthly sequence length: {args.monthly_seq_len}")
    print(f"  Prediction horizon: {args.prediction_horizon}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    # Resolve raion source configuration
    use_geographic_prior = args.use_geographic_prior
    raion_sources: List[str] = []
    if args.all_raion_sources:
        raion_sources = ALL_RAION_SOURCES.copy()
        use_geographic_prior = True  # Implicitly enable geographic prior
    elif args.raion_sources:
        raion_sources = args.raion_sources
        use_geographic_prior = True  # Implicitly enable geographic prior

    print(f"\nData Configuration (Optimization Plan):")
    print(f"  use_disaggregated_equipment: {use_disaggregated_equipment}")
    print(f"  detrend_viirs: {detrend_viirs}")
    print(f"  include_spatial_features: {args.include_spatial}")
    print(f"  start_date: {args.start_date}")
    print(f"\nGeographic Prior Configuration:")
    print(f"  use_geographic_prior: {use_geographic_prior}")
    if raion_sources:
        print(f"  raion_sources: {raion_sources}")
    else:
        print(f"  raion_sources: (none - geographic prior disabled)")

    # Create datasets
    print("\n" + "-" * 80)
    print("Creating datasets...")

    # Build daily_sources list
    # CLEANED CONFIGURATION (2026-01):
    # - All features are delta-only (no cumulative to avoid spurious correlations)
    # - Removed redundant sources: viina (→geoconfirmed_raion), firms (→firms_expanded_raion)
    # - equipment split into drones/armor/artillery (aircraft excluded)
    # - viirs excluded (lagging indicator, not predictive)

    # Non-spatial sources (45 features total, delta-only)
    non_spatial_sources = ["personnel", "drones", "armor", "artillery"]

    # Raion sources (29,841 features total, with geographic prior)
    default_raion_sources = [
        "geoconfirmed_raion", "air_raid_sirens_raion", "ucdp_raion",
        "warspotting_raion", "deepstate_raion", "firms_expanded_raion",
    ]

    # Combine sources based on mode
    if raion_sources:
        # User specified explicit raion sources
        if args.raion_only:
            # Use ONLY raion sources (pure geographic model)
            # Still include personnel as it's essential for target prediction
            daily_sources_combined = ["personnel"] + raion_sources
            print(f"\nDaily sources (raion-only mode): {daily_sources_combined}")
        else:
            # Add specified raion sources to non-spatial sources
            daily_sources_combined = non_spatial_sources + raion_sources
            print(f"\nDaily sources (combined): {daily_sources_combined}")
    elif args.all_raion_sources:
        # Use all default raion sources
        if args.raion_only:
            daily_sources_combined = ["personnel"] + default_raion_sources
            print(f"\nDaily sources (raion-only mode): {daily_sources_combined}")
        else:
            daily_sources_combined = non_spatial_sources + default_raion_sources
            print(f"\nDaily sources (full configuration): {daily_sources_combined}")
    else:
        # No raion sources - use only non-spatial sources (baseline)
        daily_sources_combined = non_spatial_sources
        print(f"\nDaily sources (non-spatial baseline): {daily_sources_combined}")

    config = MultiResolutionConfig(
        daily_sources=daily_sources_combined,
        daily_seq_len=args.daily_seq_len,
        monthly_seq_len=args.monthly_seq_len,
        prediction_horizon=args.prediction_horizon,
        use_disaggregated_equipment=use_disaggregated_equipment,
        detrend_viirs=detrend_viirs,
        include_spatial_features=args.include_spatial,
        start_date=args.start_date,
        apply_detrending=apply_detrending,
        detrending_window=args.detrending_window,
    )

    train_dataset = MultiResolutionDataset(config=config, split='train')
    val_dataset = MultiResolutionDataset(
        config=config, split='val', norm_stats=train_dataset.norm_stats
    )
    test_dataset = MultiResolutionDataset(
        config=config, split='test', norm_stats=train_dataset.norm_stats
    )

    # Get feature dimensions from dataset and create SourceConfig objects
    feature_info = train_dataset.get_feature_info()

    daily_source_configs = {
        name: SourceConfig(
            name=name,
            n_features=info['n_features'],
            resolution='daily',
            description=f'{name} daily source'
        )
        for name, info in feature_info.items()
        if info['resolution'] == 'daily'
    }
    monthly_source_configs = {
        name: SourceConfig(
            name=name,
            n_features=info['n_features'],
            resolution='monthly',
            description=f'{name} monthly source'
        )
        for name, info in feature_info.items()
        if info['resolution'] == 'monthly'
    }

    print(f"\nDaily sources: {[f'{k}({v.n_features})' for k, v in daily_source_configs.items()]}")
    print(f"Monthly sources: {[f'{k}({v.n_features})' for k, v in monthly_source_configs.items()]}")

    # Create model
    print("\n" + "-" * 80)
    print("Creating model...")

    # Check for ISW alignment flag
    use_isw_alignment = getattr(args, 'use_isw_alignment', False)

    # Auto-detect ISW from checkpoint if resuming
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"\nDetecting model configuration from checkpoint: {args.resume}")
            checkpoint_peek = torch.load(args.resume, map_location='cpu', weights_only=False)
            checkpoint_keys = checkpoint_peek.get('model_state_dict', {}).keys()
            has_isw_keys = any('isw_alignment' in k for k in checkpoint_keys)

            if has_isw_keys and not use_isw_alignment:
                print(f"  Checkpoint has ISW alignment weights - enabling ISW alignment automatically")
                use_isw_alignment = True
            elif not has_isw_keys and use_isw_alignment:
                print(f"  WARNING: --use-isw-alignment specified but checkpoint has no ISW weights")
                print(f"           ISW module will be randomly initialized")

            # Also detect other checkpoint metadata if available
            if 'epoch' in checkpoint_peek:
                print(f"  Checkpoint epoch: {checkpoint_peek['epoch']}")
            if 'best_val_loss' in checkpoint_peek:
                print(f"  Checkpoint best_val_loss: {checkpoint_peek['best_val_loss']:.4f}")

            del checkpoint_peek  # Free memory

    # Build spatial configs for raion sources with geographic priors
    custom_spatial_configs: Optional[Dict[str, SpatialSourceConfig]] = None
    if use_geographic_prior and raion_sources:
        print("\nBuilding spatial configs for raion sources...")
        custom_spatial_configs = build_spatial_configs_from_dataset(raion_sources)
        if custom_spatial_configs:
            print(f"  Spatial configs created for {len(custom_spatial_configs)} sources:")
            for name, config in custom_spatial_configs.items():
                print(f"    - {name}: {config.n_raions} raions, "
                      f"{config.features_per_raion} features/raion, "
                      f"geographic_prior={config.use_geographic_prior}")
        else:
            print("  WARNING: No spatial configs could be created (data may not be loaded yet)")
            print("           Geographic prior will be disabled for this run")
            use_geographic_prior = False

    model = MultiResolutionHAN(
        daily_source_configs=daily_source_configs,
        monthly_source_configs=monthly_source_configs,
        d_model=args.d_model,
        nhead=args.nhead,
        num_daily_layers=args.num_daily_layers,
        num_monthly_layers=args.num_monthly_layers,
        num_fusion_layers=args.num_fusion_layers,
        dropout=args.dropout,
        use_isw_alignment=use_isw_alignment,
        isw_dim=1024,  # Voyage embedding dimension
        use_geographic_prior=use_geographic_prior,
        custom_spatial_configs=custom_spatial_configs,
    )

    if use_isw_alignment:
        print(f"ISW alignment enabled (1024-dim embeddings)")
    if use_geographic_prior:
        print(f"Geographic prior enabled for {len(custom_spatial_configs or {})} raion sources")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create run manager for organized output (probe integration)
    run_manager = None
    if not args.no_run_manager:
        run_manager = TrainingRunManager(run_id=args.run_id)
        run_manager.setup()

        # Save initial metadata
        run_manager.update_metadata(
            d_model=args.d_model,
            use_multi_resolution=True,
            detrend_viirs=detrend_viirs,
            use_disaggregated_equipment=use_disaggregated_equipment,
            effective_daily_sources=list(daily_source_configs.keys()),
            effective_monthly_sources=list(monthly_source_configs.keys()),
            n_daily_sources=len(daily_source_configs),
            n_monthly_sources=len(monthly_source_configs),
            daily_seq_len=args.daily_seq_len,
            monthly_seq_len=args.monthly_seq_len,
            n_train_samples=len(train_dataset),
            n_val_samples=len(val_dataset),
            n_test_samples=len(test_dataset),
            han_n_params=n_params,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            use_geographic_prior=use_geographic_prior,
            raion_sources=raion_sources if raion_sources else None,
            n_raion_sources=len(custom_spatial_configs) if custom_spatial_configs else 0,
        )

        # Save full training config
        training_config = {
            'daily_seq_len': args.daily_seq_len,
            'monthly_seq_len': args.monthly_seq_len,
            'prediction_horizon': args.prediction_horizon,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_daily_layers': args.num_daily_layers,
            'num_monthly_layers': args.num_monthly_layers,
            'num_fusion_layers': args.num_fusion_layers,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'accumulation_steps': args.accumulation_steps,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'patience': args.patience,
            'warmup_epochs': args.warmup_epochs,
            'use_disaggregated_equipment': use_disaggregated_equipment,
            'detrend_viirs': detrend_viirs,
            'use_isw_alignment': use_isw_alignment,
            'isw_weight': args.isw_weight if use_isw_alignment else None,
            'use_geographic_prior': use_geographic_prior,
            'raion_sources': raion_sources if raion_sources else None,
            'raion_spatial_configs': {
                name: {
                    'n_raions': cfg.n_raions,
                    'features_per_raion': cfg.features_per_raion,
                    'use_geographic_prior': cfg.use_geographic_prior,
                }
                for name, cfg in (custom_spatial_configs or {}).items()
            } if custom_spatial_configs else None,
        }
        run_manager.save_config(training_config)

        print(f"\nRun management enabled:")
        print(f"  Run ID: {run_manager.run_id}")
        print(f"  Run directory: {run_manager.run_dir}")

    # Create trainer
    trainer = MultiResolutionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        run_manager=run_manager,
        use_amp=use_amp,
        gradient_checkpointing=gradient_checkpointing,
        use_temporal_reg=use_temporal_reg,
        temporal_corr_weight=args.temporal_corr_weight,
        temporal_smooth_weight=args.temporal_smooth_weight,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "-" * 80)
    history = trainer.train(verbose=True)

    # Final evaluation
    print("\n" + "-" * 80)
    print("Final Evaluation")
    print("-" * 80)

    # Load best checkpoint
    best_checkpoint = Path(args.checkpoint_dir) / 'best_checkpoint.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))

    test_metrics = trainer.evaluate(test_dataset)

    print("\nTest Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save training summary
    summary = {
        'config': vars(args),
        'history': history,
        'test_metrics': test_metrics,
        'n_params': n_params,
    }

    summary_path = Path(args.checkpoint_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTraining summary saved to {summary_path}")

    # Finalize run manager with training results
    if run_manager is not None:
        # Update metadata with final results
        run_manager.update_metadata(
            han_best_epoch=trainer.best_epoch if hasattr(trainer, 'best_epoch') else 0,
            han_best_val_loss=trainer.best_val_loss,
            stage3_complete=True,
        )

        # Mark stage 3 complete with metrics
        run_manager.mark_stage_complete(
            stage=3,
            duration=history.get('total_time', 0) if isinstance(history, dict) else 0,
            metrics={
                'best_val_loss': trainer.best_val_loss,
                'test_metrics': test_metrics,
                'n_epochs_trained': len(history.get('train_loss', [])) if isinstance(history, dict) else 0,
            }
        )

        # Save summary to run directory
        run_summary_path = run_manager.get_stage_dir(3) / 'training_summary.json'
        with open(run_summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        run_manager.finalize()

        print(f"\nRun finalized:")
        print(f"  Run ID: {run_manager.run_id}")
        print(f"  Run directory: {run_manager.run_dir}")
        print(f"  Use this run ID with probes: --training-run-id {run_manager.run_id}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
