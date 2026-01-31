"""
Integration module for training improvements.

This module provides drop-in integration of the training improvements
into the existing MultiResolutionTrainer without major refactoring.

Usage:
    from training_improvements_integration import apply_training_improvements

    # In train_multi_resolution.py, after creating trainer:
    trainer = MultiResolutionTrainer(...)
    apply_training_improvements(trainer, config)

Or use the improved trainer directly:
    from training_improvements_integration import ImprovedMultiResolutionTrainer

    trainer = ImprovedMultiResolutionTrainer(...)

Author: Claude (synthesized from multiple AI proposals)
Date: 2026-01-31
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from .training_improvements import (
        LatentStatePredictor,
        LowRankFeatureDecoder,
        PCGradSurgery,
        SoftplusKendallLoss,
        AvailabilityGatedLoss,
        CrossResolutionCycleConsistency,
        LatentPredictiveCodingLoss,
        PhysicalConsistencyConstraint,
        ImprovedTrainingConfig,
    )
except ImportError:
    from training_improvements import (
        LatentStatePredictor,
        LowRankFeatureDecoder,
        PCGradSurgery,
        SoftplusKendallLoss,
        AvailabilityGatedLoss,
        CrossResolutionCycleConsistency,
        LatentPredictiveCodingLoss,
        PhysicalConsistencyConstraint,
        ImprovedTrainingConfig,
    )


# =============================================================================
# IMPROVED FORECAST HEAD REPLACEMENT
# =============================================================================

class ImprovedForecastModule(nn.Module):
    """
    Replacement for DailyForecastingHead that uses latent prediction.

    Instead of predicting raw high-dimensional features (causing memorization),
    this module:
    1. Predicts future latent states using LatentStatePredictor
    2. Optionally decodes to raw features using LowRankFeatureDecoder
    3. Computes predictive coding loss against teacher latents

    Args:
        d_model: Latent dimension
        horizon: Number of future timesteps (default 7 for daily)
        source_dims: Dict of source name -> feature dimension (for optional decoding)
        decoder_rank: Rank for low-rank decoder (None to skip decoding)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 7,
        source_dims: Optional[Dict[str, int]] = None,
        decoder_rank: Optional[int] = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.horizon = horizon
        self.source_dims = source_dims

        # Latent state predictor (core improvement)
        self.latent_predictor = LatentStatePredictor(
            d_model=d_model,
            horizon=horizon,
            hidden_dim=d_model * 2,
            dropout=dropout,
        )

        # Optional low-rank decoder for raw feature output
        if source_dims and decoder_rank:
            self.feature_decoder = LowRankFeatureDecoder(
                d_model=d_model,
                source_dims=source_dims,
                rank=decoder_rank,
            )
        else:
            self.feature_decoder = None

        # Track total output dimension for compatibility
        if source_dims:
            self.output_dim = sum(source_dims.values())
        else:
            self.output_dim = d_model

    def forward(
        self,
        context_repr: Tensor,
        return_latents: bool = False,
    ) -> Tensor:
        """
        Predict future states.

        Args:
            context_repr: Context representation [batch, seq_len, d_model]
            return_latents: If True, return latent predictions instead of features

        Returns:
            If return_latents: [batch, horizon, d_model]
            Else: [batch, horizon, output_dim] (raw features)
        """
        # Use last timestep as context
        if context_repr.dim() == 3:
            context_latent = context_repr[:, -1, :]  # [batch, d_model]
        else:
            context_latent = context_repr  # Already [batch, d_model]

        # Predict future latent states
        pred_latents = self.latent_predictor(context_latent)  # [batch, horizon, d_model]

        if return_latents or self.feature_decoder is None:
            return pred_latents

        # Decode to raw features
        batch_size, horizon, _ = pred_latents.shape
        decoded_features = []

        for t in range(horizon):
            t_latent = pred_latents[:, t, :]
            t_decoded = self.feature_decoder(t_latent)

            # Concatenate all source features
            features = torch.cat(list(t_decoded.values()), dim=-1)
            decoded_features.append(features)

        return torch.stack(decoded_features, dim=1)  # [batch, horizon, output_dim]

    def get_latent_predictions(self, context_repr: Tensor) -> Tensor:
        """Get latent predictions explicitly."""
        return self.forward(context_repr, return_latents=True)


# =============================================================================
# IMPROVED TRAINING STEP
# =============================================================================

class ImprovedTrainingStep:
    """
    Improved training step with PCGrad, availability gating, and consistency losses.

    This class wraps the training step logic and can be used as a mixin
    or standalone component.
    """

    def __init__(
        self,
        config: ImprovedTrainingConfig,
        task_names: List[str],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.task_names = task_names

        # Initialize PCGrad surgery
        if config.use_pcgrad:
            self.pcgrad = PCGradSurgery(task_groups=config.pcgrad_task_groups)
        else:
            self.pcgrad = None

        # Initialize improved loss function
        if config.use_availability_gating:
            base_loss = SoftplusKendallLoss(task_names) if config.use_softplus_kendall else None
            self.loss_fn = AvailabilityGatedLoss(
                task_names=task_names,
                min_availability=config.min_availability,
                base_loss=base_loss,
            ).to(device)
        elif config.use_softplus_kendall:
            self.loss_fn = SoftplusKendallLoss(task_names).to(device)
        else:
            self.loss_fn = None  # Use default

        # Initialize consistency losses
        if config.use_cycle_consistency:
            self.cycle_consistency = CrossResolutionCycleConsistency(
                weight=config.cycle_consistency_weight,
            ).to(device)
        else:
            self.cycle_consistency = None

        if config.use_physical_consistency:
            self.physical_consistency = PhysicalConsistencyConstraint(
                weight=config.physical_consistency_weight,
            ).to(device)
        else:
            self.physical_consistency = None

        # Predictive coding loss
        if config.use_latent_prediction:
            self.pc_loss = LatentPredictiveCodingLoss().to(device)
        else:
            self.pc_loss = None

    def compute_combined_loss(
        self,
        task_losses: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined loss with availability gating.

        Args:
            task_losses: Dict of task name -> loss tensor
            targets: Dict of task targets (for availability computation)
            masks: Optional availability masks

        Returns:
            Tuple of (total_loss, task_weights)
        """
        if isinstance(self.loss_fn, AvailabilityGatedLoss):
            total, weights, avail = self.loss_fn(task_losses, targets, masks)
            return total, weights
        elif self.loss_fn is not None:
            return self.loss_fn(task_losses, masks)
        else:
            # Fallback: simple sum
            total = sum(task_losses.values())
            weights = {k: 1.0 for k in task_losses}
            return total, weights

    def add_consistency_losses(
        self,
        losses: Dict[str, Tensor],
        daily_pred_latents: Optional[Tensor] = None,
        monthly_teacher_latent: Optional[Tensor] = None,
        daily_pred_features: Optional[Tensor] = None,
        monthly_pred_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Add consistency regularization losses.

        Args:
            losses: Current task losses dict (modified in place)
            daily_pred_latents: Predicted daily latents [batch, horizon, d_model]
            monthly_teacher_latent: Teacher monthly latent [batch, d_model]
            daily_pred_features: Daily feature predictions [batch, horizon, n_feat]
            monthly_pred_features: Monthly feature predictions [batch, n_feat]

        Returns:
            Updated losses dict
        """
        # Cycle consistency: daily latents -> aggregated should match monthly
        if (self.cycle_consistency is not None
                and daily_pred_latents is not None
                and monthly_teacher_latent is not None):
            cycle_loss = self.cycle_consistency(
                daily_pred_latents,
                monthly_teacher_latent.detach(),  # Stop gradient on teacher
            )
            losses['cycle_consistency'] = cycle_loss

        # Physical consistency: daily sum ≈ monthly * (7/30)
        if (self.physical_consistency is not None
                and daily_pred_features is not None
                and monthly_pred_features is not None):
            phys_loss = self.physical_consistency(
                daily_pred_features,
                monthly_pred_features.detach(),
            )
            losses['physical_consistency'] = phys_loss

        return losses

    def backward_with_pcgrad(
        self,
        model: nn.Module,
        task_losses: Dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Perform backward pass with PCGrad surgery.

        Args:
            model: The model
            task_losses: Dict of task losses
            optimizer: The optimizer
        """
        if self.pcgrad is None:
            # Standard backward
            total_loss = sum(task_losses.values())
            optimizer.zero_grad()
            total_loss.backward()
        else:
            # PCGrad surgery
            self.pcgrad.backward_with_surgery(model, task_losses, optimizer)


# =============================================================================
# PATCH FUNCTION FOR EXISTING TRAINER
# =============================================================================

def apply_training_improvements(
    trainer,
    config: Optional[ImprovedTrainingConfig] = None,
) -> None:
    """
    Apply training improvements to an existing MultiResolutionTrainer.

    This function patches the trainer in-place to use:
    - SoftplusKendallLoss (fixes negative loss)
    - AvailabilityGatedLoss (prevents task collapse)
    - PCGradSurgery (prevents gradient conflicts)
    - Consistency regularization

    Args:
        trainer: MultiResolutionTrainer instance
        config: Optional configuration (uses defaults if not provided)
    """
    config = config or ImprovedTrainingConfig()

    # Get task names from existing loss function
    if hasattr(trainer, 'multi_task_loss') and hasattr(trainer.multi_task_loss, 'task_names'):
        task_names = trainer.multi_task_loss.task_names
    else:
        task_names = ['regime', 'casualty', 'anomaly', 'forecast', 'daily_forecast', 'transition']

    # Create improved training step handler
    improved_step = ImprovedTrainingStep(
        config=config,
        task_names=task_names,
        device=trainer.device,
    )

    # Store on trainer for access
    trainer._improved_step = improved_step
    trainer._improvement_config = config

    # Patch the multi-task loss if using softplus
    if config.use_softplus_kendall and improved_step.loss_fn is not None:
        trainer._original_multi_task_loss = trainer.multi_task_loss
        trainer.multi_task_loss = improved_step.loss_fn
        print("✓ Applied SoftplusKendallLoss (fixes negative loss pathology)")

    # Store PCGrad handler
    if config.use_pcgrad:
        trainer._pcgrad = improved_step.pcgrad
        print("✓ Applied PCGrad surgery (prevents gradient conflicts)")

    # Store consistency loss modules
    if config.use_cycle_consistency:
        trainer._cycle_consistency = improved_step.cycle_consistency
        print("✓ Added cycle consistency loss")

    if config.use_physical_consistency:
        trainer._physical_consistency = improved_step.physical_consistency
        print("✓ Added physical consistency loss")

    # Note about latent prediction (requires model changes)
    if config.use_latent_prediction:
        print("⚠ Latent prediction enabled - requires ImprovedForecastModule in model")


def create_improved_forecast_head(
    d_model: int,
    daily_source_configs: Dict[str, Any],
    horizon: int = 7,
    decoder_rank: int = 32,
    dropout: float = 0.1,
) -> ImprovedForecastModule:
    """
    Factory function to create an improved forecast head.

    Args:
        d_model: Model dimension
        daily_source_configs: Dict of source name -> SourceConfig
        horizon: Forecast horizon
        decoder_rank: Rank for low-rank decoder
        dropout: Dropout probability

    Returns:
        ImprovedForecastModule instance
    """
    # Extract feature dimensions from configs
    source_dims = {
        name: cfg.n_features
        for name, cfg in daily_source_configs.items()
    }

    return ImprovedForecastModule(
        d_model=d_model,
        horizon=horizon,
        source_dims=source_dims,
        decoder_rank=decoder_rank,
        dropout=dropout,
    )


# =============================================================================
# TRAINING SCRIPT MODIFICATIONS GUIDE
# =============================================================================

INTEGRATION_GUIDE = """
# Training Improvements Integration Guide

## Quick Start (Minimal Changes)

Add these lines to train_multi_resolution.py after creating the trainer:

```python
from analysis.training_improvements_integration import (
    apply_training_improvements,
    ImprovedTrainingConfig,
)

# After: trainer = MultiResolutionTrainer(...)
config = ImprovedTrainingConfig(
    use_pcgrad=True,
    use_softplus_kendall=True,
    use_availability_gating=True,
    min_availability=0.2,
)
apply_training_improvements(trainer, config)
```

## Full Integration (With Latent Prediction)

1. Replace DailyForecastingHead in multi_resolution_han.py:

```python
from analysis.training_improvements_integration import create_improved_forecast_head

# In MultiResolutionHAN.__init__, replace:
# self.daily_forecast_head = DailyForecastingHead(...)

self.daily_forecast_head = create_improved_forecast_head(
    d_model=d_model,
    daily_source_configs=daily_source_configs,
    horizon=7,
    decoder_rank=32,
    dropout=dropout,
)
```

2. Modify _compute_losses to use latent predictive coding:

```python
# In _compute_losses, for daily_forecast task:
if hasattr(self.model.daily_forecast_head, 'get_latent_predictions'):
    # Get student predictions
    pred_latents = self.model.daily_forecast_head.get_latent_predictions(fused_repr)

    # Get teacher latents (encode actual future, stop gradient)
    with torch.no_grad():
        # This requires passing future data through encoders
        # Implementation depends on batch structure
        teacher_latents = self.model.encode_future_context(...)

    # Add predictive coding loss
    pc_loss, _ = self._improved_step.pc_loss(pred_latents, teacher_latents)
    losses['daily_forecast_pc'] = pc_loss
```

## Configuration Options

- `use_pcgrad`: Project conflicting gradients (recommended: True)
- `use_softplus_kendall`: Fix negative loss (recommended: True)
- `use_availability_gating`: Gate tasks by target availability (recommended: True)
- `min_availability`: Minimum target availability to include task (default: 0.2)
- `use_cycle_consistency`: Add daily->monthly consistency loss (optional)
- `use_physical_consistency`: Add physical constraint loss (optional)
- `pcgrad_task_groups`: How to group tasks for gradient surgery
  Default: {'stable': ['regime', 'forecast'], 'noisy': ['casualty', 'daily_forecast']}

## Expected Impact

| Issue | Fix | Expected Result |
|-------|-----|-----------------|
| Forecast 140x overfitting | LatentStatePredictor + PC loss | Val gap < 5x |
| Negative Kendall loss | SoftplusKendallLoss | Always positive loss |
| Task collapse (casualty→0) | AvailabilityGatedLoss | Maintains loss > 0.1 |
| Gradient interference | PCGradSurgery | Stable multi-task learning |
| Daily/monthly divergence | Cycle consistency | Aligned predictions |
"""


def print_integration_guide():
    """Print the integration guide."""
    print(INTEGRATION_GUIDE)


# =============================================================================
# TESTING
# =============================================================================

def test_integration():
    """Test the integration module."""
    print("Testing training improvements integration...")

    # Test ImprovedForecastModule
    print("\n1. Testing ImprovedForecastModule...")
    module = ImprovedForecastModule(
        d_model=128,
        horizon=7,
        source_dims={'equipment': 38, 'personnel': 6},
        decoder_rank=32,
    )
    context = torch.randn(4, 10, 128)
    output = module(context)
    assert output.shape == (4, 7, 44), f"Expected (4, 7, 44), got {output.shape}"
    print(f"   Feature output shape: {output.shape} ✓")

    latent_output = module(context, return_latents=True)
    assert latent_output.shape == (4, 7, 128)
    print(f"   Latent output shape: {latent_output.shape} ✓")

    # Test ImprovedTrainingStep
    print("\n2. Testing ImprovedTrainingStep...")
    config = ImprovedTrainingConfig()
    step = ImprovedTrainingStep(
        config=config,
        task_names=['regime', 'forecast', 'casualty'],
        device=torch.device('cpu'),
    )

    losses = {
        'regime': torch.tensor(0.5),
        'forecast': torch.tensor(1.0),
        'casualty': torch.tensor(0.3),
    }
    targets = {
        'regime': torch.randn(10),
        'forecast': torch.randn(10),
        'casualty': torch.randn(10),
    }
    total, weights = step.compute_combined_loss(losses, targets)
    print(f"   Total loss: {total.item():.4f} ✓")
    print(f"   Weights: {weights}")

    # Test consistency losses
    print("\n3. Testing consistency losses...")
    daily_latents = torch.randn(4, 7, 128)
    monthly_latent = torch.randn(4, 128)

    updated_losses = step.add_consistency_losses(
        losses.copy(),
        daily_pred_latents=daily_latents,
        monthly_teacher_latent=monthly_latent,
    )
    if 'cycle_consistency' in updated_losses:
        print(f"   Cycle consistency loss: {updated_losses['cycle_consistency'].item():.4f} ✓")

    print("\n✓ All integration tests passed!")
    print("\nTo see the integration guide, run: print_integration_guide()")


if __name__ == '__main__':
    test_integration()
