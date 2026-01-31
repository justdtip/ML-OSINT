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
        # New hybrid loss components
        A3DROLoss,
        SpectralDriftPenalty,
        UniformValidationLoss,
        HybridLossConfig,
        create_training_losses,
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
        # New hybrid loss components
        A3DROLoss,
        SpectralDriftPenalty,
        UniformValidationLoss,
        HybridLossConfig,
        create_training_losses,
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
            # AvailabilityGatedLoss returns (total, weights) - targets is optional
            return self.loss_fn(task_losses, targets, masks)
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
# HYBRID LOSS INTEGRATION (A³DRO + Spectral + Uniform Validation)
# =============================================================================

def apply_hybrid_loss_improvements(
    trainer,
    config: Optional[HybridLossConfig] = None,
) -> None:
    """
    Apply hybrid loss improvements to an existing MultiResolutionTrainer.

    This is the RECOMMENDED approach based on the proposal analysis.
    It replaces MultiTaskLoss with:
    - A³DRO for training (no learned weights, robust aggregation)
    - UniformValidationLoss for validation (fixed weights, comparable across epochs)
    - SpectralDriftPenalty for forecast regularization

    Args:
        trainer: MultiResolutionTrainer instance
        config: Optional HybridLossConfig (uses defaults if not provided)
    """
    config = config or HybridLossConfig()

    # Get task names from existing loss function
    if hasattr(trainer, 'multi_task_loss') and hasattr(trainer.multi_task_loss, 'task_names'):
        task_names = trainer.multi_task_loss.task_names
    else:
        task_names = ['regime', 'casualty', 'anomaly', 'forecast', 'daily_forecast', 'transition']

    # Create hybrid loss modules
    loss_modules = create_training_losses(task_names, config)

    # Store original for reference
    trainer._original_multi_task_loss = trainer.multi_task_loss

    # Replace training loss
    trainer.multi_task_loss = loss_modules['training']
    trainer._training_loss = loss_modules['training']
    trainer._hybrid_config = config

    # Add validation loss (separate from training!)
    trainer._validation_loss = loss_modules['validation']

    # Add spectral penalty if enabled
    if 'spectral' in loss_modules:
        trainer._spectral_penalty = loss_modules['spectral']
    else:
        trainer._spectral_penalty = None

    # Add cycle consistency if enabled
    if 'cycle' in loss_modules:
        trainer._cycle_consistency = loss_modules['cycle']
    else:
        trainer._cycle_consistency = None

    # Patch the validate method to use fixed weights
    _patch_validate_method(trainer)

    # Print summary
    if config.use_a3dro:
        print("✓ Applied A³DRO loss (robust aggregation, no learned weights)")
    else:
        print("✓ Applied SoftplusKendallLoss")

    print("✓ Added UniformValidationLoss (fixed weights for comparability)")

    if config.use_spectral_penalty:
        print(f"✓ Added SpectralDriftPenalty (weight={config.spectral_weight})")

    if config.use_cycle_consistency:
        print(f"✓ Added CrossResolutionCycleConsistency (weight={config.cycle_weight})")


def _patch_validate_method(trainer) -> None:
    """
    Patch the trainer's validate method to use UniformValidationLoss.

    This ensures validation loss is comparable across epochs (key audit finding).
    """
    original_validate = trainer.validate

    def patched_validate(*args, **kwargs):
        # Temporarily swap loss functions
        training_loss = trainer.multi_task_loss
        trainer.multi_task_loss = trainer._validation_loss

        try:
            result = original_validate(*args, **kwargs)
        finally:
            # Restore training loss
            trainer.multi_task_loss = training_loss

        return result

    trainer.validate = patched_validate
    trainer._original_validate = original_validate


def get_spectral_loss(trainer, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral drift penalty for forecast regularization.

    Call this in train_epoch after computing task losses:

    ```python
    if hasattr(trainer, '_spectral_penalty') and trainer._spectral_penalty is not None:
        spectral_loss = get_spectral_loss(trainer, daily_pred, daily_targets)
        total_loss = total_loss + spectral_loss
    ```

    Args:
        trainer: The trainer instance
        predictions: Predicted sequence [batch, seq_len, features]
        targets: Target sequence [batch, seq_len, features]

    Returns:
        Scalar spectral loss
    """
    if trainer._spectral_penalty is None:
        return torch.tensor(0.0, device=predictions.device)

    return trainer._spectral_penalty(predictions, targets)


class HybridTrainingStep:
    """
    Improved training step with A³DRO, spectral penalty, and cycle consistency.

    This replaces ImprovedTrainingStep with the hybrid approach.
    """

    def __init__(
        self,
        config: HybridLossConfig,
        task_names: List[str],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.task_names = task_names

        # Create loss modules
        self.loss_modules = create_training_losses(task_names, config)
        self.training_loss = self.loss_modules['training'].to(device)
        self.validation_loss = self.loss_modules['validation'].to(device)

        if 'spectral' in self.loss_modules:
            self.spectral_penalty = self.loss_modules['spectral'].to(device)
        else:
            self.spectral_penalty = None

        if 'cycle' in self.loss_modules:
            self.cycle_consistency = self.loss_modules['cycle'].to(device)
        else:
            self.cycle_consistency = None

    def compute_training_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss using A³DRO or SoftplusKendall.

        Args:
            task_losses: Dict of task name -> loss tensor
            targets: Optional targets for availability computation
            masks: Optional availability masks
            epoch: Current epoch (for A³DRO warmup)

        Returns:
            Tuple of (total_loss, effective_weights)
        """
        if isinstance(self.training_loss, AvailabilityGatedLoss):
            # Need to check if base is A³DRO
            if hasattr(self.training_loss, 'base_loss') and isinstance(self.training_loss.base_loss, A3DROLoss):
                # A³DRO needs epoch for warmup
                # AvailabilityGatedLoss wraps it, so we need to pass epoch differently
                # For now, just use standard forward
                return self.training_loss(task_losses, targets, masks)
            return self.training_loss(task_losses, targets, masks)
        elif isinstance(self.training_loss, A3DROLoss):
            return self.training_loss(task_losses, targets, masks, epoch=epoch)
        else:
            return self.training_loss(task_losses, masks)

    def compute_validation_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute validation loss using fixed uniform weights.

        Args:
            task_losses: Dict of task name -> loss tensor

        Returns:
            Tuple of (total_loss, weights)
        """
        return self.validation_loss(task_losses)

    def compute_spectral_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute spectral drift penalty for forecast sequences.

        Args:
            predictions: Predicted sequence [batch, seq_len, features]
            targets: Target sequence [batch, seq_len, features]
            mask: Optional mask [batch, seq_len]

        Returns:
            Scalar spectral loss
        """
        if self.spectral_penalty is None:
            return torch.tensor(0.0, device=predictions.device)

        return self.spectral_penalty(predictions, targets, mask)

    def compute_cycle_consistency(
        self,
        daily_latents: torch.Tensor,
        monthly_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cycle consistency between daily and monthly predictions.

        Args:
            daily_latents: Predicted daily latents [batch, horizon, d_model]
            monthly_latent: Teacher monthly latent [batch, d_model]

        Returns:
            Scalar cycle consistency loss
        """
        if self.cycle_consistency is None:
            return torch.tensor(0.0, device=daily_latents.device)

        return self.cycle_consistency(daily_latents, monthly_latent.detach())


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

    # Test HybridLossConfig and HybridTrainingStep
    print("\n4. Testing HybridTrainingStep (A³DRO + Spectral)...")
    hybrid_config = HybridLossConfig(
        use_a3dro=True,
        use_spectral_penalty=True,
        use_cycle_consistency=True,
    )
    hybrid_step = HybridTrainingStep(
        config=hybrid_config,
        task_names=['regime', 'forecast', 'casualty', 'daily_forecast'],
        device=torch.device('cpu'),
    )

    losses = {
        'regime': torch.tensor(0.5),
        'forecast': torch.tensor(1.0),
        'casualty': torch.tensor(0.3),
        'daily_forecast': torch.tensor(0.8),
    }
    targets = {k: torch.randn(10) for k in losses}

    # Test training loss (A³DRO)
    train_loss, train_weights = hybrid_step.compute_training_loss(losses, targets, epoch=5)
    print(f"   A³DRO training loss: {train_loss.item():.4f} ✓")

    # Test validation loss (Uniform)
    val_loss, val_weights = hybrid_step.compute_validation_loss(losses)
    print(f"   Uniform validation loss: {val_loss.item():.4f} ✓")
    print(f"   Validation weights: {val_weights}")

    # Test spectral penalty
    pred_seq = torch.randn(4, 7, 64)
    target_seq = torch.randn(4, 7, 64)
    spectral_loss = hybrid_step.compute_spectral_penalty(pred_seq, target_seq)
    print(f"   Spectral penalty: {spectral_loss.item():.4f} ✓")

    # Test cycle consistency
    cycle_loss = hybrid_step.compute_cycle_consistency(daily_latents, monthly_latent)
    print(f"   Cycle consistency: {cycle_loss.item():.4f} ✓")

    print("\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)
    print("\nTo see the integration guide, run: print_integration_guide()")


if __name__ == '__main__':
    test_integration()
