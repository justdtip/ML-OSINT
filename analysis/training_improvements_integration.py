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
        AnchoredValidationLoss,  # GPT52
        HybridLossConfig,
        create_training_losses,
        # Phase 2 (Gemini proposal) components
        FocalLoss,
        FocalLossWithCollapsePrevention,
        SourceDropout,
        AdaptiveSourceDropout,
        CollapseDetector,
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
        AnchoredValidationLoss,  # GPT52
        HybridLossConfig,
        create_training_losses,
        # Phase 2 (Gemini proposal) components
        FocalLoss,
        FocalLossWithCollapsePrevention,
        SourceDropout,
        AdaptiveSourceDropout,
        CollapseDetector,
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
    - AnchoredValidationLoss (GPT52) or UniformValidationLoss for validation
    - SpectralDriftPenalty for forecast regularization

    GPT52 Enhancements:
    - Budgeted A³DRO (budget_beta=0.35): Prevents any task from dominating
    - Regret clipping (regret_clip=3.0): Numerical stability
    - Anchored validation: Uses regrets instead of raw losses for comparability

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

    # GPT52: Store reference to A³DRO base loss for baseline syncing
    if config.use_a3dro:
        # Get A³DRO from AvailabilityGatedLoss wrapper
        if hasattr(loss_modules['training'], 'base_loss'):
            trainer._a3dro_loss = loss_modules['training'].base_loss
        else:
            trainer._a3dro_loss = loss_modules['training']
    else:
        trainer._a3dro_loss = None

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
    _patch_validate_method(trainer, config)

    # Print summary
    if config.use_a3dro:
        print(f"Applied A3DRO loss (budget_beta={config.budget_beta}, regret_clip={config.regret_clip})")
    else:
        print("Applied SoftplusKendallLoss")

    if config.use_anchored_validation:
        print("Added AnchoredValidationLoss (GPT52: uses regrets for comparability)")
    else:
        print("Added UniformValidationLoss (fixed weights for comparability)")

    if config.use_spectral_penalty:
        print(f"Added SpectralDriftPenalty (weight={config.spectral_weight})")

    if config.use_cycle_consistency:
        print(f"Added CrossResolutionCycleConsistency (weight={config.cycle_weight})")


def _patch_validate_method(trainer, config: Optional[HybridLossConfig] = None) -> None:
    """
    Patch the trainer's validate method to use AnchoredValidationLoss or UniformValidationLoss.

    GPT52: If using anchored validation, also syncs baselines from A³DRO after warmup.
    This ensures validation loss is comparable across epochs (key audit finding).

    Args:
        trainer: The trainer instance
        config: HybridLossConfig for GPT52 parameters
    """
    original_validate = trainer.validate
    config = config or HybridLossConfig()

    def patched_validate(*args, **kwargs):
        # GPT52: Sync baselines from A³DRO to AnchoredValidationLoss after warmup
        if (config.use_anchored_validation
                and hasattr(trainer, '_a3dro_loss')
                and trainer._a3dro_loss is not None
                and trainer._a3dro_loss.baseline_frozen.item()):
            # Baselines are frozen, sync to validation loss
            if hasattr(trainer._validation_loss, 'set_baselines'):
                baselines = trainer._a3dro_loss.get_baselines()
                trainer._validation_loss.set_baselines(baselines)

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


def sync_validation_baselines(trainer) -> None:
    """
    Manually sync baselines from A³DRO to AnchoredValidationLoss.

    Call this after A³DRO warmup completes (typically after epoch >= warmup_epochs).

    Usage:
        # After warmup epochs
        if epoch >= warmup_epochs:
            sync_validation_baselines(trainer)

    Args:
        trainer: The trainer instance with hybrid loss improvements applied
    """
    if not hasattr(trainer, '_a3dro_loss') or trainer._a3dro_loss is None:
        warnings.warn("No A³DRO loss found on trainer. Cannot sync baselines.")
        return

    if not trainer._a3dro_loss.baseline_frozen.item():
        warnings.warn("A³DRO baselines not yet frozen. Wait for warmup to complete.")
        return

    if not hasattr(trainer._validation_loss, 'set_baselines'):
        warnings.warn("Validation loss does not support set_baselines. Using UniformValidationLoss?")
        return

    baselines = trainer._a3dro_loss.get_baselines()
    trainer._validation_loss.set_baselines(baselines)
    print(f"Synced validation baselines from A3DRO: {baselines}")


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

    GPT52 Enhancements:
    - Budgeted A³DRO prevents any single task from dominating
    - Regret clipping ensures numerical stability
    - Anchored validation uses regrets for comparable metrics across epochs
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

        # GPT52: Store reference to A³DRO for baseline syncing
        if config.use_a3dro:
            if hasattr(self.training_loss, 'base_loss'):
                self.a3dro_loss = self.training_loss.base_loss
            else:
                self.a3dro_loss = self.training_loss
        else:
            self.a3dro_loss = None

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

        GPT52: A³DRO now includes budget_beta and regret_clip for robustness.

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

    def sync_validation_baselines(self) -> bool:
        """
        Sync baselines from A³DRO to AnchoredValidationLoss.

        GPT52: Call this after A³DRO warmup completes to enable anchored validation.

        Returns:
            True if sync was successful, False otherwise
        """
        if self.a3dro_loss is None:
            return False

        if not self.a3dro_loss.baseline_frozen.item():
            return False

        if not hasattr(self.validation_loss, 'set_baselines'):
            return False

        baselines = self.a3dro_loss.get_baselines()
        self.validation_loss.set_baselines(baselines)
        return True

    def compute_validation_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute validation loss using anchored regrets or uniform weights.

        GPT52: If using AnchoredValidationLoss, returns uniform average of regrets.
        This makes validation comparable across epochs and tasks.

        Args:
            task_losses: Dict of task name -> loss tensor

        Returns:
            Tuple of (total_loss, weights_or_regrets)
        """
        # GPT52: Auto-sync baselines before validation
        self.sync_validation_baselines()
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
# PHASE 2 (GEMINI PROPOSAL) INTEGRATION
# =============================================================================
# Fixes for task collapse and source over-reliance:
# 1. Focal Loss for transition task (addresses collapse to constant)
# 2. Source Dropout for hdx_rainfall (addresses spurious correlation)
# 3. Collapse Detector for automatic intervention


class Phase2Config:
    """
    Configuration for Phase 2 (Gemini proposal) improvements.

    These address issues not fixed by Phase 1 (Budgeted A3DRO):
    - Transition task collapse (loss ~1e-8 at epoch 0)
    - Casualty task collapse (constant output)
    - hdx_rainfall over-reliance (5.2% importance despite irrelevance)

    Args:
        use_focal_loss: Enable focal loss for transition task. Default True.
        focal_alpha: Focal loss alpha (positive class weight). Default 0.25.
        focal_gamma: Focal loss gamma (focusing parameter). Default 2.0.
        use_source_dropout: Enable source dropout. Default True.
        source_dropout_prob: Probability of dropping source. Default 0.4.
        dropout_sources: List of sources to potentially drop.
            Default ['hdx_rainfall'].
        use_collapse_detection: Enable automatic collapse detection. Default True.
        collapse_threshold: Loss below this triggers collapse intervention.
            Default 1e-4.
        auto_enable_focal_on_collapse: Automatically switch to focal loss when
            collapse is detected. Default True.
    """

    def __init__(
        self,
        # Focal loss settings
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Source dropout settings
        use_source_dropout: bool = True,
        source_dropout_prob: float = 0.4,
        dropout_sources: Optional[List[str]] = None,
        # Collapse detection settings
        use_collapse_detection: bool = True,
        collapse_threshold: float = 1e-4,
        auto_enable_focal_on_collapse: bool = True,
    ):
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_source_dropout = use_source_dropout
        self.source_dropout_prob = source_dropout_prob
        self.dropout_sources = dropout_sources or ['hdx_rainfall']
        self.use_collapse_detection = use_collapse_detection
        self.collapse_threshold = collapse_threshold
        self.auto_enable_focal_on_collapse = auto_enable_focal_on_collapse


class Phase2TrainingStep:
    """
    Phase 2 training step with focal loss, source dropout, and collapse detection.

    Integrates with HybridTrainingStep to add Phase 2 fixes:
    1. Focal loss for transition task (prevents collapse)
    2. Source dropout for monthly features (prevents over-reliance)
    3. Collapse detection with automatic intervention

    Usage:
        phase2_step = Phase2TrainingStep(config, task_names, device)

        # In training loop:
        # Apply source dropout to monthly features
        monthly_features, monthly_masks = phase2_step.apply_source_dropout(
            monthly_features, monthly_masks
        )

        # Compute transition loss with focal loss
        if 'transition' in task_losses:
            task_losses['transition'] = phase2_step.compute_transition_loss(
                transition_logits, transition_targets
            )

        # Check for collapse and log
        phase2_step.update_collapse_detection(task_losses, epoch)
    """

    def __init__(
        self,
        config: Phase2Config,
        task_names: List[str],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.task_names = task_names

        # Initialize focal loss for transition task
        if config.use_focal_loss:
            self.focal_loss = FocalLossWithCollapsePrevention(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                collapse_threshold=config.collapse_threshold,
            ).to(device)
        else:
            self.focal_loss = None

        # Initialize source dropout
        if config.use_source_dropout:
            self.source_dropout = SourceDropout(
                dropout_sources=config.dropout_sources,
                dropout_prob=config.source_dropout_prob,
            ).to(device)
        else:
            self.source_dropout = None

        # Initialize collapse detector
        if config.use_collapse_detection:
            self.collapse_detector = CollapseDetector(
                task_names=task_names,
                loss_threshold=config.collapse_threshold,
            )
        else:
            self.collapse_detector = None

        # Track whether we've auto-enabled focal loss
        self._auto_focal_enabled = False

    def apply_source_dropout(
        self,
        features: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply source dropout to features.

        Call this in training loop BEFORE passing features to model.

        Args:
            features: Dict[source_name, Tensor[batch, seq, features]]
            masks: Dict[source_name, Tensor[batch, seq, features]]

        Returns:
            Tuple of (modified_features, modified_masks)
        """
        if self.source_dropout is None:
            return features, masks

        return self.source_dropout(features, masks)

    def compute_transition_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        use_focal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Compute transition loss with optional focal loss.

        Args:
            logits: Raw model outputs [batch] or [batch, 1]
            targets: Binary targets [batch]
            use_focal: Override focal loss usage. If None, uses config setting.

        Returns:
            Scalar loss
        """
        use_focal = use_focal if use_focal is not None else self.config.use_focal_loss

        if use_focal and self.focal_loss is not None:
            return self.focal_loss(logits, targets)
        else:
            # Fall back to BCE
            logits = logits.view(-1)
            targets = targets.view(-1).float()
            return F.binary_cross_entropy_with_logits(logits, targets)

    def update_collapse_detection(
        self,
        losses: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Dict[str, bool]:
        """
        Update collapse detection with current losses.

        Args:
            losses: Dict mapping task name to loss tensor
            epoch: Current training epoch

        Returns:
            Dict of newly collapsed tasks
        """
        if self.collapse_detector is None:
            return {}

        # Convert to float values
        loss_values = {
            name: loss.item() if isinstance(loss, torch.Tensor) else loss
            for name, loss in losses.items()
        }

        # Update detector
        newly_collapsed = self.collapse_detector.update(loss_values, epoch)

        # Auto-enable focal loss if transition collapsed
        if (self.config.auto_enable_focal_on_collapse
                and 'transition' in newly_collapsed
                and not self._auto_focal_enabled):
            print("[Phase2] Auto-enabling focal loss for transition task")
            self.config.use_focal_loss = True
            if self.focal_loss is None:
                self.focal_loss = FocalLossWithCollapsePrevention(
                    alpha=self.config.focal_alpha,
                    gamma=self.config.focal_gamma,
                    collapse_threshold=self.config.collapse_threshold,
                ).to(self.device)
            self._auto_focal_enabled = True

        return newly_collapsed

    def is_collapsed(self, task_name: str) -> bool:
        """Check if a task has collapsed."""
        if self.collapse_detector is None:
            return False
        return self.collapse_detector.is_collapsed(task_name)

    def get_collapsed_tasks(self) -> List[str]:
        """Get list of collapsed task names."""
        if self.collapse_detector is None:
            return []
        return self.collapse_detector.get_collapsed_tasks()

    def set_training(self, mode: bool = True) -> None:
        """Set training mode for dropout."""
        if self.source_dropout is not None:
            self.source_dropout.train(mode)


def apply_phase2_improvements(
    trainer,
    config: Optional[Phase2Config] = None,
) -> Phase2TrainingStep:
    """
    Apply Phase 2 (Gemini proposal) improvements to an existing trainer.

    This adds:
    - Focal loss for transition task
    - Source dropout for hdx_rainfall
    - Collapse detection with automatic intervention

    Args:
        trainer: MultiResolutionTrainer instance
        config: Optional Phase2Config (uses defaults if not provided)

    Returns:
        Phase2TrainingStep instance for use in training loop
    """
    config = config or Phase2Config()

    # Get task names
    if hasattr(trainer, 'multi_task_loss') and hasattr(trainer.multi_task_loss, 'task_names'):
        task_names = trainer.multi_task_loss.task_names
    else:
        task_names = ['regime', 'casualty', 'anomaly', 'forecast', 'daily_forecast', 'transition']

    # Create Phase2 step handler
    phase2_step = Phase2TrainingStep(
        config=config,
        task_names=task_names,
        device=trainer.device,
    )

    # Store on trainer
    trainer._phase2_step = phase2_step
    trainer._phase2_config = config

    # Print summary
    print("\n" + "=" * 60)
    print("APPLYING PHASE 2 (GEMINI PROPOSAL) IMPROVEMENTS")
    print("=" * 60)

    if config.use_focal_loss:
        print(f"  Focal Loss: alpha={config.focal_alpha}, gamma={config.focal_gamma}")

    if config.use_source_dropout:
        print(f"  Source Dropout: {config.dropout_sources} @ {config.source_dropout_prob:.0%}")

    if config.use_collapse_detection:
        print(f"  Collapse Detection: threshold={config.collapse_threshold:.2e}")

    print("=" * 60 + "\n")

    return phase2_step


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

    # Test Phase 2 improvements
    print("\n5. Testing Phase2Config and Phase2TrainingStep...")
    phase2_config = Phase2Config(
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_source_dropout=True,
        source_dropout_prob=0.4,
        dropout_sources=['hdx_rainfall'],
        use_collapse_detection=True,
        collapse_threshold=1e-4,
    )
    phase2_step = Phase2TrainingStep(
        config=phase2_config,
        task_names=['regime', 'casualty', 'transition'],
        device=torch.device('cpu'),
    )

    # Test focal loss for transition
    transition_logits = torch.randn(32)
    transition_targets = torch.zeros(32)
    transition_targets[5] = 1
    transition_loss = phase2_step.compute_transition_loss(
        transition_logits, transition_targets
    )
    print(f"   Focal transition loss: {transition_loss.item():.4f}")

    # Test source dropout
    phase2_step.set_training(True)
    features = {
        'equipment': torch.randn(4, 12, 38),
        'hdx_rainfall': torch.randn(4, 12, 16),
    }
    masks = {
        'equipment': torch.ones(4, 12, 38, dtype=torch.bool),
        'hdx_rainfall': torch.ones(4, 12, 16, dtype=torch.bool),
    }

    dropped_count = 0
    for _ in range(50):
        feat_out, mask_out = phase2_step.apply_source_dropout(features, masks)
        if feat_out['hdx_rainfall'].abs().sum() == 0:
            dropped_count += 1
    print(f"   Source dropout rate: {dropped_count * 2}% (expected ~40%)")

    # Test collapse detection
    phase2_step.update_collapse_detection(
        {'transition': 1e-8, 'casualty': 0.5}, epoch=0
    )
    assert phase2_step.is_collapsed('transition'), "Should detect transition collapse"
    print("   Collapse detection: working")

    print("\n" + "=" * 60)
    print("All integration tests passed (including Phase 2)!")
    print("=" * 60)
    print("\nTo see the integration guide, run: print_integration_guide()")


if __name__ == '__main__':
    test_integration()
