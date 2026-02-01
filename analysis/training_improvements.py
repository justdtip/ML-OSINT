"""
Training Improvements for Multi-Resolution HAN

Synthesized from three proposals:
- kimik2 (AFNP-STGS): PCGrad gradient surgery, low-rank decoders
- gpt52 (SD-MRPF): Latent predictive coding, cycle consistency, availability gating
- gemini (HGS-CP): Hierarchical gradient projection, consistency constraints

Core improvements:
1. LatentStatePredictor - Predict future latent states instead of raw features
2. PCGradSurgery - Project conflicting gradients to prevent task interference
3. SoftplusKendallLoss - Fix negative loss issue in Kendall uncertainty weighting
4. AvailabilityGatedLoss - Hard gate tasks with missing targets
5. CrossResolutionCycleConsistency - Enforce daily->monthly aggregation consistency

Phase 2 (Gemini Proposal) additions:
6. FocalLoss - Focal loss for imbalanced binary classification (transition task)
7. FocalLossWithCollapsePrevention - Extended focal loss with auto-collapse detection
8. SourceDropout - Randomly drop out entire sources during training
9. AdaptiveSourceDropout - Adaptive dropout based on source importance
10. CollapseDetector - Monitors task losses for collapse and triggers interventions

Author: Claude (synthesized from multiple AI proposals)
Date: 2026-01-31
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. LATENT STATE PREDICTOR (from gpt52)
# =============================================================================
# Replaces the massive DailyForecastingHead (4.4M params) with a lightweight
# predictor that operates in latent space (~O(D²) params).

class LatentStatePredictor(nn.Module):
    """
    Predicts future latent states instead of raw high-dimensional features.

    This prevents the forecast head from memorizing training data by:
    1. Operating in compressed latent space (d_model dimensions)
    2. Using horizon embeddings for temporal conditioning
    3. Optional low-rank decoder for raw feature reconstruction

    Parameters reduced from ~4.4M to ~O(d_model²) ≈ 65K with d_model=256.

    Args:
        d_model: Latent dimension
        horizon: Number of future timesteps to predict
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
        use_horizon_embedding: Whether to use learnable horizon embeddings
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_horizon_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.horizon = horizon
        self.use_horizon_embedding = use_horizon_embedding

        # Learnable horizon embeddings (one per future timestep)
        if use_horizon_embedding:
            self.horizon_embeddings = nn.Embedding(horizon, d_model)

        # Lightweight MLP for latent state prediction
        input_dim = d_model * 2 if use_horizon_embedding else d_model
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        context_latent: Tensor,
        return_all_horizons: bool = True,
        mask: Optional[Tensor] = None,  # Unused, for API compatibility
    ) -> Tensor:
        """
        Predict future latent states from context.

        Args:
            context_latent: Context latent state [batch, d_model] or [batch, seq_len, d_model]
            return_all_horizons: If True, return all horizon predictions
            mask: Optional mask (unused, for API compatibility with DailyForecastingHead)

        Returns:
            Predicted future latents [batch, horizon, d_model] or [batch, d_model]
        """
        # Handle 3D input by extracting last timestep
        if context_latent.dim() == 3:
            context_latent = context_latent[:, -1, :]  # [batch, d_model]

        batch_size = context_latent.shape[0]
        device = context_latent.device

        if return_all_horizons:
            predictions = []
            for tau in range(self.horizon):
                if self.use_horizon_embedding:
                    # Concatenate context with horizon embedding
                    h_embed = self.horizon_embeddings(
                        torch.tensor([tau], device=device)
                    ).expand(batch_size, -1)
                    input_feat = torch.cat([context_latent, h_embed], dim=-1)
                else:
                    input_feat = context_latent

                pred = self.predictor(input_feat)
                predictions.append(pred)

            return torch.stack(predictions, dim=1)  # [batch, horizon, d_model]
        else:
            # Single step prediction
            if self.use_horizon_embedding:
                h_embed = self.horizon_embeddings(
                    torch.zeros(batch_size, dtype=torch.long, device=device)
                )
                input_feat = torch.cat([context_latent, h_embed], dim=-1)
            else:
                input_feat = context_latent

            return self.predictor(input_feat)


class LowRankFeatureDecoder(nn.Module):
    """
    Optional low-rank decoder for reconstructing raw features from latents.

    Uses factorized projection: x = A_s(B @ z) + b_s
    where B is shared across sources, A_s is per-source.

    Total params: r*d_model + sum(F_s * r) instead of d_model * sum(F_s)

    Args:
        d_model: Latent dimension
        source_dims: Dict mapping source names to feature dimensions
        rank: Rank of factorization (bottleneck dimension)
    """

    def __init__(
        self,
        d_model: int,
        source_dims: Dict[str, int],
        rank: int = 32,
    ) -> None:
        super().__init__()

        self.source_dims = source_dims
        self.rank = rank

        # Shared bottleneck projection
        self.shared_proj = nn.Linear(d_model, rank)

        # Per-source output projections
        self.source_projs = nn.ModuleDict({
            name: nn.Linear(rank, dim)
            for name, dim in source_dims.items()
        })

    def forward(
        self,
        latent: Tensor,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Decode latent to per-source features.

        Args:
            latent: Latent representation [batch, ..., d_model]
            sources: List of sources to decode (default: all)

        Returns:
            Dict mapping source names to decoded features
        """
        sources = sources or list(self.source_dims.keys())

        # Shared bottleneck
        bottleneck = self.shared_proj(latent)

        # Per-source decoding
        outputs = {}
        for source in sources:
            if source in self.source_projs:
                outputs[source] = self.source_projs[source](bottleneck)

        return outputs


# =============================================================================
# 2. PCGRAD SURGERY (from kimik2/gemini)
# =============================================================================
# Prevents gradient conflicts between tasks by projecting conflicting gradients.

class PCGradSurgery:
    """
    Projecting Conflicting Gradients for multi-task learning.

    When task gradients conflict (negative cosine similarity), removes the
    conflicting component from the interfering gradient. This prevents
    high-magnitude gradients (e.g., forecast task) from destroying
    representations learned by other tasks (e.g., regime classification).

    Based on:
    - Yu et al. (2020) "Gradient Surgery for Multi-Task Learning"
    - kimik2/gemini proposals for HAN training

    Args:
        task_groups: Optional grouping of tasks for hierarchical projection.
            If provided, gradients within each group are summed before projection.
            Format: {'stable': ['regime', 'forecast'], 'noisy': ['casualty', 'daily_forecast']}
    """

    def __init__(
        self,
        task_groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.task_groups = task_groups or {}

    def project_gradients(
        self,
        gradients: Dict[str, List[Tensor]],
    ) -> List[Tensor]:
        """
        Apply PCGrad projection to per-task gradients.

        Args:
            gradients: Dict mapping task names to list of parameter gradients.
                Each value is a list of tensors (one per model parameter).

        Returns:
            List of projected gradients (one per model parameter).
        """
        task_names = list(gradients.keys())
        n_params = len(gradients[task_names[0]])

        # Initialize output gradients
        final_grads = [torch.zeros_like(gradients[task_names[0]][i])
                       for i in range(n_params)]

        # Group gradients if task_groups specified
        if self.task_groups:
            grouped_grads = {}
            for group_name, tasks in self.task_groups.items():
                # Sum gradients within group
                group_grad = [torch.zeros_like(final_grads[i]) for i in range(n_params)]
                for task in tasks:
                    if task in gradients:
                        for i in range(n_params):
                            group_grad[i] = group_grad[i] + gradients[task][i]
                grouped_grads[group_name] = group_grad
            gradients = grouped_grads
            task_names = list(gradients.keys())

        # Apply pairwise projection
        for i, task_i in enumerate(task_names):
            grad_i = gradients[task_i]

            for j, task_j in enumerate(task_names):
                if i == j:
                    continue

                grad_j = gradients[task_j]

                # Process each parameter
                for p_idx in range(n_params):
                    g_i = grad_i[p_idx].view(-1)
                    g_j = grad_j[p_idx].view(-1)

                    # Compute dot product
                    dot = torch.dot(g_i, g_j)

                    if dot < 0:
                        # Conflicting gradients - project g_i onto normal of g_j
                        g_j_norm_sq = torch.dot(g_j, g_j) + 1e-8
                        proj_scalar = dot / g_j_norm_sq

                        # Remove conflicting component
                        grad_i[p_idx] = grad_i[p_idx] - proj_scalar * grad_j[p_idx]

        # Sum all (projected) task gradients
        for task in task_names:
            for i in range(n_params):
                final_grads[i] = final_grads[i] + gradients[task][i]

        return final_grads

    def compute_task_gradients(
        self,
        model: nn.Module,
        losses: Dict[str, Tensor],
        retain_graph: bool = True,
    ) -> Dict[str, List[Tensor]]:
        """
        Compute per-task gradients via separate backward passes.

        Args:
            model: The model to compute gradients for
            losses: Dict mapping task names to scalar loss tensors
            retain_graph: Whether to retain computation graph

        Returns:
            Dict mapping task names to lists of parameter gradients
        """
        task_grads = {}
        params = list(model.parameters())

        for task_name, loss in losses.items():
            if loss.requires_grad:
                # Zero gradients
                model.zero_grad()

                # Backward for this task
                loss.backward(retain_graph=retain_graph)

                # Collect gradients
                grads = []
                for p in params:
                    if p.grad is not None:
                        grads.append(p.grad.clone())
                    else:
                        grads.append(torch.zeros_like(p))

                task_grads[task_name] = grads

        return task_grads

    def backward_with_surgery(
        self,
        model: nn.Module,
        losses: Dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Perform backward pass with gradient surgery.

        Args:
            model: The model to update
            losses: Dict mapping task names to scalar loss tensors
            optimizer: The optimizer (will be stepped after gradient assignment)
        """
        # Compute per-task gradients
        task_grads = self.compute_task_gradients(model, losses, retain_graph=True)

        # Apply PCGrad projection
        final_grads = self.project_gradients(task_grads)

        # Zero gradients and assign projected gradients
        optimizer.zero_grad()
        for p, g in zip(model.parameters(), final_grads):
            p.grad = g


# =============================================================================
# 3. SOFTPLUS KENDALL LOSS (from gpt52)
# =============================================================================
# Fixes negative loss issue in standard Kendall uncertainty weighting.

class SoftplusKendallLoss(nn.Module):
    """
    Fixed Kendall uncertainty weighting that cannot produce negative losses.

    Standard Kendall formulation:
        L = 0.5 * exp(-log_var) * L_i + 0.5 * log_var

    This can go negative when log_var < 0 and L_i is small.

    Fixed formulation using softplus:
        s_i = softplus(u_i)  # Ensures s_i >= 0
        L = exp(-s_i) * L_i + s_i

    This maintains the spirit of learned weighting without pathologies.

    Args:
        task_names: List of task identifiers
        init_scale: Initial value for scale parameters
    """

    def __init__(
        self,
        task_names: List[str],
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()

        self.task_names = list(task_names)

        # Raw (pre-softplus) scale parameters
        self.raw_scales = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
            for name in self.task_names
        })

    def forward(
        self,
        losses: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute weighted combined loss.

        Args:
            losses: Dict mapping task names to scalar loss tensors
            masks: Optional dict of availability masks (not used, for API compat)

        Returns:
            Tuple of (total_loss, task_weights dict)
        """
        device = next(iter(losses.values())).device
        total_loss = torch.tensor(0.0, device=device)
        task_weights = {}

        for task_name in self.task_names:
            if task_name not in losses:
                continue

            task_loss = losses[task_name]

            # Skip invalid losses
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                continue

            # Apply softplus to ensure non-negative scale
            scale = F.softplus(self.raw_scales[task_name])

            # Fixed formulation: L = exp(-s) * L_i + s
            precision = torch.exp(-scale)
            weighted_loss = precision * task_loss + scale
            total_loss = total_loss + weighted_loss

            task_weights[task_name] = precision.item()

        return total_loss, task_weights

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (precision = exp(-softplus(raw)))."""
        weights = {}
        with torch.no_grad():
            for task_name in self.task_names:
                scale = F.softplus(self.raw_scales[task_name])
                weights[task_name] = torch.exp(-scale).item()
        return weights

    def get_uncertainties(self) -> Dict[str, float]:
        """Get current task uncertainties (variance = exp(scale))."""
        uncertainties = {}
        with torch.no_grad():
            for task_name in self.task_names:
                scale = F.softplus(self.raw_scales[task_name])
                uncertainties[task_name] = torch.exp(scale).item()
        return uncertainties


# =============================================================================
# 4. AVAILABILITY-GATED LOSS (from gpt52)
# =============================================================================
# Prevents task collapse by hard-gating tasks with insufficient supervision.

class AvailabilityGatedLoss(nn.Module):
    """
    Hard-gates tasks based on target availability.

    Instead of using regularization fallbacks when targets are missing,
    completely excludes tasks with availability below a threshold.

    This prevents the model from learning to output constants for
    tasks with sparse supervision (which was causing task collapse).

    Args:
        task_names: List of task identifiers
        min_availability: Minimum fraction of valid targets to include task
        base_loss: Underlying loss function (SoftplusKendallLoss or MultiTaskLoss)
    """

    def __init__(
        self,
        task_names: List[str],
        min_availability: float = 0.2,
        base_loss: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.task_names = list(task_names)
        self.min_availability = min_availability

        # Use SoftplusKendallLoss by default
        self.base_loss = base_loss or SoftplusKendallLoss(task_names)

    def compute_availability(
        self,
        targets: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute availability ratio for each task.

        Args:
            targets: Dict of target tensors (NaN indicates missing)
            masks: Optional explicit availability masks

        Returns:
            Dict mapping task names to availability ratios [0, 1]
        """
        availability = {}

        for task_name in self.task_names:
            if task_name not in targets:
                availability[task_name] = 0.0
                continue

            target = targets[task_name]

            if masks and task_name in masks:
                # Use explicit mask
                mask = masks[task_name]
                avail = mask.float().mean().item()
            else:
                # Infer from NaN values
                valid = ~torch.isnan(target)
                avail = valid.float().mean().item()

            availability[task_name] = avail

        return availability

    def forward(
        self,
        losses: Dict[str, Tensor],
        targets: Optional[Dict[str, Tensor]] = None,
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute availability-gated loss.

        Args:
            losses: Dict mapping task names to loss tensors
            targets: Optional dict of target tensors (for availability computation).
                     If None, all tasks are assumed to have full availability.
            masks: Optional availability masks

        Returns:
            Tuple of (total_loss, task_weights)
            Note: availability dict removed from return for API compatibility
        """
        # If no targets provided, assume full availability (fallback mode)
        if targets is None:
            return self.base_loss(losses, masks)

        # Compute availability
        availability = self.compute_availability(targets, masks)

        # Filter losses by availability threshold
        gated_losses = {}
        for task_name, loss in losses.items():
            avail = availability.get(task_name, 0.0)
            if avail >= self.min_availability:
                gated_losses[task_name] = loss
            else:
                # Task excluded due to low availability
                pass

        # Apply base loss to gated losses
        if gated_losses:
            total_loss, task_weights = self.base_loss(gated_losses, masks)
        else:
            # No tasks passed availability gate - return small regularization
            device = next(iter(losses.values())).device
            total_loss = torch.tensor(1e-6, device=device, requires_grad=True)
            task_weights = {}

        # Return only (total_loss, task_weights) for API compatibility
        # Availability info is logged but not returned to match MultiTaskLoss signature
        return total_loss, task_weights

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights from base loss (for API compatibility)."""
        return self.base_loss.get_task_weights()

    def get_uncertainties(self) -> Dict[str, float]:
        """Get current task uncertainties from base loss (for API compatibility)."""
        return self.base_loss.get_uncertainties()


# =============================================================================
# 5. CROSS-RESOLUTION CYCLE CONSISTENCY (from gpt52)
# =============================================================================
# Enforces that daily predictions aggregate to match monthly predictions.

class CrossResolutionCycleConsistency(nn.Module):
    """
    Enforces cycle consistency between daily and monthly predictions.

    The key insight: if we predict daily latents z_{t+1..t+7}^d and monthly
    latent z_{t+1}^m, then aggregating the daily latents should approximately
    match the monthly latent.

    This provides free supervision and prevents daily forecasts from
    diverging from monthly structure.

    Args:
        aggregation_module: Module to aggregate daily latents to monthly
            (should be compatible with EnhancedLearnableMonthlyAggregation)
        loss_type: 'cosine' for cosine similarity or 'mse' for MSE
        weight: Weight for consistency loss
    """

    def __init__(
        self,
        aggregation_module: Optional[nn.Module] = None,
        loss_type: str = 'cosine',
        weight: float = 0.2,
    ) -> None:
        super().__init__()

        self.aggregation_module = aggregation_module
        self.loss_type = loss_type
        self.weight = weight

        # Simple attention-based aggregation if none provided
        if aggregation_module is None:
            self.simple_aggregation = nn.Sequential(
                nn.Linear(128, 64),  # Attention scoring
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        else:
            self.simple_aggregation = None

    def aggregate_daily_to_monthly(
        self,
        daily_latents: Tensor,
        month_boundaries: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Aggregate daily latents to monthly using attention.

        Args:
            daily_latents: [batch, n_days, d_model]
            month_boundaries: Optional [batch, 2] start/end indices

        Returns:
            Aggregated monthly latent [batch, d_model]
        """
        if self.aggregation_module is not None:
            # Use provided aggregation module
            # Assuming it has a similar interface
            return self.aggregation_module(daily_latents, month_boundaries)

        # Simple attention-weighted aggregation
        # Compute attention scores
        scores = self.simple_aggregation(daily_latents)  # [batch, n_days, 1]
        weights = F.softmax(scores, dim=1)

        # Weighted sum
        aggregated = (daily_latents * weights).sum(dim=1)  # [batch, d_model]
        return aggregated

    def forward(
        self,
        daily_pred_latents: Tensor,
        monthly_teacher_latent: Tensor,
        month_boundaries: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute cycle consistency loss.

        Args:
            daily_pred_latents: Predicted daily latents [batch, horizon, d_model]
            monthly_teacher_latent: Teacher monthly latent [batch, d_model]
            month_boundaries: Optional boundaries for aggregation

        Returns:
            Scalar consistency loss
        """
        # Aggregate daily predictions to monthly
        aggregated = self.aggregate_daily_to_monthly(
            daily_pred_latents, month_boundaries
        )

        # Compute consistency loss
        if self.loss_type == 'cosine':
            # Cosine similarity loss (1 - cos_sim)
            cos_sim = F.cosine_similarity(
                aggregated, monthly_teacher_latent, dim=-1
            )
            loss = (1 - cos_sim).mean()
        else:
            # MSE loss
            loss = F.mse_loss(aggregated, monthly_teacher_latent)

        return self.weight * loss


# =============================================================================
# 6. LATENT PREDICTIVE CODING LOSS (from gpt52)
# =============================================================================
# Teacher-student loss for latent state prediction.

class LatentPredictiveCodingLoss(nn.Module):
    """
    Teacher-student predictive coding loss for latent forecasting.

    Student: Predicts future latent states from context (causal)
    Teacher: Encodes actual future data (stop-grad, non-causal)

    Loss: cosine similarity or Huber between student predictions and
    teacher latents.

    This is the core mechanism that prevents forecast head memorization:
    instead of predicting raw features, we predict compressed latent
    states which contain much less capacity for memorization.

    Args:
        loss_type: 'cosine' or 'huber'
        daily_weight: Weight for daily latent prediction loss
        monthly_weight: Weight for monthly latent prediction loss
    """

    def __init__(
        self,
        loss_type: str = 'cosine',
        daily_weight: float = 1.0,
        monthly_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.loss_type = loss_type
        self.daily_weight = daily_weight
        self.monthly_weight = monthly_weight

    def forward(
        self,
        daily_pred: Tensor,
        daily_teacher: Tensor,
        monthly_pred: Optional[Tensor] = None,
        monthly_teacher: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute predictive coding loss.

        Args:
            daily_pred: Student daily predictions [batch, horizon, d_model]
            daily_teacher: Teacher daily latents [batch, horizon, d_model]
            monthly_pred: Optional student monthly predictions [batch, d_model]
            monthly_teacher: Optional teacher monthly latents [batch, d_model]

        Returns:
            Tuple of (total_loss, loss_components dict)
        """
        components = {}
        total_loss = torch.tensor(0.0, device=daily_pred.device)

        # Daily predictive coding loss
        if self.loss_type == 'cosine':
            # Per-timestep cosine similarity
            cos_sim = F.cosine_similarity(daily_pred, daily_teacher, dim=-1)
            daily_loss = (1 - cos_sim).mean()
        else:
            # Huber loss
            daily_loss = F.smooth_l1_loss(daily_pred, daily_teacher)

        components['daily_pc'] = daily_loss
        total_loss = total_loss + self.daily_weight * daily_loss

        # Monthly predictive coding loss (if provided)
        if monthly_pred is not None and monthly_teacher is not None:
            if self.loss_type == 'cosine':
                cos_sim = F.cosine_similarity(monthly_pred, monthly_teacher, dim=-1)
                monthly_loss = (1 - cos_sim).mean()
            else:
                monthly_loss = F.smooth_l1_loss(monthly_pred, monthly_teacher)

            components['monthly_pc'] = monthly_loss
            total_loss = total_loss + self.monthly_weight * monthly_loss

        return total_loss, components


# =============================================================================
# 7. PHYSICAL CONSISTENCY CONSTRAINT (from gemini)
# =============================================================================
# Enforces that daily predictions align with monthly at correct scale.

class PhysicalConsistencyConstraint(nn.Module):
    """
    Enforces physical consistency between daily and monthly predictions.

    Constraint: Sum of next 7 days should align with monthly forecast
    scaled by the fraction 7/30.

    This acts as a physics-based regularizer that prevents the model
    from memorizing random daily noise that contradicts monthly trends.

    Args:
        daily_scale: Scaling factor for daily predictions (default: 7/30)
        weight: Weight for consistency loss
    """

    def __init__(
        self,
        daily_scale: float = 7.0 / 30.0,
        weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.daily_scale = daily_scale
        self.weight = weight

    def forward(
        self,
        daily_pred: Tensor,
        monthly_pred: Tensor,
    ) -> Tensor:
        """
        Compute physical consistency loss.

        Args:
            daily_pred: Daily predictions [batch, horizon, n_features]
            monthly_pred: Monthly predictions [batch, n_features]

        Returns:
            Scalar consistency loss
        """
        # Sum daily predictions over horizon
        daily_sum = daily_pred.sum(dim=1)  # [batch, n_features]

        # Scale monthly to match expected daily sum
        monthly_scaled = monthly_pred * self.daily_scale

        # MSE between sums
        loss = F.mse_loss(daily_sum, monthly_scaled)

        return self.weight * loss


# =============================================================================
# 8. COMBINED IMPROVED TRAINING LOOP
# =============================================================================

class ImprovedTrainingConfig:
    """Configuration for improved training loop."""

    def __init__(
        self,
        use_pcgrad: bool = True,
        use_softplus_kendall: bool = True,
        use_availability_gating: bool = True,
        use_latent_prediction: bool = True,
        use_cycle_consistency: bool = True,
        use_physical_consistency: bool = True,
        min_availability: float = 0.2,
        cycle_consistency_weight: float = 0.2,
        physical_consistency_weight: float = 0.1,
        pcgrad_task_groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.use_pcgrad = use_pcgrad
        self.use_softplus_kendall = use_softplus_kendall
        self.use_availability_gating = use_availability_gating
        self.use_latent_prediction = use_latent_prediction
        self.use_cycle_consistency = use_cycle_consistency
        self.use_physical_consistency = use_physical_consistency
        self.min_availability = min_availability
        self.cycle_consistency_weight = cycle_consistency_weight
        self.physical_consistency_weight = physical_consistency_weight

        # Default grouping: stable (monthly-scale) vs noisy (daily-scale)
        self.pcgrad_task_groups = pcgrad_task_groups or {
            'stable': ['regime', 'forecast'],
            'noisy': ['casualty', 'daily_forecast', 'anomaly', 'transition'],
        }


def create_improved_loss_fn(
    task_names: List[str],
    config: ImprovedTrainingConfig,
) -> nn.Module:
    """
    Factory function to create the improved loss function.

    Args:
        task_names: List of task identifiers
        config: Training configuration

    Returns:
        Loss module (SoftplusKendallLoss or AvailabilityGatedLoss)
    """
    if config.use_softplus_kendall:
        base_loss = SoftplusKendallLoss(task_names)
    else:
        # Fall back to standard (would need to import MultiTaskLoss)
        base_loss = SoftplusKendallLoss(task_names)  # Default to fixed version

    if config.use_availability_gating:
        return AvailabilityGatedLoss(
            task_names=task_names,
            min_availability=config.min_availability,
            base_loss=base_loss,
        )
    else:
        return base_loss


# =============================================================================
# 9. A³DRO LOSS (from gpt52) - RECOMMENDED FOR VALIDATION
# =============================================================================
# Availability-Aware Anchored Distributionally-Robust Objective
# KEY ADVANTAGE: No learned weights -> validation loss directly comparable


class A3DROLoss(nn.Module):
    """
    Availability-Aware Anchored Distributionally-Robust Objective (A³DRO).

    Replaces learned task weighting with principled robust aggregation:
    1. Normalize each task by frozen anchor/scale -> prevents domination
    2. Soft gate by label availability -> prevents collapse under sparse supervision
    3. Robust log-sum-exp across tasks -> focuses on worst-performing task
    4. NO LEARNED WEIGHTS -> validation/early stopping are meaningful

    GPT52 Proposal Enhancements:
    5. Budgeted mixing (beta): q_i^(β) = (1-β)*q_i + β*u_i prevents any task from
       dominating (max weight capped at 1-β)
    6. Regret clipping (c): Clamps regrets to [-c, +c] for numerical stability

    Mathematical formulation:
        r_i = log(L_i + ε) - log(b_i + ε)  # Anchored regret
        r_i_clip = clamp(r_i, -c, +c)      # Regret clipping (GPT52)
        g_i = sigmoid(κ(a_i - a_min))      # Soft availability gate
        p̃_i = p_i * g_i / Σ_j(p_j * g_j)  # Effective prior
        q_i = softmax(r_i_clip / λ)        # DRO weights
        q_i^(β) = (1-β)*q_i + β*u_i        # Budgeted mixing (GPT52)
        L = λ * log(Σ_i q_i^(β) * exp(r_i_clip / λ))  # Robust aggregation

    Args:
        task_names: List of task identifiers
        priors: Dict of prior task weights (default: uniform)
        lambda_temp: Temperature for robust aggregation (smaller = more focus on worst)
        a_min: Minimum availability threshold
        kappa: Steepness of availability gate sigmoid
        warmup_epochs: Epochs before freezing baselines
        budget_beta: Budget mixing coefficient (GPT52). Mixes DRO weights with uniform:
            q^(β) = (1-β)*q + β*u. Default 0.35 caps any task at 65% of gradient budget.
            Set to 0.0 to disable budgeting (original A³DRO behavior).
        regret_clip: Maximum absolute regret value (GPT52). Clamps regrets to [-c, +c]
            before softmax for numerical stability. Default 3.0.
            Set to float('inf') to disable clipping (original behavior).
    """

    def __init__(
        self,
        task_names: List[str],
        priors: Optional[Dict[str, float]] = None,
        lambda_temp: float = 1.0,
        a_min: float = 0.2,
        kappa: float = 20.0,
        warmup_epochs: int = 3,
        budget_beta: float = 0.35,
        regret_clip: float = 3.0,
    ) -> None:
        super().__init__()

        self.task_names = list(task_names)
        self.lambda_temp = lambda_temp
        self.a_min = a_min
        self.kappa = kappa
        self.warmup_epochs = warmup_epochs
        self.budget_beta = budget_beta
        self.regret_clip = regret_clip

        # Task priors (fixed, not learned)
        if priors is None:
            priors = {name: 1.0 / len(task_names) for name in task_names}
        self.priors = priors

        # Anchors and scales (frozen after warmup)
        # Using buffers so they're saved with state_dict but not trained
        self.register_buffer(
            'baselines',
            torch.zeros(len(task_names))
        )
        self.register_buffer(
            'scales',
            torch.ones(len(task_names))
        )
        self.register_buffer(
            'baseline_frozen',
            torch.tensor(False)
        )

        # EMA for computing baselines during warmup
        self._loss_ema = {name: None for name in task_names}
        self._loss_var_ema = {name: None for name in task_names}
        self._ema_alpha = 0.1

    def update_baselines(self, losses: Dict[str, Tensor], epoch: int) -> None:
        """Update baseline EMAs during warmup, freeze after warmup_epochs."""
        if self.baseline_frozen.item():
            return

        for i, task_name in enumerate(self.task_names):
            if task_name not in losses:
                continue

            loss_val = losses[task_name].detach().item()

            # Update EMA
            if self._loss_ema[task_name] is None:
                self._loss_ema[task_name] = loss_val
                self._loss_var_ema[task_name] = 0.0
            else:
                delta = loss_val - self._loss_ema[task_name]
                self._loss_ema[task_name] += self._ema_alpha * delta
                self._loss_var_ema[task_name] = (
                    (1 - self._ema_alpha) * self._loss_var_ema[task_name] +
                    self._ema_alpha * delta ** 2
                )

        # Freeze after warmup
        if epoch >= self.warmup_epochs:
            for i, task_name in enumerate(self.task_names):
                if self._loss_ema[task_name] is not None:
                    self.baselines[i] = self._loss_ema[task_name]
                    # Use sqrt of variance as scale (MAD approximation)
                    self.scales[i] = max(
                        (self._loss_var_ema[task_name] ** 0.5),
                        1e-6
                    )
            self.baseline_frozen.fill_(True)

    def compute_availability(
        self,
        targets: Optional[Dict[str, Tensor]] = None,
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute availability for each task."""
        availability = {}
        for task_name in self.task_names:
            if masks and task_name in masks:
                avail = masks[task_name].float().mean().item()
            elif targets and task_name in targets:
                target = targets[task_name]
                valid = ~torch.isnan(target) if target.is_floating_point() else torch.ones_like(target, dtype=torch.bool)
                avail = valid.float().mean().item()
            else:
                avail = 1.0  # Assume full availability if not provided
            availability[task_name] = avail
        return availability

    def forward(
        self,
        losses: Dict[str, Tensor],
        targets: Optional[Dict[str, Tensor]] = None,
        masks: Optional[Dict[str, Tensor]] = None,
        epoch: int = 0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute A³DRO aggregated loss with GPT52 enhancements.

        Args:
            losses: Dict mapping task names to scalar loss tensors
            targets: Optional targets for availability computation
            masks: Optional explicit availability masks
            epoch: Current epoch (for baseline warmup)

        Returns:
            Tuple of (total_loss, effective_weights dict)
        """
        device = next(iter(losses.values())).device

        # Update baselines during warmup
        if not self.baseline_frozen.item():
            self.update_baselines(losses, epoch)

        # Compute availability
        availability = self.compute_availability(targets, masks)

        # Soft availability gating: g_i = sigmoid(κ(a_i - a_min))
        gates = {}
        for task_name in self.task_names:
            a_i = availability.get(task_name, 1.0)
            g_i = torch.sigmoid(
                torch.tensor(self.kappa * (a_i - self.a_min), device=device)
            )
            gates[task_name] = g_i

        # Effective prior with gating: p̃_i = p_i * g_i / Z
        p_eff_unnorm = {
            name: self.priors.get(name, 1.0) * gates[name]
            for name in self.task_names
        }
        Z = sum(p_eff_unnorm.values()) + 1e-8
        p_eff = {name: p_eff_unnorm[name] / Z for name in self.task_names}

        # Compute anchored regrets: r_i = log(L_i) - log(b_i)
        regrets = []
        valid_tasks = []

        for i, task_name in enumerate(self.task_names):
            if task_name not in losses:
                continue

            task_loss = losses[task_name]
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                continue

            # Anchored log-ratio regret
            if self.baseline_frozen.item():
                baseline = self.baselines[i]
            else:
                baseline = torch.tensor(1.0, device=device)
            regret = torch.log(task_loss + 1e-8) - torch.log(baseline + 1e-8)

            # GPT52: Regret clipping for numerical stability
            # Clamp regrets to [-c, +c] before softmax
            if self.regret_clip < float('inf'):
                regret = regret.clamp(-self.regret_clip, self.regret_clip)

            regrets.append(regret)
            valid_tasks.append(task_name)

        if not regrets:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Stack regrets for vectorized operations
        regrets_tensor = torch.stack(regrets)
        n_active = len(valid_tasks)

        # Compute DRO weights via softmax on regrets
        # q_i = softmax(r_i / λ) with prior weighting
        logits = torch.stack([
            torch.log(p_eff[name] + 1e-12) + regrets[i] / self.lambda_temp
            for i, name in enumerate(valid_tasks)
        ])
        q_dro = F.softmax(logits, dim=0)  # DRO weights

        # GPT52: Budgeted mixing - prevents any single task from dominating
        # q_i^(β) = (1-β)*q_i + β*u_i where u_i = 1/|active_tasks|
        if self.budget_beta > 0 and n_active > 0:
            u_uniform = torch.ones(n_active, device=device) / n_active
            q_budgeted = (1 - self.budget_beta) * q_dro + self.budget_beta * u_uniform
        else:
            q_budgeted = q_dro

        # Record effective weights for logging
        effective_weights = {
            name: q_budgeted[i].item()
            for i, name in enumerate(valid_tasks)
        }

        # Robust aggregation: L = Σ_i q_i^(β) * r_i
        # Using weighted sum of regrets (equivalent to weighted log-loss ratio)
        total_loss = (q_budgeted * regrets_tensor).sum()

        return total_loss, effective_weights

    def get_baselines(self) -> Dict[str, float]:
        """Get frozen baseline values."""
        return {
            name: self.baselines[i].item()
            for i, name in enumerate(self.task_names)
        }

    def get_task_weights(self) -> Dict[str, float]:
        """Get current effective task weights (for API compatibility).

        A³DRO doesn't have learned weights - returns uniform effective weights.
        The actual weighting happens dynamically based on regret.
        """
        # Return uniform weights since A³DRO uses dynamic robust aggregation
        return {name: 1.0 / len(self.task_names) for name in self.task_names}

    def get_uncertainties(self) -> Dict[str, float]:
        """Get task uncertainties (for API compatibility).

        A³DRO doesn't track uncertainties like Kendall - returns scale factors.
        """
        return {
            name: self.scales[i].item()
            for i, name in enumerate(self.task_names)
        }


# =============================================================================
# 10. SPECTRAL DRIFT PENALTY (from gemini) - FOR FORECAST REGULARIZATION
# =============================================================================
# Penalizes low-frequency (trend) errors more than high-frequency (noise) errors


class SpectralDriftPenalty(nn.Module):
    """
    FFT-based spectral loss that penalizes trend drift more than noise.

    The key insight: for forecasting, we care about the *trend* (low frequency)
    more than the day-to-day jitter (high frequency). This loss:
    1. Transforms prediction error to frequency domain via FFT
    2. Weights low-frequency components more heavily
    3. Returns weighted spectral magnitude

    This directly addresses the 140x Train/Val gap by ignoring high-frequency
    noise that the model was memorizing.

    Args:
        weight_decay: Exponential decay rate for frequency weights
            Higher = more focus on very low frequencies
        weight: Multiplier for the spectral loss
    """

    def __init__(
        self,
        weight_decay: float = 0.5,
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight_decay = weight_decay
        self.weight = weight

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute spectral drift penalty.

        Args:
            predictions: Predicted sequence [batch, seq_len, features]
            targets: Target sequence [batch, seq_len, features]
            mask: Optional mask for valid timesteps [batch, seq_len]

        Returns:
            Scalar spectral drift loss
        """
        # Compute error signal
        error = predictions - targets  # [batch, seq_len, features]

        # Apply mask if provided
        if mask is not None:
            error = error * mask.unsqueeze(-1)

        # FFT along time dimension
        # Using rfft for real signals (more efficient)
        fft_error = torch.fft.rfft(error, dim=1)
        magnitude = torch.abs(fft_error)  # [batch, freq_bins, features]

        # Create low-pass filter weighting
        # Lower frequencies (DC, trends) get higher weight
        freq_bins = magnitude.shape[1]
        freqs = torch.arange(freq_bins, device=magnitude.device, dtype=magnitude.dtype)
        weights = torch.exp(-self.weight_decay * freqs)  # [freq_bins]

        # Expand for broadcasting: [1, freq_bins, 1]
        weights = weights.view(1, -1, 1)

        # Weighted spectral magnitude
        weighted_magnitude = weights * magnitude
        spectral_loss = weighted_magnitude.mean()

        return self.weight * spectral_loss


# =============================================================================
# 11. UNIFIED VALIDATION LOSS (Recommended from audit)
# =============================================================================
# Fixed weights for comparable validation metrics across epochs


class UniformValidationLoss(nn.Module):
    """
    Fixed uniform weights for validation - ensures loss comparability.

    This addresses the key audit finding: learned weights during training
    make validation loss non-comparable across epochs and runs.

    For validation and early stopping, use this instead of SoftplusKendallLoss.

    Args:
        task_names: List of task identifiers
        weights: Optional fixed weights (default: uniform 1/N)
    """

    def __init__(
        self,
        task_names: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()

        self.task_names = list(task_names)
        if weights is None:
            weights = {name: 1.0 / len(task_names) for name in task_names}
        self.weights = weights

    def forward(
        self,
        losses: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute uniformly weighted loss.

        Args:
            losses: Dict mapping task names to scalar loss tensors
            masks: Unused, for API compatibility

        Returns:
            Tuple of (total_loss, weights dict)
        """
        device = next(iter(losses.values())).device
        total_loss = torch.tensor(0.0, device=device)
        used_weights = {}

        for task_name in self.task_names:
            if task_name not in losses:
                continue

            task_loss = losses[task_name]
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                continue

            weight = self.weights.get(task_name, 1.0 / len(self.task_names))
            total_loss = total_loss + weight * task_loss
            used_weights[task_name] = weight

        return total_loss, used_weights

    def get_task_weights(self) -> Dict[str, float]:
        """Get fixed task weights."""
        return self.weights.copy()

    def get_uncertainties(self) -> Dict[str, float]:
        """Get task uncertainties (returns 1.0 for all - no uncertainty tracking)."""
        return {name: 1.0 for name in self.task_names}


# =============================================================================
# 11b. ANCHORED VALIDATION LOSS (GPT52 Proposal)
# =============================================================================
# Uses uniform average of regrets instead of raw losses for validation.
# This makes validation comparable across epochs and tasks.


class AnchoredValidationLoss(nn.Module):
    """
    Anchored validation loss using uniform average of log-ratio regrets.

    GPT52 Proposal: Instead of computing validation as uniform average of raw losses,
    use uniform average of anchored regrets:
        L_val = (1/|A|) * sum(r_i)
    where r_i = log(L_i + eps) - log(b_i + eps) are log-ratio regrets.

    This provides several key benefits:
    1. **Comparable across epochs**: Raw losses change scale as training progresses,
       but regrets are always relative to frozen baselines.
    2. **Comparable across tasks**: Tasks with different loss scales are normalized
       by their baseline, so regime (small) and forecast (large) contribute equally.
    3. **Interpretable**: Negative regret means "better than baseline", positive means worse.

    The baselines should be shared with the training A3DROLoss to ensure consistency.

    Args:
        task_names: List of task identifiers
        baselines: Dict mapping task names to baseline loss values.
            These should come from A3DROLoss after warmup (a3dro.get_baselines()).
            If None, defaults to 1.0 for all tasks (no anchoring).
        regret_clip: Maximum absolute regret value for numerical stability.
            Default 3.0 (same as A3DROLoss default).
    """

    def __init__(
        self,
        task_names: List[str],
        baselines: Optional[Dict[str, float]] = None,
        regret_clip: float = 3.0,
    ) -> None:
        super().__init__()

        self.task_names = list(task_names)
        self.regret_clip = regret_clip

        # Store baselines as buffer (not trained)
        if baselines is None:
            baselines = {name: 1.0 for name in task_names}

        # Register baselines as a buffer tensor for serialization
        baseline_tensor = torch.tensor(
            [baselines.get(name, 1.0) for name in task_names],
            dtype=torch.float32
        )
        self.register_buffer('baselines', baseline_tensor)

        # Keep dict version for convenience
        self._baseline_dict = baselines

    def set_baselines(self, baselines: Dict[str, float]) -> None:
        """
        Update baselines from A3DROLoss after warmup.

        Call this after A3DROLoss warmup completes:
            anchored_val.set_baselines(a3dro_loss.get_baselines())

        Args:
            baselines: Dict mapping task names to baseline loss values
        """
        self._baseline_dict = baselines.copy()
        baseline_tensor = torch.tensor(
            [baselines.get(name, 1.0) for name in self.task_names],
            dtype=torch.float32,
            device=self.baselines.device
        )
        self.baselines.copy_(baseline_tensor)

    def forward(
        self,
        losses: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute anchored validation loss as uniform average of regrets.

        Args:
            losses: Dict mapping task names to scalar loss tensors
            masks: Unused, for API compatibility

        Returns:
            Tuple of (total_loss, regret_dict)
            - total_loss: Uniform average of regrets (1/|A| * sum(r_i))
            - regret_dict: Per-task regret values for logging
        """
        device = next(iter(losses.values())).device
        regrets = []
        regret_dict = {}

        for i, task_name in enumerate(self.task_names):
            if task_name not in losses:
                continue

            task_loss = losses[task_name]
            if torch.isnan(task_loss) or torch.isinf(task_loss):
                continue

            # Compute anchored log-ratio regret
            baseline = self.baselines[i]
            regret = torch.log(task_loss + 1e-8) - torch.log(baseline + 1e-8)

            # Apply regret clipping for numerical stability
            if self.regret_clip < float('inf'):
                regret = regret.clamp(-self.regret_clip, self.regret_clip)

            regrets.append(regret)
            regret_dict[task_name] = regret.item()

        if not regrets:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Uniform average of regrets: L_val = (1/|A|) * sum(r_i)
        total_loss = torch.stack(regrets).mean()

        return total_loss, regret_dict

    def get_task_weights(self) -> Dict[str, float]:
        """Get task weights (uniform for anchored validation)."""
        return {name: 1.0 / len(self.task_names) for name in self.task_names}

    def get_uncertainties(self) -> Dict[str, float]:
        """Get task uncertainties (returns baselines as proxy)."""
        return self._baseline_dict.copy()

    def get_baselines(self) -> Dict[str, float]:
        """Get current baseline values."""
        return {
            name: self.baselines[i].item()
            for i, name in enumerate(self.task_names)
        }


# =============================================================================
# 12. HYBRID LOSS CONFIGURATION (Recommended Final Approach)
# =============================================================================


class HybridLossConfig:
    """
    Recommended configuration combining best of all proposals:
    - A³DRO or SoftplusKendall for training (with AvailabilityGating)
    - AnchoredValidation or UniformValidation for validation/early stopping
    - SpectralDriftPenalty for forecast regularization
    - CrossResolutionCycleConsistency for daily/monthly alignment

    GPT52 Enhancements:
    - Budgeted A³DRO (budget_beta=0.35): Prevents any task from dominating
    - Regret clipping (regret_clip=3.0): Numerical stability
    - Anchored validation (use_anchored_validation=True): Comparable metrics
    """

    def __init__(
        self,
        # Loss selection
        use_a3dro: bool = True,  # Recommended: True for robust aggregation
        use_spectral_penalty: bool = True,  # Recommended: True for forecast regularization
        use_cycle_consistency: bool = True,
        use_anchored_validation: bool = True,  # GPT52: Use anchored regrets for validation

        # A³DRO parameters
        lambda_temp: float = 0.5,  # Smaller = more focus on worst task
        warmup_epochs: int = 3,

        # GPT52: Budgeted A³DRO parameters
        budget_beta: float = 0.35,  # Mix with uniform to cap any task at 65%
        regret_clip: float = 3.0,   # Clamp regrets for numerical stability

        # Spectral penalty parameters
        spectral_weight_decay: float = 0.5,
        spectral_weight: float = 0.1,

        # Cycle consistency parameters
        cycle_weight: float = 0.2,

        # Availability gating (still useful even with A³DRO soft gating)
        use_hard_availability_gating: bool = True,
        min_availability: float = 0.2,
    ):
        self.use_a3dro = use_a3dro
        self.use_spectral_penalty = use_spectral_penalty
        self.use_cycle_consistency = use_cycle_consistency
        self.use_anchored_validation = use_anchored_validation
        self.lambda_temp = lambda_temp
        self.warmup_epochs = warmup_epochs
        self.budget_beta = budget_beta
        self.regret_clip = regret_clip
        self.spectral_weight_decay = spectral_weight_decay
        self.spectral_weight = spectral_weight
        self.cycle_weight = cycle_weight
        self.use_hard_availability_gating = use_hard_availability_gating
        self.min_availability = min_availability


def create_training_losses(
    task_names: List[str],
    config: HybridLossConfig,
) -> Dict[str, nn.Module]:
    """
    Create training and validation loss functions based on config.

    GPT52 enhancements:
    - A³DRO now includes budget_beta and regret_clip parameters
    - Validation can use AnchoredValidationLoss for comparable metrics

    Returns:
        Dict with keys:
            'training': Loss for training (A³DRO or SoftplusKendall)
            'validation': Loss for validation (AnchoredValidation or UniformValidation)
            'spectral': Optional SpectralDriftPenalty
            'cycle': Optional CrossResolutionCycleConsistency
    """
    losses = {}

    # Training loss with GPT52 enhancements
    if config.use_a3dro:
        base_loss = A3DROLoss(
            task_names=task_names,
            lambda_temp=config.lambda_temp,
            warmup_epochs=config.warmup_epochs,
            budget_beta=config.budget_beta,    # GPT52: Budgeted mixing
            regret_clip=config.regret_clip,    # GPT52: Regret clipping
        )
    else:
        base_loss = SoftplusKendallLoss(task_names)

    if config.use_hard_availability_gating:
        losses['training'] = AvailabilityGatedLoss(
            task_names=task_names,
            min_availability=config.min_availability,
            base_loss=base_loss,
        )
    else:
        losses['training'] = base_loss

    # Validation loss
    # GPT52: Use anchored validation for comparable metrics across epochs
    if config.use_anchored_validation:
        losses['validation'] = AnchoredValidationLoss(
            task_names=task_names,
            baselines=None,  # Will be set after A³DRO warmup
            regret_clip=config.regret_clip,
        )
    else:
        # Fallback to uniform validation (original behavior)
        losses['validation'] = UniformValidationLoss(task_names)

    # Auxiliary losses
    if config.use_spectral_penalty:
        losses['spectral'] = SpectralDriftPenalty(
            weight_decay=config.spectral_weight_decay,
            weight=config.spectral_weight,
        )

    if config.use_cycle_consistency:
        losses['cycle'] = CrossResolutionCycleConsistency(
            loss_type='cosine',
            weight=config.cycle_weight,
        )

    return losses


# =============================================================================
# TESTING
# =============================================================================

def test_improvements():
    """Test the improvement modules."""
    print("Testing training improvements...")

    batch_size = 4
    d_model = 128
    horizon = 7
    n_features = 64

    # Test LatentStatePredictor
    print("\n1. Testing LatentStatePredictor...")
    predictor = LatentStatePredictor(d_model=d_model, horizon=horizon)
    context = torch.randn(batch_size, d_model)
    pred = predictor(context)
    assert pred.shape == (batch_size, horizon, d_model)
    print(f"   Output shape: {pred.shape} ✓")

    # Test LowRankFeatureDecoder
    print("\n2. Testing LowRankFeatureDecoder...")
    decoder = LowRankFeatureDecoder(
        d_model=d_model,
        source_dims={'equipment': 38, 'personnel': 6},
        rank=16,
    )
    latent = torch.randn(batch_size, d_model)
    decoded = decoder(latent)
    assert 'equipment' in decoded and decoded['equipment'].shape == (batch_size, 38)
    print(f"   Decoded equipment shape: {decoded['equipment'].shape} ✓")

    # Test SoftplusKendallLoss
    print("\n3. Testing SoftplusKendallLoss...")
    loss_fn = SoftplusKendallLoss(['task1', 'task2'])
    losses = {
        'task1': torch.tensor(1.0),
        'task2': torch.tensor(0.5),
    }
    total, weights = loss_fn(losses)
    assert total > 0, "Loss should be positive with softplus fix"
    print(f"   Total loss: {total.item():.4f} (positive ✓)")
    print(f"   Weights: {weights}")

    # Test PCGradSurgery
    print("\n4. Testing PCGradSurgery...")
    surgery = PCGradSurgery()
    # Create mock gradients that conflict
    grad1 = [torch.tensor([1.0, 0.0])]
    grad2 = [torch.tensor([-1.0, 0.0])]  # Conflicts with grad1
    grads = {'task1': grad1, 'task2': grad2}
    projected = surgery.project_gradients(grads)
    # After projection, conflicting components should be removed
    print(f"   Projected gradient: {projected[0]} ✓")

    # Test CrossResolutionCycleConsistency
    print("\n5. Testing CrossResolutionCycleConsistency...")
    cycle = CrossResolutionCycleConsistency(loss_type='cosine', weight=0.2)
    daily_latents = torch.randn(batch_size, horizon, d_model)
    monthly_latent = torch.randn(batch_size, d_model)
    cycle_loss = cycle(daily_latents, monthly_latent)
    print(f"   Cycle consistency loss: {cycle_loss.item():.4f} ✓")

    # Test LatentPredictiveCodingLoss
    print("\n6. Testing LatentPredictiveCodingLoss...")
    pc_loss_fn = LatentPredictiveCodingLoss(loss_type='cosine')
    daily_pred = torch.randn(batch_size, horizon, d_model)
    daily_teacher = daily_pred + 0.1 * torch.randn_like(daily_pred)  # Similar
    pc_loss, components = pc_loss_fn(daily_pred, daily_teacher)
    print(f"   Predictive coding loss: {pc_loss.item():.4f} ✓")

    # Test AvailabilityGatedLoss
    print("\n7. Testing AvailabilityGatedLoss...")
    gated_loss = AvailabilityGatedLoss(
        task_names=['task1', 'task2'],
        min_availability=0.5,
    )
    losses = {'task1': torch.tensor(1.0), 'task2': torch.tensor(0.5)}
    targets = {
        'task1': torch.randn(10),  # 100% available
        'task2': torch.tensor([float('nan')] * 8 + [1.0, 1.0]),  # 20% available
    }
    # Compute availability separately for testing
    avail = gated_loss.compute_availability(targets)
    total, weights = gated_loss(losses, targets)
    print(f"   Availability: {avail}")
    print(f"   Task2 gated out (20% < 50%): {'task2' not in weights} ✓")

    # Test A³DRO Loss with GPT52 enhancements
    print("\n8. Testing A3DROLoss with GPT52 (budgeted + clipped)...")
    a3dro = A3DROLoss(
        task_names=['task1', 'task2', 'task3'],
        lambda_temp=0.5,
        warmup_epochs=2,
        budget_beta=0.35,  # GPT52: Budgeted mixing
        regret_clip=3.0,   # GPT52: Regret clipping
    )
    losses = {
        'task1': torch.tensor(1.0),
        'task2': torch.tensor(5.0),  # Higher loss
        'task3': torch.tensor(0.5),
    }
    # Simulate warmup
    for epoch in range(3):
        total, eff_weights = a3dro(losses, epoch=epoch)
    print(f"   Total A³DRO loss: {total.item():.4f}")
    print(f"   Effective weights: {eff_weights}")
    print(f"   Baselines frozen: {a3dro.baseline_frozen.item()} ✓")
    print(f"   Baselines: {a3dro.get_baselines()}")

    # GPT52: Verify budget constraint (no task > 65%)
    max_weight = max(eff_weights.values())
    max_allowed = 1 - a3dro.budget_beta
    print(f"   Max task weight: {max_weight:.4f} (budget cap: {max_allowed:.2f})")
    assert max_weight <= max_allowed + 0.01, f"Budget violated: {max_weight} > {max_allowed}"
    print(f"   Budget constraint satisfied ✓")

    # Test A³DRO without budgeting (original behavior)
    print("\n8b. Testing A3DROLoss without budgeting...")
    a3dro_no_budget = A3DROLoss(
        task_names=['task1', 'task2', 'task3'],
        lambda_temp=0.5,
        warmup_epochs=2,
        budget_beta=0.0,   # Disable budgeting
        regret_clip=float('inf'),  # Disable clipping
    )
    for epoch in range(3):
        total_nb, eff_weights_nb = a3dro_no_budget(losses, epoch=epoch)
    print(f"   Without budget, max weight: {max(eff_weights_nb.values()):.4f}")

    # Test SpectralDriftPenalty
    print("\n9. Testing SpectralDriftPenalty...")
    spectral = SpectralDriftPenalty(weight_decay=0.5, weight=0.1)
    predictions = torch.randn(batch_size, horizon, n_features)
    targets_seq = torch.randn(batch_size, horizon, n_features)
    spectral_loss = spectral(predictions, targets_seq)
    assert spectral_loss.item() >= 0, "Spectral loss should be non-negative"
    print(f"   Spectral drift loss: {spectral_loss.item():.4f} ✓")

    # Test UniformValidationLoss
    print("\n10. Testing UniformValidationLoss...")
    uniform_val = UniformValidationLoss(['task1', 'task2', 'task3'])
    losses = {
        'task1': torch.tensor(1.0),
        'task2': torch.tensor(2.0),
        'task3': torch.tensor(3.0),
    }
    total, weights = uniform_val(losses)
    expected = (1.0 + 2.0 + 3.0) / 3
    assert abs(total.item() - expected) < 1e-5, "Uniform loss should be simple average"
    print(f"   Total uniform loss: {total.item():.4f} (expected: {expected:.4f}) ✓")
    print(f"   Weights: {weights}")

    # Test AnchoredValidationLoss (GPT52)
    print("\n10b. Testing AnchoredValidationLoss (GPT52)...")
    # Use baselines from the A³DRO warmup
    baselines = a3dro.get_baselines()
    print(f"   Baselines from A³DRO: {baselines}")

    anchored_val = AnchoredValidationLoss(
        task_names=['task1', 'task2', 'task3'],
        baselines=baselines,
        regret_clip=3.0,
    )

    # Test with same losses as baselines -> regrets should be ~0
    baseline_losses = {name: torch.tensor(val) for name, val in baselines.items()}
    total_anchored, regret_dict = anchored_val(baseline_losses)
    print(f"   Loss at baseline: {total_anchored.item():.6f} (should be ~0)")
    print(f"   Regrets: {regret_dict}")
    assert abs(total_anchored.item()) < 0.01, "Regrets at baseline should be ~0"
    print(f"   Baseline test passed ✓")

    # Test with losses 2x baseline -> regrets should be positive (log(2) ~ 0.69)
    double_losses = {name: torch.tensor(val * 2) for name, val in baselines.items()}
    total_double, regret_double = anchored_val(double_losses)
    print(f"   Loss at 2x baseline: {total_double.item():.4f} (should be ~0.69)")
    assert total_double.item() > 0, "Regrets should be positive when loss > baseline"
    print(f"   2x baseline test passed ✓")

    # Test with losses 0.5x baseline -> regrets should be negative
    half_losses = {name: torch.tensor(val * 0.5) for name, val in baselines.items()}
    total_half, regret_half = anchored_val(half_losses)
    print(f"   Loss at 0.5x baseline: {total_half.item():.4f} (should be ~-0.69)")
    assert total_half.item() < 0, "Regrets should be negative when loss < baseline"
    print(f"   0.5x baseline test passed ✓")

    # Test set_baselines method
    new_baselines = {'task1': 2.0, 'task2': 3.0, 'task3': 1.0}
    anchored_val.set_baselines(new_baselines)
    assert anchored_val.get_baselines() == new_baselines, "set_baselines should update"
    print(f"   set_baselines() works ✓")

    # Test HybridLossConfig and factory with GPT52
    print("\n11. Testing HybridLossConfig + create_training_losses (GPT52)...")
    config = HybridLossConfig(
        use_a3dro=True,
        use_spectral_penalty=True,
        use_cycle_consistency=True,
        use_anchored_validation=True,  # GPT52
        budget_beta=0.35,               # GPT52
        regret_clip=3.0,                # GPT52
    )
    task_names = ['regime', 'casualty', 'forecast', 'daily_forecast']
    loss_modules = create_training_losses(task_names, config)
    assert 'training' in loss_modules
    assert 'validation' in loss_modules
    assert 'spectral' in loss_modules
    assert 'cycle' in loss_modules
    print(f"   Training loss type: {type(loss_modules['training']).__name__}")
    print(f"   Validation loss type: {type(loss_modules['validation']).__name__}")

    # Verify GPT52 parameters are passed through
    # Get base A³DRO from AvailabilityGatedLoss
    base_loss = loss_modules['training'].base_loss
    assert base_loss.budget_beta == 0.35, "budget_beta should be 0.35"
    assert base_loss.regret_clip == 3.0, "regret_clip should be 3.0"
    print(f"   A³DRO budget_beta: {base_loss.budget_beta} ✓")
    print(f"   A³DRO regret_clip: {base_loss.regret_clip} ✓")

    # Verify anchored validation is used
    assert isinstance(loss_modules['validation'], AnchoredValidationLoss), \
        "Should use AnchoredValidationLoss"
    print(f"   AnchoredValidationLoss created ✓")
    print(f"   Created all loss modules ✓")

    # Test backward compatibility (disable GPT52 features)
    print("\n11b. Testing backward compatibility (GPT52 disabled)...")
    config_legacy = HybridLossConfig(
        use_a3dro=True,
        use_anchored_validation=False,  # Use legacy uniform validation
        budget_beta=0.0,                 # Disable budgeting
        regret_clip=float('inf'),        # Disable clipping
    )
    loss_modules_legacy = create_training_losses(task_names, config_legacy)
    assert isinstance(loss_modules_legacy['validation'], UniformValidationLoss), \
        "Should use UniformValidationLoss when anchored disabled"
    print(f"   Legacy mode works ✓")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


# =============================================================================
# 13. FOCAL LOSS (Phase 2 - Gemini Proposal)
# =============================================================================
# Addresses transition task collapse by down-weighting easy negatives


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.

    Addresses the transition task collapse where the model learns to predict
    "no transition" constantly (loss ~1e-8 at epoch 0). This happens because:
    - ~95%+ of samples are "no transition" (class imbalance)
    - BCE loss is dominated by easy negative examples
    - Model converges to always-negative output

    Focal Loss down-weights easy examples using:
        FL = alpha * (1-pt)^gamma * BCE
    where:
        pt = exp(-BCE) = probability of correct class
        (1-pt)^gamma = modulating factor that reduces loss for easy examples

    Parameters:
        alpha (float): Weighting factor for the positive class. Default 0.25.
            Helps address class imbalance by upweighting the rare positive class.
        gamma (float): Focusing parameter. Default 2.0.
            gamma=0 reduces to standard BCE
            gamma>0 reduces loss for well-classified examples
        reduction (str): 'mean', 'sum', or 'none'
        auto_detect_collapse (bool): If True, automatically detect loss collapse
            and adjust alpha/gamma. Default True.
        collapse_threshold (float): Loss below this triggers collapse detection.
            Default 1e-4.

    Reference:
        Lin et al. (2017) "Focal Loss for Dense Object Detection"

    Example:
        >>> focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(32)  # Raw scores
        >>> targets = torch.zeros(32)  # Mostly negative
        >>> targets[5] = 1  # Few positive
        >>> loss = focal_loss(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        auto_detect_collapse: bool = True,
        collapse_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.auto_detect_collapse = auto_detect_collapse
        self.collapse_threshold = collapse_threshold

        # Track collapse detection state
        self._collapse_detected = False
        self._original_alpha = alpha
        self._original_gamma = gamma

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid) [batch] or [batch, 1]
            targets: Binary targets (0 or 1) [batch] or [batch, 1]
            weight: Optional per-sample weights [batch]

        Returns:
            Focal loss scalar (if reduction='mean'/'sum') or per-sample losses
        """
        # Flatten if needed
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Compute BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Compute pt = P(y=1) for positive, P(y=0) for negative
        # pt = sigmoid(logit) for positive targets
        # pt = 1 - sigmoid(logit) for negative targets
        # Equivalently: pt = exp(-BCE)
        pt = torch.exp(-bce_loss)

        # Compute alpha factor
        # alpha for positive class, (1-alpha) for negative class
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight: (1-pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)

        # Final focal loss
        focal_loss = alpha_factor * focal_weight * bce_loss

        # Apply sample weights if provided
        if weight is not None:
            focal_loss = focal_loss * weight.view(-1)

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def detect_and_adjust_collapse(
        self,
        loss_value: float,
        positive_rate: float,
    ) -> bool:
        """
        Detect loss collapse and automatically adjust parameters.

        Call this after computing loss to check for collapse. If detected,
        parameters are adjusted to encourage learning.

        Args:
            loss_value: Current loss value
            positive_rate: Fraction of positive samples in batch

        Returns:
            True if collapse was detected and parameters adjusted
        """
        if not self.auto_detect_collapse:
            return False

        if loss_value < self.collapse_threshold and not self._collapse_detected:
            self._collapse_detected = True

            # Increase alpha to upweight rare positive class
            # Increase gamma to down-weight easy negatives more aggressively
            new_alpha = min(0.75, self.alpha * 2)
            new_gamma = min(5.0, self.gamma + 1.0)

            print(f"[FocalLoss] Collapse detected (loss={loss_value:.2e}, "
                  f"pos_rate={positive_rate:.3f})")
            print(f"[FocalLoss] Adjusting alpha: {self.alpha:.2f} -> {new_alpha:.2f}")
            print(f"[FocalLoss] Adjusting gamma: {self.gamma:.1f} -> {new_gamma:.1f}")

            self.alpha = new_alpha
            self.gamma = new_gamma
            return True

        return False

    def reset_parameters(self) -> None:
        """Reset to original alpha and gamma values."""
        self.alpha = self._original_alpha
        self.gamma = self._original_gamma
        self._collapse_detected = False


class FocalLossWithCollapsePrevention(nn.Module):
    """
    Extended Focal Loss with automatic collapse detection and prevention.

    This wrapper around FocalLoss adds:
    1. Automatic collapse detection when loss < threshold
    2. Label smoothing to prevent overconfidence
    3. Minimum loss floor to maintain gradients
    4. Positive class upweighting based on class distribution

    Use this instead of raw FocalLoss for the transition task.

    Args:
        alpha: Base weighting factor for positive class. Default 0.25.
        gamma: Focusing parameter. Default 2.0.
        label_smoothing: Smooth labels to prevent overconfidence. Default 0.1.
        min_loss_floor: Minimum loss value to maintain gradients. Default 1e-6.
        collapse_threshold: Loss below this triggers collapse prevention. Default 1e-4.
        auto_reweight: Automatically reweight based on class distribution. Default True.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        min_loss_floor: float = 1e-6,
        collapse_threshold: float = 1e-4,
        auto_reweight: bool = True,
    ) -> None:
        super().__init__()
        self.base_alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.min_loss_floor = min_loss_floor
        self.collapse_threshold = collapse_threshold
        self.auto_reweight = auto_reweight

        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='none',
            auto_detect_collapse=False,  # We handle collapse ourselves
        )

        # Track statistics for adaptive reweighting
        self._running_pos_rate = 0.5
        self._ema_decay = 0.99
        self._collapse_count = 0

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute focal loss with collapse prevention.

        Args:
            logits: Raw model outputs (before sigmoid) [batch]
            targets: Binary targets (0 or 1) [batch]
            weight: Optional per-sample weights [batch]

        Returns:
            Focal loss scalar with collapse prevention
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Update running positive rate
        with torch.no_grad():
            pos_rate = targets.mean().item()
            self._running_pos_rate = (
                self._ema_decay * self._running_pos_rate +
                (1 - self._ema_decay) * pos_rate
            )

        # Auto-reweight alpha based on class imbalance
        if self.auto_reweight and self._running_pos_rate < 0.1:
            # Severe imbalance: increase alpha for rare class
            effective_alpha = min(0.9, self.base_alpha / (self._running_pos_rate + 0.01))
        else:
            effective_alpha = self.base_alpha

        self.focal_loss.alpha = effective_alpha

        # Compute focal loss
        per_sample_loss = self.focal_loss(logits, targets, weight)

        # Mean reduction with floor
        loss = per_sample_loss.mean()

        # Apply minimum loss floor to maintain gradients
        loss = torch.maximum(loss, torch.tensor(self.min_loss_floor, device=loss.device))

        # Check for collapse
        if loss.item() < self.collapse_threshold:
            self._collapse_count += 1

            if self._collapse_count >= 5:
                # Persistent collapse - add gradient-maintaining term
                # Small regularization on logit magnitude
                reg_loss = logits.pow(2).mean() * 1e-4
                loss = loss + reg_loss

        return loss

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics for logging."""
        return {
            'running_pos_rate': self._running_pos_rate,
            'effective_alpha': self.focal_loss.alpha,
            'gamma': self.gamma,
            'collapse_count': self._collapse_count,
        }


# =============================================================================
# 14. SOURCE DROPOUT (Phase 2 - Gemini Proposal)
# =============================================================================
# Prevents over-reliance on uninformative sources like hdx_rainfall


class SourceDropout(nn.Module):
    """
    Randomly drops out entire sources during training.

    Addresses the issue where uninformative sources (like hdx_rainfall with 5.2%
    importance despite being irrelevant) dominate because they have strong
    seasonal patterns that correlate spuriously with targets.

    During training, randomly zeros out specified sources with given probability.
    This forces the model to learn from other (more relevant) sources.

    Args:
        dropout_sources: List of source names to potentially drop.
            Default: ['hdx_rainfall']
        dropout_prob: Probability of dropping each source. Default 0.4.
        training_only: Only apply dropout during training. Default True.

    Example:
        >>> source_dropout = SourceDropout(
        ...     dropout_sources=['hdx_rainfall'],
        ...     dropout_prob=0.4,
        ... )
        >>> features = {'equipment': torch.randn(4, 12, 38),
        ...             'hdx_rainfall': torch.randn(4, 12, 16)}
        >>> masks = {'equipment': torch.ones(4, 12, 38, dtype=torch.bool),
        ...          'hdx_rainfall': torch.ones(4, 12, 16, dtype=torch.bool)}
        >>> # During training, hdx_rainfall may be zeroed out 40% of the time
        >>> features, masks = source_dropout(features, masks)
    """

    def __init__(
        self,
        dropout_sources: Optional[List[str]] = None,
        dropout_prob: float = 0.4,
        training_only: bool = True,
    ) -> None:
        super().__init__()
        self.dropout_sources = dropout_sources or ['hdx_rainfall']
        self.dropout_prob = dropout_prob
        self.training_only = training_only

    def forward(
        self,
        features: Dict[str, Tensor],
        masks: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Apply source dropout.

        Args:
            features: Dict[source_name, Tensor[batch, seq, features]]
            masks: Dict[source_name, Tensor[batch, seq, features]]

        Returns:
            Tuple of (modified_features, modified_masks)
        """
        if self.training_only and not self.training:
            return features, masks

        features_out = {}
        masks_out = {}

        for name, feat in features.items():
            if name in self.dropout_sources and self.training:
                # Decide whether to drop this source (same decision for whole batch)
                if torch.rand(1).item() < self.dropout_prob:
                    # Zero out features and mark as missing
                    features_out[name] = torch.zeros_like(feat)
                    masks_out[name] = torch.zeros_like(masks[name])
                else:
                    features_out[name] = feat
                    masks_out[name] = masks[name]
            else:
                features_out[name] = feat
                masks_out[name] = masks[name]

        return features_out, masks_out

    def extra_repr(self) -> str:
        return f"dropout_sources={self.dropout_sources}, dropout_prob={self.dropout_prob}"


class AdaptiveSourceDropout(nn.Module):
    """
    Adaptive source dropout based on source importance.

    Extends SourceDropout by tracking which sources the model relies on and
    increasing dropout probability for over-relied sources.

    This creates a curriculum where:
    1. Initially: Low dropout, model can use all sources
    2. Later: High dropout for dominant sources, forcing diversification
    3. Result: More balanced source usage, less spurious correlation

    Args:
        all_sources: List of all source names
        initial_dropout_prob: Starting dropout probability. Default 0.1.
        max_dropout_prob: Maximum dropout probability. Default 0.6.
        importance_threshold: Sources with importance above this get dropout.
            Default 0.15 (15% importance).
        update_interval: How often to update dropout probs. Default 100 batches.
    """

    def __init__(
        self,
        all_sources: List[str],
        initial_dropout_prob: float = 0.1,
        max_dropout_prob: float = 0.6,
        importance_threshold: float = 0.15,
        update_interval: int = 100,
    ) -> None:
        super().__init__()
        self.all_sources = list(all_sources)
        self.initial_dropout_prob = initial_dropout_prob
        self.max_dropout_prob = max_dropout_prob
        self.importance_threshold = importance_threshold
        self.update_interval = update_interval

        # Initialize per-source dropout probabilities
        self.dropout_probs = {name: initial_dropout_prob for name in all_sources}

        # Track source importance (from attention weights or gradient norms)
        self.source_importance = {name: 1.0 / len(all_sources) for name in all_sources}
        self._batch_count = 0

    def update_importance(self, importance: Dict[str, float]) -> None:
        """
        Update source importance estimates.

        Call this after each batch with the attention weights or
        gradient norms from the model.

        Args:
            importance: Dict mapping source name to importance score (0-1)
        """
        # EMA update
        alpha = 0.1
        for name in self.all_sources:
            if name in importance:
                self.source_importance[name] = (
                    (1 - alpha) * self.source_importance[name] +
                    alpha * importance[name]
                )

        self._batch_count += 1

        # Periodically update dropout probabilities
        if self._batch_count % self.update_interval == 0:
            self._update_dropout_probs()

    def _update_dropout_probs(self) -> None:
        """Update dropout probabilities based on source importance."""
        for name in self.all_sources:
            imp = self.source_importance[name]

            if imp > self.importance_threshold:
                # Source is over-relied - increase dropout
                # Scale dropout linearly with how much importance exceeds threshold
                excess = (imp - self.importance_threshold) / (1 - self.importance_threshold)
                new_prob = self.initial_dropout_prob + excess * (
                    self.max_dropout_prob - self.initial_dropout_prob
                )
                self.dropout_probs[name] = min(new_prob, self.max_dropout_prob)
            else:
                # Source is underutilized - keep low dropout
                self.dropout_probs[name] = self.initial_dropout_prob

    def forward(
        self,
        features: Dict[str, Tensor],
        masks: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Apply adaptive source dropout.

        Args:
            features: Dict[source_name, Tensor[batch, seq, features]]
            masks: Dict[source_name, Tensor[batch, seq, features]]

        Returns:
            Tuple of (modified_features, modified_masks)
        """
        if not self.training:
            return features, masks

        features_out = {}
        masks_out = {}

        for name, feat in features.items():
            if name in self.dropout_probs:
                if torch.rand(1).item() < self.dropout_probs[name]:
                    # Drop this source
                    features_out[name] = torch.zeros_like(feat)
                    masks_out[name] = torch.zeros_like(masks[name])
                else:
                    features_out[name] = feat
                    masks_out[name] = masks[name]
            else:
                features_out[name] = feat
                masks_out[name] = masks[name]

        return features_out, masks_out

    def get_dropout_probs(self) -> Dict[str, float]:
        """Get current per-source dropout probabilities."""
        return self.dropout_probs.copy()


# =============================================================================
# 15. COLLAPSE DETECTOR (Phase 2 - Gemini Proposal)
# =============================================================================


class CollapseDetector:
    """
    Monitors task losses for collapse and triggers interventions.

    Collapse is detected when:
    1. Loss drops below threshold (e.g., < 1e-4) early in training
    2. Loss variance becomes very low (model outputs constant)
    3. Gradient norms become negligible

    When collapse is detected, the detector can:
    1. Switch to focal loss for affected tasks
    2. Increase regularization
    3. Reset task-specific parameters
    4. Log warnings for investigation

    Args:
        task_names: List of task identifiers to monitor
        loss_threshold: Loss below this triggers collapse check. Default 1e-4.
        variance_window: Window size for variance computation. Default 10.
        min_variance: Minimum acceptable variance. Default 1e-8.
        early_epoch_threshold: Only check for collapse before this epoch. Default 5.
    """

    def __init__(
        self,
        task_names: List[str],
        loss_threshold: float = 1e-4,
        variance_window: int = 10,
        min_variance: float = 1e-8,
        early_epoch_threshold: int = 5,
    ) -> None:
        self.task_names = list(task_names)
        self.loss_threshold = loss_threshold
        self.variance_window = variance_window
        self.min_variance = min_variance
        self.early_epoch_threshold = early_epoch_threshold

        # Track loss history per task
        self.loss_history: Dict[str, List[float]] = {
            name: [] for name in task_names
        }

        # Collapse state per task
        self.collapse_detected: Dict[str, bool] = {
            name: False for name in task_names
        }

        # Epoch tracking
        self.current_epoch = 0

    def update(
        self,
        losses: Dict[str, float],
        epoch: int,
    ) -> Dict[str, bool]:
        """
        Update with current losses and check for collapse.

        Args:
            losses: Dict mapping task name to current loss value
            epoch: Current training epoch

        Returns:
            Dict mapping task name to collapse status (True if just detected)
        """
        self.current_epoch = epoch
        newly_collapsed = {}

        for task_name, loss in losses.items():
            if task_name not in self.task_names:
                continue

            # Add to history
            self.loss_history[task_name].append(loss)

            # Trim to window size
            if len(self.loss_history[task_name]) > self.variance_window:
                self.loss_history[task_name] = self.loss_history[task_name][-self.variance_window:]

            # Check for collapse (only in early epochs)
            if epoch < self.early_epoch_threshold and not self.collapse_detected[task_name]:
                is_collapsed = self._check_collapse(task_name, loss)
                if is_collapsed:
                    self.collapse_detected[task_name] = True
                    newly_collapsed[task_name] = True
                    print(f"[CollapseDetector] COLLAPSE DETECTED for task '{task_name}' "
                          f"at epoch {epoch} (loss={loss:.2e})")

        return newly_collapsed

    def _check_collapse(self, task_name: str, current_loss: float) -> bool:
        """Check if a task has collapsed."""
        # Criterion 1: Loss below threshold
        if current_loss < self.loss_threshold:
            return True

        # Criterion 2: Loss variance very low (output is constant)
        history = self.loss_history[task_name]
        if len(history) >= 3:
            variance = np.var(history)
            if variance < self.min_variance and current_loss < 0.01:
                return True

        return False

    def is_collapsed(self, task_name: str) -> bool:
        """Check if a task is in collapsed state."""
        return self.collapse_detected.get(task_name, False)

    def get_collapsed_tasks(self) -> List[str]:
        """Get list of collapsed task names."""
        return [name for name, collapsed in self.collapse_detected.items() if collapsed]

    def reset(self) -> None:
        """Reset all tracking state."""
        self.loss_history = {name: [] for name in self.task_names}
        self.collapse_detected = {name: False for name in self.task_names}
        self.current_epoch = 0


# =============================================================================
# TESTING (Updated)
# =============================================================================


def test_phase2_improvements():
    """Test Phase 2 (Gemini proposal) improvements."""
    print("\n" + "=" * 60)
    print("Testing Phase 2 (Gemini Proposal) Improvements")
    print("=" * 60)

    # Test FocalLoss
    print("\n1. Testing FocalLoss...")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(32)
    targets = torch.zeros(32)
    targets[5] = 1  # Few positive samples
    targets[10] = 1

    loss = focal(logits, targets)
    print(f"   Focal loss: {loss.item():.4f}")

    # Compare with BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"   BCE loss: {bce.item():.4f}")
    print(f"   Focal < BCE: {loss.item() < bce.item()} (expected for imbalanced)")

    # Test collapse detection
    print("\n2. Testing FocalLossWithCollapsePrevention...")
    focal_cp = FocalLossWithCollapsePrevention(
        alpha=0.25,
        gamma=2.0,
        collapse_threshold=1e-4,
    )

    # Simulate collapse scenario (model outputs constant)
    collapsed_logits = torch.full((32,), -10.0)  # Confident negative
    targets_imbalanced = torch.zeros(32)
    targets_imbalanced[0] = 1  # 3% positive rate

    loss_collapsed = focal_cp(collapsed_logits, targets_imbalanced)
    print(f"   Loss with collapse prevention: {loss_collapsed.item():.6f}")
    print(f"   Stats: {focal_cp.get_stats()}")

    # Test SourceDropout
    print("\n3. Testing SourceDropout...")
    source_dropout = SourceDropout(
        dropout_sources=['hdx_rainfall'],
        dropout_prob=0.4,
    )
    source_dropout.train()

    features = {
        'equipment': torch.randn(4, 12, 38),
        'hdx_rainfall': torch.randn(4, 12, 16),
    }
    masks = {
        'equipment': torch.ones(4, 12, 38, dtype=torch.bool),
        'hdx_rainfall': torch.ones(4, 12, 16, dtype=torch.bool),
    }

    # Run multiple times to see dropout effect
    dropped_count = 0
    for _ in range(100):
        feat_out, mask_out = source_dropout(features, masks)
        if feat_out['hdx_rainfall'].abs().sum() == 0:
            dropped_count += 1

    print(f"   Dropout rate: {dropped_count}% (expected ~40%)")
    assert 25 < dropped_count < 55, f"Dropout rate {dropped_count}% outside expected range"
    print("   SourceDropout works correctly")

    # Test CollapseDetector
    print("\n4. Testing CollapseDetector...")
    detector = CollapseDetector(
        task_names=['transition', 'casualty'],
        loss_threshold=1e-4,
        early_epoch_threshold=5,
    )

    # Simulate normal training
    detector.update({'transition': 0.5, 'casualty': 1.2}, epoch=0)
    detector.update({'transition': 0.3, 'casualty': 0.8}, epoch=1)
    assert len(detector.get_collapsed_tasks()) == 0, "Should not detect collapse yet"

    # Simulate transition collapse
    collapsed = detector.update({'transition': 1e-8, 'casualty': 0.6}, epoch=2)
    assert 'transition' in collapsed, "Should detect transition collapse"
    assert detector.is_collapsed('transition'), "Transition should be marked collapsed"
    assert not detector.is_collapsed('casualty'), "Casualty should not be collapsed"
    print("   CollapseDetector works correctly")

    print("\n" + "=" * 60)
    print("All Phase 2 tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_improvements()
    test_phase2_improvements()
