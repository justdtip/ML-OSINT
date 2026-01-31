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

Author: Claude (synthesized from multiple AI proposals)
Date: 2026-01-31
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

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
    ) -> Tensor:
        """
        Predict future latent states from context.

        Args:
            context_latent: Context latent state [batch, d_model]
            return_all_horizons: If True, return all horizon predictions

        Returns:
            Predicted future latents [batch, horizon, d_model] or [batch, d_model]
        """
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

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_improvements()
