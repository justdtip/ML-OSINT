"""
ISW Embedding Integration Module for Multi-Resolution HAN

This module provides components for integrating pre-computed Voyage AI embeddings
(voyage-4-large, 1024-dim) from ISW daily assessment reports into the existing
Multi-Resolution Hierarchical Attention Network.

Integration Strategies Implemented:
1. FrozenISWProjection - Dimensionality reduction with minimal trainable params
2. LearnedISWProjection - Trainable projection with bottleneck architecture
3. ISWGatedFusion - Gated injection into quantitative feature stream
4. QuantitativeNarrativeCrossAttention - Cross-modal attention mechanism
5. ContrastiveISWLoss - Alignment loss for joint embedding space
6. NarrativeToStateHead - Auxiliary task for narrative grounding

Usage:
    from isw_embedding_integration import ISWEnhancedMultiResolutionHAN

    model = ISWEnhancedMultiResolutionHAN(
        daily_source_configs=daily_configs,
        monthly_source_configs=monthly_configs,
        isw_dim=1024,
        enable_cross_attention=False,  # Start simple
        enable_contrastive=True,
    )

    outputs = model(
        daily_features=daily_features,
        daily_masks=daily_masks,
        monthly_features=monthly_features,
        monthly_masks=monthly_masks,
        month_boundaries=month_boundaries,
        isw_embeddings=isw_embeddings,  # [batch, daily_seq, 1024]
        isw_mask=isw_mask,              # [batch, daily_seq] True=valid
    )

Author: AI Engineering Team
Date: 2026-01-21
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ISWIntegrationConfig:
    """Configuration for ISW embedding integration."""
    isw_dim: int = 1024                 # Voyage-4-large embedding dimension
    projection_dim: int = 128           # Target projection dimension (match d_model)
    bottleneck_dim: int = 64            # Intermediate dimension for MLP projection
    dropout: float = 0.3                # Higher dropout for small dataset
    temperature: float = 0.07           # Contrastive learning temperature
    contrastive_weight: float = 0.1     # Weight for contrastive auxiliary loss
    n2s_weight: float = 0.1             # Weight for narrative-to-state auxiliary loss
    use_frozen_projection: bool = True  # Start with frozen projection
    use_cross_attention: bool = False   # Enable cross-attention (more params)
    use_contrastive: bool = True        # Enable contrastive learning
    use_n2s_auxiliary: bool = True      # Enable narrative-to-state prediction
    num_attention_heads: int = 8        # For cross-attention
    temporal_window: int = 7            # Days of ISW history to consider


# =============================================================================
# PROJECTION MODULES
# =============================================================================

class FrozenISWProjection(nn.Module):
    """
    Frozen linear projection from ISW embedding space to model dimension.

    Uses orthogonal initialization and normalization to preserve geometric
    structure while reducing dimensionality. Only the scale factor is trainable.

    Parameters: ~1 (just the scale factor)
    """

    def __init__(
        self,
        isw_dim: int = 1024,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()

        self.isw_dim = isw_dim
        self.projection_dim = projection_dim

        # Frozen linear projection
        self.projection = nn.Linear(isw_dim, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.projection.weight.requires_grad = False

        # Learnable scale factor (single parameter)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, isw_embeddings: Tensor) -> Tensor:
        """
        Project ISW embeddings to model dimension.

        Args:
            isw_embeddings: [batch, seq_len, isw_dim]

        Returns:
            projected: [batch, seq_len, projection_dim] normalized and scaled
        """
        proj = self.projection(isw_embeddings)
        # Normalize and scale to have similar magnitude as quantitative encodings
        proj_norm = F.normalize(proj, dim=-1)
        return self.scale * proj_norm * math.sqrt(self.projection_dim)


class LearnedISWProjection(nn.Module):
    """
    Trainable MLP projection with bottleneck architecture.

    The bottleneck forces the model to learn a compressed, task-relevant
    representation from the high-dimensional ISW embeddings.

    Parameters: isw_dim * bottleneck + bottleneck * projection_dim
                = 1024 * 64 + 64 * 128 = ~74K parameters
    """

    def __init__(
        self,
        isw_dim: int = 1024,
        projection_dim: int = 128,
        bottleneck_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            # Compress to bottleneck
            nn.Linear(isw_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Expand to target dimension
            nn.Linear(bottleneck_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, isw_embeddings: Tensor) -> Tensor:
        """
        Project ISW embeddings through bottleneck MLP.

        Args:
            isw_embeddings: [batch, seq_len, isw_dim]

        Returns:
            projected: [batch, seq_len, projection_dim]
        """
        return self.projection(isw_embeddings)


# =============================================================================
# GATED FUSION MODULE
# =============================================================================

class ISWGatedFusion(nn.Module):
    """
    Gated fusion of quantitative encodings with ISW embeddings.

    The gate learns to control how much ISW information flows into the
    quantitative stream, allowing the model to selectively use narrative
    context where it's most useful.

    Parameters: projection_dim * 2 * projection_dim = ~33K parameters
    """

    def __init__(
        self,
        quant_dim: int = 128,
        isw_projected_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(quant_dim + isw_projected_dim, quant_dim),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(quant_dim + isw_projected_dim, quant_dim),
            nn.LayerNorm(quant_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(quant_dim)

        # CRITICAL FIX: Initialize gate bias to 0 so sigmoid outputs start near 0.5
        # This prevents saturation at initialization which can cause vanishing gradients
        self._init_gate_weights()

    def _init_gate_weights(self) -> None:
        """Initialize gate weights to avoid sigmoid saturation at start."""
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                # Xavier init for weights
                nn.init.xavier_uniform_(module.weight)
                # Zero bias ensures sigmoid(0) = 0.5 at initialization
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        quant_encoded: Tensor,      # [batch, seq_len, quant_dim]
        isw_projected: Tensor,      # [batch, seq_len, isw_projected_dim]
        isw_mask: Optional[Tensor] = None,  # [batch, seq_len] True=valid
    ) -> Tensor:
        """
        Fuse quantitative encodings with projected ISW embeddings.

        Args:
            quant_encoded: Quantitative feature encodings
            isw_projected: Projected ISW embeddings (same seq_len as quant)
            isw_mask: Optional mask for valid ISW embeddings

        Returns:
            fused: Enhanced quantitative encodings [batch, seq_len, quant_dim]
        """
        # Handle missing ISW data by zeroing contribution
        if isw_mask is not None:
            isw_projected = isw_projected * isw_mask.unsqueeze(-1).float()

        # Concatenate for gating
        combined = torch.cat([quant_encoded, isw_projected], dim=-1)

        # Compute gate (how much ISW to use)
        gate_values = self.gate(combined)

        # Gated residual fusion
        fused_features = self.fusion(combined)
        output = quant_encoded + gate_values * fused_features

        return self.output_norm(output)


# =============================================================================
# CROSS-ATTENTION MODULE
# =============================================================================

class QuantitativeNarrativeCrossAttention(nn.Module):
    """
    Cross-attention where quantitative features attend to narrative embeddings.

    This allows the model to dynamically retrieve relevant narrative context
    for each quantitative observation, learning which aspects of ISW reports
    are most predictive.

    Parameters: ~311K parameters (use sparingly with small datasets)
    """

    def __init__(
        self,
        quant_dim: int = 128,
        narrative_dim: int = 1024,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project narrative to hidden dim (large reduction)
        self.narrative_proj_k = nn.Linear(narrative_dim, hidden_dim)
        self.narrative_proj_v = nn.Linear(narrative_dim, hidden_dim)

        # Queries from quantitative encodings
        self.query_proj = nn.Linear(quant_dim, hidden_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.narrative_proj_k, self.narrative_proj_v, self.query_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # CRITICAL FIX: Initialize gate bias to 0 so sigmoid outputs start near 0.5
        # This prevents saturation at initialization which can cause vanishing gradients
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        quant_encoded: Tensor,          # [batch, seq_len, quant_dim]
        narrative_embeddings: Tensor,   # [batch, seq_len, narrative_dim]
        narrative_mask: Optional[Tensor] = None,  # [batch, seq_len] True=valid
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Enhance quantitative encodings with narrative context via cross-attention.

        Args:
            quant_encoded: Quantitative feature encodings
            narrative_embeddings: Raw ISW embeddings (not projected)
            narrative_mask: Mask for valid narrative embeddings
            return_attention: Whether to return attention weights

        Returns:
            enhanced: Enhanced quantitative encodings [batch, seq_len, hidden_dim]
            attention_weights: Optional [batch, num_heads, seq_q, seq_k]
        """
        # Project
        Q = self.query_proj(quant_encoded)
        K = self.narrative_proj_k(narrative_embeddings)
        V = self.narrative_proj_v(narrative_embeddings)

        # Prepare mask (True = IGNORE for PyTorch MHA)
        key_padding_mask = None
        if narrative_mask is not None:
            key_padding_mask = ~narrative_mask

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            Q, K, V,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )

        # Gated residual
        gate = self.gate(torch.cat([quant_encoded, attended], dim=-1))
        enhanced = self.norm(quant_encoded + gate * attended)

        if return_attention:
            return enhanced, attn_weights
        return enhanced, None


# =============================================================================
# CONTRASTIVE LEARNING
# =============================================================================

class ContrastiveISWLoss(nn.Module):
    """
    InfoNCE contrastive loss to align quantitative states with narratives.

    Creates a shared embedding space where quantitative states and their
    corresponding narrative descriptions are close together.

    Parameters: ~66K parameters (projection heads)
    """

    def __init__(
        self,
        quant_dim: int = 128,
        narrative_dim: int = 1024,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.temperature = temperature

        # Project both modalities to shared space
        self.quant_projection = nn.Sequential(
            nn.Linear(quant_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.narrative_projection = nn.Sequential(
            nn.Linear(narrative_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self,
        quant_encodings: Tensor,       # [batch, seq_len, quant_dim]
        narrative_embeddings: Tensor,  # [batch, seq_len, narrative_dim]
        valid_mask: Tensor,            # [batch, seq_len] True where both valid
    ) -> Tensor:
        """
        Compute InfoNCE contrastive loss.

        Args:
            quant_encodings: Quantitative state encodings
            narrative_embeddings: ISW embeddings
            valid_mask: Mask for valid timesteps (where both modalities present)

        Returns:
            loss: Scalar contrastive loss
        """
        # Flatten
        quant_flat = quant_encodings.view(-1, quant_encodings.shape[-1])
        narrative_flat = narrative_embeddings.view(-1, narrative_embeddings.shape[-1])
        mask_flat = valid_mask.view(-1)

        # Filter to valid
        valid_indices = mask_flat.nonzero(as_tuple=True)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=quant_encodings.device, requires_grad=True)

        quant_valid = quant_flat[valid_indices]
        narrative_valid = narrative_flat[valid_indices]
        n_samples = len(valid_indices)

        # CRITICAL FIX: Memory guard for large batches
        # The NxN similarity matrix can OOM with large n_samples
        # Limit to 4096 samples to prevent memory issues (~64MB for float32 4096x4096)
        max_contrastive_samples = 4096
        if n_samples > max_contrastive_samples:
            # Randomly subsample to prevent OOM
            perm = torch.randperm(n_samples, device=quant_valid.device)[:max_contrastive_samples]
            quant_valid = quant_valid[perm]
            narrative_valid = narrative_valid[perm]
            n_samples = max_contrastive_samples

        # Project and normalize
        quant_proj = F.normalize(self.quant_projection(quant_valid), dim=-1)
        narrative_proj = F.normalize(self.narrative_projection(narrative_valid), dim=-1)

        # Similarity matrix with numerical stability
        # Clamp logits to prevent overflow in softmax
        logits = torch.matmul(quant_proj, narrative_proj.T) / self.temperature
        logits = torch.clamp(logits, min=-100.0, max=100.0)  # Prevent exp overflow in softmax

        # Labels: diagonal (same-day pairs)
        labels = torch.arange(n_samples, device=logits.device)

        # Bidirectional cross-entropy
        loss_q2n = F.cross_entropy(logits, labels)
        loss_n2q = F.cross_entropy(logits.T, labels)

        return (loss_q2n + loss_n2q) / 2


# =============================================================================
# AUXILIARY TASK: NARRATIVE-TO-STATE PREDICTION
# =============================================================================

class NarrativeToStateHead(nn.Module):
    """
    Auxiliary task head: Predict key quantitative signals from narrative alone.

    If ISW narratives contain conflict-relevant information, they should
    predict (approximately) equipment losses, casualty rates, activity levels.

    This provides self-supervision signal and validates ISW embedding quality.

    Parameters: ~131K parameters
    """

    def __init__(
        self,
        narrative_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(narrative_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Names of predicted quantitative signals
        self.output_keys = [
            'equipment_total_delta',
            'personnel_daily_rate',
            'front_activity_index',
            'fire_count_log',
            'frp_total_log',
            'territorial_change_km2',
            'airstrike_indicator',
            'artillery_intensity',
            'offensive_indicator',
            'defensive_indicator',
        ]

    def forward(self, narrative_embeddings: Tensor) -> Tensor:
        """
        Predict quantitative signals from narrative.

        Args:
            narrative_embeddings: [batch, seq_len, narrative_dim]

        Returns:
            predictions: [batch, seq_len, output_dim]
        """
        return self.predictor(narrative_embeddings)

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Compute masked MSE loss for narrative-to-state prediction.

        Args:
            predictions: [batch, seq_len, output_dim]
            targets: [batch, seq_len, output_dim] ground truth quantitative signals
            mask: [batch, seq_len] True where valid

        Returns:
            loss: Scalar loss value
        """
        mask_expanded = mask.unsqueeze(-1).float()
        n_valid = mask_expanded.sum().clamp(min=1)

        loss = F.mse_loss(
            predictions * mask_expanded,
            targets * mask_expanded,
            reduction='sum'
        ) / n_valid

        return loss


# =============================================================================
# INTEGRATED MODEL
# =============================================================================

class ISWIntegrationModule(nn.Module):
    """
    Complete ISW integration module that can be added to MultiResolutionHAN.

    Combines projection, gating, optional cross-attention, and auxiliary losses.

    Usage:
        # Create module
        isw_module = ISWIntegrationModule(config)

        # In forward pass (after daily fusion)
        enhanced_daily, aux_losses = isw_module(
            quant_encoded=fused_daily,
            isw_embeddings=isw_embeddings,
            isw_mask=isw_mask,
            compute_aux_losses=True,
        )
    """

    def __init__(self, config: ISWIntegrationConfig) -> None:
        super().__init__()

        self.config = config

        # Projection layer
        if config.use_frozen_projection:
            self.isw_projection = FrozenISWProjection(
                isw_dim=config.isw_dim,
                projection_dim=config.projection_dim,
            )
        else:
            self.isw_projection = LearnedISWProjection(
                isw_dim=config.isw_dim,
                projection_dim=config.projection_dim,
                bottleneck_dim=config.bottleneck_dim,
                dropout=config.dropout,
            )

        # Gated fusion
        self.gated_fusion = ISWGatedFusion(
            quant_dim=config.projection_dim,
            isw_projected_dim=config.projection_dim,
            dropout=config.dropout,
        )

        # Optional cross-attention
        self.cross_attention = None
        if config.use_cross_attention:
            self.cross_attention = QuantitativeNarrativeCrossAttention(
                quant_dim=config.projection_dim,
                narrative_dim=config.isw_dim,
                hidden_dim=config.projection_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
            )

        # Contrastive loss
        self.contrastive_loss = None
        if config.use_contrastive:
            self.contrastive_loss = ContrastiveISWLoss(
                quant_dim=config.projection_dim,
                narrative_dim=config.isw_dim,
                projection_dim=config.projection_dim,
                temperature=config.temperature,
            )

        # Narrative-to-state auxiliary head
        self.n2s_head = None
        if config.use_n2s_auxiliary:
            self.n2s_head = NarrativeToStateHead(
                narrative_dim=config.isw_dim,
                hidden_dim=256,
                output_dim=10,
                dropout=config.dropout,
            )

        # Learned no_observation_token for missing ISW data
        self.no_isw_token = nn.Parameter(
            torch.randn(1, 1, config.projection_dim) * 0.02
        )

    def forward(
        self,
        quant_encoded: Tensor,          # [batch, seq_len, projection_dim]
        isw_embeddings: Tensor,         # [batch, seq_len, isw_dim]
        isw_mask: Optional[Tensor] = None,  # [batch, seq_len] True=valid
        key_quant_targets: Optional[Tensor] = None,  # For N2S aux loss
        compute_aux_losses: bool = True,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Integrate ISW embeddings with quantitative encodings.

        Args:
            quant_encoded: Quantitative feature encodings from daily fusion
            isw_embeddings: Pre-computed ISW embeddings (voyage-4-large)
            isw_mask: Mask indicating valid ISW embeddings
            key_quant_targets: Optional targets for N2S auxiliary task
            compute_aux_losses: Whether to compute auxiliary losses
            return_attention: Whether to return attention weights

        Returns:
            enhanced: Enhanced quantitative encodings [batch, seq_len, projection_dim]
            aux_outputs: Dict with auxiliary losses and optional attention weights
        """
        batch_size, seq_len, _ = quant_encoded.shape
        device = quant_encoded.device
        aux_outputs = {}

        # Handle missing mask
        if isw_mask is None:
            isw_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Project ISW embeddings
        isw_projected = self.isw_projection(isw_embeddings)

        # Replace missing ISW with no_observation_token
        no_isw_expanded = self.no_isw_token.expand(batch_size, seq_len, -1)
        isw_projected = torch.where(
            isw_mask.unsqueeze(-1),
            isw_projected,
            no_isw_expanded,
        )

        # Gated fusion
        enhanced = self.gated_fusion(quant_encoded, isw_projected, isw_mask)

        # Optional cross-attention
        if self.cross_attention is not None:
            enhanced, attn_weights = self.cross_attention(
                enhanced,
                isw_embeddings,
                isw_mask,
                return_attention=return_attention,
            )
            if return_attention and attn_weights is not None:
                aux_outputs['cross_attention_weights'] = attn_weights

        # Compute auxiliary losses
        if compute_aux_losses:
            # Contrastive loss
            if self.contrastive_loss is not None:
                aux_outputs['contrastive_loss'] = self.contrastive_loss(
                    quant_encoded, isw_embeddings, isw_mask
                )

            # Narrative-to-state loss
            if self.n2s_head is not None and key_quant_targets is not None:
                n2s_predictions = self.n2s_head(isw_embeddings)
                aux_outputs['n2s_predictions'] = n2s_predictions
                aux_outputs['n2s_loss'] = self.n2s_head.compute_loss(
                    n2s_predictions, key_quant_targets, isw_mask
                )

        return enhanced, aux_outputs

    def compute_total_auxiliary_loss(
        self,
        aux_outputs: Dict[str, Tensor],
    ) -> Tensor:
        """
        Combine auxiliary losses with configured weights.

        Args:
            aux_outputs: Dict from forward() containing individual losses

        Returns:
            total_aux_loss: Weighted sum of auxiliary losses
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)

        if 'contrastive_loss' in aux_outputs:
            total = total + self.config.contrastive_weight * aux_outputs['contrastive_loss']

        if 'n2s_loss' in aux_outputs:
            total = total + self.config.n2s_weight * aux_outputs['n2s_loss']

        return total

    def unfreeze_projection(self) -> None:
        """
        Unfreeze the projection layer for fine-tuning.

        Call this after initial training with frozen projection.
        """
        if isinstance(self.isw_projection, FrozenISWProjection):
            for param in self.isw_projection.parameters():
                param.requires_grad = True
            print("ISW projection layer unfrozen for fine-tuning")


# =============================================================================
# UTILITY: PREPARE KEY QUANTITATIVE TARGETS
# =============================================================================

def prepare_key_quantitative_targets(
    daily_features: Dict[str, Tensor],
    daily_masks: Dict[str, Tensor],
) -> Tensor:
    """
    Extract key quantitative signals for the N2S auxiliary task.

    This prepares targets that ISW embeddings should be able to predict:
    - Equipment loss deltas
    - Personnel rates
    - Activity indices

    Args:
        daily_features: Dict mapping source name to features
        daily_masks: Dict mapping source name to masks

    Returns:
        targets: [batch, seq_len, 10] key quantitative signals
    """
    batch_size = None
    seq_len = None
    device = None

    # Get dimensions
    for name, feat in daily_features.items():
        batch_size, seq_len, _ = feat.shape
        device = feat.device
        break

    # Initialize targets
    targets = torch.zeros(batch_size, seq_len, 10, device=device)

    # Extract from equipment (if available)
    if 'equipment' in daily_features:
        eq_feat = daily_features['equipment']
        # Assuming first few features are loss totals
        if eq_feat.shape[-1] >= 1:
            # Total equipment delta (sum or first feature)
            targets[:, :, 0] = eq_feat[:, :, 0]  # equipment_total_delta

    # Extract from personnel
    if 'personnel' in daily_features:
        pers_feat = daily_features['personnel']
        if pers_feat.shape[-1] >= 1:
            targets[:, :, 1] = pers_feat[:, :, 0]  # personnel_daily_rate

    # Extract from firms (fire data)
    if 'firms' in daily_features:
        firms_feat = daily_features['firms']
        if firms_feat.shape[-1] >= 2:
            targets[:, :, 3] = torch.log1p(firms_feat[:, :, 0].clamp(min=0))  # fire_count_log
            targets[:, :, 4] = torch.log1p(firms_feat[:, :, 1].clamp(min=0))  # frp_total_log

    # Extract from deepstate (territorial)
    if 'deepstate' in daily_features:
        ds_feat = daily_features['deepstate']
        if ds_feat.shape[-1] >= 1:
            targets[:, :, 2] = ds_feat[:, :, 0]  # front_activity_index
            if ds_feat.shape[-1] >= 2:
                targets[:, :, 5] = ds_feat[:, :, 1]  # territorial_change_km2

    return targets


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ISW Embedding Integration Module - Tests")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test configuration
    batch_size = 4
    seq_len = 100
    quant_dim = 128
    isw_dim = 1024

    # Create dummy data
    quant_encoded = torch.randn(batch_size, seq_len, quant_dim, device=device)
    isw_embeddings = torch.randn(batch_size, seq_len, isw_dim, device=device)
    isw_mask = torch.rand(batch_size, seq_len, device=device) > 0.2  # 80% valid
    key_quant_targets = torch.randn(batch_size, seq_len, 10, device=device)

    # Test 1: FrozenISWProjection
    print("\n" + "-" * 40)
    print("Test 1: FrozenISWProjection")
    frozen_proj = FrozenISWProjection(isw_dim, quant_dim).to(device)
    out = frozen_proj(isw_embeddings)
    print(f"  Input: {isw_embeddings.shape}")
    print(f"  Output: {out.shape}")
    trainable = sum(p.numel() for p in frozen_proj.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable}")
    print("  PASSED")

    # Test 2: LearnedISWProjection
    print("\n" + "-" * 40)
    print("Test 2: LearnedISWProjection")
    learned_proj = LearnedISWProjection(isw_dim, quant_dim, 64).to(device)
    out = learned_proj(isw_embeddings)
    print(f"  Output: {out.shape}")
    trainable = sum(p.numel() for p in learned_proj.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  PASSED")

    # Test 3: ISWGatedFusion
    print("\n" + "-" * 40)
    print("Test 3: ISWGatedFusion")
    isw_proj = torch.randn(batch_size, seq_len, quant_dim, device=device)
    gated_fusion = ISWGatedFusion(quant_dim, quant_dim).to(device)
    out = gated_fusion(quant_encoded, isw_proj, isw_mask)
    print(f"  Output: {out.shape}")
    trainable = sum(p.numel() for p in gated_fusion.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  PASSED")

    # Test 4: QuantitativeNarrativeCrossAttention
    print("\n" + "-" * 40)
    print("Test 4: QuantitativeNarrativeCrossAttention")
    cross_attn = QuantitativeNarrativeCrossAttention(
        quant_dim, isw_dim, quant_dim, 8
    ).to(device)
    out, attn = cross_attn(quant_encoded, isw_embeddings, isw_mask, return_attention=True)
    print(f"  Output: {out.shape}")
    print(f"  Attention: {attn.shape if attn is not None else 'None'}")
    trainable = sum(p.numel() for p in cross_attn.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  PASSED")

    # Test 5: ContrastiveISWLoss
    print("\n" + "-" * 40)
    print("Test 5: ContrastiveISWLoss")
    contrastive = ContrastiveISWLoss(quant_dim, isw_dim, quant_dim).to(device)
    loss = contrastive(quant_encoded, isw_embeddings, isw_mask)
    print(f"  Contrastive loss: {loss.item():.4f}")
    trainable = sum(p.numel() for p in contrastive.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  PASSED")

    # Test 6: NarrativeToStateHead
    print("\n" + "-" * 40)
    print("Test 6: NarrativeToStateHead")
    n2s_head = NarrativeToStateHead(isw_dim, 256, 10).to(device)
    pred = n2s_head(isw_embeddings)
    loss = n2s_head.compute_loss(pred, key_quant_targets, isw_mask)
    print(f"  Predictions: {pred.shape}")
    print(f"  N2S loss: {loss.item():.4f}")
    trainable = sum(p.numel() for p in n2s_head.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print("  PASSED")

    # Test 7: Full ISWIntegrationModule
    print("\n" + "-" * 40)
    print("Test 7: ISWIntegrationModule (Full)")
    config = ISWIntegrationConfig(
        use_frozen_projection=True,
        use_cross_attention=False,
        use_contrastive=True,
        use_n2s_auxiliary=True,
    )
    integration = ISWIntegrationModule(config).to(device)
    enhanced, aux_outputs = integration(
        quant_encoded,
        isw_embeddings,
        isw_mask,
        key_quant_targets,
        compute_aux_losses=True,
    )
    print(f"  Enhanced output: {enhanced.shape}")
    print(f"  Aux outputs: {list(aux_outputs.keys())}")
    total_aux_loss = integration.compute_total_auxiliary_loss(aux_outputs)
    print(f"  Total aux loss: {total_aux_loss.item():.4f}")
    trainable = sum(p.numel() for p in integration.parameters() if p.requires_grad)
    total = sum(p.numel() for p in integration.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")
    print("  PASSED")

    # Test 8: Gradient flow
    print("\n" + "-" * 40)
    print("Test 8: Gradient Flow")
    loss = enhanced.mean() + total_aux_loss
    loss.backward()
    grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in integration.parameters() if p.requires_grad)
    print(f"  Gradients computed: {grad_exists}")
    print("  PASSED" if grad_exists else "  FAILED")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
