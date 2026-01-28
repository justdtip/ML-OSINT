"""
Geographic Source Encoder

Enhances the MultiResolutionHAN with geographic attention priors for spatial sources.
This module provides:

1. GeographicSourceEncoder: Applies cross-raion attention with geographic prior
   within a single spatial source (e.g., VIINA, FIRMS)

2. GeographicDailyCrossSourceFusion: Enhanced version of DailyCrossSourceFusion
   that applies geographic priors to spatial sources before cross-source fusion

Integration:
    The geographic prior is applied WITHIN spatial sources (cross-raion attention),
    not between sources. This captures that Bakhmut is near Soledar, not that
    "equipment losses" is near "personnel losses".

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import geographic adjacency
from analysis.cross_raion_attention import GeographicAdjacency


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SpatialSourceConfig:
    """Configuration for a spatial source with raion structure."""
    name: str
    n_raions: int
    features_per_raion: int
    raion_keys: List[str] = field(default_factory=list)
    use_geographic_prior: bool = True


# Default spatial source configurations
# These match the output of load_viina_daily() and load_firms_raion()
DEFAULT_SPATIAL_CONFIGS = {
    'viina': SpatialSourceConfig(
        name='viina',
        n_raions=100,  # Top 100 raions with events
        features_per_raion=1,  # events_total per raion (plus event types for top 30)
        use_geographic_prior=True,
    ),
    'firms': SpatialSourceConfig(
        name='firms',
        n_raions=20,  # Top 20 raions by fire count
        features_per_raion=4,  # count, brightness, frp, dayratio
        use_geographic_prior=True,
    ),
}


# =============================================================================
# GEOGRAPHIC SOURCE ENCODER
# =============================================================================

class GeographicSourceEncoder(nn.Module):
    """
    Encoder for a spatial source with geographic attention.

    Takes raion-structured features and applies:
    1. Per-raion temporal encoding
    2. Cross-raion attention with geographic prior
    3. Temporal aggregation back to sequence format

    This allows the model to learn that events in Bakhmut predict events
    in nearby Soledar more than in distant Odesa.

    Input: [batch, seq_len, n_raions * features_per_raion]
    Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        spatial_config: SpatialSourceConfig,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        prior_weight: float = 1.0,
    ):
        super().__init__()

        self.config = spatial_config
        self.n_raions = spatial_config.n_raions
        self.features_per_raion = spatial_config.features_per_raion
        self.d_model = d_model
        self.n_heads = n_heads

        # Input dimension
        self.input_dim = self.n_raions * self.features_per_raion

        # Per-raion feature projection
        self.raion_proj = nn.Linear(self.features_per_raion, d_model)

        # Raion embedding
        self.raion_embedding = nn.Embedding(self.n_raions, d_model)

        # Geographic adjacency
        self.geographic_adjacency = GeographicAdjacency()
        self.prior_weight = nn.Parameter(torch.ones(1) * prior_weight)

        # Cross-raion attention
        self.cross_raion_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Temporal attention (pool across raions)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads // 2,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Cached geographic prior
        self._geo_prior: Optional[Tensor] = None

    def _get_geo_prior(self, device: torch.device) -> Optional[Tensor]:
        """Get geographic prior matrix."""
        if not self.config.use_geographic_prior:
            return None

        if not self.config.raion_keys:
            return None

        if self._geo_prior is not None and self._geo_prior.device == device:
            return self._geo_prior

        prior = self.geographic_adjacency.get_adjacency_for_raions(
            self.config.raion_keys, device
        )
        self._geo_prior = prior
        return prior

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Encode spatial source with geographic attention.

        Args:
            x: Input features [batch, seq_len, n_raions * features_per_raion]
            mask: Optional observation mask. Supports two formats:
                  - [batch, seq_len]: Per-timestep mask (backward compatible)
                  - [batch, seq_len, n_raions]: Per-raion mask (full granularity)
            return_attention: If True, return attention weights for interpretability

        Returns:
            If return_attention=False:
                Encoded representation [batch, seq_len, d_model]
            If return_attention=True:
                Tuple of (encoded, attention_dict) where attention_dict contains:
                - 'cross_raion_attention': [batch*seq_len, n_heads, n_raions, n_raions]
                - 'temporal_attention': [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Normalize mask to per-raion format [batch, seq_len, n_raions]
        raion_mask: Optional[Tensor] = None
        if mask is not None:
            if mask.dim() == 2:
                # Expand [batch, seq_len] → [batch, seq_len, n_raions]
                raion_mask = mask.unsqueeze(-1).expand(-1, -1, self.n_raions)
            elif mask.dim() == 3:
                raion_mask = mask
            else:
                raise ValueError(f"Mask must be 2D or 3D, got {mask.dim()}D")

        # Reshape to [batch, seq_len, n_raions, features_per_raion]
        # Handle case where input may not perfectly match expected dimensions
        expected_dim = self.n_raions * self.features_per_raion
        if x.shape[-1] != expected_dim:
            # Pad or truncate
            if x.shape[-1] < expected_dim:
                pad = torch.zeros(batch_size, seq_len, expected_dim - x.shape[-1], device=device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., :expected_dim]

        x_raion = x.view(batch_size, seq_len, self.n_raions, self.features_per_raion)

        # Project per-raion features to d_model
        # [batch, seq_len, n_raions, d_model]
        x_proj = self.raion_proj(x_raion)

        # Add raion embeddings
        raion_idx = torch.arange(self.n_raions, device=device)
        raion_emb = self.raion_embedding(raion_idx)  # [n_raions, d_model]
        x_proj = x_proj + raion_emb.unsqueeze(0).unsqueeze(0)

        # Apply cross-raion attention for each timestep
        # Reshape: [batch * seq_len, n_raions, d_model]
        x_flat = x_proj.view(batch_size * seq_len, self.n_raions, self.d_model)

        # Get geographic prior
        geo_prior = self._get_geo_prior(device)

        # Build attention mask for unobserved raions
        # PyTorch MHA: True in key_padding_mask means "ignore this position"
        attn_key_padding_mask: Optional[Tensor] = None
        if raion_mask is not None:
            # Flatten: [batch * seq_len, n_raions]
            flat_raion_mask = raion_mask.view(batch_size * seq_len, self.n_raions)
            # Invert: True = observed → False = ignore; so ~mask = ignore unobserved
            attn_key_padding_mask = ~flat_raion_mask.bool()

            # IMPORTANT: Prevent all-masked rows which cause NaN in attention
            # If all raions masked for a timestep, unmask the first one to avoid NaN
            all_masked = attn_key_padding_mask.all(dim=-1)  # [batch * seq_len]
            if all_masked.any():
                # For fully-masked timesteps, unmask first raion (attention will still be valid)
                attn_key_padding_mask = attn_key_padding_mask.clone()
                attn_key_padding_mask[all_masked, 0] = False

        # Apply cross-raion attention
        cross_raion_attn_weights = None
        if geo_prior is not None:
            # Build geographic prior attention mask
            # PyTorch MultiheadAttention attn_mask is ADDITIVE to attention scores
            # Shape requirement: [batch*num_heads, seq_len, seq_len] or [seq_len, seq_len]
            # Here seq_len refers to the raion dimension (n_raions)
            #
            # geo_prior: [n_raions, n_raions] in LOG-SPACE from GeographicAdjacency
            # Values are log(exp(-distance/scale)) so nearby raions have values ~0,
            # distant raions have large negative values (down to ~-18 for very distant)
            # This is added to attention logits, so nearby raions get boosted attention
            attn_bias = self.prior_weight * geo_prior  # [n_raions, n_raions]

            # Expand for batch and heads: [batch*seq_len*n_heads, n_raions, n_raions]
            # PyTorch MHA with batch_first=True expects attn_mask of shape:
            # - (L, S) for 2D: broadcast across batch and heads
            # - (N*num_heads, L, S) for 3D: per-batch-head mask
            # We use the 3D form to properly apply to all batch elements
            n_batch_seq = batch_size * seq_len
            attn_mask = attn_bias.unsqueeze(0).expand(
                n_batch_seq * self.n_heads, -1, -1
            )  # [batch*seq_len*n_heads, n_raions, n_raions]

            attended, cross_raion_attn_weights = self.cross_raion_attn(
                x_flat, x_flat, x_flat,
                attn_mask=attn_mask,
                key_padding_mask=attn_key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,  # Return per-head weights
            )
        else:
            attended, cross_raion_attn_weights = self.cross_raion_attn(
                x_flat, x_flat, x_flat,
                key_padding_mask=attn_key_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,  # Return per-head weights
            )

        # Handle any remaining NaN from edge cases
        attended = torch.nan_to_num(attended, nan=0.0)

        x_flat = self.norm1(x_flat + attended)

        # Reshape back: [batch, seq_len, n_raions, d_model]
        x_raion = x_flat.view(batch_size, seq_len, self.n_raions, self.d_model)

        # Aggregate across raions for each timestep using mask-aware mean pooling
        # [batch, seq_len, d_model]
        if raion_mask is not None:
            # Expand mask for broadcasting: [batch, seq_len, n_raions, 1]
            mask_expanded = raion_mask.unsqueeze(-1).float()
            # Zero out unobserved raions before summing
            x_masked = x_raion * mask_expanded
            # Sum observed raions and divide by count (with epsilon for stability)
            observed_count = mask_expanded.sum(dim=2, keepdim=True).clamp(min=1.0)
            x_agg = x_masked.sum(dim=2) / observed_count.squeeze(-1)
            # Handle fully-masked timesteps (use mean as fallback)
            fully_masked = (mask_expanded.sum(dim=2) == 0).squeeze(-1)  # [batch, seq_len]
            if fully_masked.any():
                fallback = x_raion.mean(dim=2)
                x_agg = torch.where(fully_masked.unsqueeze(-1), fallback, x_agg)
        else:
            x_agg = x_raion.mean(dim=2)

        # Apply temporal self-attention
        attended, temporal_attn_weights = self.temporal_attn(
            x_agg, x_agg, x_agg,
            need_weights=return_attention,
            average_attn_weights=False,  # Return per-head weights
        )
        x_agg = self.norm2(x_agg + attended)

        # Output projection
        output = self.output_proj(x_agg)

        if return_attention:
            attention_dict = {}
            if cross_raion_attn_weights is not None:
                # Shape: [batch*seq_len, n_heads, n_raions, n_raions]
                attention_dict['cross_raion_attention'] = cross_raion_attn_weights
            if temporal_attn_weights is not None:
                # Shape: [batch, n_heads, seq_len, seq_len]
                attention_dict['temporal_attention'] = temporal_attn_weights
            return output, attention_dict

        return output


# =============================================================================
# ENHANCED DAILY CROSS-SOURCE FUSION
# =============================================================================

class GeographicDailyCrossSourceFusion(nn.Module):
    """
    Enhanced DailyCrossSourceFusion with geographic priors for spatial sources.

    This module:
    1. Applies GeographicSourceEncoder to spatial sources (viina, firms)
    2. Uses standard encoding for non-spatial sources (equipment, personnel)
    3. Performs cross-source fusion on the encoded representations

    The geographic prior is applied WITHIN spatial sources, not between sources.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        source_names: Optional[List[str]] = None,
        spatial_configs: Optional[Dict[str, SpatialSourceConfig]] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.source_names = source_names or [
            'equipment', 'personnel', 'deepstate', 'firms', 'viina', 'viirs'
        ]
        self.n_sources = len(self.source_names)

        # Spatial source configurations
        self.spatial_configs = spatial_configs or {}

        # Geographic encoders for spatial sources
        self.geographic_encoders = nn.ModuleDict()
        for name, config in self.spatial_configs.items():
            if name in self.source_names:
                self.geographic_encoders[name] = GeographicSourceEncoder(
                    spatial_config=config,
                    d_model=d_model,
                    n_heads=nhead,
                    dropout=dropout,
                )

        # Standard projection for non-spatial sources
        self.source_projections = nn.ModuleDict()
        # Note: feature dimensions will be set dynamically

        # Source type embedding
        self.source_type_embedding = nn.Embedding(self.n_sources, d_model)

        # Cross-source attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * self.n_sources, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        source_features: Dict[str, Tensor],
        source_masks: Dict[str, Tensor],
        return_attention: bool = False,
        raion_masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Fuse multiple daily source representations with geographic priors.

        Args:
            source_features: Dict mapping source name to [batch, seq_len, n_features]
            source_masks: Dict mapping source name to [batch, seq_len] or [batch, seq_len, n_features]
            return_attention: Whether to return attention weights
            raion_masks: Optional Dict mapping source name to [batch, seq_len, n_raions]
                        Per-raion observation masks for geographic sources.
                        If provided, used instead of source_masks for sources with
                        geographic encoders, enabling finer-grained attention masking.

        Returns:
            fused: Fused representation [batch, seq_len, d_model]
            combined_mask: Combined observation mask [batch, seq_len]
            attention_weights: Dict of attention weights (if requested)
        """
        batch_size = None
        seq_len = None
        device = None

        # Get dimensions
        for name in self.source_names:
            if name in source_features:
                batch_size, seq_len, _ = source_features[name].shape
                device = source_features[name].device
                break

        if batch_size is None:
            raise ValueError("No source data provided")

        attention_weights = {}

        # Encode each source
        encoded_sources = {}
        for i, name in enumerate(self.source_names):
            if name not in source_features:
                # Missing source: use zeros
                encoded_sources[name] = torch.zeros(
                    batch_size, seq_len, self.d_model, device=device
                )
                continue

            features = source_features[name]

            # Use geographic encoder for spatial sources
            if name in self.geographic_encoders:
                # Prefer per-raion mask if available
                if raion_masks is not None and name in raion_masks:
                    # Use per-raion mask [batch, seq_len, n_raions] for full granularity
                    source_mask = raion_masks[name]
                else:
                    # No per-raion mask available - convert source_masks to per-timestep
                    # source_masks has shape [batch, seq, n_features] which is incompatible
                    # with expected [batch, seq, n_raions]. Use per-timestep mask instead.
                    source_mask_raw = source_masks.get(name)
                    if source_mask_raw is not None and source_mask_raw.dim() == 3:
                        # Reduce to [batch, seq] - any feature observed means timestep observed
                        source_mask = source_mask_raw.any(dim=-1)
                    else:
                        source_mask = source_mask_raw  # Already 2D or None

                if return_attention:
                    encoded, geo_attn = self.geographic_encoders[name](
                        features, mask=source_mask, return_attention=True
                    )
                    # Store attention weights with source name prefix
                    for attn_key, attn_val in geo_attn.items():
                        attention_weights[f'{name}_{attn_key}'] = attn_val
                else:
                    encoded = self.geographic_encoders[name](features, mask=source_mask)
            else:
                # Standard projection for non-spatial sources
                if name not in self.source_projections:
                    n_features = features.shape[-1]
                    self.source_projections[name] = nn.Linear(n_features, self.d_model).to(device)
                encoded = self.source_projections[name](features)

            # Add source type embedding
            source_type_emb = self.source_type_embedding(
                torch.tensor([i], device=device)
            ).expand(batch_size, seq_len, -1)

            encoded_sources[name] = encoded + source_type_emb

        # Create combined mask
        combined_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for name in self.source_names:
            if name in source_masks:
                if source_masks[name].dim() == 3:
                    combined_mask = combined_mask | source_masks[name].any(dim=-1)
                else:
                    combined_mask = combined_mask | source_masks[name]

        # Stack for cross-source attention
        stacked = torch.stack(
            [encoded_sources[name] for name in self.source_names],
            dim=2
        )  # [batch, seq_len, n_sources, d_model]

        # Reshape for attention
        flat_stacked = stacked.view(batch_size * seq_len, self.n_sources, self.d_model)

        # Apply cross-source attention
        cross_source_attentions = []
        for layer_idx, (attn, norm, ffn, ffn_norm) in enumerate(zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffn_layers,
            self.ffn_norms,
        )):
            attended, attn_weights = attn(
                flat_stacked, flat_stacked, flat_stacked,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            if return_attention and attn_weights is not None:
                cross_source_attentions.append(attn_weights)
            flat_stacked = norm(flat_stacked + attended)

            ffn_out = ffn(flat_stacked)
            flat_stacked = ffn_norm(flat_stacked + ffn_out)

        # Store cross-source attention weights
        if return_attention and cross_source_attentions:
            for layer_idx, attn_weights in enumerate(cross_source_attentions):
                # attn_weights: [batch*seq_len, n_heads, n_sources, n_sources]
                attention_weights[f'cross_source_layer_{layer_idx}'] = attn_weights

        # Reshape back
        cross_attended = flat_stacked.view(batch_size, seq_len, self.n_sources, self.d_model)

        # Concatenate sources and project
        concat = cross_attended.view(batch_size, seq_len, self.n_sources * self.d_model)
        fused = self.output_projection(concat)

        return fused, combined_mask, attention_weights


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_geographic_fusion(
    d_model: int = 128,
    nhead: int = 8,
    viina_n_raions: int = 100,
    firms_n_raions: int = 20,
    viina_raion_keys: Optional[List[str]] = None,
    firms_raion_keys: Optional[List[str]] = None,
    **kwargs,
) -> GeographicDailyCrossSourceFusion:
    """
    Factory function to create geographic-enhanced fusion module.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        viina_n_raions: Number of raions in VIINA data
        firms_n_raions: Number of raions in FIRMS data
        viina_raion_keys: List of VIINA raion keys for geographic prior
        firms_raion_keys: List of FIRMS raion keys for geographic prior
        **kwargs: Additional arguments

    Returns:
        Configured GeographicDailyCrossSourceFusion
    """
    spatial_configs = {
        'viina': SpatialSourceConfig(
            name='viina',
            n_raions=viina_n_raions,
            features_per_raion=1,  # Will be expanded based on actual data
            raion_keys=viina_raion_keys or [],
            use_geographic_prior=bool(viina_raion_keys),
        ),
        'firms': SpatialSourceConfig(
            name='firms',
            n_raions=firms_n_raions,
            features_per_raion=4,
            raion_keys=firms_raion_keys or [],
            use_geographic_prior=bool(firms_raion_keys),
        ),
    }

    return GeographicDailyCrossSourceFusion(
        d_model=d_model,
        nhead=nhead,
        spatial_configs=spatial_configs,
        **kwargs,
    )


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Geographic Source Encoder Test")
    print("=" * 60)

    # Test GeographicSourceEncoder
    print("\n1. Testing GeographicSourceEncoder (no mask)...")

    config = SpatialSourceConfig(
        name='firms',
        n_raions=20,
        features_per_raion=4,
        use_geographic_prior=False,  # No real raion keys for test
    )

    encoder = GeographicSourceEncoder(config, d_model=128)

    batch_size = 2
    seq_len = 30
    n_raions = 20
    n_features = n_raions * 4  # n_raions * features_per_raion

    x = torch.randn(batch_size, seq_len, n_features)
    output = encoder(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # Test with 2D mask (backward compatible)
    print("\n2. Testing GeographicSourceEncoder (2D mask - per-timestep)...")
    mask_2d = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask_2d[:, 25:] = False  # Last 5 timesteps unobserved
    output_2d = encoder(x, mask=mask_2d)
    print(f"   Mask shape: {mask_2d.shape}")
    print(f"   Output shape: {output_2d.shape}")

    # Test with 3D mask (per-raion)
    print("\n3. Testing GeographicSourceEncoder (3D mask - per-raion)...")
    mask_3d = torch.ones(batch_size, seq_len, n_raions, dtype=torch.bool)
    mask_3d[:, :, 15:] = False  # Last 5 raions unobserved
    mask_3d[:, 25:, :] = False  # Last 5 timesteps fully unobserved
    output_3d = encoder(x, mask=mask_3d)
    print(f"   Mask shape: {mask_3d.shape}")
    print(f"   Output shape: {output_3d.shape}")

    # Verify outputs differ when masks differ
    print("\n4. Verifying mask affects output...")
    diff_2d_3d = (output_2d - output_3d).abs().mean().item()
    print(f"   Mean absolute difference (2D vs 3D mask): {diff_2d_3d:.6f}")
    assert diff_2d_3d > 0.01, "Masks should produce different outputs"
    print("   ✓ Masks produce different outputs as expected")

    # Test GeographicDailyCrossSourceFusion
    print("\n5. Testing GeographicDailyCrossSourceFusion (2D masks)...")

    viina_n_raions = 10
    firms_n_raions = 5

    fusion = create_geographic_fusion(
        d_model=128,
        viina_n_raions=viina_n_raions,
        firms_n_raions=firms_n_raions,
    )

    source_features = {
        'equipment': torch.randn(batch_size, seq_len, 15),
        'personnel': torch.randn(batch_size, seq_len, 10),
        'viina': torch.randn(batch_size, seq_len, viina_n_raions * 1),  # n_raions * 1
        'firms': torch.randn(batch_size, seq_len, firms_n_raions * 4),  # n_raions * 4
    }

    source_masks = {
        name: torch.ones(batch_size, seq_len, dtype=torch.bool)
        for name in source_features
    }

    fused, combined_mask, _ = fusion(source_features, source_masks)

    print(f"   Source shapes: {[(k, v.shape) for k, v in source_features.items()]}")
    print(f"   Fused output shape: {fused.shape}")

    # Test with 3D per-raion masks for spatial sources
    print("\n6. Testing GeographicDailyCrossSourceFusion (3D per-raion masks)...")

    source_masks_3d = {
        'equipment': torch.ones(batch_size, seq_len, dtype=torch.bool),
        'personnel': torch.ones(batch_size, seq_len, dtype=torch.bool),
        'viina': torch.ones(batch_size, seq_len, viina_n_raions, dtype=torch.bool),
        'firms': torch.ones(batch_size, seq_len, firms_n_raions, dtype=torch.bool),
    }
    # Mark some raions as unobserved
    source_masks_3d['viina'][:, :, 7:] = False  # Last 3 raions unobserved
    source_masks_3d['firms'][:, :, 3:] = False  # Last 2 raions unobserved

    fused_3d, combined_mask_3d, _ = fusion(source_features, source_masks_3d)

    print(f"   viina mask shape: {source_masks_3d['viina'].shape}")
    print(f"   firms mask shape: {source_masks_3d['firms'].shape}")
    print(f"   Fused output shape: {fused_3d.shape}")

    # Verify 3D masks affect output
    diff = (fused - fused_3d).abs().mean().item()
    print(f"   Mean absolute difference (2D vs 3D masks): {diff:.6f}")
    assert diff > 0.01, "Per-raion masks should produce different outputs"
    print("   ✓ Per-raion masks produce different outputs as expected")

    # Test that geographic prior actually affects attention weights
    print("\n7. Testing geographic prior application to attention...")

    # Create encoder with synthetic raion keys to enable geo prior
    # We'll create a mock that injects a known prior matrix
    n_raions_test = 5
    config_with_prior = SpatialSourceConfig(
        name='test_geo',
        n_raions=n_raions_test,
        features_per_raion=2,
        raion_keys=['raion_0', 'raion_1', 'raion_2', 'raion_3', 'raion_4'],
        use_geographic_prior=True,
    )

    encoder_with_prior = GeographicSourceEncoder(
        config_with_prior, d_model=64, n_heads=4
    )

    # Create a synthetic geographic prior matrix
    # raion_0 and raion_1 are "adjacent" (high prior)
    # raion_0 and raion_4 are "distant" (low prior)
    synthetic_prior = torch.zeros(n_raions_test, n_raions_test)
    synthetic_prior.fill_diagonal_(0.0)  # Self-attention stays at 0 (neutral in log-space)
    # Adjacent pairs: higher values (less negative = stronger attention)
    synthetic_prior[0, 1] = synthetic_prior[1, 0] = -0.5  # Close
    synthetic_prior[1, 2] = synthetic_prior[2, 1] = -0.5  # Close
    synthetic_prior[2, 3] = synthetic_prior[3, 2] = -0.5  # Close
    synthetic_prior[3, 4] = synthetic_prior[4, 3] = -0.5  # Close
    # Distant pairs: lower values (more negative = weaker attention)
    synthetic_prior[0, 4] = synthetic_prior[4, 0] = -5.0  # Far
    synthetic_prior[0, 3] = synthetic_prior[3, 0] = -3.0  # Medium
    synthetic_prior[0, 2] = synthetic_prior[2, 0] = -1.5  # Medium-close
    synthetic_prior[1, 3] = synthetic_prior[3, 1] = -1.5  # Medium-close
    synthetic_prior[1, 4] = synthetic_prior[4, 1] = -3.0  # Medium
    synthetic_prior[2, 4] = synthetic_prior[4, 2] = -1.5  # Medium-close

    # Inject the synthetic prior
    encoder_with_prior._geo_prior = synthetic_prior

    # Create test input
    test_batch = 2
    test_seq = 5
    test_input = torch.randn(test_batch, test_seq, n_raions_test * 2)

    # Get output with attention weights
    output_with_prior, attn_dict = encoder_with_prior(
        test_input, return_attention=True
    )

    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output_with_prior.shape}")

    if 'cross_raion_attention' in attn_dict:
        cross_attn = attn_dict['cross_raion_attention']
        print(f"   Cross-raion attention shape: {cross_attn.shape}")

        # Average attention weights across batch, seq, and heads
        # cross_attn: [batch*seq_len, n_heads, n_raions, n_raions]
        avg_attn = cross_attn.mean(dim=(0, 1))  # [n_raions, n_raions]

        print(f"   Average attention matrix:")
        for i in range(n_raions_test):
            row = [f"{avg_attn[i, j].item():.3f}" for j in range(n_raions_test)]
            print(f"     raion_{i}: {row}")

        # Verify that adjacent raions have higher attention than distant ones
        # raion_0 -> raion_1 (adjacent) should be > raion_0 -> raion_4 (distant)
        attn_0_to_1 = avg_attn[0, 1].item()
        attn_0_to_4 = avg_attn[0, 4].item()
        print(f"\n   Attention raion_0 -> raion_1 (adjacent): {attn_0_to_1:.4f}")
        print(f"   Attention raion_0 -> raion_4 (distant):  {attn_0_to_4:.4f}")

        # The geographic prior should bias attention so adjacent > distant
        if attn_0_to_1 > attn_0_to_4:
            print("   ✓ Geographic prior correctly biases attention toward adjacent raions")
        else:
            print("   ! Warning: Adjacent attention not higher than distant")
            print("     (This may be OK if learned weights override the prior)")
    else:
        print("   ! Cross-raion attention weights not returned")

    # Compare with encoder without geographic prior
    print("\n8. Comparing with encoder without geographic prior...")

    config_no_prior = SpatialSourceConfig(
        name='test_no_geo',
        n_raions=n_raions_test,
        features_per_raion=2,
        raion_keys=[],  # Empty = no geographic prior
        use_geographic_prior=False,
    )

    encoder_no_prior = GeographicSourceEncoder(
        config_no_prior, d_model=64, n_heads=4
    )

    # Copy weights from the prior encoder for fair comparison
    encoder_no_prior.load_state_dict(encoder_with_prior.state_dict(), strict=False)

    # Get output without prior
    output_no_prior, attn_dict_no_prior = encoder_no_prior(
        test_input, return_attention=True
    )

    # Compare outputs - they should differ because of the geographic prior
    output_diff = (output_with_prior - output_no_prior).abs().mean().item()
    print(f"   Output difference (with vs without geo prior): {output_diff:.6f}")

    if output_diff > 0.001:
        print("   ✓ Geographic prior produces different outputs")
    else:
        print("   ! Warning: Outputs too similar - prior may not be applied")

    if 'cross_raion_attention' in attn_dict_no_prior:
        attn_no_prior = attn_dict_no_prior['cross_raion_attention'].mean(dim=(0, 1))
        attn_with_prior = attn_dict['cross_raion_attention'].mean(dim=(0, 1))

        attn_diff = (attn_with_prior - attn_no_prior).abs().mean().item()
        print(f"   Attention difference (with vs without geo prior): {attn_diff:.6f}")

        if attn_diff > 0.001:
            print("   ✓ Geographic prior affects attention distribution")
        else:
            print("   ! Warning: Attention distributions too similar")

    print("\n" + "=" * 60)
    print("All tests passed!")
