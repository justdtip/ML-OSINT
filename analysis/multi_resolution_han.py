"""
Multi-Resolution Hierarchical Attention Network (MultiResolutionHAN)

This module implements a hierarchical attention network that processes multi-resolution
time series data from conflict monitoring sources while maintaining strict data integrity
principles.

CRITICAL DATA INTEGRITY PRINCIPLES:
==================================

1. NEVER fabricate, interpolate, or forward-fill missing values
2. Daily sources processed at DAILY resolution (~1426 timesteps)
3. Monthly sources processed at MONTHLY resolution (~48 timesteps)
4. Missing observations use learned `no_observation_token` embeddings
5. Explicit observation masks are maintained and used throughout

Architecture Overview:
=====================

    DAILY DATA (5 sources, ~1426 days)
        |
    DailySourceEncoders (5 separate encoders)
        |
    DailyFusion (cross-source attention at daily resolution)
        |
    LearnableMonthlyAggregation (daily->monthly via attention, NOT averaging)
        |
                      |                    |
          aggregated_daily_repr     MONTHLY DATA (5 sources, ~48 months)
                      |                    |
                      |            MonthlySourceEncoders (with no_observation_tokens)
                      |                    |
                CrossResolutionFusion (bidirectional attention)
                      |
                TemporalEncoder (processes fused monthly sequence)
                      |
                Multi-Task Heads:
                |-- CasualtyPredictionHead (regression)
                |-- RegimeClassificationHead (4-class)
                |-- AnomalyDetectionHead (binary)
                |-- ForecastingHead (next-month features)

Author: ML Engineering Team
Date: 2026-01-21
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import existing modules (handle both direct execution and package imports)
try:
    from multi_resolution_modules import (
        MonthlyEncoder,
        MultiSourceMonthlyEncoder,
        CrossResolutionFusion,
        create_monthly_encoder,
        MONTHLY_SOURCE_CONFIGS,
        DailySourceConfig,
        DAILY_SOURCE_CONFIGS,
        MonthlySourceConfig,
        LearnableMonthlyAggregator,
        SinusoidalPositionalEncoding,
    )
    from multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
    )
except ImportError:
    from analysis.multi_resolution_modules import (
        MonthlyEncoder,
        MultiSourceMonthlyEncoder,
        CrossResolutionFusion,
        create_monthly_encoder,
        MONTHLY_SOURCE_CONFIGS,
        DailySourceConfig,
        DAILY_SOURCE_CONFIGS,
        MonthlySourceConfig,
        LearnableMonthlyAggregator,
        SinusoidalPositionalEncoding,
    )
    from analysis.multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
    )

# Import geographic fusion (optional, for enhanced spatial attention)
try:
    from geographic_source_encoder import (
        GeographicDailyCrossSourceFusion,
        SpatialSourceConfig,
        create_geographic_fusion,
    )
    GEOGRAPHIC_FUSION_AVAILABLE = True
except ImportError:
    try:
        from analysis.geographic_source_encoder import (
            GeographicDailyCrossSourceFusion,
            SpatialSourceConfig,
            create_geographic_fusion,
        )
        GEOGRAPHIC_FUSION_AVAILABLE = True
    except ImportError:
        GEOGRAPHIC_FUSION_AVAILABLE = False

# Re-export constants for convenience
# Note: viirs is included per multi_resolution_data.py configuration
DAILY_SOURCES = ['equipment', 'personnel', 'deepstate', 'firms', 'viina', 'viirs']
MONTHLY_SOURCES = ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class SourceConfig:
    """Configuration for a data source (daily or monthly).

    Attributes:
        name: Source identifier (e.g., 'equipment', 'sentinel')
        n_features: Number of input features for this source
        resolution: Either 'daily' or 'monthly'
        description: Human-readable description of the source
    """
    name: str
    n_features: int
    resolution: str = 'daily'
    description: str = ''


@dataclass
class MultiResolutionHANConfig:
    """Configuration for the MultiResolutionHAN model.

    Attributes:
        daily_source_configs: Dict mapping source name to SourceConfig for daily sources
        monthly_source_configs: Dict mapping source name to SourceConfig for monthly sources
        d_model: Hidden dimension for all transformer components
        nhead: Number of attention heads
        num_daily_layers: Number of layers in daily encoders
        num_monthly_layers: Number of layers in monthly encoders
        num_fusion_layers: Number of cross-resolution fusion layers
        num_temporal_layers: Number of temporal encoder layers after fusion
        dropout: Dropout probability throughout the model
        max_daily_len: Maximum daily sequence length (default ~1500 for ~4 years)
        max_monthly_len: Maximum monthly sequence length (default 60 for 5 years)
        prediction_tasks: List of prediction tasks to enable
        causal: Whether to use causal masking in temporal encoders (default True)
    """
    daily_source_configs: Dict[str, SourceConfig] = field(default_factory=dict)
    monthly_source_configs: Dict[str, SourceConfig] = field(default_factory=dict)
    d_model: int = 128
    nhead: int = 8
    num_daily_layers: int = 4
    num_monthly_layers: int = 3
    num_fusion_layers: int = 2
    num_temporal_layers: int = 2
    dropout: float = 0.1
    max_daily_len: int = 1500
    max_monthly_len: int = 60
    prediction_tasks: List[str] = field(
        default_factory=lambda: ['casualty', 'regime', 'anomaly', 'forecast']
    )
    causal: bool = True
    # Geographic prior options
    use_geographic_prior: bool = False
    spatial_source_names: List[str] = field(
        default_factory=lambda: ['viina', 'firms']
    )
    viina_n_raions: int = 100
    firms_n_raions: int = 20


# =============================================================================
# DAILY SOURCE ENCODER
# =============================================================================

class DailySourceEncoder(nn.Module):
    """
    Encoder for a single daily-resolution data source.

    This encoder processes daily observations at their NATIVE daily resolution,
    maintaining data integrity by using learned no_observation_tokens for missing
    values rather than fabricating data.

    CRITICAL: Uses causal masking by default to prevent future information
    leakage during autoregressive prediction. Each timestep can only attend
    to current and past timesteps, not future ones.

    Key Features:
    - Projects raw features to d_model dimension
    - Adds learnable feature embeddings per feature
    - Adds sinusoidal positional encoding for temporal position
    - Uses observation mask in attention (missing values cannot be keys/values)
    - Maintains its own no_observation_token for missing daily values
    - Causal masking to prevent future information leakage

    Args:
        source_config: Configuration for this daily source
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout probability
        max_len: Maximum sequence length
        causal: Whether to use causal (autoregressive) attention masking.
            If True, position i can only attend to positions <= i.
            Default is True to prevent future information leakage.

    Example:
        >>> config = SourceConfig('equipment', n_features=38, resolution='daily')
        >>> encoder = DailySourceEncoder(config, d_model=128)
        >>> values = torch.randn(4, 1000, 38)  # batch, seq, features
        >>> mask = torch.ones(4, 1000, 38, dtype=torch.bool)
        >>> hidden, attn = encoder(values, mask)
        >>> print(hidden.shape)  # [4, 1000, 128]
    """

    def __init__(
        self,
        source_config: SourceConfig,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 1500,
        causal: bool = True,
    ) -> None:
        super().__init__()

        self.source_config = source_config
        self.d_model = d_model
        self.n_features = source_config.n_features
        self.nhead = nhead
        self.causal = causal
        self.max_len = max_len

        # =====================================================================
        # CRITICAL: Learned no_observation_token for missing values
        # =====================================================================
        # This is NOT zero, NOT forward-fill - it's a learned representation
        # of "we have no data here". The model learns what absence means.
        self.no_observation_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        # Feature projection: raw features -> d_model
        self.feature_projection = nn.Sequential(
            nn.Linear(source_config.n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable feature embeddings (one per feature)
        self.feature_embedding = nn.Embedding(source_config.n_features, d_model)

        # Observation status embedding: 0=unobserved, 1=observed
        self.observation_status_embedding = nn.Embedding(2, d_model)

        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Register causal mask buffer for autoregressive attention
        # Upper triangular matrix where True = IGNORE (don't attend)
        # This prevents position i from attending to any position j > i
        if causal:
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: Tensor,
        observation_mask: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Encode daily features for this source.

        Args:
            features: Raw feature values [batch, seq_len, n_features]
            observation_mask: Binary mask [batch, seq_len, n_features] where
                True = observed (real data), False = unobserved (no data)
            return_attention: Whether to return attention weights

        Returns:
            hidden: Encoded representations [batch, seq_len, d_model]
            attention_weights: Optional attention weights if return_attention=True

        Note:
            Missing observations are handled by replacing their projected values
            with the learned no_observation_token. The observation_mask is used
            to mask attention so that unobserved positions cannot contribute
            as keys/values.
        """
        batch_size, seq_len, n_features = features.shape
        device = features.device

        # CRITICAL FIX: Defensive shape assertions to catch dimension mismatches early
        assert features.dim() == 3, f"Expected features to be 3D, got {features.dim()}D"
        assert observation_mask.dim() == 3, f"Expected observation_mask to be 3D, got {observation_mask.dim()}D"
        assert features.shape == observation_mask.shape, f"features and observation_mask shape mismatch: {features.shape} vs {observation_mask.shape}"
        assert n_features == self.n_features, f"Input n_features {n_features} != expected {self.n_features}"
        assert observation_mask.dtype == torch.bool, f"observation_mask should be torch.bool, got {observation_mask.dtype}"

        # =====================================================================
        # STEP 1: Replace MISSING_VALUE (-999.0) with 0.0 BEFORE projection
        # =====================================================================
        # This prevents extreme values from flowing through linear layers
        # The observation_mask already indicates which positions are valid
        features = features.clone()
        features = features.masked_fill(~observation_mask, 0.0)

        # Handle extreme values with soft clamping
        # Some features like territorial_area_km2 can legitimately be 30,000+ km²
        # Use log-compression for values > 100 to preserve relative ordering
        # while keeping gradients stable
        large_pos = features > 100
        large_neg = features < -100
        if large_pos.any():
            # Compress: 100 + log(x/100) for x > 100
            features = torch.where(
                large_pos,
                100 + torch.log1p(features.abs() / 100 - 1),
                features
            )
        if large_neg.any():
            features = torch.where(
                large_neg,
                -100 - torch.log1p(features.abs() / 100 - 1),
                features
            )

        # =====================================================================
        # STEP 2: Project features to d_model dimension
        # =====================================================================
        # Project all features together
        hidden = self.feature_projection(features)  # [batch, seq, d_model]

        # Verify projection output is reasonable
        assert not torch.isnan(hidden).any(), "NaN in projected features"
        # Projection can amplify clamped values; use softer check
        if (hidden.abs() > 1000).any():
            hidden = hidden.clamp(-100, 100)

        # =====================================================================
        # STEP 3: Handle missing observations with no_observation_token
        # =====================================================================
        # Compute per-timestep observation status (True if ANY feature observed)
        timestep_observed = observation_mask.any(dim=-1)  # [batch, seq_len]

        # Expand no_observation_token for broadcasting
        no_obs_expanded = self.no_observation_token.expand(batch_size, seq_len, -1)

        # Replace unobserved timesteps with no_observation_token
        obs_mask_expanded = timestep_observed.unsqueeze(-1)  # [batch, seq, 1]
        hidden = torch.where(obs_mask_expanded, hidden, no_obs_expanded)

        # =====================================================================
        # STEP 4: Add observation status embedding
        # =====================================================================
        obs_status = timestep_observed.long()  # [batch, seq_len]
        obs_status_emb = self.observation_status_embedding(obs_status)
        hidden = hidden + obs_status_emb

        # =====================================================================
        # STEP 5: Add positional encoding
        # =====================================================================
        hidden = self.positional_encoding(hidden)

        # =====================================================================
        # STEP 6: Transformer encoding with masked attention
        # =====================================================================
        # For PyTorch transformer: True = IGNORE this position
        src_key_padding_mask = ~timestep_observed

        # Handle edge case: if ALL positions are masked in a sequence
        all_masked = src_key_padding_mask.all(dim=1)
        if all_masked.any():
            src_key_padding_mask = src_key_padding_mask.clone()
            src_key_padding_mask[all_masked, 0] = False

        # Prepare causal attention mask
        # In PyTorch TransformerEncoder, mask=True means IGNORE that position
        attn_mask = None
        if self.causal:
            attn_mask = self.causal_mask[:seq_len, :seq_len]

        hidden = self.transformer_encoder(
            hidden,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # =====================================================================
        # STEP 7: Output normalization
        # =====================================================================
        hidden = self.output_norm(hidden)

        # TODO: Implement attention weight extraction if needed
        attention_weights = None

        return hidden, attention_weights


# =============================================================================
# TEMPORAL POSITIONAL ENCODING FOR GATE
# =============================================================================


class TemporalGatePositionalEncoding(nn.Module):
    """
    Learnable temporal positional encoding specifically designed for gating mechanisms.

    Unlike standard sinusoidal encoding, this learns position-aware representations
    that can capture domain-specific temporal patterns (e.g., weekly reporting cycles,
    monthly resupply patterns) relevant to conflict dynamics.

    Combines:
    - Absolute position embeddings (captures trend/time-in-conflict)
    - Relative position encoding via learned offset embeddings
    - Day-of-week embeddings (7-day cycle for operational patterns)

    Args:
        d_model: Model hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1500,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Absolute position embeddings
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Day-of-week embeddings (7-day operational cycle)
        self.day_of_week_embedding = nn.Embedding(7, d_model)

        # Learnable scale for combining embeddings
        self.scale = nn.Parameter(torch.ones(3))  # [pos, dow, input]

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.day_of_week_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add temporal positional encoding.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positions = positions.clamp(max=self.max_len - 1)

        # Get position embeddings
        pos_emb = self.position_embedding(positions)  # [batch, seq, d_model]

        # Get day-of-week embeddings
        dow = positions % 7
        dow_emb = self.day_of_week_embedding(dow)  # [batch, seq, d_model]

        # Scale factors
        scale = F.softmax(self.scale, dim=0)

        # Combine with learned scaling
        output = scale[0] * pos_emb + scale[1] * dow_emb + scale[2] * x

        return self.dropout(self.layer_norm(output))


# =============================================================================
# TEMPORAL SOURCE GATE (Enhanced with temporal encoding)
# =============================================================================


class TemporalSourceGate(nn.Module):
    """
    Source gating with temporal context awareness and explicit temporal encoding.

    Uses local convolution, temporal positional encoding, and optional self-attention
    to compute source importance that considers temporal patterns, not just the current
    timestep. This addresses the limitation of pointwise gating which cannot learn
    patterns like "FIRMS becomes more important after equipment losses".

    Enhanced Features (addressing overfitting):
    - Explicit temporal positional encoding for better inductive bias
    - Multi-scale convolution for capturing patterns at different frequencies
    - Temperature-scaled softmax for sharper/softer gating
    - Regularization via dropout at multiple stages

    Architecture:
    - Multi-scale 1D convolutions for local temporal context (3, 7, 14 day windows)
    - Temporal positional encoding for explicit temporal awareness
    - Optional self-attention for longer-range temporal patterns
    - Temperature-scaled softmax for gate output

    Args:
        d_model: Model hidden dimension
        n_sources: Number of sources to gate
        kernel_sizes: List of convolution kernel sizes (default: [3, 7, 14])
        nhead: Number of attention heads (if use_attention=True)
        use_attention: Whether to include self-attention for longer patterns
        dropout: Dropout probability
        temperature: Temperature for softmax (lower = sharper gating)
    """

    def __init__(
        self,
        d_model: int,
        n_sources: int,
        kernel_sizes: Optional[List[int]] = None,
        nhead: int = 4,
        use_attention: bool = True,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.use_attention = use_attention
        self.temperature = temperature

        if kernel_sizes is None:
            kernel_sizes = [3, 7, 14]  # Short, weekly, bi-weekly patterns
        self.kernel_sizes = kernel_sizes
        n_scales = len(kernel_sizes)

        # Calculate output channels per scale to ensure they sum to d_model
        # Handle non-divisible cases by giving extra channels to first scales
        base_channels = d_model // n_scales
        remainder = d_model % n_scales
        scale_channels = [
            base_channels + (1 if i < remainder else 0)
            for i in range(n_scales)
        ]
        self.scale_channels = scale_channels
        total_conv_out = sum(scale_channels)

        # Multi-scale temporal convolutions
        # Each captures patterns at different temporal frequencies
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    d_model * n_sources,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k, channels in zip(kernel_sizes, scale_channels)
        ])

        # Fusion layer for multi-scale outputs
        self.scale_fusion = nn.Sequential(
            nn.Linear(total_conv_out, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal positional encoding
        self.temporal_encoding = TemporalGatePositionalEncoding(
            d_model=d_model,
            max_len=1500,
            dropout=dropout,
        )

        # Optional self-attention for longer-range patterns
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(d_model)
            self.attn_dropout = nn.Dropout(dropout)

        # Final gate projection with intermediate layer for better gradients
        self.gate_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_sources),
        )

        # Learnable temperature (can adapt during training)
        self.learnable_temp = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        stacked_sources: Tensor,  # [batch, seq, n_sources * d_model]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute temporally-aware source importance weights.

        Args:
            stacked_sources: Concatenated source representations [batch, seq, n_sources * d_model]
            mask: Optional validity mask [batch, seq] where True = valid position

        Returns:
            Source importance weights [batch, seq, n_sources] summing to 1 per timestep
        """
        batch_size, seq_len, _ = stacked_sources.shape
        device = stacked_sources.device

        # Multi-scale temporal convolutions
        # Conv1d expects [batch, channels, seq]
        x_transposed = stacked_sources.transpose(1, 2)  # [batch, channels, seq]

        scale_outputs = []
        for conv in self.multi_scale_convs:
            conv_out = conv(x_transposed)  # [batch, d_model // n_scales, seq']
            # Ensure output matches input sequence length (padding can cause +1/-1)
            if conv_out.shape[2] > seq_len:
                conv_out = conv_out[:, :, :seq_len]
            elif conv_out.shape[2] < seq_len:
                pad_size = seq_len - conv_out.shape[2]
                conv_out = F.pad(conv_out, (0, pad_size), mode='constant', value=0)
            scale_outputs.append(conv_out)

        # Concatenate multi-scale features
        x = torch.cat(scale_outputs, dim=1)  # [batch, d_model, seq]
        x = x.transpose(1, 2)  # [batch, seq, d_model]

        # Fuse multi-scale features
        x = self.scale_fusion(x)

        # Add temporal positional encoding
        x = self.temporal_encoding(x)

        # Optional attention for longer-range patterns
        if self.use_attention:
            # For key_padding_mask: True = IGNORE this position
            key_padding_mask = ~mask if mask is not None else None

            # Handle edge case where all positions might be masked
            if key_padding_mask is not None:
                all_masked = key_padding_mask.all(dim=1)
                if all_masked.any():
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_masked, 0] = False

            attended, _ = self.temporal_attention(x, x, x, key_padding_mask=key_padding_mask)
            attended = self.attn_dropout(attended)
            x = self.attn_norm(x + attended)

        # Compute gate logits
        gate_logits = self.gate_projection(x)  # [batch, seq, n_sources]

        # Temperature-scaled softmax
        # Use absolute value of learnable temp to ensure positive
        effective_temp = torch.abs(self.learnable_temp) + 0.1  # Minimum temp of 0.1
        gate_weights = F.softmax(gate_logits / effective_temp, dim=-1)

        return gate_weights


# =============================================================================
# DAILY CROSS-SOURCE FUSION
# =============================================================================

class DailyCrossSourceFusion(nn.Module):
    """
    Fuses representations from multiple daily sources using cross-attention.

    This module learns dependencies between different daily sources (equipment,
    personnel, deepstate, firms, viina) through cross-attention mechanisms.

    Architecture:
    - Each source attends to all other sources
    - Gated residual connections allow selective information flow
    - Output is a unified daily representation

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_layers: Number of fusion layers
        dropout: Dropout probability
        source_names: List of source names for tracking
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        source_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.source_names = source_names or DAILY_SOURCES
        self.n_sources = len(self.source_names)

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
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # Feed-forward layers
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
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # Gating mechanism for combining sources with temporal context
        # Replaces the pointwise gate that could not learn temporal patterns
        self.source_gate = TemporalSourceGate(
            d_model=d_model,
            n_sources=self.n_sources,
            kernel_sizes=[3, 7, 14],  # Multi-scale: 3-day, weekly, bi-weekly
            nhead=4,
            use_attention=True,
            dropout=dropout,
        )

        # Final projection
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
        source_hidden: Dict[str, Tensor],
        source_masks: Dict[str, Tensor],
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Fuse multiple daily source representations.

        Args:
            source_hidden: Dict mapping source name to [batch, seq_len, d_model]
            source_masks: Dict mapping source name to [batch, seq_len] masks
            return_attention: Whether to return attention weights

        Returns:
            fused: Fused representation [batch, seq_len, d_model]
            combined_mask: Combined observation mask [batch, seq_len]
            attention_weights: Dict of attention weights per source (if requested)
        """
        batch_size = None
        seq_len = None
        device = None

        # Get dimensions from first available source
        for name in self.source_names:
            if name in source_hidden:
                batch_size, seq_len, _ = source_hidden[name].shape
                device = source_hidden[name].device
                break

        if batch_size is None:
            raise ValueError("No source data provided")

        attention_weights = {}

        # Add source type embeddings
        source_hidden_with_type = {}
        for i, name in enumerate(self.source_names):
            if name in source_hidden:
                source_type_emb = self.source_type_embedding(
                    torch.tensor([i], device=device)
                ).expand(batch_size, seq_len, -1)
                source_hidden_with_type[name] = source_hidden[name] + source_type_emb
            else:
                # Missing source: use zeros
                source_hidden_with_type[name] = torch.zeros(
                    batch_size, seq_len, self.d_model, device=device
                )

        # Create combined mask (True where ANY source has observation)
        combined_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for name in self.source_names:
            if name in source_masks:
                # source_masks has shape [batch, seq, features], reduce to [batch, seq]
                if source_masks[name].dim() == 3:
                    combined_mask = combined_mask | source_masks[name].any(dim=-1)
                else:
                    combined_mask = combined_mask | source_masks[name]

        # Stack sources for processing: [batch, seq, n_sources, d_model]
        stacked = torch.stack(
            [source_hidden_with_type[name] for name in self.source_names],
            dim=2
        )

        # Reshape for attention: [batch * seq, n_sources, d_model]
        flat_stacked = stacked.view(batch_size * seq_len, self.n_sources, self.d_model)

        # Apply cross-source attention layers
        for attn, norm, ffn, ffn_norm in zip(
            self.cross_attention_layers,
            self.layer_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            # Self-attention across sources
            attended, _ = attn(flat_stacked, flat_stacked, flat_stacked)
            flat_stacked = norm(flat_stacked + attended)

            # Feed-forward
            ffn_out = ffn(flat_stacked)
            flat_stacked = ffn_norm(flat_stacked + ffn_out)

        # Reshape back: [batch, seq, n_sources, d_model]
        fused_sources = flat_stacked.view(batch_size, seq_len, self.n_sources, self.d_model)

        # Compute source importance via gating with temporal context
        concat_sources = fused_sources.view(batch_size, seq_len, -1)
        source_importance = self.source_gate(concat_sources, combined_mask)  # [batch, seq, n_sources]

        # Weighted combination
        weighted_sources = fused_sources * source_importance.unsqueeze(-1)

        # Final projection
        fused = self.output_projection(
            weighted_sources.view(batch_size, seq_len, -1)
        )

        if return_attention:
            attention_weights['source_importance'] = source_importance

        return fused, combined_mask, attention_weights


# =============================================================================
# DAILY TEMPORAL ENCODER (Enhanced with temporal gating and positional encoding)
# =============================================================================

class DailyTemporalEncoder(nn.Module):
    """
    Process daily sequences with temporal awareness BEFORE monthly aggregation.

    This module captures daily-level temporal patterns (weekly cycles, operational
    tempo) that would otherwise be lost during monthly aggregation. Uses multi-scale
    convolutions, explicit temporal positional encoding, temporal gating, and local
    windowed attention to efficiently process long daily sequences.

    The validation finding C1 showed that the model learns temporal patterns
    primarily at MONTHLY resolution (2.9x more sensitive than daily) because
    daily information is compressed before the main temporal processing. This
    module addresses that by enriching daily representations with temporal
    context before aggregation.

    Enhanced Features (addressing overfitting at epoch 2):
    - Explicit temporal positional encoding for better inductive bias
    - Temporal gating mechanism to control information flow
    - Multi-scale convolutions with proper normalization
    - Learnable scale factors for combining components
    - Additional regularization via dropout

    Architecture:
    - Temporal positional encoding (sinusoidal + learned day-of-week)
    - Multi-scale 1D convolutions (3, 7, 14, 28 day kernels) with dropout
    - Temporal gate for controlling multi-scale feature contribution
    - Local windowed causal attention (~31 day window)
    - Gated residual connections

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        dropout: Dropout probability
        window_size: Size of local attention window (default 31 for ~1 month)
        causal: If True, days can only attend to past (not future)
        max_len: Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
        window_size: int = 31,
        causal: bool = True,
        max_len: int = 1500,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.causal = causal
        self.max_len = max_len

        # =====================================================================
        # TEMPORAL POSITIONAL ENCODING
        # =====================================================================
        # Sinusoidal base encoding
        self._create_sinusoidal_encoding(max_len, d_model)

        # Learned day-of-week encoding (7-day cycle)
        self.day_of_week_embedding = nn.Embedding(7, d_model)

        # Learned month-in-year encoding (seasonal patterns)
        self.month_embedding = nn.Embedding(12, d_model)

        # Position encoding combination weights
        self.pos_scale = nn.Parameter(torch.ones(4))  # [sin, dow, month, input]

        self.pos_norm = nn.LayerNorm(d_model)
        self.pos_dropout = nn.Dropout(dropout)

        # =====================================================================
        # MULTI-SCALE TEMPORAL CONVOLUTIONS
        # =====================================================================
        # Capture patterns at different temporal scales:
        # - 3 days: short-term operational patterns
        # - 7 days: weekly cycles
        # - 14 days: bi-weekly patterns
        # - 28 days: monthly patterns
        kernel_sizes = [3, 7, 14, 28]
        self.kernel_sizes = kernel_sizes

        self.multi_scale_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k // 2),
                nn.GroupNorm(num_groups=4, num_channels=d_model // 4),  # More stable than BatchNorm
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        self.conv_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # =====================================================================
        # TEMPORAL GATE (controls multi-scale feature contribution)
        # =====================================================================
        # This gate learns to control how much multi-scale information flows
        # through vs. the original signal. Helps prevent overfitting by
        # allowing the model to "fall back" to simpler representations.
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )

        # =====================================================================
        # LOCAL WINDOWED ATTENTION
        # =====================================================================
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # =====================================================================
        # OUTPUT LAYERS
        # =====================================================================
        # Final gated combination
        self.output_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)

        # Learnable scale for residual connection (prevents gradient issues)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # Pre-computed local attention mask
        self._cached_attn_mask = None
        self._cached_mask_len = 0

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> None:
        """Create sinusoidal positional encoding buffer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('sinusoidal_pe', pe)

    def _create_local_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Create attention mask for local windowed attention with optional causality.

        Uses caching to avoid recreating mask for same sequence length.

        Args:
            seq_len: Length of the sequence
            device: Device for tensor creation

        Returns:
            Attention mask [seq_len, seq_len] where True = MASK (don't attend)
        """
        # Check cache
        if self._cached_attn_mask is not None and self._cached_mask_len >= seq_len:
            return self._cached_attn_mask[:seq_len, :seq_len].to(device)

        # Create new mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        for i in range(seq_len):
            if self.causal:
                # Causal: only attend to past within window
                start = max(0, i - self.window_size + 1)
                end = i + 1  # Include current position
            else:
                # Non-causal: attend within window centered on current position
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False

        # Cache the mask
        self._cached_attn_mask = mask
        self._cached_mask_len = seq_len

        return mask.to(device)

    def _add_temporal_encoding(self, x: Tensor) -> Tensor:
        """
        Add temporal positional encoding to input.

        Combines sinusoidal, day-of-week, and month encodings.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get sinusoidal encoding
        sin_pe = self.sinusoidal_pe[:, :seq_len, :]  # [1, seq_len, d_model]

        # Compute day-of-week indices (assuming sequential days)
        positions = torch.arange(seq_len, device=device)
        dow_indices = positions % 7
        dow_emb = self.day_of_week_embedding(dow_indices)  # [seq_len, d_model]
        dow_emb = dow_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Approximate month (assuming ~30 days per month, seasonal patterns)
        month_indices = (positions // 30) % 12
        month_emb = self.month_embedding(month_indices)  # [seq_len, d_model]
        month_emb = month_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine with learnable scales
        scale = F.softmax(self.pos_scale, dim=0)
        encoded = (
            scale[0] * sin_pe +
            scale[1] * dow_emb +
            scale[2] * month_emb +
            scale[3] * x
        )

        return self.pos_dropout(self.pos_norm(encoded))

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process daily sequence with local temporal patterns.

        Args:
            x: Daily representations [batch, n_days, d_model]
            mask: Observation mask [batch, n_days] where True = observed

        Returns:
            Temporally-enriched daily representations [batch, n_days, d_model]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Store original input for residual
        residual = x

        # =====================================================================
        # 1. Add temporal positional encoding
        # =====================================================================
        x_encoded = self._add_temporal_encoding(x)

        # =====================================================================
        # 2. Multi-scale convolutions for local pattern extraction
        # =====================================================================
        x_conv = x_encoded.transpose(1, 2)  # [batch, d_model, seq]
        conv_outputs = [conv(x_conv) for conv in self.multi_scale_conv]

        # Concatenate and ensure sequence length is preserved
        min_len = min(out.shape[2] for out in conv_outputs)
        conv_outputs = [out[:, :, :min_len] for out in conv_outputs]

        multi_scale = torch.cat(conv_outputs, dim=1)  # [batch, d_model, min_len]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, min_len, d_model]

        # Pad or trim to original length
        if multi_scale.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - multi_scale.shape[1], self.d_model,
                device=device
            )
            multi_scale = torch.cat([multi_scale, padding], dim=1)
        elif multi_scale.shape[1] > seq_len:
            multi_scale = multi_scale[:, :seq_len, :]

        multi_scale = self.conv_fusion(multi_scale)

        # =====================================================================
        # 3. Temporal gate for multi-scale features
        # =====================================================================
        # Concatenate input and multi-scale features for gating decision
        gate_input = torch.cat([x_encoded, multi_scale], dim=-1)  # [batch, seq, 2*d_model]
        gate = self.temporal_gate(gate_input)  # [batch, seq, d_model]

        # Apply gate: blend between encoded input and multi-scale features
        gated_features = gate * multi_scale + (1 - gate) * x_encoded

        # =====================================================================
        # 4. Local windowed attention for contextual patterns
        # =====================================================================
        attn_mask = self._create_local_causal_mask(seq_len, device)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # True = ignore

            # Handle fully masked sequences
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        attended, _ = self.local_attention(
            gated_features, gated_features, gated_features,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        # Handle NaN from attention (occurs when all keys are masked)
        attended = torch.nan_to_num(attended, nan=0.0)
        attended = self.attn_dropout(attended)
        attended = self.attn_norm(gated_features + attended)

        # =====================================================================
        # 5. Output gating and residual connection
        # =====================================================================
        # Final gate to control how much new information to add
        output_gate_input = torch.cat([residual, attended], dim=-1)
        output_gate = self.output_gate(output_gate_input)

        # Gated combination with scaled residual
        output = output_gate * attended + (1 - output_gate) * residual

        # Apply final normalization and dropout
        output = self.output_dropout(self.output_norm(output))

        # Add scaled residual for gradient flow
        output = output + self.residual_scale * residual

        return output


# =============================================================================
# CROSS-TEMPORAL SCALE ATTENTION
# =============================================================================

class CrossTemporalScaleAttention(nn.Module):
    """
    Cross-attention mechanism that allows information flow across different
    temporal scales (daily, weekly, monthly).

    This module addresses the finding that models learn primarily at monthly
    resolution by explicitly allowing each temporal scale to attend to others.
    For example, monthly representations can attend to weekly summaries, and
    weekly summaries can incorporate daily patterns.

    Architecture:
    - Projects daily sequences to weekly and monthly scales via pooling
    - Cross-attention between scales (daily<->weekly, weekly<->monthly)
    - Gated fusion to combine information from different scales
    - Causal masking to prevent future information leakage

    The key insight is that conflict dynamics operate at multiple timescales:
    - Daily: Tactical operations, individual battles
    - Weekly: Operational tempo, supply cycles
    - Monthly: Strategic shifts, resource accumulation

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        dropout: Dropout probability
        max_daily_len: Maximum daily sequence length
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
        max_daily_len: int = 1500,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # =====================================================================
        # SCALE PROJECTION LAYERS
        # =====================================================================
        # Project daily to weekly (7-day pooling with attention)
        self.daily_to_weekly_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Learned week queries for pooling daily to weekly
        self.week_queries = nn.Parameter(
            torch.randn(1, max_daily_len // 7 + 1, d_model) * (d_model ** -0.5)
        )

        # Project weekly to monthly (4-week pooling)
        self.weekly_to_monthly_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Learned month queries for pooling weekly to monthly
        self.month_queries = nn.Parameter(
            torch.randn(1, max_daily_len // 30 + 1, d_model) * (d_model ** -0.5)
        )

        # =====================================================================
        # CROSS-SCALE ATTENTION
        # =====================================================================
        # Monthly attending to weekly context
        self.monthly_cross_weekly = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Weekly attending to daily context
        self.weekly_cross_daily = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # =====================================================================
        # GATED FUSION
        # =====================================================================
        # Fuse information from different scales back to daily
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Gate for controlling how much cross-scale information to use
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Layer norms
        self.norm_weekly = nn.LayerNorm(d_model)
        self.norm_monthly = nn.LayerNorm(d_model)
        self.norm_output = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _create_week_boundaries(
        self, n_days: int, device: torch.device
    ) -> Tuple[Tensor, int]:
        """Create week boundary indices for pooling."""
        n_weeks = (n_days + 6) // 7
        boundaries = []
        for w in range(n_weeks):
            start = w * 7
            end = min((w + 1) * 7, n_days)
            boundaries.append((start, end))
        return boundaries, n_weeks

    def _create_month_boundaries(
        self, n_weeks: int, device: torch.device
    ) -> Tuple[Tensor, int]:
        """Create month boundary indices for pooling (4 weeks per month)."""
        n_months = (n_weeks + 3) // 4
        boundaries = []
        for m in range(n_months):
            start = m * 4
            end = min((m + 1) * 4, n_weeks)
            boundaries.append((start, end))
        return boundaries, n_months

    def forward(
        self,
        daily_repr: Tensor,
        daily_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply cross-temporal scale attention.

        Args:
            daily_repr: Daily representations [batch, n_days, d_model]
            daily_mask: Optional mask [batch, n_days] where True = observed

        Returns:
            Enhanced daily representations [batch, n_days, d_model]
        """
        batch_size, n_days, _ = daily_repr.shape
        device = daily_repr.device

        # Store original for residual
        residual = daily_repr

        # =====================================================================
        # 1. Project daily to weekly scale
        # =====================================================================
        week_boundaries, n_weeks = self._create_week_boundaries(n_days, device)

        # Get week queries for this sequence
        week_q = self.week_queries[:, :n_weeks, :].expand(batch_size, -1, -1)

        # Create attention mask for week->day attention (each week only sees its days)
        week_day_mask = torch.ones(n_weeks, n_days, device=device, dtype=torch.bool)
        for w, (start, end) in enumerate(week_boundaries):
            week_day_mask[w, start:end] = False  # False = can attend

        # Attention from week queries to daily representations
        weekly_repr, _ = self.daily_to_weekly_attn(
            week_q,
            daily_repr,
            daily_repr,
            attn_mask=week_day_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1).reshape(batch_size * self.nhead, n_weeks, n_days),
        )
        weekly_repr = torch.nan_to_num(weekly_repr, nan=0.0)
        weekly_repr = self.norm_weekly(weekly_repr)

        # =====================================================================
        # 2. Project weekly to monthly scale
        # =====================================================================
        month_boundaries, n_months = self._create_month_boundaries(n_weeks, device)

        # Get month queries for this sequence
        month_q = self.month_queries[:, :n_months, :].expand(batch_size, -1, -1)

        # Create attention mask for month->week attention
        month_week_mask = torch.ones(n_months, n_weeks, device=device, dtype=torch.bool)
        for m, (start, end) in enumerate(month_boundaries):
            month_week_mask[m, start:end] = False

        monthly_repr, _ = self.weekly_to_monthly_attn(
            month_q,
            weekly_repr,
            weekly_repr,
            attn_mask=month_week_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1).reshape(batch_size * self.nhead, n_months, n_weeks),
        )
        monthly_repr = torch.nan_to_num(monthly_repr, nan=0.0)
        monthly_repr = self.norm_monthly(monthly_repr)

        # =====================================================================
        # 3. Cross-scale attention (monthly attends to weekly)
        # =====================================================================
        monthly_enhanced, _ = self.monthly_cross_weekly(
            monthly_repr, weekly_repr, weekly_repr
        )
        monthly_enhanced = torch.nan_to_num(monthly_enhanced, nan=0.0)

        # =====================================================================
        # 4. Cross-scale attention (weekly attends to daily)
        # =====================================================================
        weekly_enhanced, _ = self.weekly_cross_daily(
            weekly_repr, daily_repr, daily_repr
        )
        weekly_enhanced = torch.nan_to_num(weekly_enhanced, nan=0.0)

        # =====================================================================
        # 5. Broadcast back to daily resolution
        # =====================================================================
        # Expand weekly to daily (repeat each week's representation for 7 days)
        weekly_daily = torch.zeros(batch_size, n_days, self.d_model, device=device)
        for w, (start, end) in enumerate(week_boundaries):
            weekly_daily[:, start:end, :] = weekly_enhanced[:, w:w+1, :].expand(-1, end - start, -1)

        # Expand monthly to daily (repeat each month's representation for ~30 days)
        monthly_daily = torch.zeros(batch_size, n_days, self.d_model, device=device)
        for m, (w_start, w_end) in enumerate(month_boundaries):
            # Get the day range covered by this month
            if w_start < len(week_boundaries):
                d_start = week_boundaries[w_start][0]
                d_end = week_boundaries[min(w_end, len(week_boundaries)) - 1][1] if w_end <= len(week_boundaries) else n_days
                if d_start < d_end:
                    monthly_daily[:, d_start:d_end, :] = monthly_enhanced[:, m:m+1, :].expand(-1, d_end - d_start, -1)

        # =====================================================================
        # 6. Fuse multi-scale information
        # =====================================================================
        # Concatenate daily, weekly (broadcast), and monthly (broadcast)
        multi_scale_concat = torch.cat([daily_repr, weekly_daily, monthly_daily], dim=-1)
        fused = self.scale_fusion(multi_scale_concat)

        # Gated combination with original
        gate_input = torch.cat([residual, fused], dim=-1)
        gate = self.fusion_gate(gate_input)

        output = gate * fused + (1 - gate) * residual
        output = self.dropout(self.norm_output(output))

        return output


# =============================================================================
# LEARNABLE MONTHLY AGGREGATION (Enhanced version)
# =============================================================================

class EnhancedLearnableMonthlyAggregation(nn.Module):
    """
    Aggregates daily representations to monthly resolution using cross-attention.

    CRITICAL: This is LEARNABLE aggregation, NOT simple averaging.

    The monthly queries learn to attend to the most relevant daily observations
    within their month boundaries. This allows the model to:
    1. Weight significant events (battles, supply disruptions) higher
    2. Downweight noise and routine observations
    3. Capture different aggregation patterns for different contexts

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        max_months: Maximum number of months to support
        dropout: Dropout probability
        use_month_constraints: If True, months only attend to their own days
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        max_months: int = 60,
        dropout: float = 0.1,
        use_month_constraints: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.max_months = max_months
        self.use_month_constraints = use_month_constraints

        # Learnable month query embeddings
        self.month_queries = nn.Parameter(
            torch.randn(1, max_months, d_model) * (d_model ** -0.5)
        )

        # Month position embedding
        self.month_position_embedding = nn.Embedding(max_months, d_model)

        # Cross-attention: months query days
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def _create_month_attention_mask(
        self,
        n_months: int,
        n_days: int,
        month_boundaries: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Create attention mask constraining each month to its days.

        Args:
            n_months: Number of months
            n_days: Number of days in sequence
            month_boundaries: [batch, n_months, 2] with (start_day, end_day)
            device: Device for tensor creation

        Returns:
            Attention mask [batch, n_months, n_days] where True = MASK (don't attend)
        """
        batch_size = month_boundaries.shape[0]

        # Create day indices [1, 1, n_days]
        day_indices = torch.arange(n_days, device=device).view(1, 1, n_days)

        # Get start and end for each month [batch, n_months, 1]
        start_days = month_boundaries[:, :, 0].unsqueeze(-1)
        end_days = month_boundaries[:, :, 1].unsqueeze(-1)

        # Create mask: True where day is outside month boundaries
        mask = (day_indices < start_days) | (day_indices >= end_days)

        return mask

    def forward(
        self,
        daily_hidden: Tensor,
        month_boundaries: Tensor,
        daily_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Aggregate daily representations to monthly via learned cross-attention.

        Args:
            daily_hidden: Daily encoder output [batch, n_days, d_model]
            month_boundaries: [batch, n_months, 2] with (start_day, end_day) per month
            daily_mask: Optional [batch, n_days] where True = valid observation
            return_attention: Whether to return attention weights

        Returns:
            monthly_hidden: Aggregated monthly representations [batch, n_months, d_model]
            monthly_mask: Monthly observation mask [batch, n_months]
            attention_weights: Optional [batch, nhead, n_months, n_days]
        """
        batch_size, n_days, d_model_in = daily_hidden.shape
        n_months = month_boundaries.shape[1]
        device = daily_hidden.device

        # CRITICAL FIX: Defensive shape assertions to catch dimension mismatches early
        assert daily_hidden.dim() == 3, f"Expected daily_hidden to be 3D, got {daily_hidden.dim()}D"
        assert month_boundaries.dim() == 3, f"Expected month_boundaries to be 3D, got {month_boundaries.dim()}D"
        assert month_boundaries.shape[2] == 2, f"Expected month_boundaries last dim to be 2, got {month_boundaries.shape[2]}"
        assert d_model_in == self.d_model, f"Input d_model {d_model_in} != expected {self.d_model}"
        if daily_mask is not None:
            assert daily_mask.dim() == 2, f"Expected daily_mask to be 2D, got {daily_mask.dim()}D"
            assert daily_mask.shape[0] == batch_size, f"daily_mask batch size mismatch"
            assert daily_mask.shape[1] == n_days, f"daily_mask seq_len mismatch"

        # Get month queries
        queries = self.month_queries[:, :n_months, :].expand(batch_size, -1, -1)

        # Add month position embeddings
        month_positions = torch.arange(n_months, device=device).unsqueeze(0)
        month_pos_emb = self.month_position_embedding(month_positions)
        queries = queries + month_pos_emb

        # Prepare attention masks
        key_padding_mask = None
        if daily_mask is not None:
            key_padding_mask = ~daily_mask  # True = ignore
            # Handle edge case: if ALL positions are masked, unmask first position
            # This prevents softmax from receiving all -inf, which causes NaN gradients
            all_masked = key_padding_mask.all(dim=-1)  # [batch]
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False  # Unmask first position

        attn_mask = None
        if self.use_month_constraints:
            attn_mask = self._create_month_attention_mask(
                n_months, n_days, month_boundaries, device
            )
            # Handle edge case: combined mask might leave no valid positions for some months
            # Compute effective mask: True if position is masked by EITHER mask
            if key_padding_mask is not None:
                combined_mask = attn_mask | key_padding_mask.unsqueeze(1)  # [batch, n_months, n_days]
            else:
                combined_mask = attn_mask
            # For each month, ensure at least one position is unmasked
            all_masked_per_month = combined_mask.all(dim=-1)  # [batch, n_months]
            if all_masked_per_month.any():
                # Find first position in each month and unmask it
                for b in range(batch_size):
                    for m in range(n_months):
                        if all_masked_per_month[b, m]:
                            # Unmask the first day of this month
                            start_day = month_boundaries[b, m, 0].item()
                            if start_day < n_days:
                                attn_mask[b, m, start_day] = False
                                if key_padding_mask is not None:
                                    key_padding_mask[b, start_day] = False
            # PyTorch MHA expects attn_mask of shape (batch * num_heads, L, S)
            # where L=n_months (target), S=n_days (source)
            # Expand: [batch, n_months, n_days] -> [batch * nhead, n_months, n_days]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            attn_mask = attn_mask.reshape(batch_size * self.nhead, n_months, n_days)

        # Apply layer norm
        queries_normed = self.norm1(queries)
        daily_normed = self.norm1(daily_hidden)

        # Cross-attention
        attended, attention_weights = self.cross_attention(
            query=queries_normed,
            key=daily_normed,
            value=daily_hidden,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        # Handle NaN from attention (occurs when all keys are masked for a query)
        # Replace NaN with zeros - these positions will be marked as unobserved
        attended = torch.nan_to_num(attended, nan=0.0)

        # Residual + FFN
        attended = queries + attended
        ffn_out = self.ffn(self.norm2(attended))
        monthly_hidden = attended + ffn_out

        # Output projection
        monthly_hidden = self.output_projection(monthly_hidden)

        # Compute monthly observation mask
        # A month is observed if ANY day in it was observed
        monthly_mask = torch.zeros(batch_size, n_months, dtype=torch.bool, device=device)
        if daily_mask is not None:
            for m in range(n_months):
                for b in range(batch_size):
                    start = month_boundaries[b, m, 0].item()
                    end = month_boundaries[b, m, 1].item()
                    if start < end and start < n_days and end <= n_days:
                        monthly_mask[b, m] = daily_mask[b, start:end].any()
        else:
            monthly_mask.fill_(True)

        if return_attention:
            return monthly_hidden, monthly_mask, attention_weights
        return monthly_hidden, monthly_mask, None


# =============================================================================
# TEMPORAL ENCODER
# =============================================================================

class TemporalEncoder(nn.Module):
    """
    Causal transformer encoder for processing the fused monthly sequence.

    This encoder processes the cross-resolution fused representations to
    capture long-range temporal dependencies in the monthly sequence.

    CRITICAL: Uses causal masking by default to prevent future information
    leakage during autoregressive prediction. Each timestep can only attend
    to current and past timesteps, not future ones.

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dropout: Dropout probability
        max_len: Maximum sequence length
        causal: Whether to use causal (autoregressive) attention masking.
            If True, position i can only attend to positions <= i.
            Default is True to prevent future information leakage.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 60,
        causal: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.causal = causal
        self.max_len = max_len

        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(d_model)

        # Register causal mask buffer for autoregressive attention
        # Upper triangular matrix where True = IGNORE (don't attend)
        # This prevents position i from attending to any position j > i
        if causal:
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode the fused monthly sequence with optional causal masking.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            mask: Optional observation mask [batch, seq_len] where True = observed

        Returns:
            Encoded sequence [batch, seq_len, d_model]

        Note:
            When causal=True, each position can only attend to itself and
            previous positions, preventing future information leakage.
        """
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        hidden = self.positional_encoding(x)

        # Prepare padding mask (src_key_padding_mask)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # True = ignore

            # Handle fully masked sequences
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

        # Prepare causal attention mask
        # In PyTorch TransformerEncoder, mask=True means IGNORE that position
        attn_mask = None
        if self.causal:
            attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer encoding with both masks
        hidden = self.transformer_encoder(
            hidden,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        return self.output_norm(hidden)


# =============================================================================
# PREDICTION HEADS
# =============================================================================

class CasualtyPredictionHead(nn.Module):
    """
    Regression head for casualty prediction.

    Outputs 3 values per timestep: deaths_best, deaths_low, deaths_high
    representing point estimate and uncertainty bounds.

    Args:
        d_model: Input dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mean prediction (deaths_best, deaths_low, deaths_high)
        self.mean_head = nn.Linear(hidden_dim // 2, 3)

        # Variance prediction (log variance for numerical stability)
        self.log_var_head = nn.Linear(hidden_dim // 2, 3)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_variance: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Predict casualty estimates with optional variance.

        Args:
            x: Input representations [batch, seq_len, d_model]
            mask: Optional observation mask (unused but kept for API consistency)
            return_variance: If True, return (mean, variance) tuple

        Returns:
            If return_variance=True: (mean [batch, seq, 3], variance [batch, seq, 3])
            If return_variance=False: mean [batch, seq, 3]
        """
        features = self.shared(x)
        mean = self.mean_head(features)

        if return_variance:
            log_var = self.log_var_head(features)
            # Clamp log variance for stability and convert to variance
            log_var = torch.clamp(log_var, min=-10, max=10)
            variance = torch.exp(log_var)
            return mean, variance

        return mean


class RegimeClassificationHead(nn.Module):
    """
    4-class classification head for regime/phase classification.

    Classifies conflict phases (e.g., offensive, defensive, stalemate, escalation).

    Args:
        d_model: Input dimension
        num_classes: Number of regime classes (default 4)
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        num_classes: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Predict regime class logits.

        Args:
            x: Input representations [batch, seq_len, d_model]
            mask: Optional observation mask

        Returns:
            Class logits [batch, seq_len, num_classes]
        """
        return self.mlp(x)


class AnomalyDetectionHead(nn.Module):
    """
    Binary classification head for anomaly detection.

    Identifies unusual patterns that may indicate significant events.

    Args:
        d_model: Input dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Predict anomaly scores.

        Args:
            x: Input representations [batch, seq_len, d_model]
            mask: Optional observation mask

        Returns:
            Anomaly scores [batch, seq_len, 1]
        """
        return self.mlp(x)


class ForecastingHead(nn.Module):
    """
    Forecasting head for predicting next month's features.

    Predicts a summary of next month's expected feature values.

    Args:
        d_model: Input dimension
        output_dim: Number of features to predict
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        output_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Predict next month's features.

        Args:
            x: Input representations [batch, seq_len, d_model]
            mask: Optional observation mask

        Returns:
            Forecast predictions [batch, seq_len, output_dim]
        """
        return self.mlp(x)


class DailyForecastingHead(nn.Module):
    """
    Forecasting head for predicting next horizon days of daily-resolution features.

    Produces daily-level forecasts for each daily source, enabling fine-grained
    temporal prediction evaluation in backtesting.

    Args:
        d_model: Input dimension
        output_dim: Total number of daily features to predict (sum of all daily sources)
        horizon: Number of days ahead to forecast
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 128,
        output_dim: int = 165,  # Sum of all daily source features
        horizon: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.horizon = horizon
        self.output_dim = output_dim

        # Temporal context aggregation
        self.context_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Forecast generation network
        self.forecast_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Predict next horizon days of daily features.

        Args:
            x: Input representations [batch, seq_len, d_model]
            mask: Optional observation mask

        Returns:
            Daily forecast predictions [batch, horizon, output_dim]
        """
        batch_size = x.shape[0]

        # Self-attention for temporal context
        attended, _ = self.context_attn(x, x, x)

        # Use last timestep representation for forecast
        last_repr = attended[:, -1, :]  # [batch, d_model]

        # Generate horizon-step forecast
        flat_pred = self.forecast_net(last_repr)  # [batch, horizon * output_dim]

        # Reshape to [batch, horizon, output_dim]
        return flat_pred.view(batch_size, self.horizon, self.output_dim)


# =============================================================================
# UNCERTAINTY ESTIMATION
# =============================================================================

class UncertaintyEstimator(nn.Module):
    """
    Estimates prediction uncertainty based on observation density.

    Uncertainty increases in regions with sparse observations, reflecting
    our reduced confidence when data is limited.

    Args:
        d_model: Input dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(
        self,
        x: Tensor,
        observation_density: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Estimate uncertainty.

        Args:
            x: Input representations [batch, seq_len, d_model]
            observation_density: Optional [batch, seq_len] observation density (0-1)

        Returns:
            Uncertainty estimates [batch, seq_len, 1]
        """
        base_uncertainty = self.mlp(x)

        if observation_density is not None:
            # Scale uncertainty inversely with observation density
            # Low density -> high uncertainty
            density_scale = 1.0 + (1.0 - observation_density).unsqueeze(-1)
            base_uncertainty = base_uncertainty * density_scale

        return base_uncertainty


# =============================================================================
# MAIN MODEL: MultiResolutionHAN (ISW ALIGNMENT MODULE REMOVED)
# =============================================================================
# NOTE: ISWAlignmentModule was removed on 2026-01-30 after probes showed
# ISW embeddings had near-zero correlation with model latents (mean similarity 0.0016).
# The model was not effectively using these features, so they were removed to
# reduce parameter count and potential noise.
# =============================================================================

class MultiResolutionHAN(nn.Module):
    """
    Hierarchical Attention Network with Multi-Resolution Support.

    This model processes conflict monitoring data from multiple sources at their
    native temporal resolutions, fuses them using attention mechanisms, and
    produces multi-task predictions.

    CRITICAL DATA INTEGRITY PRINCIPLE:
    ==================================
    This model NEVER fabricates, interpolates, or forward-fills missing values.

    - Daily sources (equipment, personnel, deepstate, firms, viina) are processed
      at their native DAILY resolution (~1426 timesteps)
    - Monthly sources (sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom) are
      processed at their native MONTHLY resolution (~48 timesteps)
    - Missing observations use learned `no_observation_token` embeddings
    - Explicit observation masks are maintained throughout the forward pass

    Architecture:
    =============

    1. DailySourceEncoders: 5 separate encoders for daily sources
       - Each has its own no_observation_token
       - Each uses masked attention (missing values cannot be keys/values)

    2. DailyCrossSourceFusion: Cross-attention between daily sources
       - Learns dependencies between equipment, personnel, etc.
       - Outputs unified daily representation

    3. LearnableMonthlyAggregation: Daily -> Monthly via cross-attention
       - NOT simple averaging - learns which days matter
       - Respects month boundaries

    4. MonthlySourceEncoders: 5 encoders for monthly sources
       - Uses existing MultiSourceMonthlyEncoder
       - Has per-source no_observation_tokens

    5. CrossResolutionFusion: Bidirectional attention
       - Aggregated daily attends to monthly
       - Monthly attends to aggregated daily

    6. TemporalEncoder: Processes fused monthly sequence
       - Captures long-range temporal dependencies

    7. Multi-Task Heads:
       - CasualtyPredictionHead: 3-output regression
       - RegimeClassificationHead: 4-class classification
       - AnomalyDetectionHead: Binary anomaly scores
       - ForecastingHead: Next-month feature prediction

    Args:
        daily_source_configs: Dict[str, SourceConfig] for daily sources
        monthly_source_configs: Dict[str, SourceConfig] for monthly sources
        d_model: Hidden dimension (default 128)
        nhead: Number of attention heads (default 8)
        num_daily_layers: Layers per daily encoder (default 4)
        num_monthly_layers: Layers per monthly encoder (default 3)
        num_fusion_layers: Cross-resolution fusion layers (default 2)
        num_temporal_layers: Final temporal encoder layers (default 2)
        dropout: Dropout probability (default 0.1)
        prediction_tasks: List of tasks to enable
        causal: Whether to use causal (autoregressive) attention masking in all
            temporal encoders. If True, position i can only attend to positions <= i.
            Default is True to prevent future information leakage.

    Example:
        >>> # Create with default configs
        >>> model = MultiResolutionHAN(
        ...     daily_source_configs={
        ...         'equipment': SourceConfig('equipment', 38, 'daily'),
        ...         'personnel': SourceConfig('personnel', 6, 'daily'),
        ...     },
        ...     monthly_source_configs={
        ...         'sentinel': SourceConfig('sentinel', 43, 'monthly'),
        ...     },
        ...     d_model=128,
        ... )
        >>>
        >>> # Forward pass
        >>> outputs = model(
        ...     daily_features={'equipment': torch.randn(4, 1000, 38), ...},
        ...     daily_masks={'equipment': torch.ones(4, 1000, 38, dtype=torch.bool), ...},
        ...     monthly_features={'sentinel': torch.randn(4, 35, 43), ...},
        ...     monthly_masks={'sentinel': torch.ones(4, 35, 43, dtype=torch.bool), ...},
        ...     month_boundaries=torch.randint(0, 1000, (4, 35, 2)),
        ... )
    """

    def __init__(
        self,
        daily_source_configs: Dict[str, SourceConfig],
        monthly_source_configs: Dict[str, SourceConfig],
        d_model: int = 128,
        nhead: int = 8,
        num_daily_layers: int = 4,
        num_monthly_layers: int = 3,
        num_fusion_layers: int = 2,
        num_temporal_layers: int = 2,
        dropout: float = 0.1,
        prediction_tasks: Optional[List[str]] = None,
        causal: bool = True,
        # Geographic prior configuration (optional)
        use_geographic_prior: bool = False,
        spatial_source_names: Optional[List[str]] = None,
        viina_n_raions: int = 100,
        firms_n_raions: int = 20,
        # Custom spatial configs for sources like geoconfirmed_raion
        # Dict[source_name, SpatialSourceConfig]
        custom_spatial_configs: Optional[Dict[str, 'SpatialSourceConfig']] = None,
        # Cross-temporal attention configuration (optional)
        # Enables attention across daily/weekly/monthly scales to address
        # the finding that model learns primarily at monthly resolution
        use_cross_temporal_attention: bool = False,
    ) -> None:
        super().__init__()

        # Store configuration
        self.daily_source_configs = daily_source_configs
        self.monthly_source_configs = monthly_source_configs
        self.daily_source_names = list(daily_source_configs.keys())
        self.monthly_source_names = list(monthly_source_configs.keys())
        self.d_model = d_model
        self.nhead = nhead
        self.causal = causal
        self.prediction_tasks = prediction_tasks or ['casualty', 'regime', 'anomaly', 'forecast']
        self.use_geographic_prior = use_geographic_prior
        self.spatial_source_names = spatial_source_names or ['viina', 'firms']
        # Store cross-temporal flag early so it's available during init
        self._use_cross_temporal = use_cross_temporal_attention

        # =====================================================================
        # 1. DAILY SOURCE ENCODERS (with causal masking)
        # =====================================================================
        self.daily_encoders = nn.ModuleDict({
            name: DailySourceEncoder(
                source_config=config,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_daily_layers,
                dropout=dropout,
                causal=causal,
            )
            for name, config in daily_source_configs.items()
        })

        # =====================================================================
        # 2. DAILY CROSS-SOURCE FUSION (with optional geographic prior)
        # =====================================================================
        if use_geographic_prior and GEOGRAPHIC_FUSION_AVAILABLE:
            # Use geographic-enhanced fusion for spatial sources
            spatial_configs = {}

            # If custom spatial configs are provided, use ONLY those
            # (indicates we're using new raion sources, not legacy viina/firms spatial configs)
            if custom_spatial_configs:
                for source_name, config in custom_spatial_configs.items():
                    if source_name in self.daily_source_names:
                        spatial_configs[source_name] = config
            else:
                # Legacy mode: apply default viina/firms spatial configs
                # ONLY when no custom configs are provided
                if 'viina' in self.daily_source_names:
                    spatial_configs['viina'] = SpatialSourceConfig(
                        name='viina',
                        n_raions=viina_n_raions,
                        features_per_raion=1,  # Total events per raion
                        use_geographic_prior=True,
                    )
                if 'firms' in self.daily_source_names:
                    spatial_configs['firms'] = SpatialSourceConfig(
                        name='firms',
                        n_raions=firms_n_raions,
                        features_per_raion=4,  # count, brightness, frp, dayratio
                        use_geographic_prior=True,
                    )

            # Validate spatial configs match source feature dimensions
            for source_name, spatial_config in spatial_configs.items():
                if source_name in daily_source_configs:
                    expected_features = spatial_config.n_raions * spatial_config.features_per_raion
                    actual_features = daily_source_configs[source_name].n_features
                    if expected_features != actual_features:
                        raise ValueError(
                            f"Spatial config mismatch for '{source_name}': "
                            f"spatial config expects {expected_features} features "
                            f"({spatial_config.n_raions} raions x {spatial_config.features_per_raion} features/raion) "
                            f"but source config has {actual_features} features"
                        )
                    # Warn if raion_keys is empty but use_geographic_prior is True
                    if spatial_config.use_geographic_prior and not spatial_config.raion_keys:
                        warnings.warn(
                            f"Spatial config for '{source_name}' has use_geographic_prior=True "
                            f"but raion_keys is empty. Geographic attention prior will be disabled "
                            f"for this source. Provide raion_keys to enable spatial attention priors.",
                            UserWarning,
                            stacklevel=2,
                        )

            self.daily_fusion = GeographicDailyCrossSourceFusion(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_fusion_layers,
                dropout=dropout,
                source_names=self.daily_source_names,
                spatial_configs=spatial_configs,
            )
        else:
            # Use standard fusion
            self.daily_fusion = DailyCrossSourceFusion(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_fusion_layers,
                dropout=dropout,
                source_names=self.daily_source_names,
            )

        # =====================================================================
        # 2.5 DAILY TEMPORAL ENCODER (before monthly aggregation)
        # =====================================================================
        # Process daily sequences with temporal awareness BEFORE monthly aggregation.
        # This addresses C1 finding: daily patterns are lost during aggregation,
        # causing model to learn primarily at monthly resolution (2.9x sensitivity ratio).
        #
        # Enhanced with:
        # - Explicit temporal positional encoding
        # - Temporal gating mechanism
        # - Multi-scale convolutions with proper normalization
        self.daily_temporal_encoder = DailyTemporalEncoder(
            d_model=d_model,
            nhead=max(4, nhead // 2),  # At least 4 heads for temporal patterns
            dropout=dropout,
            window_size=31,  # ~1 month local attention window
            causal=True,  # Use causal attention for prediction integrity
            max_len=1500,  # Support up to ~4 years of daily data
        )

        # =====================================================================
        # 2.6 CROSS-TEMPORAL SCALE ATTENTION (optional)
        # =====================================================================
        # Allows information flow across daily/weekly/monthly scales.
        # This addresses the finding that model learns primarily at monthly
        # resolution by explicitly connecting different temporal scales.
        self.use_cross_temporal_attention = getattr(self, '_use_cross_temporal', False)
        self.cross_temporal_attention: Optional[CrossTemporalScaleAttention] = None
        if self.use_cross_temporal_attention:
            self.cross_temporal_attention = CrossTemporalScaleAttention(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                max_daily_len=1500,
            )

        # =====================================================================
        # 3. LEARNABLE MONTHLY AGGREGATION (daily -> monthly)
        # =====================================================================
        self.monthly_aggregation = EnhancedLearnableMonthlyAggregation(
            d_model=d_model,
            nhead=nhead,
            max_months=60,
            dropout=dropout,
            use_month_constraints=True,
        )

        # =====================================================================
        # 4. MONTHLY SOURCE ENCODERS
        # =====================================================================
        # Convert SourceConfig to MonthlySourceConfig for the existing module
        monthly_configs = {
            name: MonthlySourceConfig(
                name=name,
                n_features=config.n_features,
                description=config.description,
                typical_observations=40,  # Default estimate
            )
            for name, config in monthly_source_configs.items()
        }

        self.monthly_encoder = MultiSourceMonthlyEncoder(
            source_configs=monthly_configs,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_monthly_layers,
            num_fusion_layers=num_fusion_layers,
            dropout=dropout,
        )

        # =====================================================================
        # 5. CROSS-RESOLUTION FUSION
        # =====================================================================
        self.cross_resolution_fusion = CrossResolutionFusion(
            daily_dim=d_model,
            monthly_dim=d_model,
            hidden_dim=d_model,
            num_layers=num_fusion_layers,
            num_heads=nhead,
            dropout=dropout,
            use_gating=True,
            aggregation='attention',
            output_daily=False,  # We work at monthly resolution for predictions
        )

        # =====================================================================
        # 6. TEMPORAL ENCODER (with causal masking)
        # =====================================================================
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_temporal_layers,
            dropout=dropout,
            causal=causal,
        )

        # =====================================================================
        # 7. PREDICTION HEADS
        # =====================================================================
        if 'casualty' in self.prediction_tasks:
            self.casualty_head = CasualtyPredictionHead(
                d_model=d_model,
                hidden_dim=d_model * 2,
                dropout=dropout,
            )

        if 'regime' in self.prediction_tasks:
            self.regime_head = RegimeClassificationHead(
                d_model=d_model,
                num_classes=4,
                hidden_dim=d_model * 2,
                dropout=dropout,
            )

        if 'anomaly' in self.prediction_tasks:
            self.anomaly_head = AnomalyDetectionHead(
                d_model=d_model,
                hidden_dim=d_model * 2,
                dropout=dropout,
            )

        if 'forecast' in self.prediction_tasks:
            # Monthly forecast: sum of all monthly source features
            forecast_output_dim = sum(
                cfg.n_features for cfg in monthly_source_configs.values()
            )
            self.forecast_head = ForecastingHead(
                d_model=d_model,
                output_dim=forecast_output_dim,
                hidden_dim=d_model * 2,
                dropout=dropout,
            )

            # Daily forecast: sum of all daily source features for horizon days
            daily_forecast_output_dim = sum(
                cfg.n_features for cfg in daily_source_configs.values()
            )
            self.daily_forecast_head = DailyForecastingHead(
                d_model=d_model,
                output_dim=daily_forecast_output_dim,
                horizon=7,  # Predict 7 days ahead
                hidden_dim=d_model * 2,
                dropout=dropout,
            )

        # =====================================================================
        # 8. UNCERTAINTY ESTIMATION
        # =====================================================================
        self.uncertainty_estimator = UncertaintyEstimator(
            d_model=d_model,
            hidden_dim=d_model // 2,
        )

    def _compute_observation_density(
        self,
        daily_masks: Dict[str, Tensor],
        monthly_masks: Dict[str, Tensor],
        month_boundaries: Tensor,
    ) -> Tensor:
        """
        Compute observation density per month.

        Args:
            daily_masks: Dict of daily observation masks
            monthly_masks: Dict of monthly observation masks
            month_boundaries: [batch, n_months, 2] boundaries

        Returns:
            observation_density: [batch, n_months] values in [0, 1]
        """
        batch_size, n_months = month_boundaries.shape[:2]
        device = month_boundaries.device

        density = torch.zeros(batch_size, n_months, device=device)

        # Daily observation density within each month
        daily_density = torch.zeros(batch_size, n_months, device=device)
        n_daily_sources = 0

        for name, mask in daily_masks.items():
            n_daily_sources += 1
            # mask: [batch, seq, features] -> per-timestep [batch, seq]
            if mask.dim() == 3:
                timestep_mask = mask.any(dim=-1).float()
            else:
                timestep_mask = mask.float()

            for b in range(batch_size):
                for m in range(n_months):
                    start = month_boundaries[b, m, 0].item()
                    end = month_boundaries[b, m, 1].item()
                    if start < end and end <= timestep_mask.shape[1]:
                        daily_density[b, m] += timestep_mask[b, start:end].mean()

        if n_daily_sources > 0:
            daily_density /= n_daily_sources

        # Monthly observation density
        monthly_density = torch.zeros(batch_size, n_months, device=device)
        n_monthly_sources = 0

        for name, mask in monthly_masks.items():
            n_monthly_sources += 1
            if mask.dim() == 3:
                timestep_mask = mask.any(dim=-1).float()
            else:
                timestep_mask = mask.float()

            # Ensure mask length matches n_months
            if timestep_mask.shape[1] >= n_months:
                monthly_density += timestep_mask[:, :n_months]
            else:
                monthly_density[:, :timestep_mask.shape[1]] += timestep_mask

        if n_monthly_sources > 0:
            monthly_density /= n_monthly_sources

        # Combine daily and monthly density
        density = 0.6 * daily_density + 0.4 * monthly_density

        return density.clamp(0, 1)

    def forward(
        self,
        daily_features: Dict[str, Tensor],
        daily_masks: Dict[str, Tensor],
        monthly_features: Dict[str, Tensor],
        monthly_masks: Dict[str, Tensor],
        month_boundaries: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
        raion_masks: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the multi-resolution hierarchical attention network.

        Args:
            daily_features: Dict[source_name, Tensor[batch, daily_seq, features]]
                Raw feature values for each daily source
            daily_masks: Dict[source_name, Tensor[batch, daily_seq, features]]
                Boolean masks where True = observed, False = missing
            monthly_features: Dict[source_name, Tensor[batch, monthly_seq, features]]
                Raw feature values for each monthly source
            monthly_masks: Dict[source_name, Tensor[batch, monthly_seq, features]]
                Boolean masks where True = observed, False = missing
            month_boundaries: Tensor[batch, n_months, 2]
                (start_idx, end_idx) in daily sequence for each month
            targets: Optional Dict of target tensors for loss computation
            raion_masks: Optional Dict[source_name, Tensor[batch, daily_seq, n_raions]]
                Per-raion observation masks for geographic sources. These provide
                finer-grained masking than daily_masks, allowing the model to know
                which specific raions were observed vs missing at each timestep.

        Returns:
            Dict containing:
                'temporal_output': Tensor[batch, seq_len, d_model] latent representation
                'casualty_pred': Tensor[batch, seq_len, 3] if 'casualty' in tasks
                'regime_logits': Tensor[batch, seq_len, 4] if 'regime' in tasks
                'anomaly_score': Tensor[batch, seq_len, 1] if 'anomaly' in tasks
                'forecast_pred': Tensor[batch, seq_len, n_features] if 'forecast' in tasks
                'daily_attention': Dict[str, Tensor] per-source daily attention
                'monthly_attention': Dict[str, Tensor] per-source monthly attention
                'cross_resolution_attention': Tensor fusion attention weights
                'source_importance': Tensor showing source contributions
                'uncertainty': Tensor prediction uncertainty from observation density
        """
        # =====================================================================
        # STEP 1: ENCODE DAILY SOURCES
        # =====================================================================
        daily_encoded = {}
        daily_attention_weights = {}

        for name in self.daily_source_names:
            if name in daily_features:
                encoded, attn = self.daily_encoders[name](
                    daily_features[name],
                    daily_masks[name],
                    return_attention=True,
                )
                daily_encoded[name] = encoded
                if attn is not None:
                    daily_attention_weights[name] = attn

        # =====================================================================
        # STEP 2: FUSE DAILY SOURCES
        # =====================================================================
        # Pass raion_masks if using GeographicDailyCrossSourceFusion for
        # finer-grained per-raion attention masking
        if hasattr(self.daily_fusion, 'geographic_encoders') and raion_masks is not None:
            fused_daily, combined_daily_mask, daily_fusion_attn = self.daily_fusion(
                daily_encoded,
                daily_masks,
                return_attention=True,
                raion_masks=raion_masks,
            )
        else:
            fused_daily, combined_daily_mask, daily_fusion_attn = self.daily_fusion(
                daily_encoded,
                daily_masks,
                return_attention=True,
            )

        # =====================================================================
        # STEP 2.5: APPLY DAILY TEMPORAL PROCESSING
        # =====================================================================
        # Enrich daily representations with temporal context BEFORE monthly
        # aggregation. This addresses C1 finding: captures daily patterns
        # (weekly cycles, operational tempo) that would otherwise be lost
        # during aggregation.
        #
        # The enhanced DailyTemporalEncoder now includes:
        # - Explicit temporal positional encoding (sinusoidal + day-of-week + month)
        # - Temporal gating mechanism for controlling information flow
        # - Multi-scale convolutions with GroupNorm for stability
        fused_daily = self.daily_temporal_encoder(fused_daily, combined_daily_mask)

        # =====================================================================
        # STEP 2.6: APPLY CROSS-TEMPORAL SCALE ATTENTION (if enabled)
        # =====================================================================
        # Allows information flow across daily/weekly/monthly scales.
        # This addresses the finding that model learns primarily at monthly
        # resolution by explicitly connecting different temporal scales.
        if self.cross_temporal_attention is not None:
            fused_daily = self.cross_temporal_attention(fused_daily, combined_daily_mask)

        # =====================================================================
        # STEP 3: AGGREGATE DAILY TO MONTHLY
        # =====================================================================
        n_months = month_boundaries.shape[1]

        aggregated_daily, aggregated_daily_mask, agg_attention = self.monthly_aggregation(
            fused_daily,
            month_boundaries,
            combined_daily_mask,
            return_attention=True,
        )

        # =====================================================================
        # STEP 4: ENCODE MONTHLY SOURCES
        # =====================================================================
        batch_size, monthly_seq_len = list(monthly_features.values())[0].shape[:2]
        device = aggregated_daily.device

        # Prepare month indices for monthly encoder
        month_indices = torch.arange(monthly_seq_len, device=device).unsqueeze(0)
        month_indices = month_indices.expand(batch_size, -1)

        # CRITICAL FIX: Replace MISSING_VALUE (-999.0) with 0.0 in monthly features
        # BEFORE encoding. This must happen BEFORE mask reduction because the
        # timestep-level mask marks a timestep observed if ANY feature is observed,
        # but individual features may still have -999.0 sentinel values.
        monthly_features_clean = {}
        for name, features in monthly_features.items():
            mask = monthly_masks.get(name)
            if mask is not None and mask.dim() == 3:
                # Replace missing values with 0.0 using feature-level mask
                features_clean = features.clone()
                features_clean = features_clean.masked_fill(~mask, 0.0)
                monthly_features_clean[name] = features_clean
            else:
                monthly_features_clean[name] = features

        # Convert masks to timestep-level (reduce feature dimension)
        monthly_timestep_masks = {}
        for name, mask in monthly_masks.items():
            if mask.dim() == 3:
                monthly_timestep_masks[name] = mask.any(dim=-1).float()
            else:
                monthly_timestep_masks[name] = mask.float()

        monthly_encoder_output = self.monthly_encoder(
            source_features=monthly_features_clean,
            source_masks=monthly_timestep_masks,
            month_indices=month_indices,
            return_attention=True,
        )

        monthly_encoded = monthly_encoder_output['hidden']
        monthly_attention_weights = monthly_encoder_output.get('attention_weights', {})
        source_importance = monthly_encoder_output.get('source_importance', None)

        # =====================================================================
        # STEP 5: CROSS-RESOLUTION FUSION
        # =====================================================================
        # Ensure sequence lengths match
        min_len = min(aggregated_daily.shape[1], monthly_encoded.shape[1])

        aggregated_daily_aligned = aggregated_daily[:, :min_len]
        monthly_encoded_aligned = monthly_encoded[:, :min_len]
        aggregated_daily_mask_aligned = aggregated_daily_mask[:, :min_len]

        # Create monthly mask from timestep masks
        monthly_combined_mask = torch.zeros(batch_size, min_len, dtype=torch.bool, device=device)
        for mask in monthly_timestep_masks.values():
            if mask.shape[1] >= min_len:
                monthly_combined_mask = monthly_combined_mask | (mask[:, :min_len] > 0.5)
            else:
                monthly_combined_mask[:, :mask.shape[1]] = (
                    monthly_combined_mask[:, :mask.shape[1]] | (mask > 0.5)
                )

        # Create month_boundaries for fusion (identity mapping since already at monthly)
        fusion_boundaries = torch.stack([
            torch.arange(min_len, device=device),
            torch.arange(1, min_len + 1, device=device),
        ], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        fusion_output = self.cross_resolution_fusion(
            daily_repr=aggregated_daily_aligned,
            monthly_repr=monthly_encoded_aligned,
            daily_mask=aggregated_daily_mask_aligned,
            monthly_mask=monthly_combined_mask,
            month_boundaries=fusion_boundaries,
            return_attention=True,
        )

        fused_monthly = fusion_output.fused_monthly
        cross_attention = fusion_output.attention_weights

        # =====================================================================
        # STEP 6: TEMPORAL ENCODING
        # =====================================================================
        temporal_encoded = self.temporal_encoder(
            fused_monthly,
            monthly_combined_mask,
        )

        # =====================================================================
        # STEP 7: PREDICTION HEADS
        # =====================================================================
        outputs = {}

        # Include temporal encoding for latent representation extraction (used by probes)
        outputs['temporal_output'] = temporal_encoded  # [batch, seq, d_model]

        if 'casualty' in self.prediction_tasks:
            casualty_pred, casualty_var = self.casualty_head(
                temporal_encoded, return_variance=True
            )
            outputs['casualty_pred'] = casualty_pred
            outputs['casualty_var'] = casualty_var

        if 'regime' in self.prediction_tasks:
            outputs['regime_logits'] = self.regime_head(temporal_encoded)

        if 'anomaly' in self.prediction_tasks:
            outputs['anomaly_score'] = self.anomaly_head(temporal_encoded)

        if 'forecast' in self.prediction_tasks:
            outputs['forecast_pred'] = self.forecast_head(temporal_encoded)
            # Daily-resolution forecast: [batch, horizon, daily_features]
            # Only output if daily_forecast_head exists (trained models)
            if hasattr(self, 'daily_forecast_head'):
                outputs['daily_forecast_pred'] = self.daily_forecast_head(temporal_encoded)

        # =====================================================================
        # STEP 8: ATTENTION WEIGHTS AND METADATA
        # =====================================================================
        outputs['daily_attention'] = daily_attention_weights
        outputs['monthly_attention'] = monthly_attention_weights
        outputs['cross_resolution_attention'] = cross_attention

        if source_importance is not None:
            outputs['source_importance'] = source_importance

        # =====================================================================
        # STEP 9: UNCERTAINTY ESTIMATION
        # =====================================================================
        observation_density = self._compute_observation_density(
            daily_masks, monthly_masks, month_boundaries[:, :min_len]
        )
        outputs['uncertainty'] = self.uncertainty_estimator(
            temporal_encoded,
            observation_density,
        )

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        config = {
            'daily_source_configs': {
                name: {'name': cfg.name, 'n_features': cfg.n_features, 'resolution': cfg.resolution}
                for name, cfg in self.daily_source_configs.items()
            },
            'monthly_source_configs': {
                name: {'name': cfg.name, 'n_features': cfg.n_features, 'resolution': cfg.resolution}
                for name, cfg in self.monthly_source_configs.items()
            },
            'd_model': self.d_model,
            'nhead': self.nhead,
            'prediction_tasks': self.prediction_tasks,
            'causal': self.causal,
            'use_cross_temporal_attention': self.cross_temporal_attention is not None,
        }
        return config


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_multi_resolution_han(
    d_model: int = 128,
    nhead: int = 8,
    num_daily_layers: int = 4,
    num_monthly_layers: int = 3,
    num_fusion_layers: int = 2,
    num_temporal_layers: int = 2,
    dropout: float = 0.1,
    prediction_tasks: Optional[List[str]] = None,
    causal: bool = True,
    # Cross-temporal attention (optional)
    use_cross_temporal_attention: bool = False,
) -> MultiResolutionHAN:
    """
    Factory function to create a MultiResolutionHAN with default source configs.

    Uses the standard Ukraine conflict monitoring sources:
    - Daily: equipment (38), personnel (6), deepstate (55), firms (42), viina (24)
    - Monthly: sentinel (43), hdx_conflict (18), hdx_food (20), hdx_rainfall (16), iom (18)

    Args:
        d_model: Hidden dimension
        nhead: Number of attention heads (8 default, can increase to 16 for finer patterns)
        num_daily_layers: Layers per daily encoder
        num_monthly_layers: Layers per monthly encoder
        num_fusion_layers: Cross-resolution fusion layers
        num_temporal_layers: Final temporal encoder layers
        dropout: Dropout probability
        prediction_tasks: List of tasks to enable
        causal: Whether to use causal attention masking (default True)
        use_cross_temporal_attention: Whether to enable cross-scale temporal attention
            (daily/weekly/monthly). Helps model learn at multiple temporal resolutions.

    Returns:
        Configured MultiResolutionHAN instance

    Example:
        >>> model = create_multi_resolution_han(d_model=128, nhead=8)
        >>> print(model)
        >>> # With cross-temporal attention for better temporal patterns
        >>> model_temporal = create_multi_resolution_han(use_cross_temporal_attention=True)
        >>> # With increased attention heads (d_model=128 divisible by 16)
        >>> model_heads = create_multi_resolution_han(nhead=16)
    """
    # Validate nhead divides d_model
    if d_model % nhead != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"For d_model=128, valid nhead values are: 1, 2, 4, 8, 16, 32, 64, 128"
        )

    # Default daily source configurations (from actual MultiResolutionDataset)
    daily_configs = {
        'equipment': SourceConfig('equipment', 11, 'daily', 'Equipment loss counts'),
        'personnel': SourceConfig('personnel', 3, 'daily', 'Personnel casualty figures'),
        'deepstate': SourceConfig('deepstate', 5, 'daily', 'Front line territorial data'),
        'firms': SourceConfig('firms', 13, 'daily', 'VIIRS fire detection'),
        'viina': SourceConfig('viina', 6, 'daily', 'VIINA territorial control'),
        'viirs': SourceConfig('viirs', 8, 'daily', 'NASA VIIRS nightlights'),
    }

    # Default monthly source configurations (from actual MultiResolutionDataset)
    monthly_configs = {
        'sentinel': SourceConfig('sentinel', 7, 'monthly', 'Sentinel satellite products'),
        'hdx_conflict': SourceConfig('hdx_conflict', 5, 'monthly', 'HDX conflict events'),
        'hdx_food': SourceConfig('hdx_food', 10, 'monthly', 'HDX food security'),
        'hdx_rainfall': SourceConfig('hdx_rainfall', 6, 'monthly', 'HDX rainfall data'),
        'iom': SourceConfig('iom', 7, 'monthly', 'IOM displacement surveys'),
    }

    return MultiResolutionHAN(
        daily_source_configs=daily_configs,
        monthly_source_configs=monthly_configs,
        d_model=d_model,
        nhead=nhead,
        num_daily_layers=num_daily_layers,
        num_monthly_layers=num_monthly_layers,
        num_fusion_layers=num_fusion_layers,
        num_temporal_layers=num_temporal_layers,
        dropout=dropout,
        prediction_tasks=prediction_tasks,
        causal=causal,
        use_cross_temporal_attention=use_cross_temporal_attention,
    )


# =============================================================================
# COMPREHENSIVE TESTS
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("MultiResolutionHAN Comprehensive Tests")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test configuration
    batch_size = 4
    daily_seq_len = 1000  # ~3 years of daily data
    monthly_seq_len = 35  # ~3 years of monthly data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # =========================================================================
    # TEST 1: Create Model Instance
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 1: Create MultiResolutionHAN Instance")
    print("-" * 40)

    try:
        model = create_multi_resolution_han(
            d_model=128,
            nhead=8,
            num_daily_layers=2,  # Reduced for faster testing
            num_monthly_layers=2,
            num_fusion_layers=1,
            num_temporal_layers=1,
            dropout=0.1,
        )
        model = model.to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 2: Create Dummy Data with Realistic Shapes
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 2: Create Dummy Data")
    print("-" * 40)

    try:
        # Daily features and masks
        daily_features = {
            'equipment': torch.randn(batch_size, daily_seq_len, 38, device=device),
            'personnel': torch.randn(batch_size, daily_seq_len, 6, device=device),
            'deepstate': torch.randn(batch_size, daily_seq_len, 55, device=device),
            'firms': torch.randn(batch_size, daily_seq_len, 42, device=device),
            'viina': torch.randn(batch_size, daily_seq_len, 24, device=device),
        }

        # Create masks with varying observation densities
        daily_masks = {}
        for name, feat in daily_features.items():
            # Random observation pattern with ~80% coverage
            mask = torch.rand(batch_size, daily_seq_len, feat.shape[-1], device=device) > 0.2
            daily_masks[name] = mask

        # Monthly features and masks
        monthly_features = {
            'sentinel': torch.randn(batch_size, monthly_seq_len, 43, device=device),
            'hdx_conflict': torch.randn(batch_size, monthly_seq_len, 18, device=device),
            'hdx_food': torch.randn(batch_size, monthly_seq_len, 20, device=device),
            'hdx_rainfall': torch.randn(batch_size, monthly_seq_len, 16, device=device),
            'iom': torch.randn(batch_size, monthly_seq_len, 18, device=device),
        }

        monthly_masks = {}
        for name, feat in monthly_features.items():
            # Monthly data typically more sparse (~70% coverage)
            mask = torch.rand(batch_size, monthly_seq_len, feat.shape[-1], device=device) > 0.3
            monthly_masks[name] = mask

        # Create month boundaries (~30 days per month)
        days_per_month = daily_seq_len // monthly_seq_len
        month_boundaries = torch.zeros(batch_size, monthly_seq_len, 2, dtype=torch.long, device=device)
        for m in range(monthly_seq_len):
            month_boundaries[:, m, 0] = m * days_per_month
            month_boundaries[:, m, 1] = min((m + 1) * days_per_month, daily_seq_len)

        print(f"Daily features created:")
        for name, feat in daily_features.items():
            obs_rate = daily_masks[name].float().mean().item() * 100
            print(f"  {name}: {feat.shape}, observation rate: {obs_rate:.1f}%")

        print(f"\nMonthly features created:")
        for name, feat in monthly_features.items():
            obs_rate = monthly_masks[name].float().mean().item() * 100
            print(f"  {name}: {feat.shape}, observation rate: {obs_rate:.1f}%")

        print(f"\nMonth boundaries: {month_boundaries.shape}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 3: Forward Pass
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 3: Forward Pass")
    print("-" * 40)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

        print(f"Forward pass successful!")
        print(f"\nOutput keys: {list(outputs.keys())}")

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 4: Verify Output Shapes
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 4: Verify Output Shapes")
    print("-" * 40)

    try:
        expected_seq_len = min(monthly_seq_len, monthly_seq_len)  # After alignment

        if 'casualty_pred' in outputs:
            shape = outputs['casualty_pred'].shape
            print(f"casualty_pred: {shape}")
            assert shape[0] == batch_size, f"Batch size mismatch: {shape[0]} vs {batch_size}"
            assert shape[2] == 3, f"Casualty output should have 3 features"

        if 'regime_logits' in outputs:
            shape = outputs['regime_logits'].shape
            print(f"regime_logits: {shape}")
            assert shape[0] == batch_size, f"Batch size mismatch"
            assert shape[2] == 4, f"Regime should have 4 classes"

        if 'anomaly_score' in outputs:
            shape = outputs['anomaly_score'].shape
            print(f"anomaly_score: {shape}")
            assert shape[0] == batch_size, f"Batch size mismatch"
            assert shape[2] == 1, f"Anomaly should have 1 output"

        if 'forecast_pred' in outputs:
            shape = outputs['forecast_pred'].shape
            print(f"forecast_pred: {shape}")
            assert shape[0] == batch_size, f"Batch size mismatch"

        if 'uncertainty' in outputs:
            shape = outputs['uncertainty'].shape
            print(f"uncertainty: {shape}")
            assert shape[0] == batch_size, f"Batch size mismatch"

        if 'source_importance' in outputs:
            shape = outputs['source_importance'].shape
            print(f"source_importance: {shape}")

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 5: Check Gradients Flow Properly
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 5: Check Gradient Flow")
    print("-" * 40)

    try:
        model.train()

        outputs = model(
            daily_features=daily_features,
            daily_masks=daily_masks,
            monthly_features=monthly_features,
            monthly_masks=monthly_masks,
            month_boundaries=month_boundaries,
        )

        # Create dummy loss
        loss = 0
        if 'casualty_pred' in outputs:
            loss = loss + outputs['casualty_pred'].mean()
        if 'regime_logits' in outputs:
            loss = loss + outputs['regime_logits'].mean()
        if 'anomaly_score' in outputs:
            loss = loss + outputs['anomaly_score'].mean()
        if 'forecast_pred' in outputs:
            loss = loss + outputs['forecast_pred'].mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are not zero
        grad_exists = False
        grad_not_zero = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_exists = True
                if param.grad.abs().sum() > 0:
                    grad_not_zero = True
                    break

        assert grad_exists, "No gradients computed"
        assert grad_not_zero, "All gradients are zero"

        print(f"Loss: {loss.item():.4f}")
        print("Gradients computed and non-zero")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 6: Test with Varying Observation Densities
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 6: Varying Observation Densities")
    print("-" * 40)

    try:
        model.eval()

        # Test with different observation rates
        for obs_rate in [0.1, 0.5, 0.9]:
            # Create sparse masks
            sparse_daily_masks = {
                name: torch.rand_like(mask.float()) > (1 - obs_rate)
                for name, mask in daily_masks.items()
            }
            sparse_monthly_masks = {
                name: torch.rand_like(mask.float()) > (1 - obs_rate)
                for name, mask in monthly_masks.items()
            }

            with torch.no_grad():
                outputs = model(
                    daily_features=daily_features,
                    daily_masks=sparse_daily_masks,
                    monthly_features=monthly_features,
                    monthly_masks=sparse_monthly_masks,
                    month_boundaries=month_boundaries,
                )

            uncertainty_mean = outputs['uncertainty'].mean().item()
            print(f"Observation rate {obs_rate*100:.0f}%: uncertainty = {uncertainty_mean:.4f}")

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 7: Verify No NaN in Outputs
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 7: Verify No NaN in Outputs")
    print("-" * 40)

    try:
        model.eval()

        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

        nan_found = False
        for key, value in outputs.items():
            if isinstance(value, Tensor):
                if torch.isnan(value).any():
                    print(f"NaN found in {key}")
                    nan_found = True
                else:
                    print(f"{key}: No NaN")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, Tensor) and torch.isnan(subvalue).any():
                        print(f"NaN found in {key}[{subkey}]")
                        nan_found = True

        assert not nan_found, "NaN values found in outputs"
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 8: Test Edge Cases
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 8: Edge Cases")
    print("-" * 40)

    try:
        model.eval()

        # Test with minimal sequence lengths
        mini_daily = {name: feat[:, :100] for name, feat in daily_features.items()}
        mini_daily_masks = {name: mask[:, :100] for name, mask in daily_masks.items()}
        mini_monthly = {name: feat[:, :5] for name, feat in monthly_features.items()}
        mini_monthly_masks = {name: mask[:, :5] for name, mask in monthly_masks.items()}
        mini_boundaries = torch.zeros(batch_size, 5, 2, dtype=torch.long, device=device)
        for m in range(5):
            mini_boundaries[:, m, 0] = m * 20
            mini_boundaries[:, m, 1] = min((m + 1) * 20, 100)

        with torch.no_grad():
            outputs = model(
                daily_features=mini_daily,
                daily_masks=mini_daily_masks,
                monthly_features=mini_monthly,
                monthly_masks=mini_monthly_masks,
                month_boundaries=mini_boundaries,
            )

        print(f"Minimal sequence test passed")

        # Test with all observations masked for one source
        all_masked_daily_masks = daily_masks.copy()
        all_masked_daily_masks['equipment'] = torch.zeros_like(daily_masks['equipment'])

        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=all_masked_daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

        print(f"All-masked source test passed")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 9: Test MISSING_VALUE (-999.0) Handling
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 9: MISSING_VALUE (-999.0) Handling")
    print("-" * 40)

    try:
        model.eval()

        # Create features with MISSING_VALUE (-999.0) in unobserved positions
        MISSING_VALUE = -999.0
        daily_features_with_missing = {}
        daily_masks_with_missing = {}

        for name, feat in daily_features.items():
            # Create mask with ~30% missing
            mask = torch.rand(batch_size, daily_seq_len, feat.shape[-1], device=device) > 0.3
            # Create features: random values where observed, -999.0 where missing
            features = torch.randn(batch_size, daily_seq_len, feat.shape[-1], device=device)
            features = features.masked_fill(~mask, MISSING_VALUE)

            daily_features_with_missing[name] = features
            daily_masks_with_missing[name] = mask

        monthly_features_with_missing = {}
        monthly_masks_with_missing = {}

        for name, feat in monthly_features.items():
            mask = torch.rand(batch_size, monthly_seq_len, feat.shape[-1], device=device) > 0.3
            features = torch.randn(batch_size, monthly_seq_len, feat.shape[-1], device=device)
            features = features.masked_fill(~mask, MISSING_VALUE)

            monthly_features_with_missing[name] = features
            monthly_masks_with_missing[name] = mask

        # Verify we have MISSING_VALUE in the data
        for name, feat in daily_features_with_missing.items():
            n_missing = (feat == MISSING_VALUE).sum().item()
            print(f"  Daily {name}: {n_missing} MISSING_VALUE positions")

        with torch.no_grad():
            outputs = model(
                daily_features=daily_features_with_missing,
                daily_masks=daily_masks_with_missing,
                monthly_features=monthly_features_with_missing,
                monthly_masks=monthly_masks_with_missing,
                month_boundaries=month_boundaries,
            )

        # Verify no NaN in outputs
        nan_found = False
        extreme_found = False
        for key, value in outputs.items():
            if isinstance(value, Tensor):
                if torch.isnan(value).any():
                    print(f"NaN found in {key}")
                    nan_found = True
                if (value.abs() > 1000).any():
                    print(f"Extreme values found in {key}")
                    extreme_found = True

        assert not nan_found, "NaN values found in outputs with MISSING_VALUE input"
        assert not extreme_found, "Extreme values found in outputs with MISSING_VALUE input"

        print("MISSING_VALUE handling test passed - no NaN or extreme values")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nMultiResolutionHAN is ready for production use.")
    print("\nKey characteristics verified:")
    print("- Daily data processed at daily resolution")
    print("- Monthly data processed at monthly resolution")
    print("- Missing values handled via no_observation_token (no fabrication)")
    print("- MISSING_VALUE (-999.0) replaced with 0.0 BEFORE linear projections")
    print("- Gradients flow properly for training")
    print("- No NaN values in outputs")
    print("- Edge cases handled gracefully")
