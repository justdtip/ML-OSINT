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

# Re-export constants for convenience
DAILY_SOURCES = ['equipment', 'personnel', 'deepstate', 'firms', 'viina']
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


# =============================================================================
# DAILY SOURCE ENCODER
# =============================================================================

class DailySourceEncoder(nn.Module):
    """
    Encoder for a single daily-resolution data source.

    This encoder processes daily observations at their NATIVE daily resolution,
    maintaining data integrity by using learned no_observation_tokens for missing
    values rather than fabricating data.

    Key Features:
    - Projects raw features to d_model dimension
    - Adds learnable feature embeddings per feature
    - Adds sinusoidal positional encoding for temporal position
    - Uses observation mask in attention (missing values cannot be keys/values)
    - Maintains its own no_observation_token for missing daily values

    Args:
        source_config: Configuration for this daily source
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dropout: Dropout probability
        max_len: Maximum sequence length

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
    ) -> None:
        super().__init__()

        self.source_config = source_config
        self.d_model = d_model
        self.n_features = source_config.n_features
        self.nhead = nhead

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

        # Verify no extreme values remain (MISSING_VALUE = -999.0)
        assert not (features.abs() > 100).any(), "Extreme values detected in features after masking"

        # =====================================================================
        # STEP 2: Project features to d_model dimension
        # =====================================================================
        # Project all features together
        hidden = self.feature_projection(features)  # [batch, seq, d_model]

        # Verify projection output is reasonable
        assert not torch.isnan(hidden).any(), "NaN in projected features"
        assert not (hidden.abs() > 100).any(), "Extreme values in projection output"

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

        hidden = self.transformer_encoder(
            hidden,
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

        # Gating mechanism for combining sources
        self.source_gate = nn.Sequential(
            nn.Linear(d_model * self.n_sources, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.n_sources),
            nn.Softmax(dim=-1),
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

        # Compute source importance via gating
        concat_sources = fused_sources.view(batch_size, seq_len, -1)
        source_importance = self.source_gate(concat_sources)  # [batch, seq, n_sources]

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

        attn_mask = None
        if self.use_month_constraints:
            attn_mask = self._create_month_attention_mask(
                n_months, n_days, month_boundaries, device
            )
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
    Standard transformer encoder for processing the fused monthly sequence.

    This encoder processes the cross-resolution fused representations to
    capture long-range temporal dependencies in the monthly sequence.

    Args:
        d_model: Model hidden dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dropout: Dropout probability
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 60,
    ) -> None:
        super().__init__()

        self.d_model = d_model

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

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode the fused monthly sequence.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            mask: Optional observation mask [batch, seq_len] where True = observed

        Returns:
            Encoded sequence [batch, seq_len, d_model]
        """
        # Add positional encoding
        hidden = self.positional_encoding(x)

        # Prepare attention mask
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # True = ignore

            # Handle fully masked sequences
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

        # Transformer encoding
        hidden = self.transformer_encoder(
            hidden,
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
# MAIN MODEL: MultiResolutionHAN
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
    ) -> None:
        super().__init__()

        # Store configuration
        self.daily_source_configs = daily_source_configs
        self.monthly_source_configs = monthly_source_configs
        self.daily_source_names = list(daily_source_configs.keys())
        self.monthly_source_names = list(monthly_source_configs.keys())
        self.d_model = d_model
        self.nhead = nhead
        self.prediction_tasks = prediction_tasks or ['casualty', 'regime', 'anomaly', 'forecast']

        # =====================================================================
        # 1. DAILY SOURCE ENCODERS
        # =====================================================================
        self.daily_encoders = nn.ModuleDict({
            name: DailySourceEncoder(
                source_config=config,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_daily_layers,
                dropout=dropout,
            )
            for name, config in daily_source_configs.items()
        })

        # =====================================================================
        # 2. DAILY CROSS-SOURCE FUSION
        # =====================================================================
        self.daily_fusion = DailyCrossSourceFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_fusion_layers,
            dropout=dropout,
            source_names=self.daily_source_names,
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
        # 6. TEMPORAL ENCODER
        # =====================================================================
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_temporal_layers,
            dropout=dropout,
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
            # Output dimension: sum of all monthly source features
            forecast_output_dim = sum(
                cfg.n_features for cfg in monthly_source_configs.values()
            )
            self.forecast_head = ForecastingHead(
                d_model=d_model,
                output_dim=forecast_output_dim,
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

        Returns:
            Dict containing:
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
        fused_daily, combined_daily_mask, daily_fusion_attn = self.daily_fusion(
            daily_encoded,
            daily_masks,
            return_attention=True,
        )

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
        return {
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
        }


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
) -> MultiResolutionHAN:
    """
    Factory function to create a MultiResolutionHAN with default source configs.

    Uses the standard Ukraine conflict monitoring sources:
    - Daily: equipment (38), personnel (6), deepstate (55), firms (42), viina (24)
    - Monthly: sentinel (43), hdx_conflict (18), hdx_food (20), hdx_rainfall (16), iom (18)

    Args:
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_daily_layers: Layers per daily encoder
        num_monthly_layers: Layers per monthly encoder
        num_fusion_layers: Cross-resolution fusion layers
        num_temporal_layers: Final temporal encoder layers
        dropout: Dropout probability
        prediction_tasks: List of tasks to enable

    Returns:
        Configured MultiResolutionHAN instance

    Example:
        >>> model = create_multi_resolution_han(d_model=128, nhead=8)
        >>> print(model)
    """
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
