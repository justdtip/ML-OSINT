"""
Multi-Resolution Fusion Modules for Time Series Processing.

This module provides PyTorch components for combining representations from
different temporal resolutions (e.g., daily and monthly) using attention
mechanisms. The key principle is that information flows between resolutions
via ATTENTION, not by fabricating or interpolating missing values.

The daily encoder has already processed the full daily signal; these modules
combine that learned representation with monthly context while maintaining
proper observation masks throughout.

Author: ML Engineering Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FusionOutput(NamedTuple):
    """Output container for cross-resolution fusion.

    Attributes:
        fused_monthly: Fused representations at monthly resolution.
            Shape: (batch, num_months, hidden_dim)
        fused_daily: Optional fused representations at daily resolution.
            Shape: (batch, num_days, hidden_dim) or None
        monthly_mask: Observation mask for monthly values (True = observed).
            Shape: (batch, num_months)
        daily_mask: Optional observation mask for daily values.
            Shape: (batch, num_days) or None
        attention_weights: Dictionary of attention weight tensors for analysis.
    """
    fused_monthly: Tensor
    fused_daily: Optional[Tensor]
    monthly_mask: Tensor
    daily_mask: Optional[Tensor]
    attention_weights: Dict[str, Tensor]


class PredictionOutput(NamedTuple):
    """Output container for prediction heads.

    Attributes:
        predictions: Model predictions.
        confidence: Optional confidence scores.
        auxiliary: Optional auxiliary outputs (e.g., attention for interpretability).
    """
    predictions: Tensor
    confidence: Optional[Tensor]
    auxiliary: Dict[str, Tensor]


class TaskType(Enum):
    """Supported prediction task types."""
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_HORIZON = "multi_horizon"


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional masking.

    Implements the attention mechanism from "Attention Is All You Need"
    with proper handling of observation masks to ensure the model only
    attends to actually observed values.
    """

    def __init__(
        self,
        dropout: float = 0.1,
        scale: Optional[float] = None,
    ) -> None:
        """Initialize scaled dot-product attention.

        Args:
            dropout: Dropout probability for attention weights.
            scale: Optional custom scaling factor. If None, uses 1/sqrt(d_k).
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor. Shape: (batch, heads, seq_q, d_k)
            key: Key tensor. Shape: (batch, heads, seq_k, d_k)
            value: Value tensor. Shape: (batch, heads, seq_k, d_v)
            mask: Optional boolean mask where True indicates positions to KEEP.
                Shape: (batch, 1, 1, seq_k) or (batch, heads, seq_q, seq_k)
            return_attention: Whether to return attention weights.

        Returns:
            Output tensor of shape (batch, heads, seq_q, d_v).
            If return_attention=True, also returns attention weights.
        """
        d_k = query.size(-1)
        scale = self.scale if self.scale is not None else math.sqrt(d_k)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale

        # Apply mask: set masked positions to large negative value
        if mask is not None:
            # mask is True for positions to keep, False for positions to ignore
            # Handle edge case: if ALL positions are masked for a row, unmask first position
            # This prevents softmax from receiving all -inf, which causes NaN gradients
            all_masked = ~mask.any(dim=-1, keepdim=True)  # [batch, heads, seq_q, 1]
            if all_masked.any():
                # Create modified mask that keeps first position for all-masked rows
                mask = mask.clone()
                # Expand all_masked to match mask shape and set first position to True
                first_pos_unmask = torch.zeros_like(mask)
                first_pos_unmask[..., 0] = True
                mask = mask | (all_masked & first_pos_unmask)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)

        # Safety check: replace any remaining NaN with 0 (shouldn't happen with above fix)
        attention_weights = attention_weights.masked_fill(
            torch.isnan(attention_weights), 0.0
        )

        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        if return_attention:
            return output, attention_weights
        return output


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention between two sequences.

    This module allows one sequence (query source) to attend to another
    sequence (key/value source), enabling information flow between
    different temporal resolutions without fabricating data.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initialize multi-head cross-attention.

        Args:
            query_dim: Dimension of query input.
            key_dim: Dimension of key/value input.
            hidden_dim: Internal hidden dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            bias: Whether to include bias in linear projections.

        Raises:
            ValueError: If hidden_dim is not divisible by num_heads.
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Separate projections for query source and key/value source
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.key_proj = nn.Linear(key_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(key_dim, hidden_dim, bias=bias)
        self.output_proj = nn.Linear(hidden_dim, query_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        query_seq: Tensor,
        key_value_seq: Tensor,
        key_value_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute multi-head cross-attention.

        Args:
            query_seq: Query sequence. Shape: (batch, seq_q, query_dim)
            key_value_seq: Key/value sequence. Shape: (batch, seq_kv, key_dim)
            key_value_mask: Boolean mask for key/value sequence where True
                indicates observed (valid) positions. Shape: (batch, seq_kv)
            return_attention: Whether to return attention weights.

        Returns:
            Output tensor of shape (batch, seq_q, query_dim).
            If return_attention=True, also returns attention weights
            of shape (batch, num_heads, seq_q, seq_kv).
        """
        batch_size, seq_q, _ = query_seq.shape
        seq_kv = key_value_seq.size(1)

        # Project to multi-head space
        Q = self.query_proj(query_seq)
        K = self.key_proj(key_value_seq)
        V = self.value_proj(key_value_seq)

        # Reshape for multi-head attention: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare mask for attention
        attention_mask = None
        if key_value_mask is not None:
            # Expand mask: (batch, seq_kv) -> (batch, 1, 1, seq_kv)
            attention_mask = key_value_mask.unsqueeze(1).unsqueeze(2)

        # Compute attention
        if return_attention:
            attn_output, attn_weights = self.attention(
                Q, K, V, mask=attention_mask, return_attention=True
            )
        else:
            attn_output = self.attention(Q, K, V, mask=attention_mask)

        # Reshape back: (batch, heads, seq_q, head_dim) -> (batch, seq_q, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_q, self.hidden_dim
        )

        # Final projection
        output = self.output_proj(attn_output)
        output = self.dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between two sequences.

    Enables information flow in both directions:
    - Sequence A attends to Sequence B
    - Sequence B attends to Sequence A

    This is essential for multi-resolution fusion where both daily
    and monthly representations should inform each other.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
    ) -> None:
        """Initialize bidirectional cross-attention.

        Args:
            dim_a: Dimension of sequence A (e.g., aggregated daily).
            dim_b: Dimension of sequence B (e.g., monthly).
            hidden_dim: Internal hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            use_gating: Whether to use gating mechanism for fusion.
        """
        super().__init__()

        self.dim_a = dim_a
        self.dim_b = dim_b
        self.use_gating = use_gating

        # A attends to B
        self.cross_attn_a_to_b = MultiHeadCrossAttention(
            query_dim=dim_a,
            key_dim=dim_b,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # B attends to A
        self.cross_attn_b_to_a = MultiHeadCrossAttention(
            query_dim=dim_b,
            key_dim=dim_a,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Layer norms for residual connections
        self.norm_a = nn.LayerNorm(dim_a)
        self.norm_b = nn.LayerNorm(dim_b)

        # Optional gating mechanism
        if use_gating:
            self.gate_a = nn.Sequential(
                nn.Linear(dim_a * 2, dim_a),
                nn.Sigmoid(),
            )
            self.gate_b = nn.Sequential(
                nn.Linear(dim_b * 2, dim_b),
                nn.Sigmoid(),
            )

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Dict[str, Tensor]]
    ]:
        """Compute bidirectional cross-attention.

        Args:
            seq_a: Sequence A (e.g., aggregated daily). Shape: (batch, seq_a, dim_a)
            seq_b: Sequence B (e.g., monthly). Shape: (batch, seq_b, dim_b)
            mask_a: Observation mask for sequence A. Shape: (batch, seq_a)
            mask_b: Observation mask for sequence B. Shape: (batch, seq_b)
            return_attention: Whether to return attention weight tensors.

        Returns:
            Tuple of (updated_seq_a, updated_seq_b).
            If return_attention=True, also returns dict of attention weights.
        """
        attention_weights = {}

        # A attends to B (daily attends to monthly context)
        if return_attention:
            cross_a, attn_a_to_b = self.cross_attn_a_to_b(
                seq_a, seq_b, key_value_mask=mask_b, return_attention=True
            )
            attention_weights['a_to_b'] = attn_a_to_b
        else:
            cross_a = self.cross_attn_a_to_b(seq_a, seq_b, key_value_mask=mask_b)

        # B attends to A (monthly attends to daily detail)
        if return_attention:
            cross_b, attn_b_to_a = self.cross_attn_b_to_a(
                seq_b, seq_a, key_value_mask=mask_a, return_attention=True
            )
            attention_weights['b_to_a'] = attn_b_to_a
        else:
            cross_b = self.cross_attn_b_to_a(seq_b, seq_a, key_value_mask=mask_a)

        # Fuse with gating or simple residual
        if self.use_gating:
            gate_a = self.gate_a(torch.cat([seq_a, cross_a], dim=-1))
            gate_b = self.gate_b(torch.cat([seq_b, cross_b], dim=-1))
            updated_a = self.norm_a(seq_a + gate_a * cross_a)
            updated_b = self.norm_b(seq_b + gate_b * cross_b)
        else:
            updated_a = self.norm_a(seq_a + cross_a)
            updated_b = self.norm_b(seq_b + cross_b)

        if return_attention:
            return updated_a, updated_b, attention_weights
        return updated_a, updated_b


class TemporalAggregator(nn.Module):
    """Aggregates daily representations to monthly boundaries.

    Takes daily encoder outputs and aggregates them to monthly resolution
    for fusion with monthly encoder outputs. Supports multiple aggregation
    strategies and properly handles variable-length months.
    """

    def __init__(
        self,
        daily_dim: int,
        output_dim: int,
        aggregation: str = 'attention',
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize temporal aggregator.

        Args:
            daily_dim: Dimension of daily representations.
            output_dim: Dimension of aggregated monthly output.
            aggregation: Aggregation strategy ('attention', 'mean', 'last', 'max').
            num_heads: Number of attention heads (if using attention aggregation).
            dropout: Dropout probability.
        """
        super().__init__()

        self.daily_dim = daily_dim
        self.output_dim = output_dim
        self.aggregation = aggregation

        if aggregation == 'attention':
            # Learnable query for each month
            self.month_query = nn.Parameter(torch.randn(1, 1, output_dim))
            self.cross_attn = MultiHeadCrossAttention(
                query_dim=output_dim,
                key_dim=daily_dim,
                hidden_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.pre_proj = nn.Linear(daily_dim, output_dim) if daily_dim != output_dim else nn.Identity()
        else:
            self.proj = nn.Linear(daily_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        daily_repr: Tensor,
        daily_mask: Tensor,
        month_boundaries: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Aggregate daily representations to monthly resolution.

        Args:
            daily_repr: Daily representations. Shape: (batch, num_days, daily_dim)
            daily_mask: Daily observation mask. Shape: (batch, num_days)
            month_boundaries: Tensor of (start_idx, end_idx) for each month.
                Shape: (batch, num_months, 2) or (num_months, 2) if shared.

        Returns:
            Tuple of:
            - Aggregated monthly representations. Shape: (batch, num_months, output_dim)
            - Monthly mask (True if any day in month was observed). Shape: (batch, num_months)
        """
        batch_size = daily_repr.size(0)

        # Handle shared vs per-batch boundaries
        if month_boundaries.dim() == 2:
            month_boundaries = month_boundaries.unsqueeze(0).expand(batch_size, -1, -1)

        num_months = month_boundaries.size(1)
        device = daily_repr.device

        monthly_repr_list = []
        monthly_mask_list = []

        for month_idx in range(num_months):
            start_indices = month_boundaries[:, month_idx, 0]  # (batch,)
            end_indices = month_boundaries[:, month_idx, 1]    # (batch,)

            # Get maximum range for this month across batch
            max_start = start_indices.min().item()
            max_end = end_indices.max().item()

            # Extract daily window for this month
            month_daily = []
            month_mask = []

            for b in range(batch_size):
                start = start_indices[b].item()
                end = end_indices[b].item()

                if start >= end:
                    # Empty month - use zeros
                    month_daily.append(torch.zeros(1, self.daily_dim, device=device))
                    month_mask.append(torch.zeros(1, dtype=torch.bool, device=device))
                else:
                    month_daily.append(daily_repr[b, start:end])
                    month_mask.append(daily_mask[b, start:end])

            # Pad to same length within this month
            max_len = max(m.size(0) for m in month_daily)
            padded_daily = torch.zeros(batch_size, max_len, self.daily_dim, device=device)
            padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

            for b in range(batch_size):
                seq_len = month_daily[b].size(0)
                padded_daily[b, :seq_len] = month_daily[b]
                padded_mask[b, :seq_len] = month_mask[b]

            # Aggregate based on strategy
            if self.aggregation == 'attention':
                # Use cross-attention with learnable query
                query = self.month_query.expand(batch_size, -1, -1)
                projected_daily = self.pre_proj(padded_daily)
                month_agg = self.cross_attn(
                    query, projected_daily, key_value_mask=padded_mask
                )
                month_agg = month_agg.squeeze(1)  # (batch, output_dim)
            elif self.aggregation == 'mean':
                # Masked mean
                mask_expanded = padded_mask.unsqueeze(-1).float()
                sum_repr = (padded_daily * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                month_agg = self.proj(sum_repr / count)
            elif self.aggregation == 'last':
                # Last observed value in month
                last_indices = padded_mask.long().cumsum(dim=1).argmax(dim=1)
                month_agg = self.proj(
                    padded_daily[torch.arange(batch_size), last_indices]
                )
            elif self.aggregation == 'max':
                # Max pooling over observed values
                masked_daily = padded_daily.masked_fill(
                    ~padded_mask.unsqueeze(-1), float('-inf')
                )
                month_agg = self.proj(masked_daily.max(dim=1)[0])
            else:
                raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

            monthly_repr_list.append(month_agg)
            # Month is observed if ANY day was observed
            monthly_mask_list.append(padded_mask.any(dim=1))

        # Stack to get final monthly representations
        monthly_repr = torch.stack(monthly_repr_list, dim=1)  # (batch, num_months, output_dim)
        monthly_mask = torch.stack(monthly_mask_list, dim=1)  # (batch, num_months)

        return monthly_repr, monthly_mask


class MonthlyToDailyUpsampler(nn.Module):
    """Upsamples monthly representations back to daily resolution.

    After fusion, this module can broadcast monthly context back to
    daily resolution for tasks that require daily-level predictions.
    Uses attention to create position-specific daily representations
    from the monthly context.
    """

    def __init__(
        self,
        monthly_dim: int,
        daily_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_positional: bool = True,
    ) -> None:
        """Initialize upsampler.

        Args:
            monthly_dim: Dimension of monthly representations.
            daily_dim: Dimension of output daily representations.
            hidden_dim: Internal hidden dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            use_positional: Whether to use positional encoding for daily positions.
        """
        super().__init__()

        self.monthly_dim = monthly_dim
        self.daily_dim = daily_dim
        self.use_positional = use_positional

        # Cross-attention: daily queries attend to monthly context
        self.cross_attn = MultiHeadCrossAttention(
            query_dim=daily_dim,
            key_dim=monthly_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Project monthly to daily dim for queries
        self.query_proj = nn.Linear(monthly_dim, daily_dim)

        if use_positional:
            # Learnable day-of-month embeddings (max 31 days)
            self.day_position_embed = nn.Embedding(31, daily_dim)

        self.output_proj = nn.Linear(daily_dim, daily_dim)
        self.norm = nn.LayerNorm(daily_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        monthly_repr: Tensor,
        monthly_mask: Tensor,
        month_boundaries: Tensor,
        num_days: int,
    ) -> Tuple[Tensor, Tensor]:
        """Upsample monthly representations to daily resolution.

        Args:
            monthly_repr: Monthly representations. Shape: (batch, num_months, monthly_dim)
            monthly_mask: Monthly observation mask. Shape: (batch, num_months)
            month_boundaries: Tensor of (start_idx, end_idx) for each month.
                Shape: (batch, num_months, 2) or (num_months, 2) if shared.
            num_days: Total number of days in output.

        Returns:
            Tuple of:
            - Daily representations. Shape: (batch, num_days, daily_dim)
            - Daily mask (propagated from monthly). Shape: (batch, num_days)
        """
        batch_size = monthly_repr.size(0)
        device = monthly_repr.device

        if month_boundaries.dim() == 2:
            month_boundaries = month_boundaries.unsqueeze(0).expand(batch_size, -1, -1)

        num_months = month_boundaries.size(1)

        # Initialize output tensors
        daily_repr = torch.zeros(batch_size, num_days, self.daily_dim, device=device)
        daily_mask = torch.zeros(batch_size, num_days, dtype=torch.bool, device=device)

        for month_idx in range(num_months):
            start_indices = month_boundaries[:, month_idx, 0]
            end_indices = month_boundaries[:, month_idx, 1]

            for b in range(batch_size):
                start = start_indices[b].item()
                end = end_indices[b].item()

                if start >= end:
                    continue

                days_in_month = end - start

                # Create query for each day in this month
                month_context = monthly_repr[b:b+1]  # (1, num_months, monthly_dim)
                month_mask_b = monthly_mask[b:b+1]   # (1, num_months)

                # Project to daily dim for base queries
                base_query = self.query_proj(monthly_repr[b, month_idx:month_idx+1])  # (1, daily_dim)
                base_query = base_query.unsqueeze(0).expand(1, days_in_month, -1)  # (1, days, daily_dim)

                # Add positional encoding for day-of-month
                if self.use_positional:
                    day_positions = torch.arange(days_in_month, device=device)
                    pos_embed = self.day_position_embed(day_positions)  # (days, daily_dim)
                    base_query = base_query + pos_embed.unsqueeze(0)

                # Cross-attend to all monthly context
                daily_context = self.cross_attn(
                    base_query, month_context, key_value_mask=month_mask_b
                )  # (1, days, daily_dim)

                daily_context = self.norm(daily_context + base_query)
                daily_context = self.output_proj(daily_context)
                daily_context = self.dropout(daily_context)

                # Place in output
                daily_repr[b, start:end] = daily_context.squeeze(0)
                daily_mask[b, start:end] = monthly_mask[b, month_idx]

        return daily_repr, daily_mask




class CrossResolutionFusion(nn.Module):
    """Cross-resolution fusion module for combining daily and monthly representations.

    This is the main module that orchestrates the fusion of representations from
    different temporal resolutions. It takes aggregated daily representations
    (at monthly boundaries) and monthly encoder outputs, applies bidirectional
    cross-attention, and produces fused representations for downstream tasks.

    Key Design Principles:
    1. Information flows via ATTENTION, not by fabricating data
    2. Observation masks are maintained throughout - the model always knows
       which monthly values were actually observed
    3. Bidirectional attention allows daily and monthly to inform each other
    4. Optional upsampling to daily resolution for fine-grained predictions

    Example:
        >>> fusion = CrossResolutionFusion(
        ...     daily_dim=256,
        ...     monthly_dim=128,
        ...     hidden_dim=256,
        ...     num_layers=2,
        ... )
        >>> output = fusion(
        ...     daily_repr=daily_encoder_output,
        ...     monthly_repr=monthly_encoder_output,
        ...     daily_mask=daily_obs_mask,
        ...     monthly_mask=monthly_obs_mask,
        ...     month_boundaries=boundaries,
        ... )
        >>> fused_monthly = output.fused_monthly  # For monthly prediction
    """

    def __init__(
        self,
        daily_dim: int,
        monthly_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
        aggregation: str = 'attention',
        output_daily: bool = False,
    ) -> None:
        """Initialize cross-resolution fusion module.

        Args:
            daily_dim: Dimension of daily encoder representations.
            monthly_dim: Dimension of monthly encoder representations.
            hidden_dim: Internal hidden dimension for attention.
            num_layers: Number of bidirectional cross-attention layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            use_gating: Whether to use gating in bidirectional attention.
            aggregation: Strategy for aggregating daily to monthly
                ('attention', 'mean', 'last', 'max').
            output_daily: Whether to also output at daily resolution.
        """
        super().__init__()

        self.daily_dim = daily_dim
        self.monthly_dim = monthly_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_daily = output_daily

        # Aggregate daily to monthly resolution
        self.daily_aggregator = TemporalAggregator(
            daily_dim=daily_dim,
            output_dim=hidden_dim,
            aggregation=aggregation,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        # Project monthly to hidden dim
        self.monthly_proj = nn.Linear(monthly_dim, hidden_dim)
        self.monthly_norm = nn.LayerNorm(hidden_dim)

        # Stack of bidirectional cross-attention layers
        self.fusion_layers = nn.ModuleList([
            BidirectionalCrossAttention(
                dim_a=hidden_dim,
                dim_b=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gating=use_gating,
            )
            for _ in range(num_layers)
        ])

        # Feed-forward layers after attention
        self.ffn_daily = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_monthly = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm_daily = nn.LayerNorm(hidden_dim)
        self.ffn_norm_monthly = nn.LayerNorm(hidden_dim)

        # Optional upsampler for daily output
        if output_daily:
            self.upsampler = MonthlyToDailyUpsampler(
                monthly_dim=hidden_dim,
                daily_dim=daily_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

    def forward(
        self,
        daily_repr: Tensor,
        monthly_repr: Tensor,
        daily_mask: Tensor,
        monthly_mask: Tensor,
        month_boundaries: Tensor,
        return_attention: bool = False,
    ) -> FusionOutput:
        """Fuse daily and monthly representations.

        Args:
            daily_repr: Daily encoder output. Shape: (batch, num_days, daily_dim)
            monthly_repr: Monthly encoder output. Shape: (batch, num_months, monthly_dim)
            daily_mask: Daily observation mask (True = observed).
                Shape: (batch, num_days)
            monthly_mask: Monthly observation mask (True = observed).
                Shape: (batch, num_months)
            month_boundaries: Boundaries for aggregating daily to monthly.
                Shape: (batch, num_months, 2) or (num_months, 2)
            return_attention: Whether to include attention weights in output.

        Returns:
            FusionOutput containing fused representations and masks.
        """
        attention_weights = {}

        # Step 1: Aggregate daily representations to monthly resolution
        aggregated_daily, aggregated_daily_mask = self.daily_aggregator(
            daily_repr, daily_mask, month_boundaries
        )

        # Step 2: Project monthly to hidden dimension
        monthly_hidden = self.monthly_proj(monthly_repr)
        monthly_hidden = self.monthly_norm(monthly_hidden)

        # Step 3: Apply bidirectional cross-attention layers
        daily_hidden = aggregated_daily

        for layer_idx, fusion_layer in enumerate(self.fusion_layers):
            if return_attention:
                daily_hidden, monthly_hidden, layer_attn = fusion_layer(
                    daily_hidden, monthly_hidden,
                    mask_a=aggregated_daily_mask,
                    mask_b=monthly_mask,
                    return_attention=True,
                )
                attention_weights[f'layer_{layer_idx}'] = layer_attn
            else:
                daily_hidden, monthly_hidden = fusion_layer(
                    daily_hidden, monthly_hidden,
                    mask_a=aggregated_daily_mask,
                    mask_b=monthly_mask,
                )

        # Step 4: Apply feed-forward networks
        daily_hidden = self.ffn_norm_daily(
            daily_hidden + self.ffn_daily(daily_hidden)
        )
        monthly_hidden = self.ffn_norm_monthly(
            monthly_hidden + self.ffn_monthly(monthly_hidden)
        )

        # Step 5: Combine fused representations for monthly output
        # The monthly_hidden has been enriched via cross-attention from daily
        # The daily_hidden (aggregated) has been enriched from monthly
        # We use monthly_hidden as the primary output since it maintains the
        # original monthly resolution aligned with monthly_mask
        # Note: daily_hidden may have different seq length than monthly_hidden
        # because it's based on month_boundaries, so we don't add them directly
        fused_monthly = monthly_hidden  # Already enriched via bidirectional attention

        # Step 6: Optionally upsample to daily resolution
        fused_daily = None
        output_daily_mask = None

        if self.output_daily:
            num_days = daily_repr.size(1)
            fused_daily, output_daily_mask = self.upsampler(
                fused_monthly, monthly_mask, month_boundaries, num_days
            )
            # Add original daily representation via residual
            # Project daily to hidden dim for addition
            if hasattr(self.daily_aggregator, 'pre_proj') and isinstance(
                self.daily_aggregator.pre_proj, nn.Linear
            ):
                daily_proj = nn.functional.linear(
                    daily_repr,
                    self.daily_aggregator.pre_proj.weight,
                    self.daily_aggregator.pre_proj.bias,
                )
            else:
                # Identity case - need to project if dims don't match
                if self.daily_dim != self.hidden_dim:
                    # Create projection on the fly (not ideal but handles edge case)
                    daily_proj = nn.functional.linear(
                        daily_repr,
                        torch.eye(self.hidden_dim, self.daily_dim, device=daily_repr.device)
                    )
                else:
                    daily_proj = daily_repr
            fused_daily = fused_daily + daily_proj

        return FusionOutput(
            fused_monthly=fused_monthly,
            fused_daily=fused_daily,
            monthly_mask=monthly_mask,
            daily_mask=output_daily_mask,
            attention_weights=attention_weights,
        )


class ForecastingHead(nn.Module):
    """Prediction head for time series forecasting tasks.

    Takes fused representations and produces forecasts for future time steps.
    Supports both point predictions and distributional outputs for uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        forecast_horizon: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_distribution: bool = False,
    ) -> None:
        """Initialize forecasting head.

        Args:
            input_dim: Dimension of input representations.
            forecast_horizon: Number of time steps to forecast.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            output_distribution: Whether to output distribution parameters
                (mean and variance) instead of point predictions.
        """
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.output_distribution = output_distribution

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output projection
        if output_distribution:
            self.mean_proj = nn.Linear(hidden_dim, forecast_horizon)
            self.var_proj = nn.Linear(hidden_dim, forecast_horizon)
        else:
            self.output_proj = nn.Linear(hidden_dim, forecast_horizon)

    def forward(
        self,
        representations: Tensor,
        mask: Optional[Tensor] = None,
    ) -> PredictionOutput:
        """Generate forecasts from fused representations.

        Args:
            representations: Fused representations. Shape: (batch, seq, input_dim)
            mask: Optional mask indicating valid positions. Shape: (batch, seq)

        Returns:
            PredictionOutput with forecasts and optional confidence.
        """
        # Use last valid position for forecasting
        if mask is not None:
            # Find last observed position for each batch item
            last_indices = mask.long().cumsum(dim=1).argmax(dim=1)
            batch_size = representations.size(0)
            last_repr = representations[
                torch.arange(batch_size, device=representations.device),
                last_indices
            ]
        else:
            # Use last position
            last_repr = representations[:, -1]  # (batch, input_dim)

        # Encode
        hidden = self.encoder(last_repr)  # (batch, hidden_dim)

        # Generate predictions
        if self.output_distribution:
            mean = self.mean_proj(hidden)
            log_var = self.var_proj(hidden)
            variance = F.softplus(log_var) + 1e-6  # Ensure positive

            return PredictionOutput(
                predictions=mean,
                confidence=1.0 / variance,  # Higher variance = lower confidence
                auxiliary={'variance': variance, 'log_var': log_var},
            )
        else:
            predictions = self.output_proj(hidden)

            return PredictionOutput(
                predictions=predictions,
                confidence=None,
                auxiliary={},
            )


class ClassificationHead(nn.Module):
    """Prediction head for classification tasks.

    Takes fused representations and produces class probabilities.
    Supports binary, multiclass, and multilabel classification.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        multilabel: bool = False,
        pooling: str = 'last',
    ) -> None:
        """Initialize classification head.

        Args:
            input_dim: Dimension of input representations.
            num_classes: Number of output classes.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            multilabel: Whether this is multilabel classification.
            pooling: How to pool sequence ('last', 'mean', 'max', 'attention').
        """
        super().__init__()

        self.num_classes = num_classes
        self.multilabel = multilabel
        self.pooling = pooling

        # Attention pooling
        if pooling == 'attention':
            self.attention_weights = nn.Linear(input_dim, 1)

        # Hidden layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _pool(
        self,
        representations: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Pool sequence representations to single vector.

        Args:
            representations: Shape (batch, seq, input_dim)
            mask: Shape (batch, seq)

        Returns:
            Pooled representation of shape (batch, input_dim)
        """
        if mask is None:
            mask = torch.ones(
                representations.size(0),
                representations.size(1),
                dtype=torch.bool,
                device=representations.device,
            )

        if self.pooling == 'last':
            last_indices = mask.long().cumsum(dim=1).argmax(dim=1)
            batch_size = representations.size(0)
            return representations[
                torch.arange(batch_size, device=representations.device),
                last_indices
            ]

        elif self.pooling == 'mean':
            mask_expanded = mask.unsqueeze(-1).float()
            sum_repr = (representations * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            return sum_repr / count

        elif self.pooling == 'max':
            masked_repr = representations.masked_fill(
                ~mask.unsqueeze(-1), float('-inf')
            )
            return masked_repr.max(dim=1)[0]

        elif self.pooling == 'attention':
            # Compute attention weights
            attn_scores = self.attention_weights(representations).squeeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            return (representations * attn_weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def forward(
        self,
        representations: Tensor,
        mask: Optional[Tensor] = None,
    ) -> PredictionOutput:
        """Generate class predictions from fused representations.

        Args:
            representations: Fused representations. Shape: (batch, seq, input_dim)
            mask: Optional mask indicating valid positions. Shape: (batch, seq)

        Returns:
            PredictionOutput with class probabilities and confidence.
        """
        # Pool sequence
        pooled = self._pool(representations, mask)  # (batch, input_dim)

        # Encode
        hidden = self.encoder(pooled)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(hidden)  # (batch, num_classes)

        if self.multilabel:
            probs = torch.sigmoid(logits)
            confidence = torch.abs(probs - 0.5) * 2  # Confidence based on distance from 0.5
        else:
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0]  # Max probability as confidence

        return PredictionOutput(
            predictions=probs,
            confidence=confidence,
            auxiliary={'logits': logits},
        )


class MultiHorizonForecastingHead(nn.Module):
    """Prediction head for multi-horizon forecasting.

    Produces forecasts at multiple horizons simultaneously, with
    separate uncertainty estimates for each horizon.
    """

    def __init__(
        self,
        input_dim: int,
        horizons: List[int],
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        share_encoder: bool = True,
    ) -> None:
        """Initialize multi-horizon forecasting head.

        Args:
            input_dim: Dimension of input representations.
            horizons: List of forecast horizons (e.g., [1, 3, 6, 12]).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            share_encoder: Whether to share encoder across horizons.
        """
        super().__init__()

        self.horizons = horizons
        self.share_encoder = share_encoder

        # Shared or separate encoders
        if share_encoder:
            layers = []
            current_dim = input_dim
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                current_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)

            # Separate output heads per horizon
            self.horizon_heads = nn.ModuleDict({
                str(h): nn.Linear(hidden_dim, 2)  # mean and variance
                for h in horizons
            })
        else:
            self.horizon_encoders = nn.ModuleDict()
            for h in horizons:
                layers = []
                current_dim = input_dim
                for _ in range(num_layers):
                    layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                    current_dim = hidden_dim
                layers.append(nn.Linear(hidden_dim, 2))
                self.horizon_encoders[str(h)] = nn.Sequential(*layers)

    def forward(
        self,
        representations: Tensor,
        mask: Optional[Tensor] = None,
    ) -> PredictionOutput:
        """Generate multi-horizon forecasts.

        Args:
            representations: Fused representations. Shape: (batch, seq, input_dim)
            mask: Optional mask indicating valid positions. Shape: (batch, seq)

        Returns:
            PredictionOutput with predictions dict containing forecasts per horizon.
        """
        # Get last valid representation
        if mask is not None:
            last_indices = mask.long().cumsum(dim=1).argmax(dim=1)
            batch_size = representations.size(0)
            last_repr = representations[
                torch.arange(batch_size, device=representations.device),
                last_indices
            ]
        else:
            last_repr = representations[:, -1]

        horizon_predictions = {}
        horizon_variances = {}

        if self.share_encoder:
            hidden = self.encoder(last_repr)
            for h in self.horizons:
                output = self.horizon_heads[str(h)](hidden)
                horizon_predictions[h] = output[:, 0]
                horizon_variances[h] = F.softplus(output[:, 1]) + 1e-6
        else:
            for h in self.horizons:
                output = self.horizon_encoders[str(h)](last_repr)
                horizon_predictions[h] = output[:, 0]
                horizon_variances[h] = F.softplus(output[:, 1]) + 1e-6

        # Stack predictions
        predictions = torch.stack(
            [horizon_predictions[h] for h in self.horizons], dim=1
        )
        variances = torch.stack(
            [horizon_variances[h] for h in self.horizons], dim=1
        )

        return PredictionOutput(
            predictions=predictions,
            confidence=1.0 / variances,
            auxiliary={
                'variances': variances,
                'horizon_predictions': horizon_predictions,
                'horizon_variances': horizon_variances,
            },
        )


class MultiResolutionPredictor(nn.Module):
    """Complete multi-resolution prediction model.

    Combines CrossResolutionFusion with task-specific prediction heads
    for end-to-end training and inference.

    Example:
        >>> model = MultiResolutionPredictor(
        ...     daily_dim=256,
        ...     monthly_dim=128,
        ...     hidden_dim=256,
        ...     task_type=TaskType.FORECASTING,
        ...     forecast_horizon=12,
        ... )
        >>> output = model(
        ...     daily_repr=daily_enc,
        ...     monthly_repr=monthly_enc,
        ...     daily_mask=daily_mask,
        ...     monthly_mask=monthly_mask,
        ...     month_boundaries=boundaries,
        ... )
    """

    def __init__(
        self,
        daily_dim: int,
        monthly_dim: int,
        hidden_dim: int,
        task_type: TaskType = TaskType.FORECASTING,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        # Forecasting params
        forecast_horizon: int = 12,
        output_distribution: bool = True,
        # Classification params
        num_classes: int = 2,
        multilabel: bool = False,
        # Multi-horizon params
        horizons: Optional[List[int]] = None,
    ) -> None:
        """Initialize complete multi-resolution predictor.

        Args:
            daily_dim: Dimension of daily encoder representations.
            monthly_dim: Dimension of monthly encoder representations.
            hidden_dim: Internal hidden dimension.
            task_type: Type of prediction task.
            num_fusion_layers: Number of fusion layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            forecast_horizon: Horizon for single-horizon forecasting.
            output_distribution: Whether to output distributions for forecasting.
            num_classes: Number of classes for classification.
            multilabel: Whether classification is multilabel.
            horizons: List of horizons for multi-horizon forecasting.
        """
        super().__init__()

        self.task_type = task_type

        # Fusion module
        self.fusion = CrossResolutionFusion(
            daily_dim=daily_dim,
            monthly_dim=monthly_dim,
            hidden_dim=hidden_dim,
            num_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_daily=False,  # Most tasks use monthly output
        )

        # Task-specific head
        if task_type == TaskType.FORECASTING:
            self.head = ForecastingHead(
                input_dim=hidden_dim,
                forecast_horizon=forecast_horizon,
                hidden_dim=hidden_dim,
                dropout=dropout,
                output_distribution=output_distribution,
            )
        elif task_type == TaskType.CLASSIFICATION:
            self.head = ClassificationHead(
                input_dim=hidden_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                multilabel=multilabel,
            )
        elif task_type == TaskType.MULTI_HORIZON:
            self.head = MultiHorizonForecastingHead(
                input_dim=hidden_dim,
                horizons=horizons or [1, 3, 6, 12],
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif task_type == TaskType.REGRESSION:
            self.head = ForecastingHead(
                input_dim=hidden_dim,
                forecast_horizon=1,
                hidden_dim=hidden_dim,
                dropout=dropout,
                output_distribution=output_distribution,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def forward(
        self,
        daily_repr: Tensor,
        monthly_repr: Tensor,
        daily_mask: Tensor,
        monthly_mask: Tensor,
        month_boundaries: Tensor,
        return_attention: bool = False,
    ) -> Tuple[PredictionOutput, FusionOutput]:
        """Generate predictions from multi-resolution inputs.

        Args:
            daily_repr: Daily encoder output. Shape: (batch, num_days, daily_dim)
            monthly_repr: Monthly encoder output. Shape: (batch, num_months, monthly_dim)
            daily_mask: Daily observation mask. Shape: (batch, num_days)
            monthly_mask: Monthly observation mask. Shape: (batch, num_months)
            month_boundaries: Month boundary indices. Shape: (num_months, 2) or (batch, num_months, 2)
            return_attention: Whether to return attention weights.

        Returns:
            Tuple of (PredictionOutput, FusionOutput).
        """
        # Fuse representations
        fusion_output = self.fusion(
            daily_repr=daily_repr,
            monthly_repr=monthly_repr,
            daily_mask=daily_mask,
            monthly_mask=monthly_mask,
            month_boundaries=month_boundaries,
            return_attention=return_attention,
        )

        # Generate predictions
        prediction_output = self.head(
            representations=fusion_output.fused_monthly,
            mask=fusion_output.monthly_mask,
        )

        return prediction_output, fusion_output


def create_month_boundaries(
    num_days: int,
    days_per_month: Union[int, List[int]] = 30,
) -> Tensor:
    """Utility to create month boundary tensor.

    Args:
        num_days: Total number of days.
        days_per_month: Either fixed days per month or list of days per month.

    Returns:
        Month boundaries tensor of shape (num_months, 2).
    """
    if isinstance(days_per_month, int):
        num_months = (num_days + days_per_month - 1) // days_per_month
        boundaries = []
        for i in range(num_months):
            start = i * days_per_month
            end = min((i + 1) * days_per_month, num_days)
            boundaries.append([start, end])
    else:
        boundaries = []
        current = 0
        for days in days_per_month:
            end = min(current + days, num_days)
            boundaries.append([current, end])
            current = end
            if current >= num_days:
                break

    return torch.tensor(boundaries, dtype=torch.long)


# Convenience function for testing
def _test_fusion_module():
    """Test the fusion module with dummy data."""
    batch_size = 4
    num_months = 12
    days_per_month = 30
    num_days = num_months * days_per_month  # 360 days to match exactly 12 months
    daily_dim = 256
    monthly_dim = 128
    hidden_dim = 256

    # Create dummy data
    daily_repr = torch.randn(batch_size, num_days, daily_dim)
    monthly_repr = torch.randn(batch_size, num_months, monthly_dim)

    # Create masks (some observations missing)
    daily_mask = torch.rand(batch_size, num_days) > 0.1
    monthly_mask = torch.rand(batch_size, num_months) > 0.2

    # Create month boundaries - must match num_months exactly
    month_boundaries = create_month_boundaries(num_days, days_per_month=days_per_month)

    # Test CrossResolutionFusion
    fusion = CrossResolutionFusion(
        daily_dim=daily_dim,
        monthly_dim=monthly_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=8,
        output_daily=True,
    )

    output = fusion(
        daily_repr=daily_repr,
        monthly_repr=monthly_repr,
        daily_mask=daily_mask,
        monthly_mask=monthly_mask,
        month_boundaries=month_boundaries,
        return_attention=True,
    )

    print(f"Fused monthly shape: {output.fused_monthly.shape}")
    print(f"Monthly mask shape: {output.monthly_mask.shape}")
    if output.fused_daily is not None:
        print(f"Fused daily shape: {output.fused_daily.shape}")
    print(f"Attention weights keys: {list(output.attention_weights.keys())}")

    # Test MultiResolutionPredictor
    predictor = MultiResolutionPredictor(
        daily_dim=daily_dim,
        monthly_dim=monthly_dim,
        hidden_dim=hidden_dim,
        task_type=TaskType.FORECASTING,
        forecast_horizon=6,
        output_distribution=True,
    )

    pred_output, fusion_output = predictor(
        daily_repr=daily_repr,
        monthly_repr=monthly_repr,
        daily_mask=daily_mask,
        monthly_mask=monthly_mask,
        month_boundaries=month_boundaries,
    )

    print(f"\nPredictions shape: {pred_output.predictions.shape}")
    if pred_output.confidence is not None:
        print(f"Confidence shape: {pred_output.confidence.shape}")

    print("\nAll tests passed!")


# =============================================================================
# MONTHLY ENCODER COMPONENT
# =============================================================================
#
# THE NO-FABRICATION PRINCIPLE
# ============================
#
# This section implements a critical architectural decision: we NEVER fabricate
# plausible values for missing observations. Instead, we use:
#
# 1. LEARNED NO_OBSERVATION_TOKEN: A trainable embedding that explicitly represents
#    "we have no data here". The model learns what "no data" means in context.
#
# 2. EXPLICIT OBSERVATION MASKS: Binary masks that distinguish between:
#    - "This sensor reported a value of 0" (real observation)
#    - "This sensor did not report anything" (no observation)
#
# 3. MASKED ATTENTION: Unobserved positions cannot contribute keys/values to
#    attention computation. They can only RECEIVE information (as queries),
#    never PROVIDE information.
#
# WHY THIS MATTERS
# ----------------
# Consider monthly satellite data with ~35 observations over 3 years:
# - Forward-filling would fabricate ~965 daily "observations" from thin air
# - Zeros would confuse "no data" with "measured zero intensity"
# - Our approach: 35 REAL embeddings + learned "I don't know" for gaps
#
# The model explicitly KNOWS which positions have real data vs placeholders.
# This is critical for:
# - Uncertainty quantification (can't be certain about what you didn't observe)
# - Causal reasoning (can't attribute effects to fabricated causes)
# - Model interpretability (attention shows what REAL data informed predictions)


class ObservationStatus(Enum):
    """Explicit observation status for each position."""
    UNOBSERVED = 0
    OBSERVED = 1


@dataclass
class MonthlySourceConfig:
    """Configuration for a monthly-resolution data source.

    Attributes:
        name: Identifier for the data source.
        n_features: Number of features in this source.
        description: Human-readable description.
        typical_observations: Expected number of observations over study period.
    """
    name: str
    n_features: int
    description: str
    typical_observations: int


# Registry of monthly data sources with their specifications
MONTHLY_SOURCE_CONFIGS: Dict[str, MonthlySourceConfig] = {
    'sentinel': MonthlySourceConfig(
        name='sentinel',
        n_features=43,
        description='Sentinel-1/2/3/5P satellite products',
        typical_observations=35
    ),
    'hdx_conflict': MonthlySourceConfig(
        name='hdx_conflict',
        n_features=18,
        description='HDX conflict event aggregations',
        typical_observations=40
    ),
    'hdx_food': MonthlySourceConfig(
        name='hdx_food',
        n_features=20,
        description='HDX food security indicators',
        typical_observations=40
    ),
    'hdx_rainfall': MonthlySourceConfig(
        name='hdx_rainfall',
        n_features=16,
        description='HDX dekadal rainfall data',
        typical_observations=45
    ),
    'iom': MonthlySourceConfig(
        name='iom',
        n_features=18,
        description='IOM displacement surveys',
        typical_observations=35
    ),
}


class MonthlyObservationEmbedding(nn.Module):
    """
    Embeds feature values with EXPLICIT observation status.

    This is the core component that implements the no-fabrication principle.

    For each feature at each timestep, produces an embedding that encodes:
    1. The feature VALUE (if observed) OR learned no_observation_token (if not)
    2. The SOURCE TYPE (which data source this feature comes from)
    3. The OBSERVATION STATUS (binary: observed vs not observed)
    4. TEMPORAL POSITION (which month in the sequence)

    The key insight is that unobserved positions get a LEARNED placeholder
    embedding, not zeros or forward-filled values. The model explicitly
    knows these are placeholders through the observation status embedding.

    Example:
        >>> embedding = MonthlyObservationEmbedding(
        ...     n_sources=5,
        ...     max_features_per_source=43,
        ...     d_model=64
        ... )
        >>> values = torch.randn(4, 35, 43)  # batch=4, months=35, features=43
        >>> obs_mask = (torch.rand(4, 35, 43) > 0.3).float()  # 70% observed
        >>> embeddings, attn_mask = embedding(values, obs_mask, ...)
    """

    def __init__(
        self,
        n_sources: int,
        max_features_per_source: int,
        d_model: int = 64,
        max_months: int = 60,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize observation embedding layer.

        Args:
            n_sources: Number of distinct data sources (e.g., 5 for monthly).
            max_features_per_source: Maximum features any single source has.
            d_model: Output embedding dimension.
            max_months: Maximum sequence length in months.
            dropout: Dropout rate for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.max_features_per_source = max_features_per_source
        self.n_sources = n_sources

        # ==================================================================
        # CRITICAL: Learned "no observation" token
        # ==================================================================
        # This is NOT zero, NOT forward-fill - it's a learned representation
        # of "we have no data here". The model learns what absence means.
        self.no_observation_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02  # Small init, learned during training
        )

        # Value projection for REAL observations
        # Maps scalar feature values to d_model dimension
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Source type embedding: which data source does this come from?
        self.source_embedding = nn.Embedding(n_sources, d_model)

        # Feature index embedding: which feature within the source?
        self.feature_embedding = nn.Embedding(max_features_per_source, d_model)

        # ==================================================================
        # CRITICAL: Observation status embedding
        # ==================================================================
        # Binary embedding: 0 = unobserved, 1 = observed
        # This allows the model to EXPLICITLY distinguish real data from placeholders
        self.observation_status_embedding = nn.Embedding(2, d_model)

        # Temporal position encoding (sinusoidal + learned)
        self.temporal_embedding = nn.Embedding(max_months, d_model)

        # Register sinusoidal positional encoding as buffer
        pe = self._create_sinusoidal_encoding(max_months, d_model)
        self.register_buffer('sinusoidal_pe', pe)

        # Combination layer: fuse all embeddings
        self.combination_layer = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self) -> None:
        """Initialize weights with appropriate scales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        values: Tensor,
        observation_mask: Tensor,
        source_ids: Tensor,
        feature_ids: Tensor,
        month_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Embed features with explicit observation status.

        Args:
            values: [batch, n_months, n_features] - Raw feature values.
                Unobserved positions can have any value (will be replaced).
            observation_mask: [batch, n_months, n_features] - Binary mask.
                1.0 = observed (real data), 0.0 = unobserved (no data).
            source_ids: [batch, n_months, n_features] - Source index per feature.
            feature_ids: [batch, n_months, n_features] - Feature index within source.
            month_indices: [batch, n_months] - Month position in sequence.

        Returns:
            embeddings: [batch, n_months, n_features, d_model] - Combined embeddings.
            attention_mask: [batch, n_months, n_features] - Mask for attention
                (1.0 = can be attended to, 0.0 = should be masked).
        """
        batch_size, n_months, n_features = values.shape
        device = values.device

        # ==================================================================
        # 1. VALUE EMBEDDING (observed) OR NO_OBSERVATION_TOKEN (unobserved)
        # ==================================================================
        values_expanded = values.unsqueeze(-1)  # [batch, n_months, n_features, 1]
        value_emb = self.value_projection(values_expanded)

        # Expand no_observation_token to match shape
        no_obs_emb = self.no_observation_token.expand(batch_size, n_months, n_features, -1)

        # ==================================================================
        # CRITICAL: Use mask to select between real values and placeholder
        # ==================================================================
        obs_mask_expanded = observation_mask.unsqueeze(-1)  # [batch, n_months, n_features, 1]

        value_or_placeholder = torch.where(
            obs_mask_expanded.bool(),
            value_emb,
            no_obs_emb,
        )

        # ==================================================================
        # 2. SOURCE TYPE EMBEDDING
        # ==================================================================
        source_emb = self.source_embedding(source_ids)  # [batch, n_months, n_features, d_model]

        # ==================================================================
        # 3. FEATURE INDEX EMBEDDING
        # ==================================================================
        feature_emb = self.feature_embedding(feature_ids)  # [batch, n_months, n_features, d_model]

        # ==================================================================
        # 4. OBSERVATION STATUS EMBEDDING (CRITICAL for no-fabrication)
        # ==================================================================
        obs_status = observation_mask.long()  # Convert to 0/1 integers
        obs_status_emb = self.observation_status_embedding(obs_status)

        # ==================================================================
        # 5. TEMPORAL POSITION EMBEDDING
        # ==================================================================
        temporal_learned = self.temporal_embedding(month_indices)  # [batch, n_months, d_model]
        temporal_sinusoidal = self.sinusoidal_pe[month_indices]  # [batch, n_months, d_model]
        temporal_emb = temporal_learned + temporal_sinusoidal
        temporal_emb = temporal_emb.unsqueeze(2).expand(-1, -1, n_features, -1)

        # ==================================================================
        # 6. COMBINE ALL EMBEDDINGS
        # ==================================================================
        combined = torch.cat([
            value_or_placeholder,
            source_emb,
            feature_emb,
            obs_status_emb,
            temporal_emb,
        ], dim=-1)

        embeddings = self.combination_layer(combined)
        attention_mask = observation_mask

        return embeddings, attention_mask


class ObservationMaskedTransformerLayer(nn.Module):
    """
    Transformer encoder layer where UNOBSERVED positions cannot be keys/values.

    This implements a critical constraint: information can only flow FROM
    real observations. Unobserved positions (with no_observation_token)
    can QUERY to gather information, but cannot PROVIDE information as
    keys or values.

    Why this matters:
    - Prevents the model from "learning" patterns in fabricated data
    - Attention weights only reflect influence of REAL observations
    - Uncertainty naturally increases in regions with sparse observations

    Architecture:
        1. Multi-Head Self-Attention (unobserved masked in K,V)
        2. Add & Norm
        3. Feed-Forward Network
        4. Add & Norm
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ) -> None:
        """
        Initialize transformer encoder layer with observation masking.

        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            dim_feedforward: Hidden dimension of FFN.
            dropout: Dropout rate.
            activation: Activation function ('gelu' or 'relu').
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        observation_mask: Tensor,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through transformer layer with observation masking.

        Args:
            x: [batch, seq_len, d_model] - Input embeddings.
            observation_mask: [batch, seq_len] - Binary mask (1=observed, 0=unobserved).
            return_attention: Whether to return attention weights.

        Returns:
            output: [batch, seq_len, d_model] - Transformed embeddings.
            attention_weights: [batch, nhead, seq_len, seq_len] (if return_attention).
        """
        # ==================================================================
        # CRITICAL: key_padding_mask
        # ==================================================================
        # PyTorch attention convention: True = IGNORE this position
        # We want to ignore UNOBSERVED positions (mask == 0)
        key_padding_mask = (observation_mask == 0)  # True where unobserved

        # Handle edge case: if ALL positions are masked, unmask first to prevent NaN
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Self-attention with masking
        attn_out, attn_weights = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        # Residual + LayerNorm
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        if return_attention:
            return x, attn_weights
        return x


class MonthlyEncoder(nn.Module):
    """
    Encodes monthly-resolution time series with explicit observation handling.

    This encoder processes monthly data sources (sentinel, hdx_conflict,
    hdx_food, hdx_rainfall, iom) at their NATIVE monthly resolution (~35
    timesteps), without upsampling to daily or fabricating missing observations.

    Key Design Principles:
    ----------------------
    1. NO FABRICATION: Missing observations use a learned no_observation_token,
       not zeros or forward-filled values.

    2. EXPLICIT MASKS: Observation masks distinguish "no data" from "value = 0".

    3. MASKED ATTENTION: Unobserved positions cannot contribute as keys/values,
       only as queries (to receive information, not provide it).

    4. UNCERTAINTY AWARENESS: Output includes uncertainty estimates that are
       higher in regions with sparse observations.

    Architecture:
    -------------
        Input Processing:
            values: [batch, n_months, n_features]
            observation_mask: [batch, n_months] - per-timestep observation status

        Feature Embedding:
            - Projects each feature to d_model dimension
            - Adds feature-specific embeddings
            - Uses no_observation_token for unobserved timesteps

        Feature Aggregation:
            - Uses cross-attention with learnable CLS token
            - Aggregates all features at each timestep

        Temporal Encoding:
            - Stack of ObservationMaskedTransformerLayers
            - Each layer applies masked self-attention (unobserved masked in K,V)

        Output:
            hidden: [batch, n_months, d_model] - encoded representations
            uncertainty: [batch, n_months, 1] - observation density-based uncertainty

    Example:
        >>> config = MONTHLY_SOURCE_CONFIGS['sentinel']
        >>> encoder = MonthlyEncoder(source_config=config, d_model=64)
        >>> outputs = encoder(
        ...     values=sentinel_data,           # [batch, 35, 43]
        ...     observation_mask=obs_mask,      # [batch, 35]
        ...     month_indices=torch.arange(35)  # Month positions
        ... )
        >>> hidden = outputs['hidden']          # [batch, 35, 64]
        >>> uncertainty = outputs['uncertainty'] # [batch, 35, 1]
    """

    def __init__(
        self,
        source_config: MonthlySourceConfig,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_months: int = 60,
    ) -> None:
        """
        Initialize MonthlyEncoder.

        Args:
            source_config: Configuration for the monthly data source.
            d_model: Model hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Hidden dimension of feed-forward networks.
            dropout: Dropout rate for regularization.
            max_months: Maximum sequence length in months.
        """
        super().__init__()

        self.source_config = source_config
        self.d_model = d_model
        self.n_features = source_config.n_features
        self.nhead = nhead

        # ==================================================================
        # FEATURE-LEVEL EMBEDDING
        # ==================================================================
        self.feature_projection = nn.Linear(1, d_model)

        # Learned no-observation token (shared across features of this source)
        self.no_observation_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        # Feature-specific embeddings
        self.feature_embedding = nn.Embedding(source_config.n_features, d_model)

        # Observation status embedding (0=unobserved, 1=observed)
        self.observation_status_embedding = nn.Embedding(2, d_model)

        # Temporal position embedding
        self.temporal_embedding = nn.Embedding(max_months, d_model)

        # Sinusoidal position encoding
        pe = self._create_sinusoidal_pe(max_months, d_model)
        self.register_buffer('sinusoidal_pe', pe)

        # ==================================================================
        # FEATURE FUSION (aggregate features at each timestep)
        # ==================================================================
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.feature_norm = nn.LayerNorm(d_model)

        # Learnable CLS token for feature aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ==================================================================
        # TEMPORAL TRANSFORMER ENCODER
        # ==================================================================
        self.encoder_layers = nn.ModuleList([
            ObservationMaskedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
            )
            for _ in range(num_layers)
        ])

        # ==================================================================
        # OUTPUT PROJECTION
        # ==================================================================
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

        self._init_weights()

    def _create_sinusoidal_pe(self, max_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_features(
        self,
        values: Tensor,
        observation_mask: Tensor,
        month_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Embed features with observation status.

        Args:
            values: [batch, n_months, n_features]
            observation_mask: [batch, n_months] - per-timestep mask
            month_indices: [batch, n_months]

        Returns:
            feature_embeddings: [batch, n_months, n_features, d_model]
            feature_mask: [batch, n_months, n_features] - expanded mask
        """
        batch_size, n_months, n_features = values.shape
        device = values.device

        # ==================================================================
        # STEP 0: Replace MISSING_VALUE (-999.0) with 0.0 BEFORE projection
        # ==================================================================
        # Expand observation_mask to feature level first for masking
        feature_mask = observation_mask.unsqueeze(-1).expand(-1, -1, n_features)

        # Clone values and replace missing values with 0.0
        values_clean = values.clone()
        values_clean = values_clean.masked_fill(~feature_mask.bool(), 0.0)

        # Verify no extreme values remain
        assert not (values_clean.abs() > 100).any(), "Extreme values detected in values after masking"

        # ==================================================================
        # VALUE EMBEDDING
        # ==================================================================
        values_expanded = values_clean.unsqueeze(-1)
        value_emb = self.feature_projection(values_expanded)

        # Verify projection output is reasonable
        assert not torch.isnan(value_emb).any(), "NaN in projected values"

        # ==================================================================
        # NO-OBSERVATION TOKEN (for unobserved timesteps)
        # ==================================================================
        # feature_mask already computed above in STEP 0

        # Expand no_observation_token
        no_obs = self.no_observation_token.expand(batch_size, n_months, n_features, -1)

        # Select value_emb where observed, no_obs where unobserved
        obs_mask_expanded = feature_mask.unsqueeze(-1)
        embedded_values = torch.where(
            obs_mask_expanded.bool(),
            value_emb,
            no_obs,
        )

        # ==================================================================
        # FEATURE-SPECIFIC EMBEDDING
        # ==================================================================
        feature_indices = torch.arange(n_features, device=device)
        feat_emb = self.feature_embedding(feature_indices)
        feat_emb = feat_emb.view(1, 1, n_features, self.d_model).expand(batch_size, n_months, -1, -1)

        # ==================================================================
        # OBSERVATION STATUS EMBEDDING
        # ==================================================================
        obs_status = feature_mask.long()
        obs_status_emb = self.observation_status_embedding(obs_status)

        # ==================================================================
        # TEMPORAL EMBEDDING
        # ==================================================================
        temporal_learned = self.temporal_embedding(month_indices)
        temporal_sin = self.sinusoidal_pe[month_indices]
        temporal_emb = (temporal_learned + temporal_sin).unsqueeze(2).expand(-1, -1, n_features, -1)

        # ==================================================================
        # COMBINE EMBEDDINGS
        # ==================================================================
        feature_embeddings = embedded_values + feat_emb + obs_status_emb + temporal_emb

        return feature_embeddings, feature_mask

    def _aggregate_features(
        self,
        feature_embeddings: Tensor,
        feature_mask: Tensor,
    ) -> Tensor:
        """
        Aggregate feature embeddings at each timestep using attention.

        Args:
            feature_embeddings: [batch, n_months, n_features, d_model]
            feature_mask: [batch, n_months, n_features]

        Returns:
            timestep_embeddings: [batch, n_months, d_model]
        """
        batch_size, n_months, n_features, d_model = feature_embeddings.shape

        # Reshape for batch processing: [batch * n_months, n_features, d_model]
        flat_features = feature_embeddings.view(batch_size * n_months, n_features, d_model)
        flat_mask = feature_mask.view(batch_size * n_months, n_features)

        # Expand CLS token: [batch * n_months, 1, d_model]
        cls = self.cls_token.expand(batch_size * n_months, -1, -1)

        # key_padding_mask: True = ignore (unobserved features)
        key_padding_mask = (flat_mask == 0)

        # Handle fully masked timesteps
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Cross-attention: CLS queries features
        aggregated, _ = self.feature_attention(
            query=cls,
            key=flat_features,
            value=flat_features,
            key_padding_mask=key_padding_mask,
        )

        # Reshape back: [batch, n_months, d_model]
        timestep_embeddings = aggregated.view(batch_size, n_months, d_model)
        timestep_embeddings = self.feature_norm(timestep_embeddings)

        return timestep_embeddings

    def _compute_observation_density(
        self,
        observation_mask: Tensor,
        window_size: int = 3,
    ) -> Tensor:
        """
        Compute local observation density for uncertainty scaling.

        Args:
            observation_mask: [batch, n_months]
            window_size: Size of local window

        Returns:
            density: [batch, n_months] - Local observation density (0 to 1)
        """
        # Pad for convolution
        padded = F.pad(
            observation_mask.unsqueeze(1).float(),
            (window_size // 2, window_size // 2),
            mode='replicate',
        )

        # Average pooling gives local density
        kernel = torch.ones(1, 1, window_size, device=observation_mask.device) / window_size
        density = F.conv1d(padded, kernel).squeeze(1)

        return density

    def forward(
        self,
        values: Tensor,
        observation_mask: Tensor,
        month_indices: Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Encode monthly data with observation-aware attention.

        Args:
            values: [batch, n_months, n_features] - Feature values.
                Unobserved positions can have any value (will be replaced).
            observation_mask: [batch, n_months] - Per-timestep observation mask.
                1.0 = this month has real observation.
                0.0 = this month has no observation (missing data).
            month_indices: [batch, n_months] - Position indices (0 to n_months-1).
            return_attention: Whether to return attention weights.

        Returns:
            Dictionary containing:
                'hidden': [batch, n_months, d_model] - Encoded representations.
                'uncertainty': [batch, n_months, 1] - Estimated uncertainty.
                'attention_weights': Optional list of attention weights per layer.

        Note:
            The model explicitly handles missing observations:
            - Observed positions: real value embedding + OBSERVED status
            - Unobserved positions: no_observation_token + UNOBSERVED status
            - Attention: unobserved cannot contribute as keys/values
        """
        batch_size, n_months, n_features = values.shape

        # ==================================================================
        # STEP 1: Embed features with observation status
        # ==================================================================
        feature_embeddings, feature_mask = self._embed_features(
            values, observation_mask, month_indices
        )

        # ==================================================================
        # STEP 2: Aggregate features at each timestep
        # ==================================================================
        timestep_embeddings = self._aggregate_features(feature_embeddings, feature_mask)

        # ==================================================================
        # STEP 3: Temporal encoding with masked attention
        # ==================================================================
        hidden = timestep_embeddings
        attention_weights = []

        for layer in self.encoder_layers:
            if return_attention:
                hidden, weights = layer(hidden, observation_mask, return_attention=True)
                attention_weights.append(weights)
            else:
                hidden = layer(hidden, observation_mask)

        # ==================================================================
        # STEP 4: Output projection
        # ==================================================================
        hidden = self.output_projection(hidden)

        # ==================================================================
        # STEP 5: Uncertainty estimation
        # ==================================================================
        uncertainty = self.uncertainty_head(hidden)

        # Scale uncertainty by observation sparsity
        obs_density = self._compute_observation_density(observation_mask)
        uncertainty = uncertainty * (1.0 + (1.0 - obs_density).unsqueeze(-1))

        outputs = {
            'hidden': hidden,
            'uncertainty': uncertainty,
        }

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs


class MultiSourceMonthlyEncoder(nn.Module):
    """
    Encodes multiple monthly data sources with cross-source attention.

    This encoder handles all five monthly sources:
    - sentinel: Satellite imagery (43 features)
    - hdx_conflict: Conflict events (18 features)
    - hdx_food: Food security (20 features)
    - hdx_rainfall: Precipitation (16 features)
    - iom: Displacement data (18 features)

    Each source has its own MonthlyEncoder, then cross-source attention
    fuses information across sources while respecting observation patterns.

    Key Features:
    - Per-source encoding with observation masking
    - Cross-source attention to learn source dependencies
    - Unified output combining all sources
    - Per-source importance weights for interpretability

    Example:
        >>> encoder = MultiSourceMonthlyEncoder(
        ...     source_configs=MONTHLY_SOURCE_CONFIGS,
        ...     d_model=64
        ... )
        >>> outputs = encoder(
        ...     source_features={'sentinel': ..., 'hdx_conflict': ...},
        ...     source_masks={'sentinel': ..., 'hdx_conflict': ...},
        ...     month_indices=torch.arange(35)
        ... )
        >>> hidden = outputs['hidden']  # [batch, 35, 64]
        >>> importance = outputs['source_importance']  # [batch, 35, 5]
    """

    def __init__(
        self,
        source_configs: Dict[str, MonthlySourceConfig],
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        max_months: int = 60,
    ) -> None:
        """
        Initialize multi-source encoder.

        Args:
            source_configs: Dictionary of source configurations.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Layers per source encoder.
            num_fusion_layers: Layers for cross-source fusion.
            dropout: Dropout rate.
            max_months: Maximum sequence length.
        """
        super().__init__()

        self.source_configs = source_configs
        self.source_names = list(source_configs.keys())
        self.n_sources = len(source_configs)
        self.d_model = d_model

        # ==================================================================
        # PER-SOURCE ENCODERS
        # ==================================================================
        self.source_encoders = nn.ModuleDict({
            name: MonthlyEncoder(
                source_config=config,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dropout=dropout,
                max_months=max_months,
            )
            for name, config in source_configs.items()
        })

        # ==================================================================
        # SOURCE TYPE EMBEDDING
        # ==================================================================
        self.source_type_embedding = nn.Embedding(self.n_sources, d_model)

        # ==================================================================
        # CROSS-SOURCE FUSION
        # ==================================================================
        self.cross_source_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_fusion_layers)
        ])
        self.cross_source_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_fusion_layers)
        ])

        # ==================================================================
        # OUTPUT PROJECTION
        # ==================================================================
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * self.n_sources, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Source importance head
        self.source_importance_head = nn.Sequential(
            nn.Linear(d_model * self.n_sources, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.n_sources),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        source_features: Dict[str, Tensor],
        source_masks: Dict[str, Tensor],
        month_indices: Tensor,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Encode multiple monthly sources.

        Args:
            source_features: {source_name: [batch, n_months, n_features]}
            source_masks: {source_name: [batch, n_months]} - observation masks
            month_indices: [batch, n_months] - temporal positions
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                'hidden': [batch, n_months, d_model] - Fused representation.
                'source_hidden': {source: [batch, n_months, d_model]} - Per-source.
                'source_importance': [batch, n_months, n_sources] - Source weights.
                'uncertainty': [batch, n_months, 1] - Combined uncertainty.
        """
        batch_size, n_months = month_indices.shape
        device = month_indices.device

        # ==================================================================
        # ENCODE EACH SOURCE
        # ==================================================================
        source_hidden = {}
        source_uncertainties = []

        for i, source_name in enumerate(self.source_names):
            if source_name in source_features:
                outputs = self.source_encoders[source_name](
                    values=source_features[source_name],
                    observation_mask=source_masks[source_name],
                    month_indices=month_indices,
                    return_attention=return_attention,
                )

                # Add source type embedding
                source_type_emb = self.source_type_embedding(
                    torch.tensor([i], device=device)
                ).expand(batch_size, n_months, -1)

                source_hidden[source_name] = outputs['hidden'] + source_type_emb
                source_uncertainties.append(outputs['uncertainty'])
            else:
                # Missing source: use zeros with high uncertainty
                source_hidden[source_name] = torch.zeros(
                    batch_size, n_months, self.d_model, device=device
                )
                source_uncertainties.append(
                    torch.ones(batch_size, n_months, 1, device=device) * 10.0
                )

        # ==================================================================
        # CROSS-SOURCE ATTENTION
        # ==================================================================
        stacked = torch.stack([source_hidden[s] for s in self.source_names], dim=2)
        flat_stacked = stacked.view(batch_size * n_months, self.n_sources, self.d_model)

        for attn, norm in zip(self.cross_source_attention, self.cross_source_norms):
            attended, _ = attn(flat_stacked, flat_stacked, flat_stacked)
            flat_stacked = norm(flat_stacked + attended)

        fused_sources = flat_stacked.view(batch_size, n_months, self.n_sources, self.d_model)

        # ==================================================================
        # COMBINE SOURCES
        # ==================================================================
        concat_sources = fused_sources.view(batch_size, n_months, -1)

        source_importance = self.source_importance_head(concat_sources)
        hidden = self.output_projection(concat_sources)

        stacked_uncertainties = torch.cat(source_uncertainties, dim=-1)
        uncertainty = (stacked_uncertainties * source_importance).sum(dim=-1, keepdim=True)

        return {
            'hidden': hidden,
            'source_hidden': source_hidden,
            'source_importance': source_importance,
            'uncertainty': uncertainty,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_monthly_encoder(
    source_name: str,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> MonthlyEncoder:
    """
    Factory function to create a MonthlyEncoder for a specific source.

    Args:
        source_name: One of 'sentinel', 'hdx_conflict', 'hdx_food',
            'hdx_rainfall', 'iom'.
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout rate.

    Returns:
        Configured MonthlyEncoder instance.

    Raises:
        ValueError: If source_name is not recognized.

    Example:
        >>> encoder = create_monthly_encoder('sentinel', d_model=64)
        >>> outputs = encoder(values, mask, month_indices)
    """
    if source_name not in MONTHLY_SOURCE_CONFIGS:
        raise ValueError(
            f"Unknown source: {source_name}. "
            f"Valid sources: {list(MONTHLY_SOURCE_CONFIGS.keys())}"
        )

    return MonthlyEncoder(
        source_config=MONTHLY_SOURCE_CONFIGS[source_name],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )


def create_multi_source_encoder(
    sources: Optional[List[str]] = None,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_fusion_layers: int = 2,
    dropout: float = 0.1,
) -> MultiSourceMonthlyEncoder:
    """
    Factory function to create a MultiSourceMonthlyEncoder.

    Args:
        sources: List of source names to include (default: all).
        d_model: Model dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Layers per source encoder.
        num_fusion_layers: Layers for cross-source fusion.
        dropout: Dropout rate.

    Returns:
        Configured MultiSourceMonthlyEncoder instance.

    Example:
        >>> encoder = create_multi_source_encoder(
        ...     sources=['sentinel', 'hdx_conflict'],
        ...     d_model=64
        ... )
    """
    if sources is None:
        sources = list(MONTHLY_SOURCE_CONFIGS.keys())

    configs = {s: MONTHLY_SOURCE_CONFIGS[s] for s in sources}

    return MultiSourceMonthlyEncoder(
        source_configs=configs,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_fusion_layers=num_fusion_layers,
        dropout=dropout,
    )


# =============================================================================
# DAILY SOURCE CONFIGURATIONS
# =============================================================================

@dataclass
class DailySourceConfig:
    """Configuration for a daily-resolution data source."""
    name: str
    n_features: int
    description: str = ""


# Default daily source configurations matching the Ukraine conflict dataset
DAILY_SOURCE_CONFIGS = {
    'equipment': DailySourceConfig('equipment', 38, 'Equipment loss counts by type'),
    'personnel': DailySourceConfig('personnel', 6, 'Personnel casualty figures'),
    'deepstate': DailySourceConfig('deepstate', 55, 'Front line territorial data'),
    'firms': DailySourceConfig('firms', 42, 'VIIRS fire detection satellite data'),
    'viina': DailySourceConfig('viina', 24, 'VIINA territorial control'),
}


# =============================================================================
# OUTPUT CONTAINERS FOR DAILY ENCODER
# =============================================================================

class DailyEncoderOutput(NamedTuple):
    """Output container for DailyEncoder.

    Attributes:
        daily_hidden: Full daily representations at original resolution.
            Shape: (batch, num_days, hidden_dim)
        monthly_hidden: Aggregated monthly representations.
            Shape: (batch, num_months, hidden_dim)
        aggregation_attention: Attention weights showing which days contributed
            to each month's representation. Shape: (batch, num_heads, num_months, num_days)
            This is crucial for interpretability - it shows which daily observations
            the model considered most important for each monthly representation.
    """
    daily_hidden: Tensor
    monthly_hidden: Tensor
    aggregation_attention: Optional[Tensor]


# =============================================================================
# POSITIONAL ENCODING FOR DAILY SEQUENCES
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Implements the standard transformer positional encoding from "Attention Is All
    You Need" (Vaswani et al., 2017). This encoding provides:

    1. Unique position information for each timestep
    2. Smooth interpolation between positions (important for time series)
    3. Bounded magnitude regardless of sequence length

    For time series, sinusoidal encoding is often preferred over learned encoding
    because it can generalize to sequence lengths not seen during training.

    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to support
        dropout: Dropout probability applied after adding positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term for sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Register as buffer (not a parameter, but should be saved/loaded)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            position_ids: Optional position indices. If None, uses sequential 0, 1, 2, ...

        Returns:
            Tensor of same shape with positional encoding added
        """
        if position_ids is not None:
            # Gather positional encodings for specific positions
            batch_size, seq_len = position_ids.shape
            pe_expanded = self.pe.expand(batch_size, -1, -1)
            pos_encoding = torch.gather(
                pe_expanded,
                1,
                position_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            x = x + pos_encoding
        else:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for temporal sequences.

    Unlike sinusoidal encoding, this learns position embeddings from data.
    May capture domain-specific temporal patterns but cannot generalize to
    unseen sequence lengths.

    Includes optional day-of-month encoding to capture monthly cyclical patterns
    relevant to conflict data (e.g., resupply cycles, operational patterns).

    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to support
        include_day_of_month: Whether to add day-of-month embeddings
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2000,
        include_day_of_month: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.include_day_of_month = include_day_of_month

        # Absolute position embedding
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Day-of-month embedding (captures monthly cyclical patterns)
        if include_day_of_month:
            self.day_of_month_embedding = nn.Embedding(31, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add learned positional encoding to input tensor.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            position_ids: Optional position indices [batch, seq_len].
                         If None, uses sequential positions 0, 1, 2, ...

        Returns:
            Tensor of same shape with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Create position indices if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        # Add absolute position embedding
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb

        # Add day-of-month embedding if enabled
        if self.include_day_of_month:
            day_of_month = position_ids % 31  # Approximate day within month
            dom_emb = self.day_of_month_embedding(day_of_month)
            x = x + dom_emb

        x = self.layer_norm(x)
        return self.dropout(x)


# =============================================================================
# LEARNABLE MONTHLY AGGREGATOR
# =============================================================================

class LearnableMonthlyAggregator(nn.Module):
    """
    Learnable aggregation mechanism that compresses daily representations to
    monthly boundaries using cross-attention with learned month queries.

    CRITICAL DESIGN DECISION: This is NOT a simple mean/max pooling operation.
    Instead, it learns which daily observations are most important for each month
    through cross-attention. This allows the model to:

    1. Weight significant events (major battles, supply disruptions) higher
    2. Downweight noise and routine observations
    3. Capture different aggregation patterns for different aspects of the data

    Architecture:
    -------------
    The aggregator uses cross-attention where:
    - Query: Learned month embeddings (one per month position)
    - Key/Value: Daily encoder hidden states

    Each month query attends to all daily observations and produces a weighted
    combination. The attention weights are interpretable - they show which days
    the model considers most relevant for each monthly representation.

    Month Boundary Awareness:
    -------------------------
    Two modes of operation:

    1. GLOBAL: Month queries can attend to ANY daily observation
       - Captures long-range dependencies
       - May learn seasonal patterns

    2. CONSTRAINED: Month queries only attend to days within that month
       - Enforces temporal locality
       - Prevents information leakage across months

    The constrained mode is implemented via attention masking.

    Args:
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        max_months: Maximum number of months to support
        dropout: Dropout probability
        use_month_constraints: If True, each month only attends to its own days
        pre_norm: Whether to apply LayerNorm before attention (Pre-LN transformer)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        max_months: int = 60,
        dropout: float = 0.1,
        use_month_constraints: bool = False,
        pre_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_months = max_months
        self.use_month_constraints = use_month_constraints
        self.pre_norm = pre_norm

        # Learnable month query embeddings
        # These are the "questions" each month asks of the daily observations
        self.month_queries = nn.Parameter(torch.randn(1, max_months, d_model))
        nn.init.normal_(self.month_queries, mean=0, std=d_model ** -0.5)

        # Month position embedding (captures month-of-year seasonality)
        self.month_position_embedding = nn.Embedding(max_months, d_model)

        # Cross-attention: months attend to days
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network for post-attention processing
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)

    def _create_month_attention_mask(
        self,
        n_months: int,
        n_days: int,
        month_boundaries: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create attention mask that constrains each month to attend only to its days.

        Args:
            n_months: Number of months
            n_days: Number of days in sequence
            month_boundaries: [batch, n_months, 2] tensor with (start_day, end_day)
            device: Device to create mask on

        Returns:
            Attention mask of shape [batch, n_months, n_days]
            True = MASK (don't attend), False = allow attention
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
        daily_hidden: torch.Tensor,
        n_months: int,
        daily_mask: Optional[torch.Tensor] = None,
        month_boundaries: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Aggregate daily representations to monthly using learned cross-attention.

        Args:
            daily_hidden: Daily encoder output [batch, n_days, d_model]
            n_months: Number of months to produce
            daily_mask: Optional mask [batch, n_days] where True = valid observation
                       (inverted to attention mask internally)
            month_boundaries: Optional [batch, n_months, 2] with (start_day, end_day)
                            per month. Required if use_month_constraints=True.
            return_attention_weights: If True, return attention weights for analysis

        Returns:
            monthly_hidden: Aggregated monthly representations [batch, n_months, d_model]
            attention_weights: (optional) [batch, n_heads, n_months, n_days]
        """
        batch_size, n_days, _ = daily_hidden.shape
        device = daily_hidden.device

        # Get month queries for the requested number of months
        queries = self.month_queries[:, :n_months, :].expand(batch_size, -1, -1)

        # Add month position embeddings
        month_positions = torch.arange(n_months, device=device).unsqueeze(0)
        month_pos_emb = self.month_position_embedding(month_positions)
        queries = queries + month_pos_emb

        # Prepare attention mask
        key_padding_mask = None
        if daily_mask is not None:
            key_padding_mask = ~daily_mask  # Invert: True = mask out

        attn_mask = None
        if self.use_month_constraints and month_boundaries is not None:
            attn_mask = self._create_month_attention_mask(
                n_months, n_days, month_boundaries, device
            )

        # Pre-norm if configured
        if self.pre_norm:
            queries_normed = self.norm1(queries)
            daily_normed = self.norm1(daily_hidden)
        else:
            queries_normed = queries
            daily_normed = daily_hidden

        # Cross-attention: months query days
        attended, attention_weights = self.cross_attention(
            query=queries_normed,
            key=daily_normed,
            value=daily_hidden,  # Use un-normed values
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=return_attention_weights,
            average_attn_weights=False  # Keep per-head weights
        )

        # Residual connection
        if not self.pre_norm:
            attended = self.norm1(queries + attended)
        else:
            attended = queries + attended

        # Feed-forward with residual
        if self.pre_norm:
            ffn_out = self.ffn(self.norm2(attended))
            monthly_hidden = attended + ffn_out
        else:
            ffn_out = self.ffn(attended)
            monthly_hidden = self.norm2(attended + ffn_out)

        # Final projection
        monthly_hidden = self.output_projection(monthly_hidden)

        if return_attention_weights:
            return monthly_hidden, attention_weights
        return monthly_hidden


# =============================================================================
# DAILY ENCODER
# =============================================================================

class DailyEncoder(nn.Module):
    """
    Transformer encoder for daily-resolution time series data.

    This encoder processes daily observations at their FULL native resolution
    (typically ~1000+ timesteps for the Ukraine conflict dataset). It produces:

    1. Full daily representations - for daily-level predictions
    2. Aggregated monthly representations - for fusion with monthly sources

    The monthly aggregation uses a LearnableMonthlyAggregator that learns which
    daily observations are most important for each month, rather than using
    simple mean/max pooling that would lose information.

    NO DATA FABRICATION PRINCIPLE:
    ------------------------------
    This encoder adheres strictly to the no-fabrication principle:

    - Missing daily observations are handled via a learned [NO_OBS] token, not
      by interpolating or forward-filling values
    - The attention mechanism naturally learns to ignore or downweight missing
      observations based on the mask
    - Monthly aggregation is LEARNED, not a fixed function like mean/max that
      would arbitrarily compress the signal

    The attention weights from the LearnableMonthlyAggregator are fully
    interpretable - they show exactly which daily observations the model
    considered important for each monthly representation.

    Architecture Details:
    ---------------------

    Input Processing:
    - Feature projection: Linear layer maps raw features to d_model dimension
    - Positional encoding: Sinusoidal or learned, captures temporal structure
    - Missing observation handling: Learned [NO_OBS] token for masked positions

    Transformer Encoder:
    - Standard transformer encoder with configurable layers and heads
    - Uses GELU activation (smoother gradients than ReLU)
    - Pre-LayerNorm configuration for training stability

    Monthly Aggregation:
    - LearnableMonthlyAggregator compresses daily states to monthly boundaries
    - Cross-attention allows model to learn which days matter for each month
    - Attention weights are interpretable for analysis

    Memory Considerations:
    ----------------------
    Processing 1000+ daily timesteps with full self-attention requires O(n^2)
    memory. For very long sequences, consider:

    1. Gradient checkpointing (enabled via PyTorch's checkpoint utilities)
    2. Chunked processing with overlapping windows
    3. Linear attention variants (not implemented here)

    For typical conflict datasets (~1000 days, batch_size=16, d_model=128),
    memory usage is approximately 2-4GB on GPU.

    Args:
        n_features: Total number of input features (sum across all daily sources)
        d_model: Dimension of model embeddings (default: 128)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer encoder layers (default: 4)
        dropout: Dropout probability (default: 0.1)
        max_days: Maximum number of days to support (default: 1500)
        max_months: Maximum number of months for aggregation (default: 60)
        positional_encoding: Type of positional encoding ('sinusoidal' or 'learned')
        use_month_constraints: Whether to constrain month attention to its days

    Example:
        >>> encoder = DailyEncoder(
        ...     n_features=165,  # equipment + personnel + deepstate + firms + viina
        ...     d_model=128,
        ...     n_heads=8,
        ...     n_layers=4
        ... )
        >>>
        >>> # Daily features [batch, n_days, n_features]
        >>> x = torch.randn(16, 1000, 165)
        >>> mask = torch.ones(16, 1000, dtype=torch.bool)
        >>>
        >>> # Forward pass
        >>> output = encoder(x, observation_mask=mask, n_months=33)
        >>>
        >>> print(output.daily_hidden.shape)   # [16, 1000, 128]
        >>> print(output.monthly_hidden.shape) # [16, 33, 128]
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_days: int = 1500,
        max_months: int = 60,
        positional_encoding: str = 'sinusoidal',
        use_month_constraints: bool = False
    ):
        super().__init__()

        # Store configuration
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_days = max_days
        self.max_months = max_months
        self.use_month_constraints = use_month_constraints

        # Validate configuration
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # ===================
        # INPUT PROCESSING
        # ===================

        # Project raw features to model dimension
        self.feature_projection = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        if positional_encoding == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_days,
                dropout=dropout
            )
        elif positional_encoding == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(
                d_model=d_model,
                max_len=max_days,
                include_day_of_month=True,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        # Learned embedding for missing/masked observations
        # When a day has no observation, we use this instead of zeros
        # This is CRITICAL for the no-fabrication principle
        self.no_observation_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.no_observation_token, mean=0, std=d_model ** -0.5)

        # ===================
        # TRANSFORMER ENCODER
        # ===================

        # Single encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for training stability
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False  # Disable for compatibility
        )

        # Post-encoder layer norm
        self.encoder_output_norm = nn.LayerNorm(d_model)

        # ===================
        # MONTHLY AGGREGATION
        # ===================

        self.monthly_aggregator = LearnableMonthlyAggregator(
            d_model=d_model,
            n_heads=n_heads,
            max_months=max_months,
            dropout=dropout,
            use_month_constraints=use_month_constraints,
            pre_norm=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(
        self,
        x: torch.Tensor,
        observation_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        n_months: Optional[int] = None,
        month_boundaries: Optional[torch.Tensor] = None,
        return_attention_weights: bool = True
    ) -> DailyEncoderOutput:
        """
        Encode daily observations and aggregate to monthly representations.

        Args:
            x: Daily features [batch, n_days, n_features]
            observation_mask: Optional [batch, n_days] boolean mask where
                            True = valid observation, False = missing.
                            Missing days use learned no_observation_token.
            position_ids: Optional [batch, n_days] position indices.
                         If None, uses sequential 0, 1, 2, ...
            n_months: Number of months to aggregate to. If None, inferred from
                     sequence length assuming 30 days/month.
            month_boundaries: Optional [batch, n_months, 2] with (start_day, end_day)
                            per month. Required if use_month_constraints=True.
            return_attention_weights: If True, return monthly aggregation attention.
                                     Default True for interpretability.

        Returns:
            DailyEncoderOutput containing:
                - daily_hidden: Full daily representations [batch, n_days, d_model]
                - monthly_hidden: Aggregated monthly representations [batch, n_months, d_model]
                - aggregation_attention: (if return_attention_weights=True)
                    Attention weights [batch, n_heads, n_months, n_days]
        """
        batch_size, n_days, _ = x.shape
        device = x.device

        # Infer number of months if not provided
        if n_months is None:
            n_months = max(1, n_days // 30)

        # ===================
        # INPUT PROCESSING
        # ===================

        # Project features to model dimension
        hidden = self.feature_projection(x)  # [batch, n_days, d_model]

        # Handle missing observations using learned [NO_OBS] token
        if observation_mask is not None:
            # Expand mask for broadcasting: [batch, n_days, 1]
            mask_expanded = observation_mask.unsqueeze(-1).float()

            # Replace missing observations with learned token
            no_obs_expanded = self.no_observation_token.expand(batch_size, n_days, -1)
            hidden = hidden * mask_expanded + no_obs_expanded * (1 - mask_expanded)

        # Add positional encoding
        hidden = self.positional_encoding(hidden, position_ids)

        # ===================
        # TRANSFORMER ENCODING
        # ===================

        # Create attention mask for transformer
        # For TransformerEncoder: True = IGNORE this position
        src_key_padding_mask = None
        if observation_mask is not None:
            src_key_padding_mask = ~observation_mask

        # Apply transformer encoder
        daily_hidden = self.transformer_encoder(
            hidden,
            src_key_padding_mask=src_key_padding_mask
        )

        # Apply output normalization
        daily_hidden = self.encoder_output_norm(daily_hidden)

        # ===================
        # MONTHLY AGGREGATION
        # ===================

        aggregator_output = self.monthly_aggregator(
            daily_hidden=daily_hidden,
            n_months=n_months,
            daily_mask=observation_mask,
            month_boundaries=month_boundaries,
            return_attention_weights=return_attention_weights
        )

        if return_attention_weights:
            monthly_hidden, attention_weights = aggregator_output
            return DailyEncoderOutput(
                daily_hidden=daily_hidden,
                monthly_hidden=monthly_hidden,
                aggregation_attention=attention_weights
            )
        else:
            monthly_hidden = aggregator_output
            return DailyEncoderOutput(
                daily_hidden=daily_hidden,
                monthly_hidden=monthly_hidden,
                aggregation_attention=None
            )

    def get_daily_representations(
        self,
        x: torch.Tensor,
        observation_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get only daily representations (skip monthly aggregation).

        Useful when you only need daily-level predictions and want to
        save computation.

        Args:
            x: Daily features [batch, n_days, n_features]
            observation_mask: Optional [batch, n_days] boolean mask
            position_ids: Optional [batch, n_days] position indices

        Returns:
            daily_hidden: [batch, n_days, d_model]
        """
        batch_size, n_days, _ = x.shape

        # Project features
        hidden = self.feature_projection(x)

        # Handle missing observations
        if observation_mask is not None:
            mask_expanded = observation_mask.unsqueeze(-1).float()
            no_obs_expanded = self.no_observation_token.expand(batch_size, n_days, -1)
            hidden = hidden * mask_expanded + no_obs_expanded * (1 - mask_expanded)

        # Add positional encoding
        hidden = self.positional_encoding(hidden, position_ids)

        # Transformer encoding
        src_key_padding_mask = None
        if observation_mask is not None:
            src_key_padding_mask = ~observation_mask

        daily_hidden = self.transformer_encoder(
            hidden,
            src_key_padding_mask=src_key_padding_mask
        )

        return self.encoder_output_norm(daily_hidden)


# =============================================================================
# MULTI-SOURCE DAILY ENCODER
# =============================================================================

class MultiSourceDailyEncoder(nn.Module):
    """
    Encoder for multiple daily-resolution data sources with source-specific
    processing and fusion.

    This encoder handles the heterogeneous nature of daily OSINT sources:
    - equipment: Cumulative loss counts (non-negative, monotonic)
    - personnel: Casualty figures (non-negative, monotonic)
    - deepstate: Territorial control metrics (bounded, categorical)
    - firms: Fire detections (count data, seasonal patterns)
    - viina: Control assessments (categorical/ordinal)

    Architecture:
    -------------

    1. Source-Specific Encoders: Each source has its own encoder head that
       learns source-appropriate representations.

    2. Source Fusion: Encoded sources are fused via attention or concatenation.

    3. Joint Encoding: Fused representation is processed by shared transformer.

    4. Monthly Aggregation: Joint representation aggregated to monthly.

    This design allows the model to learn source-specific patterns while
    capturing cross-source interactions.

    Args:
        source_configs: Dict mapping source name to DailySourceConfig
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        dropout: Dropout probability
        fusion_method: How to fuse sources ('attention', 'concatenate', 'sum')
    """

    def __init__(
        self,
        source_configs: Dict[str, DailySourceConfig],
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        fusion_method: str = 'attention',
        max_days: int = 1500,
        max_months: int = 60
    ):
        super().__init__()

        self.source_configs = source_configs
        self.source_names = list(source_configs.keys())
        self.n_sources = len(source_configs)
        self.d_model = d_model
        self.fusion_method = fusion_method

        # ===================
        # SOURCE-SPECIFIC ENCODERS
        # ===================

        self.source_projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(cfg.n_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for name, cfg in source_configs.items()
        })

        # Learnable source type embeddings
        self.source_embeddings = nn.Embedding(self.n_sources, d_model)

        # ===================
        # SOURCE FUSION
        # ===================

        if fusion_method == 'attention':
            # Cross-source attention
            self.source_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.source_fusion_norm = nn.LayerNorm(d_model)

        elif fusion_method == 'concatenate':
            # Concatenate then project
            self.source_fusion = nn.Sequential(
                nn.Linear(d_model * self.n_sources, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        elif fusion_method == 'sum':
            # Learned weighted sum
            self.source_weights = nn.Parameter(torch.ones(self.n_sources))
            self.source_fusion_norm = nn.LayerNorm(d_model)

        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        # ===================
        # SHARED ENCODER
        # ===================

        # Positional encoding (shared across sources)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_days,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        self.encoder_output_norm = nn.LayerNorm(d_model)

        # ===================
        # MONTHLY AGGREGATION
        # ===================

        self.monthly_aggregator = LearnableMonthlyAggregator(
            d_model=d_model,
            n_heads=n_heads,
            max_months=max_months,
            dropout=dropout,
            use_month_constraints=False,
            pre_norm=True
        )

    def _fuse_sources(
        self,
        source_hidden: Dict[str, torch.Tensor],
        source_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fuse multiple source representations into single representation.

        Args:
            source_hidden: Dict mapping source name to [batch, n_days, d_model]
            source_masks: Optional dict mapping source name to [batch, n_days] mask

        Returns:
            fused: [batch, n_days, d_model]
            combined_mask: [batch, n_days] (True where ANY source has observation)
        """
        batch_size = next(iter(source_hidden.values())).shape[0]
        n_days = next(iter(source_hidden.values())).shape[1]
        device = next(iter(source_hidden.values())).device

        # Stack source representations
        source_list = []
        for i, name in enumerate(self.source_names):
            hidden = source_hidden[name]

            # Add source type embedding
            source_emb = self.source_embeddings(
                torch.tensor([i], device=device)
            ).unsqueeze(0).unsqueeze(0)
            hidden = hidden + source_emb.expand(batch_size, n_days, 1, -1).squeeze(2)

            source_list.append(hidden)

        stacked = torch.stack(source_list, dim=2)  # [batch, n_days, n_sources, d_model]

        # Combine masks
        combined_mask = None
        if source_masks is not None:
            mask_list = [source_masks.get(name, torch.ones(batch_size, n_days, device=device, dtype=torch.bool))
                        for name in self.source_names]
            combined_mask = torch.stack(mask_list, dim=2).any(dim=2)

        # Fuse based on method
        if self.fusion_method == 'attention':
            stacked_flat = stacked.view(batch_size * n_days, self.n_sources, self.d_model)
            attended, _ = self.source_attention(stacked_flat, stacked_flat, stacked_flat)
            fused = attended.mean(dim=1).view(batch_size, n_days, self.d_model)
            fused = self.source_fusion_norm(fused)

        elif self.fusion_method == 'concatenate':
            concat = stacked.view(batch_size, n_days, -1)
            fused = self.source_fusion(concat)

        elif self.fusion_method == 'sum':
            weights = F.softmax(self.source_weights, dim=0)
            weights = weights.view(1, 1, self.n_sources, 1)
            fused = (stacked * weights).sum(dim=2)
            fused = self.source_fusion_norm(fused)

        return fused, combined_mask

    def forward(
        self,
        source_features: Dict[str, torch.Tensor],
        source_masks: Optional[Dict[str, torch.Tensor]] = None,
        n_months: Optional[int] = None,
        return_attention_weights: bool = True
    ) -> DailyEncoderOutput:
        """
        Encode multiple daily sources and aggregate to monthly.

        Args:
            source_features: Dict mapping source name to [batch, n_days, n_features]
            source_masks: Optional dict mapping source name to [batch, n_days] mask
            n_months: Number of months to aggregate to
            return_attention_weights: If True, return aggregation attention

        Returns:
            DailyEncoderOutput with daily_hidden, monthly_hidden, and aggregation_attention
        """
        first_source = next(iter(source_features.values()))
        batch_size, n_days, _ = first_source.shape

        if n_months is None:
            n_months = max(1, n_days // 30)

        # Source-specific encoding
        source_hidden = {}
        for name in self.source_names:
            if name in source_features:
                source_hidden[name] = self.source_projections[name](source_features[name])
            else:
                source_hidden[name] = torch.zeros(
                    batch_size, n_days, self.d_model, device=first_source.device
                )

        # Source fusion
        fused, combined_mask = self._fuse_sources(source_hidden, source_masks)

        # Positional encoding
        fused = self.positional_encoding(fused)

        # Transformer encoding
        src_key_padding_mask = None
        if combined_mask is not None:
            src_key_padding_mask = ~combined_mask

        daily_hidden = self.transformer_encoder(fused, src_key_padding_mask=src_key_padding_mask)
        daily_hidden = self.encoder_output_norm(daily_hidden)

        # Monthly aggregation
        aggregator_output = self.monthly_aggregator(
            daily_hidden=daily_hidden,
            n_months=n_months,
            daily_mask=combined_mask,
            return_attention_weights=return_attention_weights
        )

        if return_attention_weights:
            monthly_hidden, attention_weights = aggregator_output
            return DailyEncoderOutput(
                daily_hidden=daily_hidden,
                monthly_hidden=monthly_hidden,
                aggregation_attention=attention_weights
            )
        else:
            monthly_hidden = aggregator_output
            return DailyEncoderOutput(
                daily_hidden=daily_hidden,
                monthly_hidden=monthly_hidden,
                aggregation_attention=None
            )


# =============================================================================
# DAILY ENCODER FACTORY FUNCTIONS
# =============================================================================

def create_daily_encoder(
    n_features: int = 165,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
    positional_encoding: str = 'sinusoidal',
) -> DailyEncoder:
    """
    Factory function to create a DailyEncoder for daily-resolution sources.

    Args:
        n_features: Total number of input features.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dropout: Dropout rate.
        positional_encoding: Type of positional encoding ('sinusoidal' or 'learned').

    Returns:
        Configured DailyEncoder instance.

    Example:
        >>> encoder = create_daily_encoder(n_features=165, d_model=128)
        >>> output = encoder(x, observation_mask=mask, n_months=33)
    """
    return DailyEncoder(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        positional_encoding=positional_encoding,
    )


def create_multi_source_daily_encoder(
    sources: Optional[List[str]] = None,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    dropout: float = 0.1,
    fusion_method: str = 'attention',
) -> MultiSourceDailyEncoder:
    """
    Factory function to create a MultiSourceDailyEncoder.

    Args:
        sources: List of source names to include (default: all daily sources).
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dropout: Dropout rate.
        fusion_method: How to fuse sources ('attention', 'concatenate', 'sum').

    Returns:
        Configured MultiSourceDailyEncoder instance.

    Example:
        >>> encoder = create_multi_source_daily_encoder(
        ...     sources=['equipment', 'personnel', 'firms'],
        ...     d_model=128
        ... )
    """
    if sources is None:
        sources = list(DAILY_SOURCE_CONFIGS.keys())

    configs = {s: DAILY_SOURCE_CONFIGS[s] for s in sources}

    return MultiSourceDailyEncoder(
        source_configs=configs,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        fusion_method=fusion_method,
    )


# =============================================================================
# DAILY ENCODER TESTS
# =============================================================================

def _test_daily_encoder() -> None:
    """Test DailyEncoder functionality."""
    print("=" * 80)
    print("Testing DailyEncoder")
    print("=" * 80)

    encoder = create_daily_encoder(n_features=165, d_model=128, n_layers=4)

    batch_size = 4
    n_days = 1000
    n_features = 165
    n_months = 33

    x = torch.randn(batch_size, n_days, n_features)
    mask = torch.ones(batch_size, n_days, dtype=torch.bool)

    # Simulate some missing days
    mask[:, 100:105] = False
    mask[:, 500:510] = False
    mask[:, 750:752] = False

    output = encoder(x, observation_mask=mask, n_months=n_months)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  observation_mask: {mask.shape}")

    print(f"\nOutput shapes:")
    print(f"  daily_hidden: {output.daily_hidden.shape}")
    print(f"  monthly_hidden: {output.monthly_hidden.shape}")

    if output.aggregation_attention is not None:
        print(f"  aggregation_attention: {output.aggregation_attention.shape}")

    # Check attention sums to ~1
    if output.aggregation_attention is not None:
        attn_sum = output.aggregation_attention.sum(dim=-1).mean().item()
        print(f"\nMean attention sum (should be ~1.0): {attn_sum:.4f}")

    assert not torch.isnan(output.daily_hidden).any(), "NaN in daily_hidden!"
    assert not torch.isnan(output.monthly_hidden).any(), "NaN in monthly_hidden!"

    print("\n[PASS] DailyEncoder test passed!")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nParameter count: {n_params:,}")


def _test_multi_source_daily_encoder() -> None:
    """Test MultiSourceDailyEncoder functionality."""
    print("\n" + "=" * 80)
    print("Testing MultiSourceDailyEncoder")
    print("=" * 80)

    encoder = create_multi_source_daily_encoder(d_model=128)

    batch_size = 4
    n_days = 1000
    n_months = 33

    source_features = {}
    source_masks = {}

    for name, config in DAILY_SOURCE_CONFIGS.items():
        source_features[name] = torch.randn(batch_size, n_days, config.n_features)
        source_masks[name] = torch.ones(batch_size, n_days, dtype=torch.bool)
        # Simulate different missing patterns per source
        start = hash(name) % 100
        source_masks[name][:, start:start+20] = False

    output = encoder(
        source_features=source_features,
        source_masks=source_masks,
        n_months=n_months,
    )

    print(f"\nInput shapes:")
    for name in source_features:
        print(f"  {name}: {source_features[name].shape}")

    print(f"\nOutput shapes:")
    print(f"  daily_hidden: {output.daily_hidden.shape}")
    print(f"  monthly_hidden: {output.monthly_hidden.shape}")

    if output.aggregation_attention is not None:
        print(f"  aggregation_attention: {output.aggregation_attention.shape}")

    assert not torch.isnan(output.daily_hidden).any(), "NaN in daily_hidden!"
    assert not torch.isnan(output.monthly_hidden).any(), "NaN in monthly_hidden!"

    print("\n[PASS] MultiSourceDailyEncoder test passed!")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nParameter count: {n_params:,}")


# =============================================================================
# MONTHLY ENCODER TESTS
# =============================================================================

def _test_monthly_encoder() -> None:
    """Test MonthlyEncoder functionality."""
    print("=" * 80)
    print("Testing MonthlyEncoder")
    print("=" * 80)

    encoder = create_monthly_encoder('sentinel', d_model=64, num_layers=2)

    batch_size = 4
    n_months = 35
    n_features = 43

    values = torch.randn(batch_size, n_months, n_features)
    observation_mask = (torch.rand(batch_size, n_months) > 0.65).float()
    month_indices = torch.arange(n_months).unsqueeze(0).expand(batch_size, -1)

    outputs = encoder(
        values=values,
        observation_mask=observation_mask,
        month_indices=month_indices,
        return_attention=True,
    )

    print(f"\nInput shapes:")
    print(f"  values: {values.shape}")
    print(f"  observation_mask: {observation_mask.shape}")
    print(f"  month_indices: {month_indices.shape}")

    print(f"\nOutput shapes:")
    print(f"  hidden: {outputs['hidden'].shape}")
    print(f"  uncertainty: {outputs['uncertainty'].shape}")

    if 'attention_weights' in outputs:
        print(f"  attention_weights: {len(outputs['attention_weights'])} layers")

    obs_rate = observation_mask.mean().item()
    print(f"\nObservation statistics:")
    print(f"  Observation rate: {obs_rate:.1%}")

    obs_mask_bool = observation_mask.bool()
    obs_uncertainty = outputs['uncertainty'][obs_mask_bool.unsqueeze(-1).expand_as(outputs['uncertainty'])].mean().item()
    unobs_uncertainty = outputs['uncertainty'][~obs_mask_bool.unsqueeze(-1).expand_as(outputs['uncertainty'])].mean().item()
    print(f"  Mean uncertainty (observed): {obs_uncertainty:.3f}")
    print(f"  Mean uncertainty (unobserved): {unobs_uncertainty:.3f}")

    assert not torch.isnan(outputs['hidden']).any(), "NaN in hidden!"
    assert not torch.isnan(outputs['uncertainty']).any(), "NaN in uncertainty!"

    print("\n[PASS] MonthlyEncoder test passed!")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nParameter count: {n_params:,}")


def _test_multi_source_encoder() -> None:
    """Test MultiSourceMonthlyEncoder functionality."""
    print("\n" + "=" * 80)
    print("Testing MultiSourceMonthlyEncoder")
    print("=" * 80)

    encoder = create_multi_source_encoder(d_model=64)

    batch_size = 4
    n_months = 35

    source_features = {}
    source_masks = {}

    for name, config in MONTHLY_SOURCE_CONFIGS.items():
        source_features[name] = torch.randn(batch_size, n_months, config.n_features)
        obs_rate = 0.3 + 0.4 * torch.rand(1).item()
        source_masks[name] = (torch.rand(batch_size, n_months) < obs_rate).float()

    month_indices = torch.arange(n_months).unsqueeze(0).expand(batch_size, -1)

    outputs = encoder(
        source_features=source_features,
        source_masks=source_masks,
        month_indices=month_indices,
    )

    print(f"\nInput shapes:")
    for name in source_features:
        print(f"  {name}: {source_features[name].shape}, mask: {source_masks[name].shape}")

    print(f"\nOutput shapes:")
    print(f"  hidden: {outputs['hidden'].shape}")
    print(f"  uncertainty: {outputs['uncertainty'].shape}")
    print(f"  source_importance: {outputs['source_importance'].shape}")

    print(f"\nMean source importance:")
    mean_importance = outputs['source_importance'].mean(dim=(0, 1))
    for i, name in enumerate(encoder.source_names):
        print(f"  {name}: {mean_importance[i].item():.3f}")

    assert not torch.isnan(outputs['hidden']).any(), "NaN in hidden!"

    print("\n[PASS] MultiSourceMonthlyEncoder test passed!")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nParameter count: {n_params:,}")


if __name__ == "__main__":
    _test_fusion_module()

    print("\n" + "=" * 80)
    print("DAILY ENCODER TESTS")
    print("=" * 80)

    _test_daily_encoder()
    _test_multi_source_daily_encoder()

    print("\n" + "=" * 80)
    print("MONTHLY ENCODER TESTS")
    print("=" * 80)

    _test_monthly_encoder()
    _test_multi_source_encoder()

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
