"""
Multi-Resolution Time Series Architecture Proposal
===================================================

Problem Statement:
------------------
The codebase has two data resolution categories:
  - DAILY (~1000+ samples): equipment, personnel, deepstate, firms, viina
  - MONTHLY (~32-45 samples): sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom

Current approaches both destroy information:
  1. Aggregate daily to monthly -> loses 97% of daily signal granularity
  2. Forward-fill monthly to daily -> fabricates 97% of sparse data

GOAL: Use daily data at daily resolution AND monthly data at monthly resolution,
combining them meaningfully without fabrication.

This module proposes three concrete architectures with implementations.
"""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")


# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

class Resolution(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass
class SourceSpec:
    """Specification for a data source."""
    name: str
    resolution: Resolution
    n_features: int
    typical_samples: int  # Expected number of observations
    description: str


# Define all sources with their true resolutions
SOURCE_SPECS = {
    # DAILY RESOLUTION (~1000+ samples over war period)
    'equipment': SourceSpec('equipment', Resolution.DAILY, 38, 1000, 'Equipment loss counts'),
    'personnel': SourceSpec('personnel', Resolution.DAILY, 6, 1000, 'Personnel casualty figures'),
    'deepstate': SourceSpec('deepstate', Resolution.DAILY, 55, 1000, 'Front line territorial data'),
    'firms': SourceSpec('firms', Resolution.DAILY, 42, 1000, 'Fire detection satellite data'),
    'viina': SourceSpec('viina', Resolution.DAILY, 24, 1000, 'VIINA territorial control'),

    # MONTHLY RESOLUTION (~32-45 samples over war period)
    'sentinel': SourceSpec('sentinel', Resolution.MONTHLY, 43, 32, 'Multi-spectral satellite imagery'),
    'hdx_conflict': SourceSpec('hdx_conflict', Resolution.MONTHLY, 18, 40, 'HDX conflict events'),
    'hdx_food': SourceSpec('hdx_food', Resolution.MONTHLY, 20, 40, 'Food price indices'),
    'hdx_rainfall': SourceSpec('hdx_rainfall', Resolution.MONTHLY, 16, 45, 'Dekadal rainfall data'),
    'iom': SourceSpec('iom', Resolution.MONTHLY, 18, 35, 'IOM displacement surveys'),
}


# =============================================================================
# APPROACH 1: HIERARCHICAL MULTI-RATE ARCHITECTURE
# =============================================================================
#
# Key Insight: Process each resolution in its own encoder, then fuse at the
# coarser (monthly) level. Daily encoder aggregates its hidden states to
# monthly boundaries before fusion.
#
# Architecture:
#   Daily sources -> Daily Encoder (processes all 1000 days)
#                         |
#                         v (aggregate to monthly boundaries)
#                    [Monthly Hidden States]
#                         |
#   Monthly sources -> Monthly Encoder (processes all 32 months)
#                         |
#                         v
#                    Cross-Resolution Fusion
#                         |
#                         v
#                    Prediction Heads
#
# Pros:
#   - Daily data retains full granularity in daily encoder
#   - No fabrication of monthly data
#   - Fusion happens at consistent time points
#   - Can predict at daily OR monthly resolution
#
# Cons:
#   - Daily encoder must learn to aggregate meaningfully
#   - More complex than simple concatenation
#   - Training requires careful batching

if HAS_TORCH:

    class DailyEncoder(nn.Module):
        """
        Encodes daily time series data with learnable aggregation to monthly.

        Uses a hierarchical approach:
        1. Per-day encoding with local attention
        2. Learnable aggregation to monthly buckets
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 3,
            dropout: float = 0.1,
            days_per_month: int = 30
        ):
            super().__init__()
            self.d_model = d_model
            self.days_per_month = days_per_month

            # Feature projection
            self.feature_proj = nn.Linear(n_features, d_model)

            # Learnable positional encoding (daily)
            self.daily_pos = nn.Embedding(1500, d_model)  # Max 1500 days

            # Day-of-month encoding (for aggregation awareness)
            self.day_of_month_emb = nn.Embedding(31, d_model)

            # Daily transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.daily_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Learnable monthly aggregation via cross-attention
            # Query: month positions, Key/Value: daily encodings
            self.month_query = nn.Parameter(torch.randn(1, 50, d_model))  # Max 50 months
            self.aggregation_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

            # Layer norm for aggregated output
            self.agg_norm = nn.LayerNorm(d_model)

        def forward(
            self,
            x: torch.Tensor,
            observation_mask: torch.Tensor,
            day_indices: torch.Tensor,
            month_boundaries: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: [batch, seq_len, n_features] - daily observations
                observation_mask: [batch, seq_len] - 1 where data exists, 0 for missing
                day_indices: [batch, seq_len] - absolute day index (0-1000+)
                month_boundaries: [batch, n_months, 2] - (start_day, end_day) per month

            Returns:
                daily_hidden: [batch, seq_len, d_model] - full daily encodings
                monthly_hidden: [batch, n_months, d_model] - aggregated to monthly
            """
            batch_size, seq_len, _ = x.shape
            device = x.device
            n_months = month_boundaries.shape[1]

            # Project features
            x_proj = self.feature_proj(x)

            # Add positional encodings
            pos_emb = self.daily_pos(day_indices)
            day_of_month = day_indices % 31  # Approximate day-of-month
            dom_emb = self.day_of_month_emb(day_of_month)

            x_encoded = x_proj + pos_emb + dom_emb

            # Apply observation mask (zero out missing days for attention)
            attn_mask = (observation_mask == 0)  # True = ignore

            # Daily encoding
            daily_hidden = self.daily_encoder(
                x_encoded,
                src_key_padding_mask=attn_mask
            )

            # Aggregate to monthly via cross-attention
            # Expand month queries for batch
            month_queries = self.month_query[:, :n_months, :].expand(batch_size, -1, -1)

            # Cross-attend: months attend to days
            monthly_hidden, _ = self.aggregation_attn(
                month_queries,  # Query: what we want (monthly representations)
                daily_hidden,   # Key: what we have (daily encodings)
                daily_hidden,   # Value: what we extract
                key_padding_mask=attn_mask
            )

            monthly_hidden = self.agg_norm(monthly_hidden)

            return daily_hidden, monthly_hidden


    class MonthlyEncoder(nn.Module):
        """
        Encodes monthly time series data.

        Simpler than daily encoder since we have fewer time points.
        Includes explicit "observation present" embedding.
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.d_model = d_model

            # Feature projection
            self.feature_proj = nn.Linear(n_features, d_model)

            # Monthly positional encoding
            self.month_pos = nn.Embedding(60, d_model)  # Max 60 months

            # Observation mask embedding (learnable "no observation" token)
            self.no_obs_token = nn.Parameter(torch.randn(1, 1, d_model))

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(
            self,
            x: torch.Tensor,
            observation_mask: torch.Tensor,
            month_indices: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                x: [batch, n_months, n_features] - monthly observations
                observation_mask: [batch, n_months] - 1 where observation exists
                month_indices: [batch, n_months] - month index (0-59)

            Returns:
                hidden: [batch, n_months, d_model]
            """
            batch_size, n_months, _ = x.shape

            # Project features
            x_proj = self.feature_proj(x)

            # Add positional encoding
            pos_emb = self.month_pos(month_indices)
            x_encoded = x_proj + pos_emb

            # Replace missing observations with learned "no observation" token
            # This is CRITICAL: we don't fabricate values, we use a learned placeholder
            obs_mask_expanded = observation_mask.unsqueeze(-1)  # [batch, n_months, 1]
            no_obs_expanded = self.no_obs_token.expand(batch_size, n_months, -1)

            x_masked = torch.where(
                obs_mask_expanded.bool(),
                x_encoded,
                no_obs_expanded + pos_emb  # Still add position to no-obs token
            )

            # Encode (no key_padding_mask - we handle missing via token replacement)
            hidden = self.encoder(x_masked)

            return hidden


    class CrossResolutionFusion(nn.Module):
        """
        Fuses daily-aggregated and monthly encodings.

        Uses cross-attention where each resolution can attend to the other,
        learning complementary information.
        """

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()

            # Daily-to-Monthly attention (daily queries monthly context)
            self.daily_to_monthly = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

            # Monthly-to-Daily attention (monthly queries daily details)
            self.monthly_to_daily = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )

            # Fusion layers
            self.daily_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            self.monthly_fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        def forward(
            self,
            daily_from_daily: torch.Tensor,
            monthly_from_daily: torch.Tensor,
            monthly_from_monthly: torch.Tensor,
            monthly_obs_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Fuse representations from both resolution encoders.

            Args:
                daily_from_daily: [batch, n_days, d_model] - daily encoder output
                monthly_from_daily: [batch, n_months, d_model] - daily aggregated to monthly
                monthly_from_monthly: [batch, n_months, d_model] - monthly encoder output
                monthly_obs_mask: [batch, n_months] - which months have sparse observations

            Returns:
                fused_daily: [batch, n_days, d_model]
                fused_monthly: [batch, n_months, d_model]
            """
            # Monthly queries daily (get fine-grained details)
            monthly_context, _ = self.monthly_to_daily(
                monthly_from_monthly,  # Query
                daily_from_daily,      # Key
                daily_from_daily       # Value
            )

            # Fuse monthly: combine own encoding + context from daily
            monthly_combined = torch.cat([monthly_from_monthly, monthly_context], dim=-1)
            fused_monthly = self.monthly_fusion(monthly_combined) + monthly_from_daily

            # Daily queries monthly (get broader context) - but at aggregated level
            # First, repeat monthly to daily resolution for attention
            daily_context, _ = self.daily_to_monthly(
                monthly_from_daily,    # Query (daily aggregated to monthly)
                monthly_from_monthly,  # Key
                monthly_from_monthly   # Value
            )

            # Fuse daily at monthly level, then will be used for prediction
            daily_combined = torch.cat([monthly_from_daily, daily_context], dim=-1)
            fused_daily_monthly = self.daily_fusion(daily_combined)

            return fused_daily_monthly, fused_monthly


    class HierarchicalMultiRateModel(nn.Module):
        """
        APPROACH 1: Hierarchical Multi-Rate Architecture

        Complete model that:
        1. Processes daily sources with DailyEncoder
        2. Processes monthly sources with MonthlyEncoder
        3. Fuses at monthly resolution
        4. Predicts at desired resolution
        """

        def __init__(
            self,
            daily_sources: Dict[str, int],   # {source_name: n_features}
            monthly_sources: Dict[str, int], # {source_name: n_features}
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 3,
            dropout: float = 0.1
        ):
            super().__init__()
            self.daily_sources = list(daily_sources.keys())
            self.monthly_sources = list(monthly_sources.keys())
            self.d_model = d_model

            # Per-source daily encoders
            total_daily_features = sum(daily_sources.values())
            self.daily_encoder = DailyEncoder(
                n_features=total_daily_features,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dropout=dropout
            )

            # Per-source monthly encoders (one per source for interpretability)
            self.monthly_encoders = nn.ModuleDict()
            for name, n_feat in monthly_sources.items():
                self.monthly_encoders[name] = MonthlyEncoder(
                    n_features=n_feat,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=2,
                    dropout=dropout
                )

            # Monthly source fusion (combine all monthly encoders)
            n_monthly_sources = len(monthly_sources)
            self.monthly_source_fusion = nn.Sequential(
                nn.Linear(d_model * n_monthly_sources, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )

            # Cross-resolution fusion
            self.cross_fusion = CrossResolutionFusion(
                d_model=d_model,
                nhead=nhead,
                num_layers=2,
                dropout=dropout
            )

            # Prediction heads
            self.monthly_pred_head = nn.Linear(d_model, 1)  # Predict target at monthly
            self.daily_pred_head = nn.Linear(d_model, 1)    # Predict target at daily

        def forward(
            self,
            daily_features: torch.Tensor,
            daily_obs_mask: torch.Tensor,
            daily_indices: torch.Tensor,
            month_boundaries: torch.Tensor,
            monthly_features: Dict[str, torch.Tensor],
            monthly_obs_masks: Dict[str, torch.Tensor],
            month_indices: torch.Tensor,
            predict_resolution: str = 'monthly'
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass through hierarchical model.

            Args:
                daily_features: [batch, n_days, total_daily_features]
                daily_obs_mask: [batch, n_days] - 1 where daily observation exists
                daily_indices: [batch, n_days] - day index
                month_boundaries: [batch, n_months, 2] - (start_day, end_day)
                monthly_features: {source: [batch, n_months, n_features]}
                monthly_obs_masks: {source: [batch, n_months]} - 1 where observation exists
                month_indices: [batch, n_months]
                predict_resolution: 'daily' or 'monthly'

            Returns:
                predictions and attention weights
            """
            batch_size = daily_features.shape[0]
            n_months = month_indices.shape[1]

            # Encode daily sources
            daily_hidden, daily_aggregated = self.daily_encoder(
                daily_features, daily_obs_mask, daily_indices, month_boundaries
            )

            # Encode monthly sources separately
            monthly_encodings = []
            for source_name in self.monthly_sources:
                if source_name in monthly_features:
                    enc = self.monthly_encoders[source_name](
                        monthly_features[source_name],
                        monthly_obs_masks[source_name],
                        month_indices
                    )
                    monthly_encodings.append(enc)

            # Fuse monthly encodings
            if monthly_encodings:
                monthly_concat = torch.cat(monthly_encodings, dim=-1)
                monthly_fused = self.monthly_source_fusion(monthly_concat)
            else:
                monthly_fused = torch.zeros(batch_size, n_months, self.d_model,
                                           device=daily_features.device)

            # Compute combined observation mask for monthly
            combined_monthly_mask = torch.stack(
                [monthly_obs_masks[s] for s in self.monthly_sources if s in monthly_obs_masks],
                dim=-1
            ).any(dim=-1).float()

            # Cross-resolution fusion
            fused_daily_at_monthly, fused_monthly = self.cross_fusion(
                daily_hidden,
                daily_aggregated,
                monthly_fused,
                combined_monthly_mask
            )

            # Predictions
            outputs = {
                'daily_hidden': daily_hidden,
                'monthly_hidden': fused_monthly,
                'daily_aggregated': fused_daily_at_monthly
            }

            if predict_resolution == 'monthly':
                outputs['predictions'] = self.monthly_pred_head(fused_monthly)
            else:
                # For daily predictions, we need to upsample monthly fusion
                # This is where we're careful not to fabricate
                outputs['predictions'] = self.daily_pred_head(daily_hidden)
                outputs['monthly_predictions'] = self.monthly_pred_head(fused_monthly)

            return outputs


# =============================================================================
# APPROACH 2: OBSERVATION-MASKED ATTENTION NETWORK (OMAN)
# =============================================================================
#
# Key Insight: Use a single unified timeline but with explicit observation
# masks that distinguish "no data" from "zero value". Attention only flows
# from REAL observations.
#
# Architecture:
#   All sources aligned to daily timeline
#         |
#         v
#   Feature Embedding (with source type + observation status)
#         |
#         v
#   Masked Self-Attention (only attends to real observations)
#         |
#         v
#   Prediction Heads
#
# The key difference from forward-filling: we don't fill values, we mask
# the attention so that only real observations contribute.
#
# Pros:
#   - Single unified architecture
#   - Clean handling of irregular observations
#   - Attention weights show exactly which observations inform predictions
#
# Cons:
#   - Sparse monthly sources have very few attended positions
#   - May struggle with very long gaps between observations

if HAS_TORCH:

    class ObservationEmbedding(nn.Module):
        """
        Embeds feature values with explicit observation status.

        For each feature at each time step, produces an embedding that encodes:
        1. The feature value (if observed)
        2. The feature type (which source/feature)
        3. The observation status (observed vs not observed)
        4. Temporal position
        """

        def __init__(
            self,
            n_sources: int,
            max_features_per_source: int,
            d_model: int = 64,
            max_days: int = 1500
        ):
            super().__init__()
            self.d_model = d_model

            # Value projection (per feature type would be ideal, but simplified here)
            self.value_proj = nn.Linear(1, d_model)

            # Source embedding
            self.source_emb = nn.Embedding(n_sources, d_model)

            # Feature-within-source embedding
            self.feature_emb = nn.Embedding(max_features_per_source, d_model)

            # Observation status embedding (0=not observed, 1=observed)
            self.obs_status_emb = nn.Embedding(2, d_model)

            # Temporal positional encoding
            self.temporal_pos = nn.Embedding(max_days, d_model)

            # Learned "no observation" value representation
            self.no_obs_value = nn.Parameter(torch.randn(1, 1, d_model))

            # Combination layer
            self.combine = nn.Sequential(
                nn.Linear(d_model * 4, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )

        def forward(
            self,
            values: torch.Tensor,
            obs_mask: torch.Tensor,
            source_ids: torch.Tensor,
            feature_ids: torch.Tensor,
            day_indices: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                values: [batch, seq_len, n_features_total] - feature values
                obs_mask: [batch, seq_len, n_features_total] - 1 if observed
                source_ids: [batch, seq_len, n_features_total] - source index
                feature_ids: [batch, seq_len, n_features_total] - feature index within source
                day_indices: [batch, seq_len] - day index

            Returns:
                embeddings: [batch, seq_len * n_features_total, d_model]
                attention_mask: [batch, seq_len * n_features_total] - for masked attention
            """
            batch_size, seq_len, n_features = values.shape

            # Flatten to treat each feature at each time as a token
            values_flat = values.reshape(batch_size, -1, 1)  # [batch, seq*feat, 1]
            obs_flat = obs_mask.reshape(batch_size, -1)      # [batch, seq*feat]
            source_flat = source_ids.reshape(batch_size, -1)
            feature_flat = feature_ids.reshape(batch_size, -1)

            # Expand day indices to match features
            day_flat = day_indices.unsqueeze(-1).expand(-1, -1, n_features).reshape(batch_size, -1)

            # Get embeddings
            value_emb = self.value_proj(values_flat)
            source_emb = self.source_emb(source_flat)
            feature_emb = self.feature_emb(feature_flat)
            obs_emb = self.obs_status_emb(obs_flat.long())
            temporal_emb = self.temporal_pos(day_flat)

            # For unobserved positions, replace value embedding with learned token
            no_obs_expanded = self.no_obs_value.expand(batch_size, seq_len * n_features, -1)
            value_emb = torch.where(
                obs_flat.unsqueeze(-1).bool(),
                value_emb,
                no_obs_expanded
            )

            # Combine all embeddings
            combined = torch.cat([value_emb, source_emb, feature_emb, temporal_emb], dim=-1)
            embeddings = self.combine(combined)

            # Attention mask: only attend to observed positions
            # But allow queries from unobserved to attend to observed
            attention_mask = obs_flat  # 1 = can be attended to, 0 = cannot

            return embeddings, attention_mask


    class ObservationMaskedAttention(nn.Module):
        """
        Self-attention where unobserved positions cannot be keys/values.

        This ensures information only flows FROM real observations.
        Unobserved positions can query (to get predictions) but not contribute.
        """

        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()

            # Custom attention that respects observation mask
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(
                        d_model, nhead, dropout=dropout, batch_first=True
                    ),
                    'ffn': nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model * 4, d_model),
                        nn.Dropout(dropout)
                    ),
                    'norm1': nn.LayerNorm(d_model),
                    'norm2': nn.LayerNorm(d_model)
                }))

        def forward(
            self,
            x: torch.Tensor,
            obs_mask: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model] - input embeddings
                obs_mask: [batch, seq_len] - 1 for observed positions

            Returns:
                output: [batch, seq_len, d_model]
            """
            # Convert obs_mask to attention mask
            # key_padding_mask: True = IGNORE this position
            key_padding_mask = (obs_mask == 0)

            for layer in self.layers:
                # Self-attention with masking
                attn_out, _ = layer['self_attn'](
                    x, x, x,
                    key_padding_mask=key_padding_mask
                )
                x = layer['norm1'](x + attn_out)

                # FFN
                ffn_out = layer['ffn'](x)
                x = layer['norm2'](x + ffn_out)

            return x


    class OMAN(nn.Module):
        """
        APPROACH 2: Observation-Masked Attention Network

        Unified model that handles multi-resolution data through
        explicit observation masking rather than interpolation.
        """

        def __init__(
            self,
            source_specs: Dict[str, SourceSpec],
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            self.source_specs = source_specs
            self.source_names = list(source_specs.keys())
            self.d_model = d_model

            # Calculate feature layout
            self.feature_offsets = {}
            self.n_features_total = 0
            for name, spec in source_specs.items():
                self.feature_offsets[name] = self.n_features_total
                self.n_features_total += spec.n_features

            # Embedding layer
            max_features = max(s.n_features for s in source_specs.values())
            self.embedding = ObservationEmbedding(
                n_sources=len(source_specs),
                max_features_per_source=max_features,
                d_model=d_model
            )

            # Main attention network
            self.attention = ObservationMaskedAttention(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout
            )

            # Prediction head
            self.pred_head = nn.Linear(d_model, 1)

        def forward(
            self,
            features: Dict[str, torch.Tensor],
            obs_masks: Dict[str, torch.Tensor],
            day_indices: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.

            Args:
                features: {source_name: [batch, seq_len, n_features]}
                obs_masks: {source_name: [batch, seq_len]} - per-day observation mask
                day_indices: [batch, seq_len]

            Returns:
                predictions and hidden states
            """
            batch_size, seq_len = day_indices.shape
            device = day_indices.device

            # Concatenate all features
            all_features = []
            all_obs_masks = []
            source_ids = []
            feature_ids = []

            for i, (name, spec) in enumerate(self.source_specs.items()):
                feat = features.get(name, torch.zeros(batch_size, seq_len, spec.n_features, device=device))
                mask = obs_masks.get(name, torch.zeros(batch_size, seq_len, device=device))

                all_features.append(feat)
                # Expand day-level mask to feature level
                all_obs_masks.append(mask.unsqueeze(-1).expand(-1, -1, spec.n_features))

                # Source and feature IDs
                source_ids.append(torch.full((batch_size, seq_len, spec.n_features), i, device=device))
                feature_ids.append(
                    torch.arange(spec.n_features, device=device).view(1, 1, -1).expand(batch_size, seq_len, -1)
                )

            # Stack
            all_features = torch.cat(all_features, dim=-1)
            all_obs_masks = torch.cat(all_obs_masks, dim=-1)
            source_ids = torch.cat(source_ids, dim=-1)
            feature_ids = torch.cat(feature_ids, dim=-1)

            # Embed
            embeddings, attn_mask = self.embedding(
                all_features, all_obs_masks, source_ids, feature_ids, day_indices
            )

            # Apply masked attention
            hidden = self.attention(embeddings, attn_mask)

            # Reshape back to [batch, seq, n_features, d_model]
            hidden_reshaped = hidden.view(batch_size, seq_len, self.n_features_total, self.d_model)

            # Pool across features for temporal prediction
            hidden_pooled = hidden_reshaped.mean(dim=2)  # [batch, seq, d_model]

            predictions = self.pred_head(hidden_pooled)

            return {
                'predictions': predictions,
                'hidden': hidden_pooled,
                'attention_mask': attn_mask
            }


# =============================================================================
# APPROACH 3: MULTI-RATE STATE SPACE MODEL
# =============================================================================
#
# Key Insight: Model the system as having a continuous latent state that
# evolves at the fastest resolution (daily). Observations at different
# resolutions are partial observations of this state.
#
# Architecture:
#   Latent State z(t) evolves daily: z(t+1) = f(z(t)) + noise
#
#   Daily observations: y_daily(t) = g_daily(z(t)) + noise
#   Monthly observations: y_monthly(m) = h_monthly(z(t_m)) + noise
#
# Training: Use variational inference to learn state transitions
# and observation models simultaneously.
#
# Pros:
#   - Principled probabilistic framework
#   - Natural handling of irregular observations
#   - Provides uncertainty estimates
#   - Can do proper imputation with uncertainty
#
# Cons:
#   - More complex training (variational)
#   - Computational cost for long sequences
#   - Requires careful tuning

if HAS_TORCH:

    class StateTransitionModel(nn.Module):
        """
        Models latent state evolution: z(t+1) = f(z(t)) + noise

        Uses GRU-like gating for stable long-term dynamics.
        """

        def __init__(self, state_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            self.state_dim = state_dim

            # Gated transition (GRU-like)
            self.update_gate = nn.Sequential(
                nn.Linear(state_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
                nn.Sigmoid()
            )

            self.candidate = nn.Sequential(
                nn.Linear(state_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim),
                nn.Tanh()
            )

            # State noise (learnable diagonal covariance)
            self.log_noise_std = nn.Parameter(torch.zeros(state_dim))

        def forward(
            self,
            z: torch.Tensor,
            dt: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict next state from current state.

            Args:
                z: [batch, state_dim] - current state
                dt: [batch] - time delta (optional, for variable step sizes)

            Returns:
                z_next_mean: [batch, state_dim] - predicted mean
                z_next_std: [batch, state_dim] - predicted std
            """
            # Self-loop for gating
            z_doubled = torch.cat([z, z], dim=-1)

            update = self.update_gate(z_doubled)
            candidate = self.candidate(z_doubled)

            z_next_mean = (1 - update) * z + update * candidate
            z_next_std = torch.exp(self.log_noise_std).unsqueeze(0).expand_as(z_next_mean)

            return z_next_mean, z_next_std


    class ObservationModel(nn.Module):
        """
        Maps latent state to observation space.

        Different heads for different data sources.
        """

        def __init__(
            self,
            state_dim: int,
            source_dims: Dict[str, int],
            hidden_dim: int = 128
        ):
            super().__init__()
            self.source_names = list(source_dims.keys())

            # Per-source observation models
            self.obs_models = nn.ModuleDict()
            self.obs_log_stds = nn.ParameterDict()

            for name, dim in source_dims.items():
                self.obs_models[name] = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim)
                )
                self.obs_log_stds[name] = nn.Parameter(torch.zeros(dim))

        def forward(
            self,
            z: torch.Tensor,
            source_name: str
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Predict observation for a specific source.

            Args:
                z: [batch, state_dim] - latent state
                source_name: which source to predict

            Returns:
                obs_mean: [batch, source_dim]
                obs_std: [batch, source_dim]
            """
            obs_mean = self.obs_models[source_name](z)
            obs_std = torch.exp(self.obs_log_stds[source_name]).unsqueeze(0).expand_as(obs_mean)

            return obs_mean, obs_std


    class EncoderModel(nn.Module):
        """
        Encodes observations into approximate posterior over latent state.

        q(z_t | y_1:t) - recognition model for variational inference.
        """

        def __init__(
            self,
            source_dims: Dict[str, int],
            state_dim: int = 64,
            hidden_dim: int = 128
        ):
            super().__init__()
            self.state_dim = state_dim

            # Per-source encoders
            self.source_encoders = nn.ModuleDict()
            for name, dim in source_dims.items():
                self.source_encoders[name] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim * 2)  # mean + log_std
                )

            # Fusion for combining multiple source observations
            n_sources = len(source_dims)
            self.fusion = nn.Sequential(
                nn.Linear(state_dim * 2 * n_sources, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim * 2)
            )

            # Prior-informed combination with transition
            self.combine_with_prior = nn.Sequential(
                nn.Linear(state_dim * 4, hidden_dim),  # posterior + prior
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim * 2)
            )

        def forward(
            self,
            observations: Dict[str, torch.Tensor],
            obs_masks: Dict[str, torch.Tensor],
            prior_mean: torch.Tensor,
            prior_std: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute approximate posterior given observations and prior.

            Args:
                observations: {source: [batch, source_dim]} - observed values
                obs_masks: {source: [batch]} - 1 if observed
                prior_mean, prior_std: [batch, state_dim] - from transition

            Returns:
                posterior_mean, posterior_std: [batch, state_dim]
            """
            batch_size = prior_mean.shape[0]
            device = prior_mean.device

            # Encode available observations
            encoded = []
            for name in self.source_encoders:
                if name in observations and name in obs_masks:
                    obs = observations[name]
                    mask = obs_masks[name]

                    # Encode
                    enc = self.source_encoders[name](obs)

                    # Zero out unobserved (use prior instead)
                    enc = enc * mask.unsqueeze(-1)

                    encoded.append(enc)
                else:
                    # No observation - use zeros (will rely on prior)
                    encoded.append(torch.zeros(batch_size, self.state_dim * 2, device=device))

            # Fuse observations
            if encoded:
                fused = torch.cat(encoded, dim=-1)
                obs_posterior = self.fusion(fused)
            else:
                obs_posterior = torch.zeros(batch_size, self.state_dim * 2, device=device)

            # Combine with prior
            prior_params = torch.cat([prior_mean, prior_std], dim=-1)
            combined = torch.cat([obs_posterior, prior_params], dim=-1)
            posterior_params = self.combine_with_prior(combined)

            # Split into mean and log_std
            posterior_mean = posterior_params[:, :self.state_dim]
            posterior_log_std = posterior_params[:, self.state_dim:]
            posterior_std = F.softplus(posterior_log_std) + 1e-6

            return posterior_mean, posterior_std


    class MultiRateStateSpaceModel(nn.Module):
        """
        APPROACH 3: Multi-Rate State Space Model

        Complete variational state space model for multi-resolution data.

        Latent state evolves at daily resolution.
        Observations can arrive at different rates.
        """

        def __init__(
            self,
            source_specs: Dict[str, SourceSpec],
            state_dim: int = 64,
            hidden_dim: int = 128
        ):
            super().__init__()
            self.source_specs = source_specs
            self.state_dim = state_dim

            source_dims = {name: spec.n_features for name, spec in source_specs.items()}

            # State transition model
            self.transition = StateTransitionModel(state_dim, hidden_dim)

            # Observation model
            self.observation = ObservationModel(state_dim, source_dims, hidden_dim)

            # Encoder (recognition model)
            self.encoder = EncoderModel(source_dims, state_dim, hidden_dim)

            # Initial state prior
            self.z0_mean = nn.Parameter(torch.zeros(state_dim))
            self.z0_log_std = nn.Parameter(torch.zeros(state_dim))

            # Prediction head (from state to target)
            self.pred_head = nn.Linear(state_dim, 1)

        def forward(
            self,
            observations: Dict[str, torch.Tensor],
            obs_masks: Dict[str, torch.Tensor],
            seq_len: int,
            n_samples: int = 1
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass with variational inference.

            Args:
                observations: {source: [batch, seq_len, n_features]}
                obs_masks: {source: [batch, seq_len]} - 1 where observed
                seq_len: number of time steps
                n_samples: number of latent samples for estimation

            Returns:
                predictions, reconstructions, KL divergence
            """
            batch_size = next(iter(observations.values())).shape[0]
            device = next(iter(observations.values())).device

            # Initialize state
            z_mean = self.z0_mean.unsqueeze(0).expand(batch_size, -1)
            z_std = F.softplus(self.z0_log_std).unsqueeze(0).expand(batch_size, -1)

            # Storage
            all_z_means = []
            all_z_stds = []
            all_predictions = []
            total_kl = 0

            for t in range(seq_len):
                # Get observations at time t
                obs_t = {name: obs[:, t] for name, obs in observations.items()}
                mask_t = {name: mask[:, t] for name, mask in obs_masks.items()}

                # Prior from transition (except t=0)
                if t > 0:
                    prior_mean, prior_std = self.transition(z_mean)
                else:
                    prior_mean, prior_std = z_mean, z_std

                # Posterior from encoder
                post_mean, post_std = self.encoder(obs_t, mask_t, prior_mean, prior_std)

                # KL divergence: KL(posterior || prior)
                kl = self._kl_divergence(post_mean, post_std, prior_mean, prior_std)
                total_kl = total_kl + kl

                # Sample state (reparameterization trick)
                eps = torch.randn_like(post_mean)
                z_sample = post_mean + eps * post_std

                # Prediction from state
                pred = self.pred_head(z_sample)

                # Store
                all_z_means.append(post_mean)
                all_z_stds.append(post_std)
                all_predictions.append(pred)

                # Update for next step
                z_mean = post_mean
                z_std = post_std

            return {
                'predictions': torch.stack(all_predictions, dim=1),
                'z_means': torch.stack(all_z_means, dim=1),
                'z_stds': torch.stack(all_z_stds, dim=1),
                'kl_divergence': total_kl
            }

        def _kl_divergence(
            self,
            mu1: torch.Tensor,
            std1: torch.Tensor,
            mu2: torch.Tensor,
            std2: torch.Tensor
        ) -> torch.Tensor:
            """KL divergence between two diagonal Gaussians."""
            var1 = std1 ** 2
            var2 = std2 ** 2

            kl = 0.5 * (
                torch.log(var2 / var1) +
                (var1 + (mu1 - mu2) ** 2) / var2 -
                1
            ).sum(dim=-1).mean()

            return kl

        def impute(
            self,
            observations: Dict[str, torch.Tensor],
            obs_masks: Dict[str, torch.Tensor],
            target_source: str,
            n_samples: int = 100
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Impute missing values for a source with uncertainty.

            Returns mean and std of imputation.
            """
            with torch.no_grad():
                outputs = self.forward(observations, obs_masks,
                                      observations[target_source].shape[1], n_samples)

                # Get observation predictions from latent states
                z_means = outputs['z_means']
                batch_size, seq_len, _ = z_means.shape

                imputed_means = []
                imputed_stds = []

                for t in range(seq_len):
                    z_t = z_means[:, t]
                    obs_mean, obs_std = self.observation(z_t, target_source)
                    imputed_means.append(obs_mean)
                    imputed_stds.append(obs_std)

                return torch.stack(imputed_means, dim=1), torch.stack(imputed_stds, dim=1)


# =============================================================================
# COMPARISON SUMMARY
# =============================================================================

def print_architecture_comparison():
    """Print comparison of the three approaches."""

    print("=" * 80)
    print("MULTI-RESOLUTION TIME SERIES ARCHITECTURE COMPARISON")
    print("=" * 80)

    print("""
PROBLEM: Data at different temporal resolutions
  - DAILY: equipment, personnel, deepstate, firms, viina (~1000 samples)
  - MONTHLY: sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom (~35 samples)

Current bad approaches:
  1. Aggregate daily to monthly -> Loses 97% of daily granularity
  2. Forward-fill monthly to daily -> Fabricates 97% of sparse data


APPROACH 1: HIERARCHICAL MULTI-RATE MODEL
==========================================
Architecture:
  Daily Sources -> Daily Encoder -> Learnable Aggregation to Monthly
                                              |
  Monthly Sources -> Monthly Encoder -------> Cross-Resolution Fusion -> Predictions

Key Features:
  - Daily encoder uses full daily resolution internally
  - Learnable aggregation (cross-attention) compresses to monthly boundaries
  - Monthly encoder uses explicit "no observation" tokens (not zeros)
  - Fusion at monthly resolution allows comparison

Pros:
  + Daily data preserves full granularity in encoder
  + No fabrication - missing monthly = learned placeholder
  + Interpretable: can inspect aggregation attention weights
  + Can predict at either resolution

Cons:
  - Aggregation must learn good compression
  - Information loss when daily -> monthly
  - Complex batching for variable month lengths


APPROACH 2: OBSERVATION-MASKED ATTENTION NETWORK (OMAN)
=======================================================
Architecture:
  All Data Aligned to Daily -> Observation Embedding -> Masked Self-Attention
                                     |
                     [value, source_type, obs_status, position]
                                     |
                          Attention only from REAL observations

Key Features:
  - Single unified architecture for all sources
  - Explicit observation masks distinguish "no data" from "zero"
  - Attention key/value masking: only observed positions can contribute
  - Unobserved positions can query but not be attended to

Pros:
  + Clean, unified architecture
  + Explicit handling of missing data (not fabricated)
  + Attention weights show exactly which observations inform predictions
  + Scales to arbitrary observation patterns

Cons:
  - Monthly sources have very sparse attention (only ~35 positions out of 1000)
  - Long-range dependencies require deep attention
  - Memory cost: O(n^2) where n includes all features x all days


APPROACH 3: MULTI-RATE STATE SPACE MODEL
========================================
Architecture:
  Latent State z(t) evolves at daily resolution
         |
    z(t+1) = f(z(t)) + noise   (Transition Model)
         |
  Daily Obs: y_daily(t) = g(z(t))    (when observed)
  Monthly Obs: y_monthly(m) = h(z(t_m))  (when observed)

Key Features:
  - Continuous latent state evolves at finest resolution
  - Observations are partial views of state (observed OR not)
  - Variational inference learns state from available observations
  - Natural uncertainty quantification

Pros:
  + Principled probabilistic framework
  + Natural handling of irregular observations
  + Provides uncertainty estimates for imputation
  + Can do principled imputation (not fabrication)

Cons:
  - More complex training (variational)
  - Sequential processing (hard to parallelize)
  - Requires careful tuning of KL weight


RECOMMENDATION
==============
For this use case (Ukraine conflict OSINT), I recommend:

PRIMARY: Approach 1 (Hierarchical Multi-Rate)
  - Best preserves daily signal fidelity
  - Aggregation is interpretable (which days matter for monthly prediction)
  - Practical to implement and train

SECONDARY: Approach 3 (State Space) for uncertainty-aware imputation
  - When you need uncertainty estimates
  - When downstream tasks need all features at daily resolution

AVOID: Approach 2 (OMAN) for this specific problem
  - Sparse monthly observations get overwhelmed by dense daily
  - Attention sparsity issues with 35/1000 observed positions


IMPLEMENTATION PRIORITY
=======================
1. Start with Hierarchical Multi-Rate (Approach 1)
2. Add State Space imputation head for uncertainty when needed
3. Use observation masking concepts from OMAN within the encoders
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_architecture_comparison()

    if HAS_TORCH:
        print("\n" + "=" * 80)
        print("MODEL PARAMETER COUNTS")
        print("=" * 80)

        # Daily sources
        daily_sources = {
            'equipment': 38,
            'personnel': 6,
            'deepstate': 55,
            'firms': 42,
            'viina': 24
        }

        # Monthly sources
        monthly_sources = {
            'sentinel': 43,
            'hdx_conflict': 18,
            'hdx_food': 20,
            'hdx_rainfall': 16,
            'iom': 18
        }

        # Approach 1
        model1 = HierarchicalMultiRateModel(daily_sources, monthly_sources)
        params1 = sum(p.numel() for p in model1.parameters())
        print(f"\nApproach 1 (Hierarchical Multi-Rate): {params1:,} parameters")

        # Approach 2
        model2 = OMAN(SOURCE_SPECS)
        params2 = sum(p.numel() for p in model2.parameters())
        print(f"Approach 2 (OMAN): {params2:,} parameters")

        # Approach 3
        model3 = MultiRateStateSpaceModel(SOURCE_SPECS)
        params3 = sum(p.numel() for p in model3.parameters())
        print(f"Approach 3 (State Space): {params3:,} parameters")
