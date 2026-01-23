"""
Phase 1: Joint Interpolation Models for Same-Source Features

This module implements temporal interpolation for features with resolution > 1 day,
using joint models that exploit correlations between features observed simultaneously.

Key insight: Features from the same observation (e.g., all Sentinel-2 bands from one pass)
share the same acquisition timestamps. A joint model can:
1. Learn band-to-band correlations
2. Share encoder parameters efficiently
3. Predict all bands together with consistent temporal context

Temporal Resolutions:
- Sentinel-1 SAR: ~6 day revisit
- Sentinel-2 Optical: ~5 day revisit
- Sentinel-3 Fire: ~1-2 day revisit
- Sentinel-5P Atmospheric: ~daily (already interpolated)
- DeepState: Variable (update-driven, often 1-7 days between changes)

Architecture per source:
- Shared encoder for all features in the observation
- Cross-feature attention to learn correlations
- Decoder predicts all features for intermediate days
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from datetime import datetime, timedelta
import math

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Install with: pip install torch")

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    UNIFIED_INTERP_MODEL, UNIFIED_DELTA_MODEL, UNIFIED_HYBRID_MODEL,
    get_interp_model_path,
)

# For backward compatibility, keep BASE_DIR as an alias to PROJECT_ROOT
BASE_DIR = PROJECT_ROOT

# Import real data loaders
try:
    from interpolation_data_loaders import (
        SentinelDataLoader,
        DeepStateDataLoader,
        EquipmentDataLoader,
        FIRMSDataLoader,
        UCDPDataLoader
    )
    HAS_REAL_DATA = True
except ImportError:
    HAS_REAL_DATA = False
    print("Warning: Real data loaders not available. Using synthetic data.")

# Import training utilities for proper scheduling and accumulation
try:
    from training_utils import WarmupCosineScheduler, GradientAccumulator
    from training_config import DataConfig
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    print("Warning: Training utilities not available. Using basic scheduler.")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InterpolationConfig:
    """Configuration for a joint interpolation model."""
    name: str
    source: str
    features: List[str]
    native_resolution_days: float  # Average days between observations
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    max_gap_days: int = 10  # Maximum gap to interpolate
    dropout: float = 0.1
    # Phase 2 hierarchy support
    parent_features: Optional[List[str]] = None  # Parent features this group derives from
    child_groups: Optional[List[str]] = None     # Child model names this can condition
    hierarchy_level: int = 0                      # 0=aggregate, 1=decomposed, 2=sub-decomposed
    conditioning_dim: int = 0                     # Dimension of external conditioning (0=none)


# =============================================================================
# PHASE 1 CONFIGURATIONS (Aggregate-level interpolation)
# =============================================================================
# These models interpolate aggregate features. In Phase 2, their outputs
# will condition child models that interpolate decomposed sub-features.

INTERPOLATION_CONFIGS = {
    # --- Sentinel Satellite (Level 0: Aggregate) ---
    'sentinel2': InterpolationConfig(
        name='Sentinel-2 Optical Bands',
        source='sentinel',
        features=[
            's2_b02_blue', 's2_b03_green', 's2_b04_red', 's2_b08_nir',
            's2_b11_swir1', 's2_b12_swir2', 's2_ndvi_mean', 's2_ndwi_mean',
            's2_nbr_mean', 's2_cloud_cover'
        ],
        native_resolution_days=5.0,
        hierarchy_level=0,
        child_groups=['sentinel2_indices']  # Phase 2: conditions derived indices
    ),
    'sentinel1': InterpolationConfig(
        name='Sentinel-1 SAR',
        source='sentinel',
        features=[
            's1_vv_mean', 's1_vh_mean', 's1_vv_vh_ratio',
            's1_change_intensity', 's1_coherence', 's1_backscatter_anom'
        ],
        native_resolution_days=6.0,
        hierarchy_level=0,
        child_groups=['sentinel1_change']  # Phase 2: conditions change detection
    ),
    'sentinel3': InterpolationConfig(
        name='Sentinel-3 Fire Products',
        source='sentinel',
        features=[
            's3_frp_count', 's3_frp_total', 's3_frp_max',
            's3_otci_mean', 's3_gifapar_mean'
        ],
        native_resolution_days=1.5,
        hierarchy_level=0
    ),
    's5p_gases': InterpolationConfig(
        name='Sentinel-5P Trace Gases',
        source='sentinel',
        features=[
            's5p_no2_mean', 's5p_no2_max', 's5p_co_mean',
            's5p_co_max', 's5p_aerosol_index'
        ],
        native_resolution_days=1.0,
        hierarchy_level=0
    ),

    # --- DeepState (Level 0: Aggregate) ---
    'deepstate': InterpolationConfig(
        name='DeepState Front Line Aggregates',
        source='deepstate',
        features=[
            'poly_occupied_area', 'poly_liberated_area', 'poly_contested_area',
            'front_line_length', 'arrows_total', 'units_total'
        ],
        native_resolution_days=2.5,
        hierarchy_level=0,
        child_groups=['deepstate_arrows', 'deepstate_units', 'deepstate_polygons']
    ),

    # --- Equipment (Level 0: Aggregate totals) ---
    'equipment_totals': InterpolationConfig(
        name='Equipment Loss Totals',
        source='equipment',
        features=[
            'aircraft_total', 'heli_total', 'tank_total', 'afv_total',
            'arty_total', 'drones_total', 'air_defense_total'
        ],
        native_resolution_days=1.0,  # Daily updates
        hierarchy_level=0,
        child_groups=['equipment_aircraft', 'equipment_tanks', 'equipment_afv']
    ),
}


# =============================================================================
# PHASE 2 CONFIGURATIONS (Decomposed sub-features)
# =============================================================================
# These will be added in Phase 2 implementation. They receive conditioning
# from their parent models defined above.

PHASE2_CONFIGS = {
    # --- DeepState Decomposed ---
    'deepstate_arrows': InterpolationConfig(
        name='DeepState Attack Directions',
        source='deepstate',
        features=[
            'arrows_north', 'arrows_east', 'arrows_south', 'arrows_west',
            'arrows_nne', 'arrows_ene', 'arrows_ese', 'arrows_sse',
            'arrows_ssw', 'arrows_wsw', 'arrows_wnw', 'arrows_nnw'
        ],
        native_resolution_days=2.5,
        hierarchy_level=1,
        parent_features=['arrows_total'],  # Conditioned by total
        conditioning_dim=64  # Receives d_model from parent
    ),
    'deepstate_units': InterpolationConfig(
        name='DeepState Military Units',
        source='deepstate',
        features=[
            'units_army', 'units_division', 'units_brigade', 'units_regiment',
            'units_battalion', 'units_motorized', 'units_tank', 'units_artillery',
            'units_airborne', 'units_recon', 'units_bars', 'units_akhmat'
        ],
        native_resolution_days=2.5,
        hierarchy_level=1,
        parent_features=['units_total'],
        conditioning_dim=64
    ),
    'deepstate_polygons': InterpolationConfig(
        name='DeepState Territory Status',
        source='deepstate',
        features=[
            'poly_occupied_count', 'poly_liberated_count', 'poly_contested_count',
            'poly_unknown_count', 'poly_newly_occupied', 'poly_newly_liberated'
        ],
        native_resolution_days=2.5,
        hierarchy_level=1,
        parent_features=['poly_occupied_area', 'poly_liberated_area', 'poly_contested_area'],
        conditioning_dim=64
    ),

    # --- Equipment Decomposed ---
    'equipment_tanks': InterpolationConfig(
        name='Tank Losses by Type',
        source='equipment',
        features=[
            'tank_t62', 'tank_t64', 'tank_t72', 'tank_t80', 'tank_t90', 'tank_other'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['tank_total'],
        conditioning_dim=64
    ),
    'equipment_afv': InterpolationConfig(
        name='AFV Losses by Type',
        source='equipment',
        features=[
            'afv_bmp', 'afv_btr', 'afv_mtlb', 'afv_bmd', 'afv_other'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['afv_total'],
        conditioning_dim=64
    ),
    'equipment_aircraft': InterpolationConfig(
        name='Aircraft Losses by Type',
        source='equipment',
        features=[
            'aircraft_sukhoi', 'aircraft_mig', 'aircraft_transport', 'aircraft_awacs'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['aircraft_total'],
        conditioning_dim=64
    ),

    # --- UCDP Decomposed (event-level, needs different handling) ---
    'ucdp_geography': InterpolationConfig(
        name='UCDP Events by Oblast',
        source='ucdp',
        features=[
            'geo_donetsk', 'geo_luhansk', 'geo_kharkiv', 'geo_kherson',
            'geo_zaporizhzhya', 'geo_sumy', 'geo_dnipropetrovsk', 'geo_mykolayiv',
            'geo_chernihiv', 'geo_kyiv_oblast', 'geo_kyiv_city', 'geo_odessa', 'geo_other'
        ],
        native_resolution_days=1.0,  # Event-level aggregated to daily
        hierarchy_level=1,
        parent_features=['total_events'],
        conditioning_dim=64
    ),

    # --- FIRMS Decomposed ---
    'firms_by_intensity': InterpolationConfig(
        name='FIRMS Fires by Intensity',
        source='firms',
        features=[
            'frp_tiny', 'frp_small', 'frp_medium', 'frp_large', 'frp_very_large', 'frp_extreme'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['frp_total'],
        conditioning_dim=64
    ),
    'firms_by_time': InterpolationConfig(
        name='FIRMS Fires by Time of Day',
        source='firms',
        features=[
            'frp_morning', 'frp_afternoon', 'frp_evening', 'frp_night_period'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['fire_count'],
        conditioning_dim=64
    ),
}


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class TemporalPositionalEncoding(nn.Module):
        """
        Positional encoding that handles irregular time intervals.

        Unlike standard positional encoding which assumes fixed intervals,
        this encodes the actual day offset from reference point.
        """
        def __init__(self, d_model: int, max_days: int = 365, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            self.d_model = d_model

            # Learnable day embedding (more flexible than sinusoidal for irregular data)
            self.day_embedding = nn.Embedding(max_days * 2, d_model)  # *2 for negative offsets
            self.max_days = max_days

            # Also include sinusoidal for smooth interpolation
            pe = torch.zeros(max_days * 2, d_model)
            position = torch.arange(0, max_days * 2, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe_sinusoidal', pe)

            # Combine learnable + sinusoidal
            self.combine = nn.Linear(d_model * 2, d_model)

        def forward(self, x: torch.Tensor, day_offsets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
                day_offsets: [batch, seq_len] - days from reference (can be fractional)
            """
            # Clamp and shift to valid embedding range
            day_idx = (day_offsets + self.max_days).long().clamp(0, self.max_days * 2 - 1)

            # Get learnable embedding
            learned_pe = self.day_embedding(day_idx)

            # Get sinusoidal (interpolate for fractional days)
            sin_pe = self.pe_sinusoidal[day_idx]

            # Combine both
            combined = torch.cat([learned_pe, sin_pe], dim=-1)
            pe = self.combine(combined)

            return self.dropout(x + pe)


    class CrossFeatureAttention(nn.Module):
        """
        Learns correlations between features observed at the same timestamp.

        For example, for Sentinel-2:
        - Blue and Green bands are highly correlated
        - NDVI depends on Red and NIR
        - Cloud cover affects all bands

        Phase 2 Extension: Supports optional conditioning from:
        - Parent features (e.g., tank_total conditions tank_t72, tank_t80)
        - Daily features (e.g., FIRMS daily fires condition satellite interpolation)
        """
        def __init__(
            self,
            num_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            conditioning_dim: int = 0  # Phase 2: external conditioning dimension
        ):
            super().__init__()
            self.num_features = num_features
            self.d_model = d_model
            self.conditioning_dim = conditioning_dim

            # Project each feature to d_model
            self.feature_projection = nn.Linear(1, d_model)

            # Learnable feature embeddings (like word embeddings for features)
            self.feature_embeddings = nn.Embedding(num_features, d_model)

            # Phase 2: Conditioning projection (if conditioning is provided)
            if conditioning_dim > 0:
                self.conditioning_projection = nn.Linear(conditioning_dim, d_model)
                # Cross-attention from features to conditioning
                self.conditioning_attention = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
            else:
                self.conditioning_projection = None
                self.conditioning_attention = None

            # Self-attention across features
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            conditioning: Optional[torch.Tensor] = None  # Phase 2: [batch, cond_seq, cond_dim]
        ) -> torch.Tensor:
            """
            Args:
                x: [batch, num_features] - feature values at one timestamp
                mask: [batch, num_features] - 1 for valid, 0 for missing
                conditioning: Optional [batch, cond_seq_len, conditioning_dim] - external context

            Returns:
                [batch, num_features, d_model] - encoded feature representations
            """
            batch_size = x.size(0)

            # Project each feature: [batch, num_features, d_model]
            x_proj = self.feature_projection(x.unsqueeze(-1))

            # Add feature-specific embeddings
            feat_indices = torch.arange(self.num_features, device=x.device)
            feat_emb = self.feature_embeddings(feat_indices)  # [num_features, d_model]
            x_proj = x_proj + feat_emb.unsqueeze(0)

            # Phase 2: Apply conditioning if provided
            if conditioning is not None and self.conditioning_projection is not None:
                # Project conditioning to d_model
                cond_proj = self.conditioning_projection(conditioning)  # [batch, cond_seq, d_model]
                # Cross-attend: features query conditioning context
                x_proj, _ = self.conditioning_attention(x_proj, cond_proj, cond_proj)

            # Attention mask (True = ignore)
            attn_mask = None
            if mask is not None:
                attn_mask = (mask == 0)

            # Cross-feature attention
            encoded = self.transformer(x_proj, src_key_padding_mask=attn_mask)

            return encoded


    class GapInterpolator(nn.Module):
        """
        Interpolates feature values for days between observations.

        Takes encoded representations from before and after the gap,
        and predicts values for intermediate days using attention
        over the temporal context.
        """
        def __init__(
            self,
            num_features: int,
            d_model: int = 64,
            nhead: int = 4,
            max_gap_days: int = 10,
            dropout: float = 0.1
        ):
            super().__init__()
            self.num_features = num_features
            self.d_model = d_model
            self.max_gap_days = max_gap_days

            # Query embeddings for each day in the gap
            self.day_queries = nn.Embedding(max_gap_days, d_model)

            # Cross-attention: query days attend to observed timestamps
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )

            # Feature-wise decoder
            self.decoder = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, 1)
            )

            # Residual connection with linear interpolation
            self.blend_weight = nn.Parameter(torch.tensor(0.5))

        def forward(
            self,
            encoded_before: torch.Tensor,  # [batch, num_features, d_model]
            encoded_after: torch.Tensor,   # [batch, num_features, d_model]
            values_before: torch.Tensor,   # [batch, num_features]
            values_after: torch.Tensor,    # [batch, num_features]
            days_since_before: int,        # How many days after 'before'
            total_gap_days: int            # Total days in gap
        ) -> torch.Tensor:
            """
            Predict feature values for a specific day in the gap.

            Returns:
                [batch, num_features] - predicted feature values
            """
            batch_size = encoded_before.size(0)

            # Linear interpolation as baseline
            # Protect against division by zero (when before and after are same day)
            if total_gap_days <= 0:
                alpha = 0.5  # Midpoint if gap is zero
            else:
                alpha = days_since_before / total_gap_days
            linear_interp = (1 - alpha) * values_before + alpha * values_after

            # Query embedding for this day position
            day_idx = min(days_since_before, self.max_gap_days - 1)
            query = self.day_queries.weight[day_idx:day_idx+1]  # [1, d_model]
            query = query.unsqueeze(0).expand(batch_size, self.num_features, -1)

            # Key/Value from before and after observations
            # [batch, 2*num_features, d_model]
            kv = torch.cat([encoded_before, encoded_after], dim=1)

            # Cross-attention
            attended, _ = self.cross_attention(query, kv, kv)  # [batch, num_features, d_model]

            # Decode to feature values
            neural_pred = self.decoder(attended).squeeze(-1)  # [batch, num_features]

            # Blend neural prediction with linear interpolation
            # (allows model to learn when linear is sufficient vs when it needs adjustment)
            blend = torch.sigmoid(self.blend_weight)
            output = blend * neural_pred + (1 - blend) * linear_interp

            return output


    class JointInterpolationModel(nn.Module):
        """
        Complete joint interpolation model for a single source.

        Combines:
        1. Cross-feature attention to learn correlations
        2. Temporal context encoding
        3. Gap interpolation with attention over context

        Phase 2 Extension: Supports hierarchical conditioning where:
        - Parent model outputs condition child model interpolation
        - Daily features (FIRMS, UCDP) condition gapped satellite interpolation
        """
        def __init__(self, config: InterpolationConfig):
            super().__init__()
            self.config = config
            self.num_features = len(config.features)

            # Cross-feature encoder (with optional conditioning for Phase 2)
            self.cross_feature_attn = CrossFeatureAttention(
                num_features=self.num_features,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dropout=config.dropout,
                conditioning_dim=config.conditioning_dim  # Phase 2 hook
            )

            # Temporal position encoding
            self.temporal_encoding = TemporalPositionalEncoding(
                d_model=config.d_model,
                max_days=365,
                dropout=config.dropout
            )

            # Gap interpolator
            self.gap_interpolator = GapInterpolator(
                num_features=self.num_features,
                d_model=config.d_model,
                nhead=config.nhead,
                max_gap_days=config.max_gap_days,
                dropout=config.dropout
            )

            # Uncertainty estimation head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Linear(config.d_model // 2, 1),
                nn.Softplus()  # Positive uncertainty
            )

            # Phase 2: Output projection for conditioning child models
            # This produces embeddings that can condition downstream interpolators
            self.output_for_conditioning = nn.Linear(config.d_model, config.d_model)

        def encode_observation(
            self,
            features: torch.Tensor,  # [batch, num_features]
            day_offset: torch.Tensor,  # [batch, 1]
            mask: Optional[torch.Tensor] = None,
            conditioning: Optional[torch.Tensor] = None  # Phase 2: external conditioning
        ) -> torch.Tensor:
            """
            Encode a single observation with its temporal position.

            Args:
                features: [batch, num_features] - feature values
                day_offset: [batch, 1] - temporal position
                mask: [batch, num_features] - validity mask
                conditioning: [batch, cond_seq, cond_dim] - external context (Phase 2)

            Returns:
                [batch, num_features, d_model]
            """
            # Cross-feature encoding (with optional conditioning)
            encoded = self.cross_feature_attn(features, mask, conditioning)

            # Add temporal position (broadcast across features)
            # Squeeze to [batch] if needed (handles both [batch, 1] and [batch, 1, 1])
            day_offset_flat = day_offset.view(day_offset.size(0))  # [batch]
            day_offset_expanded = day_offset_flat.unsqueeze(1).expand(-1, self.num_features)  # [batch, num_features]
            encoded = self.temporal_encoding(encoded, day_offset_expanded)

            return encoded

        def forward(
            self,
            obs_before: torch.Tensor,     # [batch, num_features]
            obs_after: torch.Tensor,      # [batch, num_features]
            day_before: torch.Tensor,     # [batch, 1] - day offset of before obs
            day_after: torch.Tensor,      # [batch, 1] - day offset of after obs
            target_day: torch.Tensor,     # [batch, 1] - day to predict
            mask_before: Optional[torch.Tensor] = None,
            mask_after: Optional[torch.Tensor] = None,
            conditioning: Optional[torch.Tensor] = None,  # Phase 2: [batch, cond_seq, cond_dim]
            return_embedding: bool = False  # Phase 2: return embedding for child conditioning
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """
            Predict feature values for target day given surrounding observations.

            Args:
                obs_before, obs_after: Feature values at surrounding observations
                day_before, day_after: Day offsets of observations
                target_day: Day to predict
                mask_before, mask_after: Validity masks
                conditioning: External conditioning context (Phase 2)
                return_embedding: If True, also return embedding for child conditioning

            Returns:
                predictions: [batch, num_features]
                uncertainties: [batch, num_features]
                embedding: [batch, d_model] (only if return_embedding=True)
            """
            # Encode observations (with optional conditioning)
            enc_before = self.encode_observation(obs_before, day_before, mask_before, conditioning)
            enc_after = self.encode_observation(obs_after, day_after, mask_after, conditioning)

            # Calculate gap parameters
            days_since_before = (target_day - day_before).squeeze(-1).mean().int().item()
            total_gap = (day_after - day_before).squeeze(-1).mean().int().item()

            # Interpolate
            predictions = self.gap_interpolator(
                enc_before, enc_after,
                obs_before, obs_after,
                days_since_before, total_gap
            )

            # Estimate uncertainty (average of before/after encodings)
            avg_encoding = (enc_before + enc_after) / 2
            uncertainties = self.uncertainty_head(avg_encoding).squeeze(-1)

            if return_embedding:
                # Phase 2: Produce embedding for conditioning child models
                # Pool across features and project
                pooled = avg_encoding.mean(dim=1)  # [batch, d_model]
                child_conditioning = self.output_for_conditioning(pooled)  # [batch, d_model]
                return predictions, uncertainties, child_conditioning

            return predictions, uncertainties


    class InterpolationDataset(Dataset):
        """
        Dataset for training interpolation models.

        Creates training samples by:
        1. Loading actual observation timestamps
        2. For each gap > 1 day, creating samples for intermediate days
        3. Holding out some observations as targets

        CRITICAL FIX: Uses temporal split (not random) to prevent data leakage.
        Training uses earlier dates, validation uses later dates with configurable gap.
        """
        def __init__(
            self,
            config: InterpolationConfig,
            data_path: Path,
            train: bool = True,
            val_ratio: float = 0.2,
            temporal_gap: int = 7,
            norm_stats: Optional[Dict[str, np.ndarray]] = None
        ):
            """
            Initialize the interpolation dataset.

            Args:
                config: Interpolation configuration
                data_path: Path to data directory
                train: If True, return training samples; if False, return validation samples
                val_ratio: Proportion of data to use for validation
                temporal_gap: Number of days gap between train and validation sets
                              to prevent temporal leakage (default 7 for daily data)
                norm_stats: Pre-computed normalization statistics from training data.
                           If None and train=True, will compute from this data.
                           If None and train=False, will compute from all data (not recommended).
            """
            self.config = config
            self.num_features = len(config.features)
            self.train = train
            self.temporal_gap = temporal_gap
            self.norm_stats = norm_stats

            # Load data (before normalization)
            self._load_data(data_path)

            # Apply normalization - compute stats only from training data
            if self.norm_stats is None:
                # This path is for backwards compatibility or when training set
                # If this is training data, we compute and store stats
                # If validation and no stats provided, we compute from all data (with warning)
                if not train:
                    print("  WARNING: Validation dataset created without pre-computed norm_stats.")
                    print("           This may cause data leakage. Pass norm_stats from training set.")
            else:
                # Apply pre-computed normalization stats
                self.apply_norm_stats(self.norm_stats)

            # Create interpolation samples with temporal split
            self._create_samples(val_ratio)

        def _load_data(self, data_path: Path):
            """Load observation data with timestamps from real data loaders."""
            if HAS_REAL_DATA:
                self._load_real_data(data_path)
            else:
                self._load_synthetic_data()

        def _load_real_data(self, data_path: Path):
            """Load from actual data sources."""
            source = self.config.source
            config_name = self.config.name.lower()

            # Map source types to data loaders
            try:
                if source == 'sentinel':
                    loader = SentinelDataLoader().load().process()
                    # Use daily observations for better interpolation training
                    data, dates = loader.get_daily_observations()
                    feature_names = loader.feature_names
                elif source == 'deepstate':
                    loader = DeepStateDataLoader().load().process()
                    data = loader.processed_data
                    dates = loader.dates
                    feature_names = loader.feature_names
                elif source == 'equipment':
                    loader = EquipmentDataLoader().load().process()
                    # Use daily changes for equipment (captures daily loss variation)
                    data, dates = loader.get_daily_changes()
                    feature_names = loader.feature_names
                elif source == 'firms':
                    loader = FIRMSDataLoader().load().process()
                    data = loader.processed_data
                    dates = loader.dates
                    feature_names = loader.feature_names
                elif source == 'ucdp':
                    loader = UCDPDataLoader().load().process()
                    data = loader.processed_data
                    dates = loader.dates
                    feature_names = loader.feature_names
                else:
                    print(f"Unknown source '{source}', falling back to synthetic")
                    self._load_synthetic_data()
                    return
            except Exception as e:
                print(f"Error loading real data for {source}: {e}")
                print("Falling back to synthetic data")
                self._load_synthetic_data()
                return

            # Convert dates to day offsets from first date
            reference_date = datetime.strptime(dates[0], '%Y-%m-%d') if '-' in dates[0] else datetime.strptime(dates[0], '%Y-%m')
            day_offsets = []
            for d in dates:
                if '-' in d and len(d) == 10:  # YYYY-MM-DD
                    dt = datetime.strptime(d, '%Y-%m-%d')
                else:  # YYYY-MM
                    dt = datetime.strptime(d, '%Y-%m')
                day_offsets.append((dt - reference_date).days)

            # Map config features to available features
            # By default, use ALL available features (not just config-specified)
            # This maximizes data utilization for training
            n_obs = len(dates)
            n_config_features = len(self.config.features)
            n_available = len(feature_names)

            # Use all available features for training (more data = better generalization)
            # The model will be created to match this feature count
            n_use = n_available
            observations = data[:, :n_use].copy()

            # Store actual feature count for model creation
            self.actual_num_features = n_use
            self.actual_feature_names = feature_names[:n_use]

            # Handle NaN values
            observations = np.nan_to_num(observations, nan=0.0)

            # Store raw observations and day offsets for normalization computation
            self._raw_observations = observations.copy()
            self.observation_days = torch.tensor(day_offsets, dtype=torch.float32)

            # Normalize: if norm_stats not provided, compute from this data
            # (will be overwritten by apply_norm_stats if called later)
            if self.norm_stats is None:
                # Compute normalization from this data (training set should call compute_norm_stats later)
                self._computed_norm_stats = self._compute_norm_stats_internal(observations)
                observations = self._apply_norm_stats_internal(observations, self._computed_norm_stats)
            else:
                # Apply pre-computed stats
                observations = self._apply_norm_stats_internal(observations, self.norm_stats)

            self.observations = torch.tensor(observations, dtype=torch.float32)

            print(f"  Loaded {n_obs} observations from real {source} data")
            print(f"  Available features: {n_available}, Using: {n_use}, Config expects: {n_config_features}")

        def _load_synthetic_data(self):
            """Generate synthetic data for testing."""
            # Simulate observations over 2 years
            np.random.seed(42)
            n_obs = int(730 / self.config.native_resolution_days)  # ~2 years

            # Observation timestamps (with some jitter)
            base_days = np.cumsum(
                self.config.native_resolution_days +
                np.random.uniform(-1, 1, n_obs)
            ).astype(int)

            # Feature values (with realistic correlations)
            # Base signal with trend and seasonality
            t = base_days / 365
            trend = 0.1 * t
            seasonality = 0.3 * np.sin(2 * np.pi * t)

            # Correlated features
            base = trend + seasonality + np.random.randn(n_obs) * 0.1
            observations = np.zeros((n_obs, self.num_features))

            for i in range(self.num_features):
                correlation = 0.8 ** i  # Decreasing correlation
                noise = np.random.randn(n_obs) * (1 - correlation)
                observations[:, i] = base * correlation + noise

            # Store raw observations for normalization computation
            self._raw_observations = observations.copy()
            self.observation_days = torch.tensor(base_days, dtype=torch.float32)

            # Normalize: if norm_stats not provided, compute from this data
            if self.norm_stats is None:
                self._computed_norm_stats = self._compute_norm_stats_internal(observations)
                observations = self._apply_norm_stats_internal(observations, self._computed_norm_stats)
            else:
                observations = self._apply_norm_stats_internal(observations, self.norm_stats)

            self.observations = torch.tensor(observations, dtype=torch.float32)

        def _compute_norm_stats_internal(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
            """
            Compute normalization statistics (min, max) from observations.

            Args:
                observations: Raw observation array of shape (n_obs, n_features)

            Returns:
                Dictionary with 'min' and 'max' arrays for each feature
            """
            return {
                'min': observations.min(axis=0),
                'max': observations.max(axis=0)
            }

        def _apply_norm_stats_internal(
            self,
            observations: np.ndarray,
            stats: Dict[str, np.ndarray]
        ) -> np.ndarray:
            """
            Apply normalization statistics to observations.

            Args:
                observations: Raw observation array of shape (n_obs, n_features)
                stats: Dictionary with 'min' and 'max' arrays

            Returns:
                Normalized observations in [0, 1] range
            """
            normalized = observations.copy()
            col_min = stats['min']
            col_max = stats['max']
            range_vals = col_max - col_min

            # Avoid division by zero for constant features
            range_vals = np.where(range_vals > 0, range_vals, 1.0)

            normalized = (observations - col_min) / range_vals
            return normalized

        def compute_norm_stats(self, train_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
            """
            Compute normalization statistics from training data only.

            This method should be called on the training dataset to get normalization
            statistics that can then be passed to validation/test datasets.

            Args:
                train_indices: Optional list of observation indices to use for computing stats.
                              If None, uses all observations (for training set this is OK).

            Returns:
                Dictionary with 'min' and 'max' arrays for each feature
            """
            if hasattr(self, '_raw_observations'):
                raw_obs = self._raw_observations
            else:
                # Fall back to observations tensor (already normalized, less ideal)
                raw_obs = self.observations.numpy()

            if train_indices is not None:
                raw_obs = raw_obs[train_indices]

            stats = self._compute_norm_stats_internal(raw_obs)
            self._computed_norm_stats = stats
            return stats

        def apply_norm_stats(self, stats: Dict[str, np.ndarray]) -> None:
            """
            Apply pre-computed normalization statistics to this dataset.

            This method should be called on validation/test datasets with stats
            computed from the training dataset.

            Args:
                stats: Dictionary with 'min' and 'max' arrays from training data
            """
            self.norm_stats = stats
            if hasattr(self, '_raw_observations'):
                normalized = self._apply_norm_stats_internal(self._raw_observations, stats)
                self.observations = torch.tensor(normalized, dtype=torch.float32)

        def get_norm_stats(self) -> Optional[Dict[str, np.ndarray]]:
            """
            Get the normalization statistics used by this dataset.

            Returns:
                Dictionary with 'min' and 'max' arrays, or None if not computed
            """
            if hasattr(self, '_computed_norm_stats'):
                return self._computed_norm_stats
            return self.norm_stats

        def _create_samples(self, val_ratio: float):
            """
            Create training/validation samples from observations.

            CRITICAL FIX: Uses temporal split instead of random shuffle to prevent
            data leakage. Training samples use earlier dates, validation samples
            use later dates with a configurable temporal gap between them.

            Uses multi-scale sampling: not just consecutive triplets, but also
            wider context windows to improve generalization to various gap sizes.
            """
            n_obs = len(self.observation_days)
            all_samples = []

            # Multi-scale context: use different skip patterns
            # skip=1: consecutive (i, i+1, i+2)
            # skip=2: wider context (i, i+2, i+4) - learns longer-range patterns
            max_skip = min(3, (n_obs - 1) // 2)  # Don't exceed data bounds

            for skip in range(1, max_skip + 1):
                for i in range(n_obs - 2 * skip):
                    before_idx = i
                    target_idx = i + skip
                    after_idx = i + 2 * skip

                    day_before = self.observation_days[before_idx]
                    day_target = self.observation_days[target_idx]
                    day_after = self.observation_days[after_idx]

                    gap1 = (day_target - day_before).item()
                    gap2 = (day_after - day_target).item()
                    total_gap = (day_after - day_before).item()

                    # Only use if total gap is within interpolation range
                    if total_gap <= self.config.max_gap_days * 2:
                        all_samples.append({
                            'obs_before': self.observations[before_idx],
                            'obs_after': self.observations[after_idx],
                            'target': self.observations[target_idx],
                            'day_before': day_before,
                            'day_after': day_after,
                            'day_target': day_target,
                            'gap_ratio': gap1 / max(total_gap, 1)  # Position in gap (0-1)
                        })

            # CRITICAL FIX: Temporal split instead of random shuffle
            # Sort samples by target date to ensure chronological ordering
            all_samples.sort(key=lambda x: x['day_target'].item())

            n_samples = len(all_samples)

            # Find the split point based on temporal ordering
            # Training uses first (1 - val_ratio) of chronologically sorted samples
            # Validation uses last val_ratio of samples, with a temporal gap
            n_train = int(n_samples * (1 - val_ratio))

            if n_train > 0 and n_samples > n_train:
                # Find the day at the split point
                train_end_day = all_samples[n_train - 1]['day_target'].item()

                # Find the first validation sample that is at least temporal_gap days
                # after the last training sample
                val_start_idx = n_train
                for idx in range(n_train, n_samples):
                    sample_day = all_samples[idx]['day_target'].item()
                    if sample_day >= train_end_day + self.temporal_gap:
                        val_start_idx = idx
                        break

                if self.train:
                    # Training set: all samples before the split
                    self.samples = all_samples[:n_train]
                    # Shuffle training samples for better gradient updates
                    # (but only within training set, preserving train/val separation)
                    np.random.seed(42)
                    indices = np.random.permutation(len(self.samples))
                    self.samples = [self.samples[i] for i in indices]
                else:
                    # Validation set: samples after the gap
                    self.samples = all_samples[val_start_idx:]
            else:
                # Edge case: not enough samples
                if self.train:
                    self.samples = all_samples
                else:
                    self.samples = []

            # Log split information
            if len(all_samples) > 0:
                min_day = min(s['day_target'].item() for s in all_samples)
                max_day = max(s['day_target'].item() for s in all_samples)
                if len(self.samples) > 0:
                    split_min = min(s['day_target'].item() for s in self.samples)
                    split_max = max(s['day_target'].item() for s in self.samples)
                    split_type = "TRAIN" if self.train else "VAL"
                    print(f"  {split_type} split: {len(self.samples)} samples, "
                          f"days [{split_min:.0f}, {split_max:.0f}] "
                          f"(full range: [{min_day:.0f}, {max_day:.0f}])")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            return (
                sample['obs_before'],
                sample['obs_after'],
                sample['day_before'].unsqueeze(0),
                sample['day_after'].unsqueeze(0),
                sample['day_target'].unsqueeze(0),
                sample['target']
            )


    class InterpolationTrainer:
        """
        Trainer for joint interpolation models.

        Improvements for better generalization:
        1. Uncertainty-weighted loss with minimum floor (prevents collapse)
        2. Temporal smoothness regularization
        3. Early stopping with patience
        4. Gradient clipping
        5. WarmupCosineScheduler for better learning rate scheduling
        6. Optional gradient accumulation for larger effective batch sizes
        """

        def __init__(
            self,
            model: JointInterpolationModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            min_uncertainty: float = 0.05,
            smoothness_weight: float = 0.1,
            device: str = 'cpu',
            warmup_epochs: int = 10,
            total_epochs: int = 100,
            accumulation_steps: int = 1,
            use_warmup_cosine: bool = True
        ):
            self.model = model.to(device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.min_uncertainty = min_uncertainty
            self.smoothness_weight = smoothness_weight
            self.accumulation_steps = accumulation_steps
            self.use_warmup_cosine = use_warmup_cosine and HAS_TRAINING_UTILS

            # AdamW with weight decay for regularization
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Use WarmupCosineScheduler if available, otherwise fall back to ReduceLROnPlateau
            if self.use_warmup_cosine:
                self.scheduler = WarmupCosineScheduler(
                    self.optimizer,
                    warmup_epochs=warmup_epochs,
                    total_epochs=total_epochs,
                    warmup_start_lr=lr * 0.01,  # Start at 1% of base LR
                    min_lr=1e-7
                )
                self.scheduler_type = 'warmup_cosine'
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
                )
                self.scheduler_type = 'reduce_on_plateau'

            # Setup gradient accumulator if available and needed
            if HAS_TRAINING_UTILS and accumulation_steps > 1:
                self.grad_accumulator = GradientAccumulator(
                    self.optimizer,
                    accumulation_steps=accumulation_steps,
                    max_grad_norm=1.0
                )
            else:
                self.grad_accumulator = None

        def train_epoch(self) -> dict:
            """Train for one epoch with optional gradient accumulation."""
            self.model.train()
            total_nll = 0
            total_smooth = 0
            total_loss = 0

            # Reset gradient accumulator if using it
            if self.grad_accumulator is not None:
                self.grad_accumulator.zero_grad()
            else:
                self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                (obs_before, obs_after, day_before, day_after,
                 day_target, target) = [x.to(self.device) for x in batch]

                predictions, uncertainties = self.model(
                    obs_before, obs_after,
                    day_before, day_after, day_target
                )

                # Gaussian NLL loss with learned uncertainty
                nll_loss = self._gaussian_nll_loss(predictions, target, uncertainties)

                # Smoothness regularization
                smooth_loss = self._smoothness_loss(predictions, obs_before, obs_after)

                # Combined loss
                loss = nll_loss + self.smoothness_weight * smooth_loss

                # Scale loss for gradient accumulation
                if self.grad_accumulator is not None:
                    scaled_loss = loss / self.accumulation_steps
                    scaled_loss.backward()
                    # Step will be performed by accumulator when ready
                    self.grad_accumulator.step(loss)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_nll += nll_loss.item()
                total_smooth += smooth_loss.item()
                total_loss += loss.item()

            n = len(self.train_loader)
            return {
                'total': total_loss / n,
                'nll': total_nll / n,
                'smooth': total_smooth / n
            }

        def validate(self) -> Tuple[float, float]:
            """Validate model."""
            self.model.eval()
            total_loss = 0
            total_mae = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    (obs_before, obs_after, day_before, day_after,
                     day_target, target) = [x.to(self.device) for x in batch]

                    predictions, uncertainties = self.model(
                        obs_before, obs_after,
                        day_before, day_after, day_target
                    )

                    loss = self._gaussian_nll_loss(predictions, target, uncertainties)
                    mae = F.l1_loss(predictions, target)

                    total_loss += loss.item()
                    total_mae += mae.item()

            avg_loss = total_loss / len(self.val_loader)
            avg_mae = total_mae / len(self.val_loader)

            # Only step ReduceLROnPlateau here (WarmupCosine steps per epoch in train())
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(avg_loss)

            return avg_loss, avg_mae

        def _gaussian_nll_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            uncertainty: torch.Tensor
        ) -> torch.Tensor:
            """
            Gaussian negative log-likelihood with learned variance.

            Includes minimum uncertainty floor to prevent variance collapse.
            """
            # Clamp uncertainty to minimum floor (prevents collapse to zero variance)
            uncertainty_clamped = torch.clamp(uncertainty, min=self.min_uncertainty)
            variance = uncertainty_clamped ** 2 + 1e-6

            # NLL = 0.5 * (log(variance) + (pred - target)^2 / variance)
            nll = 0.5 * (torch.log(variance) + (pred - target) ** 2 / variance)

            return nll.mean()

        def _smoothness_loss(
            self,
            pred: torch.Tensor,
            obs_before: torch.Tensor,
            obs_after: torch.Tensor
        ) -> torch.Tensor:
            """
            Encourage predictions to be between before and after values.

            This is a soft constraint - interpolated values should generally
            fall between the boundary observations for most features.
            """
            # How much pred exceeds the [min, max] range of before/after
            lower = torch.min(obs_before, obs_after)
            upper = torch.max(obs_before, obs_after)

            # Penalize predictions outside the range (with soft margin)
            margin = 0.1  # Allow 10% overshoot
            range_size = (upper - lower).clamp(min=0.01)

            below_lower = F.relu(lower - pred - margin * range_size)
            above_upper = F.relu(pred - upper - margin * range_size)

            return (below_lower + above_upper).mean()

        def train(
            self,
            epochs: int = 100,
            patience: int = 20,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """
            Full training loop with early stopping.

            Args:
                epochs: Maximum epochs to train
                patience: Stop if no improvement for this many epochs
                verbose: Print progress
            """
            history = {
                'train_loss': [], 'val_loss': [], 'val_mae': [],
                'train_nll': [], 'train_smooth': [], 'learning_rate': []
            }
            best_val_mae = float('inf')
            best_epoch = 0
            patience_counter = 0

            for epoch in range(epochs):
                train_metrics = self.train_epoch()
                val_loss, val_mae = self.validate()

                # Get current learning rate for logging
                current_lr = self.optimizer.param_groups[0]['lr']

                history['train_loss'].append(train_metrics['total'])
                history['train_nll'].append(train_metrics['nll'])
                history['train_smooth'].append(train_metrics['smooth'])
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                history['learning_rate'].append(current_lr)

                # Track best by MAE (more interpretable than NLL)
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model - sanitize name for filesystem
                    safe_name = self.model.config.name.replace(' ', '_').replace('/', '_').lower()
                    INTERP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        INTERP_MODEL_DIR / f"interp_{safe_name}_best.pt"
                    )
                else:
                    patience_counter += 1

                # Step scheduler based on type
                if self.scheduler_type == 'warmup_cosine':
                    # WarmupCosineScheduler steps per epoch, not based on metrics
                    self.scheduler.step()
                # Note: ReduceLROnPlateau is stepped in validate()

                if verbose and epoch % 10 == 0:
                    marker = '*' if epoch == best_epoch else ''
                    print(f"Epoch {epoch:3d}: loss={train_metrics['total']:.4f}, "
                          f"val_mae={val_mae:.4f}, lr={current_lr:.2e} {marker}")

                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best={best_epoch}, mae={best_val_mae:.4f})")
                    break

            return history


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_all_interpolation_models() -> Dict[str, JointInterpolationModel]:
    """Create all interpolation models defined in INTERPOLATION_CONFIGS."""
    models = {}
    for name, config in INTERPOLATION_CONFIGS.items():
        models[name] = JointInterpolationModel(config)
    return models


def train_all_models(
    epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
    temporal_gap: int = 7,
    warmup_epochs: int = 10
):
    """
    Train all interpolation models with proper temporal split and normalization.

    CRITICAL FIXES APPLIED:
    1. Temporal split instead of random shuffle (prevents future data leakage)
    2. Normalization stats computed from training data only (prevents global leakage)
    3. Configurable temporal gap between train/val sets
    4. WarmupCosineScheduler for better learning rate scheduling

    Args:
        epochs: Maximum training epochs
        batch_size: Batch size for training
        verbose: Print training progress
        temporal_gap: Days gap between train and validation sets (default 7)
        warmup_epochs: Number of warmup epochs for learning rate scheduler
    """
    if not HAS_TORCH:
        print("PyTorch required for training")
        return

    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)
    print(f"Temporal gap: {temporal_gap} days (prevents train/val leakage)")
    print(f"Normalization: Computed from TRAINING data only")
    if HAS_TRAINING_UTILS:
        print(f"Scheduler: WarmupCosineScheduler (warmup={warmup_epochs} epochs)")
    else:
        print("Scheduler: ReduceLROnPlateau (training_utils not available)")
    if HAS_REAL_DATA:
        print("Data source: REAL data from interpolation_data_loaders")
    else:
        print("Data source: SYNTHETIC data (real data loaders not available)")
    print("=" * 70)

    results = {}

    for name, config in INTERPOLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training: {config.name}")
        print(f"Config features: {len(config.features)}")
        print(f"Native resolution: {config.native_resolution_days} days")
        print(f"{'='*60}")

        # CRITICAL FIX: Create training dataset first, compute norm_stats
        print("  Creating training dataset...")
        train_dataset = InterpolationDataset(
            config, DATA_DIR,
            train=True,
            temporal_gap=temporal_gap,
            norm_stats=None  # Will compute internally
        )

        # Compute normalization statistics from training data only
        norm_stats = train_dataset.compute_norm_stats()
        print(f"  Computed normalization stats from {len(train_dataset)} training samples")

        # CRITICAL FIX: Create validation dataset with pre-computed norm_stats
        print("  Creating validation dataset with training norm_stats...")
        val_dataset = InterpolationDataset(
            config, DATA_DIR,
            train=False,
            temporal_gap=temporal_gap,
            norm_stats=norm_stats  # Apply training stats to prevent leakage
        )

        # Get actual feature count from loaded data
        actual_n_features = getattr(train_dataset, 'actual_num_features', len(config.features))
        actual_feature_names = getattr(train_dataset, 'actual_feature_names', config.features)

        print(f"  Actual features loaded: {actual_n_features}")

        # Create updated config with actual features if different
        if actual_n_features != len(config.features):
            actual_config = InterpolationConfig(
                name=config.name,
                source=config.source,
                features=actual_feature_names,
                native_resolution_days=config.native_resolution_days,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                max_gap_days=config.max_gap_days,
                dropout=config.dropout,
                parent_features=config.parent_features,
                child_groups=config.child_groups,
                hierarchy_level=config.hierarchy_level,
                conditioning_dim=config.conditioning_dim,
            )
        else:
            actual_config = config

        # Create model with correct feature count
        model = JointInterpolationModel(actual_config)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Skip if not enough samples
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"  SKIPPING: Not enough samples for training/validation")
            continue

        # Create loaders (shuffle=True for training is OK since temporal split already done)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Train with improved settings including WarmupCosineScheduler
        trainer = InterpolationTrainer(
            model, train_loader, val_loader,
            lr=1e-4,
            weight_decay=0.01,
            min_uncertainty=0.05,
            smoothness_weight=0.1,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            use_warmup_cosine=True
        )
        history = trainer.train(epochs=epochs, patience=20, verbose=verbose)

        best_mae = min(history['val_mae']) if history['val_mae'] else float('inf')
        results[name] = {
            'config': actual_config,  # Use actual config with correct feature count
            'history': history,
            'final_val_mae': history['val_mae'][-1] if history['val_mae'] else float('inf'),
            'best_val_mae': best_mae,
            'norm_stats': norm_stats  # Store for inference
        }
        print(f"  Best MAE: {best_mae:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for name, res in results.items():
        print(f"  {name}: best_mae={res['best_val_mae']:.4f}")

    return results


def print_model_summary():
    """Print summary of all interpolation models."""
    print("=" * 80)
    print("PHASE 1: JOINT INTERPOLATION MODELS")
    print("=" * 80)

    total_params = 0

    for name, config in INTERPOLATION_CONFIGS.items():
        print(f"\n{config.name}")
        print("-" * 40)
        print(f"  Source: {config.source}")
        print(f"  Features: {len(config.features)}")
        print(f"  Native resolution: {config.native_resolution_days} days")
        print(f"  Max interpolation gap: {config.max_gap_days} days")
        print(f"  Features:")
        for feat in config.features[:5]:
            print(f"    - {feat}")
        if len(config.features) > 5:
            print(f"    ... and {len(config.features) - 5} more")

        if HAS_TORCH:
            model = JointInterpolationModel(config)
            n_params = sum(p.numel() for p in model.parameters())
            total_params += n_params
            print(f"  Parameters: {n_params:,}")

    if HAS_TORCH:
        print(f"\n{'='*80}")
        print(f"TOTAL PARAMETERS (all models): {total_params:,}")
        print(f"{'='*80}")

    print("""
ARCHITECTURE PER MODEL:
=======================

┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Two observations                       │
│         [Obs Before: day T-n]     [Obs After: day T+m]          │
│         features: [f1, f2, ..., fN]  features: [f1, f2, ..., fN] │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CROSS-FEATURE ATTENTION                         │
│                                                                  │
│   Each observation independently:                                │
│   - Project features to d_model                                  │
│   - Add learnable feature embeddings                             │
│   - Self-attention across features                               │
│   - Learn: "Blue correlates with Green", "NDVI = f(Red, NIR)"   │
│                                                                  │
│   Output: [batch, num_features, d_model] per observation        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               TEMPORAL POSITION ENCODING                         │
│                                                                  │
│   - Encode day offset (handles irregular intervals)              │
│   - Combines learnable + sinusoidal encodings                    │
│   - Allows model to learn temporal dynamics                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GAP INTERPOLATOR                              │
│                                                                  │
│   For target day T:                                              │
│   1. Create query embedding for day position in gap              │
│   2. Cross-attend to encoded observations (before + after)       │
│   3. Decode to feature predictions                               │
│   4. Blend with linear interpolation (learnable weight)          │
│                                                                  │
│   Key insight: When gap is small, linear may suffice            │
│               When gap is large, need neural adjustment          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                  │
│                                                                  │
│   predictions: [batch, num_features]  - predicted values         │
│   uncertainties: [batch, num_features] - prediction uncertainty  │
│                                                                  │
│   Loss: Gaussian NLL with learned variance                       │
│         (model learns when predictions are uncertain)            │
└─────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_model_summary()

    if HAS_TORCH:
        print("\n" + "=" * 80)
        print("TRAINING ALL MODELS (synthetic data)")
        print("=" * 80)

        # Quick training for architecture verification
        results = train_all_models(epochs=50, batch_size=16, verbose=True)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - RESULTS")
        print("=" * 80)
        for name, result in results.items():
            print(f"  {name}: final_val_mae = {result['final_val_mae']:.4f}")
