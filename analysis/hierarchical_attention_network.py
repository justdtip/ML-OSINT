"""
Hierarchical Multi-Head Attention Network for Ukraine Conflict OSINT Analysis

Architecture Overview:
======================

This network processes 198 decomposed features across 6 data domains using a
hierarchical attention mechanism that:

1. Groups features by domain (UCDP, FIRMS, Sentinel, DeepState, Equipment, Personnel)
2. Applies within-domain self-attention to learn feature interactions
3. Applies cross-domain attention to learn which domains matter for each prediction
4. Produces interpretable attention weights at both levels

Key Design Decisions:
- Monthly temporal resolution (32 time points)
- Source-specific normalization preserves relative scales within domains
- Resolution metadata encoded as learnable embeddings
- Missing data handled via learned mask embeddings
- Multi-task output: casualty prediction + regime classification + anomaly detection

Feature Hierarchy (198 total leaf features):
- UCDP: 33 features (events, deaths, geography decompositions)
- FIRMS: 42 features (fire counts, FRP, brightness decompositions)
- Sentinel: 43 features (S1/S2/S3/S5P decompositions)
- DeepState: 45 features (polygons, attack directions, units, airfields)
- Equipment: 29 features (aircraft, tanks, AFVs, artillery decompositions)
- Personnel: 6 features (cumulative, daily, monthly rates)
"""

import os
# Enable MPS fallback for unsupported ops (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from collections import defaultdict
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

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

BASE_DIR = PROJECT_ROOT


# =============================================================================
# DOMAIN CONFIGURATION
# =============================================================================

@dataclass
class DomainConfig:
    """Configuration for a single data domain."""
    name: str
    num_features: int
    feature_names: List[str]
    native_resolution: str  # 'event', 'daily', 'weekly', 'monthly', 'variable'
    data_type: str  # 'count', 'continuous', 'categorical', 'cumulative'
    normalization: str  # 'standard', 'minmax', 'log', 'none'


DOMAIN_CONFIGS = {
    'ucdp': DomainConfig(
        name='UCDP Conflict Events',
        num_features=33,
        feature_names=[
            # Events decomposition (8)
            'events_state_based', 'events_non_state', 'events_one_sided',
            'events_clear', 'events_uncertain',
            'events_exact_loc', 'events_approx_loc', 'events_regional_loc',
            # Deaths decomposition (7)
            'deaths_side_a', 'deaths_side_b', 'deaths_civilians', 'deaths_unknown',
            'deaths_best', 'deaths_high', 'deaths_low',
            # Geography - Oblast (13)
            'geo_donetsk', 'geo_luhansk', 'geo_kharkiv', 'geo_kherson',
            'geo_zaporizhzhya', 'geo_sumy', 'geo_dnipropetrovsk', 'geo_mykolayiv',
            'geo_chernihiv', 'geo_kyiv_oblast', 'geo_kyiv_city', 'geo_odessa', 'geo_other',
            # Geography - Front (5)
            'front_eastern', 'front_southern', 'front_northeastern', 'front_northern', 'front_rear'
        ],
        native_resolution='event',
        data_type='count',
        normalization='log'
    ),
    'firms': DomainConfig(
        name='FIRMS Fire Detections',
        num_features=42,
        feature_names=[
            # Fire count decomposition (9)
            'fires_day', 'fires_night',
            'fires_high_conf', 'fires_nominal_conf', 'fires_low_conf',
            'fires_type_0', 'fires_type_2', 'fires_type_3',
            'fires_total',
            # FRP decomposition (14)
            'frp_tiny', 'frp_small', 'frp_medium', 'frp_large', 'frp_very_large', 'frp_extreme',
            'frp_day_mean', 'frp_night_mean', 'frp_day_max', 'frp_night_max',
            'frp_morning', 'frp_afternoon', 'frp_evening', 'frp_night_period',
            # Brightness (6)
            'brightness_mean', 'brightness_max', 'brightness_std',
            'bright_t31_mean', 'bright_t31_max', 'brightness_ratio',
            # Scan/Track (4)
            'scan_mean', 'track_mean', 'pixel_area_mean', 'pixel_area_max',
            # Derived (9)
            'frp_per_fire', 'frp_total', 'fire_density',
            'day_night_ratio', 'high_intensity_pct', 'extreme_fire_count',
            'spatial_spread', 'temporal_clustering', 'persistence_index'
        ],
        native_resolution='daily',
        data_type='continuous',
        normalization='log'
    ),
    'sentinel': DomainConfig(
        name='Sentinel Satellite',
        num_features=43,
        feature_names=[
            # Sentinel-1 (8)
            's1_count', 's1_vv_mean', 's1_vh_mean', 's1_vv_vh_ratio',
            's1_change_intensity', 's1_coherence', 's1_backscatter_anom', 's1_coverage',
            # Sentinel-2 (20)
            's2_count', 's2_cloud_cover', 's2_cloud_free_count',
            's2_b02_blue', 's2_b03_green', 's2_b04_red', 's2_b08_nir',
            's2_b11_swir1', 's2_b12_swir2',
            's2_ndvi_mean', 's2_ndvi_min', 's2_ndvi_max',
            's2_ndwi_mean', 's2_nbr_mean', 's2_ndbi_mean',
            's2_burn_area', 's2_vegetation_loss', 's2_urban_change',
            's2_temporal_variance', 's2_spatial_heterogeneity',
            # Sentinel-3 (7)
            's3_frp_count', 's3_frp_total', 's3_frp_max',
            's3_otci_mean', 's3_gifapar_mean', 's3_cloud_cover', 's3_coverage',
            # Sentinel-5P (8)
            's5p_no2_mean', 's5p_no2_max', 's5p_no2_anomaly',
            's5p_co_mean', 's5p_co_max', 's5p_co_anomaly',
            's5p_aerosol_index', 's5p_coverage'
        ],
        native_resolution='variable',
        data_type='continuous',
        normalization='standard'
    ),
    'deepstate': DomainConfig(
        name='DeepState Front Line',
        num_features=45,
        feature_names=[
            # Polygon status (8)
            'poly_occupied_count', 'poly_liberated_count', 'poly_contested_count', 'poly_unknown_count',
            'poly_occupied_area', 'poly_liberated_area', 'poly_contested_area', 'poly_total_area',
            # Polygon changes (4)
            'poly_newly_occupied', 'poly_newly_liberated', 'front_line_length', 'front_line_change',
            # Attack directions (8)
            'arrows_north', 'arrows_east', 'arrows_south', 'arrows_west',
            'arrows_total', 'arrows_dominant_dir', 'arrows_spread', 'arrows_intensity',
            # Military units by echelon (6)
            'units_army', 'units_division', 'units_brigade', 'units_regiment', 'units_battalion', 'units_total',
            # Military units by type (8)
            'units_motorized', 'units_tank', 'units_artillery', 'units_airborne',
            'units_recon', 'units_bars', 'units_akhmat', 'units_other',
            # Unit dynamics (4)
            'units_new', 'units_removed', 'unit_density', 'unit_concentration',
            # Airfields (5)
            'airfields_crimea', 'airfields_east', 'airfields_north', 'airfields_west', 'airfields_total',
            # Special (2)
            'crimean_bridge_status', 'moskva_marker'
        ],
        native_resolution='variable',
        data_type='count',
        normalization='standard'
    ),
    'equipment': DomainConfig(
        name='Equipment Losses',
        num_features=29,
        feature_names=[
            # Aircraft (5)
            'aircraft_sukhoi', 'aircraft_mig', 'aircraft_transport', 'aircraft_awacs', 'aircraft_total',
            # Helicopters (6)
            'heli_ka52', 'heli_mi28', 'heli_mi24', 'heli_mi8', 'heli_other', 'heli_total',
            # Tanks (7)
            'tank_t62', 'tank_t64', 'tank_t72', 'tank_t80', 'tank_t90', 'tank_other', 'tank_total',
            # AFVs (6)
            'afv_bmp', 'afv_btr', 'afv_mtlb', 'afv_bmd', 'afv_other', 'afv_total',
            # Artillery (3)
            'arty_towed', 'arty_sp', 'arty_mrl',
            # Other (2)
            'drones_total', 'air_defense_total'
        ],
        native_resolution='daily',
        data_type='cumulative',
        normalization='log'
    ),
    'personnel': DomainConfig(
        name='Personnel Losses',
        num_features=6,
        feature_names=[
            'personnel_cumulative', 'personnel_monthly', 'personnel_daily_avg',
            'personnel_rate_change', 'personnel_acceleration', 'personnel_trend'
        ],
        native_resolution='daily',
        data_type='cumulative',
        normalization='log'
    )
}

# Total features
TOTAL_FEATURES = sum(cfg.num_features for cfg in DOMAIN_CONFIGS.values())
print(f"Total leaf features: {TOTAL_FEATURES}")


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding for temporal sequences.

        Encodes both:
        - Absolute position in sequence (trend)
        - Cyclical month-of-year (seasonality)
        """
        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Standard positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor, month_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
                month_indices: Optional [batch, seq_len] with month (1-12)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class ResolutionEncoder(nn.Module):
        """
        Encodes metadata about data resolution and availability.

        This allows the network to learn that different domains have
        different native resolutions and reliability.
        """
        def __init__(self, d_model: int, num_resolutions: int = 5):
            super().__init__()
            # Resolution types: event, daily, weekly, monthly, variable
            self.resolution_embedding = nn.Embedding(num_resolutions, d_model)
            # Data type: count, continuous, categorical, cumulative
            self.datatype_embedding = nn.Embedding(4, d_model)

            self.resolution_map = {'event': 0, 'daily': 1, 'weekly': 2, 'monthly': 3, 'variable': 4}
            self.datatype_map = {'count': 0, 'continuous': 1, 'categorical': 2, 'cumulative': 3}

        def forward(self, resolution: str, data_type: str) -> torch.Tensor:
            res_idx = torch.tensor([self.resolution_map[resolution]])
            type_idx = torch.tensor([self.datatype_map[data_type]])
            return self.resolution_embedding(res_idx) + self.datatype_embedding(type_idx)


    class DomainEncoder(nn.Module):
        """
        Encodes features within a single domain.

        Uses self-attention to learn interactions between features
        within the same domain (e.g., how different tank types relate).
        """
        def __init__(
            self,
            num_features: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.num_features = num_features
            self.d_model = d_model

            # Project input features to d_model dimension
            self.input_projection = nn.Linear(1, d_model)

            # Feature embeddings (learnable, one per feature in domain)
            self.feature_embeddings = nn.Embedding(num_features, d_model)

            # Self-attention layers within domain
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection to domain embedding
            self.output_projection = nn.Linear(d_model, d_model)

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: [batch, num_features] - raw feature values
                mask: [batch, num_features] - 1 for valid, 0 for missing

            Returns:
                domain_embedding: [batch, d_model] - compressed domain representation
                feature_attention: [batch, num_features] - attention weights per feature
            """
            batch_size = x.size(0)

            # Project each feature to d_model: [batch, num_features, d_model]
            x_proj = self.input_projection(x.unsqueeze(-1))

            # Add feature-specific embeddings
            feat_indices = torch.arange(self.num_features, device=x.device)
            feat_emb = self.feature_embeddings(feat_indices)  # [num_features, d_model]
            x_proj = x_proj + feat_emb.unsqueeze(0)

            # Apply self-attention
            # Create attention mask if needed (True = ignore)
            attn_mask = None
            if mask is not None:
                attn_mask = (mask == 0)  # True where we should mask
                # Ensure at least one position is unmasked per batch item to prevent transformer error
                # If all positions are masked, unmask the first one
                all_masked = attn_mask.all(dim=1)  # [batch]
                if all_masked.any():
                    attn_mask = attn_mask.clone()
                    attn_mask[all_masked, 0] = False  # Unmask first position

            encoded = self.transformer(x_proj, src_key_padding_mask=attn_mask)

            # Global average pooling over features (with mask)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = encoded.mean(dim=1)

            domain_emb = self.output_projection(pooled)

            # Compute feature importance (attention to CLS-like aggregation)
            feature_attention = F.softmax(encoded.mean(dim=-1), dim=-1)

            return domain_emb, feature_attention


    class CrossDomainAttention(nn.Module):
        """
        Learns relationships between different data domains.

        Uses multi-head attention where each domain embedding attends
        to all other domains to learn cross-domain dependencies.
        """
        def __init__(
            self,
            num_domains: int = 6,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.num_domains = num_domains
            self.d_model = d_model

            # Domain-level embeddings (learnable)
            self.domain_embeddings = nn.Embedding(num_domains, d_model)

            # Cross-domain attention
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
            domain_embeddings: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                domain_embeddings: [batch, num_domains, d_model]

            Returns:
                fused_embedding: [batch, d_model]
                domain_attention: [batch, num_domains] - importance of each domain
            """
            batch_size = domain_embeddings.size(0)

            # Add domain-type embeddings
            domain_indices = torch.arange(self.num_domains, device=domain_embeddings.device)
            domain_type_emb = self.domain_embeddings(domain_indices)  # [num_domains, d_model]
            x = domain_embeddings + domain_type_emb.unsqueeze(0)

            # Cross-domain attention
            encoded = self.transformer(x)

            # Compute domain importance weights
            domain_attention = F.softmax(encoded.mean(dim=-1), dim=-1)

            # Weighted combination
            fused = (encoded * domain_attention.unsqueeze(-1)).sum(dim=1)

            return fused, domain_attention


    class TemporalEncoder(nn.Module):
        """
        Encodes temporal patterns across the sequence of monthly observations.

        Uses transformer architecture with positional encoding to capture
        both long-range dependencies and seasonal patterns.
        """
        def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            max_seq_len: int = 100,
            dropout: float = 0.1
        ):
            super().__init__()
            self.d_model = d_model

            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

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
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, d_model]
                mask: [batch, seq_len] - 1 for valid, 0 for padding

            Returns:
                encoded: [batch, seq_len, d_model]
            """
            x = self.pos_encoder(x)

            attn_mask = None
            if mask is not None:
                attn_mask = (mask == 0)

            encoded = self.transformer(x, src_key_padding_mask=attn_mask)
            return encoded


    class StateTransitionNetwork(nn.Module):
        """
        Explicit state transition modeling for S(t+1) = f(S(t)).

        Predicts state changes (deltas) rather than absolute values,
        which is more appropriate for dynamical systems.
        """

        def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.state_dim = state_dim
            self.hidden_dim = hidden_dim

            # State encoder
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Transition model using GRU for sequential state evolution
            self.transition_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            # Delta predictor head - predicts change from current state
            self.delta_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, state_dim)
            )

            # State reconstructor (combines delta with current state)
            self.state_reconstructor = nn.Sequential(
                nn.Linear(state_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )

        def forward(
            self,
            current_state: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
            return_delta: bool = False
        ) -> Tuple[torch.Tensor, ...]:
            """
            Args:
                current_state: [batch, state_dim] or [batch, seq_len, state_dim]
                hidden: Optional GRU hidden state [num_layers, batch, hidden_dim]
                return_delta: Whether to return the predicted delta

            Returns:
                next_state: Predicted next state
                new_hidden: Updated GRU hidden state
                delta: (optional) Predicted state change
            """
            # Handle both single timestep and sequence inputs
            if current_state.dim() == 2:
                current_state = current_state.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False

            batch_size, seq_len, _ = current_state.shape

            # Encode current state
            encoded = self.state_encoder(current_state)  # [batch, seq_len, hidden_dim]

            # Apply transition model
            if hidden is None:
                transition_out, new_hidden = self.transition_gru(encoded)
            else:
                transition_out, new_hidden = self.transition_gru(encoded, hidden)

            # Predict state delta
            delta = self.delta_predictor(transition_out)  # [batch, seq_len, state_dim]

            # Reconstruct next state = current + learned_combination(current, delta)
            combined = torch.cat([current_state, delta], dim=-1)
            next_state = current_state + self.state_reconstructor(combined)

            if squeeze_output:
                next_state = next_state.squeeze(1)
                delta = delta.squeeze(1)

            if return_delta:
                return next_state, new_hidden, delta
            return next_state, new_hidden


    class MultiScaleTemporalEncoder(nn.Module):
        """
        Processes temporal data at multiple scales (daily, weekly, monthly context).
        Uses separate encoders for each scale then fuses via attention.
        """

        def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            scales: List[int] = None,
            nhead: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            if scales is None:
                scales = [1, 7, 30]  # daily, weekly, monthly
            self.scales = scales
            self.d_model = d_model
            self.num_scales = len(scales)

            # Per-scale convolutional encoders
            self.scale_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(input_dim, d_model, kernel_size=max(1, scale),
                              padding=max(0, scale // 2), padding_mode='replicate'),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU()
                )
                for scale in scales
            ])

            # Cross-scale attention fusion
            self.cross_scale_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )

            # Scale embeddings to differentiate scale origins
            self.scale_embeddings = nn.Embedding(len(scales), d_model)

            # Output projection
            self.output_projection = nn.Sequential(
                nn.Linear(d_model * len(scales), d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, input_dim]

            Returns:
                fused: [batch, seq_len, d_model] - multi-scale encoded features
            """
            batch_size, seq_len, input_dim = x.shape

            # Transpose for conv1d: [batch, input_dim, seq_len]
            x_t = x.transpose(1, 2)

            # Extract multi-scale features
            scale_features = []
            for i, encoder in enumerate(self.scale_encoders):
                # Apply scale-specific convolution
                feat = encoder(x_t)  # [batch, d_model, seq_len]
                feat = feat.transpose(1, 2)  # [batch, seq_len, d_model]

                # Add scale embedding
                scale_emb = self.scale_embeddings(
                    torch.tensor([i], device=x.device)
                ).unsqueeze(0).expand(batch_size, seq_len, -1)
                feat = feat + scale_emb

                scale_features.append(feat)

            # Stack for attention: [batch, num_scales * seq_len, d_model]
            # Reshape to allow cross-scale attention
            stacked = torch.stack(scale_features, dim=2)  # [batch, seq_len, num_scales, d_model]
            stacked = stacked.view(batch_size, seq_len * self.num_scales, self.d_model)

            # Cross-scale self-attention
            attended, _ = self.cross_scale_attention(stacked, stacked, stacked)

            # Reshape back and concatenate scales
            attended = attended.view(batch_size, seq_len, self.num_scales, self.d_model)

            # Concatenate scale features for each timestep
            concat_scales = attended.view(batch_size, seq_len, -1)  # [batch, seq_len, num_scales * d_model]

            # Project to output dimension
            fused = self.output_projection(concat_scales)

            return fused


    class DeltaPredictionHead(nn.Module):
        """Predicts changes/deltas rather than absolute values."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

            # Delta prediction network
            self.delta_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

            # Optional scaling factor for delta (learnable)
            self.delta_scale = nn.Parameter(torch.ones(output_dim))

        def forward(
            self,
            encoded_state: torch.Tensor,
            current_values: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                encoded_state: [batch, ..., input_dim] - encoded representation
                current_values: [batch, ..., output_dim] - current known values

            Returns:
                predicted_values: current_values + scaled_delta
                delta: the predicted change
            """
            # Predict delta
            delta = self.delta_net(encoded_state)

            # Apply learnable scaling
            scaled_delta = delta * self.delta_scale

            # Return predicted values as current + delta
            predicted_values = current_values + scaled_delta

            return predicted_values, delta


    class HierarchicalAttentionNetwork(nn.Module):
        """
        Main network combining all components.

        Architecture:
        1. Domain Encoders: Process features within each domain
        2. Cross-Domain Attention: Learn domain interactions
        3. Temporal Encoder: Capture temporal patterns
        4. Multi-Task Heads: Prediction, classification, anomaly detection
        """
        def __init__(
            self,
            domain_configs: Dict[str, DomainConfig],
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            num_temporal_layers: int = 2,
            dropout: float = 0.1,
            max_seq_len: int = 100
        ):
            super().__init__()
            self.domain_configs = domain_configs
            self.domain_names = list(domain_configs.keys())
            self.num_domains = len(domain_configs)
            self.d_model = d_model

            # Domain-specific encoders
            self.domain_encoders = nn.ModuleDict({
                name: DomainEncoder(
                    num_features=cfg.num_features,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    dropout=dropout
                )
                for name, cfg in domain_configs.items()
            })

            # Resolution encoders
            self.resolution_encoder = ResolutionEncoder(d_model)

            # Cross-domain attention
            self.cross_domain_attention = CrossDomainAttention(
                num_domains=self.num_domains,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dropout=dropout
            )

            # Temporal encoder
            self.temporal_encoder = TemporalEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_temporal_layers,
                max_seq_len=max_seq_len,
                dropout=dropout
            )

            # =========================================
            # OUTPUT HEADS
            # =========================================

            # 1. Casualty Prediction Head (regression)
            self.casualty_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 3)  # [deaths_best, deaths_low, deaths_high]
            )

            # 2. Conflict Regime Classification Head
            self.regime_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 4)  # [low_intensity, medium, high, major_offensive]
            )

            # 3. Anomaly Detection Head
            self.anomaly_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),  # Anomaly score
                nn.Sigmoid()
            )

            # 4. Next-Month Prediction Head
            self.forecast_head = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, TOTAL_FEATURES)  # Predict all features
            )

        def forward(
            self,
            features: Dict[str, torch.Tensor],
            masks: Optional[Dict[str, torch.Tensor]] = None,
            return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                features: Dict mapping domain name to [batch, seq_len, num_features]
                masks: Optional dict mapping domain name to [batch, seq_len, num_features]
                return_attention: Whether to return attention weights

            Returns:
                Dict containing:
                - casualty_pred: [batch, seq_len, 3]
                - regime_logits: [batch, seq_len, 4]
                - anomaly_score: [batch, seq_len, 1]
                - forecast: [batch, seq_len, total_features]
                - domain_attention: [batch, seq_len, num_domains] (if return_attention)
                - feature_attention: Dict[domain, [batch, seq_len, num_features]] (if return_attention)
            """
            batch_size = next(iter(features.values())).size(0)
            seq_len = next(iter(features.values())).size(1)
            device = next(iter(features.values())).device

            # Process each domain using batched operations
            domain_embeddings = []
            feature_attentions = {}

            for domain_name in self.domain_names:
                domain_feats = features[domain_name]  # [batch, seq_len, num_features]
                domain_mask = masks[domain_name] if masks else None

                # BATCHED: Reshape to process all timesteps at once
                # [batch, seq_len, num_features] -> [batch * seq_len, num_features]
                num_features = domain_feats.size(-1)
                flat_feats = domain_feats.view(batch_size * seq_len, num_features)

                flat_mask = None
                if domain_mask is not None:
                    flat_mask = domain_mask.view(batch_size * seq_len, num_features)

                # Single forward pass for all timesteps
                flat_emb, flat_attn = self.domain_encoders[domain_name](flat_feats, flat_mask)

                # Reshape back: [batch * seq_len, d_model] -> [batch, seq_len, d_model]
                domain_emb = flat_emb.view(batch_size, seq_len, self.d_model)
                domain_embeddings.append(domain_emb)

                if return_attention:
                    # Reshape attention: [batch * seq_len, num_features] -> [batch, seq_len, num_features]
                    feature_attentions[domain_name] = flat_attn.view(batch_size, seq_len, num_features)

            # Stack domain embeddings: [batch, seq_len, num_domains, d_model]
            all_domain_emb = torch.stack(domain_embeddings, dim=2)

            # BATCHED Cross-domain attention: process all timesteps in parallel
            # Reshape: [batch, seq_len, num_domains, d_model] -> [batch * seq_len, num_domains, d_model]
            flat_domain_emb = all_domain_emb.view(batch_size * seq_len, self.num_domains, self.d_model)

            # Single forward pass for cross-domain attention
            flat_fused, flat_domain_attn = self.cross_domain_attention(flat_domain_emb)

            # Reshape back: [batch * seq_len, d_model] -> [batch, seq_len, d_model]
            fused_seq = flat_fused.view(batch_size, seq_len, self.d_model)

            # Store domain attention if needed: [batch * seq_len, num_domains] -> [batch, seq_len, num_domains]
            domain_attentions = flat_domain_attn.view(batch_size, seq_len, self.num_domains)

            # Temporal encoding
            temporal_encoded = self.temporal_encoder(fused_seq)

            # Apply output heads
            outputs = {
                'casualty_pred': self.casualty_head(temporal_encoded),
                'regime_logits': self.regime_head(temporal_encoded),
                'anomaly_score': self.anomaly_head(temporal_encoded),
                'forecast': self.forecast_head(temporal_encoded)
            }

            if return_attention:
                outputs['domain_attention'] = domain_attentions  # Already [batch, seq_len, num_domains]
                outputs['feature_attention'] = feature_attentions

            return outputs


    class ConflictDataset(Dataset):
        """Dataset for loading and preprocessing conflict data."""

        def __init__(
            self,
            data_dir: Path,
            domain_configs: Dict[str, DomainConfig],
            seq_len: int = 32,
            normalize: bool = True
        ):
            self.data_dir = data_dir
            self.domain_configs = domain_configs
            self.seq_len = seq_len
            self.normalize = normalize

            # Load and preprocess data
            self._load_data()

        def _load_data(self):
            """Load all data sources and align temporally."""
            # This is a placeholder - actual implementation would load
            # and decompose all features from raw data files

            # For now, we'll create synthetic data matching the structure
            self.num_samples = 32  # Monthly data points

            self.features = {}
            self.masks = {}

            for domain_name, cfg in self.domain_configs.items():
                # Synthetic data for architecture testing
                self.features[domain_name] = torch.randn(self.num_samples, cfg.num_features)
                self.masks[domain_name] = torch.ones(self.num_samples, cfg.num_features)

            # Synthetic targets
            self.targets = {
                'casualties': torch.randn(self.num_samples, 3),
                'regime': torch.randint(0, 4, (self.num_samples,)),
                'anomaly': torch.rand(self.num_samples)
            }

        def __len__(self):
            return max(1, self.num_samples - self.seq_len + 1)

        def __getitem__(self, idx):
            # Return a sequence window
            features = {k: v[idx:idx+self.seq_len] for k, v in self.features.items()}
            masks = {k: v[idx:idx+self.seq_len] for k, v in self.masks.items()}
            targets = {k: v[idx:idx+self.seq_len] for k, v in self.targets.items()}

            return features, masks, targets


# =============================================================================
# ARCHITECTURE SUMMARY
# =============================================================================

def print_architecture_summary():
    """Print detailed architecture summary."""

    print("=" * 80)
    print("HIERARCHICAL MULTI-HEAD ATTENTION NETWORK")
    print("Ukraine Conflict OSINT Feature Analysis")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("LAYER 1: DOMAIN ENCODERS (Within-Domain Self-Attention)")
    print("=" * 80)

    for name, cfg in DOMAIN_CONFIGS.items():
        print(f"\n  {name.upper()} Domain Encoder:")
        print(f"    Input: {cfg.num_features} features")
        print(f"    Native resolution: {cfg.native_resolution}")
        print(f"    Data type: {cfg.data_type}")
        print(f"    Normalization: {cfg.normalization}")
        print(f"    Output: 64-dim domain embedding + feature attention weights")

    print("\n" + "=" * 80)
    print("LAYER 2: CROSS-DOMAIN ATTENTION")
    print("=" * 80)
    print("""
    Input: 6 domain embeddings (each 64-dim)

    Learns which domains are most important for each prediction:
    - UCDP ↔ FIRMS: Fire activity predicting casualties
    - Sentinel ↔ DeepState: Satellite imagery validating front lines
    - Equipment ↔ Personnel: Loss ratios and intensity

    Output: 64-dim fused embedding + domain attention weights
    """)

    print("\n" + "=" * 80)
    print("LAYER 3: TEMPORAL ENCODER")
    print("=" * 80)
    print("""
    Input: Sequence of fused embeddings [batch, seq_len, 64]

    Captures:
    - Seasonal patterns (weather → fire → casualty cycles)
    - Trend (escalation/de-escalation over time)
    - Regime changes (offensive/defensive periods)

    Uses positional encoding + transformer self-attention

    Output: Temporally-encoded sequence [batch, seq_len, 64]
    """)

    print("\n" + "=" * 80)
    print("LAYER 4: OUTPUT HEADS (Multi-Task)")
    print("=" * 80)
    print("""
    1. CASUALTY PREDICTION HEAD (Regression)
       Output: [deaths_best, deaths_low, deaths_high]
       Loss: MSE with uncertainty weighting

    2. REGIME CLASSIFICATION HEAD (4-class)
       Classes: [low_intensity, medium, high, major_offensive]
       Loss: Cross-entropy

    3. ANOMALY DETECTION HEAD (Binary)
       Output: Anomaly score [0, 1]
       Loss: Binary cross-entropy (self-supervised)

    4. FORECASTING HEAD (Sequence-to-Sequence)
       Output: Predicted features for next time step
       Loss: MSE for each domain
    """)

    print("\n" + "=" * 80)
    print("PARAMETER COUNT")
    print("=" * 80)

    if HAS_TORCH:
        model = HierarchicalAttentionNetwork(
            domain_configs=DOMAIN_CONFIGS,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_temporal_layers=2,
            dropout=0.1
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Breakdown by component
        print("\n  By component:")

        encoder_params = sum(sum(p.numel() for p in enc.parameters())
                           for enc in model.domain_encoders.values())
        print(f"    Domain Encoders: {encoder_params:,}")

        cross_params = sum(p.numel() for p in model.cross_domain_attention.parameters())
        print(f"    Cross-Domain Attention: {cross_params:,}")

        temporal_params = sum(p.numel() for p in model.temporal_encoder.parameters())
        print(f"    Temporal Encoder: {temporal_params:,}")

        head_params = (sum(p.numel() for p in model.casualty_head.parameters()) +
                      sum(p.numel() for p in model.regime_head.parameters()) +
                      sum(p.numel() for p in model.anomaly_head.parameters()) +
                      sum(p.numel() for p in model.forecast_head.parameters()))
        print(f"    Output Heads: {head_params:,}")

    print("\n" + "=" * 80)
    print("ATTENTION INTERPRETABILITY")
    print("=" * 80)
    print("""
    The network produces interpretable attention weights at two levels:

    1. FEATURE ATTENTION (per domain):
       Shows which features within each domain are most important.
       Example: Within FIRMS, is day/night fire ratio more important than total FRP?

    2. DOMAIN ATTENTION (cross-domain):
       Shows which data sources contribute most to predictions.
       Example: Is UCDP or DeepState more predictive of next-month casualties?

    These can be visualized as heatmaps and analyzed to understand:
    - Which features drive predictions
    - How feature importance changes over time (regime detection)
    - Anomalies where attention patterns deviate from normal
    """)

    print("\n" + "=" * 80)
    print("DATA FLOW DIAGRAM")
    print("=" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              INPUT FEATURES                                  │
    │  [UCDP:33] [FIRMS:42] [Sentinel:43] [DeepState:45] [Equipment:29] [Personnel:6]
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         DOMAIN ENCODERS                                      │
    │                                                                              │
    │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │   │  UCDP    │ │  FIRMS   │ │ Sentinel │ │DeepState │ │Equipment │ │Personnel │
    │   │ Encoder  │ │ Encoder  │ │ Encoder  │ │ Encoder  │ │ Encoder  │ │ Encoder  │
    │   │          │ │          │ │          │ │          │ │          │ │          │
    │   │ Self-Attn│ │ Self-Attn│ │ Self-Attn│ │ Self-Attn│ │ Self-Attn│ │ Self-Attn│
    │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
    │        │            │            │            │            │            │
    │        ▼            ▼            ▼            ▼            ▼            ▼
    │      [64d]        [64d]        [64d]        [64d]        [64d]        [64d]
    │        │            │            │            │            │            │
    │        └────────────┴────────────┴─────┬──────┴────────────┴────────────┘
    └────────────────────────────────────────┼────────────────────────────────────┘
                                             │
                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      CROSS-DOMAIN ATTENTION                                  │
    │                                                                              │
    │     Input: [batch, 6, 64]  (6 domain embeddings)                            │
    │                                                                              │
    │     Multi-Head Attention learns:                                            │
    │       - UCDP ↔ FIRMS correlations                                           │
    │       - Sentinel ↔ DeepState validation                                     │
    │       - Equipment ↔ Personnel ratios                                        │
    │                                                                              │
    │     Output: [batch, 64] fused + [batch, 6] domain importance                │
    └─────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        TEMPORAL ENCODER                                      │
    │                                                                              │
    │     Input: [batch, seq_len, 64] (sequence of fused embeddings)              │
    │                                                                              │
    │     ┌─────────────────────────────────────────────────────────────┐         │
    │     │  Positional Encoding (trend + seasonality)                  │         │
    │     │                                                             │         │
    │     │  May Jun Jul Aug Sep Oct Nov Dec Jan Feb Mar Apr May ...    │         │
    │     │   │   │   │   │   │   │   │   │   │   │   │   │   │         │         │
    │     │   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼         │         │
    │     │  [Transformer Self-Attention across time]                   │         │
    │     └─────────────────────────────────────────────────────────────┘         │
    │                                                                              │
    │     Output: [batch, seq_len, 64] temporally-encoded                         │
    └─────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          OUTPUT HEADS                                        │
    │                                                                              │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
    │   │  CASUALTY    │  │   REGIME     │  │   ANOMALY    │  │  FORECAST    │    │
    │   │  PREDICTION  │  │   CLASS      │  │   SCORE      │  │  NEXT MONTH  │    │
    │   │              │  │              │  │              │  │              │    │
    │   │  MLP → [3]   │  │  MLP → [4]   │  │  MLP → [1]   │  │  MLP → [198] │    │
    │   │              │  │              │  │              │  │              │    │
    │   │  best/lo/hi  │  │  lo/med/hi/  │  │  sigmoid     │  │  all feats   │    │
    │   │  casualties  │  │  offensive   │  │  0-1 score   │  │  predicted   │    │
    │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print_architecture_summary()
