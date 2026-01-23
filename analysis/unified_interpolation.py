"""
Unified Cross-Source Interpolation Model

This model learns cross-source relationships through self-supervised reconstruction:
- Given 4 out of 5 sources, predict the held-out source
- Forces the model to learn meaningful cross-source dependencies
- Produces unified daily features that can feed downstream forecasting

Architecture:
    Source-specific JIM encoders (pretrained) → Cross-Source Fusion → Reconstruction

Training:
    1. For each day, randomly mask one source
    2. Encode remaining sources with JIM models
    3. Cross-attend to fuse information
    4. Reconstruct masked source features
    5. Ground truth: actual observed values for that source

Usage:
    python unified_interpolation.py --train --epochs 100
    python unified_interpolation.py --inference --date 2024-01-15
"""

import os
# Enable MPS fallback for unsupported ops (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import random

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    UNIFIED_INTERP_MODEL, UNIFIED_DELTA_MODEL, UNIFIED_HYBRID_MODEL,
    get_interp_model_path,
)
from joint_interpolation_models import (
    JointInterpolationModel,
    InterpolationConfig,
    INTERPOLATION_CONFIGS,
)
from training_utils import WarmupCosineScheduler, GradientAccumulator, TimeSeriesAugmentation
from training_config import DataConfig
from interpolation_data_loaders import (
    SentinelDataLoader,
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader,
    PersonnelDataLoader,
    VIINADataLoader,
    HDXConflictDataLoader,
    HDXFoodPricesDataLoader,
    HDXRainfallDataLoader,
    IOMDisplacementDataLoader,
    GoogleMobilityDataLoader,
)

# MODEL_DIR is imported from config.paths
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# SOURCE CONFIGURATIONS
# =============================================================================

@dataclass
class SourceConfig:
    """Configuration for a data source in the unified model."""
    name: str
    loader_class: type
    n_features: int  # Will be set dynamically
    jim_config_key: str  # Key in INTERPOLATION_CONFIGS
    d_embed: int = 64  # Embedding dimension after JIM encoding


SOURCE_CONFIGS = {
    # ==========================================================================
    # ORIGINAL SOURCES (kept for backward compatibility)
    # ==========================================================================
    # Sentinel excluded: only 32 monthly samples + data quality issues (row 1 anomaly)
    # This was limiting all aligned data to 32 samples instead of 416
    'deepstate': SourceConfig(
        name='DeepState',
        loader_class=DeepStateDataLoader,
        n_features=55,
        jim_config_key='deepstate',
        d_embed=64
    ),
    'equipment': SourceConfig(
        name='Equipment',
        loader_class=EquipmentDataLoader,
        n_features=38,
        jim_config_key='equipment_totals',
        d_embed=64
    ),
    'firms': SourceConfig(
        name='FIRMS',
        loader_class=FIRMSDataLoader,
        n_features=42,
        jim_config_key='sentinel3',  # Use sentinel3 config as FIRMS proxy
        d_embed=64
    ),
    'ucdp': SourceConfig(
        name='UCDP',
        loader_class=UCDPDataLoader,
        n_features=48,
        jim_config_key='deepstate',  # Use deepstate config as UCDP proxy
        d_embed=64
    ),

    # ==========================================================================
    # NEW SOURCES (added for expanded coverage)
    # ==========================================================================
    'viina': SourceConfig(
        name='VIINA Territorial Control',
        loader_class=VIINADataLoader,
        n_features=24,  # Daily territorial control metrics
        jim_config_key='deepstate',  # Similar temporal pattern to DeepState
        d_embed=64
    ),
    'hdx_conflict': SourceConfig(
        name='HDX Conflict Events',
        loader_class=HDXConflictDataLoader,
        n_features=18,  # Monthly conflict event aggregates
        jim_config_key='deepstate',  # Similar domain to UCDP
        d_embed=64
    ),
    'hdx_food': SourceConfig(
        name='HDX Food Prices',
        loader_class=HDXFoodPricesDataLoader,
        n_features=20,  # Economic indicators
        jim_config_key='equipment_totals',  # Economic-style features
        d_embed=64
    ),
    'hdx_rainfall': SourceConfig(
        name='HDX Rainfall',
        loader_class=HDXRainfallDataLoader,
        n_features=16,  # Environmental data (dekadal)
        jim_config_key='sentinel3',  # Environmental satellite-like
        d_embed=64
    ),
    'iom': SourceConfig(
        name='IOM Displacement',
        loader_class=IOMDisplacementDataLoader,
        n_features=18,  # Humanitarian metrics
        jim_config_key='equipment_totals',  # Cumulative-style features
        d_embed=64
    ),
    # Note: Google Mobility data ends 2022-02-23 (pre-war baseline only)
    # Kept in separate config for pre-war analysis use cases
}


# Config for war-period sources only (excludes mobility which has no war data)
SOURCE_CONFIGS_WAR_PERIOD = {
    k: v for k, v in SOURCE_CONFIGS.items()
    if k != 'mobility'  # Mobility ends before invasion
}

# Create a subset config for models that want only the dense sources (daily data)
SOURCE_CONFIGS_DENSE = {
    k: v for k, v in SOURCE_CONFIGS.items()
    if k in ['deepstate', 'equipment', 'firms', 'viina']
    # Note: mobility excluded (no war-period data)
}

# Create a subset for sources with full war coverage (Feb 2022 - present)
SOURCE_CONFIGS_FULL_COVERAGE = {
    k: v for k, v in SOURCE_CONFIGS.items()
    if k in ['deepstate', 'equipment', 'firms', 'ucdp', 'viina', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']
}

# Pre-war baseline sources (for comparison analysis)
SOURCE_CONFIGS_PREWAR = {
    'mobility': SOURCE_CONFIGS.get('mobility'),
} if 'mobility' in SOURCE_CONFIGS else {}


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class SourceEncoder(nn.Module):
        """
        Encodes a single source's features into a fixed-size embedding.

        Can optionally use a pretrained JIM model's cross-feature attention,
        or a simple projection for sources without pretrained JIM.
        """

        def __init__(
            self,
            n_features: int,
            d_embed: int = 64,
            pretrained_jim: Optional[JointInterpolationModel] = None
        ):
            super().__init__()
            self.n_features = n_features
            self.d_embed = d_embed

            if pretrained_jim is not None:
                # Use pretrained cross-feature attention
                self.cross_feature_attn = pretrained_jim.cross_feature_attn
                # Freeze pretrained weights initially
                for param in self.cross_feature_attn.parameters():
                    param.requires_grad = False
                self.use_jim = True
            else:
                # Simple projection: features → embedding
                self.feature_proj = nn.Sequential(
                    nn.Linear(n_features, d_embed * 2),
                    nn.LayerNorm(d_embed * 2),
                    nn.GELU(),
                    nn.Linear(d_embed * 2, d_embed),
                    nn.LayerNorm(d_embed)
                )
                self.use_jim = False

            # Final projection to unified embedding space
            self.output_proj = nn.Linear(d_embed, d_embed)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, n_features] - raw feature values

            Returns:
                [batch, d_embed] - source embedding
            """
            if self.use_jim:
                # Use cross-feature attention, then pool
                # x_proj: [batch, n_features, d_model]
                encoded = self.cross_feature_attn(x)
                # Pool across features
                pooled = encoded.mean(dim=1)  # [batch, d_model]
            else:
                # Direct projection
                pooled = self.feature_proj(x)  # [batch, d_embed]

            return self.output_proj(pooled)


    class SourceDecoder(nn.Module):
        """
        Decodes a fused embedding back to source-specific features.

        Used for reconstruction loss during training.
        """

        def __init__(self, n_features: int, d_embed: int = 64):
            super().__init__()
            self.n_features = n_features

            self.decoder = nn.Sequential(
                nn.Linear(d_embed, d_embed * 2),
                nn.LayerNorm(d_embed * 2),
                nn.GELU(),
                nn.Linear(d_embed * 2, d_embed * 2),
                nn.LayerNorm(d_embed * 2),
                nn.GELU(),
                nn.Linear(d_embed * 2, n_features),
                nn.Sigmoid()  # Constrain output to [0, 1] for normalized targets
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, d_embed] - fused embedding

            Returns:
                [batch, n_features] - reconstructed features
            """
            return self.decoder(x)


    class CrossSourceAttention(nn.Module):
        """
        Cross-attention between source embeddings.

        Learns which sources inform which others:
        - Sentinel thermal → FIRMS fire detections
        - UCDP events → Equipment losses
        - DeepState territory → all others
        """

        def __init__(
            self,
            n_sources: int,
            d_embed: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.n_sources = n_sources
            self.d_embed = d_embed

            # Learnable source type embeddings
            self.source_embeddings = nn.Embedding(n_sources, d_embed)

            # Cross-source transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_embed,
                nhead=nhead,
                dim_feedforward=d_embed * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection per source
            self.output_projs = nn.ModuleList([
                nn.Linear(d_embed, d_embed) for _ in range(n_sources)
            ])

        def forward(
            self,
            source_embeddings: Dict[str, torch.Tensor],
            source_order: List[str],
            mask: Optional[Dict[str, torch.Tensor]] = None
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                source_embeddings: {source_name: [batch, d_embed]}
                source_order: List of source names in consistent order
                mask: {source_name: [batch]} - 1 for present, 0 for masked

            Returns:
                {source_name: [batch, d_embed]} - fused embeddings
            """
            batch_size = next(iter(source_embeddings.values())).size(0)
            device = next(iter(source_embeddings.values())).device

            # Stack source embeddings: [batch, n_sources, d_embed]
            stacked = torch.stack([source_embeddings[s] for s in source_order], dim=1)

            # Add source type embeddings
            source_indices = torch.arange(len(source_order), device=device)
            source_type_emb = self.source_embeddings(source_indices)  # [n_sources, d_embed]
            stacked = stacked + source_type_emb.unsqueeze(0)

            # Create attention mask if sources are masked
            attn_mask = None
            if mask is not None:
                # [batch, n_sources] - True means IGNORE
                mask_tensor = torch.stack([1 - mask.get(s, torch.ones(batch_size, device=device))
                                          for s in source_order], dim=1)
                attn_mask = mask_tensor.bool()

            # Cross-source attention
            fused = self.transformer(stacked, src_key_padding_mask=attn_mask)

            # Apply per-source output projections
            outputs = {}
            for i, source_name in enumerate(source_order):
                outputs[source_name] = self.output_projs[i](fused[:, i, :])

            return outputs


    class UnifiedInterpolationModel(nn.Module):
        """
        Unified model for cross-source interpolation with self-supervised learning.

        Architecture:
        1. Source-specific encoders (can use pretrained JIM)
        2. Cross-source attention fusion
        3. Source-specific decoders (for reconstruction)

        Training: Mask one source, predict it from others.
        Inference: Get unified daily features from all available sources.
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            d_embed: int = 64,
            nhead: int = 4,
            num_fusion_layers: int = 2,
            dropout: float = 0.1,
            pretrained_jims: Optional[Dict[str, JointInterpolationModel]] = None
        ):
            super().__init__()
            self.source_names = list(source_configs.keys())
            self.source_configs = source_configs
            self.d_embed = d_embed

            # Source encoders
            self.encoders = nn.ModuleDict()
            for name, config in source_configs.items():
                jim = pretrained_jims.get(name) if pretrained_jims else None
                self.encoders[name] = SourceEncoder(
                    n_features=config.n_features,
                    d_embed=d_embed,
                    pretrained_jim=jim
                )

            # Cross-source fusion
            self.fusion = CrossSourceAttention(
                n_sources=len(source_configs),
                d_embed=d_embed,
                nhead=nhead,
                num_layers=num_fusion_layers,
                dropout=dropout
            )

            # Source decoders (for reconstruction)
            self.decoders = nn.ModuleDict()
            for name, config in source_configs.items():
                self.decoders[name] = SourceDecoder(
                    n_features=config.n_features,
                    d_embed=d_embed
                )

            # Unified output projection (concatenates all fused embeddings)
            total_features = sum(c.n_features for c in source_configs.values())
            self.unified_proj = nn.Sequential(
                nn.Linear(d_embed * len(source_configs), d_embed * 2),
                nn.LayerNorm(d_embed * 2),
                nn.GELU(),
                nn.Linear(d_embed * 2, total_features)
            )

        def encode_sources(
            self,
            features: Dict[str, torch.Tensor],
            mask: Optional[Dict[str, torch.Tensor]] = None
        ) -> Dict[str, torch.Tensor]:
            """Encode each source into embeddings."""
            embeddings = {}
            for name in self.source_names:
                if name in features:
                    # Apply mask if present (zero out masked sources)
                    x = features[name]
                    if mask is not None and name in mask:
                        x = x * mask[name].unsqueeze(-1)
                    embeddings[name] = self.encoders[name](x)
                else:
                    # Source not available - use zeros
                    batch_size = next(iter(features.values())).size(0)
                    device = next(iter(features.values())).device
                    embeddings[name] = torch.zeros(batch_size, self.d_embed, device=device)
            return embeddings

        def forward(
            self,
            features: Dict[str, torch.Tensor],
            mask: Optional[Dict[str, torch.Tensor]] = None,
            return_reconstructions: bool = True
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass with optional reconstruction.

            Args:
                features: {source_name: [batch, n_features]}
                mask: {source_name: [batch]} - 1 for present, 0 for masked
                return_reconstructions: Whether to decode back to features

            Returns:
                Dictionary with:
                - 'fused_embeddings': {source_name: [batch, d_embed]}
                - 'reconstructions': {source_name: [batch, n_features]} (if requested)
                - 'unified': [batch, total_features] - concatenated predictions
            """
            # Encode sources
            embeddings = self.encode_sources(features, mask)

            # Cross-source fusion
            fused = self.fusion(embeddings, self.source_names, mask)

            outputs = {'fused_embeddings': fused}

            # Reconstruct features
            if return_reconstructions:
                reconstructions = {}
                for name in self.source_names:
                    reconstructions[name] = self.decoders[name](fused[name])
                outputs['reconstructions'] = reconstructions

            # Unified output (all features concatenated)
            fused_concat = torch.cat([fused[name] for name in self.source_names], dim=-1)
            outputs['unified'] = self.unified_proj(fused_concat)

            return outputs

        def get_unified_features(
            self,
            features: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            """
            Get unified feature vector from all available sources.
            Used for inference/downstream tasks.
            """
            outputs = self.forward(features, mask=None, return_reconstructions=False)
            return outputs['unified']


    class CrossSourceDataset(Dataset):
        """
        Dataset for cross-source fusion training.

        Loads aligned daily features from all sources and creates
        training samples with random source masking.

        Args:
            source_configs: Dictionary of source configurations.
            train: If True, creates training split; if False, creates validation split.
            val_ratio: Proportion of data for validation (default 0.2).
            mask_prob: Probability of masking each source during training.
            temporal_gap: Number of days gap between train and validation splits
                to prevent data leakage (default 7).
            norm_stats: Pre-computed normalization statistics from training set.
                Required for validation set to prevent data leakage.
                If None and train=True, stats are computed from training data.
                If None and train=False, raises ValueError.
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            train: bool = True,
            val_ratio: float = 0.2,
            mask_prob: float = 0.2,  # Probability of masking each source
            temporal_gap: int = 7,  # Days gap between train/val to prevent leakage
            norm_stats: Optional[Dict] = None  # Pre-computed normalization stats
        ):
            self.source_configs = source_configs
            self.train = train
            self.mask_prob = mask_prob
            self.temporal_gap = temporal_gap
            self.norm_stats = norm_stats

            # Load all sources
            self.source_data = {}
            self.source_dates = {}
            self.feature_counts = {}

            print("Loading source data...")
            for name, config in source_configs.items():
                try:
                    loader = config.loader_class().load().process()
                    if hasattr(loader, 'get_daily_observations'):
                        data, dates = loader.get_daily_observations()
                    elif hasattr(loader, 'get_daily_changes'):
                        data, dates = loader.get_daily_changes()
                    else:
                        data = loader.processed_data
                        dates = loader.dates

                    self.source_data[name] = data
                    self.source_dates[name] = dates
                    self.feature_counts[name] = data.shape[1]

                    # Update config with actual feature count
                    config.n_features = data.shape[1]

                    print(f"  {name}: {len(dates)} days, {data.shape[1]} features")
                except Exception as e:
                    print(f"  Error loading {name}: {e}")

            # Find overlapping date range (without normalization - done later)
            self._align_dates(skip_normalization=True)

            # Create train/val split with temporal gap
            n_samples = len(self.aligned_dates)
            n_val = int(n_samples * val_ratio)

            # Calculate split indices with temporal gap
            train_end_idx = n_samples - n_val - temporal_gap

            if train:
                self.start_idx = 0
                self.end_idx = train_end_idx
                # Compute normalization stats from training data only
                self._compute_norm_stats(train_end_idx)
            else:
                # Validation: start after gap from train end
                self.start_idx = train_end_idx + temporal_gap
                self.end_idx = n_samples
                # Validation requires pre-computed norm_stats to prevent leakage
                if norm_stats is None:
                    raise ValueError(
                        "norm_stats must be provided for validation dataset to prevent "
                        "data leakage. Create training dataset first and pass "
                        "train_dataset.norm_stats to validation dataset."
                    )
                self.norm_stats = norm_stats

            # Apply normalization using stored stats
            self._apply_normalization()

            print(f"Dataset: {self.end_idx - self.start_idx} samples "
                  f"({'train' if train else 'val'}), temporal_gap={temporal_gap} days")

        def _compute_norm_stats(self, train_end_idx: int):
            """
            Compute normalization statistics from training data only.

            This prevents data leakage by ensuring validation/test data
            does not influence normalization parameters.

            Args:
                train_end_idx: Index marking the end of training data.
            """
            self.norm_stats = {}
            for name, data in self.aligned_data.items():
                train_data = data[:train_end_idx]
                self.norm_stats[name] = {
                    'min': train_data.min(axis=0),
                    'max': train_data.max(axis=0)
                }

        def _apply_normalization(self):
            """
            Apply stored normalization statistics to all data.

            Uses min-max normalization: (x - min) / (max - min)
            Statistics must be pre-computed via _compute_norm_stats()
            or passed in during initialization.

            IMPORTANT: Values are clipped to [0, 1] to handle out-of-distribution
            values that may occur when validation data contains values outside
            the training range (common with forward-filled sparse sources like
            hdx_rainfall where data collection methodology may change over time).
            """
            for name, data in self.aligned_data.items():
                stats = self.norm_stats[name]
                range_val = stats['max'] - stats['min']
                # CRITICAL FIX: Use minimum range of 0.1 for constant features
                # Features like deepstate with range [0,0] would otherwise cause issues
                # Also handles hdx_rainfall total_pixels which was constant at ~39k
                range_val = np.maximum(range_val, 0.1)
                normalized = (data - stats['min']) / range_val
                # CRITICAL FIX: Clip to [0, 1] to handle out-of-distribution values
                # This prevents massive MAE when validation data has values outside
                # the training range (e.g., hdx_rainfall total_pixels changed from
                # ~39k in 2022-2024 to ~98k in 2025, causing unnormalized values
                # of ~58000 that dominated the loss)
                self.aligned_data[name] = np.clip(normalized, 0.0, 1.0)

        def _align_dates(self, skip_normalization: bool = False):
            """
            Align dates across sources with different temporal resolutions.

            Strategy:
            - Use INTERSECTION of truly daily sources as primary timeline
            - Forward-fill sparse (monthly/dekadal) sources to daily resolution
            - This allows mixing daily and periodic data sources

            Args:
                skip_normalization: If True, skip inline normalization (will be
                    done separately after train/val split is computed).
            """
            # Known daily sources (manually specified for accuracy)
            # These have ~daily observations throughout the war period
            KNOWN_DAILY = {'deepstate', 'equipment', 'firms', 'viina'}
            # Known sparse/periodic sources
            KNOWN_SPARSE = {'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom', 'ucdp'}

            daily_sources = []
            sparse_sources = []

            for name, dates in self.source_dates.items():
                if name in KNOWN_DAILY:
                    daily_sources.append(name)
                elif name in KNOWN_SPARSE:
                    sparse_sources.append(name)
                elif len(dates) > 500:  # Fallback heuristic for unknown sources
                    daily_sources.append(name)
                else:
                    sparse_sources.append(name)

            print(f"  Daily sources: {daily_sources}")
            print(f"  Sparse sources: {sparse_sources}")

            # Convert all dates to datetime.date
            def parse_date(d):
                if isinstance(d, str):
                    if len(d) == 10:
                        return datetime.strptime(d, '%Y-%m-%d').date()
                    else:
                        return datetime.strptime(d, '%Y-%m').date()
                return d.date() if hasattr(d, 'date') else d

            date_sets = {}
            for name, dates in self.source_dates.items():
                date_sets[name] = set(parse_date(d) for d in dates)

            # Use intersection of DAILY sources as primary timeline
            # (sparse sources will be forward-filled)
            if daily_sources:
                primary_dates = set.intersection(*[date_sets[s] for s in daily_sources])
            else:
                # Fallback: union of all dates
                primary_dates = set.union(*date_sets.values()) if date_sets else set()

            self.aligned_dates = sorted(list(primary_dates))

            if not self.aligned_dates:
                print("  WARNING: No common dates found!")
                return

            print(f"  Aligned dates: {len(self.aligned_dates)} "
                  f"({self.aligned_dates[0]} to {self.aligned_dates[-1]})")

            # Create date-indexed data for each source
            self.aligned_data = {}
            for name in self.source_data:
                dates = self.source_dates[name]
                data = self.source_data[name]

                # Create sorted date → data mapping
                date_data_pairs = []
                for i, d in enumerate(dates):
                    dt = parse_date(d)
                    date_data_pairs.append((dt, data[i]))
                date_data_pairs.sort(key=lambda x: x[0])

                # For sparse sources: forward-fill to daily resolution
                if name in sparse_sources:
                    aligned = self._forward_fill_sparse(date_data_pairs, self.aligned_dates, data.shape[1])
                else:
                    # Daily source: direct lookup
                    date_to_idx = {dt: i for i, (dt, _) in enumerate(date_data_pairs)}
                    aligned = []
                    for date in self.aligned_dates:
                        if date in date_to_idx:
                            aligned.append(date_data_pairs[date_to_idx[date]][1])
                        else:
                            aligned.append(np.zeros(data.shape[1], dtype=np.float32))

                self.aligned_data[name] = np.array(aligned, dtype=np.float32)

                # Note: Normalization is now handled separately via _compute_norm_stats
                # and _apply_normalization to prevent train/val data leakage.
                # The skip_normalization parameter controls this behavior.
                if not skip_normalization:
                    # Legacy behavior for backward compatibility (not recommended)
                    for i in range(self.aligned_data[name].shape[1]):
                        col = self.aligned_data[name][:, i]
                        col_min, col_max = col.min(), col.max()
                        if col_max > col_min:
                            self.aligned_data[name][:, i] = (col - col_min) / (col_max - col_min)

        def _forward_fill_sparse(self, date_data_pairs, target_dates, n_features):
            """
            Forward-fill sparse data to match target daily dates.

            For each target date, use the most recent observation that is <= target date.
            """
            aligned = []
            sparse_idx = 0
            n_sparse = len(date_data_pairs)

            for target_date in target_dates:
                # Advance sparse_idx to find most recent observation <= target_date
                while sparse_idx < n_sparse - 1 and date_data_pairs[sparse_idx + 1][0] <= target_date:
                    sparse_idx += 1

                # Use this observation if it's <= target_date, else zeros
                if sparse_idx < n_sparse and date_data_pairs[sparse_idx][0] <= target_date:
                    aligned.append(date_data_pairs[sparse_idx][1])
                else:
                    aligned.append(np.zeros(n_features, dtype=np.float32))

            return aligned

        def __len__(self):
            return self.end_idx - self.start_idx

        def __getitem__(self, idx):
            actual_idx = self.start_idx + idx

            # Get features for this day
            features = {}
            for name in self.aligned_data:
                features[name] = torch.tensor(
                    self.aligned_data[name][actual_idx],
                    dtype=torch.float32
                )

            # Create random mask (for training)
            # Each source has mask_prob chance of being masked
            mask = {}
            masked_source = None

            if self.train:
                # For training: mask exactly one source
                masked_source = random.choice(list(features.keys()))
                for name in features:
                    mask[name] = torch.tensor(0.0 if name == masked_source else 1.0)
            else:
                # For validation: no masking (or could mask systematically)
                for name in features:
                    mask[name] = torch.tensor(1.0)

            return features, mask, masked_source if masked_source else ''


    class UnifiedTrainer:
        """
        Trainer for the unified cross-source model.

        Training objective: Reconstruct masked source from others.

        Uses WarmupCosineScheduler for better training dynamics:
        - Linear warmup for stable early training
        - Cosine annealing for smooth convergence
        """

        def __init__(
            self,
            model: UnifiedInterpolationModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            device: str = 'cpu',
            epochs: int = 100,
            warmup_epochs: int = 10
        ):
            self.model = model.to(device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.epochs = epochs

            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            # Use WarmupCosineScheduler for better training dynamics
            self.scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
                warmup_start_lr=lr * 0.01,  # Start at 1% of target LR
                min_lr=1e-7
            )

        def train_epoch(self) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            total_recon_loss = 0
            source_losses = {name: 0.0 for name in self.model.source_names}
            n_batches = 0

            for features, mask, masked_sources in self.train_loader:
                # Move to device
                features = {k: v.to(self.device) for k, v in features.items()}
                mask = {k: v.to(self.device) for k, v in mask.items()}

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features, mask, return_reconstructions=True)

                # Reconstruction loss (only for masked sources)
                recon_loss = 0
                for i, (name, recon) in enumerate(outputs['reconstructions'].items()):
                    target = features[name]
                    source_mask = mask[name]

                    # Loss only on masked sources
                    # source_mask is 0 for masked, 1 for present
                    masked_samples = (source_mask == 0).float()

                    if masked_samples.sum() > 0:
                        loss = F.mse_loss(recon, target, reduction='none').mean(dim=-1)
                        # Weight by mask (only count masked sources)
                        weighted_loss = (loss * masked_samples).sum() / (masked_samples.sum() + 1e-8)
                        recon_loss = recon_loss + weighted_loss
                        source_losses[name] += weighted_loss.item()

                loss = recon_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss
                n_batches += 1

            return {
                'total': total_loss / max(n_batches, 1),
                'reconstruction': total_recon_loss / max(n_batches, 1),
                **{f'{name}_loss': v / max(n_batches, 1) for name, v in source_losses.items()}
            }

        def validate(self) -> Dict[str, float]:
            """Validate model - test reconstruction for each source."""
            self.model.eval()
            source_maes = {name: [] for name in self.model.source_names}

            with torch.no_grad():
                for features, mask, _ in self.val_loader:
                    features = {k: v.to(self.device) for k, v in features.items()}

                    # Test reconstruction for each source (mask one at a time)
                    for masked_source in self.model.source_names:
                        # Create mask
                        test_mask = {
                            name: torch.ones(features[name].size(0), device=self.device)
                            if name != masked_source else
                            torch.zeros(features[name].size(0), device=self.device)
                            for name in features
                        }

                        outputs = self.model(features, test_mask, return_reconstructions=True)

                        # Compute MAE for masked source
                        recon = outputs['reconstructions'][masked_source]
                        target = features[masked_source]
                        mae = (recon - target).abs().mean().item()
                        source_maes[masked_source].append(mae)

            results = {
                f'{name}_mae': np.mean(maes) for name, maes in source_maes.items()
            }
            results['mean_mae'] = np.mean([np.mean(maes) for maes in source_maes.values()])

            return results

        def train(
            self,
            epochs: int = 100,
            patience: int = 20,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """Full training loop."""
            history = {
                'train_loss': [],
                'val_mae': [],
            }

            best_val_mae = float('inf')
            best_epoch = 0
            patience_counter = 0

            print(f"\nTraining for up to {epochs} epochs...")
            print("-" * 70)

            for epoch in range(epochs):
                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                history['train_loss'].append(train_metrics['total'])
                history['val_mae'].append(val_metrics['mean_mae'])

                # WarmupCosineScheduler steps per epoch (not based on val metric)
                self.scheduler.step()

                if val_metrics['mean_mae'] < best_val_mae:
                    best_val_mae = val_metrics['mean_mae']
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        UNIFIED_INTERP_MODEL
                    )
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    source_maes = ' | '.join([
                        f"{name[:4]}:{val_metrics[f'{name}_mae']:.4f}"
                        for name in self.model.source_names
                    ])
                    marker = '*' if epoch == best_epoch else ''
                    print(f"Epoch {epoch:3d}: loss={train_metrics['total']:.4f}, "
                          f"val_mae={val_metrics['mean_mae']:.4f} {marker}")
                    print(f"           {source_maes}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break

            print("-" * 70)
            print(f"Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}")

            return history


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_unified_model(
    load_pretrained_jims: bool = True,
    d_embed: int = 64
) -> UnifiedInterpolationModel:
    """Create unified model, optionally with pretrained JIM encoders."""

    # Update configs with actual feature counts
    dataset = CrossSourceDataset(SOURCE_CONFIGS, train=True)

    # Load pretrained JIMs if requested
    pretrained_jims = {}
    if load_pretrained_jims:
        for name, config in SOURCE_CONFIGS.items():
            jim_path = get_interp_model_path(config.jim_config_key)
            if jim_path.exists():
                try:
                    jim_config = INTERPOLATION_CONFIGS.get(config.jim_config_key)
                    if jim_config:
                        # Create JIM with correct feature count
                        jim_config_actual = InterpolationConfig(
                            name=jim_config.name,
                            source=jim_config.source,
                            features=['f' + str(i) for i in range(config.n_features)],
                            native_resolution_days=jim_config.native_resolution_days,
                            d_model=d_embed,
                            nhead=jim_config.nhead,
                            num_layers=jim_config.num_layers,
                        )
                        jim = JointInterpolationModel(jim_config_actual)
                        jim.load_state_dict(torch.load(jim_path, map_location='cpu'))
                        pretrained_jims[name] = jim
                        print(f"Loaded pretrained JIM for {name}")
                except Exception as e:
                    print(f"Could not load JIM for {name}: {e}")

    model = UnifiedInterpolationModel(
        source_configs=SOURCE_CONFIGS,
        d_embed=d_embed,
        nhead=4,
        num_fusion_layers=2,
        dropout=0.1,
        pretrained_jims=pretrained_jims if pretrained_jims else None
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nUnified model created with {n_params:,} parameters")

    return model


def train_unified_model(
    epochs: int = 100,
    batch_size: int = 32,
    device: str = None
) -> Dict[str, Any]:
    """Train the unified cross-source model."""

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print("=" * 70)
    print("UNIFIED CROSS-SOURCE INTERPOLATION MODEL")
    print(f"Device: {device}")
    print("=" * 70)

    # Create datasets - IMPORTANT: Create training dataset first to compute
    # normalization statistics, then pass those stats to validation dataset
    # to prevent data leakage.
    train_dataset = CrossSourceDataset(
        SOURCE_CONFIGS,
        train=True,
        temporal_gap=7  # 7-day gap between train and validation
    )
    val_dataset = CrossSourceDataset(
        SOURCE_CONFIGS,
        train=False,
        temporal_gap=7,
        norm_stats=train_dataset.norm_stats  # Use training stats for validation
    )

    # Custom collate function
    def collate_fn(batch):
        features = {name: [] for name in SOURCE_CONFIGS}
        masks = {name: [] for name in SOURCE_CONFIGS}
        masked_sources = []

        for feat, mask, masked in batch:
            for name in features:
                features[name].append(feat[name])
                masks[name].append(mask[name])
            masked_sources.append(masked)

        return (
            {name: torch.stack(tensors) for name, tensors in features.items()},
            {name: torch.stack(tensors) for name, tensors in masks.items()},
            masked_sources
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    model = create_unified_model(load_pretrained_jims=True)

    # Train
    trainer = UnifiedTrainer(
        model, train_loader, val_loader,
        lr=1e-4, weight_decay=0.01, device=device,
        epochs=epochs, warmup_epochs=10
    )

    history = trainer.train(epochs=epochs, patience=20, verbose=True)

    return {
        'model': model,
        'history': history,
        'source_configs': SOURCE_CONFIGS
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Unified Cross-Source Interpolation')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device')
    parser.add_argument('--summary', action='store_true', help='Print model summary')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    if args.train:
        train_unified_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    elif args.summary:
        model = create_unified_model(load_pretrained_jims=False)
        print("\nModel architecture:")
        print(model)
    else:
        print("Usage:")
        print("  --train     Train the unified model")
        print("  --summary   Print model summary")


if __name__ == "__main__":
    main()
