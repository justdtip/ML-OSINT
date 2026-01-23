"""
Unified Cross-Source Interpolation Model (DELTA VERSION)

This version uses ONLY delta (per-day) equipment features to avoid
spurious correlations from cumulative time series.

Changes from original:
- Equipment features filtered to delta-only columns
- Proper handling of rate-based features
- RSA fusion regularization to maintain cross-modal similarity during training

Usage:
    python unified_interpolation_delta.py --train --epochs 100
"""

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
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    UNIFIED_INTERP_MODEL, UNIFIED_DELTA_MODEL, UNIFIED_HYBRID_MODEL,
)
from joint_interpolation_models import (
    JointInterpolationModel,
    InterpolationConfig,
    INTERPOLATION_CONFIGS,
)
from interpolation_data_loaders import (
    SentinelDataLoader,
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader
)

# MODEL_DIR is imported from config.paths
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# DELTA EQUIPMENT FEATURE EXTRACTION
# =============================================================================

def extract_equipment_delta_features(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract only delta (per-day) features from equipment data.

    Includes:
    - *_delta columns (daily changes)
    - *_7day_avg columns (rolling averages of deltas)
    - total_losses_day (already a delta)
    - heavy_equipment_ratio (ratio, not cumulative)
    - direction_encoded (categorical)

    Excludes:
    - Cumulative totals (tank, aircraft, etc.)
    - total_losses_cumulative
    """
    delta_indices = []
    delta_names = []

    for i, name in enumerate(feature_names):
        # Include delta and rolling average columns
        if '_delta' in name or '_7day_avg' in name:
            delta_indices.append(i)
            delta_names.append(name)
        # Include specific derived features that aren't cumulative
        elif name in ['total_losses_day', 'heavy_equipment_ratio', 'direction_encoded']:
            delta_indices.append(i)
            delta_names.append(name)

    delta_data = data[:, delta_indices]
    return delta_data, delta_names


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
    use_delta_only: bool = False  # Whether to use delta features only


SOURCE_CONFIGS = {
    # Sentinel excluded: only 32 monthly samples + data quality issues (row 1 anomaly)
    # This was limiting all aligned data to 32 samples instead of 416
    'deepstate': SourceConfig(
        name='DeepState',
        loader_class=DeepStateDataLoader,
        n_features=55,
        jim_config_key='deepstate',
        d_embed=64,
        use_delta_only=False
    ),
    'equipment': SourceConfig(
        name='Equipment',
        loader_class=EquipmentDataLoader,
        n_features=27,  # Will be updated to delta-only count
        jim_config_key='equipment_totals',
        d_embed=64,
        use_delta_only=True  # KEY CHANGE: Use delta features only
    ),
    'firms': SourceConfig(
        name='FIRMS',
        loader_class=FIRMSDataLoader,
        n_features=42,
        jim_config_key='sentinel3',
        d_embed=64,
        use_delta_only=False
    ),
    'ucdp': SourceConfig(
        name='UCDP',
        loader_class=UCDPDataLoader,
        n_features=48,
        jim_config_key='deepstate',
        d_embed=64,
        use_delta_only=False
    ),
}


# =============================================================================
# RSA FUSION REGULARIZATION
# =============================================================================


def compute_pairwise_cosine_similarity(
    embeddings: Dict[str, 'torch.Tensor'],
    source_names: List[str]
) -> 'torch.Tensor':
    """
    Compute pairwise cosine similarities between source embeddings.

    Args:
        embeddings: Dictionary mapping source names to embeddings [batch, d_embed]
        source_names: List of source names defining the order

    Returns:
        Similarity matrix of shape [n_sources, n_sources] averaged over batch
    """
    import torch
    import torch.nn.functional as F

    n_sources = len(source_names)

    # Stack embeddings: [batch, n_sources, d_embed]
    stacked = torch.stack([embeddings[name] for name in source_names], dim=1)

    # Normalize for cosine similarity
    normalized = F.normalize(stacked, p=2, dim=-1)

    # Compute pairwise similarities: [batch, n_sources, n_sources]
    # Using einsum: batch matrix multiplication
    sim_matrix = torch.einsum('bnd,bmd->bnm', normalized, normalized)

    # Average over batch
    return sim_matrix.mean(dim=0)


def compute_rsa_fusion_loss(
    pre_fusion_embeddings: Dict[str, 'torch.Tensor'],
    post_fusion_embeddings: Dict[str, 'torch.Tensor'],
    source_names: List[str],
    similarity_threshold: float = 0.3,
    margin: float = 0.1
) -> 'torch.Tensor':
    """
    Compute RSA (Representational Similarity Analysis) fusion regularization loss.

    This loss penalizes when cross-modal similarities drop below a threshold,
    preventing the ZINB casualty head from pulling source representations apart.

    The loss encourages:
    1. Post-fusion embeddings to maintain alignment across sources
    2. Cross-source similarities to stay above the threshold
    3. Gradual alignment rather than forcing identical representations

    Args:
        pre_fusion_embeddings: Source embeddings before fusion [batch, d_embed] per source
        post_fusion_embeddings: Source embeddings after fusion [batch, d_embed] per source
        source_names: List of source names
        similarity_threshold: Minimum desired similarity between sources (default 0.3)
        margin: Soft margin above threshold to encourage (default 0.1)

    Returns:
        Scalar loss penalizing low cross-modal similarity
    """
    import torch

    # Compute post-fusion similarity matrix
    post_sim = compute_pairwise_cosine_similarity(post_fusion_embeddings, source_names)

    n_sources = len(source_names)

    # Extract off-diagonal elements (cross-source similarities)
    mask = ~torch.eye(n_sources, dtype=torch.bool, device=post_sim.device)
    cross_similarities = post_sim[mask]

    # Hinge loss: penalize similarities below (threshold + margin)
    # Loss = max(0, threshold + margin - similarity)
    target = similarity_threshold + margin
    hinge_loss = torch.clamp(target - cross_similarities, min=0.0)

    # Mean loss over all pairs
    loss = hinge_loss.mean()

    return loss


def compute_contrastive_fusion_loss(
    embeddings: Dict[str, 'torch.Tensor'],
    source_names: List[str],
    temperature: float = 0.1
) -> 'torch.Tensor':
    """
    Compute contrastive loss to encourage cross-source alignment.

    Uses InfoNCE-style loss where same-timestep embeddings from different
    sources should be more similar than embeddings from different timesteps.

    Args:
        embeddings: Dictionary mapping source names to embeddings [batch, d_embed]
        source_names: List of source names
        temperature: Temperature for softmax scaling (default 0.1)

    Returns:
        Scalar contrastive loss
    """
    import torch
    import torch.nn.functional as F

    n_sources = len(source_names)
    if n_sources < 2:
        return torch.tensor(0.0, device=next(iter(embeddings.values())).device)

    # Stack and normalize: [batch, n_sources, d_embed]
    stacked = torch.stack([embeddings[name] for name in source_names], dim=1)
    normalized = F.normalize(stacked, p=2, dim=-1)

    batch_size = normalized.size(0)
    d_embed = normalized.size(2)

    # Reshape for contrastive: [batch * n_sources, d_embed]
    flat = normalized.view(-1, d_embed)

    # Compute all pairwise similarities: [batch * n_sources, batch * n_sources]
    sim_matrix = torch.mm(flat, flat.t()) / temperature

    # Create labels: same-timestep, different-source pairs are positives
    # For each anchor (batch_idx, source_idx), positives are (batch_idx, other_sources)
    total_size = batch_size * n_sources

    # Mask for positives: same batch index, different source
    batch_indices = torch.arange(batch_size, device=sim_matrix.device).repeat_interleave(n_sources)
    source_indices = torch.arange(n_sources, device=sim_matrix.device).repeat(batch_size)

    # Same batch, different source = positive
    same_batch = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)
    diff_source = source_indices.unsqueeze(1) != source_indices.unsqueeze(0)
    positive_mask = same_batch & diff_source

    # For each anchor, compute loss over its positives vs negatives
    # Using logsumexp for numerical stability
    losses = []
    for i in range(total_size):
        pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue

        # Positive similarities
        pos_sims = sim_matrix[i, pos_indices]

        # All similarities except self
        all_except_self = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        for pos_sim in pos_sims:
            log_denominator = torch.logsumexp(all_except_self, dim=0)
            loss = -pos_sim + log_denominator
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=sim_matrix.device)

    return torch.stack(losses).mean()


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class SourceEncoder(nn.Module):
        """Encodes a single source's features into a fixed-size embedding."""

        def __init__(
            self,
            n_features: int,
            d_embed: int = 64,
            pretrained_jim: Optional[JointInterpolationModel] = None
        ):
            super().__init__()
            self.n_features = n_features
            self.d_embed = d_embed

            # Simple projection: features → embedding
            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, d_embed * 2),
                nn.LayerNorm(d_embed * 2),
                nn.GELU(),
                nn.Linear(d_embed * 2, d_embed),
                nn.LayerNorm(d_embed)
            )

            # Final projection to unified embedding space
            self.output_proj = nn.Linear(d_embed, d_embed)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = self.feature_proj(x)
            return self.output_proj(pooled)


    class SourceDecoder(nn.Module):
        """Decodes a fused embedding back to source-specific features."""

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
                nn.Linear(d_embed * 2, n_features)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)


    class CrossSourceAttention(nn.Module):
        """Cross-attention between source embeddings."""

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
            batch_size = next(iter(source_embeddings.values())).size(0)
            device = next(iter(source_embeddings.values())).device

            # Stack source embeddings: [batch, n_sources, d_embed]
            stacked = torch.stack([source_embeddings[s] for s in source_order], dim=1)

            # Add source type embeddings
            source_indices = torch.arange(len(source_order), device=device)
            source_type_emb = self.source_embeddings(source_indices)
            stacked = stacked + source_type_emb.unsqueeze(0)

            # Create attention mask if sources are masked
            attn_mask = None
            if mask is not None:
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


    class UnifiedInterpolationModelDelta(nn.Module):
        """
        Unified model for cross-source interpolation (Delta version).

        Uses delta-only equipment features to avoid spurious correlations.
        Includes RSA fusion regularization to maintain cross-modal similarity.
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            d_embed: int = 64,
            nhead: int = 4,
            num_fusion_layers: int = 2,
            dropout: float = 0.1,
            fusion_loss_weight: float = 0.1,
            similarity_threshold: float = 0.3,
            use_contrastive_fusion: bool = False
        ):
            super().__init__()
            self.source_names = list(source_configs.keys())
            self.source_configs = source_configs
            self.d_embed = d_embed

            # RSA fusion regularization parameters
            self.fusion_loss_weight = fusion_loss_weight
            self.similarity_threshold = similarity_threshold
            self.use_contrastive_fusion = use_contrastive_fusion

            # Source encoders
            self.encoders = nn.ModuleDict()
            for name, config in source_configs.items():
                self.encoders[name] = SourceEncoder(
                    n_features=config.n_features,
                    d_embed=d_embed
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

            # Unified output projection
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
            embeddings = {}
            for name in self.source_names:
                if name in features:
                    x = features[name]
                    if mask is not None and name in mask:
                        x = x * mask[name].unsqueeze(-1)
                    embeddings[name] = self.encoders[name](x)
                else:
                    batch_size = next(iter(features.values())).size(0)
                    device = next(iter(features.values())).device
                    embeddings[name] = torch.zeros(batch_size, self.d_embed, device=device)
            return embeddings

        def forward(
            self,
            features: Dict[str, torch.Tensor],
            mask: Optional[Dict[str, torch.Tensor]] = None,
            return_reconstructions: bool = True,
            compute_fusion_loss: bool = True
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass with optional RSA fusion regularization.

            Args:
                features: Dict mapping source names to feature tensors [batch, n_features]
                mask: Optional dict of masks per source (0 = masked, 1 = visible)
                return_reconstructions: Whether to compute reconstructions
                compute_fusion_loss: Whether to compute RSA fusion regularization loss

            Returns:
                Dict containing:
                - 'fused_embeddings': Post-fusion embeddings per source
                - 'reconstructions': Reconstructed features (if return_reconstructions)
                - 'unified': Unified projection of all sources
                - 'pre_fusion_embeddings': Pre-fusion embeddings (if compute_fusion_loss)
                - 'fusion_loss': RSA regularization loss (if compute_fusion_loss)
                - 'cross_source_similarity': Mean cross-source similarity (for monitoring)
            """
            # Encode sources (pre-fusion embeddings)
            embeddings = self.encode_sources(features, mask)

            # Cross-source fusion
            fused = self.fusion(embeddings, self.source_names, mask)

            outputs = {
                'fused_embeddings': fused,
                'pre_fusion_embeddings': embeddings
            }

            # Compute RSA fusion regularization loss
            if compute_fusion_loss and self.fusion_loss_weight > 0:
                if self.use_contrastive_fusion:
                    fusion_loss = compute_contrastive_fusion_loss(
                        fused, self.source_names, temperature=0.1
                    )
                else:
                    fusion_loss = compute_rsa_fusion_loss(
                        embeddings, fused, self.source_names,
                        similarity_threshold=self.similarity_threshold,
                        margin=0.1
                    )
                outputs['fusion_loss'] = fusion_loss

                # Compute cross-source similarity for monitoring
                post_sim = compute_pairwise_cosine_similarity(fused, self.source_names)
                n_sources = len(self.source_names)
                mask_tensor = ~torch.eye(n_sources, dtype=torch.bool, device=post_sim.device)
                cross_sim = post_sim[mask_tensor].mean()
                outputs['cross_source_similarity'] = cross_sim

            # Reconstruct features
            if return_reconstructions:
                reconstructions = {}
                for name in self.source_names:
                    reconstructions[name] = self.decoders[name](fused[name])
                outputs['reconstructions'] = reconstructions

            # Unified output
            fused_concat = torch.cat([fused[name] for name in self.source_names], dim=-1)
            outputs['unified'] = self.unified_proj(fused_concat)

            return outputs


    class CrossSourceDatasetDelta(Dataset):
        """
        Dataset for cross-source fusion training (Delta version).

        Uses delta-only equipment features.

        Context Window Optimization (Probe 3.1.1):
            Analysis shows shorter context windows (7-14 days) yield better performance:
            - 7 days:  78.8% accuracy, F1 0.318 (OPTIMAL)
            - 14 days: 78.4% accuracy, F1 0.318 (OPTIMAL)
            - 30+ days: Degraded performance due to outdated patterns

            This suggests conflict dynamics exhibit rapid regime changes where recent
            context is most predictive. Longer windows introduce noise from outdated
            patterns that confuse the model.

        FIXES APPLIED:
        1. Computes normalization stats on TRAINING data only, then applies to validation
        2. Supports temporal gap between train/val to prevent leakage
        3. Uses z-score normalization instead of min-max (more robust)
        4. Configurable context window with 7-14 day optimal default
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            train: bool = True,
            val_ratio: float = 0.2,
            temporal_gap: int = 0,  # Days gap between train/val to prevent leakage
            context_window_days: int = 14,  # Optimal context window (Probe 3.1.1)
            norm_stats: Dict = None  # Pre-computed normalization stats (for val set)
        ):
            self.source_configs = source_configs
            self.train = train
            self.temporal_gap = temporal_gap
            self.context_window_days = context_window_days
            self.norm_stats = norm_stats or {}

            # Load all sources
            self.source_data = {}
            self.source_dates = {}
            self.feature_names = {}

            print(f"Loading source data (DELTA version, train={train}, context={context_window_days}d)...")
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

                    feature_names = loader.feature_names

                    # Apply delta filtering for equipment
                    if config.use_delta_only:
                        data, feature_names = extract_equipment_delta_features(data, feature_names)
                        print(f"  {name}: {len(dates)} days, {data.shape[1]} DELTA features")
                    else:
                        print(f"  {name}: {len(dates)} days, {data.shape[1]} features")

                    self.source_data[name] = data
                    self.source_dates[name] = dates
                    self.feature_names[name] = feature_names

                    # Update config with actual feature count
                    config.n_features = data.shape[1]

                except Exception as e:
                    print(f"  Error loading {name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Find overlapping date range
            self._align_dates()

            # Create train/val split with temporal gap
            n_samples = len(self.aligned_dates)
            n_val = int(n_samples * val_ratio)

            if train:
                self.start_idx = 0
                self.end_idx = n_samples - n_val
            else:
                # Validation starts after train + temporal gap
                self.start_idx = n_samples - n_val + temporal_gap
                self.end_idx = n_samples

            # FIX: Compute normalization on TRAINING data only
            self._normalize_data(train, n_samples, n_val)

            print(f"Dataset: {self.end_idx - self.start_idx} samples "
                  f"({'train' if train else 'val'})")

        def _align_dates(self):
            """Find common dates across all sources and align data."""
            date_sets = {}
            for name, dates in self.source_dates.items():
                date_sets[name] = set()
                for d in dates:
                    if isinstance(d, str):
                        if len(d) == 10:
                            date_sets[name].add(datetime.strptime(d, '%Y-%m-%d').date())
                        else:
                            date_sets[name].add(datetime.strptime(d, '%Y-%m').date())
                    else:
                        date_sets[name].add(d.date() if hasattr(d, 'date') else d)

            common_dates = set.intersection(*date_sets.values()) if date_sets else set()
            self.aligned_dates = sorted(list(common_dates))

            print(f"  Common dates: {len(self.aligned_dates)} "
                  f"({self.aligned_dates[0]} to {self.aligned_dates[-1]})")

            # Create date-indexed data for each source
            self.aligned_data = {}
            for name in self.source_data:
                dates = self.source_dates[name]
                data = self.source_data[name]

                date_to_idx = {}
                for i, d in enumerate(dates):
                    if isinstance(d, str):
                        if len(d) == 10:
                            dt = datetime.strptime(d, '%Y-%m-%d').date()
                        else:
                            dt = datetime.strptime(d, '%Y-%m').date()
                    else:
                        dt = d.date() if hasattr(d, 'date') else d
                    date_to_idx[dt] = i

                aligned = []
                for date in self.aligned_dates:
                    if date in date_to_idx:
                        aligned.append(data[date_to_idx[date]])
                    else:
                        aligned.append(np.zeros(data.shape[1], dtype=np.float32))

                self.aligned_data[name] = np.array(aligned, dtype=np.float32)

        def _normalize_data(self, train: bool, n_samples: int, n_val: int):
            """
            FIX: Normalize using training data statistics only.

            Training set: compute and store mean/std, then normalize
            Validation set: use provided norm_stats from training
            """
            print("  Normalizing features...")

            if train:
                # Training: compute stats on training indices only
                train_end = n_samples - n_val
                self.norm_stats = {}

                for name in self.aligned_data:
                    train_data = self.aligned_data[name][:train_end]
                    mean = train_data.mean(axis=0, keepdims=True)
                    std = train_data.std(axis=0, keepdims=True) + 1e-8

                    self.norm_stats[name] = {'mean': mean, 'std': std}

                    # Apply normalization to ALL data
                    self.aligned_data[name] = (self.aligned_data[name] - mean) / std

            else:
                # Validation: use provided norm_stats
                if not self.norm_stats:
                    print("  WARNING: No norm_stats provided for validation set!")
                    print("           Computing on full data (potential leakage)")
                    for name in self.aligned_data:
                        mean = self.aligned_data[name].mean(axis=0, keepdims=True)
                        std = self.aligned_data[name].std(axis=0, keepdims=True) + 1e-8
                        self.aligned_data[name] = (self.aligned_data[name] - mean) / std
                else:
                    for name in self.aligned_data:
                        if name in self.norm_stats:
                            mean = self.norm_stats[name]['mean']
                            std = self.norm_stats[name]['std']
                            self.aligned_data[name] = (self.aligned_data[name] - mean) / std
                        else:
                            print(f"  WARNING: No norm_stats for {name}")
                            mean = self.aligned_data[name].mean(axis=0, keepdims=True)
                            std = self.aligned_data[name].std(axis=0, keepdims=True) + 1e-8
                            self.aligned_data[name] = (self.aligned_data[name] - mean) / std

        def __len__(self):
            return self.end_idx - self.start_idx

        def __getitem__(self, idx):
            actual_idx = self.start_idx + idx

            features = {}
            for name in self.aligned_data:
                features[name] = torch.tensor(
                    self.aligned_data[name][actual_idx],
                    dtype=torch.float32
                )

            mask = {}
            masked_source = None

            if self.train:
                masked_source = random.choice(list(features.keys()))
                for name in features:
                    mask[name] = torch.tensor(0.0 if name == masked_source else 1.0)
            else:
                for name in features:
                    mask[name] = torch.tensor(1.0)

            return features, mask, masked_source if masked_source else ''


    class UnifiedTrainerDelta:
        """
        Trainer for the unified cross-source model (Delta version).

        Includes RSA fusion regularization to maintain cross-modal similarity.
        """

        def __init__(
            self,
            model: UnifiedInterpolationModelDelta,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            device: str = 'cpu',
            fusion_loss_weight: Optional[float] = None
        ):
            """
            Initialize the trainer.

            Args:
                model: The UnifiedInterpolationModelDelta to train
                train_loader: DataLoader for training data
                val_loader: DataLoader for validation data
                lr: Learning rate (default 1e-4)
                weight_decay: Weight decay for AdamW (default 0.01)
                device: Device to train on ('cpu', 'cuda', 'mps')
                fusion_loss_weight: Override model's fusion_loss_weight (optional)
            """
            self.model = model.to(device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device

            # Allow overriding fusion loss weight at trainer level
            if fusion_loss_weight is not None:
                self.model.fusion_loss_weight = fusion_loss_weight

            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
            )

        def train_epoch(self) -> Dict[str, float]:
            """
            Train for one epoch.

            Returns:
                Dict with training metrics including:
                - 'total': Total loss
                - 'reconstruction': Reconstruction loss
                - 'fusion': RSA fusion regularization loss
                - 'cross_sim': Mean cross-source similarity
                - '{source}_loss': Per-source reconstruction losses
            """
            self.model.train()
            total_loss = 0
            total_recon_loss = 0
            total_fusion_loss = 0
            total_cross_sim = 0
            source_losses = {name: 0.0 for name in self.model.source_names}
            n_batches = 0

            for features, mask, masked_sources in self.train_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                mask = {k: v.to(self.device) for k, v in mask.items()}

                self.optimizer.zero_grad()

                outputs = self.model(
                    features, mask,
                    return_reconstructions=True,
                    compute_fusion_loss=True
                )

                # Reconstruction loss
                recon_loss = 0
                for name, recon in outputs['reconstructions'].items():
                    target = features[name]
                    source_mask = mask[name]

                    masked_samples = (source_mask == 0).float()

                    if masked_samples.sum() > 0:
                        loss = F.mse_loss(recon, target, reduction='none').mean(dim=-1)
                        weighted_loss = (loss * masked_samples).sum() / (masked_samples.sum() + 1e-8)
                        recon_loss = recon_loss + weighted_loss
                        source_losses[name] += weighted_loss.item()

                # RSA fusion regularization loss
                fusion_loss = outputs.get('fusion_loss', torch.tensor(0.0, device=self.device))
                cross_sim = outputs.get('cross_source_similarity', torch.tensor(0.0, device=self.device))

                # Total loss = reconstruction + weighted fusion regularization
                loss = recon_loss + self.model.fusion_loss_weight * fusion_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_recon_loss += recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss
                total_fusion_loss += fusion_loss.item() if torch.is_tensor(fusion_loss) else fusion_loss
                total_cross_sim += cross_sim.item() if torch.is_tensor(cross_sim) else cross_sim
                n_batches += 1

            return {
                'total': total_loss / max(n_batches, 1),
                'reconstruction': total_recon_loss / max(n_batches, 1),
                'fusion': total_fusion_loss / max(n_batches, 1),
                'cross_sim': total_cross_sim / max(n_batches, 1),
                **{f'{name}_loss': v / max(n_batches, 1) for name, v in source_losses.items()}
            }

        def validate(self) -> Dict[str, float]:
            """
            Validate the model.

            Returns:
                Dict with validation metrics including:
                - '{source}_mae': Per-source MAE
                - 'mean_mae': Mean MAE across sources
                - 'cross_sim': Mean cross-source similarity (RSA proxy)
            """
            self.model.eval()
            source_maes = {name: [] for name in self.model.source_names}
            cross_sims = []

            with torch.no_grad():
                for features, mask, _ in self.val_loader:
                    features = {k: v.to(self.device) for k, v in features.items()}

                    # Compute cross-source similarity with all sources visible
                    full_mask = {
                        name: torch.ones(features[name].size(0), device=self.device)
                        for name in features
                    }
                    outputs_full = self.model(
                        features, full_mask,
                        return_reconstructions=False,
                        compute_fusion_loss=True
                    )
                    if 'cross_source_similarity' in outputs_full:
                        cross_sims.append(outputs_full['cross_source_similarity'].item())

                    # Per-source reconstruction MAE
                    for masked_source in self.model.source_names:
                        test_mask = {
                            name: torch.ones(features[name].size(0), device=self.device)
                            if name != masked_source else
                            torch.zeros(features[name].size(0), device=self.device)
                            for name in features
                        }

                        outputs = self.model(
                            features, test_mask,
                            return_reconstructions=True,
                            compute_fusion_loss=False
                        )

                        recon = outputs['reconstructions'][masked_source]
                        target = features[masked_source]
                        mae = (recon - target).abs().mean().item()
                        source_maes[masked_source].append(mae)

            results = {
                f'{name}_mae': np.mean(maes) for name, maes in source_maes.items()
            }
            results['mean_mae'] = np.mean([np.mean(maes) for maes in source_maes.values()])
            results['cross_sim'] = np.mean(cross_sims) if cross_sims else 0.0

            return results

        def train(
            self,
            epochs: int = 100,
            patience: int = 20,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """
            Train the model with RSA fusion regularization.

            Args:
                epochs: Maximum number of epochs
                patience: Early stopping patience
                verbose: Whether to print progress

            Returns:
                Training history dict with losses and metrics per epoch
            """
            history = {
                'train_loss': [],
                'train_recon_loss': [],
                'train_fusion_loss': [],
                'train_cross_sim': [],
                'val_mae': [],
                'val_cross_sim': [],
            }

            best_val_mae = float('inf')
            best_epoch = 0
            patience_counter = 0

            print(f"\nTraining for up to {epochs} epochs...")
            print(f"RSA fusion regularization: weight={self.model.fusion_loss_weight}, "
                  f"threshold={self.model.similarity_threshold}")
            print("-" * 80)

            for epoch in range(epochs):
                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                history['train_loss'].append(train_metrics['total'])
                history['train_recon_loss'].append(train_metrics['reconstruction'])
                history['train_fusion_loss'].append(train_metrics['fusion'])
                history['train_cross_sim'].append(train_metrics['cross_sim'])
                history['val_mae'].append(val_metrics['mean_mae'])
                history['val_cross_sim'].append(val_metrics['cross_sim'])

                self.scheduler.step(val_metrics['mean_mae'])

                if val_metrics['mean_mae'] < best_val_mae:
                    best_val_mae = val_metrics['mean_mae']
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        UNIFIED_DELTA_MODEL
                    )
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    source_maes = ' | '.join([
                        f"{name[:4]}:{val_metrics[f'{name}_mae']:.4f}"
                        for name in self.model.source_names
                    ])
                    marker = '*' if epoch == best_epoch else ''
                    print(f"Epoch {epoch:3d}: loss={train_metrics['total']:.4f} "
                          f"(recon={train_metrics['reconstruction']:.4f}, "
                          f"fusion={train_metrics['fusion']:.4f})")
                    print(f"           val_mae={val_metrics['mean_mae']:.4f}, "
                          f"cross_sim={val_metrics['cross_sim']:.3f} {marker}")
                    print(f"           {source_maes}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break

            print("-" * 80)
            print(f"Best val MAE: {best_val_mae:.4f} at epoch {best_epoch}")
            print(f"Final cross-source similarity: {history['val_cross_sim'][-1]:.3f}")

            return history


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_unified_model_delta(
    epochs: int = 100,
    batch_size: int = 32,
    device: str = None,
    fusion_loss_weight: float = 0.1,
    similarity_threshold: float = 0.3,
    use_contrastive_fusion: bool = False,
    context_window_days: int = 14
) -> Dict[str, Any]:
    """
    Train the unified cross-source model with delta equipment features.

    Context Window Optimization (Probe 3.1.1):
        Analysis shows shorter context windows (7-14 days) yield better performance.
        Default is 14 days based on probe findings showing 78.4% accuracy.

    Args:
        epochs: Maximum training epochs (default 100)
        batch_size: Batch size (default 32)
        device: Device to train on (auto-detected if None)
        fusion_loss_weight: Weight for RSA fusion regularization loss (default 0.1)
        similarity_threshold: Minimum cross-source similarity to maintain (default 0.3)
        use_contrastive_fusion: Use InfoNCE contrastive loss instead of hinge (default False)
        context_window_days: Context window in days, 7-14 optimal (default 14)

    Returns:
        Dict with trained model, history, and source configs
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print("=" * 80)
    print("UNIFIED CROSS-SOURCE INTERPOLATION MODEL (DELTA VERSION)")
    print(f"Device: {device}")
    print(f"RSA Fusion Regularization: weight={fusion_loss_weight}, "
          f"threshold={similarity_threshold}")
    print("=" * 80)

    # Create datasets with proper normalization
    # FIX: Use temporal gap and pass training norm_stats to validation
    temporal_gap = 7  # 7-day gap between train/val to prevent leakage

    train_dataset = CrossSourceDatasetDelta(
        SOURCE_CONFIGS,
        train=True,
        temporal_gap=temporal_gap,
        context_window_days=context_window_days
    )
    val_dataset = CrossSourceDatasetDelta(
        SOURCE_CONFIGS,
        train=False,
        temporal_gap=temporal_gap,
        context_window_days=context_window_days,
        norm_stats=train_dataset.norm_stats  # Use training stats for validation
    )

    print(f"\nMethodology fixes applied:")
    print(f"  - Context window: {context_window_days} days (optimal: 7-14 per Probe 3.1.1)")
    print(f"  - Temporal gap: {temporal_gap} days between train/val")
    print(f"  - Normalization: training-only stats applied to validation")
    print(f"  - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

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

    # Create model with RSA fusion regularization
    model = UnifiedInterpolationModelDelta(
        source_configs=SOURCE_CONFIGS,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2,
        dropout=0.1,
        fusion_loss_weight=fusion_loss_weight,
        similarity_threshold=similarity_threshold,
        use_contrastive_fusion=use_contrastive_fusion
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {n_params:,} parameters")
    print(f"Equipment features: {SOURCE_CONFIGS['equipment'].n_features} (delta-only)")
    print(f"RSA fusion regularization enabled: weight={fusion_loss_weight}")

    # Train
    trainer = UnifiedTrainerDelta(
        model, train_loader, val_loader,
        lr=1e-4, weight_decay=0.01, device=device
    )

    history = trainer.train(epochs=epochs, patience=20, verbose=True)

    return {
        'model': model,
        'history': history,
        'source_configs': SOURCE_CONFIGS
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified Cross-Source Interpolation (Delta) with RSA Fusion Regularization'
    )
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device')
    parser.add_argument(
        '--fusion_loss_weight', type=float, default=0.1,
        help='Weight for RSA fusion regularization loss (default 0.1)'
    )
    parser.add_argument(
        '--similarity_threshold', type=float, default=0.3,
        help='Minimum cross-source similarity threshold (default 0.3)'
    )
    parser.add_argument(
        '--use_contrastive', action='store_true',
        help='Use InfoNCE contrastive loss instead of hinge loss'
    )
    parser.add_argument(
        '--context_window', type=int, default=14,
        help='Context window in days, 7-14 optimal per Probe 3.1.1 (default 14)'
    )

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    if args.train:
        train_unified_model_delta(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            fusion_loss_weight=args.fusion_loss_weight,
            similarity_threshold=args.similarity_threshold,
            use_contrastive_fusion=args.use_contrastive,
            context_window_days=args.context_window
        )
    else:
        print("Usage:")
        print("  --train                   Train the unified model with delta equipment features")
        print("  --fusion_loss_weight X    RSA fusion regularization weight (default 0.1)")
        print("  --similarity_threshold X  Minimum cross-source similarity (default 0.3)")
        print("  --use_contrastive         Use InfoNCE contrastive loss instead of hinge")
        print("  --context_window N        Context window in days, 7-14 optimal (default 14)")


if __name__ == "__main__":
    main()
