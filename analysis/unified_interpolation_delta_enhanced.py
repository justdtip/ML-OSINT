"""
Unified Cross-Source Interpolation Model (DELTA VERSION - ENHANCED)

ADDITIONS based on omitted variable analysis:
1. Seasonal features (month_sin, month_cos) - r=0.82 with residual Factor_5
2. Enhanced FIRMS fire features integration

This version uses the same delta-only equipment features but adds
temporal context that the omitted variable analysis identified as
correlated with residual patterns.

Usage:
    python unified_interpolation_delta_enhanced.py --train --epochs 100
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
# DELTA EQUIPMENT FEATURE EXTRACTION (unchanged from original)
# =============================================================================

def extract_equipment_delta_features(data: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract only delta (per-day) features from equipment data."""
    delta_indices = []
    delta_names = []

    for i, name in enumerate(feature_names):
        if '_delta' in name or '_7day_avg' in name:
            delta_indices.append(i)
            delta_names.append(name)
        elif name in ['total_losses_day', 'heavy_equipment_ratio', 'direction_encoded']:
            delta_indices.append(i)
            delta_names.append(name)
        elif name.endswith('_per_day'):
            delta_indices.append(i)
            delta_names.append(name)

    if delta_indices:
        return data[:, delta_indices], delta_names
    else:
        return data, feature_names


# =============================================================================
# SEASONAL FEATURE EXTRACTION (NEW - from omitted variable analysis)
# =============================================================================

def extract_seasonal_features(dates: List) -> Tuple[np.ndarray, List[str]]:
    """
    Extract seasonal features based on omitted variable analysis findings.

    The analysis found:
    - month_sin: r=0.815 with Factor_5 (strong seasonal pattern)
    - month_cos: r=0.605 with Factor_1

    These capture cyclical patterns in conflict dynamics.
    """
    n_samples = len(dates)
    seasonal_features = np.zeros((n_samples, 4), dtype=np.float32)

    for i, d in enumerate(dates):
        if isinstance(d, str):
            if len(d) == 10:
                dt = datetime.strptime(d, '%Y-%m-%d')
            else:
                dt = datetime.strptime(d, '%Y-%m')
        elif hasattr(d, 'month'):
            dt = d
        else:
            dt = datetime.combine(d, datetime.min.time())

        month = dt.month
        day_of_year = dt.timetuple().tm_yday

        # Cyclical month encoding (strongest correlations with residual factors)
        seasonal_features[i, 0] = np.sin(2 * np.pi * month / 12)  # month_sin
        seasonal_features[i, 1] = np.cos(2 * np.pi * month / 12)  # month_cos

        # Day of year encoding (finer seasonal resolution)
        seasonal_features[i, 2] = np.sin(2 * np.pi * day_of_year / 365)  # doy_sin
        seasonal_features[i, 3] = np.cos(2 * np.pi * day_of_year / 365)  # doy_cos

    feature_names = ['month_sin', 'month_cos', 'doy_sin', 'doy_cos']
    return seasonal_features, feature_names


# =============================================================================
# SOURCE CONFIG (with seasonal source added)
# =============================================================================

@dataclass
class SourceConfig:
    name: str
    loader_class: Any
    n_features: int
    jim_config_key: str
    d_embed: int = 64
    use_delta_only: bool = False
    is_temporal: bool = False  # NEW: Flag for temporal features


SOURCE_CONFIGS = {
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
        n_features=27,
        jim_config_key='equipment_totals',
        d_embed=64,
        use_delta_only=True
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
    # NEW: Seasonal temporal features
    'seasonal': SourceConfig(
        name='Seasonal',
        loader_class=None,  # Generated, not loaded
        n_features=4,
        jim_config_key='deepstate',  # Not used
        d_embed=32,  # Smaller embedding for simple features
        use_delta_only=False,
        is_temporal=True
    ),
}


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

            self.feature_proj = nn.Sequential(
                nn.Linear(n_features, d_embed * 2),
                nn.LayerNorm(d_embed * 2),
                nn.GELU(),
                nn.Linear(d_embed * 2, d_embed),
                nn.LayerNorm(d_embed)
            )

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

            self.source_embeddings = nn.Embedding(n_sources, d_embed)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_embed,
                nhead=nhead,
                dim_feedforward=d_embed * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

            stacked = torch.stack([source_embeddings[s] for s in source_order], dim=1)

            source_indices = torch.arange(len(source_order), device=device)
            source_type_emb = self.source_embeddings(source_indices)
            stacked = stacked + source_type_emb.unsqueeze(0)

            attn_mask = None
            if mask is not None:
                mask_tensor = torch.stack([1 - mask.get(s, torch.ones(batch_size, device=device))
                                          for s in source_order], dim=1)
                attn_mask = mask_tensor.bool()

            fused = self.transformer(stacked, src_key_padding_mask=attn_mask)

            outputs = {}
            for i, source_name in enumerate(source_order):
                outputs[source_name] = self.output_projs[i](fused[:, i, :])

            return outputs


    class UnifiedInterpolationModelEnhanced(nn.Module):
        """
        Unified model for cross-source interpolation (Enhanced version).

        Includes seasonal temporal features based on omitted variable analysis.
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            d_embed: int = 64,
            nhead: int = 4,
            num_fusion_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            self.source_names = list(source_configs.keys())
            self.source_configs = source_configs
            self.d_embed = d_embed

            # Identify which sources are reconstructable vs. context-only
            self.reconstructable_sources = [
                name for name, config in source_configs.items()
                if not config.is_temporal
            ]

            # Source encoders (all sources including seasonal)
            self.encoders = nn.ModuleDict()
            for name, config in source_configs.items():
                # Use smaller embedding for temporal features
                embed_dim = config.d_embed if hasattr(config, 'd_embed') else d_embed
                self.encoders[name] = SourceEncoder(
                    n_features=config.n_features,
                    d_embed=d_embed  # Project to common embedding space
                )

            # Cross-source fusion
            self.fusion = CrossSourceAttention(
                n_sources=len(source_configs),
                d_embed=d_embed,
                nhead=nhead,
                num_layers=num_fusion_layers,
                dropout=dropout
            )

            # Source decoders (only for reconstructable sources)
            self.decoders = nn.ModuleDict()
            for name in self.reconstructable_sources:
                config = source_configs[name]
                self.decoders[name] = SourceDecoder(
                    n_features=config.n_features,
                    d_embed=d_embed
                )

        def forward(
            self,
            features: Dict[str, torch.Tensor],
            mask: Optional[Dict[str, torch.Tensor]] = None,
            return_reconstructions: bool = False
        ) -> Dict[str, torch.Tensor]:
            # Encode all sources
            embeddings = {}
            for name in self.source_names:
                if name in features:
                    embeddings[name] = self.encoders[name](features[name])

            # Cross-source fusion
            fused = self.fusion(embeddings, self.source_names, mask)

            outputs = {'fused': fused}

            # Reconstruct only the reconstructable sources
            if return_reconstructions:
                reconstructions = {}
                for name in self.reconstructable_sources:
                    if name in fused:
                        reconstructions[name] = self.decoders[name](fused[name])
                outputs['reconstructions'] = reconstructions

            return outputs


    class CrossSourceDatasetEnhanced(Dataset):
        """
        Dataset for cross-source learning with seasonal features.

        Adds temporal context features identified by omitted variable analysis.
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            train: bool = True,
            val_ratio: float = 0.2,
            temporal_gap: int = 0,
            norm_stats: Dict = None
        ):
            self.source_configs = source_configs
            self.train = train
            self.temporal_gap = temporal_gap
            self.norm_stats = norm_stats or {}

            self.source_data = {}
            self.source_dates = {}
            self.feature_names = {}

            print(f"Loading source data (ENHANCED version, train={train})...")

            # Load standard sources
            for name, config in source_configs.items():
                if config.is_temporal:
                    continue  # Skip temporal sources, generated later

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

                    if config.use_delta_only:
                        data, feature_names = extract_equipment_delta_features(data, feature_names)
                        print(f"  {name}: {len(dates)} days, {data.shape[1]} DELTA features")
                    else:
                        print(f"  {name}: {len(dates)} days, {data.shape[1]} features")

                    self.source_data[name] = data
                    self.source_dates[name] = dates
                    self.feature_names[name] = feature_names
                    config.n_features = data.shape[1]

                except Exception as e:
                    print(f"  Error loading {name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Align dates
            self._align_dates()

            # Generate seasonal features based on aligned dates
            for name, config in source_configs.items():
                if config.is_temporal:
                    seasonal_data, seasonal_names = extract_seasonal_features(self.aligned_dates)
                    self.aligned_data[name] = seasonal_data
                    self.feature_names[name] = seasonal_names
                    config.n_features = seasonal_data.shape[1]
                    print(f"  {name}: {len(self.aligned_dates)} days, {seasonal_data.shape[1]} temporal features (GENERATED)")

            # Train/val split
            n_samples = len(self.aligned_dates)
            n_val = int(n_samples * val_ratio)

            if train:
                self.start_idx = 0
                self.end_idx = n_samples - n_val
            else:
                self.start_idx = n_samples - n_val + temporal_gap
                self.end_idx = n_samples

            # Normalize
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
            """Normalize using training data statistics only."""
            print("  Normalizing features...")

            if train:
                train_end = n_samples - n_val
                self.norm_stats = {}

                for name in self.aligned_data:
                    # Don't normalize seasonal features (already bounded [-1, 1])
                    if self.source_configs.get(name, SourceConfig('', None, 0, '')).is_temporal:
                        continue

                    train_data = self.aligned_data[name][:train_end]
                    mean = train_data.mean(axis=0, keepdims=True)
                    std = train_data.std(axis=0, keepdims=True) + 1e-8

                    self.norm_stats[name] = {'mean': mean, 'std': std}
                    self.aligned_data[name] = (self.aligned_data[name] - mean) / std

            else:
                for name in self.aligned_data:
                    if self.source_configs.get(name, SourceConfig('', None, 0, '')).is_temporal:
                        continue

                    if name in self.norm_stats:
                        mean = self.norm_stats[name]['mean']
                        std = self.norm_stats[name]['std']
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
                # Only mask reconstructable sources
                reconstructable = [
                    name for name, config in self.source_configs.items()
                    if not config.is_temporal
                ]
                masked_source = random.choice(reconstructable)
                for name in features:
                    mask[name] = torch.tensor(0.0 if name == masked_source else 1.0)
            else:
                for name in features:
                    mask[name] = torch.tensor(1.0)

            return features, mask, masked_source if masked_source else ''


    class UnifiedTrainerEnhanced:
        """Trainer for the enhanced unified model."""

        def __init__(
            self,
            model: UnifiedInterpolationModelEnhanced,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 1e-3,
            device: str = 'cpu'
        ):
            self.model = model.to(device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device

            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )

            self.best_val_loss = float('inf')

        def train_epoch(self) -> float:
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch in self.train_loader:
                features = {k: v.to(self.device) for k, v in batch[0].items()}
                mask = {k: v.to(self.device) for k, v in batch[1].items()}

                self.optimizer.zero_grad()

                outputs = self.model(features, mask, return_reconstructions=True)

                # Reconstruction loss (only for reconstructable sources)
                loss = 0
                for name, recon in outputs['reconstructions'].items():
                    target = features[name]
                    source_mask = mask.get(name, torch.ones(target.size(0), device=self.device))
                    masked_loss = F.mse_loss(recon, target, reduction='none')
                    # Weight by inverse mask (higher weight for masked sources)
                    weights = 1.0 + (1.0 - source_mask.unsqueeze(-1))
                    loss += (masked_loss * weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            return total_loss / max(n_batches, 1)

        def validate(self) -> Dict[str, float]:
            self.model.eval()
            total_mae = {name: 0 for name in self.model.reconstructable_sources}
            n_batches = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    features = {k: v.to(self.device) for k, v in batch[0].items()}

                    # Test each reconstructable source
                    for test_source in self.model.reconstructable_sources:
                        test_mask = {
                            name: torch.tensor(0.0 if name == test_source else 1.0,
                                             device=self.device).expand(features[name].size(0))
                            for name in features
                        }

                        outputs = self.model(features, test_mask, return_reconstructions=True)

                        if test_source in outputs['reconstructions']:
                            recon = outputs['reconstructions'][test_source]
                            target = features[test_source]
                            mae = F.l1_loss(recon, target).item()
                            total_mae[test_source] += mae

                    n_batches += 1

            return {name: total_mae[name] / max(n_batches, 1) for name in total_mae}

        def train(self, epochs: int = 100, patience: int = 20, verbose: bool = True) -> dict:
            history = {'train_loss': [], 'val_mae': []}
            patience_counter = 0
            best_epoch = 0

            print(f"Training for up to {epochs} epochs...")
            print("-" * 70)

            for epoch in range(epochs):
                train_loss = self.train_epoch()
                val_metrics = self.validate()

                val_mae = np.mean(list(val_metrics.values()))
                history['train_loss'].append(train_loss)
                history['val_mae'].append(val_mae)

                improved = val_mae < self.best_val_loss
                if improved:
                    self.best_val_loss = val_mae
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        MODEL_DIR / 'unified_interpolation_delta_enhanced_best.pt'
                    )
                else:
                    patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == epochs - 1 or improved):
                    source_maes = ' | '.join([
                        f"{name[:4]}:{val_metrics[name]:.4f}"
                        for name in sorted(val_metrics.keys())
                    ])
                    print(f"Epoch {epoch:3d}: loss={train_loss:.4f}, val_mae={val_mae:.4f} "
                          f"{'*' if improved else ''}")
                    print(f"           {source_maes}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            print("-" * 70)
            print(f"Best val MAE: {self.best_val_loss:.4f} at epoch {best_epoch}")

            return history


def train_unified_model_enhanced(
    epochs: int = 100,
    batch_size: int = 32,
    device: str = None
) -> Dict:
    """Train the enhanced unified cross-source model."""

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            # MPS has issues with nested tensors, use CPU
            device = 'cpu'
        else:
            device = 'cpu'

    print("=" * 70)
    print("UNIFIED CROSS-SOURCE INTERPOLATION MODEL (ENHANCED VERSION)")
    print("Additions: Seasonal features (month_sin, month_cos)")
    print(f"Device: {device}")
    print("=" * 70)

    # Create datasets
    train_dataset = CrossSourceDatasetEnhanced(
        SOURCE_CONFIGS,
        train=True,
        temporal_gap=7
    )

    val_dataset = CrossSourceDatasetEnhanced(
        SOURCE_CONFIGS,
        train=False,
        temporal_gap=7,
        norm_stats=train_dataset.norm_stats
    )

    # Update configs with actual feature counts
    for name, config in SOURCE_CONFIGS.items():
        if name in train_dataset.feature_names:
            config.n_features = len(train_dataset.feature_names[name])

    def collate_fn(batch):
        features = {k: torch.stack([b[0][k] for b in batch]) for k in batch[0][0].keys()}
        masks = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1].keys()}
        return features, masks, [b[2] for b in batch]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"\nMethodology:")
    print(f"  - Temporal gap: 7 days between train/val")
    print(f"  - Seasonal features: month_sin, month_cos, doy_sin, doy_cos")
    print(f"  - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    model = UnifiedInterpolationModelEnhanced(
        source_configs=SOURCE_CONFIGS,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    # Feature summary
    print("\nFeatures per source:")
    for name, config in SOURCE_CONFIGS.items():
        temporal_flag = " (TEMPORAL)" if config.is_temporal else ""
        print(f"  {name}: {config.n_features}{temporal_flag}")

    # Train
    trainer = UnifiedTrainerEnhanced(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        device=device
    )

    history = trainer.train(epochs=epochs, patience=20, verbose=True)

    return {
        'model': model,
        'history': history,
        'source_configs': SOURCE_CONFIGS,
        'best_val_mae': trainer.best_val_loss
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Unified Cross-Source Interpolation (Enhanced)')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, help='Device')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    if args.train:
        result = train_unified_model_enhanced(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        print(f"\n=== RESULTS ===")
        print(f"Best validation MAE: {result['best_val_mae']:.4f}")
    else:
        print("Usage:")
        print("  --train     Train the enhanced model with seasonal features")


if __name__ == "__main__":
    main()
