"""
Unified Cross-Source Interpolation Model (HYBRID VERSION)

Combines:
- Delta features (daily changes) for short-term dynamics
- Normalized cumulative features (losses/day moving averages) for trend context

This addresses the tension between:
- Cumulative model: Higher reconstruction but potentially spurious trend correlations
- Delta model: Statistically valid but loses aggregate intensity signal

Usage:
    python unified_interpolation_hybrid.py --train --epochs 100
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import argparse

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
# HYBRID EQUIPMENT FEATURE EXTRACTION
# =============================================================================

def extract_hybrid_equipment_features(
    data: np.ndarray,
    feature_names: List[str],
    dates: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract hybrid features from equipment data:

    1. Delta features (*_delta) - daily changes
    2. Rolling averages (*_7day_avg) - smoothed daily rates
    3. Normalized cumulative (cumulative / days_since_conflict_start) - intensity over time
    4. Ratio features (heavy_equipment_ratio) - already normalized

    FIX: Uses actual calendar days since conflict start (Feb 24, 2022),
    NOT sample indices, to avoid temporal position leakage.

    This captures both:
    - Short-term dynamics (deltas)
    - Long-term intensity trends (normalized cumulative)
    """
    hybrid_indices = []
    hybrid_names = []

    # Equipment types that have cumulative values
    cumulative_types = [
        'aircraft', 'helicopter', 'tank', 'apc', 'field_artillery',
        'mlrs', 'anti_aircraft', 'drone', 'cruise_missile', 'ship',
        'submarine', 'special_equipment', 'vehicle', 'fuel_tank'
    ]

    for i, name in enumerate(feature_names):
        # Include delta features (daily changes)
        if '_delta' in name:
            hybrid_indices.append(i)
            hybrid_names.append(name)
        # Include rolling averages
        elif '_7day_avg' in name:
            hybrid_indices.append(i)
            hybrid_names.append(name)
        # Include ratio features
        elif name in ['heavy_equipment_ratio', 'direction_encoded']:
            hybrid_indices.append(i)
            hybrid_names.append(name)
        # Include total_losses_day (already a rate)
        elif name == 'total_losses_day':
            hybrid_indices.append(i)
            hybrid_names.append(name)

    # FIX: Use actual calendar days since conflict start, not sample indices
    # This removes the temporal position encoding that was causing leakage
    n_samples = len(data)
    conflict_start = datetime(2022, 2, 24)

    if dates is not None and len(dates) == n_samples:
        # Use actual calendar dates
        days_since_conflict = []
        for date_str in dates:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                days = (dt - conflict_start).days
                days_since_conflict.append(max(days, 1))  # Avoid div by 0
            except ValueError:
                days_since_conflict.append(1)
        days_since_start = np.array(days_since_conflict).reshape(-1, 1)
    else:
        # Fallback: estimate from conflict start date
        # Assume daily data starting from Feb 25, 2022
        print("  WARNING: No dates provided, estimating days since conflict start")
        days_since_start = np.arange(1, n_samples + 1).reshape(-1, 1)

    normalized_cumulative = []
    normalized_names = []

    for i, name in enumerate(feature_names):
        # Check if this is a raw cumulative (not delta, not avg, not ratio)
        if name in cumulative_types:
            # Normalize by actual calendar days since conflict start
            norm_values = data[:, i:i+1] / days_since_start
            normalized_cumulative.append(norm_values)
            normalized_names.append(f'{name}_per_day')

    # Combine
    hybrid_data = data[:, hybrid_indices]

    if normalized_cumulative:
        norm_data = np.concatenate(normalized_cumulative, axis=1)
        hybrid_data = np.concatenate([hybrid_data, norm_data], axis=1)
        hybrid_names.extend(normalized_names)

    return hybrid_data, hybrid_names


# =============================================================================
# SOURCE CONFIGURATIONS
# =============================================================================

@dataclass
class SourceConfig:
    """Configuration for a data source in the unified model."""
    name: str
    loader_class: type
    n_features: int
    d_embed: int = 64
    use_hybrid: bool = False  # Use hybrid feature extraction


SOURCE_CONFIGS = {
    # Sentinel excluded: only 32 monthly samples + data quality issues (row 1 anomaly)
    # 'sentinel': SourceConfig(
    #     name='Sentinel',
    #     loader_class=SentinelDataLoader,
    #     n_features=55,
    #     d_embed=64,
    #     use_hybrid=False
    # ),
    'deepstate': SourceConfig(
        name='DeepState',
        loader_class=DeepStateDataLoader,
        n_features=55,
        d_embed=64,
        use_hybrid=False
    ),
    'equipment': SourceConfig(
        name='Equipment',
        loader_class=EquipmentDataLoader,
        n_features=41,  # Will be set dynamically based on hybrid extraction
        d_embed=64,
        use_hybrid=True  # KEY: Use hybrid features
    ),
    'firms': SourceConfig(
        name='FIRMS',
        loader_class=FIRMSDataLoader,
        n_features=42,
        d_embed=64,
        use_hybrid=False
    ),
    'ucdp': SourceConfig(
        name='UCDP',
        loader_class=UCDPDataLoader,
        n_features=48,
        d_embed=64,
        use_hybrid=False
    ),
}


# =============================================================================
# NEURAL NETWORK COMPONENTS (same architecture as delta model)
# =============================================================================

if HAS_TORCH:

    class SourceEncoder(nn.Module):
        """Encodes a single source's features into a fixed-size embedding."""

        def __init__(self, n_features: int, d_embed: int = 64):
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


    class UnifiedInterpolationModelHybrid(nn.Module):
        """
        Unified model for cross-source interpolation (Hybrid version).

        Uses hybrid equipment features combining deltas and normalized cumulative.
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

            self.encoders = nn.ModuleDict()
            for name, config in source_configs.items():
                self.encoders[name] = SourceEncoder(
                    n_features=config.n_features,
                    d_embed=d_embed
                )

            self.fusion = CrossSourceAttention(
                n_sources=len(source_configs),
                d_embed=d_embed,
                nhead=nhead,
                num_layers=num_fusion_layers,
                dropout=dropout
            )

            self.decoders = nn.ModuleDict()
            for name, config in source_configs.items():
                self.decoders[name] = SourceDecoder(
                    n_features=config.n_features,
                    d_embed=d_embed
                )

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
            return_reconstructions: bool = True
        ) -> Dict[str, torch.Tensor]:
            embeddings = self.encode_sources(features, mask)
            fused = self.fusion(embeddings, self.source_names, mask)

            outputs = {'fused_embeddings': fused}

            if return_reconstructions:
                reconstructions = {}
                for name in self.source_names:
                    reconstructions[name] = self.decoders[name](fused[name])
                outputs['reconstructions'] = reconstructions

            fused_concat = torch.cat([fused[name] for name in self.source_names], dim=-1)
            outputs['unified'] = self.unified_proj(fused_concat)

            return outputs


    class CrossSourceDatasetHybrid(Dataset):
        """
        Dataset for cross-source fusion training (Hybrid version).

        FIXES APPLIED:
        1. Uses calendar days for *_per_day normalization (not sample indices)
        2. Computes normalization stats on TRAINING data only, then applies to validation
        3. Supports temporal cross-validation with configurable gap
        """

        def __init__(
            self,
            source_configs: Dict[str, SourceConfig],
            train: bool = True,
            val_ratio: float = 0.2,
            exclude_clown_units: bool = True,  # Exclude Orban/Fico markers
            temporal_gap: int = 0,  # Days gap between train/val to prevent leakage
            norm_stats: Dict = None,  # Pre-computed normalization stats (for val set)
            ablate_per_day: bool = False  # For ablation: exclude *_per_day features
        ):
            self.source_configs = source_configs
            self.train = train
            self.exclude_clown_units = exclude_clown_units
            self.temporal_gap = temporal_gap
            self.ablate_per_day = ablate_per_day

            self.source_data = {}
            self.source_dates = {}
            self.feature_names = {}
            self.norm_stats = norm_stats or {}

            print(f"Loading source data (HYBRID version, train={train})...")
            for name, config in source_configs.items():
                try:
                    loader = config.loader_class().load().process()
                    data = loader.processed_data
                    dates = loader.dates if hasattr(loader, 'dates') else None

                    # Apply hybrid extraction for equipment
                    if name == 'equipment' and config.use_hybrid:
                        if hasattr(loader, 'feature_names'):
                            # FIX #1: Pass dates for calendar-based normalization
                            data, feat_names = extract_hybrid_equipment_features(
                                data, loader.feature_names, dates=dates
                            )

                            # Ablation: Remove *_per_day features if requested
                            if ablate_per_day:
                                keep_indices = [i for i, n in enumerate(feat_names)
                                              if not n.endswith('_per_day')]
                                data = data[:, keep_indices]
                                feat_names = [feat_names[i] for i in keep_indices]
                                print(f"  ABLATION: Removed *_per_day features from {name}")

                            self.feature_names[name] = feat_names
                            # Update config with actual feature count
                            config.n_features = len(feat_names)
                        else:
                            self.feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]
                    else:
                        if hasattr(loader, 'feature_names'):
                            self.feature_names[name] = loader.feature_names
                        else:
                            self.feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]

                    self.source_data[name] = torch.tensor(data, dtype=torch.float32)

                    if dates is not None:
                        self.source_dates[name] = dates

                    print(f"  {name}: {data.shape} features")

                except Exception as e:
                    print(f"  WARNING: Could not load {name}: {e}")
                    import traceback
                    traceback.print_exc()

            # Align to common length
            min_samples = min(len(d) for d in self.source_data.values())
            self.n_samples = min_samples

            for name in self.source_data:
                self.source_data[name] = self.source_data[name][:self.n_samples]
                if name in self.source_dates:
                    self.source_dates[name] = self.source_dates[name][:self.n_samples]

            print(f"  Aligned to {self.n_samples} samples")

            # Optionally exclude clown units from DeepState
            if exclude_clown_units and 'deepstate' in self.feature_names:
                clown_idx = None
                for i, fname in enumerate(self.feature_names['deepstate']):
                    if 'clown' in fname.lower():
                        clown_idx = i
                        break
                if clown_idx is not None:
                    print(f"  Zeroing out units_clown feature (index {clown_idx})")
                    self.source_data['deepstate'][:, clown_idx] = 0

            # Temporal train/val split with optional gap
            split_idx = int(self.n_samples * (1 - val_ratio))

            if train:
                # Training set: all data up to split point
                self.indices = list(range(split_idx))
            else:
                # Validation set: starts after split point + temporal gap
                val_start = split_idx + temporal_gap
                self.indices = list(range(val_start, self.n_samples))

            # FIX #2: Compute normalization stats on TRAINING data only
            print("  Normalizing features...")
            if train:
                # Training: compute and store normalization stats
                self.norm_stats = {}
                train_indices = self.indices
                for name in self.source_data:
                    train_data = self.source_data[name][train_indices]
                    mean = train_data.mean(dim=0, keepdim=True)
                    std = train_data.std(dim=0, keepdim=True) + 1e-8
                    self.norm_stats[name] = {'mean': mean, 'std': std}
                    # Apply normalization to ALL data (will be indexed later)
                    self.source_data[name] = (self.source_data[name] - mean) / std
            else:
                # Validation: use provided norm_stats from training set
                if not self.norm_stats:
                    print("  WARNING: No norm_stats provided for validation set!")
                    print("           Computing on full data (potential leakage)")
                    for name in self.source_data:
                        mean = self.source_data[name].mean(dim=0, keepdim=True)
                        std = self.source_data[name].std(dim=0, keepdim=True) + 1e-8
                        self.source_data[name] = (self.source_data[name] - mean) / std
                else:
                    # Apply training normalization stats to validation data
                    for name in self.source_data:
                        if name in self.norm_stats:
                            mean = self.norm_stats[name]['mean']
                            std = self.norm_stats[name]['std']
                            self.source_data[name] = (self.source_data[name] - mean) / std
                        else:
                            print(f"  WARNING: No norm_stats for {name}, using local stats")
                            mean = self.source_data[name].mean(dim=0, keepdim=True)
                            std = self.source_data[name].std(dim=0, keepdim=True) + 1e-8
                            self.source_data[name] = (self.source_data[name] - mean) / std

            print(f"  {'Train' if train else 'Val'} samples: {len(self.indices)}")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            features = {name: data[real_idx] for name, data in self.source_data.items()}
            return features


    class UnifiedTrainerHybrid:
        """Trainer for hybrid unified model."""

        def __init__(
            self,
            model: UnifiedInterpolationModelHybrid,
            train_loader: DataLoader,
            val_loader: DataLoader,
            lr: float = 1e-3,
            device: torch.device = None
        ):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device or torch.device('cpu')

            self.model.to(self.device)

            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )

            self.best_val_loss = float('inf')
            self.history = {'train_loss': [], 'val_loss': []}

        def train_epoch(self, mask_ratio: float = 0.2):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                features = {k: v.to(self.device) for k, v in batch.items()}

                # Random source masking
                batch_size = next(iter(features.values())).size(0)
                mask = {}
                masked_source = np.random.choice(list(features.keys()))
                for name in features:
                    if name == masked_source:
                        mask[name] = torch.zeros(batch_size, device=self.device)
                    else:
                        mask[name] = torch.ones(batch_size, device=self.device)

                self.optimizer.zero_grad()

                outputs = self.model(features, mask=mask)
                reconstructions = outputs['reconstructions']

                # Reconstruction loss (focus on masked source)
                loss = F.mse_loss(reconstructions[masked_source], features[masked_source])

                # Add auxiliary losses for other sources
                for name in features:
                    if name != masked_source:
                        loss += 0.1 * F.mse_loss(reconstructions[name], features[name])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            return total_loss / len(self.train_loader)

        @torch.no_grad()
        def validate(self):
            self.model.eval()
            total_loss = 0

            for batch in self.val_loader:
                features = {k: v.to(self.device) for k, v in batch.items()}

                # Test each source as masked
                for masked_source in features.keys():
                    batch_size = next(iter(features.values())).size(0)
                    mask = {
                        name: torch.zeros(batch_size, device=self.device) if name == masked_source
                        else torch.ones(batch_size, device=self.device)
                        for name in features
                    }

                    outputs = self.model(features, mask=mask)
                    loss = F.mse_loss(outputs['reconstructions'][masked_source], features[masked_source])
                    total_loss += loss.item()

            return total_loss / (len(self.val_loader) * len(features))

        def train(self, epochs: int = 100, patience: int = 20):
            patience_counter = 0

            for epoch in range(epochs):
                train_loss = self.train_epoch()
                val_loss = self.validate()

                self.scheduler.step(val_loss)

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint('unified_interpolation_hybrid_best.pt')
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            return self.history

        def save_checkpoint(self, filename: str):
            if filename == 'unified_interpolation_hybrid_best.pt':
                torch.save(self.model.state_dict(), UNIFIED_HYBRID_MODEL)
            else:
                torch.save(self.model.state_dict(), MODEL_DIR / filename)


def main():
    parser = argparse.ArgumentParser(description='Hybrid Unified Interpolation Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--temporal-gap', type=int, default=7,
                       help='Days gap between train/val for temporal CV (default: 7)')
    parser.add_argument('--ablate-per-day', action='store_true',
                       help='Ablation: remove *_per_day features to test their contribution')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.train:
        # Create datasets with proper normalization handling
        print("\nCreating datasets...")
        print(f"  Temporal gap: {args.temporal_gap} days")
        if args.ablate_per_day:
            print("  ABLATION MODE: *_per_day features will be excluded")

        # FIX: Create training dataset first to get normalization stats
        train_dataset = CrossSourceDatasetHybrid(
            SOURCE_CONFIGS,
            train=True,
            exclude_clown_units=True,
            temporal_gap=args.temporal_gap,
            ablate_per_day=args.ablate_per_day
        )

        # FIX: Pass training normalization stats to validation dataset
        val_dataset = CrossSourceDatasetHybrid(
            SOURCE_CONFIGS,
            train=False,
            exclude_clown_units=True,
            temporal_gap=args.temporal_gap,
            norm_stats=train_dataset.norm_stats,  # Use training stats!
            ablate_per_day=args.ablate_per_day
        )

        # Update source configs with actual feature counts
        for name, config in SOURCE_CONFIGS.items():
            if name in train_dataset.feature_names:
                config.n_features = len(train_dataset.feature_names[name])
                print(f"  {name}: {config.n_features} features")

        def collate_fn(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )

        # Create model
        print("\nCreating hybrid model...")
        model = UnifiedInterpolationModelHybrid(
            source_configs=SOURCE_CONFIGS,
            d_embed=64,
            nhead=4,
            num_fusion_layers=2
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Train
        trainer = UnifiedTrainerHybrid(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            device=device
        )

        print(f"\nTraining for {args.epochs} epochs...")
        history = trainer.train(epochs=args.epochs, patience=20)

        print(f"\nBest validation loss: {trainer.best_val_loss:.4f}")
        print(f"Model saved to {MODEL_DIR / 'unified_interpolation_hybrid_best.pt'}")

        # Print feature summary
        print("\n=== Hybrid Equipment Features ===")
        if 'equipment' in train_dataset.feature_names:
            for i, name in enumerate(train_dataset.feature_names['equipment']):
                print(f"  {i:2d}: {name}")


if __name__ == '__main__':
    main()
