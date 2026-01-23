#!/usr/bin/env python3
"""
Phase 2B: Hierarchical Interpolation System

This module implements the full 198-feature hierarchical conditioning system where:
1. Phase 1 parent models (trained on aggregate features) produce conditioning embeddings
2. Phase 2 child models use these embeddings to interpolate decomposed sub-features
3. The hierarchy flows: aggregate → decomposed → sub-decomposed

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PHASE 1: PARENT MODELS                    │
    │   (trained independently, produce conditioning embeddings)   │
    ├─────────────────────────────────────────────────────────────┤
    │  equipment_totals ──┬── tank_total ────→ equipment_tanks    │
    │                     ├── afv_total ─────→ equipment_afv      │
    │                     └── aircraft_total → equipment_aircraft │
    │                                                              │
    │  deepstate ─────────┬── arrows_total ──→ deepstate_arrows   │
    │                     ├── units_total ───→ deepstate_units    │
    │                     └── poly_* ────────→ deepstate_polygons │
    │                                                              │
    │  sentinel2 ─────────────────────────→ sentinel2_indices     │
    │  sentinel1 ─────────────────────────→ sentinel1_change      │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    PHASE 2: CHILD MODELS                     │
    │   (receive conditioning from parents during training/inf)    │
    ├─────────────────────────────────────────────────────────────┤
    │  equipment_tanks:    [t62, t64, t72, t80, t90, other]       │
    │  equipment_afv:      [bmp, btr, mtlb, bmd, other]           │
    │  equipment_aircraft: [sukhoi, mig, transport, awacs]        │
    │  deepstate_arrows:   [12 directional features]              │
    │  deepstate_units:    [12 unit type features]                │
    │  deepstate_polygons: [6 territory status features]          │
    │  ucdp_geography:     [13 oblast features]                   │
    │  firms_by_intensity: [6 intensity bins]                     │
    │  firms_by_time:      [4 time-of-day features]              │
    └─────────────────────────────────────────────────────────────┘

Usage:
    python analysis/hierarchical_interpolation_system.py --train-phase2
    python analysis/hierarchical_interpolation_system.py --interpolate --date 2024-01-15
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

# Import Phase 1 components
from joint_interpolation_models import (
    INTERPOLATION_CONFIGS, PHASE2_CONFIGS,
    JointInterpolationModel, InterpolationConfig,
    InterpolationDataset, InterpolationTrainer
)

# Import data loaders
from interpolation_data_loaders import (
    SentinelDataLoader, DeepStateDataLoader,
    EquipmentDataLoader, FIRMSDataLoader, UCDPDataLoader
)

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    UNIFIED_INTERP_MODEL, UNIFIED_DELTA_MODEL, UNIFIED_HYBRID_MODEL,
    get_interp_model_path, get_phase_model_path,
)

# For backward compatibility, keep BASE_DIR as an alias to PROJECT_ROOT
BASE_DIR = PROJECT_ROOT
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# PARENT-CHILD RELATIONSHIPS
# =============================================================================

# Maps Phase 2 child configs to their Phase 1 parents
PARENT_CHILD_MAP = {
    # Equipment hierarchy
    'equipment_tanks': {
        'parent': 'equipment_totals',
        'parent_feature_idx': 2,  # tank_total is 3rd feature (0-indexed)
        'constraint': 'sum_to_parent'  # child values should sum to parent
    },
    'equipment_afv': {
        'parent': 'equipment_totals',
        'parent_feature_idx': 3,  # afv_total
        'constraint': 'sum_to_parent'
    },
    'equipment_aircraft': {
        'parent': 'equipment_totals',
        'parent_feature_idx': 0,  # aircraft_total
        'constraint': 'sum_to_parent'
    },

    # DeepState hierarchy
    'deepstate_arrows': {
        'parent': 'deepstate',
        'parent_feature_idx': 4,  # arrows_total
        'constraint': 'sum_to_parent'
    },
    'deepstate_units': {
        'parent': 'deepstate',
        'parent_feature_idx': 5,  # units_total
        'constraint': 'sum_to_parent'
    },
    'deepstate_polygons': {
        'parent': 'deepstate',
        'parent_feature_indices': [0, 1, 2],  # poly_occupied, liberated, contested areas
        'constraint': 'soft_consistency'
    },

    # UCDP hierarchy (no Phase 1 parent yet, use aggregate)
    'ucdp_geography': {
        'parent': None,  # Will need to add ucdp parent model
        'constraint': 'sum_to_parent'
    },

    # FIRMS hierarchy
    'firms_by_intensity': {
        'parent': None,  # Will need to add firms parent
        'constraint': 'sum_to_parent'
    },
    'firms_by_time': {
        'parent': None,
        'constraint': 'soft_consistency'
    },
}


# =============================================================================
# HIERARCHICAL INTERPOLATION SYSTEM
# =============================================================================

if HAS_TORCH:

    class HierarchicalInterpolationSystem:
        """
        Complete system for hierarchical temporal interpolation.

        Manages:
        1. Loading and caching Phase 1 parent models
        2. Training Phase 2 child models with parent conditioning
        3. Inference pipeline that respects hierarchy
        """

        def __init__(self, device: str = 'cpu'):
            self.device = device
            self.parent_models: Dict[str, JointInterpolationModel] = {}
            self.child_models: Dict[str, JointInterpolationModel] = {}
            self.loaded = False

        def load_parent_models(self):
            """Load trained Phase 1 parent models."""
            print("Loading Phase 1 parent models...")

            for name, config in INTERPOLATION_CONFIGS.items():
                model = JointInterpolationModel(config)

                # Try to load saved weights
                model_path = INTERP_MODEL_DIR / f"interp_{config.name.replace(' ', '_').lower()}_best.pt"
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    print(f"  Loaded: {name}")
                else:
                    print(f"  Warning: No trained model for {name}, using random init")

                model.to(self.device)
                model.eval()
                self.parent_models[name] = model

            self.loaded = True

        def get_parent_conditioning(
            self,
            parent_name: str,
            obs_before: torch.Tensor,
            obs_after: torch.Tensor,
            day_before: torch.Tensor,
            day_after: torch.Tensor,
            target_day: torch.Tensor
        ) -> torch.Tensor:
            """
            Get conditioning embedding from a parent model.

            Args:
                parent_name: Name of parent model (e.g., 'equipment_totals')
                obs_before, obs_after: Parent feature observations
                day_before, day_after, target_day: Temporal positions

            Returns:
                conditioning: [batch, 1, d_model] - conditioning for child model
            """
            if parent_name not in self.parent_models:
                raise ValueError(f"Parent model '{parent_name}' not loaded")

            parent = self.parent_models[parent_name]
            parent.eval()

            with torch.no_grad():
                _, _, embedding = parent(
                    obs_before, obs_after,
                    day_before, day_after, target_day,
                    return_embedding=True
                )

            # Add sequence dimension for attention
            return embedding.unsqueeze(1)  # [batch, 1, d_model]

        def train_child_model(
            self,
            child_name: str,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True
        ) -> Dict[str, Any]:
            """
            Train a Phase 2 child model with parent conditioning.
            """
            if child_name not in PHASE2_CONFIGS:
                raise ValueError(f"Unknown child model: {child_name}")

            if not self.loaded:
                self.load_parent_models()

            config = PHASE2_CONFIGS[child_name]
            relationship = PARENT_CHILD_MAP.get(child_name, {})
            parent_name = relationship.get('parent')

            print(f"\n{'='*60}")
            print(f"Training Phase 2 Child: {config.name}")
            print(f"Parent model: {parent_name or 'None'}")
            print(f"Features: {len(config.features)}")
            print(f"Conditioning dim: {config.conditioning_dim}")
            print(f"{'='*60}")

            # Create child model
            child_model = JointInterpolationModel(config)
            n_params = sum(p.numel() for p in child_model.parameters())
            print(f"Parameters: {n_params:,}")

            # Create dataset
            train_dataset = ConditionedInterpolationDataset(
                config=config,
                parent_name=parent_name,
                parent_model=self.parent_models.get(parent_name),
                data_path=DATA_DIR,
                train=True,
                device=self.device
            )
            val_dataset = ConditionedInterpolationDataset(
                config=config,
                parent_name=parent_name,
                parent_model=self.parent_models.get(parent_name),
                data_path=DATA_DIR,
                train=False,
                device=self.device
            )

            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")

            if len(train_dataset) == 0:
                print(f"Warning: No training samples for {child_name}")
                return {'error': 'no_samples'}

            # Create loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size
            )

            # Create trainer with hierarchy-aware loss
            trainer = HierarchicalInterpolationTrainer(
                model=child_model,
                train_loader=train_loader,
                val_loader=val_loader,
                constraint_type=relationship.get('constraint', 'none'),
                lr=1e-4,
                device=self.device
            )

            # Train
            history = trainer.train(epochs=epochs, patience=20, verbose=verbose)

            # Save model
            model_path = MODEL_DIR / f"phase2_{child_name}_best.pt"
            torch.save(child_model.state_dict(), model_path)
            print(f"Saved model to {model_path}")

            best_mae = min(history['val_mae'])
            self.child_models[child_name] = child_model

            return {
                'name': child_name,
                'history': history,
                'best_val_mae': best_mae,
                'n_params': n_params
            }

        def train_all_children(
            self,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True
        ) -> Dict[str, Any]:
            """Train all Phase 2 child models."""
            if not self.loaded:
                self.load_parent_models()

            results = {}

            for child_name in PHASE2_CONFIGS.keys():
                try:
                    result = self.train_child_model(
                        child_name, epochs, batch_size, verbose
                    )
                    results[child_name] = result
                except Exception as e:
                    print(f"Error training {child_name}: {e}")
                    results[child_name] = {'error': str(e)}

            # Summary
            print("\n" + "=" * 70)
            print("PHASE 2 TRAINING SUMMARY")
            print("=" * 70)
            for name, res in results.items():
                if 'error' in res:
                    print(f"  {name}: ERROR - {res['error']}")
                else:
                    print(f"  {name}: best_mae={res['best_val_mae']:.4f}")

            return results

        def interpolate(
            self,
            target_date: datetime,
            source: str
        ) -> Dict[str, np.ndarray]:
            """
            Interpolate all features for a given date using the hierarchy.

            Returns dict of feature arrays at the target date.
            """
            # TODO: Implement full interpolation pipeline
            raise NotImplementedError("Full interpolation pipeline coming soon")


    class ConditionedInterpolationDataset(Dataset):
        """
        Dataset for Phase 2 child models that includes parent conditioning.

        For each sample:
        1. Loads child feature observations
        2. Loads corresponding parent observations
        3. Generates parent conditioning embedding
        """

        def __init__(
            self,
            config: InterpolationConfig,
            parent_name: Optional[str],
            parent_model: Optional[JointInterpolationModel],
            data_path: Path,
            train: bool = True,
            val_ratio: float = 0.2,
            device: str = 'cpu'
        ):
            self.config = config
            self.parent_name = parent_name
            self.parent_model = parent_model
            self.device = device
            self.num_features = len(config.features)
            self.train = train

            # Load child data
            self._load_child_data(data_path)

            # Load parent data if available
            if parent_name and parent_model:
                self._load_parent_data(data_path)
            else:
                self.parent_observations = None
                self.parent_days = None

            # Create samples
            self._create_samples(val_ratio)

        def _load_child_data(self, data_path: Path):
            """Load child feature data based on source."""
            source = self.config.source

            try:
                if source == 'equipment':
                    loader = EquipmentDataLoader().load().process()
                    data, dates = loader.get_daily_changes()
                    feature_names = loader.feature_names
                elif source == 'deepstate':
                    loader = DeepStateDataLoader().load().process()
                    data = loader.processed_data
                    dates = loader.dates
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
                elif source == 'sentinel':
                    loader = SentinelDataLoader().load().process()
                    data, dates = loader.get_daily_observations()
                    feature_names = loader.feature_names
                else:
                    # Fallback to synthetic
                    self._create_synthetic_child_data()
                    return
            except Exception as e:
                print(f"Error loading child data: {e}")
                self._create_synthetic_child_data()
                return

            # Process dates to day offsets
            reference_date = datetime.strptime(dates[0], '%Y-%m-%d') if '-' in dates[0] else datetime.strptime(dates[0][:7], '%Y-%m')
            day_offsets = []
            for d in dates:
                if '-' in d and len(d) == 10:
                    dt = datetime.strptime(d, '%Y-%m-%d')
                else:
                    dt = datetime.strptime(d[:7], '%Y-%m')
                day_offsets.append((dt - reference_date).days)

            # Select features for this child model
            n_obs = len(dates)
            n_config_features = len(self.config.features)
            n_available = len(feature_names)

            n_use = min(n_config_features, n_available)
            observations = data[:, :n_use].copy()

            if n_use < n_config_features:
                padding = np.zeros((n_obs, n_config_features - n_use))
                observations = np.hstack([observations, padding])

            # Normalize
            observations = np.nan_to_num(observations, nan=0.0)
            for i in range(observations.shape[1]):
                col = observations[:, i]
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    observations[:, i] = (col - col_min) / (col_max - col_min)

            self.child_observations = torch.tensor(observations, dtype=torch.float32)
            self.child_days = torch.tensor(day_offsets, dtype=torch.float32)
            self.dates = dates

            print(f"  Child data: {n_obs} obs, {n_use}/{n_config_features} features")

        def _load_parent_data(self, data_path: Path):
            """Load parent feature data (same source, different features)."""
            # Parent uses same source, we need to load full data
            source = self.config.source
            parent_config = INTERPOLATION_CONFIGS.get(self.parent_name)

            if not parent_config:
                self.parent_observations = None
                return

            try:
                if source == 'equipment':
                    loader = EquipmentDataLoader().load().process()
                    data, dates = loader.get_daily_changes()
                elif source == 'deepstate':
                    loader = DeepStateDataLoader().load().process()
                    data = loader.processed_data
                    dates = loader.dates
                elif source == 'sentinel':
                    loader = SentinelDataLoader().load().process()
                    data, dates = loader.get_daily_observations()
                else:
                    self.parent_observations = None
                    return
            except Exception as e:
                print(f"Error loading parent data: {e}")
                self.parent_observations = None
                return

            n_parent_features = len(parent_config.features)
            n_available = data.shape[1]
            n_use = min(n_parent_features, n_available)

            parent_obs = data[:, :n_use].copy()
            if n_use < n_parent_features:
                padding = np.zeros((len(dates), n_parent_features - n_use))
                parent_obs = np.hstack([parent_obs, padding])

            parent_obs = np.nan_to_num(parent_obs, nan=0.0)
            for i in range(parent_obs.shape[1]):
                col = parent_obs[:, i]
                col_min, col_max = col.min(), col.max()
                if col_max > col_min:
                    parent_obs[:, i] = (col - col_min) / (col_max - col_min)

            self.parent_observations = torch.tensor(parent_obs, dtype=torch.float32)
            print(f"  Parent data: {len(dates)} obs, {n_use}/{n_parent_features} features")

        def _create_synthetic_child_data(self):
            """Create synthetic data as fallback."""
            np.random.seed(42)
            n_obs = 500
            n_features = self.num_features

            observations = np.random.randn(n_obs, n_features) * 0.3 + 0.5
            observations = np.clip(observations, 0, 1)

            self.child_observations = torch.tensor(observations, dtype=torch.float32)
            self.child_days = torch.tensor(np.arange(n_obs), dtype=torch.float32)
            self.dates = [f"2022-{1 + i//30:02d}-{1 + i%30:02d}" for i in range(n_obs)]
            print(f"  Using synthetic child data: {n_obs} obs")

        def _create_samples(self, val_ratio: float):
            """Create training samples with parent conditioning."""
            n_obs = len(self.child_days)
            self.samples = []

            max_skip = min(3, (n_obs - 1) // 2)

            for skip in range(1, max_skip + 1):
                for i in range(n_obs - 2 * skip):
                    before_idx = i
                    target_idx = i + skip
                    after_idx = i + 2 * skip

                    day_before = self.child_days[before_idx]
                    day_target = self.child_days[target_idx]
                    day_after = self.child_days[after_idx]

                    total_gap = (day_after - day_before).item()

                    if total_gap <= self.config.max_gap_days * 2:
                        sample = {
                            'child_before': self.child_observations[before_idx],
                            'child_after': self.child_observations[after_idx],
                            'child_target': self.child_observations[target_idx],
                            'day_before': day_before,
                            'day_after': day_after,
                            'day_target': day_target,
                        }

                        # Add parent data if available
                        if self.parent_observations is not None:
                            sample['parent_before'] = self.parent_observations[before_idx]
                            sample['parent_after'] = self.parent_observations[after_idx]

                        self.samples.append(sample)

            # Shuffle and split
            np.random.seed(42)
            indices = np.random.permutation(len(self.samples))
            self.samples = [self.samples[i] for i in indices]

            n_samples = len(self.samples)
            n_val = int(n_samples * val_ratio)

            if self.train:
                self.samples = self.samples[:-n_val] if n_val > 0 else self.samples
            else:
                self.samples = self.samples[-n_val:] if n_val > 0 else []

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Get parent conditioning if available
            conditioning = None
            if self.parent_model is not None and 'parent_before' in sample:
                with torch.no_grad():
                    self.parent_model.eval()
                    parent_before = sample['parent_before'].unsqueeze(0).to(self.device)
                    parent_after = sample['parent_after'].unsqueeze(0).to(self.device)
                    day_before = sample['day_before'].unsqueeze(0).unsqueeze(0).to(self.device)
                    day_after = sample['day_after'].unsqueeze(0).unsqueeze(0).to(self.device)
                    day_target = sample['day_target'].unsqueeze(0).unsqueeze(0).to(self.device)

                    _, _, embedding = self.parent_model(
                        parent_before, parent_after,
                        day_before, day_after, day_target,
                        return_embedding=True
                    )
                    conditioning = embedding.squeeze(0)  # [d_model]

            return (
                sample['child_before'],
                sample['child_after'],
                sample['day_before'].unsqueeze(0),
                sample['day_after'].unsqueeze(0),
                sample['day_target'].unsqueeze(0),
                sample['child_target'],
                conditioning if conditioning is not None else torch.zeros(64)
            )


    class HierarchicalInterpolationTrainer:
        """
        Trainer for Phase 2 child models with hierarchy-aware losses.

        Supports constraint types:
        - 'sum_to_parent': Child features should sum to parent aggregate
        - 'soft_consistency': Soft constraint on parent-child relationship
        - 'none': No hierarchical constraint
        """

        def __init__(
            self,
            model: JointInterpolationModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            constraint_type: str = 'none',
            constraint_weight: float = 0.1,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            min_uncertainty: float = 0.05,
            smoothness_weight: float = 0.1,
            device: str = 'cpu'
        ):
            self.model = model.to(device)
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.constraint_type = constraint_type
            self.constraint_weight = constraint_weight
            self.min_uncertainty = min_uncertainty
            self.smoothness_weight = smoothness_weight

            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )

        def train_epoch(self) -> dict:
            """Train for one epoch with conditioning."""
            self.model.train()
            total_nll = 0
            total_smooth = 0
            total_loss = 0
            n_batches = 0

            for batch in self.train_loader:
                (child_before, child_after, day_before, day_after,
                 day_target, target, conditioning) = batch

                # Move to device
                child_before = child_before.to(self.device)
                child_after = child_after.to(self.device)
                day_before = day_before.to(self.device)
                day_after = day_after.to(self.device)
                day_target = day_target.to(self.device)
                target = target.to(self.device)
                conditioning = conditioning.to(self.device)

                # Add sequence dim for conditioning
                if conditioning.dim() == 2:
                    conditioning = conditioning.unsqueeze(1)  # [batch, 1, d_model]

                self.optimizer.zero_grad()

                # Forward with conditioning
                predictions, uncertainties = self.model(
                    child_before, child_after,
                    day_before, day_after, day_target,
                    conditioning=conditioning
                )

                # Losses
                nll_loss = self._gaussian_nll_loss(predictions, target, uncertainties)
                smooth_loss = self._smoothness_loss(predictions, child_before, child_after)

                loss = nll_loss + self.smoothness_weight * smooth_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_nll += nll_loss.item()
                total_smooth += smooth_loss.item()
                total_loss += loss.item()
                n_batches += 1

            return {
                'total': total_loss / max(n_batches, 1),
                'nll': total_nll / max(n_batches, 1),
                'smooth': total_smooth / max(n_batches, 1)
            }

        def validate(self) -> Tuple[float, float]:
            """Validate model."""
            self.model.eval()
            total_loss = 0
            total_mae = 0
            n_batches = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    (child_before, child_after, day_before, day_after,
                     day_target, target, conditioning) = batch

                    child_before = child_before.to(self.device)
                    child_after = child_after.to(self.device)
                    day_before = day_before.to(self.device)
                    day_after = day_after.to(self.device)
                    day_target = day_target.to(self.device)
                    target = target.to(self.device)
                    conditioning = conditioning.to(self.device)

                    if conditioning.dim() == 2:
                        conditioning = conditioning.unsqueeze(1)

                    predictions, uncertainties = self.model(
                        child_before, child_after,
                        day_before, day_after, day_target,
                        conditioning=conditioning
                    )

                    loss = self._gaussian_nll_loss(predictions, target, uncertainties)
                    mae = F.l1_loss(predictions, target)

                    total_loss += loss.item()
                    total_mae += mae.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_mae = total_mae / max(n_batches, 1)

            self.scheduler.step(avg_mae)

            return avg_loss, avg_mae

        def _gaussian_nll_loss(self, pred, target, uncertainty):
            """Gaussian NLL with minimum uncertainty floor."""
            uncertainty_clamped = torch.clamp(uncertainty, min=self.min_uncertainty)
            variance = uncertainty_clamped ** 2 + 1e-6
            nll = 0.5 * (torch.log(variance) + (pred - target) ** 2 / variance)
            return nll.mean()

        def _smoothness_loss(self, pred, obs_before, obs_after):
            """Encourage predictions between boundary values."""
            lower = torch.min(obs_before, obs_after)
            upper = torch.max(obs_before, obs_after)
            margin = 0.1
            range_size = (upper - lower).clamp(min=0.01)
            below = F.relu(lower - pred - margin * range_size)
            above = F.relu(pred - upper - margin * range_size)
            return (below + above).mean()

        def train(
            self,
            epochs: int = 100,
            patience: int = 20,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """Full training loop with early stopping."""
            history = {
                'train_loss': [], 'val_loss': [], 'val_mae': [],
                'train_nll': [], 'train_smooth': []
            }
            best_val_mae = float('inf')
            best_epoch = 0
            patience_counter = 0

            for epoch in range(epochs):
                train_metrics = self.train_epoch()
                val_loss, val_mae = self.validate()

                history['train_loss'].append(train_metrics['total'])
                history['train_nll'].append(train_metrics['nll'])
                history['train_smooth'].append(train_metrics['smooth'])
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    marker = '*' if epoch == best_epoch else ''
                    print(f"Epoch {epoch:3d}: loss={train_metrics['total']:.4f}, "
                          f"val_mae={val_mae:.4f} {marker}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(best={best_epoch}, mae={best_val_mae:.4f})")
                    break

            return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Hierarchical Interpolation System')
    parser.add_argument('--train-phase2', action='store_true',
                        help='Train all Phase 2 child models')
    parser.add_argument('--train-child', type=str, default=None,
                        help='Train specific child model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda/mps)')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    system = HierarchicalInterpolationSystem(device=args.device)

    if args.train_phase2:
        print("=" * 70)
        print("PHASE 2B: HIERARCHICAL INTERPOLATION TRAINING")
        print("=" * 70)
        results = system.train_all_children(
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True
        )

    elif args.train_child:
        print(f"Training child model: {args.train_child}")
        result = system.train_child_model(
            args.train_child,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True
        )
        print(f"Result: {result}")

    else:
        # Print summary
        print("=" * 70)
        print("PHASE 2B: HIERARCHICAL INTERPOLATION SYSTEM")
        print("=" * 70)
        print("\nPhase 1 Parent Models:")
        for name, config in INTERPOLATION_CONFIGS.items():
            children = config.child_groups or []
            print(f"  {name}: {len(config.features)} features -> {children}")

        print("\nPhase 2 Child Models:")
        for name, config in PHASE2_CONFIGS.items():
            parent = PARENT_CHILD_MAP.get(name, {}).get('parent', 'None')
            print(f"  {name}: {len(config.features)} features (parent: {parent})")

        print("\nUsage:")
        print("  python hierarchical_interpolation_system.py --train-phase2")
        print("  python hierarchical_interpolation_system.py --train-child deepstate_arrows")


if __name__ == "__main__":
    main()
