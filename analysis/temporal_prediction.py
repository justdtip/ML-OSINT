"""
Temporal Prediction Model

Builds on top of the frozen unified delta model to predict future values.

Architecture:
    Frozen Unified Encoders → Fused Latent → Shared Temporal Context → Multi-Head Predictions

Features:
    - Frozen unified encoder (leverages pre-trained cross-source representations)
    - Shared LSTM/Transformer temporal context over sliding window
    - Multiple prediction heads (one per source)
    - Multi-horizon predictions (T+1, T+3, T+7)
    - Multi-task learning for regularization

Usage:
    python temporal_prediction.py --train --epochs 100
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json
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

from unified_interpolation_delta import (
    SOURCE_CONFIGS,
    UnifiedInterpolationModelDelta,
    extract_equipment_delta_features,
    MODEL_DIR
)
from interpolation_data_loaders import (
    SentinelDataLoader,
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader
)
from training_utils import WarmupCosineScheduler, GradientAccumulator, TimeSeriesAugmentation
from training_config import DataConfig, TrainingConfig

# Import centralized output path
from config.paths import TEMPORAL_PREDICTION_OUTPUT_DIR

RESULTS_DIR = TEMPORAL_PREDICTION_OUTPUT_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TemporalConfig:
    """Configuration for temporal prediction model."""
    # Sliding window size (days of history)
    window_size: int = 14

    # Prediction horizons
    horizons: List[int] = field(default_factory=lambda: [1, 3, 7])

    # Sources to exclude (e.g., Sentinel has monthly data, not daily)
    exclude_sources: List[str] = field(default_factory=lambda: ['sentinel'])

    # Temporal encoder - REDUCED for 781 training samples
    # Rule: ~50 samples/param minimum, so max ~15k trainable params
    temporal_hidden: int = 64       # Reduced from 256 (4x smaller)
    temporal_layers: int = 1        # Reduced from 2 (simpler model)
    temporal_dropout: float = 0.4   # Increased from 0.1 (aggressive regularization)
    temporal_type: str = "lstm"     # "lstm" or "transformer"

    # Prediction heads - REDUCED
    head_hidden: int = 32           # Reduced from 128 (4x smaller)

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4     # Reduced from 5e-4 (slower learning)
    weight_decay: float = 1e-2      # Increased from 1e-5 (1000x stronger L2)
    epochs: int = 200
    patience: int = 30
    warmup_epochs: int = 10  # Warmup epochs for WarmupCosineScheduler
    min_lr: float = 1e-7  # Minimum learning rate
    grad_clip: float = 0.5  # Gradient clipping threshold (reduced from 1.0)

    # Data split configuration
    temporal_gap: int = 7  # Gap in days between train/val and val/test splits
    val_ratio: float = 0.15  # Validation set ratio
    test_ratio: float = 0.1  # Held-out test set ratio

    # Uncertainty quantification
    mc_dropout_samples: int = 10  # Number of MC Dropout samples for uncertainty estimation

    # Loss weights per source (can prioritize certain predictions)
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        'equipment': 1.5,  # Prioritize equipment predictions
        'ucdp': 1.2,       # Conflict events important
        'firms': 1.0,
        'deepstate': 1.0,
        'sentinel': 0.8    # Lower priority for satellite
    })

    # Horizon weights (prioritize near-term)
    horizon_weights: Dict[int, float] = field(default_factory=lambda: {
        1: 1.0,
        3: 0.8,
        7: 0.6
    })


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class TemporalEncoder(nn.Module):
        """
        Temporal context encoder using LSTM or Transformer.

        Takes a sequence of fused latent vectors and produces a context vector.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            encoder_type: str = "lstm"
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.encoder_type = encoder_type
            self.dropout = dropout

            # Input dropout for regularization (drops entire features across time)
            self.input_dropout = nn.Dropout(dropout)

            if encoder_type == "lstm":
                self.encoder = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=False
                )
                self.output_dim = hidden_dim

            elif encoder_type == "transformer":
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                    activation='gelu'
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output_proj = nn.Linear(input_dim, hidden_dim)
                self.output_dim = hidden_dim
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

            self.norm = nn.LayerNorm(hidden_dim)
            # Output dropout after normalization
            self.output_dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: [batch, seq_len, input_dim] - sequence of latent vectors

            Returns:
                context: [batch, hidden_dim] - temporal context vector
            """
            # Apply input dropout (regularizes the frozen latent representations)
            x = self.input_dropout(x)

            if self.encoder_type == "lstm":
                output, (h_n, c_n) = self.encoder(x)
                # Use final hidden state
                context = h_n[-1]  # [batch, hidden_dim]
            else:
                output = self.encoder(x)  # [batch, seq_len, input_dim]
                # Use mean pooling over sequence
                context = self.output_proj(output.mean(dim=1))  # [batch, hidden_dim]

            # Apply output dropout after normalization
            return self.output_dropout(self.norm(context))


    class PredictionHead(nn.Module):
        """
        Prediction head for a single source.

        Predicts multiple horizons simultaneously.
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 64,
            horizons: List[int] = [1, 3, 7],
            dropout: float = 0.4  # Added dropout parameter
        ):
            super().__init__()
            self.output_dim = output_dim
            self.horizons = horizons

            # Shared feature extraction with strong dropout for regularization
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)  # Use configurable dropout
            )

            # Per-horizon output layers with dropout before final prediction
            self.horizon_heads = nn.ModuleDict({
                f"h{h}": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),  # Lighter dropout before output
                    nn.Linear(hidden_dim, output_dim)
                )
                for h in horizons
            })

        def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
            """
            Args:
                x: [batch, input_dim] - temporal context

            Returns:
                predictions: Dict[horizon, [batch, output_dim]]
            """
            shared_features = self.shared(x)

            predictions = {}
            for h in self.horizons:
                predictions[h] = self.horizon_heads[f"h{h}"](shared_features)

            return predictions


    class TemporalPredictionModel(nn.Module):
        """
        Multi-head temporal prediction model.

        Architecture:
            Frozen Unified Encoders
                    ↓
            Fused Latent [batch, seq, 5*64]
                    ↓
            Shared Temporal Encoder (LSTM/Transformer)
                    ↓
            Temporal Context [batch, hidden]
                    ↓
            ├── Equipment Head → T+1, T+3, T+7 predictions
            ├── UCDP Head → T+1, T+3, T+7 predictions
            ├── FIRMS Head → T+1, T+3, T+7 predictions
            ├── DeepState Head → T+1, T+3, T+7 predictions
            └── Sentinel Head → T+1, T+3, T+7 predictions
        """

        def __init__(
            self,
            unified_model: UnifiedInterpolationModelDelta,
            config: TemporalConfig,
            source_configs: Dict,
            prediction_sources: List[str] = None
        ):
            super().__init__()
            self.config = config
            # Sources used for prediction heads (excludes sources without enough data)
            self.source_names = prediction_sources or list(source_configs.keys())
            self.source_configs = {k: v for k, v in source_configs.items() if k in self.source_names}
            # All sources in the unified model (for latent extraction)
            self.unified_source_names = list(unified_model.source_names)

            # Freeze the unified model
            self.unified_model = unified_model
            for param in self.unified_model.parameters():
                param.requires_grad = False
            self.unified_model.eval()

            # Calculate input dimension (concatenated fused embeddings from ALL unified sources)
            d_embed = unified_model.d_embed
            n_unified_sources = len(self.unified_source_names)
            latent_dim = d_embed * n_unified_sources  # 64 * 5 = 320 (still uses all sources)

            # Shared temporal encoder
            self.temporal_encoder = TemporalEncoder(
                input_dim=latent_dim,
                hidden_dim=config.temporal_hidden,
                num_layers=config.temporal_layers,
                dropout=config.temporal_dropout,
                encoder_type=config.temporal_type
            )

            # Per-source prediction heads with dropout for regularization
            self.prediction_heads = nn.ModuleDict()
            for name, src_config in source_configs.items():
                self.prediction_heads[name] = PredictionHead(
                    input_dim=config.temporal_hidden,
                    output_dim=src_config.n_features,
                    hidden_dim=config.head_hidden,
                    horizons=config.horizons,
                    dropout=config.temporal_dropout  # Pass dropout from config
                )

        def extract_latent_sequence(
            self,
            features_sequence: List[Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
            """
            Extract latent representations for a sequence of time steps.

            Args:
                features_sequence: List of feature dicts, one per time step

            Returns:
                latent_sequence: [batch, seq_len, latent_dim]
            """
            latent_list = []

            with torch.no_grad():
                for features in features_sequence:
                    # Get fused embeddings from unified model
                    outputs = self.unified_model(features, return_reconstructions=False)
                    fused = outputs['fused_embeddings']

                    # Concatenate all source embeddings
                    concat = torch.cat([fused[name] for name in self.source_names], dim=-1)
                    latent_list.append(concat)

            return torch.stack(latent_list, dim=1)  # [batch, seq_len, latent_dim]

        def forward(
            self,
            latent_sequence: torch.Tensor
        ) -> Dict[str, Dict[int, torch.Tensor]]:
            """
            Args:
                latent_sequence: [batch, seq_len, latent_dim] - pre-computed latent sequence

            Returns:
                predictions: Dict[source_name, Dict[horizon, [batch, n_features]]]
            """
            # Temporal encoding
            context = self.temporal_encoder(latent_sequence)

            # Per-source predictions
            predictions = {}
            for name in self.source_names:
                predictions[name] = self.prediction_heads[name](context)

            return predictions

        def predict_with_uncertainty(
            self,
            latent_sequence: torch.Tensor,
            n_samples: int = 10
        ) -> Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
            """
            MC Dropout uncertainty estimation.

            Runs multiple forward passes with dropout enabled to estimate
            prediction uncertainty via Monte Carlo sampling.

            Args:
                latent_sequence: [batch, seq_len, latent_dim] - pre-computed latent sequence
                n_samples: Number of MC Dropout samples (default: 10)

            Returns:
                uncertainty_predictions: Dict[source_name, Dict[horizon, (mean, std)]]
                    where mean and std are tensors of shape [batch, n_features]
            """
            # Enable dropout for MC sampling
            self.train()
            # Keep unified model frozen in eval mode
            self.unified_model.eval()

            # Collect predictions from multiple forward passes
            all_predictions = {
                src: {h: [] for h in self.config.horizons}
                for src in self.source_names
            }

            for _ in range(n_samples):
                with torch.no_grad():
                    preds = self.forward(latent_sequence)
                    for src in self.source_names:
                        for h in self.config.horizons:
                            all_predictions[src][h].append(preds[src][h])

            # Back to eval mode
            self.eval()

            # Compute mean and std across samples
            uncertainty_predictions = {}
            for src in self.source_names:
                uncertainty_predictions[src] = {}
                for h in self.config.horizons:
                    stacked = torch.stack(all_predictions[src][h], dim=0)  # [n_samples, batch, n_features]
                    mean_pred = stacked.mean(dim=0)
                    std_pred = stacked.std(dim=0)
                    uncertainty_predictions[src][h] = (mean_pred, std_pred)

            return uncertainty_predictions


    # =============================================================================
    # DATASET
    # =============================================================================

    class TemporalPredictionDataset(Dataset):
        """
        Dataset for temporal prediction training.

        Creates sliding windows of latent representations with future targets.
        Supports three-way split (train/val/test) with configurable temporal gaps
        between splits to prevent data leakage.
        """

        def __init__(
            self,
            unified_model: UnifiedInterpolationModelDelta,
            source_configs: Dict,
            config: TemporalConfig,
            device: torch.device,
            split: str = 'train',
            temporal_gap: Optional[int] = None,
            val_ratio: Optional[float] = None,
            test_ratio: Optional[float] = None,
            norm_stats: Optional[Dict] = None
        ):
            """
            Initialize the temporal prediction dataset.

            Args:
                unified_model: Frozen unified model for latent extraction
                source_configs: Dictionary of source configurations
                config: TemporalConfig instance
                device: PyTorch device
                split: One of 'train', 'val', or 'test'
                temporal_gap: Gap in days between splits (overrides config)
                val_ratio: Validation ratio (overrides config)
                test_ratio: Test ratio (overrides config)
                norm_stats: Pre-computed normalization stats from training split.
                    Required for 'val' and 'test' splits to ensure consistent normalization.
                    Contains {'means': {...}, 'stds': {...}} for each source.
            """
            self.config = config
            self.split = split
            self.device = device

            # Use provided values or fall back to config
            self.temporal_gap = temporal_gap if temporal_gap is not None else config.temporal_gap
            self.val_ratio = val_ratio if val_ratio is not None else config.val_ratio
            self.test_ratio = test_ratio if test_ratio is not None else config.test_ratio

            # All source configs (for unified model)
            self.all_source_configs = source_configs
            self.all_source_names = list(source_configs.keys())

            # Sources to use for prediction (exclude ones without enough temporal data)
            self.prediction_source_names = [
                name for name in source_configs.keys()
                if name not in config.exclude_sources
            ]

            if config.exclude_sources:
                print(f"Excluding from predictions: {config.exclude_sources}")
                print(f"Predicting for: {self.prediction_source_names}")

            # Load all source data
            print("Loading source data for temporal prediction...")
            self.source_data = {}
            self.feature_names = {}

            # First pass: load all sources and find min samples among PREDICTION sources only
            min_prediction_samples = float('inf')

            for name, src_config in source_configs.items():
                loader = src_config.loader_class().load().process()
                data = loader.processed_data

                # Apply delta filtering for equipment
                if name == 'equipment' and src_config.use_delta_only:
                    if hasattr(loader, 'feature_names'):
                        data, feat_names = extract_equipment_delta_features(
                            data, loader.feature_names
                        )
                        self.feature_names[name] = feat_names
                    else:
                        self.feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]
                else:
                    if hasattr(loader, 'feature_names'):
                        self.feature_names[name] = loader.feature_names
                    else:
                        self.feature_names[name] = [f"feat_{i}" for i in range(data.shape[1])]

                self.source_data[name] = torch.tensor(data, dtype=torch.float32)

                # Only count prediction sources for alignment
                if name in self.prediction_source_names:
                    min_prediction_samples = min(min_prediction_samples, len(data))
                    print(f"  {name}: {data.shape} (prediction)")
                else:
                    print(f"  {name}: {data.shape} (latent only, excluded from prediction)")

            # Align based on prediction sources
            self.n_samples = int(min_prediction_samples)
            for name in self.source_data:
                # Truncate or repeat excluded sources to match
                if len(self.source_data[name]) < self.n_samples:
                    # For excluded sources with less data, repeat last sample
                    shortage = self.n_samples - len(self.source_data[name])
                    last_sample = self.source_data[name][-1:].expand(shortage, -1)
                    self.source_data[name] = torch.cat([self.source_data[name], last_sample], dim=0)
                else:
                    self.source_data[name] = self.source_data[name][:self.n_samples]

            print(f"  Aligned to {self.n_samples} samples (based on prediction sources)")

            # Normalize features for better training (store stats for denormalization)
            # CRITICAL FIX: Compute stats on TRAINING data only, share with val/test
            self.feature_means = {}
            self.feature_stds = {}
            print("  Normalizing features...")

            if split == 'train':
                # Compute split boundaries first to know training range
                n_valid = self.n_samples - config.window_size - max(config.horizons)
                available_for_splits = n_valid - 2 * self.temporal_gap
                train_ratio = 1.0 - self.val_ratio - self.test_ratio
                train_size = int(available_for_splits * train_ratio)
                train_end_idx = config.window_size + train_size

                # Compute stats ONLY on training portion
                for name in self.prediction_source_names:
                    train_data = self.source_data[name][:train_end_idx]
                    mean = train_data.mean(dim=0, keepdim=True)
                    # CRITICAL: Use minimum std of 0.1 to prevent explosion with constant features
                    std = torch.clamp(train_data.std(dim=0, keepdim=True), min=0.1)
                    self.feature_means[name] = mean
                    self.feature_stds[name] = std
                    # Apply to ALL data using training stats
                    self.source_data[name] = (self.source_data[name] - mean) / std

            else:
                # Val/Test: Use provided stats from training
                if norm_stats is None:
                    raise ValueError(
                        f"norm_stats required for '{split}' split to ensure consistent "
                        "normalization with training data. Pass the norm_stats from "
                        "the training dataset: {'means': train_ds.feature_means, 'stds': train_ds.feature_stds}"
                    )
                self.feature_means = norm_stats['means']
                self.feature_stds = norm_stats['stds']
                for name in self.prediction_source_names:
                    mean = self.feature_means[name]
                    std = self.feature_stds[name]
                    self.source_data[name] = (self.source_data[name] - mean) / std

            # Pre-compute latent representations using frozen unified model
            print("Pre-computing latent representations...")
            self.latent_data = self._precompute_latents(unified_model)
            print(f"  Latent shape: {self.latent_data.shape}")

            # Create valid indices (accounting for window and max horizon)
            max_horizon = max(config.horizons)
            self.valid_start = config.window_size
            self.valid_end = self.n_samples - max_horizon

            # Three-way split with temporal gaps
            self.indices = self._compute_split_indices()

            print(f"  {split.capitalize()} samples: {len(self.indices)}")
            print(f"  Temporal gap between splits: {self.temporal_gap} days")

        def _compute_split_indices(self) -> List[int]:
            """
            Compute indices for train/val/test split with temporal gaps.

            The data is split chronologically:
                [train_data] [gap] [val_data] [gap] [test_data]

            This prevents temporal leakage where future information could
            influence predictions on past data.

            Returns:
                List of valid indices for the requested split
            """
            n_valid = self.valid_end - self.valid_start

            # Calculate sizes accounting for gaps
            # Total = train + gap + val + gap + test
            # We need to subtract 2 gaps from available space
            available_for_splits = n_valid - 2 * self.temporal_gap

            if available_for_splits <= 0:
                raise ValueError(
                    f"Not enough data for three-way split with temporal_gap={self.temporal_gap}. "
                    f"Available samples: {n_valid}, required minimum: {2 * self.temporal_gap + 3}"
                )

            # Calculate split sizes
            train_ratio = 1.0 - self.val_ratio - self.test_ratio
            train_size = int(available_for_splits * train_ratio)
            val_size = int(available_for_splits * self.val_ratio)
            test_size = available_for_splits - train_size - val_size  # Remainder goes to test

            # Calculate split boundaries
            train_start = self.valid_start
            train_end = train_start + train_size

            val_start = train_end + self.temporal_gap  # Gap after train
            val_end = val_start + val_size

            test_start = val_end + self.temporal_gap  # Gap after val
            test_end = self.valid_end

            # Store split info for debugging/logging
            self.split_info = {
                'train': (train_start, train_end, train_size),
                'val': (val_start, val_end, val_size),
                'test': (test_start, test_end, test_size),
                'temporal_gap': self.temporal_gap,
                'total_valid': n_valid
            }

            # Return indices for the requested split
            if self.split == 'train':
                return list(range(train_start, train_end))
            elif self.split == 'val':
                return list(range(val_start, val_end))
            elif self.split == 'test':
                return list(range(test_start, test_end))
            else:
                raise ValueError(f"Unknown split: {self.split}. Must be 'train', 'val', or 'test'")

        def _precompute_latents(self, unified_model: UnifiedInterpolationModelDelta) -> torch.Tensor:
            """Pre-compute all latent representations using ALL sources (for rich embeddings)."""
            unified_model.eval()
            latents = []

            # Use the unified model's source order for consistent concatenation
            unified_source_order = unified_model.source_names

            with torch.no_grad():
                # Process in batches
                batch_size = 64
                for start in range(0, self.n_samples, batch_size):
                    end = min(start + batch_size, self.n_samples)

                    # Move features to device for forward pass (use ALL sources)
                    features = {
                        name: self.source_data[name][start:end].to(self.device)
                        for name in self.all_source_names
                    }

                    outputs = unified_model(features, return_reconstructions=False)
                    fused = outputs['fused_embeddings']

                    # Concatenate all source embeddings in unified model's order
                    concat = torch.cat([fused[name] for name in unified_source_order], dim=-1)
                    latents.append(concat.cpu())

            return torch.cat(latents, dim=0)  # [n_samples, latent_dim]

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Dict[int, torch.Tensor]]]:
            """
            Returns:
                latent_window: [window_size, latent_dim]
                targets: Dict[source_name, Dict[horizon, [n_features]]] (only prediction sources)
            """
            t = self.indices[idx]

            # Get window of latents ending at t-1 (predicting from t onwards)
            window_start = t - self.config.window_size
            latent_window = self.latent_data[window_start:t]

            # Get targets for each horizon (only for prediction sources)
            targets = {}
            for name in self.prediction_source_names:
                targets[name] = {}
                for h in self.config.horizons:
                    targets[name][h] = self.source_data[name][t + h - 1]  # T+h target

            return latent_window, targets


    # =============================================================================
    # TRAINING
    # =============================================================================

    class TemporalTrainer:
        """Trainer for temporal prediction model."""

        def __init__(
            self,
            model: TemporalPredictionModel,
            config: TemporalConfig,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            test_loader: Optional[DataLoader] = None
        ):
            self.model = model
            self.config = config
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.device = device

            # Only optimize non-frozen parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            # Use WarmupCosineScheduler instead of ReduceLROnPlateau
            self.scheduler = WarmupCosineScheduler(
                optimizer=self.optimizer,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epochs,
                warmup_start_lr=config.learning_rate * 0.01,  # Start at 1% of base LR
                min_lr=config.min_lr
            )

            self.best_val_loss = float('inf')
            self.patience_counter = 0
            self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': [], 'learning_rate': []}

        def compute_loss(
            self,
            predictions: Dict[str, Dict[int, torch.Tensor]],
            targets: Dict[str, Dict[int, torch.Tensor]]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute weighted multi-task loss."""
            total_loss = 0.0
            loss_details = {}

            for source_name in self.model.source_names:
                source_weight = self.config.source_weights.get(source_name, 1.0)

                for horizon in self.config.horizons:
                    horizon_weight = self.config.horizon_weights.get(horizon, 1.0)

                    pred = predictions[source_name][horizon]
                    target = targets[source_name][horizon]

                    # MSE loss
                    loss = F.mse_loss(pred, target)
                    weighted_loss = loss * source_weight * horizon_weight

                    total_loss += weighted_loss
                    loss_details[f"{source_name}_h{horizon}"] = loss.item()

            return total_loss, loss_details

        def train_epoch(self) -> float:
            """Train for one epoch."""
            self.model.train()
            # Keep unified model in eval mode
            self.model.unified_model.eval()

            total_loss = 0.0
            n_batches = 0

            for latent_window, targets in self.train_loader:
                latent_window = latent_window.to(self.device)
                targets = {
                    src: {h: t.to(self.device) for h, t in horizons.items()}
                    for src, horizons in targets.items()
                }

                self.optimizer.zero_grad()

                predictions = self.model(latent_window)
                loss, _ = self.compute_loss(predictions, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            return total_loss / n_batches

        @torch.no_grad()
        def validate(self) -> Tuple[float, Dict[str, float]]:
            """Validate and compute metrics."""
            self.model.eval()

            total_loss = 0.0
            all_predictions = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            all_targets = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            n_batches = 0

            for latent_window, targets in self.val_loader:
                latent_window = latent_window.to(self.device)
                targets = {
                    src: {h: t.to(self.device) for h, t in horizons.items()}
                    for src, horizons in targets.items()
                }

                predictions = self.model(latent_window)
                loss, _ = self.compute_loss(predictions, targets)

                total_loss += loss.item()
                n_batches += 1

                # Collect predictions and targets
                for src in self.model.source_names:
                    for h in self.config.horizons:
                        all_predictions[src][h].append(predictions[src][h].cpu())
                        all_targets[src][h].append(targets[src][h].cpu())

            # Compute metrics
            metrics = {}
            for src in self.model.source_names:
                for h in self.config.horizons:
                    preds = torch.cat(all_predictions[src][h], dim=0).numpy()
                    targs = torch.cat(all_targets[src][h], dim=0).numpy()

                    # RMSE
                    rmse = np.sqrt(np.mean((preds - targs) ** 2))
                    metrics[f"{src}_h{h}_rmse"] = rmse

                    # Correlation (mean across features)
                    corrs = []
                    for i in range(preds.shape[1]):
                        if np.std(preds[:, i]) > 1e-8 and np.std(targs[:, i]) > 1e-8:
                            corr = np.corrcoef(preds[:, i], targs[:, i])[0, 1]
                            if not np.isnan(corr):
                                corrs.append(corr)
                    metrics[f"{src}_h{h}_corr"] = np.mean(corrs) if corrs else 0.0

            return total_loss / n_batches, metrics

        def train(self) -> Dict[str, Any]:
            """Full training loop."""
            print(f"\nStarting training for {self.config.epochs} epochs...")
            print(f"  Device: {self.device}")
            print(f"  Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            print(f"  Using WarmupCosineScheduler with {self.config.warmup_epochs} warmup epochs")

            for epoch in range(self.config.epochs):
                train_loss = self.train_epoch()
                val_loss, val_metrics = self.validate()

                # Get current learning rate before stepping scheduler
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'].append(current_lr)

                # Step scheduler (WarmupCosineScheduler is epoch-based)
                self.scheduler.step()

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(val_metrics)

                # Early stopping with minimum improvement threshold
                min_improvement = 0.001  # Require at least 0.1% improvement
                if val_loss < self.best_val_loss * (1 - min_improvement):
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('temporal_prediction_best.pt')
                else:
                    self.patience_counter += 1

                # Calculate train/val ratio to monitor overfitting
                train_val_ratio = train_loss / val_loss if val_loss > 0 else 0

                if (epoch + 1) % 5 == 0:
                    # Sample metrics
                    eq_corr = val_metrics.get('equipment_h1_corr', 0)
                    ucdp_corr = val_metrics.get('ucdp_h1_corr', 0)
                    print(f"  Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                          f"ratio={train_val_ratio:.3f}, eq_corr={eq_corr:.3f}, ucdp_corr={ucdp_corr:.3f}, lr={current_lr:.2e}")
                    # Warn if severe overfitting
                    if train_val_ratio < 0.3:
                        print(f"    [WARNING] Train/Val ratio {train_val_ratio:.3f} indicates overfitting")

                if self.patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            # Evaluate on test set if available
            if self.test_loader is not None:
                print("\nEvaluating on held-out test set...")
                test_loss, test_metrics = self._evaluate_on_loader(self.test_loader)
                self.history['test_loss'] = test_loss
                self.history['test_metrics'] = test_metrics
                print(f"  Test loss: {test_loss:.4f}")
                for src in self.model.source_names:
                    corr = test_metrics.get(f'{src}_h1_corr', 0)
                    rmse = test_metrics.get(f'{src}_h1_rmse', 0)
                    print(f"    {src}: corr={corr:.3f}, rmse={rmse:.4f}")

            return self.history

        @torch.no_grad()
        def _evaluate_on_loader(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
            """Evaluate model on a given data loader."""
            self.model.eval()

            total_loss = 0.0
            all_predictions = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            all_targets = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            n_batches = 0

            for latent_window, targets in loader:
                latent_window = latent_window.to(self.device)
                targets = {
                    src: {h: t.to(self.device) for h, t in horizons.items()}
                    for src, horizons in targets.items()
                }

                predictions = self.model(latent_window)
                loss, _ = self.compute_loss(predictions, targets)

                total_loss += loss.item()
                n_batches += 1

                # Collect predictions and targets
                for src in self.model.source_names:
                    for h in self.config.horizons:
                        all_predictions[src][h].append(predictions[src][h].cpu())
                        all_targets[src][h].append(targets[src][h].cpu())

            # Compute metrics
            metrics = {}
            for src in self.model.source_names:
                for h in self.config.horizons:
                    preds = torch.cat(all_predictions[src][h], dim=0).numpy()
                    targs = torch.cat(all_targets[src][h], dim=0).numpy()

                    # RMSE
                    rmse = np.sqrt(np.mean((preds - targs) ** 2))
                    metrics[f"{src}_h{h}_rmse"] = rmse

                    # Correlation (mean across features)
                    corrs = []
                    for i in range(preds.shape[1]):
                        if np.std(preds[:, i]) > 1e-8 and np.std(targs[:, i]) > 1e-8:
                            corr = np.corrcoef(preds[:, i], targs[:, i])[0, 1]
                            if not np.isnan(corr):
                                corrs.append(corr)
                    metrics[f"{src}_h{h}_corr"] = np.mean(corrs) if corrs else 0.0

            return total_loss / n_batches, metrics

        def evaluate_with_uncertainty(
            self,
            loader: DataLoader,
            n_samples: Optional[int] = None
        ) -> Dict[str, Any]:
            """
            Evaluate model with MC Dropout uncertainty estimation.

            Args:
                loader: DataLoader for evaluation
                n_samples: Number of MC samples (defaults to config.mc_dropout_samples)

            Returns:
                Dictionary containing predictions, targets, uncertainties, and metrics
            """
            if n_samples is None:
                n_samples = self.config.mc_dropout_samples

            print(f"  Running MC Dropout with {n_samples} samples...")

            all_means = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            all_stds = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}
            all_targets = {src: {h: [] for h in self.config.horizons} for src in self.model.source_names}

            for latent_window, targets in loader:
                latent_window = latent_window.to(self.device)

                # Get predictions with uncertainty
                uncertainty_preds = self.model.predict_with_uncertainty(latent_window, n_samples)

                for src in self.model.source_names:
                    for h in self.config.horizons:
                        mean_pred, std_pred = uncertainty_preds[src][h]
                        all_means[src][h].append(mean_pred.cpu())
                        all_stds[src][h].append(std_pred.cpu())
                        all_targets[src][h].append(targets[src][h])

            # Compute metrics with uncertainty
            results = {'predictions': {}, 'uncertainties': {}, 'metrics': {}}

            for src in self.model.source_names:
                results['predictions'][src] = {}
                results['uncertainties'][src] = {}
                results['metrics'][src] = {}

                for h in self.config.horizons:
                    means = torch.cat(all_means[src][h], dim=0).numpy()
                    stds = torch.cat(all_stds[src][h], dim=0).numpy()
                    targs = torch.cat(all_targets[src][h], dim=0).numpy()

                    results['predictions'][src][h] = means
                    results['uncertainties'][src][h] = stds

                    # Metrics
                    rmse = np.sqrt(np.mean((means - targs) ** 2))
                    mae = np.mean(np.abs(means - targs))
                    mean_uncertainty = np.mean(stds)

                    # Calibration: check if uncertainty correlates with error
                    errors = np.abs(means - targs).mean(axis=1)
                    uncertainties = stds.mean(axis=1)
                    if np.std(uncertainties) > 1e-8:
                        calibration_corr = np.corrcoef(errors, uncertainties)[0, 1]
                    else:
                        calibration_corr = 0.0

                    results['metrics'][src][f'h{h}'] = {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mean_uncertainty': float(mean_uncertainty),
                        'calibration_corr': float(calibration_corr) if not np.isnan(calibration_corr) else 0.0
                    }

            return results

        def save_checkpoint(self, filename: str):
            """Save model checkpoint."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'best_val_loss': self.best_val_loss
            }, MODEL_DIR / filename)


    # =============================================================================
    # EVALUATION & VISUALIZATION
    # =============================================================================

    def evaluate_model(
        model: TemporalPredictionModel,
        val_loader: DataLoader,
        config: TemporalConfig,
        device: torch.device
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of temporal prediction model."""
        model.eval()

        all_predictions = {src: {h: [] for h in config.horizons} for src in model.source_names}
        all_targets = {src: {h: [] for h in config.horizons} for src in model.source_names}

        with torch.no_grad():
            for latent_window, targets in val_loader:
                latent_window = latent_window.to(device)
                predictions = model(latent_window)

                for src in model.source_names:
                    for h in config.horizons:
                        all_predictions[src][h].append(predictions[src][h].cpu())
                        all_targets[src][h].append(targets[src][h])

        results = {'by_source': {}, 'by_horizon': {}, 'summary': {}}

        for src in model.source_names:
            results['by_source'][src] = {}

            for h in config.horizons:
                preds = torch.cat(all_predictions[src][h], dim=0).numpy()
                targs = torch.cat(all_targets[src][h], dim=0).numpy()

                # Metrics
                mse = np.mean((preds - targs) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(preds - targs))

                # Per-feature correlations
                feature_corrs = []
                for i in range(preds.shape[1]):
                    if np.std(preds[:, i]) > 1e-8 and np.std(targs[:, i]) > 1e-8:
                        corr = np.corrcoef(preds[:, i], targs[:, i])[0, 1]
                        if not np.isnan(corr):
                            feature_corrs.append(corr)

                results['by_source'][src][f'h{h}'] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mean_corr': float(np.mean(feature_corrs)) if feature_corrs else 0.0,
                    'max_corr': float(np.max(feature_corrs)) if feature_corrs else 0.0,
                    'min_corr': float(np.min(feature_corrs)) if feature_corrs else 0.0,
                    'n_positive_corr': sum(1 for c in feature_corrs if c > 0)
                }

        # Aggregate by horizon
        for h in config.horizons:
            corrs = [results['by_source'][src][f'h{h}']['mean_corr'] for src in model.source_names]
            rmses = [results['by_source'][src][f'h{h}']['rmse'] for src in model.source_names]
            results['by_horizon'][f'h{h}'] = {
                'mean_corr_across_sources': float(np.mean(corrs)),
                'mean_rmse_across_sources': float(np.mean(rmses))
            }

        # Summary
        all_corrs = [
            results['by_source'][src][f'h{h}']['mean_corr']
            for src in model.source_names
            for h in config.horizons
        ]
        results['summary'] = {
            'overall_mean_corr': float(np.mean(all_corrs)),
            'best_source': max(model.source_names, key=lambda s: results['by_source'][s]['h1']['mean_corr']),
            'best_horizon': f"h{config.horizons[0]}"  # Usually h1 is best
        }

        return results


    def plot_results(
        history: Dict[str, List],
        eval_results: Dict[str, Any],
        config: TemporalConfig
    ):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            print("Matplotlib not available, skipping plots")
            return

        fig = plt.figure(figsize=(16, 12))

        # 1. Training curves
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(history['train_loss'], label='Train', alpha=0.8)
        ax1.plot(history['val_loss'], label='Val', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Correlation by source (h1)
        ax2 = fig.add_subplot(2, 3, 2)
        sources = list(eval_results['by_source'].keys())
        h1_corrs = [eval_results['by_source'][s]['h1']['mean_corr'] for s in sources]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sources)))
        bars = ax2.bar(sources, h1_corrs, color=colors)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Mean Correlation')
        ax2.set_title('T+1 Prediction Accuracy by Source')
        ax2.set_ylim(-0.2, 1.0)
        for bar, val in zip(bars, h1_corrs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        # 3. Correlation by horizon
        ax3 = fig.add_subplot(2, 3, 3)
        horizons = config.horizons
        for src in sources:
            corrs = [eval_results['by_source'][src][f'h{h}']['mean_corr'] for h in horizons]
            ax3.plot(horizons, corrs, 'o-', label=src, markersize=8)
        ax3.set_xlabel('Horizon (days)')
        ax3.set_ylabel('Mean Correlation')
        ax3.set_title('Prediction Accuracy vs Horizon')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # 4. RMSE by source
        ax4 = fig.add_subplot(2, 3, 4)
        h1_rmses = [eval_results['by_source'][s]['h1']['rmse'] for s in sources]
        bars = ax4.bar(sources, h1_rmses, color=colors)
        ax4.set_ylabel('RMSE')
        ax4.set_title('T+1 Prediction Error by Source')

        # 5. Heatmap of correlations (source x horizon)
        ax5 = fig.add_subplot(2, 3, 5)
        corr_matrix = np.array([
            [eval_results['by_source'][s][f'h{h}']['mean_corr'] for h in horizons]
            for s in sources
        ])
        im = ax5.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.8)
        ax5.set_xticks(range(len(horizons)))
        ax5.set_xticklabels([f'T+{h}' for h in horizons])
        ax5.set_yticks(range(len(sources)))
        ax5.set_yticklabels(sources)
        ax5.set_title('Correlation Heatmap')
        plt.colorbar(im, ax=ax5)

        # Add values to heatmap
        for i in range(len(sources)):
            for j in range(len(horizons)):
                ax5.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center',
                        color='black' if corr_matrix[i,j] > 0.3 else 'white', fontsize=9)

        # 6. Summary text
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
Temporal Prediction Model Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture:
  • Window size: {config.window_size} days
  • Horizons: T+{', T+'.join(map(str, config.horizons))}
  • Temporal encoder: {config.temporal_type.upper()}
  • Hidden dim: {config.temporal_hidden}

Results:
  • Overall mean correlation: {eval_results['summary']['overall_mean_corr']:.3f}
  • Best source: {eval_results['summary']['best_source']}

T+1 Performance:
"""
        for src in sources:
            corr = eval_results['by_source'][src]['h1']['mean_corr']
            rmse = eval_results['by_source'][src]['h1']['rmse']
            summary_text += f"  • {src}: r={corr:.3f}, RMSE={rmse:.3f}\n"

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'temporal_prediction_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plots to {RESULTS_DIR / 'temporal_prediction_results.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Temporal Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--window', type=int, default=14, help='Window size (days)')
    parser.add_argument('--temporal-type', choices=['lstm', 'transformer'], default='lstm',
                       help='Temporal encoder type')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/mps/cpu/auto)')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    # Device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Configuration
    config = TemporalConfig(
        window_size=args.window,
        epochs=args.epochs,
        temporal_type=args.temporal_type
    )

    # Load frozen unified model
    print("\nLoading frozen unified delta model...")
    unified_model_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'

    if not unified_model_path.exists():
        print(f"ERROR: Unified model not found at {unified_model_path}")
        print("Please train the unified delta model first.")
        return

    # Create unified model architecture
    unified_model = UnifiedInterpolationModelDelta(
        source_configs=SOURCE_CONFIGS,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2
    )

    # Load weights
    state_dict = torch.load(unified_model_path, map_location='cpu', weights_only=False)
    unified_model.load_state_dict(state_dict)
    unified_model.to(device)
    unified_model.eval()
    print("  Unified model loaded and frozen")

    if args.train:
        # Create datasets with three-way split and temporal gaps
        print("\nCreating temporal datasets with three-way split...")
        print(f"  Temporal gap between splits: {config.temporal_gap} days")
        print(f"  Val ratio: {config.val_ratio}, Test ratio: {config.test_ratio}")

        train_dataset = TemporalPredictionDataset(
            unified_model=unified_model,
            source_configs=SOURCE_CONFIGS,
            config=config,
            device=device,
            split='train'
        )

        # CRITICAL FIX: Share normalization stats from training with val/test
        # This prevents data leakage and ensures consistent feature scaling
        norm_stats = {
            'means': train_dataset.feature_means,
            'stds': train_dataset.feature_stds
        }

        val_dataset = TemporalPredictionDataset(
            unified_model=unified_model,
            source_configs=SOURCE_CONFIGS,
            config=config,
            device=device,
            split='val',
            norm_stats=norm_stats
        )
        test_dataset = TemporalPredictionDataset(
            unified_model=unified_model,
            source_configs=SOURCE_CONFIGS,
            config=config,
            device=device,
            split='test',
            norm_stats=norm_stats
        )

        # Log split information
        if hasattr(train_dataset, 'split_info'):
            info = train_dataset.split_info
            print(f"\n  Split details:")
            print(f"    Train: indices {info['train'][0]}-{info['train'][1]} ({info['train'][2]} samples)")
            print(f"    [gap of {info['temporal_gap']} days]")
            print(f"    Val:   indices {info['val'][0]}-{info['val'][1]} ({info['val'][2]} samples)")
            print(f"    [gap of {info['temporal_gap']} days]")
            print(f"    Test:  indices {info['test'][0]}-{info['test'][1]} ({info['test'][2]} samples)")

        # Get prediction sources (excluding sentinel etc.)
        prediction_sources = train_dataset.prediction_source_names

        # Custom collate function for nested dict targets
        def collate_fn(batch):
            latents = torch.stack([b[0] for b in batch])
            targets = {}
            for src in prediction_sources:
                targets[src] = {}
                for h in config.horizons:
                    targets[src][h] = torch.stack([b[1][src][h] for b in batch])
            return latents, targets

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )

        # Create temporal prediction model
        print("\nCreating temporal prediction model...")
        print(f"  Prediction sources: {prediction_sources}")
        model = TemporalPredictionModel(
            unified_model=unified_model,
            config=config,
            source_configs=SOURCE_CONFIGS,
            prediction_sources=prediction_sources
        )
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Frozen params: {total_params - trainable_params:,}")

        # Train with test loader for final evaluation
        trainer = TemporalTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            test_loader=test_loader
        )

        history = trainer.train()

        # Evaluate on validation set
        print("\nEvaluating model on validation set...")
        eval_results = evaluate_model(model, val_loader, config, device)

        # Evaluate on test set with uncertainty quantification
        print("\nEvaluating on test set with uncertainty quantification...")
        uncertainty_results = trainer.evaluate_with_uncertainty(test_loader)

        # Print uncertainty metrics
        print("\nTest Set Uncertainty Metrics:")
        for src in model.source_names:
            for h in config.horizons:
                metrics = uncertainty_results['metrics'][src][f'h{h}']
                print(f"  {src} T+{h}: RMSE={metrics['rmse']:.4f}, "
                      f"mean_uncertainty={metrics['mean_uncertainty']:.4f}, "
                      f"calibration_corr={metrics['calibration_corr']:.3f}")

        # Print results
        print("\n" + "="*60)
        print("TEMPORAL PREDICTION RESULTS")
        print("="*60)

        print("\nT+1 Performance:")
        for src in model.source_names:
            metrics = eval_results['by_source'][src]['h1']
            print(f"  {src:12}: corr={metrics['mean_corr']:.3f}, RMSE={metrics['rmse']:.3f}")

        print("\nPerformance by Horizon:")
        for h in config.horizons:
            metrics = eval_results['by_horizon'][f'h{h}']
            print(f"  T+{h}: mean_corr={metrics['mean_corr_across_sources']:.3f}, "
                  f"mean_RMSE={metrics['mean_rmse_across_sources']:.3f}")

        # Generate plots
        print("\nGenerating plots...")
        plot_results(history, eval_results, config)

        # Save results
        results_path = RESULTS_DIR / 'temporal_prediction_results.json'
        results_dict = {
            'config': {
                'window_size': config.window_size,
                'horizons': config.horizons,
                'temporal_type': config.temporal_type,
                'temporal_hidden': config.temporal_hidden,
                'temporal_gap': config.temporal_gap,
                'val_ratio': config.val_ratio,
                'test_ratio': config.test_ratio,
                'mc_dropout_samples': config.mc_dropout_samples,
                'warmup_epochs': config.warmup_epochs,
                'epochs_trained': len(history['train_loss'])
            },
            'evaluation': eval_results,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }

        # Add test metrics if available
        if 'test_loss' in history:
            results_dict['test_loss'] = history['test_loss']
            results_dict['test_metrics'] = history['test_metrics']

        # Add uncertainty metrics
        if 'uncertainty_results' in dir():
            results_dict['uncertainty_metrics'] = uncertainty_results['metrics']

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"  Saved results to {results_path}")

        print("\nDone!")


if __name__ == '__main__':
    main()
