#!/usr/bin/env python3
"""
Full Training Pipeline for Tactical State Prediction System

This script orchestrates the complete end-to-end training pipeline for the
tactical state prediction system, including:

Stage 1: Joint Interpolation Models (Phase 1)
    - Fill temporal gaps in each data source
    - Output: Interpolated daily features

Stage 2: Unified Cross-Source Model
    - Learn cross-source relationships
    - Output: Unified feature embeddings

Stage 3: Hierarchical Attention Network (HAN)
    - Multi-domain attention for state encoding
    - Output: Encoded state representations

Stage 4: Temporal Prediction Model
    - Multi-horizon forecasting (T+1, T+3, T+7)
    - Output: Feature predictions with uncertainty

Stage 5: Tactical State Predictor
    - Discrete state classification
    - State transition modeling P(S(t+1)|S(t))
    - Output: State predictions and trajectories

Usage:
    # Run full pipeline
    python train_full_pipeline.py

    # Run with custom epochs
    python train_full_pipeline.py --epochs 50

    # Quick test run
    python train_full_pipeline.py --quick

    # Resume from a specific stage
    python train_full_pipeline.py --resume 3

    # Run only a specific stage
    python train_full_pipeline.py --stage 2

Author: ML Engineering Team
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Enable MPS fallback for unsupported ops (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL,
)

# Backward compatibility alias
# ANALYSIS_DIR is now imported from config.paths

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

# Import pipeline components
# Note: DATA_DIR and MODEL_DIR are already imported from config.paths above
try:
    from joint_interpolation_models import (
        JointInterpolationModel,
        InterpolationConfig,
        InterpolationDataset,
        InterpolationTrainer,
        INTERPOLATION_CONFIGS,
    )
    HAS_INTERPOLATION = True
except ImportError as e:
    HAS_INTERPOLATION = False
    print(f"Warning: Joint interpolation models not available: {e}")

try:
    from unified_interpolation import (
        UnifiedInterpolationModel,
        CrossSourceDataset,
        UnifiedTrainer,
        SOURCE_CONFIGS,
    )
    HAS_UNIFIED = True
except ImportError as e:
    HAS_UNIFIED = False
    print(f"Warning: Unified interpolation not available: {e}")

try:
    from hierarchical_attention_network import (
        HierarchicalAttentionNetwork,
        DOMAIN_CONFIGS,
        TOTAL_FEATURES
    )
    from conflict_data_loader import create_data_loaders
    from train_han import HierarchicalAttentionTrainer
    HAS_HAN = True
except ImportError as e:
    HAS_HAN = False
    print(f"Warning: HAN not available: {e}")

try:
    from temporal_prediction import (
        TemporalPredictionModel,
        TemporalPredictionDataset,
        TemporalTrainer,
        TemporalConfig,
        evaluate_model as evaluate_temporal_model
    )
    from unified_interpolation_delta import (
        UnifiedInterpolationModelDelta,
        SOURCE_CONFIGS as DELTA_SOURCE_CONFIGS
    )
    HAS_TEMPORAL = True
except ImportError as e:
    HAS_TEMPORAL = False
    print(f"Warning: Temporal prediction not available: {e}")

try:
    from tactical_state_prediction import (
        TacticalStatePredictor,
        TacticalPredictionConfig,
        TacticalStateConfig,
        TacticalStateDataset,
        TacticalStateLoss,
        train_tactical_predictor
    )
    HAS_TACTICAL = True
except ImportError as e:
    HAS_TACTICAL = False
    print(f"Warning: Tactical state prediction not available: {e}")

try:
    from training_utils import (
        WarmupCosineScheduler,
        GradientAccumulator,
        SmartEarlyStopping,
        SWAWrapper,
        SnapshotEnsemble,
        LabelSmoothing,
        mixup_data,
        mixup_criterion,
        cutmix_data,
        CyclicLRWithRestarts,
        RegularizationScheduler
    )
    from training_config import DataConfig, TrainingConfig, ExperimentConfig
    HAS_TRAINING_UTILS = True
except ImportError as e:
    HAS_TRAINING_UTILS = False
    print(f"Warning: Training utilities not available: {e}")

# Multi-Resolution HAN (new architecture preserving daily resolution)
try:
    from multi_resolution_han import (
        MultiResolutionHAN,
        create_multi_resolution_han,
    )
    from multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
        create_multi_resolution_dataloaders,
    )
    from train_multi_resolution import (
        MultiResolutionTrainer,
        MultiTaskLoss,
        multi_resolution_collate_fn,
    )
    HAS_MULTI_RESOLUTION = True
except ImportError as e:
    HAS_MULTI_RESOLUTION = False
    print(f"Warning: Multi-resolution HAN not available: {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full training pipeline."""

    # Data settings
    resolution: str = 'weekly'
    temporal_gap: int = 14

    # Training settings per stage
    epochs_interpolation: int = 100
    epochs_unified: int = 100
    epochs_han: int = 200
    epochs_temporal: int = 150
    epochs_tactical: int = 100

    # Model settings
    d_model: int = 64
    n_states: int = 8

    # Control which stages to run
    skip_interpolation: bool = False
    skip_unified: bool = False
    skip_han: bool = False
    skip_temporal: bool = False
    skip_tactical: bool = False

    # Multi-resolution mode (Stage 3 alternative)
    # When True, uses MultiResolutionHAN which:
    # - Processes daily data at DAILY resolution (not aggregated to monthly)
    # - Processes monthly data at MONTHLY resolution
    # - Uses learned no_observation_tokens for missing data (NO interpolation)
    # - Maintains data integrity throughout
    use_multi_resolution: bool = True  # Default to new architecture

    # Checkpointing
    checkpoint_dir: str = str(PIPELINE_CHECKPOINT_DIR)
    resume_from_stage: Optional[int] = None

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    patience: int = 30

    # Early stopping flexibility
    early_stopping_strategy: str = 'smoothed'  # 'standard', 'smoothed', 'relative', 'plateau', 'combined'
    early_stopping_min_epochs: int = 50  # Don't stop before this
    early_stopping_min_delta: float = 1e-4  # Minimum improvement threshold
    early_stopping_smoothing: float = 0.9  # EMA factor for smoothed strategy
    early_stopping_relative_threshold: float = 0.1  # 10% tolerance for relative strategy
    disable_early_stopping: bool = False  # Completely disable early stopping

    # Stochastic Weight Averaging (SWA)
    use_swa: bool = True
    swa_start_pct: float = 0.75  # Start SWA at 75% of training
    swa_freq: int = 5  # Update SWA every N epochs

    # Snapshot Ensembles
    use_snapshots: bool = False
    n_snapshots: int = 5

    # Regularization
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # Learning rate scheduling
    lr_schedule: str = 'cosine'  # 'cosine', 'cosine_restarts', 'linear', 'constant'
    cosine_t0: int = 10  # Initial cycle length for cosine restarts
    cosine_t_mult: int = 2  # Cycle length multiplier

    # Device
    device: str = 'auto'

    # Logging
    verbose: bool = True
    save_history: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    @property
    def swa_start_epoch_interpolation(self) -> int:
        """Calculate SWA start epoch for interpolation stage."""
        return int(self.epochs_interpolation * self.swa_start_pct)

    @property
    def swa_start_epoch_unified(self) -> int:
        """Calculate SWA start epoch for unified stage."""
        return int(self.epochs_unified * self.swa_start_pct)

    @property
    def swa_start_epoch_han(self) -> int:
        """Calculate SWA start epoch for HAN stage."""
        return int(self.epochs_han * self.swa_start_pct)

    @property
    def swa_start_epoch_temporal(self) -> int:
        """Calculate SWA start epoch for temporal stage."""
        return int(self.epochs_temporal * self.swa_start_pct)

    @property
    def swa_start_epoch_tactical(self) -> int:
        """Calculate SWA start epoch for tactical stage."""
        return int(self.epochs_tactical * self.swa_start_pct)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, verbose: bool = True) -> logging.Logger:
    """Setup logging for the pipeline."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{timestamp}.log'

    # Create logger
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

class CheckpointManager:
    """Manages checkpoints between pipeline stages."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_stage_checkpoint(
        self,
        stage: int,
        results: Dict[str, Any],
        model_state: Optional[Dict] = None
    ) -> Path:
        """Save checkpoint for a completed stage."""
        checkpoint = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'results': self._serialize_results(results),
        }
        if model_state is not None:
            checkpoint['model_state'] = model_state

        path = self.checkpoint_dir / f'stage{stage}_checkpoint.pt'
        torch.save(checkpoint, path)
        return path

    def load_stage_checkpoint(self, stage: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a specific stage."""
        path = self.checkpoint_dir / f'stage{stage}_checkpoint.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=False)
        return None

    def get_latest_stage(self) -> int:
        """Get the latest completed stage."""
        latest = 0
        for i in range(1, 6):
            if (self.checkpoint_dir / f'stage{i}_checkpoint.pt').exists():
                latest = i
        return latest

    def save_pipeline_results(self, results: Dict[str, Any]) -> Path:
        """Save final pipeline results."""
        path = self.checkpoint_dir / 'pipeline_results.json'
        with open(path, 'w') as f:
            json.dump(self._serialize_results(results), f, indent=2)
        return path

    def _serialize_results(self, obj):
        """Recursively convert results to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._serialize_results(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_results(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Skip non-serializable objects like model configs
            return None
        else:
            return obj


# =============================================================================
# DEVICE MANAGEMENT
# =============================================================================

def get_device(device_str: str = 'auto') -> torch.device:
    """Get the appropriate device for training."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


# =============================================================================
# STAGE 1: JOINT INTERPOLATION MODELS
# =============================================================================

def train_interpolation_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, Any]:
    """
    Stage 1: Train Joint Interpolation Models

    Trains models to interpolate temporal gaps in each data source,
    producing daily features from irregularly-sampled observations.
    """
    if not HAS_INTERPOLATION:
        logger.warning("Skipping interpolation stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training joint interpolation models...")

    results = {
        'models': {},
        'histories': {},
        'metrics': {}
    }

    for name, interp_config in INTERPOLATION_CONFIGS.items():
        logger.info(f"\n  Training: {interp_config.name}")
        logger.info(f"    Features: {len(interp_config.features)}")
        logger.info(f"    Resolution: {interp_config.native_resolution_days} days")

        try:
            # Create training dataset first to compute normalization stats
            train_dataset = InterpolationDataset(
                interp_config, DATA_DIR,
                train=True,
                temporal_gap=config.temporal_gap,
                norm_stats=None
            )

            # Compute normalization statistics from training data only
            norm_stats = train_dataset.compute_norm_stats()

            # Create validation dataset with training stats
            val_dataset = InterpolationDataset(
                interp_config, DATA_DIR,
                train=False,
                temporal_gap=config.temporal_gap,
                norm_stats=norm_stats
            )

            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logger.warning(f"    Skipping {name}: insufficient data")
                continue

            # Get actual feature count from loaded data
            actual_n_features = getattr(
                train_dataset, 'actual_num_features', len(interp_config.features)
            )
            actual_feature_names = getattr(
                train_dataset, 'actual_feature_names', interp_config.features
            )

            # Create config with actual features if different
            if actual_n_features != len(interp_config.features):
                actual_config = InterpolationConfig(
                    name=interp_config.name,
                    source=interp_config.source,
                    features=actual_feature_names,
                    native_resolution_days=interp_config.native_resolution_days,
                    d_model=config.d_model,
                    nhead=interp_config.nhead,
                    num_layers=interp_config.num_layers,
                    max_gap_days=interp_config.max_gap_days,
                    dropout=interp_config.dropout,
                )
            else:
                actual_config = interp_config

            # Create model
            model = JointInterpolationModel(actual_config)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size
            )

            # Create trainer
            trainer = InterpolationTrainer(
                model, train_loader, val_loader,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                device=str(device),
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.epochs_interpolation,
                use_warmup_cosine=True
            )

            # Train
            history = trainer.train(
                epochs=config.epochs_interpolation,
                patience=config.patience,
                verbose=config.verbose
            )

            best_mae = min(history['val_mae']) if history['val_mae'] else float('inf')

            results['histories'][name] = history
            results['metrics'][name] = {
                'best_val_mae': best_mae,
                'final_val_mae': history['val_mae'][-1] if history['val_mae'] else float('inf'),
                'epochs_trained': len(history['train_loss'])
            }

            logger.info(f"    Best MAE: {best_mae:.4f}")

        except Exception as e:
            logger.error(f"    Error training {name}: {e}")
            logger.debug(traceback.format_exc())
            results['metrics'][name] = {'error': str(e)}

    return results


# =============================================================================
# STAGE 2: UNIFIED CROSS-SOURCE MODEL
# =============================================================================

def train_unified_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, Any]:
    """
    Stage 2: Train Unified Cross-Source Model

    Learns cross-source relationships through self-supervised reconstruction.
    """
    if not HAS_UNIFIED:
        logger.warning("Skipping unified stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training unified cross-source model...")

    try:
        # Create datasets with temporal split
        logger.info("  Creating datasets...")
        train_dataset = CrossSourceDataset(
            SOURCE_CONFIGS,
            train=True,
            temporal_gap=config.temporal_gap
        )
        val_dataset = CrossSourceDataset(
            SOURCE_CONFIGS,
            train=False,
            temporal_gap=config.temporal_gap,
            norm_stats=train_dataset.norm_stats
        )

        logger.info(f"    Train samples: {len(train_dataset)}")
        logger.info(f"    Val samples: {len(val_dataset)}")

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
            train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn
        )

        # Create model
        logger.info("  Creating model...")
        model = UnifiedInterpolationModel(
            source_configs=SOURCE_CONFIGS,
            d_embed=config.d_model,
            nhead=4,
            num_fusion_layers=2,
            dropout=0.1,
            pretrained_jims=None  # Could load from Stage 1 if available
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params:,}")

        # Create trainer
        trainer = UnifiedTrainer(
            model, train_loader, val_loader,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            device=str(device),
            epochs=config.epochs_unified,
            warmup_epochs=config.warmup_epochs
        )

        # Train
        history = trainer.train(
            epochs=config.epochs_unified,
            patience=config.patience,
            verbose=config.verbose
        )

        best_val_mae = min(history['val_mae']) if history['val_mae'] else float('inf')

        results = {
            'history': history,
            'metrics': {
                'best_val_mae': best_val_mae,
                'final_val_mae': history['val_mae'][-1] if history['val_mae'] else float('inf'),
                'epochs_trained': len(history['train_loss'])
            },
            'model_path': str(MODEL_DIR / 'unified_interpolation_best.pt')
        }

        logger.info(f"  Best MAE: {best_val_mae:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in unified stage: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}


# =============================================================================
# STAGE 3: HIERARCHICAL ATTENTION NETWORK
# =============================================================================

def train_han_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, Any]:
    """
    Stage 3: Train Hierarchical Attention Network

    Processes multi-domain features with hierarchical attention for
    state encoding and prediction.
    """
    if not HAS_HAN:
        logger.warning("Skipping HAN stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training Hierarchical Attention Network...")

    try:
        # Create data loaders
        logger.info("  Creating data loaders...")
        train_loader, val_loader, test_loader, norm_stats = create_data_loaders(
            DOMAIN_CONFIGS,
            batch_size=config.batch_size,
            seq_len=4,  # 4 months of context
            temporal_gap_days=config.temporal_gap,
            resolution=config.resolution
        )

        logger.info(f"    Train batches: {len(train_loader)}")
        logger.info(f"    Val batches: {len(val_loader)}")
        logger.info(f"    Test batches: {len(test_loader)}")

        # Create model
        logger.info("  Creating model...")
        model = HierarchicalAttentionNetwork(
            domain_configs=DOMAIN_CONFIGS,
            d_model=config.d_model,
            nhead=4,
            num_encoder_layers=2,
            num_temporal_layers=2,
            dropout=0.2
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params:,}")

        # Create trainer
        trainer = HierarchicalAttentionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            device=str(device),
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.epochs_han,
            accumulation_steps=8,
            use_uncertainty_loss=True
        )

        # Train
        history = trainer.train(
            epochs=config.epochs_han,
            patience=config.patience,
            verbose=config.verbose
        )

        results = {
            'history': history,
            'metrics': {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_regime_acc': history['val_regime_acc'][-1] if history['val_regime_acc'] else 0,
                'epochs_trained': len(history['train_loss'])
            },
            'model_path': str(HAN_BEST_MODEL)
        }

        logger.info(f"  Final val loss: {results['metrics']['final_val_loss']:.4f}")
        logger.info(f"  Regime accuracy: {results['metrics']['final_regime_acc']:.2%}")

        return results

    except Exception as e:
        logger.error(f"Error in HAN stage: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}


# =============================================================================
# STAGE 3 ALTERNATIVE: MULTI-RESOLUTION HAN
# =============================================================================

def train_multi_resolution_han_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, Any]:
    """
    Stage 3 (Alternative): Train Multi-Resolution Hierarchical Attention Network

    This is the preferred architecture that:
    - Processes daily data at DAILY resolution (~1400 timesteps)
    - Processes monthly data at MONTHLY resolution (~48 timesteps)
    - Uses learned no_observation_tokens for missing data (NO interpolation)
    - Maintains full data integrity - no fabrication

    Key difference from standard HAN: preserves temporal granularity instead
    of collapsing everything to monthly resolution.
    """
    if not HAS_MULTI_RESOLUTION:
        logger.warning("Skipping Multi-Resolution HAN stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training Multi-Resolution Hierarchical Attention Network...")
    logger.info("  (Using new architecture that preserves daily resolution)")

    try:
        # Create data configuration with probe-based optimizations
        # NOTE: daily_seq_len=365 is required for architecture stability (monthly aggregation)
        # The 14-day context window finding (Probe 3.1.1) should be applied via
        # attention masking or recency weighting, not sequence length reduction
        logger.info("  Creating multi-resolution datasets...")
        data_config = MultiResolutionConfig(
            daily_seq_len=365,  # Required for monthly aggregation architecture
            monthly_seq_len=12,
            prediction_horizon=1,
            detrend_viirs=True,  # Remove spurious VIIRS correlation (Probe 1.2.3)
            use_disaggregated_equipment=getattr(config, 'use_disaggregated_equipment', False),
        )

        # Split ratios for train/val/test
        val_ratio = 0.15
        test_ratio = 0.15

        # Create datasets with proper normalization handling
        train_dataset = MultiResolutionDataset(
            config=data_config,
            split='train',
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        norm_stats = train_dataset.norm_stats  # Get stats from training set

        val_dataset = MultiResolutionDataset(
            config=data_config,
            split='val',
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            norm_stats=norm_stats  # Use training stats to prevent leakage
        )
        test_dataset = MultiResolutionDataset(
            config=data_config,
            split='test',
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            norm_stats=norm_stats  # Use training stats to prevent leakage
        )

        logger.info(f"    Train samples: {len(train_dataset)}")
        logger.info(f"    Val samples: {len(val_dataset)}")
        logger.info(f"    Test samples: {len(test_dataset)}")

        # Create model
        logger.info("  Creating Multi-Resolution HAN model...")
        model = create_multi_resolution_han(
            d_model=config.d_model,
            nhead=4,
            num_daily_layers=3,
            num_monthly_layers=2,
            num_fusion_layers=2,
            num_temporal_layers=2,
            dropout=0.1,
            prediction_tasks=['casualty', 'regime', 'anomaly', 'forecast'],
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params:,}")

        # Create trainer
        logger.info("  Setting up trainer...")
        trainer = MultiResolutionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=config.batch_size,
            accumulation_steps=4,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            num_epochs=config.epochs_han,
            patience=config.patience,
            warmup_epochs=config.warmup_epochs,
            checkpoint_dir=str(MULTI_RES_CHECKPOINT_DIR),
            device=str(device),
        )

        # Train
        logger.info("  Starting training...")
        history = trainer.train()

        # Get final metrics
        final_metrics = trainer.evaluate(test_dataset)

        # Extract training history (trainer returns nested structure)
        train_losses = history.get('train_history', {}).get('total', [])
        val_losses = history.get('val_history', {}).get('total', [])

        results = {
            'history': history,
            'metrics': {
                'final_train_loss': train_losses[-1] if train_losses else 0,
                'final_val_loss': val_losses[-1] if val_losses else 0,
                'best_val_loss': history.get('best_val_loss', 0),
                'best_epoch': history.get('best_epoch', 0),
                'test_metrics': final_metrics,
                'epochs_trained': len(train_losses)
            },
            'model_path': str(MULTI_RES_CHECKPOINT_DIR / 'best_model.pt'),
            'architecture': 'MultiResolutionHAN',
            'data_integrity': {
                'daily_resolution_preserved': True,
                'no_interpolation': True,
                'uses_observation_masks': True,
            }
        }

        logger.info(f"  Final val loss: {results['metrics']['final_val_loss']:.4f}")
        logger.info(f"  Best val loss: {results['metrics']['best_val_loss']:.4f} at epoch {results['metrics']['best_epoch']}")
        logger.info(f"  Epochs trained: {results['metrics']['epochs_trained']}")
        logger.info(f"  Test metrics: {final_metrics}")

        return results

    except Exception as e:
        logger.error(f"Error in Multi-Resolution HAN stage: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}


# =============================================================================
# STAGE 4: TEMPORAL PREDICTION MODEL
# =============================================================================

def train_temporal_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, Any]:
    """
    Stage 4: Train Temporal Prediction Model

    Multi-horizon forecasting (T+1, T+3, T+7) with uncertainty estimation.
    """
    if not HAS_TEMPORAL:
        logger.warning("Skipping temporal stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training Temporal Prediction Model...")

    try:
        # Check for frozen unified model
        unified_model_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'

        if not unified_model_path.exists():
            logger.warning("  Unified delta model not found, attempting to use unified model...")
            unified_model_path = MODEL_DIR / 'unified_interpolation_best.pt'

            if not unified_model_path.exists():
                logger.error("  No unified model found. Please train Stage 2 first.")
                return {'skipped': True, 'reason': 'unified model not found'}

        # Create unified model architecture
        logger.info("  Loading frozen unified model...")
        unified_model = UnifiedInterpolationModelDelta(
            source_configs=DELTA_SOURCE_CONFIGS,
            d_embed=config.d_model,
            nhead=4,
            num_fusion_layers=2
        )

        # Load weights
        state_dict = torch.load(unified_model_path, map_location='cpu', weights_only=False)
        unified_model.load_state_dict(state_dict)
        unified_model.to(device)
        unified_model.eval()

        # Configuration
        temporal_config = TemporalConfig(
            window_size=14,
            horizons=[1, 3, 7],
            epochs=config.epochs_temporal,
            patience=config.patience,
            warmup_epochs=config.warmup_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )

        # Create datasets
        logger.info("  Creating datasets...")
        train_dataset = TemporalPredictionDataset(
            unified_model=unified_model,
            source_configs=DELTA_SOURCE_CONFIGS,
            config=temporal_config,
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
            source_configs=DELTA_SOURCE_CONFIGS,
            config=temporal_config,
            device=device,
            split='val',
            norm_stats=norm_stats
        )
        test_dataset = TemporalPredictionDataset(
            unified_model=unified_model,
            source_configs=DELTA_SOURCE_CONFIGS,
            config=temporal_config,
            device=device,
            split='test',
            norm_stats=norm_stats
        )

        # Get prediction sources
        prediction_sources = train_dataset.prediction_source_names
        logger.info(f"    Prediction sources: {prediction_sources}")

        # Custom collate function
        def collate_fn(batch):
            latents = torch.stack([b[0] for b in batch])
            targets = {}
            for src in prediction_sources:
                targets[src] = {}
                for h in temporal_config.horizons:
                    targets[src][h] = torch.stack([b[1][src][h] for b in batch])
            return latents, targets

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_fn
        )

        # Create model
        logger.info("  Creating temporal prediction model...")
        model = TemporalPredictionModel(
            unified_model=unified_model,
            config=temporal_config,
            source_configs=DELTA_SOURCE_CONFIGS,
            prediction_sources=prediction_sources
        )
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"    Total params: {n_params:,}")
        logger.info(f"    Trainable params: {trainable_params:,}")

        # Create trainer
        trainer = TemporalTrainer(
            model=model,
            config=temporal_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            test_loader=test_loader
        )

        # Train
        history = trainer.train()

        # Evaluate
        logger.info("  Evaluating on validation set...")
        eval_results = evaluate_temporal_model(model, val_loader, temporal_config, device)

        results = {
            'history': history,
            'evaluation': eval_results,
            'metrics': {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'overall_mean_corr': eval_results['summary']['overall_mean_corr'],
                'epochs_trained': len(history['train_loss'])
            },
            'model_path': str(MODEL_DIR / 'temporal_prediction_best.pt')
        }

        logger.info(f"  Overall mean correlation: {results['metrics']['overall_mean_corr']:.3f}")

        return results

    except Exception as e:
        logger.error(f"Error in temporal stage: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}


# =============================================================================
# STAGE 5: TACTICAL STATE PREDICTOR
# =============================================================================

def train_tactical_stage(
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    stage_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Stage 5: Train Tactical State Predictor

    Discrete state classification with state transition modeling P(S(t+1)|S(t)).
    """
    if not HAS_TACTICAL:
        logger.warning("Skipping tactical stage: module not available")
        return {'skipped': True, 'reason': 'module not available'}

    logger.info("Training Tactical State Predictor...")

    try:
        # Configuration
        tactical_config = TacticalPredictionConfig(
            state_config=TacticalStateConfig(
                n_states=config.n_states,
                state_embed_dim=config.d_model
            ),
            hidden_dim=128,
            context_dim=config.d_model,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.epochs_tactical,
            patience=config.patience,
            batch_size=config.batch_size
        )

        # Determine feature dimension
        # Try to use HAN embeddings if available, otherwise use raw features
        if HAS_HAN:
            feature_dim = config.d_model
        else:
            feature_dim = TOTAL_FEATURES if HAS_HAN else 198

        logger.info(f"  Feature dimension: {feature_dim}")
        logger.info(f"  Number of states: {config.n_states}")

        # Create model
        logger.info("  Creating tactical state predictor...")
        model = TacticalStatePredictor(
            feature_dim=feature_dim,
            config=tactical_config
        )

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params:,}")

        # CRITICAL FIX: Use real features from previous stages, not random noise
        # The random data had NO learnable patterns, causing worse-than-random accuracy
        logger.info("  Loading features from previous stages...")

        # Initialize stage_results if None
        if stage_results is None:
            stage_results = {}

        # Try to load features from Stage 4 (temporal prediction) outputs
        features = None
        if 'temporal' in stage_results:
            try:
                stage4_dir = Path(config.checkpoint_dir) / 'stage4_temporal'
                # Load temporal model and extract feature representations
                temporal_checkpoint = stage4_dir / 'best_model.pt'
                if temporal_checkpoint.exists():
                    logger.info(f"    Loading temporal features from {temporal_checkpoint}")
                    checkpoint = torch.load(temporal_checkpoint, map_location=device)
                    # If we have cached features, use those
                    if 'features' in checkpoint:
                        features = checkpoint['features']
                        logger.info(f"    Loaded cached features: {features.shape}")
            except Exception as e:
                logger.warning(f"    Could not load Stage 4 features: {e}")

        # Fallback: Load from unified model (Stage 2) outputs
        if features is None and 'unified' in stage_results:
            try:
                stage2_dir = Path(config.checkpoint_dir) / 'stage2_unified'
                unified_checkpoint = stage2_dir / 'best_model.pt'
                if unified_checkpoint.exists():
                    logger.info(f"    Loading unified features from {unified_checkpoint}")
                    # Load unified model and extract latent representations
                    checkpoint = torch.load(unified_checkpoint, map_location=device)
                    if 'latents' in checkpoint:
                        features = checkpoint['latents']
                        logger.info(f"    Loaded unified latents: {features.shape}")
            except Exception as e:
                logger.warning(f"    Could not load Stage 2 features: {e}")

        # Last resort: Generate structured synthetic data (not random noise)
        if features is None:
            logger.warning("  No real features available, generating structured synthetic data")
            logger.warning("  WARNING: For production use, enable previous stages to generate real features")
            n_timesteps = 500

            # Generate structured data with temporal patterns (NOT random noise)
            # This at least has learnable patterns unlike np.random.randn
            t = np.linspace(0, 10 * np.pi, n_timesteps)
            base_signals = np.column_stack([
                np.sin(t),                    # Oscillating pattern
                np.cos(t * 0.7),              # Different frequency
                np.sin(t * 1.5) * np.exp(-t / 50),  # Damped oscillation
                np.cumsum(np.random.randn(n_timesteps)) / 10,  # Random walk (trend)
            ])

            # Expand to full feature dimension with correlations
            n_features = feature_dim
            features = np.zeros((n_timesteps, n_features), dtype=np.float32)
            for i in range(n_features):
                # Each feature is a combination of base signals with noise
                weights = np.random.randn(4)
                features[:, i] = np.dot(base_signals, weights) + 0.1 * np.random.randn(n_timesteps)

            logger.info(f"    Generated structured synthetic features: {features.shape}")
        else:
            # Ensure numpy array
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()

        logger.info("  Creating datasets...")
        train_dataset = TacticalStateDataset(
            features=features,
            state_labels=None,  # Will be derived from training data
            seq_len=4,
            train=True,
            val_ratio=0.2,
            temporal_gap=config.temporal_gap,
            n_states=config.n_states
        )

        # CRITICAL FIX: Share normalization and percentile stats with validation
        val_dataset = TacticalStateDataset(
            features=features,
            state_labels=None,
            seq_len=4,
            train=False,
            val_ratio=0.2,
            temporal_gap=config.temporal_gap,
            n_states=config.n_states,
            shared_percentiles=train_dataset.percentile_boundaries,
            shared_norm_stats={'mean': train_dataset.feature_mean, 'std': train_dataset.feature_std}
        )

        logger.info(f"    Train samples: {len(train_dataset)}")
        logger.info(f"    Val samples: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Train
        history = train_tactical_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=tactical_config,
            device=str(device)
        )

        results = {
            'history': history,
            'metrics': {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_train_acc': history['train_accuracy'][-1],
                'final_val_acc': history['val_accuracy'][-1],
                'final_trans_acc': history['transition_accuracy'][-1],
                'epochs_trained': len(history['train_loss'])
            },
            'model_path': str(MODEL_DIR / 'tactical_state_predictor_best.pt')
        }

        logger.info(f"  Final val accuracy: {results['metrics']['final_val_acc']:.2%}")
        logger.info(f"  Transition accuracy: {results['metrics']['final_trans_acc']:.2%}")

        return results

    except Exception as e:
        logger.error(f"Error in tactical stage: {e}")
        logger.debug(traceback.format_exc())
        return {'error': str(e)}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """
    Run the full training pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Dictionary with results from all stages
    """
    # Setup
    # If checkpoint_dir is absolute, use as-is; otherwise make relative to ANALYSIS_DIR
    checkpoint_dir_path = Path(config.checkpoint_dir)
    if not checkpoint_dir_path.is_absolute():
        checkpoint_dir = ANALYSIS_DIR / config.checkpoint_dir
    else:
        checkpoint_dir = checkpoint_dir_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(checkpoint_dir / 'logs', config.verbose)
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    device = get_device(config.device)

    logger.info("=" * 70)
    logger.info("TACTICAL STATE PREDICTION - FULL TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"\nDevice: {device}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Save configuration
    config.save(str(checkpoint_dir / 'pipeline_config.json'))

    results = {}
    start_time = time.time()

    # Determine starting stage
    start_stage = 1
    if config.resume_from_stage is not None:
        start_stage = config.resume_from_stage
        logger.info(f"\nResuming from stage {start_stage}")

        # Load previous results
        for i in range(1, start_stage):
            prev = checkpoint_manager.load_stage_checkpoint(i)
            if prev is not None:
                stage_name = ['interpolation', 'unified', 'han', 'temporal', 'tactical'][i-1]
                results[stage_name] = prev.get('results', {})

    # Stage 1: Interpolation
    if start_stage <= 1 and not config.skip_interpolation:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: Training Joint Interpolation Models")
        logger.info("=" * 60)

        stage_start = time.time()
        results['interpolation'] = train_interpolation_stage(config, logger, device)
        results['interpolation']['duration_seconds'] = time.time() - stage_start

        checkpoint_manager.save_stage_checkpoint(1, results)
        logger.info(f"Stage 1 completed in {results['interpolation']['duration_seconds']:.1f}s")

    # Stage 2: Unified
    if start_stage <= 2 and not config.skip_unified:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Training Unified Cross-Source Model")
        logger.info("=" * 60)

        stage_start = time.time()
        results['unified'] = train_unified_stage(config, logger, device)
        results['unified']['duration_seconds'] = time.time() - stage_start

        checkpoint_manager.save_stage_checkpoint(2, results)
        logger.info(f"Stage 2 completed in {results['unified']['duration_seconds']:.1f}s")

    # Stage 3: HAN (standard or multi-resolution)
    if start_stage <= 3 and not config.skip_han:
        logger.info("\n" + "=" * 60)

        if config.use_multi_resolution:
            logger.info("STAGE 3: Training Multi-Resolution HAN (preserves daily resolution)")
            logger.info("=" * 60)
            logger.info("  -> Using NEW architecture: daily data at daily resolution")
            logger.info("  -> No interpolation or forward-fill of missing values")

            stage_start = time.time()
            results['han'] = train_multi_resolution_han_stage(config, logger, device)
        else:
            logger.info("STAGE 3: Training Hierarchical Attention Network (standard)")
            logger.info("=" * 60)
            logger.info("  -> Using standard HAN with monthly-aggregated data")

            stage_start = time.time()
            results['han'] = train_han_stage(config, logger, device)

        results['han']['duration_seconds'] = time.time() - stage_start
        results['han']['multi_resolution'] = config.use_multi_resolution

        checkpoint_manager.save_stage_checkpoint(3, results)
        logger.info(f"Stage 3 completed in {results['han']['duration_seconds']:.1f}s")

    # Stage 4: Temporal
    if start_stage <= 4 and not config.skip_temporal:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: Training Temporal Prediction Model")
        logger.info("=" * 60)

        stage_start = time.time()
        results['temporal'] = train_temporal_stage(config, logger, device)
        results['temporal']['duration_seconds'] = time.time() - stage_start

        checkpoint_manager.save_stage_checkpoint(4, results)
        logger.info(f"Stage 4 completed in {results['temporal']['duration_seconds']:.1f}s")

    # Stage 5: Tactical State
    if start_stage <= 5 and not config.skip_tactical:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: Training Tactical State Predictor")
        logger.info("=" * 60)

        stage_start = time.time()
        results['tactical'] = train_tactical_stage(config, logger, device, stage_results=results)
        results['tactical']['duration_seconds'] = time.time() - stage_start

        checkpoint_manager.save_stage_checkpoint(5, results)
        logger.info(f"Stage 5 completed in {results['tactical']['duration_seconds']:.1f}s")

    # Final evaluation and summary
    total_time = time.time() - start_time
    results['pipeline'] = {
        'total_duration_seconds': total_time,
        'completed_at': datetime.now().isoformat(),
        'device': str(device)
    }

    # Print final summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE - FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nTotal training time: {total_time/60:.1f} minutes")

    for stage_name, stage_results in results.items():
        if stage_name == 'pipeline':
            continue

        if isinstance(stage_results, dict):
            if stage_results.get('skipped'):
                logger.info(f"\n{stage_name.upper()}: SKIPPED ({stage_results.get('reason', 'unknown')})")
            elif 'error' in stage_results:
                logger.info(f"\n{stage_name.upper()}: ERROR - {stage_results['error']}")
            elif 'metrics' in stage_results:
                logger.info(f"\n{stage_name.upper()}:")
                for metric, value in stage_results['metrics'].items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: {value}")

    # Save final results
    checkpoint_manager.save_pipeline_results(results)
    logger.info(f"\nResults saved to {checkpoint_dir / 'pipeline_results.json'}")

    return results


def evaluate_full_pipeline(
    results: Dict[str, Any],
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Evaluate the complete trained pipeline.

    Args:
        results: Results from training all stages
        config: Pipeline configuration

    Returns:
        Evaluation metrics for the full system
    """
    evaluation = {
        'stages_completed': [],
        'stages_failed': [],
        'overall_metrics': {}
    }

    for stage_name, stage_results in results.items():
        if stage_name == 'pipeline':
            continue

        if isinstance(stage_results, dict):
            if stage_results.get('skipped') or 'error' in stage_results:
                evaluation['stages_failed'].append(stage_name)
            else:
                evaluation['stages_completed'].append(stage_name)

    # Aggregate metrics
    if 'tactical' in results and 'metrics' in results.get('tactical', {}):
        evaluation['overall_metrics']['state_accuracy'] = results['tactical']['metrics'].get('final_val_acc', 0)
        evaluation['overall_metrics']['transition_accuracy'] = results['tactical']['metrics'].get('final_trans_acc', 0)

    if 'temporal' in results and 'metrics' in results.get('temporal', {}):
        evaluation['overall_metrics']['prediction_correlation'] = results['temporal']['metrics'].get('overall_mean_corr', 0)

    return evaluation


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train the full tactical state prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python train_full_pipeline.py

    # Quick test run (10 epochs each stage)
    python train_full_pipeline.py --quick

    # Run with custom epochs
    python train_full_pipeline.py --epochs 50

    # Run only a specific stage
    python train_full_pipeline.py --stage 3

    # Resume from a specific stage
    python train_full_pipeline.py --resume 3

    # Skip certain stages
    python train_full_pipeline.py --skip-interpolation --skip-unified
        """
    )

    # General options
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5],
                        help='Run only this stage')
    parser.add_argument('--resume', type=int, choices=[1, 2, 3, 4, 5],
                        help='Resume from this stage')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (10 epochs each stage)')

    # Epoch overrides
    parser.add_argument('--epochs', type=int,
                        help='Override epochs for all stages')
    parser.add_argument('--epochs-interpolation', type=int, default=100)
    parser.add_argument('--epochs-unified', type=int, default=100)
    parser.add_argument('--epochs-han', type=int, default=200)
    parser.add_argument('--epochs-temporal', type=int, default=150)
    parser.add_argument('--epochs-tactical', type=int, default=100)

    # Skip stages
    parser.add_argument('--skip-interpolation', action='store_true')
    parser.add_argument('--skip-unified', action='store_true')
    parser.add_argument('--skip-han', action='store_true')
    parser.add_argument('--skip-temporal', action='store_true')
    parser.add_argument('--skip-tactical', action='store_true')

    # Multi-resolution mode (Stage 3)
    parser.add_argument('--multi-resolution', action='store_true', default=True,
                        help='Use Multi-Resolution HAN that preserves daily resolution (default: True)')
    parser.add_argument('--no-multi-resolution', action='store_true',
                        help='Use standard HAN with monthly-aggregated data')

    # Model settings
    parser.add_argument('--d-model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--n-states', type=int, default=8,
                        help='Number of tactical states')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--temporal-gap', type=int, default=14,
                        help='Temporal gap between train/val splits')

    # Early stopping flexibility
    parser.add_argument('--min-epochs', type=int, default=50,
                        help='Minimum epochs before early stopping kicks in')
    parser.add_argument('--early-stop-strategy', type=str, default='smoothed',
                        choices=['standard', 'smoothed', 'relative', 'plateau', 'combined'],
                        help='Early stopping strategy')
    parser.add_argument('--smoothing-factor', type=float, default=0.9,
                        help='EMA smoothing factor for smoothed early stopping')
    parser.add_argument('--relative-threshold', type=float, default=0.1,
                        help='Relative threshold for relative early stopping (0.1 = 10%%)')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum improvement delta for early stopping')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping entirely')

    # Stochastic Weight Averaging (SWA)
    parser.add_argument('--use-swa', action='store_true', default=True,
                        help='Enable Stochastic Weight Averaging (default: enabled)')
    parser.add_argument('--no-swa', action='store_true',
                        help='Disable Stochastic Weight Averaging')
    parser.add_argument('--swa-start-pct', type=float, default=0.75,
                        help='Start SWA at this percentage of training (default: 0.75)')
    parser.add_argument('--swa-freq', type=int, default=5,
                        help='Update SWA model every N epochs (default: 5)')

    # Snapshot Ensembles
    parser.add_argument('--use-snapshots', action='store_true',
                        help='Enable snapshot ensemble collection')
    parser.add_argument('--n-snapshots', type=int, default=5,
                        help='Number of snapshots to keep (default: 5)')

    # Regularization
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Enable label smoothing')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--use-mixup', action='store_true',
                        help='Enable mixup data augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup interpolation strength (default: 0.2)')

    # Learning rate scheduling
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'cosine_restarts', 'linear', 'constant'],
                        help='Learning rate schedule type')
    parser.add_argument('--cosine-t0', type=int, default=10,
                        help='Initial cycle length for cosine restarts')
    parser.add_argument('--cosine-t-mult', type=int, default=2,
                        help='Cycle length multiplier for cosine restarts')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage (recommended for transformer operations on Apple Silicon)')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default=str(PIPELINE_CHECKPOINT_DIR))
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()

    # Apply command line overrides
    if args.quick:
        config.epochs_interpolation = 10
        config.epochs_unified = 10
        config.epochs_han = 10
        config.epochs_temporal = 10
        config.epochs_tactical = 10
        config.patience = 5

    if args.epochs:
        config.epochs_interpolation = args.epochs
        config.epochs_unified = args.epochs
        config.epochs_han = args.epochs
        config.epochs_temporal = args.epochs
        config.epochs_tactical = args.epochs

    if args.epochs_interpolation:
        config.epochs_interpolation = args.epochs_interpolation
    if args.epochs_unified:
        config.epochs_unified = args.epochs_unified
    if args.epochs_han:
        config.epochs_han = args.epochs_han
    if args.epochs_temporal:
        config.epochs_temporal = args.epochs_temporal
    if args.epochs_tactical:
        config.epochs_tactical = args.epochs_tactical

    # Stage control
    if args.stage:
        # Run only specified stage
        config.skip_interpolation = args.stage != 1
        config.skip_unified = args.stage != 2
        config.skip_han = args.stage != 3
        config.skip_temporal = args.stage != 4
        config.skip_tactical = args.stage != 5
    else:
        config.skip_interpolation = args.skip_interpolation
        config.skip_unified = args.skip_unified
        config.skip_han = args.skip_han
        config.skip_temporal = args.skip_temporal
        config.skip_tactical = args.skip_tactical

    if args.resume:
        config.resume_from_stage = args.resume

    # Other settings
    config.d_model = args.d_model
    config.n_states = args.n_states
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.patience = args.patience
    config.temporal_gap = args.temporal_gap

    # Early stopping flexibility
    config.early_stopping_min_epochs = args.min_epochs
    config.early_stopping_strategy = args.early_stop_strategy
    config.early_stopping_smoothing = args.smoothing_factor
    config.early_stopping_relative_threshold = args.relative_threshold
    config.early_stopping_min_delta = args.min_delta
    config.disable_early_stopping = args.no_early_stop

    # Stochastic Weight Averaging (SWA)
    config.use_swa = args.use_swa and not args.no_swa
    config.swa_start_pct = args.swa_start_pct
    config.swa_freq = args.swa_freq

    # Snapshot Ensembles
    config.use_snapshots = args.use_snapshots
    config.n_snapshots = args.n_snapshots

    # Regularization
    config.use_label_smoothing = args.use_label_smoothing
    config.label_smoothing = args.label_smoothing
    config.use_mixup = args.use_mixup
    config.mixup_alpha = args.mixup_alpha

    # Learning rate scheduling
    config.lr_schedule = args.lr_schedule
    config.cosine_t0 = args.cosine_t0
    config.cosine_t_mult = args.cosine_t_mult

    # Multi-resolution architecture
    config.use_multi_resolution = args.multi_resolution and not args.no_multi_resolution

    # Force CPU if requested (useful for transformer ops on Apple Silicon)
    if args.force_cpu:
        config.device = 'cpu'
    else:
        config.device = args.device
    config.checkpoint_dir = args.checkpoint_dir
    config.verbose = not args.quiet

    # Run pipeline
    print("\nPipeline Configuration:")
    print("-" * 40)
    print(f"  Epochs (interp/unified/han/temp/tact): "
          f"{config.epochs_interpolation}/{config.epochs_unified}/"
          f"{config.epochs_han}/{config.epochs_temporal}/{config.epochs_tactical}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Number of states: {config.n_states}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LR schedule: {config.lr_schedule}")
    print(f"  Patience: {config.patience}")
    print(f"  Temporal gap: {config.temporal_gap} days")
    print(f"  Device: {config.device}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
    print(f"  Multi-resolution HAN: {'ENABLED' if config.use_multi_resolution else 'disabled (standard HAN)'}")

    # Early stopping configuration
    print("\nEarly Stopping Configuration:")
    print("-" * 40)
    if config.disable_early_stopping:
        print("  Early stopping: DISABLED")
    else:
        print(f"  Strategy: {config.early_stopping_strategy}")
        print(f"  Min epochs: {config.early_stopping_min_epochs}")
        print(f"  Patience: {config.patience}")
        print(f"  Min delta: {config.early_stopping_min_delta}")
        if config.early_stopping_strategy in ['smoothed', 'combined']:
            print(f"  Smoothing factor: {config.early_stopping_smoothing}")
        if config.early_stopping_strategy in ['relative', 'combined']:
            print(f"  Relative threshold: {config.early_stopping_relative_threshold:.0%}")

    # Regularization configuration
    print("\nRegularization Configuration:")
    print("-" * 40)
    print(f"  SWA: {'ENABLED' if config.use_swa else 'disabled'}")
    if config.use_swa:
        print(f"    Start at: {config.swa_start_pct:.0%} of training")
        print(f"    Update frequency: every {config.swa_freq} epochs")
    print(f"  Snapshots: {'ENABLED' if config.use_snapshots else 'disabled'}")
    if config.use_snapshots:
        print(f"    Max snapshots: {config.n_snapshots}")
    print(f"  Label smoothing: {'ENABLED' if config.use_label_smoothing else 'disabled'}")
    if config.use_label_smoothing:
        print(f"    Smoothing factor: {config.label_smoothing}")
    print(f"  Mixup: {'ENABLED' if config.use_mixup else 'disabled'}")
    if config.use_mixup:
        print(f"    Alpha: {config.mixup_alpha}")

    # Print stage status
    print("\nStage Status:")
    stages = [
        ('1. Interpolation', config.skip_interpolation),
        ('2. Unified', config.skip_unified),
        ('3. HAN', config.skip_han),
        ('4. Temporal', config.skip_temporal),
        ('5. Tactical', config.skip_tactical)
    ]
    for name, skipped in stages:
        status = 'SKIP' if skipped else 'RUN'
        print(f"  {name}: {status}")

    print()

    # Run
    results = run_pipeline(config)

    # Final evaluation
    evaluation = evaluate_full_pipeline(results, config)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Stages completed: {', '.join(evaluation['stages_completed']) or 'None'}")
    print(f"Stages failed: {', '.join(evaluation['stages_failed']) or 'None'}")

    if evaluation['overall_metrics']:
        print("\nOverall Metrics:")
        for metric, value in evaluation['overall_metrics'].items():
            print(f"  {metric}: {value:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
