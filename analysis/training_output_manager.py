"""
Training Output Manager for Pipeline Runs
==========================================

Manages training run directories with organized checkpoint storage,
unified configuration, and metadata for probe integration.

Directory Structure:
-------------------
analysis/training_runs/
└── run_DD-MM-YYYY_HH-MM/
    ├── config.json              # Full pipeline config
    ├── metadata.json            # Training metadata & results
    ├── stage1_interpolation/    # JIM checkpoints
    │   ├── interp_equipment_best.pt
    │   ├── interp_deepstate_best.pt
    │   └── ...
    ├── stage2_unified/          # Unified model checkpoints
    │   ├── unified_best.pt
    │   └── unified_delta_best.pt
    ├── stage3_han/              # HAN checkpoints
    │   ├── best_checkpoint.pt
    │   ├── checkpoint_epoch_10.pt
    │   └── training_summary.json
    ├── stage4_temporal/         # Temporal prediction checkpoints
    │   └── temporal_best.pt
    └── stage5_tactical/         # Tactical state predictor
        └── tactical_best.pt

Usage:
------
    from training_output_manager import TrainingRunManager

    manager = TrainingRunManager()
    manager.setup()

    # Save config at start
    manager.save_config(pipeline_config)

    # Get paths for each stage
    jim_dir = manager.get_stage_dir(1)
    unified_dir = manager.get_stage_dir(2)
    han_dir = manager.get_stage_dir(3)

    # Update metadata after training
    manager.update_metadata(stage1_complete=True, stage1_duration=123.4)

    # Finalize
    manager.finalize()

Author: ML Engineering Team
Date: 2026-01-24
"""

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.paths import ANALYSIS_DIR


# =============================================================================
# CONSTANTS
# =============================================================================

TRAINING_RUNS_DIR = ANALYSIS_DIR / "training_runs"

STAGE_NAMES = {
    1: "stage1_interpolation",
    2: "stage2_unified",
    3: "stage3_han",
    4: "stage4_temporal",
    5: "stage5_tactical",
}

STAGE_DESCRIPTIONS = {
    1: "Joint Interpolation Models (JIM) - Temporal gap filling",
    2: "Unified Cross-Source Model - Cross-source relationships",
    3: "Hierarchical Attention Network (HAN) - State encoding",
    4: "Temporal Prediction Model - Multi-horizon forecasting",
    5: "Tactical State Predictor - State classification",
}


# =============================================================================
# METADATA
# =============================================================================

@dataclass
class TrainingRunMetadata:
    """Metadata for a training run."""

    # Run identification
    run_id: str = ""
    run_timestamp: str = ""
    run_dir: str = ""

    # Pipeline configuration summary
    # NOTE: These should be populated from actual PipelineConfig at runtime,
    # not relied upon as defaults. Use None to indicate "not set".
    d_model: int = 64
    use_multi_resolution: bool = True
    detrend_viirs: Optional[bool] = None  # None = not explicitly set
    use_disaggregated_equipment: Optional[bool] = None  # None = not explicitly set

    # Data configuration (critical for probe compatibility)
    # These are populated from actual MultiResolutionConfig at runtime
    effective_daily_sources: List[str] = field(default_factory=list)
    effective_monthly_sources: List[str] = field(default_factory=list)
    n_daily_sources: int = 0
    n_monthly_sources: int = 0
    daily_seq_len: int = 365
    monthly_seq_len: int = 12
    date_range_start: str = ""
    date_range_end: str = ""

    # Dataset statistics
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0
    feature_dims_per_source: Dict[str, int] = field(default_factory=dict)

    # Stage completion status
    stage1_complete: bool = False
    stage2_complete: bool = False
    stage3_complete: bool = False
    stage4_complete: bool = False
    stage5_complete: bool = False

    # Stage durations (seconds)
    stage1_duration: float = 0.0
    stage2_duration: float = 0.0
    stage3_duration: float = 0.0
    stage4_duration: float = 0.0
    stage5_duration: float = 0.0

    # Stage metrics (best validation loss/accuracy)
    stage1_metrics: Dict[str, Any] = field(default_factory=dict)
    stage2_metrics: Dict[str, Any] = field(default_factory=dict)
    stage3_metrics: Dict[str, Any] = field(default_factory=dict)
    stage4_metrics: Dict[str, Any] = field(default_factory=dict)
    stage5_metrics: Dict[str, Any] = field(default_factory=dict)

    # Model counts
    n_jim_models: int = 0
    n_unified_models: int = 0

    # HAN-specific
    han_best_epoch: int = 0
    han_best_val_loss: float = 0.0
    han_n_params: int = 0

    # Training settings (from PipelineConfig)
    batch_size: int = 32
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    early_stopping_strategy: str = ""
    use_swa: bool = False

    # Overall
    total_duration_seconds: float = 0.0
    device: str = "cpu"
    torch_version: str = ""
    python_version: str = ""

    # Reproducibility
    git_commit_hash: str = ""
    random_seed: Optional[int] = None

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingRunMetadata':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# TRAINING RUN MANAGER
# =============================================================================

class TrainingRunManager:
    """
    Manages training run output directory structure.

    Creates organized directories for each pipeline stage with
    consistent naming and unified configuration storage.
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize the training run manager.

        Args:
            run_id: Optional custom run ID. If None, generates from timestamp.
        """
        if run_id is None:
            self.run_id = datetime.now().strftime("run_%d-%m-%Y_%H-%M")
        else:
            self.run_id = run_id

        self.run_dir = TRAINING_RUNS_DIR / self.run_id
        self.metadata = TrainingRunMetadata(
            run_id=self.run_id,
            run_timestamp=datetime.now().isoformat(),
            run_dir=str(self.run_dir),
        )
        self._setup_complete = False

    def setup(self) -> None:
        """Create the run directory structure."""
        if self._setup_complete:
            return

        # Create main run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create stage directories
        for stage_num, stage_name in STAGE_NAMES.items():
            stage_dir = self.run_dir / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)

        self._setup_complete = True

        # Get system info
        try:
            import torch
            self.metadata.torch_version = torch.__version__
        except ImportError:
            pass

        import sys
        self.metadata.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def get_stage_dir(self, stage: int) -> Path:
        """
        Get the directory path for a specific stage.

        Args:
            stage: Stage number (1-5)

        Returns:
            Path to the stage directory
        """
        if stage not in STAGE_NAMES:
            raise ValueError(f"Invalid stage: {stage}. Must be 1-5.")
        return self.run_dir / STAGE_NAMES[stage]

    def get_checkpoint_path(self, stage: int, name: str) -> Path:
        """
        Get a checkpoint path for a specific stage.

        Args:
            stage: Stage number (1-5)
            name: Checkpoint name (e.g., 'best', 'epoch_10')

        Returns:
            Full path to the checkpoint file
        """
        stage_dir = self.get_stage_dir(stage)

        # Add .pt extension if not present
        if not name.endswith('.pt'):
            name = f"{name}.pt"

        return stage_dir / name

    def save_config(self, config: Any) -> Path:
        """
        Save the full pipeline configuration.

        Args:
            config: PipelineConfig or dict-like object with to_dict() method

        Returns:
            Path to saved config file
        """
        config_path = self.run_dir / "config.json"

        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dict__'):
            config_dict = vars(config)
        else:
            config_dict = dict(config)

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        return config_path

    def load_config(self) -> Dict[str, Any]:
        """Load the pipeline configuration."""
        config_path = self.run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config found at {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def update_metadata(self, **kwargs) -> None:
        """
        Update metadata fields.

        Args:
            **kwargs: Field names and values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                # Store unknown fields in stage metrics if they look like metrics
                if key.startswith('stage') and '_' in key:
                    pass  # Ignore unknown stage fields

        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def mark_stage_complete(
        self,
        stage: int,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark a stage as complete with its metrics.

        Args:
            stage: Stage number (1-5)
            duration: Duration in seconds
            metrics: Optional metrics dict
        """
        setattr(self.metadata, f"stage{stage}_complete", True)
        setattr(self.metadata, f"stage{stage}_duration", duration)

        if metrics:
            setattr(self.metadata, f"stage{stage}_metrics", metrics)

        self._save_metadata()

    def record_error(self, stage: int, error: str) -> None:
        """Record an error for a stage."""
        self.metadata.errors.append(f"Stage {stage}: {error}")
        self._save_metadata()

    def finalize(self, total_duration: Optional[float] = None) -> Path:
        """
        Finalize the training run.

        Args:
            total_duration: Optional total duration to record

        Returns:
            Path to the metadata file
        """
        if total_duration is not None:
            self.metadata.total_duration_seconds = total_duration

        self._save_metadata()
        return self.run_dir / "metadata.json"

    @property
    def config_path(self) -> Path:
        """Path to the config file."""
        return self.run_dir / "config.json"

    @property
    def metadata_path(self) -> Path:
        """Path to the metadata file."""
        return self.run_dir / "metadata.json"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_training_runs() -> List[Dict[str, Any]]:
    """
    List all training runs with their metadata.

    Returns:
        List of metadata dictionaries, sorted by timestamp (newest first)
    """
    runs = []

    if not TRAINING_RUNS_DIR.exists():
        return runs

    for run_dir in TRAINING_RUNS_DIR.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
            continue

        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                runs.append(metadata)
            except json.JSONDecodeError:
                runs.append({'run_id': run_dir.name, 'error': 'Invalid metadata'})
        else:
            runs.append({'run_id': run_dir.name, 'error': 'No metadata'})

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.get('run_timestamp', ''), reverse=True)
    return runs


def get_training_run(run_id: str) -> Optional[TrainingRunManager]:
    """
    Get a training run manager for an existing run.

    Args:
        run_id: The run ID (e.g., 'run_24-01-2026_14-30')

    Returns:
        TrainingRunManager or None if not found
    """
    run_dir = TRAINING_RUNS_DIR / run_id
    if not run_dir.exists():
        return None

    manager = TrainingRunManager(run_id=run_id)

    # Load existing metadata
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            manager.metadata = TrainingRunMetadata.from_dict(data)

    manager._setup_complete = True
    return manager


def get_latest_training_run() -> Optional[TrainingRunManager]:
    """
    Get the most recent training run.

    Returns:
        TrainingRunManager for the latest run, or None if no runs exist
    """
    runs = list_training_runs()
    if not runs:
        return None

    return get_training_run(runs[0]['run_id'])


def copy_checkpoints_to_run(
    manager: TrainingRunManager,
    source_dirs: Optional[Dict[int, Path]] = None
) -> None:
    """
    Copy existing checkpoints into a training run directory.

    Useful for organizing legacy checkpoints into the new structure.

    Args:
        manager: TrainingRunManager to copy into
        source_dirs: Dict mapping stage numbers to source directories.
                    If None, uses default locations.
    """
    from config.paths import (
        INTERP_MODEL_DIR, MODEL_DIR, MULTI_RES_CHECKPOINT_DIR
    )

    if source_dirs is None:
        source_dirs = {
            1: INTERP_MODEL_DIR,
            2: MODEL_DIR,
            3: MULTI_RES_CHECKPOINT_DIR,
        }

    for stage, source_dir in source_dirs.items():
        if not source_dir.exists():
            continue

        dest_dir = manager.get_stage_dir(stage)

        if stage == 1:
            # Copy interpolation models
            for pt_file in source_dir.glob('interp_*_best.pt'):
                shutil.copy2(pt_file, dest_dir / pt_file.name)

        elif stage == 2:
            # Copy unified models
            for pattern in ['unified_interpolation_best.pt', 'unified_interpolation_delta_best.pt']:
                pt_file = source_dir / pattern
                if pt_file.exists():
                    shutil.copy2(pt_file, dest_dir / pt_file.name)

        elif stage == 3:
            # Copy HAN checkpoints
            for pt_file in source_dir.glob('*.pt'):
                shutil.copy2(pt_file, dest_dir / pt_file.name)
            # Also copy training_summary.json if present
            summary = source_dir / 'training_summary.json'
            if summary.exists():
                shutil.copy2(summary, dest_dir / summary.name)
