"""
Centralized output management for probe runs.

Provides consistent directory structure for comparing model variations:
    analysis/probes/runs/
        run_<dd>-<mm>-<yyyy>_<hh>-<mm>/
            metadata.json          # Run configuration and phase info
            raw_metrics/           # CSVs, JSONs, non-human-readable
            figures/               # All visualizations
            probe_battery_report.md
"""

import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"


@dataclass
class RunMetadata:
    """Metadata about a probe run for comparison across model variations."""

    # Run identification
    run_id: str = ""
    run_timestamp: str = ""

    # Model configuration
    model_checkpoint: str = ""
    d_model: int = 0
    nhead: int = 0
    num_daily_layers: int = 0
    num_monthly_layers: int = 0
    num_fusion_layers: int = 0
    num_params: int = 0

    # Data configuration
    daily_seq_len: int = 0
    monthly_seq_len: int = 0
    daily_sources: List[str] = field(default_factory=list)
    monthly_sources: List[str] = field(default_factory=list)
    detrend_viirs: bool = False
    use_disaggregated_equipment: bool = True

    # Training configuration (from checkpoint)
    training_epochs: int = 0
    best_epoch: int = 0
    best_val_loss: float = 0.0

    # Task configuration
    task_names: List[str] = field(default_factory=list)
    task_priors: Dict[str, float] = field(default_factory=dict)

    # Phase information (for tracking architectural modifications)
    phase_name: str = ""
    phase_description: str = ""
    optimizations_applied: List[str] = field(default_factory=list)

    # Training run linkage (which training run these probes were run against)
    training_run_id: str = ""

    # Environment
    device: str = ""
    torch_version: str = ""
    python_version: str = ""

    # Probe run info
    probes_completed: int = 0
    probes_failed: int = 0
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunMetadata':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RunOutputManager:
    """
    Manages output directories and files for a single probe run.

    Directory structure:
        runs/run_<dd>-<mm>-<yyyy>_<hh>-<mm>/
            metadata.json
            raw_metrics/
            figures/
            probe_battery_report.md
    """

    def __init__(self, run_id: Optional[str] = None):
        """
        Initialize output manager.

        Args:
            run_id: Optional custom run ID. If None, generates timestamp-based ID.
        """
        if run_id is None:
            now = datetime.now()
            self.run_id = f"run_{now:%d-%m-%Y_%H-%M}"
        else:
            self.run_id = run_id

        self.run_dir = RUNS_DIR / self.run_id
        self.raw_metrics_dir = self.run_dir / "raw_metrics"
        self.figures_dir = self.run_dir / "figures"
        self.metadata_path = self.run_dir / "metadata.json"
        self.report_path = self.run_dir / "probe_battery_report.md"

        self._metadata = RunMetadata(
            run_id=self.run_id,
            run_timestamp=datetime.now().isoformat(),
            torch_version=torch.__version__,
            python_version=platform.python_version(),
        )

    def setup(self) -> None:
        """Create directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metrics_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

    @property
    def metadata(self) -> RunMetadata:
        """Get run metadata."""
        return self._metadata

    def update_metadata(self, **kwargs) -> None:
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)

    def save_metadata(self) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self._metadata.to_dict(), f, indent=2, default=str)

    def load_metadata(self) -> RunMetadata:
        """Load metadata from existing run."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                data = json.load(f)
            self._metadata = RunMetadata.from_dict(data)
        return self._metadata

    def get_figure_path(self, name: str, extension: str = "png") -> Path:
        """
        Get path for a figure file.

        Args:
            name: Figure name (without extension)
            extension: File extension (default: png)

        Returns:
            Full path to figure file
        """
        # Clean up name - remove any existing extension
        name = name.rsplit('.', 1)[0] if '.' in name else name
        return self.figures_dir / f"{name}.{extension}"

    def get_metrics_path(self, name: str, extension: str = "json") -> Path:
        """
        Get path for a metrics file.

        Args:
            name: Metrics file name (without extension)
            extension: File extension (default: json)

        Returns:
            Full path to metrics file
        """
        name = name.rsplit('.', 1)[0] if '.' in name else name
        return self.raw_metrics_dir / f"{name}.{extension}"

    def save_figure(self, fig, name: str, dpi: int = 150, **kwargs) -> Path:
        """
        Save a matplotlib figure.

        Args:
            fig: Matplotlib figure
            name: Figure name
            dpi: Resolution
            **kwargs: Additional savefig arguments

        Returns:
            Path to saved figure
        """
        path = self.get_figure_path(name)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', **kwargs)
        return path

    def save_json(self, data: Any, name: str) -> Path:
        """
        Save data as JSON.

        Args:
            data: Data to save
            name: File name

        Returns:
            Path to saved file
        """
        path = self.get_metrics_path(name, "json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_csv(self, df, name: str, **kwargs) -> Path:
        """
        Save DataFrame as CSV.

        Args:
            df: Pandas DataFrame
            name: File name
            **kwargs: Additional to_csv arguments

        Returns:
            Path to saved file
        """
        path = self.get_metrics_path(name, "csv")
        df.to_csv(path, **kwargs)
        return path

    def extract_model_metadata(self, checkpoint_path: Path, model=None, config=None) -> None:
        """
        Extract metadata from model checkpoint and config.

        Args:
            checkpoint_path: Path to model checkpoint
            model: Optional loaded model
            config: Optional configuration dict
        """
        self.update_metadata(model_checkpoint=str(checkpoint_path))

        # Load checkpoint for metadata
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

                # Extract training info
                if 'epoch' in checkpoint:
                    self.update_metadata(training_epochs=checkpoint.get('epoch', 0))
                if 'best_val_loss' in checkpoint:
                    self.update_metadata(best_val_loss=checkpoint.get('best_val_loss', 0.0))

            except Exception:
                pass

        # Extract from config if provided
        if config:
            self.update_metadata(
                d_model=config.get('d_model', 0),
                nhead=config.get('nhead', 0),
                num_daily_layers=config.get('num_daily_layers', 0),
                num_monthly_layers=config.get('num_monthly_layers', 0),
                num_fusion_layers=config.get('num_fusion_layers', 0),
                daily_seq_len=config.get('daily_seq_len', 0),
                monthly_seq_len=config.get('monthly_seq_len', 0),
            )

        # Extract from model if provided
        if model is not None:
            self.update_metadata(
                num_params=sum(p.numel() for p in model.parameters()),
            )

    def extract_data_metadata(self, data_config) -> None:
        """
        Extract metadata from data configuration.

        Args:
            data_config: MultiResolutionConfig or similar
        """
        if hasattr(data_config, 'daily_sources'):
            self.update_metadata(daily_sources=list(data_config.daily_sources))
        if hasattr(data_config, 'monthly_sources'):
            self.update_metadata(monthly_sources=list(data_config.monthly_sources))
        if hasattr(data_config, 'detrend_viirs'):
            self.update_metadata(detrend_viirs=data_config.detrend_viirs)
        if hasattr(data_config, 'use_disaggregated_equipment'):
            self.update_metadata(use_disaggregated_equipment=data_config.use_disaggregated_equipment)
        if hasattr(data_config, 'daily_seq_len'):
            self.update_metadata(daily_seq_len=data_config.daily_seq_len)
        if hasattr(data_config, 'monthly_seq_len'):
            self.update_metadata(monthly_seq_len=data_config.monthly_seq_len)

    def set_phase_info(
        self,
        phase_name: str,
        phase_description: str = "",
        optimizations: Optional[List[str]] = None
    ) -> None:
        """
        Set phase information for tracking architectural modifications.

        Args:
            phase_name: Short name for the phase (e.g., "Phase0_Optimizations")
            phase_description: Description of what this phase tests
            optimizations: List of optimizations applied in this phase
        """
        self.update_metadata(
            phase_name=phase_name,
            phase_description=phase_description,
            optimizations_applied=optimizations or [],
        )

    def finalize(
        self,
        probes_completed: int = 0,
        probes_failed: int = 0,
        total_duration: float = 0.0
    ) -> None:
        """
        Finalize run and save metadata.

        Args:
            probes_completed: Number of completed probes
            probes_failed: Number of failed probes
            total_duration: Total run duration in seconds
        """
        self.update_metadata(
            probes_completed=probes_completed,
            probes_failed=probes_failed,
            total_duration_seconds=total_duration,
        )
        self.save_metadata()


def list_runs() -> List[Dict[str, Any]]:
    """
    List all available probe runs with summary info.

    Returns:
        List of run summaries
    """
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            metadata_path = run_dir / "metadata.json"
            summary = {
                "run_id": run_dir.name,
                "path": str(run_dir),
            }

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    summary.update({
                        "phase_name": metadata.get("phase_name", ""),
                        "probes_completed": metadata.get("probes_completed", 0),
                        "probes_failed": metadata.get("probes_failed", 0),
                        "d_model": metadata.get("d_model", 0),
                        "detrend_viirs": metadata.get("detrend_viirs", False),
                    })
                except Exception:
                    pass

            runs.append(summary)

    return runs


def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare metadata across multiple runs.

    Args:
        run_ids: List of run IDs to compare

    Returns:
        Comparison dictionary
    """
    comparison = {
        "runs": [],
        "differences": {},
    }

    for run_id in run_ids:
        manager = RunOutputManager(run_id)
        if manager.metadata_path.exists():
            metadata = manager.load_metadata()
            comparison["runs"].append(metadata.to_dict())

    # Find differences
    if len(comparison["runs"]) >= 2:
        keys_to_compare = [
            "d_model", "nhead", "daily_seq_len", "detrend_viirs",
            "use_disaggregated_equipment", "phase_name", "optimizations_applied",
            "best_val_loss", "probes_completed", "probes_failed"
        ]

        for key in keys_to_compare:
            values = [run.get(key) for run in comparison["runs"]]
            if len(set(str(v) for v in values)) > 1:
                comparison["differences"][key] = values

    return comparison
