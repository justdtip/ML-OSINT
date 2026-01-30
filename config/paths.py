"""
Centralized Path Configuration for ML_OSINT

All paths in the project should be derived from these constants.
NEVER use hardcoded absolute paths like '/Users/daniel.tipton/ML_OSINT'.

Usage:
    from config.paths import PROJECT_ROOT, DATA_DIR, MODEL_DIR

    data_path = DATA_DIR / "ucdp" / "ged_events.csv"
    model_path = MODEL_DIR / "han_best.pt"
"""

from pathlib import Path
import os

# Project root - determined relative to this config file
# This file is at: PROJECT_ROOT/config/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Override with environment variable if set (for deployment flexibility)
if os.environ.get("ML_OSINT_ROOT"):
    PROJECT_ROOT = Path(os.environ["ML_OSINT_ROOT"])


# ============================================================================
# PRIMARY DIRECTORIES
# ============================================================================

# Data directory - all raw and processed data
DATA_DIR = PROJECT_ROOT / "data"

# Analysis directory - core ML pipeline
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Scripts directory - download and utility scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"


# ============================================================================
# MODEL AND CHECKPOINT DIRECTORIES
# ============================================================================

# Trained model weights
MODEL_DIR = ANALYSIS_DIR / "models"

# Interpolation models (domain-specific temporal gap filling)
INTERP_MODEL_DIR = MODEL_DIR / "interpolation"

# Training checkpoints (intermediate states)
CHECKPOINT_DIR = ANALYSIS_DIR / "checkpoints"

# Multi-resolution specific checkpoints
MULTI_RES_CHECKPOINT_DIR = CHECKPOINT_DIR / "multi_resolution"

# Pipeline stage checkpoints
PIPELINE_CHECKPOINT_DIR = MODEL_DIR / "pipeline"


# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

# Unified output directory (new structure)
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Figures and visualizations
FIGURES_DIR = OUTPUT_DIR / "figures"

# Analysis reports
REPORTS_DIR = OUTPUT_DIR / "reports"

# Results (CSVs, JSONs)
RESULTS_DIR = OUTPUT_DIR / "results"

# Log files
LOG_DIR = PROJECT_ROOT / "logs"

# Analysis outputs (consolidated from various analysis scripts)
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"


# ============================================================================
# ANALYSIS OUTPUT SUBDIRECTORIES
# ============================================================================

CROSS_SOURCE_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "cross_source"
DELTA_MODEL_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "delta_model"
EMBEDDING_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "embeddings"
FEATURE_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "features"
INTERPRETABILITY_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "interpretability"
MODEL_ASSESSMENT_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "model_assessment"
MODEL_COMPARISON_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "model_comparison"
NETWORK_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "network"
TEMPORAL_PREDICTION_OUTPUT_DIR = ANALYSIS_OUTPUT_DIR / "temporal_prediction"


# ============================================================================
# PROBE DIRECTORIES
# ============================================================================

# Legacy output directory (deprecated - kept for backward compatibility)
PROBE_OUTPUT_DIR = ANALYSIS_DIR / "probes" / "outputs"

# New organized run directory structure
PROBE_RUNS_DIR = ANALYSIS_DIR / "probes" / "runs"

# Current run output directories (set at runtime by the probe runner)
# These are used by individual probe modules when saving figures/metrics
_CURRENT_PROBE_RUN_DIR: Path = None
_CURRENT_PROBE_FIGURES_DIR: Path = None
_CURRENT_PROBE_METRICS_DIR: Path = None


def set_current_probe_run(run_dir: Path, figures_dir: Path, metrics_dir: Path) -> None:
    """Set the current probe run output directories.

    Called by the MasterProbeRunner when starting a new run.
    Individual probe modules should use get_probe_*_dir() functions.

    Args:
        run_dir: Base directory for the current run
        figures_dir: Directory for figures within the run
        metrics_dir: Directory for raw metrics within the run
    """
    global _CURRENT_PROBE_RUN_DIR, _CURRENT_PROBE_FIGURES_DIR, _CURRENT_PROBE_METRICS_DIR
    _CURRENT_PROBE_RUN_DIR = run_dir
    _CURRENT_PROBE_FIGURES_DIR = figures_dir
    _CURRENT_PROBE_METRICS_DIR = metrics_dir


def get_probe_run_dir() -> Path:
    """Get the current probe run directory.

    Returns:
        Path to current run directory, or legacy PROBE_OUTPUT_DIR if not set
    """
    if _CURRENT_PROBE_RUN_DIR is not None:
        return _CURRENT_PROBE_RUN_DIR
    return PROBE_OUTPUT_DIR


def get_probe_figures_dir() -> Path:
    """Get the current probe figures directory.

    Returns:
        Path to figures directory for the current run
    """
    if _CURRENT_PROBE_FIGURES_DIR is not None:
        return _CURRENT_PROBE_FIGURES_DIR
    return PROBE_OUTPUT_DIR


def get_probe_metrics_dir() -> Path:
    """Get the current probe metrics directory.

    Returns:
        Path to raw_metrics directory for the current run
    """
    if _CURRENT_PROBE_METRICS_DIR is not None:
        return _CURRENT_PROBE_METRICS_DIR
    return PROBE_OUTPUT_DIR


# ============================================================================
# LEGACY OUTPUT DIRECTORIES (for backward compatibility)
# These will be deprecated - use OUTPUT_DIR structure instead
# ============================================================================

ANALYSIS_FIGURES_DIR = FIGURES_DIR  # Now points to outputs/figures/
ANALYSIS_REPORTS_DIR = ANALYSIS_DIR / "reports"


# ============================================================================
# DATA SOURCE DIRECTORIES
# ============================================================================

# Primary data sources
UCDP_DIR = DATA_DIR / "ucdp"
FIRMS_DIR = DATA_DIR / "firms"
DEEPSTATE_DIR = DATA_DIR / "deepstate"
SENTINEL_DIR = DATA_DIR / "sentinel"
WAR_LOSSES_DIR = DATA_DIR / "war-losses-data"
NASA_DIR = DATA_DIR / "nasa"
VIIRS_DIR = NASA_DIR / "viirs_nightlights"
HDX_DIR = DATA_DIR / "hdx" / "ukraine"
IOM_DIR = DATA_DIR / "iom"
VIINA_DIR = DATA_DIR / "viina"

# Embedding directories
WAYBACK_DIR = DATA_DIR / "wayback_archives"
# NOTE: ISW embeddings removed from model on 2026-01-30 (near-zero correlation with latents).
# Path kept for reference by analysis/validation scripts but not used in training.
ISW_EMBEDDINGS_DIR = WAYBACK_DIR / "isw_assessments" / "embeddings"
TIMELINE_DIR = DATA_DIR / "timelines"
TIMELINE_EMBEDDINGS_DIR = TIMELINE_DIR / "embeddings"
TELEGRAM_DIR = DATA_DIR / "telegram"
TELEGRAM_EMBEDDINGS_DIR = TELEGRAM_DIR / "embeddings"


# ============================================================================
# SPECIFIC FILE PATHS (commonly used)
# ============================================================================

# Data files
UCDP_EVENTS_FILE = UCDP_DIR / "ged_events.csv"
FIRMS_ARCHIVE_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
FIRMS_NRT_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_nrt_SV-C2_706038.csv"
EQUIPMENT_LOSSES_FILE = WAR_LOSSES_DIR / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"
PERSONNEL_LOSSES_FILE = WAR_LOSSES_DIR / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"
SENTINEL_RAW_FILE = SENTINEL_DIR / "sentinel_timeseries_raw.json"
SENTINEL_WEEKLY_FILE = SENTINEL_DIR / "sentinel_weekly.json"
VIIRS_STATS_FILE = VIIRS_DIR / "viirs_daily_brightness_stats.csv"

# Model files
HAN_BEST_MODEL = MODEL_DIR / "han_best.pt"
HAN_FINAL_MODEL = MODEL_DIR / "han_final.pt"
UNIFIED_INTERP_MODEL = MODEL_DIR / "unified_interpolation_best.pt"
UNIFIED_DELTA_MODEL = MODEL_DIR / "unified_interpolation_delta_best.pt"
UNIFIED_HYBRID_MODEL = MODEL_DIR / "unified_interpolation_hybrid_best.pt"
TEMPORAL_PRED_MODEL = MODEL_DIR / "temporal_prediction_best.pt"
TACTICAL_STATE_MODEL = MODEL_DIR / "tactical_state_predictor_best.pt"


# ============================================================================
# CREDENTIAL FILES
# ============================================================================

ENV_FILE = PROJECT_ROOT / ".env"
EARTHDATA_TOKEN_FILE = PROJECT_ROOT / ".earthdata_token"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_model_path(model_name: str, variant: str = "best") -> Path:
    """Get standardized model checkpoint path.

    Args:
        model_name: Name of the model (e.g., 'han', 'unified_interpolation')
        variant: 'best', 'final', or epoch number

    Returns:
        Path to the model file
    """
    if variant.isdigit():
        return MODEL_DIR / f"{model_name}_epoch_{variant}.pt"
    return MODEL_DIR / f"{model_name}_{variant}.pt"


def get_phase_model_path(phase: int, domain: str) -> Path:
    """Get path for phase-specific model.

    Args:
        phase: Phase number (2 or 3)
        domain: Domain name (e.g., 'equipment_tanks', 'deepstate_polygons')

    Returns:
        Path to the model file
    """
    return MODEL_DIR / f"phase{phase}_{domain}_best.pt"


def get_interp_model_path(domain: str) -> Path:
    """Get path for interpolation model.

    Args:
        domain: Domain name (e.g., 'sentinel-2_cloud_metrics')

    Returns:
        Path to the model file
    """
    return INTERP_MODEL_DIR / f"interp_{domain}_best.pt"


# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that critical directories exist."""
    critical_dirs = [DATA_DIR, ANALYSIS_DIR, MODEL_DIR]
    missing = [d for d in critical_dirs if not d.exists()]

    if missing:
        print(f"Warning: Missing directories: {missing}")
        return False
    return True


# Run validation on import (non-blocking)
if __name__ != "__main__":
    try:
        validate_paths()
    except Exception:
        pass  # Silently continue if validation fails during import
