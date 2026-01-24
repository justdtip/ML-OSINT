#!/usr/bin/env python3
"""
Master Probe Runner for Multi-Resolution HAN Model
===================================================

A unified script to run the complete probe battery or specific subsets
for validating the Multi-Resolution Hierarchical Attention Network.

Usage:
------
    # Run all probes
    python run_probes.py --all

    # Run specific tier
    python run_probes.py --tier 1  # Critical probes only
    python run_probes.py --tier 2  # Important probes

    # Run specific section
    python run_probes.py --section 1  # Data artifacts
    python run_probes.py --section 2  # Cross-modal fusion

    # Run specific probe by ID
    python run_probes.py --probe 1.2.1  # VIIRS-Casualty temporal

    # Run data-only probes (no model required)
    python run_probes.py --data-only

    # List available probes
    python run_probes.py --list

Author: ML Engineering Team
Date: 2026-01-23
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Suppress warnings during import
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

# Import output manager for run organization
from .output_manager import RunOutputManager, list_runs, compare_runs

# ============================================================================
# Configuration - Compute paths BEFORE any imports that might affect them
# ============================================================================

# Get the directory containing this script - compute this FIRST before any imports
# Use realpath to resolve symlinks and get the true absolute path
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ANALYSIS_DIR.parent

# Add directories to path for imports (must come after path constants)
# PROJECT_DIR is needed for 'config' module imports
# ANALYSIS_DIR is needed for 'multi_resolution_han' and related module imports
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from probes import (
    get_available_modules,
    print_availability_report,
    TIER_1_PROBES,
    TIER_2_PROBES,
    TIER_3_PROBES,
)


def prepare_batch_for_model(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a batch from the dataloader for the model's forward method.

    Maps dataloader keys to model's expected parameter names.
    The collate function returns 'month_boundary_indices' but model expects 'month_boundaries'.
    """
    result = {}
    for key, value in batch.items():
        if key == 'month_boundary_indices':
            result['month_boundaries'] = value
        elif key not in ('batch_size', 'sample_indices', 'daily_seq_lens', 'monthly_seq_lens'):
            # Skip metadata keys that aren't model inputs
            result[key] = value
    return result


@dataclass
class ProbeRunnerConfig:
    """Configuration for the master probe runner."""

    # Paths - defaults are absolute paths based on script location
    checkpoint_path: Path = field(default_factory=lambda: ANALYSIS_DIR / "checkpoints/multi_resolution/best_checkpoint.pt")
    checkpoint_dir: Path = field(default_factory=lambda: ANALYSIS_DIR / "checkpoints/multi_resolution")
    data_dir: Path = field(default_factory=lambda: PROJECT_DIR / "data")

    # Run identification (for output organization)
    run_id: Optional[str] = None  # Auto-generated if None
    phase_name: str = ""  # e.g., "Phase0_Optimizations"
    phase_description: str = ""  # Description of what this run tests
    optimizations: List[str] = field(default_factory=list)  # List of optimizations applied

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Execution options
    batch_size: int = 8
    num_samples: int = 100
    num_workers: int = 0
    verbose: bool = True
    save_intermediate: bool = True

    # Probe selection
    run_tier: Optional[int] = None  # 1, 2, or 3
    run_section: Optional[int] = None  # 1-7
    run_probe_ids: List[str] = field(default_factory=list)
    data_only: bool = False  # Run only probes that don't require model

    # Pipeline stage selection (1=JIM, 2=Unified, 3=HAN, None=all stages)
    pipeline_stage: Optional[int] = None

    # Training run linkage (for loading from specific training runs)
    training_run_id: Optional[str] = None

    # Data split for probes - 'test' is default for evaluation, 'train' covers all conflict phases
    # Use 'train' for probes that need to analyze patterns across all phases (clustering, temporal patterns)
    # Use 'all' to combine train+val+test for maximum coverage
    probe_split: str = 'train'  # 'train', 'val', 'test', or 'all'

    def __post_init__(self):
        # Convert to Path objects if strings and resolve to absolute paths
        self.checkpoint_path = Path(self.checkpoint_path).resolve()
        self.checkpoint_dir = Path(self.checkpoint_dir).resolve()
        self.data_dir = Path(self.data_dir).resolve()


# ============================================================================
# Pipeline Stage to Probe Mapping
# ============================================================================
# Maps pipeline stages to the probe IDs that are relevant for each stage
PIPELINE_STAGE_PROBES = {
    # Stage 1: Joint Interpolation Models (JIM)
    # Probes that analyze interpolation quality and JIM model behavior
    1: [
        # Data artifact probes (all use raw data, applicable to JIM output)
        "1.1.1", "1.1.2", "1.1.3", "1.1.4",  # Equipment analysis
        "1.2.1", "1.2.2", "1.2.3", "1.2.4",  # VIIRS analysis
        "1.3.1",  # Personnel quality
        "1.4.1", "1.4.2",  # Statistical correlation
        "1.5.1",  # Neural pattern mining
        # JIM-specific interpretability probes
        "2.3.1",  # JIM Module I/O Analysis
        "2.3.2",  # JIM Attention Pattern Analysis
    ],
    # Stage 2: Unified Cross-Source Model
    # Probes that analyze cross-source relationships and unified embeddings
    2: [
        # All Stage 1 probes (unified model builds on interpolated data)
        "1.1.1", "1.1.2", "1.1.3", "1.1.4",
        "1.2.1", "1.2.2", "1.2.3", "1.2.4",
        "1.3.1", "1.4.1", "1.4.2", "1.5.1",
        "2.3.1", "2.3.2",
        # Cross-source analysis probes
        "2.4.1",  # Cross-Source Latent Analysis
        "2.4.2",  # Delta Model Validation
        # Model architecture comparison
        "8.1.1",  # Model Architecture Comparison
        "8.1.2",  # Reconstruction Performance Comparison
    ],
    # Stage 3: Hierarchical Attention Network (HAN)
    # Full model - all probes applicable
    3: None,  # None means all probes are applicable
}


# ============================================================================
# Probe Registry
# ============================================================================
@dataclass
class ProbeInfo:
    """Information about a single probe."""
    id: str
    name: str
    class_name: str
    section: int
    tier: int
    requires_model: bool = True
    requires_isw: bool = False
    module: str = ""


# Map export names to actual class names in modules
CLASS_NAME_MAPPING = {
    # data_artifact_probes
    "EncodingVarianceProbe": "EncodingVarianceComparisonProbe",
    "EquipmentTemporalLagProbe": "TemporalLagAnalysisProbe",
    "GeographicVIIRSProbe": "GeographicVIIRSDecompositionProbe",
    # tactical_readiness_probes
    "TacticalReadinessProbeRunner": "TacticalReadinessProbe",
}


# Build probe registry from tier definitions
PROBE_REGISTRY: Dict[str, ProbeInfo] = {}

for tier_num, tier_probes in [(1, TIER_1_PROBES), (2, TIER_2_PROBES), (3, TIER_3_PROBES)]:
    for probe_id, probe_name, probe_class in tier_probes:
        section = int(probe_id.split('.')[0])
        # Resolve class name to actual module class name
        actual_class_name = CLASS_NAME_MAPPING.get(probe_class, probe_class)

        # Determine special requirements
        requires_isw = probe_class in [
            "ISWAlignmentProbe", "TopicExtractionProbe", "ISWPredictiveContentProbe",
            "EventResponseProbe", "LagAnalysisProbe", "SemanticAnomalyProbe",
            "CounterfactualProbe", "SemanticPredictorProbe"
        ]

        requires_model = probe_class not in [
            "EncodingVarianceProbe", "EquipmentPersonnelRedundancyProbe",
            "EquipmentCategoryDisaggregationProbe", "EquipmentTemporalLagProbe",
            "VIIRSCasualtyTemporalProbe", "VIIRSFeatureDecompositionProbe",
            "TrendConfoundingProbe", "GeographicVIIRSProbe",
            "PersonnelVIIRSMediationProbe", "DataAvailabilityAudit",
            "SectorDefinition", "EntitySchemaSpec"
        ]

        # Determine module based on probe ID (section.subsection.probe)
        # Some subsections map to different modules than their main section
        subsection = probe_id.split('.')[1] if '.' in probe_id else '0'

        if section == 1:
            # 1.4.x and 1.5.x are in statistical_analysis_probes
            if subsection in ['4', '5']:
                module = "statistical_analysis_probes"
            else:
                module = "data_artifact_probes"
        elif section == 2:
            # 2.3.x and 2.4.x are in model_interpretability_probes
            if subsection in ['3', '4']:
                module = "model_interpretability_probes"
            else:
                module = "cross_modal_fusion_probes"
        elif section == 3:
            module = "temporal_dynamics_probes"
        elif section == 4:
            module = "semantic_structure_probes"
        elif section == 5:
            module = "semantic_association_probes"
        elif section == 6:
            module = "causal_importance_probes"
        elif section == 7:
            module = "tactical_readiness_probes"
        elif section == 8:
            module = "model_assessment_probes"
        else:
            module = "unknown"

        PROBE_REGISTRY[probe_id] = ProbeInfo(
            id=probe_id,
            name=probe_name,
            class_name=actual_class_name,
            section=section,
            tier=tier_num,
            requires_model=requires_model,
            requires_isw=requires_isw,
            module=module
        )


# ============================================================================
# Probe Result
# ============================================================================
@dataclass
class MasterProbeResult:
    """Result from a probe execution."""
    probe_id: str
    probe_name: str
    status: str  # "completed", "failed", "skipped"
    duration_seconds: float
    findings: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Master Probe Runner
# ============================================================================
class MasterProbeRunner:
    """
    Orchestrates the execution of all probes in the test battery.
    """

    def __init__(self, config: ProbeRunnerConfig):
        self.config = config
        self.results: Dict[str, MasterProbeResult] = {}
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.isw_data = None
        self._start_time = None

        # Setup output manager for organized run directories
        self.output_manager = RunOutputManager(run_id=config.run_id)
        self.output_manager.setup()

        # Set global output directories for probe modules to use
        from config.paths import set_current_probe_run
        set_current_probe_run(
            run_dir=self.output_manager.run_dir,
            figures_dir=self.output_manager.figures_dir,
            metrics_dir=self.output_manager.raw_metrics_dir,
        )

        # Set phase information if provided
        if config.phase_name:
            self.output_manager.set_phase_info(
                phase_name=config.phase_name,
                phase_description=config.phase_description,
                optimizations=config.optimizations,
            )

        # Update metadata with device info
        self.output_manager.update_metadata(device=config.device)

        # Now setup logging (which needs the output dir to exist)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ProbeRunner")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler - save to raw_metrics in run directory
        log_file = self.output_manager.raw_metrics_dir / "probe_run.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _load_model(self):
        """Load the trained model from checkpoint based on pipeline stage."""
        if self.model is not None:
            return

        # Dispatch to stage-specific loader if pipeline_stage is set
        stage = self.config.pipeline_stage
        if stage == 1:
            self._load_stage1_models()
            return
        elif stage == 2:
            self._load_stage2_model()
            return
        # Stage 3 or None: load Multi-Resolution HAN (default)

        self.logger.info(f"Loading model from {self.config.checkpoint_path}")

        try:
            # Import model class and SourceConfig
            from multi_resolution_han import MultiResolutionHAN, SourceConfig
            from multi_resolution_data import (
                MultiResolutionConfig,
                MultiResolutionDataset,
                create_multi_resolution_dataloaders,
                multi_resolution_collate_fn
            )
            import json

            # Load configuration from available sources
            # Priority: 1) training_summary.json 2) training run's config.json 3) defaults
            config_dict = {}

            # Try training_summary.json first (standalone train_multi_resolution.py runs)
            training_summary_path = self.config.checkpoint_dir / "training_summary.json"
            if training_summary_path.exists():
                with open(training_summary_path, "r") as f:
                    training_summary = json.load(f)
                config_dict = training_summary.get('config', {})

            # Also check training run's config.json (train_full_pipeline.py runs)
            # This may have data config values like use_disaggregated_equipment
            if self.config.training_run_id:
                from pathlib import Path
                training_run_dir = Path(self.config.checkpoint_dir).parent.parent
                training_run_config_path = training_run_dir / "config.json"
                if training_run_config_path.exists():
                    with open(training_run_config_path, "r") as f:
                        run_config = json.load(f)
                    # Merge run config (lower priority than training_summary)
                    for key in ['use_disaggregated_equipment', 'detrend_viirs']:
                        if key in run_config and key not in config_dict:
                            config_dict[key] = run_config[key]

            # Create data configuration
            # Note: defaults are conservative for older checkpoints without these settings
            data_config = MultiResolutionConfig(
                daily_seq_len=config_dict.get('daily_seq_len', 365),
                monthly_seq_len=config_dict.get('monthly_seq_len', 12),
                prediction_horizon=config_dict.get('prediction_horizon', 1),
                use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),
                detrend_viirs=config_dict.get('detrend_viirs', True),
            )

            # First create train dataset to get normalization stats
            train_dataset = MultiResolutionDataset(data_config, split='train')
            norm_stats = train_dataset.norm_stats

            # Create dataset for probes based on config.probe_split
            # 'train' gives access to all conflict phases (Initial Invasion through Attritional Warfare)
            # 'test' is most recent data only (typically single phase)
            # 'all' combines all splits for maximum phase coverage
            probe_split = self.config.probe_split
            self.logger.info(f"Using '{probe_split}' split for probes")

            if probe_split == 'all':
                # Combine all splits - create datasets and concatenate
                from torch.utils.data import ConcatDataset
                val_dataset = MultiResolutionDataset(data_config, split='val', norm_stats=norm_stats)
                test_dataset = MultiResolutionDataset(data_config, split='test', norm_stats=norm_stats)
                self.dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
                self.logger.info(f"Combined dataset: {len(train_dataset)} train + {len(val_dataset)} val + {len(test_dataset)} test = {len(self.dataset)} samples")
            elif probe_split == 'train':
                # Use train dataset directly (covers all conflict phases)
                self.dataset = train_dataset
            elif probe_split == 'val':
                self.dataset = MultiResolutionDataset(data_config, split='val', norm_stats=norm_stats)
            else:  # 'test'
                self.dataset = MultiResolutionDataset(data_config, split='test', norm_stats=norm_stats)

            # Get feature dimensions from dataset sample
            sample = self.dataset[0]

            # Get source names dynamically from the sample
            # This ensures we match the data config (e.g., disaggregated vs aggregated equipment)
            daily_source_names = list(sample.daily_features.keys())
            monthly_source_names = list(sample.monthly_features.keys())

            self.logger.info(f"Daily sources: {daily_source_names}")
            self.logger.info(f"Monthly sources: {monthly_source_names}")

            # Build source configs dynamically from the actual data
            daily_source_configs = {}
            monthly_source_configs = {}

            for source_name in daily_source_names:
                n_features = sample.daily_features[source_name].shape[-1]
                daily_source_configs[source_name] = SourceConfig(
                    name=source_name,
                    n_features=n_features,
                    resolution='daily',
                )

            for source_name in monthly_source_names:
                n_features = sample.monthly_features[source_name].shape[-1]
                monthly_source_configs[source_name] = SourceConfig(
                    name=source_name,
                    n_features=n_features,
                    resolution='monthly',
                )

            # Create model with correct parameters
            # Note: d_model=64 matches the trained checkpoint (2.3M params)
            # Previous default of 128 caused size mismatch errors
            self.model = MultiResolutionHAN(
                daily_source_configs=daily_source_configs,
                monthly_source_configs=monthly_source_configs,
                d_model=config_dict.get('d_model', 64),  # Match trained checkpoint
                nhead=config_dict.get('nhead', 4),  # Match trained checkpoint
                num_daily_layers=config_dict.get('num_daily_layers', 3),
                num_monthly_layers=config_dict.get('num_monthly_layers', 2),
                num_fusion_layers=config_dict.get('num_fusion_layers', 2),
                num_temporal_layers=2,
                dropout=0.0,  # No dropout for inference
            )

            # Load checkpoint
            # Note: weights_only=False is needed for checkpoints saved with numpy arrays
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.config.device,
                weights_only=False
            )

            # Handle state dict (may have 'model_state_dict' key)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Load state dict (allow missing keys for flexibility)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.config.device)
            self.model.eval()

            # Also create a dataloader for batch processing
            from torch.utils.data import DataLoader
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=multi_resolution_collate_fn
            )

            self.logger.info(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            self.logger.info(f"Daily sources: {list(daily_source_configs.keys())}")
            self.logger.info(f"Monthly sources: {list(monthly_source_configs.keys())}")

            # Extract metadata for run comparison
            self.output_manager.extract_model_metadata(
                checkpoint_path=self.config.checkpoint_path,
                model=self.model,
                config=config_dict,
            )
            self.output_manager.extract_data_metadata(data_config)
            self.output_manager.update_metadata(
                daily_sources=list(daily_source_configs.keys()),
                monthly_sources=list(monthly_source_configs.keys()),
                task_names=['casualty', 'regime', 'transition', 'anomaly', 'forecast'],
            )

            # Extract training info from checkpoint
            if 'history' in checkpoint:
                history = checkpoint.get('history', {})
                self.output_manager.update_metadata(
                    training_epochs=len(history.get('train_history', {}).get('total', [])),
                )
            if 'best_epoch' in checkpoint:
                self.output_manager.update_metadata(best_epoch=checkpoint['best_epoch'])
            if 'best_val_loss' in checkpoint:
                self.output_manager.update_metadata(best_val_loss=checkpoint['best_val_loss'])

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_stage1_models(self):
        """Load Stage 1: Joint Interpolation Models (JIM)."""
        self.logger.info("Loading Stage 1: Joint Interpolation Models")

        try:
            from config.paths import INTERP_MODEL_DIR

            # Scan directory for all available JIM models
            model_files = list(INTERP_MODEL_DIR.glob('interp_*_best.pt'))
            if not model_files:
                raise ValueError(f"No JIM models found in {INTERP_MODEL_DIR}")

            # Load each model and store metadata
            self.jim_models = {}
            self.jim_model_info = {}

            for model_path in sorted(model_files):
                # Extract source name from filename (interp_<source>_best.pt)
                source_name = model_path.stem.replace('interp_', '').replace('_best', '')
                self.logger.info(f"  Loading JIM model: {source_name}")

                try:
                    state = torch.load(model_path, map_location=self.config.device, weights_only=False)

                    # Store model state and metadata (don't instantiate model class)
                    model_info = {
                        'path': str(model_path),
                        'state': state,
                    }

                    # Extract model config if available
                    if isinstance(state, dict):
                        if 'config' in state:
                            model_info['config'] = state['config']
                        if 'model_state_dict' in state:
                            model_info['n_params'] = sum(
                                v.numel() for v in state['model_state_dict'].values()
                                if hasattr(v, 'numel')
                            )

                    self.jim_model_info[source_name] = model_info
                    self.jim_models[source_name] = state  # Store raw state for probes

                except Exception as e:
                    self.logger.warning(f"    Failed to load {source_name}: {e}")

            if not self.jim_models:
                raise ValueError("No JIM models could be loaded")

            self.logger.info(f"Loaded {len(self.jim_models)} JIM models")

            # Set model to None for stage 1 (probes should use jim_models directly)
            self.model = None

            # Update metadata
            self.output_manager.update_metadata(
                pipeline_stage=1,
                stage_name="Joint Interpolation Models (JIM)",
                jim_models=list(self.jim_models.keys()),
                n_jim_models=len(self.jim_models),
                task_names=['interpolation'],
            )

        except Exception as e:
            self.logger.error(f"Failed to load Stage 1 models: {e}")
            raise

    def _load_stage2_model(self):
        """Load Stage 2: Unified Cross-Source Model."""
        self.logger.info("Loading Stage 2: Unified Cross-Source Model")

        try:
            from config.paths import MODEL_DIR

            # Find available unified models
            model_paths = [
                MODEL_DIR / 'unified_interpolation_delta_best.pt',
                MODEL_DIR / 'unified_interpolation_best.pt',
            ]

            loaded_models = {}
            for model_path in model_paths:
                if model_path.exists():
                    model_name = model_path.stem.replace('_best', '')
                    self.logger.info(f"  Loading: {model_name}")

                    state = torch.load(model_path, map_location=self.config.device, weights_only=False)

                    # Store model state and metadata
                    model_info = {
                        'path': str(model_path),
                        'state': state,
                    }

                    # Extract parameter count
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        n_params = sum(
                            v.numel() for v in state['model_state_dict'].values()
                            if hasattr(v, 'numel')
                        )
                        model_info['n_params'] = n_params
                        self.logger.info(f"    Parameters: {n_params:,}")

                    loaded_models[model_name] = model_info

            if not loaded_models:
                raise ValueError(f"No unified models found at {MODEL_DIR}")

            # Store all unified models for probes
            self.unified_models = loaded_models

            # Set model to None (probes should use unified_models directly)
            self.model = None

            self.logger.info(f"Loaded {len(loaded_models)} unified model(s): {list(loaded_models.keys())}")

            # Update metadata
            self.output_manager.update_metadata(
                pipeline_stage=2,
                stage_name="Unified Cross-Source Model",
                unified_models=list(loaded_models.keys()),
                task_names=['reconstruction', 'cross_source'],
            )

        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 model: {e}")
            raise

    def _load_isw_data(self):
        """Load ISW embedding data for semantic probes."""
        if self.isw_data is not None:
            return

        self.logger.info("Loading ISW embeddings")

        try:
            isw_path = self.config.data_dir / "wayback_archives/isw_assessments/embeddings"

            embeddings = np.load(isw_path / "isw_embedding_matrix.npy")
            with open(isw_path / "isw_date_index.json", 'r') as f:
                date_index = json.load(f)

            self.isw_data = {
                'embeddings': embeddings,
                'date_index': date_index
            }

            self.logger.info(f"Loaded {embeddings.shape[0]} ISW embeddings")

        except Exception as e:
            self.logger.warning(f"Failed to load ISW data: {e}")
            self.isw_data = None

    def get_probes_to_run(self) -> List[ProbeInfo]:
        """Determine which probes to run based on configuration."""
        probes = []

        if self.config.run_probe_ids:
            # Run specific probes by ID
            for probe_id in self.config.run_probe_ids:
                if probe_id in PROBE_REGISTRY:
                    probes.append(PROBE_REGISTRY[probe_id])
                else:
                    self.logger.warning(f"Unknown probe ID: {probe_id}")

        elif self.config.run_tier is not None:
            # Run specific tier
            tier_num = self.config.run_tier
            tier_probes = {1: TIER_1_PROBES, 2: TIER_2_PROBES, 3: TIER_3_PROBES}.get(tier_num, [])
            for probe_id, _, _ in tier_probes:
                if probe_id in PROBE_REGISTRY:
                    probes.append(PROBE_REGISTRY[probe_id])

        elif self.config.run_section is not None:
            # Run specific section
            section = self.config.run_section
            probes = [p for p in PROBE_REGISTRY.values() if p.section == section]

        else:
            # Run all probes (Tier 1, then 2, then 3)
            for tier_num in [1, 2, 3]:
                tier_probes = {1: TIER_1_PROBES, 2: TIER_2_PROBES, 3: TIER_3_PROBES}[tier_num]
                for probe_id, _, _ in tier_probes:
                    if probe_id in PROBE_REGISTRY:
                        probes.append(PROBE_REGISTRY[probe_id])

        # Filter for data-only if requested
        if self.config.data_only:
            probes = [p for p in probes if not p.requires_model]

        # Filter by pipeline stage if specified
        if self.config.pipeline_stage is not None:
            stage = self.config.pipeline_stage
            stage_probe_ids = PIPELINE_STAGE_PROBES.get(stage)
            if stage_probe_ids is not None:  # None means all probes allowed (stage 3)
                probes = [p for p in probes if p.id in stage_probe_ids]
                self.logger.info(f"Filtered to {len(probes)} probes for pipeline stage {stage}")

        return probes

    def run_probe(self, probe_info: ProbeInfo) -> MasterProbeResult:
        """Run a single probe and return results."""
        self.logger.info(f"Running probe {probe_info.id}: {probe_info.name}")
        start_time = time.time()

        try:
            # Load dependencies if needed
            if probe_info.requires_model:
                self._load_model()

            if probe_info.requires_isw:
                self._load_isw_data()
                if self.isw_data is None:
                    return MasterProbeResult(
                        probe_id=probe_info.id,
                        probe_name=probe_info.name,
                        status="skipped",
                        duration_seconds=time.time() - start_time,
                        error_message="ISW data not available"
                    )

            # Import and instantiate probe class dynamically
            module = __import__(
                f"probes.{probe_info.module}",
                fromlist=[probe_info.class_name]
            )
            probe_class = getattr(module, probe_info.class_name)

            # Run the probe based on its type
            # This is a simplified version - each probe module has its own runner
            result = self._execute_probe(probe_class, probe_info)

            duration = time.time() - start_time
            self.logger.info(f"Probe {probe_info.id} completed in {duration:.2f}s")

            return MasterProbeResult(
                probe_id=probe_info.id,
                probe_name=probe_info.name,
                status="completed",
                duration_seconds=duration,
                findings=result.get('findings', {}),
                artifacts=result.get('artifacts', {}),
                recommendations=result.get('recommendations', [])
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Probe {probe_info.id} failed: {e}")
            import traceback
            traceback.print_exc()

            return MasterProbeResult(
                probe_id=probe_info.id,
                probe_name=probe_info.name,
                status="failed",
                duration_seconds=duration,
                error_message=str(e)
            )

    def _execute_probe(self, probe_class, probe_info: ProbeInfo) -> Dict[str, Any]:
        """Execute a probe and return its results."""
        # Data artifact probes (Section 1)
        if probe_info.section == 1:
            probe = probe_class()
            result = probe.run()
            return {
                'findings': result.findings if hasattr(result, 'findings') else {},
                'artifacts': result.artifacts if hasattr(result, 'artifacts') else {},
                'recommendations': result.recommendations if hasattr(result, 'recommendations') else []
            }

        # Cross-modal fusion probes (Section 2)
        elif probe_info.section == 2:
            from probes.cross_modal_fusion_probes import RSAProbeConfig, AttentionFlowProbeConfig

            try:
                if probe_info.class_name == "RSAProbe":
                    config = RSAProbeConfig(model=self.model)
                    probe = probe_class(config)
                    result = probe.run(self.dataloader)
                elif probe_info.class_name == "AttentionFlowProbe":
                    config = AttentionFlowProbeConfig(model=self.model)
                    probe = probe_class(config)
                    result = probe.run(self.dataloader)
                elif probe_info.class_name == "AblationProbe":
                    from probes.cross_modal_fusion_probes import AblationProbeConfig
                    config = AblationProbeConfig(model=self.model)
                    probe = probe_class(config)
                    result = probe.run(self.dataloader, task_evaluators={})
                else:
                    # Checkpoint comparison - use weights_only=False for loading checkpoints
                    from probes.cross_modal_fusion_probes import CheckpointProbeConfig
                    config = CheckpointProbeConfig(
                        model=self.model,
                        checkpoint_dir=self.config.checkpoint_dir
                    )
                    probe = probe_class(config)
                    result = probe.run(self.dataloader)

                return {
                    'findings': result.interpretation if hasattr(result, 'interpretation') else {},
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': result.recommendations if hasattr(result, 'recommendations') else []
                }
            except (IndexError, ValueError, RuntimeError) as e:
                # Handle empty data or dimension mismatch errors gracefully
                return {
                    'findings': {'error': str(e), 'probe': probe_info.class_name},
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': [f'Probe {probe_info.class_name} failed: {str(e)}. Check data availability.']
                }

        # Temporal dynamics probes (Section 3)
        elif probe_info.section == 3:
            try:
                probe = probe_class(self.model, self.dataset, self.config.device)
                # Some probes accept num_samples, others don't
                try:
                    result = probe.run(num_samples=self.config.num_samples)
                except TypeError:
                    result = probe.run()
                result.save(self.output_manager.figures_dir)
                # Get figure paths - result has 'figures' dict, not 'figure_paths'
                figure_paths = list(result.figures.keys()) if hasattr(result, 'figures') else []
                return {
                    'findings': result.metrics if hasattr(result, 'metrics') else {},
                    'artifacts': {'figures': figure_paths, 'tables': []},
                    'recommendations': []
                }
            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                # Handle empty data or dimension errors (e.g., "Number of columns must be positive",
                # "expected non-empty vector")
                error_msg = str(e)
                return {
                    'findings': {'error': error_msg, 'probe': probe_info.class_name},
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': [
                        f'Probe {probe_info.class_name} failed: {error_msg}.',
                        'This may indicate insufficient data samples spanning transition periods.',
                        'Consider increasing num_samples or checking dataset date coverage.'
                    ]
                }

        # Semantic structure probes (Section 4)
        elif probe_info.section == 4:
            try:
                # Extract latent representations first
                latents = self._extract_latents()
                dates = self._get_dates()

                # Ensure latents and dates have matching dimensions
                if len(latents) != len(dates):
                    # Truncate to minimum length
                    min_len = min(len(latents), len(dates))
                    latents = latents[:min_len]
                    dates = dates[:min_len]

                if probe_info.class_name == "OperationClusteringProbe":
                    probe = probe_class(latents, dates)
                    result = probe.compute_clustering_metrics()
                elif probe_info.class_name in ["DayTypeDecodingProbe", "IntensityProbe", "GeographicFocusProbe"]:
                    # Create intensity labels from latent magnitudes as proxy
                    # Use tertiles [33, 67] instead of quartiles to ensure more samples per class
                    latent_magnitudes = np.linalg.norm(latents, axis=1)
                    labels = np.digitize(latent_magnitudes,
                                        np.percentile(latent_magnitudes, [33, 67]))

                    # Merge sparse classes: ensure each class has at least 2 samples for stratified split
                    unique_labels, label_counts = np.unique(labels, return_counts=True)
                    min_samples_per_class = 2

                    # If any class has fewer than min_samples_per_class, merge it with adjacent class
                    for i, (label, count) in enumerate(zip(unique_labels, label_counts)):
                        if count < min_samples_per_class:
                            # Merge with the nearest label (prefer lower label)
                            if label > 0:
                                labels[labels == label] = label - 1
                            elif label < len(unique_labels) - 1:
                                labels[labels == label] = label + 1

                    # Recompute unique labels after merging
                    unique_labels, label_counts = np.unique(labels, return_counts=True)

                    # Final check: if we still have classes with < 2 samples, skip stratification
                    if any(c < min_samples_per_class for c in label_counts):
                        return {
                            'findings': {
                                'error': f'Insufficient samples per class after merging. '
                                         f'Label distribution: {dict(zip(unique_labels, label_counts))}',
                                'probe': probe_info.class_name,
                                'n_samples': len(labels),
                            },
                            'artifacts': {'figures': [], 'tables': []},
                            'recommendations': [
                                'Increase num_samples to get more data points per class.',
                                'Consider using a different labeling strategy with fewer classes.'
                            ]
                        }

                    probe = probe_class(latents, labels)
                    result = probe.train_probe()
                elif probe_info.class_name == "TemporalPatternProbe":
                    probe = probe_class(latents, dates)
                    # Run all temporal tests
                    weekly = probe.test_weekly_cycle()
                    seasonal = probe.test_seasonal_pattern()
                    result = {
                        'weekly_cycle': weekly.__dict__ if hasattr(weekly, '__dict__') else weekly,
                        'seasonal_pattern': seasonal.__dict__ if hasattr(seasonal, '__dict__') else seasonal
                    }
                else:
                    # Fallback for other Section 4 probes
                    try:
                        probe = probe_class(latents, dates)
                        if hasattr(probe, 'compute_clustering_metrics'):
                            result = probe.compute_clustering_metrics()
                        elif hasattr(probe, 'train_probe'):
                            result = probe.train_probe()
                        elif hasattr(probe, 'test_weekly_cycle'):
                            result = probe.test_weekly_cycle()
                        else:
                            result = {'error': 'No suitable method found'}
                    except TypeError as e:
                        result = {'error': str(e)}

                # Convert result to dict if needed
                if hasattr(result, '__dict__'):
                    result = result.__dict__
                elif hasattr(result, 'to_dict'):
                    result = result.to_dict()

                return {
                    'findings': result if isinstance(result, dict) else {},
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': []
                }
            except (ValueError, IndexError) as e:
                error_msg = str(e)
                # Handle stratification errors specifically
                if "least populated classes" in error_msg or "minimum number of groups" in error_msg:
                    return {
                        'findings': {
                            'error': 'Stratification failed - too few samples per class',
                            'details': error_msg,
                            'probe': probe_info.class_name,
                        },
                        'artifacts': {'figures': [], 'tables': []},
                        'recommendations': [
                            'Increase num_samples to ensure at least 2 samples per class.',
                            'Consider using fewer label bins or a different labeling strategy.'
                        ]
                    }
                else:
                    return {
                        'findings': {'error': error_msg, 'probe': probe_info.class_name},
                        'artifacts': {'figures': [], 'tables': []},
                        'recommendations': [f'Probe failed: {error_msg}']
                    }

        # Semantic association probes (Section 5)
        elif probe_info.section == 5:
            latents = self._extract_latents()

            probe = probe_class()
            if hasattr(probe, 'run'):
                result = probe.run(
                    isw_embeddings=self.isw_data['embeddings'],
                    date_index=self.isw_data['date_index'],
                    model_latents=latents
                )
            else:
                result = {}

            return {
                'findings': result if isinstance(result, dict) else {},
                'artifacts': {'figures': [], 'tables': []},
                'recommendations': []
            }

        # Causal importance probes (Section 6)
        elif probe_info.section == 6:
            probe = probe_class(self.model, self.config.device)
            # Prepare a single batch with correct key names for model
            raw_batch = next(iter(self.dataloader))
            batch = prepare_batch_for_model(raw_batch)
            result = probe.run(batch)
            return {
                'findings': result.to_dict() if hasattr(result, 'to_dict') else {},
                'artifacts': {'figures': [], 'tables': []},
                'recommendations': []
            }

        # Tactical readiness probes (Section 7)
        elif probe_info.section == 7:
            try:
                # Section 7 probes have different initialization patterns
                if probe_info.class_name == "SectorCorrelationProbe":
                    # SectorCorrelationProbe requires a SectorDefinition instance
                    from probes.tactical_readiness_probes import SectorDefinition
                    sector_def = SectorDefinition()
                    probe = probe_class(sector_def)
                    # Return probe info since it needs data to run correlation analysis
                    result = {
                        'probe_type': probe_info.class_name,
                        'status': 'initialized',
                        'note': 'SectorCorrelationProbe requires FIRMS data to compute correlations. '
                                'Call compute_sector_correlations_firms() with a DataFrame.',
                        'available_methods': ['compute_sector_correlations_firms', 'generate_independence_report']
                    }
                elif probe_info.class_name == "ResolutionAnalysisProbe":
                    # ResolutionAnalysisProbe requires a DataAvailabilityAudit instance
                    from probes.tactical_readiness_probes import DataAvailabilityAudit
                    audit = DataAvailabilityAudit()
                    audit.audit_all_sources()
                    probe = probe_class(audit)
                    # Run the analysis methods
                    temporal_analysis = probe.analyze_temporal_resolution()
                    spatial_analysis = probe.analyze_spatial_resolution()
                    recommendation = probe.get_optimal_resolution_recommendation()
                    result = {
                        'temporal_resolution_analysis': {
                            res: {
                                'feasibility': tradeoff.overall_feasibility,
                                'recommendations': tradeoff.recommendations,
                            }
                            for res, tradeoff in temporal_analysis.items()
                        },
                        'spatial_resolution_analysis': {
                            res: {
                                'feasibility': tradeoff.overall_feasibility,
                                'recommendations': tradeoff.recommendations,
                            }
                            for res, tradeoff in spatial_analysis.items()
                        },
                        'optimal_recommendation': recommendation
                    }
                else:
                    # Default handling for other Section 7 probes
                    probe = probe_class()
                    # Try different methods to extract info
                    if hasattr(probe, 'run'):
                        result = probe.run()
                    elif hasattr(probe, 'audit_all_sources'):
                        audit_result = probe.audit_all_sources()
                        # Convert SourceAuditResult objects to dicts for JSON serialization
                        result = {}
                        for source_name, audit in audit_result.items():
                            if hasattr(audit, '__dict__'):
                                audit_dict = {}
                                for key, val in audit.__dict__.items():
                                    if hasattr(val, '__dict__'):
                                        audit_dict[key] = str(val)
                                    elif isinstance(val, dict):
                                        # Convert nested objects
                                        audit_dict[key] = {
                                            k: str(v) if hasattr(v, '__dict__') else v
                                            for k, v in val.items()
                                        }
                                    else:
                                        audit_dict[key] = val
                                result[source_name] = audit_dict
                            else:
                                result[source_name] = str(audit)
                    elif hasattr(probe, 'get_sectors'):
                        sectors = probe.get_sectors()
                        result = {
                            'sectors': {k: str(v) if hasattr(v, '__dict__') else v for k, v in sectors.items()}
                            if isinstance(sectors, dict) else sectors
                        }
                    elif hasattr(probe, 'get_unit_schema'):
                        result = {
                            'unit_schema': probe.get_unit_schema(),
                            'infrastructure_schema': probe.get_infrastructure_schema() if hasattr(probe, 'get_infrastructure_schema') else {}
                        }
                    else:
                        # Just return the object's dict representation if available
                        result = vars(probe) if hasattr(probe, '__dict__') else {}

                return {
                    'findings': result if isinstance(result, dict) else {},
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': []
                }
            except Exception as e:
                return {
                    'findings': {
                        'error': str(e),
                        'probe': probe_info.class_name,
                    },
                    'artifacts': {'figures': [], 'tables': []},
                    'recommendations': [f'Probe {probe_info.class_name} failed: {str(e)}']
                }

        else:
            return {'findings': {}, 'artifacts': {}, 'recommendations': []}

    def _extract_latents(self) -> np.ndarray:
        """Extract latent representations from the model.

        Returns:
            np.ndarray: 2D array of shape [n_samples, n_features] suitable for
                        scikit-learn transformers (e.g., StandardScaler) and probes.

        Note:
            Model outputs like 'casualty_pred' have shape [batch, seq, features].
            We take the last timestep to get [batch, features], then concatenate
            across batches to get [n_samples, features].
        """
        if self.model is None:
            self._load_model()

        latents = []
        with torch.no_grad():
            for batch in self.dataloader:
                # Move to device
                daily_features = {
                    k: v.to(self.config.device)
                    for k, v in batch['daily_features'].items()
                }
                daily_masks = {
                    k: v.to(self.config.device)
                    for k, v in batch['daily_masks'].items()
                }
                monthly_features = {
                    k: v.to(self.config.device)
                    for k, v in batch['monthly_features'].items()
                }
                monthly_masks = {
                    k: v.to(self.config.device)
                    for k, v in batch['monthly_masks'].items()
                }
                month_boundaries = batch['month_boundary_indices'].to(self.config.device)

                # Forward pass
                outputs = self.model(
                    daily_features=daily_features,
                    daily_masks=daily_masks,
                    monthly_features=monthly_features,
                    monthly_masks=monthly_masks,
                    month_boundaries=month_boundaries
                )

                # Get fused representation - use casualty predictions as proxy for latent
                # The model doesn't directly expose internal representations
                if 'fused_representation' in outputs:
                    latent = outputs['fused_representation'].cpu().numpy()
                elif 'temporal_output' in outputs:
                    latent = outputs['temporal_output'].cpu().numpy()
                elif 'casualty_pred' in outputs:
                    # Use casualty predictions as proxy for latent representation
                    latent = outputs['casualty_pred'].cpu().numpy()
                else:
                    # Fallback: use first available output tensor
                    for key, val in outputs.items():
                        if isinstance(val, torch.Tensor) and val.dim() >= 2:
                            latent = val.cpu().numpy()
                            break
                    else:
                        raise ValueError(f"No suitable latent representation found in outputs: {list(outputs.keys())}")

                # Convert 3D tensors [batch, seq, features] to 2D [batch, features]
                # by taking the last timestep. This is required because:
                # 1. StandardScaler requires dim <= 2
                # 2. Probes expect [n_samples, n_features] format
                # 3. The last timestep represents the final prediction state
                if latent.ndim == 3:
                    latent = latent[:, -1, :]  # Take last timestep: [batch, seq, feat] -> [batch, feat]

                latents.append(latent)

        return np.concatenate(latents, axis=0)

    def _get_dates(self) -> np.ndarray:
        """Get dates from dataset.

        Returns:
            np.ndarray: Array of dates with one entry per sample, matching
                        the number of rows returned by _extract_latents().

        Note:
            Since _extract_latents() returns one row per sample (taking the
            last timestep), we return one date per sample (the last date
            in the sequence, representing the prediction target date).
            We iterate through the dataloader to match the same batches
            as _extract_latents().
        """
        dates = []
        # Iterate through the same dataloader as _extract_latents to ensure alignment
        for batch in self.dataloader:
            batch_size = 1
            # Try to determine batch size from the batch
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if hasattr(batch[key][sub_key], 'shape'):
                            batch_size = batch[key][sub_key].shape[0]
                            break
                    break
                elif hasattr(batch[key], 'shape'):
                    batch_size = batch[key].shape[0]
                    break

            # Get dates for this batch
            if 'monthly_dates' in batch:
                # Use the batch's monthly_dates if available
                batch_dates = batch['monthly_dates']
                for i in range(batch_size):
                    if i < len(batch_dates) and len(batch_dates[i]) > 0:
                        dates.append(batch_dates[i][-1])  # Last date in sequence
                    else:
                        dates.append(len(dates))  # Fallback index
            else:
                # Fallback: use indices
                for _ in range(batch_size):
                    dates.append(len(dates))

        return np.array(dates)

    def _get_casualty_data(self) -> np.ndarray:
        """Get casualty data for label construction."""
        # This would load from the personnel data source
        # For now, return placeholder
        return np.random.randn(len(self.dataset) * 12)

    def run_all(self) -> Dict[str, MasterProbeResult]:
        """Run all selected probes."""
        probes = self.get_probes_to_run()

        self.logger.info(f"Running {len(probes)} probes")
        self.logger.info("=" * 60)

        for probe_info in probes:
            result = self.run_probe(probe_info)
            self.results[probe_info.id] = result

            if self.config.save_intermediate:
                self._save_result(result)

        # Generate final report
        self.generate_report()

        return self.results

    def run_tier(self, tier: int) -> Dict[str, MasterProbeResult]:
        """Run all probes in a specific tier."""
        self.config.run_tier = tier
        return self.run_all()

    def run_section(self, section: int) -> Dict[str, MasterProbeResult]:
        """Run all probes in a specific section."""
        self.config.run_section = section
        return self.run_all()

    def _save_result(self, result: MasterProbeResult):
        """Save individual probe result to raw_metrics."""
        result_path = self.output_manager.get_metrics_path(f"probe_{result.probe_id.replace('.', '_')}")
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def generate_report(self):
        """Generate comprehensive markdown report."""
        report_path = self.output_manager.report_path

        with open(report_path, 'w') as f:
            f.write("# Multi-Resolution HAN Probe Battery Report\n\n")
            f.write(f"**Generated:** {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

            # Summary statistics
            total = len(self.results)
            completed = sum(1 for r in self.results.values() if r.status == "completed")
            failed = sum(1 for r in self.results.values() if r.status == "failed")
            skipped = sum(1 for r in self.results.values() if r.status == "skipped")

            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Probes:** {total}\n")
            f.write(f"- **Completed:** {completed}\n")
            f.write(f"- **Failed:** {failed}\n")
            f.write(f"- **Skipped:** {skipped}\n")
            f.write(f"- **Total Duration:** {sum(r.duration_seconds for r in self.results.values()):.2f}s\n\n")

            # Key findings by tier
            f.write("## Key Findings by Tier\n\n")

            for tier in [1, 2, 3]:
                f.write(f"### Tier {tier}\n\n")
                tier_results = [
                    (probe_id, result)
                    for probe_id, result in self.results.items()
                    if PROBE_REGISTRY.get(probe_id, ProbeInfo("", "", "", 0, 0)).tier == tier
                ]

                if not tier_results:
                    f.write("*No probes run*\n\n")
                    continue

                f.write("| Probe ID | Name | Status | Duration |\n")
                f.write("|----------|------|--------|----------|\n")

                for probe_id, result in tier_results:
                    status_icon = {"completed": "✓", "failed": "✗", "skipped": "⊘"}.get(result.status, "?")
                    f.write(f"| {probe_id} | {result.probe_name} | {status_icon} {result.status} | {result.duration_seconds:.2f}s |\n")

                f.write("\n")

            # Detailed results
            f.write("## Detailed Results\n\n")

            for probe_id, result in sorted(self.results.items()):
                f.write(f"### {probe_id}: {result.probe_name}\n\n")
                f.write(f"**Status:** {result.status}\n")
                f.write(f"**Duration:** {result.duration_seconds:.2f}s\n\n")

                if result.error_message:
                    f.write(f"**Error:** {result.error_message}\n\n")

                if result.findings:
                    f.write("**Findings:**\n")
                    f.write("```json\n")
                    f.write(json.dumps(result.findings, indent=2, default=str))
                    f.write("\n```\n\n")

                if result.recommendations:
                    f.write("**Recommendations:**\n")
                    for rec in result.recommendations:
                        f.write(f"- {rec}\n")
                    f.write("\n")

                f.write("---\n\n")

        self.logger.info(f"Report generated: {report_path}")

        # Also save as JSON in raw_metrics
        json_path = self.output_manager.get_metrics_path("probe_battery_results")
        with open(json_path, 'w') as f:
            json.dump(
                {pid: r.to_dict() for pid, r in self.results.items()},
                f, indent=2, default=str
            )

        # Finalize metadata
        total = len(self.results)
        completed = sum(1 for r in self.results.values() if r.status == "completed")
        failed = sum(1 for r in self.results.values() if r.status == "failed")
        total_duration = sum(r.duration_seconds for r in self.results.values())

        self.output_manager.finalize(
            probes_completed=completed,
            probes_failed=failed,
            total_duration=total_duration,
        )

        self.logger.info(f"JSON results saved: {json_path}")


# ============================================================================
# CLI Interface
# ============================================================================
def list_probes():
    """Print all available probes."""
    print("\n" + "=" * 80)
    print("Multi-Resolution HAN Probe Battery - Available Probes")
    print("=" * 80 + "\n")

    for tier in [1, 2, 3]:
        tier_name = {1: "Critical", 2: "Important", 3: "Exploratory"}[tier]
        print(f"Tier {tier} ({tier_name}):")
        print("-" * 60)

        probes = [p for p in PROBE_REGISTRY.values() if p.tier == tier]
        for probe in sorted(probes, key=lambda x: x.id):
            req = []
            if probe.requires_model:
                req.append("model")
            if probe.requires_isw:
                req.append("ISW")
            req_str = f" [{', '.join(req)}]" if req else ""
            print(f"  {probe.id:8s} | {probe.name:45s}{req_str}")

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Multi-Resolution HAN probe battery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_probes.py --all                  # Run all probes
  python run_probes.py --tier 1               # Run critical probes only
  python run_probes.py --section 1            # Run data artifact probes
  python run_probes.py --probe 1.2.1 1.1.2    # Run specific probes
  python run_probes.py --data-only            # Run probes that don't need model
  python run_probes.py --list                 # List available probes
        """
    )

    # Probe selection
    parser.add_argument("--all", action="store_true", help="Run all probes")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run specific tier")
    parser.add_argument("--section", type=int, choices=list(range(1, 8)), help="Run specific section (1-7)")
    parser.add_argument("--probe", nargs="+", help="Run specific probe(s) by ID")
    parser.add_argument("--data-only", action="store_true", help="Run only data probes (no model required)")
    parser.add_argument("--pipeline-stage", type=int, choices=[1, 2, 3],
                        help="Run probes for specific pipeline stage: 1=JIM, 2=Unified, 3=HAN (default: 3)")
    parser.add_argument("--list", action="store_true", help="List available probes")

    # Paths - use None as default to trigger dataclass defaults
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: analysis/checkpoints/multi_resolution/best_checkpoint.pt)")
    parser.add_argument("--training-run", type=str, default=None,
                        help="Training run ID to load models from (e.g., 'run_24-01-2026_14-30')")
    parser.add_argument("--list-training-runs", action="store_true",
                        help="List all available training runs")

    # Run identification and phase tracking
    parser.add_argument("--run-id", type=str, default=None,
                        help="Custom run ID (default: auto-generated timestamp)")
    parser.add_argument("--phase", type=str, default="",
                        help="Phase name for tracking (e.g., 'Phase0_Optimizations')")
    parser.add_argument("--phase-desc", type=str, default="",
                        help="Description of what this run tests")
    parser.add_argument("--optimizations", nargs="+", default=[],
                        help="List of optimizations applied (e.g., --optimizations 'VIIRS detrending' 'Task priors')")

    # Run management
    parser.add_argument("--list-runs", action="store_true",
                        help="List all previous probe runs")
    parser.add_argument("--compare-runs", nargs="+",
                        help="Compare metadata across runs (provide run IDs)")

    # Execution options
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Data split for probes
    parser.add_argument("--probe-split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="Data split for probes. 'train' covers all conflict phases (default), "
                             "'test' is most recent only, 'all' combines all splits")

    args = parser.parse_args()

    # Handle list command
    if args.list:
        print_availability_report()
        list_probes()
        return

    # Handle list-runs command
    if args.list_runs:
        runs = list_runs()
        if not runs:
            print("No previous probe runs found.")
        else:
            print("\n" + "=" * 80)
            print("Previous Probe Runs")
            print("=" * 80 + "\n")
            for run in runs:
                phase = run.get('phase_name', 'Unknown')
                completed = run.get('probes_completed', 0)
                failed = run.get('probes_failed', 0)
                d_model = run.get('d_model', '?')
                detrend = "Yes" if run.get('detrend_viirs') else "No"
                print(f"  {run['run_id']}")
                print(f"    Phase: {phase}, d_model: {d_model}, detrend_viirs: {detrend}")
                print(f"    Probes: {completed} completed, {failed} failed")
                print()
        return

    # Handle compare-runs command
    if args.compare_runs:
        comparison = compare_runs(args.compare_runs)
        print("\n" + "=" * 80)
        print("Run Comparison")
        print("=" * 80 + "\n")
        if comparison["differences"]:
            print("Differences found:")
            for key, values in comparison["differences"].items():
                print(f"  {key}:")
                for i, val in enumerate(values):
                    run_id = args.compare_runs[i] if i < len(args.compare_runs) else f"Run {i}"
                    print(f"    {run_id}: {val}")
        else:
            print("No differences found in compared fields.")
        return

    # Handle list-training-runs command
    if args.list_training_runs:
        from training_output_manager import list_training_runs
        training_runs = list_training_runs()
        if not training_runs:
            print("No training runs found.")
        else:
            print("\n" + "=" * 80)
            print("Available Training Runs")
            print("=" * 80 + "\n")
            for run in training_runs:
                run_id = run.get('run_id', 'Unknown')
                stages_complete = sum(1 for i in range(1, 6) if run.get(f'stage{i}_complete', False))
                d_model = run.get('d_model', '?')
                multi_res = "Yes" if run.get('use_multi_resolution') else "No"
                duration = run.get('total_duration_seconds', 0)
                print(f"  {run_id}")
                print(f"    Stages complete: {stages_complete}/5, d_model: {d_model}, multi_resolution: {multi_res}")
                if duration > 0:
                    print(f"    Duration: {duration/60:.1f} minutes")
                print()
        return

    # Create configuration
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Build config kwargs
    config_kwargs = {
        "device": device,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "verbose": args.verbose,
        "run_tier": args.tier,
        "run_section": args.section,
        "run_probe_ids": args.probe or [],
        "data_only": args.data_only,
        "pipeline_stage": args.pipeline_stage,
        "probe_split": args.probe_split,
        "run_id": args.run_id,
        "phase_name": args.phase,
        "phase_description": args.phase_desc,
        "optimizations": args.optimizations,
    }

    # Only override defaults if paths were explicitly provided
    if args.checkpoint is not None:
        config_kwargs["checkpoint_path"] = Path(args.checkpoint).resolve()

    # Handle training run - load checkpoints from training run directory
    training_run_config = None
    if args.training_run is not None:
        from training_output_manager import get_training_run, TRAINING_RUNS_DIR
        training_manager = get_training_run(args.training_run)
        if training_manager is None:
            print(f"Error: Training run '{args.training_run}' not found.")
            print(f"Available runs in: {TRAINING_RUNS_DIR}")
            return

        # Load training config
        try:
            training_run_config = training_manager.load_config()
        except FileNotFoundError:
            print(f"Warning: No config.json found in training run, using defaults")

        # Set checkpoint paths from training run
        han_dir = training_manager.get_stage_dir(3)
        checkpoint_path = han_dir / "best_checkpoint.pt"
        if checkpoint_path.exists():
            config_kwargs["checkpoint_path"] = checkpoint_path
            config_kwargs["checkpoint_dir"] = han_dir
        else:
            print(f"Warning: No HAN checkpoint found in {han_dir}")

        # Link the probe run to the training run
        config_kwargs["training_run_id"] = args.training_run
        if not config_kwargs.get("phase_name"):
            config_kwargs["phase_name"] = f"Training:{args.training_run}"
        print(f"Loading models from training run: {args.training_run}")
        print(f"  Training run directory: {training_manager.run_dir}")

    config = ProbeRunnerConfig(**config_kwargs)

    # Run probes
    runner = MasterProbeRunner(config)

    if args.all or (args.tier is None and args.section is None and not args.probe):
        # Default to running all
        config.run_tier = None
        config.run_section = None
        results = runner.run_all()
    else:
        results = runner.run_all()

    # Print summary
    print("\n" + "=" * 60)
    print("Probe Battery Complete")
    print("=" * 60)
    completed = sum(1 for r in results.values() if r.status == "completed")
    failed = sum(1 for r in results.values() if r.status == "failed")
    print(f"Completed: {completed}, Failed: {failed}")
    print(f"Run ID: {runner.output_manager.run_id}")
    print(f"Results saved to: {runner.output_manager.run_dir}")
    print(f"  - Report: {runner.output_manager.report_path}")
    print(f"  - Figures: {runner.output_manager.figures_dir}")
    print(f"  - Metrics: {runner.output_manager.raw_metrics_dir}")


if __name__ == "__main__":
    main()
