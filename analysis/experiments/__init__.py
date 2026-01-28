"""
Experiments Module for ML_OSINT

This module contains experimental training scripts and ablation studies
for the Multi-Resolution HAN model.

Available experiments:
- isw_ablation_experiment: ISW integration ablation study
- modular_training_experiment: Modular data source configuration experiments

Usage:
    from analysis.experiments.modular_training_experiment import (
        ModularTrainingExperiment,
        ExperimentConfig,
    )

    config = ExperimentConfig(data_config_name='spatial_rich', epochs=100)
    experiment = ModularTrainingExperiment(config)
    results = experiment.run_single_experiment('spatial_rich')
"""

from pathlib import Path

# Module directory
EXPERIMENTS_DIR = Path(__file__).parent

__all__ = [
    'EXPERIMENTS_DIR',
]
