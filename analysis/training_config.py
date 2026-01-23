"""
Unified Configuration Module for ML Training Pipeline.

This module centralizes all hyperparameters and settings for the tactical
prediction ML system, including data processing, model architecture, and
training configurations.

Usage:
    from training_config import get_config, ExperimentConfig

    # Use a preset configuration
    config = get_config('default')

    # Or create a custom configuration
    config = ExperimentConfig(
        data=DataConfig(resolution='daily'),
        training=TrainingConfig(epochs=100)
    )

    # Save and load configurations
    config.save('experiment_config.json')
    loaded_config = ExperimentConfig.load('experiment_config.json')
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Literal
import json
import os
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL,
)


@dataclass
class DataConfig:
    """Configuration for data processing and feature engineering.

    Context Window Findings (Probe 3.1.1 - Context Window Paradox):
        Analysis revealed an inverted relationship between context window size and
        model performance. Shorter context windows yield BETTER results:

        - 7 days:  78.8% accuracy, F1 0.318 (OPTIMAL)
        - 14 days: 78.4% accuracy, F1 0.318 (OPTIMAL)
        - 30 days: 77.2% accuracy, F1 0.319
        - 60 days: 75.9% accuracy, F1 0.320
        - 90 days: 74.4% accuracy, F1 0.319
        - Full:    51.3% accuracy, F1 0.235

        This suggests conflict dynamics exhibit rapid regime changes where recent
        context (7-14 days) is most predictive. Longer windows introduce noise
        from outdated patterns that confuse the model.

    Attributes:
        resolution: Time resolution for aggregation ('daily', 'weekly', 'monthly').
        temporal_gap_days: Gap in days between training and prediction windows.
        train_ratio: Proportion of data for training set.
        val_ratio: Proportion of data for validation set.
        test_ratio: Proportion of data for test set.
        seq_len: Number of time steps in input sequences (resolution-dependent).
        context_window_days: Explicit context window in days (7-14 optimal).
            When set, overrides seq_len based on resolution. If None, seq_len
            is used directly.
        prediction_horizon: Number of time steps to predict ahead.
        use_reduced_features: Whether to use reduced feature set for efficiency.
        imputation_strategy: Strategy for handling missing values
            ('domain_specific', 'mean', 'median', 'forward_fill', 'zero').
    """
    resolution: Literal['daily', 'weekly', 'monthly'] = 'weekly'
    temporal_gap_days: int = 14
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_len: int = 14  # Updated default: 14 days for daily, 2 weeks for weekly
    context_window_days: Optional[int] = 14  # Optimal: 7-14 days per Probe 3.1.1
    prediction_horizon: int = 1
    use_reduced_features: bool = True
    imputation_strategy: Literal[
        'domain_specific', 'mean', 'median', 'forward_fill', 'zero'
    ] = 'domain_specific'

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        # Validate ratios sum to 1.0
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Data split ratios must sum to 1.0, got {total_ratio:.4f}"
            )

        # Validate positive values
        if self.temporal_gap_days < 0:
            raise ValueError("temporal_gap_days must be non-negative")
        if self.seq_len < 1:
            raise ValueError("seq_len must be at least 1")
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        if self.context_window_days is not None and self.context_window_days < 1:
            raise ValueError("context_window_days must be at least 1 when set")

    @property
    def effective_seq_len(self) -> int:
        """Calculate effective sequence length based on context_window_days and resolution.

        Returns the number of time steps based on the context window and temporal
        resolution. If context_window_days is set, it converts days to the
        appropriate number of time steps for the resolution.

        Returns:
            Number of time steps for input sequences.

        Example:
            >>> config = DataConfig(resolution='daily', context_window_days=14)
            >>> config.effective_seq_len
            14
            >>> config = DataConfig(resolution='weekly', context_window_days=14)
            >>> config.effective_seq_len
            2
        """
        if self.context_window_days is None:
            return self.seq_len

        resolution_to_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
        }
        days_per_step = resolution_to_days.get(self.resolution, 1)
        return max(1, self.context_window_days // days_per_step)


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture.

    Attributes:
        d_model: Dimension of model embeddings and hidden states.
        nhead: Number of attention heads (must divide d_model evenly).
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: Dimension of feedforward network in transformer.
        dropout: Dropout probability for regularization.
        use_state_transition: Enable state transition modeling component.
        use_multi_scale_temporal: Enable multi-scale temporal attention.
        use_delta_prediction: Predict changes rather than absolute values.
    """
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.2
    use_state_transition: bool = True
    use_multi_scale_temporal: bool = True
    use_delta_prediction: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.dim_feedforward < 1:
            raise ValueError("dim_feedforward must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training loop and optimization.

    Attributes:
        batch_size: Number of samples per gradient update step.
        accumulation_steps: Gradient accumulation steps (effective batch =
            batch_size * accumulation_steps).
        learning_rate: Initial learning rate for optimizer.
        weight_decay: L2 regularization coefficient.
        epochs: Maximum number of training epochs.
        warmup_epochs: Number of epochs for learning rate warmup.
        patience: Early stopping patience (epochs without improvement).
        min_lr: Minimum learning rate for scheduler.
        max_grad_norm: Maximum gradient norm for clipping.
        use_uncertainty_loss: Enable uncertainty-aware loss function.
        augmentation_prob: Probability of applying data augmentation.
        noise_std: Standard deviation of Gaussian noise augmentation.

        # Early stopping flexibility
        early_stopping_strategy: Strategy for early stopping
            ('standard', 'smoothed', 'relative', 'plateau', 'combined').
        early_stopping_min_epochs: Minimum epochs before early stopping can trigger.
        early_stopping_min_delta: Minimum improvement to reset patience counter.
        early_stopping_smoothing: EMA smoothing factor for 'smoothed' strategy.
        early_stopping_relative_threshold: Relative tolerance for 'relative' strategy.
        disable_early_stopping: Completely disable early stopping.

        # Stochastic Weight Averaging (SWA)
        use_swa: Enable Stochastic Weight Averaging.
        swa_start_pct: Start SWA at this percentage of total training.
        swa_freq: Update SWA model every N epochs.
        swa_lr_factor: LR multiplier during SWA phase (relative to min_lr).

        # Snapshot Ensembles
        use_snapshots: Enable snapshot ensemble collection.
        n_snapshots: Number of snapshots to keep.
        snapshot_at_restarts: Save snapshots at LR restart points.

        # Regularization
        use_label_smoothing: Enable label smoothing for classification.
        label_smoothing: Label smoothing factor (0.0 to 1.0).
        use_mixup: Enable mixup data augmentation.
        mixup_alpha: Mixup interpolation strength.
        use_cutmix: Enable cutmix data augmentation.
        cutmix_alpha: CutMix interpolation strength.

        # Learning rate scheduling
        lr_schedule: LR schedule type ('cosine', 'cosine_restarts', 'linear', 'constant').
        cosine_t0: Initial cycle length for cosine restarts.
        cosine_t_mult: Cycle length multiplier for cosine restarts.
    """
    batch_size: int = 4
    accumulation_steps: int = 8  # effective batch = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 200
    warmup_epochs: int = 10
    patience: int = 30
    min_lr: float = 1e-7
    max_grad_norm: float = 1.0
    use_uncertainty_loss: bool = True
    augmentation_prob: float = 0.5
    noise_std: float = 0.1

    # Early stopping flexibility
    early_stopping_strategy: Literal[
        'standard', 'smoothed', 'relative', 'plateau', 'combined'
    ] = 'smoothed'
    early_stopping_min_epochs: int = 50
    early_stopping_min_delta: float = 1e-4
    early_stopping_smoothing: float = 0.9
    early_stopping_relative_threshold: float = 0.1
    disable_early_stopping: bool = False

    # Stochastic Weight Averaging (SWA)
    use_swa: bool = True
    swa_start_pct: float = 0.75
    swa_freq: int = 5
    swa_lr_factor: float = 0.5

    # Snapshot Ensembles
    use_snapshots: bool = False
    n_snapshots: int = 5
    snapshot_at_restarts: bool = True

    # Regularization
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Learning rate scheduling
    lr_schedule: Literal['cosine', 'cosine_restarts', 'linear', 'constant'] = 'cosine'
    cosine_t0: int = 10
    cosine_t_mult: int = 2

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not 0.0 <= self.augmentation_prob <= 1.0:
            raise ValueError("augmentation_prob must be in [0, 1]")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

        # Validate early stopping parameters
        if self.early_stopping_min_epochs < 0:
            raise ValueError("early_stopping_min_epochs must be non-negative")
        if not 0.0 < self.early_stopping_smoothing < 1.0:
            raise ValueError("early_stopping_smoothing must be in (0, 1)")
        if self.early_stopping_relative_threshold < 0:
            raise ValueError("early_stopping_relative_threshold must be non-negative")

        # Validate SWA parameters
        if not 0.0 < self.swa_start_pct <= 1.0:
            raise ValueError("swa_start_pct must be in (0, 1]")
        if self.swa_freq < 1:
            raise ValueError("swa_freq must be at least 1")
        if self.swa_lr_factor <= 0:
            raise ValueError("swa_lr_factor must be positive")

        # Validate snapshot parameters
        if self.n_snapshots < 1:
            raise ValueError("n_snapshots must be at least 1")

        # Validate regularization parameters
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.mixup_alpha < 0:
            raise ValueError("mixup_alpha must be non-negative")
        if self.cutmix_alpha < 0:
            raise ValueError("cutmix_alpha must be non-negative")

        # Validate LR schedule parameters
        if self.cosine_t0 < 1:
            raise ValueError("cosine_t0 must be at least 1")
        if self.cosine_t_mult < 1:
            raise ValueError("cosine_t_mult must be at least 1")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.accumulation_steps

    @property
    def swa_start_epoch(self) -> int:
        """Calculate the epoch at which SWA should start."""
        return int(self.epochs * self.swa_start_pct)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configurations.

    This is the main configuration class that combines data, model, and training
    configurations along with experiment metadata.

    Attributes:
        data: Data processing configuration.
        model: Model architecture configuration.
        training: Training loop configuration.
        experiment_name: Name identifier for the experiment.
        seed: Random seed for reproducibility.
        device: Compute device ('auto', 'cuda', 'mps', 'cpu').

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name='my_experiment',
        ...     data=DataConfig(resolution='daily'),
        ...     training=TrainingConfig(epochs=100)
        ... )
        >>> config.save('config.json')
        >>> loaded = ExperimentConfig.load('config.json')
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = 'tactical_prediction'
    seed: int = 42
    device: Literal['auto', 'cuda', 'mps', 'cpu'] = 'auto'

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a nested dictionary.

        Returns:
            Dictionary representation of the complete configuration.
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from a dictionary.

        Args:
            d: Dictionary with configuration values.

        Returns:
            ExperimentConfig instance.
        """
        return cls(
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            experiment_name=d.get('experiment_name', 'tactical_prediction'),
            seed=d.get('seed', 42),
            device=d.get('device', 'auto'),
        )

    def save(self, path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            path: File path to save the configuration.
        """
        # Ensure directory exists
        dir_path = os.path.dirname(os.path.abspath(path))
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from a JSON file.

        Args:
            path: File path to load the configuration from.

        Returns:
            ExperimentConfig instance.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device string.

        Returns:
            Resolved device string ('cuda', 'mps', or 'cpu').
        """
        if self.device != 'auto':
            return self.device

        # Try to import torch and detect available devices
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'

    def __str__(self) -> str:
        """Return a formatted string representation of the configuration."""
        lines = [
            f"ExperimentConfig: {self.experiment_name}",
            f"  Seed: {self.seed}",
            f"  Device: {self.device}",
            "",
            "  Data Configuration:",
            f"    Resolution: {self.data.resolution}",
            f"    Temporal Gap: {self.data.temporal_gap_days} days",
            f"    Split: {self.data.train_ratio}/{self.data.val_ratio}/{self.data.test_ratio}",
            f"    Context Window: {self.data.context_window_days} days (effective seq_len: {self.data.effective_seq_len})",
            f"    Sequence Length (raw): {self.data.seq_len}",
            f"    Prediction Horizon: {self.data.prediction_horizon}",
            f"    Use Reduced Features: {self.data.use_reduced_features}",
            f"    Imputation Strategy: {self.data.imputation_strategy}",
            "",
            "  Model Configuration:",
            f"    d_model: {self.model.d_model}",
            f"    Attention Heads: {self.model.nhead}",
            f"    Encoder Layers: {self.model.num_encoder_layers}",
            f"    Decoder Layers: {self.model.num_decoder_layers}",
            f"    Feedforward Dim: {self.model.dim_feedforward}",
            f"    Dropout: {self.model.dropout}",
            f"    State Transition: {self.model.use_state_transition}",
            f"    Multi-Scale Temporal: {self.model.use_multi_scale_temporal}",
            f"    Delta Prediction: {self.model.use_delta_prediction}",
            "",
            "  Training Configuration:",
            f"    Batch Size: {self.training.batch_size} (effective: {self.training.effective_batch_size})",
            f"    Learning Rate: {self.training.learning_rate}",
            f"    LR Schedule: {self.training.lr_schedule}",
            f"    Weight Decay: {self.training.weight_decay}",
            f"    Epochs: {self.training.epochs}",
            f"    Warmup Epochs: {self.training.warmup_epochs}",
            f"    Min LR: {self.training.min_lr}",
            f"    Max Grad Norm: {self.training.max_grad_norm}",
            f"    Uncertainty Loss: {self.training.use_uncertainty_loss}",
            f"    Augmentation Prob: {self.training.augmentation_prob}",
            f"    Noise Std: {self.training.noise_std}",
            "",
            "  Early Stopping Configuration:",
            f"    Strategy: {self.training.early_stopping_strategy}",
            f"    Min Epochs: {self.training.early_stopping_min_epochs}",
            f"    Patience: {self.training.patience}",
            f"    Min Delta: {self.training.early_stopping_min_delta}",
            f"    Smoothing Factor: {self.training.early_stopping_smoothing}",
            f"    Relative Threshold: {self.training.early_stopping_relative_threshold}",
            f"    Disabled: {self.training.disable_early_stopping}",
            "",
            "  Regularization Configuration:",
            f"    SWA Enabled: {self.training.use_swa}",
            f"    SWA Start: {self.training.swa_start_pct:.0%} (epoch {self.training.swa_start_epoch})",
            f"    SWA Frequency: {self.training.swa_freq} epochs",
            f"    Snapshots Enabled: {self.training.use_snapshots}",
            f"    N Snapshots: {self.training.n_snapshots}",
            f"    Label Smoothing: {self.training.use_label_smoothing} ({self.training.label_smoothing})",
            f"    Mixup: {self.training.use_mixup} (alpha={self.training.mixup_alpha})",
            f"    CutMix: {self.training.use_cutmix} (alpha={self.training.cutmix_alpha})",
        ]
        return "\n".join(lines)


# Preset configurations for common use cases
PRESET_CONFIGS: Dict[str, ExperimentConfig] = {
    'default': ExperimentConfig(),

    'fast_debug': ExperimentConfig(
        experiment_name='fast_debug',
        training=TrainingConfig(
            epochs=10,
            batch_size=8,
            patience=5,
        ),
        data=DataConfig(resolution='monthly'),
    ),

    'production': ExperimentConfig(
        experiment_name='production',
        training=TrainingConfig(
            epochs=500,
            patience=50,
            learning_rate=1e-4,
            weight_decay=0.02,
        ),
        data=DataConfig(resolution='weekly'),
    ),

    'high_capacity': ExperimentConfig(
        experiment_name='high_capacity',
        model=ModelConfig(
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
        ),
        training=TrainingConfig(
            epochs=300,
            patience=40,
            batch_size=2,
            accumulation_steps=16,
        ),
    ),

    'quick_validation': ExperimentConfig(
        experiment_name='quick_validation',
        training=TrainingConfig(
            epochs=50,
            patience=10,
            batch_size=8,
        ),
        data=DataConfig(
            resolution='weekly',
            seq_len=8,
        ),
    ),

    'ablation_no_state_transition': ExperimentConfig(
        experiment_name='ablation_no_state_transition',
        model=ModelConfig(
            use_state_transition=False,
        ),
    ),

    'ablation_no_multi_scale': ExperimentConfig(
        experiment_name='ablation_no_multi_scale',
        model=ModelConfig(
            use_multi_scale_temporal=False,
        ),
    ),

    'ablation_no_delta': ExperimentConfig(
        experiment_name='ablation_no_delta',
        model=ModelConfig(
            use_delta_prediction=False,
        ),
    ),

    'long_training': ExperimentConfig(
        experiment_name='long_training',
        training=TrainingConfig(
            epochs=500,
            patience=50,
            early_stopping_strategy='smoothed',
            early_stopping_min_epochs=100,
            early_stopping_smoothing=0.95,
            early_stopping_relative_threshold=0.15,
            use_swa=True,
            swa_start_pct=0.6,
            swa_freq=3,
            use_label_smoothing=True,
            label_smoothing=0.1,
            use_mixup=True,
            mixup_alpha=0.2,
        ),
    ),

    'no_early_stop': ExperimentConfig(
        experiment_name='no_early_stop',
        training=TrainingConfig(
            epochs=300,
            disable_early_stopping=True,
            use_swa=True,
            swa_start_pct=0.7,
            use_snapshots=True,
            n_snapshots=5,
        ),
    ),

    'aggressive_regularization': ExperimentConfig(
        experiment_name='aggressive_regularization',
        model=ModelConfig(
            dropout=0.3,
        ),
        training=TrainingConfig(
            epochs=400,
            weight_decay=0.05,
            early_stopping_strategy='combined',
            early_stopping_min_epochs=80,
            use_swa=True,
            use_label_smoothing=True,
            label_smoothing=0.15,
            use_mixup=True,
            mixup_alpha=0.4,
            use_cutmix=True,
            cutmix_alpha=1.0,
        ),
    ),

    'cyclic_lr': ExperimentConfig(
        experiment_name='cyclic_lr',
        training=TrainingConfig(
            epochs=300,
            lr_schedule='cosine_restarts',
            cosine_t0=20,
            cosine_t_mult=2,
            early_stopping_strategy='plateau',
            early_stopping_min_epochs=60,
            use_snapshots=True,
            n_snapshots=5,
            snapshot_at_restarts=True,
        ),
    ),

    # Optimal context window configurations based on Probe 3.1.1 findings
    # Shorter context (7-14 days) yields better performance due to rapid
    # regime changes in conflict dynamics.
    'optimal_context': ExperimentConfig(
        experiment_name='optimal_context',
        data=DataConfig(
            resolution='daily',
            context_window_days=14,  # Optimal: 78.4% accuracy, F1 0.318
            seq_len=14,
        ),
        training=TrainingConfig(
            epochs=200,
            patience=30,
        ),
    ),

    'optimal_context_7day': ExperimentConfig(
        experiment_name='optimal_context_7day',
        data=DataConfig(
            resolution='daily',
            context_window_days=7,  # Best accuracy: 78.8%, F1 0.318
            seq_len=7,
        ),
        training=TrainingConfig(
            epochs=200,
            patience=30,
        ),
    ),

    'optimal_context_weekly': ExperimentConfig(
        experiment_name='optimal_context_weekly',
        data=DataConfig(
            resolution='weekly',
            context_window_days=14,  # 2 weeks at weekly resolution
            seq_len=2,
        ),
        training=TrainingConfig(
            epochs=200,
            patience=30,
        ),
    ),

    'optimal_context_production': ExperimentConfig(
        experiment_name='optimal_context_production',
        data=DataConfig(
            resolution='daily',
            context_window_days=14,
            seq_len=14,
        ),
        model=ModelConfig(
            d_model=128,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=512,
        ),
        training=TrainingConfig(
            epochs=500,
            patience=50,
            learning_rate=1e-4,
            weight_decay=0.02,
            use_swa=True,
            swa_start_pct=0.7,
        ),
    ),
}


def get_config(preset: str = 'default') -> ExperimentConfig:
    """Get a preset configuration by name.

    Args:
        preset: Name of the preset configuration. Available presets:
            - 'default': Standard configuration with balanced settings.
            - 'fast_debug': Quick iterations for debugging (10 epochs, monthly).
            - 'production': Full training for production (500 epochs).
            - 'high_capacity': Larger model architecture for complex patterns.
            - 'quick_validation': Fast validation runs (50 epochs).
            - 'ablation_no_state_transition': Ablation without state transition.
            - 'ablation_no_multi_scale': Ablation without multi-scale temporal.
            - 'ablation_no_delta': Ablation without delta prediction.
            - 'long_training': Extended training with SWA and mixup.
            - 'no_early_stop': Training without early stopping, uses SWA.
            - 'aggressive_regularization': Heavy regularization for overfitting.
            - 'cyclic_lr': Cosine annealing with warm restarts.
            - 'optimal_context': 14-day context window (Probe 3.1.1 optimal).
            - 'optimal_context_7day': 7-day context (best accuracy 78.8%).
            - 'optimal_context_weekly': 2-week context at weekly resolution.
            - 'optimal_context_production': Production config with optimal context.

    Returns:
        A copy of the requested ExperimentConfig preset.
        Falls back to 'default' if preset is not found.

    Example:
        >>> config = get_config('fast_debug')
        >>> config.training.epochs
        10
        >>> config = get_config('optimal_context')
        >>> config.data.context_window_days
        14
    """
    if preset not in PRESET_CONFIGS:
        import warnings
        warnings.warn(
            f"Unknown preset '{preset}', falling back to 'default'. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )

    # Return a fresh instance to prevent mutation of preset configs
    base_config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS['default'])
    return ExperimentConfig.from_dict(base_config.to_dict())


def list_presets() -> Dict[str, str]:
    """List all available preset configurations with descriptions.

    Returns:
        Dictionary mapping preset names to their experiment names.
    """
    return {name: config.experiment_name for name, config in PRESET_CONFIGS.items()}


if __name__ == '__main__':
    # Demonstrate configuration usage
    print("Available presets:", list_presets())
    print()

    # Show default configuration
    config = get_config('default')
    print(config)
    print()

    # Demonstrate save/load
    test_path = '/tmp/test_config.json'
    config.save(test_path)
    loaded = ExperimentConfig.load(test_path)
    print(f"Configuration saved and loaded successfully: {loaded.experiment_name}")
    print()

    # Show preset configurations
    print("=" * 60)
    print("Preset Configurations Summary:")
    print("=" * 60)
    for preset_name in PRESET_CONFIGS:
        preset = get_config(preset_name)
        print(f"\n{preset_name}:")
        print(f"  Epochs: {preset.training.epochs}")
        print(f"  Batch Size: {preset.training.batch_size}")
        print(f"  Resolution: {preset.data.resolution}")
        print(f"  d_model: {preset.model.d_model}")
