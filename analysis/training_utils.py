"""
Training utilities for ML_OSINT tactical prediction system.

This module provides optimized training components including:
- Learning rate scheduling with warmup
- Uncertainty-weighted multi-task loss
- Time-series specific data augmentation
- Temporal cross-validation splits
- Gradient accumulation for large batch training
- Flexible early stopping strategies
- Stochastic Weight Averaging (SWA)
- Snapshot ensembles
- Label smoothing and mixup regularization

Author: ML Engineering Team
"""

import copy
import math
from typing import List, Tuple, Iterator, Optional, Dict, Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL,
)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing decay.

    The scheduler performs linear warmup from warmup_start_lr to base_lr over
    warmup_epochs, then applies cosine annealing decay to min_lr over the
    remaining epochs.

    Args:
        optimizer: PyTorch optimizer instance.
        warmup_epochs: Number of epochs for linear warmup phase.
        total_epochs: Total number of training epochs.
        warmup_start_lr: Initial learning rate at the start of warmup. Default: 0.0.
        min_lr: Minimum learning rate at the end of cosine decay. Default: 1e-7.
        last_epoch: The index of last epoch. Default: -1.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer, warmup_epochs=10, total_epochs=100,
        ...     warmup_start_lr=1e-6, min_lr=1e-7
        ... )
        >>> for epoch in range(100):
        ...     train_one_epoch(model, optimizer)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 0.0,
        min_lr: float = 1e-7,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr

        # Validate parameters
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if total_epochs <= warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be > warmup_epochs ({warmup_epochs})"
            )
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current epoch.

        Returns:
            List of learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            # Clamp progress to [0, 1] to handle edge cases
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class UncertaintyWeightedLoss(nn.Module):
    """
    Learnable multi-task loss weighting based on task uncertainty.

    Implements the uncertainty weighting approach from Kendall et al. (2018)
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics". Each task's loss is weighted by a learnable parameter
    representing the homoscedastic uncertainty.

    The combined loss is computed as:
        L = sum_i (0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i)

    where log_var_i is a learnable parameter for task i, and L_i is the
    unweighted loss for task i.

    Args:
        task_names: List of task names/identifiers.
        init_log_var: Initial value for log variance parameters. Default: 0.0.

    Attributes:
        log_vars: nn.ParameterDict mapping task names to learnable log variances.

    Example:
        >>> loss_fn = UncertaintyWeightedLoss(['classification', 'regression'])
        >>> losses = {'classification': 0.5, 'regression': 1.2}
        >>> total_loss = loss_fn(losses)
        >>> total_loss.backward()  # Gradients flow to log_var parameters
    """

    def __init__(
        self,
        task_names: List[str],
        init_log_var: float = 0.0
    ) -> None:
        super().__init__()

        if not task_names:
            raise ValueError("task_names must be a non-empty list")

        self.task_names = list(task_names)
        self.init_log_var = init_log_var

        # Create learnable log variance parameters for each task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))
            for name in self.task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute uncertainty-weighted combined loss.

        Args:
            losses: Dictionary mapping task names to their unweighted losses.
                    Each loss should be a scalar tensor.

        Returns:
            Combined weighted loss as a scalar tensor.

        Raises:
            KeyError: If a required task name is not found in losses.
        """
        total_loss = torch.tensor(0.0, device=self._get_device())

        for task_name in self.task_names:
            if task_name not in losses:
                raise KeyError(
                    f"Task '{task_name}' not found in losses. "
                    f"Available tasks: {list(losses.keys())}"
                )

            task_loss = losses[task_name]
            log_var = self.log_vars[task_name]

            # Kendall et al. formulation:
            # L = 0.5 * exp(-log_var) * L_i + 0.5 * log_var
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
            total_loss = total_loss + weighted_loss

        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights for logging and monitoring.

        Returns:
            Dictionary mapping task names to their effective weights
            (computed as exp(-log_var)).
        """
        weights = {}
        with torch.no_grad():
            for task_name in self.task_names:
                log_var = self.log_vars[task_name]
                # Weight is the precision: exp(-log_var)
                weights[task_name] = torch.exp(-log_var).item()
        return weights

    def _get_device(self) -> torch.device:
        """Get device of the module parameters."""
        return next(self.parameters()).device


class TimeSeriesAugmentation:
    """
    Time-series specific data augmentation for tactical prediction.

    Applies a combination of augmentation techniques designed for temporal
    data while preserving the sequential nature and patterns:

    - Temporal shift: Random circular shift along time axis
    - Magnitude warping: Smooth random scaling using cubic spline
    - Jittering: Additive Gaussian noise
    - Feature dropout: Random zeroing of feature channels

    Args:
        shift_range: Maximum number of timesteps for temporal shift. Default: 2.
        warp_magnitude: Strength of magnitude warping (0-1). Default: 0.1.
        jitter_std: Standard deviation of Gaussian jitter noise. Default: 0.05.
        feature_dropout_prob: Probability of dropping each feature. Default: 0.1.
        augmentation_prob: Probability of applying augmentation. Default: 0.5.

    Example:
        >>> augmenter = TimeSeriesAugmentation(
        ...     shift_range=2, warp_magnitude=0.1,
        ...     jitter_std=0.05, augmentation_prob=0.5
        ... )
        >>> # features shape: (batch, seq_len, num_features)
        >>> augmented = augmenter(features, training=True)
    """

    def __init__(
        self,
        shift_range: int = 2,
        warp_magnitude: float = 0.1,
        jitter_std: float = 0.05,
        feature_dropout_prob: float = 0.1,
        augmentation_prob: float = 0.5
    ) -> None:
        self.shift_range = shift_range
        self.warp_magnitude = warp_magnitude
        self.jitter_std = jitter_std
        self.feature_dropout_prob = feature_dropout_prob
        self.augmentation_prob = augmentation_prob

        # Validate parameters
        if shift_range < 0:
            raise ValueError(f"shift_range must be >= 0, got {shift_range}")
        if not 0 <= warp_magnitude <= 1:
            raise ValueError(f"warp_magnitude must be in [0, 1], got {warp_magnitude}")
        if jitter_std < 0:
            raise ValueError(f"jitter_std must be >= 0, got {jitter_std}")
        if not 0 <= feature_dropout_prob <= 1:
            raise ValueError(
                f"feature_dropout_prob must be in [0, 1], got {feature_dropout_prob}"
            )
        if not 0 <= augmentation_prob <= 1:
            raise ValueError(
                f"augmentation_prob must be in [0, 1], got {augmentation_prob}"
            )

    def __call__(
        self,
        features: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Apply time-series augmentation to input features.

        Args:
            features: Input features with shape (batch, seq_len, num_features)
                      or (seq_len, num_features).
            training: If True, apply augmentation; if False, return unchanged.

        Returns:
            Augmented features with the same shape as input.
        """
        if not training:
            return features

        # Handle both batched and unbatched inputs
        squeeze_batch = False
        if features.ndim == 2:
            features = features[np.newaxis, ...]
            squeeze_batch = True

        augmented = features.copy()
        batch_size = augmented.shape[0]

        for i in range(batch_size):
            # Apply augmentation with specified probability
            if np.random.random() < self.augmentation_prob:
                sample = augmented[i]

                # Temporal shift (circular)
                if self.shift_range > 0:
                    shift = np.random.randint(-self.shift_range, self.shift_range + 1)
                    if shift != 0:
                        sample = np.roll(sample, shift, axis=0)

                # Magnitude warping
                if self.warp_magnitude > 0:
                    sample = self._magnitude_warp(sample)

                # Jittering (additive noise)
                if self.jitter_std > 0:
                    noise = np.random.normal(0, self.jitter_std, sample.shape)
                    sample = sample + noise

                # Feature dropout
                if self.feature_dropout_prob > 0:
                    sample = self._feature_dropout(sample)

                augmented[i] = sample

        if squeeze_batch:
            augmented = augmented.squeeze(0)

        return augmented.astype(features.dtype)

    def _magnitude_warp(self, x: np.ndarray) -> np.ndarray:
        """
        Apply smooth magnitude warping using cubic interpolation.

        Creates smooth random scaling factors along the time axis using
        cubic spline interpolation between a few random control points.

        Args:
            x: Input array with shape (seq_len, num_features).

        Returns:
            Warped array with same shape.
        """
        seq_len, num_features = x.shape

        # Generate random scaling factors at control points
        num_knots = 4
        knot_positions = np.linspace(0, seq_len - 1, num_knots)
        knot_values = 1.0 + np.random.uniform(
            -self.warp_magnitude, self.warp_magnitude, num_knots
        )

        # Interpolate to full sequence length
        all_positions = np.arange(seq_len)
        warp_factors = np.interp(all_positions, knot_positions, knot_values)

        # Apply warping (broadcast across features)
        warped = x * warp_factors[:, np.newaxis]

        return warped

    def _feature_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Apply feature-wise dropout (zero out random feature channels).

        Args:
            x: Input array with shape (seq_len, num_features).

        Returns:
            Array with some features zeroed out.
        """
        seq_len, num_features = x.shape

        # Generate dropout mask for features
        mask = np.random.random(num_features) > self.feature_dropout_prob

        # Ensure at least one feature remains
        if not mask.any():
            mask[np.random.randint(num_features)] = True

        # Apply mask (broadcast across time)
        dropped = x * mask[np.newaxis, :]

        return dropped


class TimeSeriesSplit:
    """
    Expanding window cross-validation splitter for time series data.

    Generates train/validation splits that respect temporal ordering:
    - Training sets expand over time (walk-forward validation)
    - Gap between train and validation prevents data leakage
    - Validation sets can be fixed or expanding size

    This is essential for temporal data where future information must not
    leak into training data.

    Args:
        n_samples: Total number of samples in the dataset.
        n_splits: Number of cross-validation folds. Default: 5.
        min_train_ratio: Minimum ratio of data for initial training set. Default: 0.5.
        gap: Number of samples to skip between train and validation. Default: 7.
        val_size: Fixed validation size. If None, expands proportionally. Default: None.

    Example:
        >>> splitter = TimeSeriesSplit(n_samples=1000, n_splits=5, gap=7)
        >>> for train_idx, val_idx in splitter.split():
        ...     X_train, y_train = X[train_idx], y[train_idx]
        ...     X_val, y_val = X[val_idx], y[val_idx]
        ...     model.fit(X_train, y_train)
        ...     score = model.evaluate(X_val, y_val)
    """

    def __init__(
        self,
        n_samples: int,
        n_splits: int = 5,
        min_train_ratio: float = 0.5,
        gap: int = 7,
        val_size: Optional[int] = None
    ) -> None:
        self.n_samples = n_samples
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.gap = gap
        self.val_size = val_size

        # Validate parameters
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")
        if n_splits <= 0:
            raise ValueError(f"n_splits must be > 0, got {n_splits}")
        if not 0 < min_train_ratio < 1:
            raise ValueError(
                f"min_train_ratio must be in (0, 1), got {min_train_ratio}"
            )
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")
        if val_size is not None and val_size <= 0:
            raise ValueError(f"val_size must be > 0 if specified, got {val_size}")

        # Compute minimum training size
        self.min_train_size = int(n_samples * min_train_ratio)

        # Validate that splits are feasible
        min_required = self.min_train_size + gap + (val_size or 1)
        if n_samples < min_required:
            raise ValueError(
                f"Not enough samples ({n_samples}) for specified parameters. "
                f"Need at least {min_required} samples."
            )

    def split(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation split indices.

        Yields:
            Tuples of (train_indices, val_indices) as numpy arrays.
        """
        indices = np.arange(self.n_samples)

        # Calculate available space for expanding beyond minimum train size
        available_for_expansion = self.n_samples - self.min_train_size - self.gap

        if self.val_size is not None:
            # Fixed validation size
            available_for_expansion -= self.val_size
            step_size = max(1, available_for_expansion // self.n_splits)

            for split_idx in range(self.n_splits):
                train_end = self.min_train_size + split_idx * step_size
                val_start = train_end + self.gap
                val_end = min(val_start + self.val_size, self.n_samples)

                # Ensure we don't exceed dataset bounds
                if val_end > self.n_samples:
                    break

                train_indices = indices[:train_end]
                val_indices = indices[val_start:val_end]

                if len(val_indices) > 0:
                    yield train_indices, val_indices
        else:
            # Proportional validation size (expanding window)
            for split_idx in range(self.n_splits):
                # Calculate split point
                split_ratio = (split_idx + 1) / (self.n_splits + 1)
                split_point = int(
                    self.min_train_size +
                    split_ratio * (self.n_samples - self.min_train_size - self.gap)
                )

                train_end = split_point
                val_start = train_end + self.gap

                # Calculate validation end for this split
                if split_idx < self.n_splits - 1:
                    next_split_ratio = (split_idx + 2) / (self.n_splits + 1)
                    next_split_point = int(
                        self.min_train_size +
                        next_split_ratio * (self.n_samples - self.min_train_size - self.gap)
                    )
                    val_end = min(next_split_point, self.n_samples)
                else:
                    val_end = self.n_samples

                train_indices = indices[:train_end]
                val_indices = indices[val_start:val_end]

                if len(val_indices) > 0:
                    yield train_indices, val_indices

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(n_samples={self.n_samples}, n_splits={self.n_splits}, "
            f"min_train_ratio={self.min_train_ratio}, gap={self.gap}, "
            f"val_size={self.val_size})"
        )


class GradientAccumulator:
    """
    Helper class for gradient accumulation to simulate larger batch sizes.

    Gradient accumulation allows training with effectively larger batch sizes
    than what fits in memory by accumulating gradients over multiple forward
    passes before performing an optimizer step.

    Args:
        optimizer: PyTorch optimizer instance.
        accumulation_steps: Number of steps to accumulate before updating. Default: 1.
        max_grad_norm: Maximum gradient norm for clipping. Default: 1.0.
                       Set to None to disable gradient clipping.

    Attributes:
        current_step: Current accumulation step counter.

    Example:
        >>> accumulator = GradientAccumulator(optimizer, accumulation_steps=4)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     scaled_loss = loss / accumulator.accumulation_steps
        ...     scaled_loss.backward()
        ...     if accumulator.step(loss):
        ...         print(f"Updated at step {accumulator.current_step}")
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0
    ) -> None:
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm

        if accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {accumulation_steps}"
            )
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0 if specified, got {max_grad_norm}"
            )

        self.current_step = 0
        self._accumulated_loss = 0.0

    def step(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients and perform optimizer step if ready.

        This method should be called after loss.backward(). It tracks
        accumulation progress and performs the optimizer step with
        gradient clipping when accumulation_steps batches have been processed.

        Args:
            loss: The loss tensor (for logging purposes, gradients should
                  already be computed via backward()).

        Returns:
            True if optimizer step was performed, False otherwise.
        """
        self.current_step += 1

        # Track accumulated loss for logging
        if isinstance(loss, torch.Tensor):
            self._accumulated_loss += loss.detach().item()
        else:
            self._accumulated_loss += float(loss)

        if self.should_step():
            # Apply gradient clipping if specified
            if self.max_grad_norm is not None:
                all_params = []
                for param_group in self.optimizer.param_groups:
                    all_params.extend(param_group['params'])
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

            # Perform optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Reset accumulated loss
            self._accumulated_loss = 0.0

            return True

        return False

    def zero_grad(self) -> None:
        """
        Zero gradients and reset accumulation counter.

        Call this at the start of training or when you need to discard
        accumulated gradients.
        """
        self.optimizer.zero_grad()
        self.current_step = 0
        self._accumulated_loss = 0.0

    def should_step(self) -> bool:
        """
        Check if optimizer step should be performed.

        Returns:
            True if current_step is a multiple of accumulation_steps.
        """
        return self.current_step % self.accumulation_steps == 0

    def get_accumulated_loss(self) -> float:
        """
        Get the accumulated loss since last optimizer step.

        Returns:
            Sum of losses accumulated since last step.
        """
        return self._accumulated_loss

    def get_effective_batch_size(self, batch_size: int) -> int:
        """
        Calculate effective batch size with accumulation.

        Args:
            batch_size: Actual batch size per forward pass.

        Returns:
            Effective batch size (batch_size * accumulation_steps).
        """
        return batch_size * self.accumulation_steps

    def __repr__(self) -> str:
        return (
            f"GradientAccumulator(accumulation_steps={self.accumulation_steps}, "
            f"max_grad_norm={self.max_grad_norm}, current_step={self.current_step})"
        )


class SmartEarlyStopping:
    """
    Flexible early stopping with multiple strategies to avoid premature stopping.

    This class provides several strategies to determine when to stop training,
    allowing for longer training while still preventing memorization:

    Strategies:
    - 'standard': Stop after patience epochs with no improvement (traditional behavior)
    - 'smoothed': Use exponential moving average of validation loss to reduce noise
    - 'relative': Only stop if val loss is X% worse than best (tolerates small regressions)
    - 'plateau': Detect true plateaus vs temporary increases using trend analysis
    - 'combined': Combines smoothed and relative strategies for maximum flexibility

    Args:
        patience: Number of epochs to wait for improvement before stopping. Default: 30.
        min_delta: Minimum change to qualify as an improvement. Default: 1e-4.
        min_epochs: Don't consider stopping until this many epochs. Default: 50.
        smoothing_factor: EMA smoothing factor for 'smoothed' strategy. Default: 0.9.
        relative_threshold: Relative tolerance for 'relative' strategy (0.1 = 10%). Default: 0.1.
        strategy: Early stopping strategy to use. Default: 'smoothed'.
        restore_best: Whether to restore best model weights when stopping. Default: True.
        verbose: Whether to print early stopping messages. Default: False.

    Example:
        >>> early_stop = SmartEarlyStopping(
        ...     patience=30, min_epochs=50, strategy='smoothed'
        ... )
        >>> for epoch in range(max_epochs):
        ...     val_loss = train_one_epoch(model)
        ...     if early_stop(epoch, val_loss, model):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
        >>> # Optionally restore best weights
        >>> if early_stop.best_state is not None:
        ...     model.load_state_dict(early_stop.best_state)
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-4,
        min_epochs: int = 50,
        smoothing_factor: float = 0.9,
        relative_threshold: float = 0.1,
        strategy: Literal['standard', 'smoothed', 'relative', 'plateau', 'combined'] = 'smoothed',
        restore_best: bool = True,
        verbose: bool = False
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if min_epochs < 0:
            raise ValueError(f"min_epochs must be >= 0, got {min_epochs}")
        if not 0.0 < smoothing_factor < 1.0:
            raise ValueError(f"smoothing_factor must be in (0, 1), got {smoothing_factor}")
        if relative_threshold < 0:
            raise ValueError(f"relative_threshold must be >= 0, got {relative_threshold}")

        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.smoothing_factor = smoothing_factor
        self.relative_threshold = relative_threshold
        self.strategy = strategy
        self.restore_best = restore_best
        self.verbose = verbose

        # State tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self.counter = 0
        self.stopped_epoch = 0

        # For smoothed strategy
        self.ema_loss: Optional[float] = None
        self.best_ema_loss = float('inf')

        # For plateau detection
        self.loss_history: List[float] = []
        self.plateau_window = 10  # Window for plateau detection

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        Check if training should stop based on the configured strategy.

        Args:
            epoch: Current epoch number (0-indexed).
            val_loss: Validation loss for the current epoch.
            model: Optional model to save best state from.

        Returns:
            True if training should stop, False otherwise.
        """
        self.loss_history.append(val_loss)

        # Don't stop before min_epochs
        if epoch < self.min_epochs:
            self._update_best(val_loss, epoch, model)
            return False

        # Update smoothed loss
        if self.ema_loss is None:
            self.ema_loss = val_loss
        else:
            self.ema_loss = (
                self.smoothing_factor * self.ema_loss +
                (1 - self.smoothing_factor) * val_loss
            )

        # Check for improvement based on strategy
        should_stop = self._check_strategy(epoch, val_loss)

        # Update best if improved
        self._update_best(val_loss, epoch, model)

        if should_stop:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(strategy: {self.strategy}, best: {self.best_loss:.6f})")

        return should_stop

    def _check_strategy(self, epoch: int, val_loss: float) -> bool:
        """Check if stopping criteria met based on selected strategy."""
        if self.strategy == 'standard':
            return self._check_standard(val_loss)
        elif self.strategy == 'smoothed':
            return self._check_smoothed()
        elif self.strategy == 'relative':
            return self._check_relative(val_loss)
        elif self.strategy == 'plateau':
            return self._check_plateau()
        elif self.strategy == 'combined':
            return self._check_combined(val_loss)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _check_standard(self, val_loss: float) -> bool:
        """Standard early stopping: stop after patience epochs without improvement."""
        if val_loss < self.best_loss - self.min_delta:
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def _check_smoothed(self) -> bool:
        """Smoothed strategy: use EMA of validation loss to reduce noise."""
        if self.ema_loss is None:
            return False

        if self.ema_loss < self.best_ema_loss - self.min_delta:
            self.best_ema_loss = self.ema_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def _check_relative(self, val_loss: float) -> bool:
        """Relative strategy: only stop if loss is X% worse than best."""
        if val_loss < self.best_loss - self.min_delta:
            self.counter = 0
        else:
            # Check if we're within relative threshold of best
            relative_diff = (val_loss - self.best_loss) / (abs(self.best_loss) + 1e-8)
            if relative_diff <= self.relative_threshold:
                # Within tolerance, don't increment counter
                pass
            else:
                self.counter += 1

        return self.counter >= self.patience

    def _check_plateau(self) -> bool:
        """Plateau detection: identify true plateaus vs temporary increases."""
        if len(self.loss_history) < self.plateau_window:
            return False

        recent = self.loss_history[-self.plateau_window:]

        # Check for improvement in recent window
        best_recent = min(recent)
        if best_recent < self.best_loss - self.min_delta:
            self.counter = 0
            return False

        # Compute trend (linear regression slope)
        x = np.arange(len(recent))
        y = np.array(recent)
        slope = np.polyfit(x, y, 1)[0]

        # If slope is positive or near zero (plateau), increment counter
        if slope >= -self.min_delta:
            self.counter += 1
        else:
            # Loss is still decreasing
            self.counter = max(0, self.counter - 1)

        return self.counter >= self.patience

    def _check_combined(self, val_loss: float) -> bool:
        """Combined strategy: requires both smoothed and relative criteria."""
        smoothed_stop = self._check_smoothed()

        # Also check relative
        relative_diff = (val_loss - self.best_loss) / (abs(self.best_loss) + 1e-8)
        within_tolerance = relative_diff <= self.relative_threshold

        # Only stop if smoothed indicates stop AND we're outside relative tolerance
        return smoothed_stop and not within_tolerance

    def _update_best(
        self,
        val_loss: float,
        epoch: int,
        model: Optional[nn.Module]
    ) -> None:
        """Update best loss and optionally save model state."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.restore_best and model is not None:
                self.best_state = copy.deepcopy(model.state_dict())

        # Also track best EMA loss
        if self.ema_loss is not None and self.ema_loss < self.best_ema_loss:
            self.best_ema_loss = self.ema_loss

    def reset(self) -> None:
        """Reset early stopping state for a new training run."""
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.stopped_epoch = 0
        self.ema_loss = None
        self.best_ema_loss = float('inf')
        self.loss_history = []

    def get_best_epoch(self) -> int:
        """Get the epoch with the best validation loss."""
        return self.best_epoch

    def __repr__(self) -> str:
        return (
            f"SmartEarlyStopping(patience={self.patience}, min_epochs={self.min_epochs}, "
            f"strategy='{self.strategy}', best_loss={self.best_loss:.6f}, "
            f"counter={self.counter})"
        )


class SWAWrapper:
    """
    Stochastic Weight Averaging (SWA) for improved generalization.

    SWA averages model weights over the last portion of training, leading to
    models that find flatter minima and generalize better. This is particularly
    useful for preventing overfitting during longer training runs.

    Reference: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima
    and Better Generalization"

    Args:
        model: The PyTorch model to apply SWA to.
        swa_start_epoch: Epoch to start collecting weights for averaging. Default: 75.
        swa_freq: Frequency (in epochs) to update SWA model. Default: 5.
        swa_lr: Optional learning rate to use during SWA phase. Default: None.

    Example:
        >>> model = MyModel()
        >>> swa = SWAWrapper(model, swa_start_epoch=75, swa_freq=5)
        >>> for epoch in range(150):
        ...     train_one_epoch(model)
        ...     swa.update(epoch)
        >>> # Get the averaged model for inference
        >>> averaged_model = swa.get_averaged_model()
        >>> # Update batch norm statistics
        >>> swa.update_bn(train_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start_epoch: int = 75,
        swa_freq: int = 5,
        swa_lr: Optional[float] = None
    ) -> None:
        if swa_start_epoch < 0:
            raise ValueError(f"swa_start_epoch must be >= 0, got {swa_start_epoch}")
        if swa_freq < 1:
            raise ValueError(f"swa_freq must be >= 1, got {swa_freq}")

        self.model = model
        self.swa_start = swa_start_epoch
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr

        # Create SWA model using PyTorch's built-in utilities
        self.swa_model = torch.optim.swa_utils.AveragedModel(model)
        self.n_averaged = 0
        self._swa_started = False

    def update(self, epoch: int) -> bool:
        """
        Update SWA model if appropriate for the current epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            True if SWA model was updated, False otherwise.
        """
        if epoch >= self.swa_start and epoch % self.swa_freq == 0:
            self.swa_model.update_parameters(self.model)
            self.n_averaged += 1
            self._swa_started = True
            return True
        return False

    def get_averaged_model(self) -> nn.Module:
        """
        Get the SWA-averaged model.

        Returns:
            The averaged model. Returns original model if SWA hasn't started.
        """
        if not self._swa_started:
            return self.model
        return self.swa_model

    def update_bn(
        self,
        loader: torch.utils.data.DataLoader,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Update batch normalization statistics for the SWA model.

        This should be called after training is complete, as SWA can invalidate
        the running statistics of batch norm layers.

        Args:
            loader: DataLoader with training data for BN statistics.
            device: Device to run update on. If None, uses model's device.
        """
        if not self._swa_started:
            return

        torch.optim.swa_utils.update_bn(loader, self.swa_model, device=device)

    def is_active(self, epoch: int) -> bool:
        """Check if SWA is active for the given epoch."""
        return epoch >= self.swa_start

    def get_swa_lr(self, base_lr: float) -> float:
        """Get the learning rate to use during SWA phase."""
        if self.swa_lr is not None:
            return self.swa_lr
        # Default: use a reduced learning rate
        return base_lr * 0.5

    def __repr__(self) -> str:
        return (
            f"SWAWrapper(swa_start={self.swa_start}, swa_freq={self.swa_freq}, "
            f"n_averaged={self.n_averaged})"
        )


class SnapshotEnsemble:
    """
    Snapshot Ensemble: save model snapshots at learning rate cycle minimums.

    This technique saves model checkpoints ("snapshots") at the minimum points
    of cyclic learning rate schedules. The final prediction averages across
    snapshots for improved robustness and uncertainty estimation.

    Reference: Huang et al. (2017) "Snapshot Ensembles: Train 1, get M for free"

    Args:
        n_snapshots: Maximum number of snapshots to keep. Default: 5.
        save_optimizer: Whether to also save optimizer state. Default: False.

    Example:
        >>> ensemble = SnapshotEnsemble(n_snapshots=5)
        >>> for epoch in range(100):
        ...     train_one_epoch(model, scheduler)
        ...     if is_lr_minimum(scheduler):
        ...         ensemble.save_snapshot(model, epoch)
        >>> # Ensemble prediction
        >>> predictions = ensemble.predict_ensemble(model, test_inputs)
    """

    def __init__(
        self,
        n_snapshots: int = 5,
        save_optimizer: bool = False
    ) -> None:
        if n_snapshots < 1:
            raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")

        self.n_snapshots = n_snapshots
        self.save_optimizer = save_optimizer
        self.snapshots: List[Dict[str, Any]] = []

    def save_snapshot(
        self,
        model: nn.Module,
        epoch: int,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save a model snapshot.

        Args:
            model: The model to snapshot.
            epoch: Current epoch number.
            optimizer: Optional optimizer to save state from.
            metrics: Optional dictionary of metrics at this snapshot.
        """
        snapshot = {
            'epoch': epoch,
            'state': copy.deepcopy(model.state_dict()),
            'metrics': metrics or {}
        }

        if self.save_optimizer and optimizer is not None:
            snapshot['optimizer_state'] = copy.deepcopy(optimizer.state_dict())

        self.snapshots.append(snapshot)

        # Keep only the most recent n_snapshots
        if len(self.snapshots) > self.n_snapshots:
            self.snapshots.pop(0)

    def predict_ensemble(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Make ensemble predictions by averaging across all snapshots.

        Args:
            model: Model architecture (weights will be loaded from snapshots).
            inputs: Input tensor for prediction.
            device: Device to run predictions on.

        Returns:
            Averaged predictions across all snapshots.
        """
        if not self.snapshots:
            # No snapshots, just use current model
            model.eval()
            with torch.no_grad():
                return model(inputs)

        if device is None:
            device = next(model.parameters()).device

        inputs = inputs.to(device)
        predictions = []

        # Save current state to restore later
        original_state = copy.deepcopy(model.state_dict())

        model.eval()
        for snapshot in self.snapshots:
            model.load_state_dict(snapshot['state'])
            model.to(device)
            with torch.no_grad():
                pred = model(inputs)
            predictions.append(pred)

        # Restore original state
        model.load_state_dict(original_state)

        return torch.stack(predictions).mean(dim=0)

    def predict_with_uncertainty(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make ensemble predictions with uncertainty estimation.

        Args:
            model: Model architecture (weights will be loaded from snapshots).
            inputs: Input tensor for prediction.
            device: Device to run predictions on.

        Returns:
            Tuple of (mean predictions, standard deviation across snapshots).
        """
        if not self.snapshots:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
            return pred, torch.zeros_like(pred)

        if device is None:
            device = next(model.parameters()).device

        inputs = inputs.to(device)
        predictions = []

        original_state = copy.deepcopy(model.state_dict())

        model.eval()
        for snapshot in self.snapshots:
            model.load_state_dict(snapshot['state'])
            model.to(device)
            with torch.no_grad():
                pred = model(inputs)
            predictions.append(pred)

        model.load_state_dict(original_state)

        stacked = torch.stack(predictions)
        return stacked.mean(dim=0), stacked.std(dim=0)

    def get_snapshot_epochs(self) -> List[int]:
        """Get the epochs at which snapshots were saved."""
        return [s['epoch'] for s in self.snapshots]

    def clear(self) -> None:
        """Clear all saved snapshots."""
        self.snapshots = []

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        epochs = self.get_snapshot_epochs()
        return f"SnapshotEnsemble(n_snapshots={self.n_snapshots}, saved={len(self)}, epochs={epochs})"


class LabelSmoothing(nn.Module):
    """
    Label smoothing loss for classification tasks.

    Label smoothing prevents the model from becoming overconfident by
    replacing hard one-hot targets with soft targets that have some
    probability mass on incorrect classes.

    This is a form of regularization that can improve generalization,
    especially when training for longer.

    Args:
        smoothing: Smoothing factor in [0, 1). Default: 0.1.
            - 0.0: No smoothing (equivalent to standard cross-entropy)
            - 0.1: 10% probability mass distributed to non-target classes
        reduction: Loss reduction method ('mean', 'sum', 'none'). Default: 'mean'.

    Example:
        >>> criterion = LabelSmoothing(smoothing=0.1)
        >>> logits = model(inputs)  # shape: (batch, num_classes)
        >>> targets = torch.tensor([0, 1, 2])  # class indices
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ) -> None:
        super().__init__()

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")

        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Predicted logits with shape (batch, num_classes).
            targets: Target class indices with shape (batch,).

        Returns:
            Label-smoothed loss value.
        """
        num_classes = logits.size(-1)

        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(
                dim=-1,
                index=targets.unsqueeze(-1),
                value=self.confidence
            )

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"LabelSmoothing(smoothing={self.smoothing}, reduction='{self.reduction}')"


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to batch of data.

    Mixup creates virtual training examples by taking convex combinations
    of pairs of examples and their labels. This is a powerful regularization
    technique that can significantly improve generalization.

    Reference: Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"

    Args:
        x: Input features with shape (batch, ...).
        y: Target values with shape (batch, ...).
        alpha: Mixup interpolation strength. Default: 0.2.
            - Higher alpha means more mixing (more regularization)
            - alpha=0 means no mixing
        device: Device to create tensors on. If None, uses x's device.

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam) where:
            - mixed_x: Mixed input features
            - y_a: Original targets
            - y_b: Shuffled targets
            - lam: Mixing coefficient

    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
        >>> outputs = model(mixed_x)
        >>> loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    """
    if device is None:
        device = x.device

    batch_size = x.size(0)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Random permutation for pairing
    index = torch.randperm(batch_size, device=device)

    # Mix the inputs
    mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y, y[index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute mixup loss as weighted combination of losses on original targets.

    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        pred: Model predictions.
        y_a: Original targets.
        y_b: Shuffled targets from mixup pairing.
        lam: Mixing coefficient.

    Returns:
        Mixup loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation for time series data.

    CutMix cuts and pastes patches between training examples, which can be
    more effective than mixup for certain types of data.

    Reference: Yun et al. (2019) "CutMix: Regularization Strategy to Train
    Strong Classifiers with Localizable Features"

    Args:
        x: Input features with shape (batch, seq_len, features) or (batch, features).
        y: Target values with shape (batch, ...).
        alpha: CutMix interpolation strength. Default: 1.0.
        device: Device to create tensors on. If None, uses x's device.

    Returns:
        Tuple of (mixed_x, y_a, y_b, lam) where:
            - mixed_x: CutMix'd input features
            - y_a: Original targets
            - y_b: Shuffled targets
            - lam: Area ratio (used for loss weighting)
    """
    if device is None:
        device = x.device

    batch_size = x.size(0)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Random permutation for pairing
    index = torch.randperm(batch_size, device=device)

    # Determine cut region
    if x.dim() == 3:
        # Time series: (batch, seq_len, features)
        seq_len = x.size(1)
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0

        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]

        # Recalculate lambda based on actual area
        lam = 1 - cut_len / seq_len
    else:
        # Flat features: use regular mixup
        mixed_x = lam * x + (1 - lam) * x[index]

    return mixed_x, y, y[index], lam


class CyclicLRWithRestarts(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warm restarts.

    This scheduler implements SGDR (Stochastic Gradient Descent with Warm Restarts),
    which periodically resets the learning rate to its initial value. This can help
    escape local minima and is useful for snapshot ensembles.

    Reference: Loshchilov & Hutter (2017) "SGDR: Stochastic Gradient Descent
    with Warm Restarts"

    Args:
        optimizer: PyTorch optimizer instance.
        T_0: Number of epochs for the first restart cycle. Default: 10.
        T_mult: Factor to increase cycle length after each restart. Default: 2.
        eta_min: Minimum learning rate. Default: 1e-7.
        last_epoch: The index of last epoch. Default: -1.

    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, T_0=10, T_mult=2)
        >>> for epoch in range(100):
        ...     train_one_epoch(model)
        ...     scheduler.step()
        ...     if scheduler.is_restart():
        ...         # Good time to save a snapshot
        ...         ensemble.save_snapshot(model, epoch)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-7,
        last_epoch: int = -1
    ) -> None:
        if T_0 < 1:
            raise ValueError(f"T_0 must be >= 1, got {T_0}")
        if T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {T_mult}")

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self._is_restart = False

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate for the next epoch."""
        if epoch is None:
            epoch = self.last_epoch + 1

        self._is_restart = False

        # Check for restart
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self._is_restart = True
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult

        super().step(epoch)

    def is_restart(self) -> bool:
        """Check if current epoch is a restart point."""
        return self._is_restart

    def is_at_minimum(self) -> bool:
        """Check if learning rate is at its minimum for current cycle."""
        return self.T_cur == self.T_i - 1


class RegularizationScheduler:
    """
    Schedule regularization strength during training.

    This class allows for dynamic adjustment of regularization parameters
    (dropout, weight decay, augmentation) during training. Can help prevent
    early overfitting while allowing finer learning in later epochs.

    Args:
        initial_dropout: Initial dropout probability. Default: 0.3.
        final_dropout: Final dropout probability. Default: 0.1.
        initial_aug_prob: Initial augmentation probability. Default: 0.7.
        final_aug_prob: Final augmentation probability. Default: 0.3.
        warmup_epochs: Epochs before starting to decrease regularization. Default: 20.
        total_epochs: Total training epochs for scheduling. Default: 200.

    Example:
        >>> reg_scheduler = RegularizationScheduler(
        ...     initial_dropout=0.3, final_dropout=0.1, total_epochs=200
        ... )
        >>> for epoch in range(200):
        ...     dropout = reg_scheduler.get_dropout(epoch)
        ...     aug_prob = reg_scheduler.get_aug_prob(epoch)
        ...     model.set_dropout(dropout)
        ...     train_one_epoch(model, aug_prob=aug_prob)
    """

    def __init__(
        self,
        initial_dropout: float = 0.3,
        final_dropout: float = 0.1,
        initial_aug_prob: float = 0.7,
        final_aug_prob: float = 0.3,
        warmup_epochs: int = 20,
        total_epochs: int = 200
    ) -> None:
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.initial_aug_prob = initial_aug_prob
        self.final_aug_prob = final_aug_prob
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_dropout(self, epoch: int) -> float:
        """Get dropout rate for the given epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_dropout

        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))

        # Cosine decay
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_dropout + (self.initial_dropout - self.final_dropout) * decay

    def get_aug_prob(self, epoch: int) -> float:
        """Get augmentation probability for the given epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_aug_prob

        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))

        # Linear decay
        return self.initial_aug_prob + (self.final_aug_prob - self.initial_aug_prob) * progress

    def __repr__(self) -> str:
        return (
            f"RegularizationScheduler(dropout={self.initial_dropout}->{self.final_dropout}, "
            f"aug_prob={self.initial_aug_prob}->{self.final_aug_prob})"
        )
