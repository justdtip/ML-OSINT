#!/usr/bin/env python3
"""
Temporal Dynamics Probes for Multi-Resolution HAN Model
========================================================

This module implements comprehensive probes for analyzing temporal dynamics
in the Multi-Resolution Hierarchical Attention Network (HAN) model.

Probe Categories:
-----------------

3.1 Context Window Effects
    - 3.1.1 ContextWindowProbe: Truncated context inference experiments
    - 3.1.2 AttentionDistanceProbe: Temporal attention pattern analysis
    - 3.1.3 PredictiveHorizonProbe: Multi-horizon prediction testing

3.2 State Transition Dynamics
    - 3.2.1 TransitionDynamicsProbe: Latent trajectory around regime transitions
    - 3.2.2 LatentVelocityProbe: Velocity-based transition prediction

Key Regime Transition Dates:
----------------------------
    - 2022-04-02: Kyiv withdrawal (Initial Invasion -> Stalemate)
    - 2022-09-01: Counteroffensive starts (Stalemate -> Counteroffensive)
    - 2022-12-01: Attritional warfare begins (Counteroffensive -> Attritional)

Model Specifications:
--------------------
    - daily_seq_len: 365 days
    - monthly_seq_len: 12 months
    - Regime classes: 4 (Initial Invasion, Stalemate, Counteroffensive, Attritional)

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from scipy.signal import find_peaks
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_resolution_han import (
    MultiResolutionHAN,
    MultiResolutionHANConfig,
    SourceConfig,
    DAILY_SOURCES,
    MONTHLY_SOURCES,
)
from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    MultiResolutionSample,
    multi_resolution_collate_fn,
)

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    MULTI_RES_CHECKPOINT_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# Task key mapping for consistent resolution between task names and output keys
from .task_key_mapping import (
    get_output_key,
    extract_task_output,
    has_task_output,
)

warnings.filterwarnings('ignore', category=UserWarning)


# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

BASE_DIR = PROJECT_ROOT
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR
CHECKPOINT_DIR = MULTI_RES_CHECKPOINT_DIR


def get_output_dir():
    """Get current output directory for figures."""
    return get_probe_figures_dir()


# Note: OUTPUT_DIR is now dynamic - use get_output_dir() instead

# Conflict start and regime transitions
CONFLICT_START = pd.Timestamp("2022-02-24")
REGIME_TRANSITIONS = {
    "kyiv_withdrawal": pd.Timestamp("2022-04-02"),
    "counteroffensive_start": pd.Timestamp("2022-09-01"),
    "attritional_warfare": pd.Timestamp("2022-12-01"),
}

# Phase definitions
PHASE_LABELS = {
    0: "Initial Invasion",
    1: "Stalemate",
    2: "Counteroffensive",
    3: "Attritional Warfare",
}

PHASE_COLORS = {
    0: "#d62728",  # Red
    1: "#7f7f7f",  # Grey
    2: "#2ca02c",  # Green
    3: "#ff7f0e",  # Orange
}

# Context lengths for truncated inference experiments
CONTEXT_LENGTHS = [7, 14, 30, 60, 90, None]  # None = full context (365 days)

# Prediction horizons for multi-horizon testing
PREDICTION_HORIZONS = [1, 3, 7, 14]  # days ahead

# Window size for transition boundary analysis
TRANSITION_WINDOW_DAYS = 14  # [-14, +14] around transition


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def date_to_phase(date: Union[str, pd.Timestamp, np.datetime64]) -> int:
    """
    Map a date to its conflict phase.

    Returns:
        0: Initial Invasion (Feb 24 - Apr 1, 2022)
        1: Stalemate (Apr 2 - Aug 31, 2022)
        2: Counteroffensive (Sep 1 - Nov 30, 2022)
        3: Attritional Warfare (Dec 1, 2022 - present)
    """
    if isinstance(date, (str, np.datetime64)):
        date = pd.to_datetime(date)

    if date < REGIME_TRANSITIONS["kyiv_withdrawal"]:
        return 0
    elif date < REGIME_TRANSITIONS["counteroffensive_start"]:
        return 1
    elif date < REGIME_TRANSITIONS["attritional_warfare"]:
        return 2
    else:
        return 3


def get_days_to_transition(date: pd.Timestamp) -> Dict[str, int]:
    """Calculate days to nearest transition (positive = future, negative = past)."""
    result = {}
    for name, trans_date in REGIME_TRANSITIONS.items():
        result[name] = (trans_date - date).days
    return result


def create_truncated_batch(
    batch: Dict[str, Any],
    context_length: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Create a truncated version of a batch with limited context.

    Args:
        batch: Original batch dictionary
        context_length: Number of days to keep (from most recent)
        device: Target device

    Returns:
        Truncated batch with masked features outside context window
    """
    truncated = {}

    # Get original sequence length
    sample_source = list(batch['daily_features'].keys())[0]
    full_seq_len = batch['daily_features'][sample_source].shape[1]

    # Calculate start index for truncation
    start_idx = max(0, full_seq_len - context_length)

    # Truncate daily features and masks
    truncated['daily_features'] = {}
    truncated['daily_masks'] = {}

    for source_name, features in batch['daily_features'].items():
        # Keep only the last context_length days
        truncated['daily_features'][source_name] = features[:, start_idx:, :].to(device)

        # For earlier positions, mask them as unobserved
        mask = batch['daily_masks'][source_name].clone()
        if start_idx > 0:
            mask[:, :start_idx, :] = False
        truncated['daily_masks'][source_name] = mask[:, start_idx:, :].to(device)

    # Monthly features remain the same
    truncated['monthly_features'] = {
        k: v.to(device) for k, v in batch['monthly_features'].items()
    }
    truncated['monthly_masks'] = {
        k: v.to(device) for k, v in batch['monthly_masks'].items()
    }

    # Adjust month boundaries for truncation
    if 'month_boundaries' in batch or 'month_boundary_indices' in batch:
        key = 'month_boundaries' if 'month_boundaries' in batch else 'month_boundary_indices'
        boundaries = batch[key].clone()
        # Shift boundaries by start_idx
        boundaries = boundaries - start_idx
        boundaries = torch.clamp(boundaries, min=0)
        truncated['month_boundaries'] = boundaries.to(device)

    return truncated


def extract_latent_representations(
    model: MultiResolutionHAN,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Extract latent representations from various model layers.

    Returns dict with:
        - 'daily_encoded': Per-source daily encoder outputs
        - 'daily_fused': Fused daily representation
        - 'monthly_aggregated': Daily-to-monthly aggregated representation
        - 'temporal_encoded': Final temporal encoder output
    """
    activations = {}
    hooks = []

    def get_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    # Register hooks on key modules
    if hasattr(model, 'daily_encoders'):
        for source_name, encoder in model.daily_encoders.items():
            hook = encoder.register_forward_hook(get_hook(f'daily_{source_name}'))
            hooks.append(hook)

    if hasattr(model, 'daily_fusion'):
        hook = model.daily_fusion.register_forward_hook(get_hook('daily_fused'))
        hooks.append(hook)

    if hasattr(model, 'monthly_aggregation'):
        hook = model.monthly_aggregation.register_forward_hook(get_hook('monthly_aggregated'))
        hooks.append(hook)

    if hasattr(model, 'temporal_encoder'):
        hook = model.temporal_encoder.register_forward_hook(get_hook('temporal_encoded'))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(
                daily_features=batch['daily_features'],
                daily_masks=batch['daily_masks'],
                monthly_features=batch['monthly_features'],
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch.get('month_boundaries', batch.get('month_boundary_indices')),
            )
    finally:
        for hook in hooks:
            hook.remove()

    return activations


def extract_attention_weights(
    model: MultiResolutionHAN,
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Extract attention weights from transformer layers.

    Returns dict mapping layer names to attention weight tensors.
    """
    attention_weights = {}
    hooks = []

    def get_attention_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                attn = output[1]
                if attn is not None:
                    attention_weights[name] = attn.detach()
        return hook

    # Enable attention weight output and register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            original_need_weights = getattr(module, 'need_weights', True)
            module.need_weights = True
            hook = module.register_forward_hook(get_attention_hook(name))
            hooks.append((hook, module, original_need_weights))

    try:
        with torch.no_grad():
            model(
                daily_features=batch['daily_features'],
                daily_masks=batch['daily_masks'],
                monthly_features=batch['monthly_features'],
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch.get('month_boundaries', batch.get('month_boundary_indices')),
            )
    finally:
        for hook, module, original in hooks:
            hook.remove()
            module.need_weights = original

    return attention_weights


# ==============================================================================
# BASE PROBE CLASS
# ==============================================================================

@dataclass
class ProbeResult:
    """Container for probe results."""
    name: str
    metrics: Dict[str, Any]
    figures: Dict[str, plt.Figure] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, output_dir: Path) -> None:
        """Save probe results to disk."""
        probe_dir = output_dir / self.name
        probe_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        with open(probe_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Save figures
        for fig_name, fig in self.figures.items():
            fig.savefig(probe_dir / f"{fig_name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        # Save raw data as numpy archives
        for data_name, data in self.raw_data.items():
            if isinstance(data, np.ndarray):
                np.save(probe_dir / f"{data_name}.npy", data)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(probe_dir / f"{data_name}.csv", index=False)

        print(f"  Saved results to {probe_dir}")


class BaseProbe(ABC):
    """Abstract base class for temporal dynamics probes."""

    def __init__(
        self,
        model: MultiResolutionHAN,
        dataset: MultiResolutionDataset,
        device: torch.device,
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.output_dir = output_dir or get_output_dir()
        self.model.eval()

    @property
    @abstractmethod
    def name(self) -> str:
        """Probe name for identification."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> ProbeResult:
        """Execute the probe and return results."""
        pass

    def prepare_batch(
        self,
        indices: List[int],
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Prepare a batch of samples with labels and dates.

        Returns:
            batch: Dict of tensors for model input
            labels: Phase labels per sample per month
            dates: Monthly dates per sample
        """
        daily_features = defaultdict(list)
        daily_masks = defaultdict(list)
        monthly_features = defaultdict(list)
        monthly_masks = defaultdict(list)
        month_boundaries = []
        all_labels = []
        all_dates = []

        for idx in indices:
            sample = self.dataset[idx]

            for source_name, features in sample.daily_features.items():
                daily_features[source_name].append(features)
                daily_masks[source_name].append(sample.daily_masks[source_name])

            for source_name, features in sample.monthly_features.items():
                monthly_features[source_name].append(features)
                monthly_masks[source_name].append(sample.monthly_masks[source_name])

            month_boundaries.append(sample.month_boundary_indices)

            # Get phase labels for each month
            if hasattr(sample, 'monthly_dates') and len(sample.monthly_dates) > 0:
                sample_labels = [date_to_phase(d) for d in sample.monthly_dates]
                sample_dates = sample.monthly_dates
            else:
                sample_labels = [0] * 12
                sample_dates = np.array([])

            all_labels.append(sample_labels)
            all_dates.append(sample_dates)

        batch = {
            'daily_features': {
                k: torch.stack(v).to(self.device) for k, v in daily_features.items()
            },
            'daily_masks': {
                k: torch.stack(v).to(self.device) for k, v in daily_masks.items()
            },
            'monthly_features': {
                k: torch.stack(v).to(self.device) for k, v in monthly_features.items()
            },
            'monthly_masks': {
                k: torch.stack(v).to(self.device) for k, v in monthly_masks.items()
            },
            'month_boundaries': torch.stack(month_boundaries).to(self.device),
        }

        labels = np.array(all_labels)
        dates = np.array(all_dates, dtype=object)

        return batch, labels, dates


# ==============================================================================
# 3.1.1 CONTEXT WINDOW PROBE
# ==============================================================================

class ContextWindowProbe(BaseProbe):
    """
    Probe 3.1.1: Truncated Context Inference

    Tests model performance with varying context lengths [7, 14, 30, 60, 90, full].
    Measures:
        - Regime classification accuracy and F1 score
        - Latent representation variance
        - Cross-context correlations
    """

    @property
    def name(self) -> str:
        return "context_window_probe"

    def run(
        self,
        num_samples: int = 100,
        batch_size: int = 8,
        context_lengths: Optional[List[Optional[int]]] = None,
    ) -> ProbeResult:
        """
        Run truncated context inference experiments.

        Args:
            num_samples: Number of samples to evaluate
            batch_size: Batch size for inference
            context_lengths: List of context lengths to test (None = full)
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        context_lengths = context_lengths or CONTEXT_LENGTHS

        # Sample indices
        n_available = len(self.dataset)
        sample_indices = np.random.choice(
            n_available, min(num_samples, n_available), replace=False
        )

        results_by_context = {}
        latent_by_context = {}

        for ctx_len in context_lengths:
            ctx_name = str(ctx_len) if ctx_len else "full"
            print(f"\n  Testing context length: {ctx_name}")

            all_predictions = []
            all_labels = []
            all_latents = []

            # Process in batches
            for batch_start in range(0, len(sample_indices), batch_size):
                batch_indices = sample_indices[batch_start:batch_start + batch_size].tolist()
                batch, labels, _ = self.prepare_batch(batch_indices)

                # Apply truncation if not full context
                if ctx_len is not None:
                    batch = create_truncated_batch(batch, ctx_len, self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        daily_features=batch['daily_features'],
                        daily_masks=batch['daily_masks'],
                        monthly_features=batch['monthly_features'],
                        monthly_masks=batch['monthly_masks'],
                        month_boundaries=batch['month_boundaries'],
                    )

                # Extract predictions and latents using centralized task key mapping
                regime_output = extract_task_output(outputs, 'regime')
                if regime_output is not None:
                    preds = regime_output.argmax(dim=-1).cpu().numpy()
                    all_predictions.extend(preds.flatten().tolist())
                    all_labels.extend(labels.flatten().tolist())

                if 'temporal_hidden' in outputs:
                    latents = outputs['temporal_hidden'].cpu().numpy()
                    all_latents.append(latents)

            # Compute metrics
            if all_predictions:
                accuracy = accuracy_score(all_labels, all_predictions)
                f1 = f1_score(all_labels, all_predictions, average='macro')

                results_by_context[ctx_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'n_samples': len(all_predictions),
                }
                print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            if all_latents:
                all_latents = np.concatenate(all_latents, axis=0)
                latent_variance = np.var(all_latents, axis=(0, 1)).mean()
                results_by_context[ctx_name]['latent_variance'] = float(latent_variance)
                latent_by_context[ctx_name] = all_latents

        # Compute cross-context correlations
        correlations = self._compute_cross_correlations(latent_by_context)

        # Generate visualizations
        figures = {}
        figures['performance_by_context'] = self._plot_performance_curves(results_by_context)
        figures['latent_variance'] = self._plot_latent_variance(results_by_context)
        if correlations:
            figures['cross_correlations'] = self._plot_correlations(correlations)

        return ProbeResult(
            name=self.name,
            metrics={
                'results_by_context': results_by_context,
                'cross_correlations': correlations,
            },
            figures=figures,
            raw_data={'latent_representations': latent_by_context.get('full', np.array([]))},
        )

    def _compute_cross_correlations(
        self,
        latent_by_context: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute correlations between latent representations at different context lengths."""
        if 'full' not in latent_by_context:
            return {}

        full_latents = latent_by_context['full'].reshape(latent_by_context['full'].shape[0], -1)
        correlations = {}

        for ctx_name, latents in latent_by_context.items():
            if ctx_name == 'full':
                continue

            truncated_latents = latents.reshape(latents.shape[0], -1)

            # Ensure same number of samples
            min_samples = min(len(full_latents), len(truncated_latents))
            if min_samples > 0:
                corr = np.corrcoef(
                    full_latents[:min_samples].flatten(),
                    truncated_latents[:min_samples].flatten()
                )[0, 1]
                correlations[ctx_name] = float(corr)

        return correlations

    def _plot_performance_curves(
        self,
        results: Dict[str, Dict],
    ) -> plt.Figure:
        """Plot performance metrics by context length."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        contexts = []
        accuracies = []
        f1_scores = []

        for ctx_name, metrics in results.items():
            if ctx_name == 'full':
                contexts.append(365)
            else:
                contexts.append(int(ctx_name))
            accuracies.append(metrics.get('accuracy', 0))
            f1_scores.append(metrics.get('f1_score', 0))

        # Sort by context length
        sorted_indices = np.argsort(contexts)
        contexts = [contexts[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]

        # Accuracy plot
        axes[0].plot(contexts, accuracies, 'o-', linewidth=2, markersize=8, color='#2ca02c')
        axes[0].set_xlabel('Context Length (days)', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Regime Classification Accuracy by Context Length', fontsize=14)
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)

        # F1 score plot
        axes[1].plot(contexts, f1_scores, 's-', linewidth=2, markersize=8, color='#d62728')
        axes[1].set_xlabel('Context Length (days)', fontsize=12)
        axes[1].set_ylabel('F1 Score (Macro)', fontsize=12)
        axes[1].set_title('F1 Score by Context Length', fontsize=14)
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_latent_variance(
        self,
        results: Dict[str, Dict],
    ) -> plt.Figure:
        """Plot latent representation variance by context length."""
        fig, ax = plt.subplots(figsize=(8, 5))

        contexts = []
        variances = []

        for ctx_name, metrics in results.items():
            if 'latent_variance' in metrics:
                if ctx_name == 'full':
                    contexts.append(365)
                else:
                    contexts.append(int(ctx_name))
                variances.append(metrics['latent_variance'])

        if contexts:
            sorted_indices = np.argsort(contexts)
            contexts = [contexts[i] for i in sorted_indices]
            variances = [variances[i] for i in sorted_indices]

            ax.bar(range(len(contexts)), variances, color='#1f77b4', alpha=0.7)
            ax.set_xticks(range(len(contexts)))
            ax.set_xticklabels([str(c) if c < 365 else 'full' for c in contexts])
            ax.set_xlabel('Context Length (days)', fontsize=12)
            ax.set_ylabel('Mean Latent Variance', fontsize=12)
            ax.set_title('Latent Representation Variance by Context Length', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_correlations(
        self,
        correlations: Dict[str, float],
    ) -> plt.Figure:
        """Plot cross-context correlations."""
        fig, ax = plt.subplots(figsize=(8, 5))

        contexts = []
        corr_values = []

        for ctx_name, corr in sorted(correlations.items(), key=lambda x: int(x[0])):
            contexts.append(int(ctx_name))
            corr_values.append(corr)

        if contexts:
            ax.plot(contexts, corr_values, 'o-', linewidth=2, markersize=10, color='#9467bd')
            ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 threshold')
            ax.set_xlabel('Context Length (days)', fontsize=12)
            ax.set_ylabel('Correlation with Full Context', fontsize=12)
            ax.set_title('Latent Representation Correlation with Full Context', fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        return fig


# ==============================================================================
# 3.1.2 ATTENTION DISTANCE PROBE
# ==============================================================================

class AttentionDistanceProbe(BaseProbe):
    """
    Probe 3.1.2: Temporal Attention Patterns

    Analyzes attention distance distributions:
        - Mean attention distance per head
        - Peak identification (important temporal positions)
        - Comparison across conflict phases
    """

    @property
    def name(self) -> str:
        return "attention_distance_probe"

    def run(
        self,
        num_samples: int = 50,
        batch_size: int = 4,
    ) -> ProbeResult:
        """
        Analyze temporal attention patterns.

        Args:
            num_samples: Number of samples to analyze
            batch_size: Batch size for inference
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        n_available = len(self.dataset)
        sample_indices = np.random.choice(
            n_available, min(num_samples, n_available), replace=False
        )

        all_attention_by_phase = defaultdict(list)
        all_distances_by_head = defaultdict(list)
        all_peak_positions = defaultdict(list)

        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start:batch_start + batch_size].tolist()
            batch, labels, dates = self.prepare_batch(batch_indices)

            # Extract attention weights
            attention_weights = extract_attention_weights(self.model, batch, self.device)

            # Analyze each attention layer
            for layer_name, attn in attention_weights.items():
                # attn shape: [batch, heads, seq_q, seq_k]
                if attn.dim() != 4:
                    continue

                batch_size_actual, n_heads, seq_q, seq_k = attn.shape

                # Compute attention distances
                for head_idx in range(n_heads):
                    head_attn = attn[:, head_idx, :, :].cpu().numpy()

                    # Compute mean attention distance for each query position
                    for b in range(batch_size_actual):
                        for q in range(seq_q):
                            attn_dist = head_attn[b, q, :]

                            # Mean attention distance
                            positions = np.arange(seq_k)
                            mean_dist = np.sum(attn_dist * np.abs(positions - q))
                            all_distances_by_head[f"{layer_name}_head{head_idx}"].append(mean_dist)

                            # Find peaks in attention
                            peaks, _ = find_peaks(attn_dist, height=0.1)
                            for peak in peaks:
                                all_peak_positions[f"{layer_name}_head{head_idx}"].append(peak - q)

                        # Store by phase
                        if b < len(labels):
                            phase = int(labels[b].flatten()[0]) if labels[b].size > 0 else 0
                            all_attention_by_phase[phase].append(head_attn[b].flatten())

        # Compute summary statistics
        distance_stats = {}
        for head_name, distances in all_distances_by_head.items():
            distance_stats[head_name] = {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'median': float(np.median(distances)),
            }

        peak_stats = {}
        for head_name, peaks in all_peak_positions.items():
            if peaks:
                peak_stats[head_name] = {
                    'mean_offset': float(np.mean(peaks)),
                    'std_offset': float(np.std(peaks)),
                    'n_peaks': len(peaks),
                }

        # Phase comparison statistics
        phase_comparison = {}
        for phase, attn_list in all_attention_by_phase.items():
            if attn_list:
                attn_array = np.concatenate(attn_list)
                phase_comparison[PHASE_LABELS[phase]] = {
                    'mean_attention': float(np.mean(attn_array)),
                    'entropy': float(stats.entropy(np.abs(attn_array) + 1e-10)),
                }

        # Generate visualizations
        figures = {}
        figures['attention_distance_histogram'] = self._plot_distance_histogram(all_distances_by_head)
        figures['peak_distribution'] = self._plot_peak_distribution(all_peak_positions)
        figures['phase_comparison'] = self._plot_phase_comparison(all_attention_by_phase)

        return ProbeResult(
            name=self.name,
            metrics={
                'distance_stats': distance_stats,
                'peak_stats': peak_stats,
                'phase_comparison': phase_comparison,
            },
            figures=figures,
            raw_data={},
        )

    def _plot_distance_histogram(
        self,
        distances_by_head: Dict[str, List[float]],
    ) -> plt.Figure:
        """Plot histogram of attention distances."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Aggregate distances across heads
        all_distances = []
        for distances in distances_by_head.values():
            all_distances.extend(distances)

        if all_distances:
            ax.hist(all_distances, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
            ax.axvline(np.mean(all_distances), color='red', linestyle='--',
                      label=f'Mean: {np.mean(all_distances):.2f}')
            ax.axvline(np.median(all_distances), color='green', linestyle='--',
                      label=f'Median: {np.median(all_distances):.2f}')
            ax.set_xlabel('Mean Attention Distance (positions)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Mean Attention Distance', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_peak_distribution(
        self,
        peak_positions: Dict[str, List[int]],
    ) -> plt.Figure:
        """Plot distribution of attention peak positions."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Aggregate peaks across heads
        all_peaks = []
        for peaks in peak_positions.values():
            all_peaks.extend(peaks)

        if all_peaks:
            ax.hist(all_peaks, bins=np.arange(-100, 101, 5), alpha=0.7,
                   color='#2ca02c', edgecolor='black')
            ax.axvline(0, color='red', linestyle='-', linewidth=2, label='Current position')
            ax.set_xlabel('Peak Offset from Query Position', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Attention Peak Positions', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_phase_comparison(
        self,
        attention_by_phase: Dict[int, List[np.ndarray]],
    ) -> plt.Figure:
        """Plot attention pattern comparison across phases."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for phase_idx, (phase, attn_list) in enumerate(sorted(attention_by_phase.items())):
            if phase_idx >= 4 or not attn_list:
                continue

            ax = axes[phase_idx]
            all_attn = np.concatenate(attn_list)

            ax.hist(all_attn, bins=50, alpha=0.7, color=PHASE_COLORS[phase],
                   edgecolor='black', density=True)
            ax.set_xlabel('Attention Weight', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{PHASE_LABELS[phase]}', fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Attention Weight Distributions by Conflict Phase', fontsize=14)
        plt.tight_layout()
        return fig


# ==============================================================================
# 3.1.3 PREDICTIVE HORIZON PROBE
# ==============================================================================

class PredictiveHorizonProbe(BaseProbe):
    """
    Probe 3.1.3: Predictive Horizon Analysis

    Tests model performance at different prediction horizons:
        - t+1, t+3, t+7, t+14 days ahead
        - Measures accuracy degradation with horizon
    """

    @property
    def name(self) -> str:
        return "predictive_horizon_probe"

    def run(
        self,
        num_samples: int = 100,
        batch_size: int = 8,
        horizons: Optional[List[int]] = None,
    ) -> ProbeResult:
        """
        Test prediction performance at multiple horizons.

        Args:
            num_samples: Number of samples to evaluate
            batch_size: Batch size for inference
            horizons: List of prediction horizons in days
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        horizons = horizons or PREDICTION_HORIZONS

        n_available = len(self.dataset)
        sample_indices = np.random.choice(
            n_available, min(num_samples, n_available), replace=False
        )

        results_by_horizon = {}

        for horizon in horizons:
            print(f"\n  Testing horizon: t+{horizon} days")

            all_predictions = []
            all_true_labels = []

            for batch_start in range(0, len(sample_indices), batch_size):
                batch_indices = sample_indices[batch_start:batch_start + batch_size].tolist()
                batch, labels, dates = self.prepare_batch(batch_indices)

                # For horizon testing, we need to adjust the target labels
                # Shift labels forward by the horizon (in monthly terms)
                # Since horizons are in days, convert to approximate months
                horizon_months = max(1, horizon // 30)

                # Truncate context to simulate predicting further ahead
                truncated_batch = self._create_horizon_adjusted_batch(
                    batch, horizon, self.device
                )

                with torch.no_grad():
                    outputs = self.model(
                        daily_features=truncated_batch['daily_features'],
                        daily_masks=truncated_batch['daily_masks'],
                        monthly_features=truncated_batch['monthly_features'],
                        monthly_masks=truncated_batch['monthly_masks'],
                        month_boundaries=truncated_batch['month_boundaries'],
                    )

                # Use centralized task key mapping
                regime_output = extract_task_output(outputs, 'regime')
                if regime_output is not None:
                    preds = regime_output.argmax(dim=-1).cpu().numpy()
                    all_predictions.extend(preds.flatten().tolist())
                    all_true_labels.extend(labels.flatten().tolist())

            if all_predictions:
                accuracy = accuracy_score(all_true_labels, all_predictions)
                f1 = f1_score(all_true_labels, all_predictions, average='macro')
                conf_matrix = confusion_matrix(all_true_labels, all_predictions)

                results_by_horizon[horizon] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'confusion_matrix': conf_matrix.tolist(),
                    'n_samples': len(all_predictions),
                }
                print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Generate visualizations
        figures = {}
        figures['horizon_performance'] = self._plot_horizon_performance(results_by_horizon)
        figures['horizon_confusion'] = self._plot_horizon_confusion(results_by_horizon)

        return ProbeResult(
            name=self.name,
            metrics={'results_by_horizon': results_by_horizon},
            figures=figures,
            raw_data={},
        )

    def _create_horizon_adjusted_batch(
        self,
        batch: Dict[str, Any],
        horizon: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Create batch adjusted for prediction horizon testing."""
        # For horizon testing, we remove the last `horizon` days from context
        # This simulates predicting further into the future

        sample_source = list(batch['daily_features'].keys())[0]
        full_seq_len = batch['daily_features'][sample_source].shape[1]

        # Calculate effective sequence length
        effective_len = max(30, full_seq_len - horizon)

        adjusted = {}
        adjusted['daily_features'] = {
            k: v[:, :effective_len, :].to(device)
            for k, v in batch['daily_features'].items()
        }
        adjusted['daily_masks'] = {
            k: v[:, :effective_len, :].to(device)
            for k, v in batch['daily_masks'].items()
        }
        adjusted['monthly_features'] = {
            k: v.to(device) for k, v in batch['monthly_features'].items()
        }
        adjusted['monthly_masks'] = {
            k: v.to(device) for k, v in batch['monthly_masks'].items()
        }

        # Adjust month boundaries
        boundaries = batch['month_boundaries'].clone()
        boundaries = torch.clamp(boundaries, max=effective_len)
        adjusted['month_boundaries'] = boundaries.to(device)

        return adjusted

    def _plot_horizon_performance(
        self,
        results: Dict[int, Dict],
    ) -> plt.Figure:
        """Plot performance degradation by prediction horizon."""
        fig, ax = plt.subplots(figsize=(10, 6))

        horizons = sorted(results.keys())
        accuracies = [results[h]['accuracy'] for h in horizons]
        f1_scores = [results[h]['f1_score'] for h in horizons]

        x = np.arange(len(horizons))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('Prediction Horizon (days)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Prediction Performance by Horizon', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f't+{h}' for h in horizons])
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _plot_horizon_confusion(
        self,
        results: Dict[int, Dict],
    ) -> plt.Figure:
        """Plot confusion matrices for each horizon."""
        horizons = sorted(results.keys())
        n_horizons = len(horizons)

        fig, axes = plt.subplots(1, n_horizons, figsize=(4 * n_horizons, 4))
        if n_horizons == 1:
            axes = [axes]

        for ax, horizon in zip(axes, horizons):
            conf_matrix = np.array(results[horizon]['confusion_matrix'])

            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[PHASE_LABELS[i] for i in range(4)],
                       yticklabels=[PHASE_LABELS[i] for i in range(4)])
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.set_title(f'Horizon t+{horizon}', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

        plt.suptitle('Confusion Matrices by Prediction Horizon', fontsize=14)
        plt.tight_layout()
        return fig


# ==============================================================================
# 3.2.1 TRANSITION DYNAMICS PROBE
# ==============================================================================

class TransitionDynamicsProbe(BaseProbe):
    """
    Probe 3.2.1: Transition Boundary Analysis

    Analyzes latent trajectories around regime transitions:
        - Extract latent states [-14, +14] days around transitions
        - Compute trajectory velocity
        - Identify pre-transition signatures
    """

    @property
    def name(self) -> str:
        return "transition_dynamics_probe"

    def run(
        self,
        window_days: int = 14,
        batch_size: int = 4,
    ) -> ProbeResult:
        """
        Analyze latent dynamics around regime transitions.

        Args:
            window_days: Days before and after transition to analyze
            batch_size: Batch size for inference
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        trajectories_by_transition = {}
        velocities_by_transition = {}

        for trans_name, trans_date in REGIME_TRANSITIONS.items():
            print(f"\n  Analyzing transition: {trans_name} ({trans_date.date()})")

            # Find samples that span this transition
            transition_samples = self._find_transition_samples(trans_date, window_days)

            if not transition_samples:
                print(f"    No samples found spanning this transition")
                continue

            print(f"    Found {len(transition_samples)} samples")

            # Extract latent trajectories
            trajectories, daily_dates = self._extract_transition_trajectories(
                transition_samples, trans_date, window_days, batch_size
            )

            if len(trajectories) > 0:
                trajectories_by_transition[trans_name] = {
                    'trajectories': trajectories,
                    'dates': daily_dates,
                    'transition_date': trans_date.isoformat(),
                }

                # Compute velocities
                velocities = self._compute_trajectory_velocities(trajectories)
                velocities_by_transition[trans_name] = {
                    'velocities': velocities,
                    'mean_velocity': float(np.mean(velocities)),
                    'peak_velocity_day': int(np.argmax(np.mean(velocities, axis=0)) - window_days),
                }
                print(f"    Mean velocity: {np.mean(velocities):.4f}")
                print(f"    Peak velocity at day: {velocities_by_transition[trans_name]['peak_velocity_day']}")

        # Guard: Handle empty trajectories
        if not trajectories_by_transition:
            print("  Warning: No trajectories extracted - no samples span the transition windows")
            return ProbeResult(
                name=self.name,
                metrics={
                    'transitions_analyzed': [],
                    'velocity_summary': {},
                    'error': 'No samples found spanning any transition window. '
                             'Dataset may not cover the required date ranges.',
                },
                figures={},
                raw_data={'trajectories': {}},
            )

        # Generate visualizations
        figures = {}
        figures['latent_trajectories'] = self._plot_latent_trajectories(
            trajectories_by_transition, window_days
        )
        figures['velocity_profiles'] = self._plot_velocity_profiles(
            velocities_by_transition, window_days
        )

        # Compute summary metrics
        metrics = {
            'transitions_analyzed': list(trajectories_by_transition.keys()),
            'velocity_summary': {
                name: {
                    'mean': data['mean_velocity'],
                    'peak_day': data['peak_velocity_day'],
                }
                for name, data in velocities_by_transition.items()
            },
        }

        return ProbeResult(
            name=self.name,
            metrics=metrics,
            figures=figures,
            raw_data={
                'trajectories': {
                    name: data['trajectories']
                    for name, data in trajectories_by_transition.items()
                },
            },
        )

    def _find_transition_samples(
        self,
        transition_date: pd.Timestamp,
        window_days: int,
    ) -> List[int]:
        """Find dataset samples that span the transition window."""
        valid_indices = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]

            if hasattr(sample, 'daily_dates') and len(sample.daily_dates) > 0:
                dates = pd.to_datetime(sample.daily_dates)
                min_date = dates.min()
                max_date = dates.max()

                # Check if sample spans the transition window
                window_start = transition_date - timedelta(days=window_days)
                window_end = transition_date + timedelta(days=window_days)

                if min_date <= window_start and max_date >= window_end:
                    valid_indices.append(idx)

        return valid_indices

    def _extract_transition_trajectories(
        self,
        sample_indices: List[int],
        transition_date: pd.Timestamp,
        window_days: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract latent trajectories around a transition."""
        all_trajectories = []
        all_dates = []

        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start:batch_start + batch_size]
            batch, _, dates = self.prepare_batch(batch_indices)

            # Extract latent representations
            latents = extract_latent_representations(self.model, batch, self.device)

            # Get temporal encoded output
            if 'temporal_encoded' in latents:
                temporal_latents = latents['temporal_encoded'].cpu().numpy()
            elif 'daily_fused' in latents:
                temporal_latents = latents['daily_fused'].cpu().numpy()
            else:
                continue

            # Extract window around transition for each sample
            for i, idx in enumerate(batch_indices):
                sample = self.dataset[idx]
                if not hasattr(sample, 'daily_dates'):
                    continue

                sample_dates = pd.to_datetime(sample.daily_dates)

                # Find indices for window around transition
                window_indices = []
                window_dates = []
                for d_offset in range(-window_days, window_days + 1):
                    target_date = transition_date + timedelta(days=d_offset)
                    # Find closest date in sample
                    date_diffs = np.abs((sample_dates - target_date).total_seconds())
                    closest_idx = np.argmin(date_diffs)
                    if date_diffs[closest_idx] < 86400:  # Within 1 day
                        window_indices.append(closest_idx)
                        window_dates.append(d_offset)

                if len(window_indices) > 0 and i < temporal_latents.shape[0]:
                    # Extract latent trajectory
                    trajectory = temporal_latents[i, window_indices, :]
                    all_trajectories.append(trajectory)
                    all_dates.append(window_dates)

        if all_trajectories:
            # Pad to same length
            max_len = max(t.shape[0] for t in all_trajectories)
            padded = np.zeros((len(all_trajectories), max_len, all_trajectories[0].shape[-1]))
            for i, t in enumerate(all_trajectories):
                padded[i, :t.shape[0], :] = t
            return padded, np.array(all_dates, dtype=object)

        return np.array([]), np.array([])

    def _compute_trajectory_velocities(
        self,
        trajectories: np.ndarray,
    ) -> np.ndarray:
        """Compute latent space velocities from trajectories."""
        # trajectories shape: [n_samples, n_timesteps, d_model]
        # velocity = ||z(t+1) - z(t)||

        velocities = np.linalg.norm(
            trajectories[:, 1:, :] - trajectories[:, :-1, :],
            axis=-1
        )
        return velocities

    def _plot_latent_trajectories(
        self,
        trajectories_by_transition: Dict[str, Dict],
        window_days: int,
    ) -> plt.Figure:
        """Plot latent trajectories around transitions using PCA."""
        from sklearn.decomposition import PCA

        # Guard: Handle empty trajectories dict
        n_transitions = len(trajectories_by_transition)
        if n_transitions == 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No transition trajectories available.\nNo samples span the transition windows.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Latent Trajectories Around Regime Transitions', fontsize=14)
            ax.axis('off')
            return fig

        fig, axes = plt.subplots(1, n_transitions,
                                 figsize=(6 * n_transitions, 5))
        if n_transitions == 1:
            axes = [axes]

        for ax, (trans_name, data) in zip(axes, trajectories_by_transition.items()):
            trajectories = data['trajectories']

            # Flatten for PCA
            n_samples, n_steps, d_model = trajectories.shape
            flat_traj = trajectories.reshape(-1, d_model)

            # Apply PCA
            pca = PCA(n_components=2)
            pca_traj = pca.fit_transform(flat_traj)
            pca_traj = pca_traj.reshape(n_samples, n_steps, 2)

            # Plot trajectories with color gradient
            cmap = plt.cm.coolwarm
            for i in range(min(n_samples, 10)):  # Plot up to 10 trajectories
                colors = cmap(np.linspace(0, 1, n_steps))
                for j in range(n_steps - 1):
                    ax.plot(pca_traj[i, j:j+2, 0], pca_traj[i, j:j+2, 1],
                           color=colors[j], alpha=0.5, linewidth=1)

                # Mark transition point (middle)
                mid = n_steps // 2
                ax.scatter(pca_traj[i, mid, 0], pca_traj[i, mid, 1],
                          color='black', s=50, marker='x', zorder=5)

            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.set_title(f'{trans_name.replace("_", " ").title()}', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(-window_days, window_days))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Days from transition')

        plt.suptitle('Latent Trajectories Around Regime Transitions (PCA)', fontsize=14)
        plt.tight_layout()
        return fig

    def _plot_velocity_profiles(
        self,
        velocities_by_transition: Dict[str, Dict],
        window_days: int,
    ) -> plt.Figure:
        """Plot velocity profiles around transitions."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Guard: Handle empty velocity data
        if not velocities_by_transition:
            ax.text(0.5, 0.5, 'No velocity profile data available.\nNo transitions were analyzed successfully.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Latent Space Velocity Profiles Around Regime Transitions', fontsize=14)
            return fig

        colors = plt.cm.tab10.colors
        for i, (trans_name, data) in enumerate(velocities_by_transition.items()):
            velocities = data['velocities']

            # Guard: Skip if velocities array is empty
            if velocities.size == 0:
                continue

            mean_vel = np.mean(velocities, axis=0)
            std_vel = np.std(velocities, axis=0)

            # Guard: Ensure mean_vel has data
            if mean_vel.size == 0:
                continue

            days = np.arange(-window_days, window_days)
            if len(mean_vel) < len(days):
                days = days[:len(mean_vel)]

            ax.plot(days, mean_vel, label=trans_name.replace('_', ' ').title(),
                   color=colors[i % len(colors)], linewidth=2)
            ax.fill_between(days, mean_vel - std_vel, mean_vel + std_vel,
                           color=colors[i % len(colors)], alpha=0.2)

        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Transition')
        ax.set_xlabel('Days from Transition', fontsize=12)
        ax.set_ylabel('Latent Velocity ||z(t+1) - z(t)||', fontsize=12)
        ax.set_title('Latent Space Velocity Profiles Around Regime Transitions', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# ==============================================================================
# 3.2.2 LATENT VELOCITY PROBE
# ==============================================================================

class LatentVelocityProbe(BaseProbe):
    """
    Probe 3.2.2: Latent Velocity Prediction

    Analyzes relationship between latent velocity and:
        - Upcoming regime transitions
        - Casualty rates
        - Other conflict intensity indicators
    """

    @property
    def name(self) -> str:
        return "latent_velocity_probe"

    def run(
        self,
        num_samples: int = 200,
        batch_size: int = 8,
    ) -> ProbeResult:
        """
        Analyze latent velocity correlations with conflict dynamics.

        Args:
            num_samples: Number of samples to analyze
            batch_size: Batch size for inference
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        n_available = len(self.dataset)
        sample_indices = np.random.choice(
            n_available, min(num_samples, n_available), replace=False
        )

        all_velocities = []
        all_phases = []
        all_days_to_transition = []
        all_dates = []

        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start:batch_start + batch_size].tolist()
            batch, labels, dates = self.prepare_batch(batch_indices)

            # Extract latent representations
            latents = extract_latent_representations(self.model, batch, self.device)

            if 'temporal_encoded' in latents:
                temporal = latents['temporal_encoded'].cpu().numpy()
            elif 'daily_fused' in latents:
                temporal = latents['daily_fused'].cpu().numpy()
            else:
                continue

            # Compute velocities
            velocities = np.linalg.norm(temporal[:, 1:, :] - temporal[:, :-1, :], axis=-1)

            for i, idx in enumerate(batch_indices):
                sample = self.dataset[idx]

                if i >= velocities.shape[0]:
                    continue

                sample_vel = velocities[i]
                all_velocities.append(np.mean(sample_vel))

                # Get phase
                if hasattr(sample, 'monthly_dates') and len(sample.monthly_dates) > 0:
                    phase = date_to_phase(sample.monthly_dates[-1])
                    all_phases.append(phase)

                    # Compute days to nearest transition
                    last_date = pd.to_datetime(sample.monthly_dates[-1])
                    days_to_trans = get_days_to_transition(last_date)
                    min_days = min(abs(d) for d in days_to_trans.values())
                    all_days_to_transition.append(min_days)
                    all_dates.append(last_date)
                else:
                    all_phases.append(0)
                    all_days_to_transition.append(365)
                    all_dates.append(None)

        # Compute correlations
        velocities_arr = np.array(all_velocities) if all_velocities else np.array([])
        phases_arr = np.array(all_phases) if all_phases else np.array([])
        days_arr = np.array(all_days_to_transition) if all_days_to_transition else np.array([])

        # Guard: Handle empty velocity data
        if len(velocities_arr) == 0:
            print("  Warning: No velocities computed - insufficient valid samples")
            return ProbeResult(
                name=self.name,
                metrics={
                    'velocity_by_phase': {},
                    'transition_correlation': {'pearson_r': 0, 'p_value': 1},
                    'overall_stats': {
                        'mean_velocity': 0.0,
                        'std_velocity': 0.0,
                        'n_samples': 0,
                    },
                    'error': 'No valid velocity samples computed. Check dataset date coverage.',
                },
                figures={},
                raw_data={
                    'velocities': velocities_arr,
                    'phases': phases_arr,
                    'days_to_transition': days_arr,
                },
            )

        # Velocity by phase
        velocity_by_phase = {}
        for phase in range(4):
            phase_mask = phases_arr == phase
            if phase_mask.sum() > 0:
                velocity_by_phase[PHASE_LABELS[phase]] = {
                    'mean': float(np.mean(velocities_arr[phase_mask])),
                    'std': float(np.std(velocities_arr[phase_mask])),
                    'n_samples': int(phase_mask.sum()),
                }

        # Velocity vs days to transition correlation
        valid_mask = days_arr < 365
        if valid_mask.sum() > 10:
            correlation, p_value = stats.pearsonr(
                velocities_arr[valid_mask],
                days_arr[valid_mask]
            )
            transition_correlation = {
                'pearson_r': float(correlation),
                'p_value': float(p_value),
            }
        else:
            transition_correlation = {'pearson_r': 0, 'p_value': 1}

        # Generate visualizations
        figures = {}
        figures['velocity_by_phase'] = self._plot_velocity_by_phase(velocity_by_phase)
        figures['velocity_vs_transition'] = self._plot_velocity_vs_transition(
            velocities_arr, days_arr, valid_mask
        )
        figures['velocity_timeline'] = self._plot_velocity_timeline(
            all_velocities, all_dates, all_phases
        )

        return ProbeResult(
            name=self.name,
            metrics={
                'velocity_by_phase': velocity_by_phase,
                'transition_correlation': transition_correlation,
                'overall_stats': {
                    'mean_velocity': float(np.mean(velocities_arr)),
                    'std_velocity': float(np.std(velocities_arr)),
                    'n_samples': len(velocities_arr),
                },
            },
            figures=figures,
            raw_data={
                'velocities': velocities_arr,
                'phases': phases_arr,
                'days_to_transition': days_arr,
            },
        )

    def _plot_velocity_by_phase(
        self,
        velocity_by_phase: Dict[str, Dict],
    ) -> plt.Figure:
        """Plot latent velocity distribution by conflict phase."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Guard: Handle empty velocity data
        if not velocity_by_phase:
            ax.text(0.5, 0.5, 'No velocity data available by phase.\nNo samples were processed successfully.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Latent Space Velocity by Conflict Phase', fontsize=14)
            return fig

        phases = list(velocity_by_phase.keys())
        means = [velocity_by_phase[p]['mean'] for p in phases]
        stds = [velocity_by_phase[p]['std'] for p in phases]

        # Guard: Map phase names to colors safely
        colors = []
        phase_labels_list = list(PHASE_LABELS.values())
        for p in phases:
            if p in phase_labels_list:
                colors.append(PHASE_COLORS[phase_labels_list.index(p)])
            else:
                colors.append('#808080')  # Default gray for unknown phases

        bars = ax.bar(phases, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                     edgecolor='black')

        ax.set_xlabel('Conflict Phase', fontsize=12)
        ax.set_ylabel('Mean Latent Velocity', fontsize=12)
        ax.set_title('Latent Space Velocity by Conflict Phase', fontsize=14)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

        # Add sample counts
        for i, (bar, phase) in enumerate(zip(bars, phases)):
            n = velocity_by_phase[phase]['n_samples']
            ax.annotate(f'n={n}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _plot_velocity_vs_transition(
        self,
        velocities: np.ndarray,
        days_to_transition: np.ndarray,
        valid_mask: np.ndarray,
    ) -> plt.Figure:
        """Plot latent velocity vs days to nearest transition."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Guard: Check if we have valid data points
        n_valid = valid_mask.sum() if valid_mask.size > 0 else 0
        if n_valid == 0:
            ax.text(0.5, 0.5, 'No valid velocity data available.\nNo samples have known transition distances.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel('Days to Nearest Transition', fontsize=12)
            ax.set_ylabel('Latent Velocity', fontsize=12)
            ax.set_title('Latent Velocity vs. Proximity to Regime Transition', fontsize=14)
            return fig

        valid_days = days_to_transition[valid_mask]
        valid_velocities = velocities[valid_mask]

        ax.scatter(valid_days, valid_velocities, alpha=0.5, c='#1f77b4', s=30)

        # Add trend line only if we have enough points
        if n_valid >= 2:
            try:
                z = np.polyfit(valid_days, valid_velocities, 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_days.min(), valid_days.max(), 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear trend')
                ax.legend()
            except (np.linalg.LinAlgError, ValueError) as e:
                # Polyfit can fail with degenerate data
                ax.text(0.02, 0.98, f'Trend line unavailable: {e}',
                       transform=ax.transAxes, fontsize=8, va='top')

        ax.set_xlabel('Days to Nearest Transition', fontsize=12)
        ax.set_ylabel('Latent Velocity', fontsize=12)
        ax.set_title('Latent Velocity vs. Proximity to Regime Transition', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_velocity_timeline(
        self,
        velocities: List[float],
        dates: List[Optional[pd.Timestamp]],
        phases: List[int],
    ) -> plt.Figure:
        """Plot velocity timeline with regime transition markers."""
        fig, ax = plt.subplots(figsize=(14, 6))

        # Filter valid dates
        valid_data = [(d, v, p) for d, v, p in zip(dates, velocities, phases) if d is not None]
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid date data', ha='center', va='center')
            return fig

        valid_dates, valid_velocities, valid_phases = zip(*sorted(valid_data))

        # Color by phase
        colors = [PHASE_COLORS[p] for p in valid_phases]

        ax.scatter(valid_dates, valid_velocities, c=colors, alpha=0.6, s=30)

        # Add transition lines
        for trans_name, trans_date in REGIME_TRANSITIONS.items():
            ax.axvline(trans_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(trans_date, ax.get_ylim()[1], trans_name.replace('_', '\n'),
                   rotation=0, ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Latent Velocity', fontsize=12)
        ax.set_title('Latent Space Velocity Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add legend
        handles = [mpatches.Patch(color=PHASE_COLORS[i], label=PHASE_LABELS[i])
                  for i in range(4)]
        ax.legend(handles=handles, loc='upper right')

        plt.tight_layout()
        return fig


# ==============================================================================
# PROBE RUNNER
# ==============================================================================

class TemporalDynamicsProbeRunner:
    """
    Orchestrates execution of all temporal dynamics probes.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path or CHECKPOINT_DIR
        self.output_dir = output_dir or get_output_dir()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.dataset = None
        self.results = {}

    def setup(self) -> None:
        """Load model and dataset."""
        print("="*60)
        print("Setting up Temporal Dynamics Probe Runner")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model path: {self.model_path}")
        print(f"Output dir: {self.output_dir}")

        # Load training configuration
        config_path = self.model_path / "training_summary.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                training_summary = json.load(f)
            config_dict = training_summary.get('config', {})
        else:
            print("  Warning: training_summary.json not found, using defaults")
            config_dict = {}

        # Create data configuration
        # NOTE: use_disaggregated_equipment and detrend_viirs must match training config
        # to ensure consistent source configuration (6 vs 8 daily sources)
        data_config = MultiResolutionConfig(
            daily_seq_len=config_dict.get('daily_seq_len', 365),
            monthly_seq_len=config_dict.get('monthly_seq_len', 12),
            prediction_horizon=config_dict.get('prediction_horizon', 1),
            use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', True),
            detrend_viirs=config_dict.get('detrend_viirs', True),
        )

        # Load dataset
        print("\n  Loading dataset...")
        self.dataset = MultiResolutionDataset(data_config, split='train')
        print(f"    Dataset size: {len(self.dataset)} samples")

        # Get feature dimensions dynamically from dataset sample
        # NOTE: Source names depend on use_disaggregated_equipment setting:
        #   - use_disaggregated_equipment=True: drones, armor, artillery (8 daily sources)
        #   - use_disaggregated_equipment=False: equipment (6 daily sources)
        sample = self.dataset[0]
        daily_source_configs = {}
        monthly_source_configs = {}

        # Build configs from actual sources in the sample (not hardcoded list)
        for source_name, tensor in sample.daily_features.items():
            n_features = tensor.shape[-1]
            daily_source_configs[source_name] = SourceConfig(
                name=source_name,
                n_features=n_features,
                resolution='daily',
            )

        for source_name, tensor in sample.monthly_features.items():
            n_features = tensor.shape[-1]
            monthly_source_configs[source_name] = SourceConfig(
                name=source_name,
                n_features=n_features,
                resolution='monthly',
            )

        print(f"    Daily sources: {list(daily_source_configs.keys())}")
        print(f"    Monthly sources: {list(monthly_source_configs.keys())}")

        # Create model
        print("\n  Creating model...")
        self.model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=config_dict.get('d_model', 64),
            nhead=config_dict.get('nhead', 4),
            num_daily_layers=config_dict.get('num_daily_layers', 3),
            num_monthly_layers=config_dict.get('num_monthly_layers', 2),
            num_fusion_layers=config_dict.get('num_fusion_layers', 2),
            num_temporal_layers=2,
            dropout=0.0,
        )

        # Load checkpoint
        checkpoint_path = self.model_path / "best_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"\n  Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            print("    Checkpoint loaded successfully")
        else:
            print(f"\n  Warning: No checkpoint found at {checkpoint_path}")
            print("    Using randomly initialized model")

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"\n  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("  Setup complete!")

    def run_all_probes(
        self,
        probes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, ProbeResult]:
        """
        Run all specified probes.

        Args:
            probes: List of probe names to run (None = all)
            **kwargs: Additional arguments passed to each probe
        """
        if self.model is None:
            self.setup()

        available_probes = {
            'context_window': ContextWindowProbe,
            'attention_distance': AttentionDistanceProbe,
            'predictive_horizon': PredictiveHorizonProbe,
            'transition_dynamics': TransitionDynamicsProbe,
            'latent_velocity': LatentVelocityProbe,
        }

        probes_to_run = probes or list(available_probes.keys())

        print("\n" + "="*60)
        print("Running Temporal Dynamics Probes")
        print("="*60)
        print(f"Probes to run: {probes_to_run}")

        for probe_name in probes_to_run:
            if probe_name not in available_probes:
                print(f"\n  Warning: Unknown probe '{probe_name}', skipping")
                continue

            probe_class = available_probes[probe_name]
            probe = probe_class(
                model=self.model,
                dataset=self.dataset,
                device=self.device,
                output_dir=self.output_dir,
            )

            try:
                result = probe.run(**kwargs)
                result.save(self.output_dir)
                self.results[probe_name] = result
            except Exception as e:
                print(f"\n  Error running probe '{probe_name}': {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*60)
        print("All probes complete!")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")

        return self.results

    def generate_summary_report(self) -> None:
        """Generate a summary report of all probe results."""
        report_path = self.output_dir / "summary_report.md"

        with open(report_path, 'w') as f:
            f.write("# Temporal Dynamics Probe Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Model: {self.model_path}\n\n")

            for probe_name, result in self.results.items():
                f.write(f"## {probe_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Timestamp:** {result.timestamp}\n\n")
                f.write("### Metrics\n\n")
                f.write("```json\n")
                f.write(json.dumps(result.metrics, indent=2, default=str))
                f.write("\n```\n\n")
                f.write(f"### Figures\n\n")
                for fig_name in result.figures.keys():
                    f.write(f"- {fig_name}.png\n")
                f.write("\n---\n\n")

        print(f"\nSummary report saved to: {report_path}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run temporal dynamics probes on Multi-Resolution HAN model"
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help="Specific probes to run (default: all)",
        choices=[
            'context_window',
            'attention_distance',
            'predictive_horizon',
            'transition_dynamics',
            'latent_velocity',
        ]
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to analyze per probe",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # Will use get_output_dir() if not specified
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    output_dir = Path(args.output_dir)

    runner = TemporalDynamicsProbeRunner(
        output_dir=output_dir,
        device=device,
    )

    runner.setup()
    results = runner.run_all_probes(
        probes=args.probes,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
    runner.generate_summary_report()

    print("\n" + "="*60)
    print("TEMPORAL DYNAMICS PROBE ANALYSIS COMPLETE")
    print("="*60)
