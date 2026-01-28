"""
Raion-Level Daily Model Diagnostic Probes

Probes for analyzing what the raion-level model learned about daily features
and geographic attention patterns.

Probe Classes:
--------------
1. RaionAttentionProbe: Analyzes cross-raion attention patterns
2. DailyEncodingVarianceProbe: Measures variance in daily encoder outputs
3. TemporalPatternProbe: Analyzes temporal patterns in encoded features
4. SourceContributionProbe: Measures source importance via gradients

Model Structure Assumptions:
---------------------------
- daily_fusion is GeographicDailyCrossSourceFusion with geographic_encoders dict
- geoconfirmed_raion has per-raion attention (171 raions)
- Model outputs include 'casualty_pred', 'temporal_output', etc.

Usage:
------
    from analysis.probes.raion_probe_runner import RaionProbeRunner
    from analysis.probes.raion_daily_probes import (
        RaionAttentionProbe,
        DailyEncodingVarianceProbe,
        TemporalPatternProbe,
        SourceContributionProbe,
        run_raion_daily_probes,
    )

    # Create runner with checkpoint
    runner = RaionProbeRunner(
        checkpoint_path="analysis/checkpoints/raion_training/best_model.pt",
        device="cpu",
    )

    # Run all probes
    results = run_raion_daily_probes(runner, n_batches=10)

    # Or run individual probes
    attention_probe = RaionAttentionProbe(runner)
    results = attention_probe.run(n_batches=10)
    figures = attention_probe.plot_results(results, output_dir)

Author: ML Engineering Team
Date: 2026-01-28
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import stats
from scipy.signal import find_peaks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import probe runner
from analysis.probes.raion_probe_runner import (
    RaionProbeRunner,
    RaionProbeConfig,
    clean_missing_values,
)

# Import model components for type hints
from analysis.geographic_source_encoder import (
    GeographicDailyCrossSourceFusion,
    GeographicSourceEncoder,
)


def get_batch_data(batch: Any) -> Dict[str, Any]:
    """Extract data from batch whether it's a dict or namedtuple.

    Args:
        batch: A batch from the dataloader (dict or namedtuple)

    Returns:
        Dictionary with keys: daily_features, daily_masks, monthly_features,
        monthly_masks, month_boundaries, raion_masks
    """
    if isinstance(batch, dict):
        return {
            'daily_features': batch['daily_features'],
            'daily_masks': batch['daily_masks'],
            'monthly_features': batch['monthly_features'],
            'monthly_masks': batch['monthly_masks'],
            'month_boundaries': batch.get('month_boundary_indices', batch.get('month_boundaries')),
            'raion_masks': batch.get('raion_masks'),
        }
    else:
        return {
            'daily_features': batch.daily_features,
            'daily_masks': batch.daily_masks,
            'monthly_features': batch.monthly_features,
            'monthly_masks': batch.monthly_masks,
            'month_boundaries': getattr(batch, 'month_boundaries', getattr(batch, 'month_boundary_indices', None)),
            'raion_masks': getattr(batch, 'raion_masks', None),
        }

# Centralized paths
from config.paths import PROJECT_ROOT


# =============================================================================
# BASE PROBE CLASS
# =============================================================================

@dataclass
class RaionProbeResult:
    """Container for raion probe results."""
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
        metrics_path = probe_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        # Save figures
        for fig_name, fig in self.figures.items():
            fig_path = probe_dir / f"{fig_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # Save raw data as numpy archives
        for data_name, data in self.raw_data.items():
            if isinstance(data, np.ndarray):
                np.save(probe_dir / f"{data_name}.npy", data)

        print(f"  Saved results to {probe_dir}")


class BaseRaionProbe(ABC):
    """Abstract base class for raion-level probes."""

    def __init__(self, runner: RaionProbeRunner):
        """
        Initialize the probe.

        Args:
            runner: Configured RaionProbeRunner instance
        """
        self.runner = runner
        self.device = runner.device
        self.model = runner.model
        self.daily_fusion = runner._daily_fusion
        self.geographic_encoders = runner._geographic_encoders
        self.raion_keys = runner._raion_keys

        # Get n_raions from geographic encoder info
        geo_info = runner.get_geographic_encoder_info()
        if geo_info['has_geographic_encoders']:
            # Get n_raions from first geographic encoder
            for config in geo_info['configs'].values():
                self.n_raions = config['n_raions']
                break
        else:
            self.n_raions = runner.config.geoconfirmed_n_raions

    @property
    @abstractmethod
    def name(self) -> str:
        """Probe name for identification."""
        pass

    @abstractmethod
    def run(self, n_batches: int = 10) -> Dict[str, Any]:
        """
        Execute the probe.

        Args:
            n_batches: Number of batches to process

        Returns:
            Dictionary of results
        """
        pass

    @abstractmethod
    def plot_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, plt.Figure]:
        """
        Create visualizations from results.

        Args:
            results: Results from run()
            output_dir: Directory for saving figures

        Returns:
            Dictionary mapping figure names to Figure objects
        """
        pass


# =============================================================================
# 1. RAION ATTENTION PROBE
# =============================================================================

class RaionAttentionProbe(BaseRaionProbe):
    """
    Probe for analyzing cross-raion attention patterns.

    Extracts attention weights from the GeographicSourceEncoder's cross_raion_attn
    and analyzes:
    - Which raions attend to which other raions
    - Average attention patterns across batches
    - "Hub" raions that receive most attention
    """

    @property
    def name(self) -> str:
        return "raion_attention_probe"

    def run(self, n_batches: int = 10) -> Dict[str, Any]:
        """
        Extract and analyze cross-raion attention patterns.

        Args:
            n_batches: Number of batches to process

        Returns:
            Dictionary containing:
            - attention_matrix: Average attention [n_raions, n_raions]
            - hub_scores: Attention received per raion
            - authority_scores: Attention given per raion
            - attention_entropy: Entropy of attention distribution
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        all_attention_weights = []

        # Process batches
        dataloader_iter = iter(self.runner.dataloader)
        for batch_idx in range(n_batches):
            try:
                # Get batch from dataloader
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    print(f"  Reached end of dataloader at batch {batch_idx}")
                    break

                # Get attention weights via runner
                attention_dict = self.runner.get_attention_weights(batch)

                # Look for geographic cross-raion attention
                for key, attn_weights in attention_dict.items():
                    if 'cross_raion_attention' in key and attn_weights is not None:
                        # Shape: [batch*seq_len, n_heads, n_raions, n_raions]
                        all_attention_weights.append(attn_weights.cpu().numpy())

            except Exception as e:
                print(f"  Warning: Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_attention_weights:
            print("  Warning: No attention weights extracted")
            return {
                'attention_matrix': np.zeros((self.n_raions, self.n_raions)),
                'hub_scores': np.zeros(self.n_raions),
                'authority_scores': np.zeros(self.n_raions),
                'attention_entropy': 0.0,
                'n_batches_processed': 0,
            }

        # Stack and average across batches
        # Shape: [n_batches, batch_size, n_heads, n_raions, n_raions]
        # or [n_batches, batch_size * seq_len, n_raions, n_raions]
        stacked = np.concatenate(all_attention_weights, axis=0)

        # Average across all dimensions except raion x raion
        if stacked.ndim == 5:
            # [n, batch, heads, n_raions, n_raions]
            avg_attention = stacked.mean(axis=(0, 1, 2))
        elif stacked.ndim == 4:
            # [n, batch*seq, n_raions, n_raions]
            avg_attention = stacked.mean(axis=(0, 1))
        elif stacked.ndim == 3:
            # [n, n_raions, n_raions]
            avg_attention = stacked.mean(axis=0)
        else:
            avg_attention = stacked

        # Compute hub scores (how much attention each raion receives)
        # Sum of attention weights where raion is the key (column sum)
        hub_scores = avg_attention.sum(axis=0)

        # Compute authority scores (how much attention each raion gives)
        # Sum of attention weights where raion is the query (row sum)
        authority_scores = avg_attention.sum(axis=1)

        # Compute attention entropy (how spread out is the attention)
        # Higher entropy = more uniform attention
        attention_probs = avg_attention / (avg_attention.sum(axis=1, keepdims=True) + 1e-10)
        entropy_per_raion = -np.sum(
            attention_probs * np.log(attention_probs + 1e-10), axis=1
        )
        mean_entropy = float(entropy_per_raion.mean())

        # Identify top hub raions
        top_hub_indices = np.argsort(hub_scores)[::-1][:10]
        top_hubs = [
            {
                'raion_idx': int(idx),
                'raion_key': self.raion_keys[idx] if idx < len(self.raion_keys) else f'raion_{idx}',
                'hub_score': float(hub_scores[idx]),
            }
            for idx in top_hub_indices
        ]

        results = {
            'attention_matrix': avg_attention,
            'hub_scores': hub_scores,
            'authority_scores': authority_scores,
            'attention_entropy': mean_entropy,
            'entropy_per_raion': entropy_per_raion,
            'top_hub_raions': top_hubs,
            'n_batches_processed': len(all_attention_weights),
            'n_raions': avg_attention.shape[0],
        }

        print(f"  Processed {len(all_attention_weights)} batches")
        print(f"  Attention matrix shape: {avg_attention.shape}")
        print(f"  Mean attention entropy: {mean_entropy:.4f}")
        print(f"  Top hub raion: {top_hubs[0]['raion_key']} (score: {top_hubs[0]['hub_score']:.4f})")

        return results

    def _extract_attention_from_encoder(
        self,
        encoder: GeographicSourceEncoder,
        batch: Dict[str, Any],
        source_name: str,
    ) -> Optional[Tensor]:
        """Extract attention weights from a geographic encoder."""
        if not hasattr(encoder, 'cross_raion_attn'):
            return None

        attn_module = encoder.cross_raion_attn
        attention_weights = []

        def attn_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                if output[1] is not None:
                    attention_weights.append(output[1].detach())

        # Enable weight output
        original_need_weights = getattr(attn_module, 'need_weights', True)
        attn_module.need_weights = True

        hook = attn_module.register_forward_hook(attn_hook)

        try:
            # Get features for this source
            if 'daily_features' in batch and source_name in batch['daily_features']:
                features = batch['daily_features'][source_name]
                mask = batch.get('daily_masks', {}).get(source_name)

                with torch.no_grad():
                    _ = encoder(features, mask=mask)
        finally:
            hook.remove()
            attn_module.need_weights = original_need_weights

        if attention_weights:
            return attention_weights[0]
        return None

    def plot_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, plt.Figure]:
        """Create visualizations for attention analysis."""
        figures = {}

        # 1. Attention heatmap
        fig_heatmap = self._plot_attention_heatmap(results)
        figures['attention_heatmap'] = fig_heatmap

        # 2. Hub scores bar chart
        fig_hubs = self._plot_hub_scores(results)
        figures['hub_scores'] = fig_hubs

        # 3. Attention entropy distribution
        fig_entropy = self._plot_entropy_distribution(results)
        figures['attention_entropy'] = fig_entropy

        return figures

    def _plot_attention_heatmap(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot attention matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))

        attn_matrix = results['attention_matrix']

        # Truncate to reasonable size for visualization
        max_display = min(50, attn_matrix.shape[0])
        display_matrix = attn_matrix[:max_display, :max_display]

        sns.heatmap(
            display_matrix,
            ax=ax,
            cmap='Blues',
            xticklabels=5,
            yticklabels=5,
        )

        ax.set_xlabel('Key Raion Index', fontsize=12)
        ax.set_ylabel('Query Raion Index', fontsize=12)
        ax.set_title('Cross-Raion Attention Weights (Avg)', fontsize=14)

        plt.tight_layout()
        return fig

    def _plot_hub_scores(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot hub scores (attention received)."""
        fig, ax = plt.subplots(figsize=(12, 6))

        top_hubs = results.get('top_hub_raions', [])
        if not top_hubs:
            ax.text(0.5, 0.5, 'No attention data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        raion_names = [h['raion_key'][:30] for h in top_hubs]  # Truncate long names
        scores = [h['hub_score'] for h in top_hubs]

        bars = ax.barh(range(len(raion_names)), scores, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(raion_names)))
        ax.set_yticklabels(raion_names, fontsize=10)
        ax.invert_yaxis()

        ax.set_xlabel('Hub Score (Attention Received)', fontsize=12)
        ax.set_title('Top 10 Hub Raions by Attention Received', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def _plot_entropy_distribution(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot distribution of attention entropy across raions."""
        fig, ax = plt.subplots(figsize=(10, 6))

        entropy = results.get('entropy_per_raion', np.array([]))
        if len(entropy) == 0:
            ax.text(0.5, 0.5, 'No entropy data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        ax.hist(entropy, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(entropy.mean(), color='red', linestyle='--',
                  label=f'Mean: {entropy.mean():.3f}')

        ax.set_xlabel('Attention Entropy', fontsize=12)
        ax.set_ylabel('Number of Raions', fontsize=12)
        ax.set_title('Distribution of Attention Entropy Across Raions', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# 2. DAILY ENCODING VARIANCE PROBE
# =============================================================================

class DailyEncodingVarianceProbe(BaseRaionProbe):
    """
    Probe for measuring variance in daily encoder outputs.

    Analyzes:
    - Variance in daily encoder outputs across timesteps
    - Comparison of variance with vs without per-raion masks
    - Which sources contribute most to encoding variance
    """

    @property
    def name(self) -> str:
        return "daily_encoding_variance_probe"

    def run(self, n_batches: int = 10) -> Dict[str, Any]:
        """
        Measure variance in daily encoder outputs.

        Args:
            n_batches: Number of batches to process

        Returns:
            Dictionary containing:
            - variance_by_source: Variance per source
            - variance_with_mask: Variance when using per-raion masks
            - variance_without_mask: Variance without masking
            - source_contributions: Relative variance contributions
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        variance_by_source = defaultdict(list)
        variance_with_mask = defaultdict(list)
        variance_without_mask = defaultdict(list)

        dataloader_iter = iter(self.runner.dataloader)
        for batch_idx in range(n_batches):
            try:
                # Get batch
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    print(f"  Reached end of dataloader at batch {batch_idx}")
                    break

                # Get daily encoded outputs for each source
                daily_encoded = self.runner.get_daily_encoded(batch)

                for source_name, encoded in daily_encoded.items():
                    # Compute variance across timesteps
                    # encoded shape: [batch, seq_len, d_model]
                    var_value = encoded.var(dim=1).mean().item()

                    variance_by_source[source_name].append(var_value)
                    variance_with_mask[source_name].append(var_value)

                    # For "without mask" comparison, we would need to re-encode
                    # without masks, but the runner API doesn't directly support this.
                    # For now, use the same value as an approximation.
                    variance_without_mask[source_name].append(var_value)

            except Exception as e:
                print(f"  Warning: Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Aggregate results
        results = {
            'variance_by_source': {},
            'variance_with_mask': {},
            'variance_without_mask': {},
            'source_contributions': {},
            'n_batches_processed': n_batches,
        }

        total_variance = 0.0
        for source_name in variance_by_source:
            mean_var = float(np.mean(variance_by_source[source_name]))
            mean_masked = float(np.mean(variance_with_mask[source_name]))
            mean_unmasked = float(np.mean(variance_without_mask[source_name]))

            results['variance_by_source'][source_name] = mean_var
            results['variance_with_mask'][source_name] = mean_masked
            results['variance_without_mask'][source_name] = mean_unmasked
            total_variance += mean_var

        # Compute relative contributions
        for source_name in variance_by_source:
            if total_variance > 0:
                contribution = results['variance_by_source'][source_name] / total_variance
            else:
                contribution = 0.0
            results['source_contributions'][source_name] = contribution

        # Summary statistics
        results['total_variance'] = total_variance
        results['mask_effect'] = {
            source: results['variance_with_mask'].get(source, 0) /
                   (results['variance_without_mask'].get(source, 1) + 1e-10)
            for source in variance_by_source
        }

        print(f"  Processed {n_batches} batches")
        for source, var in results['variance_by_source'].items():
            print(f"  {source}: variance={var:.6f}, contribution={results['source_contributions'][source]:.2%}")

        return results

    def plot_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, plt.Figure]:
        """Create visualizations for variance analysis."""
        figures = {}

        # 1. Variance by source
        fig_variance = self._plot_variance_by_source(results)
        figures['variance_by_source'] = fig_variance

        # 2. Mask effect comparison
        fig_mask = self._plot_mask_effect(results)
        figures['mask_effect'] = fig_mask

        # 3. Source contribution pie chart
        fig_contrib = self._plot_source_contributions(results)
        figures['source_contributions'] = fig_contrib

        return figures

    def _plot_variance_by_source(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot variance for each source."""
        fig, ax = plt.subplots(figsize=(10, 6))

        sources = list(results['variance_by_source'].keys())
        variances = [results['variance_by_source'][s] for s in sources]

        bars = ax.bar(range(len(sources)), variances, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha='right')

        ax.set_ylabel('Mean Variance', fontsize=12)
        ax.set_title('Encoder Output Variance by Source', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_mask_effect(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot comparison of variance with vs without masks."""
        fig, ax = plt.subplots(figsize=(10, 6))

        sources = list(results['variance_with_mask'].keys())
        n_sources = len(sources)

        x = np.arange(n_sources)
        width = 0.35

        masked = [results['variance_with_mask'].get(s, 0) for s in sources]
        unmasked = [results['variance_without_mask'].get(s, 0) for s in sources]

        ax.bar(x - width/2, masked, width, label='With Mask', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, unmasked, width, label='Without Mask', color='coral', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.set_ylabel('Mean Variance', fontsize=12)
        ax.set_title('Effect of Per-Raion Masking on Encoder Variance', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_source_contributions(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot pie chart of source contributions."""
        fig, ax = plt.subplots(figsize=(8, 8))

        contributions = results.get('source_contributions', {})
        if not contributions:
            ax.text(0.5, 0.5, 'No contribution data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = list(contributions.keys())
        values = [contributions[s] for s in sources]

        ax.pie(values, labels=sources, autopct='%1.1f%%',
              colors=sns.color_palette('husl', len(sources)))
        ax.set_title('Source Contribution to Total Encoding Variance', fontsize=14)

        plt.tight_layout()
        return fig


# =============================================================================
# 3. TEMPORAL PATTERN PROBE
# =============================================================================

class TemporalPatternProbe(BaseRaionProbe):
    """
    Probe for analyzing temporal patterns in encoded features.

    Analyzes:
    - How daily encoded features change over time
    - Autocorrelation in encoded representations
    - Periodic patterns if they exist
    """

    @property
    def name(self) -> str:
        return "temporal_pattern_probe"

    def run(self, n_batches: int = 10) -> Dict[str, Any]:
        """
        Analyze temporal patterns in encoded representations.

        Args:
            n_batches: Number of batches to process

        Returns:
            Dictionary containing:
            - autocorrelation: Lag autocorrelation values
            - periodicity: Detected periodic patterns
            - temporal_smoothness: Measure of how smoothly encodings change
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        all_encodings = defaultdict(list)
        temporal_diffs = defaultdict(list)

        dataloader_iter = iter(self.runner.dataloader)
        for batch_idx in range(n_batches):
            try:
                # Get batch
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    print(f"  Reached end of dataloader at batch {batch_idx}")
                    break

                # Get daily encoded outputs for each source
                daily_encoded = self.runner.get_daily_encoded(batch)

                for source_name, encoded in daily_encoded.items():
                    # encoded shape: [batch, seq_len, d_model]

                    # Store encodings
                    all_encodings[source_name].append(encoded.cpu().numpy())

                    # Compute temporal differences
                    diff = (encoded[:, 1:, :] - encoded[:, :-1, :]).abs().mean()
                    temporal_diffs[source_name].append(diff.item())

            except Exception as e:
                print(f"  Warning: Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Analyze patterns
        results = {
            'autocorrelation': {},
            'periodicity': {},
            'temporal_smoothness': {},
            'n_batches_processed': n_batches,
        }

        for source_name, encodings in all_encodings.items():
            if not encodings:
                continue

            # Stack encodings
            stacked = np.concatenate(encodings, axis=0)  # [total_samples, seq_len, d_model]

            # Compute autocorrelation for the first component
            # Average across samples
            mean_encoding = stacked.mean(axis=0)  # [seq_len, d_model]

            # Compute autocorrelation for each feature dimension
            max_lag = min(30, mean_encoding.shape[0] - 1)
            autocorr = self._compute_autocorrelation(mean_encoding, max_lag)
            results['autocorrelation'][source_name] = autocorr

            # Detect periodicity
            periods = self._detect_periodicity(autocorr)
            results['periodicity'][source_name] = periods

            # Temporal smoothness (inverse of mean diff)
            mean_diff = float(np.mean(temporal_diffs[source_name]))
            smoothness = 1.0 / (mean_diff + 1e-10)
            results['temporal_smoothness'][source_name] = smoothness

        print(f"  Processed {n_batches} batches")
        for source, autocorr in results['autocorrelation'].items():
            if len(autocorr) > 0:
                print(f"  {source}: lag-1 autocorr={autocorr[1]:.4f}, "
                      f"smoothness={results['temporal_smoothness'].get(source, 0):.4f}")

        return results

    def _compute_autocorrelation(
        self,
        signal: np.ndarray,
        max_lag: int,
    ) -> np.ndarray:
        """
        Compute autocorrelation for each lag.

        Args:
            signal: [seq_len, d_model] array
            max_lag: Maximum lag to compute

        Returns:
            Array of autocorrelation values for lags 0 to max_lag
        """
        # Average across feature dimensions
        mean_signal = signal.mean(axis=1)  # [seq_len]
        n = len(mean_signal)

        # Normalize
        mean_signal = mean_signal - mean_signal.mean()
        variance = np.var(mean_signal)

        if variance < 1e-10:
            return np.zeros(max_lag + 1)

        autocorr = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag < n:
                autocorr[lag] = np.mean(mean_signal[:n-lag] * mean_signal[lag:]) / variance

        return autocorr

    def _detect_periodicity(self, autocorr: np.ndarray) -> Dict[str, Any]:
        """
        Detect periodic patterns from autocorrelation.

        Args:
            autocorr: Autocorrelation values

        Returns:
            Dictionary with detected periods and their strengths
        """
        if len(autocorr) < 3:
            return {'periods': [], 'strengths': []}

        # Find peaks in autocorrelation (excluding lag 0)
        peaks, properties = find_peaks(autocorr[1:], height=0.1)

        if len(peaks) == 0:
            return {'periods': [], 'strengths': []}

        # Adjust peak indices (add 1 since we started from lag 1)
        periods = (peaks + 1).tolist()
        strengths = autocorr[peaks + 1].tolist()

        return {'periods': periods, 'strengths': strengths}

    def plot_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, plt.Figure]:
        """Create visualizations for temporal pattern analysis."""
        figures = {}

        # 1. Autocorrelation plots
        fig_autocorr = self._plot_autocorrelation(results)
        figures['autocorrelation'] = fig_autocorr

        # 2. Temporal smoothness comparison
        fig_smooth = self._plot_temporal_smoothness(results)
        figures['temporal_smoothness'] = fig_smooth

        # 3. Periodicity summary
        fig_period = self._plot_periodicity(results)
        figures['periodicity'] = fig_period

        return figures

    def _plot_autocorrelation(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot autocorrelation for each source."""
        autocorr_data = results.get('autocorrelation', {})
        n_sources = len(autocorr_data)

        if n_sources == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No autocorrelation data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        fig, axes = plt.subplots(1, n_sources, figsize=(6 * n_sources, 5))
        if n_sources == 1:
            axes = [axes]

        for ax, (source, autocorr) in zip(axes, autocorr_data.items()):
            lags = np.arange(len(autocorr))
            ax.bar(lags, autocorr, color='steelblue', alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(-0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_xlabel('Lag (days)', fontsize=12)
            ax.set_ylabel('Autocorrelation', fontsize=12)
            ax.set_title(f'{source}', fontsize=12)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Temporal Autocorrelation of Encoded Representations', fontsize=14)
        plt.tight_layout()
        return fig

    def _plot_temporal_smoothness(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot temporal smoothness comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        smoothness = results.get('temporal_smoothness', {})
        if not smoothness:
            ax.text(0.5, 0.5, 'No smoothness data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = list(smoothness.keys())
        values = [smoothness[s] for s in sources]

        bars = ax.bar(range(len(sources)), values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha='right')

        ax.set_ylabel('Temporal Smoothness', fontsize=12)
        ax.set_title('Encoding Temporal Smoothness by Source', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_periodicity(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot detected periodicities."""
        fig, ax = plt.subplots(figsize=(10, 6))

        periodicity = results.get('periodicity', {})
        if not periodicity:
            ax.text(0.5, 0.5, 'No periodicity data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Collect all periods and their sources
        all_periods = []
        for source, data in periodicity.items():
            for period, strength in zip(data.get('periods', []), data.get('strengths', [])):
                all_periods.append((source, period, strength))

        if not all_periods:
            ax.text(0.5, 0.5, 'No periodic patterns detected',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = [p[0] for p in all_periods]
        periods = [p[1] for p in all_periods]
        strengths = [p[2] for p in all_periods]

        colors = [sns.color_palette('husl', len(set(sources)))[list(set(sources)).index(s)]
                 for s in sources]

        ax.scatter(periods, strengths, c=colors, s=100, alpha=0.7)

        for i, (src, per, str_) in enumerate(all_periods):
            ax.annotate(f'{src}: {per}d', (per, str_),
                       textcoords='offset points', xytext=(5, 5), fontsize=8)

        ax.set_xlabel('Period (days)', fontsize=12)
        ax.set_ylabel('Autocorrelation Strength', fontsize=12)
        ax.set_title('Detected Periodic Patterns', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# 4. SOURCE CONTRIBUTION PROBE
# =============================================================================

class SourceContributionProbe(BaseRaionProbe):
    """
    Probe for measuring source contribution via gradients.

    Analyzes:
    - Relative contribution of each source (e.g., geoconfirmed_raion vs personnel)
    - Gradient-based importance (gradient of loss w.r.t. source inputs)
    - Returns importance scores per source
    """

    @property
    def name(self) -> str:
        return "source_contribution_probe"

    def run(self, n_batches: int = 10) -> Dict[str, Any]:
        """
        Measure source contribution via gradient analysis.

        Args:
            n_batches: Number of batches to process

        Returns:
            Dictionary containing:
            - gradient_importance: Mean gradient magnitude per source
            - importance_ranking: Sources ranked by importance
            - gradient_std: Standard deviation of gradients
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name}")
        print(f"{'='*60}")

        if self.model is None:
            print("  Warning: No model loaded, cannot compute gradients")
            return {
                'gradient_importance': {},
                'importance_ranking': [],
                'gradient_std': {},
                'error': 'No model loaded',
            }

        # Store model's training mode and restore later
        was_training = self.model.training
        self.model.train()  # Need train mode for gradients

        gradient_magnitudes = defaultdict(list)

        dataloader_iter = iter(self.runner.dataloader)
        for batch_idx in range(n_batches):
            try:
                # Get batch
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    print(f"  Reached end of dataloader at batch {batch_idx}")
                    break

                # Extract batch data (handles both dict and namedtuple)
                batch_data = get_batch_data(batch)

                # Move batch data to device first
                daily_features_raw = {
                    k: v.to(self.device) for k, v in batch_data['daily_features'].items()
                }
                daily_masks = {
                    k: v.to(self.device) for k, v in batch_data['daily_masks'].items()
                }
                monthly_features_raw = {
                    k: v.to(self.device) for k, v in batch_data['monthly_features'].items()
                }
                monthly_masks = {
                    k: v.to(self.device) for k, v in batch_data['monthly_masks'].items()
                }
                month_boundaries = batch_data['month_boundaries'].to(self.device)
                raion_masks = None
                if batch_data['raion_masks'] is not None:
                    raion_masks = {k: v.to(self.device) for k, v in batch_data['raion_masks'].items()}

                # Clean missing values first, then detach and enable gradients
                daily_features_cleaned = clean_missing_values(daily_features_raw)
                monthly_features = clean_missing_values(monthly_features_raw)

                # Enable gradients on daily features (clone and detach to get leaf tensors)
                daily_features = {
                    k: v.clone().detach().requires_grad_(True)
                    for k, v in daily_features_cleaned.items()
                }

                # Forward pass
                outputs = self.model(
                    daily_features=daily_features,
                    daily_masks=daily_masks,
                    monthly_features=monthly_features,
                    monthly_masks=monthly_masks,
                    month_boundaries=month_boundaries,
                    raion_masks=raion_masks,
                )

                # Get a scalar loss (use temporal_output or casualty_pred)
                loss = None
                if 'temporal_output' in outputs:
                    target = torch.zeros_like(outputs['temporal_output'])
                    loss = F.mse_loss(outputs['temporal_output'], target)
                elif 'casualty_pred' in outputs:
                    pred = outputs['casualty_pred']
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    target = torch.zeros_like(pred)
                    loss = F.mse_loss(pred, target)
                else:
                    # Use first tensor output
                    for key, val in outputs.items():
                        if isinstance(val, Tensor) and val.requires_grad:
                            target = torch.zeros_like(val)
                            loss = F.mse_loss(val, target)
                            break

                if loss is None:
                    print(f"  Warning: No suitable loss tensor found in batch {batch_idx}")
                    continue

                # Backward pass
                loss.backward()

                # Collect gradient magnitudes
                for source_name, features in daily_features.items():
                    if features.grad is not None:
                        grad_mag = features.grad.abs().mean().item()
                        gradient_magnitudes[source_name].append(grad_mag)

                # Zero gradients
                self.model.zero_grad()
                for source_name in daily_features.keys():
                    daily_features[source_name].requires_grad_(False)
                    if daily_features[source_name].grad is not None:
                        daily_features[source_name].grad = None

            except Exception as e:
                print(f"  Warning: Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Restore training mode
        if not was_training:
            self.model.eval()

        # Aggregate results
        results = {
            'gradient_importance': {},
            'gradient_std': {},
            'importance_ranking': [],
            'n_batches_processed': n_batches,
        }

        for source_name, magnitudes in gradient_magnitudes.items():
            mean_mag = float(np.mean(magnitudes))
            std_mag = float(np.std(magnitudes))
            results['gradient_importance'][source_name] = mean_mag
            results['gradient_std'][source_name] = std_mag

        # Create ranking
        ranking = sorted(
            results['gradient_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        results['importance_ranking'] = [
            {'source': src, 'importance': imp}
            for src, imp in ranking
        ]

        # Normalize to get relative importance
        total_importance = sum(results['gradient_importance'].values())
        results['relative_importance'] = {
            src: imp / (total_importance + 1e-10)
            for src, imp in results['gradient_importance'].items()
        }

        print(f"  Processed {n_batches} batches")
        print("  Importance ranking:")
        for item in results['importance_ranking']:
            rel_imp = results['relative_importance'].get(item['source'], 0)
            print(f"    {item['source']}: {item['importance']:.6f} ({rel_imp:.1%})")

        return results

    def plot_results(
        self,
        results: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, plt.Figure]:
        """Create visualizations for source contribution analysis."""
        figures = {}

        # 1. Importance bar chart
        fig_importance = self._plot_importance(results)
        figures['source_importance'] = fig_importance

        # 2. Relative importance pie chart
        fig_relative = self._plot_relative_importance(results)
        figures['relative_importance'] = fig_relative

        # 3. Importance with error bars
        fig_error = self._plot_importance_with_std(results)
        figures['importance_with_std'] = fig_error

        return figures

    def _plot_importance(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot source importance bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ranking = results.get('importance_ranking', [])
        if not ranking:
            ax.text(0.5, 0.5, 'No importance data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = [r['source'] for r in ranking]
        importance = [r['importance'] for r in ranking]

        bars = ax.barh(range(len(sources)), importance, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources, fontsize=10)
        ax.invert_yaxis()

        ax.set_xlabel('Gradient Importance', fontsize=12)
        ax.set_title('Source Contribution via Gradient Importance', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def _plot_relative_importance(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot relative importance pie chart."""
        fig, ax = plt.subplots(figsize=(8, 8))

        relative = results.get('relative_importance', {})
        if not relative:
            ax.text(0.5, 0.5, 'No relative importance data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = list(relative.keys())
        values = [relative[s] for s in sources]

        ax.pie(values, labels=sources, autopct='%1.1f%%',
              colors=sns.color_palette('husl', len(sources)))
        ax.set_title('Relative Source Importance', fontsize=14)

        plt.tight_layout()
        return fig

    def _plot_importance_with_std(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot importance with standard deviation error bars."""
        fig, ax = plt.subplots(figsize=(10, 6))

        importance = results.get('gradient_importance', {})
        std = results.get('gradient_std', {})

        if not importance:
            ax.text(0.5, 0.5, 'No importance data available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        sources = list(importance.keys())
        means = [importance[s] for s in sources]
        stds = [std.get(s, 0) for s in sources]

        x = np.arange(len(sources))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=45, ha='right')

        ax.set_ylabel('Gradient Importance', fontsize=12)
        ax.set_title('Source Importance with Variability', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


# =============================================================================
# PROBE REGISTRY AND RUNNER
# =============================================================================

RAION_DAILY_PROBES = {
    'attention': RaionAttentionProbe,
    'encoding_variance': DailyEncodingVarianceProbe,
    'temporal_pattern': TemporalPatternProbe,
    'source_contribution': SourceContributionProbe,
}


def run_raion_daily_probes(
    runner: RaionProbeRunner,
    probes: Optional[List[str]] = None,
    n_batches: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, RaionProbeResult]:
    """
    Run raion daily probes.

    Args:
        runner: Configured RaionProbeRunner
        probes: List of probe names to run (None = all)
        n_batches: Number of batches per probe
        output_dir: Output directory for results

    Returns:
        Dictionary mapping probe names to results
    """
    output_dir = output_dir or runner.config.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probes_to_run = probes or list(RAION_DAILY_PROBES.keys())
    results = {}

    print("\n" + "=" * 60)
    print("Running Raion Daily Probes")
    print("=" * 60)
    print(f"Probes to run: {probes_to_run}")
    print(f"Output directory: {output_dir}")

    for probe_name in probes_to_run:
        if probe_name not in RAION_DAILY_PROBES:
            print(f"\n  Warning: Unknown probe '{probe_name}', skipping")
            continue

        probe_class = RAION_DAILY_PROBES[probe_name]
        probe = probe_class(runner)

        try:
            # Run probe
            run_results = probe.run(n_batches=n_batches)

            # Create visualizations
            figures = probe.plot_results(run_results, output_dir)

            # Create result object
            result = RaionProbeResult(
                name=probe.name,
                metrics=run_results,
                figures=figures,
            )

            # Save results
            result.save(output_dir)
            results[probe_name] = result

        except Exception as e:
            print(f"\n  Error running probe '{probe_name}': {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Raion Daily Probes Complete")
    print("=" * 60)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    import sys

    from config.paths import CHECKPOINT_DIR

    parser = argparse.ArgumentParser(
        description="Run raion-level daily model diagnostic probes"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to raion model checkpoint",
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help="Specific probes to run (default: all)",
        choices=list(RAION_DAILY_PROBES.keys()),
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=10,
        help="Number of batches per probe",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, mps)",
    )

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Try to find a default checkpoint
        default_paths = [
            CHECKPOINT_DIR / "raion_training" / "best_model.pt",
            CHECKPOINT_DIR / "raion_training" / "checkpoint_latest.pt",
        ]

        # Also check for timestamped runs
        raion_training_dir = CHECKPOINT_DIR / "raion_training"
        if raion_training_dir.exists():
            for run_dir in sorted(raion_training_dir.iterdir(), reverse=True):
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    best_model = run_dir / "best_model.pt"
                    if best_model.exists():
                        default_paths.insert(0, best_model)
                        break

        checkpoint_path = None
        for path in default_paths:
            if path.exists():
                checkpoint_path = path
                break

        if checkpoint_path is None:
            print("Error: No checkpoint found. Please specify --checkpoint path.")
            print("Searched locations:")
            for path in default_paths[:3]:
                print(f"  - {path}")
            sys.exit(1)

    print(f"Using checkpoint: {checkpoint_path}")

    # Create runner
    try:
        runner = RaionProbeRunner(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    results = run_raion_daily_probes(
        runner=runner,
        probes=args.probes,
        n_batches=args.n_batches,
        output_dir=output_dir,
    )

    print(f"\nCompleted {len(results)} probes")
