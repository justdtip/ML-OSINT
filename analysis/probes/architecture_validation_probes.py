#!/usr/bin/env python3
"""
Architecture Validation Probes for Multi-Resolution HAN Model
==============================================================

This module provides probes to validate the effectiveness of new architecture
components added to the Multi-Resolution HAN:

Section 9: Architecture Component Validation
---------------------------------------------

9.1 TemporalSourceGate Validation (C2 Fix)
    - 9.1.1 GateWeightDynamicsProbe: Tests if gate weights vary meaningfully over time
    - 9.1.2 TemporalContextEffectProbe: Tests impact of temporal context on gating

9.2 DailyTemporalEncoder Validation (C1 Fix)
    - 9.2.1 MultiScalePatternProbe: Tests multi-scale conv pattern detection
    - 9.2.3 DailyEnrichmentProbe: Compares representations before/after encoding

9.3 ISWAlignmentModule Validation (C3 Fix)
    - 9.3.1 AlignmentQualityProbe: Tests alignment quality in projection space
    - 9.3.3 ConflictPhaseAlignmentProbe: Tests alignment across conflict phases

Author: ML Engineering Team
Date: 2026-01-26
"""

from __future__ import annotations

import json
import sys
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_resolution_han import (
    MultiResolutionHAN,
    MultiResolutionHANConfig,
    TemporalSourceGate,
    DailyTemporalEncoder,
    ISWAlignmentModule,
    DAILY_SOURCES,
    MONTHLY_SOURCES,
)
from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    multi_resolution_collate_fn,
)

from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR,
    MULTI_RES_CHECKPOINT_DIR,
    get_probe_figures_dir,
)

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class ArchitectureProbeResult:
    """Result from an architecture validation probe."""
    probe_id: str
    probe_name: str
    passed: bool
    metrics: Dict[str, float]
    interpretation: str
    visualizations: List[Path] = field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None


class ArchitectureProbe(ABC):
    """Base class for architecture validation probes."""

    probe_id: str = "0.0.0"
    probe_name: str = "Base Probe"
    tier: int = 2  # Priority tier (1=critical, 2=important, 3=exploratory)

    def __init__(
        self,
        model: MultiResolutionHAN,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        output_dir: Optional[Path] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir or get_probe_figures_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> ArchitectureProbeResult:
        """Execute the probe and return results."""
        pass

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save a figure and return its path."""
        subdir = self.output_dir / self.probe_id.replace(".", "_")
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path


# =============================================================================
# 9.1 TemporalSourceGate Validation Probes
# =============================================================================

class GateWeightDynamicsProbe(ArchitectureProbe):
    """
    9.1.1: Tests if gate weights vary meaningfully over time.

    The TemporalSourceGate should produce weights that change based on
    the temporal context, not just static or random weights.

    Metrics:
    - gate_temporal_variance: Variance of gate weights across time
    - gate_entropy: Entropy of gate distribution (low = selective, high = uniform)
    - gate_autocorrelation: How much gates at t correlate with gates at t+1
    """

    probe_id = "9.1.1"
    probe_name = "Gate Weight Dynamics"
    tier = 1

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        all_gate_weights = []
        all_timestamps = []

        # Hook to capture gate weights
        gate_weights_captured = []

        def capture_gate_hook(module, input, output):
            if isinstance(output, Tensor):
                gate_weights_captured.append(output.detach().cpu())

        # Find and hook the source gate
        hook_handle = None
        if hasattr(self.model, 'daily_fusion') and hasattr(self.model.daily_fusion, 'source_gate'):
            hook_handle = self.model.daily_fusion.source_gate.register_forward_hook(capture_gate_hook)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 20:  # Sample 20 batches
                    break

                # Move batch to device
                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks', 'isw_embedding', 'isw_mask',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                # Fix key name mapping
                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                try:
                    _ = self.model(**batch_gpu)
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} forward pass failed: {e}")
                    continue

        if hook_handle:
            hook_handle.remove()

        if not gate_weights_captured:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not capture gate weights. Model may not have TemporalSourceGate.",
            )

        # Concatenate captured weights
        gate_weights = torch.cat(gate_weights_captured, dim=0)  # [total_samples, seq, n_sources]

        # Compute metrics
        # 1. Temporal variance: how much do weights change over time?
        temporal_variance = gate_weights.var(dim=1).mean().item()

        # 2. Entropy: how selective is the gating?
        # High entropy = uniform weights (bad), low entropy = selective (good)
        eps = 1e-8
        entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=-1).mean().item()

        # 3. Autocorrelation: do gates evolve smoothly?
        gate_t = gate_weights[:, :-1, :]
        gate_t1 = gate_weights[:, 1:, :]
        autocorr = F.cosine_similarity(
            gate_t.reshape(-1, gate_t.shape[-1]),
            gate_t1.reshape(-1, gate_t1.shape[-1]),
            dim=-1
        ).mean().item()

        # 4. Source dominance: does any source dominate?
        mean_weights = gate_weights.mean(dim=(0, 1))
        max_dominance = mean_weights.max().item()
        n_sources = gate_weights.shape[-1]
        uniform_weight = 1.0 / n_sources

        # Pass criteria:
        # - Temporal variance should be > 0.001 (weights should vary)
        # - Entropy should be < log(n_sources) * 0.9 (some selectivity)
        # - Autocorrelation should be > 0.5 (smooth evolution)
        max_entropy = np.log(n_sources)
        passed = (
            temporal_variance > 0.001 and
            entropy < max_entropy * 0.9 and
            autocorr > 0.5
        )

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Gate weight over time for sample sequence
        ax1 = axes[0, 0]
        sample_weights = gate_weights[0].numpy()  # First sample
        for i in range(n_sources):
            ax1.plot(sample_weights[:, i], label=f'Source {i}', alpha=0.7)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Gate Weight')
        ax1.set_title('Gate Weights Over Time (Sample Sequence)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. Mean gate weights per source
        ax2 = axes[0, 1]
        source_names = DAILY_SOURCES[:n_sources] if n_sources <= len(DAILY_SOURCES) else [f'S{i}' for i in range(n_sources)]
        bars = ax2.bar(source_names, mean_weights.numpy())
        ax2.axhline(y=uniform_weight, color='r', linestyle='--', label='Uniform')
        ax2.set_xlabel('Source')
        ax2.set_ylabel('Mean Weight')
        ax2.set_title('Average Gate Weight per Source')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)

        # 3. Temporal variance distribution
        ax3 = axes[1, 0]
        per_source_var = gate_weights.var(dim=1).mean(dim=0).numpy()
        ax3.bar(source_names, per_source_var)
        ax3.set_xlabel('Source')
        ax3.set_ylabel('Temporal Variance')
        ax3.set_title('Gate Weight Temporal Variance by Source')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Entropy histogram
        ax4 = axes[1, 1]
        per_timestep_entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=-1)
        ax4.hist(per_timestep_entropy.flatten().numpy(), bins=50, alpha=0.7)
        ax4.axvline(x=max_entropy, color='r', linestyle='--', label=f'Max Entropy ({max_entropy:.2f})')
        ax4.set_xlabel('Entropy')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Gate Weight Entropy Distribution')
        ax4.legend()

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'gate_weight_dynamics')

        metrics = {
            'temporal_variance': temporal_variance,
            'mean_entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy / max_entropy,
            'autocorrelation': autocorr,
            'max_source_dominance': max_dominance,
            'uniform_weight': uniform_weight,
        }

        interpretation = (
            f"Gate Weight Dynamics Analysis:\n"
            f"- Temporal variance: {temporal_variance:.4f} (threshold: >0.001)\n"
            f"- Entropy ratio: {entropy/max_entropy:.2%} of maximum (threshold: <90%)\n"
            f"- Autocorrelation: {autocorr:.3f} (threshold: >0.5)\n"
            f"- Max source dominance: {max_dominance:.3f} vs uniform {uniform_weight:.3f}\n"
            f"\nConclusion: Gate weights {'show' if passed else 'DO NOT show'} meaningful temporal dynamics."
        )

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
            raw_data={'gate_weights_sample': sample_weights.tolist()},
        )


class TemporalContextEffectProbe(ArchitectureProbe):
    """
    9.1.2: Tests impact of temporal context on gating decisions.

    Compares gate weights when:
    1. Full temporal context is available
    2. Temporal context is ablated (zeroed past)

    If the gate is using temporal context, ablation should significantly change weights.
    """

    probe_id = "9.1.2"
    probe_name = "Temporal Context Effect"
    tier = 3

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        full_context_weights = []
        ablated_context_weights = []

        # Hook to capture gate weights
        current_weights = []

        def capture_gate_hook(module, input, output):
            if isinstance(output, Tensor):
                current_weights.append(output.detach().cpu())

        hook_handle = None
        if hasattr(self.model, 'daily_fusion') and hasattr(self.model.daily_fusion, 'source_gate'):
            hook_handle = self.model.daily_fusion.source_gate.register_forward_hook(capture_gate_hook)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 10:
                    break

                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks', 'isw_embedding', 'isw_mask',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                # Run with full context
                current_weights.clear()
                try:
                    _ = self.model(**batch_gpu)
                    if current_weights:
                        full_context_weights.append(current_weights[0])
                except Exception as e:
                    logger.warning(f"Full context forward failed: {e}")
                    continue

                # Run with ablated temporal context (zero first half)
                ablated_batch = batch_gpu.copy()
                if 'daily_features' in ablated_batch:
                    daily = ablated_batch['daily_features']
                    seq_len = daily.shape[1]
                    # Zero out first half of sequence
                    ablated_daily = daily.clone()
                    ablated_daily[:, :seq_len//2] = 0
                    ablated_batch['daily_features'] = ablated_daily

                current_weights.clear()
                try:
                    _ = self.model(**ablated_batch)
                    if current_weights:
                        ablated_context_weights.append(current_weights[0])
                except Exception as e:
                    logger.warning(f"Ablated context forward failed: {e}")
                    continue

        if hook_handle:
            hook_handle.remove()

        if not full_context_weights or not ablated_context_weights:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not capture gate weights for comparison.",
            )

        # Concatenate and compare
        full_weights = torch.cat(full_context_weights, dim=0)
        ablated_weights = torch.cat(ablated_context_weights, dim=0)

        # Compute difference metrics (focus on second half where we expect divergence)
        seq_len = full_weights.shape[1]
        second_half_start = seq_len // 2

        full_second = full_weights[:, second_half_start:]
        ablated_second = ablated_weights[:, second_half_start:]

        # Mean absolute difference
        mean_abs_diff = (full_second - ablated_second).abs().mean().item()

        # Cosine distance
        full_flat = full_second.reshape(-1, full_second.shape[-1])
        ablated_flat = ablated_second.reshape(-1, ablated_second.shape[-1])
        cosine_sim = F.cosine_similarity(full_flat, ablated_flat, dim=-1).mean().item()

        # KL divergence
        eps = 1e-8
        kl_div = (full_second * torch.log((full_second + eps) / (ablated_second + eps))).sum(dim=-1).mean().item()

        # Pass if there's meaningful difference
        passed = mean_abs_diff > 0.01 and cosine_sim < 0.99

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Weight comparison over time
        ax1 = axes[0]
        sample_full = full_weights[0, :, 0].numpy()  # First sample, first source
        sample_ablated = ablated_weights[0, :, 0].numpy()
        ax1.plot(sample_full, label='Full Context', alpha=0.7)
        ax1.plot(sample_ablated, label='Ablated Context', alpha=0.7)
        ax1.axvline(x=second_half_start, color='r', linestyle='--', label='Ablation Point')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Gate Weight (Source 0)')
        ax1.set_title('Gate Weights: Full vs Ablated Context')
        ax1.legend()

        # 2. Difference heatmap
        ax2 = axes[1]
        diff_per_source = (full_second - ablated_second).abs().mean(dim=(0, 1)).numpy()
        n_sources = len(diff_per_source)
        source_names = DAILY_SOURCES[:n_sources] if n_sources <= len(DAILY_SOURCES) else [f'S{i}' for i in range(n_sources)]
        ax2.bar(source_names, diff_per_source)
        ax2.set_xlabel('Source')
        ax2.set_ylabel('Mean Absolute Difference')
        ax2.set_title('Context Ablation Effect by Source')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Cosine similarity distribution
        ax3 = axes[2]
        per_timestep_sim = F.cosine_similarity(full_second, ablated_second, dim=-1)
        ax3.hist(per_timestep_sim.flatten().numpy(), bins=50, alpha=0.7)
        ax3.axvline(x=1.0, color='r', linestyle='--', label='Perfect Similarity')
        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Full vs Ablated Weight Similarity')
        ax3.legend()

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'temporal_context_effect')

        metrics = {
            'mean_abs_difference': mean_abs_diff,
            'cosine_similarity': cosine_sim,
            'kl_divergence': kl_div,
        }

        interpretation = (
            f"Temporal Context Effect Analysis:\n"
            f"- Mean absolute weight difference: {mean_abs_diff:.4f}\n"
            f"- Cosine similarity: {cosine_sim:.4f} (1.0 = identical)\n"
            f"- KL divergence: {kl_div:.4f}\n"
            f"\nConclusion: Temporal context {'significantly affects' if passed else 'has minimal effect on'} gate weights."
        )

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
        )


# =============================================================================
# 9.2 DailyTemporalEncoder Validation Probes
# =============================================================================

class MultiScalePatternProbe(ArchitectureProbe):
    """
    9.2.1: Tests if multi-scale convolutions capture patterns at different frequencies.

    The DailyTemporalEncoder uses convolutions with kernels of 3, 7, 14, 28 days.
    This probe checks if each scale captures distinct frequency content.
    """

    probe_id = "9.2.1"
    probe_name = "Multi-Scale Pattern Detection"
    tier = 3

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        # Hook to capture multi-scale conv outputs
        conv_outputs = defaultdict(list)

        def create_conv_hook(scale_idx):
            def hook(module, input, output):
                conv_outputs[scale_idx].append(output.detach().cpu())
            return hook

        hook_handles = []

        # Find DailyTemporalEncoder and hook its multi-scale convs
        if hasattr(self.model, 'daily_temporal_encoder'):
            encoder = self.model.daily_temporal_encoder
            if hasattr(encoder, 'multi_scale_conv'):
                for i, conv in enumerate(encoder.multi_scale_conv):
                    hook_handles.append(conv.register_forward_hook(create_conv_hook(i)))

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 10:
                    break

                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks', 'isw_embedding', 'isw_mask',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                try:
                    _ = self.model(**batch_gpu)
                except Exception:
                    continue

        for handle in hook_handles:
            handle.remove()

        if not conv_outputs:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not capture multi-scale conv outputs. DailyTemporalEncoder may not be present.",
            )

        # Analyze frequency content of each scale
        scale_names = ['3-day', '7-day', '14-day', '28-day']
        scale_frequencies = {}
        scale_correlations = np.zeros((len(conv_outputs), len(conv_outputs)))

        for scale_idx in conv_outputs:
            outputs = torch.cat(conv_outputs[scale_idx], dim=0)  # [batch, channels, seq]

            # Compute power spectrum to analyze frequency content
            outputs_np = outputs.numpy()
            power_spectra = []
            for b in range(min(outputs_np.shape[0], 10)):
                for c in range(min(outputs_np.shape[1], 5)):
                    fft = np.fft.fft(outputs_np[b, c])
                    power = np.abs(fft) ** 2
                    power_spectra.append(power[:len(power)//2])  # Positive frequencies only

            mean_power = np.mean(power_spectra, axis=0)
            # Find dominant frequency
            dominant_freq_idx = np.argmax(mean_power[1:]) + 1  # Skip DC component
            scale_frequencies[scale_idx] = {
                'dominant_freq_idx': int(dominant_freq_idx),
                'power_spectrum': mean_power.tolist(),
            }

        # Compute cross-scale correlation
        for i in conv_outputs:
            for j in conv_outputs:
                out_i = torch.cat(conv_outputs[i], dim=0).mean(dim=1).flatten()
                out_j = torch.cat(conv_outputs[j], dim=0).mean(dim=1).flatten()
                corr = np.corrcoef(out_i.numpy(), out_j.numpy())[0, 1]
                scale_correlations[i, j] = corr

        # Check if scales are sufficiently different
        off_diagonal_corrs = scale_correlations[np.triu_indices_from(scale_correlations, k=1)]
        mean_cross_corr = np.mean(np.abs(off_diagonal_corrs)) if len(off_diagonal_corrs) > 0 else 1.0

        # Scales should be moderately correlated but not identical
        passed = 0.3 < mean_cross_corr < 0.9

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Power spectra comparison
        ax1 = axes[0, 0]
        for scale_idx, freq_info in scale_frequencies.items():
            spectrum = freq_info['power_spectrum']
            if scale_idx < len(scale_names):
                ax1.plot(spectrum[:50], label=scale_names[scale_idx], alpha=0.7)
        ax1.set_xlabel('Frequency Index')
        ax1.set_ylabel('Power')
        ax1.set_title('Power Spectra by Convolution Scale')
        ax1.legend()
        ax1.set_yscale('log')

        # 2. Cross-scale correlation heatmap
        ax2 = axes[0, 1]
        n_scales = len(conv_outputs)
        labels = scale_names[:n_scales] if n_scales <= len(scale_names) else [f'Scale {i}' for i in range(n_scales)]
        im = ax2.imshow(scale_correlations, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(n_scales))
        ax2.set_yticks(range(n_scales))
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        ax2.set_title('Cross-Scale Correlation')
        plt.colorbar(im, ax=ax2)

        # 3. Sample outputs comparison
        ax3 = axes[1, 0]
        for scale_idx in list(conv_outputs.keys())[:4]:
            outputs = torch.cat(conv_outputs[scale_idx], dim=0)
            sample = outputs[0, 0].numpy()  # First sample, first channel
            if scale_idx < len(scale_names):
                ax3.plot(sample[:100], label=scale_names[scale_idx], alpha=0.7)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Activation')
        ax3.set_title('Sample Conv Outputs by Scale')
        ax3.legend()

        # 4. Variance per scale
        ax4 = axes[1, 1]
        scale_variances = []
        for scale_idx in sorted(conv_outputs.keys()):
            outputs = torch.cat(conv_outputs[scale_idx], dim=0)
            scale_variances.append(outputs.var().item())
        ax4.bar(labels[:len(scale_variances)], scale_variances)
        ax4.set_xlabel('Scale')
        ax4.set_ylabel('Output Variance')
        ax4.set_title('Activation Variance by Scale')

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'multi_scale_patterns')

        metrics = {
            'mean_cross_scale_correlation': mean_cross_corr,
            'n_scales': len(conv_outputs),
        }
        for i, var in enumerate(scale_variances):
            metrics[f'scale_{i}_variance'] = var

        interpretation = (
            f"Multi-Scale Pattern Analysis:\n"
            f"- Number of scales detected: {len(conv_outputs)}\n"
            f"- Mean cross-scale correlation: {mean_cross_corr:.3f}\n"
            f"  (Should be 0.3-0.9: moderate correlation indicates different but related patterns)\n"
            f"\nConclusion: Scales {'capture distinct patterns' if passed else 'may be redundant'}."
        )

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
            raw_data={'scale_frequencies': scale_frequencies},
        )


class DailyEnrichmentProbe(ArchitectureProbe):
    """
    9.2.3: Compares daily representations before and after the DailyTemporalEncoder.

    The encoder should enrich representations with temporal context.
    We measure:
    - Information gain (variance increase)
    - Temporal coherence improvement
    - Autocorrelation changes
    """

    probe_id = "9.2.3"
    probe_name = "Daily Temporal Enrichment"
    tier = 1

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        pre_encoder_reps = []
        post_encoder_reps = []

        # Hooks to capture before/after
        pre_hook_outputs = []
        post_hook_outputs = []

        def pre_hook(module, input, output):
            # Input to the temporal encoder
            if isinstance(input, tuple):
                pre_hook_outputs.append(input[0].detach().cpu())
            else:
                pre_hook_outputs.append(input.detach().cpu())

        def post_hook(module, input, output):
            post_hook_outputs.append(output.detach().cpu())

        pre_handle = None
        post_handle = None

        if hasattr(self.model, 'daily_temporal_encoder'):
            encoder = self.model.daily_temporal_encoder
            pre_handle = encoder.register_forward_pre_hook(lambda m, i: pre_hook(m, i, None))
            post_handle = encoder.register_forward_hook(post_hook)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 15:
                    break

                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks', 'isw_embedding', 'isw_mask',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                try:
                    _ = self.model(**batch_gpu)
                except Exception:
                    continue

        if pre_handle:
            pre_handle.remove()
        if post_handle:
            post_handle.remove()

        if not pre_hook_outputs or not post_hook_outputs:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not capture before/after representations. DailyTemporalEncoder may not be present.",
            )

        # Analyze differences
        pre_reps = torch.cat(pre_hook_outputs, dim=0)  # [total_samples, seq, d]
        post_reps = torch.cat(post_hook_outputs, dim=0)

        # 1. Variance comparison
        pre_variance = pre_reps.var(dim=-1).mean().item()
        post_variance = post_reps.var(dim=-1).mean().item()
        variance_ratio = post_variance / (pre_variance + 1e-8)

        # 2. Temporal autocorrelation
        def compute_autocorr(reps):
            reps_t = reps[:, :-1, :]
            reps_t1 = reps[:, 1:, :]
            return F.cosine_similarity(
                reps_t.reshape(-1, reps_t.shape[-1]),
                reps_t1.reshape(-1, reps_t1.shape[-1]),
                dim=-1
            ).mean().item()

        pre_autocorr = compute_autocorr(pre_reps)
        post_autocorr = compute_autocorr(post_reps)

        # 3. Representation magnitude
        pre_norm = pre_reps.norm(dim=-1).mean().item()
        post_norm = post_reps.norm(dim=-1).mean().item()

        # 4. Change magnitude
        diff = post_reps - pre_reps
        change_magnitude = diff.norm(dim=-1).mean().item()
        relative_change = change_magnitude / (pre_norm + 1e-8)

        # Pass criteria:
        # - Encoder should add information (relative_change > 0.1)
        # - Should maintain or improve temporal coherence
        passed = relative_change > 0.1 and post_autocorr >= pre_autocorr * 0.9

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Variance comparison
        ax1 = axes[0, 0]
        ax1.bar(['Before Encoder', 'After Encoder'], [pre_variance, post_variance])
        ax1.set_ylabel('Mean Variance')
        ax1.set_title(f'Representation Variance (ratio: {variance_ratio:.2f}x)')

        # 2. Autocorrelation comparison
        ax2 = axes[0, 1]
        ax2.bar(['Before Encoder', 'After Encoder'], [pre_autocorr, post_autocorr])
        ax2.set_ylabel('Temporal Autocorrelation')
        ax2.set_title('Temporal Coherence')

        # 3. Change magnitude over time
        ax3 = axes[1, 0]
        change_per_timestep = diff.norm(dim=-1).mean(dim=0).numpy()
        ax3.plot(change_per_timestep)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Change Magnitude')
        ax3.set_title('Encoder Effect Over Time')

        # 4. Sample representation comparison
        ax4 = axes[1, 1]
        sample_pre = pre_reps[0, :, 0].numpy()
        sample_post = post_reps[0, :, 0].numpy()
        ax4.plot(sample_pre[:100], label='Before', alpha=0.7)
        ax4.plot(sample_post[:100], label='After', alpha=0.7)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Representation (dim 0)')
        ax4.set_title('Sample Representation Before/After')
        ax4.legend()

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'daily_enrichment')

        metrics = {
            'pre_variance': pre_variance,
            'post_variance': post_variance,
            'variance_ratio': variance_ratio,
            'pre_autocorrelation': pre_autocorr,
            'post_autocorrelation': post_autocorr,
            'relative_change': relative_change,
            'change_magnitude': change_magnitude,
        }

        interpretation = (
            f"Daily Temporal Enrichment Analysis:\n"
            f"- Pre-encoder variance: {pre_variance:.4f}\n"
            f"- Post-encoder variance: {post_variance:.4f} ({variance_ratio:.2f}x)\n"
            f"- Pre-encoder autocorrelation: {pre_autocorr:.3f}\n"
            f"- Post-encoder autocorrelation: {post_autocorr:.3f}\n"
            f"- Relative change: {relative_change:.1%}\n"
            f"\nConclusion: Encoder {'enriches' if passed else 'does not significantly enrich'} daily representations."
        )

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
        )


# =============================================================================
# 9.3 ISWAlignmentModule Validation Probes
# =============================================================================

class AlignmentQualityProbe(ArchitectureProbe):
    """
    9.3.1: Tests the quality of alignment between model and ISW representations.

    Measures cosine similarity in the shared projection space to verify
    that the contrastive learning objective is working.
    """

    probe_id = "9.3.1"
    probe_name = "ISW Alignment Quality"
    tier = 1

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        # Check if model has ISW alignment
        if not hasattr(self.model, 'isw_alignment') or self.model.isw_alignment is None:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Model does not have ISW alignment module enabled.",
            )

        model_projections = []
        isw_projections = []

        # Load ISW embeddings
        isw_embedding_dir = PROJECT_ROOT / "data" / "wayback_archives" / "isw_assessments" / "embeddings"

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 20:
                    break

                # Check if batch has ISW embeddings
                if 'isw_embedding' not in batch or batch['isw_embedding'] is None:
                    continue

                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                # Get ISW embedding separately
                isw_emb = batch.get('isw_embedding')
                isw_mask = batch.get('isw_mask')

                if isw_emb is None:
                    continue

                isw_emb = isw_emb.to(self.device) if isinstance(isw_emb, Tensor) else isw_emb

                # Remove ISW from batch for forward pass
                batch_for_model = {k: v for k, v in batch_gpu.items()
                                   if k not in ['isw_embedding', 'isw_mask']}

                try:
                    outputs = self.model(**batch_for_model)

                    # Get model representation from outputs
                    if hasattr(outputs, 'fused_repr'):
                        model_repr = outputs.fused_repr
                    elif 'fused_repr' in outputs:
                        model_repr = outputs['fused_repr']
                    else:
                        # Use cross_resolution_fused as fallback
                        model_repr = outputs.get('cross_resolution_fused')

                    if model_repr is None:
                        continue

                    # Project through ISW alignment module
                    model_proj, isw_proj = self.model.isw_alignment(model_repr, isw_emb, isw_mask)

                    model_projections.append(model_proj.cpu())
                    isw_projections.append(isw_proj.cpu())

                except Exception as e:
                    logger.warning(f"ISW alignment forward failed: {e}")
                    continue

        if not model_projections:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not compute ISW alignments. Check if ISW embeddings are available in data.",
            )

        # Concatenate projections
        model_proj = torch.cat(model_projections, dim=0)  # [total_samples, seq, proj_dim]
        isw_proj = torch.cat(isw_projections, dim=0)

        # Compute alignment metrics
        # 1. Per-timestep cosine similarity
        cosine_sim = F.cosine_similarity(model_proj, isw_proj, dim=-1)  # [samples, seq]
        mean_cosine = cosine_sim.mean().item()
        std_cosine = cosine_sim.std().item()

        # 2. Compare same-timestep similarity vs. random pairing
        random_indices = torch.randperm(isw_proj.shape[1])
        random_isw = isw_proj[:, random_indices]
        random_cosine = F.cosine_similarity(model_proj, random_isw, dim=-1).mean().item()

        # Alignment margin: same-timestep similarity should be higher than random
        alignment_margin = mean_cosine - random_cosine

        # 3. L2 distance in projection space
        l2_dist = (model_proj - isw_proj).norm(dim=-1).mean().item()

        # Pass criteria: alignment should be better than random
        passed = alignment_margin > 0.05 and mean_cosine > 0.3

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cosine similarity distribution
        ax1 = axes[0, 0]
        ax1.hist(cosine_sim.flatten().numpy(), bins=50, alpha=0.7, label='Same Timestep')
        ax1.axvline(x=mean_cosine, color='r', linestyle='--', label=f'Mean: {mean_cosine:.3f}')
        ax1.axvline(x=random_cosine, color='g', linestyle='--', label=f'Random: {random_cosine:.3f}')
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Alignment Quality Distribution')
        ax1.legend()

        # 2. Alignment over time
        ax2 = axes[0, 1]
        mean_per_timestep = cosine_sim.mean(dim=0).numpy()
        ax2.plot(mean_per_timestep)
        ax2.axhline(y=mean_cosine, color='r', linestyle='--', label='Overall Mean')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Mean Cosine Similarity')
        ax2.set_title('Alignment Quality Over Time')
        ax2.legend()

        # 3. Projection space visualization (2D projection)
        ax3 = axes[1, 0]
        from sklearn.decomposition import PCA

        # Flatten and sample for visualization
        model_flat = model_proj.reshape(-1, model_proj.shape[-1]).numpy()
        isw_flat = isw_proj.reshape(-1, isw_proj.shape[-1]).numpy()

        n_samples = min(500, model_flat.shape[0])
        sample_idx = np.random.choice(model_flat.shape[0], n_samples, replace=False)

        combined = np.concatenate([model_flat[sample_idx], isw_flat[sample_idx]], axis=0)
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)

        ax3.scatter(combined_2d[:n_samples, 0], combined_2d[:n_samples, 1],
                   alpha=0.5, label='Model', s=10)
        ax3.scatter(combined_2d[n_samples:, 0], combined_2d[n_samples:, 1],
                   alpha=0.5, label='ISW', s=10)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('Projection Space (PCA)')
        ax3.legend()

        # 4. Alignment margin summary
        ax4 = axes[1, 1]
        bars = ax4.bar(['Same Timestep', 'Random Pairing'], [mean_cosine, random_cosine])
        bars[0].set_color('green')
        bars[1].set_color('gray')
        ax4.set_ylabel('Mean Cosine Similarity')
        ax4.set_title(f'Alignment Margin: {alignment_margin:.3f}')

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'isw_alignment_quality')

        metrics = {
            'mean_cosine_similarity': mean_cosine,
            'std_cosine_similarity': std_cosine,
            'random_baseline': random_cosine,
            'alignment_margin': alignment_margin,
            'mean_l2_distance': l2_dist,
        }

        interpretation = (
            f"ISW Alignment Quality Analysis:\n"
            f"- Mean cosine similarity: {mean_cosine:.3f} +/- {std_cosine:.3f}\n"
            f"- Random baseline: {random_cosine:.3f}\n"
            f"- Alignment margin: {alignment_margin:.3f}\n"
            f"- Mean L2 distance: {l2_dist:.3f}\n"
            f"\nConclusion: ISW alignment {'is effective' if passed else 'needs improvement'}."
        )

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
        )


class ConflictPhaseAlignmentProbe(ArchitectureProbe):
    """
    9.3.3: Tests if alignment quality varies across conflict phases.

    Different phases (Initial Invasion, Stalemate, Counteroffensive, Attritional)
    may have different ISW narrative characteristics. This probe checks if
    alignment is consistent or phase-dependent.
    """

    probe_id = "9.3.3"
    probe_name = "Conflict Phase Alignment"
    tier = 3

    # Key phase transition dates
    PHASE_DATES = {
        'Initial Invasion': (datetime(2022, 2, 24), datetime(2022, 4, 1)),
        'Stalemate': (datetime(2022, 4, 2), datetime(2022, 8, 31)),
        'Counteroffensive': (datetime(2022, 9, 1), datetime(2022, 11, 30)),
        'Attritional': (datetime(2022, 12, 1), datetime(2024, 12, 31)),
    }

    def run(self) -> ArchitectureProbeResult:
        logger.info(f"Running {self.probe_name} probe...")

        self.model.eval()

        if not hasattr(self.model, 'isw_alignment') or self.model.isw_alignment is None:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Model does not have ISW alignment module enabled.",
            )

        phase_alignments = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= 30:
                    break

                if 'isw_embedding' not in batch or batch['isw_embedding'] is None:
                    continue

                # Get dates for this batch if available
                dates = batch.get('daily_dates')
                if dates is None:
                    continue

                batch_gpu = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                    if k not in ['forecast_targets', 'forecast_masks',
                                 'target', 'regime_target', 'casualty_target', 'anomaly_target']
                }

                if 'month_boundary_indices' in batch_gpu:
                    batch_gpu['month_boundaries'] = batch_gpu.pop('month_boundary_indices')

                isw_emb = batch['isw_embedding'].to(self.device)
                isw_mask = batch.get('isw_mask')

                batch_for_model = {k: v for k, v in batch_gpu.items()
                                   if k not in ['isw_embedding', 'isw_mask', 'daily_dates']}

                try:
                    outputs = self.model(**batch_for_model)

                    if hasattr(outputs, 'fused_repr'):
                        model_repr = outputs.fused_repr
                    elif 'fused_repr' in outputs:
                        model_repr = outputs['fused_repr']
                    else:
                        model_repr = outputs.get('cross_resolution_fused')

                    if model_repr is None:
                        continue

                    model_proj, isw_proj = self.model.isw_alignment(model_repr, isw_emb, isw_mask)
                    cosine_sim = F.cosine_similarity(model_proj, isw_proj, dim=-1)

                    # Assign alignments to phases based on dates
                    for sample_idx in range(len(dates)):
                        sample_dates = dates[sample_idx]
                        for t, date in enumerate(sample_dates):
                            if isinstance(date, str):
                                date = datetime.strptime(date, '%Y-%m-%d')

                            for phase, (start, end) in self.PHASE_DATES.items():
                                if start <= date <= end:
                                    phase_alignments[phase].append(cosine_sim[sample_idx, t].item())
                                    break

                except Exception as e:
                    logger.warning(f"Phase alignment forward failed: {e}")
                    continue

        if not phase_alignments:
            return ArchitectureProbeResult(
                probe_id=self.probe_id,
                probe_name=self.probe_name,
                passed=False,
                metrics={},
                interpretation="Could not compute phase-specific alignments.",
            )

        # Compute per-phase statistics
        phase_stats = {}
        for phase, alignments in phase_alignments.items():
            if alignments:
                phase_stats[phase] = {
                    'mean': np.mean(alignments),
                    'std': np.std(alignments),
                    'count': len(alignments),
                }

        # Check for significant phase differences
        phase_means = [stats['mean'] for stats in phase_stats.values()]
        if len(phase_means) >= 2:
            mean_diff = max(phase_means) - min(phase_means)
            passed = mean_diff < 0.2  # Alignment should be consistent
        else:
            mean_diff = 0
            passed = True

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Phase alignment box plots
        ax1 = axes[0]
        phase_data = [phase_alignments[phase] for phase in self.PHASE_DATES.keys()
                      if phase in phase_alignments]
        phase_labels = [phase for phase in self.PHASE_DATES.keys() if phase in phase_alignments]

        if phase_data:
            ax1.boxplot(phase_data, labels=phase_labels)
            ax1.set_ylabel('Cosine Similarity')
            ax1.set_title('ISW Alignment by Conflict Phase')
            ax1.tick_params(axis='x', rotation=15)

        # 2. Mean alignment comparison
        ax2 = axes[1]
        if phase_stats:
            means = [phase_stats[p]['mean'] for p in phase_labels if p in phase_stats]
            stds = [phase_stats[p]['std'] for p in phase_labels if p in phase_stats]
            x_pos = range(len(means))
            ax2.bar(x_pos, means, yerr=stds, capsize=5)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(phase_labels, rotation=15)
            ax2.set_ylabel('Mean Cosine Similarity')
            ax2.set_title('Mean Alignment by Phase')

        plt.tight_layout()
        fig_path = self._save_figure(fig, 'conflict_phase_alignment')

        metrics = {
            'phase_difference': mean_diff,
        }
        for phase, stats in phase_stats.items():
            metrics[f'{phase.lower().replace(" ", "_")}_mean'] = stats['mean']
            metrics[f'{phase.lower().replace(" ", "_")}_count'] = stats['count']

        interpretation = (
            f"Conflict Phase Alignment Analysis:\n"
        )
        for phase, stats in phase_stats.items():
            interpretation += f"- {phase}: {stats['mean']:.3f} +/- {stats['std']:.3f} (n={stats['count']})\n"
        interpretation += f"\nMax phase difference: {mean_diff:.3f}\n"
        interpretation += f"Conclusion: Alignment is {'consistent' if passed else 'variable'} across phases."

        return ArchitectureProbeResult(
            probe_id=self.probe_id,
            probe_name=self.probe_name,
            passed=passed,
            metrics=metrics,
            interpretation=interpretation,
            visualizations=[fig_path],
        )


# =============================================================================
# Probe Registry
# =============================================================================

ARCHITECTURE_PROBES = {
    # Section 9.1: TemporalSourceGate
    '9.1.1': GateWeightDynamicsProbe,
    '9.1.2': TemporalContextEffectProbe,

    # Section 9.2: DailyTemporalEncoder
    '9.2.1': MultiScalePatternProbe,
    '9.2.3': DailyEnrichmentProbe,

    # Section 9.3: ISWAlignmentModule
    '9.3.1': AlignmentQualityProbe,
    '9.3.3': ConflictPhaseAlignmentProbe,
}


def run_architecture_probes(
    model: MultiResolutionHAN,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Optional[Path] = None,
    probe_ids: Optional[List[str]] = None,
) -> Dict[str, ArchitectureProbeResult]:
    """
    Run architecture validation probes.

    Args:
        model: The MultiResolutionHAN model to probe
        dataloader: DataLoader providing batches
        device: Device for computation
        output_dir: Directory for outputs
        probe_ids: Optional list of specific probe IDs to run

    Returns:
        Dictionary mapping probe IDs to results
    """
    results = {}

    probe_classes = ARCHITECTURE_PROBES
    if probe_ids:
        probe_classes = {k: v for k, v in ARCHITECTURE_PROBES.items() if k in probe_ids}

    for probe_id, probe_class in probe_classes.items():
        logger.info(f"Running probe {probe_id}: {probe_class.probe_name}")
        try:
            probe = probe_class(model, dataloader, device, output_dir)
            result = probe.run()
            results[probe_id] = result

            status = "PASS" if result.passed else "FAIL"
            logger.info(f"  [{status}] {result.probe_name}")

        except Exception as e:
            logger.error(f"  [ERROR] {probe_id}: {e}")
            results[probe_id] = ArchitectureProbeResult(
                probe_id=probe_id,
                probe_name=probe_class.probe_name,
                passed=False,
                metrics={},
                interpretation=f"Probe failed with error: {e}",
            )

    return results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run architecture validation probes")
    parser.add_argument("--checkpoint", type=str,
                       default=str(MULTI_RES_CHECKPOINT_DIR / "best_checkpoint.pt"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--probes", nargs="+", default=None)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print(f"Loading model from {args.checkpoint}")
    # Model and data loading would go here
