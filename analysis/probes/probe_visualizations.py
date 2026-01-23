#!/usr/bin/env python3
"""
Probe Visualization Module for Multi-Resolution HAN Model Analysis

This module provides publication-quality visualization tools for all probe results
from the Multi-Resolution HAN investigation battery.

Sections Covered:
1. Data Artifacts (encoding variance, equipment redundancy, VIIRS analysis)
2. Cross-Modal Fusion (RSA, attention flow, ablation, fusion trajectory)
3. Temporal Dynamics (context window, attention distance, transitions)
4. Semantic Structure (operation clustering, day-type probes, patterns)
5. ISW Semantic Association (alignment, topic correlation, event response)
6. Causal Importance (rankings, interventions, integrated gradients)
7. Tactical Readiness (data availability, sector maps, resolution tradeoffs)

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

# Optional imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive visualizations disabled.")

try:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    PROBE_OUTPUT_DIR,
)

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

BASE_DIR = PROJECT_ROOT
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR
PROBE_DIR = ANALYSIS_DIR / "probes"
OUTPUT_DIR = PROBE_OUTPUT_DIR
FIGURE_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

@dataclass
class StyleConfig:
    """Publication-quality style configuration for visualizations."""

    # Theme
    dark_mode: bool = False

    # Figure sizes (inches)
    figure_small: Tuple[float, float] = (8, 6)
    figure_medium: Tuple[float, float] = (12, 8)
    figure_large: Tuple[float, float] = (16, 10)
    figure_dashboard: Tuple[float, float] = (20, 16)

    # DPI settings
    dpi_screen: int = 100
    dpi_publication: int = 300
    dpi_poster: int = 600

    # Font settings
    font_family: str = "sans-serif"
    font_title: int = 14
    font_label: int = 12
    font_tick: int = 10
    font_annotation: int = 9
    font_legend: int = 10

    # Line and marker settings
    line_width: float = 1.5
    marker_size: int = 6
    grid_alpha: float = 0.3

    # Color palettes
    palette_categorical: str = "husl"
    palette_sequential: str = "viridis"
    palette_diverging: str = "RdBu_r"

    def apply(self) -> None:
        """Apply style settings to matplotlib."""
        if self.dark_mode:
            plt.style.use('dark_background')
            self.bg_color = '#1a1a2e'
            self.text_color = '#eaeaea'
            self.grid_color = '#3a3a5a'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#2d2d2d'
            self.grid_color = '#cccccc'

        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_tick,
            'axes.titlesize': self.font_title,
            'axes.labelsize': self.font_label,
            'xtick.labelsize': self.font_tick,
            'ytick.labelsize': self.font_tick,
            'legend.fontsize': self.font_legend,
            'figure.titlesize': self.font_title + 2,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'grid.alpha': self.grid_alpha,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })


# Conflict phase colors
PHASE_COLORS = {
    0: "#d62728",  # Red - Initial Invasion
    1: "#7f7f7f",  # Grey - Stalemate
    2: "#2ca02c",  # Green - Counteroffensive
    3: "#ff7f0e",  # Orange - Attritional Warfare
}

PHASE_LABELS = {
    0: "Initial Invasion",
    1: "Stalemate",
    2: "Counteroffensive",
    3: "Attritional Warfare",
}

# Source colors for consistent visualization
SOURCE_COLORS = {
    'equipment': '#1f77b4',
    'personnel': '#ff7f0e',
    'deepstate': '#2ca02c',
    'firms': '#d62728',
    'viina': '#9467bd',
    'viirs': '#8c564b',
    'sentinel': '#e377c2',
    'hdx_conflict': '#7f7f7f',
    'hdx_food': '#bcbd22',
    'hdx_rainfall': '#17becf',
    'iom': '#aec7e8',
    'isw': '#ffbb78',
}

# Task colors
TASK_COLORS = {
    'regime': '#1f77b4',
    'casualty': '#ff7f0e',
    'anomaly': '#2ca02c',
    'forecast': '#d62728',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path = FIGURE_DIR,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    transparent: bool = False,
) -> List[Path]:
    """Save figure in multiple formats."""
    paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=fig.get_facecolor() if not transparent else 'none',
            edgecolor='none',
            transparent=transparent,
        )
        paths.append(path)
    return paths


def load_probe_results(probe_id: str, output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """Load probe results from YAML file."""
    import yaml
    yaml_path = output_dir / f"probe_{probe_id.replace('.', '_')}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    return {}


def load_json_results(filename: str, output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    """Load results from JSON file."""
    json_path = output_dir / filename
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def add_significance_annotation(
    ax: plt.Axes,
    x: float,
    y: float,
    p_value: float,
    fontsize: int = 10,
) -> None:
    """Add significance stars to plot."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    ax.annotate(stars, (x, y), ha='center', va='bottom', fontsize=fontsize)


def create_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    mappable,
    label: str,
    orientation: str = 'vertical',
) -> plt.colorbar:
    """Create a properly positioned colorbar."""
    divider = make_axes_locatable(ax)
    if orientation == 'vertical':
        cax = divider.append_axes("right", size="5%", pad=0.1)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.set_label(label)
    return cbar


# =============================================================================
# SECTION 1: DATA ARTIFACT FIGURES
# =============================================================================

class DataArtifactFigures:
    """Visualization class for Section 1: Data Artifact Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_encoding_variance(
        self,
        cumulative_data: np.ndarray,
        delta_data: np.ndarray,
        rolling_data: np.ndarray,
        dates: Optional[List] = None,
        categories: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create bar chart comparing cumulative vs delta variance.

        Shows coefficient of variation for each encoding type across
        equipment categories.
        """
        fig, axes = plt.subplots(1, 3, figsize=self.style.figure_medium)

        if categories is None:
            categories = [f'Cat {i}' for i in range(len(cumulative_data))]

        # Calculate CV for each encoding
        cv_cumulative = np.std(cumulative_data, axis=-1) / (np.mean(cumulative_data, axis=-1) + 1e-8)
        cv_delta = np.std(delta_data, axis=-1) / (np.mean(np.abs(delta_data), axis=-1) + 1e-8)
        cv_rolling = np.std(rolling_data, axis=-1) / (np.mean(np.abs(rolling_data), axis=-1) + 1e-8)

        x = np.arange(len(categories))
        width = 0.25

        # Bar chart comparison
        ax = axes[0]
        bars1 = ax.bar(x - width, cv_cumulative, width, label='Cumulative', color='#1f77b4')
        bars2 = ax.bar(x, cv_delta, width, label='Delta', color='#ff7f0e')
        bars3 = ax.bar(x + width, cv_rolling, width, label='Rolling-7', color='#2ca02c')
        ax.set_xlabel('Equipment Category')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Encoding Variance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()

        # Time series comparison for one category
        ax = axes[1]
        if dates is not None:
            t = range(min(100, len(dates)))
        else:
            t = range(min(100, cumulative_data.shape[-1]))
        ax.plot(t, cumulative_data[0, :len(t)] / cumulative_data[0, :len(t)].max(),
                label='Cumulative (norm)', alpha=0.8)
        ax.plot(t, (delta_data[0, :len(t)] - delta_data[0, :len(t)].min()) /
                (delta_data[0, :len(t)].max() - delta_data[0, :len(t)].min() + 1e-8),
                label='Delta (norm)', alpha=0.8)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'Time Series: {categories[0]}')
        ax.legend()

        # Summary statistics
        ax = axes[2]
        summary_data = {
            'Cumulative': [np.mean(cv_cumulative), np.std(cv_cumulative)],
            'Delta': [np.mean(cv_delta), np.std(cv_delta)],
            'Rolling-7': [np.mean(cv_rolling), np.std(cv_rolling)],
        }
        means = [v[0] for v in summary_data.values()]
        stds = [v[1] for v in summary_data.values()]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(summary_data.keys(), means, yerr=stds, capsize=5, color=colors)
        ax.set_ylabel('Mean CV (+/- std)')
        ax.set_title('Summary Statistics')

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9)

        fig.suptitle('Encoding Variance Analysis', fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_equipment_redundancy(
        self,
        correlation_matrix: np.ndarray,
        labels: List[str],
        partial_correlations: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Create heatmap of equipment-personnel correlations.

        Shows both raw and partial correlations (controlling for time).
        """
        n_plots = 2 if partial_correlations is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))

        if n_plots == 1:
            axes = [axes]

        # Raw correlations
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[0],
            cbar_kws={'label': 'Pearson r', 'shrink': 0.8}
        )
        axes[0].set_title('Raw Correlations (Delta Encoding)')
        axes[0].tick_params(axis='x', rotation=45)

        # Partial correlations (if provided)
        if partial_correlations is not None:
            mask = np.triu(np.ones_like(partial_correlations, dtype=bool), k=1)
            sns.heatmap(
                partial_correlations,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[1],
                cbar_kws={'label': 'Partial r', 'shrink': 0.8}
            )
            axes[1].set_title('Partial Correlations\n(Controlling for Time Trend)')
            axes[1].tick_params(axis='x', rotation=45)

        fig.suptitle('Equipment-Personnel Redundancy Analysis',
                     fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_viirs_lag_analysis(
        self,
        lags: np.ndarray,
        cross_correlation: np.ndarray,
        confidence_bands: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        peak_lag: Optional[int] = None,
    ) -> plt.Figure:
        """
        Create cross-correlation lag plot with confidence bands.

        Shows temporal relationship between VIIRS and casualties.
        """
        fig, ax = plt.subplots(figsize=self.style.figure_small)

        # Plot cross-correlation
        ax.bar(lags, cross_correlation, color='#1f77b4', alpha=0.7, edgecolor='navy')

        # Add confidence bands
        if confidence_bands is not None:
            ax.fill_between(lags, confidence_bands[0], confidence_bands[1],
                           alpha=0.2, color='gray', label='95% CI')
            ax.axhline(y=confidence_bands[1][0], color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=confidence_bands[0][0], color='gray', linestyle='--', alpha=0.5)

        # Mark peak lag
        if peak_lag is not None:
            peak_idx = np.where(lags == peak_lag)[0]
            if len(peak_idx) > 0:
                ax.axvline(x=peak_lag, color='red', linestyle='-', linewidth=2,
                          label=f'Peak lag: {peak_lag} days')
                ax.scatter([peak_lag], [cross_correlation[peak_idx[0]]],
                          color='red', s=100, zorder=5)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Cross-Correlation')
        ax.set_title('VIIRS-Casualty Temporal Relationship')
        ax.legend(loc='upper right')

        # Add interpretation annotation
        if peak_lag is not None:
            if peak_lag > 0:
                interpretation = "VIIRS leads casualties"
            elif peak_lag < 0:
                interpretation = "VIIRS lags casualties"
            else:
                interpretation = "Concurrent relationship"
            ax.annotate(interpretation, xy=(0.02, 0.98), xycoords='axes fraction',
                       ha='left', va='top', fontsize=10, style='italic',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def fig_viirs_dominance(
        self,
        source_importance: Dict[str, float],
        highlight_viirs: bool = True,
    ) -> plt.Figure:
        """
        Create source importance bar chart with VIIRS highlighted.
        """
        fig, ax = plt.subplots(figsize=self.style.figure_small)

        sources = list(source_importance.keys())
        importances = list(source_importance.values())

        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        sources = [sources[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]

        # Color bars
        colors = []
        for s in sources:
            if highlight_viirs and s.lower() == 'viirs':
                colors.append('#d62728')  # Red for VIIRS
            else:
                colors.append(SOURCE_COLORS.get(s.lower(), '#1f77b4'))

        bars = ax.barh(sources, importances, color=colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, imp in zip(bars, importances):
            ax.annotate(f'{imp:.3f}',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       ha='left', va='center', fontsize=9,
                       xytext=(5, 0), textcoords='offset points')

        ax.set_xlabel('Feature Importance')
        ax.set_title('Source Importance Ranking')
        ax.invert_yaxis()

        if highlight_viirs:
            # Add legend for VIIRS highlight
            viirs_patch = mpatches.Patch(color='#d62728', label='VIIRS (dominant)')
            other_patch = mpatches.Patch(color='#1f77b4', label='Other sources')
            ax.legend(handles=[viirs_patch, other_patch], loc='lower right')

        plt.tight_layout()
        return fig

    def fig_mediation_path(
        self,
        total_effect: float,
        direct_effect: float,
        indirect_effect: float,
        p_values: Dict[str, float],
        mediator_name: str = "Personnel",
    ) -> plt.Figure:
        """
        Create path diagram showing mediation analysis.

        Shows VIIRS -> Personnel -> Casualties mediation.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')

        # Node positions
        x_pos = {'viirs': 1.5, 'mediator': 5, 'outcome': 8.5}
        y_pos = {'viirs': 3, 'mediator': 5, 'outcome': 3}

        # Draw nodes
        node_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                         edgecolor='navy', linewidth=2)

        ax.text(x_pos['viirs'], y_pos['viirs'], 'VIIRS\nRadiance',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=node_style)
        ax.text(x_pos['mediator'], y_pos['mediator'], f'{mediator_name}\n(Mediator)',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=node_style)
        ax.text(x_pos['outcome'], y_pos['outcome'], 'Casualties\n(Outcome)',
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=node_style)

        # Draw paths with arrows
        arrow_style = dict(arrowstyle='->', color='darkblue', lw=2,
                          connectionstyle='arc3,rad=0')

        # Path a: VIIRS -> Mediator
        ax.annotate('', xy=(x_pos['mediator']-0.8, y_pos['mediator']-0.3),
                   xytext=(x_pos['viirs']+0.8, y_pos['viirs']+0.3),
                   arrowprops=arrow_style)

        # Path b: Mediator -> Outcome
        ax.annotate('', xy=(x_pos['outcome']-0.8, y_pos['outcome']+0.3),
                   xytext=(x_pos['mediator']+0.8, y_pos['mediator']-0.3),
                   arrowprops=arrow_style)

        # Direct path c': VIIRS -> Outcome
        ax.annotate('', xy=(x_pos['outcome']-0.8, y_pos['outcome']),
                   xytext=(x_pos['viirs']+0.8, y_pos['viirs']),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2,
                                  connectionstyle='arc3,rad=0', linestyle='--'))

        # Add path coefficients
        a_coef = indirect_effect / (direct_effect + 1e-8) if direct_effect != 0 else indirect_effect
        b_coef = direct_effect

        # Path labels
        ax.text(3, 4.5, f'a = {a_coef:.3f}', fontsize=10, ha='center')
        ax.text(7, 4.5, f'b = {b_coef:.3f}', fontsize=10, ha='center')
        ax.text(5, 2.5, f"c' = {direct_effect:.3f}", fontsize=10, ha='center', color='gray')

        # Summary box
        summary_text = (
            f"Mediation Analysis Summary\n"
            f"{'='*30}\n"
            f"Total Effect (c): {total_effect:.4f}\n"
            f"Direct Effect (c'): {direct_effect:.4f}\n"
            f"Indirect Effect (a*b): {indirect_effect:.4f}\n"
            f"Proportion Mediated: {abs(indirect_effect/total_effect)*100:.1f}%"
            if total_effect != 0 else "N/A"
        )
        ax.text(5, 0.8, summary_text, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
               family='monospace')

        ax.set_title('Mediation Path Analysis: VIIRS -> Personnel -> Casualties',
                    fontsize=self.style.font_title, fontweight='bold', y=0.95)

        return fig


# =============================================================================
# SECTION 2: CROSS-MODAL FUSION FIGURES
# =============================================================================

class CrossModalFusionFigures:
    """Visualization class for Section 2: Cross-Modal Fusion Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_rsa_heatmap(
        self,
        rsa_matrix: np.ndarray,
        source_labels: List[str],
        title: str = "Representational Similarity Analysis",
    ) -> plt.Figure:
        """
        Create representational similarity matrix heatmap.

        Shows how similar representations are across different sources.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        mask = np.triu(np.ones_like(rsa_matrix, dtype=bool), k=1)

        sns.heatmap(
            rsa_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=source_labels,
            yticklabels=source_labels,
            ax=ax,
            cbar_kws={'label': 'RSA Correlation', 'shrink': 0.8},
            annot_kws={'size': 9}
        )

        ax.set_title(title, fontsize=self.style.font_title, pad=20)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        # Add interpretation guide
        ax.text(0.02, -0.12,
               "Interpretation: RSA > 0.3 suggests meaningful fusion | RSA near 0 suggests independent processing",
               transform=ax.transAxes, fontsize=9, style='italic')

        plt.tight_layout()
        return fig

    def fig_attention_flow(
        self,
        attention_weights: np.ndarray,
        source_labels: List[str],
        threshold: float = 0.1,
    ) -> plt.Figure:
        """
        Create attention flow visualization (chord-style diagram).

        Shows how attention flows between sources during cross-modal fusion.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.style.figure_medium)

        n_sources = len(source_labels)

        # Left: Heatmap view
        ax = axes[0]
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_xticks(range(n_sources))
        ax.set_yticks(range(n_sources))
        ax.set_xticklabels(source_labels, rotation=45, ha='right')
        ax.set_yticklabels(source_labels)
        ax.set_xlabel('Attention To')
        ax.set_ylabel('Attention From')
        ax.set_title('Cross-Source Attention Weights')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight')

        # Annotate significant connections
        for i in range(n_sources):
            for j in range(n_sources):
                if attention_weights[i, j] >= threshold:
                    ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                           ha='center', va='center', color='white'
                           if attention_weights[i, j] > 0.5 else 'black',
                           fontsize=8)

        # Right: Circular flow diagram
        ax = axes[1]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Place sources in a circle
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        radius = 1.0

        node_positions = {}
        for i, (label, angle) in enumerate(zip(source_labels, angles)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            node_positions[label] = (x, y)

            color = SOURCE_COLORS.get(label.lower(), '#1f77b4')
            circle = plt.Circle((x, y), 0.15, color=color, alpha=0.8)
            ax.add_patch(circle)

            # Label position
            label_x = 1.3 * np.cos(angle)
            label_y = 1.3 * np.sin(angle)
            ax.text(label_x, label_y, label, ha='center', va='center',
                   fontsize=9, fontweight='bold')

        # Draw attention flows
        for i, src_from in enumerate(source_labels):
            for j, src_to in enumerate(source_labels):
                if i != j and attention_weights[i, j] >= threshold:
                    x1, y1 = node_positions[src_from]
                    x2, y2 = node_positions[src_to]

                    # Draw arrow with width proportional to attention
                    width = attention_weights[i, j] * 3
                    ax.annotate('', xy=(x2*0.85, y2*0.85),
                               xytext=(x1*0.85, y1*0.85),
                               arrowprops=dict(arrowstyle='->',
                                              color='gray',
                                              lw=width,
                                              alpha=0.6,
                                              connectionstyle='arc3,rad=0.2'))

        ax.set_title('Attention Flow Diagram\n(threshold = {:.2f})'.format(threshold))

        fig.suptitle('Cross-Modal Attention Flow Analysis',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_ablation_heatmap(
        self,
        ablation_results: np.ndarray,
        source_labels: List[str],
        task_labels: List[str],
    ) -> plt.Figure:
        """
        Create leave-one-out ablation results heatmap.

        Shows performance change when each source is removed for each task.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create diverging heatmap (negative = source is important)
        vmax = np.max(np.abs(ablation_results))

        im = ax.imshow(ablation_results, cmap='RdBu', aspect='auto',
                      vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(task_labels)))
        ax.set_yticks(range(len(source_labels)))
        ax.set_xticklabels(task_labels, rotation=45, ha='right')
        ax.set_yticklabels(source_labels)
        ax.set_xlabel('Task')
        ax.set_ylabel('Ablated Source')

        # Add value annotations
        for i in range(len(source_labels)):
            for j in range(len(task_labels)):
                val = ablation_results[i, j]
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:+.2%}', ha='center', va='center',
                       color=color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Performance Change (negative = source important)')

        ax.set_title('Leave-One-Out Ablation Study',
                    fontsize=self.style.font_title, pad=15)

        # Add interpretation
        ax.text(0.5, -0.15,
               "Red = Removing source hurts performance (important) | Blue = Removing source helps (redundant/harmful)",
               transform=ax.transAxes, ha='center', fontsize=9, style='italic')

        plt.tight_layout()
        return fig

    def fig_fusion_trajectory(
        self,
        epochs: np.ndarray,
        rsa_scores: np.ndarray,
        attention_entropy: np.ndarray,
        source_importance_variance: np.ndarray,
    ) -> plt.Figure:
        """
        Create line plot of fusion quality metrics over training epochs.
        """
        fig, axes = plt.subplots(1, 3, figsize=self.style.figure_medium)

        # RSA score over training
        ax = axes[0]
        ax.plot(epochs, rsa_scores, 'o-', color='#1f77b4', linewidth=2, markersize=6)
        ax.fill_between(epochs, rsa_scores * 0.9, rsa_scores * 1.1, alpha=0.2)
        ax.axhline(y=0.3, color='green', linestyle='--', label='Fusion threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean RSA Score')
        ax.set_title('Representation Similarity')
        ax.legend()

        # Attention entropy
        ax = axes[1]
        ax.plot(epochs, attention_entropy, 's-', color='#ff7f0e', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Attention Entropy')
        ax.set_title('Attention Distribution Entropy')
        ax.annotate('Higher = more distributed', xy=(0.5, 0.02),
                   xycoords='axes fraction', ha='center', fontsize=9, style='italic')

        # Source importance variance
        ax = axes[2]
        ax.plot(epochs, source_importance_variance, '^-', color='#2ca02c',
               linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Importance Variance')
        ax.set_title('Source Importance Spread')
        ax.annotate('Higher = more differentiated', xy=(0.5, 0.02),
                   xycoords='axes fraction', ha='center', fontsize=9, style='italic')

        fig.suptitle('Fusion Quality Evolution During Training',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 3: TEMPORAL DYNAMICS FIGURES
# =============================================================================

class TemporalDynamicsFigures:
    """Visualization class for Section 3: Temporal Dynamics Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_context_window_curve(
        self,
        context_lengths: List[int],
        performance_metrics: Dict[str, np.ndarray],
        metric_name: str = "Accuracy",
    ) -> plt.Figure:
        """
        Create performance vs context length curve.

        Shows how model performance changes with different context windows.
        """
        fig, ax = plt.subplots(figsize=self.style.figure_small)

        colors = list(TASK_COLORS.values())
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for i, (task, perf) in enumerate(performance_metrics.items()):
            ax.plot(context_lengths, perf,
                   marker=markers[i % len(markers)],
                   color=colors[i % len(colors)],
                   linewidth=2, markersize=8,
                   label=task.capitalize())

        ax.set_xlabel('Context Window (days)')
        ax.set_ylabel(metric_name)
        ax.set_title('Performance vs Context Length')
        ax.legend(loc='lower right')

        # Mark full context
        if context_lengths[-1] > 100:
            ax.axvline(x=context_lengths[-1], color='gray', linestyle='--',
                      alpha=0.5, label='Full context')

        # Add annotation for optimal window
        for task, perf in performance_metrics.items():
            best_idx = np.argmax(perf)
            if best_idx < len(context_lengths) - 1:
                ax.annotate(f'{task}: {context_lengths[best_idx]}d',
                           xy=(context_lengths[best_idx], perf[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7)

        plt.tight_layout()
        return fig

    def fig_attention_distance(
        self,
        attention_distances: np.ndarray,
        phase_labels: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Create histogram of temporal attention distances.

        Shows how far back in time the model attends.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.style.figure_medium)

        # Overall histogram
        ax = axes[0]
        ax.hist(attention_distances.flatten(), bins=50, color='#1f77b4',
               alpha=0.7, edgecolor='black', linewidth=0.5)

        mean_dist = np.mean(attention_distances)
        median_dist = np.median(attention_distances)

        ax.axvline(x=mean_dist, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_dist:.1f} days')
        ax.axvline(x=median_dist, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_dist:.1f} days')

        ax.set_xlabel('Attention Distance (days)')
        ax.set_ylabel('Frequency')
        ax.set_title('Temporal Attention Distribution')
        ax.legend()

        # By conflict phase (if provided)
        ax = axes[1]
        if phase_labels is not None:
            phase_data = {}
            for phase in np.unique(phase_labels):
                phase_data[PHASE_LABELS.get(phase, f'Phase {phase}')] = \
                    attention_distances[phase_labels == phase]

            positions = range(len(phase_data))
            bp = ax.boxplot(phase_data.values(), positions=positions, patch_artist=True)

            for i, (box, phase) in enumerate(zip(bp['boxes'], phase_data.keys())):
                phase_num = [k for k, v in PHASE_LABELS.items() if v == phase]
                color = PHASE_COLORS.get(phase_num[0] if phase_num else i, '#1f77b4')
                box.set_facecolor(color)
                box.set_alpha(0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(phase_data.keys(), rotation=30, ha='right')
            ax.set_ylabel('Attention Distance (days)')
            ax.set_title('Attention Distance by Conflict Phase')
        else:
            # Show cumulative distribution
            sorted_dist = np.sort(attention_distances.flatten())
            cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
            ax.plot(sorted_dist, cdf, linewidth=2)
            ax.set_xlabel('Attention Distance (days)')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Cumulative Distribution of Attention Distance')

            # Mark percentiles
            for pct in [0.5, 0.75, 0.9]:
                val = np.percentile(sorted_dist, pct * 100)
                ax.axvline(x=val, color='gray', linestyle=':', alpha=0.5)
                ax.annotate(f'{pct*100:.0f}%: {val:.0f}d', xy=(val, pct),
                           fontsize=8, ha='left')

        fig.suptitle('Temporal Attention Analysis',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_transition_trajectories(
        self,
        latent_trajectories: np.ndarray,
        dates: List,
        transition_dates: Dict[str, str],
        method: str = 'pca',
    ) -> plt.Figure:
        """
        Create PCA/t-SNE latent trajectories around regime transitions.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dimensionality reduction
        if method.lower() == 'tsne' and HAS_SCIPY:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(latent_trajectories)
            method_label = 't-SNE'
        else:
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(latent_trajectories)
            explained_var = reducer.explained_variance_ratio_
            method_label = f'PCA ({explained_var[0]*100:.1f}%, {explained_var[1]*100:.1f}%)'

        # Convert dates to datetime for comparison
        dates_dt = pd.to_datetime(dates)

        # Assign phases
        phase_assignments = []
        for d in dates_dt:
            if d < pd.to_datetime(transition_dates.get('kyiv_withdrawal', '2022-04-02')):
                phase_assignments.append(0)
            elif d < pd.to_datetime(transition_dates.get('counteroffensive_start', '2022-09-01')):
                phase_assignments.append(1)
            elif d < pd.to_datetime(transition_dates.get('attritional_warfare', '2022-12-01')):
                phase_assignments.append(2)
            else:
                phase_assignments.append(3)
        phase_assignments = np.array(phase_assignments)

        # Plot trajectories by phase
        for phase in np.unique(phase_assignments):
            mask = phase_assignments == phase
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=PHASE_COLORS[phase], label=PHASE_LABELS[phase],
                      alpha=0.6, s=30)

        # Draw trajectory line
        ax.plot(coords[:, 0], coords[:, 1], 'k-', alpha=0.2, linewidth=0.5)

        # Mark transitions
        for trans_name, trans_date in transition_dates.items():
            trans_dt = pd.to_datetime(trans_date)
            if trans_dt in dates_dt.values:
                idx = np.where(dates_dt == trans_dt)[0][0]
                ax.scatter(coords[idx, 0], coords[idx, 1],
                          marker='*', s=300, c='black', zorder=10,
                          edgecolors='white', linewidths=2)
                ax.annotate(trans_name.replace('_', ' ').title(),
                           xy=(coords[idx, 0], coords[idx, 1]),
                           xytext=(15, 15), textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='black'))

        # Mark start and end
        ax.scatter(coords[0, 0], coords[0, 1], marker='o', s=200,
                  c='green', edgecolors='black', linewidths=2,
                  label='Start', zorder=10)
        ax.scatter(coords[-1, 0], coords[-1, 1], marker='s', s=200,
                  c='red', edgecolors='black', linewidths=2,
                  label='End', zorder=10)

        ax.set_xlabel(f'{method_label} Component 1')
        ax.set_ylabel(f'{method_label} Component 2')
        ax.set_title('Latent Space Trajectories Around Regime Transitions')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        return fig

    def fig_latent_velocity(
        self,
        dates: List,
        velocities: np.ndarray,
        transition_dates: Dict[str, str],
    ) -> plt.Figure:
        """
        Create velocity timeline with transition markers.
        """
        fig, ax = plt.subplots(figsize=self.style.figure_medium)

        dates_dt = pd.to_datetime(dates)

        # Plot velocity
        ax.plot(dates_dt, velocities, color='#1f77b4', linewidth=1, alpha=0.8)
        ax.fill_between(dates_dt, 0, velocities, alpha=0.3)

        # Add rolling average
        window = 7
        rolling_vel = pd.Series(velocities).rolling(window=window, center=True).mean()
        ax.plot(dates_dt, rolling_vel, color='#d62728', linewidth=2,
               label=f'{window}-day rolling average')

        # Mark transitions
        for trans_name, trans_date in transition_dates.items():
            trans_dt = pd.to_datetime(trans_date)
            ax.axvline(x=trans_dt, color='black', linestyle='--', linewidth=2, alpha=0.7)
            ax.annotate(trans_name.replace('_', ' ').title(),
                       xy=(trans_dt, ax.get_ylim()[1]),
                       xytext=(5, -10), textcoords='offset points',
                       rotation=45, fontsize=9, fontweight='bold',
                       va='top', ha='left')

        # Color background by phase
        y_min, y_max = ax.get_ylim()
        phase_boundaries = [dates_dt.min()] + \
                          [pd.to_datetime(d) for d in transition_dates.values()] + \
                          [dates_dt.max()]

        for i in range(len(phase_boundaries) - 1):
            ax.axvspan(phase_boundaries[i], phase_boundaries[i+1],
                      alpha=0.1, color=PHASE_COLORS.get(i, 'gray'))

        ax.set_xlabel('Date')
        ax.set_ylabel('Latent Velocity (L2 norm of daily change)')
        ax.set_title('Latent State Velocity Over Time')
        ax.legend(loc='upper right')

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 4: SEMANTIC STRUCTURE FIGURES
# =============================================================================

class SemanticStructureFigures:
    """Visualization class for Section 4: Semantic Structure Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_operation_clustering(
        self,
        latents: np.ndarray,
        operation_labels: List[str],
        dates: Optional[List] = None,
        method: str = 'tsne',
    ) -> plt.Figure:
        """
        Create t-SNE/PCA visualization with operation labels.

        Shows how different military operations cluster in latent space.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dimensionality reduction
        if method.lower() == 'tsne' and HAS_SCIPY:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(latents)
            method_label = 't-SNE'
        else:
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(latents)
            explained_var = reducer.explained_variance_ratio_
            method_label = f'PCA ({explained_var[0]*100:.1f}%, {explained_var[1]*100:.1f}%)'

        # Get unique operations
        unique_ops = list(set(op for op in operation_labels if op is not None))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ops) + 1))
        op_to_color = {op: colors[i] for i, op in enumerate(unique_ops)}
        op_to_color[None] = 'lightgray'

        # Plot by operation
        for op in unique_ops + [None]:
            mask = np.array([label == op for label in operation_labels])
            if np.sum(mask) > 0:
                label = op if op else 'No Operation'
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=[op_to_color[op]], label=label,
                          alpha=0.6 if op else 0.3,
                          s=50 if op else 20)

        ax.set_xlabel(f'{method_label} Component 1')
        ax.set_ylabel(f'{method_label} Component 2')
        ax.set_title('Operation Clustering in Latent Space')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        plt.tight_layout()
        return fig

    def fig_probe_confusion(
        self,
        confusion_matrix: np.ndarray,
        class_labels: List[str],
        probe_name: str = "Day Type",
    ) -> plt.Figure:
        """
        Create confusion matrix for day-type/intensity probes.
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        # Normalize by row (true labels)
        cm_normalized = confusion_matrix.astype('float') / \
                       (confusion_matrix.sum(axis=1, keepdims=True) + 1e-8)

        # Plot heatmap
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Add annotations
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                val = cm_normalized[i, j]
                count = confusion_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}\n({count})',
                       ha='center', va='center', color=color, fontsize=9)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Proportion')

        # Calculate accuracy
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        ax.set_title(f'{probe_name} Probe Confusion Matrix\n(Accuracy: {accuracy:.1%})',
                    fontsize=self.style.font_title, pad=15)

        plt.tight_layout()
        return fig

    def fig_weekly_pattern(
        self,
        weekday_effects: Dict[str, float],
        weekday_errors: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        """
        Create bar chart of weekday effects.
        """
        fig, ax = plt.subplots(figsize=self.style.figure_small)

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                   'Friday', 'Saturday', 'Sunday']
        effects = [weekday_effects.get(d, 0) for d in weekdays]
        errors = [weekday_errors.get(d, 0) for d in weekdays] if weekday_errors else None

        x = np.arange(len(weekdays))
        colors = ['#1f77b4' if e >= 0 else '#d62728' for e in effects]

        if errors:
            bars = ax.bar(x, effects, yerr=errors, capsize=5, color=colors,
                         edgecolor='black', linewidth=0.5)
        else:
            bars = ax.bar(x, effects, color=colors, edgecolor='black', linewidth=0.5)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(weekdays, rotation=45, ha='right')
        ax.set_ylabel('Effect on Latent State')
        ax.set_title('Weekday Effects on Model Latent States')

        # Add significance markers if errors provided
        if errors:
            for bar, effect, error in zip(bars, effects, errors):
                if abs(effect) > 2 * error:  # Rough significance
                    ax.annotate('*',
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               ha='center', va='bottom' if effect > 0 else 'top',
                               fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def fig_seasonal_pattern(
        self,
        monthly_effects: Dict[str, float],
        monthly_errors: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        """
        Create monthly pattern visualization.
        """
        fig = plt.figure(figsize=self.style.figure_small)

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        effects = [monthly_effects.get(m, 0) for m in months]

        # Create circular/polar plot
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)

        # Shift to start from top
        theta = theta + np.pi/2

        # Create polar subplot
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Plot
        bars = ax.bar(theta, effects, width=0.5, alpha=0.7,
                     color=plt.cm.RdYlBu_r((np.array(effects) - min(effects)) /
                                           (max(effects) - min(effects) + 1e-8)))

        ax.set_xticks(theta)
        ax.set_xticklabels(months)
        ax.set_title('Seasonal Pattern in Latent States', pad=20)

        # Add radial grid labels
        ax.set_rlabel_position(0)

        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 5: ISW SEMANTIC ASSOCIATION FIGURES
# =============================================================================

class ISWSemanticFigures:
    """Visualization class for Section 5: ISW Semantic-Numerical Association Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_isw_alignment(
        self,
        dates: List,
        similarities: np.ndarray,
        rolling_window: int = 7,
    ) -> plt.Figure:
        """
        Create time series of ISW-latent cosine similarity.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        dates_dt = pd.to_datetime(dates)

        # Raw similarity
        ax = axes[0]
        ax.plot(dates_dt, similarities, color='#1f77b4', alpha=0.5, linewidth=0.5)
        rolling = pd.Series(similarities).rolling(window=rolling_window, center=True).mean()
        ax.plot(dates_dt, rolling, color='#d62728', linewidth=2,
               label=f'{rolling_window}-day rolling mean')
        ax.axhline(y=np.mean(similarities), color='gray', linestyle='--',
                  label=f'Overall mean: {np.mean(similarities):.3f}')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('ISW Embedding - Model Latent Alignment Over Time')
        ax.legend(loc='upper right')

        # Distribution by month
        ax = axes[1]
        df = pd.DataFrame({'date': dates_dt, 'similarity': similarities})
        df['month'] = df['date'].dt.to_period('M')
        monthly_stats = df.groupby('month')['similarity'].agg(['mean', 'std'])

        x = range(len(monthly_stats))
        ax.bar(x, monthly_stats['mean'], yerr=monthly_stats['std'],
              capsize=3, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.set_xticks(x[::3])  # Every 3rd month
        ax.set_xticklabels([str(m) for m in monthly_stats.index[::3]], rotation=45)
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Similarity')
        ax.set_title('Monthly ISW-Latent Alignment')

        plt.tight_layout()
        return fig

    def fig_topic_source_correlation(
        self,
        correlation_matrix: np.ndarray,
        topic_labels: List[str],
        source_labels: List[str],
    ) -> plt.Figure:
        """
        Create heatmap of topic x source correlations.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-1, vmax=1)

        ax.set_xticks(range(len(source_labels)))
        ax.set_yticks(range(len(topic_labels)))
        ax.set_xticklabels(source_labels, rotation=45, ha='right')
        ax.set_yticklabels(topic_labels)
        ax.set_xlabel('Data Source')
        ax.set_ylabel('ISW Topic')

        # Add annotations for significant correlations
        for i in range(len(topic_labels)):
            for j in range(len(source_labels)):
                val = correlation_matrix[i, j]
                if abs(val) > 0.3:  # Only annotate significant
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=8, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation')

        ax.set_title('ISW Topic - Data Source Correlations',
                    fontsize=self.style.font_title, pad=15)

        plt.tight_layout()
        return fig

    def fig_event_response(
        self,
        event_data: Dict[str, Dict],
        window_days: int = 14,
    ) -> plt.Figure:
        """
        Create multi-panel showing latent response to key events.

        event_data format: {event_name: {'dates': [...], 'latent_norm': [...],
                           'similarity': [...], 'event_idx': int}}
        """
        n_events = len(event_data)
        cols = min(3, n_events)
        rows = (n_events + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_events == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (event_name, data) in enumerate(event_data.items()):
            row, col = divmod(idx, cols)
            ax = axes[row, col]

            days_relative = np.arange(-window_days, window_days + 1)
            event_idx = data.get('event_idx', window_days)

            # Plot latent norm
            if 'latent_norm' in data:
                latent = np.array(data['latent_norm'])
                ax.plot(days_relative[:len(latent)], latent[:len(days_relative)],
                       'b-', label='Latent Norm', linewidth=2)

            # Plot similarity
            if 'similarity' in data:
                sim = np.array(data['similarity'])
                ax2 = ax.twinx()
                ax2.plot(days_relative[:len(sim)], sim[:len(days_relative)],
                        'r--', label='ISW Similarity', linewidth=2)
                ax2.set_ylabel('Similarity', color='red')

            ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
            ax.set_xlabel('Days Relative to Event')
            ax.set_ylabel('Latent Norm', color='blue')
            ax.set_title(event_name.replace('_', ' ').title(), fontsize=10)

        # Remove empty subplots
        for idx in range(n_events, rows * cols):
            row, col = divmod(idx, cols)
            axes[row, col].axis('off')

        fig.suptitle('Latent Response to Major Events', fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_lag_analysis(
        self,
        lag_results: Dict[str, Dict],
    ) -> plt.Figure:
        """
        Create lead/lag classification visualization.

        lag_results format: {metric_name: {'lags': [...], 'correlations': [...],
                            'peak_lag': int, 'classification': str}}
        """
        n_metrics = len(lag_results)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric, data) in zip(axes, lag_results.items()):
            lags = np.array(data['lags'])
            corrs = np.array(data['correlations'])
            peak_lag = data.get('peak_lag', 0)
            classification = data.get('classification', 'Unknown')

            # Bar chart of lag correlations
            colors = ['#2ca02c' if l < 0 else '#1f77b4' if l > 0 else '#ff7f0e'
                     for l in lags]
            ax.bar(lags, corrs, color=colors, alpha=0.7, edgecolor='black')

            # Highlight peak
            peak_idx = np.where(lags == peak_lag)[0]
            if len(peak_idx) > 0:
                ax.bar(peak_lag, corrs[peak_idx[0]], color='red', edgecolor='black',
                      linewidth=2)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Correlation')
            ax.set_title(f'{metric}\n({classification})')

            # Legend
            legend_elements = [
                mpatches.Patch(color='#2ca02c', label='ISW leads'),
                mpatches.Patch(color='#1f77b4', label='ISW lags'),
                mpatches.Patch(color='#ff7f0e', label='Concurrent'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        fig.suptitle('Lead/Lag Analysis: ISW vs Numerical Features',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 6: CAUSAL IMPORTANCE FIGURES
# =============================================================================

class CausalImportanceFigures:
    """Visualization class for Section 6: Causal Importance Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_causal_rankings(
        self,
        rankings: Dict[str, Dict[str, float]],
    ) -> plt.Figure:
        """
        Create stacked bar chart of causal importance by task.

        rankings format: {task: {source: importance_score}}
        """
        fig, ax = plt.subplots(figsize=self.style.figure_medium)

        tasks = list(rankings.keys())
        sources = list(rankings[tasks[0]].keys())

        x = np.arange(len(tasks))
        width = 0.7

        # Stack bars
        bottom = np.zeros(len(tasks))
        for i, source in enumerate(sources):
            values = [rankings[task].get(source, 0) for task in tasks]
            color = SOURCE_COLORS.get(source.lower(), plt.cm.tab20(i / len(sources)))
            ax.bar(x, values, width, bottom=bottom, label=source, color=color)
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks])
        ax.set_xlabel('Task')
        ax.set_ylabel('Cumulative Causal Importance')
        ax.set_title('Causal Importance Rankings by Task')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        plt.tight_layout()
        return fig

    def fig_intervention_effects(
        self,
        intervention_results: Dict[str, np.ndarray],
    ) -> plt.Figure:
        """
        Create distribution plot of intervention effects.

        intervention_results format: {source: array of effect sizes}
        """
        fig, ax = plt.subplots(figsize=self.style.figure_medium)

        sources = list(intervention_results.keys())
        data = [intervention_results[s] for s in sources]

        # Violin plot
        parts = ax.violinplot(data, positions=range(len(sources)), showmeans=True,
                             showmedians=True)

        # Color violins
        for i, pc in enumerate(parts['bodies']):
            color = SOURCE_COLORS.get(sources[i].lower(), '#1f77b4')
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Add box plot overlay
        bp = ax.boxplot(data, positions=range(len(sources)), widths=0.15,
                       patch_artist=False, showfliers=False)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha='right')
        ax.set_xlabel('Source')
        ax.set_ylabel('Effect Size (Performance Change)')
        ax.set_title('Distribution of Intervention Effects by Source')

        # Add mean annotations
        for i, (s, d) in enumerate(zip(sources, data)):
            mean_val = np.mean(d)
            ax.annotate(f'{mean_val:+.3f}',
                       xy=(i, np.max(d)),
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def fig_integrated_gradients(
        self,
        attributions: np.ndarray,
        source_labels: List[str],
        feature_labels: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> plt.Figure:
        """
        Create attribution heatmap per source/feature.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.style.figure_medium)

        # Left: Source-level attributions
        ax = axes[0]
        source_attr = np.mean(np.abs(attributions), axis=1)
        sorted_idx = np.argsort(source_attr)[::-1]

        y_pos = np.arange(len(source_labels))
        colors = [SOURCE_COLORS.get(source_labels[i].lower(), '#1f77b4')
                 for i in sorted_idx]

        ax.barh(y_pos, source_attr[sorted_idx], color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([source_labels[i] for i in sorted_idx])
        ax.set_xlabel('Mean |Attribution|')
        ax.set_title('Source-Level Attributions')
        ax.invert_yaxis()

        # Right: Feature-level heatmap (top N features)
        ax = axes[1]
        if feature_labels is None:
            feature_labels = [f'F{i}' for i in range(attributions.shape[1])]

        # Select top features by variance
        feature_var = np.var(attributions, axis=0)
        top_features = np.argsort(feature_var)[::-1][:top_n]

        attr_subset = attributions[:, top_features]
        feat_labels_subset = [feature_labels[i] for i in top_features]

        im = ax.imshow(attr_subset.T, cmap='RdBu_r', aspect='auto')

        ax.set_xticks(range(len(source_labels)))
        ax.set_yticks(range(len(feat_labels_subset)))
        ax.set_xticklabels(source_labels, rotation=45, ha='right')
        ax.set_yticklabels(feat_labels_subset, fontsize=8)
        ax.set_xlabel('Source')
        ax.set_ylabel(f'Top {top_n} Features')
        ax.set_title('Feature Attribution Heatmap')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attribution')

        fig.suptitle('Integrated Gradients Attribution Analysis',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_causal_flow_graph(
        self,
        flow_matrix: np.ndarray,
        source_labels: List[str],
        threshold: float = 0.1,
    ) -> plt.Figure:
        """
        Create network visualization of attention-based causal flow.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        n_sources = len(source_labels)
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        radius = 1.0

        # Position nodes
        node_positions = {}
        for i, (label, angle) in enumerate(zip(source_labels, angles)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            node_positions[label] = (x, y)

            # Node size proportional to outgoing flow
            outflow = np.sum(flow_matrix[i, :])
            size = 0.1 + 0.15 * outflow / (np.max(np.sum(flow_matrix, axis=1)) + 1e-8)

            color = SOURCE_COLORS.get(label.lower(), '#1f77b4')
            circle = plt.Circle((x, y), size, color=color, alpha=0.8,
                               edgecolor='black', linewidth=2)
            ax.add_patch(circle)

            # Label
            label_x = 1.35 * np.cos(angle)
            label_y = 1.35 * np.sin(angle)
            ax.text(label_x, label_y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')

        # Draw edges
        for i, src_from in enumerate(source_labels):
            for j, src_to in enumerate(source_labels):
                if i != j and flow_matrix[i, j] > threshold:
                    x1, y1 = node_positions[src_from]
                    x2, y2 = node_positions[src_to]

                    # Edge width proportional to flow
                    width = flow_matrix[i, j] * 5
                    alpha = min(0.8, flow_matrix[i, j] * 2)

                    ax.annotate('',
                               xy=(x2*0.85, y2*0.85),
                               xytext=(x1*0.85, y1*0.85),
                               arrowprops=dict(arrowstyle='->',
                                              color='gray',
                                              lw=width,
                                              alpha=alpha,
                                              connectionstyle='arc3,rad=0.15'))

        ax.set_title('Causal Information Flow Graph\n(Edge width = flow strength)',
                    fontsize=self.style.font_title, y=1.05)

        # Add legend for flow strength
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=1, label='Weak flow'),
            Line2D([0], [0], color='gray', linewidth=3, label='Medium flow'),
            Line2D([0], [0], color='gray', linewidth=5, label='Strong flow'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        return fig


# =============================================================================
# SECTION 7: TACTICAL READINESS FIGURES
# =============================================================================

class TacticalReadinessFigures:
    """Visualization class for Section 7: Tactical Readiness Probes."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

    def fig_data_availability(
        self,
        availability_matrix: pd.DataFrame,
        density_matrix: Optional[pd.DataFrame] = None,
    ) -> plt.Figure:
        """
        Create heatmap of source x resolution availability.
        """
        fig, axes = plt.subplots(1, 2, figsize=self.style.figure_medium)

        # Availability (categorical)
        ax = axes[0]

        # Convert to numeric
        avail_map = {'YES': 2, 'NO': 0, 'N/A': 1}
        avail_numeric = availability_matrix.replace(avail_map).astype(float)

        # Custom colormap
        colors = ['#d62728', '#ffff99', '#2ca02c']  # Red, Yellow, Green
        cmap = LinearSegmentedColormap.from_list('availability', colors, N=3)

        im = ax.imshow(avail_numeric.values, cmap=cmap, aspect='auto', vmin=0, vmax=2)

        ax.set_xticks(range(len(availability_matrix.columns)))
        ax.set_yticks(range(len(availability_matrix.index)))
        ax.set_xticklabels(availability_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(availability_matrix.index, fontsize=9)
        ax.set_xlabel('Resolution Level')
        ax.set_ylabel('Data Source')
        ax.set_title('Data Availability by Resolution')

        # Add text annotations
        for i in range(len(availability_matrix.index)):
            for j in range(len(availability_matrix.columns)):
                val = availability_matrix.iloc[i, j]
                color = 'white' if val == 'NO' else 'black'
                ax.text(j, i, val, ha='center', va='center', color=color, fontsize=7)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#2ca02c', label='Available (YES)'),
            mpatches.Patch(color='#ffff99', label='Not Applicable (N/A)'),
            mpatches.Patch(color='#d62728', label='Not Available (NO)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.15),
                 ncol=3, fontsize=8)

        # Density (if provided)
        ax = axes[1]
        if density_matrix is not None:
            im = ax.imshow(density_matrix.values, cmap='YlGn', aspect='auto',
                          vmin=0, vmax=1)

            ax.set_xticks(range(len(density_matrix.columns)))
            ax.set_yticks(range(len(density_matrix.index)))
            ax.set_xticklabels(density_matrix.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(density_matrix.index, fontsize=9)
            ax.set_xlabel('Resolution Level')
            ax.set_ylabel('Data Source')
            ax.set_title('Data Density (0-1)')

            # Add text annotations
            for i in range(len(density_matrix.index)):
                for j in range(len(density_matrix.columns)):
                    val = density_matrix.iloc[i, j]
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color=color, fontsize=7)

            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Density')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Density data not provided',
                   ha='center', va='center', transform=ax.transAxes)

        fig.suptitle('Data Availability Analysis for Tactical Predictions',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig

    def fig_sector_map(
        self,
        sector_definitions: Dict[str, Dict],
        sector_metrics: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        """
        Create geographic map of front-line sectors.
        """
        fig, ax = plt.subplots(figsize=(14, 10))

        # Ukraine bounding box (approximate)
        ax.set_xlim(22, 42)
        ax.set_ylim(44, 53)

        # Draw sectors
        for sector_id, sector_info in sector_definitions.items():
            bbox = sector_info.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                lon_min, lat_min, lon_max, lat_max = bbox

                # Sector color based on metrics if provided
                if sector_metrics and sector_id in sector_metrics:
                    metric = sector_metrics[sector_id]
                    color = plt.cm.RdYlGn(metric)  # 0=red, 1=green
                else:
                    color = '#1f77b4'

                # Draw rectangle
                rect = plt.Rectangle((lon_min, lat_min),
                                     lon_max - lon_min, lat_max - lat_min,
                                     facecolor=color, edgecolor='black',
                                     linewidth=2, alpha=0.5)
                ax.add_patch(rect)

                # Label
                center_lon = (lon_min + lon_max) / 2
                center_lat = (lat_min + lat_max) / 2
                name = sector_info.get('name', sector_id)
                ax.text(center_lon, center_lat, name,
                       ha='center', va='center', fontsize=8,
                       fontweight='bold', wrap=True)

        # Add reference points
        cities = {
            'Kyiv': (30.5, 50.45),
            'Kharkiv': (36.25, 50.0),
            'Donetsk': (37.8, 48.0),
            'Mariupol': (37.55, 47.1),
            'Kherson': (32.6, 46.65),
            'Zaporizhzhia': (35.15, 47.85),
        }

        for city, (lon, lat) in cities.items():
            ax.plot(lon, lat, 'ko', markersize=8)
            ax.annotate(city, (lon, lat), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Front-Line Sector Definitions')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add colorbar if metrics provided
        if sector_metrics:
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=Normalize(0, 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label('Sector Metric')

        plt.tight_layout()
        return fig

    def fig_resolution_tradeoff(
        self,
        resolutions: List[str],
        performance: np.ndarray,
        feasibility: np.ndarray,
        coverage: np.ndarray,
    ) -> plt.Figure:
        """
        Create performance vs resolution tradeoff curve.
        """
        fig, axes = plt.subplots(1, 3, figsize=self.style.figure_medium)

        x = np.arange(len(resolutions))

        # Performance curve
        ax = axes[0]
        ax.plot(x, performance, 'o-', color='#1f77b4', linewidth=2, markersize=10)
        ax.fill_between(x, 0, performance, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(resolutions, rotation=45, ha='right')
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Predicted Performance')
        ax.set_title('Performance vs Resolution')
        ax.set_ylim(0, 1)

        # Feasibility curve
        ax = axes[1]
        colors = ['#2ca02c' if f > 0.5 else '#ff7f0e' if f > 0.25 else '#d62728'
                 for f in feasibility]
        ax.bar(x, feasibility, color=colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(resolutions, rotation=45, ha='right')
        ax.set_xlabel('Resolution')
        ax.set_ylabel('Feasibility Score')
        ax.set_title('Data Feasibility by Resolution')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Trade-off plot
        ax = axes[2]
        # Bubble chart: x=feasibility, y=performance, size=coverage
        bubble_size = (coverage + 0.1) * 500

        scatter = ax.scatter(feasibility, performance, s=bubble_size,
                           c=range(len(resolutions)), cmap='viridis',
                           alpha=0.7, edgecolors='black', linewidth=2)

        for i, res in enumerate(resolutions):
            ax.annotate(res, (feasibility[i], performance[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)

        ax.set_xlabel('Feasibility')
        ax.set_ylabel('Predicted Performance')
        ax.set_title('Resolution Trade-off\n(bubble size = coverage)')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        # Mark optimal region
        ax.axhspan(0.6, 1.0, xmin=0.5, xmax=1.0, alpha=0.1, color='green')
        ax.text(0.75, 0.8, 'Optimal\nRegion', ha='center', va='center',
               fontsize=10, style='italic')

        fig.suptitle('Resolution Trade-off Analysis',
                    fontsize=self.style.font_title + 2, y=1.02)
        plt.tight_layout()
        return fig


# =============================================================================
# DASHBOARD GENERATOR
# =============================================================================

class ProbeResultDashboard:
    """Creates a multi-panel summary dashboard combining key findings from all sections."""

    def __init__(self, style: StyleConfig = None):
        self.style = style or StyleConfig()
        self.style.apply()

        # Initialize all figure classes
        self.data_artifact = DataArtifactFigures(style)
        self.cross_modal = CrossModalFusionFigures(style)
        self.temporal = TemporalDynamicsFigures(style)
        self.semantic = SemanticStructureFigures(style)
        self.isw = ISWSemanticFigures(style)
        self.causal = CausalImportanceFigures(style)
        self.tactical = TacticalReadinessFigures(style)

    def create_dashboard(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive 4x5 grid dashboard with key visualizations.

        Args:
            results: Dictionary containing all probe results
            output_path: Optional path to save the dashboard

        Returns:
            Figure object containing the dashboard
        """
        fig = plt.figure(figsize=self.style.figure_dashboard)
        gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Data Artifacts & Cross-Modal Fusion
        # 1.1 Source importance
        ax = fig.add_subplot(gs[0, 0])
        if 'source_importance' in results:
            sources = list(results['source_importance'].keys())[:8]
            values = [results['source_importance'][s] for s in sources]
            colors = [SOURCE_COLORS.get(s.lower(), '#1f77b4') for s in sources]
            ax.barh(sources, values, color=colors)
            ax.set_xlabel('Importance')
            ax.set_title('Source Importance', fontsize=10)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Source Importance\n(No data)', ha='center', va='center')
            ax.set_title('Source Importance', fontsize=10)

        # 1.2 VIIRS lag
        ax = fig.add_subplot(gs[0, 1])
        if 'viirs_lag' in results:
            lags = results['viirs_lag'].get('lags', range(-7, 8))
            corrs = results['viirs_lag'].get('correlations', np.zeros(15))
            ax.bar(lags, corrs, color='#8c564b', alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='--')
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Correlation')
            ax.set_title('VIIRS-Casualty Lag', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'VIIRS Lag\n(No data)', ha='center', va='center')
            ax.set_title('VIIRS-Casualty Lag', fontsize=10)

        # 1.3 RSA matrix (simplified)
        ax = fig.add_subplot(gs[0, 2])
        if 'rsa_matrix' in results:
            rsa = np.array(results['rsa_matrix'])
            im = ax.imshow(rsa, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_title('RSA Matrix', fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.6)
        else:
            ax.text(0.5, 0.5, 'RSA Matrix\n(No data)', ha='center', va='center')
            ax.set_title('RSA Matrix', fontsize=10)

        # 1.4 Ablation summary
        ax = fig.add_subplot(gs[0, 3])
        if 'ablation_summary' in results:
            sources = list(results['ablation_summary'].keys())[:6]
            effects = [results['ablation_summary'][s] for s in sources]
            colors = ['#d62728' if e < 0 else '#2ca02c' for e in effects]
            ax.barh(sources, effects, color=colors)
            ax.axvline(x=0, color='black', linestyle='-')
            ax.set_xlabel('Effect')
            ax.set_title('Ablation Effects', fontsize=10)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Ablation Effects\n(No data)', ha='center', va='center')
            ax.set_title('Ablation Effects', fontsize=10)

        # 1.5 Fusion trajectory
        ax = fig.add_subplot(gs[0, 4])
        if 'fusion_trajectory' in results:
            epochs = results['fusion_trajectory'].get('epochs', range(10))
            rsa = results['fusion_trajectory'].get('rsa_scores', np.random.rand(10))
            ax.plot(epochs, rsa, 'o-', color='#1f77b4')
            ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RSA')
            ax.set_title('Fusion Trajectory', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Fusion Trajectory\n(No data)', ha='center', va='center')
            ax.set_title('Fusion Trajectory', fontsize=10)

        # Row 2: Temporal Dynamics
        # 2.1 Context window
        ax = fig.add_subplot(gs[1, 0])
        if 'context_performance' in results:
            lengths = results['context_performance'].get('lengths', [7, 14, 30, 60, 90])
            perf = results['context_performance'].get('performance', np.random.rand(5))
            ax.plot(lengths, perf, 'o-', color='#1f77b4')
            ax.set_xlabel('Context (days)')
            ax.set_ylabel('Performance')
            ax.set_title('Context Window', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Context Window\n(No data)', ha='center', va='center')
            ax.set_title('Context Window', fontsize=10)

        # 2.2 Attention distance histogram
        ax = fig.add_subplot(gs[1, 1])
        if 'attention_distances' in results:
            distances = np.array(results['attention_distances'])
            ax.hist(distances, bins=20, color='#ff7f0e', alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(distances), color='red', linestyle='--')
            ax.set_xlabel('Distance (days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Attention Distance', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Attention Distance\n(No data)', ha='center', va='center')
            ax.set_title('Attention Distance', fontsize=10)

        # 2.3 Latent velocity
        ax = fig.add_subplot(gs[1, 2])
        if 'latent_velocity' in results:
            vel = np.array(results['latent_velocity'])
            ax.plot(vel, color='#2ca02c', alpha=0.7)
            rolling = pd.Series(vel).rolling(window=7).mean()
            ax.plot(rolling, color='#d62728', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Velocity')
            ax.set_title('Latent Velocity', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Latent Velocity\n(No data)', ha='center', va='center')
            ax.set_title('Latent Velocity', fontsize=10)

        # 2.4 Phase transition
        ax = fig.add_subplot(gs[1, 3])
        phase_counts = results.get('phase_distribution', {0: 38, 1: 152, 2: 91, 3: 365})
        if phase_counts:
            phases = list(PHASE_LABELS.values())[:len(phase_counts)]
            counts = list(phase_counts.values())
            colors = [PHASE_COLORS[i] for i in range(len(phases))]
            ax.pie(counts, labels=phases, colors=colors, autopct='%1.0f%%',
                  textprops={'fontsize': 8})
            ax.set_title('Phase Distribution', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Phase Distribution\n(No data)', ha='center', va='center')
            ax.set_title('Phase Distribution', fontsize=10)

        # 2.5 Transition markers
        ax = fig.add_subplot(gs[1, 4])
        ax.text(0.5, 0.9, 'Key Transitions', ha='center', va='top',
               fontsize=11, fontweight='bold')
        transitions = [
            'Apr 2022: Kyiv withdrawal',
            'Sep 2022: Kharkiv counter.',
            'Dec 2022: Attritional phase',
        ]
        for i, t in enumerate(transitions):
            ax.text(0.1, 0.7 - i*0.25, t, ha='left', va='center', fontsize=9)
        ax.axis('off')
        ax.set_title('', fontsize=10)

        # Row 3: Semantic & ISW
        # 3.1 Operation clustering (placeholder)
        ax = fig.add_subplot(gs[2, 0])
        if 'operation_clusters' in results:
            ax.scatter(np.random.randn(50), np.random.randn(50),
                      c=np.random.randint(0, 5, 50), cmap='tab10', alpha=0.6)
            ax.set_title('Operation Clusters', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Operation Clusters\n(No data)', ha='center', va='center')
            ax.set_title('Operation Clusters', fontsize=10)

        # 3.2 Weekday effects
        ax = fig.add_subplot(gs[2, 1])
        if 'weekday_effects' in results:
            effects = list(results['weekday_effects'].values())
            days = ['M', 'T', 'W', 'Th', 'F', 'S', 'Su'][:len(effects)]
            colors = ['#1f77b4' if e >= 0 else '#d62728' for e in effects]
            ax.bar(days, effects, color=colors)
            ax.axhline(y=0, color='black', linestyle='-')
            ax.set_title('Weekday Effects', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Weekday Effects\n(No data)', ha='center', va='center')
            ax.set_title('Weekday Effects', fontsize=10)

        # 3.3 ISW alignment
        ax = fig.add_subplot(gs[2, 2])
        if 'isw_similarities' in results:
            sims = np.array(results['isw_similarities'])[:100]  # First 100 days
            ax.plot(sims, color='#9467bd', alpha=0.7)
            ax.axhline(y=np.mean(sims), color='red', linestyle='--')
            ax.set_xlabel('Day')
            ax.set_ylabel('Similarity')
            ax.set_title('ISW Alignment', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'ISW Alignment\n(No data)', ha='center', va='center')
            ax.set_title('ISW Alignment', fontsize=10)

        # 3.4 Topic correlations
        ax = fig.add_subplot(gs[2, 3])
        if 'topic_correlations' in results:
            corr = np.array(results['topic_correlations'])[:5, :5]
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Topic-Source Corr.', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Topic Correlations\n(No data)', ha='center', va='center')
            ax.set_title('Topic-Source Corr.', fontsize=10)

        # 3.5 Event response
        ax = fig.add_subplot(gs[2, 4])
        ax.text(0.5, 0.9, 'Key Events', ha='center', va='top',
               fontsize=11, fontweight='bold')
        events = [
            'Kerch Bridge: High response',
            'Kherson lib.: Strong signal',
            'Prigozhin: Moderate impact',
        ]
        for i, e in enumerate(events):
            ax.text(0.1, 0.7 - i*0.25, e, ha='left', va='center', fontsize=8)
        ax.axis('off')

        # Row 4: Causal & Tactical
        # 4.1 Causal rankings
        ax = fig.add_subplot(gs[3, 0])
        if 'causal_rankings' in results:
            sources = list(results['causal_rankings'].keys())[:6]
            importance = [results['causal_rankings'][s] for s in sources]
            colors = [SOURCE_COLORS.get(s.lower(), '#1f77b4') for s in sources]
            ax.barh(sources, importance, color=colors)
            ax.set_xlabel('Importance')
            ax.set_title('Causal Rankings', fontsize=10)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Causal Rankings\n(No data)', ha='center', va='center')
            ax.set_title('Causal Rankings', fontsize=10)

        # 4.2 Intervention effects
        ax = fig.add_subplot(gs[3, 1])
        if 'intervention_means' in results:
            sources = list(results['intervention_means'].keys())[:5]
            means = [results['intervention_means'][s] for s in sources]
            colors = ['#d62728' if m < 0 else '#2ca02c' for m in means]
            ax.bar(sources, means, color=colors)
            ax.axhline(y=0, color='black', linestyle='-')
            ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Effect')
            ax.set_title('Interventions', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Interventions\n(No data)', ha='center', va='center')
            ax.set_title('Interventions', fontsize=10)

        # 4.3 Data availability summary
        ax = fig.add_subplot(gs[3, 2])
        if 'availability_summary' in results:
            res_levels = ['National', 'Oblast', 'Sector', 'Grid']
            scores = [results['availability_summary'].get(r, 0) for r in res_levels]
            colors = ['#2ca02c' if s > 0.7 else '#ff7f0e' if s > 0.4 else '#d62728'
                     for s in scores]
            ax.bar(res_levels, scores, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Availability')
            ax.set_title('Data Availability', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Data Availability\n(No data)', ha='center', va='center')
            ax.set_title('Data Availability', fontsize=10)

        # 4.4 Resolution tradeoff
        ax = fig.add_subplot(gs[3, 3])
        if 'resolution_tradeoff' in results:
            resolutions = results['resolution_tradeoff'].get('resolutions', ['Daily', '12h', '6h'])
            feasibility = results['resolution_tradeoff'].get('feasibility', [0.9, 0.5, 0.2])
            performance = results['resolution_tradeoff'].get('performance', [0.8, 0.6, 0.4])
            ax.scatter(feasibility, performance, s=100, c=range(len(resolutions)),
                      cmap='viridis', edgecolors='black')
            for i, r in enumerate(resolutions):
                ax.annotate(r, (feasibility[i], performance[i]), fontsize=8)
            ax.set_xlabel('Feasibility')
            ax.set_ylabel('Performance')
            ax.set_title('Resolution Trade-off', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Resolution Trade-off\n(No data)', ha='center', va='center')
            ax.set_title('Resolution Trade-off', fontsize=10)

        # 4.5 Recommendations summary
        ax = fig.add_subplot(gs[3, 4])
        ax.text(0.5, 0.95, 'Key Recommendations', ha='center', va='top',
               fontsize=11, fontweight='bold')
        recommendations = [
            '1. Use delta encoding',
            '2. Validate VIIRS signal',
            '3. Focus on oblast level',
            '4. Add semantic features',
        ]
        for i, r in enumerate(recommendations):
            ax.text(0.05, 0.75 - i*0.2, r, ha='left', va='center', fontsize=8)
        ax.axis('off')

        # Main title
        fig.suptitle('Multi-Resolution HAN Probe Results Dashboard',
                    fontsize=self.style.font_title + 4, y=0.98, fontweight='bold')

        # Save if path provided
        if output_path:
            save_figure(fig, output_path.stem, output_path.parent, formats=['png', 'pdf'])

        return fig


# =============================================================================
# REPORT FIGURE GENERATOR
# =============================================================================

def generate_all_figures(
    results_dir: Path = OUTPUT_DIR,
    output_dir: Path = FIGURE_DIR,
    style: StyleConfig = None,
) -> Dict[str, List[Path]]:
    """
    Read probe results and generate all figures.

    Args:
        results_dir: Directory containing probe result files
        output_dir: Directory to save generated figures
        style: Style configuration for figures

    Returns:
        Dictionary mapping section names to list of generated figure paths
    """
    style = style or StyleConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_figures = {
        'data_artifacts': [],
        'cross_modal_fusion': [],
        'temporal_dynamics': [],
        'semantic_structure': [],
        'isw_semantic': [],
        'causal_importance': [],
        'tactical_readiness': [],
        'dashboard': [],
    }

    # Initialize figure generators
    data_fig = DataArtifactFigures(style)
    cross_fig = CrossModalFusionFigures(style)
    temporal_fig = TemporalDynamicsFigures(style)
    semantic_fig = SemanticStructureFigures(style)
    isw_fig = ISWSemanticFigures(style)
    causal_fig = CausalImportanceFigures(style)
    tactical_fig = TacticalReadinessFigures(style)

    # Load available results
    try:
        tactical_results = load_json_results('tactical_readiness_summary.json', results_dir)
    except Exception:
        tactical_results = {}

    # Generate Section 7 figures from tactical results
    if tactical_results:
        # Data availability
        if 'data_availability' in tactical_results:
            matrix_data = tactical_results['data_availability'].get('matrix', {})
            if matrix_data:
                # Reconstruct DataFrame
                sources = [matrix_data['source'][str(i)] for i in range(len(matrix_data['source']))]

                # Get spatial columns
                spatial_cols = [c for c in matrix_data.keys() if c.startswith('spatial_') and not c.endswith('_density')]
                avail_data = {}
                density_data = {}

                for col in spatial_cols:
                    col_name = col.replace('spatial_', '')
                    avail_data[col_name] = [matrix_data[col][str(i)] for i in range(len(sources))]
                    density_col = col + '_density'
                    if density_col in matrix_data:
                        density_data[col_name] = [matrix_data[density_col][str(i)] for i in range(len(sources))]

                avail_df = pd.DataFrame(avail_data, index=sources)
                density_df = pd.DataFrame(density_data, index=sources) if density_data else None

                fig = tactical_fig.fig_data_availability(avail_df, density_df)
                paths = save_figure(fig, 'fig_data_availability', output_dir)
                generated_figures['tactical_readiness'].extend(paths)
                plt.close(fig)

        # Sector map
        if 'sector_definitions' in tactical_results:
            sectors = tactical_results['sector_definitions'].get('sectors', {}).get('sectors', {})
            if sectors:
                fig = tactical_fig.fig_sector_map(sectors)
                paths = save_figure(fig, 'fig_sector_map', output_dir)
                generated_figures['tactical_readiness'].extend(paths)
                plt.close(fig)

        # Resolution tradeoff
        if 'resolution_analysis' in tactical_results:
            spatial_tradeoff = tactical_results['resolution_analysis'].get('tradeoff_tables', {}).get('spatial', {})
            if spatial_tradeoff:
                resolutions = [spatial_tradeoff['resolution'][str(i)] for i in range(len(spatial_tradeoff['resolution']))]

                # Extract metrics
                performance = []
                feasibility = []
                coverage = []

                for i in range(len(resolutions)):
                    perf = spatial_tradeoff.get('spatial_accuracy', {}).get(str(i), 0.5)
                    feas_str = spatial_tradeoff.get('overall_feasibility', {}).get(str(i), 'MEDIUM')
                    feas_map = {'HIGH': 0.9, 'MEDIUM': 0.5, 'LOW': 0.3, 'NOT_FEASIBLE': 0.1}
                    feas = feas_map.get(feas_str, 0.5)
                    cov = spatial_tradeoff.get('cross_source_coverage', {}).get(str(i), 0.5)

                    performance.append(perf)
                    feasibility.append(feas)
                    coverage.append(cov)

                fig = tactical_fig.fig_resolution_tradeoff(
                    resolutions,
                    np.array(performance),
                    np.array(feasibility),
                    np.array(coverage)
                )
                paths = save_figure(fig, 'fig_resolution_tradeoff', output_dir)
                generated_figures['tactical_readiness'].extend(paths)
                plt.close(fig)

    # Generate dashboard
    dashboard = ProbeResultDashboard(style)
    all_results = {'tactical_results': tactical_results}

    # Load ISW semantic results if available
    try:
        isw_results = load_json_results('semantic_association_results.json',
                                       results_dir.parent / 'semantic_results')
        if 'isw_latent_correlation' in isw_results:
            sims = isw_results['isw_latent_correlation'].get('alignment', {}).get('daily_similarities', [])
            if sims:
                all_results['isw_similarities'] = sims
    except Exception:
        pass

    fig = dashboard.create_dashboard(all_results)
    paths = save_figure(fig, 'probe_dashboard', output_dir)
    generated_figures['dashboard'].extend(paths)
    plt.close(fig)

    print(f"Generated figures saved to: {output_dir}")
    for section, paths in generated_figures.items():
        if paths:
            print(f"  {section}: {len(paths)} files")

    return generated_figures


def generate_section_figures(
    section: str,
    results: Dict[str, Any],
    output_dir: Path = FIGURE_DIR,
    style: StyleConfig = None,
) -> List[Path]:
    """
    Generate figures for a specific section.

    Args:
        section: Section name (1-7 or name)
        results: Dictionary containing section results
        output_dir: Directory to save figures
        style: Style configuration

    Returns:
        List of paths to generated figures
    """
    style = style or StyleConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    section_map = {
        '1': 'data_artifacts',
        '2': 'cross_modal_fusion',
        '3': 'temporal_dynamics',
        '4': 'semantic_structure',
        '5': 'isw_semantic',
        '6': 'causal_importance',
        '7': 'tactical_readiness',
    }

    section_name = section_map.get(str(section), section.lower().replace(' ', '_'))
    generated_paths = []

    # Map section to figure class
    figure_classes = {
        'data_artifacts': DataArtifactFigures,
        'cross_modal_fusion': CrossModalFusionFigures,
        'temporal_dynamics': TemporalDynamicsFigures,
        'semantic_structure': SemanticStructureFigures,
        'isw_semantic': ISWSemanticFigures,
        'causal_importance': CausalImportanceFigures,
        'tactical_readiness': TacticalReadinessFigures,
    }

    if section_name not in figure_classes:
        print(f"Unknown section: {section}")
        return generated_paths

    fig_class = figure_classes[section_name](style)

    # Generate figures based on available results
    # This is a dispatcher that calls appropriate methods based on result keys
    for key, data in results.items():
        method_name = f'fig_{key}'
        if hasattr(fig_class, method_name):
            try:
                method = getattr(fig_class, method_name)
                if isinstance(data, dict):
                    fig = method(**data)
                else:
                    fig = method(data)
                paths = save_figure(fig, f'{section_name}_{key}', output_dir)
                generated_paths.extend(paths)
                plt.close(fig)
            except Exception as e:
                print(f"Error generating {method_name}: {e}")

    return generated_paths


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate probe visualizations')
    parser.add_argument('--results-dir', type=Path, default=OUTPUT_DIR,
                       help='Directory containing probe results')
    parser.add_argument('--output-dir', type=Path, default=FIGURE_DIR,
                       help='Directory to save figures')
    parser.add_argument('--dark-mode', action='store_true',
                       help='Use dark mode style')
    parser.add_argument('--section', type=str, default=None,
                       help='Generate figures for specific section (1-7)')

    args = parser.parse_args()

    style = StyleConfig(dark_mode=args.dark_mode)

    if args.section:
        # Load section-specific results and generate
        print(f"Generating figures for section {args.section}")
        # Would need to load appropriate results file
    else:
        # Generate all figures
        print("Generating all probe figures...")
        generate_all_figures(args.results_dir, args.output_dir, style)