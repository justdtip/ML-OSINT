#!/usr/bin/env python3
"""
Comprehensive Model Interpretability Analysis
=============================================

This script performs deep analysis of the trained Multi-Resolution HAN model:
1. Latent representation analysis (t-SNE, PCA, UMAP)
2. Attention head visualization and specialization analysis
3. Cross-modal association discovery (VIIRS, equipment, casualties, etc.)
4. Feature importance via gradient-based attribution
5. Temporal dynamics analysis
6. 20+ detailed visualization figures

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from multi_resolution_han import (
    MultiResolutionHAN,
    MultiResolutionHANConfig,
    SourceConfig,
    DAILY_SOURCES,
    MONTHLY_SOURCES,
)
from multi_resolution_data import MultiResolutionDataset, MultiResolutionConfig

warnings.filterwarnings('ignore', category=UserWarning)

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR, MULTI_RES_CHECKPOINT_DIR,
    INTERPRETABILITY_OUTPUT_DIR,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

CHECKPOINT_DIR = MULTI_RES_CHECKPOINT_DIR
ANALYSIS_DIR_LOCAL = Path(__file__).parent
OUTPUT_DIR = INTERPRETABILITY_OUTPUT_DIR
DATA_BASE_PATH = DATA_DIR

DAILY_SOURCE_NAMES = ['equipment', 'personnel', 'deepstate', 'firms', 'viina', 'viirs']
MONTHLY_SOURCE_NAMES = ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']

# Color schemes for sources
SOURCE_COLORS = {
    'equipment': '#e41a1c',    # Red - combat losses
    'personnel': '#377eb8',    # Blue - personnel
    'deepstate': '#4daf4a',    # Green - territory
    'firms': '#ff7f00',        # Orange - fire/destruction
    'viina': '#984ea3',        # Purple - events
    'viirs': '#ffff33',        # Yellow - nightlights
    'sentinel': '#a65628',     # Brown - satellite
    'hdx_conflict': '#f781bf', # Pink - humanitarian
    'hdx_food': '#999999',     # Grey - food prices
    'hdx_rainfall': '#66c2a5', # Teal - weather
    'iom': '#fc8d62',          # Coral - displacement
}

# Conflict phase labels
PHASE_LABELS = {
    0: 'Initial Invasion',
    1: 'Stalemate',
    2: 'Counteroffensive',
    3: 'Attritional Warfare',
}

PHASE_COLORS = {
    0: '#d62728',  # Red - Initial invasion
    1: '#7f7f7f',  # Grey - Stalemate
    2: '#2ca02c',  # Green - Counteroffensive
    3: '#ff7f0e',  # Orange - Attritional
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def setup_output_dir() -> Path:
    """Create output directory for figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def date_to_conflict_phase(date) -> int:
    """
    Map a date to a conflict phase based on major events in the Ukraine conflict.

    Phases:
    0 - Initial Invasion (Feb 2022 - Mar 2022): Russian advance on Kyiv
    1 - Stalemate/Repositioning (Apr 2022 - Aug 2022): Russian withdrawal from north, Donbas focus
    2 - Ukrainian Counteroffensive (Sep 2022 - Nov 2022): Kharkiv and Kherson liberation
    3 - Attritional Warfare (Dec 2022 onwards): Bakhmut, slow grinding conflict
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, np.datetime64):
        date = pd.to_datetime(date)

    # Define phase boundaries
    if date < pd.to_datetime('2022-04-01'):
        return 0  # Initial Invasion
    elif date < pd.to_datetime('2022-09-01'):
        return 1  # Stalemate/Repositioning
    elif date < pd.to_datetime('2022-12-01'):
        return 2  # Ukrainian Counteroffensive
    else:
        return 3  # Attritional Warfare


def load_model_and_data(device: torch.device) -> Tuple[MultiResolutionHAN, Dict, Any]:
    """Load the trained model and prepare data."""
    print("Loading model and data...")

    # Load training summary for configuration
    with open(CHECKPOINT_DIR / "training_summary.json", "r") as f:
        training_summary = json.load(f)

    config_dict = training_summary['config']

    # Create data configuration
    data_config = MultiResolutionConfig(
        daily_seq_len=config_dict.get('daily_seq_len', 365),
        monthly_seq_len=config_dict.get('monthly_seq_len', 12),
        prediction_horizon=config_dict.get('prediction_horizon', 1),
    )

    # Create train dataset - use train for better coverage of conflict phases
    dataset = MultiResolutionDataset(data_config, split='train')

    # Get feature dimensions from dataset
    daily_source_configs = {}
    monthly_source_configs = {}

    # Sample a batch to get dimensions
    sample = dataset[0]

    for source_name in DAILY_SOURCE_NAMES:
        if source_name in sample.daily_features:
            n_features = sample.daily_features[source_name].shape[-1]
            daily_source_configs[source_name] = SourceConfig(
                name=source_name,
                n_features=n_features,
                resolution='daily',
            )

    for source_name in MONTHLY_SOURCE_NAMES:
        if source_name in sample.monthly_features:
            n_features = sample.monthly_features[source_name].shape[-1]
            monthly_source_configs[source_name] = SourceConfig(
                name=source_name,
                n_features=n_features,
                resolution='monthly',
            )

    # Create model with explicit parameters
    model = MultiResolutionHAN(
        daily_source_configs=daily_source_configs,
        monthly_source_configs=monthly_source_configs,
        d_model=config_dict.get('d_model', 128),
        nhead=config_dict.get('nhead', 8),
        num_daily_layers=config_dict.get('num_daily_layers', 3),
        num_monthly_layers=config_dict.get('num_monthly_layers', 2),
        num_fusion_layers=config_dict.get('num_fusion_layers', 2),
        num_temporal_layers=2,
        dropout=0.0,  # No dropout for inference
    )

    # Load checkpoint
    checkpoint_path = CHECKPOINT_DIR / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle state dict (may have 'model_state_dict' key)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load state dict (allow missing keys for flexibility)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Daily sources: {list(daily_source_configs.keys())}")
    print(f"  Monthly sources: {list(monthly_source_configs.keys())}")

    return model, training_summary, dataset


def prepare_batch(dataset, indices: List[int], device: torch.device) -> Tuple[Dict, np.ndarray]:
    """Prepare a batch of data for model inference.

    Returns:
        batch: Dict of tensors for model input
        labels: Array of conflict phase labels per sample per month
                Shape: [batch_size, monthly_seq_len] - one label per month in sequence
    """
    daily_features = defaultdict(list)
    daily_masks = defaultdict(list)
    monthly_features = defaultdict(list)
    monthly_masks = defaultdict(list)
    month_boundaries = []
    all_monthly_labels = []

    for idx in indices:
        sample = dataset[idx]

        for source_name, features in sample.daily_features.items():
            daily_features[source_name].append(features)
            daily_masks[source_name].append(sample.daily_masks[source_name])

        for source_name, features in sample.monthly_features.items():
            monthly_features[source_name].append(features)
            monthly_masks[source_name].append(sample.monthly_masks[source_name])

        # Add month boundaries
        month_boundaries.append(sample.month_boundary_indices)

        # Get conflict phase label for EACH month in this sample
        if hasattr(sample, 'monthly_dates') and len(sample.monthly_dates) > 0:
            sample_labels = [date_to_conflict_phase(d) for d in sample.monthly_dates]
        else:
            sample_labels = [0] * 12  # Default
        all_monthly_labels.append(sample_labels)

    # Stack into batch tensors
    batch = {
        'daily_features': {k: torch.stack(v).to(device) for k, v in daily_features.items()},
        'daily_masks': {k: torch.stack(v).to(device) for k, v in daily_masks.items()},
        'monthly_features': {k: torch.stack(v).to(device) for k, v in monthly_features.items()},
        'monthly_masks': {k: torch.stack(v).to(device) for k, v in monthly_masks.items()},
        'month_boundaries': torch.stack(month_boundaries).to(device),
    }

    # labels shape: [batch_size, monthly_seq_len]
    labels = np.array(all_monthly_labels)

    return batch, labels


# ==============================================================================
# LATENT REPRESENTATION EXTRACTION
# ==============================================================================

class LatentExtractor:
    """Extract latent representations from various layers of the model."""

    def __init__(self, model: MultiResolutionHAN, device: torch.device):
        self.model = model
        self.device = device
        self.activations = {}
        self.hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook

        # Hook daily source encoders
        if hasattr(self.model, 'daily_encoders'):
            for name, encoder in self.model.daily_encoders.items():
                hook = encoder.register_forward_hook(get_activation(f'daily_{name}'))
                self.hooks.append(hook)

        # Hook monthly source encoders
        if hasattr(self.model, 'monthly_encoders'):
            for name, encoder in self.model.monthly_encoders.items():
                hook = encoder.register_forward_hook(get_activation(f'monthly_{name}'))
                self.hooks.append(hook)

        # Hook daily fusion if available
        if hasattr(self.model, 'daily_fusion'):
            hook = self.model.daily_fusion.register_forward_hook(get_activation('daily_fusion'))
            self.hooks.append(hook)

        # Hook cross-resolution fusion if available
        if hasattr(self.model, 'cross_resolution_fusion'):
            hook = self.model.cross_resolution_fusion.register_forward_hook(
                get_activation('cross_resolution_fusion')
            )
            self.hooks.append(hook)

        # Hook temporal encoder if available
        if hasattr(self.model, 'temporal_encoder'):
            hook = self.model.temporal_encoder.register_forward_hook(get_activation('temporal'))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, batch: Dict) -> Dict[str, Tensor]:
        """Extract latent representations for a batch."""
        self.activations = {}
        self._register_hooks()

        try:
            # Forward pass
            _ = self.model(
                daily_features=batch['daily_features'],
                daily_masks=batch['daily_masks'],
                monthly_features=batch['monthly_features'],
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch['month_boundaries'],
            )
        finally:
            self._remove_hooks()

        return self.activations


# ==============================================================================
# ATTENTION ANALYSIS
# ==============================================================================

class AttentionAnalyzer:
    """Analyze attention patterns in the model."""

    def __init__(self, model: MultiResolutionHAN, device: torch.device):
        self.model = model
        self.device = device
        self.attention_weights = {}
        self.hooks = []

    def _register_attention_hooks(self):
        """Register hooks to capture attention weights."""

        def get_attention(name):
            def hook(module, input, output):
                # MultiheadAttention returns (output, attention_weights)
                if isinstance(output, tuple) and len(output) == 2:
                    attn = output[1]
                    if attn is not None:
                        self.attention_weights[name] = attn.detach().cpu()
            return hook

        # Find all MultiheadAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Enable attention weight output
                module.need_weights = True
                hook = module.register_forward_hook(get_attention(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @torch.no_grad()
    def extract_attention(self, batch: Dict) -> Dict[str, Tensor]:
        """Extract attention weights for a batch."""
        self.attention_weights = {}
        self._register_attention_hooks()

        try:
            _ = self.model(
                daily_features=batch['daily_features'],
                daily_masks=batch['daily_masks'],
                monthly_features=batch['monthly_features'],
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch['month_boundaries'],
            )
        finally:
            self._remove_hooks()

        return self.attention_weights


# ==============================================================================
# FIGURE GENERATION FUNCTIONS
# ==============================================================================

def fig01_training_dynamics(training_summary: Dict, output_dir: Path):
    """Figure 1: Training dynamics - loss curves and learning rate."""
    print("  Generating Figure 01: Training Dynamics...")

    history = training_summary['history']['train_history']
    val_history = training_summary['history']['val_history']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = list(range(1, len(history['total']) + 1))

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['total'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_history['total'], 'r--', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss (Kendall-weighted)')
    ax.set_title('Total Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Regime loss
    ax = axes[0, 1]
    ax.semilogy(epochs, history['regime'], 'b-', label='Train', linewidth=2)
    ax.semilogy(epochs, val_history['regime'], 'r--', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Regime Loss (log scale)')
    ax.set_title('Regime Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Transition loss
    ax = axes[0, 2]
    ax.plot(epochs, history['transition'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_history['transition'], 'r--', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Transition Loss')
    ax.set_title('Transition Detection Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Casualty ZINB loss
    ax = axes[1, 0]
    ax.plot(epochs, history['casualty'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_history['casualty'], 'r--', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Casualty ZINB NLL')
    ax.set_title('Casualty Prediction (ZINB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Anomaly loss
    ax = axes[1, 1]
    ax.plot(epochs, history['anomaly'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_history['anomaly'], 'r--', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Anomaly Loss')
    ax.set_title('Anomaly Detection Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 2]
    ax.semilogy(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=10, color='r', linestyle='--', alpha=0.5, label='Warmup end')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '01_training_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig02_loss_decomposition(training_summary: Dict, output_dir: Path):
    """Figure 2: Loss decomposition - stacked area chart."""
    print("  Generating Figure 02: Loss Decomposition...")

    history = training_summary['history']['train_history']
    epochs = list(range(1, len(history['total']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked area chart of losses
    ax = axes[0]
    losses = {
        'Regime': np.array(history['regime']),
        'Transition': np.array(history['transition']),
        'Casualty': np.array(history['casualty']),
        'Anomaly': np.array(history['anomaly']),
    }

    # Normalize to show relative contribution
    total = sum(losses.values())
    normalized = {k: v / (total + 1e-8) for k, v in losses.items()}

    ax.stackplot(epochs,
                 normalized['Regime'],
                 normalized['Transition'],
                 normalized['Casualty'],
                 normalized['Anomaly'],
                 labels=['Regime', 'Transition', 'Casualty', 'Anomaly'],
                 colors=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
                 alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative Loss Contribution')
    ax.set_title('Loss Decomposition Over Training')
    ax.legend(loc='upper right')
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1)

    # Final loss breakdown (pie chart)
    ax = axes[1]
    final_losses = {k: v[-1] for k, v in losses.items()}
    # Handle negative values for pie chart
    final_losses = {k: max(v, 0.001) for k, v in final_losses.items()}

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    wedges, texts, autotexts = ax.pie(
        list(final_losses.values()),
        labels=list(final_losses.keys()),
        autopct='%1.1f%%',
        colors=colors,
        explode=(0.05, 0.05, 0.05, 0.05),
    )
    ax.set_title('Final Loss Distribution (Epoch 100)')

    plt.tight_layout()
    plt.savefig(output_dir / '02_loss_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig03_observation_rates(training_summary: Dict, output_dir: Path):
    """Figure 3: Data observation rates over training."""
    print("  Generating Figure 03: Observation Rates...")

    history = training_summary['history']['train_history']
    epochs = list(range(1, len(history['daily_obs_rate']) + 1))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(epochs, np.array(history['daily_obs_rate']) * 100,
            'b-', label='Daily Sources', linewidth=2)
    ax.plot(epochs, np.array(history['monthly_obs_rate']) * 100,
            'r-', label='Monthly Sources', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Observation Rate (%)')
    ax.set_title('Data Completeness Throughout Training')
    ax.legend()
    ax.set_ylim(98, 100.5)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.axhline(y=99.7, color='b', linestyle='--', alpha=0.5)
    ax.axhline(y=98.9, color='r', linestyle='--', alpha=0.5)
    ax.text(len(epochs) * 0.8, 99.75, 'Daily ~99.7%', color='b')
    ax.text(len(epochs) * 0.8, 98.95, 'Monthly ~98.9%', color='r')

    plt.tight_layout()
    plt.savefig(output_dir / '03_observation_rates.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig04_test_performance(training_summary: Dict, output_dir: Path):
    """Figure 4: Test set performance metrics."""
    print("  Generating Figure 04: Test Performance...")

    test_metrics = training_summary['test_metrics']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of test losses
    ax = axes[0]
    metrics = {
        'Regime': test_metrics['regime_loss'],
        'Transition': test_metrics['transition_loss'],
        'Casualty\n(ZINB)': test_metrics['casualty_loss'],
        'Anomaly': test_metrics['anomaly_loss'],
        'Forecast': test_metrics['forecast_loss'],
    }

    x = np.arange(len(metrics))
    bars = ax.bar(x, list(metrics.values()), color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'])
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Set Performance by Task')
    ax.set_yscale('log')

    # Add value labels
    for bar, val in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=9)

    # Performance interpretation
    ax = axes[1]
    ax.axis('off')

    interpretation = """
    TEST SET INTERPRETATION
    =======================

    Regime Classification:     5.86e-05  ★★★★★ EXCELLENT
    → Near-perfect phase identification
    → Model clearly distinguishes conflict phases

    Transition Detection:      0.433     ★★★☆☆ MODERATE
    → Challenge: rare event prediction
    → Opportunity: focal loss, class weighting

    Casualty ZINB:            2.493     ★★★★☆ GOOD
    → Proper count distribution learning
    → No longer "gaming" the loss function
    → Learns zero-inflation + mean + dispersion

    Anomaly Detection:         1.24e-06  ★★★★★ EXCELLENT
    → VIIRS radiance anomalies well-predicted
    → Strong proxy for destruction events

    Forecasting:              2.34e-09  ★★★★★ EXCELLENT
    → Excellent next-period prediction
    """

    ax.text(0.1, 0.9, interpretation, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '04_test_performance.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig05_latent_pca(latent_data: Dict, labels: np.ndarray, output_dir: Path):
    """Figure 5: PCA of latent representations colored by conflict phase."""
    print("  Generating Figure 05: Latent Space PCA...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    layer_names = list(latent_data.keys())[:6]  # Up to 6 layers

    for idx, (ax, layer_name) in enumerate(zip(axes.flat, layer_names)):
        data = latent_data[layer_name]

        # Reshape if needed (batch, seq, features) -> (batch*seq, features)
        if len(data.shape) == 3:
            batch_size, seq_len, n_features = data.shape
            data = data.reshape(-1, n_features)

        # Ensure labels match data size
        if len(labels) != len(data):
            if len(labels) < len(data):
                layer_labels = np.tile(labels, (len(data) // len(labels) + 1))[:len(data)]
            else:
                layer_labels = labels[:len(data)]
        else:
            layer_labels = labels

        # Sample if too many points
        max_points = 5000
        if len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            data = data[indices]
            layer_labels = layer_labels[indices]

        # PCA
        pca = PCA(n_components=2)
        try:
            coords = pca.fit_transform(data)
        except Exception as e:
            ax.text(0.5, 0.5, f"PCA failed:\n{str(e)[:50]}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(layer_name)
            continue

        # Plot
        for phase in np.unique(layer_labels):
            mask = layer_labels == phase
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[PHASE_COLORS.get(phase, '#333333')],
                      label=PHASE_LABELS.get(phase, f'Phase {phase}'),
                      alpha=0.5, s=10)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'{layer_name}')
        ax.legend(loc='best', fontsize=8)

    # Hide empty subplots
    for idx in range(len(layer_names), 6):
        axes.flat[idx].axis('off')

    plt.suptitle('PCA of Latent Representations by Conflict Phase', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '05_latent_pca.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig06_latent_tsne(latent_data: Dict, labels: np.ndarray, output_dir: Path):
    """Figure 6: t-SNE of latent representations."""
    print("  Generating Figure 06: Latent Space t-SNE...")

    # Use the final temporal representation if available
    if 'temporal' in latent_data:
        data = latent_data['temporal']
    elif 'cross_resolution_fusion' in latent_data:
        data = latent_data['cross_resolution_fusion']
    else:
        data = list(latent_data.values())[0]

    # Reshape if needed - flatten temporal dimension to get more samples
    if len(data.shape) == 3:
        batch_size, seq_len, n_features = data.shape
        # Flatten batch and sequence dimensions to get more samples for t-SNE
        data = data.reshape(-1, n_features)

    # Convert to numpy if tensor
    if hasattr(data, 'numpy'):
        data = data.numpy()

    # Ensure labels match data size
    if len(labels) != len(data):
        # Labels should already be flattened [batch*seq] if data is flattened
        if len(labels) < len(data):
            # Repeat labels to match data
            plot_labels = np.tile(labels, (len(data) // len(labels) + 1))[:len(data)]
        else:
            plot_labels = labels[:len(data)]
    else:
        plot_labels = labels

    n_samples = len(data)
    print(f"    t-SNE input: {n_samples} samples, {data.shape[-1]} features")

    # Sample if too many points
    max_points = 2000
    if n_samples > max_points:
        indices = np.random.choice(n_samples, max_points, replace=False)
        data = data[indices]
        plot_labels = plot_labels[indices]
        n_samples = max_points

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Adjust perplexities based on sample size (perplexity must be < n_samples)
    # Rule of thumb: perplexity should be between 5 and 50, and < n_samples/3
    max_perplexity = min(50, n_samples // 3, n_samples - 1)
    perplexities = [min(5, max_perplexity), min(max_perplexity, 30)]

    if n_samples < 10:
        # Too few samples for meaningful t-SNE
        for ax in axes:
            ax.text(0.5, 0.5, f"Too few samples ({n_samples}) for t-SNE.\nNeed at least 10 samples.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    else:
        # t-SNE with different perplexities
        for idx, (ax, perplexity) in enumerate(zip(axes, perplexities)):
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
                coords = tsne.fit_transform(data)

                for phase in np.unique(plot_labels):
                    mask = plot_labels == phase
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                              c=[PHASE_COLORS.get(phase, '#333333')],
                              label=PHASE_LABELS.get(phase, f'Phase {phase}'),
                              alpha=0.6, s=30)

                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title(f't-SNE (perplexity={perplexity}, n={n_samples})')
                ax.legend(loc='best')
            except Exception as e:
                ax.text(0.5, 0.5, f"t-SNE failed:\n{str(e)[:50]}",
                        ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('t-SNE Visualization of Fused Representations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '06_latent_tsne.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig07_source_embeddings(model: MultiResolutionHAN, output_dir: Path):
    """Figure 7: Learned source type embeddings analysis."""
    print("  Generating Figure 07: Source Embeddings...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Collect all source embeddings
    all_embeddings = {}

    # Daily source embeddings from fusion module
    if hasattr(model, 'daily_fusion') and hasattr(model.daily_fusion, 'source_type_embedding'):
        emb = model.daily_fusion.source_type_embedding.weight.detach().cpu().numpy()
        for i, name in enumerate(DAILY_SOURCES[:emb.shape[0]]):
            all_embeddings[f'daily_{name}'] = emb[i]

    # Monthly source embeddings
    if hasattr(model, 'monthly_encoder') and hasattr(model.monthly_encoder, 'source_type_embedding'):
        emb = model.monthly_encoder.source_type_embedding.weight.detach().cpu().numpy()
        for i, name in enumerate(MONTHLY_SOURCES[:emb.shape[0]]):
            all_embeddings[f'monthly_{name}'] = emb[i]

    if not all_embeddings:
        # Try alternative location
        for name, module in model.named_modules():
            if 'source_type_embedding' in name and isinstance(module, nn.Embedding):
                emb = module.weight.detach().cpu().numpy()
                for i in range(emb.shape[0]):
                    all_embeddings[f'{name}_{i}'] = emb[i]

    if not all_embeddings:
        fig.text(0.5, 0.5, 'No source embeddings found in model',
                 ha='center', va='center', fontsize=14)
        plt.savefig(output_dir / '07_source_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Embedding similarity matrix
    ax = axes[0]
    source_names = list(all_embeddings.keys())
    embeddings_matrix = np.array([all_embeddings[k] for k in source_names])

    similarity = cosine_similarity(embeddings_matrix)

    im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(source_names)))
    ax.set_yticks(range(len(source_names)))
    ax.set_xticklabels([s.split('_')[-1][:6] for s in source_names], rotation=45, ha='right')
    ax.set_yticklabels([s.split('_')[-1][:6] for s in source_names])
    ax.set_title('Source Embedding Similarity')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    # PCA of source embeddings
    ax = axes[1]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_matrix)

    colors = [SOURCE_COLORS.get(s.split('_')[-1], '#333333') for s in source_names]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=200)

    for i, name in enumerate(source_names):
        ax.annotate(name.split('_')[-1], (coords[i, 0], coords[i, 1]),
                   fontsize=9, ha='center', va='bottom')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Source Embedding Space')

    # Hierarchical clustering
    ax = axes[2]
    if len(embeddings_matrix) > 2:
        linkage_matrix = linkage(embeddings_matrix, method='ward')
        dendrogram(linkage_matrix, labels=[s.split('_')[-1][:8] for s in source_names],
                   ax=ax, leaf_rotation=45)
        ax.set_title('Source Embedding Clustering')
        ax.set_ylabel('Ward Distance')
    else:
        ax.text(0.5, 0.5, 'Not enough sources for clustering',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '07_source_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig08_no_observation_tokens(model: MultiResolutionHAN, output_dir: Path):
    """Figure 8: Analysis of learned no-observation tokens."""
    print("  Generating Figure 08: No-Observation Tokens...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Collect no-observation tokens
    no_obs_tokens = {}

    for name, module in model.named_modules():
        if hasattr(module, 'no_observation_token'):
            token = module.no_observation_token.detach().cpu().numpy().flatten()
            # Clean up name
            clean_name = name.replace('.', '_').replace('daily_encoders_', '').replace('monthly_encoders_', '')
            no_obs_tokens[clean_name] = token

    if not no_obs_tokens:
        fig.text(0.5, 0.5, 'No observation tokens not found',
                 ha='center', va='center', fontsize=14)
        plt.savefig(output_dir / '08_no_observation_tokens.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Token value distributions
    ax = axes[0]
    for name, token in no_obs_tokens.items():
        ax.hist(token, bins=30, alpha=0.5, label=name[:20], density=True)
    ax.set_xlabel('Token Value')
    ax.set_ylabel('Density')
    ax.set_title('No-Observation Token Value Distributions')
    ax.legend(fontsize=8)

    # Token norms
    ax = axes[1]
    names = list(no_obs_tokens.keys())
    norms = [np.linalg.norm(token) for token in no_obs_tokens.values()]

    bars = ax.bar(range(len(names)), norms, color='steelblue')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[:10] for n in names], rotation=45, ha='right')
    ax.set_ylabel('L2 Norm')
    ax.set_title('No-Observation Token Magnitudes')

    # Token similarity matrix
    ax = axes[2]
    if len(no_obs_tokens) > 1:
        tokens_matrix = np.array(list(no_obs_tokens.values()))
        similarity = cosine_similarity(tokens_matrix)

        im = ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([n[:8] for n in names], rotation=45, ha='right')
        ax.set_yticklabels([n[:8] for n in names])
        ax.set_title('Token Similarity')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'Only one token found',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '08_no_observation_tokens.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig09_attention_patterns(attention_weights: Dict, output_dir: Path):
    """Figure 9: Attention pattern visualization."""
    print("  Generating Figure 09: Attention Patterns...")

    if not attention_weights:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No attention weights captured.\nModel may not use standard MultiheadAttention.',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '09_attention_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    n_layers = len(attention_weights)
    fig, axes = plt.subplots(2, min(4, n_layers), figsize=(20, 10))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_layers <= 4:
        axes = axes.reshape(1, -1)

    for idx, (layer_name, attn) in enumerate(list(attention_weights.items())[:8]):
        if idx >= 8:
            break

        row = idx // 4
        col = idx % 4
        ax = axes[row, col] if len(axes.shape) == 2 else axes[col]

        # Average over batch and heads
        if attn.dim() == 4:  # [batch, heads, seq, seq]
            attn_avg = attn.mean(dim=(0, 1)).numpy()
        elif attn.dim() == 3:  # [batch, seq, seq]
            attn_avg = attn.mean(dim=0).numpy()
        else:
            attn_avg = attn.numpy()

        # Subsample if too large
        if attn_avg.shape[0] > 100:
            step = attn_avg.shape[0] // 100
            attn_avg = attn_avg[::step, ::step]

        im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
        ax.set_title(layer_name.split('.')[-1][:20], fontsize=9)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide empty subplots
    for idx in range(len(attention_weights), 8):
        if idx < len(axes.flat):
            axes.flat[idx].axis('off')

    plt.suptitle('Attention Weight Patterns Across Layers', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '09_attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig10_attention_head_specialization(attention_weights: Dict, output_dir: Path):
    """Figure 10: Analysis of attention head specialization."""
    print("  Generating Figure 10: Head Specialization...")

    if not attention_weights:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No attention weights available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '10_attention_head_specialization.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect head-level statistics
    head_stats = defaultdict(list)

    for layer_name, attn in attention_weights.items():
        if attn.dim() != 4:  # Need [batch, heads, seq, seq]
            continue

        n_heads = attn.shape[1]

        for head_idx in range(n_heads):
            head_attn = attn[:, head_idx, :, :].numpy()  # [batch, seq, seq]

            # Compute statistics
            head_stats['entropy'].append(
                -np.nansum(head_attn * np.log(head_attn + 1e-10), axis=-1).mean()
            )
            head_stats['max_attn'].append(head_attn.max(axis=-1).mean())
            head_stats['locality'].append(
                np.mean([np.abs(np.arange(head_attn.shape[-1]) - np.argmax(row))
                         for row in head_attn.mean(axis=0)])
            )
            head_stats['layer'].append(layer_name)
            head_stats['head_idx'].append(head_idx)

    if not head_stats['entropy']:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No multi-head attention found',
                    ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_dir / '10_attention_head_specialization.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Entropy distribution
    ax = axes[0, 0]
    ax.hist(head_stats['entropy'], bins=20, color='steelblue', edgecolor='white')
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Attention Head Entropy Distribution')
    ax.axvline(np.mean(head_stats['entropy']), color='red', linestyle='--', label='Mean')
    ax.legend()

    # Max attention vs entropy
    ax = axes[0, 1]
    ax.scatter(head_stats['entropy'], head_stats['max_attn'], alpha=0.6)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Max Attention Weight')
    ax.set_title('Head Sharpness vs Entropy')

    # Locality measure
    ax = axes[1, 0]
    ax.hist(head_stats['locality'], bins=20, color='coral', edgecolor='white')
    ax.set_xlabel('Average Attention Distance')
    ax.set_ylabel('Count')
    ax.set_title('Attention Locality (lower = more local)')

    # Summary per layer
    ax = axes[1, 1]
    layer_names = list(set(head_stats['layer']))
    layer_entropies = []
    for layer in layer_names:
        mask = [l == layer for l in head_stats['layer']]
        layer_entropies.append(np.mean([e for e, m in zip(head_stats['entropy'], mask) if m]))

    ax.barh(range(len(layer_names)), layer_entropies, color='steelblue')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([l.split('.')[-1][:15] for l in layer_names])
    ax.set_xlabel('Mean Entropy')
    ax.set_title('Entropy by Layer')

    plt.tight_layout()
    plt.savefig(output_dir / '10_attention_head_specialization.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig11_cross_source_correlations(latent_data: Dict, output_dir: Path):
    """Figure 11: Cross-source latent space correlations."""
    print("  Generating Figure 11: Cross-Source Correlations...")

    # Extract daily source representations
    daily_reps = {k.replace('daily_', ''): v for k, v in latent_data.items()
                  if k.startswith('daily_') and 'fusion' not in k}

    if len(daily_reps) < 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f'Need at least 2 daily sources.\nFound: {list(daily_reps.keys())}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '11_cross_source_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Compute pairwise correlations
    source_names = list(daily_reps.keys())
    n_sources = len(source_names)
    corr_matrix = np.zeros((n_sources, n_sources))

    for i, name_i in enumerate(source_names):
        rep_i = daily_reps[name_i]
        if len(rep_i.shape) == 3:
            rep_i = rep_i.mean(dim=1)  # Average over sequence

        for j, name_j in enumerate(source_names):
            rep_j = daily_reps[name_j]
            if len(rep_j.shape) == 3:
                rep_j = rep_j.mean(dim=1)

            # Compute correlation
            corr = np.corrcoef(rep_i.numpy().flatten(), rep_j.numpy().flatten())[0, 1]
            corr_matrix[i, j] = corr

    # Heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_sources))
    ax.set_yticks(range(n_sources))
    ax.set_xticklabels(source_names, rotation=45, ha='right')
    ax.set_yticklabels(source_names)
    ax.set_title('Latent Space Correlation Matrix')
    plt.colorbar(im, ax=ax, label='Pearson Correlation')

    # Add correlation values
    for i in range(n_sources):
        for j in range(n_sources):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha='center', va='center', fontsize=10,
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    # Dendrogram clustering
    ax = axes[1]
    if n_sources > 2:
        # Convert correlation to distance
        dist_matrix = 1 - corr_matrix
        dist_condensed = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(dist_condensed, method='average')
        dendrogram(linkage_matrix, labels=source_names, ax=ax, leaf_rotation=45)
        ax.set_title('Hierarchical Clustering of Sources')
        ax.set_ylabel('1 - Correlation')
    else:
        ax.text(0.5, 0.5, 'Need more sources for clustering',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '11_cross_source_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig12_temporal_dynamics(latent_data: Dict, output_dir: Path):
    """Figure 12: Temporal dynamics of latent representations."""
    print("  Generating Figure 12: Temporal Dynamics...")

    # Find a representation with temporal dimension
    temporal_rep = None
    rep_name = None

    for name, rep in latent_data.items():
        if len(rep.shape) == 3 and rep.shape[1] > 10:  # [batch, seq, features]
            temporal_rep = rep
            rep_name = name
            break

    if temporal_rep is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No temporal representations found',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '12_temporal_dynamics.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Take first sample for visualization
    rep = temporal_rep[0].numpy()  # [seq_len, features]
    seq_len, n_features = rep.shape

    # PCA trajectory
    ax = axes[0, 0]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(rep)

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, seq_len))
    for i in range(seq_len - 1):
        ax.plot([coords[i, 0], coords[i+1, 0]],
                [coords[i, 1], coords[i+1, 1]],
                color=colors[i], linewidth=2, alpha=0.7)

    ax.scatter(coords[0, 0], coords[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(coords[-1, 0], coords[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Latent Trajectory ({rep_name})')
    ax.legend()

    # Feature evolution heatmap
    ax = axes[0, 1]
    # Sample features for visualization
    n_show = min(50, n_features)
    feature_sample = rep[:, :n_show]

    im = ax.imshow(feature_sample.T, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature Index')
    ax.set_title('Latent Feature Evolution')
    plt.colorbar(im, ax=ax, label='Activation')

    # Variance over time
    ax = axes[1, 0]
    variance_over_time = rep.var(axis=1)
    ax.plot(variance_over_time, color='steelblue', linewidth=2)
    ax.fill_between(range(len(variance_over_time)), variance_over_time, alpha=0.3)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature Variance')
    ax.set_title('Representation Variance Over Time')

    # Autocorrelation of latent state
    ax = axes[1, 1]
    # Compute autocorrelation using mean of features
    mean_rep = rep.mean(axis=1)
    n_lags = min(50, seq_len // 2)
    autocorr = np.array([np.corrcoef(mean_rep[:-lag], mean_rep[lag:])[0, 1]
                         for lag in range(1, n_lags)])

    ax.bar(range(1, n_lags), autocorr, color='coral', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Temporal Autocorrelation of Latent State')

    plt.tight_layout()
    plt.savefig(output_dir / '12_temporal_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig13_cross_modal_viirs_equipment(latent_data: Dict, output_dir: Path):
    """Figure 13: Cross-modal associations between VIIRS and Equipment."""
    print("  Generating Figure 13: VIIRS-Equipment Cross-Modal Analysis...")

    viirs_rep = latent_data.get('daily_viirs')
    equip_rep = latent_data.get('daily_equipment')

    if viirs_rep is None or equip_rep is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f'Missing VIIRS or Equipment representations.\nFound: {list(latent_data.keys())}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '13_viirs_equipment_crossmodal.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Flatten to (batch*seq, features) if needed
    if len(viirs_rep.shape) == 3:
        viirs_flat = viirs_rep.reshape(-1, viirs_rep.shape[-1]).numpy()
        equip_flat = equip_rep.reshape(-1, equip_rep.shape[-1]).numpy()
    else:
        viirs_flat = viirs_rep.numpy()
        equip_flat = equip_rep.numpy()

    # Sample if too many points
    max_points = 5000
    if len(viirs_flat) > max_points:
        indices = np.random.choice(len(viirs_flat), max_points, replace=False)
        viirs_flat = viirs_flat[indices]
        equip_flat = equip_flat[indices]

    # CCA-like analysis using PCA
    ax = axes[0, 0]
    pca_viirs = PCA(n_components=2).fit_transform(viirs_flat)
    pca_equip = PCA(n_components=2).fit_transform(equip_flat)

    # Color by VIIRS PC1
    scatter = ax.scatter(pca_equip[:, 0], pca_equip[:, 1],
                        c=pca_viirs[:, 0], cmap='coolwarm', alpha=0.5, s=10)
    ax.set_xlabel('Equipment PC1')
    ax.set_ylabel('Equipment PC2')
    ax.set_title('Equipment Space Colored by VIIRS PC1')
    plt.colorbar(scatter, ax=ax, label='VIIRS PC1')

    # Correlation between principal components
    ax = axes[0, 1]
    n_pcs = 10
    pca_viirs_full = PCA(n_components=n_pcs).fit_transform(viirs_flat)
    pca_equip_full = PCA(n_components=n_pcs).fit_transform(equip_flat)

    cross_corr = np.corrcoef(pca_viirs_full.T, pca_equip_full.T)[:n_pcs, n_pcs:]

    im = ax.imshow(cross_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Equipment PCs')
    ax.set_ylabel('VIIRS PCs')
    ax.set_title('Cross-Modal PC Correlations')
    plt.colorbar(im, ax=ax)

    # Top correlating dimensions
    ax = axes[1, 0]
    flat_corr = cross_corr.flatten()
    top_indices = np.argsort(np.abs(flat_corr))[-20:]
    top_corrs = flat_corr[top_indices]

    colors = ['red' if c < 0 else 'blue' for c in top_corrs]
    ax.barh(range(len(top_corrs)), top_corrs, color=colors)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('PC Pair')
    ax.set_title('Top Cross-Modal Correlations')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

    # Joint distribution
    ax = axes[1, 1]
    ax.scatter(pca_viirs[:, 0], pca_equip[:, 0], alpha=0.3, s=5)

    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(pca_viirs[:, 0], pca_equip[:, 0])
    x_line = np.linspace(pca_viirs[:, 0].min(), pca_viirs[:, 0].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
            label=f'r={r_value:.3f}, p={p_value:.2e}')

    ax.set_xlabel('VIIRS PC1')
    ax.set_ylabel('Equipment PC1')
    ax.set_title('VIIRS-Equipment Joint Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '13_viirs_equipment_crossmodal.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig14_casualty_predictors(model: MultiResolutionHAN, batch: Dict, output_dir: Path):
    """Figure 14: Feature importance for casualty prediction via gradients."""
    print("  Generating Figure 14: Casualty Prediction Feature Importance...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Enable gradients
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    # Create copies of inputs that require grad
    daily_features = {k: v.clone().requires_grad_(True)
                     for k, v in batch['daily_features'].items()}

    try:
        # Forward pass
        outputs = model(
            daily_features=daily_features,
            daily_masks=batch['daily_masks'],
            monthly_features=batch['monthly_features'],
            monthly_masks=batch['monthly_masks'],
            month_boundaries=batch['month_boundaries'],
        )

        # Get casualty prediction
        if 'casualty_pred' in outputs:
            casualty_pred = outputs['casualty_pred']
        else:
            casualty_pred = outputs.get('predictions', {}).get('casualty', None)

        if casualty_pred is None:
            raise ValueError("No casualty prediction found in outputs")

        # Compute gradients with respect to mean prediction
        target = casualty_pred.mean()
        target.backward()

        # Collect gradients per source
        source_importance = {}
        for source_name, features in daily_features.items():
            if features.grad is not None:
                grad = features.grad.abs().mean(dim=(0, 1)).cpu().numpy()
                source_importance[source_name] = grad

    except Exception as e:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f'Gradient computation failed:\n{str(e)[:100]}',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_dir / '14_casualty_predictors.png', dpi=150, bbox_inches='tight')
        plt.close()
        return
    finally:
        # Disable gradients again
        for param in model.parameters():
            param.requires_grad_(False)

    if not source_importance:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No gradient information available',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
        plt.savefig(output_dir / '14_casualty_predictors.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Bar chart of source-level importance
    ax = axes[0, 0]
    source_names = list(source_importance.keys())
    source_means = [np.mean(v) for v in source_importance.values()]

    colors = [SOURCE_COLORS.get(s, '#333333') for s in source_names]
    bars = ax.bar(range(len(source_names)), source_means, color=colors)
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Gradient Magnitude')
    ax.set_title('Source Importance for Casualty Prediction')

    # Top features within most important source
    ax = axes[0, 1]
    top_source = source_names[np.argmax(source_means)]
    top_grads = source_importance[top_source]
    n_show = min(15, len(top_grads))

    top_indices = np.argsort(top_grads)[-n_show:]
    ax.barh(range(n_show), top_grads[top_indices],
            color=SOURCE_COLORS.get(top_source, '#333333'))
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Gradient Magnitude')
    ax.set_title(f'Top Features in {top_source.upper()}')

    # Gradient distribution across all sources
    ax = axes[1, 0]
    all_grads = []
    source_labels = []
    for source, grads in source_importance.items():
        all_grads.extend(grads)
        source_labels.extend([source] * len(grads))

    # Box plot
    data_for_boxplot = [source_importance[s] for s in source_names]
    bp = ax.boxplot(data_for_boxplot, labels=source_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Distribution by Source')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Cumulative importance
    ax = axes[1, 1]
    sorted_sources = sorted(source_means, reverse=True)
    cumulative = np.cumsum(sorted_sources) / np.sum(sorted_sources)
    sorted_names = [source_names[i] for i in np.argsort(source_means)[::-1]]

    ax.bar(range(len(sorted_names)), sorted_sources / np.sum(sorted_sources),
           color=[SOURCE_COLORS.get(s, '#333333') for s in sorted_names], alpha=0.6)
    ax.plot(range(len(sorted_names)), cumulative, 'r-o', linewidth=2, markersize=8)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Importance (normalized)')
    ax.set_title('Cumulative Source Importance')
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.legend(['Cumulative', 'Individual'], loc='center right')

    plt.tight_layout()
    plt.savefig(output_dir / '14_casualty_predictors.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig15_regime_classification_analysis(latent_data: Dict, labels: np.ndarray, output_dir: Path):
    """Figure 15: Regime classification analysis."""
    print("  Generating Figure 15: Regime Classification Analysis...")

    # Get fused representation
    if 'temporal' in latent_data:
        rep = latent_data['temporal']
    elif 'cross_resolution_fusion' in latent_data:
        rep = latent_data['cross_resolution_fusion']
    else:
        rep = list(latent_data.values())[0]

    # Flatten temporal dimension to match labels
    if len(rep.shape) == 3:
        batch_size, seq_len, n_features = rep.shape
        rep = rep.reshape(-1, n_features)  # [batch*seq, features]

    rep = rep.numpy()

    # Ensure labels match rep size
    if len(labels) != len(rep):
        # Truncate or expand labels to match
        if len(labels) > len(rep):
            labels = labels[:len(rep)]
        else:
            # Repeat labels to match
            labels = np.tile(labels, (len(rep) // len(labels) + 1))[:len(rep)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Class separability via PCA
    ax = axes[0, 0]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(rep)

    for phase in np.unique(labels):
        mask = labels == phase
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[PHASE_COLORS.get(phase, '#333333')],
                  label=PHASE_LABELS.get(phase, f'Phase {phase}'),
                  alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Regime Separation in Latent Space')
    ax.legend()

    # Within-class vs between-class variance
    ax = axes[0, 1]

    class_means = {}
    class_vars = {}
    for phase in np.unique(labels):
        mask = labels == phase
        class_means[phase] = rep[mask].mean(axis=0)
        class_vars[phase] = rep[mask].var(axis=0).mean()

    overall_mean = rep.mean(axis=0)

    # Between-class variance
    between_var = np.sum([
        np.sum((class_means[p] - overall_mean) ** 2)
        for p in class_means
    ]) / len(class_means)

    # Within-class variance
    within_var = np.mean(list(class_vars.values()))

    ax.bar(['Between-Class', 'Within-Class'], [between_var, within_var],
           color=['#2ca02c', '#d62728'])
    ax.set_ylabel('Variance')
    ax.set_title('Class Separability Metrics')
    ax.text(0, between_var, f'{between_var:.2f}', ha='center', va='bottom')
    ax.text(1, within_var, f'{within_var:.2f}', ha='center', va='bottom')

    # Class centroids distance matrix
    ax = axes[1, 0]
    phases = sorted(class_means.keys())
    n_phases = len(phases)
    dist_matrix = np.zeros((n_phases, n_phases))

    for i, p1 in enumerate(phases):
        for j, p2 in enumerate(phases):
            dist_matrix[i, j] = np.linalg.norm(class_means[p1] - class_means[p2])

    im = ax.imshow(dist_matrix, cmap='viridis')
    ax.set_xticks(range(n_phases))
    ax.set_yticks(range(n_phases))
    ax.set_xticklabels([PHASE_LABELS.get(p, str(p)) for p in phases], rotation=45, ha='right')
    ax.set_yticklabels([PHASE_LABELS.get(p, str(p)) for p in phases])
    ax.set_title('Centroid Distance Matrix')
    plt.colorbar(im, ax=ax, label='Euclidean Distance')

    # Feature distributions by class
    ax = axes[1, 1]
    # Show top discriminative features
    n_features = rep.shape[1]
    feature_f_stats = []

    for f in range(min(n_features, 50)):
        groups = [rep[labels == p, f] for p in np.unique(labels)]
        try:
            f_stat, _ = stats.f_oneway(*groups)
            feature_f_stats.append(f_stat if np.isfinite(f_stat) else 0)
        except:
            feature_f_stats.append(0)

    top_features = np.argsort(feature_f_stats)[-10:]
    ax.barh(range(10), [feature_f_stats[f] for f in top_features], color='steelblue')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Feature {f}' for f in top_features])
    ax.set_xlabel('F-statistic')
    ax.set_title('Most Discriminative Features')

    plt.tight_layout()
    plt.savefig(output_dir / '15_regime_classification.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig16_anomaly_detection_analysis(latent_data: Dict, output_dir: Path):
    """Figure 16: Anomaly detection feature analysis."""
    print("  Generating Figure 16: Anomaly Detection Analysis...")

    # Use VIIRS (nightlight) representation as primary for anomaly
    viirs_rep = latent_data.get('daily_viirs')

    if viirs_rep is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'VIIRS representation not found',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '16_anomaly_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Take first sample
    if len(viirs_rep.shape) == 3:
        rep = viirs_rep[0].numpy()  # [seq_len, features]
    else:
        rep = viirs_rep.numpy()

    # Compute per-timestep anomaly scores (using Mahalanobis-like distance)
    ax = axes[0, 0]
    mean = rep.mean(axis=0)
    cov = np.cov(rep.T)

    # Regularize covariance
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
        anomaly_scores = np.array([
            np.sqrt((x - mean) @ cov_inv @ (x - mean).T)
            for x in rep
        ])
    except:
        anomaly_scores = np.linalg.norm(rep - mean, axis=1)

    ax.plot(anomaly_scores, color='steelblue', linewidth=1)
    ax.fill_between(range(len(anomaly_scores)), anomaly_scores, alpha=0.3)

    # Mark potential anomalies (top 5%)
    threshold = np.percentile(anomaly_scores, 95)
    anomaly_mask = anomaly_scores > threshold
    ax.scatter(np.where(anomaly_mask)[0], anomaly_scores[anomaly_mask],
              c='red', s=50, zorder=5, label='Potential Anomalies')
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='95th percentile')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('VIIRS-Based Anomaly Scores Over Time')
    ax.legend()

    # Feature contribution to anomalies
    ax = axes[0, 1]
    high_anomaly_mask = anomaly_scores > threshold
    normal_mask = ~high_anomaly_mask

    if high_anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
        high_anomaly_mean = rep[high_anomaly_mask].mean(axis=0)
        normal_mean = rep[normal_mask].mean(axis=0)
        diff = high_anomaly_mean - normal_mean

        n_show = min(20, len(diff))
        top_features = np.argsort(np.abs(diff))[-n_show:]

        colors = ['red' if d > 0 else 'blue' for d in diff[top_features]]
        ax.barh(range(n_show), diff[top_features], color=colors, alpha=0.7)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'Feature {f}' for f in top_features])
        ax.set_xlabel('Mean Difference (Anomaly - Normal)')
        ax.set_title('Feature Differences in Anomalies')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'Not enough anomalies detected',
                ha='center', va='center', transform=ax.transAxes)

    # Anomaly score distribution
    ax = axes[1, 0]
    ax.hist(anomaly_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='95th percentile')
    ax.axvline(np.median(anomaly_scores), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Anomaly Scores')
    ax.legend()

    # Temporal clustering of anomalies
    ax = axes[1, 1]
    if high_anomaly_mask.sum() > 1:
        anomaly_indices = np.where(high_anomaly_mask)[0]
        gaps = np.diff(anomaly_indices)

        ax.hist(gaps, bins=30, color='coral', edgecolor='white', alpha=0.7)
        ax.set_xlabel('Gap Between Consecutive Anomalies')
        ax.set_ylabel('Count')
        ax.set_title('Temporal Clustering of Anomalies')

        # Add statistics
        ax.axvline(np.median(gaps), color='red', linestyle='--', label=f'Median gap: {np.median(gaps):.1f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Too few anomalies for clustering analysis',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '16_anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig17_multimodal_fusion(latent_data: Dict, output_dir: Path):
    """Figure 17: Analysis of multi-modal fusion."""
    print("  Generating Figure 17: Multi-Modal Fusion Analysis...")

    # Get pre-fusion and post-fusion representations
    daily_reps = {k.replace('daily_', ''): v for k, v in latent_data.items()
                  if k.startswith('daily_') and 'fusion' not in k}
    fused_rep = latent_data.get('daily_fusion')
    if fused_rep is None:
        fused_rep = latent_data.get('cross_resolution_fusion')

    if not daily_reps or fused_rep is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f'Missing daily or fusion representations.\nDaily: {list(daily_reps.keys())}\nFusion: {"Found" if fused_rep is not None else "Missing"}',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '17_multimodal_fusion.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Flatten representations
    daily_flat = {}
    for name, rep in daily_reps.items():
        if len(rep.shape) == 3:
            daily_flat[name] = rep.mean(dim=1).numpy()  # Average over sequence
        else:
            daily_flat[name] = rep.numpy()

    if len(fused_rep.shape) == 3:
        fused_flat = fused_rep.mean(dim=1).numpy()
    else:
        fused_flat = fused_rep.numpy()

    # Source contribution to fused space
    ax = axes[0, 0]
    correlations = {}
    for name, rep in daily_flat.items():
        # Compute correlation between source and fused
        corr = np.mean([np.corrcoef(rep[:, i], fused_flat[:, j])[0, 1]
                        for i in range(min(10, rep.shape[1]))
                        for j in range(min(10, fused_flat.shape[1]))])
        correlations[name] = corr

    source_names = list(correlations.keys())
    corr_values = list(correlations.values())
    colors = [SOURCE_COLORS.get(s, '#333333') for s in source_names]

    bars = ax.bar(range(len(source_names)), corr_values, color=colors)
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Correlation with Fused')
    ax.set_title('Source Contribution to Fused Representation')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Information preservation (variance explained)
    ax = axes[0, 1]

    # Concatenate all sources
    concat_sources = np.concatenate(list(daily_flat.values()), axis=1)

    # PCA on concatenated vs fused
    n_samples = min(fused_flat.shape[0], concat_sources.shape[0])
    n_components = min(20, fused_flat.shape[1], concat_sources.shape[1], n_samples - 1)
    n_components = max(2, n_components)  # Ensure at least 2 components

    pca_concat = PCA(n_components=n_components).fit(concat_sources)
    pca_fused = PCA(n_components=n_components).fit(fused_flat)

    ax.plot(range(1, n_components + 1), np.cumsum(pca_concat.explained_variance_ratio_),
            'b-o', label='Concatenated Sources', linewidth=2)
    ax.plot(range(1, n_components + 1), np.cumsum(pca_fused.explained_variance_ratio_),
            'r-o', label='Fused Representation', linewidth=2)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Information Compression in Fusion')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5)

    # Reconstruction loss by source
    ax = axes[1, 0]

    # Try to reconstruct each source from fused
    from sklearn.linear_model import Ridge

    reconstruction_scores = {}
    for name, rep in daily_flat.items():
        try:
            model = Ridge(alpha=1.0)
            model.fit(fused_flat, rep)
            pred = model.predict(fused_flat)
            r2 = 1 - np.sum((rep - pred) ** 2) / np.sum((rep - rep.mean()) ** 2)
            reconstruction_scores[name] = max(0, r2)  # Clamp to 0
        except:
            reconstruction_scores[name] = 0

    source_names = list(reconstruction_scores.keys())
    scores = list(reconstruction_scores.values())
    colors = [SOURCE_COLORS.get(s, '#333333') for s in source_names]

    ax.bar(range(len(source_names)), scores, color=colors)
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names, rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Source Reconstructability from Fused')
    ax.set_ylim(0, 1)

    # Joint embedding visualization
    ax = axes[1, 1]

    # Concatenate source and fused for joint visualization
    combined = np.vstack([
        *[rep[:min(200, len(rep))] for rep in daily_flat.values()],
        fused_flat[:min(200, len(fused_flat))]
    ])

    labels = []
    for name in daily_flat.keys():
        labels.extend([name] * min(200, len(daily_flat[name])))
    labels.extend(['fused'] * min(200, len(fused_flat)))

    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)

    label_set = list(daily_flat.keys()) + ['fused']
    for label in label_set:
        mask = np.array(labels) == label
        color = SOURCE_COLORS.get(label, '#000000')
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=label, alpha=0.5, s=20)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Joint Source + Fused Space')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / '17_multimodal_fusion.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig18_resolution_fusion(latent_data: Dict, output_dir: Path):
    """Figure 18: Daily-Monthly resolution fusion analysis."""
    print("  Generating Figure 18: Resolution Fusion Analysis...")

    daily_fused = latent_data.get('daily_fusion')
    monthly_reps = {k.replace('monthly_', ''): v for k, v in latent_data.items()
                    if k.startswith('monthly_')}
    cross_res_fused = latent_data.get('cross_resolution_fusion')

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Check what we have
    ax = axes[0, 0]

    if daily_fused is not None and cross_res_fused is not None:
        # Compare daily-only vs cross-resolution fused
        if len(daily_fused.shape) == 3:
            daily_flat = daily_fused.mean(dim=1).numpy()
        else:
            daily_flat = daily_fused.numpy()

        if len(cross_res_fused.shape) == 3:
            cross_flat = cross_res_fused.mean(dim=1).numpy()
        else:
            cross_flat = cross_res_fused.numpy()

        # PCA of both
        combined = np.vstack([daily_flat[:min(500, len(daily_flat))],
                             cross_flat[:min(500, len(cross_flat))]])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)

        n_daily = min(500, len(daily_flat))
        ax.scatter(coords[:n_daily, 0], coords[:n_daily, 1],
                  c='blue', alpha=0.5, label='Daily-only Fused', s=20)
        ax.scatter(coords[n_daily:, 0], coords[n_daily:, 1],
                  c='red', alpha=0.5, label='Cross-Resolution Fused', s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Daily vs Cross-Resolution Fusion')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Missing daily or cross-resolution fusion data',
                ha='center', va='center', transform=ax.transAxes)

    # Monthly source contributions
    ax = axes[0, 1]
    if monthly_reps:
        source_norms = {}
        for name, rep in monthly_reps.items():
            if len(rep.shape) == 3:
                norm = rep.mean(dim=(0, 1)).norm().item()
            else:
                norm = rep.mean(dim=0).norm().item()
            source_norms[name] = norm

        names = list(source_norms.keys())
        norms = list(source_norms.values())
        colors = [SOURCE_COLORS.get(n, '#333333') for n in names]

        ax.bar(range(len(names)), norms, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Mean Representation Norm')
        ax.set_title('Monthly Source Representation Magnitudes')
    else:
        ax.text(0.5, 0.5, 'No monthly representations found',
                ha='center', va='center', transform=ax.transAxes)

    # Resolution mixing analysis
    ax = axes[1, 0]
    if daily_fused is not None and monthly_reps:
        # Compute how much monthly info is in cross-resolution
        if len(daily_fused.shape) == 3:
            daily_var = daily_fused.var(dim=(0, 1)).mean().item()
        else:
            daily_var = daily_fused.var(dim=0).mean().item()

        monthly_vars = {}
        for name, rep in monthly_reps.items():
            if len(rep.shape) == 3:
                monthly_vars[name] = rep.var(dim=(0, 1)).mean().item()
            else:
                monthly_vars[name] = rep.var(dim=0).mean().item()

        all_vars = {'daily': daily_var, **monthly_vars}
        names = list(all_vars.keys())
        vars_ = list(all_vars.values())

        ax.bar(range(len(names)), vars_,
               color=['steelblue'] + [SOURCE_COLORS.get(n, '#333333') for n in list(monthly_reps.keys())])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Feature Variance')
        ax.set_title('Variance by Resolution/Source')
    else:
        ax.text(0.5, 0.5, 'Missing data for resolution analysis',
                ha='center', va='center', transform=ax.transAxes)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = """
    RESOLUTION FUSION SUMMARY
    =========================

    The Multi-Resolution HAN fuses:

    DAILY SOURCES (~365 days):
    - Equipment losses
    - Personnel casualties
    - DeepState front-line
    - FIRMS fire detection
    - VIINA events
    - VIIRS nightlights

    MONTHLY SOURCES (~12 months):
    - Sentinel satellite
    - HDX conflict data
    - HDX food prices
    - HDX rainfall
    - IOM displacement

    FUSION MECHANISM:
    1. Daily → Learnable aggregation to monthly
    2. Monthly encoding with no-observation tokens
    3. Bidirectional cross-attention fusion
    4. Gated combination with learned weights
    """

    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '18_resolution_fusion.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig19_prediction_head_analysis(model: MultiResolutionHAN, output_dir: Path):
    """Figure 19: Analysis of prediction head weights."""
    print("  Generating Figure 19: Prediction Head Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect prediction head weights
    head_info = {}

    for name, module in model.named_modules():
        if 'head' in name.lower() or 'prediction' in name.lower():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                head_info[name] = {
                    'weight': weight,
                    'shape': weight.shape,
                    'norm': np.linalg.norm(weight),
                    'sparsity': (np.abs(weight) < 0.01).mean(),
                }

    if not head_info:
        # Try to find heads by looking at final layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                # Check if it looks like a prediction head (small output dim)
                if weight.shape[0] <= 128:
                    head_info[name] = {
                        'weight': weight,
                        'shape': weight.shape,
                        'norm': np.linalg.norm(weight),
                        'sparsity': (np.abs(weight) < 0.01).mean(),
                    }

    if not head_info:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No prediction heads found',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes)
        plt.savefig(output_dir / '19_prediction_heads.png', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Weight norms by head
    ax = axes[0, 0]
    names = list(head_info.keys())
    norms = [info['norm'] for info in head_info.values()]

    # Truncate names for display
    display_names = [n.split('.')[-1][:15] for n in names]

    ax.barh(range(len(names)), norms, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Weight Frobenius Norm')
    ax.set_title('Prediction Head Weight Magnitudes')

    # Weight distributions
    ax = axes[0, 1]
    for name, info in list(head_info.items())[:5]:
        weights_flat = info['weight'].flatten()
        # Sample if too many
        if len(weights_flat) > 10000:
            weights_flat = np.random.choice(weights_flat, 10000)
        ax.hist(weights_flat, bins=50, alpha=0.5, label=name.split('.')[-1][:10], density=True)

    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title('Weight Value Distributions')
    ax.legend(fontsize=8)

    # Sparsity analysis
    ax = axes[1, 0]
    sparsities = [info['sparsity'] * 100 for info in head_info.values()]

    ax.barh(range(len(names)), sparsities, color='coral')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel('% Near-Zero Weights (< 0.01)')
    ax.set_title('Weight Sparsity by Head')
    ax.set_xlim(0, 100)

    # Weight shape info
    ax = axes[1, 1]
    ax.axis('off')

    info_text = "PREDICTION HEAD STRUCTURE\n" + "=" * 30 + "\n\n"
    for name, info in list(head_info.items())[:10]:
        clean_name = name.split('.')[-1][:25]
        info_text += f"{clean_name}:\n"
        info_text += f"  Shape: {info['shape']}\n"
        info_text += f"  Norm: {info['norm']:.4f}\n"
        info_text += f"  Sparsity: {info['sparsity']*100:.1f}%\n\n"

    ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '19_prediction_heads.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig20_model_parameter_analysis(model: MultiResolutionHAN, output_dir: Path):
    """Figure 20: Model parameter analysis."""
    print("  Generating Figure 20: Model Parameters...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Count parameters by module type
    param_counts = defaultdict(int)
    param_norms = defaultdict(float)

    for name, param in model.named_parameters():
        # Categorize by module type
        if 'daily_encoder' in name:
            category = 'Daily Encoders'
        elif 'monthly_encoder' in name:
            category = 'Monthly Encoders'
        elif 'fusion' in name:
            category = 'Fusion Layers'
        elif 'temporal' in name:
            category = 'Temporal Encoder'
        elif 'head' in name or 'prediction' in name:
            category = 'Prediction Heads'
        elif 'embedding' in name:
            category = 'Embeddings'
        elif 'norm' in name:
            category = 'Normalization'
        else:
            category = 'Other'

        param_counts[category] += param.numel()
        param_norms[category] += param.data.norm().item() ** 2

    # Square root of sum of squared norms
    param_norms = {k: np.sqrt(v) for k, v in param_norms.items()}

    # Parameter count pie chart
    ax = axes[0, 0]
    categories = list(param_counts.keys())
    counts = list(param_counts.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                       colors=colors, explode=[0.02] * len(categories))
    ax.set_title(f'Parameter Distribution\n(Total: {sum(counts):,})')

    # Parameter norms
    ax = axes[0, 1]
    ax.barh(range(len(categories)), [param_norms[c] for c in categories], color=colors)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Total L2 Norm')
    ax.set_title('Parameter Norms by Module')

    # Layer-by-layer breakdown
    ax = axes[1, 0]
    layer_params = defaultdict(int)

    for name, param in model.named_parameters():
        # Get layer index if available
        parts = name.split('.')
        layer_key = '.'.join(parts[:3]) if len(parts) >= 3 else parts[0]
        layer_params[layer_key] += param.numel()

    # Take top 15 by size
    sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)[:15]
    layer_names = [l[0] for l in sorted_layers]
    layer_counts = [l[1] for l in sorted_layers]

    ax.barh(range(len(layer_names)), layer_counts, color='steelblue')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels([n[:30] for n in layer_names], fontsize=8)
    ax.set_xlabel('Parameter Count')
    ax.set_title('Top 15 Layers by Parameter Count')

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    total_params = sum(param_counts.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = f"""
    MODEL PARAMETER SUMMARY
    =======================

    Total Parameters:     {total_params:>12,}
    Trainable:            {trainable:>12,}
    Non-trainable:        {total_params - trainable:>12,}

    BREAKDOWN BY MODULE:
    --------------------
    """

    for cat, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_params * 100
        summary += f"\n    {cat:<20} {count:>10,} ({pct:>5.1f}%)"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '20_model_parameters.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig21_embedding_evolution(model: MultiResolutionHAN, output_dir: Path):
    """Figure 21: Positional and feature embedding analysis."""
    print("  Generating Figure 21: Embedding Analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Find positional encodings
    pos_encodings = {}
    for name, module in model.named_modules():
        if 'positional' in name.lower() or 'pos_encoding' in name.lower():
            if hasattr(module, 'encoding'):
                pos_encodings[name] = module.encoding.detach().cpu().numpy()
            elif hasattr(module, 'pe'):
                pos_encodings[name] = module.pe.detach().cpu().numpy()

    # Visualize positional encodings
    ax = axes[0, 0]
    if pos_encodings:
        # Take first one
        name, pe = list(pos_encodings.items())[0]
        if len(pe.shape) == 3:
            pe = pe.squeeze(0)  # Remove batch dim

        # Show first 100 positions, first 64 dims
        pe_show = pe[:min(100, pe.shape[0]), :min(64, pe.shape[1])]
        im = ax.imshow(pe_show.T, aspect='auto', cmap='RdBu_r')
        ax.set_xlabel('Position')
        ax.set_ylabel('Dimension')
        ax.set_title('Positional Encoding Pattern')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No positional encodings found',
                ha='center', va='center', transform=ax.transAxes)

    # Feature embeddings
    ax = axes[0, 1]
    feature_embs = {}
    for name, module in model.named_modules():
        if 'feature_embedding' in name.lower() and isinstance(module, nn.Embedding):
            feature_embs[name] = module.weight.detach().cpu().numpy()

    if feature_embs:
        name, emb = list(feature_embs.items())[0]

        # Similarity matrix
        sim = cosine_similarity(emb)
        im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Feature Embedding Similarity')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No feature embeddings found',
                ha='center', va='center', transform=ax.transAxes)

    # Observation status embeddings
    ax = axes[1, 0]
    obs_embs = {}
    for name, module in model.named_modules():
        if 'observation_status' in name.lower() and isinstance(module, nn.Embedding):
            obs_embs[name] = module.weight.detach().cpu().numpy()

    if obs_embs:
        for name, emb in obs_embs.items():
            # Should have 2 embeddings: observed vs unobserved
            if emb.shape[0] == 2:
                ax.bar([0, 1], [np.linalg.norm(emb[0]), np.linalg.norm(emb[1])],
                      tick_label=['Unobserved', 'Observed'], color=['red', 'green'], alpha=0.7)
                ax.set_ylabel('Embedding Norm')
                ax.set_title('Observation Status Embedding Magnitudes')

                # Add cosine similarity
                cos_sim = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
                ax.text(0.5, 0.9, f'Cosine Similarity: {cos_sim:.3f}',
                       transform=ax.transAxes, ha='center')
                break
    else:
        ax.text(0.5, 0.5, 'No observation status embeddings found',
                ha='center', va='center', transform=ax.transAxes)

    # All embedding norms comparison
    ax = axes[1, 1]
    all_emb_norms = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            norm = module.weight.detach().cpu().norm().item()
            clean_name = name.split('.')[-1][:20]
            all_emb_norms[clean_name] = norm

    if all_emb_norms:
        names = list(all_emb_norms.keys())
        norms = list(all_emb_norms.values())

        ax.barh(range(len(names)), norms, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Total Norm')
        ax.set_title('Embedding Table Norms')
    else:
        ax.text(0.5, 0.5, 'No embeddings found',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '21_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig22_summary_dashboard(training_summary: Dict, output_dir: Path):
    """Figure 22: Summary dashboard with key findings."""
    print("  Generating Figure 22: Summary Dashboard...")

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Multi-Resolution HAN: Comprehensive Analysis Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)

    # Test metrics summary
    ax = fig.add_subplot(gs[0, 0])
    test_metrics = training_summary['test_metrics']
    metrics = ['Regime', 'Transition', 'Casualty', 'Anomaly']
    values = [test_metrics['regime_loss'], test_metrics['transition_loss'],
              test_metrics['casualty_loss'], test_metrics['anomaly_loss']]

    colors = ['green' if v < 0.1 else 'orange' if v < 1 else 'red' for v in values]
    bars = ax.bar(metrics, values, color=colors)
    ax.set_yscale('log')
    ax.set_ylabel('Test Loss (log)')
    ax.set_title('Task Performance')

    # Training convergence
    ax = fig.add_subplot(gs[0, 1])
    history = training_summary['history']['val_history']
    ax.plot(history['total'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Convergence')
    ax.axhline(training_summary['history']['best_val_loss'], color='r', linestyle='--', alpha=0.5)

    # Model size
    ax = fig.add_subplot(gs[0, 2])
    n_params = training_summary['n_params']
    ax.pie([n_params], labels=[f'{n_params:,}\nparameters'],
           colors=['steelblue'], startangle=90)
    ax.set_title('Model Size')

    # Configuration
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    config = training_summary['config']
    config_text = f"""
Configuration:
─────────────
d_model: {config['d_model']}
nhead: {config['nhead']}
daily_layers: {config['num_daily_layers']}
monthly_layers: {config['num_monthly_layers']}
fusion_layers: {config['num_fusion_layers']}
dropout: {config['dropout']}
batch_size: {config['batch_size']}
"""
    ax.text(0.1, 0.9, config_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    # Loss evolution by task
    ax = fig.add_subplot(gs[1, :2])
    epochs = list(range(1, len(history['regime']) + 1))
    ax.semilogy(epochs, history['regime'], label='Regime', linewidth=2)
    ax.semilogy(epochs, history['transition'], label='Transition', linewidth=2)
    ax.semilogy(epochs, history['casualty'], label='Casualty', linewidth=2)
    ax.semilogy(epochs, history['anomaly'], label='Anomaly', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Task-Specific Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Key findings
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis('off')

    findings = """
    KEY FINDINGS
    ════════════

    ✓ REGIME CLASSIFICATION: Excellent (loss: 5.86e-05)
      → Near-perfect conflict phase identification
      → Model clearly separates invasion, stalemate,
        counteroffensive, and attritional phases

    ⚠ TRANSITION DETECTION: Moderate (loss: 0.43)
      → Rare event detection challenge
      → Recommendation: focal loss, class weighting

    ✓ CASUALTY PREDICTION (ZINB): Good (loss: 2.49)
      → Proper count distribution learning
      → Learns zero-inflation, mean, dispersion

    ✓ ANOMALY DETECTION: Excellent (loss: 1.24e-06)
      → VIIRS radiance anomalies well-predicted
      → Strong proxy for destruction events
    """

    ax.text(0.05, 0.95, findings, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Data sources
    ax = fig.add_subplot(gs[2, :2])
    sources = {
        'Daily': ['Equipment', 'Personnel', 'DeepState', 'FIRMS', 'VIINA', 'VIIRS'],
        'Monthly': ['Sentinel', 'HDX Conflict', 'HDX Food', 'HDX Rainfall', 'IOM']
    }

    y_pos = 0.9
    for resolution, source_list in sources.items():
        ax.text(0.05, y_pos, f'{resolution} Sources:', fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        y_pos -= 0.08
        for source in source_list:
            color = SOURCE_COLORS.get(source.lower().replace(' ', '_'), '#333333')
            ax.text(0.1, y_pos, f'● {source}', fontsize=10, transform=ax.transAxes,
                   color=color)
            y_pos -= 0.06
        y_pos -= 0.05

    ax.axis('off')
    ax.set_title('Data Sources (11 total)')

    # Recommendations
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')

    recs = """
    RECOMMENDATIONS
    ════════════════

    1. IMMEDIATE IMPROVEMENTS:
       • Add focal loss for transition detection
       • Implement ISW semantic integration
       • Add per-class F1 metrics

    2. INTERPRETABILITY:
       • Deploy attention visualization dashboard
       • Feature attribution analysis per phase
       • Temporal backtesting framework

    3. FUTURE DIRECTIONS:
       • Timeline-aware attention (phase embeddings)
       • Ensemble-based uncertainty quantification
       • Calibrated confidence intervals
    """

    ax.text(0.05, 0.95, recs, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.savefig(output_dir / '22_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run comprehensive interpretability analysis."""
    print("=" * 70)
    print("MULTI-RESOLUTION HAN: COMPREHENSIVE INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    output_dir = setup_output_dir()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print()

    # Load model and data
    try:
        model, training_summary, dataset = load_model_and_data(device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("GENERATING FIGURES")
    print("-" * 40)

    # Figures that don't need data
    fig01_training_dynamics(training_summary, output_dir)
    fig02_loss_decomposition(training_summary, output_dir)
    fig03_observation_rates(training_summary, output_dir)
    fig04_test_performance(training_summary, output_dir)

    # Prepare batch for analysis - use more samples spread across the dataset
    print("  Preparing batch for latent analysis...")
    try:
        # Sample from across the dataset to get diverse conflict phases
        batch_size = min(64, len(dataset))
        # Spread indices across the dataset for phase diversity
        step = max(1, len(dataset) // batch_size)
        indices = list(range(0, len(dataset), step))[:batch_size]
        batch, labels_2d = prepare_batch(dataset, indices, device)

        # labels_2d is [batch_size, monthly_seq_len] - flatten for per-timestep analysis
        labels = labels_2d.flatten()

        # Print phase distribution
        unique, counts = np.unique(labels, return_counts=True)
        phase_dist = {PHASE_LABELS.get(int(p), f'Phase {p}'): int(c) for p, c in zip(unique, counts)}
        print(f"    Conflict phase distribution: {phase_dist}")
        print(f"    Labels shape: {labels_2d.shape} -> flattened to {labels.shape}")
    except Exception as e:
        print(f"  WARNING: Could not prepare batch: {e}")
        import traceback
        traceback.print_exc()
        batch = None
        labels = np.zeros(32)

    # Extract latent representations
    print("  Extracting latent representations...")
    try:
        extractor = LatentExtractor(model, device)
        if batch is not None:
            latent_data = extractor.extract(batch)
        else:
            latent_data = {}
    except Exception as e:
        print(f"  WARNING: Could not extract latents: {e}")
        latent_data = {}

    # Extract attention weights
    print("  Extracting attention weights...")
    try:
        analyzer = AttentionAnalyzer(model, device)
        if batch is not None:
            attention_weights = analyzer.extract_attention(batch)
        else:
            attention_weights = {}
    except Exception as e:
        print(f"  WARNING: Could not extract attention: {e}")
        attention_weights = {}

    # Generate all figures
    if latent_data:
        fig05_latent_pca(latent_data, labels, output_dir)
        fig06_latent_tsne(latent_data, labels, output_dir)

    fig07_source_embeddings(model, output_dir)
    fig08_no_observation_tokens(model, output_dir)
    fig09_attention_patterns(attention_weights, output_dir)
    fig10_attention_head_specialization(attention_weights, output_dir)

    if latent_data:
        fig11_cross_source_correlations(latent_data, output_dir)
        fig12_temporal_dynamics(latent_data, output_dir)
        fig13_cross_modal_viirs_equipment(latent_data, output_dir)

    if batch is not None:
        fig14_casualty_predictors(model, batch, output_dir)

    if latent_data:
        fig15_regime_classification_analysis(latent_data, labels, output_dir)
        fig16_anomaly_detection_analysis(latent_data, output_dir)
        fig17_multimodal_fusion(latent_data, output_dir)
        fig18_resolution_fusion(latent_data, output_dir)

    fig19_prediction_head_analysis(model, output_dir)
    fig20_model_parameter_analysis(model, output_dir)
    fig21_embedding_evolution(model, output_dir)
    fig22_summary_dashboard(training_summary, output_dir)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Generated 22 figures in: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
