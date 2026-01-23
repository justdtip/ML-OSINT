#!/usr/bin/env python3
"""
Comprehensive Network Analysis for ML_OSINT Pipeline

Produces detailed visualizations for:
1. INTRA-NETWORK ANALYSIS: Within each JIM model
   - Feature embedding structure
   - Cross-feature attention patterns
   - Temporal encoding learned patterns
   - Uncertainty calibration

2. INTER-NETWORK ANALYSIS: Relationships between JIM models
   - Weight correlation across models
   - Feature embedding alignment
   - Source-specific learned representations
   - Model similarity clustering

3. UNIFIED MODEL ANALYSIS: Cross-source fusion
   - Cross-source attention heatmaps
   - Source embedding space visualization
   - Reconstruction accuracy by source
   - Learned source relationships

Output: All figures saved to outputs/analysis/network/
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Setup paths
ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR,
    NETWORK_OUTPUT_DIR,
)

# Create output directory for network figures
NETWORK_FIGURES_DIR = NETWORK_OUTPUT_DIR
NETWORK_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not available - some analyses will be skipped")

# Local imports
from unified_interpolation import (
    UnifiedInterpolationModel, SOURCE_CONFIGS, CrossSourceDataset
)
from joint_interpolation_models import (
    JointInterpolationModel, InterpolationConfig, INTERPOLATION_CONFIGS
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
if HAS_SEABORN:
    sns.set_palette("husl")

# Color schemes
SOURCE_COLORS = {
    'sentinel': '#3498db',   # Blue
    'deepstate': '#2ecc71',  # Green
    'equipment': '#e74c3c',  # Red
    'firms': '#f39c12',      # Orange
    'ucdp': '#9b59b6',       # Purple
}

PHASE_COLORS = {
    'phase1': '#1abc9c',
    'phase2': '#3498db',
    'phase3': '#e74c3c',
}


class ComprehensiveNetworkAnalyzer:
    """Full pipeline analysis with extensive visualizations."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.jim_models = {}
        self.jim_states = {}
        self.unified_model = None
        self.unified_state = None

    def load_all_models(self):
        """Load all trained models."""
        print("=" * 70)
        print("LOADING TRAINED MODELS")
        print("=" * 70)

        # Load JIM models (Phase 1 interpolation models)
        jim_paths = list(INTERP_MODEL_DIR.glob("interp_*_best.pt"))
        print(f"\nFound {len(jim_paths)} JIM models")

        for path in jim_paths:
            name = path.stem.replace('interp_', '').replace('_best', '')
            try:
                state = torch.load(path, map_location=self.device)
                self.jim_states[name] = state
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Error loading {name}: {e}")

        # Load unified model
        unified_path = MODEL_DIR / "unified_interpolation_best.pt"
        if unified_path.exists():
            try:
                self.unified_state = torch.load(unified_path, map_location=self.device)
                print(f"\nLoaded unified model: {unified_path.name}")
            except Exception as e:
                print(f"Error loading unified model: {e}")

        print(f"\nTotal models loaded: {len(self.jim_states)} JIM + {'1' if self.unified_state else '0'} Unified")

    # =========================================================================
    # INTRA-NETWORK ANALYSIS
    # =========================================================================

    def analyze_intra_network(self):
        """Analyze patterns within individual JIM models."""
        print("\n" + "=" * 70)
        print("INTRA-NETWORK ANALYSIS")
        print("=" * 70)

        self._plot_feature_embeddings_by_source()
        self._plot_attention_patterns_by_model()
        self._plot_temporal_encoding_patterns()
        self._plot_uncertainty_head_analysis()
        self._plot_decoder_weight_structure()

    def _plot_feature_embeddings_by_source(self):
        """Plot feature embedding structure for each source."""
        print("\n  Plotting feature embeddings by source...")

        # Group models by source
        source_embeddings = defaultdict(list)

        for name, state in self.jim_states.items():
            # Determine source
            source = None
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break
            if source is None:
                source = 'other'

            # Get feature embeddings
            for key in state:
                if 'feature_embeddings.weight' in key or 'feature_embedding.weight' in key:
                    emb = state[key].cpu().numpy()
                    source_embeddings[source].append({
                        'name': name,
                        'embeddings': emb,
                        'n_features': emb.shape[0],
                        'd_model': emb.shape[1]
                    })
                    break

        # Create figure - one subplot per source
        sources_with_data = [s for s in SOURCE_COLORS.keys() if source_embeddings[s]]
        n_sources = len(sources_with_data)

        if n_sources == 0:
            print("    No feature embeddings found")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, source in enumerate(sources_with_data):
            if idx >= 6:
                break
            ax = axes[idx]

            # Combine all embeddings for this source
            all_emb = []
            all_labels = []
            model_ids = []

            for i, data in enumerate(source_embeddings[source]):
                emb = data['embeddings']
                all_emb.append(emb)
                all_labels.extend([f"M{i}_F{j}" for j in range(emb.shape[0])])
                model_ids.extend([i] * emb.shape[0])

            if not all_emb:
                continue

            combined = np.vstack(all_emb)

            if HAS_SKLEARN and combined.shape[0] >= 3:
                # PCA projection
                pca = PCA(n_components=min(2, combined.shape[1]))
                projected = pca.fit_transform(combined)

                # Color by model
                scatter = ax.scatter(
                    projected[:, 0],
                    projected[:, 1] if projected.shape[1] > 1 else np.zeros(len(projected)),
                    c=model_ids,
                    cmap='tab10',
                    alpha=0.7,
                    s=30
                )

                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                if projected.shape[1] > 1:
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.set_title(f'{source.upper()}\n{len(source_embeddings[source])} models, {combined.shape[0]} features')

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Model ID')
            else:
                # Fallback: show embedding norms
                norms = np.linalg.norm(combined, axis=1)
                ax.hist(norms, bins=30, alpha=0.7, color=SOURCE_COLORS.get(source, 'gray'))
                ax.set_xlabel('Embedding L2 Norm')
                ax.set_ylabel('Count')
                ax.set_title(f'{source.upper()} Embedding Norms')

        # Hide unused axes
        for idx in range(len(sources_with_data), 6):
            axes[idx].set_visible(False)

        plt.suptitle('Feature Embeddings by Source (PCA Projection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '01_intra_feature_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 01_intra_feature_embeddings.png")

    def _plot_attention_patterns_by_model(self):
        """Plot self-attention weight patterns for each model."""
        print("\n  Plotting attention patterns...")

        # Collect attention weights
        attention_data = []

        for name, state in self.jim_states.items():
            for key in state:
                if 'self_attn.in_proj_weight' in key:
                    w = state[key].cpu().numpy()

                    # Determine source
                    source = 'other'
                    for src in SOURCE_COLORS.keys():
                        if src in name.lower():
                            source = src
                            break

                    attention_data.append({
                        'name': name[:25],
                        'source': source,
                        'weights': w,
                        'mean_magnitude': np.abs(w).mean(),
                        'std': w.std()
                    })
                    break

        if not attention_data:
            print("    No attention weights found")
            return

        # Sort by source then by mean magnitude
        attention_data.sort(key=lambda x: (x['source'], -x['mean_magnitude']))

        # Create figure with multiple views
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Attention magnitude by source (box plot)
        ax1 = fig.add_subplot(gs[0, 0])
        source_magnitudes = defaultdict(list)
        for data in attention_data:
            source_magnitudes[data['source']].append(data['mean_magnitude'])

        sources = list(source_magnitudes.keys())
        magnitudes = [source_magnitudes[s] for s in sources]
        colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

        bp = ax1.boxplot(magnitudes, labels=[s.upper() for s in sources], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Mean |Attention Weight|')
        ax1.set_title('Attention Magnitude by Source')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Attention weight distribution (violin plot or histogram)
        ax2 = fig.add_subplot(gs[0, 1:])
        all_weights = []
        all_sources = []
        for data in attention_data[:10]:  # First 10 models
            weights_flat = data['weights'].flatten()
            all_weights.extend(weights_flat[:1000])  # Sample
            all_sources.extend([data['source']] * min(1000, len(weights_flat)))

        if HAS_SEABORN and all_weights:
            import pandas as pd
            df = pd.DataFrame({'weight': all_weights, 'source': all_sources})
            sns.violinplot(data=df, x='source', y='weight', ax=ax2,
                          palette=SOURCE_COLORS, alpha=0.7)
            ax2.set_title('Attention Weight Distribution by Source')
        else:
            ax2.hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
            ax2.set_title('Attention Weight Distribution (All Models)')
        ax2.set_xlabel('Source')
        ax2.set_ylabel('Weight Value')

        # 3-8. Individual attention heatmaps for representative models
        for idx, data in enumerate(attention_data[:6]):
            row = 1 + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            w = data['weights']
            # Show first 32x32 if larger
            w_show = w[:min(32, w.shape[0]), :min(32, w.shape[1])]

            im = ax.imshow(np.abs(w_show), cmap='YlOrRd', aspect='auto')
            ax.set_title(f"{data['name']}\n({data['source'].upper()})", fontsize=9)
            ax.set_xlabel('Dim')
            ax.set_ylabel('Dim')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.suptitle('Self-Attention Weight Analysis', fontsize=14, fontweight='bold')
        plt.savefig(NETWORK_FIGURES_DIR / '02_intra_attention_patterns.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 02_intra_attention_patterns.png")

    def _plot_temporal_encoding_patterns(self):
        """Analyze learned temporal encodings."""
        print("\n  Plotting temporal encoding patterns...")

        # Collect temporal encodings
        day_embeddings = []
        sinusoidal_pe = []
        combine_weights = []

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            for key in state:
                if 'day_embedding.weight' in key:
                    emb = state[key].cpu().numpy()
                    day_embeddings.append({
                        'name': name, 'source': source,
                        'embeddings': emb[:30]  # First 30 days
                    })
                elif 'pe_sinusoidal' in key:
                    pe = state[key].cpu().numpy()
                    sinusoidal_pe.append({
                        'name': name, 'source': source,
                        'pe': pe[:100]  # First 100 positions
                    })
                elif 'temporal_encoding.combine.weight' in key:
                    w = state[key].cpu().numpy()
                    combine_weights.append({
                        'name': name, 'source': source,
                        'weights': w
                    })

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Day embedding evolution (mean across models)
        ax1 = axes[0, 0]
        if day_embeddings:
            source_day_patterns = defaultdict(list)
            for data in day_embeddings:
                daily_avg = data['embeddings'].mean(axis=1)
                source_day_patterns[data['source']].append(daily_avg)

            for source, patterns in source_day_patterns.items():
                mean_pattern = np.mean(patterns, axis=0)
                std_pattern = np.std(patterns, axis=0)
                days = np.arange(len(mean_pattern))

                ax1.plot(days, mean_pattern, color=SOURCE_COLORS.get(source, 'gray'),
                        label=source.upper(), linewidth=2)
                ax1.fill_between(days, mean_pattern - std_pattern, mean_pattern + std_pattern,
                                color=SOURCE_COLORS.get(source, 'gray'), alpha=0.2)

            ax1.set_xlabel('Day Index')
            ax1.set_ylabel('Mean Embedding Value')
            ax1.set_title('Learned Day Embeddings by Source')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No day embeddings found', ha='center', va='center')

        # 2. Sinusoidal PE structure
        ax2 = axes[0, 1]
        if sinusoidal_pe:
            # Show first model's PE
            pe = sinusoidal_pe[0]['pe']
            im = ax2.imshow(pe[:50, :16].T, aspect='auto', cmap='RdBu_r')
            ax2.set_xlabel('Position (Day)')
            ax2.set_ylabel('Dimension')
            ax2.set_title('Sinusoidal Positional Encoding')
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 'No sinusoidal PE found', ha='center', va='center')

        # 3. Temporal combine layer weight distribution
        ax3 = axes[1, 0]
        if combine_weights:
            source_weights = defaultdict(list)
            for data in combine_weights:
                source_weights[data['source']].append(np.abs(data['weights']).mean())

            sources = list(source_weights.keys())
            means = [np.mean(source_weights[s]) for s in sources]
            stds = [np.std(source_weights[s]) for s in sources]
            colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

            bars = ax3.bar(sources, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
            ax3.set_ylabel('Mean |Combine Weight|')
            ax3.set_title('Temporal Combine Layer Weights')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No combine weights found', ha='center', va='center')

        # 4. Day embedding clustering
        ax4 = axes[1, 1]
        if day_embeddings and HAS_SKLEARN:
            # Combine all day embeddings
            all_day_emb = []
            all_sources = []
            for data in day_embeddings:
                emb = data['embeddings']
                all_day_emb.append(emb.mean(axis=0))  # Mean per model
                all_sources.append(data['source'])

            if len(all_day_emb) >= 3:
                combined = np.vstack(all_day_emb)
                pca = PCA(n_components=2)
                projected = pca.fit_transform(combined)

                for i, (x, y) in enumerate(projected):
                    source = all_sources[i]
                    ax4.scatter(x, y, c=SOURCE_COLORS.get(source, 'gray'),
                               s=100, alpha=0.7, edgecolors='black')

                # Legend
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=c, markersize=10, label=s.upper())
                          for s, c in SOURCE_COLORS.items() if s in all_sources]
                ax4.legend(handles=handles)
                ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax4.set_title('Day Embedding Clustering (PCA)')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center')

        plt.suptitle('Temporal Encoding Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '03_intra_temporal_encoding.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 03_intra_temporal_encoding.png")

    def _plot_uncertainty_head_analysis(self):
        """Analyze uncertainty estimation heads."""
        print("\n  Plotting uncertainty head analysis...")

        uncertainty_data = []

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            unc_weights = []
            unc_bias = None

            for key in state:
                if 'uncertainty_head' in key:
                    if 'weight' in key:
                        unc_weights.append(state[key].cpu().numpy())
                    elif 'bias' in key and len(state[key].shape) == 1 and state[key].shape[0] == 1:
                        unc_bias = state[key].cpu().numpy().item()

            if unc_weights:
                uncertainty_data.append({
                    'name': name,
                    'source': source,
                    'weights': unc_weights,
                    'bias': unc_bias,
                    'total_params': sum(w.size for w in unc_weights)
                })

        if not uncertainty_data:
            print("    No uncertainty heads found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Output bias distribution by source
        ax1 = axes[0, 0]
        source_biases = defaultdict(list)
        for data in uncertainty_data:
            if data['bias'] is not None:
                source_biases[data['source']].append(data['bias'])

        if source_biases:
            sources = list(source_biases.keys())
            biases = [source_biases[s] for s in sources]
            colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

            bp = ax1.boxplot(biases, labels=[s.upper() for s in sources], patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax1.set_ylabel('Output Bias')
            ax1.set_title('Uncertainty Head Output Bias by Source')
            ax1.tick_params(axis='x', rotation=45)

        # 2. Weight magnitude comparison
        ax2 = axes[0, 1]
        source_magnitudes = defaultdict(list)
        for data in uncertainty_data:
            mag = np.mean([np.abs(w).mean() for w in data['weights']])
            source_magnitudes[data['source']].append(mag)

        sources = list(source_magnitudes.keys())
        mags = [np.mean(source_magnitudes[s]) for s in sources]
        stds = [np.std(source_magnitudes[s]) for s in sources]
        colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

        bars = ax2.bar(sources, mags, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax2.set_ylabel('Mean |Weight|')
        ax2.set_title('Uncertainty Head Weight Magnitudes')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Weight distribution (histogram)
        ax3 = axes[1, 0]
        all_weights = []
        for data in uncertainty_data[:10]:
            for w in data['weights']:
                all_weights.extend(w.flatten()[:500])

        ax3.hist(all_weights, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax3.axvline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Uncertainty Head Weight Distribution')

        # 4. Bias vs Weight Magnitude scatter
        ax4 = axes[1, 1]
        for data in uncertainty_data:
            if data['bias'] is not None:
                mag = np.mean([np.abs(w).mean() for w in data['weights']])
                ax4.scatter(mag, data['bias'],
                           c=SOURCE_COLORS.get(data['source'], 'gray'),
                           s=80, alpha=0.7, edgecolors='black')

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=c, markersize=10, label=s.upper())
                  for s, c in SOURCE_COLORS.items()]
        ax4.legend(handles=handles)
        ax4.set_xlabel('Mean |Weight|')
        ax4.set_ylabel('Output Bias')
        ax4.set_title('Uncertainty: Weight Magnitude vs Bias')
        ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)

        plt.suptitle('Uncertainty Estimation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '04_intra_uncertainty_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 04_intra_uncertainty_analysis.png")

    def _plot_decoder_weight_structure(self):
        """Analyze decoder network structure."""
        print("\n  Plotting decoder weight structure...")

        decoder_data = []

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            decoder_weights = []
            for key in sorted(state.keys()):
                if 'decoder' in key and 'weight' in key:
                    w = state[key].cpu().numpy()
                    decoder_weights.append({
                        'layer': key,
                        'shape': w.shape,
                        'mean': np.abs(w).mean(),
                        'std': w.std(),
                        'sparsity': (np.abs(w) < 0.01).mean()
                    })

            if decoder_weights:
                decoder_data.append({
                    'name': name,
                    'source': source,
                    'layers': decoder_weights
                })

        if not decoder_data:
            print("    No decoder weights found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Layer-wise weight magnitude progression
        ax1 = axes[0, 0]
        source_progressions = defaultdict(list)
        for data in decoder_data:
            mags = [l['mean'] for l in data['layers']]
            source_progressions[data['source']].append(mags)

        for source, progressions in source_progressions.items():
            if progressions:
                # Pad to same length
                max_len = max(len(p) for p in progressions)
                padded = [p + [np.nan] * (max_len - len(p)) for p in progressions]
                mean_prog = np.nanmean(padded, axis=0)

                ax1.plot(range(len(mean_prog)), mean_prog,
                        color=SOURCE_COLORS.get(source, 'gray'),
                        label=source.upper(), linewidth=2, marker='o')

        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Mean |Weight|')
        ax1.set_title('Decoder Weight Magnitude by Layer')
        ax1.legend()

        # 2. Sparsity by layer
        ax2 = axes[0, 1]
        source_sparsity = defaultdict(list)
        for data in decoder_data:
            sparsities = [l['sparsity'] for l in data['layers']]
            source_sparsity[data['source']].append(sparsities)

        for source, sparsities in source_sparsity.items():
            if sparsities:
                max_len = max(len(s) for s in sparsities)
                padded = [s + [np.nan] * (max_len - len(s)) for s in sparsities]
                mean_sparsity = np.nanmean(padded, axis=0)

                ax2.plot(range(len(mean_sparsity)), mean_sparsity * 100,
                        color=SOURCE_COLORS.get(source, 'gray'),
                        label=source.upper(), linewidth=2, marker='s')

        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Sparsity (%)')
        ax2.set_title('Decoder Weight Sparsity by Layer')
        ax2.legend()

        # 3. Total decoder parameters by source
        ax3 = axes[1, 0]
        source_params = defaultdict(list)
        for data in decoder_data:
            total = sum(np.prod(l['shape']) for l in data['layers'])
            source_params[data['source']].append(total)

        sources = list(source_params.keys())
        means = [np.mean(source_params[s]) for s in sources]
        colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

        ax3.bar(sources, [m/1000 for m in means], color=colors, alpha=0.7)
        ax3.set_ylabel('Parameters (thousands)')
        ax3.set_title('Decoder Parameters by Source')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Weight std distribution
        ax4 = axes[1, 1]
        for source in SOURCE_COLORS.keys():
            stds = []
            for data in decoder_data:
                if data['source'] == source:
                    stds.extend([l['std'] for l in data['layers']])
            if stds:
                ax4.hist(stds, bins=20, alpha=0.5, label=source.upper(),
                        color=SOURCE_COLORS[source])

        ax4.set_xlabel('Weight Std Dev')
        ax4.set_ylabel('Count')
        ax4.set_title('Decoder Weight Standard Deviation')
        ax4.legend()

        plt.suptitle('Decoder Network Structure Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '05_intra_decoder_structure.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 05_intra_decoder_structure.png")

    # =========================================================================
    # INTER-NETWORK ANALYSIS
    # =========================================================================

    def analyze_inter_network(self):
        """Analyze relationships between different JIM models."""
        print("\n" + "=" * 70)
        print("INTER-NETWORK ANALYSIS")
        print("=" * 70)

        self._plot_model_similarity_matrix()
        self._plot_weight_correlation_heatmap()
        self._plot_embedding_alignment()
        self._plot_model_clustering()

    def _plot_model_similarity_matrix(self):
        """Plot similarity between models based on weights."""
        print("\n  Plotting model similarity matrix...")

        # Extract representative weights from each model
        model_vectors = {}

        for name, state in self.jim_states.items():
            # Flatten key weights into a vector
            weights = []
            for key in sorted(state.keys()):
                if 'weight' in key:
                    w = state[key].cpu().numpy().flatten()
                    weights.extend(w[:100])  # First 100 elements

            if weights:
                model_vectors[name] = np.array(weights[:500])  # Cap at 500

        if len(model_vectors) < 2:
            print("    Insufficient models for similarity analysis")
            return

        # Compute similarity matrix
        names = list(model_vectors.keys())
        vectors = np.vstack([model_vectors[n] for n in names])

        # Pad vectors to same length
        max_len = max(len(model_vectors[n]) for n in names)
        vectors_padded = np.zeros((len(names), max_len))
        for i, n in enumerate(names):
            v = model_vectors[n]
            vectors_padded[i, :len(v)] = v

        if HAS_SKLEARN:
            similarity = cosine_similarity(vectors_padded)
        else:
            # Manual cosine similarity
            norms = np.linalg.norm(vectors_padded, axis=1, keepdims=True)
            normalized = vectors_padded / (norms + 1e-8)
            similarity = normalized @ normalized.T

        fig, ax = plt.subplots(figsize=(16, 14))

        # Create heatmap
        im = ax.imshow(similarity, cmap='RdYlGn', vmin=-1, vmax=1)

        # Add source-based color bars
        source_colors_list = []
        for name in names:
            for src, color in SOURCE_COLORS.items():
                if src in name.lower():
                    source_colors_list.append(color)
                    break
            else:
                source_colors_list.append('gray')

        # Add colored rectangles on the side
        for i, color in enumerate(source_colors_list):
            rect = Rectangle((-1.5, i - 0.5), 1, 1, facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            rect2 = Rectangle((i - 0.5, len(names) + 0.5), 1, 1, facecolor=color, edgecolor='none')
            ax.add_patch(rect2)

        # Labels
        short_names = [n[:20] for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short_names, rotation=90, fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)

        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title('Model Weight Similarity Matrix', fontsize=14, fontweight='bold')

        # Legend for source colors
        legend_handles = [mpatches.Patch(color=c, label=s.upper()) for s, c in SOURCE_COLORS.items()]
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '06_inter_model_similarity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 06_inter_model_similarity.png")

    def _plot_weight_correlation_heatmap(self):
        """Plot correlation between specific weight types across models."""
        print("\n  Plotting weight correlation heatmap...")

        # Group by source and extract attention weights
        source_attention = defaultdict(list)

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            for key in state:
                if 'self_attn.in_proj_weight' in key:
                    w = state[key].cpu().numpy().flatten()[:200]
                    source_attention[source].append({
                        'name': name,
                        'weights': w
                    })
                    break

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (source, data) in enumerate(source_attention.items()):
            if idx >= 6:
                break
            ax = axes[idx]

            if len(data) < 2:
                ax.text(0.5, 0.5, f'Only {len(data)} model(s)', ha='center', va='center')
                ax.set_title(f'{source.upper()}')
                continue

            # Compute correlation between models within this source
            names = [d['name'][:15] for d in data]
            weights = np.vstack([d['weights'] for d in data])

            corr = np.corrcoef(weights)

            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, fontsize=7)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_title(f'{source.upper()} Attention Correlation')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused
        for idx in range(len(source_attention), 6):
            axes[idx].set_visible(False)

        plt.suptitle('Within-Source Model Weight Correlation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '07_inter_weight_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 07_inter_weight_correlation.png")

    def _plot_embedding_alignment(self):
        """Plot how embeddings align across models."""
        print("\n  Plotting embedding alignment...")

        if not HAS_SKLEARN:
            print("    sklearn required for embedding alignment")
            return

        # Collect all embeddings
        all_embeddings = []
        all_sources = []
        all_names = []

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            for key in state:
                if 'feature_embeddings.weight' in key:
                    emb = state[key].cpu().numpy()
                    # Mean embedding per model
                    mean_emb = emb.mean(axis=0)
                    all_embeddings.append(mean_emb)
                    all_sources.append(source)
                    all_names.append(name)
                    break

        if len(all_embeddings) < 3:
            print("    Insufficient embeddings for alignment analysis")
            return

        # Stack and project
        embeddings = np.vstack(all_embeddings)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # PCA projection
        ax1 = axes[0]
        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(embeddings)

        for i, (x, y) in enumerate(pca_proj):
            source = all_sources[i]
            ax1.scatter(x, y, c=SOURCE_COLORS.get(source, 'gray'),
                       s=100, alpha=0.7, edgecolors='black')

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('Model Embedding Alignment (PCA)')

        # t-SNE projection
        ax2 = axes[1]
        if len(embeddings) >= 5:
            try:
                tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1),
                           random_state=42, n_iter=1000)
                tsne_proj = tsne.fit_transform(embeddings)

                for i, (x, y) in enumerate(tsne_proj):
                    source = all_sources[i]
                    ax2.scatter(x, y, c=SOURCE_COLORS.get(source, 'gray'),
                               s=100, alpha=0.7, edgecolors='black')

                ax2.set_xlabel('t-SNE 1')
                ax2.set_ylabel('t-SNE 2')
                ax2.set_title('Model Embedding Alignment (t-SNE)')
            except Exception as e:
                ax2.text(0.5, 0.5, f't-SNE failed: {e}', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for t-SNE', ha='center', va='center')

        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=c, markersize=10, label=s.upper())
                  for s, c in SOURCE_COLORS.items()]
        axes[0].legend(handles=handles, loc='best')
        axes[1].legend(handles=handles, loc='best')

        plt.suptitle('Cross-Model Embedding Alignment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '08_inter_embedding_alignment.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 08_inter_embedding_alignment.png")

    def _plot_model_clustering(self):
        """Cluster models by their learned representations."""
        print("\n  Plotting model clustering...")

        if not HAS_SKLEARN:
            print("    sklearn required for clustering")
            return

        # Build feature vectors for each model
        model_features = {}

        for name, state in self.jim_states.items():
            features = []

            # Extract various statistics from weights
            for key in state:
                if 'weight' in key:
                    w = state[key].cpu().numpy()
                    features.extend([
                        np.abs(w).mean(),
                        w.std(),
                        (np.abs(w) < 0.01).mean(),  # Sparsity
                    ])
                    if len(features) >= 30:
                        break

            if features:
                model_features[name] = np.array(features[:30])

        if len(model_features) < 4:
            print("    Insufficient models for clustering")
            return

        names = list(model_features.keys())
        X = np.vstack([model_features[n] for n in names])

        # Determine sources
        sources = []
        for name in names:
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break
            sources.append(source)

        # Cluster
        n_clusters = min(5, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Project with PCA
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Color by source
        ax1 = axes[0]
        for i, (x, y) in enumerate(projected):
            ax1.scatter(x, y, c=SOURCE_COLORS.get(sources[i], 'gray'),
                       s=100, alpha=0.7, edgecolors='black')

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=c, markersize=10, label=s.upper())
                  for s, c in SOURCE_COLORS.items()]
        ax1.legend(handles=handles)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('Models Colored by Source')

        # Color by cluster
        ax2 = axes[1]
        scatter = ax2.scatter(projected[:, 0], projected[:, 1],
                             c=clusters, cmap='Set1', s=100, alpha=0.7, edgecolors='black')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax2.set_title(f'Models Colored by Cluster (K={n_clusters})')
        plt.colorbar(scatter, ax=ax2, label='Cluster')

        plt.suptitle('Model Clustering Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '09_inter_model_clustering.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 09_inter_model_clustering.png")

    # =========================================================================
    # UNIFIED MODEL ANALYSIS
    # =========================================================================

    def analyze_unified_model(self):
        """Analyze the unified cross-source fusion model."""
        print("\n" + "=" * 70)
        print("UNIFIED MODEL ANALYSIS")
        print("=" * 70)

        if self.unified_state is None:
            print("  Unified model not loaded - skipping")
            return

        self._plot_unified_architecture_diagram()
        self._plot_cross_source_attention()
        self._plot_source_embeddings()
        self._plot_encoder_decoder_analysis()
        self._plot_reconstruction_analysis()

    def _plot_unified_architecture_diagram(self):
        """Create architecture diagram for unified model."""
        print("\n  Plotting unified architecture diagram...")

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        ax.axis('off')

        # Title
        ax.text(50, 57, 'Unified Cross-Source Interpolation Model',
               fontsize=18, fontweight='bold', ha='center')
        ax.text(50, 54, f'Parameters: {sum(v.numel() for v in self.unified_state.values()):,}',
               fontsize=12, ha='center', color='gray')

        # Source inputs
        sources = ['Sentinel', 'DeepState', 'Equipment', 'FIRMS', 'UCDP']
        source_keys = ['sentinel', 'deepstate', 'equipment', 'firms', 'ucdp']

        for i, (source, key) in enumerate(zip(sources, source_keys)):
            x = 10 + i * 18
            y = 45

            # Source box
            rect = FancyBboxPatch((x-6, y-3), 12, 6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=SOURCE_COLORS[key],
                                  edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, source, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

            # Arrow to encoder
            ax.annotate('', xy=(x, 35), xytext=(x, 42),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Encoder layer
        for i, (source, key) in enumerate(zip(sources, source_keys)):
            x = 10 + i * 18
            y = 32

            rect = FancyBboxPatch((x-5, y-2.5), 10, 5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightblue',
                                  edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, 'Encoder', ha='center', va='center', fontsize=9)

        # Arrows to fusion
        for i in range(5):
            x = 10 + i * 18
            ax.annotate('', xy=(50, 22), xytext=(x, 29.5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Cross-source fusion
        rect = FancyBboxPatch((30, 17), 40, 10,
                              boxstyle="round,pad=0.2",
                              facecolor='#9b59b6',
                              edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(50, 22, 'Cross-Source Attention', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white')
        ax.text(50, 19, '(Transformer, 4 heads, 2 layers)', ha='center', va='center',
               fontsize=9, color='white')

        # Arrows to decoders
        for i in range(5):
            x = 10 + i * 18
            ax.annotate('', xy=(x, 10), xytext=(50, 17),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Decoder layer
        for i, (source, key) in enumerate(zip(sources, source_keys)):
            x = 10 + i * 18
            y = 7

            rect = FancyBboxPatch((x-5, y-2.5), 10, 5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgreen',
                                  edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x, y, 'Decoder', ha='center', va='center', fontsize=9)

        # Reconstruction outputs
        for i, (source, key) in enumerate(zip(sources, source_keys)):
            x = 10 + i * 18
            ax.annotate('', xy=(x, 0), xytext=(x, 4.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.text(x, -2, f'Recon\n{source[:4]}', ha='center', va='top', fontsize=8)

        # Legend
        ax.text(95, 50, 'Legend:', fontsize=10, fontweight='bold')
        for i, (src, color) in enumerate(SOURCE_COLORS.items()):
            rect = Rectangle((92, 47-i*3), 2, 2, facecolor=color)
            ax.add_patch(rect)
            ax.text(95, 48-i*3, src.upper(), fontsize=8, va='center')

        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '10_unified_architecture.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 10_unified_architecture.png")

    def _plot_cross_source_attention(self):
        """Analyze cross-source attention patterns."""
        print("\n  Plotting cross-source attention...")

        # Extract fusion layer weights
        fusion_weights = {}
        for key in self.unified_state:
            if 'fusion' in key and 'weight' in key:
                fusion_weights[key] = self.unified_state[key].cpu().numpy()

        if not fusion_weights:
            print("    No fusion weights found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Source embeddings (if present)
        ax1 = axes[0, 0]
        source_emb_key = None
        for key in self.unified_state:
            if 'source_embeddings.weight' in key:
                source_emb_key = key
                break

        if source_emb_key:
            emb = self.unified_state[source_emb_key].cpu().numpy()
            im = ax1.imshow(emb, cmap='viridis', aspect='auto')
            ax1.set_yticks(range(5))
            ax1.set_yticklabels(['Sentinel', 'DeepState', 'Equipment', 'FIRMS', 'UCDP'])
            ax1.set_xlabel('Embedding Dimension')
            ax1.set_title('Learned Source Type Embeddings')
            plt.colorbar(im, ax=ax1)
        else:
            ax1.text(0.5, 0.5, 'No source embeddings found', ha='center', va='center')

        # 2. Attention projection weights (Q, K, V)
        ax2 = axes[0, 1]
        attn_proj_key = None
        for key in fusion_weights:
            if 'in_proj_weight' in key:
                attn_proj_key = key
                break

        if attn_proj_key:
            w = fusion_weights[attn_proj_key]
            # Split into Q, K, V
            d_model = w.shape[0] // 3

            im = ax2.imshow(np.abs(w[:, :32]), aspect='auto', cmap='magma')
            ax2.axhline(d_model - 0.5, color='white', linestyle='--', linewidth=2)
            ax2.axhline(2*d_model - 0.5, color='white', linestyle='--', linewidth=2)

            ax2.text(-2, d_model//2, 'Q', ha='right', va='center', fontsize=12, fontweight='bold')
            ax2.text(-2, d_model + d_model//2, 'K', ha='right', va='center', fontsize=12, fontweight='bold')
            ax2.text(-2, 2*d_model + d_model//2, 'V', ha='right', va='center', fontsize=12, fontweight='bold')

            ax2.set_xlabel('Input Dimension')
            ax2.set_ylabel('Output Dimension')
            ax2.set_title('Cross-Source Attention Projection (Q/K/V)')
            plt.colorbar(im, ax=ax2)
        else:
            ax2.text(0.5, 0.5, 'No attention projection found', ha='center', va='center')

        # 3. Output projection weights
        ax3 = axes[1, 0]
        out_proj_keys = [k for k in fusion_weights if 'out_proj' in k]
        if out_proj_keys:
            # Show distribution of output projection weights
            all_weights = []
            for key in out_proj_keys:
                all_weights.extend(fusion_weights[key].flatten())

            ax3.hist(all_weights, bins=50, alpha=0.7, edgecolor='black', density=True)
            ax3.axvline(0, color='red', linestyle='--')
            ax3.set_xlabel('Weight Value')
            ax3.set_ylabel('Density')
            ax3.set_title('Output Projection Weight Distribution')
        else:
            ax3.text(0.5, 0.5, 'No output projection found', ha='center', va='center')

        # 4. Feedforward weights
        ax4 = axes[1, 1]
        ff_weights = []
        for key in fusion_weights:
            if 'linear' in key or 'fc' in key:
                ff_weights.append({
                    'name': key.split('.')[-2],
                    'mean': np.abs(fusion_weights[key]).mean(),
                    'std': fusion_weights[key].std()
                })

        if ff_weights:
            names = [w['name'] for w in ff_weights]
            means = [w['mean'] for w in ff_weights]
            stds = [w['std'] for w in ff_weights]

            x = np.arange(len(names))
            ax4.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            ax4.set_xticks(x)
            ax4.set_xticklabels(names, rotation=45, fontsize=8)
            ax4.set_ylabel('Mean |Weight|')
            ax4.set_title('Feedforward Layer Weights')
        else:
            ax4.text(0.5, 0.5, 'No feedforward weights found', ha='center', va='center')

        plt.suptitle('Cross-Source Attention Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '11_unified_cross_attention.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 11_unified_cross_attention.png")

    def _plot_source_embeddings(self):
        """Detailed analysis of source embeddings."""
        print("\n  Plotting source embedding analysis...")

        # Find source-related embeddings
        source_data = {}
        for key in self.unified_state:
            if 'source_embeddings' in key:
                source_data['type_embeddings'] = self.unified_state[key].cpu().numpy()
            elif 'encoder' in key and 'weight' in key:
                parts = key.split('.')
                if 'encoders' in parts:
                    idx = parts.index('encoders')
                    if idx + 1 < len(parts):
                        source_name = parts[idx + 1]
                        if source_name not in source_data:
                            source_data[source_name] = {}
                        source_data[source_name][key] = self.unified_state[key].cpu().numpy()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Source type embedding similarity
        ax = axes[0]
        if 'type_embeddings' in source_data:
            emb = source_data['type_embeddings']
            if HAS_SKLEARN:
                sim = cosine_similarity(emb)
            else:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                sim = (emb / (norms + 1e-8)) @ (emb / (norms + 1e-8)).T

            sources = ['Sentinel', 'DeepState', 'Equipment', 'FIRMS', 'UCDP']
            im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(sources, rotation=45)
            ax.set_yticklabels(sources)
            ax.set_title('Source Embedding Similarity')
            plt.colorbar(im, ax=ax, label='Cosine Similarity')

            # Annotate values
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No type embeddings', ha='center', va='center')

        # 2-6. Per-source encoder weight analysis
        source_keys = ['sentinel', 'deepstate', 'equipment', 'firms', 'ucdp']
        for idx, source in enumerate(source_keys):
            ax = axes[idx + 1]

            if source in source_data and source_data[source]:
                # Aggregate encoder weights
                weights = []
                for key, w in source_data[source].items():
                    weights.append({
                        'layer': key.split('.')[-1],
                        'mean': np.abs(w).mean(),
                        'shape': w.shape
                    })

                if weights:
                    layers = [w['layer'][:10] for w in weights[:8]]
                    means = [w['mean'] for w in weights[:8]]

                    bars = ax.bar(range(len(layers)), means,
                                 color=SOURCE_COLORS[source], alpha=0.7)
                    ax.set_xticks(range(len(layers)))
                    ax.set_xticklabels(layers, rotation=45, fontsize=7)
                    ax.set_ylabel('Mean |Weight|')
                    ax.set_title(f'{source.upper()} Encoder')
            else:
                ax.text(0.5, 0.5, f'No {source} encoder data', ha='center', va='center')

        plt.suptitle('Source Encoder Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '12_unified_source_embeddings.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 12_unified_source_embeddings.png")

    def _plot_encoder_decoder_analysis(self):
        """Compare encoder and decoder structures."""
        print("\n  Plotting encoder-decoder analysis...")

        encoder_stats = defaultdict(list)
        decoder_stats = defaultdict(list)

        for key in self.unified_state:
            w = self.unified_state[key].cpu().numpy()

            # Determine source
            source = 'shared'
            for src in SOURCE_COLORS.keys():
                if src in key:
                    source = src
                    break

            if 'encoder' in key and 'weight' in key:
                encoder_stats[source].append({
                    'key': key,
                    'mean': np.abs(w).mean(),
                    'std': w.std(),
                    'params': w.size
                })
            elif 'decoder' in key and 'weight' in key:
                decoder_stats[source].append({
                    'key': key,
                    'mean': np.abs(w).mean(),
                    'std': w.std(),
                    'params': w.size
                })

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Encoder vs Decoder weight magnitude by source
        ax1 = axes[0, 0]
        sources = list(set(list(encoder_stats.keys()) + list(decoder_stats.keys())))
        sources = [s for s in sources if s in SOURCE_COLORS]

        x = np.arange(len(sources))
        width = 0.35

        enc_means = [np.mean([s['mean'] for s in encoder_stats.get(src, [])]) if encoder_stats.get(src) else 0
                    for src in sources]
        dec_means = [np.mean([s['mean'] for s in decoder_stats.get(src, [])]) if decoder_stats.get(src) else 0
                    for src in sources]

        ax1.bar(x - width/2, enc_means, width, label='Encoder', alpha=0.7, color='steelblue')
        ax1.bar(x + width/2, dec_means, width, label='Decoder', alpha=0.7, color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.upper() for s in sources])
        ax1.set_ylabel('Mean |Weight|')
        ax1.set_title('Encoder vs Decoder Weight Magnitude')
        ax1.legend()

        # 2. Parameter distribution
        ax2 = axes[0, 1]
        enc_params = [sum(s['params'] for s in encoder_stats.get(src, [])) for src in sources]
        dec_params = [sum(s['params'] for s in decoder_stats.get(src, [])) for src in sources]

        ax2.bar(x - width/2, [p/1000 for p in enc_params], width, label='Encoder', alpha=0.7, color='steelblue')
        ax2.bar(x + width/2, [p/1000 for p in dec_params], width, label='Decoder', alpha=0.7, color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.upper() for s in sources])
        ax2.set_ylabel('Parameters (K)')
        ax2.set_title('Parameter Count by Component')
        ax2.legend()

        # 3. Weight std comparison
        ax3 = axes[1, 0]
        enc_stds = [np.mean([s['std'] for s in encoder_stats.get(src, [])]) if encoder_stats.get(src) else 0
                   for src in sources]
        dec_stds = [np.mean([s['std'] for s in decoder_stats.get(src, [])]) if decoder_stats.get(src) else 0
                   for src in sources]

        ax3.bar(x - width/2, enc_stds, width, label='Encoder', alpha=0.7, color='steelblue')
        ax3.bar(x + width/2, dec_stds, width, label='Decoder', alpha=0.7, color='coral')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.upper() for s in sources])
        ax3.set_ylabel('Weight Std Dev')
        ax3.set_title('Weight Variance by Component')
        ax3.legend()

        # 4. Summary pie chart
        ax4 = axes[1, 1]
        total_enc = sum(sum(s['params'] for s in encoder_stats.get(src, [])) for src in encoder_stats)
        total_dec = sum(sum(s['params'] for s in decoder_stats.get(src, [])) for src in decoder_stats)

        # Count fusion params
        fusion_params = 0
        for key in self.unified_state:
            if 'fusion' in key:
                fusion_params += self.unified_state[key].numel()

        sizes = [total_enc, fusion_params, total_dec]
        labels = [f'Encoders\n{total_enc//1000}K', f'Fusion\n{fusion_params//1000}K', f'Decoders\n{total_dec//1000}K']
        colors = ['steelblue', '#9b59b6', 'coral']

        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Parameter Distribution')

        plt.suptitle('Encoder-Decoder Architecture Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '13_unified_encoder_decoder.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 13_unified_encoder_decoder.png")

    def _plot_reconstruction_analysis(self):
        """Analyze reconstruction capabilities."""
        print("\n  Plotting reconstruction analysis...")

        # This would require running inference - for now, analyze decoder structure
        decoder_output_weights = {}

        for key in self.unified_state:
            if 'decoder' in key and 'weight' in key:
                parts = key.split('.')
                if 'decoders' in parts:
                    idx = parts.index('decoders')
                    if idx + 1 < len(parts):
                        source = parts[idx + 1]
                        if source not in decoder_output_weights:
                            decoder_output_weights[source] = []
                        decoder_output_weights[source].append({
                            'key': key,
                            'weight': self.unified_state[key].cpu().numpy()
                        })

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        source_keys = ['sentinel', 'deepstate', 'equipment', 'firms', 'ucdp']

        for idx, source in enumerate(source_keys):
            ax = axes[idx]

            if source in decoder_output_weights and decoder_output_weights[source]:
                # Get final layer weights (usually largest output dim)
                layers = decoder_output_weights[source]
                final_layer = max(layers, key=lambda x: x['weight'].shape[0])
                w = final_layer['weight']

                # Show weight structure
                if w.shape[0] <= 100 and w.shape[1] <= 100:
                    im = ax.imshow(np.abs(w), aspect='auto', cmap='YlOrRd')
                    ax.set_xlabel('Input Dim')
                    ax.set_ylabel('Output Feature')
                    ax.set_title(f'{source.upper()} Decoder Output\n({w.shape[0]} features)')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    # Too large - show histogram
                    ax.hist(w.flatten(), bins=50, alpha=0.7,
                           color=SOURCE_COLORS[source], edgecolor='black')
                    ax.set_xlabel('Weight Value')
                    ax.set_ylabel('Count')
                    ax.set_title(f'{source.upper()} Decoder Weights\n({w.shape})')
            else:
                ax.text(0.5, 0.5, f'No {source} decoder', ha='center', va='center')

        # Summary in last panel
        ax = axes[5]
        # Feature count comparison
        feature_counts = {}
        for source, layers in decoder_output_weights.items():
            if layers:
                final = max(layers, key=lambda x: x['weight'].shape[0])
                feature_counts[source] = final['weight'].shape[0]

        if feature_counts:
            sources = list(feature_counts.keys())
            counts = [feature_counts[s] for s in sources]
            colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

            ax.bar(sources, counts, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Output Features')
            ax.set_title('Reconstruction Feature Counts')
            ax.tick_params(axis='x', rotation=45)

            # Add total
            ax.text(0.95, 0.95, f'Total: {sum(counts)}', transform=ax.transAxes,
                   ha='right', va='top', fontsize=12, fontweight='bold')

        plt.suptitle('Decoder Reconstruction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(NETWORK_FIGURES_DIR / '14_unified_reconstruction.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 14_unified_reconstruction.png")

    # =========================================================================
    # SUMMARY DASHBOARD
    # =========================================================================

    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard."""
        print("\n" + "=" * 70)
        print("CREATING SUMMARY DASHBOARD")
        print("=" * 70)

        fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

        # Row 1: Overview
        ax1 = fig.add_subplot(gs[0, 0])
        self._dashboard_model_counts(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._dashboard_parameter_summary(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._dashboard_source_coverage(ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        self._dashboard_architecture_summary(ax4)

        # Row 2: JIM Analysis Summary
        ax5 = fig.add_subplot(gs[1, :2])
        self._dashboard_jim_weight_summary(ax5)

        ax6 = fig.add_subplot(gs[1, 2:])
        self._dashboard_jim_embedding_summary(ax6)

        # Row 3: Unified Model Summary
        ax7 = fig.add_subplot(gs[2, :2])
        self._dashboard_unified_structure(ax7)

        ax8 = fig.add_subplot(gs[2, 2:])
        self._dashboard_cross_source_summary(ax8)

        # Row 4: Pipeline Overview
        ax9 = fig.add_subplot(gs[3, :])
        self._dashboard_pipeline_diagram(ax9)

        plt.suptitle('ML_OSINT Network Analysis Summary Dashboard',
                    fontsize=20, fontweight='bold', y=0.98)

        plt.savefig(NETWORK_FIGURES_DIR / '00_summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: 00_summary_dashboard.png")

    def _dashboard_model_counts(self, ax):
        """Model count summary."""
        counts = {
            'JIM Models': len(self.jim_states),
            'Unified': 1 if self.unified_state else 0,
        }

        # Group JIM by source
        source_counts = defaultdict(int)
        for name in self.jim_states:
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source_counts[src] += 1
                    break

        colors = ['#3498db', '#9b59b6']
        bars = ax.bar(list(counts.keys()), list(counts.values()), color=colors, alpha=0.7)

        for bar, count in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_ylabel('Count')
        ax.set_title('Trained Models', fontweight='bold')

    def _dashboard_parameter_summary(self, ax):
        """Parameter count summary."""
        jim_params = sum(sum(v.numel() for v in state.values()) for state in self.jim_states.values())
        unified_params = sum(v.numel() for v in self.unified_state.values()) if self.unified_state else 0

        sizes = [jim_params, unified_params]
        labels = [f'JIM\n{jim_params//1000}K', f'Unified\n{unified_params//1000}K']
        colors = ['#3498db', '#9b59b6']

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Parameters\nTotal: {(jim_params+unified_params)//1000}K', fontweight='bold')

    def _dashboard_source_coverage(self, ax):
        """Source coverage pie chart."""
        source_counts = defaultdict(int)
        for name in self.jim_states:
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source_counts[src] += 1
                    break

        if source_counts:
            sources = list(source_counts.keys())
            counts = [source_counts[s] for s in sources]
            colors = [SOURCE_COLORS[s] for s in sources]

            ax.pie(counts, labels=[s.upper() for s in sources], colors=colors,
                  autopct='%1.0f%%', startangle=90)
            ax.set_title('JIM Models by Source', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No source data', ha='center', va='center')

    def _dashboard_architecture_summary(self, ax):
        """Architecture summary text."""
        ax.axis('off')

        text = """Architecture Summary

JIM (Joint Interpolation Models):
  - Per-source temporal interpolation
  - Cross-feature attention
  - Uncertainty estimation
  - Gap filling for irregular data

Unified Cross-Source Fusion:
  - 5 source encoders
  - Transformer cross-attention
  - Self-supervised reconstruction
  - Learns source relationships

Pipeline: JIM → Unified → HAN
"""
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace')
        ax.set_title('Architecture', fontweight='bold')

    def _dashboard_jim_weight_summary(self, ax):
        """JIM weight statistics summary."""
        source_stats = defaultdict(list)

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            total_params = sum(v.numel() for v in state.values())
            mean_weight = np.mean([np.abs(v.cpu().numpy()).mean() for v in state.values()])

            source_stats[source].append({
                'params': total_params,
                'mean_weight': mean_weight
            })

        sources = list(source_stats.keys())
        avg_params = [np.mean([s['params'] for s in source_stats[src]])/1000 for src in sources]
        avg_weights = [np.mean([s['mean_weight'] for s in source_stats[src]]) for src in sources]
        colors = [SOURCE_COLORS.get(s, 'gray') for s in sources]

        x = np.arange(len(sources))
        width = 0.35

        ax2 = ax.twinx()

        bars1 = ax.bar(x - width/2, avg_params, width, color=colors, alpha=0.7, label='Params (K)')
        bars2 = ax2.bar(x + width/2, avg_weights, width, color=colors, alpha=0.4,
                       hatch='///', label='Mean |W|')

        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in sources])
        ax.set_ylabel('Avg Parameters (K)')
        ax2.set_ylabel('Avg |Weight|')
        ax.set_title('JIM Model Statistics by Source', fontweight='bold')

        # Combined legend
        ax.legend([bars1, bars2], ['Params (K)', 'Mean |W|'], loc='upper right')

    def _dashboard_jim_embedding_summary(self, ax):
        """JIM embedding summary."""
        if not HAS_SKLEARN:
            ax.text(0.5, 0.5, 'sklearn required', ha='center', va='center')
            return

        # Collect embeddings
        embeddings = []
        sources = []

        for name, state in self.jim_states.items():
            source = 'other'
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source = src
                    break

            for key in state:
                if 'feature_embeddings.weight' in key:
                    emb = state[key].cpu().numpy().mean(axis=0)
                    embeddings.append(emb)
                    sources.append(source)
                    break

        if len(embeddings) >= 3:
            X = np.vstack(embeddings)
            pca = PCA(n_components=2)
            projected = pca.fit_transform(X)

            for i, (x, y) in enumerate(projected):
                ax.scatter(x, y, c=SOURCE_COLORS.get(sources[i], 'gray'),
                          s=60, alpha=0.7, edgecolors='black')

            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=c, markersize=8, label=s.upper())
                      for s, c in SOURCE_COLORS.items() if s in sources]
            ax.legend(handles=handles, loc='best', fontsize=8)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')

        ax.set_title('JIM Embedding Space (PCA)', fontweight='bold')

    def _dashboard_unified_structure(self, ax):
        """Unified model structure summary."""
        if not self.unified_state:
            ax.text(0.5, 0.5, 'No unified model', ha='center', va='center')
            return

        # Count parameters by component
        component_params = defaultdict(int)

        for key in self.unified_state:
            if 'encoder' in key:
                component_params['Encoders'] += self.unified_state[key].numel()
            elif 'decoder' in key:
                component_params['Decoders'] += self.unified_state[key].numel()
            elif 'fusion' in key:
                component_params['Fusion'] += self.unified_state[key].numel()
            else:
                component_params['Other'] += self.unified_state[key].numel()

        components = list(component_params.keys())
        params = [component_params[c]/1000 for c in components]
        colors = ['steelblue', 'coral', '#9b59b6', 'gray'][:len(components)]

        bars = ax.bar(components, params, color=colors, alpha=0.7)
        ax.set_ylabel('Parameters (K)')
        ax.set_title('Unified Model Components', fontweight='bold')

        for bar, p in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{p:.0f}K', ha='center', va='bottom', fontsize=9)

    def _dashboard_cross_source_summary(self, ax):
        """Cross-source relationship summary."""
        if not self.unified_state:
            ax.text(0.5, 0.5, 'No unified model', ha='center', va='center')
            return

        # Find source embeddings
        for key in self.unified_state:
            if 'source_embeddings.weight' in key:
                emb = self.unified_state[key].cpu().numpy()

                if HAS_SKLEARN:
                    sim = cosine_similarity(emb)
                else:
                    norms = np.linalg.norm(emb, axis=1, keepdims=True)
                    sim = (emb / (norms + 1e-8)) @ (emb / (norms + 1e-8)).T

                sources = ['Sent', 'Deep', 'Equip', 'FIRMS', 'UCDP']
                im = ax.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
                ax.set_xticks(range(5))
                ax.set_yticks(range(5))
                ax.set_xticklabels(sources, fontsize=8)
                ax.set_yticklabels(sources, fontsize=8)
                ax.set_title('Learned Source Similarity', fontweight='bold')
                plt.colorbar(im, ax=ax, shrink=0.8)
                return

        ax.text(0.5, 0.5, 'No source embeddings', ha='center', va='center')

    def _dashboard_pipeline_diagram(self, ax):
        """Pipeline overview diagram."""
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.axis('off')

        # Components
        components = [
            ('Raw Data\n5 Sources', 10, '#ecf0f1'),
            ('JIM Models\nInterpolation', 30, '#3498db'),
            ('Unified Fusion\nCross-Source', 50, '#9b59b6'),
            ('Aggregation\nDaily→Monthly', 70, '#1abc9c'),
            ('HAN\nForecasting', 90, '#e74c3c'),
        ]

        for label, x, color in components:
            rect = FancyBboxPatch((x-8, 6), 16, 8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=color,
                                  edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, 10, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')

        # Arrows
        for i in range(len(components) - 1):
            x1 = components[i][1] + 8
            x2 = components[i+1][1] - 8
            ax.annotate('', xy=(x2, 10), xytext=(x1, 10),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.set_title('ML_OSINT Processing Pipeline', fontsize=12, fontweight='bold', y=0.95)

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE NETWORK ANALYSIS")
        print("=" * 70)
        print(f"Output directory: {NETWORK_FIGURES_DIR}")

        self.load_all_models()

        # Run all analyses
        self.create_summary_dashboard()
        self.analyze_intra_network()
        self.analyze_inter_network()
        self.analyze_unified_model()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nGenerated figures in: {NETWORK_FIGURES_DIR}")

        # List all figures
        figures = sorted(NETWORK_FIGURES_DIR.glob("*.png"))
        print(f"\n{len(figures)} figures generated:")
        for f in figures:
            print(f"  - {f.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Network Analysis')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['all', 'intra', 'inter', 'unified', 'dashboard'],
                       help='Which analysis to run')
    args = parser.parse_args()

    analyzer = ComprehensiveNetworkAnalyzer(device=args.device)

    if args.analysis == 'all':
        analyzer.run_full_analysis()
    else:
        analyzer.load_all_models()

        if args.analysis == 'intra':
            analyzer.analyze_intra_network()
        elif args.analysis == 'inter':
            analyzer.analyze_inter_network()
        elif args.analysis == 'unified':
            analyzer.analyze_unified_model()
        elif args.analysis == 'dashboard':
            analyzer.create_summary_dashboard()


if __name__ == "__main__":
    main()
