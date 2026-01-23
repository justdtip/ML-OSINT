#!/usr/bin/env python3
"""
Detailed Network Analysis and Visualization

Produces comprehensive figures showing what the trained models learned:
1. Attention weight heatmaps (cross-feature correlations)
2. Feature embedding clustering (t-SNE/PCA)
3. Temporal encoding patterns
4. Uncertainty calibration
5. Domain-specific learned representations
6. Cross-source correlation structure
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Models and data
from hierarchical_attention_network import (
    HierarchicalAttentionNetwork, DOMAIN_CONFIGS, TOTAL_FEATURES
)
from joint_interpolation_models import (
    INTERPOLATION_CONFIGS, PHASE2_CONFIGS, JointInterpolationModel
)
from conflict_data_loader import create_data_loaders

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR,
)

BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = ANALYSIS_FIGURES_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class NetworkAnalyzer:
    """Comprehensive analysis of trained network weights and behavior."""

    def __init__(self, device='cpu'):
        self.device = device
        self.han_model = None
        self.phase1_models = {}
        self.phase2_models = {}
        self.phase3_models = {}

    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")

        # HAN
        han_path = MODEL_DIR / "han_best.pt"
        if han_path.exists():
            self.han_model = HierarchicalAttentionNetwork(
                domain_configs=DOMAIN_CONFIGS,
                d_model=32, nhead=2,
                num_encoder_layers=1, num_temporal_layers=1,
                dropout=0.35
            )
            checkpoint = torch.load(han_path, map_location=self.device)
            self.han_model.load_state_dict(checkpoint['model_state_dict'])
            self.han_model.eval()
            print(f"  Loaded HAN model")

        # Phase 3 models
        for model_path in MODEL_DIR.glob("phase3_*.pt"):
            name = model_path.stem.replace('phase3_', '').replace('_best', '')
            state = torch.load(model_path, map_location=self.device)
            self.phase3_models[name] = state
        print(f"  Loaded {len(self.phase3_models)} Phase 3 models")

        # Phase 1 models
        for model_path in INTERP_MODEL_DIR.glob("interp_*.pt"):
            name = model_path.stem.replace('interp_', '').replace('_best', '')
            state = torch.load(model_path, map_location=self.device)
            self.phase1_models[name] = state
        print(f"  Loaded {len(self.phase1_models)} Phase 1 models")

    def analyze_han_attention(self):
        """Analyze HAN cross-domain attention patterns."""
        if self.han_model is None:
            print("HAN model not loaded")
            return

        print("\n" + "="*70)
        print("ANALYZING HAN ATTENTION PATTERNS")
        print("="*70)

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Domain encoder attention weights
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_domain_encoder_weights(ax1)

        # 2. Feature embeddings per domain
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_domain_feature_norms(ax2)

        # 3. Cross-domain fusion weights
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_cross_domain_fusion(ax3)

        # 4. Regime classifier weights
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_regime_classifier(ax4)

        # 5. Anomaly detector weights
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_anomaly_detector(ax5)

        # 6. Forecast head analysis
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_forecast_head(ax6)

        plt.suptitle('Hierarchical Attention Network Analysis', fontsize=16, fontweight='bold')
        plt.savefig(OUTPUT_DIR / 'han_attention_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'han_attention_analysis.png'}")

    def _plot_domain_encoder_weights(self, ax):
        """Plot domain encoder transformer weights."""
        # Extract weights from each domain encoder
        domain_weights = {}

        for name, encoder in self.han_model.domain_encoders.items():
            # Get transformer attention weights
            state = encoder.state_dict()

            # Self-attention in_proj weight contains Q, K, V projections
            if 'transformer.layers.0.self_attn.in_proj_weight' in state:
                w = state['transformer.layers.0.self_attn.in_proj_weight'].cpu().numpy()
                domain_weights[name] = np.abs(w).mean(axis=1)  # Average over output dim

        if domain_weights:
            domains = list(domain_weights.keys())
            weights_matrix = np.array([domain_weights[d][:32] for d in domains])  # First 32 dims

            im = ax.imshow(weights_matrix, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(domains)))
            ax.set_yticklabels([d.upper() for d in domains])
            ax.set_xlabel('Attention Dimension')
            ax.set_title('Domain Encoder Attention Weight Magnitudes')
            plt.colorbar(im, ax=ax, label='|Weight|')
        else:
            ax.text(0.5, 0.5, 'No encoder weights found', ha='center', va='center')
            ax.set_title('Domain Encoder Weights')

    def _plot_domain_feature_norms(self, ax):
        """Plot feature embedding norms by domain."""
        norms = {}

        for name, encoder in self.han_model.domain_encoders.items():
            state = encoder.state_dict()
            if 'feature_embeddings.weight' in state:
                emb = state['feature_embeddings.weight'].cpu().numpy()
                norms[name] = np.linalg.norm(emb, axis=1)

        if norms:
            domains = list(norms.keys())

            # Box plot of norms
            data = [norms[d] for d in domains]
            bp = ax.boxplot(data, labels=[d[:8].upper() for d in domains], patch_artist=True)

            colors = plt.cm.Set2(np.linspace(0, 1, len(domains)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            ax.set_ylabel('Embedding L2 Norm')
            ax.set_title('Feature Embedding Norms by Domain')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No embeddings found', ha='center', va='center')

    def _plot_cross_domain_fusion(self, ax):
        """Plot cross-domain attention fusion patterns."""
        state = self.han_model.state_dict()

        # Cross-domain attention weights
        if 'cross_domain_attention.in_proj_weight' in state:
            w = state['cross_domain_attention.in_proj_weight'].cpu().numpy()

            # Split into Q, K, V (each is d_model x d_model)
            d_model = w.shape[0] // 3
            Q, K, V = w[:d_model], w[d_model:2*d_model], w[2*d_model:]

            # Plot attention weight structure
            fig_w = np.concatenate([Q, K, V], axis=1)

            im = ax.imshow(np.abs(fig_w), aspect='auto', cmap='magma')
            ax.axvline(d_model - 0.5, color='white', linewidth=2, linestyle='--')
            ax.axvline(2*d_model - 0.5, color='white', linewidth=2, linestyle='--')

            ax.set_xlabel('Dimension (Q | K | V)')
            ax.set_ylabel('Output Dimension')
            ax.set_title('Cross-Domain Attention Projection Weights')

            # Add labels
            ax.text(d_model//2, -2, 'Query', ha='center', fontsize=10, fontweight='bold')
            ax.text(d_model + d_model//2, -2, 'Key', ha='center', fontsize=10, fontweight='bold')
            ax.text(2*d_model + d_model//2, -2, 'Value', ha='center', fontsize=10, fontweight='bold')

            plt.colorbar(im, ax=ax, label='|Weight|')
        else:
            ax.text(0.5, 0.5, 'Cross-domain attention not found', ha='center', va='center')
            ax.set_title('Cross-Domain Fusion')

    def _plot_regime_classifier(self, ax):
        """Plot regime classifier weights."""
        state = self.han_model.state_dict()

        # Find regime head weights
        regime_weights = None
        for key in state:
            if 'regime_head' in key and 'weight' in key and len(state[key].shape) == 2:
                regime_weights = state[key].cpu().numpy()
                break

        if regime_weights is not None:
            regime_labels = ['Low', 'Medium', 'High', 'Offensive']

            # If we have the final layer (4 x d_model)
            if regime_weights.shape[0] == 4:
                im = ax.imshow(regime_weights, aspect='auto', cmap='RdYlGn_r')
                ax.set_yticks(range(4))
                ax.set_yticklabels(regime_labels)
                ax.set_xlabel('Hidden Dimension')
                ax.set_title('Regime Classifier Weights')
                plt.colorbar(im, ax=ax)
            else:
                # Plot weight magnitude histogram
                ax.hist(regime_weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Count')
                ax.set_title('Regime Head Weight Distribution')
        else:
            ax.text(0.5, 0.5, 'Regime classifier not found', ha='center', va='center')
            ax.set_title('Regime Classifier')

    def _plot_anomaly_detector(self, ax):
        """Plot anomaly detector weights."""
        state = self.han_model.state_dict()

        anomaly_weights = []
        for key in sorted(state.keys()):
            if 'anomaly_head' in key and 'weight' in key:
                anomaly_weights.append(state[key].cpu().numpy())

        if anomaly_weights:
            # Plot the final layer weights
            w = anomaly_weights[-1].flatten()

            # Color by sign
            colors = ['red' if x < 0 else 'green' for x in w]
            ax.bar(range(len(w)), np.abs(w), color=colors, alpha=0.7)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('|Weight|')
            ax.set_title('Anomaly Detector Final Layer\n(Green=+, Red=-)')
        else:
            ax.text(0.5, 0.5, 'Anomaly detector not found', ha='center', va='center')
            ax.set_title('Anomaly Detector')

    def _plot_forecast_head(self, ax):
        """Plot forecast head weight analysis."""
        state = self.han_model.state_dict()

        forecast_weights = []
        for key in sorted(state.keys()):
            if 'forecast_head' in key and 'weight' in key:
                forecast_weights.append(state[key].cpu().numpy())

        if forecast_weights:
            # Analyze weight statistics across layers
            stats = []
            for i, w in enumerate(forecast_weights):
                stats.append({
                    'layer': i,
                    'mean': np.abs(w).mean(),
                    'std': w.std(),
                    'max': np.abs(w).max(),
                    'sparsity': (np.abs(w) < 0.01).mean()
                })

            x = [s['layer'] for s in stats]
            ax.bar(x, [s['mean'] for s in stats], alpha=0.7, label='Mean |W|')
            ax.errorbar(x, [s['mean'] for s in stats],
                       yerr=[s['std'] for s in stats], fmt='none', color='black', capsize=3)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Weight Statistics')
            ax.set_title('Forecast Head Weight Statistics')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Forecast head not found', ha='center', va='center')
            ax.set_title('Forecast Head')

    def analyze_feature_embeddings(self):
        """Analyze learned feature embeddings across all models."""
        print("\n" + "="*70)
        print("ANALYZING FEATURE EMBEDDINGS")
        print("="*70)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Collect embeddings from Phase 3 models by source
        embeddings_by_source = {}

        for name, state in self.phase3_models.items():
            # Determine source
            source = None
            for src in ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']:
                if src in name:
                    source = src
                    break

            if source is None:
                continue

            # Get feature embeddings
            if 'cross_feature_attn.feature_embeddings.weight' in state:
                emb = state['cross_feature_attn.feature_embeddings.weight'].cpu().numpy()
                if source not in embeddings_by_source:
                    embeddings_by_source[source] = []
                embeddings_by_source[source].append((name, emb))

        # Plot embeddings for each source
        sources = ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']

        for idx, source in enumerate(sources):
            ax = axes[idx // 3, idx % 3]

            if source in embeddings_by_source and embeddings_by_source[source]:
                # Combine embeddings from all models for this source
                all_emb = []
                labels = []

                for model_name, emb in embeddings_by_source[source]:
                    all_emb.append(emb)
                    labels.extend([f'{model_name[:15]}_{i}' for i in range(emb.shape[0])])

                combined = np.vstack(all_emb)

                # PCA for visualization
                from sklearn.decomposition import PCA

                if combined.shape[0] >= 2:
                    pca = PCA(n_components=min(2, combined.shape[1]))
                    projected = pca.fit_transform(combined)

                    # Color by model
                    colors = []
                    model_names = []
                    start = 0
                    for model_name, emb in embeddings_by_source[source]:
                        n = emb.shape[0]
                        colors.extend([len(model_names)] * n)
                        model_names.append(model_name[:20])
                        start += n

                    scatter = ax.scatter(projected[:, 0],
                                        projected[:, 1] if projected.shape[1] > 1 else np.zeros(len(projected)),
                                        c=colors, cmap='Set1', alpha=0.7, s=50)

                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                    if projected.shape[1] > 1:
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    ax.set_title(f'{source.upper()} Feature Embeddings\n({combined.shape[0]} features)')

                    # Add legend
                    if len(model_names) <= 5:
                        handles = [plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=plt.cm.Set1(i/len(model_names)),
                                            markersize=8, label=n)
                                  for i, n in enumerate(model_names)]
                        ax.legend(handles=handles, fontsize=7, loc='best')
            else:
                ax.text(0.5, 0.5, f'No {source} embeddings', ha='center', va='center')
                ax.set_title(f'{source.upper()} Embeddings')

        # Use last subplot for summary
        ax = axes[1, 2]
        self._plot_embedding_summary(ax, embeddings_by_source)

        plt.suptitle('Learned Feature Embeddings Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'feature_embeddings_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'feature_embeddings_analysis.png'}")

    def _plot_embedding_summary(self, ax, embeddings_by_source):
        """Summary statistics of embeddings."""
        stats = []

        for source, models in embeddings_by_source.items():
            for model_name, emb in models:
                stats.append({
                    'source': source,
                    'n_features': emb.shape[0],
                    'd_model': emb.shape[1],
                    'mean_norm': np.linalg.norm(emb, axis=1).mean(),
                    'std_norm': np.linalg.norm(emb, axis=1).std()
                })

        if stats:
            sources = list(set(s['source'] for s in stats))
            x = np.arange(len(sources))

            mean_norms = [np.mean([s['mean_norm'] for s in stats if s['source'] == src]) for src in sources]
            std_norms = [np.mean([s['std_norm'] for s in stats if s['source'] == src]) for src in sources]
            n_features = [sum(s['n_features'] for s in stats if s['source'] == src) for src in sources]

            bars = ax.bar(x, mean_norms, yerr=std_norms, capsize=5, alpha=0.7)

            # Add feature counts as text
            for i, (bar, n) in enumerate(zip(bars, n_features)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'n={n}', ha='center', va='bottom', fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels([s.upper() for s in sources], rotation=45)
            ax.set_ylabel('Mean Embedding Norm')
            ax.set_title('Embedding Norms by Source')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    def analyze_temporal_patterns(self):
        """Analyze learned temporal encoding patterns."""
        print("\n" + "="*70)
        print("ANALYZING TEMPORAL PATTERNS")
        print("="*70)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Day embedding patterns from interpolation models
        ax = axes[0, 0]
        self._plot_day_embeddings(ax)

        # 2. Sinusoidal PE patterns
        ax = axes[0, 1]
        self._plot_sinusoidal_pe(ax)

        # 3. Gap interpolator day queries
        ax = axes[1, 0]
        self._plot_gap_queries(ax)

        # 4. Temporal encoding combined
        ax = axes[1, 1]
        self._plot_temporal_combined(ax)

        plt.suptitle('Temporal Encoding Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'temporal_patterns_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'temporal_patterns_analysis.png'}")

    def _plot_day_embeddings(self, ax):
        """Plot learned day embeddings."""
        # Collect day embeddings from Phase 3 models
        day_embs = []

        for name, state in list(self.phase3_models.items())[:5]:  # First 5 models
            if 'temporal_encoding.day_embedding.weight' in state:
                emb = state['temporal_encoding.day_embedding.weight'].cpu().numpy()
                day_embs.append((name[:15], emb))

        if day_embs:
            # Plot embeddings for first 30 days
            days_to_show = 30

            for model_name, emb in day_embs:
                # Average across embedding dimensions
                daily_avg = emb[:days_to_show].mean(axis=1)
                ax.plot(range(days_to_show), daily_avg, label=model_name, alpha=0.7)

            ax.set_xlabel('Day Index')
            ax.set_ylabel('Mean Embedding Value')
            ax.set_title('Learned Day Embeddings (First 30 Days)')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No day embeddings found', ha='center', va='center')

    def _plot_sinusoidal_pe(self, ax):
        """Plot sinusoidal positional encodings."""
        for name, state in list(self.phase3_models.items())[:1]:
            if 'temporal_encoding.pe_sinusoidal' in state:
                pe = state['temporal_encoding.pe_sinusoidal'].cpu().numpy()

                # Show first 100 days, first 16 dimensions
                im = ax.imshow(pe[:100, :16].T, aspect='auto', cmap='RdBu_r')
                ax.set_xlabel('Day Index')
                ax.set_ylabel('Dimension')
                ax.set_title('Sinusoidal Positional Encoding')
                plt.colorbar(im, ax=ax)
                return

        ax.text(0.5, 0.5, 'No PE found', ha='center', va='center')
        ax.set_title('Sinusoidal PE')

    def _plot_gap_queries(self, ax):
        """Plot gap interpolator day queries."""
        queries = []

        for name, state in self.phase3_models.items():
            if 'gap_interpolator.day_queries.weight' in state:
                q = state['gap_interpolator.day_queries.weight'].cpu().numpy()
                queries.append(q)

        if queries:
            # Average across models
            avg_queries = np.mean(queries, axis=0)

            im = ax.imshow(avg_queries.T, aspect='auto', cmap='viridis')
            ax.set_xlabel('Day Position in Gap')
            ax.set_ylabel('Query Dimension')
            ax.set_title('Gap Interpolator Day Queries\n(Averaged across models)')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No gap queries found', ha='center', va='center')

    def _plot_temporal_combined(self, ax):
        """Plot combined temporal encoding output."""
        # Get combine layer weights
        combine_weights = []

        for name, state in self.phase3_models.items():
            if 'temporal_encoding.combine.weight' in state:
                w = state['temporal_encoding.combine.weight'].cpu().numpy()
                combine_weights.append(np.abs(w).mean(axis=0))

        if combine_weights:
            avg_weights = np.mean(combine_weights, axis=0)

            # Split into day embedding and sinusoidal parts
            mid = len(avg_weights) // 2

            x = np.arange(mid)
            ax.bar(x - 0.2, avg_weights[:mid], width=0.4, label='Day Embedding', alpha=0.7)
            ax.bar(x + 0.2, avg_weights[mid:mid*2], width=0.4, label='Sinusoidal PE', alpha=0.7)

            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('Average |Weight|')
            ax.set_title('Temporal Combine Layer Weights')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No combine weights found', ha='center', va='center')

    def analyze_cross_feature_attention(self):
        """Analyze cross-feature attention patterns."""
        print("\n" + "="*70)
        print("ANALYZING CROSS-FEATURE ATTENTION")
        print("="*70)

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Group models by source
        sources = ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']

        for idx, source in enumerate(sources):
            row, col = idx // 3, idx % 3
            ax = fig.add_subplot(gs[row, col])
            self._plot_source_attention(ax, source)

        # Summary plot
        ax = fig.add_subplot(gs[1, 2])
        self._plot_attention_summary(ax)

        # Attention head diversity
        ax = fig.add_subplot(gs[2, :])
        self._plot_attention_head_diversity(ax)

        plt.suptitle('Cross-Feature Attention Analysis', fontsize=14, fontweight='bold')
        plt.savefig(OUTPUT_DIR / 'cross_feature_attention.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'cross_feature_attention.png'}")

    def _plot_source_attention(self, ax, source):
        """Plot attention patterns for a specific source."""
        # Collect attention weights
        attn_weights = []

        for name, state in self.phase3_models.items():
            if source not in name:
                continue

            # Get self-attention weights
            for key in state:
                if 'cross_feature_attn.transformer.layers.0.self_attn.in_proj_weight' in key:
                    w = state[key].cpu().numpy()
                    attn_weights.append(w)
                    break

        if attn_weights:
            # Average across models
            avg_weights = np.mean(np.abs(attn_weights), axis=0)

            # Show as heatmap
            im = ax.imshow(avg_weights[:32, :32], cmap='YlOrRd')
            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('Output Dimension')
            ax.set_title(f'{source.upper()} Self-Attention')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, f'No {source} attention found', ha='center', va='center')
            ax.set_title(f'{source.upper()} Attention')

    def _plot_attention_summary(self, ax):
        """Summary of attention patterns."""
        summary = {}

        for source in ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']:
            weights = []
            for name, state in self.phase3_models.items():
                if source not in name:
                    continue
                for key in state:
                    if 'self_attn.in_proj_weight' in key:
                        w = state[key].cpu().numpy()
                        weights.append(np.abs(w).mean())
                        break

            if weights:
                summary[source] = np.mean(weights)

        if summary:
            sources = list(summary.keys())
            values = [summary[s] for s in sources]

            bars = ax.bar(sources, values, alpha=0.7)
            ax.set_xticklabels([s.upper() for s in sources], rotation=45)
            ax.set_ylabel('Mean |Attention Weight|')
            ax.set_title('Attention Magnitude by Source')
        else:
            ax.text(0.5, 0.5, 'No summary data', ha='center', va='center')

    def _plot_attention_head_diversity(self, ax):
        """Analyze attention head specialization."""
        # Collect attention patterns across all models
        head_patterns = []
        model_labels = []

        for name, state in list(self.phase3_models.items()):
            for key in state:
                if 'self_attn.out_proj.weight' in key:
                    w = state[key].cpu().numpy()
                    # Reshape to see head patterns
                    head_patterns.append(np.abs(w).flatten()[:64])  # First 64 elements
                    model_labels.append(name[:12])
                    break

        if head_patterns and len(head_patterns) >= 2:
            patterns = np.vstack(head_patterns)

            # Compute correlation between models
            corr = np.corrcoef(patterns)

            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title('Model Attention Pattern Similarity')

            # Add labels for every 5th model
            tick_positions = list(range(0, len(model_labels), max(1, len(model_labels)//10)))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([model_labels[i] for i in tick_positions], rotation=90, fontsize=7)
            ax.set_yticklabels([model_labels[i] for i in tick_positions], fontsize=7)

            plt.colorbar(im, ax=ax, label='Correlation')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for diversity analysis', ha='center', va='center')

    def analyze_uncertainty_learning(self):
        """Analyze learned uncertainty estimation."""
        print("\n" + "="*70)
        print("ANALYZING UNCERTAINTY ESTIMATION")
        print("="*70)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Uncertainty head weights
        ax = axes[0, 0]
        self._plot_uncertainty_weights(ax)

        # 2. Uncertainty head bias
        ax = axes[0, 1]
        self._plot_uncertainty_bias(ax)

        # 3. Weight distribution comparison
        ax = axes[1, 0]
        self._plot_uncertainty_distribution(ax)

        # 4. Source-wise uncertainty analysis
        ax = axes[1, 1]
        self._plot_source_uncertainty(ax)

        plt.suptitle('Uncertainty Estimation Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'uncertainty_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'uncertainty_analysis.png'}")

    def _plot_uncertainty_weights(self, ax):
        """Plot uncertainty head weight patterns."""
        unc_weights = []
        labels = []

        for name, state in list(self.phase3_models.items())[:10]:
            if 'uncertainty_head.0.weight' in state:
                w = state['uncertainty_head.0.weight'].cpu().numpy()
                unc_weights.append(np.abs(w).mean(axis=0))
                labels.append(name[:15])

        if unc_weights:
            data = np.vstack(unc_weights)

            im = ax.imshow(data, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Input Dimension')
            ax.set_title('Uncertainty Head Layer 1 Weights')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No uncertainty weights found', ha='center', va='center')

    def _plot_uncertainty_bias(self, ax):
        """Plot uncertainty head biases."""
        biases = []
        labels = []

        for name, state in self.phase3_models.items():
            if 'uncertainty_head.2.bias' in state:
                b = state['uncertainty_head.2.bias'].cpu().numpy().item()
                biases.append(b)
                labels.append(name[:15])

        if biases:
            # Sort by bias value
            sorted_idx = np.argsort(biases)
            biases = [biases[i] for i in sorted_idx]
            labels = [labels[i] for i in sorted_idx]

            colors = ['red' if b < 0 else 'green' for b in biases]
            ax.barh(range(len(biases)), biases, color=colors, alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Final Bias Value')
            ax.set_title('Uncertainty Head Output Bias')
        else:
            ax.text(0.5, 0.5, 'No bias found', ha='center', va='center')

    def _plot_uncertainty_distribution(self, ax):
        """Plot weight distribution for uncertainty vs other heads."""
        unc_weights_all = []
        other_weights_all = []

        for name, state in self.phase3_models.items():
            for key in state:
                w = state[key].cpu().numpy().flatten()
                if 'uncertainty_head' in key and 'weight' in key:
                    unc_weights_all.extend(w)
                elif 'decoder' in key and 'weight' in key:
                    other_weights_all.extend(w)

        if unc_weights_all and other_weights_all:
            ax.hist(unc_weights_all, bins=50, alpha=0.5, label='Uncertainty', density=True)
            ax.hist(other_weights_all, bins=50, alpha=0.5, label='Decoder', density=True)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.set_title('Weight Distribution: Uncertainty vs Decoder')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    def _plot_source_uncertainty(self, ax):
        """Plot uncertainty characteristics by source."""
        source_stats = {}

        for source in ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']:
            weights = []
            for name, state in self.phase3_models.items():
                if source not in name:
                    continue
                if 'uncertainty_head.0.weight' in state:
                    w = state['uncertainty_head.0.weight'].cpu().numpy()
                    weights.append(np.abs(w).mean())

            if weights:
                source_stats[source] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights)
                }

        if source_stats:
            sources = list(source_stats.keys())
            means = [source_stats[s]['mean'] for s in sources]
            stds = [source_stats[s]['std'] for s in sources]

            bars = ax.bar(sources, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xticklabels([s.upper() for s in sources], rotation=45)
            ax.set_ylabel('Mean |Uncertainty Weight|')
            ax.set_title('Uncertainty Estimation by Source')
        else:
            ax.text(0.5, 0.5, 'No source data', ha='center', va='center')

    def analyze_model_predictions(self):
        """Run predictions and analyze model behavior."""
        print("\n" + "="*70)
        print("ANALYZING MODEL PREDICTIONS")
        print("="*70)

        if self.han_model is None:
            print("HAN model not loaded, skipping prediction analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Load validation data
        _, val_loader = create_data_loaders(DOMAIN_CONFIGS, batch_size=4, seq_len=4)

        # Get predictions
        predictions = []
        targets = []
        regimes = []
        anomalies = []

        self.han_model.to(self.device)
        self.han_model.eval()
        with torch.no_grad():
            for features, masks, target in val_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}

                outputs = self.han_model(features, masks, return_attention=False)

                predictions.append(outputs['forecast'][:, -1, :].cpu().numpy())

                if 'regime_logits' in outputs:
                    regime_probs = F.softmax(outputs['regime_logits'][:, -1, :], dim=-1)
                    regimes.append(regime_probs.cpu().numpy())

                if 'anomaly_score' in outputs:
                    anomalies.append(outputs['anomaly_score'][:, -1].cpu().numpy())

                # Get targets
                target_features = torch.cat([
                    target['next_features'][d][:, 0, :]
                    for d in self.han_model.domain_names
                ], dim=-1)
                targets.append(target_features.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # 1. Prediction vs Target scatter
        ax = axes[0, 0]
        self._plot_prediction_scatter(ax, predictions, targets)

        # 2. Error distribution
        ax = axes[0, 1]
        self._plot_error_distribution(ax, predictions, targets)

        # 3. Regime predictions
        ax = axes[1, 0]
        if regimes:
            self._plot_regime_predictions(ax, np.vstack(regimes))
        else:
            ax.text(0.5, 0.5, 'No regime predictions', ha='center', va='center')
            ax.set_title('Regime Predictions')

        # 4. Anomaly scores
        ax = axes[1, 1]
        if anomalies:
            self._plot_anomaly_scores(ax, np.concatenate(anomalies))
        else:
            ax.text(0.5, 0.5, 'No anomaly scores', ha='center', va='center')
            ax.set_title('Anomaly Scores')

        plt.suptitle('Model Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'prediction_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'prediction_analysis.png'}")

    def _plot_prediction_scatter(self, ax, predictions, targets):
        """Scatter plot of predictions vs targets."""
        # Sample for visualization
        n_points = min(5000, predictions.size)

        pred_flat = predictions.flatten()[:n_points]
        tgt_flat = targets.flatten()[:n_points]

        ax.scatter(tgt_flat, pred_flat, alpha=0.3, s=5)

        # Perfect prediction line
        lims = [min(tgt_flat.min(), pred_flat.min()),
                max(tgt_flat.max(), pred_flat.max())]
        ax.plot(lims, lims, 'r--', label='Perfect', linewidth=2)

        # Correlation
        corr = np.corrcoef(tgt_flat, pred_flat)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top')

        ax.set_xlabel('Target')
        ax.set_ylabel('Prediction')
        ax.set_title('Prediction vs Target')
        ax.legend()

    def _plot_error_distribution(self, ax, predictions, targets):
        """Plot error distribution."""
        errors = predictions - targets

        ax.hist(errors.flatten(), bins=100, alpha=0.7, density=True, edgecolor='black')

        # Add statistics
        mean_err = errors.mean()
        std_err = errors.std()
        mae = np.abs(errors).mean()

        ax.axvline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.4f}')
        ax.axvline(0, color='black', linewidth=0.5)

        ax.text(0.95, 0.95, f'MAE: {mae:.4f}\nStd: {std_err:.4f}',
               transform=ax.transAxes, ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()

    def _plot_regime_predictions(self, ax, regimes):
        """Plot regime prediction distribution."""
        regime_labels = ['Low', 'Medium', 'High', 'Offensive']

        # Average probabilities
        avg_probs = regimes.mean(axis=0)

        # Also show predicted class distribution
        predicted = regimes.argmax(axis=1)
        pred_counts = np.bincount(predicted, minlength=4) / len(predicted)

        x = np.arange(len(regime_labels))
        width = 0.35

        ax.bar(x - width/2, avg_probs, width, label='Avg Probability', alpha=0.7)
        ax.bar(x + width/2, pred_counts, width, label='Prediction Freq', alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(regime_labels)
        ax.set_ylabel('Proportion')
        ax.set_title('Regime Classification')
        ax.legend()

    def _plot_anomaly_scores(self, ax, anomalies):
        """Plot anomaly score distribution."""
        ax.hist(anomalies, bins=50, alpha=0.7, edgecolor='black', density=True)

        # Threshold line
        threshold = 0.5
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')

        # Count anomalies
        n_anomalies = (anomalies > threshold).sum()
        pct = n_anomalies / len(anomalies) * 100

        ax.text(0.95, 0.95, f'Anomalies: {n_anomalies} ({pct:.1f}%)',
               transform=ax.transAxes, ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Anomaly Score Distribution')
        ax.legend()

    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        print("\n" + "="*70)
        print("CREATING SUMMARY DASHBOARD")
        print("="*70)

        fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Model overview
        ax1 = fig.add_subplot(gs[0, 0])
        self._summary_model_counts(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._summary_parameter_counts(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self._summary_feature_coverage(ax3)

        ax4 = fig.add_subplot(gs[0, 3])
        self._summary_data_sources(ax4)

        # Row 2: Weight statistics
        ax5 = fig.add_subplot(gs[1, :2])
        self._summary_weight_stats(ax5)

        ax6 = fig.add_subplot(gs[1, 2:])
        self._summary_layer_activations(ax6)

        # Row 3: Learned patterns
        ax7 = fig.add_subplot(gs[2, 0])
        self._summary_embedding_clusters(ax7)

        ax8 = fig.add_subplot(gs[2, 1])
        self._summary_attention_patterns(ax8)

        ax9 = fig.add_subplot(gs[2, 2])
        self._summary_temporal_patterns(ax9)

        ax10 = fig.add_subplot(gs[2, 3])
        self._summary_uncertainty_patterns(ax10)

        # Row 4: Architecture diagram
        ax11 = fig.add_subplot(gs[3, :])
        self._summary_architecture(ax11)

        plt.suptitle('Network Analysis Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {OUTPUT_DIR / 'summary_dashboard.png'}")

    def _summary_model_counts(self, ax):
        """Summary of model counts."""
        counts = {
            'HAN': 1 if self.han_model else 0,
            'Phase 1': len(self.phase1_models),
            'Phase 2': len(self.phase2_models),
            'Phase 3': len(self.phase3_models)
        }

        colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
        bars = ax.bar(list(counts.keys()), list(counts.values()), color=colors)

        for bar, count in zip(bars, counts.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Count')
        ax.set_title('Trained Models')
        ax.set_ylim(0, max(counts.values()) * 1.2)

    def _summary_parameter_counts(self, ax):
        """Summary of parameter counts."""
        param_counts = {}

        if self.han_model:
            param_counts['HAN'] = sum(p.numel() for p in self.han_model.parameters())

        # Phase 3 models
        p3_params = []
        for state in self.phase3_models.values():
            count = sum(v.numel() for v in state.values())
            p3_params.append(count)
        if p3_params:
            param_counts['Phase 3 (avg)'] = np.mean(p3_params)
            param_counts['Phase 3 (total)'] = sum(p3_params)

        if param_counts:
            labels = list(param_counts.keys())
            values = [v / 1000 for v in param_counts.values()]  # Convert to thousands

            bars = ax.bar(labels, values, alpha=0.7)
            ax.set_ylabel('Parameters (thousands)')
            ax.set_title('Parameter Counts')
            ax.tick_params(axis='x', rotation=30)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}K', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No models loaded', ha='center', va='center')

    def _summary_feature_coverage(self, ax):
        """Summary of feature coverage."""
        features_by_source = {}

        for name, state in self.phase3_models.items():
            for src in ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']:
                if src in name:
                    if 'cross_feature_attn.feature_embeddings.weight' in state:
                        n_feat = state['cross_feature_attn.feature_embeddings.weight'].shape[0]
                        features_by_source.setdefault(src, 0)
                        features_by_source[src] += n_feat
                    break

        if features_by_source:
            sources = list(features_by_source.keys())
            values = list(features_by_source.values())

            colors = plt.cm.Pastel1(np.linspace(0, 1, len(sources)))
            wedges, texts, autotexts = ax.pie(values, labels=[s.upper() for s in sources],
                                              autopct='%1.0f%%', colors=colors)
            ax.set_title('Feature Coverage by Source')
        else:
            ax.text(0.5, 0.5, 'No feature data', ha='center', va='center')

    def _summary_data_sources(self, ax):
        """Summary of data sources."""
        # Data source characteristics
        sources = {
            'FIRMS': {'type': 'Daily', 'features': 24},
            'UCDP': {'type': 'Event', 'features': 28},
            'Equipment': {'type': 'Daily', 'features': 22},
            'DeepState': {'type': 'Irregular', 'features': 7},
            'Sentinel': {'type': 'Monthly', 'features': 9}
        }

        names = list(sources.keys())
        features = [s['features'] for s in sources.values()]
        types = [s['type'] for s in sources.values()]

        colors = {'Daily': '#2ecc71', 'Event': '#e74c3c', 'Irregular': '#f39c12', 'Monthly': '#3498db'}
        bar_colors = [colors[t] for t in types]

        bars = ax.barh(names, features, color=bar_colors, alpha=0.7)

        # Add type labels
        for bar, t in zip(bars, types):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   t, va='center', fontsize=9)

        ax.set_xlabel('Features')
        ax.set_title('Data Sources')

    def _summary_weight_stats(self, ax):
        """Summary of weight statistics across models."""
        stats = []

        for name, state in self.phase3_models.items():
            model_weights = []
            for key, tensor in state.items():
                if 'weight' in key:
                    model_weights.extend(tensor.cpu().numpy().flatten())

            if model_weights:
                weights = np.array(model_weights)
                stats.append({
                    'name': name[:15],
                    'mean': np.abs(weights).mean(),
                    'std': weights.std(),
                    'min': weights.min(),
                    'max': weights.max(),
                    'sparsity': (np.abs(weights) < 0.01).mean()
                })

        if stats:
            names = [s['name'] for s in stats]
            means = [s['mean'] for s in stats]
            stds = [s['std'] for s in stats]

            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=2, alpha=0.7)
            ax.set_xticks(x[::3])  # Every 3rd label
            ax.set_xticklabels([names[i] for i in range(0, len(names), 3)], rotation=45, fontsize=8)
            ax.set_ylabel('Mean |Weight|')
            ax.set_title('Weight Statistics Across Phase 3 Models')
        else:
            ax.text(0.5, 0.5, 'No weight data', ha='center', va='center')

    def _summary_layer_activations(self, ax):
        """Summary of layer types and their characteristics."""
        layer_types = {}

        for name, state in self.phase3_models.items():
            for key in state.keys():
                # Extract layer type
                parts = key.split('.')
                if len(parts) >= 2:
                    layer_type = parts[0]
                    layer_types.setdefault(layer_type, 0)
                    layer_types[layer_type] += 1

        if layer_types:
            # Sort by count
            sorted_types = sorted(layer_types.items(), key=lambda x: -x[1])[:10]

            names = [t[0] for t in sorted_types]
            counts = [t[1] for t in sorted_types]

            ax.barh(names, counts, alpha=0.7)
            ax.set_xlabel('Parameter Count')
            ax.set_title('Layer Types in Phase 3 Models')
        else:
            ax.text(0.5, 0.5, 'No layer data', ha='center', va='center')

    def _summary_embedding_clusters(self, ax):
        """Show embedding space structure."""
        all_embeddings = []
        labels = []

        for name, state in self.phase3_models.items():
            if 'cross_feature_attn.feature_embeddings.weight' in state:
                emb = state['cross_feature_attn.feature_embeddings.weight'].cpu().numpy()
                all_embeddings.append(emb.mean(axis=0))  # Mean embedding per model

                for src in ['ucdp', 'firms', 'equipment', 'deepstate', 'sentinel']:
                    if src in name:
                        labels.append(src)
                        break
                else:
                    labels.append('other')

        if len(all_embeddings) >= 2:
            from sklearn.decomposition import PCA

            data = np.vstack(all_embeddings)
            pca = PCA(n_components=2)
            projected = pca.fit_transform(data)

            # Color by source
            source_colors = {'ucdp': 'red', 'firms': 'orange', 'equipment': 'blue',
                           'deepstate': 'green', 'sentinel': 'purple', 'other': 'gray'}
            colors = [source_colors.get(l, 'gray') for l in labels]

            ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=50, alpha=0.7)

            # Legend
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=c, markersize=8, label=l.upper())
                      for l, c in source_colors.items() if l in labels]
            ax.legend(handles=handles, fontsize=8)

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Model Embedding Space')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    def _summary_attention_patterns(self, ax):
        """Attention pattern summary."""
        patterns = []

        for name, state in self.phase3_models.items():
            for key in state:
                if 'self_attn.in_proj_weight' in key:
                    w = state[key].cpu().numpy()
                    patterns.append(np.abs(w).mean())
                    break

        if patterns:
            ax.hist(patterns, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(patterns), color='red', linestyle='--',
                      label=f'Mean: {np.mean(patterns):.3f}')
            ax.set_xlabel('Mean |Attention Weight|')
            ax.set_ylabel('Count')
            ax.set_title('Attention Weight Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No attention data', ha='center', va='center')

    def _summary_temporal_patterns(self, ax):
        """Temporal encoding pattern summary."""
        day_embs = []

        for name, state in self.phase3_models.items():
            if 'temporal_encoding.day_embedding.weight' in state:
                emb = state['temporal_encoding.day_embedding.weight'].cpu().numpy()
                day_embs.append(emb[:30].mean(axis=1))  # First 30 days

        if day_embs:
            data = np.vstack(day_embs)

            # Plot mean and std
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            ax.fill_between(range(30), mean - std, mean + std, alpha=0.3)
            ax.plot(range(30), mean, 'b-', linewidth=2)

            ax.set_xlabel('Day')
            ax.set_ylabel('Embedding Value')
            ax.set_title('Day Embedding Pattern')
        else:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center')

    def _summary_uncertainty_patterns(self, ax):
        """Uncertainty estimation summary."""
        biases = []

        for name, state in self.phase3_models.items():
            if 'uncertainty_head.2.bias' in state:
                b = state['uncertainty_head.2.bias'].cpu().numpy().item()
                biases.append(b)

        if biases:
            ax.hist(biases, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(biases), color='red', linestyle='--',
                      label=f'Mean: {np.mean(biases):.3f}')
            ax.set_xlabel('Output Bias')
            ax.set_ylabel('Count')
            ax.set_title('Uncertainty Head Bias')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No uncertainty data', ha='center', va='center')

    def _summary_architecture(self, ax):
        """Draw architecture summary."""
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.axis('off')

        # Draw components
        components = [
            ('Data Sources', 5, 10, '#3498db'),
            ('Feature Extraction\n(90 features)', 25, 10, '#2ecc71'),
            ('Phase 1-3\nInterpolation\n(29 models)', 45, 10, '#e74c3c'),
            ('HAN\nCross-Domain', 65, 10, '#9b59b6'),
            ('Outputs\n(Forecast/Regime/\nAnomaly)', 85, 10, '#f39c12')
        ]

        for label, x, y, color in components:
            rect = plt.Rectangle((x-8, y-4), 16, 8, facecolor=color, alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # Draw arrows
        for i in range(len(components) - 1):
            ax.annotate('', xy=(components[i+1][1]-8, 10), xytext=(components[i][1]+8, 10),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

        ax.set_title('Network Architecture Overview', fontsize=12, fontweight='bold', pad=20)

    def run_full_analysis(self):
        """Run all analyses and generate figures."""
        print("="*70)
        print("COMPREHENSIVE NETWORK ANALYSIS")
        print("="*70)
        print(f"Output directory: {OUTPUT_DIR}")

        self.load_models()

        # Run all analyses
        self.analyze_han_attention()
        self.analyze_feature_embeddings()
        self.analyze_temporal_patterns()
        self.analyze_cross_feature_attention()
        self.analyze_uncertainty_learning()
        self.analyze_model_predictions()
        self.create_summary_dashboard()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nGenerated figures in: {OUTPUT_DIR}")
        for f in sorted(OUTPUT_DIR.glob("*.png")):
            print(f"  - {f.name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Network Analysis and Visualization')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/mps)')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['all', 'han', 'embeddings', 'temporal', 'attention',
                               'uncertainty', 'predictions', 'dashboard'],
                       help='Which analysis to run')
    args = parser.parse_args()

    analyzer = NetworkAnalyzer(device=args.device)

    if args.analysis == 'all':
        analyzer.run_full_analysis()
    else:
        analyzer.load_models()

        if args.analysis == 'han':
            analyzer.analyze_han_attention()
        elif args.analysis == 'embeddings':
            analyzer.analyze_feature_embeddings()
        elif args.analysis == 'temporal':
            analyzer.analyze_temporal_patterns()
        elif args.analysis == 'attention':
            analyzer.analyze_cross_feature_attention()
        elif args.analysis == 'uncertainty':
            analyzer.analyze_uncertainty_learning()
        elif args.analysis == 'predictions':
            analyzer.analyze_model_predictions()
        elif args.analysis == 'dashboard':
            analyzer.create_summary_dashboard()


if __name__ == "__main__":
    main()
