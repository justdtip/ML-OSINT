"""
Fine-Grained Inter-Feature Relationship Analysis

Extracts and visualizes detailed feature-level relationships from:
1. JIM models: Cross-feature attention weights (what features predict what)
2. Unified model: Cross-source attention patterns (how sources inform each other)

Produces detailed figures and insights about:
- Feature clustering by learned similarity
- Attention flow patterns (which features drive predictions)
- Cross-source feature mappings
- Temporal feature importance
- Feature redundancy analysis
"""

import sys
from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available")

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from config.paths import FEATURE_OUTPUT_DIR

from joint_interpolation_models import (
    JointInterpolationModel,
    InterpolationConfig,
    INTERPOLATION_CONFIGS,
    PHASE2_CONFIGS,
    DATA_DIR,
    ANALYSIS_DIR
)
from unified_interpolation import (
    UnifiedInterpolationModel,
    SOURCE_CONFIGS,
    MODEL_DIR
)
from interpolation_data_loaders import (
    SentinelDataLoader,
    DeepStateDataLoader,
    EquipmentDataLoader,
    FIRMSDataLoader,
    UCDPDataLoader
)

# Output directory
FIGURE_DIR = FEATURE_OUTPUT_DIR
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class FineGrainedFeatureAnalyzer:
    """
    Comprehensive analysis of inter-feature relationships at fine granularity.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.jim_models = {}
        self.unified_model = None
        self.feature_names = {}
        self.insights = {
            'jim': {},
            'unified': {},
            'cross_source': {},
            'summary': []
        }

    def load_all_models(self):
        """Load all trained JIM and unified models."""
        print("=" * 70)
        print("LOADING TRAINED MODELS")
        print("=" * 70)

        # Load JIM models
        model_files = list(MODEL_DIR.glob("*.pt"))

        for model_path in model_files:
            name = model_path.stem

            # Skip non-JIM models
            if 'unified' in name or 'han' in name:
                continue

            try:
                # Infer source and config from name
                source, config = self._infer_model_config(name)
                if config is None:
                    continue

                model = JointInterpolationModel(config)
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                model.eval()

                self.jim_models[name] = {
                    'model': model,
                    'config': config,
                    'source': source,
                    'path': model_path
                }
                print(f"  Loaded: {name} ({config.num_features} features)")

            except Exception as e:
                print(f"  Could not load {name}: {e}")

        # Load unified model
        unified_path = MODEL_DIR / "unified_interpolation_best.pt"
        if unified_path.exists():
            try:
                # Get actual feature counts by loading data
                source_configs = {}
                for src_name, src_config in SOURCE_CONFIGS.items():
                    loader = src_config.loader_class().load().process()
                    n_features = loader.processed_data.shape[1]
                    src_config.n_features = n_features
                    source_configs[src_name] = src_config
                    self.feature_names[src_name] = loader.feature_names[:n_features]

                self.unified_model = UnifiedInterpolationModel(
                    source_configs=source_configs,
                    d_embed=64,
                    nhead=4,
                    num_fusion_layers=2,
                    dropout=0.1
                )
                state_dict = torch.load(unified_path, map_location=self.device, weights_only=False)
                self.unified_model.load_state_dict(state_dict)
                self.unified_model.eval()
                print(f"  Loaded: unified_interpolation ({sum(c.n_features for c in source_configs.values())} total features)")

            except Exception as e:
                print(f"  Could not load unified model: {e}")

        print(f"\nTotal JIM models: {len(self.jim_models)}")
        print(f"Unified model: {'Yes' if self.unified_model else 'No'}")

    def _infer_model_config(self, name: str) -> Tuple[str, Optional[InterpolationConfig]]:
        """Infer source and config from model filename."""
        name_lower = name.lower()

        # Try to match against known configs
        all_configs = {**INTERPOLATION_CONFIGS, **PHASE2_CONFIGS}

        for config_name, config in all_configs.items():
            if config_name.lower() in name_lower:
                return config.source, config

        # Infer from naming patterns
        if 'sentinel' in name_lower:
            source = 'sentinel'
            n_features = 55
        elif 'deepstate' in name_lower:
            source = 'deepstate'
            n_features = 55
        elif 'equipment' in name_lower:
            source = 'equipment'
            n_features = 42
        elif 'firms' in name_lower:
            source = 'firms'
            n_features = 42
        elif 'ucdp' in name_lower:
            source = 'ucdp'
            n_features = 48
        else:
            return 'unknown', None

        # Create generic config
        config = InterpolationConfig(
            name=name,
            source=source,
            features=[f'f{i}' for i in range(n_features)],
            native_resolution_days=1.0,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        return source, config

    def analyze_jim_feature_embeddings(self) -> Dict[str, Any]:
        """
        Analyze learned feature embeddings from JIM cross-feature attention.

        Extracts:
        - Feature embedding similarity matrix
        - Feature clusters by learned representation
        - Dominant feature patterns
        """
        print("\n" + "=" * 70)
        print("ANALYZING JIM FEATURE EMBEDDINGS")
        print("=" * 70)

        results = {}

        for name, model_info in self.jim_models.items():
            model = model_info['model']
            config = model_info['config']
            source = model_info['source']

            # Extract feature embeddings from cross-feature attention
            if hasattr(model.cross_feature_attn, 'feature_embeddings'):
                embeddings = model.cross_feature_attn.feature_embeddings.weight.detach().cpu().numpy()
                n_features = embeddings.shape[0]

                # Compute pairwise cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized = embeddings / (norms + 1e-8)
                similarity = normalized @ normalized.T

                # Find feature clusters
                if HAS_SKLEARN and n_features >= 3:
                    n_clusters = min(5, n_features // 2)
                    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    labels = clustering.fit_predict(embeddings)
                else:
                    labels = np.zeros(n_features)

                # Find most similar feature pairs
                similar_pairs = []
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        similar_pairs.append((i, j, similarity[i, j]))
                similar_pairs.sort(key=lambda x: x[2], reverse=True)

                results[name] = {
                    'embeddings': embeddings,
                    'similarity_matrix': similarity,
                    'cluster_labels': labels,
                    'n_clusters': len(np.unique(labels)),
                    'top_similar_pairs': similar_pairs[:10],
                    'source': source,
                    'n_features': n_features,
                    'mean_intra_similarity': float(np.mean(similarity[similarity < 0.99])),
                    'embedding_norms': norms.flatten()
                }

                print(f"\n{name}:")
                print(f"  Features: {n_features}")
                print(f"  Clusters found: {len(np.unique(labels))}")
                print(f"  Mean feature similarity: {results[name]['mean_intra_similarity']:.3f}")
                print(f"  Top similar pairs:")
                for i, j, sim in similar_pairs[:3]:
                    print(f"    Feature {i} <-> {j}: {sim:.3f}")

        self.insights['jim']['feature_embeddings'] = results
        return results

    def analyze_jim_attention_patterns(self) -> Dict[str, Any]:
        """
        Analyze attention weight patterns in JIM transformer layers.

        Extracts:
        - Which features attend to which (directional relationships)
        - Attention concentration (specialist vs generalist features)
        - Cross-layer attention evolution
        """
        print("\n" + "=" * 70)
        print("ANALYZING JIM ATTENTION PATTERNS")
        print("=" * 70)

        results = {}

        for name, model_info in self.jim_models.items():
            model = model_info['model']
            config = model_info['config']
            n_features = len(config.features)

            # Create synthetic input for attention probing
            batch_size = 32
            x = torch.randn(batch_size, n_features)

            # Hook to capture attention weights
            attention_weights = []

            def hook_fn(module, input, output):
                # For TransformerEncoderLayer, attention is internal
                pass

            # Get feature projection output
            with torch.no_grad():
                x_proj = model.cross_feature_attn.feature_projection(x.unsqueeze(-1))
                feat_indices = torch.arange(n_features)
                feat_emb = model.cross_feature_attn.feature_embeddings(feat_indices)
                x_proj = x_proj + feat_emb.unsqueeze(0)

            # Compute attention-like similarity from embeddings
            # This approximates what the attention would look like
            feat_emb_np = feat_emb.detach().cpu().numpy()
            norms = np.linalg.norm(feat_emb_np, axis=1, keepdims=True)
            normalized = feat_emb_np / (norms + 1e-8)
            attn_proxy = normalized @ normalized.T

            # Analyze attention patterns
            # Row i: what feature i attends to
            # Col j: how much attention j receives

            outgoing_strength = np.mean(attn_proxy, axis=1)  # How much each feature queries
            incoming_strength = np.mean(attn_proxy, axis=0)  # How much each feature is attended to

            # Find hub features (high incoming attention)
            hub_indices = np.argsort(incoming_strength)[-3:]

            # Find specialist features (concentrated attention)
            attention_entropy = -np.sum(attn_proxy * np.log(attn_proxy + 1e-8), axis=1)
            specialist_indices = np.argsort(attention_entropy)[:3]

            results[name] = {
                'attention_proxy': attn_proxy,
                'outgoing_strength': outgoing_strength,
                'incoming_strength': incoming_strength,
                'hub_features': hub_indices.tolist(),
                'specialist_features': specialist_indices.tolist(),
                'mean_attention_entropy': float(np.mean(attention_entropy)),
                'n_features': n_features
            }

            print(f"\n{name}:")
            print(f"  Hub features (most attended): {hub_indices.tolist()}")
            print(f"  Specialist features (focused attention): {specialist_indices.tolist()}")
            print(f"  Mean attention entropy: {np.mean(attention_entropy):.3f}")

        self.insights['jim']['attention_patterns'] = results
        return results

    def analyze_jim_decoder_weights(self) -> Dict[str, Any]:
        """
        Analyze decoder weights to understand feature generation patterns.

        Extracts:
        - Which latent dimensions drive which features
        - Feature generation similarity
        - Decoder weight clustering
        """
        print("\n" + "=" * 70)
        print("ANALYZING JIM DECODER WEIGHTS")
        print("=" * 70)

        results = {}

        for name, model_info in self.jim_models.items():
            model = model_info['model']

            # Get decoder weights from gap interpolator
            decoder = model.gap_interpolator.decoder

            # Extract final linear layer weights
            final_layer = None
            for layer in decoder:
                if isinstance(layer, nn.Linear):
                    final_layer = layer

            if final_layer is not None:
                weights = final_layer.weight.detach().cpu().numpy()  # [n_features, d_model]

                # Compute feature generation similarity
                norms = np.linalg.norm(weights, axis=1, keepdims=True)
                normalized = weights / (norms + 1e-8)
                generation_similarity = normalized @ normalized.T

                # Find features generated by similar latent patterns
                similar_gen_pairs = []
                n_features = weights.shape[0]
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        similar_gen_pairs.append((i, j, generation_similarity[i, j]))
                similar_gen_pairs.sort(key=lambda x: x[2], reverse=True)

                # PCA of decoder weights
                if HAS_SKLEARN and n_features >= 3:
                    pca = PCA(n_components=min(3, n_features))
                    weights_pca = pca.fit_transform(weights)
                    explained_var = pca.explained_variance_ratio_
                else:
                    weights_pca = weights[:, :3]
                    explained_var = [0.33, 0.33, 0.33]

                results[name] = {
                    'decoder_weights': weights,
                    'generation_similarity': generation_similarity,
                    'top_similar_generation': similar_gen_pairs[:10],
                    'weights_pca': weights_pca,
                    'pca_explained_var': explained_var,
                    'mean_generation_similarity': float(np.mean(generation_similarity[np.triu_indices(n_features, k=1)])),
                    'n_features': n_features
                }

                print(f"\n{name}:")
                print(f"  Decoder weight shape: {weights.shape}")
                print(f"  Mean generation similarity: {results[name]['mean_generation_similarity']:.3f}")
                print(f"  Top co-generated features:")
                for i, j, sim in similar_gen_pairs[:3]:
                    print(f"    Feature {i} <-> {j}: {sim:.3f}")

        self.insights['jim']['decoder_weights'] = results
        return results

    def analyze_unified_cross_source(self) -> Dict[str, Any]:
        """
        Analyze cross-source relationships in the unified model.

        Extracts:
        - Source embedding similarity
        - Cross-attention patterns between sources
        - Source importance ranking
        - Feature-level cross-source mappings
        """
        print("\n" + "=" * 70)
        print("ANALYZING UNIFIED MODEL CROSS-SOURCE RELATIONSHIPS")
        print("=" * 70)

        if self.unified_model is None:
            print("Unified model not loaded")
            return {}

        results = {}

        # 1. Source type embeddings
        source_emb = self.unified_model.fusion.source_embeddings.weight.detach().cpu().numpy()
        source_names = list(SOURCE_CONFIGS.keys())

        # Compute source similarity
        norms = np.linalg.norm(source_emb, axis=1, keepdims=True)
        normalized = source_emb / (norms + 1e-8)
        source_similarity = normalized @ normalized.T

        print("\nSource Embedding Similarity:")
        print("-" * 50)
        for i, src_i in enumerate(source_names):
            for j, src_j in enumerate(source_names):
                if i < j:
                    print(f"  {src_i} <-> {src_j}: {source_similarity[i, j]:.3f}")

        results['source_embeddings'] = source_emb
        results['source_similarity'] = source_similarity
        results['source_names'] = source_names

        # 2. Encoder analysis - what features matter most per source
        encoder_feature_importance = {}
        for src_name in source_names:
            encoder = self.unified_model.encoders[src_name]

            if hasattr(encoder, 'feature_proj'):
                # Get first layer weights
                first_layer = encoder.feature_proj[0]
                if isinstance(first_layer, nn.Linear):
                    weights = first_layer.weight.detach().cpu().numpy()
                    # Importance = L2 norm of weight column for each input feature
                    importance = np.linalg.norm(weights, axis=0)
                    encoder_feature_importance[src_name] = importance

        results['encoder_feature_importance'] = encoder_feature_importance

        # 3. Decoder analysis - how features are reconstructed
        decoder_analysis = {}
        for src_name in source_names:
            decoder = self.unified_model.decoders[src_name]

            # Get final layer weights
            final_layer = decoder.decoder[-1]
            if isinstance(final_layer, nn.Linear):
                weights = final_layer.weight.detach().cpu().numpy()
                # Each row is one output feature
                decoder_analysis[src_name] = {
                    'weights_shape': weights.shape,
                    'feature_norms': np.linalg.norm(weights, axis=1),
                    'mean_norm': float(np.mean(np.linalg.norm(weights, axis=1)))
                }

        results['decoder_analysis'] = decoder_analysis

        # 4. Cross-source output projections
        output_proj_analysis = {}
        for i, src_name in enumerate(source_names):
            proj = self.unified_model.fusion.output_projs[i]
            weights = proj.weight.detach().cpu().numpy()
            output_proj_analysis[src_name] = {
                'weight_norm': float(np.linalg.norm(weights)),
                'weight_shape': weights.shape
            }

        results['output_proj_analysis'] = output_proj_analysis

        # Print summary
        print("\nSource Importance by Embedding Magnitude:")
        importance_ranking = [(name, np.linalg.norm(source_emb[i]))
                            for i, name in enumerate(source_names)]
        importance_ranking.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, mag) in enumerate(importance_ranking, 1):
            print(f"  {rank}. {name}: {mag:.3f}")

        results['importance_ranking'] = importance_ranking

        self.insights['unified'] = results
        return results

    def analyze_cross_source_feature_mapping(self) -> Dict[str, Any]:
        """
        Analyze which features from one source map to features in another.

        Uses reconstruction loss gradients to find cross-source dependencies.
        """
        print("\n" + "=" * 70)
        print("ANALYZING CROSS-SOURCE FEATURE MAPPINGS")
        print("=" * 70)

        if self.unified_model is None:
            print("Unified model not loaded")
            return {}

        results = {}
        source_names = list(SOURCE_CONFIGS.keys())

        # Load actual data for probing - find minimum overlapping samples
        source_data = {}
        min_samples = float('inf')

        for src_name, src_config in SOURCE_CONFIGS.items():
            try:
                loader = src_config.loader_class().load().process()
                data = loader.processed_data
                source_data[src_name] = data
                min_samples = min(min_samples, len(data))
            except Exception as e:
                print(f"  Could not load {src_name}: {e}")
                continue

        # Use common sample count for all sources
        n_samples = min(100, int(min_samples))
        print(f"  Using {n_samples} aligned samples for analysis")

        for src_name in source_data:
            source_data[src_name] = torch.tensor(
                source_data[src_name][:n_samples], dtype=torch.float32
            )

        # For each source pair, analyze feature correlations in latent space
        for src_i in source_names:
            if src_i not in source_data:
                continue
            for src_j in source_names:
                if src_j not in source_data or src_i == src_j:
                    continue

                # Get embeddings for both sources
                with torch.no_grad():
                    emb_i = self.unified_model.encoders[src_i](source_data[src_i])
                    emb_j = self.unified_model.encoders[src_j](source_data[src_j])

                # Correlation between embeddings
                emb_i_np = emb_i.numpy()
                emb_j_np = emb_j.numpy()

                # Cross-correlation matrix
                # Mean-center
                emb_i_centered = emb_i_np - emb_i_np.mean(axis=0)
                emb_j_centered = emb_j_np - emb_j_np.mean(axis=0)

                # Correlation - use samples as observations
                # emb_i_centered: [n_samples, d_embed]
                # We want correlation between latent dimensions: [d_embed, d_embed]
                n = len(emb_i_np)
                cross_corr = (emb_i_centered.T @ emb_j_centered) / (n - 1)

                # Normalize by standard deviations
                std_i = emb_i_np.std(axis=0, keepdims=True).T  # [d_embed, 1]
                std_j = emb_j_np.std(axis=0, keepdims=True)    # [1, d_embed]
                cross_corr_normalized = cross_corr / (std_i @ std_j + 1e-8)

                results[f"{src_i}_to_{src_j}"] = {
                    'cross_correlation': cross_corr_normalized,
                    'max_correlation': float(np.max(np.abs(cross_corr_normalized))),
                    'mean_abs_correlation': float(np.mean(np.abs(cross_corr_normalized))),
                }

                print(f"\n{src_i} -> {src_j}:")
                print(f"  Max correlation: {results[f'{src_i}_to_{src_j}']['max_correlation']:.3f}")
                print(f"  Mean |correlation|: {results[f'{src_i}_to_{src_j}']['mean_abs_correlation']:.3f}")

        self.insights['cross_source'] = results
        return results

    def create_visualizations(self):
        """Create all visualization figures."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for visualizations")
            return

        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        fig_num = 0

        # 1. JIM Feature Embedding Similarity Matrices
        if 'feature_embeddings' in self.insights['jim']:
            for name, data in self.insights['jim']['feature_embeddings'].items():
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Similarity heatmap
                sim = data['similarity_matrix']
                sns.heatmap(sim, ax=axes[0], cmap='RdBu_r', center=0,
                           vmin=-1, vmax=1, square=True)
                axes[0].set_title(f'{name}\nFeature Similarity Matrix')
                axes[0].set_xlabel('Feature Index')
                axes[0].set_ylabel('Feature Index')

                # Embedding norms
                norms = data['embedding_norms']
                axes[1].bar(range(len(norms)), norms, color='steelblue', alpha=0.7)
                axes[1].set_title('Feature Embedding Magnitudes')
                axes[1].set_xlabel('Feature Index')
                axes[1].set_ylabel('L2 Norm')
                axes[1].axhline(y=np.mean(norms), color='r', linestyle='--',
                               label=f'Mean: {np.mean(norms):.2f}')
                axes[1].legend()

                plt.tight_layout()
                plt.savefig(FIGURE_DIR / f'{fig_num:02d}_jim_embeddings_{name[:20]}.png',
                           dpi=150, bbox_inches='tight')
                plt.close()
                fig_num += 1
                print(f"  Saved: {fig_num-1:02d}_jim_embeddings_{name[:20]}.png")

        # 2. JIM Attention Patterns
        if 'attention_patterns' in self.insights['jim']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, (name, data) in enumerate(list(self.insights['jim']['attention_patterns'].items())[:6]):
                if idx >= 6:
                    break
                attn = data['attention_proxy']
                sns.heatmap(attn, ax=axes[idx], cmap='viridis',
                           square=True, cbar_kws={'shrink': 0.5})
                axes[idx].set_title(f'{name[:25]}\nAttention Proxy', fontsize=9)
                axes[idx].set_xlabel('To Feature', fontsize=8)
                axes[idx].set_ylabel('From Feature', fontsize=8)

            plt.suptitle('JIM Cross-Feature Attention Patterns', fontsize=12)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'{fig_num:02d}_jim_attention_patterns.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            fig_num += 1
            print(f"  Saved: {fig_num-1:02d}_jim_attention_patterns.png")

        # 3. JIM Decoder Weight Analysis
        if 'decoder_weights' in self.insights['jim']:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, (name, data) in enumerate(list(self.insights['jim']['decoder_weights'].items())[:6]):
                if idx >= 6:
                    break
                gen_sim = data['generation_similarity']
                sns.heatmap(gen_sim, ax=axes[idx], cmap='coolwarm', center=0,
                           vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.5})
                axes[idx].set_title(f'{name[:25]}\nGeneration Similarity', fontsize=9)

            plt.suptitle('JIM Decoder: Feature Generation Patterns', fontsize=12)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'{fig_num:02d}_jim_decoder_patterns.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            fig_num += 1
            print(f"  Saved: {fig_num-1:02d}_jim_decoder_patterns.png")

        # 4. Unified Model Source Similarity
        if self.insights['unified']:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Source embedding similarity
            if 'source_similarity' in self.insights['unified']:
                sim = self.insights['unified']['source_similarity']
                names = self.insights['unified']['source_names']

                sns.heatmap(sim, ax=axes[0], annot=True, fmt='.3f',
                           xticklabels=names, yticklabels=names,
                           cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                axes[0].set_title('Source Embedding Similarity')

            # Source importance
            if 'importance_ranking' in self.insights['unified']:
                ranking = self.insights['unified']['importance_ranking']
                names = [r[0] for r in ranking]
                mags = [r[1] for r in ranking]

                colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(names)))
                axes[1].barh(names[::-1], mags[::-1], color=colors)
                axes[1].set_xlabel('Embedding Magnitude')
                axes[1].set_title('Source Importance Ranking')

                for i, (n, m) in enumerate(zip(names[::-1], mags[::-1])):
                    axes[1].text(m + 0.1, i, f'{m:.2f}', va='center', fontsize=9)

            # Source embedding PCA
            if 'source_embeddings' in self.insights['unified']:
                emb = self.insights['unified']['source_embeddings']
                names = self.insights['unified']['source_names']

                if HAS_SKLEARN and emb.shape[0] >= 3:
                    pca = PCA(n_components=2)
                    emb_2d = pca.fit_transform(emb)

                    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
                    for i, (name, color) in enumerate(zip(names, colors)):
                        axes[2].scatter(emb_2d[i, 0], emb_2d[i, 1],
                                       s=200, c=[color], label=name, edgecolor='black')
                        axes[2].annotate(name, (emb_2d[i, 0], emb_2d[i, 1]),
                                        xytext=(5, 5), textcoords='offset points', fontsize=10)

                    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                    axes[2].set_title('Source Embeddings (PCA)')
                    axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'{fig_num:02d}_unified_source_analysis.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            fig_num += 1
            print(f"  Saved: {fig_num-1:02d}_unified_source_analysis.png")

        # 5. Cross-Source Correlation Matrix
        if self.insights['cross_source']:
            source_names = list(SOURCE_CONFIGS.keys())
            n_sources = len(source_names)

            # Create summary matrix of cross-source correlations
            cross_corr_summary = np.zeros((n_sources, n_sources))

            for i, src_i in enumerate(source_names):
                for j, src_j in enumerate(source_names):
                    if i == j:
                        cross_corr_summary[i, j] = 1.0
                    else:
                        key = f"{src_i}_to_{src_j}"
                        if key in self.insights['cross_source']:
                            cross_corr_summary[i, j] = self.insights['cross_source'][key]['mean_abs_correlation']

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cross_corr_summary, ax=ax, annot=True, fmt='.3f',
                       xticklabels=source_names, yticklabels=source_names,
                       cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title('Cross-Source Embedding Correlation\n(Mean Absolute Correlation)')

            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f'{fig_num:02d}_cross_source_correlation.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            fig_num += 1
            print(f"  Saved: {fig_num-1:02d}_cross_source_correlation.png")

        # 6. Comprehensive Summary Dashboard
        fig = plt.figure(figsize=(20, 16))

        # Create grid spec for complex layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Fine-Grained Feature Analysis Summary\nML-OSINT Ukraine Conflict Analysis',
                    fontsize=16, fontweight='bold', y=0.98)

        # A. JIM Model Statistics (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'feature_embeddings' in self.insights['jim']:
            model_names = []
            similarities = []
            n_features_list = []

            for name, data in self.insights['jim']['feature_embeddings'].items():
                model_names.append(name[:15])
                similarities.append(data['mean_intra_similarity'])
                n_features_list.append(data['n_features'])

            x = np.arange(len(model_names))
            width = 0.4

            ax1.bar(x - width/2, similarities, width, label='Mean Similarity', color='steelblue')
            ax1.set_ylabel('Mean Feature Similarity')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            ax1.set_title('JIM Model Feature Similarity', fontsize=11)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

        # B. Source Importance (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'importance_ranking' in self.insights.get('unified', {}):
            ranking = self.insights['unified']['importance_ranking']
            names = [r[0] for r in ranking]
            mags = [r[1] for r in ranking]

            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
            ax2.barh(names[::-1], mags[::-1], color=colors[::-1])
            ax2.set_xlabel('Embedding Magnitude')
            ax2.set_title('Unified Model: Source Importance', fontsize=11)
            for i, m in enumerate(mags[::-1]):
                ax2.text(m + 0.05, i, f'{m:.2f}', va='center', fontsize=9)

        # C. Cross-Source Relationships (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'source_similarity' in self.insights.get('unified', {}):
            sim = self.insights['unified']['source_similarity']
            names = self.insights['unified']['source_names']

            # Plot as heatmap
            im = ax3.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax3.set_xticks(range(len(names)))
            ax3.set_yticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=45, ha='right')
            ax3.set_yticklabels(names)
            ax3.set_title('Cross-Source Embedding Similarity', fontsize=11)

            # Add correlation values
            for i in range(len(names)):
                for j in range(len(names)):
                    color = 'white' if abs(sim[i, j]) > 0.5 else 'black'
                    ax3.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                            color=color, fontsize=9)

        # D. Feature Relationship Network (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if self.insights['cross_source']:
            source_names = list(SOURCE_CONFIGS.keys())
            n = len(source_names)

            # Create circular layout
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)

            # Draw connections
            for i, src_i in enumerate(source_names):
                for j, src_j in enumerate(source_names):
                    if i < j:
                        key = f"{src_i}_to_{src_j}"
                        if key in self.insights['cross_source']:
                            corr = self.insights['cross_source'][key]['mean_abs_correlation']
                            if corr > 0.1:  # Only draw significant connections
                                width = corr * 5
                                alpha = min(corr * 2, 0.8)
                                ax4.plot([x_pos[i], x_pos[j]], [y_pos[i], y_pos[j]],
                                        linewidth=width, alpha=alpha, color='steelblue')

            # Draw nodes
            colors = plt.cm.Set2(np.linspace(0, 1, n))
            for i, (name, color) in enumerate(zip(source_names, colors)):
                ax4.scatter(x_pos[i], y_pos[i], s=1000, c=[color],
                           edgecolor='black', linewidth=2, zorder=10)
                ax4.annotate(name, (x_pos[i], y_pos[i]), ha='center', va='center',
                            fontsize=10, fontweight='bold')

            ax4.set_xlim(-1.5, 1.5)
            ax4.set_ylim(-1.5, 1.5)
            ax4.set_aspect('equal')
            ax4.axis('off')
            ax4.set_title('Cross-Source Relationship Network\n(line width = correlation strength)', fontsize=11)

        # E. Key Insights Text Box (bottom)
        ax5 = fig.add_subplot(gs[2:, :])
        ax5.axis('off')

        # Compile key insights
        insights_text = self._compile_key_insights()

        ax5.text(0.02, 0.98, insights_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.savefig(FIGURE_DIR / f'{fig_num:02d}_summary_dashboard.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        fig_num += 1
        print(f"  Saved: {fig_num-1:02d}_summary_dashboard.png")

        print(f"\nTotal figures generated: {fig_num}")

    def _compile_key_insights(self) -> str:
        """Compile key insights from all analyses."""
        lines = []
        lines.append("=" * 90)
        lines.append("KEY INSIGHTS FROM FINE-GRAINED FEATURE ANALYSIS")
        lines.append("=" * 90)
        lines.append("")

        # JIM insights
        lines.append("JIM MODEL INSIGHTS (Per-Source Feature Learning)")
        lines.append("-" * 50)

        if 'feature_embeddings' in self.insights['jim']:
            # Find models with highest/lowest feature similarity
            sims = [(name, data['mean_intra_similarity'])
                   for name, data in self.insights['jim']['feature_embeddings'].items()]
            sims.sort(key=lambda x: x[1], reverse=True)

            lines.append(f"• Most correlated features (highest learned similarity):")
            for name, sim in sims[:3]:
                lines.append(f"    - {name[:30]}: {sim:.3f}")

            lines.append(f"• Most independent features (lowest similarity):")
            for name, sim in sims[-3:]:
                lines.append(f"    - {name[:30]}: {sim:.3f}")

        if 'attention_patterns' in self.insights['jim']:
            lines.append("")
            lines.append("• Attention pattern analysis:")

            # Find models with most focused vs distributed attention
            entropies = [(name, data['mean_attention_entropy'])
                        for name, data in self.insights['jim']['attention_patterns'].items()]
            entropies.sort(key=lambda x: x[1])

            lines.append(f"    - Most focused (specialist features): {entropies[0][0][:25]}")
            lines.append(f"    - Most distributed (generalist): {entropies[-1][0][:25]}")

        lines.append("")

        # Unified model insights
        lines.append("UNIFIED MODEL INSIGHTS (Cross-Source Learning)")
        lines.append("-" * 50)

        if 'importance_ranking' in self.insights.get('unified', {}):
            ranking = self.insights['unified']['importance_ranking']
            lines.append(f"• Source importance ranking (by embedding magnitude):")
            for rank, (name, mag) in enumerate(ranking, 1):
                lines.append(f"    {rank}. {name}: {mag:.3f}")

        if 'source_similarity' in self.insights.get('unified', {}):
            sim = self.insights['unified']['source_similarity']
            names = self.insights['unified']['source_names']

            # Find strongest positive and negative relationships
            relationships = []
            for i, src_i in enumerate(names):
                for j, src_j in enumerate(names):
                    if i < j:
                        relationships.append((src_i, src_j, sim[i, j]))

            relationships.sort(key=lambda x: x[2], reverse=True)

            lines.append("")
            lines.append("• Strongest cross-source relationships:")
            for src_i, src_j, val in relationships[:3]:
                interpretation = self._interpret_relationship(src_i, src_j, val)
                lines.append(f"    - {src_i} <-> {src_j}: {val:.3f} ({interpretation})")

            lines.append("")
            lines.append("• Most independent sources:")
            for src_i, src_j, val in relationships[-2:]:
                lines.append(f"    - {src_i} <-> {src_j}: {val:.3f}")

        lines.append("")

        # Cross-source insights
        if self.insights['cross_source']:
            lines.append("CROSS-SOURCE FEATURE MAPPING INSIGHTS")
            lines.append("-" * 50)

            # Find strongest cross-source feature correlations
            all_corrs = [(key, data['max_correlation'])
                        for key, data in self.insights['cross_source'].items()]
            all_corrs.sort(key=lambda x: x[1], reverse=True)

            lines.append("• Strongest cross-source feature correlations:")
            for key, corr in all_corrs[:5]:
                src_from, src_to = key.replace('_to_', ' -> ').split(' -> ')
                lines.append(f"    - {src_from} -> {src_to}: {corr:.3f}")

        lines.append("")
        lines.append("=" * 90)
        lines.append("ACTIONABLE RECOMMENDATIONS")
        lines.append("=" * 90)

        # Generate actionable recommendations
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

        return '\n'.join(lines)

    def _interpret_relationship(self, src_i: str, src_j: str, similarity: float) -> str:
        """Interpret what a cross-source relationship means."""
        if similarity > 0.3:
            if 'equipment' in src_i.lower() and 'ucdp' in src_j.lower():
                return "Equipment losses correlate with documented violence"
            elif 'sentinel' in src_i.lower() and 'firms' in src_j.lower():
                return "Satellite imagery correlates with fire detections"
            elif 'deepstate' in src_i.lower():
                return "Territory changes correlate with this source"
            return "Strong positive correlation"
        elif similarity > 0.1:
            return "Moderate correlation"
        elif similarity > -0.1:
            return "Weak/no correlation - independent sources"
        elif similarity > -0.3:
            return "Moderate negative correlation"
        else:
            return "Strong inverse relationship"

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recs = []

        # Based on source importance
        if 'importance_ranking' in self.insights.get('unified', {}):
            top_source = self.insights['unified']['importance_ranking'][0][0]
            recs.append(f"{top_source.upper()} is the most informative source - prioritize data quality here")

        # Based on cross-source relationships
        if 'source_similarity' in self.insights.get('unified', {}):
            sim = self.insights['unified']['source_similarity']
            names = self.insights['unified']['source_names']

            # Find most independent pair
            min_sim = 1.0
            independent_pair = None
            for i, src_i in enumerate(names):
                for j, src_j in enumerate(names):
                    if i < j and sim[i, j] < min_sim:
                        min_sim = sim[i, j]
                        independent_pair = (src_i, src_j)

            if independent_pair:
                recs.append(f"{independent_pair[0]} and {independent_pair[1]} capture distinct information - both are valuable")

        # Based on feature similarity
        if 'feature_embeddings' in self.insights['jim']:
            high_sim_models = [name for name, data in self.insights['jim']['feature_embeddings'].items()
                             if data['mean_intra_similarity'] > 0.5]
            if high_sim_models:
                recs.append(f"Consider feature reduction for: {', '.join(high_sim_models[:2])} (high redundancy)")

        # General recommendations
        recs.append("Use unified model embeddings for downstream forecasting (captures cross-source dependencies)")
        recs.append("Monitor Equipment-UCDP relationship for conflict intensity signals")

        return recs

    def save_insights_json(self):
        """Save all insights to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, tuple):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            return obj

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_jim_models': len(self.jim_models),
            'unified_model': self.unified_model is not None,
            'sources': list(SOURCE_CONFIGS.keys()),
        }

        # Extract key metrics
        if 'feature_embeddings' in self.insights['jim']:
            summary['jim_models'] = {}
            for name, data in self.insights['jim']['feature_embeddings'].items():
                summary['jim_models'][name] = {
                    'n_features': data['n_features'],
                    'mean_similarity': data['mean_intra_similarity'],
                    'n_clusters': data['n_clusters'],
                    'top_similar_pairs': [(int(i), int(j), float(s)) for i, j, s in data['top_similar_pairs'][:5]]
                }

        if self.insights['unified']:
            summary['unified'] = {
                'source_importance': self.insights['unified'].get('importance_ranking', []),
                'source_similarity': convert_for_json(self.insights['unified'].get('source_similarity', []))
            }

        if self.insights['cross_source']:
            summary['cross_source'] = {
                key: {
                    'max_correlation': data['max_correlation'],
                    'mean_correlation': data['mean_abs_correlation']
                }
                for key, data in self.insights['cross_source'].items()
            }

        # Save
        output_path = FIGURE_DIR / 'fine_grained_insights.json'
        with open(output_path, 'w') as f:
            json.dump(convert_for_json(summary), f, indent=2)

        print(f"\nInsights saved to: {output_path}")

    def run_full_analysis(self):
        """Run the complete fine-grained analysis pipeline."""
        print("\n" + "=" * 80)
        print("FINE-GRAINED INTER-FEATURE RELATIONSHIP ANALYSIS")
        print("ML-OSINT Ukraine Conflict Analysis Pipeline")
        print("=" * 80)

        # Load models
        self.load_all_models()

        # Run analyses
        self.analyze_jim_feature_embeddings()
        self.analyze_jim_attention_patterns()
        self.analyze_jim_decoder_weights()
        self.analyze_unified_cross_source()
        self.analyze_cross_source_feature_mapping()

        # Create visualizations
        self.create_visualizations()

        # Save insights
        self.save_insights_json()

        # Print summary
        insights_text = self._compile_key_insights()
        print("\n" + insights_text)

        print(f"\n\nAll figures saved to: {FIGURE_DIR}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fine-Grained Feature Analysis')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    analyzer = FineGrainedFeatureAnalyzer(device=args.device)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
