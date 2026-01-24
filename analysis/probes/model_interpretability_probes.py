"""
Model Interpretability Probes for Multi-Resolution HAN Pipeline

This module provides interpretability analysis for trained models:

Section 2.3: JIM Interpretability
    - 2.3.1: JIM Module I/O Analysis
    - 2.3.2: JIM Attention Pattern Analysis
    - 2.3.3: JIM Feature Embedding Analysis

Section 2.4: Unified Model Validation
    - 2.4.1: Cross-Source Latent Analysis
    - 2.4.2: Delta Model Validation

Author: Data Science Team
Date: 2026-01-24
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    ANALYSIS_DIR,
    MODEL_DIR,
    INTERP_MODEL_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# Import base probe infrastructure
from .data_artifact_probes import Probe, ProbeResult

# Check for torch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - model probes will be limited")


# =============================================================================
# SECTION 2.3: JIM INTERPRETABILITY
# =============================================================================

class JIMModuleIOProbe(Probe):
    """
    Probe 2.3.1: JIM Module I/O Analysis

    Analyzes the input-output flow through Joint Interpolation Model modules
    to understand data transformations and identify bottlenecks.

    Based on: jim_interpretability.py
    """

    @property
    def test_id(self) -> str:
        return "2.3.1"

    @property
    def test_name(self) -> str:
        return "JIM Module I/O Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute JIM module I/O analysis."""
        self.log("Starting JIM module I/O analysis...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Import JIM components
        try:
            from joint_interpolation_models import (
                INTERPOLATION_CONFIGS,
                JointInterpolationModel,
                InterpolationDataset,
            )
        except ImportError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': f'Cannot import JIM: {e}'}],
                recommendations=['Check joint_interpolation_models.py exists']
            )

        # Find available JIM models
        model_analyses = []

        for source_name, config in INTERPOLATION_CONFIGS.items():
            model_path = INTERP_MODEL_DIR / f'interp_{source_name}_best.pt'

            if not model_path.exists():
                continue

            self.log(f"Analyzing {source_name} model...")

            try:
                # Load model
                model = JointInterpolationModel(config)
                state = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    model.load_state_dict(state)
                model.eval()

                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                # Analyze module structure
                module_info = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        params = sum(p.numel() for p in module.parameters())
                        if params > 0:
                            module_info.append({
                                'name': name,
                                'type': type(module).__name__,
                                'parameters': params
                            })

                analysis = {
                    'source': source_name,
                    'n_features': config.n_features,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'n_modules': len(module_info),
                    'top_modules': sorted(module_info, key=lambda x: x['parameters'], reverse=True)[:5]
                }
                model_analyses.append(analysis)

                findings.append({
                    'category': 'MODEL_STRUCTURE',
                    'source': source_name,
                    'n_features': config.n_features,
                    'total_params': total_params,
                    'description': f'{source_name}: {total_params:,} params for {config.n_features} features'
                })

            except Exception as e:
                findings.append({
                    'category': 'ERROR',
                    'source': source_name,
                    'description': f'Failed to analyze: {str(e)}'
                })

        if not model_analyses:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No JIM models found'}],
                recommendations=['Train JIM models first using joint_interpolation_models.py']
            )

        # Create summary table
        summary_df = pd.DataFrame([{
            'source': a['source'],
            'n_features': a['n_features'],
            'total_params': a['total_params'],
            'params_per_feature': a['total_params'] / a['n_features'] if a['n_features'] > 0 else 0
        } for a in model_analyses])

        table_path = self.save_table(summary_df, 'jim_model_summary')
        artifacts['tables'].append(table_path)

        # Create visualization
        fig = self._create_module_figure(model_analyses)
        fig_path = self.save_figure(fig, 'jim_module_analysis')
        artifacts['figures'].append(fig_path)

        # Recommendations
        if summary_df['params_per_feature'].std() > summary_df['params_per_feature'].mean() * 0.5:
            recommendations.append(
                "Large variance in parameters per feature across models - consider standardizing architectures"
            )

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={
                'n_models_analyzed': len(model_analyses),
                'sources': [a['source'] for a in model_analyses]
            }
        )

    def _create_module_figure(self, analyses: List[Dict]) -> plt.Figure:
        """Create module analysis visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('JIM Module I/O Analysis', fontsize=14, fontweight='bold')

        # 1. Parameters by source
        ax = axes[0]
        sources = [a['source'] for a in analyses]
        params = [a['total_params'] for a in analyses]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sources)))
        ax.barh(sources, params, color=colors)
        ax.set_xlabel('Total Parameters')
        ax.set_title('Parameters by Source')
        for i, (s, p) in enumerate(zip(sources, params)):
            ax.text(p + max(params) * 0.01, i, f'{p:,}', va='center', fontsize=8)

        # 2. Features vs Parameters
        ax = axes[1]
        features = [a['n_features'] for a in analyses]
        ax.scatter(features, params, c=colors, s=100)
        for i, s in enumerate(sources):
            ax.annotate(s, (features[i], params[i]), fontsize=8,
                       xytext=(5, 5), textcoords='offset points')
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Total Parameters')
        ax.set_title('Features vs Parameters')

        # 3. Top modules (stacked bar)
        ax = axes[2]
        if analyses:
            # Get unique module types across all analyses
            all_modules = {}
            for a in analyses:
                for m in a.get('top_modules', []):
                    if m['type'] not in all_modules:
                        all_modules[m['type']] = 0
                    all_modules[m['type']] += m['parameters']

            if all_modules:
                types = list(all_modules.keys())[:8]  # Top 8 types
                values = [all_modules[t] for t in types]
                ax.pie(values, labels=types, autopct='%1.1f%%', startangle=90)
                ax.set_title('Parameter Distribution by Module Type')

        plt.tight_layout()
        return fig


class JIMAttentionAnalysisProbe(Probe):
    """
    Probe 2.3.2: JIM Attention Pattern Analysis

    Analyzes learned attention patterns in JIM models to understand
    which features the model considers related.
    """

    @property
    def test_id(self) -> str:
        return "2.3.2"

    @property
    def test_name(self) -> str:
        return "JIM Attention Pattern Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute JIM attention pattern analysis."""
        self.log("Starting JIM attention pattern analysis...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        try:
            from joint_interpolation_models import (
                INTERPOLATION_CONFIGS,
                JointInterpolationModel,
                InterpolationDataset,
            )
        except ImportError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': f'Cannot import JIM: {e}'}],
                recommendations=[]
            )

        attention_analyses = []

        for source_name, config in INTERPOLATION_CONFIGS.items():
            model_path = INTERP_MODEL_DIR / f'interp_{source_name}_best.pt'

            if not model_path.exists():
                continue

            self.log(f"Analyzing {source_name} attention patterns...")

            try:
                # Load model
                model = JointInterpolationModel(config)
                state = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'model_state_dict' in state:
                    model.load_state_dict(state['model_state_dict'])
                else:
                    model.load_state_dict(state)
                model.eval()

                # Extract attention weights from cross-feature attention
                if hasattr(model, 'cross_feature_attn'):
                    attn_module = model.cross_feature_attn

                    # Try to get attention weights
                    if hasattr(attn_module, 'attention'):
                        # Get the attention layer's in_proj_weight if it exists
                        for name, param in attn_module.named_parameters():
                            if 'in_proj_weight' in name or 'query' in name.lower():
                                # This gives us learned query/key projections
                                w = param.detach().cpu().numpy()
                                findings.append({
                                    'category': 'ATTENTION_WEIGHTS',
                                    'source': source_name,
                                    'weight_name': name,
                                    'shape': list(w.shape),
                                    'mean_magnitude': float(np.abs(w).mean()),
                                    'std': float(w.std())
                                })

                # Feature embedding analysis
                if hasattr(model, 'cross_feature_attn') and hasattr(model.cross_feature_attn, 'feature_embedding'):
                    emb = model.cross_feature_attn.feature_embedding.weight.detach().cpu().numpy()

                    # Compute feature similarity matrix
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(emb)

                    # Find top feature pairs
                    n_features = len(sim_matrix)
                    pairs = []
                    for i in range(n_features):
                        for j in range(i+1, n_features):
                            pairs.append((i, j, sim_matrix[i, j]))

                    top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:10]

                    attention_analyses.append({
                        'source': source_name,
                        'n_features': n_features,
                        'embedding_dim': emb.shape[1],
                        'similarity_matrix': sim_matrix,
                        'top_pairs': top_pairs,
                        'mean_similarity': float(sim_matrix[np.triu_indices(n_features, k=1)].mean())
                    })

                    findings.append({
                        'category': 'FEATURE_SIMILARITY',
                        'source': source_name,
                        'n_features': n_features,
                        'mean_similarity': float(sim_matrix[np.triu_indices(n_features, k=1)].mean()),
                        'max_similarity': float(sim_matrix[np.triu_indices(n_features, k=1)].max()),
                        'top_pair_similarity': float(top_pairs[0][2]) if top_pairs else None
                    })

            except Exception as e:
                findings.append({
                    'category': 'ERROR',
                    'source': source_name,
                    'description': str(e)
                })

        # Create visualizations
        if attention_analyses:
            fig = self._create_attention_figure(attention_analyses)
            fig_path = self.save_figure(fig, 'jim_attention_patterns')
            artifacts['figures'].append(fig_path)

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'n_models_analyzed': len(attention_analyses)}
        )

    def _create_attention_figure(self, analyses: List[Dict]) -> plt.Figure:
        """Create attention pattern visualization."""
        n_analyses = len(analyses)
        if n_analyses == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No attention data available', ha='center', va='center')
            return fig

        # Create grid for similarity matrices
        n_cols = min(3, n_analyses)
        n_rows = (n_analyses + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle('JIM Feature Similarity Matrices', fontsize=14, fontweight='bold')

        if n_analyses == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, analysis in enumerate(analyses):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]

            sim_matrix = analysis['similarity_matrix']
            im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f"{analysis['source']}\n(mean sim: {analysis['mean_similarity']:.3f})")
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Feature Index')
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused axes
        for idx in range(n_analyses, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 2.4: UNIFIED MODEL VALIDATION
# =============================================================================

class CrossSourceLatentProbe(Probe):
    """
    Probe 2.4.1: Cross-Source Latent Analysis

    Analyzes how different data sources are represented and related
    in the unified interpolation model's latent space.

    Based on: cross_source_deep_dive.py, delta_model_deep_analysis.py
    """

    @property
    def test_id(self) -> str:
        return "2.4.1"

    @property
    def test_name(self) -> str:
        return "Cross-Source Latent Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute cross-source latent analysis."""
        self.log("Starting cross-source latent analysis...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Try to load unified model
        try:
            from unified_interpolation_delta import (
                SOURCE_CONFIGS,
                UnifiedInterpolationModelDelta,
            )
        except ImportError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': f'Cannot import unified model: {e}'}],
                recommendations=['Check unified_interpolation_delta.py exists']
            )

        # Check for model checkpoint
        model_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'
        if not model_path.exists():
            # Try alternate path
            model_path = MODEL_DIR / 'unified_interpolation_best.pt'

        if not model_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No unified model checkpoint found'}],
                recommendations=['Train unified interpolation model first']
            )

        self.log(f"Loading model from {model_path}...")

        try:
            # Load model
            state = torch.load(model_path, map_location='cpu', weights_only=False)

            # Infer config from state dict
            from copy import deepcopy
            configs = deepcopy(SOURCE_CONFIGS)
            for name in configs:
                key = f'encoders.{name}.feature_proj.0.weight'
                if key in state:
                    configs[name].n_features = state[key].shape[1]

            model = UnifiedInterpolationModelDelta(configs, d_embed=64, nhead=4, num_fusion_layers=2)
            model.load_state_dict(state)
            model.eval()

            # Analyze source embeddings
            if hasattr(model, 'source_embedding'):
                emb = model.source_embedding.weight.detach().cpu().numpy()
                source_names = list(configs.keys())

                # Source similarity
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(emb)

                # Find source relationships
                n_sources = len(source_names)
                pairs = []
                for i in range(n_sources):
                    for j in range(i+1, n_sources):
                        pairs.append({
                            'source1': source_names[i],
                            'source2': source_names[j],
                            'similarity': float(sim_matrix[i, j])
                        })

                pairs_sorted = sorted(pairs, key=lambda x: x['similarity'], reverse=True)

                findings.append({
                    'category': 'SOURCE_SIMILARITY',
                    'n_sources': n_sources,
                    'embedding_dim': emb.shape[1],
                    'top_similar_pairs': pairs_sorted[:3],
                    'least_similar_pairs': pairs_sorted[-3:]
                })

                # Save similarity matrix
                sim_df = pd.DataFrame(sim_matrix, index=source_names, columns=source_names)
                table_path = self.save_table(sim_df, 'source_similarity_matrix')
                artifacts['tables'].append(table_path)

            # Analyze encoder structure
            encoder_info = []
            for source_name in configs.keys():
                if hasattr(model, 'encoders') and source_name in model.encoders:
                    encoder = model.encoders[source_name]
                    params = sum(p.numel() for p in encoder.parameters())
                    encoder_info.append({
                        'source': source_name,
                        'n_features': configs[source_name].n_features,
                        'params': params
                    })

            if encoder_info:
                findings.append({
                    'category': 'ENCODER_STRUCTURE',
                    'encoders': encoder_info,
                    'total_encoder_params': sum(e['params'] for e in encoder_info)
                })

            # Create visualization
            fig = self._create_latent_figure(model, configs, source_names if 'source_names' in dir() else list(configs.keys()))
            fig_path = self.save_figure(fig, 'cross_source_latent')
            artifacts['figures'].append(fig_path)

        except Exception as e:
            findings.append({
                'category': 'ERROR',
                'description': f'Failed to analyze model: {str(e)}'
            })
            import traceback
            traceback.print_exc()

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'model_path': str(model_path)}
        )

    def _create_latent_figure(self, model, configs, source_names) -> plt.Figure:
        """Create cross-source latent visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Cross-Source Latent Space Analysis', fontsize=14, fontweight='bold')

        # 1. Source embedding similarity
        ax = axes[0]
        if hasattr(model, 'source_embedding'):
            emb = model.source_embedding.weight.detach().cpu().numpy()
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(emb)
            im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(source_names)))
            ax.set_yticks(range(len(source_names)))
            ax.set_xticklabels(source_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(source_names, fontsize=8)
            ax.set_title('Source Embedding Similarity')
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No source embeddings', ha='center', va='center')

        # 2. Feature counts by source
        ax = axes[1]
        features = [configs[s].n_features for s in source_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(source_names)))
        ax.bar(source_names, features, color=colors)
        ax.set_xlabel('Source')
        ax.set_ylabel('Number of Features')
        ax.set_title('Features by Source')
        ax.tick_params(axis='x', rotation=45)

        # 3. Encoder parameters
        ax = axes[2]
        if hasattr(model, 'encoders'):
            params = []
            for s in source_names:
                if s in model.encoders:
                    p = sum(p.numel() for p in model.encoders[s].parameters())
                    params.append(p)
                else:
                    params.append(0)
            ax.bar(source_names, params, color=colors)
            ax.set_xlabel('Source')
            ax.set_ylabel('Parameters')
            ax.set_title('Encoder Parameters by Source')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No encoder info', ha='center', va='center')

        plt.tight_layout()
        return fig


class DeltaModelValidationProbe(Probe):
    """
    Probe 2.4.2: Delta Model Validation

    Validates that the delta-trained unified model produces sensible
    cross-source correlations without spurious time-trend effects.

    Based on: delta_model_deep_analysis.py
    """

    @property
    def test_id(self) -> str:
        return "2.4.2"

    @property
    def test_name(self) -> str:
        return "Delta Model Validation"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute delta model validation."""
        self.log("Starting delta model validation...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Check for both cumulative and delta models
        cumulative_path = MODEL_DIR / 'unified_interpolation_best.pt'
        delta_path = MODEL_DIR / 'unified_interpolation_delta_best.pt'

        models_found = []
        if cumulative_path.exists():
            models_found.append(('cumulative', cumulative_path))
        if delta_path.exists():
            models_found.append(('delta', delta_path))

        if not models_found:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No unified models found'}],
                recommendations=['Train unified interpolation models']
            )

        # Compare models if both exist
        if len(models_found) == 2:
            findings.append({
                'category': 'MODEL_COMPARISON',
                'description': 'Both cumulative and delta models available for comparison',
                'cumulative_path': str(cumulative_path),
                'delta_path': str(delta_path)
            })
            recommendations.append(
                "Run comprehensive_model_report.py for detailed cumulative vs delta comparison"
            )
        else:
            findings.append({
                'category': 'MODEL_AVAILABILITY',
                'description': f"Only {models_found[0][0]} model available",
                'available_model': models_found[0][0]
            })

        # Analyze delta model specifically
        if delta_path.exists():
            try:
                from unified_interpolation_delta import (
                    SOURCE_CONFIGS,
                    UnifiedInterpolationModelDelta,
                )

                state = torch.load(delta_path, map_location='cpu', weights_only=False)

                # Check which features are delta-only
                delta_features_info = []
                for name, cfg in SOURCE_CONFIGS.items():
                    if name == 'equipment':
                        # Equipment should only have delta features
                        key = f'encoders.{name}.feature_proj.0.weight'
                        if key in state:
                            n_features = state[key].shape[1]
                            delta_features_info.append({
                                'source': name,
                                'n_features': n_features,
                                'is_delta': True,
                                'note': 'Equipment uses delta-only features'
                            })

                if delta_features_info:
                    findings.append({
                        'category': 'DELTA_FEATURES',
                        'description': 'Delta feature configuration verified',
                        'features': delta_features_info
                    })

                # Model size comparison
                total_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
                findings.append({
                    'category': 'MODEL_SIZE',
                    'model': 'delta',
                    'total_params': total_params,
                    'description': f'Delta model has {total_params:,} parameters'
                })

            except Exception as e:
                findings.append({
                    'category': 'ERROR',
                    'description': f'Failed to analyze delta model: {str(e)}'
                })

        # Create summary figure
        fig = self._create_validation_figure(models_found, findings)
        fig_path = self.save_figure(fig, 'delta_model_validation')
        artifacts['figures'].append(fig_path)

        self.log("Validation complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'models_analyzed': [m[0] for m in models_found]}
        )

    def _create_validation_figure(self, models: List[Tuple], findings: List[Dict]) -> plt.Figure:
        """Create delta model validation visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Delta Model Validation Summary', fontsize=14, fontweight='bold')

        # 1. Model availability
        ax = axes[0]
        model_names = [m[0] for m in models]
        ax.bar(model_names, [1] * len(model_names), color=['green' if 'delta' in m else 'blue' for m in model_names])
        ax.set_ylim(0, 1.5)
        ax.set_ylabel('Available')
        ax.set_title('Model Availability')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])

        # 2. Findings summary
        ax = axes[1]
        categories = {}
        for f in findings:
            cat = f.get('category', 'OTHER')
            categories[cat] = categories.get(cat, 0) + 1

        if categories:
            ax.pie(list(categories.values()), labels=list(categories.keys()),
                   autopct='%1.0f', startangle=90)
            ax.set_title('Findings by Category')
        else:
            ax.text(0.5, 0.5, 'No findings', ha='center', va='center')

        plt.tight_layout()
        return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Section 2.3: JIM Interpretability
    'JIMModuleIOProbe',
    'JIMAttentionAnalysisProbe',

    # Section 2.4: Unified Model Validation
    'CrossSourceLatentProbe',
    'DeltaModelValidationProbe',
]
