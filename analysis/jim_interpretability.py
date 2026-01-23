"""
Joint Interpolation Models (JIM) - Interpretability Analysis

This script provides deep interpretability analysis of trained JIM models:
1. Input-Output Inventory - What flows through each module
2. Attention Analysis - What feature correlations are learned
3. Embedding Visualization - How features are represented
4. Uncertainty Analysis - Where the model is confident/uncertain
5. Module-by-Module Data Flow - How information transforms

Usage:
    python jim_interpretability.py --model sentinel2
    python jim_interpretability.py --all
    python jim_interpretability.py --live  # Analyze during training
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import json
from datetime import datetime

# Add analysis directory to path
ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

# Import JIM components
from joint_interpolation_models import (
    INTERPOLATION_CONFIGS,
    JointInterpolationModel,
    InterpolationDataset,
    InterpolationConfig,
    DATA_DIR
)
from config.paths import INTERP_MODEL_DIR


@dataclass
class ModuleIO:
    """Captures input-output shapes and statistics for a module."""
    name: str
    input_shape: Tuple
    output_shape: Tuple
    input_stats: Dict[str, float]  # mean, std, min, max
    output_stats: Dict[str, float]
    parameters: int
    description: str


@dataclass
class AttentionAnalysis:
    """Analysis of attention patterns."""
    feature_names: List[str]
    attention_weights: np.ndarray  # [num_features, num_features]
    top_correlations: List[Tuple[str, str, float]]  # (feat1, feat2, weight)
    feature_importance: np.ndarray  # How much each feature attends/is attended


class JIMInterpreter:
    """
    Comprehensive interpretability analysis for Joint Interpolation Models.
    """

    def __init__(self, model: JointInterpolationModel, config: InterpolationConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device if HAS_TORCH else 'cpu'
        self.hooks = []
        self.activations = {}
        self.gradients = {}

    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        self.hooks = []
        self.activations = {}

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = [o.detach().cpu() if torch.is_tensor(o) else o for o in output]
                else:
                    self.activations[name] = output.detach().cpu() if torch.is_tensor(output) else output
            return hook

        # Register hooks on key modules
        modules_to_hook = {
            'cross_feature_attn': self.model.cross_feature_attn,
            'cross_feature_attn.feature_projection': self.model.cross_feature_attn.feature_projection,
            'cross_feature_attn.transformer': self.model.cross_feature_attn.transformer,
            'temporal_encoding': self.model.temporal_encoding,
            'gap_interpolator': self.model.gap_interpolator,
            'uncertainty_head': self.model.uncertainty_head,
        }

        for name, module in modules_to_hook.items():
            hook = module.register_forward_hook(get_activation(name))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def inventory_io(self, sample_batch: Tuple) -> Dict[str, ModuleIO]:
        """
        Create comprehensive input-output inventory for all modules.

        Returns dict mapping module name to ModuleIO dataclass.
        """
        self._register_hooks()
        self.model.eval()

        # Unpack batch
        obs_before, obs_after, day_before, day_after, day_target, target = sample_batch
        obs_before = obs_before.to(self.device)
        obs_after = obs_after.to(self.device)
        day_before = day_before.to(self.device)
        day_after = day_after.to(self.device)
        day_target = day_target.to(self.device)

        # Forward pass to capture activations
        with torch.no_grad():
            predictions, uncertainties = self.model(
                obs_before, obs_after,
                day_before, day_after, day_target
            )

        inventory = {}

        # 1. Input Layer Analysis
        inventory['input_obs_before'] = ModuleIO(
            name='Input: Observation Before',
            input_shape=tuple(obs_before.shape),
            output_shape=tuple(obs_before.shape),
            input_stats=self._tensor_stats(obs_before),
            output_stats=self._tensor_stats(obs_before),
            parameters=0,
            description=f"Raw feature values at time T-n. Shape: [batch, {self.config.features[0] if self.config.features else 'features'}...]"
        )

        inventory['input_obs_after'] = ModuleIO(
            name='Input: Observation After',
            input_shape=tuple(obs_after.shape),
            output_shape=tuple(obs_after.shape),
            input_stats=self._tensor_stats(obs_after),
            output_stats=self._tensor_stats(obs_after),
            parameters=0,
            description="Raw feature values at time T+m"
        )

        inventory['input_temporal'] = ModuleIO(
            name='Input: Temporal Positions',
            input_shape=(tuple(day_before.shape), tuple(day_after.shape), tuple(day_target.shape)),
            output_shape=(tuple(day_before.shape), tuple(day_after.shape), tuple(day_target.shape)),
            input_stats={
                'day_before_mean': day_before.mean().item(),
                'day_after_mean': day_after.mean().item(),
                'day_target_mean': day_target.mean().item(),
                'gap_mean': (day_after - day_before).mean().item()
            },
            output_stats={},
            parameters=0,
            description="Day offsets: before, after, and target interpolation day"
        )

        # 2. Feature Projection
        if 'cross_feature_attn.feature_projection' in self.activations:
            proj_out = self.activations['cross_feature_attn.feature_projection']
            inventory['feature_projection'] = ModuleIO(
                name='Feature Projection (Linear)',
                input_shape=(obs_before.shape[0], obs_before.shape[1], 1),  # [batch, n_feat, 1]
                output_shape=tuple(proj_out.shape),
                input_stats=self._tensor_stats(obs_before.unsqueeze(-1)),
                output_stats=self._tensor_stats(proj_out),
                parameters=sum(p.numel() for p in self.model.cross_feature_attn.feature_projection.parameters()),
                description=f"Projects each scalar feature to d_model={self.config.d_model} dimensional embedding"
            )

        # 3. Feature Embeddings (learnable)
        feat_emb = self.model.cross_feature_attn.feature_embeddings.weight
        inventory['feature_embeddings'] = ModuleIO(
            name='Feature Embeddings (Learnable)',
            input_shape=(len(self.config.features),),  # Feature indices
            output_shape=tuple(feat_emb.shape),
            input_stats={'num_features': len(self.config.features)},
            output_stats=self._tensor_stats(feat_emb),
            parameters=feat_emb.numel(),
            description="Learnable embedding for each feature type (like word embeddings)"
        )

        # 4. Cross-Feature Transformer
        if 'cross_feature_attn.transformer' in self.activations:
            transformer_out = self.activations['cross_feature_attn.transformer']
            inventory['cross_feature_transformer'] = ModuleIO(
                name='Cross-Feature Transformer',
                input_shape=tuple(self.activations.get('cross_feature_attn.feature_projection', torch.zeros(1)).shape),
                output_shape=tuple(transformer_out.shape),
                input_stats=self._tensor_stats(self.activations.get('cross_feature_attn.feature_projection', torch.zeros(1))),
                output_stats=self._tensor_stats(transformer_out),
                parameters=sum(p.numel() for p in self.model.cross_feature_attn.transformer.parameters()),
                description="Self-attention across features to learn correlations"
            )

        # 5. Temporal Encoding
        if 'temporal_encoding' in self.activations:
            temp_out = self.activations['temporal_encoding']
            inventory['temporal_encoding'] = ModuleIO(
                name='Temporal Positional Encoding',
                input_shape=tuple(self.activations.get('cross_feature_attn.transformer', torch.zeros(1)).shape),
                output_shape=tuple(temp_out.shape) if torch.is_tensor(temp_out) else 'N/A',
                input_stats=self._tensor_stats(self.activations.get('cross_feature_attn.transformer', torch.zeros(1))),
                output_stats=self._tensor_stats(temp_out) if torch.is_tensor(temp_out) else {},
                parameters=sum(p.numel() for p in self.model.temporal_encoding.parameters()),
                description="Adds temporal position information based on day offset"
            )

        # 6. Gap Interpolator
        if 'gap_interpolator' in self.activations:
            gap_out = self.activations['gap_interpolator']
            inventory['gap_interpolator'] = ModuleIO(
                name='Gap Interpolator',
                input_shape='(enc_before, enc_after, values_before, values_after)',
                output_shape=tuple(gap_out.shape) if torch.is_tensor(gap_out) else 'N/A',
                input_stats={},
                output_stats=self._tensor_stats(gap_out) if torch.is_tensor(gap_out) else {},
                parameters=sum(p.numel() for p in self.model.gap_interpolator.parameters()),
                description="Attention-based interpolation using day queries"
            )

        # 7. Uncertainty Head
        inventory['uncertainty_head'] = ModuleIO(
            name='Uncertainty Estimation Head',
            input_shape='[batch, num_features, d_model]',
            output_shape=tuple(uncertainties.shape),
            input_stats={},
            output_stats=self._tensor_stats(uncertainties),
            parameters=sum(p.numel() for p in self.model.uncertainty_head.parameters()),
            description="Predicts uncertainty (std) for each feature prediction"
        )

        # 8. Final Output
        inventory['output_predictions'] = ModuleIO(
            name='Output: Predictions',
            input_shape='gap_interpolator output',
            output_shape=tuple(predictions.shape),
            input_stats={},
            output_stats=self._tensor_stats(predictions),
            parameters=0,
            description="Interpolated feature values for target day"
        )

        inventory['output_uncertainties'] = ModuleIO(
            name='Output: Uncertainties',
            input_shape='uncertainty_head input',
            output_shape=tuple(uncertainties.shape),
            input_stats={},
            output_stats=self._tensor_stats(uncertainties),
            parameters=0,
            description="Predicted standard deviation for each feature"
        )

        self._remove_hooks()
        return inventory

    def _tensor_stats(self, tensor) -> Dict[str, float]:
        """Compute statistics for a tensor."""
        if not torch.is_tensor(tensor):
            return {}
        t = tensor.float()
        return {
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'shape': list(t.shape)
        }

    def analyze_attention(self, dataloader: DataLoader, n_batches: int = 10) -> AttentionAnalysis:
        """
        Analyze attention patterns in the cross-feature transformer.

        Extracts attention weights to understand:
        - Which features attend to which other features
        - Feature importance (how much each is attended)
        - Learned correlations
        """
        self.model.eval()

        # We need to modify the transformer to capture attention weights
        # Access the transformer encoder layers
        encoder = self.model.cross_feature_attn.transformer

        all_attention_weights = []

        # Hook to capture attention
        attention_weights = []

        def attention_hook(module, input, output):
            # For TransformerEncoderLayer, we need to access self_attn
            pass

        # Alternative: manually compute attention for analysis
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break

                obs_before, obs_after, day_before, day_after, day_target, target = batch
                obs_before = obs_before.to(self.device)

                # Get feature projections + embeddings
                x_proj = self.model.cross_feature_attn.feature_projection(obs_before.unsqueeze(-1))
                feat_indices = torch.arange(self.model.cross_feature_attn.num_features, device=self.device)
                feat_emb = self.model.cross_feature_attn.feature_embeddings(feat_indices)
                x_proj = x_proj + feat_emb.unsqueeze(0)

                # Compute attention manually (Q, K, V)
                # For a standard transformer layer: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
                d_k = self.config.d_model // self.config.nhead

                # Approximate attention by computing correlation of projected features
                # Normalize and compute similarity
                x_norm = F.normalize(x_proj, dim=-1)
                attn_approx = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [batch, n_feat, n_feat]
                all_attention_weights.append(attn_approx.cpu().numpy())

        # Average attention across batches
        avg_attention = np.mean(np.concatenate(all_attention_weights, axis=0), axis=0)

        # Extract top correlations
        n_feat = len(self.config.features)
        correlations = []
        for i in range(n_feat):
            for j in range(i+1, n_feat):
                correlations.append((
                    self.config.features[i],
                    self.config.features[j],
                    avg_attention[i, j]
                ))

        # Sort by attention weight
        correlations.sort(key=lambda x: -x[2])

        # Feature importance: sum of attention received
        importance = avg_attention.sum(axis=0)
        importance = importance / importance.sum()  # Normalize

        return AttentionAnalysis(
            feature_names=self.config.features,
            attention_weights=avg_attention,
            top_correlations=correlations[:20],  # Top 20
            feature_importance=importance
        )

    def analyze_uncertainty(self, dataloader: DataLoader, n_batches: int = 20) -> Dict[str, Any]:
        """
        Analyze uncertainty predictions across different conditions.

        Questions answered:
        - Which features have highest/lowest uncertainty?
        - Does uncertainty increase with gap size?
        - Are there systematic patterns in uncertainty?
        """
        self.model.eval()

        all_uncertainties = []
        all_gap_sizes = []
        all_errors = []
        feature_uncertainties = {f: [] for f in self.config.features}

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break

                obs_before, obs_after, day_before, day_after, day_target, target = batch
                obs_before = obs_before.to(self.device)
                obs_after = obs_after.to(self.device)
                day_before = day_before.to(self.device)
                day_after = day_after.to(self.device)
                day_target = day_target.to(self.device)
                target = target.to(self.device)

                predictions, uncertainties = self.model(
                    obs_before, obs_after,
                    day_before, day_after, day_target
                )

                # Compute actual errors
                errors = torch.abs(predictions - target)

                # Gap sizes
                gaps = (day_after - day_before).squeeze()

                all_uncertainties.append(uncertainties.cpu().numpy())
                all_gap_sizes.append(gaps.cpu().numpy())
                all_errors.append(errors.cpu().numpy())

                # Per-feature uncertainties
                for i, feat in enumerate(self.config.features[:uncertainties.shape[1]]):
                    feature_uncertainties[feat].extend(uncertainties[:, i].cpu().numpy().tolist())

        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_errors = np.concatenate(all_errors, axis=0)
        all_gap_sizes = np.concatenate(all_gap_sizes, axis=0) if all_gap_sizes else np.array([])

        # Compute correlation between uncertainty and actual error
        uncertainty_error_corr = np.corrcoef(
            all_uncertainties.flatten(),
            all_errors.flatten()
        )[0, 1] if all_uncertainties.size > 0 else 0

        # Per-feature analysis
        feature_analysis = {}
        for feat, uncerts in feature_uncertainties.items():
            if uncerts:
                feature_analysis[feat] = {
                    'mean_uncertainty': np.mean(uncerts),
                    'std_uncertainty': np.std(uncerts),
                    'min_uncertainty': np.min(uncerts),
                    'max_uncertainty': np.max(uncerts)
                }

        # Rank features by uncertainty
        ranked_features = sorted(
            feature_analysis.items(),
            key=lambda x: x[1]['mean_uncertainty'],
            reverse=True
        )

        return {
            'uncertainty_error_correlation': uncertainty_error_corr,
            'mean_uncertainty': all_uncertainties.mean(),
            'mean_error': all_errors.mean(),
            'calibration': 'Under-confident' if uncertainty_error_corr < 0.3 else 'Well-calibrated' if uncertainty_error_corr < 0.7 else 'Over-confident',
            'highest_uncertainty_features': ranked_features[:5],
            'lowest_uncertainty_features': ranked_features[-5:],
            'feature_analysis': feature_analysis
        }

    def analyze_embeddings(self) -> Dict[str, Any]:
        """
        Analyze learned feature embeddings.

        Questions answered:
        - How are features clustered in embedding space?
        - Which features have similar representations?
        - What semantic structure has been learned?
        """
        embeddings = self.model.cross_feature_attn.feature_embeddings.weight.detach().cpu().numpy()

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarities = normalized @ normalized.T

        # Find most similar pairs
        n_feat = len(self.config.features)
        similar_pairs = []
        for i in range(n_feat):
            for j in range(i+1, n_feat):
                similar_pairs.append((
                    self.config.features[i],
                    self.config.features[j],
                    similarities[i, j]
                ))

        similar_pairs.sort(key=lambda x: -x[2])

        # Embedding statistics per feature
        embedding_stats = {}
        for i, feat in enumerate(self.config.features):
            emb = embeddings[i]
            embedding_stats[feat] = {
                'norm': np.linalg.norm(emb),
                'mean': emb.mean(),
                'std': emb.std(),
                'max_abs': np.abs(emb).max()
            }

        # Find clusters using simple k-means-like grouping
        # Group by highest similarity neighbor
        clusters = {}
        assigned = set()
        for i in range(n_feat):
            if self.config.features[i] in assigned:
                continue
            cluster = [self.config.features[i]]
            assigned.add(self.config.features[i])

            for j in range(n_feat):
                if i != j and self.config.features[j] not in assigned:
                    if similarities[i, j] > 0.7:  # High similarity threshold
                        cluster.append(self.config.features[j])
                        assigned.add(self.config.features[j])

            if len(cluster) > 1:
                clusters[f"cluster_{len(clusters)}"] = cluster

        return {
            'embedding_dim': embeddings.shape[1],
            'most_similar_pairs': similar_pairs[:10],
            'most_dissimilar_pairs': similar_pairs[-10:],
            'embedding_stats': embedding_stats,
            'learned_clusters': clusters,
            'similarity_matrix': similarities
        }

    def trace_data_flow(self, sample_batch: Tuple) -> List[Dict[str, Any]]:
        """
        Trace data transformation through the entire network.

        Returns a list of transformation stages with before/after statistics.
        """
        self._register_hooks()
        self.model.eval()

        obs_before, obs_after, day_before, day_after, day_target, target = sample_batch
        obs_before = obs_before.to(self.device)
        obs_after = obs_after.to(self.device)
        day_before = day_before.to(self.device)
        day_after = day_after.to(self.device)
        day_target = day_target.to(self.device)

        with torch.no_grad():
            predictions, uncertainties = self.model(
                obs_before, obs_after,
                day_before, day_after, day_target
            )

        stages = []

        # Stage 1: Raw Input
        stages.append({
            'stage': 1,
            'name': 'Raw Input',
            'description': 'Original feature values from observations',
            'shape': list(obs_before.shape),
            'stats': self._tensor_stats(obs_before),
            'interpretation': f"Each sample has {obs_before.shape[1]} features with values in [{obs_before.min():.3f}, {obs_before.max():.3f}]"
        })

        # Stage 2: Feature Projection
        if 'cross_feature_attn.feature_projection' in self.activations:
            proj = self.activations['cross_feature_attn.feature_projection']
            stages.append({
                'stage': 2,
                'name': 'Feature Projection',
                'description': f'Linear projection from scalar to d_model={self.config.d_model}',
                'shape': list(proj.shape),
                'stats': self._tensor_stats(proj),
                'interpretation': f"Each feature now has {self.config.d_model} dimensions for attention"
            })

        # Stage 3: Feature Embeddings Added
        feat_emb = self.model.cross_feature_attn.feature_embeddings.weight
        stages.append({
            'stage': 3,
            'name': 'Feature Embeddings Added',
            'description': 'Learnable feature-type embeddings added to projections',
            'shape': list(feat_emb.shape),
            'stats': self._tensor_stats(feat_emb),
            'interpretation': "Feature identity encoded - model knows which feature it's processing"
        })

        # Stage 4: Cross-Feature Attention
        if 'cross_feature_attn.transformer' in self.activations:
            attn_out = self.activations['cross_feature_attn.transformer']
            stages.append({
                'stage': 4,
                'name': 'Cross-Feature Attention',
                'description': f'{self.config.num_layers}-layer transformer with {self.config.nhead} heads',
                'shape': list(attn_out.shape),
                'stats': self._tensor_stats(attn_out),
                'interpretation': "Features have exchanged information via attention"
            })

        # Stage 5: Temporal Encoding
        if 'temporal_encoding' in self.activations:
            temp_out = self.activations['temporal_encoding']
            if torch.is_tensor(temp_out):
                stages.append({
                    'stage': 5,
                    'name': 'Temporal Position Encoding',
                    'description': 'Day offset encoded into representation',
                    'shape': list(temp_out.shape),
                    'stats': self._tensor_stats(temp_out),
                    'interpretation': "Time position now encoded - model knows when this observation occurred"
                })

        # Stage 6: Gap Interpolation
        stages.append({
            'stage': 6,
            'name': 'Gap Interpolation',
            'description': 'Attention-based interpolation between before/after',
            'shape': list(predictions.shape),
            'stats': self._tensor_stats(predictions),
            'interpretation': f"Predicted values for target day (gap={day_after.mean() - day_before.mean():.1f} days)"
        })

        # Stage 7: Uncertainty Estimation
        stages.append({
            'stage': 7,
            'name': 'Uncertainty Estimation',
            'description': 'Predicted standard deviation per feature',
            'shape': list(uncertainties.shape),
            'stats': self._tensor_stats(uncertainties),
            'interpretation': f"Model confidence: avg uncertainty = {uncertainties.mean():.4f}"
        })

        self._remove_hooks()
        return stages

    def generate_report(self, dataloader: DataLoader) -> str:
        """Generate a comprehensive interpretability report."""
        report = []
        report.append("=" * 80)
        report.append(f"INTERPRETABILITY REPORT: {self.config.name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        # Get a sample batch
        sample_batch = next(iter(dataloader))
        sample_batch = tuple(x[:4] for x in sample_batch)  # Use smaller batch for analysis

        # 1. Input-Output Inventory
        report.append("\n" + "=" * 60)
        report.append("1. INPUT-OUTPUT INVENTORY")
        report.append("=" * 60)

        inventory = self.inventory_io(sample_batch)
        for name, io in inventory.items():
            report.append(f"\n{io.name}")
            report.append("-" * 40)
            report.append(f"  Input shape:  {io.input_shape}")
            report.append(f"  Output shape: {io.output_shape}")
            report.append(f"  Parameters:   {io.parameters:,}")
            report.append(f"  Description:  {io.description}")
            if io.output_stats:
                stats = io.output_stats
                if 'mean' in stats:
                    report.append(f"  Output stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")

        # 2. Attention Analysis
        report.append("\n" + "=" * 60)
        report.append("2. ATTENTION ANALYSIS")
        report.append("=" * 60)

        attn = self.analyze_attention(dataloader)
        report.append("\nTop 10 Feature Correlations (by attention):")
        for f1, f2, weight in attn.top_correlations[:10]:
            report.append(f"  {f1} <-> {f2}: {weight:.4f}")

        report.append("\nFeature Importance (attention received):")
        sorted_importance = sorted(zip(attn.feature_names, attn.feature_importance),
                                   key=lambda x: -x[1])
        for feat, imp in sorted_importance[:10]:
            report.append(f"  {feat}: {imp:.4f}")

        # 3. Embedding Analysis
        report.append("\n" + "=" * 60)
        report.append("3. EMBEDDING ANALYSIS")
        report.append("=" * 60)

        emb = self.analyze_embeddings()
        report.append(f"\nEmbedding dimension: {emb['embedding_dim']}")

        report.append("\nMost Similar Feature Pairs:")
        for f1, f2, sim in emb['most_similar_pairs'][:5]:
            report.append(f"  {f1} <-> {f2}: {sim:.4f}")

        if emb['learned_clusters']:
            report.append("\nLearned Feature Clusters:")
            for cluster_name, features in emb['learned_clusters'].items():
                report.append(f"  {cluster_name}: {', '.join(features)}")

        # 4. Uncertainty Analysis
        report.append("\n" + "=" * 60)
        report.append("4. UNCERTAINTY ANALYSIS")
        report.append("=" * 60)

        uncert = self.analyze_uncertainty(dataloader)
        report.append(f"\nUncertainty-Error Correlation: {uncert['uncertainty_error_correlation']:.4f}")
        report.append(f"Calibration: {uncert['calibration']}")
        report.append(f"Mean Uncertainty: {uncert['mean_uncertainty']:.4f}")
        report.append(f"Mean Error: {uncert['mean_error']:.4f}")

        report.append("\nHighest Uncertainty Features:")
        for feat, stats in uncert['highest_uncertainty_features']:
            report.append(f"  {feat}: {stats['mean_uncertainty']:.4f} +/- {stats['std_uncertainty']:.4f}")

        # 5. Data Flow Trace
        report.append("\n" + "=" * 60)
        report.append("5. DATA FLOW TRACE")
        report.append("=" * 60)

        flow = self.trace_data_flow(sample_batch)
        for stage in flow:
            report.append(f"\nStage {stage['stage']}: {stage['name']}")
            report.append(f"  Shape: {stage['shape']}")
            report.append(f"  {stage['description']}")
            report.append(f"  -> {stage['interpretation']}")

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)


def analyze_trained_model(model_name: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a trained JIM model.

    Args:
        model_name: Name from INTERPOLATION_CONFIGS (e.g., 'sentinel2')
        verbose: Print report to stdout

    Returns:
        Dictionary with all analysis results
    """
    if model_name not in INTERPOLATION_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(INTERPOLATION_CONFIGS.keys())}")
        return {}

    config = INTERPOLATION_CONFIGS[model_name]

    # Load dataset to get actual feature count
    print(f"Loading data for {model_name}...")
    train_dataset = InterpolationDataset(config, DATA_DIR, train=True)
    val_dataset = InterpolationDataset(config, DATA_DIR, train=False)

    # Get actual features
    actual_n_features = getattr(train_dataset, 'actual_num_features', len(config.features))
    actual_feature_names = getattr(train_dataset, 'actual_feature_names', config.features)

    # Create config with actual features
    if actual_n_features != len(config.features):
        actual_config = InterpolationConfig(
            name=config.name,
            source=config.source,
            features=actual_feature_names,
            native_resolution_days=config.native_resolution_days,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            max_gap_days=config.max_gap_days,
            dropout=config.dropout,
            parent_features=config.parent_features,
            child_groups=config.child_groups,
            hierarchy_level=config.hierarchy_level,
            conditioning_dim=config.conditioning_dim,
        )
    else:
        actual_config = config

    # Create model
    model = JointInterpolationModel(actual_config)

    # Try to load trained weights
    safe_name = actual_config.name.replace(' ', '_').replace('/', '_').lower()
    model_path = INTERP_MODEL_DIR / f"interp_{safe_name}_best.pt"

    if model_path.exists():
        print(f"Loading trained weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"No trained weights found at {model_path}")
        print("Analyzing untrained model (random weights)")

    # Create dataloader
    dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create interpreter and run analysis
    interpreter = JIMInterpreter(model, actual_config)

    results = {
        'config': {
            'name': actual_config.name,
            'source': actual_config.source,
            'num_features': len(actual_config.features),
            'features': actual_config.features,
            'd_model': actual_config.d_model,
            'nhead': actual_config.nhead,
            'num_layers': actual_config.num_layers
        }
    }

    # Get sample batch for IO inventory
    sample_batch = next(iter(dataloader))

    print("Analyzing input-output inventory...")
    results['io_inventory'] = interpreter.inventory_io(sample_batch)

    print("Analyzing attention patterns...")
    results['attention'] = interpreter.analyze_attention(dataloader)

    print("Analyzing embeddings...")
    results['embeddings'] = interpreter.analyze_embeddings()

    print("Analyzing uncertainty...")
    results['uncertainty'] = interpreter.analyze_uncertainty(dataloader)

    print("Tracing data flow...")
    results['data_flow'] = interpreter.trace_data_flow(sample_batch)

    if verbose:
        report = interpreter.generate_report(dataloader)
        print("\n" + report)

    return results


def analyze_all_models(verbose: bool = True) -> Dict[str, Any]:
    """Analyze all configured JIM models."""
    all_results = {}

    for name in INTERPOLATION_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print("=" * 60)

        try:
            results = analyze_trained_model(name, verbose=False)
            all_results[name] = results

            if verbose:
                # Print summary
                print(f"  Features: {results['config']['num_features']}")
                if results.get('uncertainty'):
                    print(f"  Uncertainty-Error Correlation: {results['uncertainty']['uncertainty_error_correlation']:.4f}")
                    print(f"  Calibration: {results['uncertainty']['calibration']}")
                if results.get('attention'):
                    top_corr = results['attention'].top_correlations[0] if results['attention'].top_correlations else ('N/A', 'N/A', 0)
                    print(f"  Top correlation: {top_corr[0]} <-> {top_corr[1]} ({top_corr[2]:.4f})")
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            all_results[name] = {'error': str(e)}

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='JIM Interpretability Analysis')
    parser.add_argument('--model', type=str, help='Model name to analyze (e.g., sentinel2)')
    parser.add_argument('--all', action='store_true', help='Analyze all models')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--save', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required for analysis")
        return

    if args.all:
        results = analyze_all_models(verbose=not args.quiet)
    elif args.model:
        results = analyze_trained_model(args.model, verbose=not args.quiet)
    else:
        # Default: analyze first available model
        print("No model specified. Available models:")
        for name, config in INTERPOLATION_CONFIGS.items():
            print(f"  {name}: {config.name}")
        print("\nUse --model <name> or --all")
        return

    if args.save:
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, ModuleIO):
                return {
                    'name': obj.name,
                    'input_shape': str(obj.input_shape),
                    'output_shape': str(obj.output_shape),
                    'parameters': obj.parameters,
                    'description': obj.description
                }
            elif isinstance(obj, AttentionAnalysis):
                return {
                    'feature_names': obj.feature_names,
                    'top_correlations': obj.top_correlations,
                    'feature_importance': obj.feature_importance.tolist()
                }
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif isinstance(obj, tuple):
                return [make_serializable(i) for i in obj]
            return obj

        serializable = make_serializable(results)
        with open(args.save, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
