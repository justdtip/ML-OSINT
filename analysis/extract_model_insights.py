#!/usr/bin/env python3
"""
Model Insights Extraction

Extracts interpretable insights from trained JIM and Unified models:
1. Cross-feature correlations learned by JIM models
2. Cross-source relationships learned by Unified model
3. Temporal patterns and seasonality
4. Source importance rankings
5. Feature importance within each source
6. Prediction confidence patterns

Output: Detailed text report + summary figures
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import json

ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

from config.paths import NETWORK_OUTPUT_DIR, MODEL_DIR, INTERP_MODEL_DIR

FIGURES_DIR = NETWORK_OUTPUT_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Source colors
SOURCE_COLORS = {
    'sentinel': '#3498db',
    'deepstate': '#2ecc71',
    'equipment': '#e74c3c',
    'firms': '#f39c12',
    'ucdp': '#9b59b6',
}

SOURCE_DESCRIPTIONS = {
    'sentinel': 'Satellite imagery (SAR, optical, atmospheric)',
    'deepstate': 'Front line positions, military units, territory control',
    'equipment': 'Russian equipment losses (tanks, aircraft, etc.)',
    'firms': 'Fire/thermal anomaly detections',
    'ucdp': 'Conflict events and casualties',
}


class ModelInsightsExtractor:
    """Extract interpretable insights from trained models."""

    def __init__(self):
        self.jim_states = {}
        self.unified_state = None
        self.insights = {
            'jim': {},
            'unified': {},
            'cross_source': {},
            'summary': {}
        }

    def load_models(self):
        """Load all trained model states."""
        print("=" * 70)
        print("LOADING MODELS FOR INSIGHT EXTRACTION")
        print("=" * 70)

        # Load JIM models
        for path in INTERP_MODEL_DIR.glob("interp_*_best.pt"):
            name = path.stem.replace('interp_', '').replace('_best', '')
            self.jim_states[name] = torch.load(path, map_location='cpu')

        print(f"Loaded {len(self.jim_states)} JIM models")

        # Load unified model
        unified_path = MODEL_DIR / "unified_interpolation_best.pt"
        if unified_path.exists():
            self.unified_state = torch.load(unified_path, map_location='cpu')
            print("Loaded unified cross-source model")

    def extract_jim_insights(self):
        """Extract insights from JIM models."""
        print("\n" + "=" * 70)
        print("EXTRACTING JIM MODEL INSIGHTS")
        print("=" * 70)

        # Group by source
        source_models = defaultdict(list)
        for name, state in self.jim_states.items():
            for src in SOURCE_COLORS.keys():
                if src in name.lower():
                    source_models[src].append((name, state))
                    break

        for source, models in source_models.items():
            print(f"\n--- {source.upper()} ({len(models)} models) ---")
            self.insights['jim'][source] = self._analyze_source_models(source, models)

    def _analyze_source_models(self, source: str, models: List[Tuple[str, dict]]) -> Dict:
        """Analyze all models for a specific source."""
        insights = {
            'n_models': len(models),
            'total_parameters': 0,
            'feature_correlations': [],
            'temporal_patterns': [],
            'uncertainty_characteristics': {},
            'model_details': []
        }

        all_attention_patterns = []
        all_embedding_norms = []
        all_uncertainty_biases = []

        for name, state in models:
            model_info = {'name': name}

            # Count parameters
            n_params = sum(v.numel() for v in state.values())
            insights['total_parameters'] += n_params
            model_info['parameters'] = n_params

            # Extract attention patterns
            for key in state:
                if 'self_attn.in_proj_weight' in key:
                    w = state[key].numpy()
                    # Compute attention correlation structure
                    d_model = w.shape[0] // 3
                    Q, K, V = w[:d_model], w[d_model:2*d_model], w[2*d_model:]

                    # Q-K correlation indicates learned feature relationships
                    qk_corr = np.abs(Q @ K.T).mean()
                    all_attention_patterns.append(qk_corr)
                    model_info['attention_strength'] = float(qk_corr)

                # Feature embeddings
                if 'feature_embeddings.weight' in key:
                    emb = state[key].numpy()
                    norms = np.linalg.norm(emb, axis=1)
                    all_embedding_norms.extend(norms)
                    model_info['n_features'] = emb.shape[0]
                    model_info['embedding_dim'] = emb.shape[1]

                    # Find which features have strongest embeddings
                    top_indices = np.argsort(norms)[-3:]
                    model_info['strongest_features'] = top_indices.tolist()

                # Uncertainty head
                if 'uncertainty_head' in key and 'bias' in key and state[key].numel() == 1:
                    bias = state[key].item()
                    all_uncertainty_biases.append(bias)
                    model_info['uncertainty_bias'] = bias

            insights['model_details'].append(model_info)

        # Aggregate statistics
        if all_attention_patterns:
            insights['mean_attention_strength'] = float(np.mean(all_attention_patterns))
            insights['attention_std'] = float(np.std(all_attention_patterns))

        if all_embedding_norms:
            insights['mean_embedding_norm'] = float(np.mean(all_embedding_norms))
            insights['embedding_norm_std'] = float(np.std(all_embedding_norms))

        if all_uncertainty_biases:
            insights['mean_uncertainty_bias'] = float(np.mean(all_uncertainty_biases))
            # Negative bias = model tends to be overconfident
            # Positive bias = model tends to be underconfident
            if np.mean(all_uncertainty_biases) < -0.1:
                insights['uncertainty_interpretation'] = "Models tend to be OVERCONFIDENT"
            elif np.mean(all_uncertainty_biases) > 0.1:
                insights['uncertainty_interpretation'] = "Models tend to be UNDERCONFIDENT"
            else:
                insights['uncertainty_interpretation'] = "Models have balanced confidence"

        return insights

    def extract_unified_insights(self):
        """Extract insights from the unified cross-source model."""
        if not self.unified_state:
            print("\nNo unified model loaded")
            return

        print("\n" + "=" * 70)
        print("EXTRACTING UNIFIED MODEL INSIGHTS")
        print("=" * 70)

        insights = {
            'total_parameters': sum(v.numel() for v in self.unified_state.values()),
            'source_relationships': {},
            'source_importance': {},
            'cross_source_patterns': []
        }

        # Extract source embeddings
        source_emb = None
        for key in self.unified_state:
            if 'source_embeddings.weight' in key:
                source_emb = self.unified_state[key].numpy()
                break

        if source_emb is not None:
            sources = ['sentinel', 'deepstate', 'equipment', 'firms', 'ucdp']

            # Compute source similarity matrix
            if HAS_SKLEARN:
                sim = cosine_similarity(source_emb)
            else:
                norms = np.linalg.norm(source_emb, axis=1, keepdims=True)
                sim = (source_emb / (norms + 1e-8)) @ (source_emb / (norms + 1e-8)).T

            print("\n  SOURCE SIMILARITY MATRIX (Cosine Similarity):")
            print("  " + "-" * 60)
            header = "           " + "  ".join([f"{s[:6]:>8}" for s in sources])
            print(f"  {header}")

            for i, src1 in enumerate(sources):
                row = f"  {src1:>8}  "
                for j, src2 in enumerate(sources):
                    row += f"{sim[i,j]:>8.3f}  "
                print(row)

            # Find strongest relationships
            relationships = []
            for i in range(len(sources)):
                for j in range(i+1, len(sources)):
                    relationships.append((sources[i], sources[j], sim[i, j]))

            relationships.sort(key=lambda x: -x[2])
            insights['source_relationships'] = relationships

            print("\n  STRONGEST CROSS-SOURCE RELATIONSHIPS:")
            for src1, src2, score in relationships[:5]:
                interpretation = self._interpret_relationship(src1, src2, score)
                print(f"    {src1.upper()} <-> {src2.upper()}: {score:.3f}")
                print(f"      Interpretation: {interpretation}")
                insights['cross_source_patterns'].append({
                    'sources': (src1, src2),
                    'similarity': float(score),
                    'interpretation': interpretation
                })

            # Source importance (based on embedding norm)
            norms = np.linalg.norm(source_emb, axis=1)
            importance_order = np.argsort(-norms)
            print("\n  SOURCE IMPORTANCE RANKING (by embedding magnitude):")
            for rank, idx in enumerate(importance_order):
                print(f"    {rank+1}. {sources[idx].upper()}: {norms[idx]:.3f}")
                insights['source_importance'][sources[idx]] = {
                    'rank': rank + 1,
                    'magnitude': float(norms[idx])
                }

        # Analyze encoder/decoder asymmetry
        encoder_params = defaultdict(int)
        decoder_params = defaultdict(int)

        for key in self.unified_state:
            for src in SOURCE_COLORS.keys():
                if src in key:
                    if 'encoder' in key:
                        encoder_params[src] += self.unified_state[key].numel()
                    elif 'decoder' in key:
                        decoder_params[src] += self.unified_state[key].numel()
                    break

        print("\n  ENCODER/DECODER PARAMETER DISTRIBUTION:")
        for src in SOURCE_COLORS.keys():
            enc = encoder_params.get(src, 0)
            dec = decoder_params.get(src, 0)
            ratio = enc / max(dec, 1)
            print(f"    {src.upper()}: Encoder={enc//1000}K, Decoder={dec//1000}K, Ratio={ratio:.2f}")

        self.insights['unified'] = insights

    def _interpret_relationship(self, src1: str, src2: str, score: float) -> str:
        """Provide interpretation for cross-source relationships."""
        interpretations = {
            ('sentinel', 'firms'): "Satellite thermal signatures correlate with fire detections",
            ('sentinel', 'deepstate'): "Satellite imagery reflects territorial changes",
            ('sentinel', 'equipment'): "Damage visible in satellite correlates with equipment losses",
            ('sentinel', 'ucdp'): "Satellite detects aftermath of conflict events",
            ('firms', 'deepstate'): "Fire activity concentrated along front lines",
            ('firms', 'equipment'): "Fires often accompany equipment destruction",
            ('firms', 'ucdp'): "Fire detections correlate with combat events",
            ('deepstate', 'equipment'): "Territorial gains/losses correlate with equipment attrition",
            ('deepstate', 'ucdp'): "Front line changes driven by documented conflict events",
            ('equipment', 'ucdp'): "Equipment losses reflect intensity of documented combat",
        }

        key = tuple(sorted([src1, src2]))
        base = interpretations.get(key, f"Learned correlation between {src1} and {src2}")

        if score > 0.8:
            return f"STRONG: {base}"
        elif score > 0.5:
            return f"MODERATE: {base}"
        elif score > 0.2:
            return f"WEAK: {base}"
        else:
            return f"MINIMAL/INVERSE: Sources capture distinct information"

    def extract_cross_source_insights(self):
        """Extract insights about cross-source relationships."""
        print("\n" + "=" * 70)
        print("CROSS-SOURCE PATTERN ANALYSIS")
        print("=" * 70)

        # Compare JIM model characteristics across sources
        source_characteristics = {}

        for source, data in self.insights['jim'].items():
            source_characteristics[source] = {
                'n_models': data['n_models'],
                'params_per_model': data['total_parameters'] / max(data['n_models'], 1),
                'attention_strength': data.get('mean_attention_strength', 0),
                'uncertainty': data.get('mean_uncertainty_bias', 0),
            }

        print("\n  SOURCE CHARACTERISTIC COMPARISON:")
        print("  " + "-" * 70)
        print(f"  {'Source':<12} {'Models':>8} {'Params/Model':>14} {'Attn Strength':>14} {'Uncertainty':>12}")
        print("  " + "-" * 70)

        for src in SOURCE_COLORS.keys():
            if src in source_characteristics:
                c = source_characteristics[src]
                print(f"  {src.upper():<12} {c['n_models']:>8} {c['params_per_model']/1000:>12.1f}K "
                      f"{c['attention_strength']:>14.4f} {c['uncertainty']:>12.3f}")

        # Identify which sources have strongest learned patterns
        print("\n  KEY FINDINGS:")

        # Strongest attention (most internal feature correlation)
        if source_characteristics:
            strongest_attn = max(source_characteristics.items(),
                               key=lambda x: x[1].get('attention_strength', 0))
            print(f"    - Strongest internal correlations: {strongest_attn[0].upper()}")
            print(f"      (Features within {strongest_attn[0]} have strong learned relationships)")

            # Most uncertain
            uncertainties = [(s, c.get('uncertainty', 0)) for s, c in source_characteristics.items()]
            most_uncertain = max(uncertainties, key=lambda x: x[1])
            most_confident = min(uncertainties, key=lambda x: x[1])

            if most_uncertain[1] > 0:
                print(f"    - Most uncertain predictions: {most_uncertain[0].upper()}")
                print(f"      (Model knows when {most_uncertain[0]} data is unreliable)")

            if most_confident[1] < 0:
                print(f"    - Most confident predictions: {most_confident[0].upper()}")
                print(f"      (May need calibration - risk of overconfidence)")

        self.insights['cross_source'] = source_characteristics

    def generate_summary(self):
        """Generate executive summary of insights."""
        print("\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)

        summary = []

        # Total model count and parameters
        total_jim = len(self.jim_states)
        total_jim_params = sum(
            self.insights['jim'].get(src, {}).get('total_parameters', 0)
            for src in SOURCE_COLORS.keys()
        )
        unified_params = self.insights.get('unified', {}).get('total_parameters', 0)

        summary.append(f"TRAINED MODELS: {total_jim} JIM models + 1 Unified model")
        summary.append(f"TOTAL PARAMETERS: {(total_jim_params + unified_params) / 1e6:.2f}M")
        summary.append("")

        # Key findings from JIM
        summary.append("JIM MODEL INSIGHTS:")
        for src in SOURCE_COLORS.keys():
            if src in self.insights['jim']:
                data = self.insights['jim'][src]
                n = data['n_models']
                interp = data.get('uncertainty_interpretation', 'Unknown')
                summary.append(f"  {src.upper()}: {n} models, {interp.lower()}")

        summary.append("")

        # Key findings from Unified
        if 'unified' in self.insights and self.insights['unified']:
            summary.append("UNIFIED MODEL INSIGHTS:")
            patterns = self.insights['unified'].get('cross_source_patterns', [])
            if patterns:
                summary.append("  Strongest learned cross-source relationships:")
                for p in patterns[:3]:
                    src1, src2 = p['sources']
                    summary.append(f"    - {src1.upper()} <-> {src2.upper()}: {p['interpretation']}")

            importance = self.insights['unified'].get('source_importance', {})
            if importance:
                ranked = sorted(importance.items(), key=lambda x: x[1]['rank'])
                top = [r[0].upper() for r in ranked[:2]]
                summary.append(f"  Most important sources: {', '.join(top)}")

        summary.append("")
        summary.append("ACTIONABLE INSIGHTS:")

        # Generate actionable insights
        if self.insights.get('unified', {}).get('cross_source_patterns'):
            patterns = self.insights['unified']['cross_source_patterns']
            if patterns and patterns[0]['similarity'] > 0.7:
                src1, src2 = patterns[0]['sources']
                summary.append(f"  1. {src1.upper()} and {src2.upper()} are highly correlated -")
                summary.append(f"     consider combining for improved predictions")

        # Uncertainty insights
        for src, data in self.insights.get('jim', {}).items():
            interp = data.get('uncertainty_interpretation', '')
            if 'OVERCONFIDENT' in interp:
                summary.append(f"  2. {src.upper()} models may be overconfident -")
                summary.append(f"     apply temperature scaling or recalibration")
                break

        for line in summary:
            print(f"  {line}")

        self.insights['summary'] = summary

    def create_insight_figures(self):
        """Create visualization figures for insights."""
        print("\n" + "=" * 70)
        print("GENERATING INSIGHT FIGURES")
        print("=" * 70)

        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Source model count and parameters
        ax1 = fig.add_subplot(gs[0, 0])
        sources = list(SOURCE_COLORS.keys())
        counts = [self.insights['jim'].get(s, {}).get('n_models', 0) for s in sources]
        colors = [SOURCE_COLORS[s] for s in sources]

        bars = ax1.bar(sources, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Models')
        ax1.set_title('JIM Models per Source', fontweight='bold')
        ax1.set_xticklabels([s.upper() for s in sources], rotation=45)

        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        str(count), ha='center', va='bottom', fontweight='bold')

        # 2. Attention strength by source
        ax2 = fig.add_subplot(gs[0, 1])
        attn_strengths = [self.insights['jim'].get(s, {}).get('mean_attention_strength', 0) for s in sources]

        bars = ax2.bar(sources, attn_strengths, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Mean Attention Strength')
        ax2.set_title('Learned Feature Correlations', fontweight='bold')
        ax2.set_xticklabels([s.upper() for s in sources], rotation=45)

        # 3. Uncertainty bias by source
        ax3 = fig.add_subplot(gs[0, 2])
        unc_biases = [self.insights['jim'].get(s, {}).get('mean_uncertainty_bias', 0) for s in sources]

        bar_colors = ['red' if b < -0.1 else 'green' if b > 0.1 else 'gray' for b in unc_biases]
        bars = ax3.bar(sources, unc_biases, color=bar_colors, alpha=0.7, edgecolor='black')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.axhline(-0.1, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.axhline(0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        ax3.set_ylabel('Uncertainty Bias')
        ax3.set_title('Model Confidence\n(negative=overconfident)', fontweight='bold')
        ax3.set_xticklabels([s.upper() for s in sources], rotation=45)

        # 4. Cross-source similarity matrix (if unified model exists)
        ax4 = fig.add_subplot(gs[1, 0])
        if self.unified_state is not None:
            # Find source embeddings
            for key in self.unified_state:
                if 'source_embeddings.weight' in key:
                    emb = self.unified_state[key].numpy()
                    if HAS_SKLEARN:
                        sim = cosine_similarity(emb)
                    else:
                        norms = np.linalg.norm(emb, axis=1, keepdims=True)
                        sim = (emb / (norms + 1e-8)) @ (emb / (norms + 1e-8)).T

                    im = ax4.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1)
                    ax4.set_xticks(range(5))
                    ax4.set_yticks(range(5))
                    ax4.set_xticklabels([s[:4].upper() for s in sources], fontsize=9)
                    ax4.set_yticklabels([s[:4].upper() for s in sources], fontsize=9)
                    ax4.set_title('Cross-Source Similarity', fontweight='bold')
                    plt.colorbar(im, ax=ax4, shrink=0.8)

                    # Annotate
                    for i in range(5):
                        for j in range(5):
                            ax4.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=8)
                    break
        else:
            ax4.text(0.5, 0.5, 'No unified model', ha='center', va='center')
            ax4.set_title('Cross-Source Similarity', fontweight='bold')

        # 5. Source importance ranking
        ax5 = fig.add_subplot(gs[1, 1])
        if self.insights.get('unified', {}).get('source_importance'):
            importance = self.insights['unified']['source_importance']
            sorted_sources = sorted(importance.items(), key=lambda x: x[1]['magnitude'], reverse=True)
            names = [s[0] for s in sorted_sources]
            mags = [s[1]['magnitude'] for s in sorted_sources]
            colors_sorted = [SOURCE_COLORS[s] for s in names]

            bars = ax5.barh(names, mags, color=colors_sorted, alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Embedding Magnitude')
            ax5.set_title('Source Importance\n(Unified Model)', fontweight='bold')
            ax5.set_yticklabels([s.upper() for s in names])
        else:
            ax5.text(0.5, 0.5, 'No importance data', ha='center', va='center')

        # 6. Relationship strength distribution
        ax6 = fig.add_subplot(gs[1, 2])
        if self.insights.get('unified', {}).get('source_relationships'):
            rels = self.insights['unified']['source_relationships']
            labels = [f"{r[0][:3]}-{r[1][:3]}" for r in rels]
            values = [r[2] for r in rels]
            colors_rel = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in values]

            bars = ax6.barh(labels, values, color=colors_rel, alpha=0.7, edgecolor='black')
            ax6.axvline(0.5, color='green', linestyle='--', alpha=0.5)
            ax6.axvline(0.2, color='orange', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Similarity')
            ax6.set_title('Cross-Source Relationships', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No relationship data', ha='center', va='center')

        # 7. Parameters distribution
        ax7 = fig.add_subplot(gs[2, 0])
        jim_params = sum(self.insights['jim'].get(s, {}).get('total_parameters', 0) for s in sources)
        unified_params = self.insights.get('unified', {}).get('total_parameters', 0)

        sizes = [jim_params, unified_params]
        labels = [f'JIM\n{jim_params//1000}K', f'Unified\n{unified_params//1000}K']
        colors_pie = ['#3498db', '#9b59b6']

        ax7.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax7.set_title('Parameter Distribution', fontweight='bold')

        # 8. Summary text
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')

        summary_text = "KEY INSIGHTS\n" + "=" * 50 + "\n\n"

        # Add key findings
        if self.insights.get('unified', {}).get('cross_source_patterns'):
            patterns = self.insights['unified']['cross_source_patterns']
            summary_text += "STRONGEST CROSS-SOURCE RELATIONSHIPS:\n"
            for p in patterns[:3]:
                src1, src2 = p['sources']
                summary_text += f"  {src1.upper()} <-> {src2.upper()}: {p['similarity']:.3f}\n"
            summary_text += "\n"

        # Uncertainty insights
        summary_text += "UNCERTAINTY ANALYSIS:\n"
        for src in sources:
            if src in self.insights.get('jim', {}):
                interp = self.insights['jim'][src].get('uncertainty_interpretation', 'Unknown')
                summary_text += f"  {src.upper()}: {interp}\n"

        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('ML_OSINT Model Insights Summary', fontsize=16, fontweight='bold')
        plt.savefig(FIGURES_DIR / '15_model_insights_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 15_model_insights_summary.png")

    def save_insights_json(self):
        """Save insights to JSON file."""
        output_path = FIGURES_DIR / 'model_insights.json'

        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj

        with open(output_path, 'w') as f:
            json.dump(convert(self.insights), f, indent=2)

        print(f"\n  Saved insights to: {output_path}")

    def run(self):
        """Run full insight extraction pipeline."""
        self.load_models()
        self.extract_jim_insights()
        self.extract_unified_insights()
        self.extract_cross_source_insights()
        self.generate_summary()
        self.create_insight_figures()
        self.save_insights_json()

        print("\n" + "=" * 70)
        print("INSIGHT EXTRACTION COMPLETE")
        print("=" * 70)


def main():
    extractor = ModelInsightsExtractor()
    extractor.run()


if __name__ == "__main__":
    main()
