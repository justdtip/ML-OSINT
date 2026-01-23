#!/usr/bin/env python3
"""
Stage 3: Latent Factor Extraction

Extracts latent factors from the cross-source residual covariance matrix.
These factors represent empirical "omitted variables" that affect multiple sources.

The factors are characterized by:
- Which sources they most affect (loading patterns)
- Temporal dynamics (persistence, trends, volatility)
- Potential interpretations based on their properties
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from .residual_extraction import ResidualExtractionResults


@dataclass
class FactorExtractionResults:
    """Container for factor extraction outputs."""
    factor_scores: np.ndarray  # (n_samples, n_factors)
    factor_loadings: np.ndarray  # (n_features, n_factors)
    variance_explained: np.ndarray  # (n_factors,)
    source_loadings: Dict[str, np.ndarray]  # source -> mean abs loading per factor
    feature_info: List[Tuple[str, int, str]]  # (source, idx, name) for each feature
    factor_characterizations: List[Dict]  # Per-factor analysis
    metadata: Dict = field(default_factory=dict)

    def save(self, output_dir: Path):
        """Save results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.savez(
            output_dir / 'latent_factors.npz',
            factor_scores=self.factor_scores,
            factor_loadings=self.factor_loadings,
            variance_explained=self.variance_explained
        )

        # Convert for JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        results = {
            'source_loadings': convert_for_json(self.source_loadings),
            'feature_info': convert_for_json(self.feature_info),
            'factor_characterizations': convert_for_json(self.factor_characterizations),
            'variance_explained': convert_for_json(self.variance_explained),
            'metadata': convert_for_json(self.metadata)
        }

        with open(output_dir / 'factor_characterization.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved factor extraction results to {output_dir}")

    @classmethod
    def load(cls, output_dir: Path) -> 'FactorExtractionResults':
        """Load results from disk."""
        output_dir = Path(output_dir)

        arrays = np.load(output_dir / 'latent_factors.npz')

        with open(output_dir / 'factor_characterization.json') as f:
            data = json.load(f)

        # Convert feature_info back to tuples
        feature_info = [tuple(f) for f in data['feature_info']]

        # Convert source_loadings values to arrays
        source_loadings = {k: np.array(v) for k, v in data['source_loadings'].items()}

        return cls(
            factor_scores=arrays['factor_scores'],
            factor_loadings=arrays['factor_loadings'],
            variance_explained=arrays['variance_explained'],
            source_loadings=source_loadings,
            feature_info=feature_info,
            factor_characterizations=data['factor_characterizations'],
            metadata=data.get('metadata', {})
        )


def construct_residual_matrix(
    residuals: Dict[str, np.ndarray],
    feature_names: Dict[str, List[str]]
) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """
    Stack residuals from all sources into a single matrix.

    Args:
        residuals: Dict mapping source names to residual arrays (n_samples, n_features)
        feature_names: Dict mapping source names to feature name lists

    Returns:
        R: Combined matrix (n_samples, total_features)
        feature_info: List of (source, feature_idx, feature_name) tuples
    """
    sources = list(residuals.keys())

    # Stack horizontally
    R = np.hstack([residuals[s] for s in sources])

    # Track which features belong to which source
    feature_info = []
    for source in sources:
        n_features = residuals[source].shape[1]
        names = feature_names.get(source, [f'feat_{i}' for i in range(n_features)])
        for i in range(n_features):
            name = names[i] if i < len(names) else f'feat_{i}'
            feature_info.append((source, i, name))

    return R, feature_info


def extract_residual_factors(
    residuals: Dict[str, np.ndarray],
    feature_names: Dict[str, List[str]],
    n_factors: int = 5,
    method: str = 'pca'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], List[Tuple[str, int, str]]]:
    """
    Extract latent factors from residual covariance structure.

    Args:
        residuals: Dict of residual arrays
        feature_names: Dict of feature name lists
        n_factors: Number of factors to extract
        method: 'pca' or 'factor_analysis'

    Returns:
        scores: Factor time series (n_samples, n_factors)
        loadings: Factor loadings (n_features, n_factors)
        variance_explained: Variance explained per factor
        source_loadings: Mean absolute loading per source per factor
        feature_info: Feature metadata
    """
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    R, feature_info = construct_residual_matrix(residuals, feature_names)
    sources = list(residuals.keys())

    # Standardize
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)

    # Handle NaN/Inf
    R_scaled = np.nan_to_num(R_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Extract factors
    if method == 'pca':
        model = PCA(n_components=n_factors)
        scores = model.fit_transform(R_scaled)
        loadings = model.components_.T  # (n_features, n_factors)
        variance_explained = model.explained_variance_ratio_
    else:
        model = FactorAnalysis(n_components=n_factors, random_state=42)
        scores = model.fit_transform(R_scaled)
        loadings = model.components_.T
        # Approximate variance explained
        total_var = R_scaled.var(axis=0).sum()
        factor_var = np.var(scores, axis=0)
        variance_explained = factor_var / max(total_var, 1e-10)

    # Compute mean absolute loading per source per factor
    source_loadings = {s: np.zeros(n_factors) for s in sources}
    feature_counts = {s: 0 for s in sources}

    for idx, (source, feat_idx, feat_name) in enumerate(feature_info):
        for f in range(n_factors):
            source_loadings[source][f] += np.abs(loadings[idx, f])
        feature_counts[source] += 1

    # Normalize by number of features per source
    for s in sources:
        if feature_counts[s] > 0:
            source_loadings[s] /= feature_counts[s]

    return scores, loadings, variance_explained, source_loadings, feature_info


def characterize_factors(
    factor_scores: np.ndarray,
    factor_loadings: np.ndarray,
    variance_explained: np.ndarray,
    source_loadings: Dict[str, np.ndarray],
    feature_info: List[Tuple[str, int, str]],
    timestamps: List[str]
) -> List[Dict]:
    """
    Characterize each extracted factor:
    - Which sources it most affects
    - Temporal dynamics
    - Top contributing features
    - Potential interpretations

    Returns:
        List of characterization dicts, one per factor
    """
    n_factors = factor_scores.shape[1]
    sources = list(source_loadings.keys())
    characterizations = []

    for f in range(n_factors):
        factor_score = factor_scores[:, f]

        # Source sensitivity (which sources does this factor affect most?)
        sensitivity = {s: float(source_loadings[s][f]) for s in sources}
        dominant_source = max(sensitivity, key=sensitivity.get)

        # Top contributing features for this factor
        feature_contributions = []
        for idx, (source, feat_idx, feat_name) in enumerate(feature_info):
            loading = factor_loadings[idx, f]
            feature_contributions.append({
                'source': source,
                'feature': feat_name,
                'loading': float(loading),
                'abs_loading': float(abs(loading))
            })
        feature_contributions.sort(key=lambda x: x['abs_loading'], reverse=True)
        top_features = feature_contributions[:10]

        # Temporal properties
        # Lag-1 autocorrelation
        if len(factor_score) > 1:
            autocorr_lag1 = float(np.corrcoef(factor_score[:-1], factor_score[1:])[0, 1])
        else:
            autocorr_lag1 = 0.0

        # Trend (Spearman correlation with time)
        time_indices = np.arange(len(factor_score))
        trend_corr, trend_pval = stats.spearmanr(time_indices, factor_score)
        trend_corr = float(trend_corr) if not np.isnan(trend_corr) else 0.0
        trend_pval = float(trend_pval) if not np.isnan(trend_pval) else 1.0

        # Volatility clustering (autocorrelation of squared scores)
        squared_scores = factor_score ** 2
        if len(squared_scores) > 1:
            volatility_persistence = float(np.corrcoef(squared_scores[:-1], squared_scores[1:])[0, 1])
        else:
            volatility_persistence = 0.0

        # Statistics
        factor_mean = float(np.mean(factor_score))
        factor_std = float(np.std(factor_score))
        factor_skew = float(stats.skew(factor_score))
        factor_kurtosis = float(stats.kurtosis(factor_score))

        # Interpretation
        interpretation = interpret_factor(
            sensitivity, autocorr_lag1, trend_corr, volatility_persistence,
            factor_skew, top_features
        )

        characterizations.append({
            'factor_index': f,
            'variance_explained': float(variance_explained[f]),
            'cumulative_variance': float(np.sum(variance_explained[:f+1])),
            'source_sensitivity': sensitivity,
            'dominant_source': dominant_source,
            'top_features': top_features,
            'temporal_properties': {
                'autocorrelation_lag1': autocorr_lag1,
                'trend_correlation': trend_corr,
                'trend_pvalue': trend_pval,
                'volatility_persistence': volatility_persistence
            },
            'statistics': {
                'mean': factor_mean,
                'std': factor_std,
                'skewness': factor_skew,
                'kurtosis': factor_kurtosis
            },
            'interpretation': interpretation
        })

    return characterizations


def interpret_factor(
    source_sensitivity: Dict[str, float],
    autocorr: float,
    trend: float,
    volatility: float,
    skewness: float,
    top_features: List[Dict]
) -> List[str]:
    """
    Generate interpretation based on factor properties.
    """
    interpretations = []

    # Source pattern
    sensitivities = list(source_sensitivity.values())
    max_sens = max(sensitivities)
    median_sens = np.median(sensitivities)

    if max_sens > 2 * median_sens:
        dominant = max(source_sensitivity, key=source_sensitivity.get)
        interpretations.append(f'primarily_affects_{dominant}')
    else:
        interpretations.append('affects_multiple_sources_equally')

    # Check if factor is dominated by specific source types
    source_types = set(f['source'] for f in top_features[:5])
    if len(source_types) == 1:
        interpretations.append(f'source_specific_{list(source_types)[0]}')
    elif len(source_types) == 2:
        interpretations.append(f'shared_between_{"-".join(sorted(source_types))}')

    # Temporal pattern
    if autocorr > 0.7:
        interpretations.append('highly_persistent')
    elif autocorr > 0.3:
        interpretations.append('moderately_persistent')
    elif autocorr < -0.3:
        interpretations.append('oscillating')
    else:
        interpretations.append('transient')

    # Trend
    if abs(trend) > 0.5:
        direction = 'increasing' if trend > 0 else 'decreasing'
        interpretations.append(f'strongly_trending_{direction}')
    elif abs(trend) > 0.3:
        direction = 'increasing' if trend > 0 else 'decreasing'
        interpretations.append(f'trending_{direction}')

    # Volatility
    if volatility > 0.5:
        interpretations.append('volatility_clustering')

    # Distribution shape
    if abs(skewness) > 1:
        direction = 'right' if skewness > 0 else 'left'
        interpretations.append(f'skewed_{direction}')

    # Feature-based interpretation
    feature_keywords = []
    for f in top_features[:5]:
        name = f['feature'].lower()
        if 'fire' in name or 'frp' in name or 'brightness' in name:
            feature_keywords.append('fire_activity')
        elif 'death' in name or 'casualties' in name or 'lethality' in name:
            feature_keywords.append('casualties')
        elif 'territory' in name or 'area' in name or 'control' in name:
            feature_keywords.append('territorial')
        elif 'event' in name or 'conflict' in name:
            feature_keywords.append('conflict_events')
        elif 'tank' in name or 'aircraft' in name or 'artillery' in name or 'equipment' in name:
            feature_keywords.append('military_equipment')
        elif 'arrow' in name or 'direction' in name:
            feature_keywords.append('movement_direction')

    unique_keywords = list(set(feature_keywords))
    if unique_keywords:
        interpretations.append(f'related_to_{"-".join(unique_keywords[:2])}')

    return interpretations


def run_factor_extraction(
    residual_results: ResidualExtractionResults,
    n_factors: int = 5,
    method: str = 'pca',
    use_masked: bool = True
) -> FactorExtractionResults:
    """
    Run complete factor extraction on residuals.

    Args:
        residual_results: Results from Stage 1
        n_factors: Number of factors to extract
        method: 'pca' or 'factor_analysis'
        use_masked: Use masked reconstruction residuals (recommended)

    Returns:
        FactorExtractionResults
    """
    print("\n" + "=" * 70)
    print("STAGE 3: LATENT FACTOR EXTRACTION")
    print("=" * 70)

    residuals = residual_results.masked_residuals if use_masked else residual_results.full_residuals
    residual_type = "masked" if use_masked else "full"

    print(f"\nExtracting {n_factors} factors from {residual_type} residuals")
    print(f"Method: {method.upper()}")
    print(f"Sources: {list(residuals.keys())}")

    # Count total features
    total_features = sum(r.shape[1] for r in residuals.values())
    n_samples = len(residual_results.timestamps)
    print(f"Total features: {total_features}")
    print(f"Samples: {n_samples}")

    # Extract factors
    print("\n--- Extracting Latent Factors ---")
    scores, loadings, var_explained, source_loadings, feature_info = extract_residual_factors(
        residuals,
        residual_results.feature_names,
        n_factors=n_factors,
        method=method
    )

    print(f"\nVariance explained by factors:")
    cumulative = 0
    for i, ve in enumerate(var_explained):
        cumulative += ve
        print(f"  Factor {i+1}: {ve*100:.1f}% (cumulative: {cumulative*100:.1f}%)")

    # Characterize factors
    print("\n--- Characterizing Factors ---")
    characterizations = characterize_factors(
        scores, loadings, var_explained, source_loadings,
        feature_info, residual_results.timestamps
    )

    for char in characterizations:
        f_idx = char['factor_index']
        print(f"\nFactor {f_idx + 1}:")
        print(f"  Variance explained: {char['variance_explained']*100:.1f}%")
        print(f"  Dominant source: {char['dominant_source']}")
        print(f"  Persistence (lag-1 ACF): {char['temporal_properties']['autocorrelation_lag1']:.2f}")
        print(f"  Trend correlation: {char['temporal_properties']['trend_correlation']:.2f}")
        print(f"  Top features:")
        for feat in char['top_features'][:3]:
            print(f"    {feat['source']}/{feat['feature']}: {feat['loading']:.3f}")
        print(f"  Interpretation: {', '.join(char['interpretation'][:3])}")

    # Compile metadata
    metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'residual_type': residual_type,
        'n_factors': n_factors,
        'method': method,
        'n_samples': n_samples,
        'total_features': total_features,
        'total_variance_explained': float(np.sum(var_explained)),
        'sources': list(residuals.keys())
    }

    return FactorExtractionResults(
        factor_scores=scores,
        factor_loadings=loadings,
        variance_explained=var_explained,
        source_loadings=source_loadings,
        feature_info=feature_info,
        factor_characterizations=characterizations,
        metadata=metadata
    )


def generate_factor_plots(
    results: FactorExtractionResults,
    timestamps: List[str],
    output_dir: Path
):
    """Generate visualization plots for factor analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_factors = results.factor_scores.shape[1]
    sources = list(results.source_loadings.keys())

    # Color palette
    colors = {
        'deepstate': '#2ecc71',
        'equipment': '#3498db',
        'firms': '#e74c3c',
        'ucdp': '#9b59b6'
    }

    # 1. Factor Scores Time Series
    fig, axes = plt.subplots(n_factors, 1, figsize=(14, 3 * n_factors), sharex=True)
    if n_factors == 1:
        axes = [axes]

    # Parse timestamps for x-axis
    try:
        from datetime import datetime as dt
        dates = [dt.strptime(t, '%Y-%m-%d') for t in timestamps]
        x_axis = dates
        use_dates = True
    except Exception:
        x_axis = range(len(timestamps))
        use_dates = False

    for i, ax in enumerate(axes):
        scores = results.factor_scores[:, i]
        var_exp = results.variance_explained[i]

        ax.plot(x_axis, scores, 'b-', linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(x_axis, 0, scores, alpha=0.3,
                       where=(scores > 0), color='blue')
        ax.fill_between(x_axis, 0, scores, alpha=0.3,
                       where=(scores < 0), color='red')

        ax.set_ylabel(f'Factor {i+1}\n({var_exp*100:.1f}%)')
        ax.grid(True, alpha=0.3)

        # Add dominant source annotation
        char = results.factor_characterizations[i]
        ax.text(0.02, 0.95, f"Dominant: {char['dominant_source']}",
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel('Date' if use_dates else 'Sample')
    if use_dates:
        fig.autofmt_xdate()

    plt.suptitle('Latent Factor Time Series', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_timeseries.png', dpi=150)
    plt.close()
    print(f"  Saved: factor_timeseries.png")

    # 2. Source Loadings Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create matrix: sources x factors
    loading_matrix = np.array([results.source_loadings[s] for s in sources])

    im = ax.imshow(loading_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_factors))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels([f'Factor {i+1}' for i in range(n_factors)])
    ax.set_yticklabels([s.upper() for s in sources])

    # Add values as text
    for i in range(len(sources)):
        for j in range(n_factors):
            text = ax.text(j, i, f'{loading_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=10,
                          color='white' if loading_matrix[i, j] > 0.3 else 'black')

    plt.colorbar(im, label='Mean Absolute Loading')
    ax.set_title('Factor Loadings by Source')
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_loadings_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: factor_loadings_heatmap.png")

    # 3. Source Sensitivity Radar Chart
    fig, axes = plt.subplots(1, min(n_factors, 3), figsize=(5 * min(n_factors, 3), 5),
                            subplot_kw=dict(projection='polar'))
    if n_factors == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(sources), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for i, ax in enumerate(axes[:min(n_factors, 3)]):
        values = [results.source_loadings[s][i] for s in sources]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, 'b-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([s.upper() for s in sources])
        ax.set_title(f'Factor {i+1}\n({results.variance_explained[i]*100:.1f}%)')

    plt.suptitle('Source Sensitivity per Factor', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'source_sensitivity_radar.png', dpi=150)
    plt.close()
    print(f"  Saved: source_sensitivity_radar.png")

    # 4. Variance Explained Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(n_factors)
    individual = results.variance_explained * 100
    cumulative = np.cumsum(results.variance_explained) * 100

    bars = ax.bar(x, individual, color='steelblue', alpha=0.7, label='Individual')
    ax.plot(x, cumulative, 'ro-', markersize=8, linewidth=2, label='Cumulative')

    ax.set_xlabel('Factor')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'F{i+1}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Variance Explained by Latent Factors')

    # Add percentage labels on bars
    for bar, pct in zip(bars, individual):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_explained.png', dpi=150)
    plt.close()
    print(f"  Saved: variance_explained.png")

    # 5. Top Feature Loadings per Factor
    fig, axes = plt.subplots(1, min(n_factors, 3), figsize=(6 * min(n_factors, 3), 8))
    if n_factors == 1:
        axes = [axes]

    for i, ax in enumerate(axes[:min(n_factors, 3)]):
        char = results.factor_characterizations[i]
        top_feats = char['top_features'][:10]

        names = [f"{f['source'][:3]}/{f['feature'][:15]}" for f in top_feats]
        loadings = [f['loading'] for f in top_feats]
        colors_list = ['green' if l > 0 else 'red' for l in loadings]

        y_pos = range(len(names))
        ax.barh(y_pos, loadings, color=colors_list, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Loading')
        ax.set_title(f'Factor {i+1} Top Features')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'top_feature_loadings.png', dpi=150)
    plt.close()
    print(f"  Saved: top_feature_loadings.png")


def main():
    """Run Stage 3: Latent Factor Extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3: Latent Factor Extraction')
    parser.add_argument('--input-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'results'),
                       help='Directory containing Stage 1 results')
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'results'),
                       help='Output directory for results')
    parser.add_argument('--figures-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'figures'),
                       help='Output directory for figures')
    parser.add_argument('--n-factors', type=int, default=5,
                       help='Number of factors to extract')
    parser.add_argument('--method', type=str, default='pca',
                       choices=['pca', 'factor_analysis'],
                       help='Factor extraction method')
    parser.add_argument('--use-full', action='store_true',
                       help='Use full reconstruction residuals instead of masked')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 3: LATENT FACTOR EXTRACTION")
    print("=" * 70)
    print("\nExtracting latent factors from residual covariance structure.")
    print("Factors represent empirical 'omitted variables' affecting multiple sources.\n")

    # Load Stage 1 results
    print("Loading residuals from Stage 1...")
    input_dir = Path(args.input_dir)
    residual_results = ResidualExtractionResults.load(input_dir)
    print(f"  Loaded {residual_results.metadata.get('n_samples', 'unknown')} samples")

    # Run factor extraction
    results = run_factor_extraction(
        residual_results,
        n_factors=args.n_factors,
        method=args.method,
        use_masked=not args.use_full
    )

    # Save results
    output_dir = Path(args.output_dir)
    results.save(output_dir)

    # Generate plots
    if not args.skip_plots:
        print("\n--- Generating Plots ---")
        figures_dir = Path(args.figures_dir)
        generate_factor_plots(results, residual_results.timestamps, figures_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("FACTOR EXTRACTION COMPLETE")
    print("=" * 70)

    print("\n--- Summary ---")
    print(f"Extracted {args.n_factors} factors explaining {results.metadata['total_variance_explained']*100:.1f}% of variance")

    print("\n--- Factor Interpretations ---")
    for char in results.factor_characterizations:
        f_idx = char['factor_index']
        print(f"\nFactor {f_idx + 1} ({char['variance_explained']*100:.1f}% variance):")
        print(f"  Dominant: {char['dominant_source']}")
        print(f"  Properties: {', '.join(char['interpretation'][:4])}")

    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {args.figures_dir}")
    print("\nNext step: Run Stage 4 (Candidate Variable Correlation)")


if __name__ == '__main__':
    main()
