"""
Stage 5: Granger Causality Testing

For candidate variables showing significant correlation in Stage 4,
test whether they Granger-cause the residuals (i.e., provide predictive
information beyond the residuals' own history).

Granger causality tests: "Does knowing Z at time t-k help predict the
part of Y that existing sources couldn't predict?"
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Ensure output directories exist
(OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)


@dataclass
class GrangerCausalityResults:
    """Results from Granger causality testing."""
    granger_results: List[Dict]  # Results for each tested pair
    summary_by_candidate: Dict[str, Dict]  # Aggregated results per candidate
    summary_by_source: Dict[str, Dict]  # Aggregated results per residual source
    optimal_lags: Dict[str, int]  # Best lag per candidate
    bidirectional_tests: List[Dict]  # Tests for reverse causality
    metadata: Dict

    def save(self, output_dir: Path):
        """Save results to files."""
        output_dir = Path(output_dir)

        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        results = {
            'granger_results': convert_for_json(self.granger_results),
            'summary_by_candidate': convert_for_json(self.summary_by_candidate),
            'summary_by_source': convert_for_json(self.summary_by_source),
            'optimal_lags': convert_for_json(self.optimal_lags),
            'bidirectional_tests': convert_for_json(self.bidirectional_tests),
            'metadata': convert_for_json(self.metadata)
        }

        with open(output_dir / "results" / "granger_causality.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved Granger causality results to {output_dir / 'results' / 'granger_causality.json'}")

    @classmethod
    def load(cls, output_dir: Path) -> 'GrangerCausalityResults':
        """Load results from files."""
        with open(output_dir / "results" / "granger_causality.json") as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# GRANGER CAUSALITY TESTS
# =============================================================================

def granger_causality_test(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 15,
    significance_level: float = 0.05
) -> Dict:
    """
    Test if x Granger-causes y.

    Uses statsmodels grangercausalitytests which performs:
    - F-test (ssr_ftest)
    - Chi-squared test (ssr_chi2test)
    - Likelihood ratio test (lrtest)
    - Parameter F-test (params_ftest)

    Args:
        y: Target time series (residual magnitude)
        x: Candidate time series
        max_lag: Maximum lag to test
        significance_level: Threshold for significance

    Returns:
        Dict with test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.stattools import adfuller

    # Ensure arrays are 1D
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()

    # Check for stationarity (Granger test assumes stationarity)
    try:
        adf_y = adfuller(y, maxlag=max_lag)
        adf_x = adfuller(x, maxlag=max_lag)
        y_stationary = adf_y[1] < 0.05
        x_stationary = adf_x[1] < 0.05
    except:
        y_stationary = True
        x_stationary = True

    # If non-stationary, difference the series
    if not y_stationary:
        y = np.diff(y)
    if not x_stationary:
        x = np.diff(x)

    # Ensure same length after differencing
    min_len = min(len(y), len(x))
    y = y[-min_len:]
    x = x[-min_len:]

    # Stack for grangercausalitytests (expects [y, x] matrix)
    data = np.column_stack([y, x])

    # Run Granger tests for each lag
    results_by_lag = {}
    best_lag = 1
    best_pvalue = 1.0

    try:
        # grangercausalitytests returns results for lags 1 to max_lag
        gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        for lag in range(1, max_lag + 1):
            if lag in gc_results:
                # Extract F-test p-value (most commonly used)
                f_test = gc_results[lag][0]['ssr_ftest']
                f_stat, p_value, df_denom, df_num = f_test

                results_by_lag[lag] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'df_denom': int(df_denom),
                    'df_num': int(df_num),
                    'significant': p_value < significance_level
                }

                if p_value < best_pvalue:
                    best_pvalue = p_value
                    best_lag = lag

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'y_stationary': y_stationary,
            'x_stationary': x_stationary
        }

    return {
        'success': True,
        'results_by_lag': results_by_lag,
        'best_lag': best_lag,
        'best_pvalue': float(best_pvalue),
        'granger_causes': best_pvalue < significance_level,
        'y_stationary': y_stationary,
        'x_stationary': x_stationary,
        'n_samples': len(y)
    }


def test_bidirectional_causality(
    y: np.ndarray,
    x: np.ndarray,
    max_lag: int = 15
) -> Dict:
    """
    Test causality in both directions: x -> y and y -> x.

    Returns:
        Dict with forward and reverse causality results
    """
    forward = granger_causality_test(y, x, max_lag)
    reverse = granger_causality_test(x, y, max_lag)

    return {
        'forward': forward,  # x causes y
        'reverse': reverse,  # y causes x
        'direction': determine_causality_direction(forward, reverse)
    }


def determine_causality_direction(forward: Dict, reverse: Dict) -> str:
    """Determine the direction of causality based on test results."""
    if not forward.get('success') or not reverse.get('success'):
        return 'inconclusive'

    fwd_causes = forward.get('granger_causes', False)
    rev_causes = reverse.get('granger_causes', False)

    if fwd_causes and not rev_causes:
        return 'candidate_causes_residual'
    elif rev_causes and not fwd_causes:
        return 'residual_causes_candidate'
    elif fwd_causes and rev_causes:
        # Both directions significant - compare p-values
        if forward['best_pvalue'] < reverse['best_pvalue']:
            return 'bidirectional_candidate_dominant'
        else:
            return 'bidirectional_residual_dominant'
    else:
        return 'no_causality'


# =============================================================================
# LOAD CANDIDATE DATA
# =============================================================================

def load_candidate_data(timestamps: List[str]) -> pd.DataFrame:
    """Load all candidate variables aligned to timestamps."""
    # Import from Stage 4
    from candidate_correlation import (
        load_temporal_features,
        load_lunar_features,
        load_war_losses_data,
        load_viina_events,
        load_era5_weather,
        load_hdx_conflict_events
    )

    candidates_list = []

    # Temporal
    temporal_df = load_temporal_features(timestamps)
    candidates_list.append(temporal_df)

    # Lunar
    lunar_df = load_lunar_features(timestamps)
    if not lunar_df.empty:
        candidates_list.append(lunar_df)

    # War losses
    losses_df = load_war_losses_data(timestamps)
    if not losses_df.empty:
        candidates_list.append(losses_df)

    # VIINA
    viina_df = load_viina_events(timestamps)
    if not viina_df.empty:
        candidates_list.append(viina_df)

    # ERA5
    era5_df = load_era5_weather(timestamps)
    if not era5_df.empty:
        candidates_list.append(era5_df)

    # HDX
    hdx_df = load_hdx_conflict_events(timestamps)
    if not hdx_df.empty:
        candidates_list.append(hdx_df)

    return pd.concat(candidates_list, axis=1)


# =============================================================================
# MAIN GRANGER TESTING
# =============================================================================

def run_granger_tests(
    residuals: Dict[str, np.ndarray],
    candidates: pd.DataFrame,
    significant_pairs: List[Dict],
    max_lag: int = 15,
    top_n: int = 30
) -> Tuple[List[Dict], Dict, Dict, Dict, List[Dict]]:
    """
    Run Granger causality tests on significant correlation pairs.

    Args:
        residuals: Dict of residual arrays per source
        candidates: DataFrame of candidate variables
        significant_pairs: List of significant correlations from Stage 4
        max_lag: Maximum lag for Granger tests
        top_n: Number of top pairs to test

    Returns:
        granger_results, summary_by_candidate, summary_by_source, optimal_lags, bidirectional_tests
    """
    granger_results = []
    summary_by_candidate = {}
    summary_by_source = {s: {'tested': 0, 'significant': 0, 'pairs': []} for s in residuals.keys()}
    optimal_lags = {}
    bidirectional_tests = []

    # Filter to residual-type pairs (not factor pairs)
    residual_pairs = [p for p in significant_pairs if p['type'] == 'residual'][:top_n]

    # Also test top factor correlations against all residuals
    factor_pairs = [p for p in significant_pairs if p['type'] == 'factor'][:10]

    print(f"Testing {len(residual_pairs)} residual pairs + {len(factor_pairs)} factor-derived pairs")

    tested_combinations = set()

    # Test residual pairs
    for pair in residual_pairs:
        source = pair['source']
        candidate_name = pair['candidate']

        combo_key = (source, candidate_name)
        if combo_key in tested_combinations:
            continue
        tested_combinations.add(combo_key)

        if candidate_name not in candidates.columns:
            continue

        # Get residual magnitude
        res_magnitude = np.linalg.norm(residuals[source], axis=1)
        candidate_values = candidates[candidate_name].values

        # Handle NaNs
        valid_mask = ~np.isnan(candidate_values)
        if valid_mask.sum() < len(candidate_values) * 0.8:
            continue
        candidate_values = pd.Series(candidate_values).interpolate().fillna(method='bfill').fillna(method='ffill').values

        # Run Granger test
        result = granger_causality_test(res_magnitude, candidate_values, max_lag)

        result['source'] = source
        result['candidate'] = candidate_name
        result['correlation'] = pair['correlation']
        granger_results.append(result)

        # Update summaries
        if result.get('success'):
            summary_by_source[source]['tested'] += 1
            if result.get('granger_causes'):
                summary_by_source[source]['significant'] += 1
                summary_by_source[source]['pairs'].append({
                    'candidate': candidate_name,
                    'p_value': result['best_pvalue'],
                    'lag': result['best_lag']
                })

            if candidate_name not in summary_by_candidate:
                summary_by_candidate[candidate_name] = {
                    'tested_sources': 0,
                    'significant_sources': 0,
                    'sources': []
                }
            summary_by_candidate[candidate_name]['tested_sources'] += 1
            if result.get('granger_causes'):
                summary_by_candidate[candidate_name]['significant_sources'] += 1
                summary_by_candidate[candidate_name]['sources'].append({
                    'source': source,
                    'p_value': result['best_pvalue'],
                    'lag': result['best_lag']
                })

            # Track optimal lag
            if result.get('granger_causes'):
                if candidate_name not in optimal_lags:
                    optimal_lags[candidate_name] = result['best_lag']

    # Test factor-correlated candidates against all residual sources
    for pair in factor_pairs:
        candidate_name = pair['candidate']

        if candidate_name not in candidates.columns:
            continue

        candidate_values = candidates[candidate_name].values
        candidate_values = pd.Series(candidate_values).interpolate().fillna(method='bfill').fillna(method='ffill').values

        for source, res in residuals.items():
            combo_key = (source, candidate_name)
            if combo_key in tested_combinations:
                continue
            tested_combinations.add(combo_key)

            res_magnitude = np.linalg.norm(res, axis=1)

            result = granger_causality_test(res_magnitude, candidate_values, max_lag)
            result['source'] = source
            result['candidate'] = candidate_name
            result['from_factor'] = pair['index']
            granger_results.append(result)

            if result.get('success'):
                summary_by_source[source]['tested'] += 1
                if result.get('granger_causes'):
                    summary_by_source[source]['significant'] += 1

    # Run bidirectional tests for top significant pairs
    top_significant = [r for r in granger_results if r.get('granger_causes')][:10]

    for result in top_significant:
        source = result['source']
        candidate_name = result['candidate']

        res_magnitude = np.linalg.norm(residuals[source], axis=1)
        candidate_values = candidates[candidate_name].values
        candidate_values = pd.Series(candidate_values).interpolate().fillna(method='bfill').fillna(method='ffill').values

        bidir = test_bidirectional_causality(res_magnitude, candidate_values, max_lag)
        bidir['source'] = source
        bidir['candidate'] = candidate_name
        bidirectional_tests.append(bidir)

    return granger_results, summary_by_candidate, summary_by_source, optimal_lags, bidirectional_tests


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_granger_visualizations(
    results: GrangerCausalityResults,
    output_dir: Path
):
    """Create visualization plots for Granger causality results."""
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"

    # Color palette
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

    # 1. Granger causality summary by source
    fig, ax = plt.subplots(figsize=(10, 6))

    sources = list(results.summary_by_source.keys())
    tested = [results.summary_by_source[s]['tested'] for s in sources]
    significant = [results.summary_by_source[s]['significant'] for s in sources]

    x = np.arange(len(sources))
    width = 0.35

    bars1 = ax.bar(x - width/2, tested, width, label='Tested', color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, significant, width, label='Significant (p < 0.05)', color='#2ecc71', alpha=0.7)

    ax.set_xlabel('Residual Source')
    ax.set_ylabel('Number of Candidate Pairs')
    ax.set_title('Granger Causality Tests by Residual Source')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sources])
    ax.legend()

    # Add count labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / 'granger_by_source.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Top Granger-causing candidates
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get candidates that Granger-cause multiple sources
    candidate_scores = []
    for cand, info in results.summary_by_candidate.items():
        if info['significant_sources'] > 0:
            # Average p-value across significant sources
            avg_pval = np.mean([s['p_value'] for s in info['sources']]) if info['sources'] else 1.0
            candidate_scores.append({
                'candidate': cand,
                'n_sources': info['significant_sources'],
                'avg_pvalue': avg_pval,
                'score': info['significant_sources'] * (1 - avg_pval)
            })

    candidate_scores.sort(key=lambda x: x['score'], reverse=True)
    top_candidates = candidate_scores[:15]

    if top_candidates:
        labels = [c['candidate'][:30] for c in top_candidates]
        n_sources = [c['n_sources'] for c in top_candidates]
        scores = [c['score'] for c in top_candidates]

        y_pos = range(len(labels))
        bars = ax.barh(y_pos, scores, color='#2ecc71', alpha=0.8)

        # Add n_sources annotation
        for i, (bar, ns) in enumerate(zip(bars, n_sources)):
            ax.annotate(f'({ns} sources)', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       ha='left', va='center', fontsize=9, color='#666')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Granger Score (n_sources × (1 - avg_pvalue))')
        ax.set_title('Top Candidates that Granger-Cause Residuals')

    plt.tight_layout()
    plt.savefig(figures_dir / 'granger_top_candidates.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Optimal lag distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    if results.optimal_lags:
        lags = list(results.optimal_lags.values())
        ax.hist(lags, bins=range(1, max(lags)+2), color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Optimal Lag (days)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Optimal Lags for Granger Causality')
        ax.axvline(x=np.median(lags), color='red', linestyle='--', label=f'Median: {np.median(lags):.0f} days')
        ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / 'granger_optimal_lags.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Bidirectional causality results
    fig, ax = plt.subplots(figsize=(12, 6))

    if results.bidirectional_tests:
        directions = [t['direction'] for t in results.bidirectional_tests]
        direction_counts = pd.Series(directions).value_counts()

        colors_map = {
            'candidate_causes_residual': '#2ecc71',
            'residual_causes_candidate': '#e74c3c',
            'bidirectional_candidate_dominant': '#f39c12',
            'bidirectional_residual_dominant': '#9b59b6',
            'no_causality': '#95a5a6',
            'inconclusive': '#bdc3c7'
        }

        bars = ax.bar(range(len(direction_counts)), direction_counts.values,
                     color=[colors_map.get(d, '#333') for d in direction_counts.index])
        ax.set_xticks(range(len(direction_counts)))
        ax.set_xticklabels([d.replace('_', '\n') for d in direction_counts.index], fontsize=9)
        ax.set_ylabel('Count')
        ax.set_title('Bidirectional Granger Causality Results')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / 'granger_bidirectional.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Granger causality visualizations to {figures_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_granger_causality(
    residual_results_path: Path = None,
    correlation_results_path: Path = None,
    max_lag: int = 15,
    output_dir: Path = None
) -> GrangerCausalityResults:
    """
    Run Stage 5: Granger Causality Testing.

    Args:
        residual_results_path: Path to masked_residuals.npz from Stage 1
        correlation_results_path: Path to candidate_correlations.json from Stage 4
        max_lag: Maximum lag for Granger tests
        output_dir: Output directory

    Returns:
        GrangerCausalityResults
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if residual_results_path is None:
        residual_results_path = output_dir / "results" / "masked_residuals.npz"

    if correlation_results_path is None:
        correlation_results_path = output_dir / "results" / "candidate_correlations.json"

    print("=" * 60)
    print("STAGE 5: GRANGER CAUSALITY TESTING")
    print("=" * 60)

    # Load residuals
    print("\n1. Loading residuals...")
    residual_data = np.load(residual_results_path)
    residuals = {key: residual_data[key] for key in residual_data.files}
    print(f"   Sources: {list(residuals.keys())}")

    # Load correlation results
    print("\n2. Loading Stage 4 correlation results...")
    with open(correlation_results_path) as f:
        corr_results = json.load(f)
    significant_pairs = corr_results['significant_pairs']
    print(f"   {len(significant_pairs)} significant correlations from Stage 4")

    # Load timestamps
    print("\n3. Loading timestamps...")
    with open(output_dir / "results" / "residual_metadata.json") as f:
        metadata = json.load(f)
    timestamps = metadata['timestamps']
    print(f"   Date range: {timestamps[0]} to {timestamps[-1]}")

    # Load candidate data
    print("\n4. Loading candidate variables...")
    candidates = load_candidate_data(timestamps)
    print(f"   Loaded {len(candidates.columns)} candidates")

    # Run Granger tests
    print("\n5. Running Granger causality tests...")
    print(f"   Max lag: {max_lag} days")

    granger_results, summary_by_candidate, summary_by_source, optimal_lags, bidirectional_tests = run_granger_tests(
        residuals, candidates, significant_pairs, max_lag
    )

    # Count significant results
    n_significant = sum(1 for r in granger_results if r.get('granger_causes'))
    print(f"\n   Tested {len(granger_results)} pairs")
    print(f"   Found {n_significant} significant Granger-causal relationships")

    # Print top findings
    print("\n   Top Granger-causing candidates:")
    for cand, info in sorted(summary_by_candidate.items(),
                             key=lambda x: x[1]['significant_sources'],
                             reverse=True)[:10]:
        if info['significant_sources'] > 0:
            print(f"   - {cand}: causes {info['significant_sources']}/{info['tested_sources']} sources")

    # Print bidirectional summary
    print("\n   Bidirectional causality:")
    directions = [t['direction'] for t in bidirectional_tests]
    for direction in set(directions):
        count = directions.count(direction)
        print(f"   - {direction}: {count}")

    # Create results object
    results = GrangerCausalityResults(
        granger_results=granger_results,
        summary_by_candidate=summary_by_candidate,
        summary_by_source=summary_by_source,
        optimal_lags=optimal_lags,
        bidirectional_tests=bidirectional_tests,
        metadata={
            'n_samples': len(timestamps),
            'max_lag': max_lag,
            'n_pairs_tested': len(granger_results),
            'n_significant': n_significant,
            'date_range': [timestamps[0], timestamps[-1]]
        }
    )

    # Save results
    print("\n6. Saving results...")
    results.save(output_dir)

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_granger_visualizations(results, output_dir)

    print("\n" + "=" * 60)
    print("STAGE 5 COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_granger_causality()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTotal pairs tested: {results.metadata['n_pairs_tested']}")
    print(f"Significant Granger-causal pairs: {results.metadata['n_significant']}")

    print("\nBy residual source:")
    for source, info in results.summary_by_source.items():
        pct = 100 * info['significant'] / max(info['tested'], 1)
        print(f"  {source}: {info['significant']}/{info['tested']} ({pct:.1f}%)")

    print("\nTop candidates that Granger-cause residuals:")
    sorted_candidates = sorted(
        results.summary_by_candidate.items(),
        key=lambda x: x[1]['significant_sources'],
        reverse=True
    )
    for cand, info in sorted_candidates[:5]:
        if info['significant_sources'] > 0:
            avg_lag = np.mean([s['lag'] for s in info['sources']]) if info['sources'] else 0
            print(f"  {cand}: {info['significant_sources']} sources, avg lag {avg_lag:.1f} days")
