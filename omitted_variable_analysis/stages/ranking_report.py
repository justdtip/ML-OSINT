"""
Stage 6: Ranking and Reporting

Combines all results from previous stages to produce a final ranking of candidate
omitted variables and generates a comprehensive report.

Key outputs:
1. Ranked list of candidate variables by predictive value
2. Summary of leakage vs. independent variables
3. Recommendations for model improvement
4. Final comprehensive report
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class RankingResults:
    """Results from ranking and reporting stage."""
    timestamp: str
    ranked_candidates: List[Dict]
    independent_candidates: List[Dict]
    leakage_candidates: List[Dict]
    factor_interpretations: Dict[str, Dict]
    recommendations: List[str]
    summary_stats: Dict[str, Any]


def load_all_results(output_dir: Path) -> Dict:
    """Load results from all previous stages."""
    results = {}

    # Stage 1: Residual metadata
    residual_meta_path = output_dir / "results" / "residual_metadata.json"
    if residual_meta_path.exists():
        with open(residual_meta_path) as f:
            results['residual_metadata'] = json.load(f)

    # Stage 2: Temporal analysis
    temporal_path = output_dir / "results" / "temporal_analysis.json"
    if temporal_path.exists():
        with open(temporal_path) as f:
            results['temporal_analysis'] = json.load(f)

    # Stage 3: Factor characterization
    factor_path = output_dir / "results" / "factor_characterization.json"
    if factor_path.exists():
        with open(factor_path) as f:
            results['factor_characterization'] = json.load(f)

    # Stage 4: Candidate correlations
    corr_path = output_dir / "results" / "candidate_correlations.json"
    if corr_path.exists():
        with open(corr_path) as f:
            results['candidate_correlations'] = json.load(f)

    # Stage 5: Granger causality
    granger_path = output_dir / "results" / "granger_causality.json"
    if granger_path.exists():
        with open(granger_path) as f:
            results['granger_causality'] = json.load(f)

    # Independent correlations
    indep_path = output_dir / "results" / "independent_correlations.csv"
    if indep_path.exists():
        results['independent_correlations'] = pd.read_csv(indep_path)

    return results


def classify_leakage_risk(variable: str, source: str) -> str:
    """
    Classify whether a variable has leakage risk.

    Returns:
        'high': Same data provenance as model features
        'medium': Related but not identical source
        'low': Truly independent data source
    """
    # High leakage risk - same source as model features
    high_risk_keywords = [
        'equipment', 'losses', 'cumulative', 'personnel',
        'tank_total', 'aircraft_total', 'vehicle_total'
    ]

    # Sources that share provenance with model
    high_risk_sources = ['war_losses', 'equipment_losses', 'general_staff']

    # Independent sources
    independent_sources = ['FIRMS', 'UCDP', 'Calendar', 'Lunar', 'ERA5', 'VIINA']

    # Check source
    if source in independent_sources:
        return 'low'

    if source in high_risk_sources:
        return 'high'

    # Check variable name
    var_lower = variable.lower()
    for keyword in high_risk_keywords:
        if keyword in var_lower:
            return 'high'

    # Temporal features are low risk
    if any(x in var_lower for x in ['month', 'week', 'day', 'lunar', 'season']):
        return 'low'

    return 'medium'


def rank_candidates(
    correlations: pd.DataFrame,
    granger_results: List[Dict],
    factor_characterization: Dict
) -> List[Dict]:
    """
    Rank candidate variables by their potential value for model improvement.

    Ranking criteria:
    1. Granger causality (predictive power)
    2. Correlation strength with factors
    3. Independence from existing features (low leakage risk)
    4. Coverage of unexplained variance in important factors
    """
    # Create a scoring DataFrame
    candidates = []

    # Process correlations
    if isinstance(correlations, pd.DataFrame):
        for _, row in correlations.iterrows():
            candidate = {
                'variable': row['variable'],
                'source': row['source'],
                'factor': row['factor'],
                'correlation': abs(row['correlation']),
                'p_value': row['p_value'],
                'n_obs': row.get('n_obs', 0),
                'granger_causal': False,
                'optimal_lag': None,
                'leakage_risk': classify_leakage_risk(row['variable'], row['source'])
            }
            candidates.append(candidate)

    # Add Granger causality info
    granger_lookup = {}
    if granger_results:
        for gr in granger_results:
            key = (gr.get('candidate'), gr.get('target'))
            granger_lookup[key] = gr

    for c in candidates:
        key = (c['variable'], c['factor'])
        if key in granger_lookup:
            c['granger_causal'] = True
            c['optimal_lag'] = granger_lookup[key].get('optimal_lag')

    # Calculate composite score
    for c in candidates:
        # Base score from correlation strength
        score = c['correlation'] * 100

        # Bonus for Granger causality
        if c['granger_causal']:
            score += 20

        # Penalty for leakage risk
        if c['leakage_risk'] == 'high':
            score *= 0.3  # Heavy penalty
        elif c['leakage_risk'] == 'medium':
            score *= 0.7  # Moderate penalty

        # Bonus for independent sources
        if c['leakage_risk'] == 'low':
            score *= 1.2

        c['composite_score'] = score

    # Sort by composite score
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)

    return candidates


def generate_factor_interpretations(
    factor_char: Dict,
    independent_corrs: pd.DataFrame
) -> Dict[str, Dict]:
    """Generate interpretations of what each factor might represent."""
    interpretations = {}

    n_factors = factor_char.get('n_factors', 5)
    factor_stats = factor_char.get('factor_statistics', {})

    for i in range(1, n_factors + 1):
        factor_name = f'Factor_{i}'

        # Get correlations for this factor
        factor_corrs = independent_corrs[
            independent_corrs['factor'] == factor_name
        ].sort_values('correlation', key=abs, ascending=False)

        # Top correlates
        top_correlates = []
        for _, row in factor_corrs.head(3).iterrows():
            top_correlates.append({
                'variable': row['variable'],
                'source': row['source'],
                'r': row['correlation']
            })

        # Interpret based on correlations
        interpretation = "Unidentified latent driver"

        if factor_corrs.empty:
            interpretation = "No significant independent correlates found"
        else:
            top_row = factor_corrs.iloc[0]

            # Seasonal patterns
            if 'month' in top_row['variable']:
                interpretation = "Seasonal/cyclical pattern in conflict dynamics"
            elif 'fire' in top_row['variable'] or 'frp' in top_row['variable']:
                interpretation = "Military activity intensity (satellite-detected fires)"
            elif 'fatalities' in top_row['variable'] or 'event' in top_row['variable']:
                interpretation = "Conflict intensity from independent UCDP tracking"
            elif 'lunar' in top_row['variable']:
                interpretation = "Lunar cycle effects (possible night operations)"

        interpretations[factor_name] = {
            'interpretation': interpretation,
            'top_correlates': top_correlates,
            'variance_explained': factor_stats.get(factor_name, {}).get('variance_explained', 0)
        }

    return interpretations


def generate_recommendations(
    ranked_candidates: List[Dict],
    independent_candidates: List[Dict]
) -> List[str]:
    """Generate actionable recommendations for model improvement."""
    recommendations = []

    # Filter to top independent candidates
    top_independent = [c for c in ranked_candidates if c['leakage_risk'] == 'low'][:10]

    if not top_independent:
        recommendations.append(
            "No independent candidate variables found with strong correlations. "
            "Consider collecting additional external data sources."
        )
        return recommendations

    # Recommendation 1: Top candidates to add
    if top_independent:
        top_3 = top_independent[:3]
        vars_str = ", ".join([c['variable'] for c in top_3])
        recommendations.append(
            f"Consider adding these independent variables to the model: {vars_str}. "
            f"These show significant correlation with residual factors and low leakage risk."
        )

    # Recommendation 2: Satellite fire data
    fire_candidates = [c for c in top_independent if 'fire' in c['variable'].lower() or 'frp' in c['variable'].lower()]
    if fire_candidates:
        recommendations.append(
            "NASA FIRMS satellite fire data shows strong correlation with residual factors. "
            "This is truly independent (satellite-based) and captures military activity intensity "
            "not currently in the model."
        )

    # Recommendation 3: Seasonal features
    seasonal = [c for c in top_independent if 'month' in c['variable'].lower()]
    if seasonal:
        recommendations.append(
            "Strong seasonal patterns detected in residuals. Consider adding cyclical "
            "month encoding (sin/cos) to capture seasonal effects on conflict dynamics."
        )

    # Recommendation 4: UCDP data
    ucdp_candidates = [c for c in top_independent if c['source'] == 'UCDP']
    if ucdp_candidates:
        recommendations.append(
            "UCDP conflict event data (Uppsala University) provides independent conflict tracking. "
            "Consider incorporating UCDP fatality estimates as an external validation signal."
        )

    # Recommendation 5: Leakage warning
    high_risk = [c for c in ranked_candidates if c['leakage_risk'] == 'high']
    if high_risk:
        recommendations.append(
            f"WARNING: {len(high_risk)} candidate variables have HIGH leakage risk "
            "(share data provenance with existing model features). "
            "Adding these would not improve generalization."
        )

    return recommendations


def generate_report(
    results: Dict,
    ranked_candidates: List[Dict],
    factor_interpretations: Dict,
    recommendations: List[str],
    output_dir: Path
) -> str:
    """Generate comprehensive markdown report."""
    report = []
    report.append("# Omitted Variable Analysis - Final Report")
    report.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Executive Summary
    report.append("## Executive Summary\n")

    n_candidates = len(ranked_candidates)
    n_independent = len([c for c in ranked_candidates if c['leakage_risk'] == 'low'])
    n_granger = len([c for c in ranked_candidates if c['granger_causal']])

    report.append(f"This analysis identified **{n_candidates}** candidate variables that correlate ")
    report.append(f"with residual patterns in the prediction model. Of these:\n")
    report.append(f"- **{n_independent}** are from independent data sources (low leakage risk)\n")
    report.append(f"- **{n_granger}** show Granger-causal relationships (predictive power)\n\n")

    # Key Findings
    report.append("## Key Findings\n")

    # Top independent candidates
    report.append("### Top Independent Candidate Variables\n")
    report.append("These variables show significant correlation with residual factors and ")
    report.append("come from data sources independent of the model's training data:\n\n")

    report.append("| Rank | Variable | Source | Factor | Correlation | Granger Causal |\n")
    report.append("|------|----------|--------|--------|-------------|----------------|\n")

    for i, c in enumerate([c for c in ranked_candidates if c['leakage_risk'] == 'low'][:10], 1):
        granger = "✓" if c['granger_causal'] else ""
        report.append(f"| {i} | {c['variable']} | {c['source']} | {c['factor']} | {c['correlation']:.3f} | {granger} |\n")

    report.append("\n")

    # Factor Interpretations
    report.append("### Factor Interpretations\n")
    report.append("Based on correlations with independent data sources:\n\n")

    for factor, interp in factor_interpretations.items():
        report.append(f"**{factor}** (explains {interp['variance_explained']:.1%} of residual variance):\n")
        report.append(f"- Interpretation: {interp['interpretation']}\n")
        if interp['top_correlates']:
            report.append("- Top correlates:\n")
            for tc in interp['top_correlates']:
                report.append(f"  - {tc['variable']} ({tc['source']}): r={tc['r']:.3f}\n")
        report.append("\n")

    # Leakage Analysis
    report.append("### Leakage Analysis\n")

    high_risk = [c for c in ranked_candidates if c['leakage_risk'] == 'high']
    if high_risk:
        report.append(f"**{len(high_risk)} variables identified with HIGH leakage risk:**\n\n")
        report.append("These variables share data provenance with existing model features ")
        report.append("and should NOT be added to improve the model:\n\n")
        for c in high_risk[:5]:
            report.append(f"- {c['variable']} (r={c['correlation']:.3f} with {c['factor']})\n")
        report.append("\n")

    # Recommendations
    report.append("## Recommendations\n")
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}\n\n")

    # Methodology Summary
    report.append("## Methodology\n")
    report.append("This analysis followed a 6-stage pipeline:\n\n")
    report.append("1. **Residual Extraction**: Extract prediction residuals from all data sources\n")
    report.append("2. **Temporal Structure Analysis**: Identify autocorrelation and cross-correlation patterns\n")
    report.append("3. **Latent Factor Extraction**: Use PCA to identify common latent factors in residuals\n")
    report.append("4. **Candidate Variable Correlation**: Test correlations with external candidate variables\n")
    report.append("5. **Granger Causality Testing**: Test predictive relationships\n")
    report.append("6. **Ranking and Reporting**: Combine results and generate recommendations\n")

    # Data Sources
    report.append("\n## Data Sources Evaluated\n")
    report.append("### Independent Sources (Low Leakage Risk)\n")
    report.append("- **FIRMS**: NASA VIIRS satellite fire detection\n")
    report.append("- **UCDP**: Uppsala Conflict Data Program event tracking\n")
    report.append("- **Calendar**: Seasonal/cyclical temporal features\n")
    report.append("- **Lunar**: Lunar phase data\n")
    report.append("- **ERA5**: ECMWF weather reanalysis (pending)\n\n")

    report.append("### High Leakage Risk Sources\n")
    report.append("- War losses/equipment data (shares provenance with model features)\n")
    report.append("- Cumulative casualty counts (derived from same source)\n")

    report_text = "\n".join(report)

    # Save report
    report_path = output_dir / "reports" / "stage-6-brief.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)

    return report_text


def create_visualizations(
    ranked_candidates: List[Dict],
    factor_interpretations: Dict,
    output_dir: Path
):
    """Create summary visualizations."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Candidate ranking by score and leakage risk
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top candidates bar chart
    ax = axes[0]
    top_candidates = ranked_candidates[:15]

    colors = []
    for c in top_candidates:
        if c['leakage_risk'] == 'low':
            colors.append('green')
        elif c['leakage_risk'] == 'medium':
            colors.append('orange')
        else:
            colors.append('red')

    labels = [f"{c['variable'][:25]}\n({c['source']})" for c in top_candidates]
    scores = [c['composite_score'] for c in top_candidates]

    bars = ax.barh(range(len(top_candidates)), scores, color=colors)
    ax.set_yticks(range(len(top_candidates)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Composite Score')
    ax.set_title('Top 15 Candidate Variables by Composite Score')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Low Risk (Independent)'),
        Patch(facecolor='orange', label='Medium Risk'),
        Patch(facecolor='red', label='High Risk (Leakage)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Leakage risk pie chart
    ax = axes[1]
    risk_counts = {'low': 0, 'medium': 0, 'high': 0}
    for c in ranked_candidates:
        risk_counts[c['leakage_risk']] += 1

    colors_pie = ['green', 'orange', 'red']
    labels_pie = [f'Low ({risk_counts["low"]})',
                  f'Medium ({risk_counts["medium"]})',
                  f'High ({risk_counts["high"]})']
    sizes = [risk_counts['low'], risk_counts['medium'], risk_counts['high']]

    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%')
    ax.set_title('Candidate Variables by Leakage Risk')

    plt.tight_layout()
    plt.savefig(figures_dir / 'candidate_ranking_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Factor correlations heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get independent candidates only
    independent = [c for c in ranked_candidates if c['leakage_risk'] == 'low']

    # Create correlation matrix
    factors = sorted(set(c['factor'] for c in independent))
    variables = list(set(c['variable'] for c in independent))[:20]  # Limit to 20

    matrix = np.zeros((len(variables), len(factors)))
    var_to_idx = {v: i for i, v in enumerate(variables)}
    factor_to_idx = {f: i for i, f in enumerate(factors)}

    for c in independent:
        if c['variable'] in var_to_idx and c['factor'] in factor_to_idx:
            matrix[var_to_idx[c['variable']], factor_to_idx[c['factor']]] = c['correlation']

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=45, ha='right')
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables, fontsize=8)

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Independent Variable Correlations with Residual Factors')

    plt.tight_layout()
    plt.savefig(figures_dir / 'independent_correlations_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_ranking_report(
    output_dir: Path = None
) -> RankingResults:
    """
    Run Stage 6: Ranking and Reporting.

    Combines all previous stage results to produce final rankings and report.
    """
    if output_dir is None:
        output_dir = Path("omitted_variable_analysis/outputs")

    print("="*60)
    print("Stage 6: Ranking and Reporting")
    print("="*60)

    # Load all previous results
    print("\nLoading results from previous stages...")
    results = load_all_results(output_dir)

    # Check what we have
    print(f"  Loaded: {list(results.keys())}")

    # Get independent correlations
    independent_corrs = results.get('independent_correlations', pd.DataFrame())
    if independent_corrs.empty:
        print("WARNING: No independent correlations found")
    else:
        print(f"  Independent correlations: {len(independent_corrs)}")

    # Get Granger results
    granger_results = results.get('granger_causality', {}).get('significant_relationships', [])
    print(f"  Granger-causal relationships: {len(granger_results)}")

    # Get factor characterization
    factor_char = results.get('factor_characterization', {})

    # Rank candidates
    print("\nRanking candidate variables...")
    ranked_candidates = rank_candidates(
        independent_corrs,
        granger_results,
        factor_char
    )
    print(f"  Total ranked candidates: {len(ranked_candidates)}")

    # Separate by leakage risk
    independent_candidates = [c for c in ranked_candidates if c['leakage_risk'] == 'low']
    leakage_candidates = [c for c in ranked_candidates if c['leakage_risk'] == 'high']

    print(f"  Independent (low risk): {len(independent_candidates)}")
    print(f"  High leakage risk: {len(leakage_candidates)}")

    # Generate factor interpretations
    print("\nGenerating factor interpretations...")
    factor_interpretations = generate_factor_interpretations(
        factor_char,
        independent_corrs
    )

    # Generate recommendations
    print("\nGenerating recommendations...")
    recommendations = generate_recommendations(ranked_candidates, independent_candidates)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec[:80]}...")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(ranked_candidates, factor_interpretations, output_dir)

    # Generate report
    print("\nGenerating final report...")
    report = generate_report(
        results,
        ranked_candidates,
        factor_interpretations,
        recommendations,
        output_dir
    )

    # Save results
    results_path = output_dir / "results" / "ranking_results.json"
    ranking_data = {
        'timestamp': datetime.now().isoformat(),
        'n_candidates': len(ranked_candidates),
        'n_independent': len(independent_candidates),
        'n_leakage': len(leakage_candidates),
        'top_independent': independent_candidates[:10],
        'factor_interpretations': factor_interpretations,
        'recommendations': recommendations
    }
    with open(results_path, 'w') as f:
        json.dump(ranking_data, f, indent=2)

    print(f"\nSaved results to {results_path}")
    print(f"Saved report to {output_dir / 'reports' / 'stage-6-brief.md'}")

    # Summary stats
    summary_stats = {
        'total_candidates': len(ranked_candidates),
        'independent_candidates': len(independent_candidates),
        'leakage_candidates': len(leakage_candidates),
        'granger_causal': len([c for c in ranked_candidates if c['granger_causal']]),
        'top_score': ranked_candidates[0]['composite_score'] if ranked_candidates else 0
    }

    return RankingResults(
        timestamp=datetime.now().isoformat(),
        ranked_candidates=ranked_candidates,
        independent_candidates=independent_candidates,
        leakage_candidates=leakage_candidates,
        factor_interpretations=factor_interpretations,
        recommendations=recommendations,
        summary_stats=summary_stats
    )


if __name__ == "__main__":
    results = run_ranking_report()
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\nTotal candidates analyzed: {results.summary_stats['total_candidates']}")
    print(f"Independent (actionable): {results.summary_stats['independent_candidates']}")
    print(f"High leakage risk (skip): {results.summary_stats['leakage_candidates']}")
    print(f"With Granger causality: {results.summary_stats['granger_causal']}")
    print("\nTop 5 recommended variables to add:")
    for i, c in enumerate(results.independent_candidates[:5], 1):
        print(f"  {i}. {c['variable']} ({c['source']}) - r={c['correlation']:.3f} with {c['factor']}")
