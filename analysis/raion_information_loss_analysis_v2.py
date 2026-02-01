"""
Raion-Level Aggregation Information Loss Analysis (v2)

This script provides a comprehensive analysis of information loss due to raion-level
aggregation in the ML_OSINT Multi-Resolution HAN.

Key improvements over v1:
1. Proper temporal aggregation (daily resolution as used in training)
2. Intraclass correlation coefficient (ICC) for variance decomposition
3. Corrected entropy calculation
4. Proper interpretation of spatial patterns at raion-day level

Author: Data Science Team
Date: 2026-02-01
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Import centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
    UCDP_EVENTS_FILE, FIGURES_DIR,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ucdp_events() -> pd.DataFrame:
    """Load UCDP conflict events with geographic coordinates."""
    print("Loading UCDP events...")

    if not UCDP_EVENTS_FILE.exists():
        raise FileNotFoundError(f"UCDP data not found at {UCDP_EVENTS_FILE}")

    df = pd.read_csv(UCDP_EVENTS_FILE, low_memory=False)

    # Parse dates
    df['date'] = pd.to_datetime(df['date_start'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date', 'latitude', 'longitude'])

    # Filter to Ukraine conflict (2022+)
    df = df[df['date'] >= '2022-02-24'].copy()
    df = df[df['country'] == 'Ukraine'].copy()

    print(f"  Loaded {len(df)} UCDP events in Ukraine (2022+)")
    return df


def load_firms_fires() -> pd.DataFrame:
    """Load FIRMS fire hotspot data with coordinates."""
    print("Loading FIRMS fire data...")

    firms_path = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"

    if not firms_path.exists():
        raise FileNotFoundError(f"FIRMS data not found at {firms_path}")

    df = pd.read_csv(firms_path, low_memory=False)

    df['date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df = df.dropna(subset=['date', 'latitude', 'longitude'])
    df = df[df['date'] >= '2022-02-24'].copy()

    print(f"  Loaded {len(df)} FIRMS fire detections (2022+)")
    return df


def load_geoconfirmed_events() -> pd.DataFrame:
    """Load Geoconfirmed equipment loss events with coordinates."""
    print("Loading Geoconfirmed events...")

    geoconfirmed_path = DATA_DIR / "geoconfirmed" / "geoconfirmed_Ukraine.json"

    if not geoconfirmed_path.exists():
        raise FileNotFoundError(f"Geoconfirmed data not found at {geoconfirmed_path}")

    with open(geoconfirmed_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = data.get('placemarks', [])
    else:
        items = data

    records = []
    for item in items:
        coords = item.get('coordinates', [])
        if len(coords) >= 2:
            records.append({
                'lat': coords[0],
                'lon': coords[1],
                'date': item.get('date'),
                'name': item.get('name', ''),
                'description': item.get('description', ''),
                'gear': item.get('gear', ''),
            })

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'lat', 'lon'])
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    df = df[df['date'] >= '2022-02-24'].copy()

    print(f"  Loaded {len(df)} Geoconfirmed events (2022+)")
    return df


def load_raion_boundaries():
    """Load raion boundary data for spatial assignment."""
    print("Loading raion boundaries...")

    try:
        from analysis.loaders.raion_spatial_loader import RaionBoundaryManager
        manager = RaionBoundaryManager(frontline_only=True)
        manager.load()

        raion_info = []
        for raion_key, raion in manager.raions.items():
            raion_info.append({
                'raion_key': raion_key,
                'raion_name': raion.name,
                'oblast': raion.oblast,
                'centroid_lon': raion.centroid[0],
                'centroid_lat': raion.centroid[1],
            })

        df = pd.DataFrame(raion_info)
        print(f"  Loaded {len(df)} raion boundaries")
        return df, manager
    except Exception as e:
        print(f"  Warning: Could not load raion boundaries: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None


def assign_points_to_raions(points_df, manager, lat_col='latitude', lon_col='longitude'):
    """Assign each point to its containing raion."""
    assignments = []
    total = len(points_df)

    for idx, (_, row) in enumerate(points_df.iterrows()):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Processed {idx}/{total} points...")

        raion_key = manager.get_raion_for_point(row[lat_col], row[lon_col])
        assignments.append(raion_key)

    return pd.Series(assignments, index=points_df.index)


# =============================================================================
# INTRACLASS CORRELATION COEFFICIENT (ICC)
# =============================================================================

def compute_icc(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, float]:
    """
    Compute Intraclass Correlation Coefficient (ICC) for clustered data.

    ICC measures the proportion of variance attributable to between-group differences.
    - ICC near 1: High clustering (groups are very different, within-group similar)
    - ICC near 0: Low clustering (groups are similar, within-group variable)

    For raion aggregation:
    - High ICC: Raion-level aggregation captures most of the variance (good)
    - Low ICC: Significant within-raion variation is lost (bad)

    Uses ICC(1,k) - the reliability of group means for k observations.
    """
    df_valid = df.dropna(subset=[group_col, value_col])

    if len(df_valid) < 10:
        return {'icc': np.nan, 'f_statistic': np.nan, 'p_value': np.nan}

    groups = df_valid.groupby(group_col)[value_col]

    # Get group statistics
    n_groups = groups.ngroups
    if n_groups < 2:
        return {'icc': np.nan, 'f_statistic': np.nan, 'p_value': np.nan}

    group_means = groups.mean()
    group_sizes = groups.size()
    grand_mean = df_valid[value_col].mean()

    # Between-group sum of squares
    ss_between = (group_sizes * (group_means - grand_mean) ** 2).sum()
    df_between = n_groups - 1

    # Within-group sum of squares
    ss_within = 0
    for name, group_data in groups:
        ss_within += ((group_data - group_means[name]) ** 2).sum()
    df_within = len(df_valid) - n_groups

    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # Average group size (for unbalanced design)
    n_mean = len(df_valid) / n_groups

    # F-statistic and p-value
    if ms_within > 0:
        f_stat = ms_between / ms_within
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
    else:
        f_stat = np.inf
        p_value = 0.0

    # ICC(1,k) formula
    if ms_between + ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (n_mean - 1) * ms_within)
    else:
        icc = 0.0

    # Bound ICC to [0, 1]
    icc = max(0, min(1, icc))

    return {
        'icc': icc,
        'f_statistic': f_stat,
        'p_value': p_value,
        'n_groups': n_groups,
        'n_obs': len(df_valid),
        'ms_between': ms_between,
        'ms_within': ms_within,
    }


# =============================================================================
# DAILY AGGREGATION ANALYSIS
# =============================================================================

def analyze_daily_raion_aggregation(
    df: pd.DataFrame,
    raion_col: str,
    value_cols: List[str],
    source_name: str
) -> Dict[str, Dict]:
    """
    Analyze information loss when aggregating to daily-raion level.

    This matches what the Multi-Resolution HAN actually does:
    - Events are aggregated by raion-day
    - We measure how much variance is captured at raion vs within-raion level
    """
    results = {}

    df_valid = df.dropna(subset=[raion_col]).copy()
    df_valid['date_only'] = df_valid['date'].dt.date

    for col in value_cols:
        if col not in df_valid.columns:
            continue

        # Aggregate to daily-raion level
        daily_raion = df_valid.groupby([raion_col, 'date_only'])[col].agg(['sum', 'count', 'mean']).reset_index()
        daily_raion.columns = [raion_col, 'date_only', 'daily_sum', 'daily_count', 'daily_mean']

        # Now analyze variance structure across raions for the same dates
        # ICC: how much variance is between raions vs within raions (across dates)
        icc_by_date = compute_icc(daily_raion, raion_col, 'daily_sum')

        # Alternative: ICC by raion across dates (temporal clustering)
        icc_by_raion = compute_icc(daily_raion, 'date_only', 'daily_sum')

        # Count statistics
        events_per_raion_day = daily_raion['daily_count'].describe()

        # Coefficient of variation within vs between raions
        raion_means = daily_raion.groupby(raion_col)['daily_sum'].mean()
        raion_stds = daily_raion.groupby(raion_col)['daily_sum'].std()

        cv_between = raion_means.std() / raion_means.mean() if raion_means.mean() > 0 else 0
        cv_within = (raion_stds / raion_means).mean() if (raion_means > 0).all() else 0

        results[col] = {
            'icc_spatial': icc_by_date['icc'],
            'icc_temporal': icc_by_raion['icc'],
            'f_stat_spatial': icc_by_date['f_statistic'],
            'p_value_spatial': icc_by_date['p_value'],
            'n_raions': daily_raion[raion_col].nunique(),
            'n_raion_days': len(daily_raion),
            'cv_between_raions': cv_between,
            'cv_within_raions': cv_within,
            'events_per_raion_day_mean': events_per_raion_day['mean'],
            'events_per_raion_day_std': events_per_raion_day['std'],
        }

    return results


# =============================================================================
# SPATIAL HETEROGENEITY ANALYSIS
# =============================================================================

def analyze_spatial_heterogeneity(
    df: pd.DataFrame,
    raion_col: str,
    value_col: str,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> Dict[str, float]:
    """
    Analyze spatial heterogeneity within raions.

    Measures:
    1. Average pairwise distance between events in same raion
    2. Spatial spread (std of coordinates) within raions
    3. Coefficient of variation of event locations
    """
    df_valid = df.dropna(subset=[raion_col, lat_col, lon_col]).copy()

    if len(df_valid) < 10:
        return {}

    raion_stats = []

    for raion in df_valid[raion_col].unique():
        if raion is None:
            continue

        raion_data = df_valid[df_valid[raion_col] == raion]

        if len(raion_data) < 5:
            continue

        # Spatial spread (standard deviation of coordinates)
        lat_std = raion_data[lat_col].std()
        lon_std = raion_data[lon_col].std()
        spatial_spread_km = np.sqrt(lat_std**2 + lon_std**2) * 111  # Approximate km

        # Centroid
        centroid_lat = raion_data[lat_col].mean()
        centroid_lon = raion_data[lon_col].mean()

        # Distance from centroid
        distances = np.sqrt(
            (raion_data[lat_col] - centroid_lat)**2 +
            (raion_data[lon_col] - centroid_lon)**2
        ) * 111  # km

        mean_dist = distances.mean()
        max_dist = distances.max()

        raion_stats.append({
            'raion': raion,
            'n_events': len(raion_data),
            'spatial_spread_km': spatial_spread_km,
            'mean_dist_from_centroid_km': mean_dist,
            'max_dist_from_centroid_km': max_dist,
        })

    if not raion_stats:
        return {}

    stats_df = pd.DataFrame(raion_stats)

    return {
        'mean_spatial_spread_km': stats_df['spatial_spread_km'].mean(),
        'std_spatial_spread_km': stats_df['spatial_spread_km'].std(),
        'mean_centroid_dist_km': stats_df['mean_dist_from_centroid_km'].mean(),
        'max_centroid_dist_km': stats_df['max_dist_from_centroid_km'].max(),
        'n_raions_analyzed': len(stats_df),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(
    variance_results: Dict[str, Dict],
    output_dir: Path
):
    """Create visualizations for the analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. ICC comparison across sources
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sources = list(variance_results.keys())
    spatial_iccs = []
    temporal_iccs = []

    for source in sources:
        source_results = variance_results[source]
        if source_results:
            spatial_vals = [r['icc_spatial'] for r in source_results.values()
                           if not np.isnan(r.get('icc_spatial', np.nan))]
            temporal_vals = [r['icc_temporal'] for r in source_results.values()
                           if not np.isnan(r.get('icc_temporal', np.nan))]

            spatial_iccs.append(np.mean(spatial_vals) if spatial_vals else 0)
            temporal_iccs.append(np.mean(temporal_vals) if temporal_vals else 0)

    # Spatial ICC
    ax1 = axes[0]
    bars1 = ax1.bar(sources, spatial_iccs, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax1.set_ylabel('Spatial ICC (between-raion clustering)')
    ax1.set_title('Information Captured by Raion Aggregation')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
    ax1.legend()

    # Add value labels
    for bar, val in zip(bars1, spatial_iccs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom')

    # Temporal ICC
    ax2 = axes[1]
    bars2 = ax2.bar(sources, temporal_iccs, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax2.set_ylabel('Temporal ICC (date-level clustering)')
    ax2.set_title('Temporal Pattern Clustering')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
    ax2.legend()

    for bar, val in zip(bars2, temporal_iccs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'icc_comparison.png', dpi=150)
    plt.close()

    print(f"  Saved ICC comparison figure to {output_dir / 'icc_comparison.png'}")

    # 2. Variance decomposition pie charts
    fig, axes = plt.subplots(1, len(sources), figsize=(4*len(sources), 4))

    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        source_results = variance_results[source]
        if source_results:
            avg_icc = np.mean([r['icc_spatial'] for r in source_results.values()
                              if not np.isnan(r.get('icc_spatial', np.nan))])

            within = 1 - avg_icc
            between = avg_icc

            ax.pie([between, within],
                   labels=['Between raions\n(captured)', 'Within raions\n(lost)'],
                   autopct='%1.1f%%',
                   colors=['#4CAF50', '#f44336'])
            ax.set_title(f'{source}\nVariance Decomposition')

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_decomposition.png', dpi=150)
    plt.close()

    print(f"  Saved variance decomposition figure to {output_dir / 'variance_decomposition.png'}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report_v2(
    variance_results: Dict[str, Dict],
    spatial_results: Dict[str, Dict],
    output_path: Path
) -> str:
    """Generate comprehensive analysis report."""

    report = []
    report.append("# Raion-Level Aggregation Information Loss Analysis")
    report.append("")
    report.append("**Analysis Date**: 2026-02-01")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("This analysis investigates whether aggregating OSINT data at the raion")
    report.append("(district) level causes significant information loss for the Multi-Resolution HAN.")
    report.append("The analysis uses Intraclass Correlation Coefficient (ICC) to properly measure")
    report.append("the proportion of variance captured by raion-level aggregation.")
    report.append("")

    # Key findings
    all_spatial_iccs = []
    for source_results in variance_results.values():
        for feat_results in source_results.values():
            if not np.isnan(feat_results.get('icc_spatial', np.nan)):
                all_spatial_iccs.append(feat_results['icc_spatial'])

    avg_icc = np.mean(all_spatial_iccs) if all_spatial_iccs else 0

    report.append("### Key Findings")
    report.append("")
    report.append(f"- **Average Spatial ICC**: {avg_icc:.3f} ({avg_icc*100:.1f}% of variance captured)")
    report.append(f"- **Information Preserved**: {avg_icc*100:.1f}%")
    report.append(f"- **Information Lost**: {(1-avg_icc)*100:.1f}%")
    report.append("")

    if avg_icc < 0.3:
        report.append("**Assessment**: LOW - Raion aggregation loses significant spatial information.")
        report.append("Consider using finer granularity or preserving sub-raion patterns.")
    elif avg_icc < 0.6:
        report.append("**Assessment**: MODERATE - Raion aggregation captures a reasonable amount of variance.")
        report.append("Some local patterns may be lost but major trends are preserved.")
    else:
        report.append("**Assessment**: HIGH - Raion aggregation preserves most of the spatial variance.")
        report.append("This level of aggregation is appropriate for the available data.")
    report.append("")

    # Methodology
    report.append("## Methodology")
    report.append("")
    report.append("### Intraclass Correlation Coefficient (ICC)")
    report.append("")
    report.append("ICC measures the proportion of variance attributable to between-group differences:")
    report.append("")
    report.append("$$ICC = \\frac{MS_{between} - MS_{within}}{MS_{between} + (k-1) \\cdot MS_{within}}$$")
    report.append("")
    report.append("Where:")
    report.append("- $MS_{between}$: Mean squares between raions")
    report.append("- $MS_{within}$: Mean squares within raions")
    report.append("- $k$: Average number of observations per raion")
    report.append("")
    report.append("**Interpretation**:")
    report.append("- ICC > 0.7: Strong clustering - raion aggregation works well")
    report.append("- ICC 0.4-0.7: Moderate clustering - some information loss")
    report.append("- ICC < 0.4: Weak clustering - significant information loss")
    report.append("")

    # Results by source
    report.append("## Results by Data Source")
    report.append("")

    for source_name, source_results in variance_results.items():
        if not source_results:
            continue

        report.append(f"### {source_name}")
        report.append("")

        # Table header
        report.append("| Feature | Spatial ICC | Temporal ICC | F-stat | p-value | N Raions | CV Between | CV Within |")
        report.append("|---------|-------------|--------------|--------|---------|----------|------------|-----------|")

        for feat_name, feat_results in source_results.items():
            spatial_icc = feat_results.get('icc_spatial', np.nan)
            temporal_icc = feat_results.get('icc_temporal', np.nan)
            f_stat = feat_results.get('f_stat_spatial', np.nan)
            p_val = feat_results.get('p_value_spatial', np.nan)
            n_raions = feat_results.get('n_raions', 0)
            cv_between = feat_results.get('cv_between_raions', np.nan)
            cv_within = feat_results.get('cv_within_raions', np.nan)

            spatial_str = f"{spatial_icc:.3f}" if not np.isnan(spatial_icc) else "N/A"
            temporal_str = f"{temporal_icc:.3f}" if not np.isnan(temporal_icc) else "N/A"
            f_str = f"{f_stat:.2f}" if not np.isnan(f_stat) else "N/A"
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            cv_b_str = f"{cv_between:.2f}" if not np.isnan(cv_between) else "N/A"
            cv_w_str = f"{cv_within:.2f}" if not np.isnan(cv_within) else "N/A"

            report.append(f"| {feat_name[:25]} | {spatial_str} | {temporal_str} | {f_str} | {p_str} | {n_raions} | {cv_b_str} | {cv_w_str} |")

        # Source summary
        spatial_vals = [r['icc_spatial'] for r in source_results.values()
                       if not np.isnan(r.get('icc_spatial', np.nan))]
        avg_spatial = np.mean(spatial_vals) if spatial_vals else 0

        report.append("")
        report.append(f"**Source Summary**: Average Spatial ICC = {avg_spatial:.3f}")

        if avg_spatial < 0.3:
            report.append("- Assessment: LOW clustering - significant information loss from raion aggregation")
        elif avg_spatial < 0.6:
            report.append("- Assessment: MODERATE clustering - some information preserved")
        else:
            report.append("- Assessment: HIGH clustering - raion aggregation is appropriate")
        report.append("")

        # Spatial heterogeneity results
        if source_name in spatial_results and spatial_results[source_name]:
            spatial = spatial_results[source_name]
            report.append("**Spatial Heterogeneity Within Raions**:")
            report.append(f"- Mean spatial spread: {spatial.get('mean_spatial_spread_km', 0):.1f} km")
            report.append(f"- Mean distance from centroid: {spatial.get('mean_centroid_dist_km', 0):.1f} km")
            report.append(f"- Max distance from centroid: {spatial.get('max_centroid_dist_km', 0):.1f} km")
            report.append("")

    # Conclusions and Recommendations
    report.append("## Conclusions and Recommendations")
    report.append("")

    # Determine recommendations by source
    high_loss_sources = []
    low_loss_sources = []

    for source_name, source_results in variance_results.items():
        if source_results:
            spatial_vals = [r['icc_spatial'] for r in source_results.values()
                           if not np.isnan(r.get('icc_spatial', np.nan))]
            if spatial_vals:
                avg = np.mean(spatial_vals)
                if avg < 0.3:
                    high_loss_sources.append((source_name, avg))
                elif avg > 0.6:
                    low_loss_sources.append((source_name, avg))

    report.append("### Assessment by Source")
    report.append("")

    if high_loss_sources:
        report.append("**Sources with HIGH information loss** (ICC < 0.3):")
        for source, icc in high_loss_sources:
            report.append(f"- {source}: ICC = {icc:.3f}")
        report.append("")
        report.append("These sources have significant within-raion variation that is lost through")
        report.append("aggregation. Consider:")
        report.append("- Using sub-raion grids or tiles")
        report.append("- Preserving variance statistics (not just sums)")
        report.append("- Including spatial distribution features")
        report.append("")

    if low_loss_sources:
        report.append("**Sources with LOW information loss** (ICC > 0.6):")
        for source, icc in low_loss_sources:
            report.append(f"- {source}: ICC = {icc:.3f}")
        report.append("")
        report.append("Raion-level aggregation is appropriate for these sources.")
        report.append("")

    # Interpretation
    report.append("### Interpretation of Results")
    report.append("")
    report.append("The ICC values measure how similar observations are within the same raion")
    report.append("compared to observations in different raions:")
    report.append("")
    report.append("1. **Spatial ICC**: Measures between-raion clustering")
    report.append("   - High values: Events cluster strongly by raion (aggregation OK)")
    report.append("   - Low values: Events don't cluster by raion (aggregation loses info)")
    report.append("")
    report.append("2. **Temporal ICC**: Measures temporal clustering")
    report.append("   - High values: Strong day-to-day correlations")
    report.append("   - Low values: Events are temporally independent")
    report.append("")
    report.append("3. **Coefficient of Variation**:")
    report.append("   - CV Between: Variation across raion means")
    report.append("   - CV Within: Variation within raions")
    report.append("   - High CV_between/CV_within ratio indicates good aggregation")
    report.append("")

    # Technical notes
    report.append("### Technical Notes")
    report.append("")
    report.append("- Analysis uses daily-aggregated data matching the Multi-Resolution HAN pipeline")
    report.append("- ICC is computed using one-way random effects model")
    report.append("- F-statistics test whether ICC is significantly different from zero")
    report.append("- Spatial distances approximate (assume 111 km per degree)")
    report.append("")

    # Save report
    report_text = "\n".join(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_path}")

    return report_text


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run complete raion information loss analysis (v2)."""
    print("=" * 80)
    print("RAION-LEVEL AGGREGATION INFORMATION LOSS ANALYSIS (v2)")
    print("=" * 80)
    print()

    variance_results: Dict[str, Dict] = {}
    spatial_results: Dict[str, Dict] = {}

    # Load raion boundaries
    raion_df, raion_manager = load_raion_boundaries()

    if raion_manager is None:
        print("ERROR: Cannot proceed without raion boundaries")
        return

    # ==========================================================================
    # Analyze UCDP Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("ANALYZING UCDP DATA")
    print("-" * 60)

    try:
        ucdp_df = load_ucdp_events()

        print("  Assigning events to raions...")
        ucdp_df['raion'] = assign_points_to_raions(ucdp_df, raion_manager)
        assigned_pct = 100 * ucdp_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of events assigned to raions")

        ucdp_features = ['best_est', 'deaths_a', 'deaths_b', 'deaths_civilians']

        print("  Computing ICC analysis...")
        variance_results['UCDP'] = analyze_daily_raion_aggregation(
            ucdp_df, 'raion', ucdp_features, 'UCDP'
        )

        print("  Computing spatial heterogeneity...")
        spatial_results['UCDP'] = analyze_spatial_heterogeneity(
            ucdp_df, 'raion', 'best_est'
        )

    except Exception as e:
        print(f"  ERROR analyzing UCDP data: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # Analyze FIRMS Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("ANALYZING FIRMS DATA")
    print("-" * 60)

    try:
        firms_df = load_firms_fires()

        print("  Assigning fires to raions...")
        firms_df['raion'] = assign_points_to_raions(firms_df, raion_manager)
        assigned_pct = 100 * firms_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of fires assigned to raions")

        firms_features = ['brightness', 'bright_t31', 'frp']

        print("  Computing ICC analysis...")
        variance_results['FIRMS'] = analyze_daily_raion_aggregation(
            firms_df, 'raion', firms_features, 'FIRMS'
        )

        print("  Computing spatial heterogeneity...")
        spatial_results['FIRMS'] = analyze_spatial_heterogeneity(
            firms_df, 'raion', 'frp'
        )

    except Exception as e:
        print(f"  ERROR analyzing FIRMS data: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # Analyze Geoconfirmed Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("ANALYZING GEOCONFIRMED DATA")
    print("-" * 60)

    try:
        geo_df = load_geoconfirmed_events()

        print("  Assigning events to raions...")
        geo_df['raion'] = assign_points_to_raions(geo_df, raion_manager)
        assigned_pct = 100 * geo_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of events assigned to raions")

        # Create numeric feature for analysis
        geo_df['event_count'] = 1

        geo_features = ['event_count']

        print("  Computing ICC analysis...")
        variance_results['Geoconfirmed'] = analyze_daily_raion_aggregation(
            geo_df, 'raion', geo_features, 'Geoconfirmed'
        )

        print("  Computing spatial heterogeneity...")
        spatial_results['Geoconfirmed'] = analyze_spatial_heterogeneity(
            geo_df, 'raion', 'event_count'
        )

    except Exception as e:
        print(f"  ERROR analyzing Geoconfirmed data: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================================================
    # Create Visualizations
    # ==========================================================================
    print("\n" + "-" * 60)
    print("CREATING VISUALIZATIONS")
    print("-" * 60)

    try:
        output_dir = ANALYSIS_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        create_visualizations(variance_results, output_dir)
    except Exception as e:
        print(f"  ERROR creating visualizations: {e}")

    # ==========================================================================
    # Generate Report
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    output_path = ANALYSIS_OUTPUT_DIR / "raion_information_loss_report.md"

    report = generate_report_v2(
        variance_results,
        spatial_results,
        output_path
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Quick summary
    all_spatial_iccs = []
    for source_results in variance_results.values():
        for feat_results in source_results.values():
            if not np.isnan(feat_results.get('icc_spatial', np.nan)):
                all_spatial_iccs.append(feat_results['icc_spatial'])

    if all_spatial_iccs:
        avg_icc = np.mean(all_spatial_iccs)
        print(f"\nOverall average Spatial ICC: {avg_icc:.3f}")
        print(f"Information captured by raion aggregation: {avg_icc*100:.1f}%")
        print(f"Information lost: {(1-avg_icc)*100:.1f}%")

    print(f"\nFull report saved to: {output_path}")


if __name__ == '__main__':
    main()
