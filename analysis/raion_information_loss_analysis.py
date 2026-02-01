"""
Raion-Level Aggregation Information Loss Analysis

This script investigates whether raion-level aggregation in the ML_OSINT project
causes significant information loss. It performs:

1. Variance decomposition analysis (within-raion vs between-raion)
2. Entropy analysis (raw vs aggregated data)
3. Temporal pattern preservation (ACF comparison)
4. Spatial correlation analysis (Moran's I within raions)

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

# Import centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
    UCDP_EVENTS_FILE,
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

    # Filter to Ukraine
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

    # Parse dates
    df['date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df = df.dropna(subset=['date', 'latitude', 'longitude'])

    # Filter to conflict period
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

    # Handle different JSON formats
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

    # Filter to conflict period
    df = df[df['date'] >= '2022-02-24'].copy()

    print(f"  Loaded {len(df)} Geoconfirmed events (2022+)")
    return df


def load_raion_boundaries() -> pd.DataFrame:
    """Load raion boundary data for spatial assignment."""
    print("Loading raion boundaries...")

    try:
        from analysis.loaders.raion_spatial_loader import RaionBoundaryManager
        manager = RaionBoundaryManager(frontline_only=True)
        manager.load()

        # Get raion info from the manager's raions dict
        raion_info = []
        for raion_key, raion in manager.raions.items():
            raion_info.append({
                'raion_key': raion_key,
                'raion_name': raion.name,
                'oblast': raion.oblast,
                'centroid_lon': raion.centroid[0],
                'centroid_lat': raion.centroid[1],
                'bbox_min_lon': raion.bbox[0],
                'bbox_min_lat': raion.bbox[1],
                'bbox_max_lon': raion.bbox[2],
                'bbox_max_lat': raion.bbox[3],
            })

        df = pd.DataFrame(raion_info)
        print(f"  Loaded {len(df)} raion boundaries")
        return df, manager
    except Exception as e:
        print(f"  Warning: Could not load raion boundaries: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None


def assign_points_to_raions(
    points_df: pd.DataFrame,
    manager,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.Series:
    """Assign each point to its containing raion."""
    # Use manager's lookup method
    assignments = []
    total = len(points_df)

    for idx, (_, row) in enumerate(points_df.iterrows()):
        if idx % 10000 == 0 and idx > 0:
            print(f"    Processed {idx}/{total} points...")

        lat, lon = row[lat_col], row[lon_col]

        # Use manager's get_raion_for_point method
        raion_key = manager.get_raion_for_point(lat, lon)
        assignments.append(raion_key)

    return pd.Series(assignments, index=points_df.index)


# =============================================================================
# VARIANCE DECOMPOSITION ANALYSIS
# =============================================================================

@dataclass
class VarianceDecomposition:
    """Results of variance decomposition analysis."""
    source_name: str
    feature_name: str
    total_variance: float
    within_variance: float
    between_variance: float
    variance_ratio: float  # between / total
    n_raions: int
    n_observations: int

    @property
    def information_preserved(self) -> float:
        """Fraction of information preserved by raion aggregation."""
        return self.variance_ratio

    @property
    def information_lost(self) -> float:
        """Fraction of information lost by raion aggregation."""
        return 1.0 - self.variance_ratio


def compute_variance_decomposition(
    df: pd.DataFrame,
    raion_col: str,
    value_cols: List[str],
    source_name: str
) -> List[VarianceDecomposition]:
    """
    Compute within-raion vs between-raion variance decomposition.

    Uses one-way ANOVA-style variance decomposition:
    - Total variance = variance of all observations
    - Between variance = variance of raion means
    - Within variance = average variance within each raion

    High between/total ratio = aggregation preserves most information
    Low ratio = significant information loss
    """
    results = []

    # Filter to valid raion assignments
    df_valid = df.dropna(subset=[raion_col])

    if len(df_valid) < 10:
        print(f"  Warning: Too few valid observations for {source_name}")
        return results

    for col in value_cols:
        if col not in df_valid.columns:
            continue

        # Get values and raion groups
        values = df_valid[col].astype(float)
        raions = df_valid[raion_col]

        # Skip if too few unique values
        if values.nunique() < 2:
            continue

        # Total variance
        total_var = values.var()

        if total_var == 0:
            continue

        # Group by raion
        grouped = df_valid.groupby(raion_col)[col]

        # Between-raion variance (variance of group means)
        group_means = grouped.mean()
        between_var = group_means.var()

        # Within-raion variance (pooled within-group variance)
        within_vars = grouped.var()
        group_sizes = grouped.size()

        # Weighted average of within-group variances
        if len(within_vars) > 0:
            within_var = (within_vars * (group_sizes - 1)).sum() / (group_sizes.sum() - len(group_sizes))
            within_var = max(0, within_var)  # Ensure non-negative
        else:
            within_var = 0

        # Variance ratio (proportion explained by between-group)
        # Using ANOVA-style: between / total
        variance_ratio = between_var / total_var if total_var > 0 else 0

        results.append(VarianceDecomposition(
            source_name=source_name,
            feature_name=col,
            total_variance=total_var,
            within_variance=within_var,
            between_variance=between_var,
            variance_ratio=min(1.0, variance_ratio),  # Cap at 1.0
            n_raions=raions.nunique(),
            n_observations=len(values),
        ))

    return results


# =============================================================================
# ENTROPY ANALYSIS
# =============================================================================

def compute_entropy(values: np.ndarray, n_bins: int = 50) -> float:
    """
    Compute Shannon entropy of a continuous distribution.

    Uses histogram binning to estimate probability distribution.
    """
    values = values[~np.isnan(values)]

    if len(values) < 10:
        return 0.0

    # Create histogram
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    hist = hist[hist > 0]  # Remove zero bins

    # Normalize to probability
    hist = hist / hist.sum()

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


@dataclass
class EntropyAnalysis:
    """Results of entropy analysis comparing raw vs aggregated data."""
    source_name: str
    feature_name: str
    raw_entropy: float
    aggregated_entropy: float
    information_loss: float  # raw - aggregated
    relative_loss: float  # (raw - aggregated) / raw


def compute_entropy_analysis(
    df: pd.DataFrame,
    raion_col: str,
    value_cols: List[str],
    source_name: str,
    agg_func: str = 'sum'
) -> List[EntropyAnalysis]:
    """
    Compare entropy of raw data vs raion-aggregated data.

    Information loss = H(raw) - H(aggregated)
    """
    results = []

    df_valid = df.dropna(subset=[raion_col])

    for col in value_cols:
        if col not in df_valid.columns:
            continue

        raw_values = df_valid[col].astype(float).values

        if len(raw_values) < 10:
            continue

        # Raw entropy
        raw_entropy = compute_entropy(raw_values)

        # Aggregated data (by raion and date if date column exists)
        if 'date' in df_valid.columns:
            # Aggregate by raion-day
            agg_values = df_valid.groupby([raion_col, df_valid['date'].dt.date])[col].agg(agg_func).values
        else:
            # Aggregate by raion only
            agg_values = df_valid.groupby(raion_col)[col].agg(agg_func).values

        if len(agg_values) < 10:
            continue

        aggregated_entropy = compute_entropy(agg_values)

        information_loss = raw_entropy - aggregated_entropy
        relative_loss = information_loss / raw_entropy if raw_entropy > 0 else 0

        results.append(EntropyAnalysis(
            source_name=source_name,
            feature_name=col,
            raw_entropy=raw_entropy,
            aggregated_entropy=aggregated_entropy,
            information_loss=information_loss,
            relative_loss=relative_loss,
        ))

    return results


# =============================================================================
# TEMPORAL PATTERN PRESERVATION
# =============================================================================

def compute_acf(values: np.ndarray, n_lags: int = 30) -> np.ndarray:
    """Compute autocorrelation function."""
    values = values - values.mean()
    n = len(values)

    if n < n_lags + 1:
        return np.array([])

    acf = np.correlate(values, values, mode='full')
    acf = acf[n-1:n+n_lags] / acf[n-1]

    return acf


@dataclass
class TemporalPreservation:
    """Results of temporal pattern preservation analysis."""
    source_name: str
    feature_name: str
    raw_acf_decay: float  # Decay rate of raw ACF
    aggregated_acf_decay: float  # Decay rate of aggregated ACF
    acf_correlation: float  # Correlation between raw and agg ACF
    pattern_preserved: bool  # Whether temporal structure is preserved


def compute_temporal_preservation(
    df: pd.DataFrame,
    raion_col: str,
    value_col: str,
    source_name: str,
    date_col: str = 'date'
) -> Optional[TemporalPreservation]:
    """
    Check if temporal autocorrelation is preserved after aggregation.
    """
    df_valid = df.dropna(subset=[raion_col, date_col])

    if len(df_valid) < 60:  # Need enough data for ACF
        return None

    # Daily aggregation for raw data (national level)
    raw_daily = df_valid.groupby(df_valid[date_col].dt.date)[value_col].sum()
    raw_daily = raw_daily.reindex(pd.date_range(raw_daily.index.min(), raw_daily.index.max())).fillna(0)

    if len(raw_daily) < 60:
        return None

    raw_acf = compute_acf(raw_daily.values, n_lags=30)

    # Aggregated by raion-day, then averaged across raions
    raion_daily = df_valid.groupby([raion_col, df_valid[date_col].dt.date])[value_col].sum()
    raion_daily = raion_daily.unstack(level=0).fillna(0)

    # Average across raions (simulates aggregation)
    agg_daily = raion_daily.mean(axis=1)
    agg_daily = agg_daily.reindex(pd.date_range(agg_daily.index.min(), agg_daily.index.max())).fillna(0)

    if len(agg_daily) < 60:
        return None

    agg_acf = compute_acf(agg_daily.values, n_lags=30)

    if len(raw_acf) == 0 or len(agg_acf) == 0:
        return None

    # Compute decay rate (how fast ACF drops)
    raw_decay = np.abs(np.diff(raw_acf[:10])).mean() if len(raw_acf) >= 10 else 0
    agg_decay = np.abs(np.diff(agg_acf[:10])).mean() if len(agg_acf) >= 10 else 0

    # Correlation between ACF curves
    min_len = min(len(raw_acf), len(agg_acf))
    if min_len > 1:
        acf_corr, _ = stats.pearsonr(raw_acf[:min_len], agg_acf[:min_len])
    else:
        acf_corr = 0

    # Pattern is preserved if ACF correlation is high
    pattern_preserved = acf_corr > 0.8

    return TemporalPreservation(
        source_name=source_name,
        feature_name=value_col,
        raw_acf_decay=raw_decay,
        aggregated_acf_decay=agg_decay,
        acf_correlation=acf_corr,
        pattern_preserved=pattern_preserved,
    )


# =============================================================================
# SPATIAL CORRELATION ANALYSIS
# =============================================================================

def compute_moran_i(
    values: np.ndarray,
    coords: np.ndarray,
    distance_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation statistic.

    Returns (moran_i, p_value)
    """
    n = len(values)

    if n < 10:
        return 0.0, 1.0

    # Compute spatial weights (binary based on distance)
    distances = squareform(pdist(coords))
    W = (distances < distance_threshold).astype(float)
    np.fill_diagonal(W, 0)

    # Row-standardize weights
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]

    # Standardize values
    z = values - values.mean()

    # Moran's I
    numerator = (W * np.outer(z, z)).sum()
    denominator = (z ** 2).sum()

    if denominator == 0:
        return 0.0, 1.0

    moran_i = (n / W.sum()) * (numerator / denominator)

    # Approximate p-value using normal approximation
    E_I = -1 / (n - 1)
    moran_i_z = (moran_i - E_I) / 0.2  # Approximate std
    p_value = 2 * (1 - stats.norm.cdf(abs(moran_i_z)))

    return moran_i, p_value


@dataclass
class SpatialCorrelation:
    """Results of spatial correlation analysis within raions."""
    source_name: str
    raion_key: str
    moran_i: float
    p_value: float
    n_points: int
    is_significant: bool  # p < 0.05


def compute_spatial_correlation_within_raions(
    df: pd.DataFrame,
    raion_col: str,
    value_col: str,
    source_name: str,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    min_points: int = 20
) -> List[SpatialCorrelation]:
    """
    Compute spatial autocorrelation within each raion.

    Tests whether nearby locations within a raion have similar values.
    High Moran's I indicates points within raion ARE similar (aggregation ok).
    Low Moran's I indicates points within raion are DIFFERENT (aggregation loses info).
    """
    results = []

    df_valid = df.dropna(subset=[raion_col, lat_col, lon_col, value_col])

    for raion in df_valid[raion_col].unique():
        if raion is None:
            continue

        raion_data = df_valid[df_valid[raion_col] == raion]

        if len(raion_data) < min_points:
            continue

        values = raion_data[value_col].values.astype(float)
        coords = raion_data[[lat_col, lon_col]].values

        moran_i, p_value = compute_moran_i(values, coords)

        results.append(SpatialCorrelation(
            source_name=source_name,
            raion_key=raion,
            moran_i=moran_i,
            p_value=p_value,
            n_points=len(raion_data),
            is_significant=p_value < 0.05,
        ))

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    variance_results: Dict[str, List[VarianceDecomposition]],
    entropy_results: Dict[str, List[EntropyAnalysis]],
    temporal_results: Dict[str, List[TemporalPreservation]],
    spatial_results: Dict[str, List[SpatialCorrelation]],
    output_path: Path
) -> str:
    """Generate comprehensive analysis report."""

    report = []
    report.append("# Raion-Level Aggregation Information Loss Analysis")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("This analysis investigates whether aggregating OSINT data at the raion")
    report.append("(district) level causes significant information loss for the Multi-Resolution HAN.")
    report.append("")

    # Variance summary
    report.append("## 1. Variance Decomposition Analysis")
    report.append("")
    report.append("### Methodology")
    report.append("We decompose total variance into within-raion and between-raion components.")
    report.append("The **variance ratio** (between/total) indicates how much information is")
    report.append("preserved by aggregation:")
    report.append("- High ratio (>0.7): Aggregation preserves most information")
    report.append("- Medium ratio (0.4-0.7): Moderate information loss")
    report.append("- Low ratio (<0.4): Significant information loss")
    report.append("")

    report.append("### Results by Data Source")
    report.append("")

    for source_name, results in variance_results.items():
        if not results:
            continue

        report.append(f"#### {source_name}")
        report.append("")
        report.append("| Feature | Total Var | Between Var | Within Var | Ratio | Info Preserved |")
        report.append("|---------|-----------|-------------|------------|-------|----------------|")

        # Sort by variance ratio (descending)
        results_sorted = sorted(results, key=lambda x: x.variance_ratio, reverse=True)

        for r in results_sorted[:10]:  # Top 10 features
            preserved_pct = r.variance_ratio * 100
            status = "HIGH" if r.variance_ratio > 0.7 else "MED" if r.variance_ratio > 0.4 else "LOW"
            report.append(
                f"| {r.feature_name[:30]} | {r.total_variance:.4f} | "
                f"{r.between_variance:.4f} | {r.within_variance:.4f} | "
                f"{r.variance_ratio:.3f} | {preserved_pct:.1f}% ({status}) |"
            )

        # Summary statistics
        avg_ratio = np.mean([r.variance_ratio for r in results])
        report.append("")
        report.append(f"**Average variance ratio**: {avg_ratio:.3f} ({avg_ratio*100:.1f}% information preserved)")
        report.append(f"**Features analyzed**: {len(results)}")
        report.append("")

    # Entropy summary
    report.append("## 2. Entropy Analysis")
    report.append("")
    report.append("### Methodology")
    report.append("We compare Shannon entropy of raw data vs raion-aggregated data.")
    report.append("**Information loss** = H(raw) - H(aggregated)")
    report.append("")

    report.append("### Results by Data Source")
    report.append("")

    for source_name, results in entropy_results.items():
        if not results:
            continue

        report.append(f"#### {source_name}")
        report.append("")
        report.append("| Feature | Raw Entropy | Agg Entropy | Info Loss | Relative Loss |")
        report.append("|---------|-------------|-------------|-----------|---------------|")

        results_sorted = sorted(results, key=lambda x: x.relative_loss, reverse=True)

        for r in results_sorted[:10]:
            report.append(
                f"| {r.feature_name[:30]} | {r.raw_entropy:.3f} | "
                f"{r.aggregated_entropy:.3f} | {r.information_loss:.3f} | "
                f"{r.relative_loss*100:.1f}% |"
            )

        avg_loss = np.mean([r.relative_loss for r in results])
        report.append("")
        report.append(f"**Average relative entropy loss**: {avg_loss*100:.1f}%")
        report.append("")

    # Temporal summary
    report.append("## 3. Temporal Pattern Preservation")
    report.append("")
    report.append("### Methodology")
    report.append("We compare autocorrelation functions (ACF) before and after aggregation.")
    report.append("High ACF correlation indicates temporal patterns are preserved.")
    report.append("")

    report.append("### Results by Data Source")
    report.append("")

    for source_name, results in temporal_results.items():
        if not results:
            continue

        report.append(f"#### {source_name}")
        report.append("")

        preserved_count = sum(1 for r in results if r.pattern_preserved)
        total_count = len(results)

        report.append(f"- **Patterns preserved**: {preserved_count}/{total_count} features")

        for r in results:
            status = "PRESERVED" if r.pattern_preserved else "DEGRADED"
            report.append(f"- {r.feature_name}: ACF correlation = {r.acf_correlation:.3f} ({status})")

        report.append("")

    # Spatial summary
    report.append("## 4. Spatial Correlation Analysis (Moran's I)")
    report.append("")
    report.append("### Methodology")
    report.append("We compute Moran's I spatial autocorrelation within each raion.")
    report.append("- **High positive Moran's I**: Points within raion are similar (aggregation OK)")
    report.append("- **Low/negative Moran's I**: Points within raion are different (aggregation loses info)")
    report.append("")

    report.append("### Results by Data Source")
    report.append("")

    for source_name, results in spatial_results.items():
        if not results:
            continue

        report.append(f"#### {source_name}")
        report.append("")

        # Compute summary statistics
        moran_values = [r.moran_i for r in results]
        significant_count = sum(1 for r in results if r.is_significant)
        positive_count = sum(1 for r in results if r.moran_i > 0 and r.is_significant)

        avg_moran = np.mean(moran_values) if moran_values else 0

        report.append(f"- **Average Moran's I**: {avg_moran:.3f}")
        report.append(f"- **Significant correlations**: {significant_count}/{len(results)} raions")
        report.append(f"- **Positive spatial clustering**: {positive_count}/{len(results)} raions")
        report.append("")

        # Interpretation
        if avg_moran > 0.3:
            report.append("**Interpretation**: Strong spatial clustering within raions suggests")
            report.append("raion aggregation preserves spatial structure well.")
        elif avg_moran > 0.1:
            report.append("**Interpretation**: Moderate spatial clustering. Some local patterns")
            report.append("may be lost but overall structure is preserved.")
        else:
            report.append("**Interpretation**: Weak spatial clustering within raions. Events are")
            report.append("spatially heterogeneous, suggesting significant information loss from aggregation.")
        report.append("")

    # Overall conclusions
    report.append("## 5. Overall Conclusions")
    report.append("")

    # Compute overall metrics
    all_variance_ratios = []
    for results in variance_results.values():
        all_variance_ratios.extend([r.variance_ratio for r in results])

    all_entropy_losses = []
    for results in entropy_results.values():
        all_entropy_losses.extend([r.relative_loss for r in results])

    all_temporal_preserved = []
    for results in temporal_results.values():
        all_temporal_preserved.extend([r.pattern_preserved for r in results])

    all_moran_values = []
    for results in spatial_results.values():
        all_moran_values.extend([r.moran_i for r in results])

    report.append("### Summary Statistics")
    report.append("")

    if all_variance_ratios:
        avg_vr = np.mean(all_variance_ratios)
        report.append(f"- **Overall variance ratio**: {avg_vr:.3f} ({avg_vr*100:.1f}% between-raion)")

    if all_entropy_losses:
        avg_el = np.mean(all_entropy_losses)
        report.append(f"- **Average entropy loss**: {avg_el*100:.1f}%")

    if all_temporal_preserved:
        pct_preserved = 100 * sum(all_temporal_preserved) / len(all_temporal_preserved)
        report.append(f"- **Temporal patterns preserved**: {pct_preserved:.0f}%")

    if all_moran_values:
        avg_mi = np.mean(all_moran_values)
        report.append(f"- **Average Moran's I**: {avg_mi:.3f}")

    report.append("")
    report.append("### Recommendations")
    report.append("")

    # Determine overall assessment
    high_info_loss_sources = []
    low_info_loss_sources = []

    for source_name, results in variance_results.items():
        if results:
            avg_ratio = np.mean([r.variance_ratio for r in results])
            if avg_ratio < 0.4:
                high_info_loss_sources.append(source_name)
            elif avg_ratio > 0.7:
                low_info_loss_sources.append(source_name)

    if high_info_loss_sources:
        report.append("**Sources with HIGH information loss** (consider finer granularity):")
        for s in high_info_loss_sources:
            report.append(f"- {s}")
        report.append("")

    if low_info_loss_sources:
        report.append("**Sources with LOW information loss** (raion aggregation is appropriate):")
        for s in low_info_loss_sources:
            report.append(f"- {s}")
        report.append("")

    report.append("### Statistical Significance")
    report.append("")
    report.append("All analyses use standard statistical methods:")
    report.append("- Variance decomposition: ANOVA-style within/between decomposition")
    report.append("- Entropy: Shannon entropy with histogram binning (50 bins)")
    report.append("- Moran's I: Binary spatial weights with normal approximation p-values")
    report.append("- Significance threshold: p < 0.05")
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
    """Run complete raion information loss analysis."""
    print("=" * 80)
    print("RAION-LEVEL AGGREGATION INFORMATION LOSS ANALYSIS")
    print("=" * 80)
    print()

    # Results storage
    variance_results: Dict[str, List[VarianceDecomposition]] = {}
    entropy_results: Dict[str, List[EntropyAnalysis]] = {}
    temporal_results: Dict[str, List[TemporalPreservation]] = {}
    spatial_results: Dict[str, List[SpatialCorrelation]] = {}

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

        # Assign to raions
        print("  Assigning events to raions...")
        ucdp_df['raion'] = assign_points_to_raions(ucdp_df, raion_manager)
        assigned_pct = 100 * ucdp_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of events assigned to raions")

        # Define numeric features for analysis
        ucdp_features = ['best_est', 'deaths_a', 'deaths_b', 'deaths_civilians']

        # Variance decomposition
        print("  Computing variance decomposition...")
        variance_results['UCDP'] = compute_variance_decomposition(
            ucdp_df, 'raion', ucdp_features, 'UCDP'
        )

        # Entropy analysis
        print("  Computing entropy analysis...")
        entropy_results['UCDP'] = compute_entropy_analysis(
            ucdp_df, 'raion', ucdp_features, 'UCDP'
        )

        # Temporal preservation
        print("  Computing temporal preservation...")
        temporal_results['UCDP'] = []
        for feat in ucdp_features:
            result = compute_temporal_preservation(ucdp_df, 'raion', feat, 'UCDP')
            if result:
                temporal_results['UCDP'].append(result)

        # Spatial correlation
        print("  Computing spatial correlation...")
        spatial_results['UCDP'] = compute_spatial_correlation_within_raions(
            ucdp_df, 'raion', 'best_est', 'UCDP'
        )

    except Exception as e:
        print(f"  ERROR analyzing UCDP data: {e}")

    # ==========================================================================
    # Analyze FIRMS Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("ANALYZING FIRMS DATA")
    print("-" * 60)

    try:
        firms_df = load_firms_fires()

        # Assign to raions
        print("  Assigning fires to raions...")
        firms_df['raion'] = assign_points_to_raions(firms_df, raion_manager)
        assigned_pct = 100 * firms_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of fires assigned to raions")

        # Define numeric features
        firms_features = ['brightness', 'bright_t31', 'frp', 'confidence']
        # Convert confidence to numeric
        if 'confidence' in firms_df.columns:
            firms_df['confidence_num'] = firms_df['confidence'].map({'l': 0, 'n': 1, 'h': 2})
            firms_features = ['brightness', 'bright_t31', 'frp', 'confidence_num']

        # Variance decomposition
        print("  Computing variance decomposition...")
        variance_results['FIRMS'] = compute_variance_decomposition(
            firms_df, 'raion', firms_features, 'FIRMS'
        )

        # Entropy analysis
        print("  Computing entropy analysis...")
        entropy_results['FIRMS'] = compute_entropy_analysis(
            firms_df, 'raion', firms_features, 'FIRMS'
        )

        # Temporal preservation
        print("  Computing temporal preservation...")
        temporal_results['FIRMS'] = []
        for feat in firms_features[:3]:  # Skip categorical
            result = compute_temporal_preservation(firms_df, 'raion', feat, 'FIRMS')
            if result:
                temporal_results['FIRMS'].append(result)

        # Spatial correlation
        print("  Computing spatial correlation...")
        spatial_results['FIRMS'] = compute_spatial_correlation_within_raions(
            firms_df, 'raion', 'frp', 'FIRMS'
        )

    except Exception as e:
        print(f"  ERROR analyzing FIRMS data: {e}")

    # ==========================================================================
    # Analyze Geoconfirmed Data
    # ==========================================================================
    print("\n" + "-" * 60)
    print("ANALYZING GEOCONFIRMED DATA")
    print("-" * 60)

    try:
        geo_df = load_geoconfirmed_events()

        # Assign to raions
        print("  Assigning events to raions...")
        geo_df['raion'] = assign_points_to_raions(geo_df, raion_manager)
        assigned_pct = 100 * geo_df['raion'].notna().mean()
        print(f"  {assigned_pct:.1f}% of events assigned to raions")

        # Create binary features for analysis
        geo_df['has_tank'] = geo_df['gear'].str.lower().str.contains('tank', na=False).astype(int)
        geo_df['has_artillery'] = geo_df['gear'].str.lower().str.contains('artillery|howitzer', na=False).astype(int)
        geo_df['has_drone'] = geo_df['gear'].str.lower().str.contains('drone|uav', na=False).astype(int)
        geo_df['event_count'] = 1

        geo_features = ['event_count', 'has_tank', 'has_artillery', 'has_drone']

        # Variance decomposition
        print("  Computing variance decomposition...")
        variance_results['Geoconfirmed'] = compute_variance_decomposition(
            geo_df, 'raion', geo_features, 'Geoconfirmed'
        )

        # Entropy analysis
        print("  Computing entropy analysis...")
        entropy_results['Geoconfirmed'] = compute_entropy_analysis(
            geo_df, 'raion', geo_features, 'Geoconfirmed'
        )

        # Temporal preservation
        print("  Computing temporal preservation...")
        temporal_results['Geoconfirmed'] = []
        result = compute_temporal_preservation(geo_df, 'raion', 'event_count', 'Geoconfirmed')
        if result:
            temporal_results['Geoconfirmed'].append(result)

        # Spatial correlation
        print("  Computing spatial correlation...")
        spatial_results['Geoconfirmed'] = compute_spatial_correlation_within_raions(
            geo_df, 'raion', 'event_count', 'Geoconfirmed'
        )

    except Exception as e:
        print(f"  ERROR analyzing Geoconfirmed data: {e}")

    # ==========================================================================
    # Generate Report
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    output_path = ANALYSIS_OUTPUT_DIR / "raion_information_loss_report.md"

    report = generate_report(
        variance_results,
        entropy_results,
        temporal_results,
        spatial_results,
        output_path
    )

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Quick summary
    all_variance_ratios = []
    for results in variance_results.values():
        all_variance_ratios.extend([r.variance_ratio for r in results])

    if all_variance_ratios:
        avg_vr = np.mean(all_variance_ratios)
        print(f"\nOverall average variance ratio: {avg_vr:.3f}")
        print(f"Information preserved by raion aggregation: {avg_vr*100:.1f}%")
        print(f"Information lost: {(1-avg_vr)*100:.1f}%")

    print(f"\nFull report saved to: {output_path}")


if __name__ == '__main__':
    main()
