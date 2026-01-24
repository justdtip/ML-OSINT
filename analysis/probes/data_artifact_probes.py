"""
Data Artifact Probes for Multi-Resolution HAN Model Investigation

This module implements comprehensive statistical probes to investigate data artifacts
in the MultiResolutionHAN model, specifically addressing:
- Equipment Signal Degradation Analysis (1.1.x)
- VIIRS Dominance Investigation (1.2.x)
- Personnel Data Quality Check (1.3.x)

Each probe follows a standardized output format for reproducibility and documentation.

Author: Data Science Team
Date: 2026-01-23
"""

from __future__ import annotations

import os
import json
import warnings

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    warnings.warn("PyYAML not available - will use JSON for serialization")
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate, correlation_lags
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch

# Statistical imports
try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available - some probes will be limited")

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.mediation import Mediation
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import ccf, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available - mediation analysis will be limited")

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR as CONFIG_DATA_DIR,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    MULTI_RES_CHECKPOINT_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# =============================================================================
# CONSTANTS AND PATHS
# =============================================================================

BASE_DIR = PROJECT_ROOT
DATA_DIR = CONFIG_DATA_DIR
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR
CHECKPOINT_DIR = MULTI_RES_CHECKPOINT_DIR


def get_output_dir():
    """Get the current output directory for figures."""
    return get_probe_figures_dir()


# Note: OUTPUT_DIR references in this file now use get_output_dir() dynamically
# The directory is created by the MasterProbeRunner before probes run

# Source configurations
DAILY_SOURCES = ['equipment', 'personnel', 'deepstate', 'firms', 'viina', 'viirs']
MONTHLY_SOURCES = ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']

# Equipment categories for disaggregated analysis
EQUIPMENT_CATEGORIES = {
    'tanks': ['tank'],
    'apcs': ['APC'],
    'artillery': ['field_artillery', 'MRL'],
    'aircraft': ['aircraft', 'helicopter'],
    'air_defense': ['anti_aircraft_warfare'],
    'drones': ['drone'],
    'naval': ['naval_ship'],
    'vehicles': ['vehicles_and_fuel_tanks', 'special_equipment'],
    'missiles': ['cruise_missiles']
}

# VIIRS feature names
VIIRS_FEATURES = [
    'viirs_radiance_mean', 'viirs_radiance_std', 'viirs_radiance_anomaly',
    'viirs_clear_sky_pct', 'viirs_coverage_count', 'viirs_radiance_p50',
    'viirs_radiance_p90', 'viirs_moon_illumination'
]


# =============================================================================
# PROBE OUTPUT FORMAT
# =============================================================================

@dataclass
class ProbeResult:
    """Standardized output format for probe results."""
    test_id: str
    test_name: str
    findings: List[Dict[str, Any]]
    artifacts: Dict[str, List[str]] = field(default_factory=lambda: {'figures': [], 'tables': []})
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Convert result to YAML string (or JSON if PyYAML not available)."""
        if HAS_YAML:
            return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
        else:
            return json.dumps(asdict(self), indent=2, default=str)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save result to YAML/JSON file."""
        if path is None:
            ext = '.yaml' if HAS_YAML else '.json'
            path = get_probe_metrics_dir() / f"probe_{self.test_id.replace('.', '_')}{ext}"
        with open(path, 'w') as f:
            f.write(self.to_yaml())
        return path

    def __str__(self) -> str:
        return self.to_yaml()


# =============================================================================
# BASE PROBE CLASS
# =============================================================================

class Probe(ABC):
    """Base class for all data artifact probes."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.result: Optional[ProbeResult] = None

    @property
    @abstractmethod
    def test_id(self) -> str:
        """Unique identifier for this probe (e.g., '1.1.1')."""
        pass

    @property
    @abstractmethod
    def test_name(self) -> str:
        """Human-readable name for this probe."""
        pass

    @abstractmethod
    def run(self, data: Dict[str, Any]) -> ProbeResult:
        """Execute the probe and return results."""
        pass

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.test_id}] {message}")

    def save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save a matplotlib figure and return the path."""
        path = get_output_dir() / f"{self.test_id.replace('.', '_')}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return str(path)

    def save_table(self, df: pd.DataFrame, name: str) -> str:
        """Save a DataFrame as CSV and return the path."""
        path = get_probe_metrics_dir() / f"{self.test_id.replace('.', '_')}_{name}.csv"
        df.to_csv(path, index=True)
        return str(path)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_equipment_raw() -> pd.DataFrame:
    """Load raw equipment loss data."""
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        raise FileNotFoundError(f"Equipment data not found at {equip_path}")

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    return df


def load_personnel_raw() -> pd.DataFrame:
    """Load raw personnel loss data."""
    personnel_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"

    if not personnel_path.exists():
        raise FileNotFoundError(f"Personnel data not found at {personnel_path}")

    with open(personnel_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

    return df


def load_viirs_raw() -> pd.DataFrame:
    """Load raw VIIRS nightlight data."""
    viirs_path = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"

    if not viirs_path.exists():
        raise FileNotFoundError(f"VIIRS data not found at {viirs_path}")

    df = pd.read_csv(viirs_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Aggregate across tiles
    daily_agg = df.groupby('date').agg({
        'radiance_mean': 'mean',
        'radiance_std': 'mean',
        'radiance_min': 'min',
        'radiance_max': 'max',
        'pct_clear_sky': 'mean',
        'radiance_p50': 'mean',
        'radiance_p90': 'mean',
        'moon_illumination_pct': 'mean',
    }).reset_index()

    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)

    return daily_agg


def compute_partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation between x and y, controlling for z.

    Uses the formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Returns:
        partial_r: Partial correlation coefficient
        p_value: Significance of partial correlation (approximate)
    """
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]

    if len(x) < 4:
        return np.nan, 1.0

    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)

    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denominator < 1e-10:
        return np.nan, 1.0

    partial_r = (r_xy - r_xz * r_yz) / denominator

    # Approximate p-value using t-distribution
    n = len(x)
    df = n - 3  # degrees of freedom for partial correlation
    if df <= 0:
        return partial_r, 1.0

    t_stat = partial_r * np.sqrt(df / (1 - partial_r**2 + 1e-10))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return partial_r, p_value


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Compute mutual information between x and y using k-nearest neighbors.

    Returns:
        mi: Mutual information estimate
    """
    if not HAS_SKLEARN:
        warnings.warn("sklearn not available, returning NaN for mutual information")
        return np.nan

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < n_neighbors + 1:
        return np.nan

    # Mutual information requires X as 2D array
    X = x.reshape(-1, 1)

    mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)[0]

    return mi


def cross_correlation_at_lags(x: np.ndarray, y: np.ndarray, max_lag: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between x and y at various lags.

    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to compute (both positive and negative)

    Returns:
        lags: Array of lag values from -max_lag to +max_lag
        correlations: Cross-correlation values at each lag
    """
    # Remove NaN and align
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    if len(x) < 2 * max_lag:
        return np.array([]), np.array([])

    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    # Compute cross-correlation
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            # x leads y (y lags x)
            correlations[i] = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            # y leads x (x lags y)
            correlations[i] = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            correlations[i] = np.corrcoef(x, y)[0, 1]

    return lags, correlations


# =============================================================================
# SECTION 1.1: EQUIPMENT SIGNAL DEGRADATION ANALYSIS
# =============================================================================

class EncodingVarianceComparisonProbe(Probe):
    """
    Probe 1.1.1: Compare variance of raw cumulative vs delta vs rolling delta encodings.

    This probe investigates whether equipment loss signal is degraded by encoding choice.
    """

    @property
    def test_id(self) -> str:
        return "1.1.1"

    @property
    def test_name(self) -> str:
        return "Encoding Variance Comparison"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting encoding variance comparison...")

        # Load equipment data
        df = load_equipment_raw()

        # Define equipment columns
        equip_cols = ['tank', 'APC', 'field_artillery', 'MRL', 'aircraft',
                      'helicopter', 'drone', 'naval_ship', 'anti_aircraft_warfare']
        equip_cols = [c for c in equip_cols if c in df.columns]

        findings = []
        figures = []
        tables = []

        # Compute different encodings
        encodings = {}
        for col in equip_cols:
            cumulative = df[col].values.astype(float)
            delta = np.diff(cumulative, prepend=cumulative[0])
            rolling_delta_7 = pd.Series(delta).rolling(7, min_periods=1).mean().values
            rolling_delta_14 = pd.Series(delta).rolling(14, min_periods=1).mean().values

            encodings[col] = {
                'cumulative': cumulative,
                'delta': delta,
                'rolling_7': rolling_delta_7,
                'rolling_14': rolling_delta_14
            }

        # Compute variance statistics
        variance_stats = []
        for col in equip_cols:
            for enc_type, values in encodings[col].items():
                # Skip NaN
                valid_vals = values[~np.isnan(values)]
                if len(valid_vals) > 0:
                    variance_stats.append({
                        'equipment_type': col,
                        'encoding': enc_type,
                        'variance': np.var(valid_vals),
                        'std': np.std(valid_vals),
                        'mean': np.mean(valid_vals),
                        'cv': np.std(valid_vals) / (np.abs(np.mean(valid_vals)) + 1e-10),  # coefficient of variation
                        'range': np.max(valid_vals) - np.min(valid_vals)
                    })

        variance_df = pd.DataFrame(variance_stats)
        tables.append(self.save_table(variance_df, 'encoding_variance_stats'))

        # Key finding: Compare cumulative vs delta variance ratios
        cumulative_cv_mean = variance_df[variance_df['encoding'] == 'cumulative']['cv'].mean()
        delta_cv_mean = variance_df[variance_df['encoding'] == 'delta']['cv'].mean()
        rolling_7_cv_mean = variance_df[variance_df['encoding'] == 'rolling_7']['cv'].mean()

        findings.append({
            'key_result': f"Coefficient of Variation - Cumulative: {cumulative_cv_mean:.4f}, Delta: {delta_cv_mean:.4f}, Rolling-7: {rolling_7_cv_mean:.4f}",
            'significance': f"Delta encoding has {delta_cv_mean/cumulative_cv_mean:.2f}x higher relative variance",
            'interpretation': "Cumulative encoding has very low CV due to monotonic increase, potentially masking daily variation signal"
        })

        # Stationarity test (ADF) for each encoding
        if HAS_STATSMODELS:
            adf_results = []
            for col in equip_cols[:3]:  # Test top 3 for efficiency
                for enc_type in ['cumulative', 'delta', 'rolling_7']:
                    values = encodings[col][enc_type]
                    valid_vals = values[~np.isnan(values)]
                    if len(valid_vals) > 20:
                        adf_stat, adf_p, _, _, _, _ = adfuller(valid_vals, maxlag=7)
                        adf_results.append({
                            'equipment_type': col,
                            'encoding': enc_type,
                            'adf_statistic': adf_stat,
                            'adf_pvalue': adf_p,
                            'is_stationary': adf_p < 0.05
                        })

            adf_df = pd.DataFrame(adf_results)
            tables.append(self.save_table(adf_df, 'stationarity_tests'))

            # Finding about stationarity
            cumulative_stationary = adf_df[(adf_df['encoding'] == 'cumulative') & (adf_df['is_stationary'])].shape[0]
            delta_stationary = adf_df[(adf_df['encoding'] == 'delta') & (adf_df['is_stationary'])].shape[0]

            findings.append({
                'key_result': f"Stationarity: Cumulative {cumulative_stationary}/{len(equip_cols[:3])} stationary, Delta {delta_stationary}/{len(equip_cols[:3])} stationary",
                'significance': "ADF test p < 0.05 indicates stationarity",
                'interpretation': "Non-stationary cumulative series may cause trend confounding in neural networks"
            })

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Variance by encoding type
        ax = axes[0, 0]
        pivot_var = variance_df.pivot(index='equipment_type', columns='encoding', values='variance')
        pivot_var.plot(kind='bar', ax=ax, logy=True)
        ax.set_title('Variance by Equipment Type and Encoding')
        ax.set_ylabel('Variance (log scale)')
        ax.legend(title='Encoding')
        ax.tick_params(axis='x', rotation=45)

        # Plot 2: Coefficient of Variation
        ax = axes[0, 1]
        pivot_cv = variance_df.pivot(index='equipment_type', columns='encoding', values='cv')
        pivot_cv.plot(kind='bar', ax=ax)
        ax.set_title('Coefficient of Variation by Encoding')
        ax.set_ylabel('CV')
        ax.legend(title='Encoding')
        ax.tick_params(axis='x', rotation=45)

        # Plot 3: Example time series for 'tank'
        ax = axes[1, 0]
        if 'tank' in encodings:
            dates = df['date'].values
            ax.plot(dates, encodings['tank']['cumulative'], label='Cumulative', alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Tank Losses (Cumulative)')
            ax.set_title('Tank Losses: Cumulative Encoding')
            ax.legend()

        # Plot 4: Delta encoding for 'tank'
        ax = axes[1, 1]
        if 'tank' in encodings:
            ax.plot(dates, encodings['tank']['delta'], label='Daily Delta', alpha=0.5)
            ax.plot(dates, encodings['tank']['rolling_7'], label='7-day Rolling', linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Tank Losses (Daily)')
            ax.set_title('Tank Losses: Delta Encodings')
            ax.legend()

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'encoding_comparison'))

        # Recommendations
        recommendations = [
            "Consider using delta encoding instead of cumulative to capture daily variation",
            "Apply 7-day rolling average to reduce noise while preserving signal",
            "Test model performance with different encodings in ablation study",
            "Consider log-transform for delta values to handle large spikes"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_equipment_types': len(equip_cols),
                'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
                'n_observations': len(df)
            }
        )

        return self.result


class EquipmentPersonnelRedundancyProbe(Probe):
    """
    Probe 1.1.2: Compute correlation, partial correlation, and mutual information
    between equipment and personnel losses.
    """

    @property
    def test_id(self) -> str:
        return "1.1.2"

    @property
    def test_name(self) -> str:
        return "Equipment-Personnel Redundancy Test"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting equipment-personnel redundancy analysis...")

        # Load data
        equipment_df = load_equipment_raw()
        personnel_df = load_personnel_raw()

        # Merge on date
        merged = equipment_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')

        # Compute personnel daily change
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        # Equipment columns
        equip_cols = ['tank', 'APC', 'field_artillery', 'MRL', 'aircraft',
                      'helicopter', 'drone', 'naval_ship', 'anti_aircraft_warfare']
        equip_cols = [c for c in equip_cols if c in merged.columns]

        # Compute delta for equipment
        for col in equip_cols:
            merged[f'{col}_delta'] = merged[col].diff().fillna(0)

        findings = []
        figures = []
        tables = []

        # Compute correlations
        correlation_results = []

        for col in equip_cols:
            equip_delta = merged[f'{col}_delta'].values
            personnel_delta = merged['personnel_delta'].values

            # Pearson correlation
            r_pearson, p_pearson = pearsonr(equip_delta, personnel_delta)

            # Spearman correlation
            r_spearman, p_spearman = spearmanr(equip_delta, personnel_delta)

            # Mutual information
            mi = compute_mutual_information(equip_delta, personnel_delta)

            correlation_results.append({
                'equipment_type': col,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'spearman_p': p_spearman,
                'mutual_info': mi
            })

        corr_df = pd.DataFrame(correlation_results)
        tables.append(self.save_table(corr_df, 'equipment_personnel_correlations'))

        # Key findings
        highest_corr = corr_df.loc[corr_df['pearson_r'].abs().idxmax()]
        mean_corr = corr_df['pearson_r'].mean()

        findings.append({
            'key_result': f"Mean Pearson correlation between equipment delta and personnel delta: {mean_corr:.4f}",
            'significance': f"Highest correlation: {highest_corr['equipment_type']} (r={highest_corr['pearson_r']:.4f}, p={highest_corr['pearson_p']:.2e})",
            'interpretation': "Positive correlations suggest equipment and personnel losses are temporally synchronized"
        })

        # Partial correlations controlling for time trend
        time_index = np.arange(len(merged))
        partial_corr_results = []

        for col in equip_cols:
            equip_delta = merged[f'{col}_delta'].values
            personnel_delta = merged['personnel_delta'].values

            partial_r, partial_p = compute_partial_correlation(
                equip_delta, personnel_delta, time_index
            )

            partial_corr_results.append({
                'equipment_type': col,
                'partial_r': partial_r,
                'partial_p': partial_p,
                'raw_r': corr_df[corr_df['equipment_type'] == col]['pearson_r'].values[0],
                'reduction': 1 - abs(partial_r) / (abs(corr_df[corr_df['equipment_type'] == col]['pearson_r'].values[0]) + 1e-10)
            })

        partial_df = pd.DataFrame(partial_corr_results)
        tables.append(self.save_table(partial_df, 'partial_correlations'))

        mean_reduction = partial_df['reduction'].mean()
        findings.append({
            'key_result': f"Mean correlation reduction after controlling for time: {mean_reduction:.1%}",
            'significance': f"Partial correlation analysis with time as confound",
            'interpretation': "High reduction suggests time trend explains much of the correlation (redundancy may be spurious)"
        })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Correlation heatmap
        ax = axes[0, 0]
        corr_matrix = merged[[f'{c}_delta' for c in equip_cols] + ['personnel_delta']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Correlation Matrix: Equipment Delta vs Personnel Delta')

        # Plot 2: Scatter plot for highest correlation equipment
        ax = axes[0, 1]
        top_equip = highest_corr['equipment_type']
        ax.scatter(merged[f'{top_equip}_delta'], merged['personnel_delta'], alpha=0.3)
        ax.set_xlabel(f'{top_equip} daily losses')
        ax.set_ylabel('Personnel daily losses')
        ax.set_title(f'{top_equip} vs Personnel (r={highest_corr["pearson_r"]:.3f})')

        # Plot 3: Comparison of raw vs partial correlations
        ax = axes[1, 0]
        x = np.arange(len(equip_cols))
        width = 0.35
        ax.bar(x - width/2, partial_df['raw_r'], width, label='Raw r', alpha=0.8)
        ax.bar(x + width/2, partial_df['partial_r'], width, label='Partial r (time)', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(equip_cols, rotation=45, ha='right')
        ax.set_ylabel('Correlation')
        ax.set_title('Raw vs Partial Correlation (controlling for time)')
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Plot 4: Mutual information comparison
        ax = axes[1, 1]
        ax.bar(range(len(equip_cols)), corr_df['mutual_info'])
        ax.set_xticks(range(len(equip_cols)))
        ax.set_xticklabels(equip_cols, rotation=45, ha='right')
        ax.set_ylabel('Mutual Information (nats)')
        ax.set_title('Mutual Information: Equipment Delta vs Personnel Delta')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'redundancy_analysis'))

        recommendations = [
            "Consider removing highly correlated equipment types to reduce redundancy",
            "Use partial correlations to assess true predictive value beyond time trends",
            "Investigate whether equipment losses predict personnel losses at specific lags",
            "Consider PCA or similar dimensionality reduction for equipment features"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_observations': len(merged),
                'n_equipment_types': len(equip_cols),
                'date_range': f"{merged['date'].min().date()} to {merged['date'].max().date()}"
            }
        )

        return self.result


class EquipmentCategoryDisaggregationProbe(Probe):
    """
    Probe 1.1.3: Per-category (tanks, APCs, etc.) correlation with casualties.
    """

    @property
    def test_id(self) -> str:
        return "1.1.3"

    @property
    def test_name(self) -> str:
        return "Equipment Category Disaggregation"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting equipment category disaggregation analysis...")

        # Load data
        equipment_df = load_equipment_raw()
        personnel_df = load_personnel_raw()

        # Merge
        merged = equipment_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        findings = []
        figures = []
        tables = []

        # Analyze by category
        category_results = []

        for category, cols in EQUIPMENT_CATEGORIES.items():
            available_cols = [c for c in cols if c in merged.columns]
            if not available_cols:
                continue

            # Sum category losses
            merged[f'{category}_total'] = merged[available_cols].sum(axis=1)
            merged[f'{category}_delta'] = merged[f'{category}_total'].diff().fillna(0)

            cat_delta = merged[f'{category}_delta'].values
            personnel_delta = merged['personnel_delta'].values

            # Correlations
            r_pearson, p_pearson = pearsonr(cat_delta, personnel_delta)
            r_spearman, p_spearman = spearmanr(cat_delta, personnel_delta)
            mi = compute_mutual_information(cat_delta, personnel_delta)

            # Compute category contribution (% of total equipment losses)
            total_cat = merged[f'{category}_total'].max()

            category_results.append({
                'category': category,
                'equipment_types': available_cols,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'spearman_r': r_spearman,
                'mutual_info': mi,
                'total_losses': total_cat,
                'daily_mean': np.mean(cat_delta),
                'daily_std': np.std(cat_delta)
            })

        cat_df = pd.DataFrame(category_results)
        tables.append(self.save_table(cat_df, 'category_correlations'))

        # Sort by correlation
        cat_df_sorted = cat_df.sort_values('pearson_r', ascending=False)

        findings.append({
            'key_result': f"Highest personnel correlation: {cat_df_sorted.iloc[0]['category']} (r={cat_df_sorted.iloc[0]['pearson_r']:.4f})",
            'significance': f"p-value = {cat_df_sorted.iloc[0]['pearson_p']:.2e}",
            'interpretation': "This equipment category most closely tracks personnel casualties"
        })

        findings.append({
            'key_result': f"Lowest personnel correlation: {cat_df_sorted.iloc[-1]['category']} (r={cat_df_sorted.iloc[-1]['pearson_r']:.4f})",
            'significance': f"p-value = {cat_df_sorted.iloc[-1]['pearson_p']:.2e}",
            'interpretation': "This equipment category may be less informative for casualty prediction"
        })

        # Time-lagged correlations for top categories
        lag_analysis = []
        for _, row in cat_df_sorted.head(3).iterrows():
            category = row['category']
            cat_delta = merged[f'{category}_delta'].values
            personnel_delta = merged['personnel_delta'].values

            lags, correlations = cross_correlation_at_lags(cat_delta, personnel_delta, max_lag=14)

            if len(lags) > 0:
                best_lag = lags[np.argmax(np.abs(correlations))]
                best_corr = correlations[np.argmax(np.abs(correlations))]

                lag_analysis.append({
                    'category': category,
                    'best_lag': best_lag,
                    'best_lag_corr': best_corr,
                    'lag_interpretation': 'equipment leads' if best_lag < 0 else ('concurrent' if best_lag == 0 else 'equipment lags')
                })

        lag_df = pd.DataFrame(lag_analysis)
        tables.append(self.save_table(lag_df, 'lag_analysis'))

        if len(lag_df) > 0:
            findings.append({
                'key_result': f"Optimal lag analysis: {lag_df.to_dict('records')}",
                'significance': "Positive lag = equipment losses precede personnel casualties",
                'interpretation': "Lag structure indicates whether equipment provides leading signal"
            })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Correlation by category
        ax = axes[0, 0]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(cat_df)))
        bars = ax.barh(cat_df_sorted['category'], cat_df_sorted['pearson_r'], color=colors)
        ax.set_xlabel('Pearson Correlation with Personnel Losses')
        ax.set_title('Equipment Category Correlation with Casualties')
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Plot 2: Mutual information by category
        ax = axes[0, 1]
        ax.barh(cat_df_sorted['category'], cat_df_sorted['mutual_info'])
        ax.set_xlabel('Mutual Information (nats)')
        ax.set_title('Mutual Information with Personnel Losses')

        # Plot 3: Total losses by category
        ax = axes[1, 0]
        ax.barh(cat_df_sorted['category'], cat_df_sorted['total_losses'])
        ax.set_xlabel('Total Cumulative Losses')
        ax.set_title('Total Losses by Equipment Category')

        # Plot 4: Lag correlation plot for top category
        if len(lag_df) > 0:
            ax = axes[1, 1]
            top_cat = cat_df_sorted.iloc[0]['category']
            cat_delta = merged[f'{top_cat}_delta'].values
            personnel_delta = merged['personnel_delta'].values
            lags, correlations = cross_correlation_at_lags(cat_delta, personnel_delta, max_lag=14)

            if len(lags) > 0:
                ax.plot(lags, correlations, 'b-o')
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
                ax.set_xlabel('Lag (days)')
                ax.set_ylabel('Cross-correlation')
                ax.set_title(f'{top_cat} vs Personnel: Cross-correlation at Lags')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'category_analysis'))

        recommendations = [
            f"Focus model attention on {cat_df_sorted.iloc[0]['category']} - highest correlation with casualties",
            f"Consider dropping or downweighting {cat_df_sorted.iloc[-1]['category']} - minimal correlation",
            "Use lag structure in feature engineering for predictive modeling",
            "Investigate why some categories correlate more strongly (tactical significance)"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_categories': len(cat_df),
                'date_range': f"{merged['date'].min().date()} to {merged['date'].max().date()}"
            }
        )

        return self.result


class TemporalLagAnalysisProbe(Probe):
    """
    Probe 1.1.4: Cross-correlation at lags [-30 to +30] days for equipment vs casualties.
    """

    @property
    def test_id(self) -> str:
        return "1.1.4"

    @property
    def test_name(self) -> str:
        return "Temporal Lag Analysis"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting temporal lag analysis...")

        # Load data
        equipment_df = load_equipment_raw()
        personnel_df = load_personnel_raw()

        merged = equipment_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        equip_cols = ['tank', 'APC', 'field_artillery', 'MRL', 'aircraft',
                      'helicopter', 'drone', 'anti_aircraft_warfare']
        equip_cols = [c for c in equip_cols if c in merged.columns]

        findings = []
        figures = []
        tables = []

        # Compute total equipment losses
        merged['total_equipment'] = merged[equip_cols].sum(axis=1)
        merged['total_equipment_delta'] = merged['total_equipment'].diff().fillna(0)

        # Cross-correlation analysis for multiple equipment types
        lag_results = []
        max_lag = 30

        for col in equip_cols + ['total_equipment']:
            if col == 'total_equipment':
                delta_col = 'total_equipment_delta'
            else:
                merged[f'{col}_delta'] = merged[col].diff().fillna(0)
                delta_col = f'{col}_delta'

            equip_delta = merged[delta_col].values
            personnel_delta = merged['personnel_delta'].values

            lags, correlations = cross_correlation_at_lags(equip_delta, personnel_delta, max_lag=max_lag)

            if len(lags) > 0:
                # Find optimal lag
                best_idx = np.argmax(np.abs(correlations))
                best_lag = lags[best_idx]
                best_corr = correlations[best_idx]

                # Concurrent correlation (lag=0)
                zero_idx = np.where(lags == 0)[0]
                concurrent_corr = correlations[zero_idx[0]] if len(zero_idx) > 0 else np.nan

                lag_results.append({
                    'equipment_type': col,
                    'optimal_lag': best_lag,
                    'optimal_correlation': best_corr,
                    'concurrent_correlation': concurrent_corr,
                    'lag_type': 'leading' if best_lag < 0 else ('concurrent' if best_lag == 0 else 'lagging'),
                    'correlations_array': correlations.tolist()
                })

        lag_df = pd.DataFrame(lag_results)
        # Save without the array column
        lag_df_save = lag_df.drop(columns=['correlations_array'])
        tables.append(self.save_table(lag_df_save, 'lag_analysis'))

        # Key findings
        leading = lag_df[lag_df['optimal_lag'] < 0]
        lagging = lag_df[lag_df['optimal_lag'] > 0]
        concurrent = lag_df[lag_df['optimal_lag'] == 0]

        findings.append({
            'key_result': f"Leading equipment (precede casualties): {list(leading['equipment_type'])}",
            'significance': f"Average lead time: {leading['optimal_lag'].mean():.1f} days" if len(leading) > 0 else "N/A",
            'interpretation': "These equipment types may have predictive value for upcoming casualties"
        })

        findings.append({
            'key_result': f"Lagging equipment (follow casualties): {list(lagging['equipment_type'])}",
            'significance': f"Average lag time: {lagging['optimal_lag'].mean():.1f} days" if len(lagging) > 0 else "N/A",
            'interpretation': "These equipment losses may be consequence of casualty-causing events"
        })

        # Total equipment analysis
        total_row = lag_df[lag_df['equipment_type'] == 'total_equipment'].iloc[0]
        findings.append({
            'key_result': f"Total equipment optimal lag: {total_row['optimal_lag']} days (r={total_row['optimal_correlation']:.4f})",
            'significance': f"Concurrent correlation: {total_row['concurrent_correlation']:.4f}",
            'interpretation': "Overall equipment-personnel temporal relationship"
        })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cross-correlation for total equipment
        ax = axes[0, 0]
        total_corrs = lag_df[lag_df['equipment_type'] == 'total_equipment']['correlations_array'].values[0]
        lags_plot = np.arange(-max_lag, max_lag + 1)
        ax.plot(lags_plot, total_corrs, 'b-o', markersize=3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Lag 0')
        ax.axvline(x=total_row['optimal_lag'], color='g', linestyle='--', linewidth=1, label=f'Optimal lag={total_row["optimal_lag"]}')
        ax.fill_between(lags_plot, 0, total_corrs, alpha=0.3)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('Total Equipment vs Personnel: Cross-correlation')
        ax.legend()

        # Plot 2: Optimal lag by equipment type
        ax = axes[0, 1]
        colors = ['green' if l < 0 else ('blue' if l == 0 else 'red') for l in lag_df_save['optimal_lag']]
        ax.barh(lag_df_save['equipment_type'], lag_df_save['optimal_lag'], color=colors)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('Optimal Lag (days)')
        ax.set_title('Optimal Lag by Equipment Type')
        ax.legend(handles=[
            plt.Line2D([0], [0], color='green', lw=4, label='Leading'),
            plt.Line2D([0], [0], color='blue', lw=4, label='Concurrent'),
            plt.Line2D([0], [0], color='red', lw=4, label='Lagging')
        ])

        # Plot 3: Correlation at optimal lag vs concurrent
        ax = axes[1, 0]
        x = np.arange(len(lag_df_save))
        width = 0.35
        ax.bar(x - width/2, lag_df_save['concurrent_correlation'], width, label='Concurrent (lag=0)', alpha=0.8)
        ax.bar(x + width/2, lag_df_save['optimal_correlation'], width, label='Optimal lag', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lag_df_save['equipment_type'], rotation=45, ha='right')
        ax.set_ylabel('Correlation')
        ax.set_title('Concurrent vs Optimal Lag Correlation')
        ax.legend()

        # Plot 4: Multi-equipment cross-correlation overlay
        ax = axes[1, 1]
        for _, row in lag_df.head(5).iterrows():
            ax.plot(lags_plot, row['correlations_array'], label=row['equipment_type'], alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('Cross-correlation Curves by Equipment Type')
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'temporal_lag_analysis'))

        recommendations = [
            "Use equipment types with negative optimal lag as leading indicators",
            "Consider time-shifted features in model to capture predictive relationships",
            "Investigate causal mechanisms for leading vs lagging equipment types",
            f"Total equipment shows optimal lag of {total_row['optimal_lag']} days - consider this in prediction horizon"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'max_lag': max_lag,
                'n_equipment_types': len(equip_cols),
                'date_range': f"{merged['date'].min().date()} to {merged['date'].max().date()}"
            }
        )

        return self.result


# =============================================================================
# SECTION 1.2: VIIRS DOMINANCE INVESTIGATION
# =============================================================================

class VIIRSCasualtyTemporalProbe(Probe):
    """
    Probe 1.2.1: VIIRS-Casualty Temporal Relationship via cross-correlation.
    """

    @property
    def test_id(self) -> str:
        return "1.2.1"

    @property
    def test_name(self) -> str:
        return "VIIRS-Casualty Temporal Relationship"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting VIIRS-Casualty temporal analysis...")

        # Load data
        try:
            viirs_df = load_viirs_raw()
        except FileNotFoundError as e:
            self.log(f"Warning: {e}")
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'key_result': 'VIIRS data not available', 'significance': 'N/A', 'interpretation': str(e)}],
                recommendations=['Ensure VIIRS data is downloaded and available']
            )

        personnel_df = load_personnel_raw()

        # Merge
        merged = viirs_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        findings = []
        figures = []
        tables = []

        # Analyze each VIIRS feature
        viirs_features = ['radiance_mean', 'radiance_std', 'pct_clear_sky', 'radiance_p50', 'radiance_p90']
        viirs_features = [f for f in viirs_features if f in merged.columns]

        lag_results = []
        max_lag = 30

        for feature in viirs_features:
            viirs_vals = merged[feature].values
            personnel_delta = merged['personnel_delta'].values

            # Also compute first difference of VIIRS for trend-independent analysis
            viirs_delta = np.diff(viirs_vals, prepend=viirs_vals[0])

            # Cross-correlation with raw VIIRS
            lags, correlations = cross_correlation_at_lags(viirs_vals, personnel_delta, max_lag=max_lag)

            # Cross-correlation with VIIRS delta
            lags_delta, correlations_delta = cross_correlation_at_lags(viirs_delta, personnel_delta, max_lag=max_lag)

            if len(lags) > 0:
                best_idx = np.argmax(np.abs(correlations))
                best_idx_delta = np.argmax(np.abs(correlations_delta))

                lag_results.append({
                    'viirs_feature': feature,
                    'optimal_lag_raw': lags[best_idx],
                    'optimal_corr_raw': correlations[best_idx],
                    'optimal_lag_delta': lags_delta[best_idx_delta],
                    'optimal_corr_delta': correlations_delta[best_idx_delta],
                    'concurrent_corr_raw': correlations[len(correlations)//2],
                    'concurrent_corr_delta': correlations_delta[len(correlations_delta)//2],
                    'correlations_raw': correlations.tolist(),
                    'correlations_delta': correlations_delta.tolist()
                })

        lag_df = pd.DataFrame(lag_results)
        lag_df_save = lag_df.drop(columns=['correlations_raw', 'correlations_delta'])
        tables.append(self.save_table(lag_df_save, 'viirs_temporal_analysis'))

        # Classify temporal relationship
        radiance_row = lag_df[lag_df['viirs_feature'] == 'radiance_mean'].iloc[0] if 'radiance_mean' in lag_df['viirs_feature'].values else lag_df.iloc[0]

        if radiance_row['optimal_lag_raw'] < -3:
            relationship = "LEADING (VIIRS precedes casualties)"
        elif radiance_row['optimal_lag_raw'] > 3:
            relationship = "LAGGING (VIIRS follows casualties)"
        else:
            relationship = "CONCURRENT (approximately synchronized)"

        findings.append({
            'key_result': f"VIIRS-Casualty temporal relationship: {relationship}",
            'significance': f"Optimal lag for radiance_mean: {radiance_row['optimal_lag_raw']} days (r={radiance_row['optimal_corr_raw']:.4f})",
            'interpretation': "Leading indicator suggests VIIRS may have predictive value; lagging suggests confounding"
        })

        # Compare raw vs delta correlations
        mean_corr_raw = lag_df['concurrent_corr_raw'].abs().mean()
        mean_corr_delta = lag_df['concurrent_corr_delta'].abs().mean()

        findings.append({
            'key_result': f"Mean concurrent correlation - Raw: {mean_corr_raw:.4f}, Delta: {mean_corr_delta:.4f}",
            'significance': f"Correlation {'increased' if mean_corr_delta > mean_corr_raw else 'decreased'} by {abs(mean_corr_delta - mean_corr_raw):.4f} after differencing",
            'interpretation': "Lower delta correlation suggests trend confounding; higher suggests robust signal"
        })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cross-correlation for radiance_mean
        ax = axes[0, 0]
        if len(lag_df) > 0 and 'radiance_mean' in lag_df['viirs_feature'].values:
            row = lag_df[lag_df['viirs_feature'] == 'radiance_mean'].iloc[0]
            lags_plot = np.arange(-max_lag, max_lag + 1)
            ax.plot(lags_plot, row['correlations_raw'], 'b-o', markersize=3, label='Raw VIIRS')
            ax.plot(lags_plot, row['correlations_delta'], 'r-s', markersize=3, label='VIIRS Delta')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
            ax.fill_between(lags_plot, 0, row['correlations_raw'], alpha=0.2, color='blue')
            ax.set_xlabel('Lag (days)')
            ax.set_ylabel('Cross-correlation')
            ax.set_title('VIIRS Radiance Mean vs Personnel: Cross-correlation')
            ax.legend()

        # Plot 2: Optimal lag by VIIRS feature
        ax = axes[0, 1]
        x = np.arange(len(lag_df_save))
        width = 0.35
        ax.barh(lag_df_save['viirs_feature'], lag_df_save['optimal_lag_raw'], height=0.4, label='Raw', alpha=0.8)
        ax.barh([f + 0.4 for f in range(len(lag_df_save))], lag_df_save['optimal_lag_delta'], height=0.4, label='Delta', alpha=0.8)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('Optimal Lag (days)')
        ax.set_title('Optimal Lag by VIIRS Feature')
        ax.legend()

        # Plot 3: Raw vs Delta correlation comparison
        ax = axes[1, 0]
        ax.bar(x - width/2, lag_df_save['concurrent_corr_raw'].abs(), width, label='Raw', alpha=0.8)
        ax.bar(x + width/2, lag_df_save['concurrent_corr_delta'].abs(), width, label='Delta', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lag_df_save['viirs_feature'], rotation=45, ha='right')
        ax.set_ylabel('|Correlation|')
        ax.set_title('Concurrent Correlation: Raw vs Delta VIIRS')
        ax.legend()

        # Plot 4: Time series visualization
        ax = axes[1, 1]
        ax2 = ax.twinx()
        if 'radiance_mean' in merged.columns:
            ax.plot(merged['date'], merged['radiance_mean'], 'b-', alpha=0.7, label='VIIRS Radiance')
            ax.set_ylabel('VIIRS Radiance Mean', color='blue')
        ax2.plot(merged['date'], merged['personnel_delta'].rolling(7).mean(), 'r-', alpha=0.7, label='Personnel Delta (7d MA)')
        ax2.set_ylabel('Personnel Daily Losses', color='red')
        ax.set_xlabel('Date')
        ax.set_title('VIIRS Radiance vs Personnel Losses Over Time')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'viirs_temporal_relationship'))

        recommendations = [
            f"VIIRS shows {relationship.lower()} - adjust model accordingly",
            "Consider both raw and differenced VIIRS features in model",
            "Investigate physical mechanism linking nightlights to conflict intensity",
            "Test VIIRS importance with and without trend removal"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_viirs_features': len(viirs_features),
                'n_observations': len(merged),
                'date_range': f"{merged['date'].min().date()} to {merged['date'].max().date()}"
            }
        )

        return self.result


class VIIRSFeatureDecompositionProbe(Probe):
    """
    Probe 1.2.2: VIIRS Feature Decomposition - gradient magnitude per feature.

    Note: This probe requires model checkpoints for gradient analysis.
    Falls back to correlation-based importance if model not available.
    """

    @property
    def test_id(self) -> str:
        return "1.2.2"

    @property
    def test_name(self) -> str:
        return "VIIRS Feature Decomposition"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting VIIRS feature decomposition...")

        try:
            viirs_df = load_viirs_raw()
        except FileNotFoundError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'key_result': 'VIIRS data not available', 'significance': 'N/A', 'interpretation': str(e)}],
                recommendations=['Ensure VIIRS data is available']
            )

        personnel_df = load_personnel_raw()
        merged = viirs_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        findings = []
        figures = []
        tables = []

        # Feature importance via correlation and mutual information
        viirs_features = [c for c in viirs_df.columns if c != 'date']
        importance_results = []

        for feature in viirs_features:
            if feature not in merged.columns:
                continue

            viirs_vals = merged[feature].values
            personnel_delta = merged['personnel_delta'].values

            # Correlation-based importance
            r, p = pearsonr(viirs_vals, personnel_delta)
            rho, p_spearman = spearmanr(viirs_vals, personnel_delta)

            # Mutual information
            mi = compute_mutual_information(viirs_vals, personnel_delta)

            # Variance of feature
            feature_var = np.var(viirs_vals)
            feature_std = np.std(viirs_vals)

            importance_results.append({
                'feature': feature,
                'pearson_r': r,
                'pearson_p': p,
                'spearman_rho': rho,
                'mutual_info': mi,
                'feature_variance': feature_var,
                'feature_std': feature_std,
                'importance_score': abs(r) * mi if not np.isnan(mi) else abs(r)
            })

        importance_df = pd.DataFrame(importance_results)
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        tables.append(self.save_table(importance_df, 'viirs_feature_importance'))

        # Key findings
        top_feature = importance_df.iloc[0]
        findings.append({
            'key_result': f"Most important VIIRS feature: {top_feature['feature']}",
            'significance': f"r={top_feature['pearson_r']:.4f}, MI={top_feature['mutual_info']:.4f}",
            'interpretation': "This feature contributes most to casualty prediction signal"
        })

        # Feature redundancy analysis
        viirs_only = merged[[c for c in viirs_features if c in merged.columns]]
        corr_matrix = viirs_only.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.loc[col1, col2]) > 0.8:
                    high_corr_pairs.append({
                        'feature_1': col1,
                        'feature_2': col2,
                        'correlation': corr_matrix.loc[col1, col2]
                    })

        if high_corr_pairs:
            findings.append({
                'key_result': f"Found {len(high_corr_pairs)} highly correlated VIIRS feature pairs (|r| > 0.8)",
                'significance': str(high_corr_pairs[:3]),
                'interpretation': "High redundancy among VIIRS features - consider dimensionality reduction"
            })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Feature importance bar chart
        ax = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        ax.barh(importance_df['feature'], importance_df['importance_score'], color=colors)
        ax.set_xlabel('Importance Score (|r| * MI)')
        ax.set_title('VIIRS Feature Importance for Casualty Prediction')

        # Plot 2: Correlation matrix
        ax = axes[0, 1]
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('VIIRS Feature Correlation Matrix')

        # Plot 3: Pearson vs Spearman correlation
        ax = axes[1, 0]
        ax.scatter(importance_df['pearson_r'], importance_df['spearman_rho'], s=100)
        for idx, row in importance_df.iterrows():
            ax.annotate(row['feature'][:15], (row['pearson_r'], row['spearman_rho']), fontsize=8)
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
        ax.set_xlabel('Pearson r')
        ax.set_ylabel('Spearman rho')
        ax.set_title('Pearson vs Spearman: VIIRS Features')

        # Plot 4: Mutual information vs correlation
        ax = axes[1, 1]
        valid_mi = importance_df[~importance_df['mutual_info'].isna()]
        ax.scatter(valid_mi['pearson_r'].abs(), valid_mi['mutual_info'], s=100)
        for idx, row in valid_mi.iterrows():
            ax.annotate(row['feature'][:15], (abs(row['pearson_r']), row['mutual_info']), fontsize=8)
        ax.set_xlabel('|Pearson r|')
        ax.set_ylabel('Mutual Information (nats)')
        ax.set_title('Linear vs Non-linear Importance')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'viirs_feature_decomposition'))

        recommendations = [
            f"Prioritize {top_feature['feature']} in model feature selection",
            "Apply PCA to reduce VIIRS feature redundancy",
            "Investigate non-linear relationships for features with high MI but low r",
            "Consider creating composite VIIRS index from top features"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_features': len(viirs_features),
                'n_high_corr_pairs': len(high_corr_pairs)
            }
        )

        return self.result


class TrendConfoundingProbe(Probe):
    """
    Probe 1.2.3: Trend Confounding Test - compare VIIRS importance before/after detrending.
    """

    @property
    def test_id(self) -> str:
        return "1.2.3"

    @property
    def test_name(self) -> str:
        return "Trend Confounding Test"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting trend confounding analysis...")

        try:
            viirs_df = load_viirs_raw()
        except FileNotFoundError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'key_result': 'VIIRS data not available', 'significance': 'N/A', 'interpretation': str(e)}],
                recommendations=['Ensure VIIRS data is available']
            )

        personnel_df = load_personnel_raw()
        merged = viirs_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        findings = []
        figures = []
        tables = []

        # Detrending methods
        viirs_features = ['radiance_mean', 'radiance_std', 'pct_clear_sky']
        viirs_features = [f for f in viirs_features if f in merged.columns]

        detrend_results = []

        for feature in viirs_features:
            raw_vals = merged[feature].values
            personnel_delta = merged['personnel_delta'].values
            time_index = np.arange(len(merged))

            # Method 1: First differencing
            diff_vals = np.diff(raw_vals, prepend=raw_vals[0])
            diff_personnel = np.diff(personnel_delta, prepend=personnel_delta[0])

            # Method 2: Linear detrending
            slope, intercept, _, _, _ = stats.linregress(time_index, raw_vals)
            linear_trend = slope * time_index + intercept
            detrended_linear = raw_vals - linear_trend

            slope_p, intercept_p, _, _, _ = stats.linregress(time_index, personnel_delta)
            linear_trend_p = slope_p * time_index + intercept_p
            detrended_linear_p = personnel_delta - linear_trend_p

            # Correlations before and after detrending
            r_raw, p_raw = pearsonr(raw_vals, personnel_delta)
            r_diff, p_diff = pearsonr(diff_vals, diff_personnel)
            r_linear, p_linear = pearsonr(detrended_linear, detrended_linear_p)

            # Partial correlation controlling for time
            r_partial, p_partial = compute_partial_correlation(raw_vals, personnel_delta, time_index)

            detrend_results.append({
                'feature': feature,
                'r_raw': r_raw,
                'p_raw': p_raw,
                'r_first_diff': r_diff,
                'p_first_diff': p_diff,
                'r_linear_detrend': r_linear,
                'p_linear_detrend': p_linear,
                'r_partial_time': r_partial,
                'p_partial_time': p_partial,
                'correlation_reduction': 1 - abs(r_diff) / (abs(r_raw) + 1e-10)
            })

        detrend_df = pd.DataFrame(detrend_results)
        tables.append(self.save_table(detrend_df, 'trend_confounding_analysis'))

        # Key findings
        mean_reduction = detrend_df['correlation_reduction'].mean()

        if mean_reduction > 0.5:
            trend_assessment = "STRONG trend confounding detected"
        elif mean_reduction > 0.2:
            trend_assessment = "MODERATE trend confounding detected"
        else:
            trend_assessment = "MINIMAL trend confounding"

        findings.append({
            'key_result': trend_assessment,
            'significance': f"Mean correlation reduction after detrending: {mean_reduction:.1%}",
            'interpretation': "High reduction indicates VIIRS-casualty correlation is largely spurious (both driven by time)"
        })

        # Feature-specific findings
        for _, row in detrend_df.iterrows():
            findings.append({
                'key_result': f"{row['feature']}: raw r={row['r_raw']:.4f}, detrended r={row['r_first_diff']:.4f}",
                'significance': f"Correlation reduction: {row['correlation_reduction']:.1%}",
                'interpretation': "Large reduction = trend-driven; small reduction = robust signal"
            })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Raw vs detrended correlations
        ax = axes[0, 0]
        x = np.arange(len(detrend_df))
        width = 0.2
        ax.bar(x - 1.5*width, detrend_df['r_raw'], width, label='Raw', alpha=0.8)
        ax.bar(x - 0.5*width, detrend_df['r_first_diff'], width, label='First Diff', alpha=0.8)
        ax.bar(x + 0.5*width, detrend_df['r_linear_detrend'], width, label='Linear Detrend', alpha=0.8)
        ax.bar(x + 1.5*width, detrend_df['r_partial_time'], width, label='Partial (time)', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(detrend_df['feature'], rotation=45, ha='right')
        ax.set_ylabel('Correlation')
        ax.set_title('Correlation Before and After Detrending')
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Plot 2: Correlation reduction
        ax = axes[0, 1]
        colors = ['red' if r > 0.5 else ('orange' if r > 0.2 else 'green') for r in detrend_df['correlation_reduction']]
        ax.barh(detrend_df['feature'], detrend_df['correlation_reduction'], color=colors)
        ax.axvline(x=0.5, color='r', linestyle='--', label='Strong confounding')
        ax.axvline(x=0.2, color='orange', linestyle='--', label='Moderate confounding')
        ax.set_xlabel('Correlation Reduction')
        ax.set_title('Trend Confounding Assessment')
        ax.legend()

        # Plot 3: Example time series (radiance_mean)
        ax = axes[1, 0]
        if 'radiance_mean' in merged.columns:
            feature = 'radiance_mean'
            raw_vals = merged[feature].values
            time_index = np.arange(len(merged))
            slope, intercept, _, _, _ = stats.linregress(time_index, raw_vals)
            linear_trend = slope * time_index + intercept

            ax.plot(merged['date'], raw_vals, 'b-', alpha=0.5, label='Raw')
            ax.plot(merged['date'], linear_trend, 'r--', linewidth=2, label='Linear Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Radiance Mean')
            ax.set_title('VIIRS Radiance Mean with Linear Trend')
            ax.legend()

        # Plot 4: Detrended time series
        ax = axes[1, 1]
        if 'radiance_mean' in merged.columns:
            detrended = raw_vals - linear_trend
            ax.plot(merged['date'], detrended, 'g-', alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Detrended Radiance')
            ax.set_title('VIIRS Radiance Mean (Detrended)')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'trend_confounding'))

        recommendations = [
            "Use first-differenced VIIRS features to remove trend confounding",
            "Consider seasonal decomposition for more sophisticated detrending",
            "Test model performance with and without detrended features",
            "If confounding is strong, VIIRS dominance may be an artifact - verify with ablation"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'mean_correlation_reduction': mean_reduction,
                'trend_assessment': trend_assessment
            }
        )

        return self.result


class GeographicVIIRSDecompositionProbe(Probe):
    """
    Probe 1.2.4: Geographic VIIRS Decomposition - check if regional breakdown exists.
    """

    @property
    def test_id(self) -> str:
        return "1.2.4"

    @property
    def test_name(self) -> str:
        return "Geographic VIIRS Decomposition"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting geographic VIIRS decomposition...")

        viirs_path = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"

        if not viirs_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'key_result': 'VIIRS data not available', 'significance': 'N/A', 'interpretation': 'File not found'}],
                recommendations=['Ensure VIIRS data is available']
            )

        # Load raw data with tile information
        df = pd.read_csv(viirs_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        findings = []
        figures = []
        tables = []

        # Check if tile information exists
        if 'tile' not in df.columns:
            findings.append({
                'key_result': 'No geographic tile information in VIIRS data',
                'significance': 'Cannot perform regional decomposition',
                'interpretation': 'Data is already aggregated at country level'
            })
        else:
            # Analyze by tile
            tiles = df['tile'].unique()
            tile_stats = []

            for tile in tiles:
                tile_data = df[df['tile'] == tile]

                tile_stats.append({
                    'tile': tile,
                    'n_observations': len(tile_data),
                    'date_range': f"{tile_data['date'].min().date()} to {tile_data['date'].max().date()}",
                    'mean_radiance': tile_data['radiance_mean'].mean() if 'radiance_mean' in tile_data.columns else np.nan,
                    'std_radiance': tile_data['radiance_mean'].std() if 'radiance_mean' in tile_data.columns else np.nan,
                    'pct_clear_sky': tile_data['pct_clear_sky'].mean() if 'pct_clear_sky' in tile_data.columns else np.nan
                })

            tile_df = pd.DataFrame(tile_stats)
            tables.append(self.save_table(tile_df, 'tile_statistics'))

            findings.append({
                'key_result': f"Found {len(tiles)} geographic tiles: {list(tiles)}",
                'significance': 'Regional breakdown is available',
                'interpretation': 'Can analyze regional variation in nightlight patterns'
            })

            # Compare tiles
            if len(tile_df) > 1:
                radiance_range = tile_df['mean_radiance'].max() - tile_df['mean_radiance'].min()
                findings.append({
                    'key_result': f"Radiance variation across tiles: {radiance_range:.2f}",
                    'significance': f"Range from {tile_df['mean_radiance'].min():.2f} to {tile_df['mean_radiance'].max():.2f}",
                    'interpretation': 'Large variation suggests regional differences in conflict intensity or urbanization'
                })

        # Load personnel data for regional correlation (if available)
        personnel_df = load_personnel_raw()

        if 'tile' in df.columns:
            # Compute per-tile correlation with casualties
            tile_corr_results = []

            for tile in tiles:
                tile_data = df[df['tile'] == tile].copy()
                tile_agg = tile_data.groupby('date')['radiance_mean'].mean().reset_index()

                merged = tile_agg.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
                merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

                if len(merged) > 10:
                    r, p = pearsonr(merged['radiance_mean'], merged['personnel_delta'])
                    tile_corr_results.append({
                        'tile': tile,
                        'correlation': r,
                        'p_value': p,
                        'n_obs': len(merged)
                    })

            if tile_corr_results:
                tile_corr_df = pd.DataFrame(tile_corr_results)
                tables.append(self.save_table(tile_corr_df, 'tile_casualty_correlations'))

                best_tile = tile_corr_df.loc[tile_corr_df['correlation'].abs().idxmax()]
                findings.append({
                    'key_result': f"Tile with highest casualty correlation: {best_tile['tile']} (r={best_tile['correlation']:.4f})",
                    'significance': f"p-value: {best_tile['p_value']:.2e}",
                    'interpretation': 'This region may be most relevant for casualty prediction'
                })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if 'tile' in df.columns and len(tiles) > 1:
            # Plot 1: Radiance by tile
            ax = axes[0, 0]
            tile_df_sorted = tile_df.sort_values('mean_radiance', ascending=False)
            ax.barh(tile_df_sorted['tile'], tile_df_sorted['mean_radiance'])
            ax.set_xlabel('Mean Radiance')
            ax.set_title('Mean Radiance by Geographic Tile')

            # Plot 2: Time series by tile
            ax = axes[0, 1]
            for tile in tiles[:4]:  # Top 4 tiles
                tile_data = df[df['tile'] == tile]
                daily_mean = tile_data.groupby('date')['radiance_mean'].mean()
                ax.plot(daily_mean.index, daily_mean.values, label=tile, alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Radiance Mean')
            ax.set_title('Radiance Time Series by Tile')
            ax.legend()

            # Plot 3: Correlation by tile (if available)
            if tile_corr_results:
                ax = axes[1, 0]
                ax.barh(tile_corr_df['tile'], tile_corr_df['correlation'])
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Correlation with Casualties')
                ax.set_title('Casualty Correlation by Tile')

            # Plot 4: Clear sky percentage by tile
            ax = axes[1, 1]
            ax.barh(tile_df['tile'], tile_df['pct_clear_sky'])
            ax.set_xlabel('Mean Clear Sky %')
            ax.set_title('Data Quality (Clear Sky) by Tile')
        else:
            # No tile data - show aggregate analysis
            ax = axes[0, 0]
            ax.text(0.5, 0.5, 'No geographic tile breakdown available\nData is aggregated at country level',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Geographic Decomposition')

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'geographic_decomposition'))

        recommendations = [
            "Consider using tile-specific features for regional conflict modeling",
            "Weight tiles by their casualty correlation in ensemble",
            "Investigate why certain tiles show stronger correlation with casualties",
            "Create regional VIIRS composite index from most predictive tiles"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'n_tiles': len(tiles) if 'tile' in df.columns else 0
            }
        )

        return self.result


# =============================================================================
# SECTION 1.3: PERSONNEL DATA QUALITY CHECK
# =============================================================================

class PersonnelVIIRSMediationProbe(Probe):
    """
    Probe 1.3.1: Personnel-VIIRS Mediation Analysis.

    Tests mediation model: VIIRS -> Personnel -> Casualties vs VIIRS -> Casualties
    """

    @property
    def test_id(self) -> str:
        return "1.3.1"

    @property
    def test_name(self) -> str:
        return "Personnel-VIIRS Mediation Analysis"

    def run(self, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        self.log("Starting Personnel-VIIRS mediation analysis...")

        try:
            viirs_df = load_viirs_raw()
        except FileNotFoundError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'key_result': 'VIIRS data not available', 'significance': 'N/A', 'interpretation': str(e)}],
                recommendations=['Ensure VIIRS data is available']
            )

        personnel_df = load_personnel_raw()

        # Merge all data
        merged = viirs_df.merge(personnel_df[['date', 'personnel']], on='date', how='inner')
        merged['personnel_delta'] = merged['personnel'].diff().fillna(0)

        # Create lagged personnel for outcome
        merged['personnel_delta_lead1'] = merged['personnel_delta'].shift(-1)
        merged = merged.dropna()

        findings = []
        figures = []
        tables = []

        # Simple mediation analysis using correlations
        # Path a: VIIRS -> Personnel (mediator)
        # Path b: Personnel -> Casualties (outcome)
        # Path c: VIIRS -> Casualties (total effect)
        # Path c': VIIRS -> Casualties controlling for Personnel (direct effect)

        viirs_feature = 'radiance_mean' if 'radiance_mean' in merged.columns else merged.columns[1]

        X = merged[viirs_feature].values  # Independent: VIIRS
        M = merged['personnel_delta'].values  # Mediator: Personnel delta
        Y = merged['personnel_delta_lead1'].values  # Outcome: Next-day casualties

        # Standardize for comparison
        X = (X - X.mean()) / (X.std() + 1e-10)
        M = (M - M.mean()) / (M.std() + 1e-10)
        Y = (Y - Y.mean()) / (Y.std() + 1e-10)

        # Path coefficients via OLS
        if HAS_STATSMODELS:
            # Path a: X -> M
            X_const = add_constant(X)
            model_a = OLS(M, X_const).fit()
            a = model_a.params[1]
            a_se = model_a.bse[1]
            a_p = model_a.pvalues[1]

            # Path c: X -> Y (total effect)
            model_c = OLS(Y, X_const).fit()
            c = model_c.params[1]
            c_se = model_c.bse[1]
            c_p = model_c.pvalues[1]

            # Path b and c': X + M -> Y
            XM = np.column_stack([X, M])
            XM_const = add_constant(XM)
            model_bc = OLS(Y, XM_const).fit()
            c_prime = model_bc.params[1]  # Direct effect
            b = model_bc.params[2]  # M -> Y
            c_prime_se = model_bc.bse[1]
            c_prime_p = model_bc.pvalues[1]
            b_se = model_bc.bse[2]
            b_p = model_bc.pvalues[2]

            # Indirect effect (mediation)
            indirect = a * b

            # Sobel test for indirect effect significance
            sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
            sobel_z = indirect / sobel_se
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

            # Proportion mediated
            if abs(c) > 1e-10:
                proportion_mediated = indirect / c
            else:
                proportion_mediated = 0

            mediation_results = {
                'path_a_VIIRS_to_Personnel': a,
                'path_a_p': a_p,
                'path_b_Personnel_to_Casualties': b,
                'path_b_p': b_p,
                'path_c_total_effect': c,
                'path_c_p': c_p,
                'path_c_prime_direct_effect': c_prime,
                'path_c_prime_p': c_prime_p,
                'indirect_effect': indirect,
                'sobel_z': sobel_z,
                'sobel_p': sobel_p,
                'proportion_mediated': proportion_mediated
            }

            mediation_df = pd.DataFrame([mediation_results])
            tables.append(self.save_table(mediation_df, 'mediation_analysis'))

            findings.append({
                'key_result': f"Total effect (VIIRS -> Casualties): {c:.4f} (p={c_p:.2e})",
                'significance': f"Direct effect: {c_prime:.4f}, Indirect (via Personnel): {indirect:.4f}",
                'interpretation': f"Personnel mediates {proportion_mediated*100:.1f}% of VIIRS effect on casualties"
            })

            if sobel_p < 0.05:
                findings.append({
                    'key_result': f"Significant mediation detected (Sobel z={sobel_z:.2f}, p={sobel_p:.2e})",
                    'significance': "Personnel significantly mediates VIIRS-Casualty relationship",
                    'interpretation': "VIIRS affects casualties partly through personnel losses"
                })
            else:
                findings.append({
                    'key_result': f"No significant mediation (Sobel z={sobel_z:.2f}, p={sobel_p:.2e})",
                    'significance': "Personnel does not significantly mediate VIIRS-Casualty relationship",
                    'interpretation': "VIIRS and personnel may have independent effects on casualties"
                })

        else:
            # Fallback: correlation-based approximation
            r_xm, _ = pearsonr(X, M)
            r_my, _ = pearsonr(M, Y)
            r_xy, _ = pearsonr(X, Y)

            # Partial correlation X-Y controlling for M
            r_xy_m, _ = compute_partial_correlation(X, Y, M)

            indirect_approx = r_xm * r_my
            proportion_mediated_approx = indirect_approx / r_xy if abs(r_xy) > 1e-10 else 0

            findings.append({
                'key_result': f"Correlation-based mediation approximation",
                'significance': f"r(VIIRS,Personnel)={r_xm:.4f}, r(Personnel,Casualties)={r_my:.4f}",
                'interpretation': f"Approximate proportion mediated: {proportion_mediated_approx*100:.1f}%"
            })

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Path diagram (conceptual)
        ax = axes[0, 0]
        ax.text(0.1, 0.5, 'VIIRS\n(X)', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax.text(0.5, 0.8, 'Personnel\n(M)', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax.text(0.9, 0.5, 'Casualties\n(Y)', ha='center', va='center', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightyellow'))

        if HAS_STATSMODELS:
            # Arrows with coefficients
            ax.annotate('', xy=(0.4, 0.75), xytext=(0.2, 0.55),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            ax.text(0.25, 0.68, f'a={a:.3f}', fontsize=10, color='blue')

            ax.annotate('', xy=(0.8, 0.55), xytext=(0.6, 0.75),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.text(0.7, 0.68, f'b={b:.3f}', fontsize=10, color='green')

            ax.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax.text(0.5, 0.4, f"c'={c_prime:.3f}", fontsize=10, color='red')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Mediation Path Diagram')

        # Plot 2: Scatter X vs M
        ax = axes[0, 1]
        ax.scatter(X, M, alpha=0.3)
        ax.set_xlabel('VIIRS (standardized)')
        ax.set_ylabel('Personnel Delta (standardized)')
        ax.set_title(f'Path a: VIIRS -> Personnel (r={r_xm:.3f})' if not HAS_STATSMODELS else f'Path a: VIIRS -> Personnel (a={a:.3f})')

        # Add regression line
        z = np.polyfit(X, M, 1)
        p = np.poly1d(z)
        ax.plot(sorted(X), p(sorted(X)), 'r--', linewidth=2)

        # Plot 3: Scatter M vs Y
        ax = axes[1, 0]
        ax.scatter(M, Y, alpha=0.3)
        ax.set_xlabel('Personnel Delta (standardized)')
        ax.set_ylabel('Casualties Lead1 (standardized)')
        ax.set_title(f'Path b: Personnel -> Casualties')

        z = np.polyfit(M, Y, 1)
        p = np.poly1d(z)
        ax.plot(sorted(M), p(sorted(M)), 'r--', linewidth=2)

        # Plot 4: Effect decomposition
        ax = axes[1, 1]
        if HAS_STATSMODELS:
            effects = ['Total (c)', 'Direct (c\')', 'Indirect (ab)']
            values = [c, c_prime, indirect]
            colors = ['blue', 'red', 'green']
            ax.bar(effects, values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Effect Size')
            ax.set_title('Effect Decomposition')

            # Add significance markers
            for i, (eff, val) in enumerate(zip(effects, values)):
                p_val = [c_p, c_prime_p, sobel_p][i]
                marker = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
                ax.text(i, val + 0.01, marker, ha='center', fontsize=14)

        plt.tight_layout()
        figures.append(self.save_figure(fig, 'mediation_analysis'))

        recommendations = [
            "Consider both VIIRS and Personnel as separate predictors (complementary information)",
            "If mediation is strong, Personnel may be redundant with VIIRS",
            "Test reverse mediation (Personnel -> VIIRS -> Casualties) for comparison",
            "Use causal inference methods for stronger claims about mediation"
        ]

        self.result = ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts={'figures': figures, 'tables': tables},
            recommendations=recommendations,
            metadata={
                'viirs_feature': viirs_feature,
                'n_observations': len(merged),
                'statsmodels_available': HAS_STATSMODELS
            }
        )

        return self.result


# =============================================================================
# PROBE SUITE RUNNER
# =============================================================================

class DataArtifactProbeSuite:
    """
    Orchestrates execution of all data artifact probes.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.probes = {
            '1.1.1': EncodingVarianceComparisonProbe(verbose=verbose),
            '1.1.2': EquipmentPersonnelRedundancyProbe(verbose=verbose),
            '1.1.3': EquipmentCategoryDisaggregationProbe(verbose=verbose),
            '1.1.4': TemporalLagAnalysisProbe(verbose=verbose),
            '1.2.1': VIIRSCasualtyTemporalProbe(verbose=verbose),
            '1.2.2': VIIRSFeatureDecompositionProbe(verbose=verbose),
            '1.2.3': TrendConfoundingProbe(verbose=verbose),
            '1.2.4': GeographicVIIRSDecompositionProbe(verbose=verbose),
            '1.3.1': PersonnelVIIRSMediationProbe(verbose=verbose),
        }
        self.results: Dict[str, ProbeResult] = {}

    def run_probe(self, probe_id: str, data: Optional[Dict[str, Any]] = None) -> ProbeResult:
        """Run a single probe by ID."""
        if probe_id not in self.probes:
            raise ValueError(f"Unknown probe ID: {probe_id}. Available: {list(self.probes.keys())}")

        probe = self.probes[probe_id]
        result = probe.run(data)
        self.results[probe_id] = result
        result.save()

        return result

    def run_all(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, ProbeResult]:
        """Run all probes in sequence."""
        print("=" * 80)
        print("DATA ARTIFACT PROBE SUITE")
        print("=" * 80)

        for probe_id in sorted(self.probes.keys()):
            print(f"\n{'='*40}")
            print(f"Running Probe {probe_id}: {self.probes[probe_id].test_name}")
            print(f"{'='*40}")

            try:
                result = self.run_probe(probe_id, data)
                print(f"Completed. Findings: {len(result.findings)}")
            except Exception as e:
                print(f"ERROR: {e}")
                self.results[probe_id] = ProbeResult(
                    test_id=probe_id,
                    test_name=self.probes[probe_id].test_name,
                    findings=[{'key_result': 'FAILED', 'significance': str(e), 'interpretation': 'Probe execution failed'}],
                    recommendations=['Debug and fix the probe implementation']
                )

        # Save summary
        self._save_summary()

        return self.results

    def run_section(self, section: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, ProbeResult]:
        """Run all probes in a section (e.g., '1.1' for equipment probes)."""
        section_probes = {k: v for k, v in self.probes.items() if k.startswith(section)}

        for probe_id in sorted(section_probes.keys()):
            try:
                self.run_probe(probe_id, data)
            except Exception as e:
                print(f"ERROR in {probe_id}: {e}")

        return {k: v for k, v in self.results.items() if k.startswith(section)}

    def _save_summary(self) -> Path:
        """Save a summary of all probe results."""
        summary = {
            'execution_time': datetime.now().isoformat(),
            'probes_executed': len(self.results),
            'results': {}
        }

        for probe_id, result in self.results.items():
            summary['results'][probe_id] = {
                'test_name': result.test_name,
                'n_findings': len(result.findings),
                'key_findings': [f['key_result'] for f in result.findings[:2]],
                'n_recommendations': len(result.recommendations)
            }

        ext = '.yaml' if HAS_YAML else '.json'
        summary_path = get_probe_metrics_dir() / f'probe_suite_summary{ext}'
        with open(summary_path, 'w') as f:
            if HAS_YAML:
                yaml.dump(summary, f, default_flow_style=False)
            else:
                json.dump(summary, f, indent=2, default=str)

        return summary_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DATA ARTIFACT PROBES FOR MULTI-RESOLUTION HAN MODEL")
    print("=" * 80)

    # Create and run the probe suite
    suite = DataArtifactProbeSuite(verbose=True)

    # Run all probes
    results = suite.run_all()

    # Print summary
    print("\n" + "=" * 80)
    print("PROBE SUITE SUMMARY")
    print("=" * 80)

    for probe_id, result in sorted(results.items()):
        print(f"\n[{probe_id}] {result.test_name}")
        print("-" * 40)
        for finding in result.findings[:2]:
            print(f"  - {finding['key_result']}")

    print(f"\nAll results saved to: {get_output_dir()}")
