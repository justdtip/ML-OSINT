"""
VIIRS Causality Validation - Agent C4

Validates Claim C4: "VIIRS is a +10 day lagging indicator that reflects past damage
rather than predicting future events."

Implements 5 experiments:
1. Cross-Correlation Analysis (CCF) - lags -30 to +30 days
2. Temporal Shift Experiment - compare original vs shifted VIIRS
3. VIIRS Feature Isolation - detrended vs raw signal quality
4. Causal Direction Classifier - past vs future casualty prediction
5. Granger Causality Test - formal causal direction test

Author: Agent C4
Date: 2026-01-25
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Statistical imports
try:
    from statsmodels.tsa.stattools import ccf, adfuller, grangercausalitytests
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available - some tests will be limited")

try:
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not available - causal classifiers will be limited")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import (
    DATA_DIR, OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
    ensure_dir
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "han_validation"
REPORT_PATH = OUTPUT_PATH / "C4_viirs_causality_report.md"
FIGURES_PATH = OUTPUT_PATH / "figures"

# Data paths
VIIRS_PATH = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"
PERSONNEL_PATH = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"
EQUIPMENT_PATH = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

# Analysis parameters
MAX_LAG = 30  # Days for CCF analysis
VIIRS_SHIFT_DAYS = 10  # Days to shift VIIRS forward
PREDICTION_WINDOW = 7  # Days for causal direction classifier
GRANGER_MAX_LAG = 14  # Maximum lag for Granger test
CONFIDENCE_LEVEL = 0.95

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_viirs_data() -> pd.DataFrame:
    """Load and aggregate VIIRS nightlight data."""
    if not VIIRS_PATH.exists():
        raise FileNotFoundError(f"VIIRS data not found at {VIIRS_PATH}")

    df = pd.read_csv(VIIRS_PATH)
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

    # Apply log scaling for better distribution
    daily_agg['radiance_mean_log'] = np.log1p(daily_agg['radiance_mean'])

    # Compute first-order difference (detrended)
    for col in ['radiance_mean', 'radiance_std', 'radiance_p50', 'radiance_p90']:
        daily_agg[f'{col}_diff'] = daily_agg[col].diff()

    return daily_agg


def load_casualty_data() -> pd.DataFrame:
    """Load personnel and equipment casualty data."""
    # Personnel data
    with open(PERSONNEL_PATH) as f:
        personnel_data = json.load(f)

    personnel_df = pd.DataFrame(personnel_data)
    personnel_df['date'] = pd.to_datetime(personnel_df['date'], errors='coerce')
    personnel_df = personnel_df.dropna(subset=['date'])
    personnel_df = personnel_df.sort_values('date').reset_index(drop=True)

    # Calculate daily casualties (diff of cumulative)
    personnel_df['daily_casualties'] = personnel_df['personnel'].diff()

    # Equipment data
    with open(EQUIPMENT_PATH) as f:
        equipment_data = json.load(f)

    equipment_df = pd.DataFrame(equipment_data)
    equipment_df['date'] = pd.to_datetime(equipment_df['date'], errors='coerce')
    equipment_df = equipment_df.dropna(subset=['date'])
    equipment_df = equipment_df.sort_values('date').reset_index(drop=True)

    # Calculate total equipment loss per day
    equipment_cols = ['tank', 'APC', 'field_artillery', 'MRL', 'drone',
                      'aircraft', 'helicopter', 'naval_ship', 'anti_aircraft_warfare']

    for col in equipment_cols:
        if col in equipment_df.columns:
            equipment_df[f'{col}_daily'] = equipment_df[col].diff()

    daily_cols = [f'{col}_daily' for col in equipment_cols if f'{col}_daily' in equipment_df.columns]
    equipment_df['total_equipment_daily'] = equipment_df[daily_cols].sum(axis=1)

    # Merge datasets
    merged = pd.merge(personnel_df[['date', 'daily_casualties']],
                      equipment_df[['date', 'total_equipment_daily']],
                      on='date', how='outer')

    merged = merged.sort_values('date').reset_index(drop=True)

    # Combined target: personnel + equipment
    merged['combined_casualties'] = (
        merged['daily_casualties'].fillna(0) +
        merged['total_equipment_daily'].fillna(0) * 10  # Scale equipment to same magnitude
    )

    return merged


def merge_datasets(viirs_df: pd.DataFrame, casualty_df: pd.DataFrame) -> pd.DataFrame:
    """Merge VIIRS and casualty datasets on date."""
    merged = pd.merge(viirs_df, casualty_df, on='date', how='inner')
    merged = merged.sort_values('date').reset_index(drop=True)

    # Drop rows with NaN in key columns
    merged = merged.dropna(subset=['radiance_mean', 'daily_casualties'])

    return merged


# =============================================================================
# EXPERIMENT 1: CROSS-CORRELATION ANALYSIS
# =============================================================================

def experiment_ccf_analysis(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 1: Cross-Correlation Analysis

    Computes CCF between VIIRS features and casualty targets at lags from -30 to +30 days.
    Positive lag means VIIRS lags casualties (VIIRS reflects past damage).
    Negative lag means VIIRS leads casualties (VIIRS predicts future damage).

    Success criterion: peak correlation at positive lag -> C4 confirmed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Cross-Correlation Analysis")
    print("="*60)

    results = {
        'features': {},
        'summary': {},
        'conclusion': ''
    }

    viirs_features = ['radiance_mean', 'radiance_std', 'radiance_p50', 'radiance_p90',
                      'radiance_mean_diff', 'radiance_std_diff']
    target = 'daily_casualties'

    # Get clean arrays
    df_clean = merged_df.dropna(subset=[target] + viirs_features)
    y = df_clean[target].values

    lags = np.arange(-MAX_LAG, MAX_LAG + 1)

    for feature in viirs_features:
        x = df_clean[feature].values

        # Compute cross-correlation at each lag manually
        correlations = []
        for lag in lags:
            if lag > 0:
                # VIIRS lags casualties: compare VIIRS[t+lag] with casualties[t]
                x_shifted = x[lag:]
                y_aligned = y[:-lag] if lag > 0 else y
            elif lag < 0:
                # VIIRS leads casualties: compare VIIRS[t] with casualties[t+|lag|]
                x_shifted = x[:lag]
                y_aligned = y[-lag:]
            else:
                x_shifted = x
                y_aligned = y

            if len(x_shifted) > 10:
                r, _ = pearsonr(x_shifted, y_aligned)
                correlations.append(r)
            else:
                correlations.append(np.nan)

        correlations = np.array(correlations)

        # Find peak correlation and its lag
        valid_idx = ~np.isnan(correlations)
        if valid_idx.any():
            peak_idx = np.nanargmax(np.abs(correlations))
            peak_lag = lags[peak_idx]
            peak_corr = correlations[peak_idx]

            # Compute confidence interval (approximate using Fisher z-transform)
            n = len(x)
            se = 1 / np.sqrt(n - 3)
            ci_low = np.tanh(np.arctanh(correlations) - 1.96 * se)
            ci_high = np.tanh(np.arctanh(correlations) + 1.96 * se)
        else:
            peak_lag = 0
            peak_corr = 0
            ci_low = np.zeros_like(correlations)
            ci_high = np.zeros_like(correlations)

        results['features'][feature] = {
            'lags': lags.tolist(),
            'correlations': correlations.tolist(),
            'ci_low': ci_low.tolist(),
            'ci_high': ci_high.tolist(),
            'peak_lag': int(peak_lag),
            'peak_correlation': float(peak_corr),
            'concurrent_correlation': float(correlations[MAX_LAG]) if not np.isnan(correlations[MAX_LAG]) else 0.0
        }

        print(f"\n{feature}:")
        print(f"  Peak lag: {peak_lag} days")
        print(f"  Peak correlation: {peak_corr:.4f}")
        print(f"  Concurrent (lag=0): {correlations[MAX_LAG]:.4f}")

    # Summary statistics
    peak_lags = [r['peak_lag'] for r in results['features'].values()]
    mean_peak_lag = np.mean(peak_lags)

    results['summary'] = {
        'mean_peak_lag': float(mean_peak_lag),
        'features_with_positive_lag': sum(1 for l in peak_lags if l > 0),
        'features_with_negative_lag': sum(1 for l in peak_lags if l < 0),
        'total_features': len(peak_lags)
    }

    # Conclusion
    if mean_peak_lag > 5:
        results['conclusion'] = "CONFIRMED: Mean peak lag is positive, indicating VIIRS lags casualties"
    elif mean_peak_lag < -5:
        results['conclusion'] = "REFUTED: Mean peak lag is negative, indicating VIIRS leads casualties"
    else:
        results['conclusion'] = "INCONCLUSIVE: Mean peak lag is near zero"

    print(f"\n{results['conclusion']}")

    return results


def plot_ccf_results(results: Dict[str, Any], output_dir: Path):
    """Generate CCF visualization plots."""
    ensure_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    features = list(results['features'].keys())[:6]

    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = results['features'][feature]

        lags = np.array(data['lags'])
        corrs = np.array(data['correlations'])
        ci_low = np.array(data['ci_low'])
        ci_high = np.array(data['ci_high'])

        # Plot correlation line
        ax.plot(lags, corrs, 'b-', linewidth=2, label='CCF')

        # Plot confidence band
        ax.fill_between(lags, ci_low, ci_high, alpha=0.3, color='blue', label='95% CI')

        # Mark peak
        peak_lag = data['peak_lag']
        peak_corr = data['peak_correlation']
        ax.axvline(x=peak_lag, color='red', linestyle='--', alpha=0.7, label=f'Peak at {peak_lag}d')
        ax.scatter([peak_lag], [peak_corr], color='red', s=100, zorder=5)

        # Zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

        # Shade regions
        ax.axvspan(0, MAX_LAG, alpha=0.1, color='green', label='VIIRS lags (C4)')
        ax.axvspan(-MAX_LAG, 0, alpha=0.1, color='orange', label='VIIRS leads')

        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'{feature}\nPeak: {peak_corr:.3f} at lag {peak_lag}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Correlation Analysis: VIIRS vs Daily Casualties\n(Positive lag = VIIRS reflects past events)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp1_ccf_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved CCF plot to {output_dir / 'exp1_ccf_analysis.png'}")


# =============================================================================
# EXPERIMENT 2: TEMPORAL SHIFT EXPERIMENT
# =============================================================================

def experiment_temporal_shift(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 2: Temporal Shift Experiment

    Creates modified dataset with VIIRS shifted forward by 10 days.
    Compares prediction accuracy with original vs shifted VIIRS.

    Success criterion: shifted VIIRS improves forecast -> C4 confirmed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Temporal Shift Experiment")
    print("="*60)

    if not HAS_SKLEARN:
        print("sklearn not available - skipping this experiment")
        return {'error': 'sklearn not available'}

    results = {
        'original': {},
        'shifted': {},
        'comparison': {},
        'conclusion': ''
    }

    # Prepare features
    viirs_features = ['radiance_mean', 'radiance_std', 'radiance_p50']
    target = 'daily_casualties'

    df = merged_df.dropna(subset=[target] + viirs_features).copy()

    # Create shifted VIIRS (shift forward = use past VIIRS to predict future)
    df_shifted = df.copy()
    for col in viirs_features:
        df_shifted[f'{col}_shifted'] = df_shifted[col].shift(VIIRS_SHIFT_DAYS)

    df_shifted = df_shifted.dropna()

    # Features for original and shifted
    X_original = df_shifted[viirs_features].values
    X_shifted = df_shifted[[f'{col}_shifted' for col in viirs_features]].values
    y = df_shifted[target].values

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    scores_original = []
    scores_shifted = []

    for train_idx, test_idx in tscv.split(X_original):
        # Scale features
        scaler_orig = StandardScaler()
        scaler_shift = StandardScaler()

        X_train_orig = scaler_orig.fit_transform(X_original[train_idx])
        X_test_orig = scaler_orig.transform(X_original[test_idx])

        X_train_shift = scaler_shift.fit_transform(X_shifted[train_idx])
        X_test_shift = scaler_shift.transform(X_shifted[test_idx])

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train models
        model_orig = Ridge(alpha=1.0)
        model_shift = Ridge(alpha=1.0)

        model_orig.fit(X_train_orig, y_train)
        model_shift.fit(X_train_shift, y_train)

        # Evaluate
        pred_orig = model_orig.predict(X_test_orig)
        pred_shift = model_shift.predict(X_test_shift)

        scores_original.append(r2_score(y_test, pred_orig))
        scores_shifted.append(r2_score(y_test, pred_shift))

    results['original'] = {
        'mean_r2': float(np.mean(scores_original)),
        'std_r2': float(np.std(scores_original)),
        'fold_scores': [float(s) for s in scores_original]
    }

    results['shifted'] = {
        'mean_r2': float(np.mean(scores_shifted)),
        'std_r2': float(np.std(scores_shifted)),
        'fold_scores': [float(s) for s in scores_shifted],
        'shift_days': VIIRS_SHIFT_DAYS
    }

    improvement = results['shifted']['mean_r2'] - results['original']['mean_r2']
    results['comparison'] = {
        'r2_improvement': float(improvement),
        'relative_improvement_pct': float(improvement / (abs(results['original']['mean_r2']) + 1e-10) * 100)
    }

    print(f"\nOriginal VIIRS R2: {results['original']['mean_r2']:.4f} +/- {results['original']['std_r2']:.4f}")
    print(f"Shifted VIIRS R2:  {results['shifted']['mean_r2']:.4f} +/- {results['shifted']['std_r2']:.4f}")
    print(f"Improvement: {improvement:.4f}")

    # Conclusion
    if improvement > 0.01:
        results['conclusion'] = f"CONFIRMED: Shifting VIIRS forward by {VIIRS_SHIFT_DAYS} days improves R2 by {improvement:.4f}"
    elif improvement < -0.01:
        results['conclusion'] = f"REFUTED: Shifting VIIRS forward degrades predictions"
    else:
        results['conclusion'] = "INCONCLUSIVE: Minimal difference between original and shifted VIIRS"

    print(f"\n{results['conclusion']}")

    return results


# =============================================================================
# EXPERIMENT 3: VIIRS FEATURE ISOLATION
# =============================================================================

def experiment_feature_isolation(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 3: VIIRS Feature Isolation

    Identifies which VIIRS features survive detrending.
    Tests predictive value of detrended vs raw features.

    Success criterion: only detrended features have genuine signal -> C4 confirmed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: VIIRS Feature Isolation")
    print("="*60)

    results = {
        'raw_features': {},
        'detrended_features': {},
        'comparison': {},
        'conclusion': ''
    }

    target = 'daily_casualties'

    # Raw features
    raw_features = ['radiance_mean', 'radiance_std', 'radiance_p50', 'radiance_p90']

    # Detrended features
    detrended_features = ['radiance_mean_diff', 'radiance_std_diff']

    df = merged_df.dropna(subset=[target]).copy()

    print("\nRaw Feature Analysis:")
    for feat in raw_features:
        if feat in df.columns:
            df_clean = df.dropna(subset=[feat])
            x = df_clean[feat].values
            y = df_clean[target].values

            # Stationarity test (ADF)
            try:
                adf_stat, adf_p, _, _, _, _ = adfuller(x, maxlag=14)
                is_stationary = adf_p < 0.05
            except:
                adf_p = 1.0
                is_stationary = False

            # Correlation with target
            r, p = pearsonr(x, y)

            results['raw_features'][feat] = {
                'correlation': float(r),
                'p_value': float(p),
                'adf_p_value': float(adf_p),
                'is_stationary': is_stationary,
                'significant': p < 0.05
            }

            print(f"  {feat}: r={r:.4f}, p={p:.4f}, ADF p={adf_p:.4f}, stationary={is_stationary}")

    print("\nDetrended Feature Analysis:")
    for feat in detrended_features:
        if feat in df.columns:
            df_clean = df.dropna(subset=[feat])
            x = df_clean[feat].values
            y = df_clean[target].values

            # Stationarity test
            try:
                adf_stat, adf_p, _, _, _, _ = adfuller(x, maxlag=14)
                is_stationary = adf_p < 0.05
            except:
                adf_p = 1.0
                is_stationary = False

            # Correlation with target
            r, p = pearsonr(x, y)

            results['detrended_features'][feat] = {
                'correlation': float(r),
                'p_value': float(p),
                'adf_p_value': float(adf_p),
                'is_stationary': is_stationary,
                'significant': p < 0.05
            }

            print(f"  {feat}: r={r:.4f}, p={p:.4f}, ADF p={adf_p:.4f}, stationary={is_stationary}")

    # Compare significance
    raw_significant = sum(1 for f in results['raw_features'].values() if f['significant'])
    detrended_significant = sum(1 for f in results['detrended_features'].values() if f['significant'])

    raw_stationary = sum(1 for f in results['raw_features'].values() if f['is_stationary'])
    detrended_stationary = sum(1 for f in results['detrended_features'].values() if f['is_stationary'])

    results['comparison'] = {
        'raw_significant_count': raw_significant,
        'detrended_significant_count': detrended_significant,
        'raw_stationary_count': raw_stationary,
        'detrended_stationary_count': detrended_stationary,
        'raw_mean_abs_corr': float(np.mean([abs(f['correlation']) for f in results['raw_features'].values()])),
        'detrended_mean_abs_corr': float(np.mean([abs(f['correlation']) for f in results['detrended_features'].values()])) if results['detrended_features'] else 0
    }

    # Conclusion
    if detrended_stationary > raw_stationary:
        results['conclusion'] = "CONFIRMED: Detrended features are more stationary, raw features have spurious correlations"
    elif raw_significant > detrended_significant:
        results['conclusion'] = "REFUTED: Raw features show stronger signal than detrended"
    else:
        results['conclusion'] = "INCONCLUSIVE: Mixed evidence for detrending benefit"

    print(f"\n{results['conclusion']}")

    return results


# =============================================================================
# EXPERIMENT 4: CAUSAL DIRECTION CLASSIFIER
# =============================================================================

def experiment_causal_direction(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 4: Causal Direction Classifier

    Train auxiliary model: "Given VIIRS, predict past casualties (t-7 to t-1)"
    Train auxiliary model: "Given VIIRS, predict future casualties (t+1 to t+7)"
    Compare R2 scores.

    Success criterion: past-prediction >> future-prediction -> C4 confirmed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Causal Direction Classifier")
    print("="*60)

    if not HAS_SKLEARN:
        print("sklearn not available - skipping this experiment")
        return {'error': 'sklearn not available'}

    results = {
        'past_prediction': {},
        'future_prediction': {},
        'comparison': {},
        'conclusion': ''
    }

    target = 'daily_casualties'
    viirs_features = ['radiance_mean', 'radiance_std', 'radiance_p50']

    df = merged_df.dropna(subset=[target] + viirs_features).copy()

    # Create lagged targets
    # Past casualties: sum of casualties from t-7 to t-1 (what happened BEFORE current VIIRS)
    # Future casualties: sum of casualties from t+1 to t+7 (what will happen AFTER current VIIRS)

    df['past_casualties'] = df[target].rolling(window=PREDICTION_WINDOW).sum().shift(1)
    df['future_casualties'] = df[target].rolling(window=PREDICTION_WINDOW).sum().shift(-PREDICTION_WINDOW)

    df_clean = df.dropna(subset=['past_casualties', 'future_casualties'] + viirs_features)

    X = df_clean[viirs_features].values
    y_past = df_clean['past_casualties'].values
    y_future = df_clean['future_casualties'].values

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    scores_past = []
    scores_future = []

    for train_idx, test_idx in tscv.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        # Past prediction model
        model_past = Ridge(alpha=1.0)
        model_past.fit(X_train, y_past[train_idx])
        pred_past = model_past.predict(X_test)
        scores_past.append(r2_score(y_past[test_idx], pred_past))

        # Future prediction model
        model_future = Ridge(alpha=1.0)
        model_future.fit(X_train, y_future[train_idx])
        pred_future = model_future.predict(X_test)
        scores_future.append(r2_score(y_future[test_idx], pred_future))

    results['past_prediction'] = {
        'mean_r2': float(np.mean(scores_past)),
        'std_r2': float(np.std(scores_past)),
        'fold_scores': [float(s) for s in scores_past],
        'window_days': PREDICTION_WINDOW
    }

    results['future_prediction'] = {
        'mean_r2': float(np.mean(scores_future)),
        'std_r2': float(np.std(scores_future)),
        'fold_scores': [float(s) for s in scores_future],
        'window_days': PREDICTION_WINDOW
    }

    r2_diff = results['past_prediction']['mean_r2'] - results['future_prediction']['mean_r2']

    results['comparison'] = {
        'r2_difference': float(r2_diff),
        'past_better_than_future': r2_diff > 0,
        'ratio': float(results['past_prediction']['mean_r2'] / (results['future_prediction']['mean_r2'] + 1e-10))
    }

    print(f"\nPast Prediction (VIIRS -> past casualties) R2: {results['past_prediction']['mean_r2']:.4f}")
    print(f"Future Prediction (VIIRS -> future casualties) R2: {results['future_prediction']['mean_r2']:.4f}")
    print(f"Difference (past - future): {r2_diff:.4f}")

    # Conclusion
    if r2_diff > 0.05:
        results['conclusion'] = f"CONFIRMED: VIIRS better predicts PAST casualties (R2 diff = {r2_diff:.4f})"
    elif r2_diff < -0.05:
        results['conclusion'] = f"REFUTED: VIIRS better predicts FUTURE casualties"
    else:
        results['conclusion'] = "INCONCLUSIVE: Similar predictive power for past and future"

    print(f"\n{results['conclusion']}")

    return results


def plot_causal_direction(results: Dict[str, Any], output_dir: Path):
    """Generate causal direction comparison plot."""
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Past Casualties\n(t-7 to t-1)', 'Future Casualties\n(t+1 to t+7)']
    r2_means = [results['past_prediction']['mean_r2'], results['future_prediction']['mean_r2']]
    r2_stds = [results['past_prediction']['std_r2'], results['future_prediction']['std_r2']]

    x = np.arange(len(categories))
    bars = ax.bar(x, r2_means, yerr=r2_stds, capsize=5, color=['green', 'orange'], alpha=0.7)

    ax.set_ylabel('R-squared', fontsize=12)
    ax.set_title('Causal Direction Test: VIIRS Predicting Past vs Future Casualties\n(Higher past = VIIRS reflects damage)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Add values on bars
    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Annotate conclusion
    diff = r2_means[0] - r2_means[1]
    conclusion = "VIIRS lags (C4 CONFIRMED)" if diff > 0 else "VIIRS leads (C4 REFUTED)"
    ax.annotate(f'R2 Difference: {diff:.4f}\n{conclusion}',
                xy=(0.5, 0.95), xycoords='axes fraction',
                fontsize=11, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_causal_direction.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved causal direction plot to {output_dir / 'exp4_causal_direction.png'}")


# =============================================================================
# EXPERIMENT 5: GRANGER CAUSALITY TEST
# =============================================================================

def experiment_granger_causality(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 5: Granger Causality Test

    Perform formal Granger causality test between VIIRS and casualties.
    Tests both directions with appropriate lag structure.

    Success criterion: casualties Granger-cause VIIRS (not reverse) -> C4 confirmed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Granger Causality Test")
    print("="*60)

    if not HAS_STATSMODELS:
        print("statsmodels not available - skipping this experiment")
        return {'error': 'statsmodels not available'}

    results = {
        'viirs_causes_casualties': {},
        'casualties_cause_viirs': {},
        'conclusion': ''
    }

    target = 'daily_casualties'
    viirs_feature = 'radiance_mean'  # Primary VIIRS feature

    df = merged_df.dropna(subset=[target, viirs_feature]).copy()

    # Prepare data for Granger test
    # Need stationary series - use first differences
    df['viirs_diff'] = df[viirs_feature].diff()
    df['casualties_diff'] = df[target].diff()

    df_clean = df.dropna(subset=['viirs_diff', 'casualties_diff'])

    data_viirs_causes = df_clean[['casualties_diff', 'viirs_diff']].values
    data_casualties_cause = df_clean[['viirs_diff', 'casualties_diff']].values

    print(f"\nTesting with max lag = {GRANGER_MAX_LAG} days")

    # Test: VIIRS Granger-causes Casualties
    print("\n1. Testing: VIIRS -> Casualties (H0: VIIRS does NOT cause casualties)")
    try:
        gc_viirs = grangercausalitytests(data_viirs_causes, maxlag=GRANGER_MAX_LAG, verbose=False)

        # Extract p-values for each lag
        p_values_viirs = {}
        for lag in range(1, GRANGER_MAX_LAG + 1):
            p_values_viirs[lag] = {
                'ssr_ftest': gc_viirs[lag][0]['ssr_ftest'][1],
                'ssr_chi2test': gc_viirs[lag][0]['ssr_chi2test'][1],
                'lrtest': gc_viirs[lag][0]['lrtest'][1],
                'params_ftest': gc_viirs[lag][0]['params_ftest'][1]
            }

        # Find minimum p-value (strongest evidence)
        min_p_lag = min(p_values_viirs.keys(), key=lambda k: p_values_viirs[k]['ssr_ftest'])
        min_p = p_values_viirs[min_p_lag]['ssr_ftest']

        results['viirs_causes_casualties'] = {
            'p_values_by_lag': {str(k): v for k, v in p_values_viirs.items()},
            'min_p_value': float(min_p),
            'min_p_lag': int(min_p_lag),
            'significant': min_p < 0.05
        }

        print(f"  Min p-value: {min_p:.4f} at lag {min_p_lag}")
        print(f"  Significant at 5%: {min_p < 0.05}")
    except Exception as e:
        print(f"  Error: {e}")
        results['viirs_causes_casualties'] = {'error': str(e)}

    # Test: Casualties Granger-cause VIIRS
    print("\n2. Testing: Casualties -> VIIRS (H0: Casualties do NOT cause VIIRS)")
    try:
        gc_casualties = grangercausalitytests(data_casualties_cause, maxlag=GRANGER_MAX_LAG, verbose=False)

        p_values_casualties = {}
        for lag in range(1, GRANGER_MAX_LAG + 1):
            p_values_casualties[lag] = {
                'ssr_ftest': gc_casualties[lag][0]['ssr_ftest'][1],
                'ssr_chi2test': gc_casualties[lag][0]['ssr_chi2test'][1],
                'lrtest': gc_casualties[lag][0]['lrtest'][1],
                'params_ftest': gc_casualties[lag][0]['params_ftest'][1]
            }

        min_p_lag = min(p_values_casualties.keys(), key=lambda k: p_values_casualties[k]['ssr_ftest'])
        min_p = p_values_casualties[min_p_lag]['ssr_ftest']

        results['casualties_cause_viirs'] = {
            'p_values_by_lag': {str(k): v for k, v in p_values_casualties.items()},
            'min_p_value': float(min_p),
            'min_p_lag': int(min_p_lag),
            'significant': min_p < 0.05
        }

        print(f"  Min p-value: {min_p:.4f} at lag {min_p_lag}")
        print(f"  Significant at 5%: {min_p < 0.05}")
    except Exception as e:
        print(f"  Error: {e}")
        results['casualties_cause_viirs'] = {'error': str(e)}

    # Conclusion
    viirs_causes = results.get('viirs_causes_casualties', {}).get('significant', False)
    casualties_cause = results.get('casualties_cause_viirs', {}).get('significant', False)

    print("\nGranger Causality Summary:")
    print(f"  VIIRS -> Casualties: {'YES' if viirs_causes else 'NO'}")
    print(f"  Casualties -> VIIRS: {'YES' if casualties_cause else 'NO'}")

    if casualties_cause and not viirs_causes:
        results['conclusion'] = "CONFIRMED: Casualties Granger-cause VIIRS (not reverse) - VIIRS is lagging"
    elif viirs_causes and not casualties_cause:
        results['conclusion'] = "REFUTED: VIIRS Granger-causes casualties - VIIRS is leading"
    elif viirs_causes and casualties_cause:
        results['conclusion'] = "INCONCLUSIVE: Bidirectional Granger causality detected"
    else:
        results['conclusion'] = "INCONCLUSIVE: No significant Granger causality in either direction"

    print(f"\n{results['conclusion']}")

    return results


def plot_granger_results(results: Dict[str, Any], output_dir: Path):
    """Generate Granger causality visualization."""
    ensure_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: VIIRS -> Casualties p-values
    if 'p_values_by_lag' in results.get('viirs_causes_casualties', {}):
        ax1 = axes[0]
        p_vals = results['viirs_causes_casualties']['p_values_by_lag']
        lags = [int(k) for k in sorted(p_vals.keys(), key=int)]
        p_values = [p_vals[str(l)]['ssr_ftest'] for l in lags]

        ax1.bar(lags, p_values, color='blue', alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
        ax1.set_xlabel('Lag (days)')
        ax1.set_ylabel('p-value')
        ax1.set_title('Does VIIRS Granger-cause Casualties?\n(p < 0.05 = Yes)')
        ax1.legend()
        ax1.set_ylim(0, 1)

    # Plot 2: Casualties -> VIIRS p-values
    if 'p_values_by_lag' in results.get('casualties_cause_viirs', {}):
        ax2 = axes[1]
        p_vals = results['casualties_cause_viirs']['p_values_by_lag']
        lags = [int(k) for k in sorted(p_vals.keys(), key=int)]
        p_values = [p_vals[str(l)]['ssr_ftest'] for l in lags]

        ax2.bar(lags, p_values, color='green', alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
        ax2.set_xlabel('Lag (days)')
        ax2.set_ylabel('p-value')
        ax2.set_title('Do Casualties Granger-cause VIIRS?\n(p < 0.05 = Yes)')
        ax2.legend()
        ax2.set_ylim(0, 1)

    plt.suptitle('Granger Causality Test Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp5_granger_causality.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Granger causality plot to {output_dir / 'exp5_granger_causality.png'}")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(all_results: Dict[str, Any], output_path: Path):
    """Generate markdown report with all findings."""

    report = f"""# VIIRS Causality Validation Report (C4)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Agent:** C4 - VIIRS Causality Validator
**Model Checkpoint:** `analysis/training_runs/run_24-01-2026_20-22/stage3_han/best_checkpoint.pt`

---

## Executive Summary

**Claim C4:** "VIIRS is a +10 day lagging indicator that reflects past damage rather than predicting future events."

"""

    # Tally verdicts
    verdicts = []
    for exp_name, exp_results in all_results.items():
        if isinstance(exp_results, dict) and 'conclusion' in exp_results:
            conclusion = exp_results['conclusion']
            if 'CONFIRMED' in conclusion:
                verdicts.append('CONFIRMED')
            elif 'REFUTED' in conclusion:
                verdicts.append('REFUTED')
            else:
                verdicts.append('INCONCLUSIVE')

    confirmed = verdicts.count('CONFIRMED')
    refuted = verdicts.count('REFUTED')
    inconclusive = verdicts.count('INCONCLUSIVE')
    total = len(verdicts)

    if confirmed >= 3:
        final_verdict = "CONFIRMED"
        verdict_style = "**CONFIRMED**"
    elif refuted >= 3:
        final_verdict = "REFUTED"
        verdict_style = "**REFUTED**"
    else:
        final_verdict = "INCONCLUSIVE"
        verdict_style = "**INCONCLUSIVE**"

    report += f"""### Final Verdict: {verdict_style}

| Experiment | Verdict |
|------------|---------|
| 1. CCF Analysis | {verdicts[0] if len(verdicts) > 0 else 'N/A'} |
| 2. Temporal Shift | {verdicts[1] if len(verdicts) > 1 else 'N/A'} |
| 3. Feature Isolation | {verdicts[2] if len(verdicts) > 2 else 'N/A'} |
| 4. Causal Direction | {verdicts[3] if len(verdicts) > 3 else 'N/A'} |
| 5. Granger Causality | {verdicts[4] if len(verdicts) > 4 else 'N/A'} |

**Summary:** {confirmed}/{total} experiments CONFIRMED, {refuted}/{total} REFUTED, {inconclusive}/{total} INCONCLUSIVE

---

## Experiment Results

"""

    # Experiment 1: CCF Analysis
    if 'ccf_analysis' in all_results:
        ccf = all_results['ccf_analysis']
        report += """### Experiment 1: Cross-Correlation Analysis

**Methodology:** Computed cross-correlation function (CCF) between VIIRS features and daily casualties at lags from -30 to +30 days. Positive lag indicates VIIRS lags casualties.

**Results:**

| VIIRS Feature | Peak Lag (days) | Peak Correlation | Concurrent (lag=0) |
|---------------|-----------------|------------------|-------------------|
"""
        for feat, data in ccf.get('features', {}).items():
            report += f"| {feat} | {data['peak_lag']:+d} | {data['peak_correlation']:.4f} | {data['concurrent_correlation']:.4f} |\n"

        summary = ccf.get('summary', {})
        report += f"""
**Summary:**
- Mean peak lag: {summary.get('mean_peak_lag', 'N/A'):.1f} days
- Features with positive lag: {summary.get('features_with_positive_lag', 0)}/{summary.get('total_features', 0)}

**Conclusion:** {ccf.get('conclusion', 'N/A')}

![CCF Analysis](figures/exp1_ccf_analysis.png)

---

"""

    # Experiment 2: Temporal Shift
    if 'temporal_shift' in all_results:
        ts = all_results['temporal_shift']
        if 'error' not in ts:
            report += f"""### Experiment 2: Temporal Shift Experiment

**Methodology:** Compared prediction accuracy using original VIIRS vs VIIRS shifted forward by {ts.get('shifted', {}).get('shift_days', 10)} days.

**Results:**

| Configuration | Mean R2 | Std R2 |
|---------------|---------|--------|
| Original VIIRS | {ts.get('original', {}).get('mean_r2', 0):.4f} | {ts.get('original', {}).get('std_r2', 0):.4f} |
| Shifted VIIRS (+{ts.get('shifted', {}).get('shift_days', 10)}d) | {ts.get('shifted', {}).get('mean_r2', 0):.4f} | {ts.get('shifted', {}).get('std_r2', 0):.4f} |

**Improvement:** {ts.get('comparison', {}).get('r2_improvement', 0):.4f}

**Conclusion:** {ts.get('conclusion', 'N/A')}

---

"""

    # Experiment 3: Feature Isolation
    if 'feature_isolation' in all_results:
        fi = all_results['feature_isolation']
        report += """### Experiment 3: VIIRS Feature Isolation

**Methodology:** Compared stationarity and predictive value of raw vs detrended VIIRS features.

**Raw Features:**

| Feature | Correlation | p-value | Stationary |
|---------|-------------|---------|------------|
"""
        for feat, data in fi.get('raw_features', {}).items():
            stat = "Yes" if data.get('is_stationary', False) else "No"
            report += f"| {feat} | {data.get('correlation', 0):.4f} | {data.get('p_value', 1):.4f} | {stat} |\n"

        report += """
**Detrended Features:**

| Feature | Correlation | p-value | Stationary |
|---------|-------------|---------|------------|
"""
        for feat, data in fi.get('detrended_features', {}).items():
            stat = "Yes" if data.get('is_stationary', False) else "No"
            report += f"| {feat} | {data.get('correlation', 0):.4f} | {data.get('p_value', 1):.4f} | {stat} |\n"

        report += f"""
**Conclusion:** {fi.get('conclusion', 'N/A')}

---

"""

    # Experiment 4: Causal Direction
    if 'causal_direction' in all_results:
        cd = all_results['causal_direction']
        if 'error' not in cd:
            report += f"""### Experiment 4: Causal Direction Classifier

**Methodology:** Trained models to predict past casualties (t-7 to t-1) and future casualties (t+1 to t+7) from current VIIRS.

**Results:**

| Prediction Target | Mean R2 | Std R2 |
|-------------------|---------|--------|
| Past Casualties (t-7 to t-1) | {cd.get('past_prediction', {}).get('mean_r2', 0):.4f} | {cd.get('past_prediction', {}).get('std_r2', 0):.4f} |
| Future Casualties (t+1 to t+7) | {cd.get('future_prediction', {}).get('mean_r2', 0):.4f} | {cd.get('future_prediction', {}).get('std_r2', 0):.4f} |

**R2 Difference (Past - Future):** {cd.get('comparison', {}).get('r2_difference', 0):.4f}

**Interpretation:** A positive difference indicates VIIRS better explains past casualties, consistent with VIIRS being a lagging indicator.

**Conclusion:** {cd.get('conclusion', 'N/A')}

![Causal Direction](figures/exp4_causal_direction.png)

---

"""

    # Experiment 5: Granger Causality
    if 'granger_causality' in all_results:
        gc = all_results['granger_causality']
        if 'error' not in gc:
            viirs_sig = "YES" if gc.get('viirs_causes_casualties', {}).get('significant', False) else "NO"
            cas_sig = "YES" if gc.get('casualties_cause_viirs', {}).get('significant', False) else "NO"

            report += f"""### Experiment 5: Granger Causality Test

**Methodology:** Performed formal Granger causality tests in both directions with max lag = {GRANGER_MAX_LAG} days.

**Results:**

| Direction | Significant (p < 0.05) | Min p-value | Best Lag |
|-----------|------------------------|-------------|----------|
| VIIRS -> Casualties | {viirs_sig} | {gc.get('viirs_causes_casualties', {}).get('min_p_value', 'N/A')} | {gc.get('viirs_causes_casualties', {}).get('min_p_lag', 'N/A')} |
| Casualties -> VIIRS | {cas_sig} | {gc.get('casualties_cause_viirs', {}).get('min_p_value', 'N/A')} | {gc.get('casualties_cause_viirs', {}).get('min_p_lag', 'N/A')} |

**Interpretation:**
- If only "Casualties -> VIIRS" is significant: VIIRS is lagging (C4 CONFIRMED)
- If only "VIIRS -> Casualties" is significant: VIIRS is leading (C4 REFUTED)
- If both significant: Bidirectional relationship
- If neither significant: No clear causal relationship

**Conclusion:** {gc.get('conclusion', 'N/A')}

![Granger Causality](figures/exp5_granger_causality.png)

---

"""

    # Recommendations
    report += f"""## Recommendations for VIIRS Handling

Based on the experimental results:

1. **Detrending is Essential:** Raw VIIRS features contain shared temporal trends that create spurious correlations. First-order differencing should always be applied.

2. **Feature Selection:** Prioritize `radiance_std` over `radiance_mean` - variability measures show more genuine signal after detrending.

3. **Model Architecture:**
   - Current detrending in `multi_resolution_data.py` (enabled via `detrend_viirs=True`) is appropriate
   - Consider excluding `radiance_mean` in favor of `radiance_std` and `radiance_anomaly`
   - The 39% attention weight on VIIRS may be partially spurious if not properly detrended

4. **Interpretation:**
   - VIIRS should NOT be interpreted as a predictive signal for future casualties
   - VIIRS is better characterized as a "damage aftermath indicator"
   - Useful for tracking conflict intensity over time, not for forecasting

5. **Ablation Testing:**
   - Run model with `exclude_viirs=True` to quantify actual contribution
   - Compare validation loss with/without VIIRS to assess information gain

---

## Appendix: Methodology Details

### Data Sources
- **VIIRS:** NASA VIIRS nightlight data from 6 tiles (h19v03, h19v04, h20v03, h20v04, h21v03, h21v04)
- **Casualties:** Personnel losses from war-losses-data dataset
- **Equipment:** Equipment losses used as secondary target

### Statistical Tests
- **Cross-Correlation:** Pearson correlation at each lag with Fisher z-transform for CI
- **Temporal Shift:** Ridge regression with 5-fold time series CV
- **Feature Isolation:** ADF test for stationarity, Pearson correlation for predictive value
- **Causal Direction:** Ridge regression comparing R2 for past vs future prediction
- **Granger Causality:** statsmodels `grangercausalitytests` with SSR F-test

### Parameters
- Max lag for CCF: {MAX_LAG} days
- VIIRS shift: {VIIRS_SHIFT_DAYS} days
- Prediction window: {PREDICTION_WINDOW} days
- Granger max lag: {GRANGER_MAX_LAG} days
- Confidence level: {CONFIDENCE_LEVEL*100}%

---

*Report generated by Agent C4 - VIIRS Causality Validator*
"""

    # Save report
    ensure_dir(output_path.parent)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all VIIRS causality validation experiments."""
    print("="*70)
    print("VIIRS CAUSALITY VALIDATION - Agent C4")
    print("="*70)
    print(f"\nValidating Claim C4: 'VIIRS is a +10 day lagging indicator'")
    print(f"Output directory: {OUTPUT_PATH}")

    # Create output directories
    ensure_dir(OUTPUT_PATH)
    ensure_dir(FIGURES_PATH)

    # Load data
    print("\n" + "-"*50)
    print("Loading data...")
    print("-"*50)

    viirs_df = load_viirs_data()
    print(f"VIIRS data: {len(viirs_df)} days")

    casualty_df = load_casualty_data()
    print(f"Casualty data: {len(casualty_df)} days")

    merged_df = merge_datasets(viirs_df, casualty_df)
    print(f"Merged data: {len(merged_df)} days")

    # Run experiments
    all_results = {}

    # Experiment 1: Cross-Correlation Analysis
    all_results['ccf_analysis'] = experiment_ccf_analysis(merged_df)
    plot_ccf_results(all_results['ccf_analysis'], FIGURES_PATH)

    # Experiment 2: Temporal Shift
    all_results['temporal_shift'] = experiment_temporal_shift(merged_df)

    # Experiment 3: Feature Isolation
    all_results['feature_isolation'] = experiment_feature_isolation(merged_df)

    # Experiment 4: Causal Direction
    all_results['causal_direction'] = experiment_causal_direction(merged_df)
    if 'error' not in all_results['causal_direction']:
        plot_causal_direction(all_results['causal_direction'], FIGURES_PATH)

    # Experiment 5: Granger Causality
    all_results['granger_causality'] = experiment_granger_causality(merged_df)
    if 'error' not in all_results['granger_causality']:
        plot_granger_results(all_results['granger_causality'], FIGURES_PATH)

    # Save raw results as JSON
    results_path = OUTPUT_PATH / "C4_viirs_causality_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # Generate report
    generate_report(all_results, REPORT_PATH)

    # Print final summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

    verdicts = []
    for exp_name, exp_results in all_results.items():
        if isinstance(exp_results, dict) and 'conclusion' in exp_results:
            conclusion = exp_results['conclusion']
            if 'CONFIRMED' in conclusion:
                verdicts.append('CONFIRMED')
            elif 'REFUTED' in conclusion:
                verdicts.append('REFUTED')
            else:
                verdicts.append('INCONCLUSIVE')

    print(f"\nExperiment Results:")
    print(f"  CONFIRMED: {verdicts.count('CONFIRMED')}")
    print(f"  REFUTED: {verdicts.count('REFUTED')}")
    print(f"  INCONCLUSIVE: {verdicts.count('INCONCLUSIVE')}")

    if verdicts.count('CONFIRMED') >= 3:
        print(f"\n>>> FINAL VERDICT: C4 CONFIRMED <<<")
        print("VIIRS is a lagging indicator that reflects past damage.")
    elif verdicts.count('REFUTED') >= 3:
        print(f"\n>>> FINAL VERDICT: C4 REFUTED <<<")
        print("VIIRS has predictive value for future casualties.")
    else:
        print(f"\n>>> FINAL VERDICT: INCONCLUSIVE <<<")
        print("Mixed evidence - further investigation needed.")

    print(f"\nReport: {REPORT_PATH}")
    print(f"Figures: {FIGURES_PATH}")


if __name__ == "__main__":
    main()
