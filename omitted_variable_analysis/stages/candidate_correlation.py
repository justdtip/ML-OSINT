"""
Stage 4: Candidate Variable Correlation

Correlates extracted factors and residuals with candidate external variables
to identify what the omitted variables might represent.

Candidate sources:
- Temporal features (day of week, month, trend)
- War losses data (equipment, personnel - daily)
- VIINA events (conflict events - daily aggregatable)
- Lunar/astronomical features
- ERA5 weather (when available)
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
class CandidateCorrelationResults:
    """Results from candidate variable correlation analysis."""
    factor_correlations: Dict[str, List[Dict]]  # candidate -> list of {correlation, pvalue} per factor
    residual_correlations: Dict[str, Dict[str, Dict]]  # source -> candidate -> {correlation, pvalue}
    significant_pairs: List[Dict]  # Sorted by |correlation|
    candidate_metadata: Dict[str, Dict]  # Info about each candidate source
    lagged_correlations: Dict[str, Dict]  # Best lag correlations
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
            'factor_correlations': convert_for_json(self.factor_correlations),
            'residual_correlations': convert_for_json(self.residual_correlations),
            'significant_pairs': convert_for_json(self.significant_pairs),
            'candidate_metadata': convert_for_json(self.candidate_metadata),
            'lagged_correlations': convert_for_json(self.lagged_correlations),
            'metadata': convert_for_json(self.metadata)
        }

        with open(output_dir / "results" / "candidate_correlations.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved correlation results to {output_dir / 'results' / 'candidate_correlations.json'}")

    @classmethod
    def load(cls, output_dir: Path) -> 'CandidateCorrelationResults':
        """Load results from files."""
        with open(output_dir / "results" / "candidate_correlations.json") as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# CANDIDATE DATA LOADERS
# =============================================================================

def load_temporal_features(timestamps: List[str]) -> pd.DataFrame:
    """Generate temporal features from timestamps."""
    dates = pd.to_datetime(timestamps)

    df = pd.DataFrame({
        'date': dates,
        'day_of_week': dates.dayofweek,
        'day_of_month': dates.day,
        'month': dates.month,
        'days_since_start': (dates - dates.min()).days,
        'is_weekend': dates.dayofweek >= 5,
        'is_month_start': dates.day <= 5,
        'is_month_end': dates.day >= 26,
        'week_of_year': dates.isocalendar().week.astype(int),
        'quarter': dates.quarter,
    })

    # Add sin/cos encoding for cyclical features
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df.set_index('date')


def load_lunar_features(timestamps: List[str]) -> pd.DataFrame:
    """Calculate lunar phase features."""
    try:
        import ephem
    except ImportError:
        print("ephem not installed, using simplified lunar calculation")
        return _load_lunar_simple(timestamps)

    dates = pd.to_datetime(timestamps)
    features = []

    for date in dates:
        moon = ephem.Moon(date.strftime('%Y/%m/%d'))
        features.append({
            'date': date,
            'moon_phase': moon.phase / 100,  # 0-1
            'moon_illumination': moon.phase / 100,
        })

    return pd.DataFrame(features).set_index('date')


def _load_lunar_simple(timestamps: List[str]) -> pd.DataFrame:
    """Simplified lunar phase calculation (no external dependency)."""
    dates = pd.to_datetime(timestamps)

    # Synodic month = 29.53 days
    # Reference new moon: Jan 6, 2000
    ref_new_moon = pd.Timestamp('2000-01-06')
    synodic_month = 29.53059

    features = []
    for date in dates:
        days_since_ref = (date - ref_new_moon).days
        phase = (days_since_ref % synodic_month) / synodic_month
        # Convert to illumination (0 at new, 1 at full)
        illumination = (1 - np.cos(2 * np.pi * phase)) / 2

        features.append({
            'date': date,
            'moon_phase': phase,
            'moon_illumination': illumination,
            'moon_phase_sin': np.sin(2 * np.pi * phase),
            'moon_phase_cos': np.cos(2 * np.pi * phase),
        })

    return pd.DataFrame(features).set_index('date')


def load_war_losses_data(timestamps: List[str]) -> pd.DataFrame:
    """Load daily war losses data (equipment and personnel)."""
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"
    personnel_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"

    target_dates = pd.to_datetime(timestamps)

    features = {}

    # Equipment losses
    if equip_path.exists():
        with open(equip_path) as f:
            equip_data = json.load(f)

        equip_df = pd.DataFrame(equip_data)
        equip_df['date'] = pd.to_datetime(equip_df['date'])
        equip_df = equip_df.set_index('date')

        # Calculate daily changes (losses are cumulative)
        equip_cols = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery',
                      'MRL', 'drone', 'naval ship', 'anti-aircraft warfare']

        for col in equip_cols:
            if col in equip_df.columns:
                col_clean = col.replace(' ', '_').replace('-', '_')
                # Cumulative value
                features[f'equip_{col_clean}_cumulative'] = equip_df[col]
                # Daily change
                features[f'equip_{col_clean}_daily'] = equip_df[col].diff().fillna(0)

        # Total equipment (sum of key categories)
        key_cols = [c for c in ['tank', 'APC', 'aircraft', 'helicopter'] if c in equip_df.columns]
        if key_cols:
            features['equip_total_key'] = equip_df[key_cols].sum(axis=1)
            features['equip_total_key_daily'] = features['equip_total_key'].diff().fillna(0)

    # Personnel losses
    if personnel_path.exists():
        with open(personnel_path) as f:
            personnel_data = json.load(f)

        pers_df = pd.DataFrame(personnel_data)
        pers_df['date'] = pd.to_datetime(pers_df['date'])
        pers_df = pers_df.set_index('date')

        if 'personnel' in pers_df.columns:
            features['personnel_cumulative'] = pers_df['personnel']
            features['personnel_daily'] = pers_df['personnel'].diff().fillna(0)

        if 'POW' in pers_df.columns:
            features['pow_cumulative'] = pers_df['POW']

    if not features:
        return pd.DataFrame()

    # Combine into dataframe
    df = pd.DataFrame(features)

    # Align to target timestamps
    df = df.reindex(target_dates)

    # Forward fill then backward fill for any gaps
    df = df.ffill().bfill()

    return df


def load_viina_events(timestamps: List[str]) -> pd.DataFrame:
    """Load and aggregate VIINA conflict events to daily counts."""
    viina_dir = DATA_DIR / "viina" / "extracted"

    if not viina_dir.exists():
        viina_dir = DATA_DIR / "viina" / "VIINA-main" / "Data"

    target_dates = pd.to_datetime(timestamps)
    date_range = (target_dates.min(), target_dates.max())

    all_events = []

    # Load event files for relevant years
    for year in range(date_range[0].year, date_range[1].year + 1):
        # Try different file patterns
        for pattern in [f"event_info_latest_{year}.csv", f"event_1pd_latest_{year}.csv"]:
            event_file = viina_dir / pattern
            if event_file.exists():
                try:
                    # Read only needed columns to save memory
                    df = pd.read_csv(event_file, usecols=['date', 'geonameid', 'ADM1_NAME', 'source'])
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                    df = df.dropna(subset=['date'])
                    df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
                    all_events.append(df)
                    break  # Use first matching file per year
                except Exception as e:
                    print(f"Error loading {event_file}: {e}")
                    continue

    if not all_events:
        print("No VIINA event data found")
        return pd.DataFrame()

    events = pd.concat(all_events, ignore_index=True)

    # Aggregate to daily counts
    daily = events.groupby('date').agg({
        'geonameid': 'count',  # Total events
        'ADM1_NAME': lambda x: x.nunique(),  # Unique regions affected
        'source': lambda x: x.nunique(),  # Unique sources reporting
    }).rename(columns={
        'geonameid': 'viina_event_count',
        'ADM1_NAME': 'viina_regions_affected',
        'source': 'viina_sources_reporting'
    })

    # Add rolling statistics
    daily['viina_event_count_7d_mean'] = daily['viina_event_count'].rolling(7, min_periods=1).mean()
    daily['viina_event_count_7d_std'] = daily['viina_event_count'].rolling(7, min_periods=1).std().fillna(0)

    # Align to target timestamps
    daily = daily.reindex(target_dates)
    daily = daily.ffill().bfill().fillna(0)

    return daily


def load_era5_weather(timestamps: List[str]) -> pd.DataFrame:
    """Load ERA5 weather data if available."""
    era5_dir = DATA_DIR / "era5"
    daily_file = era5_dir / "era5_ukraine_daily.csv"

    if not daily_file.exists():
        # Check if point files exist and aggregate
        points_dir = era5_dir / "points"
        if points_dir.exists():
            csv_files = list(points_dir.glob("era5_*.csv"))
            if len(csv_files) > 100:  # Enough points downloaded
                print(f"Found {len(csv_files)} ERA5 point files, aggregating...")
                return _aggregate_era5_points(csv_files, timestamps)

        print("ERA5 data not available yet")
        return pd.DataFrame()

    target_dates = pd.to_datetime(timestamps)

    df = pd.read_csv(daily_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Align to target timestamps
    df = df.reindex(target_dates)
    df = df.ffill().bfill()

    return df


def _aggregate_era5_points(csv_files: List[Path], timestamps: List[str]) -> pd.DataFrame:
    """Aggregate ERA5 point data to daily Ukraine-wide values."""
    target_dates = pd.to_datetime(timestamps)

    all_data = []
    for f in csv_files[:50]:  # Sample first 50 files for speed
        try:
            df = pd.read_csv(f)
            if 'valid_time' in df.columns:
                df['date'] = pd.to_datetime(df['valid_time']).dt.date
            elif 'time' in df.columns:
                df['date'] = pd.to_datetime(df['time']).dt.date
            all_data.append(df)
        except Exception as e:
            continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    # Aggregate by date
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['lat', 'lon', 'latitude', 'longitude']]

    daily = combined.groupby('date')[numeric_cols].mean()

    # Rename columns
    daily.columns = [f'era5_{c}' for c in daily.columns]

    # Align to target timestamps
    daily = daily.reindex(target_dates)
    daily = daily.ffill().bfill()

    return daily


def load_hdx_conflict_events(timestamps: List[str]) -> pd.DataFrame:
    """Load HDX conflict events data."""
    hdx_file = DATA_DIR / "hdx" / "ukraine" / "conflict_events_2022_present.csv"

    if not hdx_file.exists():
        return pd.DataFrame()

    target_dates = pd.to_datetime(timestamps)

    df = pd.read_csv(hdx_file)

    # Parse dates
    if 'reference_period_start' in df.columns:
        df['date'] = pd.to_datetime(df['reference_period_start'])

    if 'date' not in df.columns:
        return pd.DataFrame()

    # Aggregate by date
    daily = df.groupby('date').agg({
        'events': 'sum' if 'events' in df.columns else 'count',
        'fatalities': 'sum' if 'fatalities' in df.columns else lambda x: 0,
    }).rename(columns={
        'events': 'hdx_conflict_events',
        'fatalities': 'hdx_fatalities'
    })

    # Align
    daily = daily.reindex(target_dates)
    daily = daily.ffill().bfill().fillna(0)

    return daily


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlations(
    factor_scores: np.ndarray,
    residuals: Dict[str, np.ndarray],
    candidates: pd.DataFrame,
    timestamps: List[str],
    max_lag: int = 30
) -> Tuple[Dict, Dict, List, Dict]:
    """
    Compute correlations between factors/residuals and candidate variables.

    Returns:
        factor_correlations: dict[candidate] -> list of {correlation, pvalue} per factor
        residual_correlations: dict[source] -> dict[candidate] -> {correlation, pvalue}
        significant_pairs: list sorted by |correlation|
        lagged_correlations: best lag correlations
    """
    from scipy import stats

    factor_correlations = {}
    residual_correlations = {s: {} for s in residuals.keys()}
    significant_pairs = []
    lagged_correlations = {}

    n_factors = factor_scores.shape[1]
    sources = list(residuals.keys())

    for col in candidates.columns:
        candidate_values = candidates[col].values

        # Skip if too many NaNs
        valid_mask = ~np.isnan(candidate_values)
        if valid_mask.sum() < len(candidate_values) * 0.5:
            continue

        # Interpolate NaNs
        if np.isnan(candidate_values).any():
            candidate_values = pd.Series(candidate_values).interpolate().fillna(method='bfill').fillna(method='ffill').values

        # Factor correlations
        factor_corrs = []
        for f in range(n_factors):
            try:
                corr, pval = stats.pearsonr(factor_scores[:, f], candidate_values)
                factor_corrs.append({'correlation': float(corr), 'pvalue': float(pval)})

                if pval < 0.05 and abs(corr) > 0.1:
                    significant_pairs.append({
                        'type': 'factor',
                        'index': f,
                        'candidate': col,
                        'correlation': float(corr),
                        'pvalue': float(pval),
                        'lag': 0
                    })
            except:
                factor_corrs.append({'correlation': 0.0, 'pvalue': 1.0})

        factor_correlations[col] = factor_corrs

        # Residual magnitude correlations
        for source in sources:
            res_magnitude = np.linalg.norm(residuals[source], axis=1)

            try:
                corr, pval = stats.pearsonr(res_magnitude, candidate_values)
                residual_correlations[source][col] = {
                    'correlation': float(corr),
                    'pvalue': float(pval)
                }

                if pval < 0.05 and abs(corr) > 0.1:
                    significant_pairs.append({
                        'type': 'residual',
                        'source': source,
                        'candidate': col,
                        'correlation': float(corr),
                        'pvalue': float(pval),
                        'lag': 0
                    })
            except:
                residual_correlations[source][col] = {'correlation': 0.0, 'pvalue': 1.0}

        # Lagged correlations (check if candidate leads/lags factors)
        best_lag_corrs = {}
        for f in range(n_factors):
            best_corr = 0
            best_lag = 0
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Candidate leads factor
                    x = candidate_values[:lag]
                    y = factor_scores[-lag:, f]
                elif lag > 0:
                    # Factor leads candidate
                    x = candidate_values[lag:]
                    y = factor_scores[:-lag, f]
                else:
                    x = candidate_values
                    y = factor_scores[:, f]

                if len(x) < 10:
                    continue

                try:
                    corr, _ = stats.pearsonr(x, y)
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                except:
                    continue

            best_lag_corrs[f'factor_{f}'] = {'correlation': float(best_corr), 'lag': best_lag}

        lagged_correlations[col] = best_lag_corrs

    # Sort significant pairs by absolute correlation
    significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return factor_correlations, residual_correlations, significant_pairs, lagged_correlations


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_correlation_visualizations(
    results: CandidateCorrelationResults,
    factor_scores: np.ndarray,
    timestamps: List[str],
    output_dir: Path
):
    """Create visualization plots for correlation analysis."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    figures_dir = output_dir / "figures"

    # Color palette
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

    # 1. Factor-candidate correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    candidates = list(results.factor_correlations.keys())
    n_factors = len(list(results.factor_correlations.values())[0])

    # Build correlation matrix
    corr_matrix = np.zeros((len(candidates), n_factors))
    for i, cand in enumerate(candidates):
        for j, fc in enumerate(results.factor_correlations[cand]):
            corr_matrix[i, j] = fc['correlation']

    # Only show candidates with at least one notable correlation
    notable_mask = np.abs(corr_matrix).max(axis=1) > 0.15
    if notable_mask.sum() > 0:
        notable_candidates = [c for c, m in zip(candidates, notable_mask) if m]
        notable_matrix = corr_matrix[notable_mask]

        im = ax.imshow(notable_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)

        ax.set_xticks(range(n_factors))
        ax.set_xticklabels([f'Factor {i+1}' for i in range(n_factors)])
        ax.set_yticks(range(len(notable_candidates)))
        ax.set_yticklabels(notable_candidates, fontsize=8)

        # Add correlation values
        for i in range(len(notable_candidates)):
            for j in range(n_factors):
                val = notable_matrix[i, j]
                color = 'white' if abs(val) > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=7)

        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Factor-Candidate Correlations\n(showing |r| > 0.15)')

    plt.tight_layout()
    plt.savefig(figures_dir / 'factor_candidate_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Top significant correlations bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    top_pairs = results.significant_pairs[:20]
    if top_pairs:
        labels = []
        corrs = []
        colors_list = []

        for pair in top_pairs:
            if pair['type'] == 'factor':
                label = f"F{pair['index']+1} ↔ {pair['candidate']}"
                color = colors[pair['index'] % len(colors)]
            else:
                label = f"{pair['source']} ↔ {pair['candidate']}"
                color = '#95a5a6'
            labels.append(label)
            corrs.append(pair['correlation'])
            colors_list.append(color)

        y_pos = range(len(labels))
        ax.barh(y_pos, corrs, color=colors_list, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Correlation')
        ax.set_title('Top 20 Significant Correlations')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlim(-1, 1)

    plt.tight_layout()
    plt.savefig(figures_dir / 'top_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Lagged correlation analysis for top candidates
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Find candidates with strongest lagged correlations
    lag_strengths = []
    for cand, lag_corrs in results.lagged_correlations.items():
        max_corr = max(abs(v['correlation']) for v in lag_corrs.values())
        lag_strengths.append((cand, max_corr, lag_corrs))

    lag_strengths.sort(key=lambda x: x[1], reverse=True)

    for idx, (cand, _, lag_corrs) in enumerate(lag_strengths[:6]):
        ax = axes[idx]

        factors = sorted(lag_corrs.keys())
        corrs = [lag_corrs[f]['correlation'] for f in factors]
        lags = [lag_corrs[f]['lag'] for f in factors]

        x = range(len(factors))
        bars = ax.bar(x, corrs, color=[colors[i % len(colors)] for i in range(len(factors))], alpha=0.8)

        # Add lag annotations
        for i, (bar, lag) in enumerate(zip(bars, lags)):
            if abs(lag) > 0:
                ax.annotate(f'lag={lag}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'F{i+1}' for i in range(len(factors))], fontsize=9)
        ax.set_title(cand[:30], fontsize=10)
        ax.set_ylabel('Correlation')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylim(-0.8, 0.8)

    plt.suptitle('Best Lagged Correlations by Candidate', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'lagged_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Time series comparison for top correlations
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    dates = pd.to_datetime(timestamps)

    for idx, (ax, pair) in enumerate(zip(axes, results.significant_pairs[:3])):
        if pair['type'] == 'factor':
            factor_idx = pair['index']
            ax.plot(dates, factor_scores[:, factor_idx],
                   color=colors[factor_idx], label=f'Factor {factor_idx+1}', linewidth=1.5)

        ax.set_ylabel(f"Factor {pair.get('index', '?')+1 if pair['type']=='factor' else pair['source']}")
        ax.set_title(f"{pair['candidate']} (r={pair['correlation']:.3f})")
        ax.legend(loc='upper left')

        # Add secondary axis note
        ax.text(0.99, 0.95, f"Candidate: {pair['candidate']}",
               transform=ax.transAxes, ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.suptitle('Factor Time Series vs Top Correlated Candidates', fontsize=12)
    plt.tight_layout()
    plt.savefig(figures_dir / 'factor_candidate_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation visualizations to {figures_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_candidate_correlation(
    factor_results_path: Path = None,
    residual_results_path: Path = None,
    max_lag: int = 30,
    output_dir: Path = None
) -> CandidateCorrelationResults:
    """
    Run Stage 4: Candidate Variable Correlation.

    Args:
        factor_results_path: Path to latent_factors.npz from Stage 3
        residual_results_path: Path to masked_residuals.npz from Stage 1
        max_lag: Maximum lag for lagged correlations
        output_dir: Output directory

    Returns:
        CandidateCorrelationResults
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    if factor_results_path is None:
        factor_results_path = output_dir / "results" / "latent_factors.npz"

    if residual_results_path is None:
        residual_results_path = output_dir / "results" / "masked_residuals.npz"

    print("=" * 60)
    print("STAGE 4: CANDIDATE VARIABLE CORRELATION")
    print("=" * 60)

    # Load factor scores
    print("\n1. Loading factor scores...")
    factor_data = np.load(factor_results_path)
    factor_scores = factor_data['factor_scores']
    print(f"   Factor scores shape: {factor_scores.shape}")

    # Load residuals
    print("\n2. Loading residuals...")
    residual_data = np.load(residual_results_path)
    residuals = {key: residual_data[key] for key in residual_data.files}
    print(f"   Sources: {list(residuals.keys())}")

    # Load timestamps
    print("\n3. Loading timestamps...")
    with open(output_dir / "results" / "residual_metadata.json") as f:
        metadata = json.load(f)
    timestamps = metadata['timestamps']
    print(f"   Date range: {timestamps[0]} to {timestamps[-1]} ({len(timestamps)} samples)")

    # Load candidate variables
    print("\n4. Loading candidate variables...")
    candidates_list = []
    candidate_metadata = {}

    # Temporal features
    print("   - Temporal features...")
    temporal_df = load_temporal_features(timestamps)
    candidates_list.append(temporal_df)
    candidate_metadata['temporal'] = {
        'description': 'Time-based patterns',
        'n_features': len(temporal_df.columns),
        'source': 'computed_from_timestamps'
    }

    # Lunar features
    print("   - Lunar features...")
    lunar_df = load_lunar_features(timestamps)
    if not lunar_df.empty:
        candidates_list.append(lunar_df)
        candidate_metadata['lunar'] = {
            'description': 'Moon phase features',
            'n_features': len(lunar_df.columns),
            'source': 'astronomical_calculations'
        }

    # War losses
    print("   - War losses data...")
    losses_df = load_war_losses_data(timestamps)
    if not losses_df.empty:
        candidates_list.append(losses_df)
        candidate_metadata['war_losses'] = {
            'description': 'Daily equipment and personnel losses',
            'n_features': len(losses_df.columns),
            'source': 'war-losses-data/russia_losses_*.json'
        }
        print(f"     Loaded {len(losses_df.columns)} war losses features")

    # VIINA events
    print("   - VIINA conflict events...")
    viina_df = load_viina_events(timestamps)
    if not viina_df.empty:
        candidates_list.append(viina_df)
        candidate_metadata['viina'] = {
            'description': 'Daily conflict event counts',
            'n_features': len(viina_df.columns),
            'source': 'viina/extracted/event_*.csv'
        }
        print(f"     Loaded {len(viina_df.columns)} VIINA features")

    # ERA5 weather
    print("   - ERA5 weather data...")
    era5_df = load_era5_weather(timestamps)
    if not era5_df.empty:
        candidates_list.append(era5_df)
        candidate_metadata['era5'] = {
            'description': 'Weather conditions',
            'n_features': len(era5_df.columns),
            'source': 'era5/era5_ukraine_daily.csv'
        }
        print(f"     Loaded {len(era5_df.columns)} ERA5 features")

    # HDX conflict events
    print("   - HDX conflict events...")
    hdx_df = load_hdx_conflict_events(timestamps)
    if not hdx_df.empty:
        candidates_list.append(hdx_df)
        candidate_metadata['hdx'] = {
            'description': 'HDX aggregated conflict data',
            'n_features': len(hdx_df.columns),
            'source': 'hdx/ukraine/conflict_events_*.csv'
        }
        print(f"     Loaded {len(hdx_df.columns)} HDX features")

    # Combine all candidates
    all_candidates = pd.concat(candidates_list, axis=1)
    print(f"\n   Total candidate features: {len(all_candidates.columns)}")

    # Compute correlations
    print("\n5. Computing correlations...")
    factor_corrs, residual_corrs, sig_pairs, lagged_corrs = compute_correlations(
        factor_scores, residuals, all_candidates, timestamps, max_lag
    )

    print(f"   Found {len(sig_pairs)} significant correlations (p < 0.05, |r| > 0.1)")

    # Print top correlations
    print("\n   Top 10 correlations:")
    for i, pair in enumerate(sig_pairs[:10]):
        if pair['type'] == 'factor':
            print(f"   {i+1}. Factor {pair['index']+1} ↔ {pair['candidate']}: r={pair['correlation']:.3f}")
        else:
            print(f"   {i+1}. {pair['source']} ↔ {pair['candidate']}: r={pair['correlation']:.3f}")

    # Create results object
    results = CandidateCorrelationResults(
        factor_correlations=factor_corrs,
        residual_correlations=residual_corrs,
        significant_pairs=sig_pairs,
        candidate_metadata=candidate_metadata,
        lagged_correlations=lagged_corrs,
        metadata={
            'n_samples': len(timestamps),
            'n_factors': factor_scores.shape[1],
            'n_candidates': len(all_candidates.columns),
            'max_lag': max_lag,
            'date_range': [timestamps[0], timestamps[-1]],
            'sources_analyzed': list(residuals.keys())
        }
    )

    # Save results
    print("\n6. Saving results...")
    results.save(output_dir)

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_correlation_visualizations(results, factor_scores, timestamps, output_dir)

    print("\n" + "=" * 60)
    print("STAGE 4 COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_candidate_correlation()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nCandidate sources loaded:")
    for source, meta in results.candidate_metadata.items():
        print(f"  - {source}: {meta['n_features']} features")

    print(f"\nSignificant correlations found: {len(results.significant_pairs)}")

    # Group by factor
    factor_hits = {}
    for pair in results.significant_pairs:
        if pair['type'] == 'factor':
            f = pair['index']
            if f not in factor_hits:
                factor_hits[f] = []
            factor_hits[f].append(pair)

    print("\nTop correlates per factor:")
    for f in sorted(factor_hits.keys()):
        top = factor_hits[f][0]
        print(f"  Factor {f+1}: {top['candidate']} (r={top['correlation']:.3f})")
