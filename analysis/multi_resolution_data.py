"""
Multi-Resolution Time Series DataLoader Infrastructure

This module provides PyTorch DataLoader infrastructure for handling multi-resolution
time series data with explicit observation masks. It handles:

- Daily resolution sources (~1000+ timesteps): equipment, personnel, deepstate, firms, viina, viirs
- Monthly resolution sources (~35 timesteps): sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom

Key Features:
- Explicit observation masks (True = real observation, False = no observation)
- NO forward-filling or interpolation of missing values
- Proper temporal alignment between daily and monthly timelines
- Month boundary indices for daily-to-monthly alignment
- Variable-length sequence handling via custom collate function

Author: Data Engineering Pipeline
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Import centralized path configuration
from config.paths import (
    PROJECT_ROOT, DATA_DIR as CONFIG_DATA_DIR, ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    UCDP_DIR, FIRMS_DIR, DEEPSTATE_DIR, SENTINEL_DIR,
    WAR_LOSSES_DIR, VIIRS_DIR, HDX_DIR, IOM_DIR, VIINA_DIR,
    UCDP_EVENTS_FILE, FIRMS_ARCHIVE_FILE, FIRMS_NRT_FILE,
    EQUIPMENT_LOSSES_FILE, PERSONNEL_LOSSES_FILE,
    SENTINEL_RAW_FILE, SENTINEL_WEEKLY_FILE,
)

# Paths - use centralized config with backward compatible aliases
BASE_DIR = PROJECT_ROOT  # Alias for backward compatibility
DATA_DIR = CONFIG_DATA_DIR  # Use centralized config
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR  # Use centralized config

# Missing value sentinel - chosen to be clearly distinguishable from real data
MISSING_VALUE = -999.0


class Resolution(Enum):
    """Temporal resolution for data sources."""
    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass
class SourceConfig:
    """Configuration for a data source."""
    name: str
    resolution: Resolution
    feature_names: List[str]
    loader_func: str  # Name of the loading function
    normalization: str = "standard"  # standard, log, minmax

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclass
class MultiResolutionConfig:
    """Configuration for multi-resolution dataset."""
    # Daily sources
    daily_sources: List[str] = field(default_factory=lambda: [
        "equipment", "personnel", "deepstate", "firms", "viina", "viirs"
    ])

    # Monthly sources
    monthly_sources: List[str] = field(default_factory=lambda: [
        "sentinel", "hdx_conflict", "hdx_food", "hdx_rainfall", "iom"
    ])

    # Date range
    start_date: str = "2022-02-24"
    end_date: Optional[str] = None  # None = use latest available

    # Sequence parameters
    daily_seq_len: int = 365  # ~1 year of daily data
    monthly_seq_len: int = 12  # 1 year of monthly data
    prediction_horizon: int = 1  # Predict next month

    # Sample stride for validation/test sets to reduce temporal overlap
    # e.g., stride=3 means only every 3rd valid sample is used
    # Set to None to auto-compute based on monthly_seq_len
    val_sample_stride: Optional[int] = None

    # Missing value handling
    missing_value: float = MISSING_VALUE

    # VIIRS configuration
    # VIIRS radiance LAGS casualties by ~10 days (not leading), so correlation
    # with casualty data is an artifact of shared temporal trend, not genuine
    # predictive signal. These options help address this issue.
    detrend_viirs: bool = True  # Apply first-order differencing to remove trend (Probe 1.2.3)
    exclude_viirs: bool = False  # Completely exclude VIIRS from the dataset

    # Equipment disaggregation (Probe 1.1.2)
    # Drones have highest mutual information (MI=0.449) and lead casualties by 7-27 days.
    # When True, replaces aggregated "equipment" with separate drones/armor/artillery/aircraft.
    use_disaggregated_equipment: bool = False  # Set True for optimized source separation

    def get_effective_daily_sources(self) -> List[str]:
        """
        Get effective daily sources list based on configuration options.

        Handles:
        - Equipment disaggregation: replaces 'equipment' with drones/armor/artillery/aircraft
        - VIIRS exclusion: removes 'viirs' from sources

        Returns:
            List of daily source names to use for training/inference.
        """
        sources = list(self.daily_sources)

        # Handle equipment disaggregation
        if self.use_disaggregated_equipment and "equipment" in sources:
            idx = sources.index("equipment")
            sources = sources[:idx] + ["drones", "armor", "artillery", "aircraft"] + sources[idx+1:]

        # Handle VIIRS exclusion
        if self.exclude_viirs and "viirs" in sources:
            sources.remove("viirs")

        return sources


# =============================================================================
# DATA LOADING FUNCTIONS WITH OBSERVATION MASKS
# =============================================================================

def load_equipment_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load equipment loss data at daily resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        warnings.warn(f"Equipment data not found at {equip_path}")
        return pd.DataFrame(), np.array([])

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Define feature columns
    feature_cols = [
        'aircraft', 'helicopter', 'tank', 'APC', 'field_artillery',
        'MRL', 'drone', 'naval_ship', 'anti_aircraft_warfare',
        'special_equipment', 'vehicles_and_fuel_tanks', 'cruise_missiles'
    ]

    # Normalize column names (handle spaces)
    df.columns = df.columns.str.replace(' ', '_')

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to complete date range - DO NOT FILL MISSING VALUES
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask: True where we have real observations
    # A row is observed if ANY feature column has a non-null value
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Select and prepare features - keep NaN as NaN (will be converted to missing_value later)
    features_df = df[['date'] + available_cols].copy()

    return features_df, observation_mask


def load_personnel_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load personnel loss data at daily resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    personnel_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"

    if not personnel_path.exists():
        warnings.warn(f"Personnel data not found at {personnel_path}")
        return pd.DataFrame(), np.array([])

    with open(personnel_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Define feature columns
    feature_cols = ['personnel', 'POW']

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Calculate daily change where we have consecutive observations
    if 'personnel' in df.columns:
        df['personnel_daily'] = df['personnel'].diff()
        available_cols.append('personnel_daily')

    features_df = df[['date'] + available_cols].copy()

    return features_df, observation_mask


def load_deepstate_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load DeepState front line data at daily resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    wayback_dir = DATA_DIR / "deepstate" / "wayback_snapshots"

    if not wayback_dir.exists():
        warnings.warn(f"DeepState data not found at {wayback_dir}")
        return pd.DataFrame(), np.array([])

    snapshots = []

    for snapshot_file in sorted(wayback_dir.glob("*.json")):
        try:
            with open(snapshot_file) as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                features = data
            elif isinstance(data, dict) and 'features' in data:
                features = data['features']
            elif isinstance(data, dict) and 'map' in data:
                features = data['map'].get('features', [])
            else:
                continue

            # Count different feature types
            polygons = sum(1 for f in features if f.get('geometry', {}).get('type') == 'Polygon')
            multipolygons = sum(1 for f in features if f.get('geometry', {}).get('type') == 'MultiPolygon')
            points = sum(1 for f in features if f.get('geometry', {}).get('type') == 'Point')
            lines = sum(1 for f in features if f.get('geometry', {}).get('type') in ('LineString', 'MultiLineString'))

            # Extract date from filename
            # Handle formats like: deepstate_wayback_20220510044041.json or deepstate_20220510_...
            filename = snapshot_file.stem
            if filename.startswith('_'):  # Skip index files
                continue

            # Try to extract date from various formats
            date = None
            import re
            # Look for 8-digit date pattern (YYYYMMDD)
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                try:
                    date = pd.to_datetime(date_match.group(1)[:8], format='%Y%m%d')
                except ValueError:
                    pass

            if date is None:
                continue

            snapshots.append({
                'date': date,
                'polygons': polygons,
                'multipolygons': multipolygons,
                'points': points,
                'lines': lines,
                'total_features': len(features)
            })
        except Exception:
            continue

    if not snapshots:
        return pd.DataFrame(), np.array([])

    df = pd.DataFrame(snapshots)
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

    # Validate dates
    df = df.dropna(subset=['date'])
    if df.empty:
        return pd.DataFrame(), np.array([])

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    # Validate start/end dates
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame(), np.array([])

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    feature_cols = ['polygons', 'multipolygons', 'points', 'lines', 'total_features']
    observation_mask = df[feature_cols].notna().any(axis=1).values

    features_df = df[['date'] + feature_cols].copy()

    return features_df, observation_mask


def load_firms_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS fire detection data at daily resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    firms_path = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"

    if not firms_path.exists():
        warnings.warn(f"FIRMS data not found at {firms_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(firms_path)
    df['date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Aggregate by day
    daily_agg = df.groupby(df['date'].dt.date).agg({
        'brightness': ['count', 'mean', 'max', 'std'],
        'bright_t31': ['mean', 'max'],
        'frp': ['sum', 'mean', 'max'],
        'confidence': lambda x: (x == 'h').sum(),  # High confidence count
        'daynight': lambda x: (x == 'D').sum(),  # Daytime count
        'scan': 'mean',
        'track': 'mean'
    }).reset_index()

    # Flatten column names
    daily_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                         for col in daily_agg.columns]
    daily_agg = daily_agg.rename(columns={
        'date': 'date',
        'brightness_count': 'fire_count',
        'brightness_mean': 'brightness_mean',
        'brightness_max': 'brightness_max',
        'brightness_std': 'brightness_std',
        'bright_t31_mean': 'bright_t31_mean',
        'bright_t31_max': 'bright_t31_max',
        'frp_sum': 'frp_total',
        'frp_mean': 'frp_mean',
        'frp_max': 'frp_max',
        'confidence_<lambda>': 'high_conf_fires',
        'daynight_<lambda>': 'day_fires',
        'scan_mean': 'scan_mean',
        'track_mean': 'track_mean'
    })

    daily_agg['date'] = pd.to_datetime(daily_agg['date'])

    # Create complete daily date range
    if start_date is None:
        start_date = daily_agg['date'].min()
    if end_date is None:
        end_date = daily_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    daily_agg = daily_agg.set_index('date')
    daily_agg = daily_agg.reindex(date_range)
    daily_agg.index.name = 'date'
    daily_agg = daily_agg.reset_index()

    # Create observation mask
    feature_cols = ['fire_count', 'brightness_mean', 'brightness_max', 'brightness_std',
                    'bright_t31_mean', 'bright_t31_max', 'frp_total', 'frp_mean', 'frp_max',
                    'high_conf_fires', 'day_fires', 'scan_mean', 'track_mean']
    available_cols = [c for c in feature_cols if c in daily_agg.columns]
    observation_mask = daily_agg[available_cols].notna().any(axis=1).values

    features_df = daily_agg[['date'] + available_cols].copy()

    return features_df, observation_mask


def load_viirs_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    detrend: bool = False
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load NASA VIIRS nightlight data at daily resolution.

    This function loads VIIRS daily brightness statistics from 6 tiles covering
    Ukraine (h19v03, h19v04, h20v03, h20v04, h21v03, h21v04) and aggregates them
    into daily features suitable for the multi-resolution pipeline.

    Features computed:
        - viirs_radiance_mean: Mean radiance across tiles (log-scaled)
        - viirs_radiance_std: Standard deviation of radiance (variability indicator)
        - viirs_radiance_anomaly: Deviation from rolling 7-day baseline (z-score)
        - viirs_clear_sky_pct: Mean percentage of clear sky pixels
        - viirs_coverage_count: Number of valid tiles on that date (max 6)

    Detrending:
        When detrend=True, applies first-order differencing to remove shared
        temporal trends. This is important because VIIRS radiance LAGs casualties
        by ~10 days (not leading), so correlation is an artifact of shared trend,
        not genuine predictive signal.

        Detrended formula: viirs_detrended[t] = viirs[t] - viirs[t-1]

        This preserves observation masks - a timestep is only observed if BOTH
        the current and previous timesteps have valid observations.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        detrend: If True, apply first-order differencing to remove temporal trends.
                 Default False for backward compatibility.

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    viirs_path = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"

    if not viirs_path.exists():
        warnings.warn(f"VIIRS nightlight data not found at {viirs_path}")
        return pd.DataFrame(), np.array([])

    # Load raw data
    df = pd.read_csv(viirs_path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Aggregate across all 6 tiles for each date
    # Use mean for radiance metrics, sum for counts
    daily_agg = df.groupby('date').agg({
        # Mean aggregation for radiance features
        'radiance_mean': 'mean',
        'radiance_std': 'mean',
        'radiance_min': 'min',
        'radiance_max': 'max',
        'radiance_sum': 'sum',
        # Percentiles - mean across tiles
        'radiance_p10': 'mean',
        'radiance_p25': 'mean',
        'radiance_p50': 'mean',
        'radiance_p75': 'mean',
        'radiance_p90': 'mean',
        # Quality/coverage metrics
        'pct_clear_sky': 'mean',
        'moon_illumination_pct': 'mean',
        'lunar_zenith_mean': 'mean',
        # Count number of tiles per day
        'tile': 'count',
    }).reset_index()

    daily_agg = daily_agg.rename(columns={'tile': 'tile_count'})

    # Apply log-scaling to radiance_mean for better distribution
    # Add small epsilon to avoid log(0)
    daily_agg['radiance_mean_log'] = np.log1p(daily_agg['radiance_mean'])

    # Compute rolling 7-day baseline and anomaly
    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
    daily_agg['radiance_rolling_mean'] = daily_agg['radiance_mean'].rolling(
        window=7, min_periods=3, center=False
    ).mean()
    daily_agg['radiance_rolling_std'] = daily_agg['radiance_mean'].rolling(
        window=7, min_periods=3, center=False
    ).std()

    # Compute anomaly as z-score relative to rolling window
    # Negative values indicate darker than usual (potential destruction/outages)
    daily_agg['radiance_anomaly'] = (
        daily_agg['radiance_mean'] - daily_agg['radiance_rolling_mean']
    ) / daily_agg['radiance_rolling_std'].replace(0, np.nan)

    # Fill NaN anomaly values (from early dates with insufficient window)
    daily_agg['radiance_anomaly'] = daily_agg['radiance_anomaly'].fillna(0)

    # Select and rename final features for the pipeline
    feature_df = pd.DataFrame({
        'date': daily_agg['date'],
        'viirs_radiance_mean': daily_agg['radiance_mean_log'],  # Log-scaled
        'viirs_radiance_std': daily_agg['radiance_std'],
        'viirs_radiance_anomaly': daily_agg['radiance_anomaly'],
        'viirs_clear_sky_pct': daily_agg['pct_clear_sky'],
        'viirs_coverage_count': daily_agg['tile_count'],
        # Additional features that may be useful
        'viirs_radiance_p50': daily_agg['radiance_p50'],
        'viirs_radiance_p90': daily_agg['radiance_p90'],
        'viirs_moon_illumination': daily_agg['moon_illumination_pct'],
    })

    # Create complete daily date range
    if start_date is None:
        start_date = feature_df['date'].min()
    if end_date is None:
        end_date = feature_df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to complete date range - DO NOT FILL MISSING VALUES
    feature_df = feature_df.set_index('date')
    feature_df = feature_df.reindex(date_range)
    feature_df.index.name = 'date'
    feature_df = feature_df.reset_index()

    # Create observation mask: True where we have real observations
    feature_cols = [c for c in feature_df.columns if c != 'date']
    observation_mask = feature_df[feature_cols].notna().any(axis=1).values

    # Apply detrending if requested (first-order differencing)
    if detrend:
        # Detrending removes shared temporal trends that cause spurious correlations
        # VIIRS lags casualties by ~10 days, so correlation is trend-driven, not predictive
        detrended_df = feature_df.copy()

        # Apply first-order differencing to all numeric features
        for col in feature_cols:
            # viirs_detrended[t] = viirs[t] - viirs[t-1]
            detrended_df[col] = feature_df[col].diff()

        # Update observation mask: require BOTH current AND previous observation
        # A detrended value is only valid if both t and t-1 had valid observations
        prev_mask = np.roll(observation_mask, 1)
        prev_mask[0] = False  # First element has no previous value
        detrended_mask = observation_mask & prev_mask

        # First row will always be NaN after diff(), so mask it out
        detrended_mask[0] = False

        feature_df = detrended_df
        observation_mask = detrended_mask

    return feature_df, observation_mask


def load_viina_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load VIINA conflict event data at daily resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    viina_dir = DATA_DIR / "viina" / "extracted"

    if not viina_dir.exists():
        warnings.warn(f"VIINA data not found at {viina_dir}")
        return pd.DataFrame(), np.array([])

    # Load event info files for all years
    all_events = []
    for year in range(2022, 2027):
        event_file = viina_dir / f"event_info_latest_{year}.csv"
        if event_file.exists():
            try:
                df = pd.read_csv(event_file, low_memory=False)
                all_events.append(df)
            except Exception:
                continue

    if not all_events:
        return pd.DataFrame(), np.array([])

    events_df = pd.concat(all_events, ignore_index=True)

    # Parse date (YYYYMMDD format)
    events_df['date'] = pd.to_datetime(events_df['date'].astype(str), format='%Y%m%d', errors='coerce')
    events_df = events_df.dropna(subset=['date'])

    # Aggregate by day
    daily_agg = events_df.groupby(events_df['date'].dt.date).agg({
        'event_id': 'count',  # Total events
        'GEO_PRECISION': lambda x: (x == 'ADM3').sum(),  # High precision count
    }).reset_index()

    daily_agg.columns = ['date', 'event_count', 'high_precision_events']
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])

    # Also aggregate by region if available
    if 'ADM1_NAME' in events_df.columns:
        region_counts = events_df.groupby([events_df['date'].dt.date, 'ADM1_NAME']).size().unstack(fill_value=0)
        region_counts = region_counts.reset_index()
        region_counts['date'] = pd.to_datetime(region_counts['date'])
        # Add top regions as features
        top_regions = ['Kharkiv', 'Donets\'k', 'Zaporizhzhia', 'Kherson', 'Luhans\'k']
        for region in top_regions:
            col_name = f'events_{region.lower().replace("\'", "")}'
            if region in region_counts.columns:
                daily_agg = daily_agg.merge(
                    region_counts[['date', region]].rename(columns={region: col_name}),
                    on='date', how='left'
                )

    # Create complete daily date range
    if start_date is None:
        start_date = daily_agg['date'].min()
    if end_date is None:
        end_date = daily_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    daily_agg = daily_agg.set_index('date')
    daily_agg = daily_agg.reindex(date_range)
    daily_agg.index.name = 'date'
    daily_agg = daily_agg.reset_index()

    # Create observation mask
    feature_cols = [c for c in daily_agg.columns if c != 'date']
    observation_mask = daily_agg[feature_cols].notna().any(axis=1).values

    features_df = daily_agg.copy()

    return features_df, observation_mask


def load_sentinel_monthly(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Sentinel satellite data at monthly resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    merged_path = ANALYSIS_DIR / "sentinel_osint_merged.csv"

    if not merged_path.exists():
        warnings.warn(f"Sentinel merged data not found at {merged_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(merged_path)
    df['date'] = pd.to_datetime(df['date'])

    # Feature columns (Sentinel-specific)
    feature_cols = ['s1_radar', 's2_optical', 's3_fire', 's5p_co', 's5p_no2',
                    's2_avg_cloud', 's2_cloud_free']
    available_cols = [c for c in feature_cols if c in df.columns]

    # Create complete monthly date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    observation_mask = df[available_cols].notna().any(axis=1).values

    features_df = df[['date'] + available_cols].copy()

    return features_df, observation_mask


def load_hdx_conflict_monthly(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load HDX conflict events data at monthly resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    conflict_path = DATA_DIR / "hdx" / "ukraine" / "conflict_events_2022_present.csv"

    if not conflict_path.exists():
        warnings.warn(f"HDX conflict data not found at {conflict_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(conflict_path)

    # Parse dates
    df['period_start'] = pd.to_datetime(df['reference_period_start'], errors='coerce')
    df['period_end'] = pd.to_datetime(df['reference_period_end'], errors='coerce')
    df = df.dropna(subset=['period_start'])

    # Create month column
    df['month'] = df['period_start'].dt.to_period('M').dt.to_timestamp()

    # Aggregate by month and event type
    monthly_agg = df.groupby('month').agg({
        'events': 'sum',
        'fatalities': 'sum'
    }).reset_index()

    # Also get event type breakdown
    event_type_agg = df.pivot_table(
        index='month',
        columns='event_type',
        values='events',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Merge
    monthly_agg = monthly_agg.merge(event_type_agg, on='month', how='left')
    monthly_agg = monthly_agg.rename(columns={'month': 'date'})

    # Create complete monthly date range
    if start_date is None:
        start_date = monthly_agg['date'].min()
    if end_date is None:
        end_date = monthly_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex without filling
    monthly_agg = monthly_agg.set_index('date')
    monthly_agg = monthly_agg.reindex(date_range)
    monthly_agg.index.name = 'date'
    monthly_agg = monthly_agg.reset_index()

    # Create observation mask
    feature_cols = [c for c in monthly_agg.columns if c != 'date']
    observation_mask = monthly_agg[feature_cols].notna().any(axis=1).values

    features_df = monthly_agg.copy()

    return features_df, observation_mask


def load_hdx_food_monthly(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load HDX food prices data at monthly resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    food_path = DATA_DIR / "hdx" / "ukraine" / "food_prices_2022_present.csv"

    if not food_path.exists():
        warnings.warn(f"HDX food prices data not found at {food_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(food_path)

    # Parse dates
    df['period_start'] = pd.to_datetime(df['reference_period_start'], errors='coerce')
    df = df.dropna(subset=['period_start', 'price'])

    # Create month column
    df['month'] = df['period_start'].dt.to_period('M').dt.to_timestamp()

    # Aggregate by month - compute price statistics across all commodities
    monthly_agg = df.groupby('month').agg({
        'price': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()

    # Flatten columns
    monthly_agg.columns = ['date', 'price_mean', 'price_std', 'price_min', 'price_max', 'price_count']

    # Also compute price indices for key commodities
    key_commodities = ['Bread', 'Wheat flour', 'Potatoes', 'Oil', 'Milk']
    for commodity in key_commodities:
        commodity_df = df[df['commodity_name'].str.contains(commodity, case=False, na=False)]
        if not commodity_df.empty:
            commodity_agg = commodity_df.groupby('month')['price'].mean().reset_index()
            commodity_agg.columns = ['date', f'price_{commodity.lower().replace(" ", "_")}']
            monthly_agg = monthly_agg.merge(commodity_agg, on='date', how='left')

    # Create complete monthly date range
    if start_date is None:
        start_date = monthly_agg['date'].min()
    if end_date is None:
        end_date = monthly_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex without filling
    monthly_agg = monthly_agg.set_index('date')
    monthly_agg = monthly_agg.reindex(date_range)
    monthly_agg.index.name = 'date'
    monthly_agg = monthly_agg.reset_index()

    # Create observation mask
    feature_cols = [c for c in monthly_agg.columns if c != 'date']
    observation_mask = monthly_agg[feature_cols].notna().any(axis=1).values

    features_df = monthly_agg.copy()

    return features_df, observation_mask


def load_hdx_rainfall_monthly(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load HDX rainfall data at monthly resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    rainfall_path = DATA_DIR / "hdx" / "ukraine" / "rainfall_2022_present.csv"

    if not rainfall_path.exists():
        warnings.warn(f"HDX rainfall data not found at {rainfall_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(rainfall_path)

    # Parse dates
    df['period_start'] = pd.to_datetime(df['reference_period_start'], errors='coerce')
    df = df.dropna(subset=['period_start'])

    # Create month column
    df['month'] = df['period_start'].dt.to_period('M').dt.to_timestamp()

    # Aggregate by month
    monthly_agg = df.groupby('month').agg({
        'rainfall': ['mean', 'std', 'min', 'max'],
        'rainfall_long_term_average': 'mean',
        'rainfall_anomaly_pct': 'mean'
    }).reset_index()

    # Flatten columns
    monthly_agg.columns = [
        'date', 'rainfall_mean', 'rainfall_std', 'rainfall_min', 'rainfall_max',
        'rainfall_lta', 'rainfall_anomaly'
    ]

    # Create complete monthly date range
    if start_date is None:
        start_date = monthly_agg['date'].min()
    if end_date is None:
        end_date = monthly_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex without filling
    monthly_agg = monthly_agg.set_index('date')
    monthly_agg = monthly_agg.reindex(date_range)
    monthly_agg.index.name = 'date'
    monthly_agg = monthly_agg.reset_index()

    # Create observation mask
    feature_cols = [c for c in monthly_agg.columns if c != 'date']
    observation_mask = monthly_agg[feature_cols].notna().any(axis=1).values

    features_df = monthly_agg.copy()

    return features_df, observation_mask


def load_iom_monthly(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load IOM displacement data at monthly resolution.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    iom_path = DATA_DIR / "iom" / "ukr-iom-dtm-from-api-admin-0-to-1.csv"

    if not iom_path.exists():
        warnings.warn(f"IOM data not found at {iom_path}")
        return pd.DataFrame(), np.array([])

    # Skip the second header row (HXL tags)
    df = pd.read_csv(iom_path, skiprows=[1])

    # Parse dates
    df['date'] = pd.to_datetime(df['reportingDate'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Create month column
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Aggregate by month
    monthly_agg = df.groupby('month').agg({
        'numPresentIdpInd': ['sum', 'mean', 'max'],
        'numberMales': 'sum',
        'numberFemales': 'sum',
        'roundNumber': 'max'
    }).reset_index()

    # Flatten columns
    monthly_agg.columns = [
        'date', 'idp_total', 'idp_mean', 'idp_max',
        'idp_males', 'idp_females', 'round_number'
    ]

    # Calculate gender ratio
    monthly_agg['idp_female_ratio'] = monthly_agg['idp_females'] / (
        monthly_agg['idp_males'] + monthly_agg['idp_females'] + 1e-8
    )

    # Create complete monthly date range
    if start_date is None:
        start_date = monthly_agg['date'].min()
    if end_date is None:
        end_date = monthly_agg['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Reindex without filling
    monthly_agg = monthly_agg.set_index('date')
    monthly_agg = monthly_agg.reindex(date_range)
    monthly_agg.index.name = 'date'
    monthly_agg = monthly_agg.reset_index()

    # Create observation mask
    feature_cols = [c for c in monthly_agg.columns if c != 'date']
    observation_mask = monthly_agg[feature_cols].notna().any(axis=1).values

    features_df = monthly_agg.copy()

    return features_df, observation_mask


# =============================================================================
# DISAGGREGATED EQUIPMENT LOADERS
# =============================================================================

def load_drones_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load drone/UAV loss data at daily resolution.

    This source shows the HIGHEST signal in probe analysis:
    - r=0.289 with personnel losses
    - MI=0.47 nats (highest mutual information)
    - Drones lead casualties by 7 days (r=0.32 at optimal lag)
    - Partial correlation survives time control better than other equipment

    Features:
        - drone: Total UAV/drone losses (cumulative)
        - drone_daily: Daily change in drone losses
        - cruise_missiles: Cruise missile losses (cumulative)
        - cruise_missiles_daily: Daily change in cruise missile losses

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        warnings.warn(f"Equipment data not found at {equip_path}")
        return pd.DataFrame(), np.array([])

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Normalize column names (handle spaces and hyphens)
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')

    # Define drone-related feature columns
    feature_cols = ['drone', 'cruise_missiles']

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to complete date range - DO NOT FILL MISSING VALUES
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask: True where we have real observations
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Calculate daily changes where we have consecutive observations
    for col in available_cols:
        if col in df.columns:
            df[f'{col}_daily'] = df[col].diff()

    # Select final features
    final_cols = []
    for col in available_cols:
        final_cols.append(col)
        if f'{col}_daily' in df.columns:
            final_cols.append(f'{col}_daily')

    features_df = df[['date'] + final_cols].copy()

    return features_df, observation_mask


def load_armor_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load armored vehicle loss data at daily resolution.

    Features:
        - tank: Tank losses (cumulative)
        - tank_daily: Daily change in tank losses
        - APC: Armored Personnel Carrier / Infantry Fighting Vehicle losses (cumulative)
        - APC_daily: Daily change in APC losses

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        warnings.warn(f"Equipment data not found at {equip_path}")
        return pd.DataFrame(), np.array([])

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Normalize column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')

    # Define armor-related feature columns
    feature_cols = ['tank', 'APC']

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Calculate daily changes
    for col in available_cols:
        if col in df.columns:
            df[f'{col}_daily'] = df[col].diff()

    # Select final features
    final_cols = []
    for col in available_cols:
        final_cols.append(col)
        if f'{col}_daily' in df.columns:
            final_cols.append(f'{col}_daily')

    features_df = df[['date'] + final_cols].copy()

    return features_df, observation_mask


def load_artillery_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load artillery loss data at daily resolution.

    Features:
        - field_artillery: Field artillery losses (cumulative)
        - field_artillery_daily: Daily change in field artillery losses
        - MRL: Multiple Rocket Launcher losses (cumulative)
        - MRL_daily: Daily change in MRL losses
        - anti_aircraft_warfare: Anti-aircraft system losses (cumulative)
        - anti_aircraft_warfare_daily: Daily change in AA losses
        - mobile_SRBM_system: Short-range ballistic missile system losses (cumulative)
        - mobile_SRBM_system_daily: Daily change in SRBM losses

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        warnings.warn(f"Equipment data not found at {equip_path}")
        return pd.DataFrame(), np.array([])

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Normalize column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')

    # Define artillery-related feature columns
    feature_cols = ['field_artillery', 'MRL', 'anti_aircraft_warfare', 'mobile_SRBM_system']

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Calculate daily changes
    for col in available_cols:
        if col in df.columns:
            df[f'{col}_daily'] = df[col].diff()

    # Select final features
    final_cols = []
    for col in available_cols:
        final_cols.append(col)
        if f'{col}_daily' in df.columns:
            final_cols.append(f'{col}_daily')

    features_df = df[['date'] + final_cols].copy()

    return features_df, observation_mask


def load_aircraft_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load aircraft loss data at daily resolution.

    Features:
        - aircraft: Fixed-wing aircraft losses (cumulative)
        - aircraft_daily: Daily change in aircraft losses
        - helicopter: Helicopter losses (cumulative)
        - helicopter_daily: Daily change in helicopter losses

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with features, observation mask array)
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        warnings.warn(f"Equipment data not found at {equip_path}")
        return pd.DataFrame(), np.array([])

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Normalize column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')

    # Define aircraft-related feature columns
    feature_cols = ['aircraft', 'helicopter']

    # Create complete daily date range
    if start_date is None:
        start_date = df['date'].min()
    if end_date is None:
        end_date = df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    df = df.set_index('date')
    df = df.reindex(date_range)
    df.index.name = 'date'
    df = df.reset_index()

    # Create observation mask
    available_cols = [c for c in feature_cols if c in df.columns]
    observation_mask = df[available_cols].notna().any(axis=1).values

    # Calculate daily changes
    for col in available_cols:
        if col in df.columns:
            df[f'{col}_daily'] = df[col].diff()

    # Select final features
    final_cols = []
    for col in available_cols:
        final_cols.append(col)
        if f'{col}_daily' in df.columns:
            final_cols.append(f'{col}_daily')

    features_df = df[['date'] + final_cols].copy()

    return features_df, observation_mask


# =============================================================================
# LOADER REGISTRY
# =============================================================================

LOADER_REGISTRY: Dict[str, callable] = {
    # Daily sources - aggregated equipment (legacy)
    "equipment": load_equipment_daily,
    # Daily sources - disaggregated equipment (NEW - from probe analysis)
    # Drones show HIGHEST signal: r=0.289 with personnel, MI=0.47 nats
    # 7-day lead correlation with casualties (r=0.32)
    "drones": load_drones_daily,
    "armor": load_armor_daily,
    "artillery": load_artillery_daily,
    "aircraft": load_aircraft_daily,
    # Daily sources - other
    "personnel": load_personnel_daily,
    "deepstate": load_deepstate_daily,
    "firms": load_firms_daily,
    "viina": load_viina_daily,
    "viirs": load_viirs_daily,
    # Monthly sources
    "sentinel": load_sentinel_monthly,
    "hdx_conflict": load_hdx_conflict_monthly,
    "hdx_food": load_hdx_food_monthly,
    "hdx_rainfall": load_hdx_rainfall_monthly,
    "iom": load_iom_monthly,
}

SOURCE_RESOLUTIONS: Dict[str, Resolution] = {
    # Daily sources - aggregated equipment (legacy)
    "equipment": Resolution.DAILY,
    # Daily sources - disaggregated equipment (NEW)
    "drones": Resolution.DAILY,
    "armor": Resolution.DAILY,
    "artillery": Resolution.DAILY,
    "aircraft": Resolution.DAILY,
    # Daily sources - other
    "personnel": Resolution.DAILY,
    "deepstate": Resolution.DAILY,
    "firms": Resolution.DAILY,
    "viina": Resolution.DAILY,
    "viirs": Resolution.DAILY,
    # Monthly sources
    "sentinel": Resolution.MONTHLY,
    "hdx_conflict": Resolution.MONTHLY,
    "hdx_food": Resolution.MONTHLY,
    "hdx_rainfall": Resolution.MONTHLY,
    "iom": Resolution.MONTHLY,
}


# =============================================================================
# TEMPORAL ALIGNMENT UTILITIES
# =============================================================================

@dataclass
class TemporalAlignment:
    """Container for temporal alignment information."""
    # Daily timeline
    daily_dates: np.ndarray  # Array of daily timestamps
    n_daily: int

    # Monthly timeline
    monthly_dates: np.ndarray  # Array of monthly timestamps (first of month)
    n_monthly: int

    # Month boundary indices for daily data
    # month_boundaries[i] = (start_idx, end_idx) for month i in daily timeline
    month_boundaries: List[Tuple[int, int]]

    # Mapping from daily index to monthly index
    daily_to_monthly: np.ndarray  # Shape: [n_daily], values are monthly indices

    def get_daily_indices_for_month(self, month_idx: int) -> Tuple[int, int]:
        """Get the daily index range for a given month."""
        if month_idx < 0 or month_idx >= len(self.month_boundaries):
            raise IndexError(f"Month index {month_idx} out of range")
        return self.month_boundaries[month_idx]

    def get_month_for_daily_idx(self, daily_idx: int) -> int:
        """Get the monthly index for a given daily index."""
        if daily_idx < 0 or daily_idx >= self.n_daily:
            raise IndexError(f"Daily index {daily_idx} out of range")
        return self.daily_to_monthly[daily_idx]


def compute_temporal_alignment(
    start_date: datetime,
    end_date: datetime
) -> TemporalAlignment:
    """
    Compute temporal alignment between daily and monthly timelines.

    Args:
        start_date: Start of the time range
        end_date: End of the time range

    Returns:
        TemporalAlignment object with all alignment information
    """
    # Create daily date range
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_daily = len(daily_dates)

    # Create monthly date range (first of each month)
    monthly_dates = pd.date_range(
        start=start_date.replace(day=1),
        end=end_date,
        freq='MS'
    )
    n_monthly = len(monthly_dates)

    # Compute month boundaries in daily timeline
    month_boundaries = []
    daily_to_monthly = np.zeros(n_daily, dtype=np.int64)

    for month_idx, month_start in enumerate(monthly_dates):
        # Find all daily indices belonging to this month
        month_mask = (daily_dates.year == month_start.year) & (daily_dates.month == month_start.month)
        month_daily_indices = np.where(month_mask)[0]

        if len(month_daily_indices) > 0:
            start_idx = month_daily_indices[0]
            end_idx = month_daily_indices[-1] + 1  # Exclusive end
            month_boundaries.append((start_idx, end_idx))
            daily_to_monthly[month_daily_indices] = month_idx
        else:
            # Empty month (shouldn't happen with proper date ranges)
            month_boundaries.append((0, 0))

    return TemporalAlignment(
        daily_dates=daily_dates.values,
        n_daily=n_daily,
        monthly_dates=monthly_dates.values,
        n_monthly=n_monthly,
        month_boundaries=month_boundaries,
        daily_to_monthly=daily_to_monthly
    )


# =============================================================================
# MULTI-RESOLUTION DATASET
# =============================================================================

@dataclass
class MultiResolutionSample:
    """A single sample from the multi-resolution dataset."""
    # Daily data: Dict[source_name, Tensor[daily_seq_len, n_features]]
    daily_features: Dict[str, torch.Tensor]
    # Daily masks: Dict[source_name, Tensor[daily_seq_len, n_features]]
    daily_masks: Dict[str, torch.Tensor]

    # Monthly data: Dict[source_name, Tensor[monthly_seq_len, n_features]]
    monthly_features: Dict[str, torch.Tensor]
    # Monthly masks: Dict[source_name, Tensor[monthly_seq_len, n_features]]
    monthly_masks: Dict[str, torch.Tensor]

    # Temporal alignment for this sample
    month_boundary_indices: torch.Tensor  # [n_months, 2] start/end indices in daily seq

    # Metadata
    daily_dates: np.ndarray
    monthly_dates: np.ndarray
    sample_idx: int


class MultiResolutionDataset(Dataset):
    """
    PyTorch Dataset for multi-resolution time series with observation masks.

    This dataset handles:
    - Daily resolution sources (~1000+ timesteps)
    - Monthly resolution sources (~35 timesteps)
    - Explicit observation masks (True = observed, False = missing)
    - NO forward-filling or interpolation
    - Proper temporal alignment between resolutions

    Attributes:
        config: MultiResolutionConfig with dataset parameters
        daily_data: Dict mapping source name to (features_df, mask_array) for daily sources
        monthly_data: Dict mapping source name to (features_df, mask_array) for monthly sources
        alignment: TemporalAlignment object for daily/monthly conversion
        norm_stats: Normalization statistics (computed from training data)
    """

    def __init__(
        self,
        config: Optional[MultiResolutionConfig] = None,
        split: str = 'train',
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        seed: int = 42
    ):
        """
        Initialize the multi-resolution dataset.

        Args:
            config: Dataset configuration (uses defaults if None)
            split: One of 'train', 'val', 'test'
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            norm_stats: Pre-computed normalization stats (required for val/test)
            seed: Random seed for reproducibility
        """
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.config = config or MultiResolutionConfig()
        self.split = split
        self.seed = seed

        print(f"Initializing MultiResolutionDataset (split={split})...")

        # Load all data sources
        self._load_all_sources()

        # Compute temporal alignment
        self._compute_alignment()

        # Compute split indices
        self._compute_splits(val_ratio, test_ratio)

        # Handle normalization
        if split == 'train':
            self.norm_stats = self._compute_normalization_stats()
        else:
            if norm_stats is None:
                raise ValueError(
                    f"norm_stats must be provided for {split} split to prevent data leakage"
                )
            self.norm_stats = norm_stats

        # Apply normalization
        self._apply_normalization()

        # Convert to tensors
        self._convert_to_tensors()

        print(f"Dataset initialized: {len(self)} samples in {split} split")

    def _load_all_sources(self) -> None:
        """Load all configured data sources."""
        self.daily_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
        self.monthly_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}

        # Determine date range from all sources
        all_dates = []

        # Load daily sources (using effective sources for disaggregation support)
        effective_daily_sources = self.config.get_effective_daily_sources()
        for source_name in effective_daily_sources:

            if source_name not in LOADER_REGISTRY:
                warnings.warn(f"Unknown daily source: {source_name}")
                continue

            print(f"  Loading daily source: {source_name}...")
            loader_func = LOADER_REGISTRY[source_name]

            # Handle VIIRS detrending option
            if source_name == "viirs" and self.config.detrend_viirs:
                print(f"    Applying detrending to VIIRS data (first-order differencing)")
                df, mask = loader_func(detrend=True)
            else:
                df, mask = loader_func()

            if df.empty:
                warnings.warn(f"Empty data for source: {source_name}")
                continue

            self.daily_data[source_name] = (df, mask)
            all_dates.extend(df['date'].dropna().tolist())

        # Load monthly sources
        for source_name in self.config.monthly_sources:
            if source_name not in LOADER_REGISTRY:
                warnings.warn(f"Unknown monthly source: {source_name}")
                continue

            print(f"  Loading monthly source: {source_name}...")
            loader_func = LOADER_REGISTRY[source_name]
            df, mask = loader_func()

            if df.empty:
                warnings.warn(f"Empty data for source: {source_name}")
                continue

            self.monthly_data[source_name] = (df, mask)
            all_dates.extend(df['date'].dropna().tolist())

        if not all_dates:
            raise ValueError("No data loaded from any source")

        # Determine common date range
        self.start_date = max(pd.Timestamp(self.config.start_date), min(all_dates))
        if self.config.end_date:
            self.end_date = min(pd.Timestamp(self.config.end_date), max(all_dates))
        else:
            self.end_date = max(all_dates)

        print(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")

    def _compute_alignment(self) -> None:
        """Compute temporal alignment between daily and monthly data."""
        self.alignment = compute_temporal_alignment(self.start_date, self.end_date)

        print(f"  Daily timesteps: {self.alignment.n_daily}")
        print(f"  Monthly timesteps: {self.alignment.n_monthly}")
        print(f"  Month boundaries computed: {len(self.alignment.month_boundaries)}")

        # Realign all data sources to the common date range
        self._realign_sources()

    def _realign_sources(self) -> None:
        """Realign all data sources to the common date range."""
        daily_date_range = pd.DatetimeIndex(self.alignment.daily_dates)
        monthly_date_range = pd.DatetimeIndex(self.alignment.monthly_dates)

        # Realign daily sources
        realigned_daily = {}
        for source_name, (df, mask) in self.daily_data.items():
            # Reindex to common daily range
            df_aligned = df.set_index('date').reindex(daily_date_range)
            df_aligned.index.name = 'date'
            df_aligned = df_aligned.reset_index()

            # Create new mask based on non-null values
            feature_cols = [c for c in df_aligned.columns if c != 'date']
            new_mask = df_aligned[feature_cols].notna().any(axis=1).values

            realigned_daily[source_name] = (df_aligned, new_mask)

        self.daily_data = realigned_daily

        # Realign monthly sources
        realigned_monthly = {}
        for source_name, (df, mask) in self.monthly_data.items():
            # Reindex to common monthly range
            df_aligned = df.set_index('date').reindex(monthly_date_range)
            df_aligned.index.name = 'date'
            df_aligned = df_aligned.reset_index()

            # Create new mask based on non-null values
            feature_cols = [c for c in df_aligned.columns if c != 'date']
            new_mask = df_aligned[feature_cols].notna().any(axis=1).values

            realigned_monthly[source_name] = (df_aligned, new_mask)

        self.monthly_data = realigned_monthly

        print(f"  Sources realigned to common date range")

    def _compute_splits(self, val_ratio: float, test_ratio: float) -> None:
        """Compute train/val/test split indices based on DAILY timeline.

        This enables daily-granularity sampling, providing ~30x more training samples
        compared to monthly-granularity sampling.
        """
        n_days = self.alignment.n_daily
        n_months = self.alignment.n_monthly

        # Calculate split sizes based on DAYS (not months)
        n_test_days = max(1, int(n_days * test_ratio))
        n_val_days = max(1, int(n_days * val_ratio))
        n_train_days = n_days - n_test_days - n_val_days

        # Temporal splits (no shuffling - preserve time order)
        train_end_day = n_train_days
        val_end_day = train_end_day + n_val_days

        # Log detailed split info for debugging
        print(f"  Dataset has {n_days} total days ({n_months} months)")
        print(f"  Temporal splits (DAILY): Train days 0-{train_end_day-1}, "
              f"Val {train_end_day}-{val_end_day-1}, Test {val_end_day}-{n_days-1}")

        # Store daily index ranges for each split
        self.train_day_indices = np.arange(0, train_end_day)
        self.val_day_indices = np.arange(train_end_day, val_end_day)
        self.test_day_indices = np.arange(val_end_day, n_days)

        # Set current split's daily indices
        if self.split == 'train':
            self.day_indices = self.train_day_indices
        elif self.split == 'val':
            self.day_indices = self.val_day_indices
        else:
            self.day_indices = self.test_day_indices

        # Also compute month indices for backward compatibility with normalization
        # Map daily boundaries to months
        self.train_month_indices = self._days_to_months(self.train_day_indices)
        self.val_month_indices = self._days_to_months(self.val_day_indices)
        self.test_month_indices = self._days_to_months(self.test_day_indices)

        # Valid sample indices (where we have enough history and horizon)
        self._compute_valid_samples()

        print(f"  Split sizes - Train: {len(self.train_day_indices)} days, "
              f"Val: {len(self.val_day_indices)} days, Test: {len(self.test_day_indices)} days")

    def _days_to_months(self, day_indices: np.ndarray) -> np.ndarray:
        """Convert day indices to unique month indices for normalization."""
        if len(day_indices) == 0:
            return np.array([], dtype=int)

        month_indices = set()
        for day_idx in day_indices:
            # Find which month this day belongs to
            for month_idx, (start, end) in enumerate(self.alignment.month_boundaries):
                if start <= day_idx < end:
                    month_indices.add(month_idx)
                    break
        return np.array(sorted(month_indices), dtype=int)

    def _compute_valid_samples(self) -> None:
        """Compute valid sample indices at DAILY granularity.

        Each sample is anchored to a specific day. We check that:
        1. We have enough daily history (daily_seq_len days before)
        2. We have enough monthly history (monthly_seq_len months of context)
        3. We have enough horizon for prediction targets

        For validation and test sets, applies a stride to reduce temporal overlap.
        """
        all_valid = []

        for day_idx in self.day_indices:
            # Check if we have enough daily history
            daily_start = day_idx - self.config.daily_seq_len + 1
            if daily_start < 0:
                continue

            # Find which month this day belongs to
            target_month_idx = None
            for month_idx, (start, end) in enumerate(self.alignment.month_boundaries):
                if start <= day_idx < end:
                    target_month_idx = month_idx
                    break

            if target_month_idx is None:
                continue

            # Check if we have enough monthly history
            monthly_start = target_month_idx - self.config.monthly_seq_len + 1
            monthly_end = target_month_idx + self.config.prediction_horizon

            if monthly_start < 0 or monthly_end > self.alignment.n_monthly:
                continue

            # This day has enough history - it's a valid sample
            all_valid.append(day_idx)

        # Apply stride for val/test to reduce overlap between samples
        if self.split in ('val', 'test') and len(all_valid) > 1:
            # Auto-compute stride if not specified
            stride = self.config.val_sample_stride
            if stride is None:
                # For daily sampling, use a moderate stride
                # Target: meaningful validation coverage without extreme overlap
                # We want at least 10-20 samples if possible

                # Start with stride that gives ~50% overlap
                target_stride = max(1, self.config.daily_seq_len // 2)

                # But ensure we have at least 10 samples
                n_potential = len(all_valid)
                min_samples = min(10, n_potential)

                # Find the largest stride that still gives us min_samples
                for test_stride in range(target_stride, 0, -1):
                    n_samples = len(all_valid[::test_stride])
                    if n_samples >= min_samples:
                        stride = test_stride
                        break
                else:
                    stride = 1  # No stride - use all samples

            # Take every stride-th sample
            strided_valid = all_valid[::stride]

            # Log the impact
            actual_overlap = max(0, self.config.daily_seq_len - stride) / self.config.daily_seq_len * 100
            print(f"  {self.split} samples: {len(all_valid)} total, stride={stride} "
                  f"-> {len(strided_valid)} samples ({actual_overlap:.0f}% overlap)")

            self.valid_indices = np.array(strided_valid)
        else:
            self.valid_indices = np.array(all_valid)

        print(f"  Valid samples in {self.split}: {len(self.valid_indices)}")

    def _compute_normalization_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute normalization statistics from training data only."""
        stats = {}

        # Get training month indices for computing stats
        train_months = self.train_month_indices

        # Stats for daily sources
        for source_name, (df, mask) in self.daily_data.items():
            feature_cols = [c for c in df.columns if c != 'date']

            # Get daily indices corresponding to training months
            train_daily_mask = np.zeros(len(df), dtype=bool)
            for month_idx in train_months:
                if month_idx < len(self.alignment.month_boundaries):
                    start, end = self.alignment.month_boundaries[month_idx]
                    if start < len(train_daily_mask) and end <= len(train_daily_mask):
                        train_daily_mask[start:end] = True

            # Compute stats only on observed training data
            train_data = df.loc[train_daily_mask & mask, feature_cols].values

            if len(train_data) > 0:
                mean = np.nanmean(train_data, axis=0)
                std = np.nanstd(train_data, axis=0)
                std = np.maximum(std, 0.1)  # Prevent division by zero
                stats[source_name] = {'mean': mean, 'std': std, 'type': 'standard'}
            else:
                # No training data - use defaults
                n_features = len(feature_cols)
                stats[source_name] = {
                    'mean': np.zeros(n_features),
                    'std': np.ones(n_features),
                    'type': 'standard'
                }

        # Stats for monthly sources
        for source_name, (df, mask) in self.monthly_data.items():
            feature_cols = [c for c in df.columns if c != 'date']

            # Get training month data
            train_mask = np.zeros(len(df), dtype=bool)
            train_mask[:len(train_months)] = True
            train_mask = train_mask & mask

            train_data = df.loc[train_mask, feature_cols].values

            if len(train_data) > 0:
                mean = np.nanmean(train_data, axis=0)
                std = np.nanstd(train_data, axis=0)
                std = np.maximum(std, 0.1)
                stats[source_name] = {'mean': mean, 'std': std, 'type': 'standard'}
            else:
                n_features = len(feature_cols)
                stats[source_name] = {
                    'mean': np.zeros(n_features),
                    'std': np.ones(n_features),
                    'type': 'standard'
                }

        return stats

    def _apply_normalization(self) -> None:
        """Apply normalization to all data using pre-computed stats."""
        # Normalize daily sources
        for source_name, (df, mask) in self.daily_data.items():
            if source_name not in self.norm_stats:
                continue

            feature_cols = [c for c in df.columns if c != 'date']
            stats = self.norm_stats[source_name]

            # Create normalized copy
            df_norm = df.copy()
            values = df_norm[feature_cols].values.astype(np.float32)

            # Normalize observed values only
            mean = stats['mean']
            std = stats['std']

            # Apply normalization
            normalized = (values - mean) / std

            # Replace NaN with missing value sentinel
            normalized = np.where(np.isnan(normalized), self.config.missing_value, normalized)

            df_norm[feature_cols] = normalized
            self.daily_data[source_name] = (df_norm, mask)

        # Normalize monthly sources
        for source_name, (df, mask) in self.monthly_data.items():
            if source_name not in self.norm_stats:
                continue

            feature_cols = [c for c in df.columns if c != 'date']
            stats = self.norm_stats[source_name]

            df_norm = df.copy()
            values = df_norm[feature_cols].values.astype(np.float32)

            mean = stats['mean']
            std = stats['std']

            normalized = (values - mean) / std
            normalized = np.where(np.isnan(normalized), self.config.missing_value, normalized)

            df_norm[feature_cols] = normalized
            self.monthly_data[source_name] = (df_norm, mask)

    def _convert_to_tensors(self) -> None:
        """Convert DataFrames to tensors for efficient access."""
        self.daily_tensors: Dict[str, torch.Tensor] = {}
        self.daily_mask_tensors: Dict[str, torch.Tensor] = {}

        for source_name, (df, mask) in self.daily_data.items():
            feature_cols = [c for c in df.columns if c != 'date']
            self.daily_tensors[source_name] = torch.tensor(
                df[feature_cols].values, dtype=torch.float32
            )
            # Create feature-level mask (expand observation mask to all features)
            expanded_mask = np.tile(mask[:, np.newaxis], (1, len(feature_cols)))
            # Also mark missing values in the data
            data_mask = df[feature_cols].values != self.config.missing_value
            combined_mask = expanded_mask & data_mask
            self.daily_mask_tensors[source_name] = torch.tensor(
                combined_mask, dtype=torch.bool
            )

        self.monthly_tensors: Dict[str, torch.Tensor] = {}
        self.monthly_mask_tensors: Dict[str, torch.Tensor] = {}

        for source_name, (df, mask) in self.monthly_data.items():
            feature_cols = [c for c in df.columns if c != 'date']
            self.monthly_tensors[source_name] = torch.tensor(
                df[feature_cols].values, dtype=torch.float32
            )
            expanded_mask = np.tile(mask[:, np.newaxis], (1, len(feature_cols)))
            data_mask = df[feature_cols].values != self.config.missing_value
            combined_mask = expanded_mask & data_mask
            self.monthly_mask_tensors[source_name] = torch.tensor(
                combined_mask, dtype=torch.bool
            )

    def __len__(self) -> int:
        """Return number of valid samples in this split."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> MultiResolutionSample:
        """
        Get a sample at the given index (DAILY granularity).

        Each sample is anchored to a specific day. The sample includes:
        - daily_seq_len days of daily features ending at the anchor day
        - monthly_seq_len months of monthly features ending at the anchor day's month

        Args:
            idx: Sample index

        Returns:
            MultiResolutionSample with all data and masks
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Get the target DAY index (not month)
        target_day_idx = self.valid_indices[idx]

        # Compute daily sequence range
        daily_end = target_day_idx + 1  # Exclusive - include the target day
        daily_start = max(0, daily_end - self.config.daily_seq_len)

        # Get daily features and masks
        daily_features = {}
        daily_masks = {}

        for source_name in self.daily_tensors:
            daily_features[source_name] = self.daily_tensors[source_name][
                daily_start:daily_end
            ]
            daily_masks[source_name] = self.daily_mask_tensors[source_name][
                daily_start:daily_end
            ]

        # Find which month the target day belongs to
        target_month_idx = None
        for month_idx, (start, end) in enumerate(self.alignment.month_boundaries):
            if start <= target_day_idx < end:
                target_month_idx = month_idx
                break

        if target_month_idx is None:
            raise ValueError(f"Day {target_day_idx} not found in any month boundary")

        # Compute monthly sequence range
        monthly_start = target_month_idx - self.config.monthly_seq_len + 1
        monthly_end = target_month_idx + 1  # Exclusive

        # Get monthly features and masks
        monthly_features = {}
        monthly_masks = {}

        for source_name in self.monthly_tensors:
            monthly_features[source_name] = self.monthly_tensors[source_name][
                monthly_start:monthly_end
            ]
            monthly_masks[source_name] = self.monthly_mask_tensors[source_name][
                monthly_start:monthly_end
            ]

        # Compute month boundary indices relative to the daily sequence
        # These indicate where each month starts/ends within the daily sequence
        daily_seq_len = daily_end - daily_start
        month_boundary_indices = []

        for month_idx in range(monthly_start, monthly_end):
            if month_idx >= 0 and month_idx < len(self.alignment.month_boundaries):
                abs_start, abs_end = self.alignment.month_boundaries[month_idx]
                # Convert to relative indices within the daily sequence
                rel_start = abs_start - daily_start
                rel_end = abs_end - daily_start

                # Clip to valid range within the daily sequence
                rel_start = max(0, min(rel_start, daily_seq_len))
                rel_end = max(0, min(rel_end, daily_seq_len))

                # Ensure start <= end
                if rel_start > rel_end:
                    rel_start = rel_end

                month_boundary_indices.append([rel_start, rel_end])
            else:
                # Month outside bounds - mark as empty
                month_boundary_indices.append([0, 0])

        month_boundary_tensor = torch.tensor(month_boundary_indices, dtype=torch.long)

        # Get date arrays
        daily_dates = self.alignment.daily_dates[daily_start:daily_end]
        monthly_dates = self.alignment.monthly_dates[monthly_start:monthly_end]

        return MultiResolutionSample(
            daily_features=daily_features,
            daily_masks=daily_masks,
            monthly_features=monthly_features,
            monthly_masks=monthly_masks,
            month_boundary_indices=month_boundary_tensor,
            daily_dates=daily_dates,
            monthly_dates=monthly_dates,
            sample_idx=idx
        )

    def get_feature_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about features for each source."""
        info = {}

        for source_name, (df, _) in self.daily_data.items():
            feature_cols = [c for c in df.columns if c != 'date']
            info[source_name] = {
                'resolution': 'daily',
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'n_timesteps': len(df)
            }

        for source_name, (df, _) in self.monthly_data.items():
            feature_cols = [c for c in df.columns if c != 'date']
            info[source_name] = {
                'resolution': 'monthly',
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'n_timesteps': len(df)
            }

        return info


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def multi_resolution_collate_fn(
    batch: List[MultiResolutionSample]
) -> Dict[str, Any]:
    """
    Custom collate function for multi-resolution data.

    Handles variable-length sequences by padding to the maximum length in the batch.

    Args:
        batch: List of MultiResolutionSample objects

    Returns:
        Dictionary with batched tensors and metadata
    """
    if not batch:
        raise ValueError("Empty batch")

    batch_size = len(batch)

    # Get source names from first sample
    daily_sources = list(batch[0].daily_features.keys())
    monthly_sources = list(batch[0].monthly_features.keys())

    # Find max sequence lengths in batch
    max_daily_len = max(sample.daily_features[daily_sources[0]].shape[0]
                        for sample in batch) if daily_sources else 0
    max_monthly_len = max(sample.monthly_features[monthly_sources[0]].shape[0]
                          for sample in batch) if monthly_sources else 0

    # Initialize output dictionaries
    batched_daily_features = {}
    batched_daily_masks = {}
    batched_monthly_features = {}
    batched_monthly_masks = {}

    # Batch daily data
    for source_name in daily_sources:
        n_features = batch[0].daily_features[source_name].shape[1]

        # Initialize with padding value (missing_value for features, False for masks)
        features_batch = torch.full(
            (batch_size, max_daily_len, n_features),
            fill_value=MISSING_VALUE,
            dtype=torch.float32
        )
        masks_batch = torch.zeros(
            (batch_size, max_daily_len, n_features),
            dtype=torch.bool
        )

        for i, sample in enumerate(batch):
            seq_len = sample.daily_features[source_name].shape[0]
            features_batch[i, :seq_len] = sample.daily_features[source_name]
            masks_batch[i, :seq_len] = sample.daily_masks[source_name]

        batched_daily_features[source_name] = features_batch
        batched_daily_masks[source_name] = masks_batch

    # Batch monthly data
    for source_name in monthly_sources:
        n_features = batch[0].monthly_features[source_name].shape[1]

        features_batch = torch.full(
            (batch_size, max_monthly_len, n_features),
            fill_value=MISSING_VALUE,
            dtype=torch.float32
        )
        masks_batch = torch.zeros(
            (batch_size, max_monthly_len, n_features),
            dtype=torch.bool
        )

        for i, sample in enumerate(batch):
            seq_len = sample.monthly_features[source_name].shape[0]
            features_batch[i, :seq_len] = sample.monthly_features[source_name]
            masks_batch[i, :seq_len] = sample.monthly_masks[source_name]

        batched_monthly_features[source_name] = features_batch
        batched_monthly_masks[source_name] = masks_batch

    # Batch month boundary indices
    max_n_months = max(sample.month_boundary_indices.shape[0] for sample in batch)
    month_boundaries_batch = torch.zeros(
        (batch_size, max_n_months, 2),
        dtype=torch.long
    )

    for i, sample in enumerate(batch):
        n_months = sample.month_boundary_indices.shape[0]
        month_boundaries_batch[i, :n_months] = sample.month_boundary_indices

    # Sequence lengths for masking in attention
    daily_seq_lens = torch.tensor([
        sample.daily_features[daily_sources[0]].shape[0] if daily_sources else 0
        for sample in batch
    ], dtype=torch.long)

    monthly_seq_lens = torch.tensor([
        sample.monthly_features[monthly_sources[0]].shape[0] if monthly_sources else 0
        for sample in batch
    ], dtype=torch.long)

    return {
        'daily_features': batched_daily_features,
        'daily_masks': batched_daily_masks,
        'monthly_features': batched_monthly_features,
        'monthly_masks': batched_monthly_masks,
        'month_boundary_indices': month_boundaries_batch,
        'daily_seq_lens': daily_seq_lens,
        'monthly_seq_lens': monthly_seq_lens,
        'batch_size': batch_size,
        'sample_indices': [sample.sample_idx for sample in batch]
    }


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_multi_resolution_dataloaders(
    config: Optional[MultiResolutionConfig] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Dict[str, np.ndarray]]]:
    """
    Create train, validation, and test data loaders for multi-resolution data.

    This function ensures proper data handling:
    1. Training dataset is created first (computes normalization stats)
    2. Normalization stats from training are passed to val/test datasets
    3. Temporal ordering is preserved (no shuffling across time)

    Args:
        config: Dataset configuration (uses defaults if None)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        norm_stats: Normalization statistics from training data
    """
    config = config or MultiResolutionConfig()

    print("=" * 80)
    print("Creating Multi-Resolution DataLoaders")
    print("=" * 80)

    # Create training dataset first (computes normalization stats)
    print("\n--- Training Dataset ---")
    train_dataset = MultiResolutionDataset(
        config=config,
        split='train',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    norm_stats = train_dataset.norm_stats

    # Create validation dataset
    print("\n--- Validation Dataset ---")
    val_dataset = MultiResolutionDataset(
        config=config,
        split='val',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        norm_stats=norm_stats,
        seed=seed
    )

    # Create test dataset
    print("\n--- Test Dataset ---")
    test_dataset = MultiResolutionDataset(
        config=config,
        split='test',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        norm_stats=norm_stats,
        seed=seed
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle within training set
        num_workers=num_workers,
        collate_fn=multi_resolution_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_resolution_collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multi_resolution_collate_fn,
        pin_memory=True
    )

    print("\n" + "=" * 80)
    print("DataLoader Summary")
    print("=" * 80)
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, norm_stats


# =============================================================================
# MAIN - TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-RESOLUTION DATA LOADER TEST")
    print("=" * 80)

    # Create configuration
    config = MultiResolutionConfig(
        daily_seq_len=180,  # 6 months of daily data
        monthly_seq_len=6,   # 6 months of monthly data
        prediction_horizon=1
    )

    # Create data loaders
    train_loader, val_loader, test_loader, norm_stats = create_multi_resolution_dataloaders(
        config=config,
        batch_size=2,
        num_workers=0
    )

    # Print feature information
    print("\n" + "=" * 80)
    print("Feature Information")
    print("=" * 80)

    feature_info = train_loader.dataset.get_feature_info()
    for source_name, info in feature_info.items():
        print(f"\n{source_name} ({info['resolution']}):")
        print(f"  Features: {info['n_features']}")
        print(f"  Timesteps: {info['n_timesteps']}")
        print(f"  Feature names: {info['feature_names'][:5]}...")

    # Test one batch
    print("\n" + "=" * 80)
    print("Sample Batch")
    print("=" * 80)

    for batch in train_loader:
        print(f"\nBatch size: {batch['batch_size']}")

        print("\nDaily features:")
        for source_name, tensor in batch['daily_features'].items():
            mask = batch['daily_masks'][source_name]
            obs_rate = mask.float().mean().item() * 100
            print(f"  {source_name}: {tensor.shape}, observation rate: {obs_rate:.1f}%")

        print("\nMonthly features:")
        for source_name, tensor in batch['monthly_features'].items():
            mask = batch['monthly_masks'][source_name]
            obs_rate = mask.float().mean().item() * 100
            print(f"  {source_name}: {tensor.shape}, observation rate: {obs_rate:.1f}%")

        print(f"\nMonth boundary indices: {batch['month_boundary_indices'].shape}")
        print(f"Daily sequence lengths: {batch['daily_seq_lens']}")
        print(f"Monthly sequence lengths: {batch['monthly_seq_lens']}")

        break

    # Print normalization stats
    print("\n" + "=" * 80)
    print("Normalization Statistics (Training Data)")
    print("=" * 80)

    for source_name, stats in norm_stats.items():
        print(f"\n{source_name}:")
        print(f"  Type: {stats['type']}")
        print(f"  Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
        print(f"  Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
