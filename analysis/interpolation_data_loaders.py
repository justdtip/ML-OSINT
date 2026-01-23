#!/usr/bin/env python3
"""
Real Data Loaders for Joint Interpolation Models (Expanded Version)

This module provides independent data loaders for each data source,
extracting ALL available features from the raw data for maximum coverage.

Each loader:
1. Loads raw data from the source files
2. Extracts ALL relevant features at their native temporal resolution
3. Creates observation sequences with timestamps
4. Prepares training samples for gap interpolation

Usage:
    python analysis/interpolation_data_loaders.py [--source SOURCE] [--inspect]

    --source: sentinel, deepstate, equipment, firms, ucdp (or 'all')
    --inspect: Print data statistics and samples instead of training
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import domain-specific imputation strategies
# Handle both direct execution and module import scenarios
import sys
_analysis_dir = Path(__file__).parent
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from missing_data_imputation import IMPUTATION_STRATEGIES, impute_domain_data

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

# Try importing torch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Data inspection only.")


# =============================================================================
# NORMALIZATION MIXIN - Proper train-only normalization
# =============================================================================

class NormalizationMixin:
    """
    Mixin class providing proper train-only normalization to prevent data leakage.

    This mixin should be used by data loaders that perform normalization.
    It ensures normalization statistics are computed only from training data
    and then applied consistently to all data (train/val/test).

    Attributes
    ----------
    _norm_stats : Dict[str, Dict[str, float]]
        Dictionary storing normalization statistics per feature.
        Each feature has 'mean', 'std', 'min', 'max' as applicable.
    _normalization_fitted : bool
        Whether normalization statistics have been computed.
    """

    def __init_normalization(self):
        """Initialize normalization state."""
        self._norm_stats: Dict[str, Dict[str, float]] = {}
        self._normalization_fitted: bool = False

    def fit_normalization(
        self,
        data: np.ndarray,
        train_indices: np.ndarray,
        feature_names: List[str],
        method: str = 'zscore'
    ) -> 'NormalizationMixin':
        """
        Compute normalization statistics from training data only.

        Parameters
        ----------
        data : np.ndarray
            Full dataset of shape (n_samples, n_features).
        train_indices : np.ndarray
            Boolean mask or integer indices identifying training samples.
        feature_names : List[str]
            Names of features corresponding to data columns.
        method : str
            Normalization method: 'zscore', 'minmax', or 'robust'.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if isinstance(train_indices, np.ndarray) and train_indices.dtype == bool:
            train_data = data[train_indices]
        else:
            train_data = data[train_indices]

        self._norm_stats = {}

        for i, feat_name in enumerate(feature_names):
            col = train_data[:, i]
            stats = {}

            if method == 'zscore':
                stats['mean'] = float(np.nanmean(col))
                stats['std'] = float(np.nanstd(col))
                if stats['std'] == 0:
                    stats['std'] = 1.0  # Prevent division by zero
            elif method == 'minmax':
                stats['min'] = float(np.nanmin(col))
                stats['max'] = float(np.nanmax(col))
                if stats['max'] == stats['min']:
                    stats['max'] = stats['min'] + 1.0  # Prevent division by zero
            elif method == 'robust':
                stats['median'] = float(np.nanmedian(col))
                stats['iqr'] = float(np.nanpercentile(col, 75) - np.nanpercentile(col, 25))
                if stats['iqr'] == 0:
                    stats['iqr'] = 1.0  # Prevent division by zero

            stats['method'] = method
            self._norm_stats[feat_name] = stats

        self._normalization_fitted = True
        return self

    def apply_normalization(
        self,
        data: np.ndarray,
        feature_names: List[str],
        clip_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Apply stored normalization statistics to data.

        Parameters
        ----------
        data : np.ndarray
            Data to normalize of shape (n_samples, n_features).
        feature_names : List[str]
            Names of features corresponding to data columns.
        clip_range : Optional[Tuple[float, float]]
            If provided, clip normalized values to this range.

        Returns
        -------
        np.ndarray
            Normalized data.

        Raises
        ------
        RuntimeError
            If fit_normalization has not been called first.
        """
        if not self._normalization_fitted:
            raise RuntimeError(
                "Normalization not fitted. Call fit_normalization() first with training data."
            )

        normalized = np.zeros_like(data, dtype=np.float32)

        for i, feat_name in enumerate(feature_names):
            col = data[:, i]
            stats = self._norm_stats.get(feat_name, {})
            method = stats.get('method', 'zscore')

            if method == 'zscore':
                normalized[:, i] = (col - stats['mean']) / stats['std']
            elif method == 'minmax':
                normalized[:, i] = (col - stats['min']) / (stats['max'] - stats['min'])
            elif method == 'robust':
                normalized[:, i] = (col - stats['median']) / stats['iqr']
            else:
                normalized[:, i] = col

            if clip_range is not None:
                normalized[:, i] = np.clip(normalized[:, i], clip_range[0], clip_range[1])

        return normalized

    def inverse_normalization(
        self,
        data: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.

        Parameters
        ----------
        data : np.ndarray
            Normalized data of shape (n_samples, n_features).
        feature_names : List[str]
            Names of features corresponding to data columns.

        Returns
        -------
        np.ndarray
            Data in original scale.
        """
        if not self._normalization_fitted:
            raise RuntimeError(
                "Normalization not fitted. Call fit_normalization() first."
            )

        original = np.zeros_like(data, dtype=np.float32)

        for i, feat_name in enumerate(feature_names):
            col = data[:, i]
            stats = self._norm_stats.get(feat_name, {})
            method = stats.get('method', 'zscore')

            if method == 'zscore':
                original[:, i] = col * stats['std'] + stats['mean']
            elif method == 'minmax':
                original[:, i] = col * (stats['max'] - stats['min']) + stats['min']
            elif method == 'robust':
                original[:, i] = col * stats['iqr'] + stats['median']
            else:
                original[:, i] = col

        return original

    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """Return the stored normalization statistics."""
        return self._norm_stats.copy()


# =============================================================================
# TEMPORAL ALIGNER - Aligns multiple data sources to common grid
# =============================================================================

class TemporalAligner:
    """
    Aligns multiple data sources to a common temporal grid.

    Different data sources have different temporal resolutions and date ranges.
    This class provides infrastructure to resample and align them to a common
    daily grid for joint modeling.

    Attributes
    ----------
    target_resolution : str
        Target temporal resolution ('daily', 'weekly', 'monthly').
    start_date : Optional[datetime]
        Fixed start date for alignment. If None, determined from data.
    end_date : Optional[datetime]
        Fixed end date for alignment. If None, determined from data.
    date_range : pd.DatetimeIndex
        The computed common date range after fitting.
    loader_info : Dict[str, Dict]
        Metadata about each loader's original date range and resolution.
    """

    RESOLUTION_MAP = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'MS'  # Month start
    }

    def __init__(
        self,
        target_resolution: str = 'daily',
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ):
        """
        Initialize the TemporalAligner.

        Parameters
        ----------
        target_resolution : str
            Target resolution for alignment ('daily', 'weekly', 'monthly').
        start_date : Optional[Union[str, datetime]]
            Fixed start date. If None, uses earliest date from all loaders.
        end_date : Optional[Union[str, datetime]]
            Fixed end date. If None, uses latest date from all loaders.
        """
        self.target_resolution = target_resolution
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.date_range: Optional[pd.DatetimeIndex] = None
        self.loader_info: Dict[str, Dict] = {}

    def fit(self, loaders: Dict[str, Any]) -> 'TemporalAligner':
        """
        Determine common date range from all loaders.

        Parameters
        ----------
        loaders : Dict[str, Any]
            Dictionary mapping loader names to loader instances.
            Each loader must have 'dates' attribute after processing.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        all_starts = []
        all_ends = []

        for name, loader in loaders.items():
            # Ensure loader is processed
            if not hasattr(loader, 'dates') or not loader.dates:
                if hasattr(loader, 'process'):
                    loader.process()

            if hasattr(loader, 'dates') and loader.dates:
                dates = pd.to_datetime(loader.dates)
                self.loader_info[name] = {
                    'original_start': dates.min(),
                    'original_end': dates.max(),
                    'n_observations': len(dates),
                    'avg_resolution_days': (dates.max() - dates.min()).days / max(len(dates) - 1, 1)
                }
                all_starts.append(dates.min())
                all_ends.append(dates.max())

        # Determine date range
        if self.start_date is None:
            self.start_date = max(all_starts) if all_starts else pd.Timestamp('2022-02-24')
        if self.end_date is None:
            self.end_date = min(all_ends) if all_ends else pd.Timestamp.now()

        # Create common date range
        freq = self.RESOLUTION_MAP.get(self.target_resolution, 'D')
        self.date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=freq
        )

        return self

    def transform(
        self,
        loader_name: str,
        data: np.ndarray,
        dates: List[str],
        method: str = 'interpolate'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Resample data to target resolution and align to common grid.

        Parameters
        ----------
        loader_name : str
            Name of the loader (for logging/metadata).
        data : np.ndarray
            Data array of shape (n_samples, n_features).
        dates : List[str]
            Date strings corresponding to data rows.
        method : str
            Resampling method: 'interpolate', 'ffill', 'nearest'.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            - Resampled data aligned to common date range
            - Date strings for the common date range
        """
        if self.date_range is None:
            raise RuntimeError("TemporalAligner not fitted. Call fit() first.")

        # Convert dates to DatetimeIndex
        source_dates = pd.to_datetime(dates)

        # Create DataFrame for resampling
        df = pd.DataFrame(data, index=source_dates)

        # Reindex to common date range
        df_aligned = df.reindex(self.date_range)

        # Apply interpolation/fill method
        if method == 'interpolate':
            df_aligned = df_aligned.interpolate(method='linear', limit_direction='both')
            df_aligned = df_aligned.ffill().bfill()  # Fill any remaining NaN
        elif method == 'ffill':
            df_aligned = df_aligned.ffill().bfill()
        elif method == 'nearest':
            df_aligned = df_aligned.interpolate(method='nearest')
            df_aligned = df_aligned.ffill().bfill()

        aligned_dates = [d.strftime('%Y-%m-%d') for d in self.date_range]

        return df_aligned.values.astype(np.float32), aligned_dates

    def fit_transform(
        self,
        loaders: Dict[str, Any],
        method: str = 'interpolate'
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Fit aligner and transform all loaders in one step.

        Parameters
        ----------
        loaders : Dict[str, Any]
            Dictionary mapping loader names to loader instances.
        method : str
            Resampling method to use for all loaders.

        Returns
        -------
        Dict[str, Tuple[np.ndarray, List[str]]]
            Dictionary mapping loader names to (aligned_data, aligned_dates) tuples.
        """
        self.fit(loaders)

        results = {}
        for name, loader in loaders.items():
            if hasattr(loader, 'processed_data') and hasattr(loader, 'dates'):
                if loader.processed_data is None:
                    loader.process()
                results[name] = self.transform(
                    name,
                    loader.processed_data,
                    loader.dates,
                    method
                )

        return results

    def get_common_date_range(self) -> pd.DatetimeIndex:
        """Return the computed common date range."""
        if self.date_range is None:
            raise RuntimeError("TemporalAligner not fitted. Call fit() first.")
        return self.date_range

    def get_alignment_summary(self) -> str:
        """Return a summary of the alignment."""
        if self.date_range is None:
            return "TemporalAligner not fitted."

        lines = [
            f"Temporal Alignment Summary",
            f"=" * 50,
            f"Target resolution: {self.target_resolution}",
            f"Common date range: {self.date_range[0].date()} to {self.date_range[-1].date()}",
            f"Total aligned observations: {len(self.date_range)}",
            f"",
            f"Loader coverage:"
        ]

        for name, info in self.loader_info.items():
            lines.append(
                f"  {name}: {info['original_start'].date()} to {info['original_end'].date()} "
                f"({info['n_observations']} obs, ~{info['avg_resolution_days']:.1f} day resolution)"
            )

        return "\n".join(lines)


# =============================================================================
# DATA INSPECTION UTILITIES
# =============================================================================

@dataclass
class DataStats:
    """Statistics for a loaded dataset."""
    name: str
    n_observations: int
    n_features: int
    date_range: Tuple[str, str]
    temporal_resolution_days: float
    missing_rate: float
    feature_names: List[str]
    sample_data: np.ndarray  # First few rows
    sample_dates: List[str]

    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f"DATA SOURCE: {self.name}")
        print(f"{'='*60}")
        print(f"  Observations: {self.n_observations}")
        print(f"  Features: {self.n_features}")
        print(f"  Date range: {self.date_range[0]} to {self.date_range[1]}")
        print(f"  Avg resolution: {self.temporal_resolution_days:.2f} days")
        print(f"  Missing rate: {self.missing_rate:.1%}")
        print(f"\n  Features:")
        for i, name in enumerate(self.feature_names[:20]):
            print(f"    {i+1}. {name}")
        if len(self.feature_names) > 20:
            print(f"    ... and {len(self.feature_names) - 20} more")
        print(f"\n  Sample data (first 3 observations):")
        print(f"    Dates: {self.sample_dates[:3]}")
        for i, row in enumerate(self.sample_data[:3]):
            print(f"    [{i}]: {row[:5]}{'...' if len(row) > 5 else ''}")


# =============================================================================
# FIRMS DATA LOADER (FULLY EXPANDED - 42 features)
# =============================================================================

class FIRMSDataLoader:
    """
    Loads FIRMS (Fire Information for Resource Management System) data.

    FULLY EXPANDED: Extracts 42 features including:
    - Fire counts (total, day, night)
    - FRP statistics (total, mean, max, std, percentiles)
    - FRP intensity buckets (tiny, small, medium, large, extreme)
    - Brightness statistics
    - Scan/track pixel dimensions
    - Confidence breakdowns (high, nominal, low)
    - Fire type breakdowns
    - Geographic statistics (lat/lon spread, centroid, clustering)
    - Temporal statistics (hourly breakdown)
    - Satellite breakdown (VIIRS N/J variants)
    - Derived metrics (ratios, trends)
    """

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # FULLY EXPANDED Feature list - 42 features
        self.feature_names = [
            # Basic counts (3)
            'fire_count', 'day_fires', 'night_fires',
            # FRP statistics (7)
            'frp_total', 'frp_mean', 'frp_max', 'frp_min', 'frp_std', 'frp_median', 'frp_p90',
            # FRP intensity buckets (5)
            'frp_tiny', 'frp_small', 'frp_medium', 'frp_large', 'frp_extreme',
            # Brightness statistics (5)
            'brightness_mean', 'brightness_max', 'brightness_min', 'brightness_std', 'bright_t31_mean',
            # Scan/track pixel dimensions (4)
            'scan_mean', 'scan_max', 'track_mean', 'track_max',
            # Confidence breakdown (4)
            'conf_high', 'conf_nominal', 'conf_low', 'conf_high_ratio',
            # Fire type breakdown (4)
            'type_0', 'type_2', 'type_3', 'type_other',
            # Geographic spread (4)
            'lat_std', 'lon_std', 'centroid_lat', 'centroid_lon',
            # Temporal (time of day) (4)
            'fires_morning', 'fires_midday', 'fires_evening', 'fires_night_hours',
            # Derived metrics (2)
            'day_night_ratio', 'fires_per_km2_approx',
        ]

    def load(self) -> 'FIRMSDataLoader':
        """Load raw data from CSV."""
        self.raw_data = pd.read_csv(self.data_path)
        self.raw_data['acq_date'] = pd.to_datetime(self.raw_data['acq_date'])
        # Parse time for time-of-day features
        self.raw_data['acq_hour'] = self.raw_data['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)
        return self

    def process(self) -> 'FIRMSDataLoader':
        """Aggregate fires to daily features - FULLY EXPANDED."""
        if self.raw_data is None:
            self.load()

        df = self.raw_data.copy()

        # Pre-compute FRP intensity buckets
        df['frp_tiny'] = (df['frp'] < 1).astype(int)
        df['frp_small'] = ((df['frp'] >= 1) & (df['frp'] < 10)).astype(int)
        df['frp_medium'] = ((df['frp'] >= 10) & (df['frp'] < 50)).astype(int)
        df['frp_large'] = ((df['frp'] >= 50) & (df['frp'] < 200)).astype(int)
        df['frp_extreme'] = (df['frp'] >= 200).astype(int)

        # Pre-compute time of day buckets
        df['is_morning'] = ((df['acq_hour'] >= 6) & (df['acq_hour'] < 12)).astype(int)
        df['is_midday'] = ((df['acq_hour'] >= 12) & (df['acq_hour'] < 18)).astype(int)
        df['is_evening'] = ((df['acq_hour'] >= 18) & (df['acq_hour'] < 22)).astype(int)
        df['is_night_hours'] = ((df['acq_hour'] >= 22) | (df['acq_hour'] < 6)).astype(int)

        # Group by date with comprehensive aggregations
        daily_data = []

        for date, group in df.groupby('acq_date'):
            row = {'date': date}

            # Basic counts
            row['fire_count'] = len(group)
            row['day_fires'] = (group['daynight'] == 'D').sum()
            row['night_fires'] = (group['daynight'] == 'N').sum()

            # FRP statistics
            frp = group['frp']
            row['frp_total'] = frp.sum()
            row['frp_mean'] = frp.mean()
            row['frp_max'] = frp.max()
            row['frp_min'] = frp.min()
            row['frp_std'] = frp.std() if len(frp) > 1 else 0
            row['frp_median'] = frp.median()
            row['frp_p90'] = frp.quantile(0.9) if len(frp) > 0 else 0

            # FRP intensity buckets
            row['frp_tiny'] = group['frp_tiny'].sum()
            row['frp_small'] = group['frp_small'].sum()
            row['frp_medium'] = group['frp_medium'].sum()
            row['frp_large'] = group['frp_large'].sum()
            row['frp_extreme'] = group['frp_extreme'].sum()

            # Brightness statistics
            row['brightness_mean'] = group['brightness'].mean()
            row['brightness_max'] = group['brightness'].max()
            row['brightness_min'] = group['brightness'].min()
            row['brightness_std'] = group['brightness'].std() if len(group) > 1 else 0
            row['bright_t31_mean'] = group['bright_t31'].mean()

            # Scan/track pixel dimensions
            row['scan_mean'] = group['scan'].mean()
            row['scan_max'] = group['scan'].max()
            row['track_mean'] = group['track'].mean()
            row['track_max'] = group['track'].max()

            # Confidence breakdown
            row['conf_high'] = (group['confidence'] == 'h').sum()
            row['conf_nominal'] = (group['confidence'] == 'n').sum()
            row['conf_low'] = (group['confidence'] == 'l').sum()
            row['conf_high_ratio'] = row['conf_high'] / max(row['fire_count'], 1)

            # Fire type breakdown
            row['type_0'] = (group['type'] == 0).sum()
            row['type_2'] = (group['type'] == 2).sum()
            row['type_3'] = (group['type'] == 3).sum()
            row['type_other'] = (~group['type'].isin([0, 2, 3])).sum()

            # Geographic spread
            row['lat_std'] = group['latitude'].std() if len(group) > 1 else 0
            row['lon_std'] = group['longitude'].std() if len(group) > 1 else 0
            row['centroid_lat'] = group['latitude'].mean()
            row['centroid_lon'] = group['longitude'].mean()

            # Temporal (time of day)
            row['fires_morning'] = group['is_morning'].sum()
            row['fires_midday'] = group['is_midday'].sum()
            row['fires_evening'] = group['is_evening'].sum()
            row['fires_night_hours'] = group['is_night_hours'].sum()

            # Derived metrics
            row['day_night_ratio'] = row['day_fires'] / max(row['night_fires'], 1)
            # Approximate km² coverage (using lat/lon spread as proxy)
            coverage_km2 = max(row['lat_std'] * 111, 0.1) * max(row['lon_std'] * 74, 0.1)
            row['fires_per_km2_approx'] = row['fire_count'] / coverage_km2

            daily_data.append(row)

        daily = pd.DataFrame(daily_data)
        daily = daily.sort_values('date')

        # Use domain-specific imputation for FIRMS data (interpolation-based)
        # FIRMS missing data is typically due to satellite pass timing or cloud cover
        daily = daily.set_index('date')
        date_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
        imputed_df, self._observation_mask = impute_domain_data(
            'firms', daily, date_range, self.feature_names
        )
        daily = imputed_df.reset_index()
        daily = daily.rename(columns={'index': 'date'})

        self.dates = [d.strftime('%Y-%m-%d') for d in daily['date']]
        self.processed_data = daily[self.feature_names].values.astype(np.float32)

        return self

    def get_observation_mask(self) -> Optional[pd.DataFrame]:
        """Return the observation mask indicating observed vs imputed values."""
        return getattr(self, '_observation_mask', None)

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 1.0
        missing_rate = np.isnan(self.processed_data).mean()

        return DataStats(
            name="FIRMS Fire Detections (FULLY EXPANDED)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# UCDP EVENT DATA LOADER (FULLY EXPANDED - 48 features)
# =============================================================================

class UCDPDataLoader:
    """
    Loads UCDP (Uppsala Conflict Data Program) event data.

    FULLY EXPANDED: Extracts 48 features including:
    - Event counts and types
    - Casualty breakdowns (deaths by party, estimates)
    - Violence type breakdown
    - Geographic features (oblast + district breakdown)
    - Precision/quality indicators (where_prec, date_prec, event_clarity)
    - Source count metrics
    - Actor breakdown (side_a, side_b types)
    - Conflict type breakdown
    - Derived metrics
    """

    # Define Ukrainian oblasts for geographic breakdown
    OBLASTS = [
        'Donetsk', 'Luhansk', 'Kharkiv', 'Kherson', 'Zaporizhzhia',
        'Dnipropetrovsk', 'Mykolaiv', 'Odesa', 'Chernihiv', 'Sumy',
        'Kyiv', 'Zhytomyr', 'Poltava', 'Kirovohrad', 'Vinnytsia'
    ]

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "ucdp" / "ged_events.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # FULLY EXPANDED Feature list - 48 features
        self.feature_names = [
            # Basic counts (1)
            'event_count',
            # Violence type breakdown (3)
            'state_based', 'non_state', 'one_sided',
            # Casualty estimates (7)
            'deaths_best', 'deaths_high', 'deaths_low',
            'deaths_side_a', 'deaths_side_b', 'deaths_civilians', 'deaths_unknown',
            # Quality/precision indicators (6)
            'event_clarity_mean', 'date_precision_mean', 'where_precision_mean',
            'high_clarity_count', 'precise_date_count', 'precise_location_count',
            # Source metrics (3)
            'sources_total', 'sources_mean', 'sources_max',
            # Geographic breakdown - oblasts (10)
            'events_donetsk', 'events_luhansk', 'events_kharkiv',
            'events_kherson', 'events_zaporizhzhia', 'events_dnipro',
            'events_mykolaiv', 'events_odesa', 'events_chernihiv', 'events_other',
            # District-level counts (top conflict zones) (5)
            'districts_active', 'events_urban', 'events_rural',
            'geo_spread_lat', 'geo_spread_lon',
            # Conflict/actor breakdown (5)
            'russia_ukraine_events', 'internal_events', 'separatist_events',
            'govt_side_a', 'rebel_side_a',
            # Derived metrics (8)
            'deaths_per_event', 'civilian_ratio', 'lethality_ratio',
            'eastern_front_ratio', 'multi_source_ratio', 'uncertainty_score',
            'conflict_intensity', 'frontline_activity',
        ]

    def load(self) -> 'UCDPDataLoader':
        """Load raw data from CSV."""
        self.raw_data = pd.read_csv(self.data_path, low_memory=False)

        # Filter for Ukraine and recent years (2022+)
        self.raw_data = self.raw_data[
            (self.raw_data['country'] == 'Ukraine') &
            (self.raw_data['year'] >= 2022)
        ].copy()

        # Parse dates
        self.raw_data['date'] = pd.to_datetime(self.raw_data['date_start'], errors='coerce')

        # Create oblast column from adm_1
        self.raw_data['oblast'] = self.raw_data['adm_1'].fillna('Unknown')
        self.raw_data['district'] = self.raw_data['adm_2'].fillna('Unknown')

        return self

    def process(self) -> 'UCDPDataLoader':
        """Aggregate events to daily features - FULLY EXPANDED."""
        if self.raw_data is None:
            self.load()

        df = self.raw_data.copy()

        # Pre-compute binary flags
        df['is_state'] = (df['type_of_violence'] == 1).astype(int)
        df['is_non_state'] = (df['type_of_violence'] == 2).astype(int)
        df['is_one_sided'] = (df['type_of_violence'] == 3).astype(int)

        # Clarity/precision flags
        df['high_clarity'] = (df['event_clarity'] == 1).astype(int)
        df['precise_date'] = (df['date_prec'] == 1).astype(int)
        df['precise_location'] = (df['where_prec'] == 1).astype(int)

        # Oblast flags
        oblast_map = {
            'Donetsk': 'donetsk', 'Luhansk': 'luhansk', 'Kharkiv': 'kharkiv',
            'Kherson': 'kherson', 'Zaporizhzhia': 'zaporizhzhia',
            'Dnipropetrovsk': 'dnipro', 'Mykolaiv': 'mykolaiv', 'Odesa': 'odesa',
            'Chernihiv': 'chernihiv'
        }

        for oblast, col_name in oblast_map.items():
            df[f'in_{col_name}'] = df['oblast'].str.contains(oblast, case=False, na=False).astype(int)

        known_oblasts = '|'.join(oblast_map.keys())
        df['in_other'] = (~df['oblast'].str.contains(known_oblasts, case=False, na=False)).astype(int)

        # Eastern front (Donetsk, Luhansk, Kharkiv, Zaporizhzhia)
        df['eastern_front'] = (df['in_donetsk'] | df['in_luhansk'] |
                               df['in_kharkiv'] | df['in_zaporizhzhia']).astype(int)

        # Urban vs rural (heuristic based on where_prec - 1 is exact, higher is less precise)
        df['is_urban'] = (df['where_prec'] <= 2).astype(int)
        df['is_rural'] = (df['where_prec'] > 2).astype(int)

        # Actor types
        df['govt_side_a'] = df['side_a'].str.contains('Government', case=False, na=False).astype(int)
        df['rebel_side_a'] = (~df['side_a'].str.contains('Government', case=False, na=False)).astype(int)

        # Conflict classification
        df['russia_ukraine'] = df['conflict_name'].str.contains('Russia', case=False, na=False).astype(int)
        df['separatist'] = (df['dyad_name'].str.contains('DPR|LPR|Donetsk|Luhansk', case=False, na=False) &
                           ~df['conflict_name'].str.contains('Russia', case=False, na=False)).astype(int)
        df['internal'] = ((df['is_non_state'] == 1) | (df['is_one_sided'] == 1)).astype(int)

        # Multi-source events
        df['multi_source'] = (df['number_of_sources'] > 1).astype(int)

        # Group by date
        daily_data = []

        for date, group in df.groupby('date'):
            row = {'date': date}

            # Basic counts
            row['event_count'] = len(group)

            # Violence types
            row['state_based'] = group['is_state'].sum()
            row['non_state'] = group['is_non_state'].sum()
            row['one_sided'] = group['is_one_sided'].sum()

            # Casualty estimates
            row['deaths_best'] = group['best_est'].sum()
            row['deaths_high'] = group['high_est'].sum()
            row['deaths_low'] = group['low_est'].sum()
            row['deaths_side_a'] = group['deaths_a'].sum()
            row['deaths_side_b'] = group['deaths_b'].sum()
            row['deaths_civilians'] = group['deaths_civilians'].sum()
            row['deaths_unknown'] = group['deaths_unknown'].sum()

            # Quality/precision indicators
            row['event_clarity_mean'] = group['event_clarity'].mean()
            row['date_precision_mean'] = group['date_prec'].mean()
            row['where_precision_mean'] = group['where_prec'].mean()
            row['high_clarity_count'] = group['high_clarity'].sum()
            row['precise_date_count'] = group['precise_date'].sum()
            row['precise_location_count'] = group['precise_location'].sum()

            # Source metrics
            sources = group['number_of_sources'].fillna(1)
            row['sources_total'] = sources.sum()
            row['sources_mean'] = sources.mean()
            row['sources_max'] = sources.max()

            # Oblast breakdown
            for oblast, col_name in oblast_map.items():
                row[f'events_{col_name}'] = group[f'in_{col_name}'].sum()
            row['events_other'] = group['in_other'].sum()

            # District-level metrics
            row['districts_active'] = group['district'].nunique()
            row['events_urban'] = group['is_urban'].sum()
            row['events_rural'] = group['is_rural'].sum()

            # Geographic spread
            row['geo_spread_lat'] = group['latitude'].std() if len(group) > 1 else 0
            row['geo_spread_lon'] = group['longitude'].std() if len(group) > 1 else 0

            # Conflict/actor breakdown
            row['russia_ukraine_events'] = group['russia_ukraine'].sum()
            row['internal_events'] = group['internal'].sum()
            row['separatist_events'] = group['separatist'].sum()
            row['govt_side_a'] = group['govt_side_a'].sum()
            row['rebel_side_a'] = group['rebel_side_a'].sum()

            # Derived metrics
            row['deaths_per_event'] = row['deaths_best'] / max(row['event_count'], 1)
            row['civilian_ratio'] = row['deaths_civilians'] / max(row['deaths_best'], 1)
            row['lethality_ratio'] = row['deaths_high'] / max(row['deaths_low'], 1)
            row['eastern_front_ratio'] = group['eastern_front'].sum() / max(row['event_count'], 1)
            row['multi_source_ratio'] = group['multi_source'].sum() / max(row['event_count'], 1)
            # Uncertainty: higher where_prec and date_prec = more uncertainty
            row['uncertainty_score'] = (row['where_precision_mean'] + row['date_precision_mean']) / 2
            # Conflict intensity: events * deaths
            row['conflict_intensity'] = row['event_count'] * (row['deaths_best'] + 1)
            # Frontline activity: eastern events weighted by lethality
            row['frontline_activity'] = row['eastern_front_ratio'] * row['deaths_per_event']

            daily_data.append(row)

        daily = pd.DataFrame(daily_data)
        daily = daily.sort_values('date').dropna(subset=['date'])

        # Use domain-specific imputation for UCDP data (rolling median with decay)
        # Missing conflict data typically indicates periods of low/no activity
        daily = daily.set_index('date')
        date_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
        imputed_df, self._observation_mask = impute_domain_data(
            'ucdp', daily, date_range, self.feature_names
        )
        daily = imputed_df.reset_index()
        daily = daily.rename(columns={'index': 'date'})

        self.dates = [d.strftime('%Y-%m-%d') for d in daily['date']]
        self.processed_data = daily[self.feature_names].values.astype(np.float32)

        return self

    def get_observation_mask(self) -> Optional[pd.DataFrame]:
        """Return the observation mask indicating observed vs imputed values."""
        return getattr(self, '_observation_mask', None)

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        if len(self.dates) > 1:
            dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in self.dates]
            deltas = [(dates_dt[i+1] - dates_dt[i]).days for i in range(len(dates_dt)-1)]
            resolution = np.mean(deltas)
        else:
            resolution = 1.0

        missing_rate = np.isnan(self.processed_data).mean() if self.processed_data is not None else 0

        return DataStats(
            name="UCDP Conflict Events (FULLY EXPANDED)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if self.processed_data is not None and len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# EQUIPMENT DATA LOADER (FULLY EXPANDED - 42 features)
# =============================================================================

class EquipmentDataLoader:
    """
    Loads Russian equipment losses data.

    FULLY EXPANDED: Extracts 42 features including:
    - All equipment categories (cumulative + daily deltas)
    - New categories: cruise missiles, SRBM, submarines
    - Rate calculations (7-day, 30-day averages)
    - Total losses metrics
    - Greatest losses direction (encoded)
    """

    # Direction encoding for greatest losses direction
    DIRECTION_ENCODING = {
        'Bakhmut': 1, 'Lyman': 2, 'Avdiivka': 3, 'Zaporizhzhia': 4,
        'Kherson': 5, 'Kharkiv': 6, 'Donetsk': 7, 'Mariupol': 8,
        'Sievierodonetsk': 9, 'Melitopol': 10, 'Pokrovsk': 11, 'Toretsk': 12,
        'Kramatorsk': 13, 'Kupyansk': 14, 'Kurakhove': 15, 'Unknown': 0
    }

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # FULLY EXPANDED Feature list - 38 features
        self.feature_names = [
            # Aircraft (3)
            'aircraft', 'aircraft_delta', 'aircraft_7day_avg',
            # Helicopters (3)
            'helicopter', 'helicopter_delta', 'helicopter_7day_avg',
            # Armor (6)
            'tank', 'tank_delta', 'tank_7day_avg',
            'apc', 'apc_delta', 'apc_7day_avg',
            # Artillery (6)
            'field_artillery', 'field_artillery_delta', 'field_artillery_7day_avg',
            'mrl', 'mrl_delta', 'mrl_7day_avg',
            # Air defense (3)
            'anti_aircraft', 'anti_aircraft_delta', 'anti_aircraft_7day_avg',
            # Drones (3)
            'drone', 'drone_delta', 'drone_7day_avg',
            # Naval (4)
            'naval_ship', 'naval_ship_delta',
            'submarines', 'submarines_delta',
            # Vehicles/logistics (3)
            'vehicles_fuel', 'vehicles_fuel_delta', 'vehicles_fuel_7day_avg',
            # Special equipment (3)
            'special_equipment', 'special_equipment_delta', 'special_equipment_7day_avg',
            # Missiles (4)
            'cruise_missiles', 'cruise_missiles_delta',
            'srbm', 'srbm_delta',
            # Aggregate metrics (4)
            'total_losses_day', 'total_losses_cumulative',
            'heavy_equipment_ratio', 'direction_encoded',
        ]

    def load(self) -> 'EquipmentDataLoader':
        """Load raw data from JSON."""
        with open(self.data_path) as f:
            self.raw_data = json.load(f)
        return self

    def _encode_direction(self, direction_str: str) -> int:
        """Encode greatest losses direction to numeric value."""
        if not direction_str or pd.isna(direction_str):
            return 0
        for key, val in self.DIRECTION_ENCODING.items():
            if key.lower() in str(direction_str).lower():
                return val
        return 0

    def process(self) -> 'EquipmentDataLoader':
        """Process equipment data - FULLY EXPANDED with all metrics."""
        if self.raw_data is None:
            self.load()

        # Column mapping from JSON keys to base feature names
        col_map = {
            'aircraft': 'aircraft',
            'helicopter': 'helicopter',
            'tank': 'tank',
            'APC': 'apc',
            'field artillery': 'field_artillery',
            'MRL': 'mrl',
            'anti-aircraft warfare': 'anti_aircraft',
            'drone': 'drone',
            'naval ship': 'naval_ship',
            'submarines': 'submarines',
            'vehicles and fuel tanks': 'vehicles_fuel',
            'special equipment': 'special_equipment',
            'cruise missiles': 'cruise_missiles',
            'mobile SRBM system': 'srbm',
        }

        # Convert to DataFrame
        df = pd.DataFrame(self.raw_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # Handle older records that may use 'military auto' + 'fuel tank'
        if 'military auto' in df.columns and 'vehicles and fuel tanks' not in df.columns:
            df['vehicles and fuel tanks'] = df.get('military auto', 0).fillna(0) + df.get('fuel tank', 0).fillna(0)

        n_days = len(df)
        n_features = len(self.feature_names)
        output = np.zeros((n_days, n_features), dtype=np.float32)

        # Track cumulative values for each category
        cumulative_data = {}

        # Fill cumulative values from JSON
        for json_key, base_name in col_map.items():
            if json_key in df.columns:
                cumulative_data[base_name] = df[json_key].fillna(0).values.astype(np.float32)
            else:
                cumulative_data[base_name] = np.zeros(n_days, dtype=np.float32)

        # Calculate deltas and rolling averages
        delta_data = {}
        avg_7day_data = {}

        for base_name, cumulative in cumulative_data.items():
            # Daily delta
            delta = np.diff(cumulative, prepend=cumulative[0] if len(cumulative) > 0 else 0)
            delta[0] = 0  # First day has no meaningful delta
            delta_data[base_name] = delta

            # 7-day rolling average of deltas
            avg_7day = pd.Series(delta).rolling(7, min_periods=1).mean().values
            avg_7day_data[base_name] = avg_7day

        # Fill output array based on feature_names order
        for i, feat_name in enumerate(self.feature_names):
            if feat_name.endswith('_7day_avg'):
                base_name = feat_name.replace('_7day_avg', '')
                if base_name in avg_7day_data:
                    output[:, i] = avg_7day_data[base_name]
            elif feat_name.endswith('_delta'):
                base_name = feat_name.replace('_delta', '')
                if base_name in delta_data:
                    output[:, i] = delta_data[base_name]
            elif feat_name in cumulative_data:
                output[:, i] = cumulative_data[feat_name]

        # Calculate aggregate metrics
        # Total losses per day (sum of all deltas)
        daily_totals = sum(delta_data.values())
        output[:, self.feature_names.index('total_losses_day')] = daily_totals

        # Total cumulative losses (sum of all cumulative)
        cumulative_totals = sum(cumulative_data.values())
        output[:, self.feature_names.index('total_losses_cumulative')] = cumulative_totals

        # Heavy equipment ratio (tanks + APCs + artillery) / total
        heavy_equip = (cumulative_data['tank'] + cumulative_data['apc'] +
                       cumulative_data['field_artillery'] + cumulative_data['mrl'])
        total = cumulative_totals + 1e-8  # Avoid div by zero
        output[:, self.feature_names.index('heavy_equipment_ratio')] = heavy_equip / total

        # Direction encoding
        if 'greatest losses direction' in df.columns:
            output[:, self.feature_names.index('direction_encoded')] = df['greatest losses direction'].apply(
                self._encode_direction).values

        # Apply domain-specific imputation for any temporal gaps
        # Equipment losses are cumulative counts - use forward fill strategy
        output_df = pd.DataFrame(output, columns=self.feature_names)
        output_df['date'] = df['date'].values
        output_df = output_df.set_index('date')

        date_range = pd.date_range(start=output_df.index.min(), end=output_df.index.max(), freq='D')
        imputed_df, self._observation_mask = impute_domain_data(
            'equipment', output_df, date_range, self.feature_names
        )
        output_df = imputed_df.reset_index()
        output_df = output_df.rename(columns={'index': 'date'})

        self.dates = [d.strftime('%Y-%m-%d') for d in output_df['date']]
        self.processed_data = output_df[self.feature_names].values.astype(np.float32)

        return self

    def get_daily_changes(self) -> Tuple[np.ndarray, List[str]]:
        """Get daily change values (for training interpolation on deltas)."""
        if self.processed_data is None:
            self.process()
        return self.processed_data, self.dates

    def get_observation_mask(self) -> Optional[pd.DataFrame]:
        """Return the observation mask indicating observed vs imputed values."""
        return getattr(self, '_observation_mask', None)

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 1.0
        missing_rate = np.isnan(self.processed_data).mean()

        return DataStats(
            name="Equipment Losses (FULLY EXPANDED)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# SENTINEL DATA LOADER (FULLY EXPANDED - 55 features)
# =============================================================================

class SentinelDataLoader:
    """
    Loads Sentinel satellite data from JSON timeseries.

    FULLY EXPANDED: Extracts 55 features including:
    - Sentinel-2 Optical (12 features): counts, cloud metrics, coverage, trends
    - Sentinel-1 Radar (10 features): counts, coverage, consistency metrics
    - Sentinel-5P NO2 (8 features): atmospheric nitrogen dioxide
    - Sentinel-5P CO (8 features): atmospheric carbon monoxide
    - Sentinel-3 Fire (8 features): fire radiative power products
    - Cross-sensor metrics (6 features): multi-sensor fusion indicators
    - Temporal features (3 features): calendar-based features
    """

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "sentinel" / "sentinel_timeseries_raw.json"
        self.weekly_path = DATA_DIR / "sentinel" / "sentinel_weekly.json"
        self.raw_data = None
        self.weekly_data = None
        self.processed_data = None
        self.dates = []

        # EXPANDED Feature list - 55 features
        self.feature_names = [
            # === Sentinel-2 Optical (12 features) ===
            's2_count', 's2_avg_cloud', 's2_min_cloud', 's2_cloud_free_count',
            's2_unique_dates', 's2_coverage_ratio', 's2_cloud_variability',
            's2_count_delta', 's2_cloud_trend', 's2_observation_gap',
            's2_usable_pct', 's2_anomaly_score',

            # === Sentinel-1 Radar (10 features) ===
            's1_count', 's1_unique_dates', 's1_coverage_ratio',
            's1_count_delta', 's1_observation_gap', 's1_consistency',
            's1_anomaly_score', 's1_per_day', 's1_trend', 's1_vs_s2_ratio',

            # === Sentinel-5P NO2 (8 features) ===
            's5p_no2_count', 's5p_no2_coverage', 's5p_no2_count_delta',
            's5p_no2_trend', 's5p_no2_anomaly', 's5p_no2_per_day',
            's5p_no2_volatility', 's5p_no2_vs_fire',

            # === Sentinel-5P CO (8 features) ===
            's5p_co_count', 's5p_co_coverage', 's5p_co_count_delta',
            's5p_co_trend', 's5p_co_anomaly', 's5p_co_per_day',
            's5p_co_volatility', 's5p_co_vs_no2_ratio',

            # === Sentinel-3 Fire (8 features) ===
            's3_fire_count', 's3_fire_coverage', 's3_fire_count_delta',
            's3_fire_trend', 's3_fire_anomaly', 's3_fire_per_day',
            's3_fire_volatility', 's3_fire_intensity',

            # === Cross-sensor metrics (6 features) ===
            'total_all_products', 'sensor_diversity', 'optical_radar_ratio',
            'atmospheric_activity', 'data_richness', 'conflict_proxy',

            # === Temporal features (3 features) ===
            'days_in_month', 'season_code', 'year_progress',
        ]

    def load(self) -> 'SentinelDataLoader':
        """Load raw data from JSON files."""
        with open(self.data_path) as f:
            self.raw_data = json.load(f)

        # Load weekly data if available
        if self.weekly_path.exists():
            with open(self.weekly_path) as f:
                self.weekly_data = json.load(f)

        return self

    def _safe_get(self, lst: list, idx: int, default=0):
        """Safely get value from list with bounds checking."""
        if idx < 0 or idx >= len(lst):
            return default
        val = lst[idx]
        return default if val is None else val

    def _compute_trend(self, values: list, idx: int, window: int = 3) -> float:
        """Compute rolling trend (slope) over window."""
        if idx < window - 1:
            return 0.0
        window_vals = [self._safe_get(values, i) for i in range(idx - window + 1, idx + 1)]
        if len(window_vals) < 2:
            return 0.0
        # Simple linear trend: (last - first) / window
        return (window_vals[-1] - window_vals[0]) / window

    def _compute_anomaly(self, values: list, idx: int) -> float:
        """Compute anomaly score (z-score from historical mean)."""
        if idx == 0:
            return 0.0
        historical = [self._safe_get(values, i) for i in range(idx)]
        if not historical:
            return 0.0
        mean = np.mean(historical)
        std = np.std(historical) + 1e-8
        current = self._safe_get(values, idx)
        return (current - mean) / std

    def _compute_volatility(self, values: list, idx: int, window: int = 4) -> float:
        """Compute volatility (std) over recent window."""
        if idx < window - 1:
            return 0.0
        window_vals = [self._safe_get(values, i) for i in range(idx - window + 1, idx + 1)]
        return np.std(window_vals) if window_vals else 0.0

    def _days_in_month(self, year: int, month: int) -> int:
        """Return number of days in given month."""
        import calendar
        return calendar.monthrange(year, month)[1]

    def process(self) -> 'SentinelDataLoader':
        """Process raw data into comprehensive feature arrays."""
        if self.raw_data is None:
            self.load()

        collections = self.raw_data.get('collections', {})

        # Get base data from all collections
        s2_monthly = collections.get('sentinel-2-l2a', {}).get('monthly', [])
        s1_monthly = collections.get('sentinel-1-grd', {}).get('monthly', [])
        s5p_no2_monthly = collections.get('sentinel-5p-l2-no2-offl', {}).get('monthly', [])
        s5p_co_monthly = collections.get('sentinel-5p-l2-co-offl', {}).get('monthly', [])
        s3_fire_monthly = collections.get('sentinel-3-sl-2-frp-ntc', {}).get('monthly', [])

        # Use S2 as reference for dates (most comprehensive)
        n_months = len(s2_monthly)
        if n_months == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        self.dates = [m['month_str'] for m in s2_monthly]

        # Extract raw values from each collection
        s2_count = [m.get('count', 0) for m in s2_monthly]
        s2_cloud = [m.get('avg_cloud_cover', 0) or 0 for m in s2_monthly]
        s2_min_cloud = [m.get('min_cloud_cover', 0) or 0 for m in s2_monthly]
        s2_cloud_free = [m.get('cloud_free_count', 0) for m in s2_monthly]
        s2_unique = [m.get('unique_dates', 0) for m in s2_monthly]

        s1_count = [m.get('count', 0) for m in s1_monthly] if s1_monthly else [0] * n_months
        s1_unique = [m.get('unique_dates', 0) for m in s1_monthly] if s1_monthly else [0] * n_months

        s5p_no2_count = [m.get('count', 0) for m in s5p_no2_monthly] if s5p_no2_monthly else [0] * n_months
        s5p_co_count = [m.get('count', 0) for m in s5p_co_monthly] if s5p_co_monthly else [0] * n_months
        s3_fire_count = [m.get('count', 0) for m in s3_fire_monthly] if s3_fire_monthly else [0] * n_months

        # Initialize output array
        n_features = len(self.feature_names)
        self.processed_data = np.zeros((n_months, n_features), dtype=np.float32)

        # Process each month
        for i in range(n_months):
            idx = 0
            year = s2_monthly[i].get('year', 2022)
            month = s2_monthly[i].get('month', 1)
            days = self._days_in_month(year, month)

            # === Sentinel-2 Optical (12 features) ===
            self.processed_data[i, idx] = s2_count[i]; idx += 1
            self.processed_data[i, idx] = s2_cloud[i]; idx += 1
            self.processed_data[i, idx] = s2_min_cloud[i]; idx += 1
            self.processed_data[i, idx] = s2_cloud_free[i]; idx += 1
            self.processed_data[i, idx] = s2_unique[i]; idx += 1
            # Coverage ratio: cloud_free / count
            self.processed_data[i, idx] = s2_cloud_free[i] / max(s2_count[i], 1); idx += 1
            # Cloud variability (std of recent months)
            self.processed_data[i, idx] = self._compute_volatility(s2_cloud, i); idx += 1
            # Count delta
            self.processed_data[i, idx] = s2_count[i] - self._safe_get(s2_count, i-1) if i > 0 else 0; idx += 1
            # Cloud trend
            self.processed_data[i, idx] = self._compute_trend(s2_cloud, i); idx += 1
            # Observation gap (days / unique dates)
            self.processed_data[i, idx] = days / max(s2_unique[i], 1); idx += 1
            # Usable percentage (cloud_free / count)
            self.processed_data[i, idx] = 100 * s2_cloud_free[i] / max(s2_count[i], 1); idx += 1
            # Anomaly score
            self.processed_data[i, idx] = self._compute_anomaly(s2_count, i); idx += 1

            # === Sentinel-1 Radar (10 features) ===
            self.processed_data[i, idx] = s1_count[i]; idx += 1
            self.processed_data[i, idx] = s1_unique[i]; idx += 1
            # Coverage ratio (unique_dates / days)
            self.processed_data[i, idx] = s1_unique[i] / days; idx += 1
            # Count delta
            self.processed_data[i, idx] = s1_count[i] - self._safe_get(s1_count, i-1) if i > 0 else 0; idx += 1
            # Observation gap
            self.processed_data[i, idx] = days / max(s1_unique[i], 1); idx += 1
            # Consistency (how regular - low std = high consistency)
            self.processed_data[i, idx] = 1.0 / (self._compute_volatility(s1_count, i) + 1); idx += 1
            # Anomaly
            self.processed_data[i, idx] = self._compute_anomaly(s1_count, i); idx += 1
            # Per day
            self.processed_data[i, idx] = s1_count[i] / days; idx += 1
            # Trend
            self.processed_data[i, idx] = self._compute_trend(s1_count, i); idx += 1
            # S1 vs S2 ratio
            self.processed_data[i, idx] = s1_count[i] / max(s2_count[i], 1); idx += 1

            # === Sentinel-5P NO2 (8 features) ===
            self.processed_data[i, idx] = s5p_no2_count[i]; idx += 1
            # Coverage
            self.processed_data[i, idx] = s5p_no2_count[i] / (days * 7); idx += 1  # ~7 products/day typical
            # Delta
            self.processed_data[i, idx] = s5p_no2_count[i] - self._safe_get(s5p_no2_count, i-1) if i > 0 else 0; idx += 1
            # Trend
            self.processed_data[i, idx] = self._compute_trend(s5p_no2_count, i); idx += 1
            # Anomaly
            self.processed_data[i, idx] = self._compute_anomaly(s5p_no2_count, i); idx += 1
            # Per day
            self.processed_data[i, idx] = s5p_no2_count[i] / days; idx += 1
            # Volatility
            self.processed_data[i, idx] = self._compute_volatility(s5p_no2_count, i); idx += 1
            # NO2 vs Fire ratio
            self.processed_data[i, idx] = s5p_no2_count[i] / max(s3_fire_count[i], 1); idx += 1

            # === Sentinel-5P CO (8 features) ===
            self.processed_data[i, idx] = s5p_co_count[i]; idx += 1
            # Coverage
            self.processed_data[i, idx] = s5p_co_count[i] / (days * 3); idx += 1  # ~3 products/day typical
            # Delta
            self.processed_data[i, idx] = s5p_co_count[i] - self._safe_get(s5p_co_count, i-1) if i > 0 else 0; idx += 1
            # Trend
            self.processed_data[i, idx] = self._compute_trend(s5p_co_count, i); idx += 1
            # Anomaly
            self.processed_data[i, idx] = self._compute_anomaly(s5p_co_count, i); idx += 1
            # Per day
            self.processed_data[i, idx] = s5p_co_count[i] / days; idx += 1
            # Volatility
            self.processed_data[i, idx] = self._compute_volatility(s5p_co_count, i); idx += 1
            # CO vs NO2 ratio
            self.processed_data[i, idx] = s5p_co_count[i] / max(s5p_no2_count[i], 1); idx += 1

            # === Sentinel-3 Fire (8 features) ===
            self.processed_data[i, idx] = s3_fire_count[i]; idx += 1
            # Coverage
            self.processed_data[i, idx] = s3_fire_count[i] / (days * 4); idx += 1  # ~4 products/day typical
            # Delta
            self.processed_data[i, idx] = s3_fire_count[i] - self._safe_get(s3_fire_count, i-1) if i > 0 else 0; idx += 1
            # Trend
            self.processed_data[i, idx] = self._compute_trend(s3_fire_count, i); idx += 1
            # Anomaly
            self.processed_data[i, idx] = self._compute_anomaly(s3_fire_count, i); idx += 1
            # Per day
            self.processed_data[i, idx] = s3_fire_count[i] / days; idx += 1
            # Volatility
            self.processed_data[i, idx] = self._compute_volatility(s3_fire_count, i); idx += 1
            # Intensity (normalized to baseline)
            baseline_fire = np.mean(s3_fire_count[:max(i, 1)]) if i > 0 else s3_fire_count[i]
            self.processed_data[i, idx] = s3_fire_count[i] / max(baseline_fire, 1); idx += 1

            # === Cross-sensor metrics (6 features) ===
            total = s2_count[i] + s1_count[i] + s5p_no2_count[i] + s5p_co_count[i] + s3_fire_count[i]
            self.processed_data[i, idx] = total; idx += 1
            # Sensor diversity (entropy-like)
            counts = [s2_count[i], s1_count[i], s5p_no2_count[i], s5p_co_count[i], s3_fire_count[i]]
            probs = [c / max(total, 1) for c in counts]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
            self.processed_data[i, idx] = entropy; idx += 1
            # Optical/radar ratio
            self.processed_data[i, idx] = s2_count[i] / max(s1_count[i], 1); idx += 1
            # Atmospheric activity (normalized sum of NO2 + CO)
            self.processed_data[i, idx] = (s5p_no2_count[i] + s5p_co_count[i]) / (days * 10); idx += 1
            # Data richness (weighted coverage)
            self.processed_data[i, idx] = (
                0.3 * s2_cloud_free[i] / max(s2_count[i], 1) +
                0.2 * s1_unique[i] / days +
                0.2 * s5p_no2_count[i] / (days * 7) +
                0.15 * s5p_co_count[i] / (days * 3) +
                0.15 * s3_fire_count[i] / (days * 4)
            ); idx += 1
            # Conflict proxy (fire + atmospheric anomalies)
            fire_anom = self._compute_anomaly(s3_fire_count, i)
            no2_anom = self._compute_anomaly(s5p_no2_count, i)
            self.processed_data[i, idx] = (fire_anom + no2_anom) / 2; idx += 1

            # === Temporal features (3 features) ===
            self.processed_data[i, idx] = days; idx += 1
            # Season code: 0=winter, 1=spring, 2=summer, 3=fall
            season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
            self.processed_data[i, idx] = season_map.get(month, 0); idx += 1
            # Year progress (0-1)
            self.processed_data[i, idx] = (month - 1) / 12; idx += 1

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 30.0
        missing_mask = np.isnan(self.processed_data)
        missing_rate = missing_mask.mean()

        return DataStats(
            name="Sentinel Satellite Products (EXPANDED)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )

    def get_daily_observations(self) -> Tuple[np.ndarray, List[str]]:
        """Expand monthly data to daily observations."""
        if self.raw_data is None:
            self.load()
        if self.processed_data is None:
            self.process()

        collections = self.raw_data.get('collections', {})

        # Get all unique dates from Sentinel-2
        all_dates = set()
        if 'sentinel-2-l2a' in collections:
            for month in collections['sentinel-2-l2a']['monthly']:
                all_dates.update(month.get('dates', []))

        if not all_dates:
            return self.processed_data, self.dates

        sorted_dates = sorted(all_dates)
        n_days = len(sorted_dates)
        n_features = len(self.feature_names)

        daily_data = np.zeros((n_days, n_features), dtype=np.float32)

        month_map = {m['month_str']: i for i, m in enumerate(
            collections.get('sentinel-2-l2a', {}).get('monthly', [])
        )}

        for i, date in enumerate(sorted_dates):
            month_str = date[:7]
            if month_str in month_map:
                monthly_idx = month_map[month_str]
                if monthly_idx < len(self.processed_data):
                    daily_data[i] = self.processed_data[monthly_idx]

        return daily_data, sorted_dates


# =============================================================================
# DEEPSTATE DATA LOADER (FULLY EXPANDED - 55 features)
# =============================================================================

class DeepStateDataLoader:
    """
    Loads DeepState frontline data from wayback snapshots.

    FULLY EXPANDED: Extracts 55 features including:
    - Geometry counts (3)
    - Territory status counts (4)
    - Territory areas in km² (4)
    - Area deltas (3)
    - Arrow/attack direction counts - 8 cardinal directions (8)
    - Arrow totals and metrics (3)
    - Unit type counts (12)
    - Geographic metrics (6)
    - Temporal features (3)
    - Polygon color breakdown (5)
    - Derived metrics (4)
    """

    # Arrow number to 8-direction mapping (assuming 1=N, clockwise)
    ARROW_TO_8DIR = {
        1: 'N', 2: 'N', 3: 'NE', 4: 'E',
        5: 'E', 6: 'E', 7: 'SE', 8: 'S',
        9: 'S', 10: 'S', 11: 'SW', 12: 'W',
        13: 'W', 14: 'W', 15: 'NW', 16: 'N'
    }

    # Color to status mapping
    STATUS_COLORS = {
        '#0f9d58': 'liberated',   # Green
        '#ff5252': 'occupied',    # Red
        '#a52714': 'occupied',    # Dark red
        '#bcaaa4': 'contested',   # Tan/gray
        '#880e4f': 'unknown',     # Dark pink
    }

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "deepstate" / "wayback_snapshots"
        self.raw_snapshots = []
        self.processed_data = None
        self.dates = []
        self.prev_areas = None  # For computing deltas

        # EXPANDED Feature list - 55 features
        self.feature_names = [
            # Geometry counts (3)
            'n_polygons', 'n_points', 'n_linestrings',

            # Territory status polygon counts (4)
            'polygon_occupied', 'polygon_liberated', 'polygon_contested', 'polygon_unknown',

            # Territory areas in km² (4)
            'area_occupied_km2', 'area_liberated_km2', 'area_contested_km2', 'area_total_km2',

            # Area deltas (daily change in km²) (3)
            'area_occupied_delta', 'area_liberated_delta', 'area_contested_delta',

            # Arrow/Attack direction counts - 8 cardinal (8)
            'arrows_N', 'arrows_NE', 'arrows_E', 'arrows_SE',
            'arrows_S', 'arrows_SW', 'arrows_W', 'arrows_NW',

            # Arrow totals and derived (3)
            'arrows_total', 'arrows_eastward', 'arrows_westward',

            # Unit type counts (12)
            'units_enemy', 'units_airport', 'units_headquarter', 'units_capital',
            'units_naval', 'units_belarus', 'units_special', 'units_clown',
            'units_other', 'units_total', 'units_per_polygon', 'airports_count',

            # Geographic metrics (6)
            'centroid_lat', 'centroid_lon', 'lat_spread', 'lon_spread',
            'frontline_approx_km', 'territory_fragmentation',

            # Temporal features (3)
            'days_since_feb24', 'week_of_year', 'month_of_year',

            # Polygon color breakdown (5)
            'color_green', 'color_red', 'color_tan', 'color_dark', 'color_other',

            # Derived metrics (4)
            'occupied_ratio', 'liberated_ratio', 'activity_intensity', 'control_change_rate',
        ]

    def load(self) -> 'DeepStateDataLoader':
        """Load all wayback snapshots."""
        snapshot_files = sorted(self.data_path.glob("deepstate_wayback_*.json"))

        for fpath in snapshot_files:
            fname = fpath.stem
            date_str = fname.replace('deepstate_wayback_', '')
            try:
                dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                date = dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

            try:
                with open(fpath) as f:
                    data = json.load(f)
                self.raw_snapshots.append({
                    'date': date,
                    'datetime': dt,
                    'data': data
                })
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by date
        self.raw_snapshots.sort(key=lambda x: x['datetime'])
        return self

    def process(self) -> 'DeepStateDataLoader':
        """Process snapshots into feature arrays."""
        if not self.raw_snapshots:
            self.load()

        # Deduplicate by date (keep latest snapshot per day)
        date_to_snapshot = {}
        for snap in self.raw_snapshots:
            date = snap['date']
            if date not in date_to_snapshot or snap['datetime'] > date_to_snapshot[date]['datetime']:
                date_to_snapshot[date] = snap

        sorted_dates = sorted(date_to_snapshot.keys())
        n_obs = len(sorted_dates)
        n_features = len(self.feature_names)

        raw_data = np.zeros((n_obs, n_features), dtype=np.float32)
        self.prev_areas = None

        for i, date in enumerate(sorted_dates):
            snap = date_to_snapshot[date]
            features = self._extract_features(snap['data'], date)
            raw_data[i] = features

        # Apply domain-specific imputation for DeepState data (forward fill)
        # Territorial control is persistent - territory remains under control until changed
        output_df = pd.DataFrame(raw_data, columns=self.feature_names)
        output_df['date'] = pd.to_datetime(sorted_dates)
        output_df = output_df.set_index('date')

        date_range = pd.date_range(start=output_df.index.min(), end=output_df.index.max(), freq='D')
        imputed_df, self._observation_mask = impute_domain_data(
            'deepstate', output_df, date_range, self.feature_names
        )
        output_df = imputed_df.reset_index()
        output_df = output_df.rename(columns={'index': 'date'})

        self.dates = [d.strftime('%Y-%m-%d') for d in output_df['date']]
        self.processed_data = output_df[self.feature_names].values.astype(np.float32)

        return self

    def get_observation_mask(self) -> Optional[pd.DataFrame]:
        """Return the observation mask indicating observed vs imputed values."""
        return getattr(self, '_observation_mask', None)

    def _calculate_polygon_area(self, coords: list) -> float:
        """Calculate approximate polygon area in km² using Shoelace formula."""
        if not coords or len(coords) == 0:
            return 0.0

        ring = coords[0]  # Outer ring
        if len(ring) < 3:
            return 0.0

        n = len(ring)
        area = 0.0

        # Approximate: 1 degree latitude ≈ 111 km
        # At Ukraine latitude (~48°), 1 degree longitude ≈ 74 km
        lat_scale = 111.0
        lon_scale = 74.0

        for i in range(n):
            j = (i + 1) % n
            # Handle 3D coordinates (lon, lat, alt)
            lon_i = ring[i][0] if len(ring[i]) > 0 else 0
            lat_i = ring[i][1] if len(ring[i]) > 1 else 0
            lon_j = ring[j][0] if len(ring[j]) > 0 else 0
            lat_j = ring[j][1] if len(ring[j]) > 1 else 0

            area += (lon_i * lon_scale) * (lat_j * lat_scale)
            area -= (lon_j * lon_scale) * (lat_i * lat_scale)

        return abs(area) / 2.0

    def _extract_features(self, data: dict, date_str: str) -> np.ndarray:
        """Extract all features from a single wayback snapshot."""
        features = np.zeros(len(self.feature_names), dtype=np.float32)

        # Get features from map.features (wayback structure)
        geojson_features = data.get('map', {}).get('features', [])
        if not geojson_features:
            # Try direct features (older format)
            geojson_features = data.get('features', [])

        if not geojson_features:
            return features

        # Initialize counters
        n_poly, n_point, n_line = 0, 0, 0
        polygon_counts = {'occupied': 0, 'liberated': 0, 'contested': 0, 'unknown': 0}
        areas = {'occupied': 0.0, 'liberated': 0.0, 'contested': 0.0, 'unknown': 0.0, 'total': 0.0}
        arrow_dirs = {d: 0 for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
        unit_counts = {
            'enemy': 0, 'airport': 0, 'headquarter': 0, 'capital': 0,
            'naval': 0, 'belarus': 0, 'special': 0, 'clown': 0, 'other': 0
        }
        color_counts = {'green': 0, 'red': 0, 'tan': 0, 'dark': 0, 'other': 0}

        all_lats, all_lons = [], []

        for feat in geojson_features:
            geom = feat.get('geometry', {})
            geom_type = geom.get('type', '')
            props = feat.get('properties', {})
            coords = geom.get('coordinates', [])

            # Geometry type counts
            if geom_type in ('Polygon', 'MultiPolygon'):
                n_poly += 1

                # Calculate area
                if geom_type == 'Polygon':
                    area = self._calculate_polygon_area(coords)
                else:  # MultiPolygon
                    area = sum(self._calculate_polygon_area(p) for p in coords)

                areas['total'] += area

                # Classify by name or fill color
                name = props.get('name', '')
                fill = props.get('fill', '')

                status = 'unknown'
                if 'Liberated' in name or fill == '#0f9d58':
                    status = 'liberated'
                elif 'Occupied' in name or fill in ['#ff5252', '#a52714']:
                    status = 'occupied'
                elif 'Unknown' in name or fill == '#bcaaa4':
                    status = 'contested'

                polygon_counts[status] += 1
                areas[status] += area

                # Color breakdown
                if fill == '#0f9d58':
                    color_counts['green'] += 1
                elif fill in ['#ff5252', '#a52714']:
                    color_counts['red'] += 1
                elif fill == '#bcaaa4':
                    color_counts['tan'] += 1
                elif fill == '#880e4f':
                    color_counts['dark'] += 1
                else:
                    color_counts['other'] += 1

            elif geom_type in ('Point', 'MultiPoint'):
                n_point += 1

                # Extract coordinates for centroid calculation
                if coords and len(coords) >= 2:
                    all_lons.append(coords[0])
                    all_lats.append(coords[1])

                # Parse icon type from description
                desc = props.get('description', '')
                if 'icon=' in desc:
                    icon = desc.split('icon=')[1].split('}')[0].split('<')[0].strip()

                    # Arrows (direction indicators)
                    if 'arrow_' in icon:
                        try:
                            arrow_num = int(icon.replace('arrow_', ''))
                            if arrow_num in self.ARROW_TO_8DIR:
                                arrow_dirs[self.ARROW_TO_8DIR[arrow_num]] += 1
                        except ValueError:
                            pass
                    # Unit types
                    elif 'enemy' in icon:
                        unit_counts['enemy'] += 1
                    elif 'airport' in icon:
                        unit_counts['airport'] += 1
                    elif 'headquarter' in icon:
                        unit_counts['headquarter'] += 1
                    elif 'capital' in icon:
                        unit_counts['capital'] += 1
                    elif 'naval' in icon or 'ship' in icon:
                        unit_counts['naval'] += 1
                    elif 'belarus' in icon or 'blo' in icon:
                        unit_counts['belarus'] += 1
                    elif 'special' in icon:
                        unit_counts['special'] += 1
                    elif 'clown' in icon:
                        unit_counts['clown'] += 1
                    else:
                        unit_counts['other'] += 1

            elif geom_type in ('LineString', 'MultiLineString'):
                n_line += 1

        # Fill feature array
        idx = 0

        # Geometry counts (3)
        features[idx] = n_poly; idx += 1
        features[idx] = n_point; idx += 1
        features[idx] = n_line; idx += 1

        # Territory status polygon counts (4)
        features[idx] = polygon_counts['occupied']; idx += 1
        features[idx] = polygon_counts['liberated']; idx += 1
        features[idx] = polygon_counts['contested']; idx += 1
        features[idx] = polygon_counts['unknown']; idx += 1

        # Territory areas in km² (4)
        features[idx] = areas['occupied']; idx += 1
        features[idx] = areas['liberated']; idx += 1
        features[idx] = areas['contested']; idx += 1
        features[idx] = areas['total']; idx += 1

        # Area deltas (3) - calculated from previous observation
        if self.prev_areas is not None:
            features[idx] = areas['occupied'] - self.prev_areas['occupied']; idx += 1
            features[idx] = areas['liberated'] - self.prev_areas['liberated']; idx += 1
            features[idx] = areas['contested'] - self.prev_areas['contested']; idx += 1
        else:
            idx += 3
        self.prev_areas = areas.copy()

        # Arrow directions (8)
        for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
            features[idx] = arrow_dirs[d]; idx += 1

        # Arrow totals and derived (3)
        arrows_total = sum(arrow_dirs.values())
        features[idx] = arrows_total; idx += 1
        features[idx] = arrow_dirs['E'] + arrow_dirs['NE'] + arrow_dirs['SE']; idx += 1  # Eastward
        features[idx] = arrow_dirs['W'] + arrow_dirs['NW'] + arrow_dirs['SW']; idx += 1  # Westward

        # Unit counts (12)
        for utype in ['enemy', 'airport', 'headquarter', 'capital', 'naval', 'belarus', 'special', 'clown', 'other']:
            features[idx] = unit_counts[utype]; idx += 1
        units_total = sum(unit_counts.values())
        features[idx] = units_total; idx += 1
        features[idx] = units_total / max(n_poly, 1); idx += 1  # Units per polygon
        features[idx] = unit_counts['airport']; idx += 1  # Duplicate for airports_count

        # Geographic metrics (6)
        if all_lats and all_lons:
            features[idx] = np.mean(all_lats); idx += 1  # Centroid lat
            features[idx] = np.mean(all_lons); idx += 1  # Centroid lon
            features[idx] = np.std(all_lats) * 111 if len(all_lats) > 1 else 0; idx += 1  # Lat spread in km
            features[idx] = np.std(all_lons) * 74 if len(all_lons) > 1 else 0; idx += 1  # Lon spread in km
        else:
            idx += 4

        # Approximate frontline length (perimeter proxy)
        features[idx] = np.sqrt(areas['total']) * 4 if areas['total'] > 0 else 0; idx += 1

        # Territory fragmentation (more polygons = more fragmented)
        features[idx] = n_poly / max(areas['total'] / 1000, 1); idx += 1

        # Temporal features (3)
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            feb24 = datetime(2022, 2, 24)
            features[idx] = (dt - feb24).days; idx += 1
            features[idx] = dt.isocalendar()[1]; idx += 1  # Week of year
            features[idx] = dt.month; idx += 1
        except ValueError:
            idx += 3

        # Color breakdown (5)
        for color in ['green', 'red', 'tan', 'dark', 'other']:
            features[idx] = color_counts[color]; idx += 1

        # Derived metrics (4)
        total_area = areas['total'] if areas['total'] > 0 else 1
        features[idx] = areas['occupied'] / total_area; idx += 1  # Occupied ratio
        features[idx] = areas['liberated'] / total_area; idx += 1  # Liberated ratio
        features[idx] = arrows_total + units_total; idx += 1  # Activity intensity
        features[idx] = abs(features[11]) + abs(features[12]) + abs(features[13]); idx += 1  # Control change rate

        return features

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        if len(self.dates) > 1:
            dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in self.dates]
            deltas = [(dates_dt[i+1] - dates_dt[i]).days for i in range(len(dates_dt)-1)]
            resolution = np.mean(deltas)
        else:
            resolution = 1.0

        missing_rate = np.isnan(self.processed_data).mean()

        return DataStats(
            name="DeepState Frontline Data (EXPANDED)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# PERSONNEL DATA LOADER (NEW)
# =============================================================================

class PersonnelDataLoader:
    """
    Loads Russian personnel losses data.

    Extracts features including:
    - Cumulative personnel losses
    - Daily casualty rates
    - POW counts
    - Rolling averages
    - Acceleration metrics
    """

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 12 features
        self.feature_names = [
            # Cumulative (2)
            'personnel_cumulative', 'pow_cumulative',
            # Daily rates (2)
            'personnel_daily', 'pow_daily',
            # Rolling averages (3)
            'personnel_7day_avg', 'personnel_30day_avg', 'personnel_trend',
            # Derived metrics (5)
            'days_of_war', 'avg_daily_rate', 'acceleration',
            'personnel_per_100days', 'intensity_ratio',
        ]

    def load(self) -> 'PersonnelDataLoader':
        """Load raw data from JSON."""
        with open(self.data_path) as f:
            self.raw_data = json.load(f)
        return self

    def process(self) -> 'PersonnelDataLoader':
        """Process personnel data into feature arrays."""
        if self.raw_data is None:
            self.load()

        df = pd.DataFrame(self.raw_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Fill missing values
        df['personnel'] = df['personnel'].ffill().fillna(0)
        df['POW'] = df['POW'].ffill().fillna(0)

        n_days = len(df)
        n_features = len(self.feature_names)
        output = np.zeros((n_days, n_features), dtype=np.float32)

        personnel = df['personnel'].values
        pow_vals = df['POW'].values

        # Calculate daily changes
        personnel_daily = np.diff(personnel, prepend=personnel[0])
        pow_daily = np.diff(pow_vals, prepend=pow_vals[0])

        # Calculate rolling averages
        personnel_7day = pd.Series(personnel_daily).rolling(7, min_periods=1).mean().values
        personnel_30day = pd.Series(personnel_daily).rolling(30, min_periods=1).mean().values

        # Trend (7-day avg - 30-day avg, positive = accelerating)
        trend = personnel_7day - personnel_30day

        # Days of war
        feb24 = datetime(2022, 2, 24)
        days_of_war = np.array([(d - feb24).days for d in df['date']])

        # Average daily rate
        avg_daily_rate = personnel / np.maximum(days_of_war, 1)

        # Acceleration (change in daily rate)
        acceleration = np.diff(personnel_daily, prepend=0)

        # Personnel per 100 days
        personnel_per_100 = personnel / np.maximum(days_of_war / 100, 0.01)

        # Intensity ratio (current 7-day avg vs overall avg)
        intensity_ratio = personnel_7day / np.maximum(avg_daily_rate, 1)

        # Fill output array
        output[:, 0] = personnel  # Cumulative
        output[:, 1] = pow_vals
        output[:, 2] = personnel_daily
        output[:, 3] = pow_daily
        output[:, 4] = personnel_7day
        output[:, 5] = personnel_30day
        output[:, 6] = trend
        output[:, 7] = days_of_war
        output[:, 8] = avg_daily_rate
        output[:, 9] = acceleration
        output[:, 10] = personnel_per_100
        output[:, 11] = intensity_ratio

        self.dates = [d.strftime('%Y-%m-%d') for d in df['date']]
        self.processed_data = output

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 1.0
        missing_rate = np.isnan(self.processed_data).mean()

        return DataStats(
            name="Personnel Losses Data",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# VIINA TERRITORIAL CONTROL DATA LOADER (NEW - 91M records)
# =============================================================================

class VIINADataLoader(NormalizationMixin):
    """
    Loads VIINA (Violent Incident Information from News Articles) territorial control data.

    This dataset tracks daily territorial control status for ~33,000 Ukrainian localities
    from multiple sources (Wikipedia, DeepState, ISW, etc.).

    Features extracted:
    - Daily control status counts (UA/RU/contested)
    - Control changes (gains/losses)
    - Regional breakdowns
    - Source agreement metrics

    Note: This loader uses the NormalizationMixin to prevent data leakage.
    Call fit_normalization() with training indices before apply_normalization().
    """

    # War start date for filtering
    WAR_START = datetime(2022, 2, 24)

    # Feature groups for different normalization strategies
    LOCALITY_COUNT_FEATURES = [
        'localities_ua_control', 'localities_ru_control',
        'localities_contested', 'localities_unknown',
        'ua_control_7day_avg', 'ru_control_7day_avg', 'total_localities'
    ]
    PERCENTAGE_FEATURES = [
        'pct_ua_control', 'pct_ru_control', 'pct_contested',
        'sources_agree_pct', 'wiki_dsm_agree', 'wiki_isw_agree',
        'dsm_isw_agree', 'data_completeness'
    ]
    CHANGE_COUNT_FEATURES = [
        'localities_gained_ua', 'localities_lost_ua',
        'localities_gained_ru', 'localities_lost_ru',
        'front_activity_index'
    ]
    DERIVED_FEATURES = ['daily_change_7day_avg', 'control_volatility', 'control_momentum']

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR / "viina" / "extracted"
        self.raw_data = None
        self.processed_data = None
        self._raw_processed_data = None  # Store raw data before normalization
        self.dates = []
        self._norm_stats: Dict[str, Dict[str, float]] = {}
        self._normalization_fitted: bool = False
        # Feature list - 24 features
        self.feature_names = [
            # Control status counts (4)
            'localities_ua_control', 'localities_ru_control',
            'localities_contested', 'localities_unknown',
            # Control percentages (3)
            'pct_ua_control', 'pct_ru_control', 'pct_contested',
            # Daily changes (4)
            'localities_gained_ua', 'localities_lost_ua',
            'localities_gained_ru', 'localities_lost_ru',
            # Source agreement metrics (4)
            'sources_agree_pct', 'wiki_dsm_agree', 'wiki_isw_agree', 'dsm_isw_agree',
            # Rolling metrics (4)
            'ua_control_7day_avg', 'ru_control_7day_avg',
            'daily_change_7day_avg', 'control_volatility',
            # Derived metrics (5)
            'front_activity_index', 'control_momentum',
            'days_since_war_start', 'total_localities', 'data_completeness',
        ]

    def load(self) -> 'VIINADataLoader':
        """Load control data from extracted CSVs."""
        all_data = []

        # Load all control_latest files (2022-2026)
        for year in range(2022, 2027):
            fpath = self.data_dir / f"control_latest_{year}.csv"
            if fpath.exists():
                df = pd.read_csv(fpath, low_memory=False)
                all_data.append(df)

        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            # Parse date (format: YYYYMMDD as int)
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'].astype(str), format='%Y%m%d')
            # Filter to war period
            self.raw_data = self.raw_data[self.raw_data['date'] >= self.WAR_START]

        return self

    def process(self) -> 'VIINADataLoader':
        """Process raw data into daily aggregated features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Map status codes
        status_map = {'UA': 'ua', 'RU': 'ru', 'CONTESTED': 'contested', 'GZ': 'contested'}
        df['status_clean'] = df['status'].map(lambda x: status_map.get(x, 'unknown'))

        # Calculate source agreement
        df['wiki_dsm_agree'] = (df['status_wiki'] == df['status_dsm']).astype(int)
        df['wiki_isw_agree'] = (df['status_wiki'] == df['status_isw']).astype(int)
        df['dsm_isw_agree'] = (df['status_dsm'] == df['status_isw']).astype(int)

        # Aggregate by date
        daily_data = []
        prev_day_status = None

        for date, group in df.groupby('date'):
            row = {'date': date}
            total = len(group)

            # Control status counts
            status_counts = group['status_clean'].value_counts()
            row['localities_ua_control'] = status_counts.get('ua', 0)
            row['localities_ru_control'] = status_counts.get('ru', 0)
            row['localities_contested'] = status_counts.get('contested', 0)
            row['localities_unknown'] = status_counts.get('unknown', 0)

            # Percentages
            row['pct_ua_control'] = row['localities_ua_control'] / max(total, 1) * 100
            row['pct_ru_control'] = row['localities_ru_control'] / max(total, 1) * 100
            row['pct_contested'] = row['localities_contested'] / max(total, 1) * 100

            # Calculate daily changes (if we have previous day)
            if prev_day_status is not None:
                merged = group[['geonameid', 'status_clean']].merge(
                    prev_day_status[['geonameid', 'status_clean']],
                    on='geonameid',
                    suffixes=('_today', '_yesterday'),
                    how='inner'
                )
                changes = merged[merged['status_clean_today'] != merged['status_clean_yesterday']]

                row['localities_gained_ua'] = ((changes['status_clean_today'] == 'ua') &
                                               (changes['status_clean_yesterday'] != 'ua')).sum()
                row['localities_lost_ua'] = ((changes['status_clean_yesterday'] == 'ua') &
                                             (changes['status_clean_today'] != 'ua')).sum()
                row['localities_gained_ru'] = ((changes['status_clean_today'] == 'ru') &
                                               (changes['status_clean_yesterday'] != 'ru')).sum()
                row['localities_lost_ru'] = ((changes['status_clean_yesterday'] == 'ru') &
                                             (changes['status_clean_today'] != 'ru')).sum()
            else:
                row['localities_gained_ua'] = 0
                row['localities_lost_ua'] = 0
                row['localities_gained_ru'] = 0
                row['localities_lost_ru'] = 0

            # Source agreement
            row['sources_agree_pct'] = (group['wiki_dsm_agree'].mean() +
                                        group['wiki_isw_agree'].mean() +
                                        group['dsm_isw_agree'].mean()) / 3 * 100
            row['wiki_dsm_agree'] = group['wiki_dsm_agree'].mean() * 100
            row['wiki_isw_agree'] = group['wiki_isw_agree'].mean() * 100
            row['dsm_isw_agree'] = group['dsm_isw_agree'].mean() * 100

            # Temporal
            row['days_since_war_start'] = (date - self.WAR_START).days
            row['total_localities'] = total
            row['data_completeness'] = (total - row['localities_unknown']) / max(total, 1) * 100

            daily_data.append(row)
            prev_day_status = group[['geonameid', 'status_clean']].copy()

        daily_df = pd.DataFrame(daily_data).sort_values('date')

        # Calculate rolling metrics
        daily_df['ua_control_7day_avg'] = daily_df['localities_ua_control'].rolling(7, min_periods=1).mean()
        daily_df['ru_control_7day_avg'] = daily_df['localities_ru_control'].rolling(7, min_periods=1).mean()

        net_change = (daily_df['localities_gained_ua'] - daily_df['localities_lost_ua'])
        daily_df['daily_change_7day_avg'] = net_change.rolling(7, min_periods=1).mean()
        daily_df['control_volatility'] = net_change.rolling(7, min_periods=1).std().fillna(0)

        # Derived metrics
        daily_df['front_activity_index'] = (daily_df['localities_gained_ua'] + daily_df['localities_lost_ua'] +
                                            daily_df['localities_gained_ru'] + daily_df['localities_lost_ru'])
        daily_df['control_momentum'] = daily_df['daily_change_7day_avg'] / daily_df['control_volatility'].replace(0, 1)

        # Use domain-specific imputation for VIINA data (deepstate strategy - forward fill)
        # Territorial control is persistent
        daily_df = daily_df.set_index('date')
        date_range = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
        imputed_df, self._observation_mask = impute_domain_data(
            'deepstate', daily_df, date_range, self.feature_names
        )
        daily_df = imputed_df.reset_index()
        daily_df = daily_df.rename(columns={'index': 'date'})

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]

        # Store RAW (non-normalized) data for proper fit/transform pattern
        # This prevents data leakage - normalization should be fit on training data only
        self._raw_processed_data = daily_df[self.feature_names].values.astype(np.float32)

        # By default, store raw data as processed_data
        # Users should call fit_normalization() then apply_normalization() for proper ML workflow
        self.processed_data = self._raw_processed_data.copy()

        return self

    def fit_normalization(self, train_mask: np.ndarray) -> 'VIINADataLoader':
        """
        Fit normalization statistics using only training data.

        This method computes normalization statistics from training data only,
        preventing data leakage from test/validation sets.

        Parameters
        ----------
        train_mask : np.ndarray
            Boolean mask identifying training samples, or integer indices.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if self._raw_processed_data is None:
            self.process()

        # Get training data
        if isinstance(train_mask, np.ndarray) and train_mask.dtype == bool:
            train_data = self._raw_processed_data[train_mask]
        else:
            train_data = self._raw_processed_data[train_mask]

        self._norm_stats = {}

        # Compute statistics for each feature group from TRAINING data only
        max_localities = train_data[:, self.feature_names.index('total_localities')].max()
        if max_localities <= 0:
            max_localities = 33000  # Default

        max_days = 1500  # Expected max duration (~4 years of war)

        for i, feat in enumerate(self.feature_names):
            stats = {}

            if feat in self.LOCALITY_COUNT_FEATURES:
                stats['type'] = 'locality_count'
                stats['max_localities'] = float(max_localities)
            elif feat in self.PERCENTAGE_FEATURES:
                stats['type'] = 'percentage'
                stats['divisor'] = 100.0
            elif feat == 'days_since_war_start':
                stats['type'] = 'days'
                stats['max_days'] = float(max_days)
            elif feat in self.CHANGE_COUNT_FEATURES:
                stats['type'] = 'change_count'
                stats['max_change'] = 100.0
            elif feat in self.DERIVED_FEATURES:
                stats['type'] = 'derived'
                col = train_data[:, i]
                stats['mean'] = float(np.nanmean(col))
                stats['std'] = float(np.nanstd(col))
                if stats['std'] == 0:
                    stats['std'] = 1.0

            self._norm_stats[feat] = stats

        self._normalization_fitted = True
        return self

    def apply_normalization(self, data: np.ndarray = None) -> np.ndarray:
        """
        Apply fitted normalization to data.

        Parameters
        ----------
        data : np.ndarray, optional
            Data to normalize. If None, normalizes self._raw_processed_data.

        Returns
        -------
        np.ndarray
            Normalized data with values in [0, 1] range.

        Raises
        ------
        RuntimeError
            If fit_normalization has not been called first.
        """
        if not self._normalization_fitted:
            raise RuntimeError(
                "Normalization not fitted. Call fit_normalization(train_mask) first."
            )

        if data is None:
            data = self._raw_processed_data

        normalized = data.copy().astype(np.float32)

        for i, feat in enumerate(self.feature_names):
            stats = self._norm_stats.get(feat, {})
            norm_type = stats.get('type', 'none')

            if norm_type == 'locality_count':
                normalized[:, i] = data[:, i] / stats['max_localities']
            elif norm_type == 'percentage':
                normalized[:, i] = data[:, i] / stats['divisor']
            elif norm_type == 'days':
                normalized[:, i] = data[:, i] / stats['max_days']
            elif norm_type == 'change_count':
                normalized[:, i] = np.clip(data[:, i] / stats['max_change'], 0, 1)
            elif norm_type == 'derived':
                col = data[:, i]
                normalized[:, i] = np.clip(
                    (col - stats['mean']) / (3 * stats['std']) + 0.5, 0, 1
                )

        # Update processed_data with normalized values
        self.processed_data = normalized

        return normalized

    def get_raw_data(self) -> np.ndarray:
        """
        Return the raw (non-normalized) processed data.

        Returns
        -------
        np.ndarray
            Raw processed data before normalization.
        """
        if self._raw_processed_data is None:
            self.process()
        return self._raw_processed_data.copy()

    def get_observation_mask(self) -> Optional[pd.DataFrame]:
        """Return the observation mask indicating observed vs imputed values."""
        return getattr(self, '_observation_mask', None)

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 1.0
        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="VIINA Territorial Control",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# HDX CONFLICT EVENTS DATA LOADER (NEW - 20K records)
# =============================================================================

class HDXConflictDataLoader:
    """
    Loads HDX (Humanitarian Data Exchange) conflict events data for Ukraine.

    Monthly aggregated conflict event counts and fatalities by region and event type.
    """

    WAR_START = datetime(2022, 2, 24)

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "hdx" / "ukraine" / "conflict_events_2022_present.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 18 features
        self.feature_names = [
            # Event counts by type (6)
            'events_total', 'events_civilian_targeting', 'events_battles',
            'events_explosions', 'events_protests', 'events_other',
            # Fatalities (3)
            'fatalities_total', 'fatalities_per_event', 'fatalities_max_event',
            # Regional breakdown (5)
            'events_donetsk', 'events_kharkiv', 'events_kherson',
            'events_zaporizhzhia', 'events_other_regions',
            # Derived metrics (4)
            'intensity_index', 'regional_spread', 'days_in_period',
            'events_per_day',
        ]

    def load(self) -> 'HDXConflictDataLoader':
        """Load raw data from CSV."""
        self.raw_data = pd.read_csv(self.data_path)
        # Parse dates
        self.raw_data['date_start'] = pd.to_datetime(self.raw_data['reference_period_start'])
        self.raw_data['date_end'] = pd.to_datetime(self.raw_data['reference_period_end'])
        # Filter to war period
        self.raw_data = self.raw_data[self.raw_data['date_start'] >= self.WAR_START]
        return self

    def process(self) -> 'HDXConflictDataLoader':
        """Process raw data into period-based features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Aggregate by period (month)
        period_data = []

        for period_start, group in df.groupby('date_start'):
            row = {'date': period_start}

            period_end = group['date_end'].max()
            days_in_period = max((period_end - period_start).days, 1)

            # Event counts by type
            row['events_total'] = group['events'].sum()
            event_types = group.groupby('event_type')['events'].sum()
            row['events_civilian_targeting'] = event_types.get('civilian_targeting', 0)
            row['events_battles'] = event_types.get('battles', 0)
            row['events_explosions'] = event_types.get('explosions_remote_violence', 0)
            row['events_protests'] = event_types.get('protests', 0)
            row['events_other'] = row['events_total'] - sum([
                row['events_civilian_targeting'], row['events_battles'],
                row['events_explosions'], row['events_protests']
            ])

            # Fatalities
            row['fatalities_total'] = group['fatalities'].sum()
            row['fatalities_per_event'] = row['fatalities_total'] / max(row['events_total'], 1)
            row['fatalities_max_event'] = group['fatalities'].max()

            # Regional breakdown
            region_events = group.groupby('admin1_name')['events'].sum()
            row['events_donetsk'] = region_events.get('Donetska', 0)
            row['events_kharkiv'] = region_events.get('Kharkivska', 0)
            row['events_kherson'] = region_events.get('Khersonska', 0)
            row['events_zaporizhzhia'] = region_events.get('Zaporizka', 0)
            row['events_other_regions'] = row['events_total'] - sum([
                row['events_donetsk'], row['events_kharkiv'],
                row['events_kherson'], row['events_zaporizhzhia']
            ])

            # Derived metrics
            row['intensity_index'] = row['events_total'] * (row['fatalities_per_event'] + 1)
            row['regional_spread'] = group['admin1_name'].nunique()
            row['days_in_period'] = days_in_period
            row['events_per_day'] = row['events_total'] / days_in_period

            period_data.append(row)

        daily_df = pd.DataFrame(period_data).sort_values('date')
        daily_df = daily_df.fillna(0)

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]
        self.processed_data = daily_df[self.feature_names].values.astype(np.float32)

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        if len(self.dates) > 1:
            dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in self.dates]
            deltas = [(dates_dt[i+1] - dates_dt[i]).days for i in range(len(dates_dt)-1)]
            resolution = np.mean(deltas) if deltas else 30.0
        else:
            resolution = 30.0

        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="HDX Conflict Events",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# HDX FOOD PRICES DATA LOADER (NEW - 33K records)
# =============================================================================

class HDXFoodPricesDataLoader:
    """
    Loads HDX food price data for Ukraine.

    Tracks commodity prices across markets, useful for economic impact analysis.
    """

    WAR_START = datetime(2022, 2, 24)

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "hdx" / "ukraine" / "food_prices_2022_present.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 20 features
        self.feature_names = [
            # Price statistics (6)
            'avg_price', 'median_price', 'min_price', 'max_price',
            'price_std', 'price_range',
            # Category breakdown (5)
            'cereals_avg', 'vegetables_avg', 'meat_avg', 'dairy_avg', 'oils_avg',
            # Price changes (3)
            'price_change_pct', 'price_7day_trend', 'price_volatility',
            # Market coverage (3)
            'markets_reporting', 'commodities_tracked', 'regions_covered',
            # Derived (3)
            'inflation_proxy', 'food_security_index', 'price_anomaly_score',
        ]

    def load(self) -> 'HDXFoodPricesDataLoader':
        """Load raw data from CSV."""
        self.raw_data = pd.read_csv(self.data_path)
        self.raw_data['date_start'] = pd.to_datetime(self.raw_data['reference_period_start'])
        self.raw_data = self.raw_data[self.raw_data['date_start'] >= self.WAR_START]
        return self

    def process(self) -> 'HDXFoodPricesDataLoader':
        """Process raw data into period-based features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Map commodity categories
        category_map = {
            'cereals and tubers': 'cereals',
            'vegetables and fruits': 'vegetables',
            'meat, fish and eggs': 'meat',
            'milk and dairy': 'dairy',
            'oil and fats': 'oils'
        }
        df['category'] = df['commodity_category'].map(lambda x: category_map.get(x, 'other'))

        # Aggregate by period
        period_data = []
        prev_avg = None
        price_history = []

        for period_start, group in df.groupby('date_start'):
            row = {'date': period_start}

            prices = group['price'].dropna()

            # Price statistics
            row['avg_price'] = prices.mean() if len(prices) > 0 else 0
            row['median_price'] = prices.median() if len(prices) > 0 else 0
            row['min_price'] = prices.min() if len(prices) > 0 else 0
            row['max_price'] = prices.max() if len(prices) > 0 else 0
            row['price_std'] = prices.std() if len(prices) > 1 else 0
            row['price_range'] = row['max_price'] - row['min_price']

            # Category averages
            cat_prices = group.groupby('category')['price'].mean()
            row['cereals_avg'] = cat_prices.get('cereals', 0)
            row['vegetables_avg'] = cat_prices.get('vegetables', 0)
            row['meat_avg'] = cat_prices.get('meat', 0)
            row['dairy_avg'] = cat_prices.get('dairy', 0)
            row['oils_avg'] = cat_prices.get('oils', 0)

            # Price changes
            if prev_avg is not None and prev_avg > 0:
                row['price_change_pct'] = (row['avg_price'] - prev_avg) / prev_avg * 100
            else:
                row['price_change_pct'] = 0

            price_history.append(row['avg_price'])
            if len(price_history) >= 7:
                recent = price_history[-7:]
                row['price_7day_trend'] = (recent[-1] - recent[0]) / max(recent[0], 1) * 100
                row['price_volatility'] = np.std(recent)
            else:
                row['price_7day_trend'] = 0
                row['price_volatility'] = 0

            # Coverage metrics
            row['markets_reporting'] = group['market_name'].nunique()
            row['commodities_tracked'] = group['commodity_name'].nunique()
            row['regions_covered'] = group['admin1_name'].nunique()

            # Derived metrics
            row['inflation_proxy'] = row['price_change_pct']
            row['food_security_index'] = 100 - min(abs(row['price_change_pct']) * 2, 50) - min(row['price_volatility'], 50)
            if len(price_history) > 1:
                mean_hist = np.mean(price_history)
                std_hist = np.std(price_history) if len(price_history) > 1 else 1
                row['price_anomaly_score'] = abs(row['avg_price'] - mean_hist) / max(std_hist, 1)
            else:
                row['price_anomaly_score'] = 0

            period_data.append(row)
            prev_avg = row['avg_price']

        daily_df = pd.DataFrame(period_data).sort_values('date')
        daily_df = daily_df.fillna(0)

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]
        self.processed_data = daily_df[self.feature_names].values.astype(np.float32)

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 30.0  # Monthly
        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="HDX Food Prices",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# HDX RAINFALL DATA LOADER (NEW - 57K records)
# =============================================================================

class HDXRainfallDataLoader:
    """
    Loads HDX rainfall data for Ukraine.

    Dekadal (10-day) rainfall measurements with anomaly analysis.
    Useful for agricultural impact and seasonal analysis.
    """

    WAR_START = datetime(2022, 2, 24)

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "hdx" / "ukraine" / "rainfall_2022_present.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 16 features
        self.feature_names = [
            # Rainfall statistics (5)
            'rainfall_mean', 'rainfall_median', 'rainfall_max', 'rainfall_min', 'rainfall_std',
            # Anomaly metrics (4)
            'anomaly_pct_mean', 'anomaly_pct_std', 'above_normal_pct', 'below_normal_pct',
            # Long-term comparison (3)
            'lta_mean', 'rainfall_vs_lta_ratio', 'deviation_from_normal',
            # Coverage (2)
            'regions_reporting', 'total_pixels',
            # Derived (2)
            'drought_risk_index', 'flood_risk_index',
        ]

    def load(self) -> 'HDXRainfallDataLoader':
        """Load raw data from CSV."""
        self.raw_data = pd.read_csv(self.data_path)
        self.raw_data['date_start'] = pd.to_datetime(self.raw_data['reference_period_start'])
        self.raw_data = self.raw_data[self.raw_data['date_start'] >= self.WAR_START]
        return self

    def process(self) -> 'HDXRainfallDataLoader':
        """Process raw data into period-based features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Aggregate by period (dekad)
        period_data = []

        for period_start, group in df.groupby('date_start'):
            row = {'date': period_start}

            rainfall = group['rainfall'].dropna()
            anomaly = group['rainfall_anomaly_pct'].dropna()
            lta = group['rainfall_long_term_average'].dropna()

            # Rainfall statistics
            row['rainfall_mean'] = rainfall.mean() if len(rainfall) > 0 else 0
            row['rainfall_median'] = rainfall.median() if len(rainfall) > 0 else 0
            row['rainfall_max'] = rainfall.max() if len(rainfall) > 0 else 0
            row['rainfall_min'] = rainfall.min() if len(rainfall) > 0 else 0
            row['rainfall_std'] = rainfall.std() if len(rainfall) > 1 else 0

            # Anomaly metrics
            row['anomaly_pct_mean'] = anomaly.mean() if len(anomaly) > 0 else 0
            row['anomaly_pct_std'] = anomaly.std() if len(anomaly) > 1 else 0
            row['above_normal_pct'] = (anomaly > 100).mean() * 100 if len(anomaly) > 0 else 0
            row['below_normal_pct'] = (anomaly < 100).mean() * 100 if len(anomaly) > 0 else 0

            # Long-term comparison
            row['lta_mean'] = lta.mean() if len(lta) > 0 else 0
            row['rainfall_vs_lta_ratio'] = row['rainfall_mean'] / max(row['lta_mean'], 0.1)
            row['deviation_from_normal'] = row['rainfall_mean'] - row['lta_mean']

            # Coverage
            row['regions_reporting'] = group['admin1_name'].nunique()
            row['total_pixels'] = group['number_pixels'].sum()

            # Risk indices
            row['drought_risk_index'] = max(0, 100 - row['anomaly_pct_mean']) if row['anomaly_pct_mean'] < 100 else 0
            row['flood_risk_index'] = max(0, row['anomaly_pct_mean'] - 100) if row['anomaly_pct_mean'] > 100 else 0

            period_data.append(row)

        daily_df = pd.DataFrame(period_data).sort_values('date')
        daily_df = daily_df.fillna(0)

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]
        self.processed_data = daily_df[self.feature_names].values.astype(np.float32)

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 10.0  # Dekadal
        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="HDX Rainfall Data",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# IOM DISPLACEMENT DATA LOADER (NEW - 29K records)
# =============================================================================

class IOMDisplacementDataLoader:
    """
    Loads IOM (International Organization for Migration) displacement data.

    Tracks internally displaced persons (IDPs) across Ukraine by region.
    """

    WAR_START = datetime(2022, 2, 24)

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or DATA_DIR / "iom" / "ukr-iom-dtm-from-api-admin-0-to-1.csv"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 18 features
        self.feature_names = [
            # Total IDPs (3)
            'total_idps', 'idps_male', 'idps_female',
            # Regional breakdown (6)
            'idps_kyiv', 'idps_lviv', 'idps_dnipro',
            'idps_kharkiv', 'idps_zaporizhzhia', 'idps_other',
            # Origin tracking (3)
            'from_donetsk', 'from_luhansk', 'from_other_conflict',
            # Metrics (3)
            'regions_receiving', 'avg_per_region', 'max_single_region',
            # Derived (3)
            'displacement_intensity', 'gender_ratio', 'round_number',
        ]

    def load(self) -> 'IOMDisplacementDataLoader':
        """Load raw data from CSV."""
        # Skip the HXL tag row (row 1)
        self.raw_data = pd.read_csv(self.data_path, skiprows=[1])
        self.raw_data['date'] = pd.to_datetime(self.raw_data['reportingDate'])
        self.raw_data = self.raw_data[self.raw_data['date'] >= self.WAR_START]
        return self

    def process(self) -> 'IOMDisplacementDataLoader':
        """Process raw data into round-based features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Convert numeric columns
        for col in ['numPresentIdpInd', 'numberMales', 'numberFemales', 'roundNumber']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Aggregate by reporting date
        period_data = []

        for date, group in df.groupby('date'):
            row = {'date': date}

            # Total IDPs
            row['total_idps'] = group['numPresentIdpInd'].sum()
            row['idps_male'] = group['numberMales'].sum()
            row['idps_female'] = group['numberFemales'].sum()

            # Regional breakdown (destination)
            region_idps = group.groupby('admin1Name')['numPresentIdpInd'].sum()
            row['idps_kyiv'] = region_idps.get('Kyivska', 0) + region_idps.get('Kyiv', 0)
            row['idps_lviv'] = region_idps.get('Lvivska', 0)
            row['idps_dnipro'] = region_idps.get('Dnipropetrovska', 0)
            row['idps_kharkiv'] = region_idps.get('Kharkivska', 0)
            row['idps_zaporizhzhia'] = region_idps.get('Zaporizka', 0)
            row['idps_other'] = row['total_idps'] - sum([
                row['idps_kyiv'], row['idps_lviv'], row['idps_dnipro'],
                row['idps_kharkiv'], row['idps_zaporizhzhia']
            ])

            # Origin tracking
            if 'idpOriginAdmin1Name' in group.columns:
                origin_idps = group.groupby('idpOriginAdmin1Name')['numPresentIdpInd'].sum()
                row['from_donetsk'] = origin_idps.get('Donetska', 0)
                row['from_luhansk'] = origin_idps.get('Luhanska', 0)
                row['from_other_conflict'] = row['total_idps'] - row['from_donetsk'] - row['from_luhansk']
            else:
                row['from_donetsk'] = 0
                row['from_luhansk'] = 0
                row['from_other_conflict'] = 0

            # Metrics
            row['regions_receiving'] = group['admin1Name'].nunique()
            row['avg_per_region'] = row['total_idps'] / max(row['regions_receiving'], 1)
            row['max_single_region'] = region_idps.max() if len(region_idps) > 0 else 0

            # Derived
            row['displacement_intensity'] = row['total_idps'] / 1000  # Normalized
            row['gender_ratio'] = row['idps_female'] / max(row['idps_male'], 1)
            row['round_number'] = group['roundNumber'].max() if 'roundNumber' in group.columns else 0

            period_data.append(row)

        daily_df = pd.DataFrame(period_data).sort_values('date')
        daily_df = daily_df.fillna(0)

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]
        self.processed_data = daily_df[self.feature_names].values.astype(np.float32)

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        if len(self.dates) > 1:
            dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in self.dates]
            deltas = [(dates_dt[i+1] - dates_dt[i]).days for i in range(len(dates_dt)-1)]
            resolution = np.mean(deltas) if deltas else 30.0
        else:
            resolution = 30.0

        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="IOM Displacement Data",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# GOOGLE MOBILITY DATA LOADER (NEW - Wayback Archive)
# =============================================================================

class GoogleMobilityDataLoader:
    """
    Loads Google Community Mobility Reports from Wayback Machine archives.

    Tracks mobility changes across different location types.

    NOTE: Ukraine data in Google Mobility Reports ends on 2022-02-23 (one day
    before the invasion). This provides pre-war baseline data but no wartime
    mobility tracking. The loader includes data from 2022-01-01 onwards to
    capture the immediate pre-war baseline for comparison with other datasets.
    """

    # Data ends 2022-02-23, so we capture pre-war baseline (Jan-Feb 2022)
    DATA_START = datetime(2022, 1, 1)
    WAR_START = datetime(2022, 2, 24)  # For reference - data ends before this

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR / "wayback_archives" / "google_mobility" / "ukraine_only"
        self.raw_data = None
        self.processed_data = None
        self.dates = []
        # Feature list - 18 features
        self.feature_names = [
            # Mobility categories (6)
            'retail_recreation', 'grocery_pharmacy', 'parks',
            'transit_stations', 'workplaces', 'residential',
            # Regional averages (6)
            'kyiv_avg', 'lviv_avg', 'dnipro_avg',
            'kharkiv_avg', 'odesa_avg', 'national_avg',
            # Derived metrics (6)
            'economic_activity_index', 'social_distancing_index',
            'mobility_volatility', 'regional_disparity',
            'pre_war_baseline', 'days_to_invasion',
        ]

    def load(self) -> 'GoogleMobilityDataLoader':
        """Load all Ukraine mobility CSV files."""
        all_data = []

        for fpath in sorted(self.data_dir.glob("ukraine_*.csv")):
            try:
                df = pd.read_csv(fpath)
                all_data.append(df)
            except Exception:
                continue

        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
            # Filter to capture pre-war baseline (Jan-Feb 2022)
            # Note: Data ends 2022-02-23, no post-invasion data exists
            self.raw_data = self.raw_data[self.raw_data['date'] >= self.DATA_START]
            # Deduplicate
            self.raw_data = self.raw_data.drop_duplicates(subset=['date', 'sub_region_1'])

        return self

    def process(self) -> 'GoogleMobilityDataLoader':
        """Process raw data into daily national features."""
        if self.raw_data is None:
            self.load()

        if self.raw_data is None or len(self.raw_data) == 0:
            self.processed_data = np.zeros((0, len(self.feature_names)), dtype=np.float32)
            self.dates = []
            return self

        df = self.raw_data.copy()

        # Column mapping
        mobility_cols = {
            'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
            'grocery_and_pharmacy_percent_change_from_baseline': 'grocery_pharmacy',
            'parks_percent_change_from_baseline': 'parks',
            'transit_stations_percent_change_from_baseline': 'transit_stations',
            'workplaces_percent_change_from_baseline': 'workplaces',
            'residential_percent_change_from_baseline': 'residential',
        }

        # Aggregate by date
        daily_data = []
        baseline_values = None

        for date, group in df.groupby('date'):
            row = {'date': date}

            # National averages for each mobility type
            for orig_col, new_col in mobility_cols.items():
                if orig_col in group.columns:
                    row[new_col] = group[orig_col].mean()
                else:
                    row[new_col] = 0

            # Regional breakdown
            region_data = group.groupby('sub_region_1')
            for region, col in [('Kyiv City', 'kyiv_avg'), ('Lviv Oblast', 'lviv_avg'),
                               ('Dnipropetrovsk Oblast', 'dnipro_avg'),
                               ('Kharkiv Oblast', 'kharkiv_avg'), ('Odessa Oblast', 'odesa_avg')]:
                try:
                    reg_group = group[group['sub_region_1'] == region]
                    if len(reg_group) > 0:
                        row[col] = sum(reg_group[c].mean() for c in mobility_cols.keys() if c in reg_group.columns) / len(mobility_cols)
                    else:
                        row[col] = 0
                except Exception:
                    row[col] = 0

            row['national_avg'] = sum(row[c] for c in mobility_cols.values()) / len(mobility_cols)

            # Derived metrics
            # Economic activity: retail + workplaces + transit
            row['economic_activity_index'] = (row['retail_recreation'] + row['workplaces'] + row['transit_stations']) / 3
            # Social distancing: residential increase, others decrease
            row['social_distancing_index'] = row['residential'] - row['national_avg']

            # Volatility (requires history)
            if baseline_values is None:
                baseline_values = row['national_avg']
                row['mobility_volatility'] = 0
            else:
                row['mobility_volatility'] = abs(row['national_avg'] - baseline_values)

            # Pre-war baseline indicator (1.0 for all data since it's pre-war)
            row['pre_war_baseline'] = 1.0
            # Days until invasion (negative = before invasion)
            row['days_to_invasion'] = (self.WAR_START - date).days

            # Regional disparity
            regional_vals = [row['kyiv_avg'], row['lviv_avg'], row['dnipro_avg'],
                           row['kharkiv_avg'], row['odesa_avg']]
            row['regional_disparity'] = np.std([v for v in regional_vals if v != 0]) if any(regional_vals) else 0

            daily_data.append(row)

        daily_df = pd.DataFrame(daily_data).sort_values('date')
        daily_df = daily_df.fillna(0)

        self.dates = [d.strftime('%Y-%m-%d') for d in daily_df['date']]
        self.processed_data = daily_df[self.feature_names].values.astype(np.float32)

        return self

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded data."""
        if self.processed_data is None:
            self.process()

        resolution = 1.0  # Daily
        missing_rate = np.isnan(self.processed_data).mean() if len(self.processed_data) > 0 else 0

        return DataStats(
            name="Google Mobility (Wayback Archive)",
            n_observations=len(self.dates),
            n_features=len(self.feature_names),
            date_range=(self.dates[0], self.dates[-1]) if self.dates else ('', ''),
            temporal_resolution_days=resolution,
            missing_rate=missing_rate,
            feature_names=self.feature_names,
            sample_data=self.processed_data[:5] if len(self.processed_data) > 0 else np.array([]),
            sample_dates=self.dates[:5]
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Inspect OSINT data sources')
    parser.add_argument('--source', default='all',
                       choices=['sentinel', 'deepstate', 'equipment', 'firms', 'ucdp', 'personnel',
                               'viina', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom', 'mobility', 'all'],
                       help='Data source to inspect')
    parser.add_argument('--inspect', action='store_true', help='Print detailed statistics')
    args = parser.parse_args()

    loaders = {
        'sentinel': SentinelDataLoader,
        'deepstate': DeepStateDataLoader,
        'equipment': EquipmentDataLoader,
        'firms': FIRMSDataLoader,
        'ucdp': UCDPDataLoader,
        'personnel': PersonnelDataLoader,
        'viina': VIINADataLoader,
        'hdx_conflict': HDXConflictDataLoader,
        'hdx_food': HDXFoodPricesDataLoader,
        'hdx_rainfall': HDXRainfallDataLoader,
        'iom': IOMDisplacementDataLoader,
        'mobility': GoogleMobilityDataLoader,
    }

    if args.source == 'all':
        sources = loaders.keys()
    else:
        sources = [args.source]

    total_features = 0
    print("\n" + "=" * 70)
    print("EXPANDED DATA LOADER SUMMARY")
    print("=" * 70)

    for source in sources:
        try:
            loader = loaders[source]()
            stats = loader.get_stats()
            stats.print_summary()
            total_features += stats.n_features
        except Exception as e:
            print(f"\nError loading {source}: {e}")

    print("\n" + "=" * 70)
    print(f"TOTAL FEATURES ACROSS ALL SOURCES: {total_features}")
    print("=" * 70)


if __name__ == "__main__":
    main()
