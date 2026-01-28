"""
FIRMS (Fire Information for Resource Management System) Spatial Data Loader

Extracts spatially-tiled fire hotspot features from NASA FIRMS data including:
- Regional fire counts (hotspots per geographic tile)
- Regional brightness statistics (mean, max fire radiative power)
- Temporal patterns (day/night distribution)

Data source: NASA FIRMS VIIRS (2022-02-24 to present)
- 245,456 archive hotspots + NRT updates
- 89.9% temporal coverage (1182 of 1315 days)
- Strong conflict correlation: 35% Donbas, 23% South

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import centralized paths
from config.paths import DATA_DIR, FIRMS_DIR

# Import region definitions
try:
    from .deepstate_spatial_loader import UKRAINE_REGIONS, assign_region
except ImportError:
    from analysis.loaders.deepstate_spatial_loader import UKRAINE_REGIONS, assign_region


# =============================================================================
# CONSTANTS
# =============================================================================

# FIRMS data files
FIRMS_ARCHIVE_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
FIRMS_NRT_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_nrt_SV-C2_706038.csv"

# Feature columns in FIRMS data
FIRMS_COLUMNS = [
    'latitude', 'longitude', 'brightness', 'scan', 'track',
    'acq_date', 'acq_time', 'satellite', 'confidence',
    'bright_t31', 'frp', 'daynight'
]


# =============================================================================
# MAIN LOADER CLASS
# =============================================================================

class FIRMSSpatialLoader:
    """
    Loader for NASA FIRMS fire hotspot data with spatial tiling.

    Features extracted per region (6 regions):
    - fire_count: Number of hotspots
    - fire_brightness_mean: Mean brightness temperature (K)
    - fire_brightness_max: Maximum brightness temperature (K)
    - fire_frp_mean: Mean fire radiative power (MW)
    - fire_frp_sum: Total fire radiative power (MW)
    - fire_day_ratio: Proportion of daytime detections

    Total: 6 regions × 6 features = 36 spatial features per day

    Usage:
        loader = FIRMSSpatialLoader()
        df = loader.load_daily_features(start_date, end_date)
    """

    def __init__(
        self,
        archive_file: Optional[Path] = None,
        nrt_file: Optional[Path] = None,
    ):
        """
        Initialize the FIRMS spatial loader.

        Args:
            archive_file: Path to FIRMS archive CSV
            nrt_file: Path to FIRMS near-real-time CSV
        """
        self.archive_file = archive_file or FIRMS_ARCHIVE_FILE
        self.nrt_file = nrt_file or FIRMS_NRT_FILE

        self._data: Optional[pd.DataFrame] = None

    def _load_raw_data(self) -> pd.DataFrame:
        """Load and combine archive and NRT data."""
        if self._data is not None:
            return self._data

        dfs = []

        # Load archive
        if self.archive_file.exists():
            archive_df = pd.read_csv(self.archive_file)
            archive_df['source'] = 'archive'
            dfs.append(archive_df)
            print(f"  Loaded {len(archive_df):,} archive hotspots")
        else:
            warnings.warn(f"FIRMS archive not found: {self.archive_file}")

        # Load NRT
        if self.nrt_file.exists():
            nrt_df = pd.read_csv(self.nrt_file)
            nrt_df['source'] = 'nrt'
            dfs.append(nrt_df)
            print(f"  Loaded {len(nrt_df):,} NRT hotspots")
        else:
            warnings.warn(f"FIRMS NRT not found: {self.nrt_file}")

        if not dfs:
            return pd.DataFrame()

        # Combine
        self._data = pd.concat(dfs, ignore_index=True)

        # Parse dates
        self._data['acq_date'] = pd.to_datetime(self._data['acq_date'], errors='coerce')
        self._data = self._data.dropna(subset=['acq_date', 'latitude', 'longitude'])

        # Assign regions
        self._data['region'] = self._data.apply(
            lambda row: assign_region(row['latitude'], row['longitude']),
            axis=1
        )

        # Filter to Ukraine bounds (rough)
        ukraine_mask = (
            (self._data['latitude'] >= 44.0) & (self._data['latitude'] <= 53.0) &
            (self._data['longitude'] >= 22.0) & (self._data['longitude'] <= 41.0)
        )
        self._data = self._data[ukraine_mask].copy()

        print(f"  Total hotspots in Ukraine: {len(self._data):,}")

        return self._data

    def load_daily_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute daily aggregated features per region.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with daily spatial features
        """
        print("Loading FIRMS spatial features...")
        data = self._load_raw_data()

        if data.empty:
            return pd.DataFrame()

        # Filter date range
        if start_date:
            data = data[data['acq_date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['acq_date'] <= pd.to_datetime(end_date)]

        if data.empty:
            return pd.DataFrame()

        # Get date range
        min_date = data['acq_date'].min()
        max_date = data['acq_date'].max()
        date_range = pd.date_range(min_date, max_date, freq='D')

        regions = list(UKRAINE_REGIONS.keys())
        records = []

        print(f"  Computing daily features for {len(date_range)} days...")

        for date in date_range:
            date_val = date.date()
            day_data = data[data['acq_date'].dt.date == date_val]

            record = {'date': date}

            for region in regions:
                region_data = day_data[day_data['region'] == region]

                if len(region_data) > 0:
                    record[f'fire_count_{region}'] = len(region_data)
                    record[f'fire_brightness_mean_{region}'] = region_data['brightness'].mean()
                    record[f'fire_brightness_max_{region}'] = region_data['brightness'].max()

                    if 'frp' in region_data.columns:
                        frp_valid = region_data['frp'].dropna()
                        if len(frp_valid) > 0:
                            record[f'fire_frp_mean_{region}'] = frp_valid.mean()
                            record[f'fire_frp_sum_{region}'] = frp_valid.sum()
                        else:
                            record[f'fire_frp_mean_{region}'] = 0.0
                            record[f'fire_frp_sum_{region}'] = 0.0
                    else:
                        record[f'fire_frp_mean_{region}'] = 0.0
                        record[f'fire_frp_sum_{region}'] = 0.0

                    if 'daynight' in region_data.columns:
                        day_count = (region_data['daynight'] == 'D').sum()
                        record[f'fire_day_ratio_{region}'] = day_count / len(region_data)
                    else:
                        record[f'fire_day_ratio_{region}'] = 0.5
                else:
                    # No fires in this region on this day
                    record[f'fire_count_{region}'] = 0
                    record[f'fire_brightness_mean_{region}'] = 0.0
                    record[f'fire_brightness_max_{region}'] = 0.0
                    record[f'fire_frp_mean_{region}'] = 0.0
                    record[f'fire_frp_sum_{region}'] = 0.0
                    record[f'fire_day_ratio_{region}'] = 0.0

            # Aggregate totals
            record['fire_count_total'] = len(day_data)
            if len(day_data) > 0:
                record['fire_brightness_mean_total'] = day_data['brightness'].mean()
                if 'frp' in day_data.columns:
                    record['fire_frp_sum_total'] = day_data['frp'].sum()
                else:
                    record['fire_frp_sum_total'] = 0.0
            else:
                record['fire_brightness_mean_total'] = 0.0
                record['fire_frp_sum_total'] = 0.0

            records.append(record)

        result_df = pd.DataFrame(records)
        print(f"  Generated {len(result_df)} daily records with {len(result_df.columns)} features")

        return result_df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        regions = list(UKRAINE_REGIONS.keys())

        features = []
        for region in regions:
            features.extend([
                f'fire_count_{region}',
                f'fire_brightness_mean_{region}',
                f'fire_brightness_max_{region}',
                f'fire_frp_mean_{region}',
                f'fire_frp_sum_{region}',
                f'fire_day_ratio_{region}',
            ])

        # Totals
        features.extend([
            'fire_count_total',
            'fire_brightness_mean_total',
            'fire_frp_sum_total',
        ])

        return features


# =============================================================================
# INTEGRATION FUNCTIONS FOR MODULAR DATA SYSTEM
# =============================================================================

def load_firms_tiled(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS data with regional tiling for modular data system.

    Args:
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Tuple of (DataFrame with tiled features, observation mask array)
    """
    loader = FIRMSSpatialLoader()
    df = loader.load_daily_features(start_date, end_date)

    if df.empty:
        return pd.DataFrame(), np.array([])

    # Create observation mask (1 where we have any fires)
    mask = (df['fire_count_total'] > 0).astype(float).values

    return df, mask


def load_firms_aggregated(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS data with aggregated (non-tiled) features.

    Args:
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        Tuple of (DataFrame with aggregated features, observation mask array)
    """
    loader = FIRMSSpatialLoader()
    df = loader.load_daily_features(start_date, end_date)

    if df.empty:
        return pd.DataFrame(), np.array([])

    # Keep only aggregated columns
    keep_cols = ['date', 'fire_count_total', 'fire_brightness_mean_total', 'fire_frp_sum_total']
    df = df[[c for c in keep_cols if c in df.columns]]

    # Create observation mask
    mask = (df['fire_count_total'] > 0).astype(float).values

    return df, mask


# =============================================================================
# REGISTRATION WITH LOADER REGISTRY
# =============================================================================

FIRMS_SPATIAL_LOADER = {
    'name': 'firms_spatial',
    'loader_fn': load_firms_tiled,
    'resolution': 'daily',
    'feature_count': 39,  # 6 regions × 6 features + 3 totals
    'description': 'NASA FIRMS fire hotspots with regional tiling',
    'spatial_modes': ['aggregated', 'tiled'],
}


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("FIRMS Spatial Loader Test")
    print("=" * 60)

    loader = FIRMSSpatialLoader()
    df = loader.load_daily_features()

    print(f"\nLoaded {len(df)} days of features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFeature columns ({len(df.columns)}):")
    for col in df.columns[:20]:
        print(f"  {col}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more")

    print(f"\nSample data (first 5 rows):")
    print(df.head())

    # Show regional distribution
    print(f"\nRegional fire count totals:")
    regions = list(UKRAINE_REGIONS.keys())
    for region in regions:
        col = f'fire_count_{region}'
        if col in df.columns:
            total = df[col].sum()
            print(f"  {region}: {total:,.0f}")
