"""
VIIRS Nighttime Lights Spatial Data Loader

Provides tile-based regional VIIRS nightlight features. Each VIIRS tile covers a
10° x 10° area, giving 6 regional aggregates for Ukraine.

Note: This is tile-level spatial granularity, not true raion-level. For raion-level
nightlight analysis, raw pixel-level HDF5 data would be required.

Tile-to-Region Mapping:
- h19v03: Northwestern Ukraine (Kyiv, Zhytomyr, Chernihiv)
- h19v04: Southwestern Ukraine (Odesa, Mykolaiv, Kherson west)
- h20v03: Northeastern Ukraine (Kharkiv, Sumy, Poltava)
- h20v04: Southeastern Ukraine (Zaporizhia, Dnipro, Kherson east)
- h21v03: Eastern Ukraine (Luhansk north, Donetsk north)
- h21v04: Far Eastern Ukraine (Donetsk south, Luhansk south)

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import centralized paths
from config.paths import DATA_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

VIIRS_DATA_FILE = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"

# Tile to region mapping (approximate geographic coverage)
TILE_REGIONS = {
    'h19v03': 'northwest',   # Kyiv, Zhytomyr, Chernihiv, Rivne
    'h19v04': 'southwest',   # Odesa, Mykolaiv, Kherson (west), Crimea (west)
    'h20v03': 'northeast',   # Kharkiv, Sumy, Poltava, Cherkasy
    'h20v04': 'southeast',   # Zaporizhia, Dnipro, Kherson (east), Crimea (east)
    'h21v03': 'east_north',  # Luhansk (north), Donetsk (north)
    'h21v04': 'east_south',  # Donetsk (south), Luhansk (south), Sea of Azov
}

# Frontline-relevant regions (conflict areas)
FRONTLINE_REGIONS = ['northeast', 'southeast', 'east_north', 'east_south']


# =============================================================================
# MAIN LOADER CLASS
# =============================================================================

class VIIRSSpatialLoader:
    """
    Loader for NASA VIIRS nightlight data with tile-based regional aggregation.

    Features extracted per region (6 regions):
    - radiance_mean: Mean nighttime radiance (log-scaled)
    - radiance_std: Standard deviation (variability)
    - radiance_anomaly: Z-score deviation from 7-day baseline
    - clear_sky_pct: Clear sky percentage (quality indicator)

    Total: 6 regions × 4 features = 24 spatial features per day
    Plus 4 national aggregate features = 28 total features
    """

    def __init__(
        self,
        data_file: Optional[Path] = None,
        frontline_only: bool = False,
    ):
        """
        Initialize the VIIRS spatial loader.

        Args:
            data_file: Path to VIIRS daily stats CSV
            frontline_only: If True, only include frontline regions
        """
        self.data_file = data_file or VIIRS_DATA_FILE
        self.frontline_only = frontline_only

        self._data: Optional[pd.DataFrame] = None

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw VIIRS data."""
        if self._data is not None:
            return self._data

        if not self.data_file.exists():
            warnings.warn(f"VIIRS data not found: {self.data_file}")
            return pd.DataFrame()

        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Add region mapping
        df['region'] = df['tile'].map(TILE_REGIONS)

        self._data = df
        return df

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
            DataFrame with daily regional features
        """
        print("Loading VIIRS spatial features...")
        data = self._load_raw_data()

        if data.empty:
            return pd.DataFrame()

        # Filter date range
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]

        if data.empty:
            return pd.DataFrame()

        # Get regions to include
        if self.frontline_only:
            regions = FRONTLINE_REGIONS
        else:
            regions = list(TILE_REGIONS.values())

        # Compute per-region daily features
        records = []
        dates = data['date'].unique()

        print(f"  Processing {len(dates)} days across {len(regions)} regions...")

        for date in sorted(dates):
            day_data = data[data['date'] == date]
            record = {'date': date}

            for region in regions:
                region_data = day_data[day_data['region'] == region]

                if len(region_data) > 0:
                    # Mean radiance (log-scaled)
                    radiance = region_data['radiance_mean'].mean()
                    record[f'viirs_radiance_{region}'] = np.log1p(radiance)

                    # Standard deviation
                    record[f'viirs_std_{region}'] = region_data['radiance_std'].mean()

                    # Clear sky percentage
                    record[f'viirs_clear_{region}'] = region_data['pct_clear_sky'].mean()

                    # Moon illumination (affects visibility)
                    record[f'viirs_moon_{region}'] = region_data['moon_illumination_pct'].mean()
                else:
                    record[f'viirs_radiance_{region}'] = np.nan
                    record[f'viirs_std_{region}'] = np.nan
                    record[f'viirs_clear_{region}'] = np.nan
                    record[f'viirs_moon_{region}'] = np.nan

            # National aggregates
            record['viirs_radiance_total'] = np.log1p(day_data['radiance_mean'].mean())
            record['viirs_std_total'] = day_data['radiance_std'].mean()
            record['viirs_clear_total'] = day_data['pct_clear_sky'].mean()
            record['viirs_coverage'] = len(day_data)  # Number of tiles available

            records.append(record)

        result_df = pd.DataFrame(records)

        # Compute anomaly features (7-day rolling z-score)
        result_df = result_df.sort_values('date').reset_index(drop=True)

        for region in regions:
            col = f'viirs_radiance_{region}'
            if col in result_df.columns:
                rolling_mean = result_df[col].rolling(7, min_periods=3).mean()
                rolling_std = result_df[col].rolling(7, min_periods=3).std().replace(0, np.nan)
                result_df[f'viirs_anomaly_{region}'] = (result_df[col] - rolling_mean) / rolling_std
                result_df[f'viirs_anomaly_{region}'] = result_df[f'viirs_anomaly_{region}'].fillna(0)

        print(f"  Generated {len(result_df)} daily records with {len(result_df.columns)} features")

        return result_df

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        regions = FRONTLINE_REGIONS if self.frontline_only else list(TILE_REGIONS.values())

        features = []
        for region in regions:
            features.extend([
                f'viirs_radiance_{region}',
                f'viirs_std_{region}',
                f'viirs_clear_{region}',
                f'viirs_moon_{region}',
                f'viirs_anomaly_{region}',
            ])

        # Totals
        features.extend([
            'viirs_radiance_total',
            'viirs_std_total',
            'viirs_clear_total',
            'viirs_coverage',
        ])

        return features


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def load_viirs_tiled(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    frontline_only: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load VIIRS data with tile-based regional features.

    Args:
        start_date: Optional start date
        end_date: Optional end date
        frontline_only: Only include frontline regions

    Returns:
        Tuple of (DataFrame with tiled features, observation mask array)
    """
    loader = VIIRSSpatialLoader(frontline_only=frontline_only)
    df = loader.load_daily_features(start_date, end_date)

    if df.empty:
        return pd.DataFrame(), np.array([])

    # Create observation mask (1 where we have coverage)
    mask = (df['viirs_coverage'] > 0).astype(float).values

    return df, mask


# =============================================================================
# REGISTRATION
# =============================================================================

VIIRS_SPATIAL_LOADER = {
    'name': 'viirs_spatial',
    'loader_fn': load_viirs_tiled,
    'resolution': 'daily',
    'feature_count': 34,  # 6 regions × 5 features + 4 totals
    'description': 'NASA VIIRS nightlights with tile-based regional aggregation',
}


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("VIIRS Spatial Loader Test")
    print("=" * 60)

    loader = VIIRSSpatialLoader()
    df = loader.load_daily_features()

    if not df.empty:
        print(f"\nLoaded {len(df)} days of features")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Feature columns: {len(df.columns)}")

        print(f"\nSample columns:")
        for col in df.columns[:15]:
            print(f"  {col}")
        if len(df.columns) > 15:
            print(f"  ... and {len(df.columns) - 15} more")

        # Show regional coverage
        print(f"\nRegional radiance means:")
        for region in TILE_REGIONS.values():
            col = f'viirs_radiance_{region}'
            if col in df.columns:
                mean_val = df[col].mean()
                print(f"  {region}: {mean_val:.3f}")
    else:
        print("No data loaded")
