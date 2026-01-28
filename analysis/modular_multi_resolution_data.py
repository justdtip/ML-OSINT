"""
Modular Multi-Resolution Dataset with Configurable Data Sources.

This module extends the base MultiResolutionDataset to support modular data source
configuration, including selective enabling/disabling of sources and spatial feature
modes (aggregated, tiled, full).

Key Features:
- Integrates with ModularDataConfig for flexible source configuration
- Supports VIIRS per-tile features (6 regional tiles instead of 1 aggregated)
- Supports FIRMS with spatial binning (regional grid cells)
- Supports DeepState territorial polygons as features
- Backward compatible: default config reproduces current behavior

Usage:
    from modular_data_config import get_data_source_config
    from modular_multi_resolution_data import ModularMultiResolutionDataset

    # Use a preset config
    data_config = get_data_source_config('spatial_rich')

    # Create dataset with modular configuration
    dataset = ModularMultiResolutionDataset(
        config=multi_res_config,
        data_source_config=data_config,
        split='train',
    )

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Import base multi-resolution data module (handle both direct and package imports)
try:
    from multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
        MultiResolutionSample,
        multi_resolution_collate_fn,
        MISSING_VALUE,
        Resolution,
        TemporalAlignment,
        compute_temporal_alignment,
        LOADER_REGISTRY,
        SOURCE_RESOLUTIONS,
    )
except ImportError:
    from analysis.multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
        MultiResolutionSample,
        multi_resolution_collate_fn,
        MISSING_VALUE,
        Resolution,
        TemporalAlignment,
        compute_temporal_alignment,
        LOADER_REGISTRY,
        SOURCE_RESOLUTIONS,
    )

# Import modular configuration (handle both direct and package imports)
try:
    from modular_data_config import (
        ModularDataConfig,
        DataSourceEntry,
        SpatialMode,
        TemporalResolution,
        get_data_source_config,
        VIIRS_TILES,
        FIRMS_GRID_CELLS,
        DEEPSTATE_CONTROL_TYPES,
    )
except ImportError:
    from analysis.modular_data_config import (
        ModularDataConfig,
        DataSourceEntry,
        SpatialMode,
        TemporalResolution,
        get_data_source_config,
        VIIRS_TILES,
        FIRMS_GRID_CELLS,
        DEEPSTATE_CONTROL_TYPES,
    )

# Import centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, VIIRS_DIR, FIRMS_DIR, DEEPSTATE_DIR,
    ANALYSIS_DIR, ISW_EMBEDDINGS_DIR,
)


# =============================================================================
# SPATIAL FEATURE LOADERS
# =============================================================================

def load_viirs_tiled(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    detrend: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load VIIRS brightness data with per-region tiled features.

    Instead of aggregating all brightness values into a single feature,
    this function computes separate brightness statistics for each of the
    6 Ukrainian regional tiles defined in VIIRS_TILES.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter
        detrend: Apply first-order differencing to remove trends

    Returns:
        Tuple of (DataFrame with tiled features, observation mask array)
        Features include: {tile_name}_mean, {tile_name}_max, {tile_name}_std
        for each of the 6 tiles.
    """
    # Load the base VIIRS data with spatial columns
    viirs_path = VIIRS_DIR / "viirs_a2_gap_filled_stats.csv"

    if not viirs_path.exists():
        # Fall back to the daily stats file
        viirs_path = VIIRS_DIR / "viirs_daily_brightness_stats.csv"

    if not viirs_path.exists():
        warnings.warn(f"VIIRS data not found at {viirs_path}")
        return pd.DataFrame(), np.array([])

    df = pd.read_csv(viirs_path)

    # Check for required spatial columns
    has_spatial = 'lat' in df.columns and 'lon' in df.columns

    if not has_spatial:
        # Fall back to non-tiled loading if no spatial data
        warnings.warn(
            "VIIRS data does not have lat/lon columns. "
            "Falling back to aggregated features."
        )
        from multi_resolution_data import load_viirs_daily
        return load_viirs_daily(start_date, end_date, detrend=detrend)

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Assign each point to a tile based on lat/lon
    def assign_tile(row):
        for tile_name, tile_info in VIIRS_TILES.items():
            lat_range = tile_info['lat_range']
            lon_range = tile_info['lon_range']
            if (lat_range[0] <= row['lat'] <= lat_range[1] and
                lon_range[0] <= row['lon'] <= lon_range[1]):
                return tile_name
        return 'other'  # Points outside defined tiles

    df['tile'] = df.apply(assign_tile, axis=1)

    # Aggregate by date and tile
    daily_tiled = df.groupby(['date', 'tile']).agg({
        'brightness': ['mean', 'max', 'std', 'count']
    }).reset_index()

    # Flatten column names
    daily_tiled.columns = ['date', 'tile', 'brightness_mean', 'brightness_max',
                           'brightness_std', 'brightness_count']

    # Pivot to wide format (one column per tile per stat)
    feature_df = daily_tiled.pivot(
        index='date',
        columns='tile',
        values=['brightness_mean', 'brightness_max', 'brightness_std']
    )

    # Flatten hierarchical columns
    feature_df.columns = [f'{tile}_{stat.replace("brightness_", "")}'
                          for stat, tile in feature_df.columns]
    feature_df = feature_df.reset_index()

    # Create complete daily date range
    if start_date is None:
        start_date = feature_df['date'].min()
    if end_date is None:
        end_date = feature_df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex without filling
    feature_df = feature_df.set_index('date')
    feature_df = feature_df.reindex(date_range)
    feature_df.index.name = 'date'
    feature_df = feature_df.reset_index()

    # Create observation mask
    feature_cols = [c for c in feature_df.columns if c != 'date']
    observation_mask = feature_df[feature_cols].notna().any(axis=1).values

    # Apply detrending if requested
    if detrend:
        detrended_df = feature_df.copy()
        for col in feature_cols:
            detrended_df[col] = feature_df[col].diff()

        # Update mask for detrending
        prev_mask = np.roll(observation_mask, 1)
        prev_mask[0] = False
        observation_mask = observation_mask & prev_mask
        observation_mask[0] = False

        feature_df = detrended_df

    return feature_df, observation_mask


def load_firms_tiled(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS fire hotspot data with regional spatial binning.

    Instead of aggregating all fire detections into a single count,
    this function computes fire statistics for each regional tile.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with tiled features, observation mask array)
        Features include: {tile_name}_count, {tile_name}_frp_mean, {tile_name}_frp_max
        for each tile.
    """
    firms_archive = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
    firms_nrt = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_nrt_SV-C2_706038.csv"

    dfs = []
    for path in [firms_archive, firms_nrt]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Failed to load FIRMS data from {path}: {e}")

    if not dfs:
        warnings.warn("FIRMS data not found")
        return pd.DataFrame(), np.array([])

    fires_df = pd.concat(dfs, ignore_index=True)

    # Parse dates
    fires_df['date'] = pd.to_datetime(fires_df['acq_date'], errors='coerce')
    fires_df = fires_df.dropna(subset=['date', 'latitude', 'longitude'])

    # Assign each fire to a tile
    def assign_tile(row):
        for tile_name, tile_info in VIIRS_TILES.items():
            lat_range = tile_info['lat_range']
            lon_range = tile_info['lon_range']
            if (lat_range[0] <= row['latitude'] <= lat_range[1] and
                lon_range[0] <= row['longitude'] <= lon_range[1]):
                return tile_name
        return 'other'

    fires_df['tile'] = fires_df.apply(assign_tile, axis=1)

    # Aggregate by date and tile
    daily_tiled = fires_df.groupby(['date', 'tile']).agg({
        'latitude': 'count',  # Fire count
        'frp': ['mean', 'max', 'sum'],  # Fire radiative power stats
        'confidence': 'mean',  # Average confidence
    }).reset_index()

    # Flatten column names
    daily_tiled.columns = ['date', 'tile', 'fire_count', 'frp_mean', 'frp_max',
                           'frp_sum', 'confidence_mean']

    # Pivot to wide format
    feature_df = daily_tiled.pivot(
        index='date',
        columns='tile',
        values=['fire_count', 'frp_mean', 'frp_max']
    )

    # Flatten hierarchical columns
    feature_df.columns = [f'{tile}_{stat}' for stat, tile in feature_df.columns]
    feature_df = feature_df.reset_index()
    feature_df = feature_df.fillna(0)  # No fires = 0 count

    # Create complete daily date range
    if start_date is None:
        start_date = feature_df['date'].min()
    if end_date is None:
        end_date = feature_df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex
    feature_df = feature_df.set_index('date')
    feature_df = feature_df.reindex(date_range, fill_value=0)
    feature_df.index.name = 'date'
    feature_df = feature_df.reset_index()

    # Observation mask: True everywhere (fires can be zero but that's observed)
    observation_mask = np.ones(len(feature_df), dtype=bool)

    return feature_df, observation_mask


def load_deepstate_tiled(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load DeepState territorial control data with per-region features.

    Instead of aggregating control metrics nationally, this function computes
    territorial control statistics for each regional tile.

    Args:
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Tuple of (DataFrame with tiled features, observation mask array)
        Features include: {tile_name}_controlled_pct, {tile_name}_contested_pct
        for each tile.
    """
    wayback_dir = DEEPSTATE_DIR / "wayback_snapshots"

    if not wayback_dir.exists():
        warnings.warn(f"DeepState data not found at {wayback_dir}")
        return pd.DataFrame(), np.array([])

    snapshots = []

    for snapshot_file in sorted(wayback_dir.glob("*.json")):
        try:
            with open(snapshot_file) as f:
                data = json.load(f)

            # Extract date from filename (format: YYYYMMDD_HHMMSS.json)
            date_str = snapshot_file.stem[:8]
            date = pd.to_datetime(date_str, format='%Y%m%d')

            # Parse GeoJSON features
            if isinstance(data, dict) and 'features' in data:
                features = data['features']
            elif isinstance(data, list):
                features = data
            else:
                continue

            # Compute per-tile statistics
            tile_stats = {tile: {'russian': 0, 'ukrainian': 0, 'contested': 0, 'total': 0}
                         for tile in VIIRS_TILES.keys()}

            for feature in features:
                if not isinstance(feature, dict):
                    continue

                props = feature.get('properties', {})
                geom = feature.get('geometry', {})

                # Get centroid for tile assignment
                if geom.get('type') == 'Polygon':
                    coords = geom.get('coordinates', [[]])[0]
                    if coords:
                        lons = [c[0] for c in coords]
                        lats = [c[1] for c in coords]
                        centroid_lon = np.mean(lons)
                        centroid_lat = np.mean(lats)
                    else:
                        continue
                else:
                    continue

                # Assign to tile
                tile_name = None
                for t_name, t_info in VIIRS_TILES.items():
                    lat_range = t_info['lat_range']
                    lon_range = t_info['lon_range']
                    if (lat_range[0] <= centroid_lat <= lat_range[1] and
                        lon_range[0] <= centroid_lon <= lon_range[1]):
                        tile_name = t_name
                        break

                if tile_name is None:
                    continue

                # Determine control status
                control = props.get('control', '').lower()
                if 'russian' in control or 'occupied' in control:
                    tile_stats[tile_name]['russian'] += 1
                elif 'ukrainian' in control or 'liberated' in control:
                    tile_stats[tile_name]['ukrainian'] += 1
                elif 'contested' in control or 'active' in control:
                    tile_stats[tile_name]['contested'] += 1
                tile_stats[tile_name]['total'] += 1

            # Create row for this date
            row = {'date': date}
            for tile_name, stats in tile_stats.items():
                total = max(stats['total'], 1)  # Avoid division by zero
                row[f'{tile_name}_russian_pct'] = stats['russian'] / total
                row[f'{tile_name}_ukrainian_pct'] = stats['ukrainian'] / total
                row[f'{tile_name}_contested_pct'] = stats['contested'] / total

            snapshots.append(row)

        except Exception as e:
            warnings.warn(f"Failed to parse {snapshot_file}: {e}")
            continue

    if not snapshots:
        return pd.DataFrame(), np.array([])

    feature_df = pd.DataFrame(snapshots)
    feature_df = feature_df.sort_values('date').reset_index(drop=True)

    # Create complete daily date range
    if start_date is None:
        start_date = feature_df['date'].min()
    if end_date is None:
        end_date = feature_df['date'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex and forward-fill (control status changes are discrete events)
    feature_df = feature_df.set_index('date')
    feature_df = feature_df.reindex(date_range)
    feature_df = feature_df.ffill()  # Forward fill control status
    feature_df.index.name = 'date'
    feature_df = feature_df.reset_index()

    # Observation mask
    feature_cols = [c for c in feature_df.columns if c != 'date']
    observation_mask = feature_df[feature_cols].notna().any(axis=1).values

    return feature_df, observation_mask


# Register spatial loaders
SPATIAL_LOADER_REGISTRY = {
    'viirs': {
        SpatialMode.TILED: load_viirs_tiled,
        SpatialMode.AGGREGATED: LOADER_REGISTRY.get('viirs'),
    },
    'firms': {
        SpatialMode.TILED: load_firms_tiled,
        SpatialMode.AGGREGATED: LOADER_REGISTRY.get('firms'),
    },
    'deepstate': {
        SpatialMode.TILED: load_deepstate_tiled,
        SpatialMode.AGGREGATED: LOADER_REGISTRY.get('deepstate'),
    },
}


# =============================================================================
# MODULAR MULTI-RESOLUTION DATASET
# =============================================================================

class ModularMultiResolutionDataset(MultiResolutionDataset):
    """
    Extended MultiResolutionDataset with modular data source configuration.

    This class extends the base MultiResolutionDataset to support:
    - Selective enabling/disabling of data sources
    - Spatial feature modes (aggregated, tiled, full)
    - Per-source configuration (detrending, lag, normalization)

    Backward Compatibility:
        When no data_source_config is provided (or 'baseline' preset is used),
        the behavior is identical to the original MultiResolutionDataset.

    Args:
        config: Base MultiResolutionConfig for sequence lengths, date range, etc.
        data_source_config: ModularDataConfig for per-source settings.
            If None, uses 'baseline' preset (current behavior).
        split: One of 'train', 'val', 'test'
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        norm_stats: Pre-computed normalization stats (required for val/test)
        seed: Random seed for reproducibility

    Example:
        >>> from modular_data_config import get_data_source_config
        >>>
        >>> # Spatial-rich configuration with VIIRS tiled
        >>> data_config = get_data_source_config('spatial_rich')
        >>>
        >>> dataset = ModularMultiResolutionDataset(
        ...     data_source_config=data_config,
        ...     split='train',
        ... )
        >>> print(f"Features: {dataset.get_feature_info().keys()}")
    """

    def __init__(
        self,
        config: Optional[MultiResolutionConfig] = None,
        data_source_config: Optional[ModularDataConfig] = None,
        split: str = 'train',
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        norm_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        seed: int = 42
    ):
        """Initialize the modular multi-resolution dataset."""
        # Store modular config before calling parent __init__
        self.data_source_config = data_source_config or get_data_source_config('baseline')

        # Validate data source config
        if not isinstance(self.data_source_config, ModularDataConfig):
            raise TypeError(
                f"data_source_config must be ModularDataConfig, got {type(data_source_config)}"
            )

        print(f"\nModular Data Config: {self.data_source_config.config_name}")
        print(f"  Enabled daily sources: {self.data_source_config.enabled_daily_sources}")
        print(f"  Enabled monthly sources: {self.data_source_config.enabled_monthly_sources}")
        if self.data_source_config.spatial_features_enabled:
            print(f"  Spatial sources: {list(self.data_source_config.spatial_sources.keys())}")

        # Override the config's daily/monthly sources based on modular config
        # This ensures the parent class loads the correct sources
        if config is None:
            config = MultiResolutionConfig()

        # Update config sources based on modular config
        config.daily_sources = self._get_effective_daily_sources()
        config.monthly_sources = self._get_effective_monthly_sources()

        # Sync detrending and exclude settings
        viirs_cfg = self.data_source_config.viirs
        config.detrend_viirs = viirs_cfg.detrend if viirs_cfg.enabled else False
        config.exclude_viirs = not viirs_cfg.enabled

        config.use_disaggregated_equipment = self.data_source_config.use_disaggregated_equipment

        # Call parent constructor
        # Note: We override _load_all_sources to use our modular config
        super().__init__(
            config=config,
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            norm_stats=norm_stats,
            seed=seed,
        )

    def _get_effective_daily_sources(self) -> List[str]:
        """Get list of enabled daily sources based on modular config."""
        sources = []

        for source_name, cfg in self.data_source_config.daily_sources.items():
            if cfg.enabled:
                sources.append(source_name)

        return sources

    def _get_effective_monthly_sources(self) -> List[str]:
        """Get list of enabled monthly sources based on modular config."""
        sources = []

        for source_name, cfg in self.data_source_config.monthly_sources.items():
            if cfg.enabled:
                sources.append(source_name)

        return sources

    def _load_all_sources(self) -> None:
        """Load all configured data sources with modular spatial handling."""
        self.daily_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}
        self.monthly_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}

        all_dates = []

        # Load daily sources
        for source_name in self.config.daily_sources:
            source_cfg = self.data_source_config.get_source_config(source_name)

            if source_cfg is None or not source_cfg.enabled:
                continue

            print(f"  Loading daily source: {source_name} (spatial={source_cfg.spatial_mode.value})...")

            # Determine which loader to use based on spatial mode
            loader_func = self._get_loader_for_source(source_name, source_cfg)

            if loader_func is None:
                warnings.warn(f"No loader found for source: {source_name}")
                continue

            # Call loader with appropriate arguments
            try:
                if source_name == 'viirs' and source_cfg.detrend:
                    df, mask = loader_func(detrend=True)
                else:
                    # Check if loader accepts detrend argument
                    import inspect
                    sig = inspect.signature(loader_func)
                    if 'detrend' in sig.parameters:
                        df, mask = loader_func(detrend=source_cfg.detrend)
                    else:
                        df, mask = loader_func()

            except Exception as e:
                warnings.warn(f"Failed to load {source_name}: {e}")
                continue

            if df.empty:
                warnings.warn(f"Empty data for source: {source_name}")
                continue

            # Apply temporal lag if configured
            if source_cfg.lag_days > 0:
                df = self._apply_temporal_lag(df, source_cfg.lag_days)
                mask = np.roll(mask, source_cfg.lag_days)
                mask[:source_cfg.lag_days] = False

            self.daily_data[source_name] = (df, mask)
            all_dates.extend(df['date'].dropna().tolist())

        # Load monthly sources
        for source_name in self.config.monthly_sources:
            source_cfg = self.data_source_config.get_source_config(source_name)

            if source_cfg is None or not source_cfg.enabled:
                continue

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
        print(f"  Loaded {len(self.daily_data)} daily sources, {len(self.monthly_data)} monthly sources")

    def _get_loader_for_source(
        self,
        source_name: str,
        source_cfg: DataSourceEntry
    ) -> Optional[callable]:
        """Get the appropriate loader function for a source based on its spatial mode."""
        spatial_mode = source_cfg.spatial_mode

        # Check if we have a spatial loader for this source
        if source_name in SPATIAL_LOADER_REGISTRY:
            spatial_loaders = SPATIAL_LOADER_REGISTRY[source_name]
            if spatial_mode in spatial_loaders:
                return spatial_loaders[spatial_mode]
            # Fall back to aggregated if tiled not available
            if SpatialMode.AGGREGATED in spatial_loaders:
                return spatial_loaders[SpatialMode.AGGREGATED]

        # Use standard loader from registry
        if source_name in LOADER_REGISTRY:
            return LOADER_REGISTRY[source_name]

        return None

    def _apply_temporal_lag(self, df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
        """Apply temporal lag to a dataframe (shift features forward in time)."""
        df = df.copy()
        feature_cols = [c for c in df.columns if c != 'date']
        for col in feature_cols:
            df[col] = df[col].shift(lag_days)
        return df

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration for logging."""
        return {
            'config_name': self.data_source_config.config_name,
            'description': self.data_source_config.description,
            'enabled_daily_sources': self.data_source_config.enabled_daily_sources,
            'enabled_monthly_sources': self.data_source_config.enabled_monthly_sources,
            'spatial_features_enabled': self.data_source_config.spatial_features_enabled,
            'spatial_sources': list(self.data_source_config.spatial_sources.keys()),
            'use_disaggregated_equipment': self.data_source_config.use_disaggregated_equipment,
            'split': self.split,
            'n_samples': len(self),
        }


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_modular_dataloaders(
    multi_res_config: Optional[MultiResolutionConfig] = None,
    data_source_config: Optional[ModularDataConfig] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Dict[str, np.ndarray]], ModularDataConfig]:
    """
    Create train, validation, and test data loaders with modular configuration.

    This function extends create_multi_resolution_dataloaders to support
    modular data source configuration.

    Args:
        multi_res_config: Base MultiResolutionConfig for sequence lengths, etc.
        data_source_config: ModularDataConfig for per-source settings.
            If None, uses 'baseline' preset.
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
        data_source_config: The ModularDataConfig used (for logging)

    Example:
        >>> from modular_data_config import get_data_source_config
        >>>
        >>> data_config = get_data_source_config('spatial_rich')
        >>> train_loader, val_loader, test_loader, norm_stats, config = \\
        ...     create_modular_dataloaders(data_source_config=data_config)
    """
    multi_res_config = multi_res_config or MultiResolutionConfig()
    data_source_config = data_source_config or get_data_source_config('baseline')

    print("=" * 80)
    print("Creating Modular Multi-Resolution DataLoaders")
    print(f"Configuration: {data_source_config.config_name}")
    print("=" * 80)

    # Create training dataset first (computes normalization stats)
    print("\n--- Training Dataset ---")
    train_dataset = ModularMultiResolutionDataset(
        config=multi_res_config,
        data_source_config=data_source_config,
        split='train',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    norm_stats = train_dataset.norm_stats

    # Create validation dataset
    print("\n--- Validation Dataset ---")
    val_dataset = ModularMultiResolutionDataset(
        config=multi_res_config,
        data_source_config=data_source_config,
        split='val',
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        norm_stats=norm_stats,
        seed=seed
    )

    # Create test dataset
    print("\n--- Test Dataset ---")
    test_dataset = ModularMultiResolutionDataset(
        config=multi_res_config,
        data_source_config=data_source_config,
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
        shuffle=True,
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
    print("DataLoaders created successfully")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print("=" * 80)

    return train_loader, val_loader, test_loader, norm_stats, data_source_config


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Testing Modular Multi-Resolution Dataset")
    print("=" * 60)

    # Test baseline config
    print("\n1. Testing baseline configuration...")
    baseline_config = get_data_source_config('baseline')
    print(baseline_config)

    try:
        train_ds = ModularMultiResolutionDataset(
            data_source_config=baseline_config,
            split='train',
        )
        print(f"\nBaseline dataset created: {len(train_ds)} samples")
        print(f"Feature info: {train_ds.get_feature_info().keys()}")

        # Get a sample
        sample = train_ds[0]
        print(f"Sample daily sources: {list(sample.daily_features.keys())}")
        print(f"Sample monthly sources: {list(sample.monthly_features.keys())}")

    except Exception as e:
        print(f"Error creating baseline dataset: {e}")

    # Test spatial-rich config
    print("\n" + "=" * 60)
    print("\n2. Testing spatial_rich configuration...")
    spatial_config = get_data_source_config('spatial_rich')
    print(spatial_config)

    try:
        train_ds_spatial = ModularMultiResolutionDataset(
            data_source_config=spatial_config,
            split='train',
        )
        print(f"\nSpatial-rich dataset created: {len(train_ds_spatial)} samples")
        print(f"Feature info keys: {list(train_ds_spatial.get_feature_info().keys())}")

    except Exception as e:
        print(f"Error creating spatial dataset: {e}")

    # Test ablation config
    print("\n" + "=" * 60)
    print("\n3. Testing ablation (equipment only) configuration...")
    ablation_config = get_data_source_config('ablation_equipment_only')
    print(ablation_config)

    try:
        train_ds_ablation = ModularMultiResolutionDataset(
            data_source_config=ablation_config,
            split='train',
        )
        print(f"\nAblation dataset created: {len(train_ds_ablation)} samples")
        print(f"Feature info keys: {list(train_ds_ablation.get_feature_info().keys())}")

    except Exception as e:
        print(f"Error creating ablation dataset: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
