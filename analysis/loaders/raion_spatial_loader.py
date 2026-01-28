"""
Raion (District) Spatial Data Loader

Provides point-in-polygon assignment of spatial data sources (FIRMS, DeepState)
to Ukraine's 629 raions (administrative level 2 districts).

Key Features:
- Efficient point-in-polygon using matplotlib.path.Path (no shapely dependency)
- Pre-computed raion lookup with bounding box acceleration
- Filters to conflict-relevant raions (325 in 13 frontline oblasts)
- Configurable active raion filtering by data density

Data sources:
- Raion boundaries: GADM Ukraine admin level 2 (data/boundaries/ukraine_raions.geojson)
- FIRMS hotspots: NASA VIIRS fire data (data/firms/)
- DeepState: Unit positions (deepstate-map-data/)

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

# Import centralized paths
from config.paths import DATA_DIR, FIRMS_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

# Raion boundaries file
RAION_BOUNDARIES_FILE = DATA_DIR / "boundaries" / "ukraine_raions.geojson"
FRONTLINE_RAIONS_FILE = DATA_DIR / "boundaries" / "frontline_raions.json"

# FIRMS data files
FIRMS_ARCHIVE_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
FIRMS_NRT_FILE = FIRMS_DIR / "DL_FIRE_SV-C2_706038" / "fire_nrt_SV-C2_706038.csv"

# Frontline oblasts (13 with significant conflict activity)
FRONTLINE_OBLASTS = [
    "Donetsk", "Luhansk", "Zaporizhia", "Kherson", "Kharkiv",
    "Dnipropetrovsk", "Mykolaiv", "Sumy", "Chernihiv", "Kyiv",
    "Zhytomyr", "Crimea", "Sevastopol"
]


# =============================================================================
# RAION BOUNDARY MANAGEMENT
# =============================================================================

@dataclass
class RaionBoundary:
    """Represents a raion with its polygon boundary and metadata."""
    name: str
    oblast: str
    gid: str
    polygon_paths: List[MplPath]  # List for MultiPolygons
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    centroid: Tuple[float, float]  # lon, lat

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if point is inside this raion (with bbox acceleration)."""
        # Quick bbox check first
        if not (self.bbox[0] <= lon <= self.bbox[2] and
                self.bbox[1] <= lat <= self.bbox[3]):
            return False

        # Full polygon check
        point = (lon, lat)
        for path in self.polygon_paths:
            if path.contains_point(point):
                return True
        return False


class RaionBoundaryManager:
    """
    Manages raion boundary data with efficient point-in-polygon lookup.

    Uses a spatial index (grid-based) for fast lookups of which raions
    might contain a given point.
    """

    def __init__(
        self,
        boundaries_file: Optional[Path] = None,
        frontline_only: bool = True,
        min_observations: int = 10,
    ):
        """
        Initialize the raion boundary manager.

        Args:
            boundaries_file: Path to raion boundaries GeoJSON
            frontline_only: If True, only load frontline oblast raions
            min_observations: Minimum observations to consider a raion active
        """
        self.boundaries_file = boundaries_file or RAION_BOUNDARIES_FILE
        self.frontline_only = frontline_only
        self.min_observations = min_observations

        self.raions: Dict[str, RaionBoundary] = {}
        self._grid_index: Dict[Tuple[int, int], List[str]] = {}
        self._grid_resolution = 0.5  # degrees per grid cell

        self._loaded = False

    def load(self) -> None:
        """Load raion boundaries from GeoJSON."""
        if self._loaded:
            return

        if not self.boundaries_file.exists():
            warnings.warn(f"Raion boundaries not found: {self.boundaries_file}")
            return

        print(f"Loading raion boundaries from {self.boundaries_file}...")

        with open(self.boundaries_file) as f:
            data = json.load(f)

        loaded = 0
        skipped = 0

        for feature in data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})

            raion_name = props.get('NAME_2', '')
            oblast_name = props.get('NAME_1', '')
            gid = props.get('GID_2', '')

            # Filter to frontline oblasts if requested
            if self.frontline_only:
                # Normalize oblast name for matching
                oblast_normalized = oblast_name.replace("'", "").replace("Oblast", "").strip()
                if not any(fl.lower() in oblast_normalized.lower() or
                          oblast_normalized.lower() in fl.lower()
                          for fl in FRONTLINE_OBLASTS):
                    skipped += 1
                    continue

            # Parse geometry
            try:
                paths, bbox, centroid = self._parse_geometry(geom)
                if not paths:
                    skipped += 1
                    continue
            except Exception as e:
                warnings.warn(f"Failed to parse geometry for {raion_name}: {e}")
                skipped += 1
                continue

            # Create raion object
            raion = RaionBoundary(
                name=raion_name,
                oblast=oblast_name,
                gid=gid,
                polygon_paths=paths,
                bbox=bbox,
                centroid=centroid,
            )

            # Use unique key (some raion names are duplicated across oblasts)
            key = f"{oblast_name}_{raion_name}"
            self.raions[key] = raion
            loaded += 1

            # Add to spatial index
            self._add_to_grid_index(key, bbox)

        self._loaded = True
        print(f"  Loaded {loaded} raions (skipped {skipped})")

    def _parse_geometry(
        self, geom: Dict[str, Any]
    ) -> Tuple[List[MplPath], Tuple[float, float, float, float], Tuple[float, float]]:
        """
        Parse GeoJSON geometry into matplotlib Paths.

        Returns:
            Tuple of (list of paths, bbox, centroid)
        """
        geom_type = geom.get('type', '')
        coords = geom.get('coordinates', [])

        paths = []
        all_lons = []
        all_lats = []

        if geom_type == 'Polygon':
            # Polygon: list of rings, first is exterior
            if coords:
                exterior = coords[0]
                if exterior:
                    ring = np.array(exterior)
                    paths.append(MplPath(ring))
                    all_lons.extend(ring[:, 0])
                    all_lats.extend(ring[:, 1])

        elif geom_type == 'MultiPolygon':
            # MultiPolygon: list of polygons
            for polygon in coords:
                if polygon:
                    exterior = polygon[0]
                    if exterior:
                        ring = np.array(exterior)
                        paths.append(MplPath(ring))
                        all_lons.extend(ring[:, 0])
                        all_lats.extend(ring[:, 1])

        if not paths:
            return [], (0, 0, 0, 0), (0, 0)

        # Compute bbox and centroid
        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
        centroid = (np.mean(all_lons), np.mean(all_lats))

        return paths, bbox, centroid

    def _add_to_grid_index(self, key: str, bbox: Tuple[float, float, float, float]) -> None:
        """Add raion to grid-based spatial index."""
        min_lon, min_lat, max_lon, max_lat = bbox

        # Get grid cells that intersect this bbox
        min_x = int(min_lon / self._grid_resolution)
        max_x = int(max_lon / self._grid_resolution) + 1
        min_y = int(min_lat / self._grid_resolution)
        max_y = int(max_lat / self._grid_resolution) + 1

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                cell = (x, y)
                if cell not in self._grid_index:
                    self._grid_index[cell] = []
                self._grid_index[cell].append(key)

    def get_raion_for_point(self, lat: float, lon: float) -> Optional[str]:
        """
        Find which raion contains the given point.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Raion key if found, None otherwise
        """
        if not self._loaded:
            self.load()

        # Get grid cell for point
        cell = (int(lon / self._grid_resolution), int(lat / self._grid_resolution))

        # Check candidate raions in this cell
        candidates = self._grid_index.get(cell, [])
        for key in candidates:
            raion = self.raions.get(key)
            if raion and raion.contains_point(lon, lat):
                return key

        return None

    def get_raion_names(self) -> List[str]:
        """Get list of all raion keys."""
        if not self._loaded:
            self.load()
        return list(self.raions.keys())

    def get_active_raions(
        self,
        points: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        min_observations: Optional[int] = None,
    ) -> List[str]:
        """
        Get raions with at least min_observations points.

        Args:
            points: DataFrame with lat/lon columns
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            min_observations: Minimum observations (uses instance default if None)

        Returns:
            List of active raion keys
        """
        if not self._loaded:
            self.load()

        min_obs = min_observations or self.min_observations

        # Count points per raion
        counts: Dict[str, int] = {}
        for _, row in points.iterrows():
            raion = self.get_raion_for_point(row[lat_col], row[lon_col])
            if raion:
                counts[raion] = counts.get(raion, 0) + 1

        # Filter to active raions
        active = [r for r, c in counts.items() if c >= min_obs]
        return sorted(active, key=lambda x: counts[x], reverse=True)


# =============================================================================
# FIRMS RAION LOADER
# =============================================================================

class FIRMSRaionLoader:
    """
    Loader for NASA FIRMS fire hotspot data with raion-level aggregation.

    Features extracted per raion:
    - fire_count: Number of hotspots
    - fire_brightness_mean: Mean brightness temperature (K)
    - fire_frp_sum: Total fire radiative power (MW)
    - fire_day_ratio: Proportion of daytime detections

    Total: N_active_raions × 4 features per day
    """

    def __init__(
        self,
        archive_file: Optional[Path] = None,
        nrt_file: Optional[Path] = None,
        raion_manager: Optional[RaionBoundaryManager] = None,
        max_raions: int = 50,
    ):
        """
        Initialize the FIRMS raion loader.

        Args:
            archive_file: Path to FIRMS archive CSV
            nrt_file: Path to FIRMS near-real-time CSV
            raion_manager: Pre-configured raion boundary manager
            max_raions: Maximum number of raions to include
        """
        self.archive_file = archive_file or FIRMS_ARCHIVE_FILE
        self.nrt_file = nrt_file or FIRMS_NRT_FILE
        self.raion_manager = raion_manager or RaionBoundaryManager()
        self.max_raions = max_raions

        self._data: Optional[pd.DataFrame] = None
        self._active_raions: Optional[List[str]] = None

    def _load_raw_data(self) -> pd.DataFrame:
        """Load and combine archive and NRT FIRMS data."""
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

        # Filter to Ukraine bounds
        ukraine_mask = (
            (self._data['latitude'] >= 44.0) & (self._data['latitude'] <= 53.0) &
            (self._data['longitude'] >= 22.0) & (self._data['longitude'] <= 41.0)
        )
        self._data = self._data[ukraine_mask].copy()

        print(f"  Total hotspots in Ukraine: {len(self._data):,}")

        return self._data

    def _assign_raions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign each hotspot to a raion."""
        print("  Assigning hotspots to raions (this may take a moment)...")

        # Load raion boundaries
        self.raion_manager.load()

        # Assign each point to a raion
        raions = []
        for _, row in data.iterrows():
            raion = self.raion_manager.get_raion_for_point(
                row['latitude'], row['longitude']
            )
            raions.append(raion)

        data = data.copy()
        data['raion'] = raions

        assigned = data['raion'].notna().sum()
        print(f"  Assigned {assigned:,}/{len(data):,} points to raions")

        return data

    def load_daily_features(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute daily aggregated features per raion.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with daily raion-level features
        """
        print("Loading FIRMS raion-level features...")
        data = self._load_raw_data()

        if data.empty:
            return pd.DataFrame()

        # Assign raions
        data = self._assign_raions(data)
        data = data.dropna(subset=['raion'])

        # Filter date range
        if start_date:
            data = data[data['acq_date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['acq_date'] <= pd.to_datetime(end_date)]

        if data.empty:
            return pd.DataFrame()

        # Get active raions (by fire count)
        raion_counts = data.groupby('raion').size().sort_values(ascending=False)
        active_raions = raion_counts.head(self.max_raions).index.tolist()
        self._active_raions = active_raions

        print(f"  Active raions: {len(active_raions)}")

        # Get date range
        min_date = data['acq_date'].min()
        max_date = data['acq_date'].max()
        date_range = pd.date_range(min_date, max_date, freq='D')

        records = []
        print(f"  Computing daily features for {len(date_range)} days...")

        for date in date_range:
            date_val = date.date()
            day_data = data[data['acq_date'].dt.date == date_val]

            record = {'date': date}

            for raion in active_raions:
                raion_data = day_data[day_data['raion'] == raion]

                # Short name for column
                raion_short = raion.replace(" ", "_").replace("'", "")[:30]

                if len(raion_data) > 0:
                    record[f'fire_count_{raion_short}'] = len(raion_data)
                    record[f'fire_brightness_{raion_short}'] = raion_data['brightness'].mean()

                    if 'frp' in raion_data.columns:
                        frp_valid = raion_data['frp'].dropna()
                        record[f'fire_frp_{raion_short}'] = frp_valid.sum() if len(frp_valid) > 0 else 0.0
                    else:
                        record[f'fire_frp_{raion_short}'] = 0.0

                    if 'daynight' in raion_data.columns:
                        day_count = (raion_data['daynight'] == 'D').sum()
                        record[f'fire_dayratio_{raion_short}'] = day_count / len(raion_data)
                    else:
                        record[f'fire_dayratio_{raion_short}'] = 0.5
                else:
                    record[f'fire_count_{raion_short}'] = 0
                    record[f'fire_brightness_{raion_short}'] = 0.0
                    record[f'fire_frp_{raion_short}'] = 0.0
                    record[f'fire_dayratio_{raion_short}'] = 0.0

            # Aggregated totals
            record['fire_count_total'] = len(day_data)
            if len(day_data) > 0:
                record['fire_brightness_total'] = day_data['brightness'].mean()
                if 'frp' in day_data.columns:
                    record['fire_frp_total'] = day_data['frp'].sum()
                else:
                    record['fire_frp_total'] = 0.0
            else:
                record['fire_brightness_total'] = 0.0
                record['fire_frp_total'] = 0.0

            records.append(record)

        result_df = pd.DataFrame(records)
        print(f"  Generated {len(result_df)} daily records with {len(result_df.columns)} features")

        return result_df

    def get_active_raions(self) -> List[str]:
        """Get list of active raions (must call load_daily_features first)."""
        return self._active_raions or []


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def load_firms_raion(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_raions: int = 50,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS data with raion-level aggregation for modular data system.

    Args:
        start_date: Optional start date
        end_date: Optional end date
        max_raions: Maximum number of raions to include

    Returns:
        Tuple of (DataFrame with raion features, observation mask array)
    """
    loader = FIRMSRaionLoader(max_raions=max_raions)
    df = loader.load_daily_features(start_date, end_date)

    if df.empty:
        return pd.DataFrame(), np.array([])

    # Create observation mask (1 where we have any fires)
    mask = (df['fire_count_total'] > 0).astype(float).values

    return df, mask


def get_raion_boundary_manager(frontline_only: bool = True) -> RaionBoundaryManager:
    """Get a configured raion boundary manager."""
    manager = RaionBoundaryManager(frontline_only=frontline_only)
    manager.load()
    return manager


# =============================================================================
# REGISTRATION
# =============================================================================

RAION_SPATIAL_LOADER = {
    'name': 'firms_raion',
    'loader_fn': load_firms_raion,
    'resolution': 'daily',
    'feature_count': 203,  # 50 raions × 4 features + 3 totals
    'description': 'NASA FIRMS fire hotspots with raion-level aggregation',
}


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Raion Spatial Loader Test")
    print("=" * 60)

    # Test boundary manager
    print("\n1. Testing RaionBoundaryManager...")
    manager = RaionBoundaryManager(frontline_only=True)
    manager.load()

    print(f"   Loaded {len(manager.raions)} frontline raions")

    # Test point lookup (Bakhmut area)
    test_lat, test_lon = 48.595, 38.0
    raion = manager.get_raion_for_point(test_lat, test_lon)
    print(f"   Point ({test_lat}, {test_lon}) -> Raion: {raion}")

    # Test FIRMS raion loader
    print("\n2. Testing FIRMSRaionLoader...")
    loader = FIRMSRaionLoader(raion_manager=manager, max_raions=20)
    df = loader.load_daily_features()

    if not df.empty:
        print(f"\n   Loaded {len(df)} days of features")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Feature columns: {len(df.columns)}")
        print(f"\n   Sample columns:")
        for col in df.columns[:15]:
            print(f"     {col}")
        if len(df.columns) > 15:
            print(f"     ... and {len(df.columns) - 15} more")

        # Show active raions
        active = loader.get_active_raions()
        print(f"\n   Active raions ({len(active)}):")
        for r in active[:10]:
            print(f"     {r}")
        if len(active) > 10:
            print(f"     ... and {len(active) - 10} more")
    else:
        print("   No data loaded")
