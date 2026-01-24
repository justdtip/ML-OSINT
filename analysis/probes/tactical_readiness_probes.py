"""
Tactical Prediction Readiness Probes for Multi-Resolution HAN Model

This module provides diagnostic probes to assess whether the current Multi-Resolution
HAN model can be adapted for finer spatial and temporal resolution predictions,
specifically for tactical-level forecasting in military conflict scenarios.

Current Model Capabilities:
- National-level daily/monthly predictions
- Data sources: DeepState, FIRMS, VIIRS, Equipment losses, Personnel losses,
  Sentinel (6 tiles), UCDP, HDX

Key Questions Addressed:
1. Can we move to spatial decomposition (Oblast -> Sector -> Grid)?
2. Can we move to finer temporal resolution (12h, 6h, hourly)?
3. Is entity-level tracking feasible (units, infrastructure)?

Probe Categories:
- 7.1 Spatial Decomposition Potential
- 7.2 Entity-Level Readiness
- 7.3 Prediction Resolution Requirements

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from shapely.geometry import Polygon, Point, MultiPolygon, box
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    warnings.warn("Shapely not available. Polygon operations will use bounding boxes.")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR as CONFIG_DATA_DIR,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = PROJECT_ROOT
DATA_DIR = CONFIG_DATA_DIR
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR


def get_output_dir():
    """Get current output directory for figures."""
    return get_probe_figures_dir()


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SpatialGranularity(Enum):
    """Spatial resolution levels"""
    NATIONAL = "national"
    OBLAST = "oblast"
    RAION = "raion"
    SECTOR = "sector"
    GRID_10KM = "grid_10km"
    GRID_1KM = "grid_1km"
    COORDINATE = "coordinate"


class TemporalGranularity(Enum):
    """Temporal resolution levels"""
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    TWELVE_HOUR = "12h"
    SIX_HOUR = "6h"
    HOURLY = "hourly"


class DataSource(Enum):
    """Available data sources"""
    DEEPSTATE = "deepstate"
    FIRMS = "firms"
    VIIRS = "viirs"
    SENTINEL = "sentinel"
    EQUIPMENT = "equipment"
    PERSONNEL = "personnel"
    UCDP = "ucdp"
    VIINA = "viina"
    HDX_CONFLICT = "hdx_conflict"
    HDX_FOOD = "hdx_food"


# Ukrainian Oblasts with approximate bounding boxes (lon_min, lat_min, lon_max, lat_max)
OBLAST_BBOXES = {
    "Donetsk": (36.7, 46.8, 39.0, 49.3),
    "Luhansk": (38.0, 48.0, 40.2, 50.1),
    "Kharkiv": (34.5, 48.4, 38.5, 50.4),
    "Zaporizhzhia": (34.0, 46.5, 36.9, 48.2),
    "Kherson": (32.0, 45.8, 35.5, 47.7),
    "Dnipropetrovsk": (33.0, 47.8, 36.3, 49.5),
    "Mykolaiv": (30.2, 46.2, 33.4, 48.0),
    "Sumy": (32.7, 50.0, 35.6, 52.4),
    "Chernihiv": (30.8, 50.6, 33.5, 52.4),
    "Kyiv_Oblast": (29.2, 49.4, 32.2, 51.6),
    "Crimea": (32.4, 44.4, 36.7, 46.2),
}

# Tactical sector definitions with approximate bounding polygons
# These are the key operational sectors along the front line
TACTICAL_SECTORS = {
    "kharkiv": {
        "name": "Kharkiv Sector",
        "description": "Northern Kharkiv oblast front including Vovchansk and border areas",
        "bbox": (36.5, 49.5, 38.2, 50.4),
        "polygon_coords": [
            (36.5, 50.4), (38.2, 50.4), (38.2, 49.5), (36.5, 49.5)
        ],
        "key_locations": ["Vovchansk", "Kupiansk", "Kharkiv city"],
        "active_since": "2022-02-24"
    },
    "luhansk_svatove_kreminna": {
        "name": "Svatove-Kreminna Axis",
        "description": "Luhansk oblast western front, Svatove-Kreminna line",
        "bbox": (37.5, 48.8, 38.8, 49.8),
        "polygon_coords": [
            (37.5, 49.8), (38.8, 49.8), (38.8, 48.8), (37.5, 48.8)
        ],
        "key_locations": ["Svatove", "Kreminna", "Starobilsk"],
        "active_since": "2022-09-01"
    },
    "donetsk_north": {
        "name": "Bakhmut-Siversk Sector",
        "description": "Northern Donetsk oblast including Bakhmut salient and Siversk",
        "bbox": (37.5, 48.3, 38.5, 49.0),
        "polygon_coords": [
            (37.5, 49.0), (38.5, 49.0), (38.5, 48.3), (37.5, 48.3)
        ],
        "key_locations": ["Bakhmut", "Siversk", "Chasiv Yar", "Soledar"],
        "active_since": "2022-05-01"
    },
    "donetsk_central": {
        "name": "Avdiivka-Marinka Sector",
        "description": "Central Donetsk oblast around Avdiivka and western Donetsk city",
        "bbox": (37.2, 47.7, 38.2, 48.3),
        "polygon_coords": [
            (37.2, 48.3), (38.2, 48.3), (38.2, 47.7), (37.2, 47.7)
        ],
        "key_locations": ["Avdiivka", "Marinka", "Donetsk", "Kurakhove"],
        "active_since": "2022-02-24"
    },
    "donetsk_south": {
        "name": "Vuhledar Sector",
        "description": "Southern Donetsk oblast including Vuhledar and Velyka Novosilka",
        "bbox": (36.5, 47.0, 37.8, 47.8),
        "polygon_coords": [
            (36.5, 47.8), (37.8, 47.8), (37.8, 47.0), (36.5, 47.0)
        ],
        "key_locations": ["Vuhledar", "Velyka Novosilka", "Pavlivka"],
        "active_since": "2022-02-24"
    },
    "zaporizhzhia": {
        "name": "Zaporizhzhia Sector",
        "description": "Zaporizhzhia oblast front including Orikhiv direction",
        "bbox": (35.0, 46.8, 36.8, 48.0),
        "polygon_coords": [
            (35.0, 48.0), (36.8, 48.0), (36.8, 46.8), (35.0, 46.8)
        ],
        "key_locations": ["Orikhiv", "Tokmak", "Robotyne", "Melitopol"],
        "active_since": "2022-02-24"
    },
    "kherson": {
        "name": "Kherson Sector",
        "description": "Kherson oblast including Dnipro river line",
        "bbox": (32.5, 46.0, 35.0, 47.5),
        "polygon_coords": [
            (32.5, 47.5), (35.0, 47.5), (35.0, 46.0), (32.5, 46.0)
        ],
        "key_locations": ["Kherson", "Nova Kakhovka", "Kinburn Spit"],
        "active_since": "2022-02-24"
    },
    "kursk": {
        "name": "Kursk Incursion Sector",
        "description": "Kursk oblast (Russia) Ukrainian incursion zone",
        "bbox": (34.5, 51.0, 36.5, 52.0),
        "polygon_coords": [
            (34.5, 52.0), (36.5, 52.0), (36.5, 51.0), (34.5, 51.0)
        ],
        "key_locations": ["Sudzha", "Korenevo", "Lgov"],
        "active_since": "2024-08-06"
    }
}

# VIIRS tile definitions (from context - 6 tiles)
VIIRS_TILES = {
    "h19v03": {"name": "Northwest Ukraine", "bbox": (22.0, 48.0, 32.0, 56.0)},
    "h20v03": {"name": "Northeast Ukraine", "bbox": (32.0, 48.0, 42.0, 56.0)},
    "h19v04": {"name": "Southwest Ukraine", "bbox": (22.0, 40.0, 32.0, 48.0)},
    "h20v04": {"name": "Southeast Ukraine", "bbox": (32.0, 40.0, 42.0, 48.0)},
    "h21v03": {"name": "Eastern Russia Border", "bbox": (42.0, 48.0, 52.0, 56.0)},
    "h21v04": {"name": "Crimea/Black Sea", "bbox": (42.0, 40.0, 52.0, 48.0)},
}


# =============================================================================
# 7.1.1 DATA AVAILABILITY AUDIT
# =============================================================================

@dataclass
class GranularitySupport:
    """Support level for a given granularity"""
    granularity: Union[SpatialGranularity, TemporalGranularity]
    is_available: bool
    data_density: Optional[float] = None  # observations per unit (e.g., per day per sector)
    limiting_factors: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SourceAuditResult:
    """Complete audit result for a data source"""
    source: DataSource
    spatial_granularity: Dict[str, GranularitySupport] = field(default_factory=dict)
    temporal_granularity: Dict[str, GranularitySupport] = field(default_factory=dict)
    native_resolution: str = ""
    total_records: int = 0
    date_range: Tuple[str, str] = ("", "")
    has_coordinates: bool = False
    coordinate_precision: str = ""
    recommended_use_level: str = ""


class DataAvailabilityAudit:
    """
    Audit each data source for regional granularity capabilities.

    Section 7.1.1: Regional Signal Availability
    - Check: National, Oblast, Raion, Coordinate level availability
    - Flag limiting factors for each source
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.audit_results: Dict[str, SourceAuditResult] = {}

    def audit_all_sources(self) -> Dict[str, SourceAuditResult]:
        """Run audit on all available data sources."""
        self.audit_results = {
            "deepstate": self._audit_deepstate(),
            "firms": self._audit_firms(),
            "sentinel": self._audit_sentinel(),
            "equipment": self._audit_equipment(),
            "personnel": self._audit_personnel(),
            "ucdp": self._audit_ucdp(),
            "viina": self._audit_viina(),
        }
        return self.audit_results

    def _audit_deepstate(self) -> SourceAuditResult:
        """Audit DeepState front line data."""
        result = SourceAuditResult(source=DataSource.DEEPSTATE)

        daily_dir = self.data_dir / "deepstate" / "daily"
        wayback_dir = self.data_dir / "deepstate" / "wayback_snapshots"

        # Count available files
        daily_files = list(daily_dir.glob("*.geojson")) if daily_dir.exists() else []
        wayback_files = list(wayback_dir.glob("*.json")) if wayback_dir.exists() else []

        result.total_records = len(daily_files) + len(wayback_files)

        # Sample a file to check structure
        sample_file = None
        if daily_files:
            sample_file = daily_files[0]
        elif wayback_files:
            sample_file = wayback_files[0]

        if sample_file:
            try:
                with open(sample_file, 'r') as f:
                    data = json.load(f)

                result.has_coordinates = True
                result.coordinate_precision = "Polygon vertices (high precision ~10m)"
                result.native_resolution = "Daily snapshots with coordinate polygons"

                # Extract date range from filenames
                dates = []
                for f in daily_files:
                    match = re.search(r'(\d{8})', f.name)
                    if match:
                        dates.append(match.group(1))
                if dates:
                    result.date_range = (min(dates), max(dates))

            except Exception as e:
                result.native_resolution = f"Error reading: {e}"

        # Spatial granularity assessment
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Aggregated front line metrics available"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=True,
                data_density=1.0,
                notes="Polygons can be filtered by oblast bounding box"
            ),
            SpatialGranularity.SECTOR.value: GranularitySupport(
                granularity=SpatialGranularity.SECTOR,
                is_available=True,
                data_density=0.8,
                notes="Front line polygons align well with tactical sectors"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=True,
                data_density=1.0,
                notes="Full coordinate data available for all features"
            ),
        }

        # Temporal granularity assessment
        result.temporal_granularity = {
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=1.0,
                notes="Daily snapshots available since July 2024"
            ),
            TemporalGranularity.TWELVE_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.TWELVE_HOUR,
                is_available=False,
                limiting_factors=["Single daily snapshot only", "No intraday updates"],
                notes="Would require more frequent scraping"
            ),
            TemporalGranularity.HOURLY.value: GranularitySupport(
                granularity=TemporalGranularity.HOURLY,
                is_available=False,
                limiting_factors=["Map updates ~1x daily", "No sub-daily resolution"],
                notes="Not feasible without live API access"
            ),
        }

        result.recommended_use_level = "SECTOR-DAILY: Excellent for sector-level daily tracking"

        return result

    def _audit_firms(self) -> SourceAuditResult:
        """Audit FIRMS fire detection data."""
        result = SourceAuditResult(source=DataSource.FIRMS)

        firms_dir = self.data_dir / "firms" / "DL_FIRE_SV-C2_706038"
        archive_file = firms_dir / "fire_archive_SV-C2_706038.csv"
        nrt_file = firms_dir / "fire_nrt_SV-C2_706038.csv"

        if archive_file.exists():
            try:
                # Read sample to check structure
                df = pd.read_csv(archive_file, nrows=10000)
                result.total_records = sum(1 for _ in open(archive_file)) - 1

                result.has_coordinates = True
                result.coordinate_precision = "Point coordinates (375m VIIRS resolution)"
                result.native_resolution = "Individual fire detections with lat/lon"

                if 'latitude' in df.columns and 'longitude' in df.columns:
                    lat_precision = df['latitude'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0).mean()
                    result.coordinate_precision = f"Point coordinates (~{int(lat_precision)} decimal places, ~375m resolution)"

                if 'acq_date' in df.columns:
                    df_full = pd.read_csv(archive_file, usecols=['acq_date'])
                    dates = pd.to_datetime(df_full['acq_date'])
                    result.date_range = (dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d'))

            except Exception as e:
                result.native_resolution = f"Error: {e}"

        # Spatial granularity
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Aggregated fire counts available"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=True,
                data_density=0.9,
                notes="Point data can be filtered by oblast"
            ),
            SpatialGranularity.SECTOR.value: GranularitySupport(
                granularity=SpatialGranularity.SECTOR,
                is_available=True,
                data_density=0.7,
                notes="Sector-level aggregation possible from coordinates"
            ),
            SpatialGranularity.GRID_10KM.value: GranularitySupport(
                granularity=SpatialGranularity.GRID_10KM,
                is_available=True,
                data_density=0.5,
                limiting_factors=["Sparse coverage in non-combat areas"],
                notes="Grid cells may have zero observations"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=True,
                data_density=1.0,
                notes="Full lat/lon available for every detection"
            ),
        }

        # Temporal granularity
        result.temporal_granularity = {
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=1.0,
                notes="Date field available for all records"
            ),
            TemporalGranularity.TWELVE_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.TWELVE_HOUR,
                is_available=True,
                data_density=0.8,
                notes="Day/night classification (D/N) enables 12h resolution"
            ),
            TemporalGranularity.SIX_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.SIX_HOUR,
                is_available=True,
                data_density=0.6,
                limiting_factors=["Satellite overpasses not evenly distributed"],
                notes="acq_time field provides ~4 hour precision"
            ),
            TemporalGranularity.HOURLY.value: GranularitySupport(
                granularity=TemporalGranularity.HOURLY,
                is_available=False,
                limiting_factors=["Satellite revisit ~4x per day", "Gaps between overpasses"],
                notes="acq_time gives hour but sparse temporal coverage"
            ),
        }

        result.recommended_use_level = "SECTOR-12H: Good for sector-level sub-daily intensity"

        return result

    def _audit_sentinel(self) -> SourceAuditResult:
        """Audit Sentinel satellite data."""
        result = SourceAuditResult(source=DataSource.SENTINEL)

        sentinel_dir = self.data_dir / "sentinel"
        timeseries_file = sentinel_dir / "sentinel_timeseries_raw.json"

        if timeseries_file.exists():
            try:
                with open(timeseries_file, 'r') as f:
                    data = json.load(f)

                result.native_resolution = "Monthly aggregated scene counts by collection"
                result.has_coordinates = True
                result.coordinate_precision = f"Bounding box: {data.get('bbox', 'unknown')}"

                result.date_range = (
                    data.get('start_date', ''),
                    data.get('end_date', '')
                )

                # Count total scenes
                total = 0
                for collection, coll_data in data.get('collections', {}).items():
                    for month_data in coll_data.get('monthly', []):
                        total += month_data.get('count', 0)
                result.total_records = total

            except Exception as e:
                result.native_resolution = f"Error: {e}"

        # Spatial granularity (6 tiles mentioned in context)
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Aggregated across all tiles"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=True,
                data_density=0.7,
                limiting_factors=["Tile boundaries don't align with oblasts"],
                notes="6 tiles cover Ukraine, can approximate oblast coverage"
            ),
            SpatialGranularity.SECTOR.value: GranularitySupport(
                granularity=SpatialGranularity.SECTOR,
                is_available=False,
                limiting_factors=[
                    "Current data is tile-aggregated",
                    "Would need raw scene downloads",
                    "Cloud cover limits usable observations"
                ],
                notes="Requires scene-level processing for sector analysis"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=False,
                limiting_factors=[
                    "Only metadata currently available",
                    "Pixel-level analysis requires full scene download"
                ],
                notes="Theoretically available at 10-60m resolution"
            ),
        }

        # Temporal granularity
        result.temporal_granularity = {
            TemporalGranularity.MONTHLY.value: GranularitySupport(
                granularity=TemporalGranularity.MONTHLY,
                is_available=True,
                data_density=1.0,
                notes="Current aggregation level"
            ),
            TemporalGranularity.WEEKLY.value: GranularitySupport(
                granularity=TemporalGranularity.WEEKLY,
                is_available=True,
                data_density=0.8,
                notes="Weekly available from raw dates"
            ),
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=0.5,
                limiting_factors=["5-day revisit for Sentinel-2", "Cloud cover gaps"],
                notes="~19 unique dates per month (every ~1.5 days)"
            ),
            TemporalGranularity.HOURLY.value: GranularitySupport(
                granularity=TemporalGranularity.HOURLY,
                is_available=False,
                limiting_factors=["Polar orbit satellites", "No hourly revisit capability"],
                notes="Not physically possible with current constellation"
            ),
        }

        result.recommended_use_level = "TILE-WEEKLY: Good for regional weekly trends"

        return result

    def _audit_equipment(self) -> SourceAuditResult:
        """Audit equipment losses data."""
        result = SourceAuditResult(source=DataSource.EQUIPMENT)

        equipment_dir = self.data_dir / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data"
        equipment_file = equipment_dir / "russia_losses_equipment.json"
        oryx_file = equipment_dir / "russia_losses_equipment_oryx.json"

        if equipment_file.exists():
            try:
                with open(equipment_file, 'r') as f:
                    data = json.load(f)
                result.total_records = len(data)
                result.native_resolution = "Daily cumulative equipment counts (national level)"
                result.has_coordinates = False

                if data:
                    dates = [item.get('date', '') for item in data if 'date' in item]
                    if dates:
                        result.date_range = (min(dates), max(dates))

            except Exception as e:
                result.native_resolution = f"Error: {e}"

        # Spatial granularity (very limited - national only)
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Official UA MOD reports at national level"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=False,
                limiting_factors=[
                    "No geographic attribution in official data",
                    "Would require Oryx photo geolocation"
                ],
                notes="Oryx data has some unit annotations that could be geolocated"
            ),
            SpatialGranularity.SECTOR.value: GranularitySupport(
                granularity=SpatialGranularity.SECTOR,
                is_available=False,
                limiting_factors=["No sector-level reporting"],
                notes="Infeasible with current data"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=False,
                limiting_factors=["No coordinate data"],
                notes="Would require manual geolocation of Oryx images"
            ),
        }

        # Temporal granularity
        result.temporal_granularity = {
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=1.0,
                notes="Daily reports from UA MOD"
            ),
            TemporalGranularity.TWELVE_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.TWELVE_HOUR,
                is_available=False,
                limiting_factors=["Single daily report"],
                notes="Not available"
            ),
            TemporalGranularity.HOURLY.value: GranularitySupport(
                granularity=TemporalGranularity.HOURLY,
                is_available=False,
                limiting_factors=["Single daily report"],
                notes="Not available"
            ),
        }

        result.recommended_use_level = "NATIONAL-DAILY: Limited to national-level daily trends"

        return result

    def _audit_personnel(self) -> SourceAuditResult:
        """Audit personnel losses data."""
        result = SourceAuditResult(source=DataSource.PERSONNEL)

        personnel_dir = self.data_dir / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data"
        personnel_file = personnel_dir / "russia_losses_personnel.json"

        if personnel_file.exists():
            try:
                with open(personnel_file, 'r') as f:
                    data = json.load(f)
                result.total_records = len(data)
                result.native_resolution = "Daily cumulative personnel counts (national level)"
                result.has_coordinates = False

                if data:
                    dates = [item.get('date', '') for item in data if 'date' in item]
                    if dates:
                        result.date_range = (min(dates), max(dates))

            except Exception as e:
                result.native_resolution = f"Error: {e}"

        # Same limitations as equipment
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Official estimates at national level"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=False,
                limiting_factors=["No geographic breakdown available"],
                notes="Would require open-source investigation"
            ),
            SpatialGranularity.SECTOR.value: GranularitySupport(
                granularity=SpatialGranularity.SECTOR,
                is_available=False,
                limiting_factors=["No sector-level data"],
                notes="Infeasible"
            ),
        }

        result.temporal_granularity = {
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=1.0,
                notes="Daily cumulative counts"
            ),
            TemporalGranularity.TWELVE_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.TWELVE_HOUR,
                is_available=False,
                limiting_factors=["Single daily update"],
                notes="Not available"
            ),
        }

        result.recommended_use_level = "NATIONAL-DAILY: Limited to national-level daily trends"

        return result

    def _audit_ucdp(self) -> SourceAuditResult:
        """Audit UCDP conflict events data."""
        result = SourceAuditResult(source=DataSource.UCDP)

        ucdp_file = self.data_dir / "ucdp" / "ged_events.csv"

        if ucdp_file.exists():
            try:
                df = pd.read_csv(ucdp_file, nrows=1000)
                result.total_records = sum(1 for _ in open(ucdp_file)) - 1
                result.native_resolution = "Individual conflict events with coordinates"

                result.has_coordinates = 'latitude' in df.columns and 'longitude' in df.columns
                if result.has_coordinates:
                    result.coordinate_precision = "Event coordinates (precision varies)"

                if 'date_start' in df.columns:
                    df_dates = pd.read_csv(ucdp_file, usecols=['date_start'])
                    dates = pd.to_datetime(df_dates['date_start'], errors='coerce')
                    result.date_range = (
                        dates.min().strftime('%Y-%m-%d') if pd.notna(dates.min()) else '',
                        dates.max().strftime('%Y-%m-%d') if pd.notna(dates.max()) else ''
                    )

            except Exception as e:
                result.native_resolution = f"Error: {e}"

        # Spatial granularity - UCDP has location data
        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Aggregated event counts"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=True,
                data_density=0.8,
                notes="Admin region data available (adm_1, adm_2)"
            ),
            SpatialGranularity.RAION.value: GranularitySupport(
                granularity=SpatialGranularity.RAION,
                is_available=True,
                data_density=0.5,
                limiting_factors=["Some events have low precision"],
                notes="where_prec field indicates location precision"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=True,
                data_density=0.7,
                limiting_factors=["Precision varies (where_prec 1-7)"],
                notes="High precision for some, approximate for others"
            ),
        }

        # Temporal granularity
        result.temporal_granularity = {
            TemporalGranularity.MONTHLY.value: GranularitySupport(
                granularity=TemporalGranularity.MONTHLY,
                is_available=True,
                data_density=1.0,
                notes="Standard aggregation level"
            ),
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=0.8,
                limiting_factors=["Some multi-day events", "Publication lag"],
                notes="date_start/date_end available"
            ),
            TemporalGranularity.HOURLY.value: GranularitySupport(
                granularity=TemporalGranularity.HOURLY,
                is_available=False,
                limiting_factors=["No time-of-day data"],
                notes="Only date precision available"
            ),
        }

        result.recommended_use_level = "OBLAST-DAILY: Good for oblast-level daily event analysis"

        return result

    def _audit_viina(self) -> SourceAuditResult:
        """Audit VIINA territorial control data."""
        result = SourceAuditResult(source=DataSource.VIINA)

        viina_dir = self.data_dir / "viina"

        # Check for extracted CSV files
        viina_files = list(viina_dir.rglob("*.csv")) if viina_dir.exists() else []

        if viina_files:
            try:
                sample_file = viina_files[0]
                df = pd.read_csv(sample_file, nrows=1000)
                result.total_records = sum(len(pd.read_csv(f)) for f in viina_files)
                result.native_resolution = "Locality-level territorial control"
                result.has_coordinates = 'lat' in df.columns or 'latitude' in df.columns

            except Exception as e:
                result.native_resolution = f"Error: {e}"
        else:
            result.native_resolution = "Data files not found"

        result.spatial_granularity = {
            SpatialGranularity.NATIONAL.value: GranularitySupport(
                granularity=SpatialGranularity.NATIONAL,
                is_available=True,
                data_density=1.0,
                notes="Aggregated control percentages"
            ),
            SpatialGranularity.OBLAST.value: GranularitySupport(
                granularity=SpatialGranularity.OBLAST,
                is_available=True,
                data_density=0.9,
                notes="Locality data has oblast attribution"
            ),
            SpatialGranularity.COORDINATE.value: GranularitySupport(
                granularity=SpatialGranularity.COORDINATE,
                is_available=True,
                data_density=0.8,
                notes="Locality coordinates available"
            ),
        }

        result.temporal_granularity = {
            TemporalGranularity.DAILY.value: GranularitySupport(
                granularity=TemporalGranularity.DAILY,
                is_available=True,
                data_density=1.0,
                notes="Daily control snapshots"
            ),
            TemporalGranularity.TWELVE_HOUR.value: GranularitySupport(
                granularity=TemporalGranularity.TWELVE_HOUR,
                is_available=False,
                limiting_factors=["Single daily snapshot"],
                notes="Not available"
            ),
        }

        result.recommended_use_level = "OBLAST-DAILY: Good for oblast-level control tracking"

        return result

    def generate_availability_matrix(self) -> pd.DataFrame:
        """Generate a data availability matrix across all sources and granularities."""
        if not self.audit_results:
            self.audit_all_sources()

        rows = []

        for source_name, result in self.audit_results.items():
            row = {"source": source_name}

            # Spatial granularities
            for gran in SpatialGranularity:
                key = gran.value
                if key in result.spatial_granularity:
                    support = result.spatial_granularity[key]
                    row[f"spatial_{key}"] = "YES" if support.is_available else "NO"
                    row[f"spatial_{key}_density"] = support.data_density if support.data_density else 0
                else:
                    row[f"spatial_{key}"] = "N/A"
                    row[f"spatial_{key}_density"] = 0

            # Temporal granularities
            for gran in TemporalGranularity:
                key = gran.value
                if key in result.temporal_granularity:
                    support = result.temporal_granularity[key]
                    row[f"temporal_{key}"] = "YES" if support.is_available else "NO"
                    row[f"temporal_{key}_density"] = support.data_density if support.data_density else 0
                else:
                    row[f"temporal_{key}"] = "N/A"
                    row[f"temporal_{key}_density"] = 0

            row["has_coordinates"] = result.has_coordinates
            row["recommended_level"] = result.recommended_use_level

            rows.append(row)

        return pd.DataFrame(rows)

    def get_limiting_factors_summary(self) -> Dict[str, List[str]]:
        """Summarize all limiting factors by source."""
        if not self.audit_results:
            self.audit_all_sources()

        summary = {}

        for source_name, result in self.audit_results.items():
            factors = []

            for gran_dict in [result.spatial_granularity, result.temporal_granularity]:
                for gran_name, support in gran_dict.items():
                    if support.limiting_factors:
                        factors.extend([f"[{gran_name}] {f}" for f in support.limiting_factors])

            summary[source_name] = factors

        return summary


# =============================================================================
# 7.1.2 SECTOR DEFINITION
# =============================================================================

@dataclass
class TacticalSector:
    """Definition of a tactical sector with polygon geometry."""
    name: str
    sector_id: str
    description: str
    bbox: Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    polygon: Optional[Any] = None  # Shapely Polygon if available
    key_locations: List[str] = field(default_factory=list)
    active_since: str = ""

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if a point is within this sector."""
        if self.polygon is not None and HAS_SHAPELY:
            return self.polygon.contains(Point(lon, lat))
        else:
            # Fallback to bounding box
            lon_min, lat_min, lon_max, lat_max = self.bbox
            return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "sector_id": self.sector_id,
            "description": self.description,
            "bbox": self.bbox,
            "key_locations": self.key_locations,
            "active_since": self.active_since,
        }
        if self.polygon is not None and HAS_SHAPELY:
            result["polygon_wkt"] = self.polygon.wkt
        return result


class SectorDefinition:
    """
    Section 7.1.2: Front-Line Sector Definition

    Define tactical sectors from DeepState data and create bounding polygons
    for filtering FIRMS/VIIRS/DeepState per sector.
    """

    def __init__(self):
        self.sectors: Dict[str, TacticalSector] = {}
        self.oblasts: Dict[str, TacticalSector] = {}
        self._initialize_sectors()
        self._initialize_oblasts()

    def _initialize_sectors(self):
        """Initialize tactical sector definitions."""
        for sector_id, sector_data in TACTICAL_SECTORS.items():
            polygon = None
            if HAS_SHAPELY:
                # Create polygon from coordinates
                coords = sector_data.get("polygon_coords", [])
                if coords:
                    # Close the polygon
                    if coords[0] != coords[-1]:
                        coords = coords + [coords[0]]
                    polygon = Polygon(coords)
                else:
                    # Create from bounding box
                    polygon = box(*sector_data["bbox"])

            self.sectors[sector_id] = TacticalSector(
                name=sector_data["name"],
                sector_id=sector_id,
                description=sector_data["description"],
                bbox=sector_data["bbox"],
                polygon=polygon,
                key_locations=sector_data.get("key_locations", []),
                active_since=sector_data.get("active_since", "")
            )

    def _initialize_oblasts(self):
        """Initialize oblast definitions."""
        for oblast_name, bbox in OBLAST_BBOXES.items():
            polygon = None
            if HAS_SHAPELY:
                polygon = box(*bbox)

            self.oblasts[oblast_name.lower()] = TacticalSector(
                name=oblast_name,
                sector_id=f"oblast_{oblast_name.lower()}",
                description=f"{oblast_name} oblast administrative boundary",
                bbox=bbox,
                polygon=polygon
            )

    def get_sector(self, sector_id: str) -> Optional[TacticalSector]:
        """Get a sector by ID."""
        return self.sectors.get(sector_id)

    def get_oblast(self, oblast_name: str) -> Optional[TacticalSector]:
        """Get an oblast by name."""
        return self.oblasts.get(oblast_name.lower())

    def classify_point(self, lon: float, lat: float) -> Dict[str, Optional[str]]:
        """Classify a point into sector and oblast."""
        result = {"sector": None, "oblast": None}

        for sector_id, sector in self.sectors.items():
            if sector.contains_point(lon, lat):
                result["sector"] = sector_id
                break

        for oblast_name, oblast in self.oblasts.items():
            if oblast.contains_point(lon, lat):
                result["oblast"] = oblast_name
                break

        return result

    def filter_firms_by_sector(
        self,
        df: pd.DataFrame,
        sector_id: str,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude'
    ) -> pd.DataFrame:
        """Filter FIRMS data to a specific sector."""
        sector = self.sectors.get(sector_id)
        if sector is None:
            raise ValueError(f"Unknown sector: {sector_id}")

        if HAS_SHAPELY and sector.polygon is not None:
            # Use shapely for accurate filtering
            mask = df.apply(
                lambda row: sector.polygon.contains(Point(row[lon_col], row[lat_col])),
                axis=1
            )
        else:
            # Use bounding box
            lon_min, lat_min, lon_max, lat_max = sector.bbox
            mask = (
                (df[lon_col] >= lon_min) & (df[lon_col] <= lon_max) &
                (df[lat_col] >= lat_min) & (df[lat_col] <= lat_max)
            )

        return df[mask].copy()

    def filter_deepstate_by_sector(
        self,
        geojson_data: Dict,
        sector_id: str
    ) -> Dict:
        """Filter DeepState GeoJSON features to a specific sector."""
        sector = self.sectors.get(sector_id)
        if sector is None:
            raise ValueError(f"Unknown sector: {sector_id}")

        filtered_features = []

        for feature in geojson_data.get("features", []):
            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type", "")

            # Check if any part of the geometry intersects the sector
            if HAS_SHAPELY and sector.polygon is not None:
                try:
                    from shapely.geometry import shape
                    geom = shape(geometry)
                    if sector.polygon.intersects(geom):
                        filtered_features.append(feature)
                except Exception:
                    # Fallback to bounding box check
                    if self._geometry_intersects_bbox(geometry, sector.bbox):
                        filtered_features.append(feature)
            else:
                if self._geometry_intersects_bbox(geometry, sector.bbox):
                    filtered_features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": filtered_features,
            "sector": sector_id
        }

    def _geometry_intersects_bbox(
        self,
        geometry: Dict,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check if a GeoJSON geometry intersects a bounding box."""
        lon_min, lat_min, lon_max, lat_max = bbox

        def check_coords(coords):
            if isinstance(coords[0], (int, float)):
                # Single coordinate pair [lon, lat]
                lon, lat = coords[0], coords[1]
                return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
            else:
                # Nested coordinates
                return any(check_coords(c) for c in coords)

        coords = geometry.get("coordinates", [])
        return check_coords(coords) if coords else False

    def export_sector_definitions(self, output_path: Optional[Path] = None) -> Dict:
        """Export all sector definitions to JSON."""
        output = {
            "generated_at": datetime.now().isoformat(),
            "shapely_available": HAS_SHAPELY,
            "sectors": {sid: s.to_dict() for sid, s in self.sectors.items()},
            "oblasts": {oid: o.to_dict() for oid, o in self.oblasts.items()},
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)

        return output

    def generate_sector_coverage_report(self) -> pd.DataFrame:
        """Generate a report on sector spatial coverage."""
        rows = []

        for sector_id, sector in self.sectors.items():
            lon_min, lat_min, lon_max, lat_max = sector.bbox
            area_approx = (lon_max - lon_min) * (lat_max - lat_min) * 111 * 111 * np.cos(np.radians((lat_min + lat_max) / 2))

            rows.append({
                "sector_id": sector_id,
                "name": sector.name,
                "lon_min": lon_min,
                "lat_min": lat_min,
                "lon_max": lon_max,
                "lat_max": lat_max,
                "approx_area_km2": round(area_approx, 1),
                "key_locations": ", ".join(sector.key_locations),
                "active_since": sector.active_since
            })

        return pd.DataFrame(rows)


# =============================================================================
# 7.1.3 SECTOR CORRELATION PROBE
# =============================================================================

class SectorCorrelationProbe:
    """
    Section 7.1.3: Sector Independence Test

    Compute correlation between sectors for each feature to identify
    sector-specific vs national signals.
    """

    def __init__(self, sector_definition: SectorDefinition):
        self.sector_def = sector_definition
        self.correlation_results: Dict[str, pd.DataFrame] = {}

    def compute_sector_correlations_firms(
        self,
        df: pd.DataFrame,
        date_col: str = 'acq_date',
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        value_col: str = 'frp'
    ) -> Dict[str, Any]:
        """
        Compute correlations between sectors for FIRMS data.

        Returns:
            Dictionary with correlation matrix, sector independence scores,
            and identification of sector-specific vs national signals.
        """
        # Assign each point to a sector
        df = df.copy()
        df['sector'] = df.apply(
            lambda row: self.sector_def.classify_point(row[lon_col], row[lat_col])['sector'],
            axis=1
        )

        # Filter to points within defined sectors
        df_sectored = df[df['sector'].notna()].copy()

        if df_sectored.empty:
            return {"error": "No points fall within defined sectors"}

        # Aggregate by date and sector
        df_sectored[date_col] = pd.to_datetime(df_sectored[date_col])

        sector_daily = df_sectored.groupby([date_col, 'sector']).agg({
            value_col: ['sum', 'mean', 'count']
        }).reset_index()
        sector_daily.columns = [date_col, 'sector', f'{value_col}_sum', f'{value_col}_mean', 'fire_count']

        # Pivot to wide format for correlation
        pivot_sum = sector_daily.pivot(index=date_col, columns='sector', values=f'{value_col}_sum').fillna(0)
        pivot_count = sector_daily.pivot(index=date_col, columns='sector', values='fire_count').fillna(0)

        # Compute correlation matrices
        corr_sum = pivot_sum.corr()
        corr_count = pivot_count.corr()

        # Compute independence scores (1 - mean absolute correlation with other sectors)
        independence_scores = {}
        for sector in corr_sum.columns:
            other_corrs = corr_sum.loc[sector].drop(sector).abs()
            independence_scores[sector] = 1 - other_corrs.mean() if len(other_corrs) > 0 else 1.0

        # Identify sector-specific vs national signals
        signal_classification = {}
        for sector in corr_sum.columns:
            mean_corr = corr_sum.loc[sector].drop(sector).mean() if len(corr_sum.columns) > 1 else 0
            if mean_corr > 0.7:
                signal_classification[sector] = "NATIONAL (high correlation with other sectors)"
            elif mean_corr > 0.4:
                signal_classification[sector] = "MIXED (moderate correlation)"
            else:
                signal_classification[sector] = "SECTOR-SPECIFIC (low correlation)"

        self.correlation_results['firms'] = {
            'correlation_matrix_sum': corr_sum,
            'correlation_matrix_count': corr_count,
            'independence_scores': independence_scores,
            'signal_classification': signal_classification,
            'sector_counts': sector_daily.groupby('sector')['fire_count'].sum().to_dict(),
            'date_range': (df_sectored[date_col].min(), df_sectored[date_col].max()),
            'total_observations': len(df_sectored)
        }

        return self.correlation_results['firms']

    def compute_temporal_correlation_by_sector(
        self,
        df: pd.DataFrame,
        sector_id: str,
        date_col: str = 'acq_date',
        value_col: str = 'frp',
        lags: List[int] = [1, 2, 3, 7, 14]
    ) -> Dict[str, float]:
        """Compute autocorrelation at various lags for a specific sector."""
        sector = self.sector_def.get_sector(sector_id)
        if sector is None:
            return {"error": f"Unknown sector: {sector_id}"}

        # Filter to sector
        df_sector = self.sector_def.filter_firms_by_sector(df, sector_id)

        if df_sector.empty:
            return {"error": "No data in sector"}

        # Aggregate by date
        df_sector[date_col] = pd.to_datetime(df_sector[date_col])
        daily = df_sector.groupby(date_col)[value_col].sum().sort_index()

        # Compute autocorrelations
        autocorrs = {}
        for lag in lags:
            if len(daily) > lag:
                autocorrs[f"lag_{lag}d"] = daily.autocorr(lag=lag)
            else:
                autocorrs[f"lag_{lag}d"] = np.nan

        return autocorrs

    def generate_independence_report(self) -> pd.DataFrame:
        """Generate a report on sector independence across all computed correlations."""
        rows = []

        for source, results in self.correlation_results.items():
            if 'error' in results:
                continue

            for sector, score in results.get('independence_scores', {}).items():
                rows.append({
                    'source': source,
                    'sector': sector,
                    'independence_score': score,
                    'signal_type': results.get('signal_classification', {}).get(sector, 'Unknown'),
                    'observation_count': results.get('sector_counts', {}).get(sector, 0)
                })

        return pd.DataFrame(rows)


# =============================================================================
# 7.2 ENTITY-LEVEL READINESS
# =============================================================================

@dataclass
class EntityStateVector:
    """Specification for an entity state vector."""
    entity_type: str
    attributes: Dict[str, Dict[str, Any]]
    data_sources: List[str]
    feasibility_score: float  # 0-1, how feasible to populate
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "data_sources": self.data_sources,
            "feasibility_score": self.feasibility_score,
            "notes": self.notes
        }


class EntitySchemaSpec:
    """
    Section 7.2: Entity-Level Readiness

    7.2.1: Audit for unit-level information availability
    7.2.2: Specify entity state vectors for units and infrastructure
    """

    def __init__(self):
        self.unit_schema = self._define_unit_schema()
        self.infrastructure_schema = self._define_infrastructure_schema()
        self.data_source_audit = self._audit_entity_data_sources()

    def _define_unit_schema(self) -> EntityStateVector:
        """Define the unit entity state vector (Section 7.2.2)."""
        return EntityStateVector(
            entity_type="military_unit",
            attributes={
                "strength": {
                    "description": "Estimated personnel strength",
                    "type": "float",
                    "range": [0, 1],  # Normalized 0=depleted, 1=full strength
                    "update_frequency": "daily",
                    "source_availability": "LOW",
                    "notes": "Requires milblogger analysis, Telegram monitoring"
                },
                "equipment_count": {
                    "description": "Major equipment items remaining",
                    "type": "int",
                    "update_frequency": "weekly",
                    "source_availability": "MEDIUM",
                    "notes": "Oryx tracks some unit attributions"
                },
                "position": {
                    "description": "Estimated center of operations (lat, lon)",
                    "type": "tuple[float, float]",
                    "update_frequency": "daily",
                    "source_availability": "MEDIUM",
                    "notes": "DeepState unit markers, milblogger reports"
                },
                "days_in_contact": {
                    "description": "Days since unit entered active combat",
                    "type": "int",
                    "update_frequency": "daily",
                    "source_availability": "LOW",
                    "notes": "Requires tracking deployment history"
                },
                "loss_rate": {
                    "description": "7-day rolling loss rate (equipment/day)",
                    "type": "float",
                    "update_frequency": "weekly",
                    "source_availability": "LOW",
                    "notes": "Oryx + milblogger correlation needed"
                },
                "unit_type": {
                    "description": "Unit classification (motorized_rifle, tank, VDV, etc)",
                    "type": "categorical",
                    "update_frequency": "static",
                    "source_availability": "HIGH",
                    "notes": "Well-documented in open sources"
                },
                "echelon": {
                    "description": "Organizational level (brigade, regiment, battalion)",
                    "type": "categorical",
                    "update_frequency": "static",
                    "source_availability": "HIGH",
                    "notes": "Well-documented"
                },
                "affiliation": {
                    "description": "Command structure (Western MD, Southern MD, etc)",
                    "type": "categorical",
                    "update_frequency": "static",
                    "source_availability": "HIGH",
                    "notes": "Well-documented"
                }
            },
            data_sources=[
                "DeepState unit markers",
                "Oryx (unit annotations on some losses)",
                "UA General Staff briefings",
                "Milblogger reports (Telegram)",
                "ISW assessments"
            ],
            feasibility_score=0.35,
            notes="Unit tracking is partially feasible. Static attributes (type, echelon) "
                  "are well-documented. Dynamic attributes (strength, losses) require "
                  "extensive OSINT integration and are often estimates."
        )

    def _define_infrastructure_schema(self) -> EntityStateVector:
        """Define the infrastructure entity state vector."""
        return EntityStateVector(
            entity_type="infrastructure",
            attributes={
                "type": {
                    "description": "Infrastructure type",
                    "type": "categorical",
                    "values": ["airfield", "depot", "bridge", "rail_junction", "HQ", "SAM_site"],
                    "update_frequency": "static",
                    "source_availability": "HIGH",
                    "notes": "Well-mapped in OSM and military sources"
                },
                "status": {
                    "description": "Operational status",
                    "type": "categorical",
                    "values": ["operational", "damaged", "destroyed", "unknown"],
                    "update_frequency": "event-driven",
                    "source_availability": "MEDIUM",
                    "notes": "Strike reports, satellite imagery"
                },
                "last_activity": {
                    "description": "Last observed activity date",
                    "type": "date",
                    "update_frequency": "irregular",
                    "source_availability": "LOW",
                    "notes": "Requires regular satellite monitoring"
                },
                "strategic_value": {
                    "description": "Strategic importance score",
                    "type": "float",
                    "range": [0, 1],
                    "update_frequency": "monthly",
                    "source_availability": "MEDIUM",
                    "notes": "Can be derived from location, type, capacity"
                },
                "position": {
                    "description": "Location (lat, lon)",
                    "type": "tuple[float, float]",
                    "update_frequency": "static",
                    "source_availability": "HIGH",
                    "notes": "Well-mapped"
                },
                "capacity": {
                    "description": "Functional capacity (type-dependent metric)",
                    "type": "float",
                    "update_frequency": "event-driven",
                    "source_availability": "LOW",
                    "notes": "Difficult to assess remotely"
                }
            },
            data_sources=[
                "DeepState airfield markers",
                "OpenStreetMap",
                "Satellite imagery (Sentinel, commercial)",
                "Strike reports",
                "Social media geolocations"
            ],
            feasibility_score=0.55,
            notes="Infrastructure tracking is more feasible than unit tracking. "
                  "Static attributes are well-mapped. Status updates require "
                  "satellite imagery analysis or strike report correlation."
        )

    def _audit_entity_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Audit data sources for entity-level information (Section 7.2.1)."""
        return {
            "oryx": {
                "description": "Visual confirmation of equipment losses",
                "entity_coverage": "PARTIAL",
                "unit_annotations": True,
                "notes": "Some losses have unit attribution in description text",
                "update_frequency": "Daily",
                "feasibility": "MEDIUM - requires NLP extraction of unit names"
            },
            "ua_general_staff": {
                "description": "Official Ukrainian military briefings",
                "entity_coverage": "MINIMAL",
                "unit_annotations": False,
                "notes": "National-level aggregates only, no unit breakdown",
                "update_frequency": "Daily",
                "feasibility": "LOW - no entity-level data"
            },
            "deepstate": {
                "description": "Front line map with unit markers",
                "entity_coverage": "GOOD",
                "unit_annotations": True,
                "notes": "256+ unit markers with type/echelon information",
                "update_frequency": "Daily",
                "feasibility": "HIGH - structured unit marker data"
            },
            "milbloggers": {
                "description": "Telegram/social media reports",
                "entity_coverage": "VARIABLE",
                "unit_annotations": True,
                "notes": "Rich unit-level detail but unstructured",
                "update_frequency": "Real-time",
                "feasibility": "LOW - requires NLP + verification"
            },
            "isw": {
                "description": "ISW daily assessments",
                "entity_coverage": "PARTIAL",
                "unit_annotations": True,
                "notes": "Mentions specific units in context",
                "update_frequency": "Daily",
                "feasibility": "MEDIUM - semi-structured, NLP extractable"
            }
        }

    def assess_entity_tracking_feasibility(self) -> Dict[str, Any]:
        """Overall assessment of entity-level tracking feasibility."""
        unit_attrs = self.unit_schema.attributes
        infra_attrs = self.infrastructure_schema.attributes

        # Count attributes by availability
        def count_by_availability(attrs):
            counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for attr in attrs.values():
                avail = attr.get("source_availability", "LOW")
                counts[avail] = counts.get(avail, 0) + 1
            return counts

        unit_counts = count_by_availability(unit_attrs)
        infra_counts = count_by_availability(infra_attrs)

        return {
            "unit_tracking": {
                "feasibility_score": self.unit_schema.feasibility_score,
                "attribute_availability": unit_counts,
                "recommended_approach": "Focus on position tracking and static attributes. "
                                        "Dynamic attributes (strength, losses) require "
                                        "significant NLP and manual verification.",
                "data_sources_required": self.unit_schema.data_sources
            },
            "infrastructure_tracking": {
                "feasibility_score": self.infrastructure_schema.feasibility_score,
                "attribute_availability": infra_counts,
                "recommended_approach": "Start with airfields (DeepState markers available). "
                                        "Expand to other types with satellite imagery.",
                "data_sources_required": self.infrastructure_schema.data_sources
            },
            "overall_recommendation": "Entity-level tracking is PARTIALLY FEASIBLE. "
                                      "Recommend starting with infrastructure entities "
                                      "and DeepState unit markers as proof of concept. "
                                      "Full unit state tracking requires significant "
                                      "NLP pipeline development."
        }

    def export_schemas(self, output_path: Optional[Path] = None) -> Dict:
        """Export entity schemas to JSON."""
        output = {
            "generated_at": datetime.now().isoformat(),
            "unit_schema": self.unit_schema.to_dict(),
            "infrastructure_schema": self.infrastructure_schema.to_dict(),
            "data_source_audit": self.data_source_audit,
            "feasibility_assessment": self.assess_entity_tracking_feasibility()
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)

        return output


# =============================================================================
# 7.3 RESOLUTION ANALYSIS PROBE
# =============================================================================

@dataclass
class ResolutionTradeoff:
    """Tradeoff analysis for a specific resolution."""
    resolution: Union[SpatialGranularity, TemporalGranularity]
    data_availability: Dict[str, float]  # source -> availability score (0-1)
    expected_performance: Dict[str, float]  # metric -> expected value
    limiting_factors: List[str]
    recommendations: List[str]
    overall_feasibility: str  # "HIGH", "MEDIUM", "LOW", "NOT_FEASIBLE"


class ResolutionAnalysisProbe:
    """
    Section 7.3: Prediction Resolution Requirements

    7.3.1: Temporal resolution analysis (12h, 6h, hourly)
    7.3.2: Spatial resolution analysis (Oblast, Sector, Grid, Point)
    """

    def __init__(self, data_availability_audit: DataAvailabilityAudit):
        self.audit = data_availability_audit
        self.temporal_analysis: Dict[str, ResolutionTradeoff] = {}
        self.spatial_analysis: Dict[str, ResolutionTradeoff] = {}

    def analyze_temporal_resolution(self) -> Dict[str, ResolutionTradeoff]:
        """
        Section 7.3.1: Assess requirements for 12h, 6h, hourly predictions.
        """
        if not self.audit.audit_results:
            self.audit.audit_all_sources()

        resolutions = [
            TemporalGranularity.TWELVE_HOUR,
            TemporalGranularity.SIX_HOUR,
            TemporalGranularity.HOURLY
        ]

        for resolution in resolutions:
            # Compute data availability per source
            availability = {}
            limiting = []

            for source_name, result in self.audit.audit_results.items():
                if resolution.value in result.temporal_granularity:
                    support = result.temporal_granularity[resolution.value]
                    availability[source_name] = support.data_density if support.is_available else 0.0
                    if support.limiting_factors:
                        limiting.extend([f"[{source_name}] {f}" for f in support.limiting_factors])
                else:
                    availability[source_name] = 0.0

            # Estimate performance degradation
            mean_availability = np.mean(list(availability.values()))

            expected_perf = {
                "forecast_accuracy_degradation": round(max(0, (1 - mean_availability) * 0.3), 2),
                "coverage_completeness": round(mean_availability, 2),
                "prediction_confidence": round(mean_availability * 0.8, 2),
            }

            # Generate recommendations
            recommendations = []
            if mean_availability < 0.3:
                recommendations.append("NOT RECOMMENDED: Insufficient data density")
            elif mean_availability < 0.5:
                recommendations.append("CAUTION: Sparse data coverage, high uncertainty")
            elif mean_availability < 0.7:
                recommendations.append("FEASIBLE with caveats: Some sources limited")
            else:
                recommendations.append("GOOD: Adequate data coverage")

            # Specific recommendations
            if resolution == TemporalGranularity.TWELVE_HOUR:
                if 'firms' in availability and availability['firms'] > 0.5:
                    recommendations.append("FIRMS day/night split enables 12h analysis")
            elif resolution == TemporalGranularity.HOURLY:
                recommendations.append("CRITICAL: Most sources lack hourly data")
                recommendations.append("Consider synthetic hourly interpolation with uncertainty")

            # Determine overall feasibility
            if mean_availability >= 0.6:
                feasibility = "HIGH"
            elif mean_availability >= 0.4:
                feasibility = "MEDIUM"
            elif mean_availability >= 0.2:
                feasibility = "LOW"
            else:
                feasibility = "NOT_FEASIBLE"

            self.temporal_analysis[resolution.value] = ResolutionTradeoff(
                resolution=resolution,
                data_availability=availability,
                expected_performance=expected_perf,
                limiting_factors=limiting,
                recommendations=recommendations,
                overall_feasibility=feasibility
            )

        return self.temporal_analysis

    def analyze_spatial_resolution(self) -> Dict[str, ResolutionTradeoff]:
        """
        Section 7.3.2: Assess requirements for Oblast, Sector, Grid cell, Point level.
        """
        if not self.audit.audit_results:
            self.audit.audit_all_sources()

        resolutions = [
            SpatialGranularity.OBLAST,
            SpatialGranularity.SECTOR,
            SpatialGranularity.GRID_10KM,
            SpatialGranularity.COORDINATE
        ]

        for resolution in resolutions:
            availability = {}
            limiting = []

            for source_name, result in self.audit.audit_results.items():
                if resolution.value in result.spatial_granularity:
                    support = result.spatial_granularity[resolution.value]
                    availability[source_name] = support.data_density if support.is_available else 0.0
                    if support.limiting_factors:
                        limiting.extend([f"[{source_name}] {f}" for f in support.limiting_factors])
                else:
                    availability[source_name] = 0.0

            mean_availability = np.mean(list(availability.values()))

            expected_perf = {
                "spatial_accuracy": round(mean_availability * 0.9, 2),
                "data_density_per_unit": round(mean_availability, 2),
                "cross_source_coverage": round(sum(1 for v in availability.values() if v > 0.3) / len(availability), 2),
            }

            recommendations = []

            if resolution == SpatialGranularity.OBLAST:
                recommendations.append("RECOMMENDED: Good coverage across most sources")
                recommendations.append("Use for: Regional trend analysis, operational planning")
            elif resolution == SpatialGranularity.SECTOR:
                if mean_availability > 0.5:
                    recommendations.append("FEASIBLE: Key sources (DeepState, FIRMS) support sectors")
                    recommendations.append("Use for: Tactical-level prediction, sector comparison")
                else:
                    recommendations.append("CAUTION: Equipment/Personnel lack sector data")
            elif resolution == SpatialGranularity.GRID_10KM:
                recommendations.append("HIGH RESOLUTION: May have sparse coverage")
                recommendations.append("Consider aggregation strategies for sparse cells")
            elif resolution == SpatialGranularity.COORDINATE:
                recommendations.append("POINT-LEVEL: Use for specific event analysis")
                recommendations.append("Not recommended for prediction at this granularity")

            if mean_availability >= 0.6:
                feasibility = "HIGH"
            elif mean_availability >= 0.4:
                feasibility = "MEDIUM"
            elif mean_availability >= 0.2:
                feasibility = "LOW"
            else:
                feasibility = "NOT_FEASIBLE"

            self.spatial_analysis[resolution.value] = ResolutionTradeoff(
                resolution=resolution,
                data_availability=availability,
                expected_performance=expected_perf,
                limiting_factors=limiting,
                recommendations=recommendations,
                overall_feasibility=feasibility
            )

        return self.spatial_analysis

    def generate_resolution_tradeoff_tables(self) -> Dict[str, pd.DataFrame]:
        """Generate comprehensive tradeoff tables for temporal and spatial resolution."""
        if not self.temporal_analysis:
            self.analyze_temporal_resolution()
        if not self.spatial_analysis:
            self.analyze_spatial_resolution()

        # Temporal resolution table
        temporal_rows = []
        for res_name, tradeoff in self.temporal_analysis.items():
            row = {
                "resolution": res_name,
                "overall_feasibility": tradeoff.overall_feasibility,
            }
            for source, avail in tradeoff.data_availability.items():
                row[f"{source}_availability"] = avail
            for metric, value in tradeoff.expected_performance.items():
                row[metric] = value
            temporal_rows.append(row)

        # Spatial resolution table
        spatial_rows = []
        for res_name, tradeoff in self.spatial_analysis.items():
            row = {
                "resolution": res_name,
                "overall_feasibility": tradeoff.overall_feasibility,
            }
            for source, avail in tradeoff.data_availability.items():
                row[f"{source}_availability"] = avail
            for metric, value in tradeoff.expected_performance.items():
                row[metric] = value
            spatial_rows.append(row)

        return {
            "temporal": pd.DataFrame(temporal_rows),
            "spatial": pd.DataFrame(spatial_rows)
        }

    def get_optimal_resolution_recommendation(self) -> Dict[str, Any]:
        """Recommend optimal temporal and spatial resolution based on analysis."""
        if not self.temporal_analysis:
            self.analyze_temporal_resolution()
        if not self.spatial_analysis:
            self.analyze_spatial_resolution()

        # Find best temporal resolution that is feasible
        temporal_ranking = ["daily", "12h", "6h", "hourly"]
        best_temporal = "daily"  # Default
        for res in temporal_ranking:
            if res in self.temporal_analysis:
                if self.temporal_analysis[res].overall_feasibility in ["HIGH", "MEDIUM"]:
                    best_temporal = res
                    break

        # Find best spatial resolution that is feasible
        spatial_ranking = ["sector", "oblast", "grid_10km", "coordinate"]
        best_spatial = "oblast"  # Default
        for res in spatial_ranking:
            if res in self.spatial_analysis:
                if self.spatial_analysis[res].overall_feasibility in ["HIGH", "MEDIUM"]:
                    best_spatial = res
                    break

        return {
            "recommended_temporal": best_temporal,
            "recommended_spatial": best_spatial,
            "temporal_feasibility": self.temporal_analysis.get(best_temporal, {}).overall_feasibility if best_temporal in self.temporal_analysis else "HIGH",
            "spatial_feasibility": self.spatial_analysis.get(best_spatial, {}).overall_feasibility if best_spatial in self.spatial_analysis else "HIGH",
            "combined_recommendation": f"SECTOR-DAILY predictions are recommended as the optimal "
                                        f"balance of granularity and data availability. "
                                        f"12H resolution is achievable for FIRMS-based features.",
            "upgrade_path": [
                "Phase 1: Implement SECTOR-DAILY predictions",
                "Phase 2: Add 12H resolution for fire/thermal features",
                "Phase 3: Investigate entity-level tracking with DeepState unit data",
                "Phase 4: Consider grid-cell analysis for high-activity sectors"
            ]
        }


# =============================================================================
# MAIN PROBE RUNNER
# =============================================================================

class TacticalReadinessProbe:
    """
    Main orchestrator for all tactical prediction readiness probes.

    Runs all probes and generates comprehensive reports.
    """

    def __init__(self, data_dir: Path = DATA_DIR, output_dir: Path = None):
        self.data_dir = data_dir
        if output_dir is None:
            output_dir = get_output_dir()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize probe components
        self.data_audit = DataAvailabilityAudit(data_dir)
        self.sector_def = SectorDefinition()
        self.sector_correlation = SectorCorrelationProbe(self.sector_def)
        self.entity_schema = EntitySchemaSpec()
        self.resolution_probe = ResolutionAnalysisProbe(self.data_audit)

    def run_all_probes(self, export_results: bool = True) -> Dict[str, Any]:
        """Run all tactical readiness probes and return comprehensive results."""
        print("=" * 70)
        print("TACTICAL PREDICTION READINESS ASSESSMENT")
        print("Multi-Resolution HAN Model - Finer Resolution Feasibility")
        print("=" * 70)
        print()

        results = {}

        # 7.1.1 Data Availability Audit
        print("[7.1.1] Running Data Availability Audit...")
        self.data_audit.audit_all_sources()
        availability_matrix = self.data_audit.generate_availability_matrix()
        limiting_factors = self.data_audit.get_limiting_factors_summary()
        results["data_availability"] = {
            "matrix": availability_matrix.to_dict(),
            "limiting_factors": limiting_factors
        }
        print(f"  - Audited {len(self.data_audit.audit_results)} data sources")
        print()

        # 7.1.2 Sector Definition
        print("[7.1.2] Generating Sector Definitions...")
        sector_report = self.sector_def.generate_sector_coverage_report()
        sector_export = self.sector_def.export_sector_definitions()
        results["sector_definitions"] = {
            "coverage_report": sector_report.to_dict(),
            "sectors": sector_export
        }
        print(f"  - Defined {len(self.sector_def.sectors)} tactical sectors")
        print(f"  - Defined {len(self.sector_def.oblasts)} oblast regions")
        print()

        # 7.1.3 Sector Correlation (if FIRMS data available)
        print("[7.1.3] Sector Independence Analysis...")
        firms_file = self.data_dir / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
        if firms_file.exists():
            try:
                print("  - Loading FIRMS data for correlation analysis...")
                firms_df = pd.read_csv(firms_file)
                correlation_results = self.sector_correlation.compute_sector_correlations_firms(firms_df)
                independence_report = self.sector_correlation.generate_independence_report()
                results["sector_correlation"] = {
                    "independence_report": independence_report.to_dict() if not independence_report.empty else {},
                    "signal_classification": correlation_results.get("signal_classification", {})
                }
                print(f"  - Analyzed correlations across {len(correlation_results.get('sector_counts', {}))} sectors")
            except Exception as e:
                print(f"  - Warning: Could not complete correlation analysis: {e}")
                results["sector_correlation"] = {"error": str(e)}
        else:
            print("  - FIRMS data not available, skipping correlation analysis")
            results["sector_correlation"] = {"error": "FIRMS data not found"}
        print()

        # 7.2 Entity Schema
        print("[7.2] Entity-Level Readiness Assessment...")
        entity_assessment = self.entity_schema.assess_entity_tracking_feasibility()
        entity_export = self.entity_schema.export_schemas()
        results["entity_readiness"] = entity_assessment
        print(f"  - Unit tracking feasibility: {entity_assessment['unit_tracking']['feasibility_score']:.0%}")
        print(f"  - Infrastructure tracking feasibility: {entity_assessment['infrastructure_tracking']['feasibility_score']:.0%}")
        print()

        # 7.3 Resolution Analysis
        print("[7.3] Resolution Requirements Analysis...")
        temporal_analysis = self.resolution_probe.analyze_temporal_resolution()
        spatial_analysis = self.resolution_probe.analyze_spatial_resolution()
        tradeoff_tables = self.resolution_probe.generate_resolution_tradeoff_tables()
        optimal_recommendation = self.resolution_probe.get_optimal_resolution_recommendation()

        results["resolution_analysis"] = {
            "temporal": {k: {"feasibility": v.overall_feasibility, "recommendations": v.recommendations}
                        for k, v in temporal_analysis.items()},
            "spatial": {k: {"feasibility": v.overall_feasibility, "recommendations": v.recommendations}
                       for k, v in spatial_analysis.items()},
            "optimal_recommendation": optimal_recommendation,
            "tradeoff_tables": {
                "temporal": tradeoff_tables["temporal"].to_dict(),
                "spatial": tradeoff_tables["spatial"].to_dict()
            }
        }
        print(f"  - Recommended resolution: {optimal_recommendation['recommended_spatial']}-{optimal_recommendation['recommended_temporal']}")
        print()

        # Export results if requested
        if export_results:
            self._export_results(results, availability_matrix, sector_report, tradeoff_tables)

        # Print summary
        self._print_summary(results)

        return results

    def _export_results(
        self,
        results: Dict,
        availability_matrix: pd.DataFrame,
        sector_report: pd.DataFrame,
        tradeoff_tables: Dict[str, pd.DataFrame]
    ):
        """Export all results to files."""
        print("Exporting results...")

        # JSON summary
        summary_path = self.output_dir / "tactical_readiness_summary.json"

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        with open(summary_path, 'w') as f:
            json.dump(convert_types(results), f, indent=2, default=str)
        print(f"  - Summary: {summary_path}")

        # CSV tables
        availability_matrix.to_csv(self.output_dir / "data_availability_matrix.csv", index=False)
        print(f"  - Availability matrix: {self.output_dir / 'data_availability_matrix.csv'}")

        sector_report.to_csv(self.output_dir / "sector_coverage_report.csv", index=False)
        print(f"  - Sector report: {self.output_dir / 'sector_coverage_report.csv'}")

        tradeoff_tables["temporal"].to_csv(self.output_dir / "temporal_resolution_tradeoffs.csv", index=False)
        tradeoff_tables["spatial"].to_csv(self.output_dir / "spatial_resolution_tradeoffs.csv", index=False)
        print(f"  - Resolution tradeoffs: {self.output_dir / 'temporal_resolution_tradeoffs.csv'}")
        print(f"  - Resolution tradeoffs: {self.output_dir / 'spatial_resolution_tradeoffs.csv'}")

        # Sector definitions JSON
        self.sector_def.export_sector_definitions(self.output_dir / "sector_definitions.json")
        print(f"  - Sector definitions: {self.output_dir / 'sector_definitions.json'}")

        # Entity schemas JSON
        self.entity_schema.export_schemas(self.output_dir / "entity_schemas.json")
        print(f"  - Entity schemas: {self.output_dir / 'entity_schemas.json'}")

        print()

    def _print_summary(self, results: Dict):
        """Print executive summary of probe results."""
        print("=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        print()

        print("DATA AVAILABILITY:")
        print("-" * 40)
        for source, audit in self.data_audit.audit_results.items():
            print(f"  {source:15} | Coords: {'YES' if audit.has_coordinates else 'NO':3} | "
                  f"Level: {audit.recommended_use_level}")
        print()

        print("RESOLUTION FEASIBILITY:")
        print("-" * 40)
        res = results.get("resolution_analysis", {})
        for temporal, data in res.get("temporal", {}).items():
            print(f"  Temporal {temporal:6} | {data['feasibility']:12}")
        for spatial, data in res.get("spatial", {}).items():
            print(f"  Spatial {spatial:8} | {data['feasibility']:12}")
        print()

        print("OPTIMAL RECOMMENDATION:")
        print("-" * 40)
        opt = res.get("optimal_recommendation", {})
        print(f"  {opt.get('combined_recommendation', 'N/A')}")
        print()

        print("ENTITY TRACKING:")
        print("-" * 40)
        entity = results.get("entity_readiness", {})
        print(f"  Unit tracking:     {entity.get('unit_tracking', {}).get('feasibility_score', 0):.0%} feasible")
        print(f"  Infrastructure:    {entity.get('infrastructure_tracking', {}).get('feasibility_score', 0):.0%} feasible")
        print()

        print("UPGRADE PATH:")
        print("-" * 40)
        for step in opt.get("upgrade_path", []):
            print(f"  - {step}")
        print()

        print("=" * 70)
        print(f"Full results exported to: {self.output_dir}")
        print("=" * 70)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Run tactical readiness probes from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tactical Prediction Readiness Probes for Multi-Resolution HAN"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to output directory"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting results to files"
    )

    args = parser.parse_args()

    probe = TacticalReadinessProbe(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    results = probe.run_all_probes(export_results=not args.no_export)

    return results


if __name__ == "__main__":
    main()
