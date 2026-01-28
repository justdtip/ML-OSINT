"""
Data Loaders Package

This package contains specialized data loaders for various spatial and temporal
data sources used in the Multi-Resolution HAN training pipeline.

Available Loaders:
- deepstate_spatial_loader: Military unit positions, frontline metrics, attack directions
- firms_spatial_loader: Fire hotspot data with regional binning

Usage:
    from analysis.loaders import load_deepstate_spatial, load_firms_tiled

    # Load DeepState spatial features
    df, mask = load_deepstate_spatial(start_date, end_date, spatial_mode='tiled')

    # Load FIRMS tiled features
    df, mask = load_firms_tiled(start_date, end_date)
"""

from .deepstate_spatial_loader import (
    DeepStateSpatialLoader,
    load_deepstate_spatial,
    load_deepstate_unit_positions,
    DEEPSTATE_SPATIAL_LOADER,
    UKRAINE_REGIONS,
)

from .firms_spatial_loader import (
    FIRMSSpatialLoader,
    load_firms_tiled,
    load_firms_aggregated,
    FIRMS_SPATIAL_LOADER,
)

from .raion_spatial_loader import (
    RaionBoundaryManager,
    FIRMSRaionLoader,
    load_firms_raion,
    get_raion_boundary_manager,
    RAION_SPATIAL_LOADER,
)

from .viirs_spatial_loader import (
    VIIRSSpatialLoader,
    load_viirs_tiled,
    VIIRS_SPATIAL_LOADER,
    TILE_REGIONS,
)

from .new_source_raion_loaders import (
    GeoconfirmedRaionLoader,
    AirRaidSirensRaionLoader,
    UCDPRaionLoader,
    WarspottingRaionLoader,
    DeepStateRaionLoader,
    FIRMSExpandedRaionLoader,
    CombinedRaionLoader,
)

from .raion_adapter import (
    # Adapted loaders for LOADER_REGISTRY integration
    load_geoconfirmed_raion_adapted,
    load_air_raid_sirens_raion_adapted,
    load_ucdp_raion_adapted,
    load_warspotting_raion_adapted,
    load_deepstate_raion_adapted,
    load_firms_expanded_raion_adapted,
    load_combined_raion_adapted,
    # Per-raion mask retrieval for GeographicSourceEncoder
    get_per_raion_mask,
    get_all_raion_mask_sources,
    clear_raion_mask_registry,
    RaionMaskInfo,
    RAION_ADAPTER_REGISTRY,
)

__all__ = [
    # DeepState
    'DeepStateSpatialLoader',
    'load_deepstate_spatial',
    'load_deepstate_unit_positions',
    'DEEPSTATE_SPATIAL_LOADER',
    'UKRAINE_REGIONS',
    # FIRMS (regional)
    'FIRMSSpatialLoader',
    'load_firms_tiled',
    'load_firms_aggregated',
    'FIRMS_SPATIAL_LOADER',
    # Raion-level
    'RaionBoundaryManager',
    'FIRMSRaionLoader',
    'load_firms_raion',
    'get_raion_boundary_manager',
    'RAION_SPATIAL_LOADER',
    # VIIRS (tile-based)
    'VIIRSSpatialLoader',
    'load_viirs_tiled',
    'VIIRS_SPATIAL_LOADER',
    'TILE_REGIONS',
    # New raion-level sources (expanded features)
    'GeoconfirmedRaionLoader',   # 50 features
    'AirRaidSirensRaionLoader',  # 30 features
    'UCDPRaionLoader',           # 35 features
    'WarspottingRaionLoader',    # 33 features
    'DeepStateRaionLoader',      # 48 features
    'FIRMSExpandedRaionLoader',  # 35 features
    'CombinedRaionLoader',       # Unified loader (231 features total)
    # Raion adapters for LOADER_REGISTRY integration
    'load_geoconfirmed_raion_adapted',
    'load_air_raid_sirens_raion_adapted',
    'load_ucdp_raion_adapted',
    'load_warspotting_raion_adapted',
    'load_deepstate_raion_adapted',
    'load_firms_expanded_raion_adapted',
    'load_combined_raion_adapted',
    # Per-raion mask utilities
    'get_per_raion_mask',
    'get_all_raion_mask_sources',
    'clear_raion_mask_registry',
    'RaionMaskInfo',
    'RAION_ADAPTER_REGISTRY',
]
