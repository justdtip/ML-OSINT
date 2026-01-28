"""
Raion Data Adapter

Adapts raion-level loaders to the MultiResolutionDataset LOADER_REGISTRY interface.

The raion loaders produce:
    - features: [n_days, n_raions, n_features_per_raion]
    - mask: [n_days, n_raions]

The LOADER_REGISTRY expects:
    - df: DataFrame with 'date' column and flattened features
    - mask: 1D array [n_days]

This adapter:
1. Flattens spatial data: [n_days, n_raions, features] -> [n_days, n_raions * features]
2. Preserves per-raion masks in a separate registry for GeographicSourceEncoder
3. Creates fallback 1D masks (any raion observed = timestep observed)

Usage:
    from analysis.loaders.raion_adapter import (
        load_geoconfirmed_raion_adapted,
        load_firms_expanded_raion_adapted,
        get_per_raion_mask,  # For GeographicSourceEncoder
    )

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from analysis.loaders.new_source_raion_loaders import (
    GeoconfirmedRaionLoader,
    AirRaidSirensRaionLoader,
    UCDPRaionLoader,
    WarspottingRaionLoader,
    DeepStateRaionLoader,
    FIRMSExpandedRaionLoader,
    CombinedRaionLoader,
)


# =============================================================================
# PER-RAION MASK REGISTRY
# =============================================================================

# Global registry to store per-raion masks for each adapted source
# Key: source_name, Value: RaionMaskInfo
_RAION_MASK_REGISTRY: Dict[str, 'RaionMaskInfo'] = {}


@dataclass
class RaionMaskInfo:
    """Information about per-raion masks for a source."""
    source_name: str
    mask: np.ndarray  # [n_days, n_raions]
    raion_keys: List[str]
    feature_names: List[str]
    n_raions: int
    n_features_per_raion: int
    dates: List[datetime]


def get_per_raion_mask(source_name: str) -> Optional[RaionMaskInfo]:
    """
    Retrieve per-raion mask info for a source.

    This is used by GeographicSourceEncoder to get the full-granularity
    per-raion observation mask instead of the flattened 1D mask.

    Args:
        source_name: Name of the adapted source (e.g., 'geoconfirmed_raion')

    Returns:
        RaionMaskInfo with per-raion mask, or None if not found
    """
    return _RAION_MASK_REGISTRY.get(source_name)


def get_all_raion_mask_sources() -> List[str]:
    """Get list of all sources with per-raion masks."""
    return list(_RAION_MASK_REGISTRY.keys())


def clear_raion_mask_registry() -> None:
    """Clear the registry (useful for testing)."""
    _RAION_MASK_REGISTRY.clear()


# =============================================================================
# ADAPTER FUNCTIONS
# =============================================================================

def _adapt_raion_loader(
    loader_class,
    source_name: str,
    start_date: Optional[str] = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generic adapter for raion-level loaders.

    Converts:
        features [n_days, n_raions, n_features] -> df [n_days, n_raions * n_features]
        mask [n_days, n_raions] -> mask_1d [n_days] (any observed)

    Also stores per-raion mask in registry for GeographicSourceEncoder.

    Args:
        loader_class: The raion loader class to use
        source_name: Name for registry (e.g., 'geoconfirmed_raion')
        start_date: Start date for data range
        end_date: End date for data range (None = use latest)

    Returns:
        df: DataFrame with 'date' and flattened feature columns
        mask_1d: 1D observation mask [n_days]
    """
    loader = loader_class()

    # Parse dates
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp("2022-02-24")
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    # Load features (active_raions=None lets loader determine from data)
    features, mask, raion_keys, dates = loader.load_raion_features(
        start_date=start,
        end_date=end,
        active_raions=None,  # Auto-determine from data
    )

    # Handle empty data
    if len(features) == 0 or len(raion_keys) == 0:
        print(f"    Warning: No data for {source_name} in date range")
        empty_df = pd.DataFrame(columns=['date'])
        empty_mask = np.array([], dtype=bool)
        return empty_df, empty_mask

    n_days, n_raions, n_features = features.shape

    # Store per-raion mask in registry
    _RAION_MASK_REGISTRY[source_name] = RaionMaskInfo(
        source_name=source_name,
        mask=mask,  # [n_days, n_raions]
        raion_keys=raion_keys,
        feature_names=loader.FEATURE_NAMES,
        n_raions=n_raions,
        n_features_per_raion=n_features,
        dates=dates,
    )

    # Flatten features: [n_days, n_raions, n_features] -> [n_days, n_raions * n_features]
    features_flat = features.reshape(n_days, -1)

    # Create flattened column names
    columns = ['date']
    for raion_idx, raion_key in enumerate(raion_keys):
        for feat_name in loader.FEATURE_NAMES:
            columns.append(f"{raion_key}_{feat_name}")

    # Build DataFrame
    df = pd.DataFrame(
        np.column_stack([np.array(dates), features_flat]),
        columns=['date'] + columns[1:],
    )
    df['date'] = pd.to_datetime(df['date'])

    # Convert feature columns to float
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create 1D mask: any raion observed = timestep observed
    mask_1d = mask.any(axis=1).astype(bool)

    print(f"    Adapted {source_name}: {n_days} days, {n_raions} raions, "
          f"{n_features} features/raion = {n_raions * n_features} total features")

    return df, mask_1d


# =============================================================================
# INDIVIDUAL SOURCE ADAPTERS
# =============================================================================

def load_geoconfirmed_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Geoconfirmed data adapted for MultiResolutionDataset.

    Features: 50 per raion (event types, munition categories, damage assessments)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        GeoconfirmedRaionLoader,
        'geoconfirmed_raion',
        start_date,
        end_date,
    )


def load_air_raid_sirens_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Air Raid Sirens data adapted for MultiResolutionDataset.

    Features: 30 per raion (alert patterns, durations, periodicity)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        AirRaidSirensRaionLoader,
        'air_raid_sirens_raion',
        start_date,
        end_date,
    )


def load_ucdp_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load UCDP data adapted for MultiResolutionDataset.

    Features: 35 per raion (conflict events, casualties, actor types)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        UCDPRaionLoader,
        'ucdp_raion',
        start_date,
        end_date,
    )


def load_warspotting_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load Warspotting data adapted for MultiResolutionDataset.

    Features: 33 per raion (equipment losses by type, verification status)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        WarspottingRaionLoader,
        'warspotting_raion',
        start_date,
        end_date,
    )


def load_deepstate_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load DeepState data adapted for MultiResolutionDataset.

    Features: 48 per raion (territorial control, frontline metrics)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        DeepStateRaionLoader,
        'deepstate_raion',
        start_date,
        end_date,
    )


def load_firms_expanded_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load FIRMS expanded data adapted for MultiResolutionDataset.

    Features: 35 per raion (fire intensity, spatial clustering, temporal patterns)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    return _adapt_raion_loader(
        FIRMSExpandedRaionLoader,
        'firms_expanded_raion',
        start_date,
        end_date,
    )


def load_combined_raion_adapted(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
    min_raion_observations: int = 20,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load combined data from all raion sources adapted for MultiResolutionDataset.

    Features: 231 per raion (all sources combined: geoconfirmed, air_raid,
    ucdp, warspotting, deepstate, firms)

    Returns:
        df: DataFrame with date and flattened raion features
        mask: 1D observation mask
    """
    loader = CombinedRaionLoader()

    # Parse dates
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp("2022-02-24")
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()

    # Load all sources
    results = loader.load_all_sources(
        start_date=start,
        end_date=end,
        min_raion_observations=min_raion_observations,
    )

    # Find reference dates and raions (use first non-empty source)
    ref_dates = None
    ref_raions = None
    for source_name, (features, mask, raions, dates) in results.items():
        if len(features) > 0:
            ref_dates = dates
            ref_raions = raions
            break

    if ref_dates is None:
        print("    Warning: No data for combined_raion in date range")
        empty_df = pd.DataFrame(columns=['date'])
        empty_mask = np.array([], dtype=bool)
        return empty_df, empty_mask

    n_days = len(ref_dates)
    n_raions = len(ref_raions)

    # Concatenate features from all sources
    all_features = []
    all_feature_names = []
    combined_mask = np.zeros((n_days, n_raions), dtype=bool)

    # Define source order and their loaders for feature names
    source_info = [
        ('geoconfirmed', GeoconfirmedRaionLoader.FEATURE_NAMES),
        ('air_raid', AirRaidSirensRaionLoader.FEATURE_NAMES),
        ('ucdp', UCDPRaionLoader.FEATURE_NAMES),
        ('warspotting', WarspottingRaionLoader.FEATURE_NAMES),
        ('deepstate', DeepStateRaionLoader.FEATURE_NAMES),
        ('firms', FIRMSExpandedRaionLoader.FEATURE_NAMES),
    ]

    for source_name, feature_names in source_info:
        if source_name in results:
            features, mask, source_raions, _ = results[source_name]
            if len(features) > 0:
                # Align features and mask to reference raions
                n_features_src = features.shape[2]
                aligned_features = np.zeros((n_days, n_raions, n_features_src))
                aligned_mask = np.zeros((n_days, n_raions), dtype=bool)

                # Build raion index mapping
                source_raion_to_idx = {r: i for i, r in enumerate(source_raions)}
                for ref_idx, ref_raion in enumerate(ref_raions):
                    if ref_raion in source_raion_to_idx:
                        src_idx = source_raion_to_idx[ref_raion]
                        aligned_features[:, ref_idx, :] = features[:, src_idx, :]
                        aligned_mask[:, ref_idx] = mask[:, src_idx]

                all_features.append(aligned_features)
                all_feature_names.extend([f"{source_name}_{f}" for f in feature_names])
                combined_mask |= aligned_mask

                # Register per-source mask
                _RAION_MASK_REGISTRY[f'{source_name}_raion'] = RaionMaskInfo(
                    source_name=f'{source_name}_raion',
                    mask=aligned_mask,
                    raion_keys=ref_raions,
                    feature_names=feature_names,
                    n_raions=n_raions,
                    n_features_per_raion=len(feature_names),
                    dates=ref_dates,
                )

    if not all_features:
        print("    Warning: No valid sources for combined_raion")
        empty_df = pd.DataFrame(columns=['date'])
        empty_mask = np.array([], dtype=bool)
        return empty_df, empty_mask

    # Stack: [n_days, n_raions, total_features]
    combined_features = np.concatenate(all_features, axis=2)
    n_features = combined_features.shape[2]

    # Register combined mask
    _RAION_MASK_REGISTRY['combined_raion'] = RaionMaskInfo(
        source_name='combined_raion',
        mask=combined_mask,
        raion_keys=ref_raions,
        feature_names=all_feature_names,
        n_raions=n_raions,
        n_features_per_raion=n_features,
        dates=ref_dates,
    )

    # Flatten: [n_days, n_raions * n_features]
    features_flat = combined_features.reshape(n_days, -1)

    # Create column names
    columns = ['date']
    for raion_key in ref_raions:
        for feat_name in all_feature_names:
            columns.append(f"{raion_key}_{feat_name}")

    # Build DataFrame
    df = pd.DataFrame(
        np.column_stack([np.array(ref_dates), features_flat]),
        columns=['date'] + columns[1:],
    )
    df['date'] = pd.to_datetime(df['date'])

    # Convert feature columns to float
    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create 1D mask
    mask_1d = combined_mask.any(axis=1).astype(bool)

    print(f"    Adapted combined_raion: {n_days} days, {n_raions} raions, "
          f"{n_features} features/raion = {n_raions * n_features} total features")

    return df, mask_1d


# =============================================================================
# REGISTRY UPDATE HELPER
# =============================================================================

# Source name -> (loader_func, resolution)
RAION_ADAPTER_REGISTRY = {
    'geoconfirmed_raion': load_geoconfirmed_raion_adapted,
    'air_raid_sirens_raion': load_air_raid_sirens_raion_adapted,
    'ucdp_raion': load_ucdp_raion_adapted,
    'warspotting_raion': load_warspotting_raion_adapted,
    'deepstate_raion': load_deepstate_raion_adapted,
    'firms_expanded_raion': load_firms_expanded_raion_adapted,
    'combined_raion': load_combined_raion_adapted,
}


def get_raion_source_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration info for all raion sources.

    Returns dict mapping source_name to:
        - n_raions: Number of raions
        - n_features_per_raion: Features per raion
        - feature_names: List of feature names
        - raion_keys: List of raion keys (after loading)
    """
    configs = {}

    # These are estimates - actual values come from loaded data
    source_features = {
        'geoconfirmed_raion': (50, GeoconfirmedRaionLoader.FEATURE_NAMES),
        'air_raid_sirens_raion': (30, AirRaidSirensRaionLoader.FEATURE_NAMES),
        'ucdp_raion': (35, UCDPRaionLoader.FEATURE_NAMES),
        'warspotting_raion': (33, WarspottingRaionLoader.FEATURE_NAMES),
        'deepstate_raion': (48, DeepStateRaionLoader.FEATURE_NAMES),
        'firms_expanded_raion': (35, FIRMSExpandedRaionLoader.FEATURE_NAMES),
        'combined_raion': (231, CombinedRaionLoader.FEATURE_NAMES),
    }

    for name, (n_features, feature_names) in source_features.items():
        configs[name] = {
            'n_features_per_raion': n_features,
            'feature_names': feature_names,
            # n_raions and raion_keys populated after loading
            'n_raions': None,
            'raion_keys': None,
        }

    return configs


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Raion Data Adapter Test")
    print("=" * 60)

    # Test one adapter
    print("\n1. Testing geoconfirmed_raion adapter...")
    try:
        df, mask = load_geoconfirmed_raion_adapted(
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Columns (first 5): {list(df.columns[:5])}")

        # Check registry
        info = get_per_raion_mask('geoconfirmed_raion')
        if info:
            print(f"   Registry: {info.n_raions} raions, {info.n_features_per_raion} features/raion")
            print(f"   Per-raion mask shape: {info.mask.shape}")
        else:
            print("   Registry: NOT FOUND")
    except Exception as e:
        print(f"   Error: {e}")

    # Test combined adapter
    print("\n2. Testing combined_raion adapter...")
    try:
        df, mask = load_combined_raion_adapted(
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Mask shape: {mask.shape}")

        info = get_per_raion_mask('combined_raion')
        if info:
            print(f"   Registry: {info.n_raions} raions, {info.n_features_per_raion} features/raion")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Listing all registered sources...")
    for source in get_all_raion_mask_sources():
        info = get_per_raion_mask(source)
        if info:
            print(f"   {source}: {info.n_raions} raions, {info.n_features_per_raion} features")

    print("\n" + "=" * 60)
    print("Test complete!")
