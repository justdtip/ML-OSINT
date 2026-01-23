"""
Real Data Loader for Hierarchical Attention Network

Loads and preprocesses actual OSINT data from all sources:
- UCDP conflict events
- FIRMS fire detections
- Sentinel satellite data
- DeepState front line data
- Equipment losses
- Personnel losses

Handles:
- Temporal alignment (different resolutions to monthly aggregation)
- Missing data imputation
- Domain-specific normalization
- Train/val/test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime, timedelta
from collections import defaultdict

# PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

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


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ucdp_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load and aggregate UCDP conflict events to specified resolution.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with aggregated UCDP data
    """
    ucdp_path = DATA_DIR / "ucdp" / "ged_events.csv"

    if not ucdp_path.exists():
        print(f"UCDP data not found at {ucdp_path}")
        return pd.DataFrame()

    df = pd.read_csv(ucdp_path)

    # Parse dates
    df['date'] = pd.to_datetime(df['date_start'], format='mixed', errors='coerce')
    df = df.dropna(subset=['date'])

    # Filter to Ukraine conflict (2022+)
    df = df[df['date'] >= '2022-02-01']

    # Create period for aggregation based on resolution
    if resolution == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
        freq = 'W-MON'
    else:  # monthly (default)
        df['period'] = df['date'].dt.to_period('M')
        freq = 'MS'

    # Aggregate by period
    aggregated = df.groupby('period').agg({
        # Events by violence type
        'type_of_violence': lambda x: {
            'state_based': (x == 1).sum(),
            'non_state': (x == 2).sum(),
            'one_sided': (x == 3).sum()
        },
        # Deaths
        'deaths_a': 'sum',
        'deaths_b': 'sum',
        'deaths_civilians': 'sum',
        'deaths_unknown': 'sum',
        'best_est': 'sum',
        'high_est': 'sum',
        'low_est': 'sum',
        # Event counts
        'id': 'count'
    }).reset_index()

    # Flatten violence type dict
    aggregated['events_state_based'] = aggregated['type_of_violence'].apply(lambda x: x.get('state_based', 0) if isinstance(x, dict) else 0)
    aggregated['events_non_state'] = aggregated['type_of_violence'].apply(lambda x: x.get('non_state', 0) if isinstance(x, dict) else 0)
    aggregated['events_one_sided'] = aggregated['type_of_violence'].apply(lambda x: x.get('one_sided', 0) if isinstance(x, dict) else 0)

    aggregated = aggregated.rename(columns={
        'id': 'total_events',
        'deaths_a': 'deaths_side_a',
        'deaths_b': 'deaths_side_b',
        'best_est': 'deaths_best',
        'high_est': 'deaths_high',
        'low_est': 'deaths_low'
    })

    aggregated['date'] = aggregated['period'].dt.to_timestamp()

    return aggregated[['date', 'total_events', 'events_state_based', 'events_non_state',
                       'events_one_sided', 'deaths_side_a', 'deaths_side_b',
                       'deaths_civilians', 'deaths_unknown', 'deaths_best',
                       'deaths_high', 'deaths_low']]


def load_firms_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load and aggregate FIRMS fire data to specified resolution.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with aggregated FIRMS data
    """
    firms_path = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"

    if not firms_path.exists():
        print(f"FIRMS data not found at {firms_path}")
        return pd.DataFrame()

    df = pd.read_csv(firms_path)

    # Parse dates
    df['date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Filter to conflict period
    df = df[df['date'] >= '2022-02-01']

    # Create period for aggregation based on resolution
    if resolution == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
    else:  # monthly (default)
        df['period'] = df['date'].dt.to_period('M')

    # Aggregate by period
    aggregated = df.groupby('period').agg({
        'brightness': ['mean', 'max', 'std'],
        'bright_t31': ['mean', 'max'],
        'frp': ['sum', 'mean', 'max', 'count'],
        'confidence': lambda x: (x == 'h').sum(),  # High confidence count
        'daynight': lambda x: (x == 'D').sum(),  # Day count
        'scan': 'mean',
        'track': 'mean'
    }).reset_index()

    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                          for col in aggregated.columns]

    aggregated = aggregated.rename(columns={
        'period': 'period',
        'frp_count': 'fire_count',
        'frp_sum': 'frp_total',
        'frp_mean': 'frp_mean',
        'frp_max': 'frp_max',
        'confidence_<lambda>': 'high_conf_fires',
        'daynight_<lambda>': 'day_fires'
    })

    aggregated['date'] = aggregated['period'].dt.to_timestamp()

    return aggregated


def load_sentinel_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load Sentinel satellite data from merged dataset.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with Sentinel data (re-aggregated if weekly)
    """
    merged_path = ANALYSIS_DIR / "sentinel_osint_merged.csv"

    if not merged_path.exists():
        print(f"Sentinel merged data not found at {merged_path}")
        return pd.DataFrame()

    df = pd.read_csv(merged_path)
    df['date'] = pd.to_datetime(df['date'])

    # Rename columns to match expected schema
    column_map = {
        's1_radar': 's1_count',
        's2_optical': 's2_count',
        's3_fire': 's3_frp_count',
        's5p_co': 's5p_co_mean',
        's5p_no2': 's5p_no2_mean',
        's2_avg_cloud': 's2_cloud_cover',
        's2_cloud_free': 's2_cloud_free_count'
    }
    df = df.rename(columns=column_map)

    # The merged data is already monthly, but if weekly is requested,
    # we need to either interpolate or return as-is with a warning
    if resolution == 'weekly':
        # For now, return monthly data with warning
        # Future: implement weekly aggregation from raw Sentinel data
        print("Warning: Sentinel data is monthly, weekly resolution not yet implemented")

    return df


def load_deepstate_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load and aggregate DeepState front line data to specified resolution.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with aggregated DeepState data
    """
    wayback_dir = DATA_DIR / "deepstate" / "wayback_snapshots"

    if not wayback_dir.exists():
        print(f"DeepState data not found at {wayback_dir}")
        return pd.DataFrame()

    snapshots = []

    for snapshot_file in sorted(wayback_dir.glob("*.json")):
        try:
            with open(snapshot_file) as f:
                data = json.load(f)

            # Handle both list and dict formats
            if isinstance(data, list):
                features = data
            elif isinstance(data, dict) and 'features' in data:
                features = data['features']
            elif isinstance(data, dict) and 'map' in data:
                features = data['map'].get('features', [])
            else:
                continue

            # Count features
            polygons = sum(1 for f in features if f.get('geometry', {}).get('type') == 'Polygon')
            points = sum(1 for f in features if f.get('geometry', {}).get('type') == 'Point')

            # Extract date from filename
            # Format: deepstate_YYYYMMDD_HHMMSS.json or similar
            date_str = snapshot_file.stem.replace('deepstate_', '').split('_')[0]
            try:
                date = pd.to_datetime(date_str, format='%Y%m%d')
            except:
                continue

            snapshots.append({
                'date': date,
                'polygons': polygons,
                'points': points,
                'total_features': len(features)
            })

        except Exception as e:
            continue

    if not snapshots:
        return pd.DataFrame()

    df = pd.DataFrame(snapshots)

    # Create period for aggregation based on resolution
    if resolution == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
    else:  # monthly (default)
        df['period'] = df['date'].dt.to_period('M')

    # Aggregate (take last value of period for territorial data)
    aggregated = df.groupby('period').last().reset_index()
    aggregated['date'] = aggregated['period'].dt.to_timestamp()

    return aggregated[['date', 'polygons', 'points', 'total_features']]


def load_equipment_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load equipment loss data.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with equipment loss data
    """
    equip_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_equipment.json"

    if not equip_path.exists():
        print(f"Equipment data not found at {equip_path}")
        return pd.DataFrame()

    with open(equip_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Create period for aggregation based on resolution
    if resolution == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
    else:  # monthly (default)
        df['period'] = df['date'].dt.to_period('M')

    # Get end values (cumulative) for each period
    aggregated = df.groupby('period').last().reset_index()
    aggregated['date'] = aggregated['period'].dt.to_timestamp()

    # Select relevant columns
    cols = ['date']
    for col in ['aircraft', 'helicopter', 'tank', 'APC', 'field_artillery',
                'MRL', 'drone', 'naval_ship', 'anti_aircraft_warfare',
                'special_equipment', 'vehicles_and_fuel_tanks', 'cruise_missiles']:
        if col in aggregated.columns:
            cols.append(col)

    return aggregated[cols]


def load_personnel_data(resolution: str = 'monthly') -> pd.DataFrame:
    """
    Load personnel loss data.

    Args:
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        DataFrame with personnel loss data
    """
    personnel_path = DATA_DIR / "war-losses-data" / "2022-Ukraine-Russia-War-Dataset" / "data" / "russia_losses_personnel.json"

    if not personnel_path.exists():
        print(f"Personnel data not found at {personnel_path}")
        return pd.DataFrame()

    with open(personnel_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Create period for aggregation based on resolution
    if resolution == 'weekly':
        df['period'] = df['date'].dt.to_period('W')
        period_label = 'weekly'
    else:  # monthly (default)
        df['period'] = df['date'].dt.to_period('M')
        period_label = 'monthly'

    # Get values for each period
    aggregated = df.groupby('period').agg({
        'personnel': 'last',  # Cumulative at end of period
        'POW': 'last' if 'POW' in df.columns else lambda x: 0
    }).reset_index()

    # Calculate period change
    aggregated[f'personnel_{period_label}'] = aggregated['personnel'].diff().fillna(aggregated['personnel'])

    aggregated['date'] = aggregated['period'].dt.to_timestamp()

    # Return with consistent column name for period change
    result = aggregated[['date', 'personnel']].copy()
    result['personnel_monthly'] = aggregated[f'personnel_{period_label}']  # Keep column name for compatibility

    return result


# =============================================================================
# DOMAIN FEATURE EXTRACTION
# =============================================================================

def extract_domain_features(
    ucdp: pd.DataFrame,
    firms: pd.DataFrame,
    sentinel: pd.DataFrame,
    deepstate: pd.DataFrame,
    equipment: pd.DataFrame,
    personnel: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Extract features for each domain aligned to common timeline.

    Returns dict mapping domain name to [n_timesteps, n_features] array.
    """
    # Find common date range
    all_dates = []
    for df in [ucdp, firms, sentinel, deepstate, equipment, personnel]:
        if not df.empty and 'date' in df.columns:
            all_dates.extend(df['date'].tolist())

    if not all_dates:
        print("No data available")
        return {}

    min_date = max(df['date'].min() for df in [ucdp, firms, sentinel, deepstate, equipment, personnel]
                   if not df.empty and 'date' in df.columns)
    max_date = min(df['date'].max() for df in [ucdp, firms, sentinel, deepstate, equipment, personnel]
                   if not df.empty and 'date' in df.columns)

    # Create monthly date range
    date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
    n_months = len(date_range)

    print(f"Common date range: {min_date.date()} to {max_date.date()} ({n_months} months)")

    features = {}

    # UCDP features (33)
    ucdp_features = np.zeros((n_months, 33))
    if not ucdp.empty:
        ucdp = ucdp.set_index('date').reindex(date_range).fillna(0)
        # Events decomposition (8)
        ucdp_features[:, 0] = ucdp.get('events_state_based', 0)
        ucdp_features[:, 1] = ucdp.get('events_non_state', 0)
        ucdp_features[:, 2] = ucdp.get('events_one_sided', 0)
        ucdp_features[:, 3] = ucdp.get('total_events', 0)  # clear
        ucdp_features[:, 4] = 0  # uncertain
        ucdp_features[:, 5] = ucdp.get('total_events', 0)  # exact_loc (placeholder)
        ucdp_features[:, 6] = 0  # approx_loc
        ucdp_features[:, 7] = 0  # regional_loc
        # Deaths decomposition (7)
        ucdp_features[:, 8] = ucdp.get('deaths_side_a', 0)
        ucdp_features[:, 9] = ucdp.get('deaths_side_b', 0)
        ucdp_features[:, 10] = ucdp.get('deaths_civilians', 0)
        ucdp_features[:, 11] = ucdp.get('deaths_unknown', 0)
        ucdp_features[:, 12] = ucdp.get('deaths_best', 0)
        ucdp_features[:, 13] = ucdp.get('deaths_high', 0)
        ucdp_features[:, 14] = ucdp.get('deaths_low', 0)
        # Geography placeholders (13 + 5)
        # These would need full UCDP data with location info
    features['ucdp'] = ucdp_features

    # FIRMS features (42)
    firms_features = np.zeros((n_months, 42))
    if not firms.empty:
        firms = firms.set_index('date').reindex(date_range).fillna(0)
        # Fire count decomposition (9)
        firms_features[:, 0] = firms.get('day_fires', 0)
        firms_features[:, 1] = firms.get('fire_count', 0) - firms.get('day_fires', 0)  # night
        firms_features[:, 2] = firms.get('high_conf_fires', 0)
        firms_features[:, 3] = firms.get('fire_count', 0) * 0.6  # nominal (estimate)
        firms_features[:, 4] = firms.get('fire_count', 0) * 0.1  # low conf (estimate)
        firms_features[:, 8] = firms.get('fire_count', 0)
        # FRP decomposition (14)
        firms_features[:, 9] = firms.get('frp_mean', 0) * 0.2  # tiny
        firms_features[:, 10] = firms.get('frp_mean', 0) * 0.3  # small
        firms_features[:, 11] = firms.get('frp_mean', 0) * 0.25  # medium
        firms_features[:, 12] = firms.get('frp_mean', 0) * 0.15  # large
        firms_features[:, 13] = firms.get('frp_mean', 0) * 0.08  # very large
        firms_features[:, 14] = firms.get('frp_mean', 0) * 0.02  # extreme
        firms_features[:, 15] = firms.get('frp_mean', 0)  # day mean
        firms_features[:, 16] = firms.get('frp_mean', 0) * 0.8  # night mean
        firms_features[:, 17] = firms.get('frp_max', 0)  # day max
        firms_features[:, 18] = firms.get('frp_max', 0) * 0.8  # night max
        # Brightness (6)
        firms_features[:, 23] = firms.get('brightness_mean', 0)
        firms_features[:, 24] = firms.get('brightness_max', 0)
        firms_features[:, 25] = firms.get('brightness_std', 0) if 'brightness_std' in firms.columns else 0
        firms_features[:, 26] = firms.get('bright_t31_mean', 0) if 'bright_t31_mean' in firms.columns else 0
        firms_features[:, 27] = firms.get('bright_t31_max', 0) if 'bright_t31_max' in firms.columns else 0
        # Scan/Track (4)
        firms_features[:, 29] = firms.get('scan_mean', 0) if 'scan_mean' in firms.columns else 0
        firms_features[:, 30] = firms.get('track_mean', 0) if 'track_mean' in firms.columns else 0
        # Derived (9)
        firms_features[:, 33] = firms.get('frp_total', 0)  # frp_total
        firms_features[:, 35] = firms.get('day_fires', 0) / (firms.get('fire_count', 1) - firms.get('day_fires', 0) + 1)  # day/night ratio
    features['firms'] = firms_features

    # Sentinel features (43)
    sentinel_features = np.zeros((n_months, 43))
    if not sentinel.empty:
        sentinel = sentinel.set_index('date').reindex(date_range).fillna(0)
        # Sentinel-1 (8)
        sentinel_features[:, 0] = sentinel.get('s1_count', 0)
        # Sentinel-2 (20)
        sentinel_features[:, 8] = sentinel.get('s2_count', 0)
        sentinel_features[:, 9] = sentinel.get('s2_cloud_cover', 0)
        sentinel_features[:, 10] = sentinel.get('s2_cloud_free_count', 0)
        # Sentinel-3 (7)
        sentinel_features[:, 28] = sentinel.get('s3_frp_count', 0)
        # Sentinel-5P (8)
        sentinel_features[:, 35] = sentinel.get('s5p_no2_mean', 0)
        sentinel_features[:, 38] = sentinel.get('s5p_co_mean', 0)
    features['sentinel'] = sentinel_features

    # DeepState features (45)
    deepstate_features = np.zeros((n_months, 45))
    if not deepstate.empty:
        deepstate = deepstate.set_index('date').reindex(date_range).ffill().fillna(0)
        # Polygon counts (8)
        deepstate_features[:, 0] = deepstate.get('polygons', 0) * 0.15  # occupied count
        deepstate_features[:, 1] = deepstate.get('polygons', 0) * 0.48  # liberated count
        deepstate_features[:, 2] = deepstate.get('polygons', 0) * 0.17  # contested count
        deepstate_features[:, 3] = deepstate.get('polygons', 0) * 0.20  # unknown count
        # Unit counts
        deepstate_features[:, 23] = deepstate.get('points', 0)  # units_total
    features['deepstate'] = deepstate_features

    # Equipment features (29)
    equipment_features = np.zeros((n_months, 29))
    if not equipment.empty:
        equipment = equipment.set_index('date').reindex(date_range).ffill().fillna(0)
        # Aircraft (5)
        if 'aircraft' in equipment.columns:
            equipment_features[:, 4] = equipment['aircraft']  # aircraft_total
        # Helicopters (6)
        if 'helicopter' in equipment.columns:
            equipment_features[:, 10] = equipment['helicopter']  # heli_total
        # Tanks (7)
        if 'tank' in equipment.columns:
            equipment_features[:, 17] = equipment['tank']  # tank_total
        # AFVs (6)
        if 'APC' in equipment.columns:
            equipment_features[:, 23] = equipment['APC']  # afv_total
        # Artillery (3)
        if 'field_artillery' in equipment.columns:
            equipment_features[:, 24] = equipment['field_artillery']  # arty_towed
        if 'MRL' in equipment.columns:
            equipment_features[:, 26] = equipment['MRL']  # arty_mrl
        # Other (2)
        if 'drone' in equipment.columns:
            equipment_features[:, 27] = equipment['drone']  # drones_total
        if 'anti_aircraft_warfare' in equipment.columns:
            equipment_features[:, 28] = equipment['anti_aircraft_warfare']  # air_defense_total
    features['equipment'] = equipment_features

    # Personnel features (6)
    personnel_features = np.zeros((n_months, 6))
    if not personnel.empty:
        personnel = personnel.set_index('date').reindex(date_range).ffill().fillna(0)
        personnel_features[:, 0] = personnel.get('personnel', 0)  # cumulative
        personnel_features[:, 1] = personnel.get('personnel_monthly', 0)  # monthly
        # Calculate daily average
        personnel_features[:, 2] = personnel.get('personnel_monthly', 0) / 30  # daily_avg
        # Rate change
        personnel_features[:, 3] = np.gradient(personnel.get('personnel_monthly', np.zeros(n_months)))  # rate_change
    features['personnel'] = personnel_features

    return features, date_range


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_features(
    features: Dict[str, np.ndarray],
    domain_configs: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Apply domain-specific normalization.

    Returns:
        normalized_features: Dict of normalized arrays
        stats: Dict of normalization statistics for inverse transform
    """
    normalized = {}
    stats = {}

    for domain_name, data in features.items():
        config = domain_configs.get(domain_name)
        if config is None:
            normalized[domain_name] = data
            continue

        norm_type = config.normalization if hasattr(config, 'normalization') else 'standard'

        if norm_type == 'log':
            # Log transform for count/cumulative data
            # Clip negative values to 0 before log (handles edge cases)
            data_clipped = np.clip(data, 0, None)
            data_log = np.log1p(data_clipped)  # log(1 + x) for stability
            mean = data_log.mean(axis=0)
            std = data_log.std(axis=0) + 1e-8
            normalized[domain_name] = (data_log - mean) / std
            stats[domain_name] = {'type': 'log', 'mean': mean, 'std': std}

        elif norm_type == 'minmax':
            min_val = data.min(axis=0)
            max_val = data.max(axis=0) + 1e-8
            normalized[domain_name] = (data - min_val) / (max_val - min_val)
            stats[domain_name] = {'type': 'minmax', 'min': min_val, 'max': max_val}

        else:  # standard
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            normalized[domain_name] = (data - mean) / std
            stats[domain_name] = {'type': 'standard', 'mean': mean, 'std': std}

    return normalized, stats


def compute_normalization_stats(
    features: Dict[str, np.ndarray],
    domain_configs: Dict[str, Any],
    train_indices: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute normalization statistics from TRAINING DATA ONLY.

    This prevents data leakage by ensuring validation/test sets
    are normalized using only statistics computed from training data.

    Args:
        features: Dict mapping domain name to [n_timesteps, n_features] array
        domain_configs: Domain configuration with normalization type
        train_indices: Indices of training samples to compute stats from

    Returns:
        stats: Dict of normalization statistics per domain
    """
    stats = {}

    for domain_name, data in features.items():
        config = domain_configs.get(domain_name)
        if config is None:
            stats[domain_name] = {'type': 'none'}
            continue

        norm_type = config.normalization if hasattr(config, 'normalization') else 'standard'

        # Extract ONLY training data for computing statistics
        train_data = data[train_indices]

        if norm_type == 'log':
            # Log transform for count/cumulative data
            train_data_clipped = np.clip(train_data, 0, None)
            train_data_log = np.log1p(train_data_clipped)
            mean = train_data_log.mean(axis=0)
            # CRITICAL FIX: Use minimum std of 0.1 to prevent explosion with constant features
            # 1e-8 is too small - if val data differs even slightly from mean,
            # dividing by 1e-8 causes values in millions/trillions
            std = np.maximum(train_data_log.std(axis=0), 0.1)
            stats[domain_name] = {'type': 'log', 'mean': mean, 'std': std}

        elif norm_type == 'minmax':
            min_val = train_data.min(axis=0)
            # CRITICAL FIX: Ensure meaningful range for constant features
            range_val = train_data.max(axis=0) - min_val
            range_val = np.maximum(range_val, 0.1)  # Minimum range of 0.1
            max_val = min_val + range_val
            stats[domain_name] = {'type': 'minmax', 'min': min_val, 'max': max_val}

        else:  # standard
            mean = train_data.mean(axis=0)
            # CRITICAL FIX: Use minimum std of 0.1 to prevent explosion with constant features
            # Features like deepstate with range [0,0] would otherwise cause division by ~0
            std = np.maximum(train_data.std(axis=0), 0.1)
            stats[domain_name] = {'type': 'standard', 'mean': mean, 'std': std}

    return stats


def apply_normalization(
    features: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    Apply pre-computed normalization statistics to any data split.

    This allows validation/test sets to be normalized using
    statistics computed from training data only.

    Args:
        features: Dict mapping domain name to [n_timesteps, n_features] array
        stats: Pre-computed normalization statistics from compute_normalization_stats

    Returns:
        normalized: Dict of normalized arrays
    """
    normalized = {}

    for domain_name, data in features.items():
        domain_stats = stats.get(domain_name, {'type': 'none'})
        norm_type = domain_stats.get('type', 'none')

        if norm_type == 'log':
            data_clipped = np.clip(data, 0, None)
            data_log = np.log1p(data_clipped)
            mean = domain_stats['mean']
            std = domain_stats['std']
            result = (data_log - mean) / std
            # Clip to reasonable range to prevent explosion from OOD values
            normalized[domain_name] = np.clip(result, -10.0, 10.0)

        elif norm_type == 'minmax':
            min_val = domain_stats['min']
            max_val = domain_stats['max']
            result = (data - min_val) / (max_val - min_val)
            # Clip to [0, 1] plus small margin for OOD values
            normalized[domain_name] = np.clip(result, -0.1, 1.1)

        elif norm_type == 'standard':
            mean = domain_stats['mean']
            std = domain_stats['std']
            result = (data - mean) / std
            # Clip to reasonable range to prevent explosion from OOD values
            normalized[domain_name] = np.clip(result, -10.0, 10.0)

        else:  # none
            normalized[domain_name] = data

    return normalized


# =============================================================================
# PYTORCH DATASET
# =============================================================================

if HAS_TORCH:

    class RealConflictDataset(Dataset):
        """
        PyTorch Dataset for real conflict data.

        Loads all sources, aligns temporally, normalizes, and provides
        windowed sequences for training.

        Key features:
        - Temporal splits with configurable gaps to prevent leakage
        - Normalization stats computed from training data only
        - Support for different temporal resolutions (monthly, weekly)
        """
        def __init__(
            self,
            domain_configs: Dict[str, Any],
            seq_len: int = 12,
            prediction_horizon: int = 1,
            split: str = 'train',
            val_ratio: float = 0.2,
            test_ratio: float = 0.1,
            temporal_gap_days: int = 14,
            resolution: str = 'monthly',
            norm_stats: Optional[Dict] = None
        ):
            """
            Initialize the dataset.

            Args:
                domain_configs: Domain configuration dictionary
                seq_len: Sequence length for input windows
                prediction_horizon: Number of steps to predict ahead
                split: One of 'train', 'val', 'test'
                val_ratio: Fraction of data for validation
                test_ratio: Fraction of data for testing
                temporal_gap_days: Days of gap between splits to prevent leakage
                resolution: Temporal resolution ('monthly' or 'weekly')
                norm_stats: Pre-computed normalization stats (required for val/test)
            """
            if split not in ('train', 'val', 'test'):
                raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

            self.domain_configs = domain_configs
            self.seq_len = seq_len
            self.prediction_horizon = prediction_horizon
            self.split = split
            self.temporal_gap_days = temporal_gap_days
            self.resolution = resolution

            # Load all data sources
            print(f"Loading data sources (resolution={resolution})...")
            ucdp = load_ucdp_data(resolution=resolution)
            firms = load_firms_data(resolution=resolution)
            sentinel = load_sentinel_data(resolution=resolution)
            deepstate = load_deepstate_data(resolution=resolution)
            equipment = load_equipment_data(resolution=resolution)
            personnel = load_personnel_data(resolution=resolution)

            # Extract and align features
            print("Extracting features...")
            self.features, self.date_range = extract_domain_features(
                ucdp, firms, sentinel, deepstate, equipment, personnel
            )

            if not self.features:
                print("Warning: No features extracted, using synthetic data")
                self._create_synthetic_data()
                return

            # Compute temporal split indices BEFORE normalization
            n_samples = len(self.date_range)
            train_indices, val_indices, test_indices = self._compute_temporal_split(
                n_samples, val_ratio, test_ratio
            )

            # Handle normalization based on split
            if split == 'train':
                # Compute normalization stats from training data only
                print("Computing normalization stats from training data...")
                self.norm_stats = compute_normalization_stats(
                    self.features, domain_configs, train_indices
                )
            else:
                # Use provided stats for val/test
                if norm_stats is None:
                    raise ValueError(
                        f"norm_stats must be provided for {split} split to prevent data leakage"
                    )
                self.norm_stats = norm_stats

            # Apply normalization using the stats
            print(f"Applying normalization to {split} split...")
            self.features = apply_normalization(self.features, self.norm_stats)

            # Convert to tensors
            for domain_name in self.features:
                self.features[domain_name] = torch.tensor(
                    self.features[domain_name], dtype=torch.float32
                )

            # Set the indices for this split
            if split == 'train':
                self.indices = train_indices
            elif split == 'val':
                self.indices = val_indices
            else:  # test
                self.indices = test_indices

            # For backward compatibility, also set start_idx and end_idx
            if len(self.indices) > 0:
                self.start_idx = self.indices[0]
                self.end_idx = self.indices[-1] + 1
            else:
                self.start_idx = 0
                self.end_idx = 0

            print(f"Dataset ({split}): {len(self.indices)} samples "
                  f"(indices {self.start_idx}:{self.end_idx} of {n_samples})")

        def _compute_temporal_split(
            self,
            n_samples: int,
            val_ratio: float,
            test_ratio: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Compute temporal split indices with gaps between splits.

            The gaps prevent information leakage between adjacent time periods.
            For monthly resolution, a 14-day gap effectively means we skip
            a transition period.

            Returns:
                train_indices, val_indices, test_indices as numpy arrays
            """
            # Calculate gap in number of timesteps based on resolution
            if self.resolution == 'weekly':
                gap_timesteps = max(1, self.temporal_gap_days // 7)
            else:  # monthly
                # For monthly, gap_days=14 means roughly half a month gap
                # We use at least 1 timestep gap
                gap_timesteps = max(1, self.temporal_gap_days // 30)

            min_usable = self.seq_len + self.prediction_horizon + 1

            # Calculate split sizes accounting for gaps
            total_gap = 2 * gap_timesteps  # gap before val, gap before test
            usable_samples = n_samples - total_gap

            if usable_samples < min_usable * 3:
                # Dataset too small for proper split with gaps
                print(f"Warning: Small dataset ({n_samples} samples), reducing gaps")
                gap_timesteps = 0
                usable_samples = n_samples

            if usable_samples < min_usable * 2:
                # Still too small - use overlapping splits
                print(f"Warning: Very small dataset ({n_samples} samples), using overlapping splits")
                all_indices = np.arange(n_samples)
                return all_indices, all_indices, all_indices

            # Calculate split sizes
            n_test = max(min_usable, int(usable_samples * test_ratio))
            n_val = max(min_usable, int(usable_samples * val_ratio))
            n_train = usable_samples - n_test - n_val

            # Ensure train has enough samples
            if n_train < min_usable:
                n_train = min_usable
                remaining = usable_samples - n_train
                n_val = remaining // 2
                n_test = remaining - n_val

            # Create indices with gaps
            # Layout: [train_data][gap][val_data][gap][test_data]
            train_end = n_train
            val_start = train_end + gap_timesteps
            val_end = val_start + n_val
            test_start = val_end + gap_timesteps
            test_end = min(test_start + n_test, n_samples)

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            test_indices = np.arange(test_start, test_end)

            print(f"Temporal split with {gap_timesteps} timestep gap:")
            print(f"  Train: indices 0-{train_end-1} ({n_train} samples)")
            print(f"  Val: indices {val_start}-{val_end-1} ({n_val} samples)")
            print(f"  Test: indices {test_start}-{test_end-1} ({test_end - test_start} samples)")

            return train_indices, val_indices, test_indices

        def _create_synthetic_data(self):
            """Create synthetic data for testing when real data unavailable."""
            n_samples = 32
            self.date_range = pd.date_range('2022-05-01', periods=n_samples, freq='MS')

            # Feature counts from domain configs
            feature_counts = {
                'ucdp': 33, 'firms': 42, 'sentinel': 43,
                'deepstate': 45, 'equipment': 29, 'personnel': 6
            }

            self.features = {}
            for domain_name, n_feat in feature_counts.items():
                self.features[domain_name] = torch.randn(n_samples, n_feat)

            self.norm_stats = {}

            # Create simple split for synthetic data
            if self.split == 'train':
                self.indices = np.arange(0, 20)
            elif self.split == 'val':
                self.indices = np.arange(20, 26)
            else:  # test
                self.indices = np.arange(26, 32)

            self.start_idx = self.indices[0] if len(self.indices) > 0 else 0
            self.end_idx = self.indices[-1] + 1 if len(self.indices) > 0 else 0

        def __len__(self):
            """Return number of valid samples in this split."""
            # We need seq_len + prediction_horizon consecutive indices
            # Count how many starting positions we have
            valid_samples = 0
            for i, idx in enumerate(self.indices):
                # Check if we have enough future indices for sequence and prediction
                end_needed = idx + self.seq_len + self.prediction_horizon
                if end_needed <= len(self.date_range):
                    valid_samples += 1
                else:
                    break  # Indices are sorted, so no more valid samples after this

            return max(0, valid_samples)

        def __getitem__(self, idx):
            """Get a sample at the given index."""
            if idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

            # Get the actual data index from our split indices
            actual_idx = self.indices[idx]

            # Get sequence windows
            features = {}
            masks = {}
            for domain_name, data in self.features.items():
                features[domain_name] = data[actual_idx:actual_idx + self.seq_len]
                # Create mask (1 for valid, 0 for missing)
                masks[domain_name] = (features[domain_name] != 0).float()

            # Targets for next timestep
            target_idx = actual_idx + self.seq_len
            targets = {
                'next_features': {
                    domain_name: data[target_idx:target_idx + self.prediction_horizon]
                    for domain_name, data in self.features.items()
                }
            }

            # Add regime label (simplified: based on personnel loss rate)
            if 'personnel' in self.features:
                monthly_loss = self.features['personnel'][target_idx, 1]  # monthly column
                if monthly_loss < -0.5:  # Below average
                    regime = 0  # low intensity
                elif monthly_loss < 0.5:
                    regime = 1  # medium
                elif monthly_loss < 1.5:
                    regime = 2  # high
                else:
                    regime = 3  # major offensive
            else:
                regime = 1

            targets['regime'] = torch.tensor(regime)

            return features, masks, targets


def create_data_loaders(
    domain_configs: Dict[str, Any],
    batch_size: int = 4,
    seq_len: int = 12,
    num_workers: int = 0,
    temporal_gap_days: int = 14,
    resolution: str = 'monthly'
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders.

    This function ensures proper data handling:
    1. Training dataset is created first (computes normalization stats)
    2. Normalization stats from training are passed to val/test datasets
    3. Temporal gaps are maintained between splits to prevent leakage

    Args:
        domain_configs: Domain configuration dictionary
        batch_size: Batch size for data loaders
        seq_len: Sequence length for input windows
        num_workers: Number of worker processes for data loading
        temporal_gap_days: Days of gap between splits to prevent leakage
        resolution: Temporal resolution ('monthly' or 'weekly')

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        norm_stats: Normalization statistics computed from training data
    """
    # Create training dataset FIRST - this computes normalization stats
    print("Creating training dataset...")
    train_dataset = RealConflictDataset(
        domain_configs,
        seq_len=seq_len,
        split='train',
        temporal_gap_days=temporal_gap_days,
        resolution=resolution
    )

    # Get normalization stats from training dataset
    norm_stats = train_dataset.norm_stats

    # Create validation dataset with training normalization stats
    print("Creating validation dataset...")
    val_dataset = RealConflictDataset(
        domain_configs,
        seq_len=seq_len,
        split='val',
        temporal_gap_days=temporal_gap_days,
        resolution=resolution,
        norm_stats=norm_stats  # Pass training stats to prevent leakage
    )

    # Create test dataset with training normalization stats
    print("Creating test dataset...")
    test_dataset = RealConflictDataset(
        domain_configs,
        seq_len=seq_len,
        split='test',
        temporal_gap_days=temporal_gap_days,
        resolution=resolution,
        norm_stats=norm_stats  # Pass training stats to prevent leakage
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, norm_stats


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CONFLICT DATA LOADER TEST")
    print("=" * 80)

    # Import domain configs
    from hierarchical_attention_network import DOMAIN_CONFIGS

    # Create training dataset first
    print("\nCreating training dataset...")
    train_dataset = RealConflictDataset(
        DOMAIN_CONFIGS,
        seq_len=6,
        split='train',
        temporal_gap_days=14,
        resolution='monthly'
    )

    print(f"\nTraining dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        # Get a sample
        features, masks, targets = train_dataset[0]

        print("\nSample shapes:")
        for domain_name, data in features.items():
            print(f"  {domain_name}: {data.shape}")

        print("\nMask shapes:")
        for domain_name, mask in masks.items():
            non_zero = mask.sum().item()
            total = mask.numel()
            print(f"  {domain_name}: {mask.shape} ({non_zero}/{total} non-zero)")

    # Create validation dataset using training stats
    print("\nCreating validation dataset with training normalization stats...")
    val_dataset = RealConflictDataset(
        DOMAIN_CONFIGS,
        seq_len=6,
        split='val',
        temporal_gap_days=14,
        resolution='monthly',
        norm_stats=train_dataset.norm_stats
    )
    print(f"Validation dataset length: {len(val_dataset)}")

    # Create test dataset using training stats
    print("\nCreating test dataset with training normalization stats...")
    test_dataset = RealConflictDataset(
        DOMAIN_CONFIGS,
        seq_len=6,
        split='test',
        temporal_gap_days=14,
        resolution='monthly',
        norm_stats=train_dataset.norm_stats
    )
    print(f"Test dataset length: {len(test_dataset)}")

    # Create loaders using the convenience function
    print("\n" + "=" * 80)
    print("Testing create_data_loaders function...")
    print("=" * 80)
    train_loader, val_loader, test_loader, norm_stats = create_data_loaders(
        DOMAIN_CONFIGS,
        batch_size=2,
        seq_len=6,
        temporal_gap_days=14,
        resolution='monthly'
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Normalization stats domains: {list(norm_stats.keys())}")

    # Test one batch
    for features, masks, targets in train_loader:
        print("\nBatch shapes:")
        for domain_name, data in features.items():
            print(f"  {domain_name}: {data.shape}")
        break

    # Verify normalization stats are being computed correctly
    print("\n" + "=" * 80)
    print("Normalization Stats Summary:")
    print("=" * 80)
    for domain, stats in norm_stats.items():
        norm_type = stats.get('type', 'unknown')
        print(f"\n{domain} ({norm_type}):")
        if norm_type == 'standard':
            print(f"  mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
            print(f"  std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
        elif norm_type == 'log':
            print(f"  log mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
            print(f"  log std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
        elif norm_type == 'minmax':
            print(f"  min range: [{stats['min'].min():.4f}, {stats['min'].max():.4f}]")
            print(f"  max range: [{stats['max'].min():.4f}, {stats['max'].max():.4f}]")
