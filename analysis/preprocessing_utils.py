"""
Preprocessing utilities for temporal deconfounding.

This module provides detrending functions to remove slow-moving trends
from time series features while preserving daily fluctuations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class DetrendingConfig:
    """Configuration for feature detrending."""
    enabled: bool = True
    method: str = 'rolling_mean'  # 'rolling_mean' or 'diff'
    default_window: int = 14
    min_periods: int = 1

    # Per-source window sizes (days)
    source_windows: Dict[str, int] = field(default_factory=lambda: {
        'personnel': 14,
        'drones': 14,
        'armor': 14,
        'artillery': 14,
        'equipment': 14,
        'deepstate_raion': 7,
        'firms_expanded_raion': 7,
        'geoconfirmed_raion': 7,
        'ucdp_raion': 7,
        'air_raid_sirens_raion': 7,
        'warspotting_raion': 7,
    })

    # Column patterns to EXCLUDE from detrending (already processed)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        'rolling', 'volatility', 'momentum', 'avg', 'std'
    ])


def detrend_features(
    features: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    config: DetrendingConfig,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Apply detrending to all feature DataFrames.

    Subtracts rolling mean to remove slow trends while preserving
    daily fluctuations needed for tactical prediction.

    Args:
        features: Dict mapping source name to (DataFrame, observation_mask)
        config: DetrendingConfig with window sizes and options

    Returns:
        Dict with detrended features (masks unchanged for rolling_mean method)
    """
    if not config.enabled:
        return features

    detrended = {}

    for source_name, (df, mask) in features.items():
        if df.empty:
            detrended[source_name] = (df, mask)
            continue

        feature_cols = [c for c in df.columns if c != 'date']

        # Identify columns to exclude from detrending
        exclude_cols = set()
        for pattern in config.exclude_patterns:
            exclude_cols.update([c for c in feature_cols if pattern.lower() in c.lower()])

        detrend_cols = [c for c in feature_cols if c not in exclude_cols]

        if not detrend_cols:
            detrended[source_name] = (df, mask)
            continue

        # Get window size for this source
        window = config.source_windows.get(source_name, config.default_window)

        # Apply detrending
        df_copy = df.copy()

        if config.method == 'rolling_mean':
            # Subtract rolling mean (keeps daily fluctuations)
            rolling_mean = df_copy[detrend_cols].rolling(
                window=window,
                min_periods=config.min_periods,
                center=False
            ).mean()
            df_copy[detrend_cols] = df_copy[detrend_cols] - rolling_mean
            # Fill NaN from early rows with 0 (no trend to remove yet)
            df_copy[detrend_cols] = df_copy[detrend_cols].fillna(0)
            new_mask = mask  # Mask unchanged

        elif config.method == 'diff':
            # First-order differencing
            df_copy[detrend_cols] = df_copy[detrend_cols].diff()
            df_copy[detrend_cols] = df_copy[detrend_cols].fillna(0)
            # Update mask: need both current and previous observation
            prev_mask = np.roll(mask, 1)
            prev_mask[0] = False
            new_mask = mask & prev_mask
        else:
            raise ValueError(f"Unknown detrending method: {config.method}")

        detrended[source_name] = (df_copy, new_mask)

    return detrended
