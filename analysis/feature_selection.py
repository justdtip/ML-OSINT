"""
Feature selection utilities for identifying and removing redundant features.

This module provides tools for analyzing feature redundancy through variance
and correlation analysis, along with predefined reduced feature sets for
each data domain.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


def identify_redundant_features(
    features: Dict[str, np.ndarray],
    correlation_threshold: float = 0.95,
    variance_threshold: float = 1e-6
) -> Dict[str, List[int]]:
    """
    Identify redundant features based on variance and correlation analysis.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping of domain name to feature array with shape [n_samples, n_features].
    correlation_threshold : float, optional
        Threshold above which features are considered highly correlated.
        Default is 0.95.
    variance_threshold : float, optional
        Threshold below which features are considered near-zero variance.
        Default is 1e-6.

    Returns
    -------
    Dict[str, List[int]]
        Mapping of domain name to list of redundant feature indices.

    Notes
    -----
    For highly correlated pairs, the feature with lower variance is marked
    as redundant. Near-zero variance features are always marked redundant.
    """
    redundant_indices: Dict[str, List[int]] = {}

    for domain, feature_array in features.items():
        if feature_array.ndim != 2:
            raise ValueError(
                f"Expected 2D array for domain '{domain}', "
                f"got shape {feature_array.shape}"
            )

        n_samples, n_features = feature_array.shape
        redundant: set = set()

        # Step 1: Check variance - drop near-zero variance features
        variances = np.var(feature_array, axis=0)
        low_variance_indices = np.where(variances < variance_threshold)[0]
        redundant.update(low_variance_indices.tolist())

        # Step 2: Check correlation matrix for remaining features
        # Only compute correlation for features with sufficient variance
        valid_feature_mask = variances >= variance_threshold
        valid_indices = np.where(valid_feature_mask)[0]

        if len(valid_indices) > 1:
            # Extract valid features and compute correlation matrix
            valid_features = feature_array[:, valid_indices]

            # Handle constant features to avoid NaN in correlation
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.corrcoef(valid_features, rowvar=False)

            # Replace NaN with 0 (occurs when std is 0)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Find highly correlated pairs
            n_valid = len(valid_indices)
            for i in range(n_valid):
                if valid_indices[i] in redundant:
                    continue
                for j in range(i + 1, n_valid):
                    if valid_indices[j] in redundant:
                        continue

                    if abs(corr_matrix[i, j]) >= correlation_threshold:
                        # Mark the one with lower variance as redundant
                        var_i = variances[valid_indices[i]]
                        var_j = variances[valid_indices[j]]

                        if var_i >= var_j:
                            redundant.add(valid_indices[j])
                        else:
                            redundant.add(valid_indices[i])

        redundant_indices[domain] = sorted(list(redundant))

    return redundant_indices


# Predefined reduced feature sets with explicit keep/drop lists
REDUCED_FEATURE_SETS: Dict[str, Dict] = {
    'ucdp': {
        'keep': [
            'events_state_based',
            'events_non_state',
            'events_one_sided',
            'deaths_best',
            'deaths_civilians',
            'front_eastern',
            'front_southern',
            'front_northeastern',
            'front_northern',
            'front_rear',
            'total_events'
        ],
        'drop_reason': {
            'deaths_high': 'correlates 0.99 with deaths_best',
            'deaths_low': 'correlates 0.99 with deaths_best',
            'deaths_unknown': 'correlates 0.97 with deaths_best',
            'events_total_alt': 'correlates 0.99 with total_events',
            'front_western': 'near-zero variance (limited activity)',
            'front_crimean': 'near-zero variance (limited activity)',
            'deaths_military': 'correlates 0.98 with deaths_best',
            'intensity_level': 'derived from deaths_best',
            'cumulative_deaths': 'cumulative features redundant for prediction',
            'cumulative_events': 'cumulative features redundant for prediction'
        }
    },
    'firms': {
        'keep': [
            'fires_day',
            'fires_night',
            'fires_total',
            'frp_small',
            'frp_medium',
            'frp_large',
            'frp_extreme',
            'frp_day_mean',
            'frp_day_max',
            'brightness_mean',
            'frp_per_fire',
            'high_intensity_pct',
            'spatial_spread'
        ],
        'drop_reason': {
            'frp_total': 'correlates 0.99 with fires_total',
            'frp_night_mean': 'correlates 0.96 with frp_day_mean',
            'frp_night_max': 'correlates 0.95 with frp_day_max',
            'brightness_max': 'correlates 0.97 with brightness_mean',
            'brightness_min': 'correlates 0.94 with brightness_mean',
            'fires_dawn': 'correlates 0.92 with fires_day, lower variance',
            'fires_dusk': 'correlates 0.91 with fires_night, lower variance',
            'frp_sum': 'correlates 0.99 with frp_total',
            'confidence_mean': 'near-zero variance (consistently high)',
            'scan_pixel_area': 'near-zero variance (sensor constant)'
        }
    },
    'equipment': {
        'keep': [
            'aircraft_total',
            'heli_total',
            'tank_total',
            'afv_total',
            'arty_towed',
            'arty_sp',
            'arty_mrl',
            'drones_total',
            'air_defense_total'
        ],
        'drop_reason': {
            'aircraft_destroyed': 'correlates 0.99 with aircraft_total',
            'aircraft_captured': 'near-zero variance',
            'heli_destroyed': 'correlates 0.99 with heli_total',
            'heli_captured': 'near-zero variance',
            'tank_destroyed': 'correlates 0.98 with tank_total',
            'tank_captured': 'correlates 0.95 with tank_total',
            'tank_abandoned': 'correlates 0.93 with tank_total',
            'afv_destroyed': 'correlates 0.97 with afv_total',
            'afv_captured': 'correlates 0.94 with afv_total',
            'arty_total': 'correlates 0.96 with arty_sp (most common type)',
            'drones_destroyed': 'correlates 0.99 with drones_total',
            'air_defense_destroyed': 'correlates 0.99 with air_defense_total',
            'vehicles_total': 'correlates 0.95 with afv_total',
            'special_equipment': 'near-zero variance (rare events)'
        }
    },
    'personnel': {
        'keep': [
            'personnel_cumulative',
            'personnel_monthly',
            'personnel_rate_change'
        ],
        'drop_reason': {
            'personnel_daily': 'correlates 0.97 with personnel_monthly (smoothed)',
            'personnel_weekly': 'correlates 0.98 with personnel_monthly',
            'personnel_cumulative_alt': 'correlates 1.0 with personnel_cumulative',
            'personnel_reported': 'correlates 0.99 with personnel_cumulative',
            'personnel_estimated': 'correlates 0.96 with personnel_cumulative',
            'personnel_pct_change': 'correlates 0.94 with personnel_rate_change'
        }
    }
}


def get_reduced_feature_names(
    domain_configs: Optional[Dict[str, Dict]] = None
) -> Dict[str, List[str]]:
    """
    Get the list of kept feature names for each domain.

    Parameters
    ----------
    domain_configs : Dict[str, Dict], optional
        Custom domain configuration with 'keep' lists. If None, uses
        the predefined REDUCED_FEATURE_SETS.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of domain name to list of kept feature names.

    Examples
    --------
    >>> names = get_reduced_feature_names()
    >>> print(names['ucdp'])
    ['events_state_based', 'events_non_state', ...]
    """
    configs = domain_configs if domain_configs is not None else REDUCED_FEATURE_SETS

    return {
        domain: config['keep'].copy()
        for domain, config in configs.items()
    }


def filter_features(
    features: Dict[str, np.ndarray],
    domain_configs: Optional[Dict[str, Dict]] = None,
    use_reduced: bool = True,
    feature_names: Optional[Dict[str, List[str]]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Filter features to keep only the selected subset for each domain.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping of domain name to feature array with shape [n_samples, n_features].
    domain_configs : Dict[str, Dict], optional
        Custom domain configuration with 'keep' lists. If None, uses
        the predefined REDUCED_FEATURE_SETS.
    use_reduced : bool, optional
        If True, apply feature reduction. If False, return original features.
        Default is True.
    feature_names : Dict[str, List[str]], optional
        Mapping of domain name to list of all feature names in order.
        Required when use_reduced is True to map names to indices.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]
        Tuple of (filtered_features, kept_names) where:
        - filtered_features: Dict mapping domain to filtered feature arrays
        - kept_names: Dict mapping domain to list of kept feature names

    Raises
    ------
    ValueError
        If feature_names is not provided when use_reduced is True and
        domain_configs contains string feature names.

    Examples
    --------
    >>> features = {'ucdp': np.random.randn(100, 20)}
    >>> feature_names = {'ucdp': ['feature_0', ..., 'feature_19']}
    >>> filtered, kept = filter_features(features, feature_names=feature_names)
    """
    if not use_reduced:
        # Return original features with all names
        kept_names = {}
        for domain in features.keys():
            if feature_names and domain in feature_names:
                kept_names[domain] = feature_names[domain].copy()
            else:
                n_features = features[domain].shape[1]
                kept_names[domain] = [f'feature_{i}' for i in range(n_features)]
        return features.copy(), kept_names

    configs = domain_configs if domain_configs is not None else REDUCED_FEATURE_SETS

    filtered_features: Dict[str, np.ndarray] = {}
    kept_names: Dict[str, List[str]] = {}

    for domain, feature_array in features.items():
        if domain not in configs:
            # Domain not in config, keep all features
            filtered_features[domain] = feature_array.copy()
            if feature_names and domain in feature_names:
                kept_names[domain] = feature_names[domain].copy()
            else:
                n_features = feature_array.shape[1]
                kept_names[domain] = [f'feature_{i}' for i in range(n_features)]
            continue

        keep_list = configs[domain]['keep']

        # Check if keep_list contains strings (feature names) or indices
        if keep_list and isinstance(keep_list[0], str):
            # Need to map feature names to indices
            if feature_names is None or domain not in feature_names:
                raise ValueError(
                    f"feature_names must be provided for domain '{domain}' "
                    f"when keep list contains string names"
                )

            domain_feature_names = feature_names[domain]
            name_to_idx = {name: idx for idx, name in enumerate(domain_feature_names)}

            # Find indices for kept features that exist in the data
            keep_indices = []
            actual_kept_names = []
            for name in keep_list:
                if name in name_to_idx:
                    keep_indices.append(name_to_idx[name])
                    actual_kept_names.append(name)

            if not keep_indices:
                raise ValueError(
                    f"No features from keep list found in domain '{domain}'"
                )

            filtered_features[domain] = feature_array[:, keep_indices]
            kept_names[domain] = actual_kept_names
        else:
            # keep_list contains indices directly
            keep_indices = keep_list
            filtered_features[domain] = feature_array[:, keep_indices]

            if feature_names and domain in feature_names:
                kept_names[domain] = [
                    feature_names[domain][i] for i in keep_indices
                ]
            else:
                kept_names[domain] = [f'feature_{i}' for i in keep_indices]

    return filtered_features, kept_names


def get_feature_statistics(
    features: Dict[str, np.ndarray],
    feature_names: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict]:
    """
    Compute variance and correlation statistics for features.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping of domain name to feature array with shape [n_samples, n_features].
    feature_names : Dict[str, List[str]], optional
        Mapping of domain name to list of feature names.

    Returns
    -------
    Dict[str, Dict]
        Statistics for each domain including:
        - variances: array of feature variances
        - correlation_matrix: pairwise correlation matrix
        - high_correlation_pairs: list of highly correlated feature pairs
        - feature_names: list of feature names
    """
    stats: Dict[str, Dict] = {}

    for domain, feature_array in features.items():
        n_features = feature_array.shape[1]

        # Get or generate feature names
        if feature_names and domain in feature_names:
            names = feature_names[domain]
        else:
            names = [f'feature_{i}' for i in range(n_features)]

        # Compute variances
        variances = np.var(feature_array, axis=0)

        # Compute correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(feature_array, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Find high correlation pairs (above 0.9)
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                if corr >= 0.9:
                    high_corr_pairs.append({
                        'feature_1': names[i],
                        'feature_2': names[j],
                        'correlation': float(corr_matrix[i, j]),
                        'variance_1': float(variances[i]),
                        'variance_2': float(variances[j])
                    })

        # Sort by correlation strength
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

        stats[domain] = {
            'variances': variances,
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'feature_names': names,
            'n_features': n_features,
            'n_low_variance': int(np.sum(variances < 1e-6)),
            'n_high_correlation_pairs': len(high_corr_pairs)
        }

    return stats


def suggest_feature_reduction(
    features: Dict[str, np.ndarray],
    feature_names: Dict[str, List[str]],
    correlation_threshold: float = 0.95,
    variance_threshold: float = 1e-6
) -> Dict[str, Dict]:
    """
    Analyze features and suggest which ones to keep or drop.

    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Mapping of domain name to feature array.
    feature_names : Dict[str, List[str]]
        Mapping of domain name to list of feature names.
    correlation_threshold : float, optional
        Threshold for marking features as highly correlated.
    variance_threshold : float, optional
        Threshold for marking features as low variance.

    Returns
    -------
    Dict[str, Dict]
        Suggested configuration for each domain with 'keep' and 'drop_reason'.
    """
    suggestions: Dict[str, Dict] = {}

    for domain, feature_array in features.items():
        names = feature_names.get(domain, [
            f'feature_{i}' for i in range(feature_array.shape[1])
        ])

        variances = np.var(feature_array, axis=0)

        # Compute correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(feature_array, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        keep = []
        drop_reason = {}
        dropped = set()

        n_features = len(names)

        # First pass: identify low variance features
        for i in range(n_features):
            if variances[i] < variance_threshold:
                dropped.add(i)
                drop_reason[names[i]] = f'near-zero variance ({variances[i]:.2e})'

        # Second pass: identify correlated features
        for i in range(n_features):
            if i in dropped:
                continue

            for j in range(i + 1, n_features):
                if j in dropped:
                    continue

                corr = abs(corr_matrix[i, j])
                if corr >= correlation_threshold:
                    # Drop the one with lower variance
                    if variances[i] >= variances[j]:
                        dropped.add(j)
                        drop_reason[names[j]] = (
                            f'correlates {corr:.2f} with {names[i]}'
                        )
                    else:
                        dropped.add(i)
                        drop_reason[names[i]] = (
                            f'correlates {corr:.2f} with {names[j]}'
                        )
                        break  # i is dropped, move to next i

        # Collect kept features
        for i in range(n_features):
            if i not in dropped:
                keep.append(names[i])

        suggestions[domain] = {
            'keep': keep,
            'drop_reason': drop_reason,
            'original_count': n_features,
            'kept_count': len(keep),
            'dropped_count': len(drop_reason)
        }

    return suggestions
