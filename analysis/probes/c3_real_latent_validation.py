#!/usr/bin/env python3
"""
C3: ISW-Latent Alignment Validation with REAL Model Latents

This script resolves the INCONCLUSIVE verdict from the previous C3 probe by:
1. Extracting ACTUAL latent representations via model inference
2. Re-running the borderline experiments with real latents

The key limitation of the previous probe was using proxy latents (month queries)
instead of actual model inference outputs. This script fixes that.

Author: Agent C3
Date: 2026-01-25
"""

import os
import sys
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, OUTPUT_DIR, ensure_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = PROJECT_ROOT / "analysis" / "training_runs" / "run_24-01-2026_20-22" / "stage3_han" / "best_checkpoint.pt"
ISW_EMBEDDINGS_PATH = DATA_DIR / "wayback_archives" / "isw_assessments" / "embeddings" / "isw_embeddings.npz"
OUTPUT_DIR_PATH = OUTPUT_DIR / "analysis" / "han_validation"

# Key events for analysis
MAJOR_EVENTS = {
    "kerch_bridge": {"date": "2022-10-08", "description": "Kerch Bridge attack"},
    "kherson_withdrawal": {"date": "2022-11-11", "description": "Russian withdrawal from Kherson"},
    "prigozhin_mutiny": {"date": "2023-06-23", "description": "Prigozhin mutiny"},
    "bakhmut_fall": {"date": "2023-05-20", "description": "Fall of Bakhmut"},
    "ukraine_offensive": {"date": "2023-06-04", "description": "Ukraine counteroffensive begins"},
}

# Validation thresholds
CORRELATION_THRESHOLD = 0.1
R2_THRESHOLD = 0.1


# =============================================================================
# DATA LOADING
# =============================================================================

def load_isw_embeddings() -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load ISW embeddings from npz file."""
    print("Loading ISW embeddings...")

    data = np.load(ISW_EMBEDDINGS_PATH, allow_pickle=True)
    daily_embeddings = {key: data[key] for key in data.files}
    dates = sorted(daily_embeddings.keys())

    print(f"  Loaded {len(daily_embeddings)} daily ISW embeddings")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Embedding dimension: {daily_embeddings[dates[0]].shape}")

    # Aggregate to monthly embeddings (to match model output resolution)
    print("  Aggregating to monthly resolution...")
    monthly_embeddings = {}

    for date in dates:
        # Extract month (first of month as key)
        dt = datetime.strptime(date, "%Y-%m-%d")
        month_key = dt.replace(day=1).strftime("%Y-%m-%d")

        if month_key not in monthly_embeddings:
            monthly_embeddings[month_key] = []
        monthly_embeddings[month_key].append(daily_embeddings[date])

    # Average embeddings within each month
    for month_key in monthly_embeddings:
        monthly_embeddings[month_key] = np.mean(monthly_embeddings[month_key], axis=0)

    monthly_dates = sorted(monthly_embeddings.keys())
    print(f"  Created {len(monthly_embeddings)} monthly ISW embeddings")
    print(f"  Monthly date range: {monthly_dates[0]} to {monthly_dates[-1]}")

    return monthly_embeddings, monthly_dates


def extract_real_model_latents() -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """
    Extract REAL latent representations by running model inference.

    Returns:
        Tuple of:
        - latents_by_date: Dict mapping date strings to latent vectors
        - latent_matrix: 2D array of latents [n_samples, d_model]
        - dates: List of date strings corresponding to latent_matrix rows
    """
    print("\n" + "="*60)
    print("EXTRACTING REAL MODEL LATENTS VIA INFERENCE")
    print("="*60)

    # Import model and data modules
    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
    from multi_resolution_han import MultiResolutionHAN, MultiResolutionHANConfig
    from multi_resolution_data import (
        MultiResolutionDataset,
        MultiResolutionConfig,
        multi_resolution_collate_fn
    )
    from torch.utils.data import DataLoader

    # Load checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

    # Get model state dict keys to infer configuration
    state_dict = checkpoint['model_state_dict']

    # Detect daily sources from state dict
    daily_sources = []
    for key in state_dict.keys():
        if key.startswith('daily_encoders.') and '.no_observation_token' in key:
            source = key.split('.')[1]
            if source not in daily_sources:
                daily_sources.append(source)

    # Detect monthly sources - this model aggregates daily to monthly, so may not have separate monthly encoders
    monthly_sources = []
    for key in state_dict.keys():
        if key.startswith('monthly_encoders.') and '.no_observation_token' in key:
            source = key.split('.')[1]
            if source not in monthly_sources:
                monthly_sources.append(source)

    print(f"  Detected daily sources: {daily_sources}")
    print(f"  Detected monthly sources: {monthly_sources}")

    # If no monthly sources detected, use empty list (model uses daily-to-monthly aggregation)
    if not monthly_sources:
        print("  Note: Model uses daily-to-monthly aggregation (no separate monthly encoders)")

    # Check for d_model from first layer
    d_model = 64  # Default
    for key in state_dict.keys():
        if 'no_observation_token' in key:
            d_model = state_dict[key].shape[-1]
            break
    print(f"  Model d_model: {d_model}")

    # Detect if disaggregated equipment is used
    use_disaggregated = 'drones' in daily_sources or 'armor' in daily_sources

    # Check for detrend_viirs based on viirs presence
    exclude_viirs = 'viirs' not in daily_sources
    detrend_viirs = True  # Default to True as per training

    # Create data configuration matching model
    # Use the default monthly_sources from config since this model aggregates daily to monthly
    data_config = MultiResolutionConfig(
        use_disaggregated_equipment=use_disaggregated,
        detrend_viirs=detrend_viirs,
        exclude_viirs=exclude_viirs,
    )

    print(f"\nLoading dataset with config:")
    print(f"  use_disaggregated_equipment: {use_disaggregated}")
    print(f"  detrend_viirs: {detrend_viirs}")
    print(f"  exclude_viirs: {exclude_viirs}")
    print(f"  effective daily sources: {data_config.get_effective_daily_sources()}")
    print(f"  monthly sources: {data_config.monthly_sources}")

    # Load train dataset (which is the largest split)
    dataset = MultiResolutionDataset(data_config, split='train')
    print(f"  Dataset size: {len(dataset)} samples")

    # Get feature dimensions from dataset sample
    sample = dataset[0]
    daily_source_names = list(sample.daily_features.keys())
    monthly_source_names = list(sample.monthly_features.keys())

    print(f"\nDataset daily sources: {daily_source_names}")
    print(f"Dataset monthly sources: {monthly_source_names}")

    # Build source configs dynamically from the actual data
    from multi_resolution_han import SourceConfig

    daily_source_configs = {}
    monthly_source_configs = {}

    for source_name in daily_source_names:
        n_features = sample.daily_features[source_name].shape[-1]
        daily_source_configs[source_name] = SourceConfig(
            name=source_name,
            n_features=n_features,
            resolution='daily',
        )

    for source_name in monthly_source_names:
        n_features = sample.monthly_features[source_name].shape[-1]
        monthly_source_configs[source_name] = SourceConfig(
            name=source_name,
            n_features=n_features,
            resolution='monthly',
        )

    # Create and load model
    print("\nCreating model...")
    model = MultiResolutionHAN(
        daily_source_configs=daily_source_configs,
        monthly_source_configs=monthly_source_configs,
        d_model=d_model,
        nhead=4,
        num_daily_layers=3,
        num_monthly_layers=2,
        num_fusion_layers=2,
        num_temporal_layers=2,
        dropout=0.0,  # No dropout for inference
    )

    # Load weights
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  Warning: missing keys: {load_result.missing_keys[:5]}...")
    if load_result.unexpected_keys:
        print(f"  Warning: unexpected keys: {load_result.unexpected_keys[:5]}...")

    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multi_resolution_collate_fn,
        num_workers=0,
    )

    # Extract latents
    print("\nExtracting latents via forward pass...")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model = model.to(device)

    # Collect all monthly latents indexed by date
    # Since samples overlap, we'll average latents for the same month
    monthly_latents_collection = {}  # date -> list of latent vectors

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            daily_features = {
                k: v.to(device) for k, v in batch['daily_features'].items()
            }
            daily_masks = {
                k: v.to(device) for k, v in batch['daily_masks'].items()
            }
            monthly_features = {
                k: v.to(device) for k, v in batch['monthly_features'].items()
            }
            monthly_masks = {
                k: v.to(device) for k, v in batch['monthly_masks'].items()
            }
            month_boundaries = batch['month_boundary_indices'].to(device)

            # Forward pass
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries
            )

            # Extract temporal_output as latent representation [batch, seq, d_model]
            if 'temporal_output' in outputs:
                latent = outputs['temporal_output'].cpu().numpy()
            elif 'fused_representation' in outputs:
                latent = outputs['fused_representation'].cpu().numpy()
            else:
                # Use casualty predictions as fallback
                latent = outputs['casualty_pred'].cpu().numpy()

            # latent shape: [batch, seq, d_model]
            batch_size = latent.shape[0]
            seq_len = latent.shape[1]

            # Extract dates for each sample in batch
            for sample_idx in range(batch_size):
                if 'monthly_dates' in batch and sample_idx < len(batch['monthly_dates']):
                    date_array = batch['monthly_dates'][sample_idx]
                    if date_array is not None:
                        # Match each timestep's latent to its month date
                        for t in range(min(seq_len, len(date_array))):
                            date_str = pd.Timestamp(date_array[t]).strftime('%Y-%m-%d')
                            latent_vec = latent[sample_idx, t, :]
                            if date_str not in monthly_latents_collection:
                                monthly_latents_collection[date_str] = []
                            monthly_latents_collection[date_str].append(latent_vec)

            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {batch_idx + 1} batches...")

    # Average latents for each unique month
    latents_by_date = {}
    for date, latent_list in monthly_latents_collection.items():
        latents_by_date[date] = np.mean(latent_list, axis=0)

    valid_dates = sorted(latents_by_date.keys())
    latent_matrix = np.array([latents_by_date[d] for d in valid_dates])

    print(f"\n  Unique monthly latents: {len(latents_by_date)}")
    print(f"  Latent matrix shape: {latent_matrix.shape}")
    print(f"  Date range: {min(valid_dates)} to {max(valid_dates)}")

    return latents_by_date, latent_matrix, valid_dates


# =============================================================================
# EXPERIMENT 1: BIDIRECTIONAL PREDICTION (Re-run with real latents)
# =============================================================================

def experiment_bidirectional_prediction(
    isw_embeddings: Dict[str, np.ndarray],
    model_latents: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Train linear probes to predict one from the other.
    Low R^2 in both directions confirms decorrelation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Bidirectional Prediction Test (REAL LATENTS)")
    print("="*60)

    # Find overlapping dates
    common_dates = sorted(set(isw_embeddings.keys()) & set(model_latents.keys()))
    print(f"Overlapping dates: {len(common_dates)}")

    if len(common_dates) < 30:
        print("ERROR: Insufficient overlapping dates")
        return {"status": "insufficient_data", "verdict": "INCONCLUSIVE"}

    # Stack embeddings
    isw_matrix = np.array([isw_embeddings[d] for d in common_dates])
    latent_matrix = np.array([model_latents[d] for d in common_dates])

    print(f"ISW matrix shape: {isw_matrix.shape}")
    print(f"Latent matrix shape: {latent_matrix.shape}")

    # Reduce dimensionality for stable regression
    n_components = min(20, isw_matrix.shape[1], latent_matrix.shape[1], len(common_dates) // 3)
    print(f"Using {n_components} PCA components")

    pca_isw = PCA(n_components=n_components)
    pca_latent = PCA(n_components=n_components)

    isw_reduced = pca_isw.fit_transform(isw_matrix)
    latent_reduced = pca_latent.fit_transform(latent_matrix)

    results = {}

    # Split data (use time-based split for temporal data)
    n = len(common_dates)
    train_idx = int(n * 0.7)

    X_train_isw, X_test_isw = isw_reduced[:train_idx], isw_reduced[train_idx:]
    X_train_lat, X_test_lat = latent_reduced[:train_idx], latent_reduced[train_idx:]

    print(f"Train samples: {len(X_train_isw)}, Test samples: {len(X_test_isw)}")

    # Direction 1: Latent -> ISW
    print("\n1. Predicting ISW from Latent:")
    model_lat_to_isw = Ridge(alpha=1.0)
    model_lat_to_isw.fit(X_train_lat, X_train_isw)
    pred_isw = model_lat_to_isw.predict(X_test_lat)

    r2_lat_to_isw = r2_score(X_test_isw, pred_isw, multioutput='variance_weighted')
    mse_lat_to_isw = mean_squared_error(X_test_isw, pred_isw)

    results['latent_to_isw_r2'] = float(r2_lat_to_isw)
    results['latent_to_isw_mse'] = float(mse_lat_to_isw)

    print(f"  R^2: {r2_lat_to_isw:.4f}")
    print(f"  MSE: {mse_lat_to_isw:.4f}")

    # Direction 2: ISW -> Latent
    print("\n2. Predicting Latent from ISW:")
    model_isw_to_lat = Ridge(alpha=1.0)
    model_isw_to_lat.fit(X_train_isw, X_train_lat)
    pred_lat = model_isw_to_lat.predict(X_test_isw)

    r2_isw_to_lat = r2_score(X_test_lat, pred_lat, multioutput='variance_weighted')
    mse_isw_to_lat = mean_squared_error(X_test_lat, pred_lat)

    results['isw_to_latent_r2'] = float(r2_isw_to_lat)
    results['isw_to_latent_mse'] = float(mse_isw_to_lat)

    print(f"  R^2: {r2_isw_to_lat:.4f}")
    print(f"  MSE: {mse_isw_to_lat:.4f}")

    # Cross-validation for more robust estimates
    print("\n3. Cross-validation (5-fold):")
    from sklearn.model_selection import cross_val_score

    # CV for Latent -> ISW (using mean of ISW as single target)
    isw_mean = isw_reduced.mean(axis=1)
    latent_mean = latent_reduced.mean(axis=1)

    cv_lat_to_isw = cross_val_score(
        Ridge(alpha=1.0), latent_reduced, isw_mean, cv=5, scoring='r2'
    )
    cv_isw_to_lat = cross_val_score(
        Ridge(alpha=1.0), isw_reduced, latent_mean, cv=5, scoring='r2'
    )

    results['cv_latent_to_isw_r2_mean'] = float(cv_lat_to_isw.mean())
    results['cv_latent_to_isw_r2_std'] = float(cv_lat_to_isw.std())
    results['cv_isw_to_latent_r2_mean'] = float(cv_isw_to_lat.mean())
    results['cv_isw_to_latent_r2_std'] = float(cv_isw_to_lat.std())

    print(f"  Latent->ISW CV R^2: {cv_lat_to_isw.mean():.4f} +/- {cv_lat_to_isw.std():.4f}")
    print(f"  ISW->Latent CV R^2: {cv_isw_to_lat.mean():.4f} +/- {cv_isw_to_lat.std():.4f}")

    # Verdict
    max_r2 = max(r2_lat_to_isw, r2_isw_to_lat)
    max_cv_r2 = max(cv_lat_to_isw.mean(), cv_isw_to_lat.mean())

    results['max_r2'] = float(max_r2)
    results['max_cv_r2'] = float(max_cv_r2)

    # Use the more robust CV estimate for verdict
    decisive_r2 = max(max_r2, max_cv_r2)
    results['decisive_r2'] = float(decisive_r2)

    results['verdict'] = "CONFIRMED" if decisive_r2 < R2_THRESHOLD else "REFUTED"

    print(f"\n** Max R^2 (test): {max_r2:.4f}")
    print(f"** Max R^2 (CV): {max_cv_r2:.4f}")
    print(f"** Decisive R^2: {decisive_r2:.4f}")
    print(f"** Threshold: {R2_THRESHOLD}")
    print(f"** Verdict: {results['verdict']}")

    return results


# =============================================================================
# EXPERIMENT 2: EVENT-TRIGGERED RESPONSE (Re-run with real latents)
# =============================================================================

def experiment_event_response(
    isw_embeddings: Dict[str, np.ndarray],
    model_latents: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Analyze latent shifts around major events.
    Compare ISW embedding changes to model latent changes.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Event-Triggered Latent Response (REAL LATENTS)")
    print("="*60)

    results = {'events': {}}

    window_size = 7  # Days before/after event

    for event_name, event_info in MAJOR_EVENTS.items():
        event_date = event_info['date']
        print(f"\nAnalyzing: {event_info['description']} ({event_date})")

        # Generate date windows
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        before_dates = [(event_dt - timedelta(days=i)).strftime("%Y-%m-%d")
                       for i in range(1, window_size + 1)]
        after_dates = [(event_dt + timedelta(days=i)).strftime("%Y-%m-%d")
                      for i in range(1, window_size + 1)]

        # ISW embeddings analysis
        isw_before = [isw_embeddings.get(d) for d in before_dates if d in isw_embeddings]
        isw_after = [isw_embeddings.get(d) for d in after_dates if d in isw_embeddings]

        # Model latents analysis
        latent_before = [model_latents.get(d) for d in before_dates if d in model_latents]
        latent_after = [model_latents.get(d) for d in after_dates if d in model_latents]

        event_result = {
            'date': event_date,
            'description': event_info['description'],
            'n_isw_before': len(isw_before),
            'n_isw_after': len(isw_after),
            'n_latent_before': len(latent_before),
            'n_latent_after': len(latent_after),
        }

        if len(isw_before) >= 3 and len(isw_after) >= 3:
            isw_before_mean = np.mean(isw_before, axis=0)
            isw_after_mean = np.mean(isw_after, axis=0)
            isw_shift = np.linalg.norm(isw_after_mean - isw_before_mean)
            isw_shift_normalized = isw_shift / (np.linalg.norm(isw_before_mean) + 1e-10)

            event_result['isw_shift'] = float(isw_shift)
            event_result['isw_shift_normalized'] = float(isw_shift_normalized)
            print(f"  ISW shift: {isw_shift:.4f} (normalized: {isw_shift_normalized:.4f})")
        else:
            print(f"  ISW: insufficient data ({len(isw_before)} before, {len(isw_after)} after)")

        if len(latent_before) >= 3 and len(latent_after) >= 3:
            latent_before_mean = np.mean(latent_before, axis=0)
            latent_after_mean = np.mean(latent_after, axis=0)
            latent_shift = np.linalg.norm(latent_after_mean - latent_before_mean)
            latent_shift_normalized = latent_shift / (np.linalg.norm(latent_before_mean) + 1e-10)

            event_result['latent_shift'] = float(latent_shift)
            event_result['latent_shift_normalized'] = float(latent_shift_normalized)
            print(f"  Latent shift: {latent_shift:.4f} (normalized: {latent_shift_normalized:.4f})")
        else:
            print(f"  Latent: insufficient data ({len(latent_before)} before, {len(latent_after)} after)")

        results['events'][event_name] = event_result

    # Compute correlation between ISW and latent shifts
    isw_shifts = []
    latent_shifts = []
    for event_name, event_result in results['events'].items():
        if 'isw_shift_normalized' in event_result and 'latent_shift_normalized' in event_result:
            isw_shifts.append(event_result['isw_shift_normalized'])
            latent_shifts.append(event_result['latent_shift_normalized'])

    if len(isw_shifts) >= 3:
        shift_correlation, shift_p = stats.pearsonr(isw_shifts, latent_shifts)
        results['shift_correlation'] = float(shift_correlation)
        results['shift_correlation_p'] = float(shift_p)

        print(f"\nShift correlation across events: {shift_correlation:.4f} (p={shift_p:.4e})")

        # Spearman correlation (more robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(isw_shifts, latent_shifts)
        results['spearman_correlation'] = float(spearman_r)
        results['spearman_p'] = float(spearman_p)
        print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.4e})")

        # Use stricter threshold for event correlation
        # p > 0.05 indicates non-significant correlation
        is_significant = shift_p < 0.05
        correlation_strong = abs(shift_correlation) > 0.5

        if is_significant and correlation_strong:
            results['verdict'] = "REFUTED"
        else:
            results['verdict'] = "CONFIRMED"
    else:
        results['verdict'] = "INCONCLUSIVE"
        print("\nInsufficient data to compute shift correlation")

    print(f"** Verdict: {results['verdict']}")

    return results


# =============================================================================
# ADDITIONAL: CANONICAL CORRELATION ANALYSIS
# =============================================================================

def experiment_canonical_correlation(
    isw_embeddings: Dict[str, np.ndarray],
    model_latents: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Canonical Correlation Analysis between ISW and model latents.
    This finds the best linear combinations that maximize correlation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Canonical Correlation Analysis")
    print("="*60)

    from scipy.linalg import svd

    common_dates = sorted(set(isw_embeddings.keys()) & set(model_latents.keys()))
    print(f"Overlapping dates: {len(common_dates)}")

    if len(common_dates) < 50:
        print("Insufficient data for CCA")
        return {"status": "insufficient_data", "verdict": "INCONCLUSIVE"}

    # Stack and standardize
    isw_matrix = np.array([isw_embeddings[d] for d in common_dates])
    latent_matrix = np.array([model_latents[d] for d in common_dates])

    scaler_isw = StandardScaler()
    scaler_latent = StandardScaler()

    isw_scaled = scaler_isw.fit_transform(isw_matrix)
    latent_scaled = scaler_latent.fit_transform(latent_matrix)

    # Reduce to manageable dimensions
    n_components = min(20, isw_scaled.shape[1], latent_scaled.shape[1])
    pca_isw = PCA(n_components=n_components)
    pca_latent = PCA(n_components=n_components)

    X = pca_isw.fit_transform(isw_scaled)
    Y = pca_latent.fit_transform(latent_scaled)

    print(f"Reduced dimensions: ISW {X.shape}, Latent {Y.shape}")

    # CCA via SVD
    n = X.shape[0]
    Qx, Rx = np.linalg.qr(X)
    Qy, Ry = np.linalg.qr(Y)

    # SVD of Qx' @ Qy
    U, s, Vt = svd(Qx.T @ Qy)

    canonical_correlations = np.clip(s, 0, 1)

    results = {
        'canonical_correlations': canonical_correlations.tolist(),
        'max_canonical_corr': float(canonical_correlations[0]),
        'mean_canonical_corr': float(canonical_correlations.mean()),
        'n_significant': int(np.sum(canonical_correlations > 0.1)),
    }

    print(f"\nTop 5 Canonical Correlations:")
    for i, cc in enumerate(canonical_correlations[:5]):
        print(f"  CC{i+1}: {cc:.4f}")

    print(f"\nMax canonical correlation: {results['max_canonical_corr']:.4f}")
    print(f"Mean canonical correlation: {results['mean_canonical_corr']:.4f}")
    print(f"# significant (>0.1): {results['n_significant']}")

    # Verdict based on canonical correlations
    # Even the best linear combination should show low correlation
    if results['max_canonical_corr'] < 0.3:
        results['verdict'] = "CONFIRMED"
    elif results['max_canonical_corr'] > 0.5:
        results['verdict'] = "REFUTED"
    else:
        results['verdict'] = "BORDERLINE"

    print(f"** Verdict: {results['verdict']}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run validation with real model latents."""
    print("="*60)
    print("C3: ISW-LATENT ALIGNMENT VALIDATION (REAL LATENTS)")
    print("="*60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"ISW Embeddings: {ISW_EMBEDDINGS_PATH}")
    print(f"Output: {OUTPUT_DIR_PATH}")

    ensure_dir(OUTPUT_DIR_PATH)

    # Load ISW embeddings
    isw_embeddings, isw_dates = load_isw_embeddings()

    # Extract REAL model latents via inference
    model_latents, latent_matrix, latent_dates = extract_real_model_latents()

    print(f"\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"ISW embeddings: {len(isw_embeddings)} dates")
    print(f"Model latents: {len(model_latents)} dates")

    common_dates = sorted(set(isw_embeddings.keys()) & set(model_latents.keys()))
    print(f"Overlapping dates: {len(common_dates)}")

    if len(common_dates) < 30:
        print("ERROR: Insufficient overlapping dates for analysis")
        return

    # Run experiments
    results = {}

    # Re-run borderline experiments with REAL latents
    results['bidirectional_prediction'] = experiment_bidirectional_prediction(
        isw_embeddings, model_latents
    )

    results['event_response'] = experiment_event_response(
        isw_embeddings, model_latents
    )

    # Additional experiment: CCA
    results['canonical_correlation'] = experiment_canonical_correlation(
        isw_embeddings, model_latents
    )

    # =============================================================================
    # FINAL VERDICT
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL VERDICT DETERMINATION")
    print("="*60)

    # Collect verdicts
    verdicts = []
    for exp_name in ['bidirectional_prediction', 'event_response', 'canonical_correlation']:
        if exp_name in results:
            v = results[exp_name].get('verdict', 'INCONCLUSIVE')
            verdicts.append(v)
            print(f"  {exp_name}: {v}")

    confirmed = verdicts.count('CONFIRMED')
    refuted = verdicts.count('REFUTED')

    # Decisive logic:
    # - If bidirectional_prediction shows R^2 < 0.1: strong evidence for CONFIRMED
    # - If event_response shows non-significant correlation: evidence for CONFIRMED
    # - CCA provides additional nuance

    bidirectional_r2 = results.get('bidirectional_prediction', {}).get('decisive_r2', 1.0)
    event_p = results.get('event_response', {}).get('shift_correlation_p', 0.0)
    cca_max = results.get('canonical_correlation', {}).get('max_canonical_corr', 1.0)

    print(f"\nKey metrics:")
    print(f"  Bidirectional R^2: {bidirectional_r2:.4f} (threshold: {R2_THRESHOLD})")
    print(f"  Event correlation p-value: {event_p:.4e} (significant if < 0.05)")
    print(f"  Max CCA: {cca_max:.4f} (low if < 0.3)")

    # Decision logic
    evidence_for_confirmed = 0
    evidence_for_refuted = 0

    if bidirectional_r2 < R2_THRESHOLD:
        evidence_for_confirmed += 2  # Strong evidence
    elif bidirectional_r2 > 0.2:
        evidence_for_refuted += 2

    if event_p > 0.05:
        evidence_for_confirmed += 1  # Non-significant correlation
    elif event_p < 0.01:
        evidence_for_refuted += 1

    if cca_max < 0.3:
        evidence_for_confirmed += 1
    elif cca_max > 0.5:
        evidence_for_refuted += 1

    print(f"\nEvidence score: CONFIRMED={evidence_for_confirmed}, REFUTED={evidence_for_refuted}")

    if evidence_for_confirmed > evidence_for_refuted:
        final_verdict = "CONFIRMED"
    elif evidence_for_refuted > evidence_for_confirmed:
        final_verdict = "REFUTED"
    else:
        # Tie-breaker: use bidirectional prediction as primary
        final_verdict = "CONFIRMED" if bidirectional_r2 < R2_THRESHOLD else "REFUTED"

    results['final_verdict'] = final_verdict
    results['evidence_score'] = {
        'confirmed': evidence_for_confirmed,
        'refuted': evidence_for_refuted,
    }

    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"{'='*60}")

    # Save results
    results_path = OUTPUT_DIR_PATH / "C3_real_latent_results.json"

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = main()
