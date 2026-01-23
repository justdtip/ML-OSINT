"""
Training Targets Module for Multi-Resolution HAN

This module provides real training targets instead of synthetic data:
1. Phase/Regime Classification - from curated timeline data
2. Casualty Prediction - from HDX and UCDP conflict data
3. Phase Transition Detection - binary labels from timeline
4. Nightlight Anomaly Detection - from VIIRS data changes

Data Sources:
- data/timelines/anchored/phase_labels.json - phase labels per date
- data/hdx/ukraine/conflict_events_*.csv - monthly fatalities
- data/ucdp/ged_events.csv - conflict events with deaths
- data/nasa/viirs_nightlights/ - daily radiance statistics

Usage:
    from training_targets import TargetLoader, create_training_targets

    # Create target loader
    target_loader = TargetLoader(data_dir="data")

    # Get targets for training
    targets = target_loader.get_targets_for_dates(dates, monthly=True)

Author: AI Engineering Team
Date: 2026-01-21
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, CHECKPOINT_DIR,
    MULTI_RES_CHECKPOINT_DIR, PIPELINE_CHECKPOINT_DIR,
    HAN_BEST_MODEL, HAN_FINAL_MODEL,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

# Phase name to index mapping (from timeline anchoring)
PHASE_TO_INDEX = {
    'southern_offensive': 0,
    'northeastern_offensive': 1,
    'northern_offensive': 2,
    'donbas_offensive': 3,
    'ukrainian_counteroffensive': 4,
    'bakhmut_campaign': 5,
    '2023_counteroffensive': 6,
    'avdiivka_campaign': 7,
    '2024_russian_offensive': 8,
    'kursk_operation': 9,
    'baseline': 10,
}

INDEX_TO_PHASE = {v: k for k, v in PHASE_TO_INDEX.items()}
N_PHASES = len(PHASE_TO_INDEX)


# =============================================================================
# TARGET LOADER
# =============================================================================

class TargetLoader:
    """Loads and manages real training targets."""

    def __init__(self, data_dir: str = None):
        # Use DATA_DIR from config.paths as default
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

        # Initialize data containers
        self.phase_labels: Optional[Dict[str, dict]] = None
        self.hdx_data: Optional[pd.DataFrame] = None
        self.ucdp_data: Optional[pd.DataFrame] = None
        self.viirs_data: Optional[pd.DataFrame] = None

        self._loaded = False

    def load(self) -> bool:
        """Load all target data sources."""
        success = True

        # Load phase labels
        phase_path = self.data_dir / "timelines" / "anchored" / "phase_labels.json"
        if phase_path.exists():
            with open(phase_path) as f:
                self.phase_labels = json.load(f)
            logger.info(f"Loaded phase labels for {len(self.phase_labels)} dates")
        else:
            logger.warning(f"Phase labels not found: {phase_path}")
            success = False

        # Load HDX conflict data
        hdx_path = self.data_dir / "hdx" / "ukraine" / "conflict_events_2022_present.csv"
        if hdx_path.exists():
            self.hdx_data = pd.read_csv(hdx_path)
            # Parse dates from reference_period_start
            if 'reference_period_start' in self.hdx_data.columns:
                self.hdx_data['date'] = pd.to_datetime(self.hdx_data['reference_period_start'])
                # Extract year/month for monthly aggregation
                self.hdx_data['year'] = self.hdx_data['date'].dt.year
                self.hdx_data['month'] = self.hdx_data['date'].dt.month
            elif 'event_date' in self.hdx_data.columns:
                self.hdx_data['date'] = pd.to_datetime(self.hdx_data['event_date'])
            logger.info(f"Loaded HDX data: {len(self.hdx_data)} records")
        else:
            logger.warning(f"HDX data not found: {hdx_path}")

        # Load UCDP data
        ucdp_path = self.data_dir / "ucdp" / "ged_events.csv"
        if ucdp_path.exists():
            self.ucdp_data = pd.read_csv(ucdp_path)
            if 'date_start' in self.ucdp_data.columns:
                # Handle mixed date formats in UCDP data
                self.ucdp_data['date'] = pd.to_datetime(
                    self.ucdp_data['date_start'],
                    format='mixed',
                    dayfirst=False
                )
            logger.info(f"Loaded UCDP data: {len(self.ucdp_data)} events")
        else:
            logger.warning(f"UCDP data not found: {ucdp_path}")

        # Load VIIRS nightlight data
        viirs_path = self.data_dir / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"
        if viirs_path.exists():
            self.viirs_data = pd.read_csv(viirs_path)
            self.viirs_data['date'] = pd.to_datetime(self.viirs_data['date'])
            # Aggregate across tiles (average radiance)
            self.viirs_daily = self.viirs_data.groupby('date').agg({
                'radiance_mean': 'mean',
                'radiance_std': 'mean',
                'radiance_p50': 'mean',
                'radiance_p90': 'mean',
                'pct_clear_sky': 'mean',
            }).reset_index()
            logger.info(f"Loaded VIIRS data: {len(self.viirs_daily)} days")
        else:
            logger.warning(f"VIIRS data not found: {viirs_path}")

        self._loaded = True
        return success

    def get_phase_target(self, date: str) -> Tuple[int, bool, int]:
        """
        Get phase classification target for a date.

        Returns:
            phase_index: Index of primary phase (0 to N_PHASES-1)
            is_transition: Whether this is a phase transition day
            n_active_ops: Number of active operations
        """
        if self.phase_labels is None:
            return N_PHASES - 1, False, 0  # Default to 'baseline'

        info = self.phase_labels.get(date)
        if info is None:
            return N_PHASES - 1, False, 0

        phase = info.get('primary_phase', 'baseline')
        phase_idx = PHASE_TO_INDEX.get(phase, N_PHASES - 1)
        is_transition = info.get('is_phase_transition', False)
        n_active = info.get('n_active_operations', 0)

        return phase_idx, is_transition, n_active

    def get_phase_targets_batch(
        self,
        dates: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Get phase targets for a batch of dates.

        Returns:
            Dict with:
                'phase_index': [n_dates] int array
                'is_transition': [n_dates] bool array
                'n_active_ops': [n_dates] int array
                'valid_mask': [n_dates] bool array
        """
        n_dates = len(dates)
        phase_indices = np.full(n_dates, N_PHASES - 1, dtype=np.int64)
        is_transition = np.zeros(n_dates, dtype=bool)
        n_active_ops = np.zeros(n_dates, dtype=np.int64)
        valid_mask = np.zeros(n_dates, dtype=bool)

        for i, date in enumerate(dates):
            if date and self.phase_labels and date in self.phase_labels:
                phase_idx, trans, n_ops = self.get_phase_target(date)
                phase_indices[i] = phase_idx
                is_transition[i] = trans
                n_active_ops[i] = n_ops
                valid_mask[i] = True

        return {
            'phase_index': phase_indices,
            'is_transition': is_transition,
            'n_active_ops': n_active_ops,
            'valid_mask': valid_mask,
        }

    def get_monthly_casualty_target(
        self,
        year: int,
        month: int,
    ) -> Tuple[float, float, bool]:
        """
        Get monthly casualty statistics.

        Returns:
            fatalities: Total fatalities for the month
            events: Number of conflict events
            valid: Whether data is available
        """
        if self.hdx_data is None:
            return 0.0, 0.0, False

        # Filter to month
        mask = (
            (self.hdx_data['date'].dt.year == year) &
            (self.hdx_data['date'].dt.month == month)
        )
        month_data = self.hdx_data[mask]

        if len(month_data) == 0:
            return 0.0, 0.0, False

        fatalities = month_data['fatalities'].sum() if 'fatalities' in month_data.columns else 0.0
        events = len(month_data)

        return float(fatalities), float(events), True

    def get_monthly_casualties_batch(
        self,
        year_months: List[Tuple[int, int]],
    ) -> Dict[str, np.ndarray]:
        """
        Get monthly casualty targets for multiple months.

        Args:
            year_months: List of (year, month) tuples

        Returns:
            Dict with:
                'fatalities': [n_months] float array
                'events': [n_months] float array
                'valid_mask': [n_months] bool array
        """
        n_months = len(year_months)
        fatalities = np.zeros(n_months, dtype=np.float32)
        events = np.zeros(n_months, dtype=np.float32)
        valid_mask = np.zeros(n_months, dtype=bool)

        for i, (year, month) in enumerate(year_months):
            fat, evt, valid = self.get_monthly_casualty_target(year, month)
            fatalities[i] = fat
            events[i] = evt
            valid_mask[i] = valid

        return {
            'fatalities': fatalities,
            'events': events,
            'valid_mask': valid_mask,
        }

    def get_viirs_anomaly(
        self,
        date: str,
        window_days: int = 7,
    ) -> Tuple[float, bool]:
        """
        Compute VIIRS radiance anomaly relative to recent window.

        Negative values indicate darker than usual (potential destruction/outages).

        Returns:
            anomaly_score: Z-score of radiance relative to window
            valid: Whether data is available
        """
        if self.viirs_daily is None:
            return 0.0, False

        target_date = pd.to_datetime(date)

        # Get window data
        window_start = target_date - timedelta(days=window_days)
        mask = (
            (self.viirs_daily['date'] >= window_start) &
            (self.viirs_daily['date'] <= target_date)
        )
        window_data = self.viirs_daily[mask]

        if len(window_data) < 3:
            return 0.0, False

        # Get target day
        target_mask = self.viirs_daily['date'] == target_date
        if not target_mask.any():
            return 0.0, False

        target_radiance = self.viirs_daily.loc[target_mask, 'radiance_mean'].values[0]
        window_mean = window_data['radiance_mean'].mean()
        window_std = window_data['radiance_mean'].std()

        if window_std < 0.001:
            return 0.0, True

        anomaly = (target_radiance - window_mean) / window_std

        return float(anomaly), True

    def get_viirs_features(
        self,
        dates: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Get VIIRS features for a list of dates.

        Returns:
            Dict with:
                'radiance_mean': [n_dates] mean radiance
                'radiance_p50': [n_dates] median radiance
                'anomaly_score': [n_dates] z-score anomaly
                'valid_mask': [n_dates] bool
        """
        n_dates = len(dates)
        radiance_mean = np.zeros(n_dates, dtype=np.float32)
        radiance_p50 = np.zeros(n_dates, dtype=np.float32)
        anomaly_scores = np.zeros(n_dates, dtype=np.float32)
        valid_mask = np.zeros(n_dates, dtype=bool)

        if self.viirs_daily is None:
            return {
                'radiance_mean': radiance_mean,
                'radiance_p50': radiance_p50,
                'anomaly_score': anomaly_scores,
                'valid_mask': valid_mask,
            }

        for i, date in enumerate(dates):
            if not date:
                continue

            target_date = pd.to_datetime(date)
            mask = self.viirs_daily['date'] == target_date

            if mask.any():
                row = self.viirs_daily.loc[mask].iloc[0]
                radiance_mean[i] = row['radiance_mean']
                radiance_p50[i] = row['radiance_p50']
                valid_mask[i] = True

                # Compute anomaly
                anomaly, _ = self.get_viirs_anomaly(date)
                anomaly_scores[i] = anomaly

        return {
            'radiance_mean': radiance_mean,
            'radiance_p50': radiance_p50,
            'anomaly_score': anomaly_scores,
            'valid_mask': valid_mask,
        }

    def get_combined_targets(
        self,
        daily_dates: List[str],
        monthly_year_months: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Get all targets for training.

        Args:
            daily_dates: List of daily date strings
            monthly_year_months: List of (year, month) tuples

        Returns:
            Dict with daily and monthly targets
        """
        targets = {}

        # Daily phase targets
        phase_targets = self.get_phase_targets_batch(daily_dates)
        targets['daily_phase_index'] = phase_targets['phase_index']
        targets['daily_is_transition'] = phase_targets['is_transition']
        targets['daily_n_active_ops'] = phase_targets['n_active_ops']
        targets['daily_phase_valid'] = phase_targets['valid_mask']

        # VIIRS targets
        viirs = self.get_viirs_features(daily_dates)
        targets['daily_radiance'] = viirs['radiance_mean']
        targets['daily_anomaly'] = viirs['anomaly_score']
        targets['daily_viirs_valid'] = viirs['valid_mask']

        # Monthly casualty targets
        casualty = self.get_monthly_casualties_batch(monthly_year_months)
        targets['monthly_fatalities'] = casualty['fatalities']
        targets['monthly_events'] = casualty['events']
        targets['monthly_casualty_valid'] = casualty['valid_mask']

        # Aggregate daily phases to monthly
        if len(daily_dates) > 0 and len(monthly_year_months) > 0:
            monthly_phases = self._aggregate_daily_to_monthly(
                phase_targets['phase_index'],
                daily_dates,
                monthly_year_months,
            )
            targets['monthly_phase_index'] = monthly_phases['phase_index']
            targets['monthly_phase_valid'] = monthly_phases['valid_mask']
            targets['monthly_has_transition'] = monthly_phases['has_transition']

        return targets

    def _aggregate_daily_to_monthly(
        self,
        daily_phases: np.ndarray,
        daily_dates: List[str],
        monthly_year_months: List[Tuple[int, int]],
    ) -> Dict[str, np.ndarray]:
        """Aggregate daily phase labels to monthly (mode)."""
        n_months = len(monthly_year_months)
        monthly_phases = np.full(n_months, N_PHASES - 1, dtype=np.int64)
        valid_mask = np.zeros(n_months, dtype=bool)
        has_transition = np.zeros(n_months, dtype=bool)

        # Group daily indices by month
        for i, (year, month) in enumerate(monthly_year_months):
            month_indices = []
            for j, date in enumerate(daily_dates):
                if not date:
                    continue
                try:
                    dt = datetime.strptime(date, "%Y-%m-%d")
                    if dt.year == year and dt.month == month:
                        month_indices.append(j)
                except ValueError:
                    continue

            if month_indices:
                month_phases = daily_phases[month_indices]
                # Use mode (most common phase)
                values, counts = np.unique(month_phases, return_counts=True)
                monthly_phases[i] = values[np.argmax(counts)]
                valid_mask[i] = True
                # Check for transitions (multiple unique phases)
                has_transition[i] = len(values) > 1

        return {
            'phase_index': monthly_phases,
            'valid_mask': valid_mask,
            'has_transition': has_transition,
        }


# =============================================================================
# TENSOR CONVERSION UTILITIES
# =============================================================================

def targets_to_tensors(
    targets: Dict[str, np.ndarray],
    device: torch.device = None,
) -> Dict[str, Tensor]:
    """Convert numpy target arrays to PyTorch tensors."""
    if device is None:
        device = torch.device('cpu')

    tensors = {}
    for key, value in targets.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                tensors[key] = torch.from_numpy(value).bool().to(device)
            elif np.issubdtype(value.dtype, np.integer):
                tensors[key] = torch.from_numpy(value).long().to(device)
            else:
                tensors[key] = torch.from_numpy(value).float().to(device)
        else:
            tensors[key] = value

    return tensors


# =============================================================================
# LOSS FUNCTIONS FOR REAL TARGETS
# =============================================================================

def compute_phase_loss(
    phase_logits: Tensor,      # [batch, seq, n_phases]
    phase_targets: Tensor,      # [batch, seq]
    valid_mask: Tensor,         # [batch, seq]
) -> Tensor:
    """Compute cross-entropy loss for phase classification."""
    batch_size, seq_len, n_classes = phase_logits.shape

    # Flatten
    logits_flat = phase_logits.view(-1, n_classes)
    targets_flat = phase_targets.view(-1)
    mask_flat = valid_mask.view(-1)

    # Filter to valid
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=phase_logits.device, requires_grad=True)

    loss = torch.nn.functional.cross_entropy(
        logits_flat[valid_idx],
        targets_flat[valid_idx],
    )

    return loss


def compute_transition_loss(
    transition_logits: Tensor,  # [batch, seq, 1]
    transition_targets: Tensor,  # [batch, seq]
    valid_mask: Tensor,          # [batch, seq]
) -> Tensor:
    """Compute BCE loss for phase transition detection."""
    logits_flat = transition_logits.view(-1)
    targets_flat = transition_targets.float().view(-1)
    mask_flat = valid_mask.view(-1)

    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=transition_logits.device, requires_grad=True)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_flat[valid_idx],
        targets_flat[valid_idx],
    )

    return loss


def compute_casualty_loss(
    casualty_pred: Tensor,      # [batch, seq, 3]
    casualty_var: Tensor,       # [batch, seq, 3]
    fatality_targets: Tensor,   # [batch, seq]
    valid_mask: Tensor,         # [batch, seq]
) -> Tensor:
    """Compute NLL loss for casualty prediction with learned variance."""
    # Use the first output dimension for fatality prediction
    pred = casualty_pred[:, :, 0]  # [batch, seq]
    var = casualty_var[:, :, 0]    # [batch, seq]

    # Flatten
    pred_flat = pred.view(-1)
    var_flat = var.view(-1).clamp(min=1e-6)
    targets_flat = fatality_targets.view(-1)
    mask_flat = valid_mask.view(-1)

    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=casualty_pred.device, requires_grad=True)

    # Negative log-likelihood with learned variance
    nll = 0.5 * (
        torch.log(var_flat[valid_idx]) +
        (pred_flat[valid_idx] - targets_flat[valid_idx]).pow(2) / var_flat[valid_idx]
    )

    return nll.mean()


def compute_anomaly_loss(
    anomaly_score: Tensor,      # [batch, seq, 1]
    anomaly_targets: Tensor,    # [batch, seq] z-scores
    valid_mask: Tensor,         # [batch, seq]
) -> Tensor:
    """Compute MSE loss for anomaly score prediction."""
    pred_flat = anomaly_score.view(-1)
    targets_flat = anomaly_targets.view(-1)
    mask_flat = valid_mask.view(-1)

    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=anomaly_score.device, requires_grad=True)

    loss = torch.nn.functional.mse_loss(
        pred_flat[valid_idx],
        targets_flat[valid_idx],
    )

    return loss


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Training Targets Module - Test")
    print("=" * 70)

    # Use DATA_DIR from config.paths
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        exit(1)

    print(f"Data directory: {DATA_DIR}")

    # Create and load target loader (uses DATA_DIR by default)
    loader = TargetLoader()
    success = loader.load()
    print(f"\nData loaded successfully: {success}")

    # Test phase targets
    print("\n" + "-" * 40)
    print("Test: Phase Targets")
    test_dates = ["2022-03-15", "2022-09-10", "2023-06-10", "2024-01-15"]
    for date in test_dates:
        phase_idx, is_trans, n_ops = loader.get_phase_target(date)
        phase_name = INDEX_TO_PHASE.get(phase_idx, 'unknown')
        print(f"  {date}: phase={phase_name}, transition={is_trans}, n_ops={n_ops}")

    # Test batch phase targets
    print("\n" + "-" * 40)
    print("Test: Batch Phase Targets")
    batch_targets = loader.get_phase_targets_batch(test_dates)
    print(f"  Phase indices: {batch_targets['phase_index']}")
    print(f"  Valid mask: {batch_targets['valid_mask']}")

    # Test VIIRS targets
    print("\n" + "-" * 40)
    print("Test: VIIRS Targets")
    viirs = loader.get_viirs_features(test_dates)
    print(f"  Radiance mean: {viirs['radiance_mean']}")
    print(f"  Anomaly scores: {viirs['anomaly_score']}")
    print(f"  Valid mask: {viirs['valid_mask']}")

    # Test monthly casualties
    print("\n" + "-" * 40)
    print("Test: Monthly Casualties")
    year_months = [(2022, 3), (2022, 9), (2023, 6), (2024, 1)]
    casualties = loader.get_monthly_casualties_batch(year_months)
    print(f"  Fatalities: {casualties['fatalities']}")
    print(f"  Events: {casualties['events']}")
    print(f"  Valid: {casualties['valid_mask']}")

    # Test combined targets
    print("\n" + "-" * 40)
    print("Test: Combined Targets")
    daily_dates = [f"2022-03-{d:02d}" for d in range(1, 32)]
    monthly_ym = [(2022, 3)]
    combined = loader.get_combined_targets(daily_dates, monthly_ym)
    print(f"  Keys: {list(combined.keys())}")
    print(f"  Daily phase valid count: {combined['daily_phase_valid'].sum()}")
    print(f"  Monthly phase: {INDEX_TO_PHASE.get(combined['monthly_phase_index'][0], 'unknown')}")

    # Test tensor conversion
    print("\n" + "-" * 40)
    print("Test: Tensor Conversion")
    tensors = targets_to_tensors(combined)
    for key, value in tensors.items():
        if isinstance(value, Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
