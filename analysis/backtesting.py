"""
Backtesting Framework for Multi-Resolution HAN

This module provides tools for validating the model's predictive performance
on historical data using rolling forecasts and event detection.

Usage:
    python -m analysis.backtesting --checkpoint path/to/checkpoint.pt
    python -m analysis.backtesting --checkpoint path/to/checkpoint.pt --mode events
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Optional tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        for i, item in enumerate(iterable):
            if i % 50 == 0:
                print(f"\r{desc}: {i}/{total if total else '?'}", end='', flush=True)
            yield item
        print()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.multi_resolution_data import MultiResolutionConfig, MultiResolutionDataset
from analysis.multi_resolution_han import MultiResolutionHAN, SourceConfig


# =============================================================================
# KNOWN CONFLICT PHASES AND EVENTS
# =============================================================================

# Major conflict phases with approximate date ranges
# These are based on widely reported phases of the Russo-Ukrainian War
CONFLICT_PHASES = [
    {
        'name': 'Initial Invasion',
        'start': '2022-02-24',
        'end': '2022-04-02',
        'description': 'Russian invasion, failed Kyiv offensive',
        'phase_id': 0,
    },
    {
        'name': 'Eastern Focus',
        'start': '2022-04-03',
        'end': '2022-08-28',
        'description': 'Russian focus on Donbas, Severodonetsk, Lysychansk',
        'phase_id': 1,
    },
    {
        'name': 'Ukrainian Counteroffensive',
        'start': '2022-08-29',
        'end': '2022-11-11',
        'description': 'Kharkiv and Kherson counteroffensives',
        'phase_id': 2,
    },
    {
        'name': 'Winter Stalemate',
        'start': '2022-11-12',
        'end': '2023-06-03',
        'description': 'Bakhmut battle, positional warfare',
        'phase_id': 3,
    },
    {
        'name': '2023 Counteroffensive',
        'start': '2023-06-04',
        'end': '2023-11-30',
        'description': 'Ukrainian summer counteroffensive',
        'phase_id': 4,
    },
    {
        'name': 'Attritional Warfare',
        'start': '2023-12-01',
        'end': '2024-08-05',
        'description': 'Continued positional warfare, Avdiivka',
        'phase_id': 5,
    },
    {
        'name': 'Kursk Incursion',
        'start': '2024-08-06',
        'end': '2026-12-31',  # Ongoing
        'description': 'Ukrainian incursion into Russia',
        'phase_id': 6,
    },
]

# Major events for anomaly detection validation
MAJOR_EVENTS = [
    {'date': '2022-02-24', 'name': 'Invasion begins', 'type': 'military', 'severity': 5},
    {'date': '2022-03-02', 'name': 'Kherson falls', 'type': 'territorial', 'severity': 4},
    {'date': '2022-04-02', 'name': 'Kyiv offensive fails', 'type': 'military', 'severity': 4},
    {'date': '2022-04-14', 'name': 'Moskva sunk', 'type': 'military', 'severity': 5},
    {'date': '2022-05-20', 'name': 'Mariupol falls', 'type': 'territorial', 'severity': 4},
    {'date': '2022-06-25', 'name': 'Severodonetsk falls', 'type': 'territorial', 'severity': 3},
    {'date': '2022-09-06', 'name': 'Kharkiv counteroffensive begins', 'type': 'military', 'severity': 5},
    {'date': '2022-09-11', 'name': 'Izium liberated', 'type': 'territorial', 'severity': 4},
    {'date': '2022-10-08', 'name': 'Kerch Bridge attack', 'type': 'military', 'severity': 5},
    {'date': '2022-11-11', 'name': 'Kherson liberated', 'type': 'territorial', 'severity': 5},
    {'date': '2023-05-20', 'name': 'Bakhmut falls', 'type': 'territorial', 'severity': 4},
    {'date': '2023-06-06', 'name': 'Kakhovka dam destroyed', 'type': 'infrastructure', 'severity': 5},
    {'date': '2023-06-08', 'name': 'Summer counteroffensive begins', 'type': 'military', 'severity': 4},
    {'date': '2024-02-17', 'name': 'Avdiivka falls', 'type': 'territorial', 'severity': 4},
    {'date': '2024-08-06', 'name': 'Kursk incursion begins', 'type': 'military', 'severity': 5},
]


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    checkpoint_path: str
    start_date: Optional[str] = None  # If None, use first available
    end_date: Optional[str] = None    # If None, use last available
    prediction_horizon: int = 7       # Days ahead to predict
    step_size: int = 1                # Days between predictions
    device: str = "mps"
    output_dir: str = "outputs/backtesting"
    include_spatial: bool = True      # Include spatial features (DeepState, FIRMS)


class BacktestEngine:
    """
    Engine for running backtests on the Multi-Resolution HAN model.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "mps" else "cpu")
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset config first (needed by model loading)
        self.dataset_config = MultiResolutionConfig(
            daily_seq_len=30,
            monthly_seq_len=12,
            prediction_horizon=config.prediction_horizon,
            use_disaggregated_equipment=True,
            detrend_viirs=True,
            include_spatial_features=config.include_spatial,
        )

        # Load model
        self.model, self.model_config = self._load_model()
        self.model.eval()

        # We need the full dataset for backtesting
        self.full_dataset = MultiResolutionDataset(
            config=self.dataset_config,
            split='train'  # Will manually handle date ranges
        )

        # Store date information
        self.start_date = self.full_dataset.start_date
        self.n_days = len(self.full_dataset.daily_dates) if hasattr(self.full_dataset, 'daily_dates') else 1426

    def _load_model(self) -> Tuple[MultiResolutionHAN, Dict]:
        """Load model from checkpoint."""
        print(f"Loading checkpoint: {self.config.checkpoint_path}")
        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        state_dict = checkpoint.get('model_state_dict', checkpoint)
        has_isw = any('isw_alignment' in k for k in state_dict.keys())

        # Get feature dimensions from a temporary dataset
        temp_dataset = MultiResolutionDataset(
            config=self.dataset_config, split='train'
        )
        feature_info = temp_dataset.get_feature_info()
        del temp_dataset  # Free memory

        daily_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='daily')
            for name, info in feature_info.items() if info['resolution'] == 'daily'
        }
        monthly_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='monthly')
            for name, info in feature_info.items() if info['resolution'] == 'monthly'
        }

        model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=128,
            nhead=8,
            num_daily_layers=3,
            num_monthly_layers=2,
            num_fusion_layers=2,
            dropout=0.0,
            use_isw_alignment=has_isw,
            isw_dim=1024,
        )

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {n_params:,} parameters, ISW={has_isw}")

        return model, {'has_isw': has_isw}

    def _sample_to_batch(self, sample) -> Dict[str, torch.Tensor]:
        """Convert a MultiResolutionSample to a batch dict for the model."""
        batch = {}

        batch['daily_features'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.daily_features.items()
        }

        batch['daily_masks'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.daily_masks.items()
        }

        batch['monthly_features'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.monthly_features.items()
        }

        batch['monthly_masks'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.monthly_masks.items()
        }

        batch['month_boundaries'] = sample.month_boundary_indices.unsqueeze(0).to(self.device)

        return batch

    def _idx_to_date(self, idx: int) -> datetime:
        """Convert dataset index to date."""
        return self.start_date + timedelta(days=idx)

    def _date_to_idx(self, date_str: str) -> int:
        """Convert date string to dataset index."""
        target = datetime.strptime(date_str, '%Y-%m-%d')
        delta = target - self.start_date
        return max(0, min(delta.days, self.n_days - 1))

    def _get_phase_for_date(self, date: datetime) -> Optional[Dict]:
        """Get conflict phase for a given date."""
        for phase in CONFLICT_PHASES:
            start = datetime.strptime(phase['start'], '%Y-%m-%d')
            end = datetime.strptime(phase['end'], '%Y-%m-%d')
            if start <= date <= end:
                return phase
        return None

    def run_rolling_forecast(self) -> Dict[str, Any]:
        """
        Run rolling forecast backtest with proper forecast comparison.
        """
        print("\n" + "=" * 70)
        print("ROLLING FORECAST BACKTEST")
        print("=" * 70)

        results = {
            'dates': [],
            'date_strings': [],
            'forecast_mse': [],
            'forecast_mae': [],
            'forecast_corr': [],
            'regime_predictions': [],
            'actual_phases': [],
            'casualty_predictions': [],
            'anomaly_scores': [],
        }

        n_samples = len(self.full_dataset)
        seq_len = self.dataset_config.daily_seq_len
        horizon = self.config.prediction_horizon

        start_idx = seq_len
        end_idx = n_samples - horizon

        print(f"Backtesting from index {start_idx} to {end_idx}")
        print(f"Date range: {self._idx_to_date(start_idx).strftime('%Y-%m-%d')} to {self._idx_to_date(end_idx).strftime('%Y-%m-%d')}")
        print(f"Total backtest points: {(end_idx - start_idx) // self.config.step_size}")

        with torch.no_grad():
            for idx in tqdm(range(start_idx, end_idx, self.config.step_size), desc="Rolling forecast"):
                sample = self.full_dataset[idx]
                batch = self._sample_to_batch(sample)

                try:
                    outputs = self.model(
                        daily_features=batch['daily_features'],
                        daily_masks=batch['daily_masks'],
                        monthly_features=batch['monthly_features'],
                        monthly_masks=batch['monthly_masks'],
                        month_boundaries=batch['month_boundaries'],
                    )
                except Exception as e:
                    print(f"Error at idx {idx}: {e}")
                    continue

                # Extract predictions
                # Note: Model outputs 'forecast_pred' (not 'forecast') at monthly resolution
                forecast_pred = outputs.get('forecast_pred')
                regime_logits = outputs.get('regime_logits')
                anomaly_score = outputs.get('anomaly_score')

                current_date = self._idx_to_date(idx)
                results['dates'].append(idx)
                results['date_strings'].append(current_date.strftime('%Y-%m-%d'))

                # Get actual conflict phase
                phase = self._get_phase_for_date(current_date)
                results['actual_phases'].append(phase['phase_id'] if phase else -1)

                # ===== FIXED: Forecast comparison using monthly features =====
                # forecast_pred shape: [batch, n_months, 35] where 35 = total monthly features
                if forecast_pred is not None and idx + horizon < n_samples:
                    future_sample = self.full_dataset[idx + horizon]

                    # Get predicted features (last month's prediction)
                    pred = forecast_pred[0, -1, :].cpu().numpy()

                    # Get actual future monthly features from the future sample
                    # Concatenate all monthly sources in the same order the model uses
                    actual_features = []
                    for source_name in ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']:
                        if source_name in future_sample.monthly_features:
                            # Get last month of future sample's monthly features
                            feat = future_sample.monthly_features[source_name][-1, :].numpy()
                            actual_features.append(feat)

                    if actual_features:
                        actual = np.concatenate(actual_features)

                        # Compare prediction to actual
                        min_len = min(len(pred), len(actual))
                        if min_len > 0:
                            pred_trunc = pred[:min_len]
                            actual_trunc = actual[:min_len]

                            mse = np.mean((pred_trunc - actual_trunc) ** 2)
                            mae = np.mean(np.abs(pred_trunc - actual_trunc))

                            # Correlation (with safety check)
                            if np.std(actual_trunc) > 1e-8 and np.std(pred_trunc) > 1e-8:
                                corr = np.corrcoef(pred_trunc, actual_trunc)[0, 1]
                                corr = corr if not np.isnan(corr) else 0.0
                            else:
                                corr = 0.0

                            results['forecast_mse'].append(mse)
                            results['forecast_mae'].append(mae)
                            results['forecast_corr'].append(corr)

                # Regime prediction
                if regime_logits is not None:
                    regime_pred = torch.argmax(regime_logits[0, -1, :], dim=-1).cpu().item()
                    results['regime_predictions'].append(regime_pred)
                else:
                    results['regime_predictions'].append(-1)

                # Anomaly score
                if anomaly_score is not None:
                    results['anomaly_scores'].append(anomaly_score[0, -1, :].cpu().mean().item())
                else:
                    results['anomaly_scores'].append(0.0)

        # Compute summary statistics
        summary = {
            'n_predictions': len(results['dates']),
            'forecast_mse_mean': np.mean(results['forecast_mse']) if results['forecast_mse'] else None,
            'forecast_mse_std': np.std(results['forecast_mse']) if results['forecast_mse'] else None,
            'forecast_mae_mean': np.mean(results['forecast_mae']) if results['forecast_mae'] else None,
            'forecast_corr_mean': np.mean(results['forecast_corr']) if results['forecast_corr'] else None,
            'forecast_corr_std': np.std(results['forecast_corr']) if results['forecast_corr'] else None,
        }

        print("\n" + "-" * 70)
        print("RESULTS SUMMARY")
        print("-" * 70)
        for key, value in summary.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # Generate visualizations
        self._plot_rolling_forecast_results(results, summary)
        self._plot_regime_vs_phase(results)

        return {'results': results, 'summary': summary}

    def run_daily_forecast(self) -> Dict[str, Any]:
        """
        Run daily-resolution forecast backtest.

        Evaluates the model's ability to predict daily features (equipment losses,
        personnel, fire hotspots, territorial changes) at daily granularity.

        Uses temporal_output from the model and compares against future daily features.
        """
        print("\n" + "=" * 70)
        print("DAILY RESOLUTION FORECAST BACKTEST")
        print("=" * 70)

        # Get list of daily sources from dataset
        feature_info = self.full_dataset.get_feature_info()
        daily_sources = [name for name, info in feature_info.items()
                        if info['resolution'] == 'daily']

        print(f"Daily sources to evaluate: {daily_sources}")

        results = {
            'dates': [],
            'date_strings': [],
            'daily_mse_by_source': {src: [] for src in daily_sources},
            'daily_mae_by_source': {src: [] for src in daily_sources},
            'daily_corr_by_source': {src: [] for src in daily_sources},
            'aggregate_daily_mse': [],
            'aggregate_daily_mae': [],
        }

        n_samples = len(self.full_dataset)
        seq_len = self.dataset_config.daily_seq_len
        horizon = self.config.prediction_horizon

        start_idx = seq_len
        end_idx = n_samples - horizon

        print(f"Backtesting from index {start_idx} to {end_idx}")
        print(f"Date range: {self._idx_to_date(start_idx).strftime('%Y-%m-%d')} to "
              f"{self._idx_to_date(end_idx).strftime('%Y-%m-%d')}")
        print(f"Total backtest points: {(end_idx - start_idx) // self.config.step_size}")

        with torch.no_grad():
            for idx in tqdm(range(start_idx, end_idx, self.config.step_size),
                           desc="Daily forecast"):
                sample = self.full_dataset[idx]
                batch = self._sample_to_batch(sample)

                try:
                    outputs = self.model(
                        daily_features=batch['daily_features'],
                        daily_masks=batch['daily_masks'],
                        monthly_features=batch['monthly_features'],
                        monthly_masks=batch['monthly_masks'],
                        month_boundaries=batch['month_boundaries'],
                    )
                except Exception as e:
                    print(f"Error at idx {idx}: {e}")
                    continue

                current_date = self._idx_to_date(idx)
                results['dates'].append(idx)
                results['date_strings'].append(current_date.strftime('%Y-%m-%d'))

                # Check if model has daily forecast head
                daily_forecast = outputs.get('daily_forecast_pred')

                # Get future sample for comparison
                if idx + horizon < n_samples:
                    future_sample = self.full_dataset[idx + horizon]

                    # Compare each daily source
                    all_pred = []
                    all_actual = []

                    for source_name in daily_sources:
                        if source_name not in future_sample.daily_features:
                            continue

                        # Get actual future daily features
                        # Use the last day of the future sample's daily features
                        actual_feat = future_sample.daily_features[source_name]
                        if actual_feat.ndim == 2:
                            # Get last day's features
                            actual = actual_feat[-1, :].numpy()
                        else:
                            actual = actual_feat.numpy()

                        # Get prediction for this source
                        if daily_forecast is not None:
                            # Use daily forecast head output
                            # daily_forecast: [batch, horizon, total_daily_features]
                            # Need to extract the right slice for this source
                            pred = self._extract_source_prediction(
                                daily_forecast[0, -1, :].cpu().numpy(),
                                source_name,
                                daily_sources,
                                feature_info
                            )
                        else:
                            # Use last day of input features as naive forecast
                            pred = sample.daily_features[source_name][-1, :].numpy()

                        # Compute metrics
                        min_len = min(len(pred), len(actual))
                        if min_len > 0:
                            pred_trunc = pred[:min_len]
                            actual_trunc = actual[:min_len]

                            # Skip if all values are missing (-999)
                            valid_mask = (actual_trunc > -900) & (pred_trunc > -900)
                            if valid_mask.sum() > 0:
                                pred_valid = pred_trunc[valid_mask]
                                actual_valid = actual_trunc[valid_mask]

                                mse = np.mean((pred_valid - actual_valid) ** 2)
                                mae = np.mean(np.abs(pred_valid - actual_valid))

                                results['daily_mse_by_source'][source_name].append(mse)
                                results['daily_mae_by_source'][source_name].append(mae)

                                # Correlation
                                if len(pred_valid) > 1:
                                    std_a = np.std(actual_valid)
                                    std_p = np.std(pred_valid)
                                    if std_a > 1e-8 and std_p > 1e-8:
                                        corr = np.corrcoef(pred_valid, actual_valid)[0, 1]
                                        if not np.isnan(corr):
                                            results['daily_corr_by_source'][source_name].append(corr)

                                all_pred.extend(pred_valid.tolist())
                                all_actual.extend(actual_valid.tolist())

                    # Aggregate metrics across all sources
                    if all_pred and all_actual:
                        all_pred = np.array(all_pred)
                        all_actual = np.array(all_actual)
                        results['aggregate_daily_mse'].append(
                            np.mean((all_pred - all_actual) ** 2)
                        )
                        results['aggregate_daily_mae'].append(
                            np.mean(np.abs(all_pred - all_actual))
                        )

        # Compute summary statistics
        summary = {
            'n_predictions': len(results['dates']),
            'aggregate_mse_mean': np.mean(results['aggregate_daily_mse'])
                if results['aggregate_daily_mse'] else None,
            'aggregate_mae_mean': np.mean(results['aggregate_daily_mae'])
                if results['aggregate_daily_mae'] else None,
        }

        # Per-source summaries
        for source_name in daily_sources:
            mse_vals = results['daily_mse_by_source'][source_name]
            mae_vals = results['daily_mae_by_source'][source_name]
            corr_vals = results['daily_corr_by_source'][source_name]

            if mse_vals:
                summary[f'{source_name}_mse_mean'] = np.mean(mse_vals)
                summary[f'{source_name}_mae_mean'] = np.mean(mae_vals)
            if corr_vals:
                summary[f'{source_name}_corr_mean'] = np.mean(corr_vals)

        print("\n" + "-" * 70)
        print("DAILY FORECAST RESULTS SUMMARY")
        print("-" * 70)
        for key, value in summary.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # Generate visualizations
        self._plot_daily_forecast_results(results, summary, daily_sources)

        return {'results': results, 'summary': summary}

    def _extract_source_prediction(
        self,
        flat_pred: np.ndarray,
        source_name: str,
        daily_sources: List[str],
        feature_info: Dict,
    ) -> np.ndarray:
        """Extract source-specific prediction from concatenated daily forecast."""
        # Compute offset for this source
        offset = 0
        for src in daily_sources:
            if src == source_name:
                break
            offset += feature_info[src]['n_features']

        n_features = feature_info[source_name]['n_features']
        return flat_pred[offset:offset + n_features]

    def _plot_daily_forecast_results(
        self,
        results: Dict,
        summary: Dict,
        daily_sources: List[str],
    ):
        """Generate visualizations for daily forecast results."""
        n_sources = len(daily_sources)
        n_cols = min(3, n_sources)
        n_rows = (n_sources + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle('Daily Resolution Forecast Results by Source', fontsize=14, fontweight='bold')

        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for i, source_name in enumerate(daily_sources):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[0][col]

            mse_vals = results['daily_mse_by_source'][source_name]
            if mse_vals:
                ax.plot(range(len(mse_vals)), mse_vals, alpha=0.7, linewidth=0.8)
                mean_mse = np.mean(mse_vals)
                ax.axhline(y=mean_mse, color='r', linestyle='--',
                          label=f'Mean: {mean_mse:.3f}')
                ax.set_xlabel('Prediction Index')
                ax.set_ylabel('MSE')
                ax.set_title(f'{source_name}')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{source_name}')

        # Hide unused subplots
        for i in range(len(daily_sources), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[0][col]
            ax.set_visible(False)

        plt.tight_layout()
        save_path = self.output_dir / 'daily_forecast_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.close()

    def run_event_detection_backtest(self) -> Dict[str, Any]:
        """
        Backtest event detection against known major events.
        """
        print("\n" + "=" * 70)
        print("EVENT DETECTION BACKTEST")
        print("=" * 70)

        # First, collect anomaly scores across the full dataset
        print("\nCollecting anomaly scores across timeline...")

        all_scores = []
        all_dates = []

        with torch.no_grad():
            for idx in tqdm(range(30, len(self.full_dataset)), desc="Scanning timeline"):
                sample = self.full_dataset[idx]
                batch = self._sample_to_batch(sample)

                try:
                    outputs = self.model(
                        daily_features=batch['daily_features'],
                        daily_masks=batch['daily_masks'],
                        monthly_features=batch['monthly_features'],
                        monthly_masks=batch['monthly_masks'],
                        month_boundaries=batch['month_boundaries'],
                    )

                    if 'anomaly_score' in outputs:
                        score = outputs['anomaly_score'][0, -1, :].cpu().mean().item()
                        all_scores.append(score)
                        all_dates.append(self._idx_to_date(idx))

                except Exception as e:
                    continue

        if not all_scores:
            print("No anomaly scores collected")
            return {'status': 'no_scores'}

        # Convert to numpy for analysis
        scores = np.array(all_scores)
        dates = all_dates

        # Compute threshold (mean + 2*std)
        threshold = np.mean(scores) + 2 * np.std(scores)

        print(f"\nAnomaly score statistics:")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Threshold (mean + 2*std): {threshold:.4f}")

        # Check each major event
        print("\n" + "-" * 70)
        print("EVENT DETECTION RESULTS")
        print("-" * 70)

        event_results = []
        window_days = 7  # Look for anomaly within 7 days of event

        for event in MAJOR_EVENTS:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d')

            # Find anomaly scores around this event
            nearby_scores = []
            for i, date in enumerate(dates):
                days_diff = abs((date - event_date).days)
                if days_diff <= window_days:
                    nearby_scores.append((days_diff, scores[i], date))

            if nearby_scores:
                # Get max anomaly score in window
                max_score = max(s[1] for s in nearby_scores)
                detected = max_score > threshold

                # Find when max occurred relative to event
                max_entry = max(nearby_scores, key=lambda x: x[1])
                lead_time = (event_date - max_entry[2]).days  # Positive = detected before event

                result = {
                    'event': event['name'],
                    'date': event['date'],
                    'severity': event['severity'],
                    'max_anomaly': max_score,
                    'detected': detected,
                    'lead_time': lead_time if detected else None,
                }
                event_results.append(result)

                status = "✓ DETECTED" if detected else "✗ MISSED"
                lead_str = f" (lead time: {lead_time:+d} days)" if detected and lead_time != 0 else ""
                print(f"  {event['date']} {event['name'][:30]:<30} {status}{lead_str}")
            else:
                event_results.append({
                    'event': event['name'],
                    'date': event['date'],
                    'severity': event['severity'],
                    'max_anomaly': None,
                    'detected': False,
                    'lead_time': None,
                })
                print(f"  {event['date']} {event['name'][:30]:<30} (no data)")

        # Compute detection metrics
        detected_events = [e for e in event_results if e['detected']]
        n_detected = len(detected_events)
        n_total = len([e for e in event_results if e['max_anomaly'] is not None])

        detection_rate = n_detected / n_total if n_total > 0 else 0
        avg_lead_time = np.mean([e['lead_time'] for e in detected_events if e['lead_time'] is not None]) if detected_events else 0

        print(f"\n  Detection rate: {n_detected}/{n_total} ({100*detection_rate:.1f}%)")
        print(f"  Average lead time: {avg_lead_time:.1f} days")

        # Generate visualization
        self._plot_event_detection(dates, scores, threshold, event_results)

        return {
            'scores': all_scores,
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'threshold': threshold,
            'events': event_results,
            'detection_rate': detection_rate,
            'avg_lead_time': avg_lead_time,
        }

    def _plot_rolling_forecast_results(self, results: Dict, summary: Dict):
        """Generate visualizations for rolling forecast results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Rolling Forecast Backtest Results', fontsize=14, fontweight='bold')

        # MSE over time
        ax = axes[0, 0]
        if results['forecast_mse']:
            dates_subset = results['date_strings'][:len(results['forecast_mse'])]
            ax.plot(range(len(results['forecast_mse'])), results['forecast_mse'], alpha=0.7, color='blue')
            ax.axhline(y=summary['forecast_mse_mean'], color='r', linestyle='--',
                      label=f"Mean: {summary['forecast_mse_mean']:.4f}")
            ax.set_xlabel('Prediction Index')
            ax.set_ylabel('Forecast MSE')
            ax.set_title('Forecast MSE Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No forecast data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Forecast MSE Over Time')

        # Correlation over time
        ax = axes[0, 1]
        if results['forecast_corr']:
            ax.plot(range(len(results['forecast_corr'])), results['forecast_corr'],
                   alpha=0.7, color='green')
            ax.axhline(y=summary['forecast_corr_mean'], color='r', linestyle='--',
                      label=f"Mean: {summary['forecast_corr_mean']:.4f}")
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Prediction Index')
            ax.set_ylabel('Correlation')
            ax.set_title('Forecast-Actual Correlation Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No forecast data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Forecast-Actual Correlation')

        # Anomaly scores
        ax = axes[1, 0]
        if results['anomaly_scores']:
            ax.plot(results['dates'][:len(results['anomaly_scores'])],
                   results['anomaly_scores'], alpha=0.7, color='orange')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Scores Over Time')
            ax.grid(True, alpha=0.3)

        # Regime predictions
        ax = axes[1, 1]
        if results['regime_predictions']:
            ax.scatter(results['dates'][:len(results['regime_predictions'])],
                      results['regime_predictions'], alpha=0.5, s=10, label='Predicted')
            ax.scatter(results['dates'][:len(results['actual_phases'])],
                      results['actual_phases'], alpha=0.5, s=10, color='red', label='Actual Phase')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Regime/Phase')
            ax.set_title('Regime Predictions vs Actual Phases')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'rolling_forecast_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.close()

    def _plot_regime_vs_phase(self, results: Dict):
        """Plot detailed regime predictions vs actual conflict phases."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Regime Predictions vs Conflict Phases', fontsize=14, fontweight='bold')

        # Convert dates to datetime for plotting
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in results['date_strings']]

        # Top plot: Timeline view
        ax = axes[0]

        # Plot actual phases as background colors
        phase_colors = plt.cm.Set3(np.linspace(0, 1, len(CONFLICT_PHASES)))
        for i, phase in enumerate(CONFLICT_PHASES):
            start = datetime.strptime(phase['start'], '%Y-%m-%d')
            end = datetime.strptime(phase['end'], '%Y-%m-%d')
            ax.axvspan(start, end, alpha=0.3, color=phase_colors[i], label=phase['name'])

        # Plot predicted regimes
        if results['regime_predictions']:
            regime_preds = results['regime_predictions'][:len(dates)]
            ax.scatter(dates[:len(regime_preds)], regime_preds,
                      c='blue', s=20, alpha=0.6, label='Predicted Regime')

        ax.set_xlabel('Date')
        ax.set_ylabel('Regime')
        ax.set_title('Predicted Regime Over Conflict Timeline')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend for phases
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(CONFLICT_PHASES)+1], labels[:len(CONFLICT_PHASES)+1],
                 loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Bottom plot: Confusion matrix style comparison
        ax = axes[1]

        if results['regime_predictions'] and results['actual_phases']:
            # Create comparison data
            pred = np.array(results['regime_predictions'])
            actual = np.array(results['actual_phases'])

            # Filter out invalid entries
            valid = (actual >= 0) & (pred >= 0)
            pred = pred[valid]
            actual = actual[valid]

            if len(pred) > 0:
                # Count regime predictions per actual phase
                phase_names = [p['name'][:15] for p in CONFLICT_PHASES]
                unique_regimes = sorted(set(pred))

                data = np.zeros((len(CONFLICT_PHASES), len(unique_regimes)))
                for p, a in zip(pred, actual):
                    if a < len(CONFLICT_PHASES) and p in unique_regimes:
                        regime_idx = unique_regimes.index(p)
                        data[a, regime_idx] += 1

                # Normalize per phase
                row_sums = data.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                data_norm = data / row_sums

                im = ax.imshow(data_norm, aspect='auto', cmap='Blues')
                ax.set_xticks(range(len(unique_regimes)))
                ax.set_xticklabels([f'Regime {r}' for r in unique_regimes])
                ax.set_yticks(range(len(phase_names)))
                ax.set_yticklabels(phase_names)
                ax.set_xlabel('Predicted Regime')
                ax.set_ylabel('Actual Conflict Phase')
                ax.set_title('Regime Prediction Distribution by Conflict Phase')

                # Add text annotations
                for i in range(len(phase_names)):
                    for j in range(len(unique_regimes)):
                        text = f'{data_norm[i, j]:.2f}'
                        color = 'white' if data_norm[i, j] > 0.5 else 'black'
                        ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

                plt.colorbar(im, ax=ax, label='Proportion')

        plt.tight_layout()
        save_path = self.output_dir / 'regime_vs_phase.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def _plot_event_detection(self, dates: List[datetime], scores: np.ndarray,
                             threshold: float, event_results: List[Dict]):
        """Plot anomaly scores with major events marked."""
        fig, ax = plt.subplots(figsize=(16, 6))

        # Plot anomaly scores
        ax.plot(dates, scores, alpha=0.7, color='blue', linewidth=0.8, label='Anomaly Score')
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold:.3f})')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        # Mark major events
        for event in event_results:
            if event['max_anomaly'] is not None:
                event_date = datetime.strptime(event['date'], '%Y-%m-%d')
                color = 'green' if event['detected'] else 'red'
                ax.axvline(x=event_date, color=color, alpha=0.5, linewidth=1.5)

                # Add event label
                y_pos = threshold + 0.1 * (max(scores) - threshold)
                ax.annotate(event['event'][:20], xy=(event_date, y_pos),
                           rotation=90, fontsize=7, alpha=0.8,
                           va='bottom', ha='right')

        ax.set_xlabel('Date')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Timeline with Major Events')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', alpha=0.7),
            Line2D([0], [0], color='red', linestyle='--'),
            Line2D([0], [0], color='green', alpha=0.5, linewidth=2),
            Line2D([0], [0], color='red', alpha=0.5, linewidth=2),
        ]
        ax.legend(custom_lines, ['Anomaly Score', 'Threshold', 'Detected Event', 'Missed Event'],
                 loc='upper right')

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.output_dir / 'event_detection.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Backtest Multi-Resolution HAN model')
    parser.add_argument('--checkpoint', type=str,
                       default='analysis/checkpoints/multi_resolution/best_checkpoint.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str,
                       choices=['rolling', 'events', 'daily', 'all'],
                       default='all', help='Backtesting mode (daily = daily-resolution forecasts)')
    parser.add_argument('--horizon', type=int, default=7,
                       help='Prediction horizon in days')
    parser.add_argument('--step', type=int, default=7,
                       help='Step size between predictions')
    parser.add_argument('--device', type=str, default='mps',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='outputs/backtesting',
                       help='Output directory for results')
    parser.add_argument('--include-spatial', action='store_true', default=True,
                       help='Include spatial features (default: True for spatial-trained models)')
    parser.add_argument('--no-spatial', action='store_true', default=False,
                       help='Disable spatial features (for older checkpoints)')

    args = parser.parse_args()

    # Handle spatial flag
    include_spatial = args.include_spatial and not args.no_spatial

    config = BacktestConfig(
        checkpoint_path=args.checkpoint,
        prediction_horizon=args.horizon,
        step_size=args.step,
        device=args.device,
        output_dir=args.output_dir,
        include_spatial=include_spatial,
    )

    engine = BacktestEngine(config)

    results = {}

    if args.mode in ['rolling', 'all']:
        results['rolling'] = engine.run_rolling_forecast()

    if args.mode in ['daily', 'all']:
        results['daily'] = engine.run_daily_forecast()

    if args.mode in ['events', 'all']:
        results['events'] = engine.run_event_detection_backtest()

    # Save results
    output_path = Path(args.output_dir) / 'backtest_results.json'

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print("\nBacktesting complete!")


if __name__ == '__main__':
    main()
