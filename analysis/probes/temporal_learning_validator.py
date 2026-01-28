"""
Temporal Learning Validator (C1)

This module validates whether the Multi-Resolution HAN model learns meaningful
temporal structure or treats input sequences as bags of features.

Experiments:
1. Reverse Sequence Test - Compare predictions for chronological vs reverse order
2. Synthetic Trend Injection - Test model response to linear trends
3. Temporal Ablation Gradient - Test prediction change vs context length
4. Positional Encoding Analysis - Examine sinusoidal encoding weights
5. Gradient Flow Through Temporal Layers - Check gradient magnitudes

Author: Validation Pipeline
Date: 2026-01-25
"""

import sys
from pathlib import Path

# Add analysis directory to path
ANALYSIS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(ANALYSIS_DIR.parent))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from config.paths import PROJECT_ROOT, OUTPUT_DIR

# Import model and data infrastructure
from multi_resolution_han import (
    MultiResolutionHAN,
    SourceConfig,
    create_multi_resolution_han,
)
from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    create_multi_resolution_dataloaders,
    multi_resolution_collate_fn,
)
from torch.utils.data import DataLoader


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    name: str
    verdict: str  # "CONFIRMS_C1", "REFUTES_C1", "INCONCLUSIVE"
    metrics: Dict[str, float]
    description: str
    confidence: float  # 0-1


@dataclass
class ValidationReport:
    """Complete validation report."""
    experiment_results: List[ExperimentResult]
    final_verdict: str
    confidence_level: str
    summary: str


class TemporalLearningValidator:
    """
    Validates temporal learning in Multi-Resolution HAN models.

    Tests Claim C1: "The model ignores temporal structure and treats
    input sequences as bags of features."
    """

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        # Load model
        self.model = None
        self.model_config = None
        self._load_model()

        # Load sample data for experiments
        self.sample_batch = None
        self._load_sample_data()

        # Results storage
        self.results: List[ExperimentResult] = []

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Extract model config from state dict (most reliable)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model_config = self._infer_config_from_state_dict(state_dict)

        # Create model
        self.model = self._create_model_from_config(self.model_config)

        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully. d_model={self.model.d_model}")

    def _infer_config_from_state_dict(self, state_dict: Dict) -> Dict:
        """Infer model configuration from state dictionary keys."""
        # Default configuration based on architecture analysis
        daily_sources = []
        monthly_sources = []

        for key in state_dict.keys():
            if 'daily_encoders.' in key:
                parts = key.split('.')
                source_name = parts[1]
                if source_name not in daily_sources:
                    daily_sources.append(source_name)
            elif 'monthly_encoder.' in key and 'source_encoders.' in key:
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'source_encoders' and i + 1 < len(parts):
                        source_name = parts[i + 1]
                        if source_name not in monthly_sources:
                            monthly_sources.append(source_name)

        # Get d_model from a known layer
        d_model = 128  # default
        for key, value in state_dict.items():
            if 'output_norm.weight' in key:
                d_model = value.shape[0]
                break

        # Infer feature counts from input projections
        daily_features = {}
        for source in daily_sources:
            key = f'daily_encoders.{source}.feature_projection.0.weight'
            if key in state_dict:
                daily_features[source] = state_dict[key].shape[1]
            else:
                daily_features[source] = 12  # default

        monthly_features = {}
        for source in monthly_sources:
            # Try to infer from feature_embedding which has shape [n_features, d_model]
            feature_emb_key = f'monthly_encoder.source_encoders.{source}.feature_embedding.weight'
            if feature_emb_key in state_dict:
                monthly_features[source] = state_dict[feature_emb_key].shape[0]
            else:
                # Try different key patterns
                found = False
                for pattern in [
                    f'monthly_encoder.source_encoders.{source}.input_projection.0.weight',
                    f'monthly_encoder.source_encoders.{source}.feature_projection.0.weight',
                ]:
                    if pattern in state_dict:
                        monthly_features[source] = state_dict[pattern].shape[1]
                        found = True
                        break
                if not found:
                    monthly_features[source] = 10  # default

        return {
            'daily_source_configs': {
                name: {'name': name, 'n_features': n_feat, 'resolution': 'daily'}
                for name, n_feat in daily_features.items()
            },
            'monthly_source_configs': {
                name: {'name': name, 'n_features': n_feat, 'resolution': 'monthly'}
                for name, n_feat in monthly_features.items()
            },
            'd_model': d_model,
            'nhead': 8,
            'num_daily_layers': 4,
            'num_monthly_layers': 3,
            'num_fusion_layers': 2,
            'num_temporal_layers': 2,
            'dropout': 0.1,
        }

    def _create_model_from_config(self, config: Dict) -> MultiResolutionHAN:
        """Create model from configuration dictionary."""
        daily_configs = {
            name: SourceConfig(
                name=cfg['name'],
                n_features=cfg['n_features'],
                resolution=cfg.get('resolution', 'daily'),
            )
            for name, cfg in config['daily_source_configs'].items()
        }

        monthly_configs = {
            name: SourceConfig(
                name=cfg['name'],
                n_features=cfg['n_features'],
                resolution=cfg.get('resolution', 'monthly'),
            )
            for name, cfg in config['monthly_source_configs'].items()
        }

        return MultiResolutionHAN(
            daily_source_configs=daily_configs,
            monthly_source_configs=monthly_configs,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 8),
            num_daily_layers=config.get('num_daily_layers', 4),
            num_monthly_layers=config.get('num_monthly_layers', 3),
            num_fusion_layers=config.get('num_fusion_layers', 2),
            num_temporal_layers=config.get('num_temporal_layers', 2),
            dropout=config.get('dropout', 0.1),
        )

    def _load_sample_data(self) -> None:
        """Load sample data for experiments."""
        print("Loading sample data...")

        try:
            # Try to use the actual data loader
            data_config = MultiResolutionConfig(
                daily_seq_len=365,
                monthly_seq_len=12,
            )

            dataset = MultiResolutionDataset(
                config=data_config,
                mode='train',
            )

            if len(dataset) > 0:
                # Get a sample batch
                dataloader = DataLoader(
                    dataset,
                    batch_size=4,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=multi_resolution_collate_fn,
                )
                self.sample_batch = next(iter(dataloader))
                print(f"Loaded real data batch with {len(dataset)} samples")
                return
        except Exception as e:
            print(f"Could not load real data: {e}")

        # Create synthetic data matching model configuration
        print("Creating synthetic data for experiments...")
        self.sample_batch = self._create_synthetic_batch()

    def _create_synthetic_batch(
        self,
        batch_size: int = 4,
        daily_seq_len: int = 365,
        monthly_seq_len: int = 12,
    ) -> Dict[str, Any]:
        """Create synthetic batch matching model configuration."""
        # Get source configurations
        daily_configs = self.model_config['daily_source_configs']
        monthly_configs = self.model_config['monthly_source_configs']

        # Create daily features and masks
        daily_features = {}
        daily_masks = {}
        for name, cfg in daily_configs.items():
            n_features = cfg['n_features']
            # Create realistic-looking time series with trends
            t = torch.linspace(0, 1, daily_seq_len).unsqueeze(0).unsqueeze(-1)
            trend = t.expand(batch_size, -1, n_features)
            noise = torch.randn(batch_size, daily_seq_len, n_features) * 0.1
            daily_features[name] = trend + noise
            # Random observation mask (80% observed)
            daily_masks[name] = torch.rand(batch_size, daily_seq_len, n_features) > 0.2

        # Create monthly features and masks
        monthly_features = {}
        monthly_masks = {}
        for name, cfg in monthly_configs.items():
            n_features = cfg['n_features']
            t = torch.linspace(0, 1, monthly_seq_len).unsqueeze(0).unsqueeze(-1)
            trend = t.expand(batch_size, -1, n_features)
            noise = torch.randn(batch_size, monthly_seq_len, n_features) * 0.1
            monthly_features[name] = trend + noise
            monthly_masks[name] = torch.rand(batch_size, monthly_seq_len, n_features) > 0.2

        # Create month boundaries
        days_per_month = daily_seq_len // monthly_seq_len
        month_boundaries = torch.zeros(batch_size, monthly_seq_len, 2, dtype=torch.long)
        for m in range(monthly_seq_len):
            start = m * days_per_month
            end = min((m + 1) * days_per_month, daily_seq_len)
            month_boundaries[:, m, 0] = start
            month_boundaries[:, m, 1] = end

        return {
            'daily_features': daily_features,
            'daily_masks': daily_masks,
            'monthly_features': monthly_features,
            'monthly_masks': monthly_masks,
            'month_boundaries': month_boundaries,
        }

    def _get_predictions(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Get model predictions for a batch."""
        with torch.no_grad():
            # Move batch to device
            daily_features = {k: v.to(self.device) for k, v in batch['daily_features'].items()}
            daily_masks = {k: v.to(self.device) for k, v in batch['daily_masks'].items()}
            monthly_features = {k: v.to(self.device) for k, v in batch['monthly_features'].items()}
            monthly_masks = {k: v.to(self.device) for k, v in batch['monthly_masks'].items()}
            month_boundaries = batch['month_boundaries'].to(self.device)

            outputs = self.model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

            return outputs

    def experiment_1_reverse_sequence(self) -> ExperimentResult:
        """
        Experiment 1: Reverse Sequence Test

        Feed sequences in reverse chronological order and compare predictions.
        If predictions are identical, C1 is confirmed (model ignores order).
        """
        print("\n" + "="*60)
        print("Experiment 1: Reverse Sequence Test")
        print("="*60)

        batch = self.sample_batch

        # Get predictions for original order
        pred_original = self._get_predictions(batch)

        # Create reversed batch
        reversed_daily_features = {
            k: torch.flip(v, dims=[1]) for k, v in batch['daily_features'].items()
        }
        reversed_daily_masks = {
            k: torch.flip(v, dims=[1]) for k, v in batch['daily_masks'].items()
        }
        reversed_monthly_features = {
            k: torch.flip(v, dims=[1]) for k, v in batch['monthly_features'].items()
        }
        reversed_monthly_masks = {
            k: torch.flip(v, dims=[1]) for k, v in batch['monthly_masks'].items()
        }

        # Reverse month boundaries
        daily_seq_len = list(batch['daily_features'].values())[0].shape[1]
        reversed_boundaries = batch['month_boundaries'].clone()
        reversed_boundaries[:, :, 0] = daily_seq_len - batch['month_boundaries'][:, :, 1]
        reversed_boundaries[:, :, 1] = daily_seq_len - batch['month_boundaries'][:, :, 0]
        reversed_boundaries = torch.flip(reversed_boundaries, dims=[1])

        reversed_batch = {
            'daily_features': reversed_daily_features,
            'daily_masks': reversed_daily_masks,
            'monthly_features': reversed_monthly_features,
            'monthly_masks': reversed_monthly_masks,
            'month_boundaries': reversed_boundaries,
        }

        # Get predictions for reversed order
        pred_reversed = self._get_predictions(reversed_batch)

        # Compare predictions
        metrics = {}

        if 'casualty_pred' in pred_original and 'casualty_pred' in pred_reversed:
            orig = pred_original['casualty_pred'].cpu()
            rev = torch.flip(pred_reversed['casualty_pred'].cpu(), dims=[1])

            # Mean absolute difference
            mae = torch.abs(orig - rev).mean().item()
            metrics['casualty_mae'] = mae

            # Correlation
            orig_flat = orig.flatten()
            rev_flat = rev.flatten()
            corr = torch.corrcoef(torch.stack([orig_flat, rev_flat]))[0, 1].item()
            metrics['casualty_correlation'] = corr

            print(f"  Casualty MAE (orig vs reversed): {mae:.6f}")
            print(f"  Casualty Correlation: {corr:.4f}")

        if 'temporal_output' in pred_original:
            orig = pred_original['temporal_output'].cpu()
            rev = torch.flip(pred_reversed['temporal_output'].cpu(), dims=[1])

            mae = torch.abs(orig - rev).mean().item()
            metrics['latent_mae'] = mae

            cosine_sim = nn.functional.cosine_similarity(
                orig.flatten(1), rev.flatten(1), dim=1
            ).mean().item()
            metrics['latent_cosine_sim'] = cosine_sim

            print(f"  Latent MAE: {mae:.6f}")
            print(f"  Latent Cosine Similarity: {cosine_sim:.4f}")

        # Verdict based on similarity threshold
        # High correlation (>0.95) or low MAE suggests order doesn't matter
        if metrics.get('casualty_correlation', 0) > 0.95:
            verdict = "CONFIRMS_C1"
            confidence = 0.9
            description = "Predictions nearly identical regardless of temporal order"
        elif metrics.get('casualty_correlation', 0) > 0.80:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            description = "Moderate similarity suggests partial temporal learning"
        else:
            verdict = "REFUTES_C1"
            confidence = 0.8
            description = "Significant prediction changes with reversed order"

        result = ExperimentResult(
            name="Reverse Sequence Test",
            verdict=verdict,
            metrics=metrics,
            description=description,
            confidence=confidence,
        )

        self.results.append(result)
        print(f"\n  Verdict: {verdict} (confidence: {confidence:.2f})")

        # Save visualization
        self._plot_reverse_comparison(pred_original, pred_reversed)

        return result

    def experiment_2_synthetic_trend(self) -> ExperimentResult:
        """
        Experiment 2: Synthetic Trend Injection

        Add obvious linear trends to input features and test if model responds.
        If no response to trend direction, C1 is confirmed.
        """
        print("\n" + "="*60)
        print("Experiment 2: Synthetic Trend Injection")
        print("="*60)

        batch = self.sample_batch

        # Get baseline predictions
        pred_baseline = self._get_predictions(batch)

        # Create strong upward trend
        daily_seq_len = list(batch['daily_features'].values())[0].shape[1]
        monthly_seq_len = list(batch['monthly_features'].values())[0].shape[1]

        trend_up_daily = torch.linspace(0, 5, daily_seq_len).unsqueeze(0).unsqueeze(-1)
        trend_up_monthly = torch.linspace(0, 5, monthly_seq_len).unsqueeze(0).unsqueeze(-1)

        uptrend_daily = {
            k: v + trend_up_daily.expand_as(v)
            for k, v in batch['daily_features'].items()
        }
        uptrend_monthly = {
            k: v + trend_up_monthly.expand_as(v)
            for k, v in batch['monthly_features'].items()
        }

        uptrend_batch = {
            'daily_features': uptrend_daily,
            'daily_masks': batch['daily_masks'],
            'monthly_features': uptrend_monthly,
            'monthly_masks': batch['monthly_masks'],
            'month_boundaries': batch['month_boundaries'],
        }

        pred_uptrend = self._get_predictions(uptrend_batch)

        # Create strong downward trend
        trend_down_daily = torch.linspace(5, 0, daily_seq_len).unsqueeze(0).unsqueeze(-1)
        trend_down_monthly = torch.linspace(5, 0, monthly_seq_len).unsqueeze(0).unsqueeze(-1)

        downtrend_daily = {
            k: v + trend_down_daily.expand_as(v)
            for k, v in batch['daily_features'].items()
        }
        downtrend_monthly = {
            k: v + trend_down_monthly.expand_as(v)
            for k, v in batch['monthly_features'].items()
        }

        downtrend_batch = {
            'daily_features': downtrend_daily,
            'daily_masks': batch['daily_masks'],
            'monthly_features': downtrend_monthly,
            'monthly_masks': batch['monthly_masks'],
            'month_boundaries': batch['month_boundaries'],
        }

        pred_downtrend = self._get_predictions(downtrend_batch)

        # Analyze response to trends
        metrics = {}

        if 'casualty_pred' in pred_baseline:
            base = pred_baseline['casualty_pred'].cpu()
            up = pred_uptrend['casualty_pred'].cpu()
            down = pred_downtrend['casualty_pred'].cpu()

            # Check if predictions follow trends
            # Compute trend in predictions
            def compute_trend(pred):
                """Compute linear trend coefficient."""
                seq_len = pred.shape[1]
                t = torch.arange(seq_len, dtype=torch.float)
                pred_mean = pred.mean(dim=(0, 2))  # Average across batch and features
                return torch.corrcoef(torch.stack([t, pred_mean]))[0, 1].item()

            base_trend = compute_trend(base)
            up_trend = compute_trend(up)
            down_trend = compute_trend(down)

            metrics['baseline_trend'] = base_trend
            metrics['uptrend_response'] = up_trend
            metrics['downtrend_response'] = down_trend
            metrics['trend_difference'] = up_trend - down_trend

            print(f"  Baseline prediction trend: {base_trend:.4f}")
            print(f"  Uptrend injection response: {up_trend:.4f}")
            print(f"  Downtrend injection response: {down_trend:.4f}")
            print(f"  Trend difference (up - down): {up_trend - down_trend:.4f}")

            # Also check magnitude changes
            up_mean = up.mean().item()
            down_mean = down.mean().item()
            base_mean = base.mean().item()

            metrics['up_mean_shift'] = up_mean - base_mean
            metrics['down_mean_shift'] = down_mean - base_mean

            print(f"  Up trend mean shift: {up_mean - base_mean:.4f}")
            print(f"  Down trend mean shift: {down_mean - base_mean:.4f}")

        # Verdict: if trend_difference is near zero, model ignores trend direction
        trend_diff = abs(metrics.get('trend_difference', 0))

        if trend_diff < 0.1:
            verdict = "CONFIRMS_C1"
            confidence = 0.85
            description = "Model shows no differential response to trend direction"
        elif trend_diff < 0.3:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            description = "Weak response to trend direction"
        else:
            verdict = "REFUTES_C1"
            confidence = 0.8
            description = "Model responds meaningfully to trend direction"

        result = ExperimentResult(
            name="Synthetic Trend Injection",
            verdict=verdict,
            metrics=metrics,
            description=description,
            confidence=confidence,
        )

        self.results.append(result)
        print(f"\n  Verdict: {verdict} (confidence: {confidence:.2f})")

        # Save visualization
        self._plot_trend_response(pred_baseline, pred_uptrend, pred_downtrend)

        return result

    def experiment_3_temporal_ablation(self) -> ExperimentResult:
        """
        Experiment 3: Temporal Ablation Gradient

        Mask increasing portions of history and measure prediction change.
        If shorter context performs same/better, C1 is confirmed.
        """
        print("\n" + "="*60)
        print("Experiment 3: Temporal Ablation Gradient")
        print("="*60)

        batch = self.sample_batch
        daily_seq_len = list(batch['daily_features'].values())[0].shape[1]
        monthly_seq_len = list(batch['monthly_features'].values())[0].shape[1]

        # Get full context predictions
        pred_full = self._get_predictions(batch)

        # Test different context lengths
        context_lengths = [30, 90, 180, 270, 365]
        results_by_context = {}

        for ctx_len in context_lengths:
            if ctx_len > daily_seq_len:
                continue

            # Mask early history (keep only last ctx_len days)
            masked_daily_features = {}
            masked_daily_masks = {}

            for name, features in batch['daily_features'].items():
                masked_features = features.clone()
                masked_mask = batch['daily_masks'][name].clone()

                # Zero out early history
                mask_until = daily_seq_len - ctx_len
                masked_features[:, :mask_until, :] = 0.0
                masked_mask[:, :mask_until, :] = False

                masked_daily_features[name] = masked_features
                masked_daily_masks[name] = masked_mask

            masked_batch = {
                'daily_features': masked_daily_features,
                'daily_masks': masked_daily_masks,
                'monthly_features': batch['monthly_features'],
                'monthly_masks': batch['monthly_masks'],
                'month_boundaries': batch['month_boundaries'],
            }

            pred_masked = self._get_predictions(masked_batch)
            results_by_context[ctx_len] = pred_masked

        # Analyze how predictions change with context length
        metrics = {}

        if 'casualty_pred' in pred_full:
            full_pred = pred_full['casualty_pred'].cpu()

            # Focus on last few predictions (most affected by context)
            last_pred_idx = -3  # Last 3 months
            full_last = full_pred[:, last_pred_idx:, :]

            changes = []
            for ctx_len, pred_ctx in results_by_context.items():
                ctx_last = pred_ctx['casualty_pred'][:, last_pred_idx:, :].cpu()
                mae = torch.abs(full_last - ctx_last).mean().item()
                changes.append((ctx_len, mae))
                metrics[f'mae_ctx_{ctx_len}'] = mae
                print(f"  Context {ctx_len} days - MAE from full: {mae:.6f}")

            # Check if there's a monotonic relationship
            if len(changes) >= 3:
                maes = [c[1] for c in changes]
                # Ideally, more context = lower error (negative correlation)
                ctx_lens = [c[0] for c in changes]
                corr = np.corrcoef(ctx_lens, maes)[0, 1]
                metrics['context_error_correlation'] = float(corr)
                print(f"  Context length vs error correlation: {corr:.4f}")

        # Verdict: if correlation is positive (more context = more error) or near zero,
        # model doesn't benefit from temporal context
        corr = metrics.get('context_error_correlation', 0)

        if corr >= -0.3:
            verdict = "CONFIRMS_C1"
            confidence = 0.8
            description = "Longer context does not improve predictions"
        elif corr >= -0.6:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            description = "Weak benefit from longer context"
        else:
            verdict = "REFUTES_C1"
            confidence = 0.85
            description = "Clear benefit from longer temporal context"

        result = ExperimentResult(
            name="Temporal Ablation Gradient",
            verdict=verdict,
            metrics=metrics,
            description=description,
            confidence=confidence,
        )

        self.results.append(result)
        print(f"\n  Verdict: {verdict} (confidence: {confidence:.2f})")

        # Save visualization
        self._plot_ablation_results(metrics)

        return result

    def experiment_4_positional_encoding(self) -> ExperimentResult:
        """
        Experiment 4: Positional Encoding Analysis

        Extract and analyze sinusoidal positional encoding weights.
        If learned components have near-uniform weights, C1 is confirmed.
        """
        print("\n" + "="*60)
        print("Experiment 4: Positional Encoding Analysis")
        print("="*60)

        metrics = {}

        # Find all positional encoding modules
        pe_modules = []
        for name, module in self.model.named_modules():
            if 'positional_encoding' in name.lower():
                pe_modules.append((name, module))

        print(f"  Found {len(pe_modules)} positional encoding modules")

        # Analyze each positional encoding
        for name, module in pe_modules:
            if hasattr(module, 'pe'):
                pe = module.pe.cpu()

                # Compute variance across positions
                position_variance = pe.var(dim=1).mean().item()
                metrics[f'{name}_position_variance'] = position_variance

                # Compute variance across dimensions
                dim_variance = pe.var(dim=2).mean().item()
                metrics[f'{name}_dim_variance'] = dim_variance

                # Check frequency distribution
                # In sinusoidal encoding, different dimensions have different frequencies
                # If weights are meaningful, we expect high variance

                print(f"  {name}:")
                print(f"    Position variance: {position_variance:.6f}")
                print(f"    Dimension variance: {dim_variance:.6f}")

        # Also check if there are any learned position-related parameters
        position_params = []
        for name, param in self.model.named_parameters():
            if 'position' in name.lower() or 'pos' in name.lower():
                if param.requires_grad:
                    position_params.append((name, param))

        print(f"\n  Found {len(position_params)} learnable position parameters")

        for name, param in position_params:
            variance = param.data.var().item()
            mean = param.data.mean().item()
            metrics[f'{name}_variance'] = variance
            metrics[f'{name}_mean'] = mean
            print(f"  {name}: mean={mean:.4f}, var={variance:.6f}")

        # Analyze temporal encoder specifically
        if hasattr(self.model, 'temporal_encoder'):
            te = self.model.temporal_encoder
            if hasattr(te, 'positional_encoding') and hasattr(te.positional_encoding, 'pe'):
                pe = te.positional_encoding.pe.cpu()

                # Compute correlation matrix between positions
                pe_2d = pe.squeeze(0)  # [seq_len, d_model]

                # Sample positions for visualization
                sample_positions = [0, 5, 10, 20, 30, 50]
                sample_positions = [p for p in sample_positions if p < pe_2d.shape[0]]

                if len(sample_positions) >= 2:
                    pos_vecs = pe_2d[sample_positions]
                    pos_corr = torch.corrcoef(pos_vecs)
                    avg_off_diag = (pos_corr.sum() - pos_corr.trace()) / (pos_corr.numel() - len(sample_positions))
                    metrics['temporal_pe_avg_correlation'] = avg_off_diag.item()
                    print(f"\n  Temporal encoder PE correlation (off-diagonal avg): {avg_off_diag:.4f}")

        # Verdict: High variance in PE suggests it's being used
        # Near-uniform (low variance) suggests it's ignored
        max_variance = max(
            [v for k, v in metrics.items() if 'variance' in k],
            default=0
        )

        if max_variance < 0.1:
            verdict = "CONFIRMS_C1"
            confidence = 0.7
            description = "Positional encodings show low variance, suggesting minimal use"
        elif max_variance < 0.5:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            description = "Moderate variance in positional encodings"
        else:
            verdict = "REFUTES_C1"
            confidence = 0.75
            description = "High variance in positional encodings suggests active use"

        result = ExperimentResult(
            name="Positional Encoding Analysis",
            verdict=verdict,
            metrics=metrics,
            description=description,
            confidence=confidence,
        )

        self.results.append(result)
        print(f"\n  Verdict: {verdict} (confidence: {confidence:.2f})")

        # Save visualization
        self._plot_positional_encoding()

        return result

    def experiment_5_gradient_flow(self) -> ExperimentResult:
        """
        Experiment 5: Gradient Flow Through Temporal Layers

        Compute gradient norms at temporal encoder layers.
        Vanishing gradients suggest temporal layers are not used.
        """
        print("\n" + "="*60)
        print("Experiment 5: Gradient Flow Through Temporal Layers")
        print("="*60)

        self.model.train()  # Enable gradients

        batch = self.sample_batch

        # Move to device
        daily_features = {k: v.to(self.device).requires_grad_(True) for k, v in batch['daily_features'].items()}
        daily_masks = {k: v.to(self.device) for k, v in batch['daily_masks'].items()}
        monthly_features = {k: v.to(self.device).requires_grad_(True) for k, v in batch['monthly_features'].items()}
        monthly_masks = {k: v.to(self.device) for k, v in batch['monthly_masks'].items()}
        month_boundaries = batch['month_boundaries'].to(self.device)

        # Forward pass
        outputs = self.model(
            daily_features=daily_features,
            daily_masks=daily_masks,
            monthly_features=monthly_features,
            monthly_masks=monthly_masks,
            month_boundaries=month_boundaries,
        )

        # Create dummy loss and backpropagate
        if 'casualty_pred' in outputs:
            loss = outputs['casualty_pred'].sum()
        else:
            loss = outputs['temporal_output'].sum()

        loss.backward()

        # Collect gradient statistics
        metrics = {}
        gradient_norms = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[name] = grad_norm

                # Categorize by layer type
                if 'temporal_encoder' in name:
                    key = 'temporal_encoder_grad_norm'
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(grad_norm)
                elif 'daily_encoder' in name:
                    key = 'daily_encoder_grad_norm'
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(grad_norm)
                elif 'positional' in name:
                    key = 'positional_grad_norm'
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(grad_norm)

        # Compute summary statistics
        for key in ['temporal_encoder_grad_norm', 'daily_encoder_grad_norm', 'positional_grad_norm']:
            if key in metrics and len(metrics[key]) > 0:
                mean_grad = np.mean(metrics[key])
                max_grad = np.max(metrics[key])
                metrics[f'{key}_mean'] = mean_grad
                metrics[f'{key}_max'] = max_grad
                print(f"  {key}: mean={mean_grad:.6f}, max={max_grad:.6f}")

        # Compare temporal encoder gradients to other layers
        temporal_grad_mean = metrics.get('temporal_encoder_grad_norm_mean', 0)
        daily_grad_mean = metrics.get('daily_encoder_grad_norm_mean', 0)

        if daily_grad_mean > 0:
            ratio = temporal_grad_mean / daily_grad_mean
            metrics['temporal_to_daily_grad_ratio'] = ratio
            print(f"\n  Temporal/Daily gradient ratio: {ratio:.4f}")

        # Check for vanishing gradients
        all_grads = [v for v in gradient_norms.values() if v > 0]
        if all_grads:
            min_grad = min(all_grads)
            max_grad = max(all_grads)
            grad_range = max_grad / (min_grad + 1e-10)
            metrics['gradient_range_ratio'] = grad_range
            print(f"  Gradient range (max/min): {grad_range:.2f}")

        self.model.eval()  # Back to eval mode

        # Verdict: vanishing gradients in temporal layers suggest they're not used
        temporal_grad = metrics.get('temporal_encoder_grad_norm_mean', 0)

        if temporal_grad < 1e-6:
            verdict = "CONFIRMS_C1"
            confidence = 0.9
            description = "Vanishing gradients in temporal encoder suggest it's not learning"
        elif temporal_grad < 1e-4:
            verdict = "INCONCLUSIVE"
            confidence = 0.5
            description = "Low but non-zero gradients in temporal encoder"
        else:
            verdict = "REFUTES_C1"
            confidence = 0.8
            description = "Healthy gradients flowing through temporal encoder"

        result = ExperimentResult(
            name="Gradient Flow Through Temporal Layers",
            verdict=verdict,
            metrics=metrics,
            description=description,
            confidence=confidence,
        )

        self.results.append(result)
        print(f"\n  Verdict: {verdict} (confidence: {confidence:.2f})")

        # Save visualization
        self._plot_gradient_flow(gradient_norms)

        return result

    def _plot_reverse_comparison(
        self,
        pred_original: Dict,
        pred_reversed: Dict,
    ) -> None:
        """Plot comparison of original vs reversed predictions."""
        if 'casualty_pred' not in pred_original:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        orig = pred_original['casualty_pred'][0, :, 0].cpu().numpy()
        rev = pred_reversed['casualty_pred'][0, :, 0].cpu().numpy()
        rev_flipped = np.flip(rev)

        # Plot 1: Original predictions
        axes[0, 0].plot(orig, 'b-', label='Original')
        axes[0, 0].set_title('Original Order Predictions')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Prediction')
        axes[0, 0].legend()

        # Plot 2: Reversed predictions (flipped back)
        axes[0, 1].plot(rev_flipped, 'r-', label='Reversed (flipped)')
        axes[0, 1].set_title('Reversed Order Predictions (Aligned)')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Prediction')
        axes[0, 1].legend()

        # Plot 3: Overlay comparison
        axes[1, 0].plot(orig, 'b-', label='Original', alpha=0.7)
        axes[1, 0].plot(rev_flipped, 'r--', label='Reversed', alpha=0.7)
        axes[1, 0].set_title('Overlay Comparison')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Prediction')
        axes[1, 0].legend()

        # Plot 4: Difference
        diff = orig - rev_flipped
        axes[1, 1].plot(diff, 'g-')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].set_title(f'Difference (MAE={np.abs(diff).mean():.4f})')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Difference')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp1_reverse_sequence.png', dpi=150)
        plt.close()

    def _plot_trend_response(
        self,
        pred_baseline: Dict,
        pred_uptrend: Dict,
        pred_downtrend: Dict,
    ) -> None:
        """Plot model response to synthetic trends."""
        if 'casualty_pred' not in pred_baseline:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        base = pred_baseline['casualty_pred'][0, :, 0].cpu().numpy()
        up = pred_uptrend['casualty_pred'][0, :, 0].cpu().numpy()
        down = pred_downtrend['casualty_pred'][0, :, 0].cpu().numpy()

        # Plot 1: All predictions overlaid
        axes[0, 0].plot(base, 'b-', label='Baseline', linewidth=2)
        axes[0, 0].plot(up, 'g--', label='Uptrend', alpha=0.7)
        axes[0, 0].plot(down, 'r--', label='Downtrend', alpha=0.7)
        axes[0, 0].set_title('Predictions with Trend Injection')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Prediction')
        axes[0, 0].legend()

        # Plot 2: Difference from baseline
        axes[0, 1].plot(up - base, 'g-', label='Uptrend - Baseline')
        axes[0, 1].plot(down - base, 'r-', label='Downtrend - Baseline')
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Deviation from Baseline')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Difference')
        axes[0, 1].legend()

        # Plot 3: Uptrend vs Downtrend
        axes[1, 0].plot(up - down, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Uptrend - Downtrend Response')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Difference')

        # Plot 4: Summary statistics
        stats = {
            'Baseline Mean': base.mean(),
            'Uptrend Mean': up.mean(),
            'Downtrend Mean': down.mean(),
            'Up-Down Diff': (up - down).mean(),
        }
        axes[1, 1].bar(stats.keys(), stats.values(), color=['blue', 'green', 'red', 'purple'])
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp2_trend_response.png', dpi=150)
        plt.close()

    def _plot_ablation_results(self, metrics: Dict) -> None:
        """Plot temporal ablation results."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract context lengths and MAEs
        ctx_data = [(int(k.split('_')[-1]), v)
                    for k, v in metrics.items() if 'mae_ctx' in k]
        ctx_data.sort()

        if ctx_data:
            ctx_lens, maes = zip(*ctx_data)

            ax.plot(ctx_lens, maes, 'bo-', markersize=10, linewidth=2)
            ax.set_xlabel('Context Length (days)', fontsize=12)
            ax.set_ylabel('MAE from Full Context', fontsize=12)
            ax.set_title('Prediction Error vs Context Length', fontsize=14)

            # Add correlation annotation
            if 'context_error_correlation' in metrics:
                corr = metrics['context_error_correlation']
                ax.annotate(f'Correlation: {corr:.3f}',
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=12, va='top')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp3_ablation.png', dpi=150)
        plt.close()

    def _plot_positional_encoding(self) -> None:
        """Plot positional encoding analysis."""
        if not hasattr(self.model, 'temporal_encoder'):
            return

        te = self.model.temporal_encoder
        if not hasattr(te, 'positional_encoding') or not hasattr(te.positional_encoding, 'pe'):
            return

        pe = te.positional_encoding.pe.cpu().squeeze(0).numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Positional encoding heatmap
        im = axes[0, 0].imshow(pe[:60, :64], aspect='auto', cmap='RdBu_r')
        axes[0, 0].set_title('Positional Encoding Matrix')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Position')
        plt.colorbar(im, ax=axes[0, 0])

        # Plot 2: Sample position vectors
        positions = [0, 10, 20, 30, 40, 50]
        positions = [p for p in positions if p < pe.shape[0]]
        for p in positions:
            axes[0, 1].plot(pe[p, :64], label=f'Position {p}', alpha=0.7)
        axes[0, 1].set_title('Sample Position Vectors')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()

        # Plot 3: Variance across positions for each dimension
        dim_variance = pe[:60, :].var(axis=0)
        axes[1, 0].bar(range(len(dim_variance[:64])), dim_variance[:64])
        axes[1, 0].set_title('Variance Across Positions per Dimension')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Variance')

        # Plot 4: Correlation between nearby positions
        if pe.shape[0] >= 10:
            pos_corrs = []
            for i in range(min(30, pe.shape[0] - 1)):
                corr = np.corrcoef(pe[i], pe[i + 1])[0, 1]
                pos_corrs.append(corr)
            axes[1, 1].plot(pos_corrs, 'b-')
            axes[1, 1].set_title('Correlation Between Adjacent Positions')
            axes[1, 1].set_xlabel('Position')
            axes[1, 1].set_ylabel('Correlation')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp4_positional_encoding.png', dpi=150)
        plt.close()

    def _plot_gradient_flow(self, gradient_norms: Dict) -> None:
        """Plot gradient flow analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Categorize gradients
        categories = {
            'temporal': [],
            'daily': [],
            'monthly': [],
            'fusion': [],
            'other': [],
        }

        for name, norm in gradient_norms.items():
            if 'temporal_encoder' in name:
                categories['temporal'].append((name, norm))
            elif 'daily_encoder' in name:
                categories['daily'].append((name, norm))
            elif 'monthly_encoder' in name:
                categories['monthly'].append((name, norm))
            elif 'fusion' in name:
                categories['fusion'].append((name, norm))
            else:
                categories['other'].append((name, norm))

        # Plot 1: Box plot by category
        data = []
        labels = []
        for cat, grads in categories.items():
            if grads:
                norms = [g[1] for g in grads]
                data.append(norms)
                labels.append(f'{cat}\n(n={len(norms)})')

        if data:
            axes[0].boxplot(data, labels=labels)
            axes[0].set_title('Gradient Norms by Module Category')
            axes[0].set_ylabel('Gradient Norm')
            axes[0].set_yscale('log')

        # Plot 2: Top 20 gradient norms
        sorted_grads = sorted(gradient_norms.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [g[0].split('.')[-1][:15] for g in sorted_grads]
        values = [g[1] for g in sorted_grads]

        axes[1].barh(range(len(names)), values)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=8)
        axes[1].set_title('Top 20 Gradient Norms')
        axes[1].set_xlabel('Gradient Norm')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'exp5_gradient_flow.png', dpi=150)
        plt.close()

    def run_all_experiments(self) -> ValidationReport:
        """Run all experiments and generate report."""
        print("="*60)
        print("TEMPORAL LEARNING VALIDATION - CLAIM C1")
        print("="*60)
        print("\nClaim: The model ignores temporal structure and treats")
        print("       input sequences as bags of features.")
        print("="*60)

        # Run experiments
        self.experiment_1_reverse_sequence()
        self.experiment_2_synthetic_trend()
        self.experiment_3_temporal_ablation()
        self.experiment_4_positional_encoding()
        self.experiment_5_gradient_flow()

        # Tally verdicts
        confirms = sum(1 for r in self.results if r.verdict == "CONFIRMS_C1")
        refutes = sum(1 for r in self.results if r.verdict == "REFUTES_C1")
        inconclusive = sum(1 for r in self.results if r.verdict == "INCONCLUSIVE")

        total = len(self.results)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nResults: {confirms} confirm, {refutes} refute, {inconclusive} inconclusive")

        # Determine final verdict
        if confirms >= 3:
            final_verdict = "CONFIRMED"
            confidence_level = "HIGH" if confirms >= 4 else "MODERATE"
        elif refutes >= 3:
            final_verdict = "REFUTED"
            confidence_level = "HIGH" if refutes >= 4 else "MODERATE"
        else:
            final_verdict = "INCONCLUSIVE"
            confidence_level = "LOW"

        summary = f"""
Claim C1 Validation Results
===========================

Claim: "The model ignores temporal structure and treats input sequences as bags of features."

FINAL VERDICT: {final_verdict}
Confidence Level: {confidence_level}

Experiment Results:
"""
        for r in self.results:
            summary += f"\n  - {r.name}: {r.verdict} (confidence: {r.confidence:.2f})"
            summary += f"\n    {r.description}"

        summary += f"""

Tally: {confirms}/5 confirm, {refutes}/5 refute, {inconclusive}/5 inconclusive

Interpretation:
"""
        if final_verdict == "CONFIRMED":
            summary += """
  The evidence strongly suggests the model is NOT effectively learning temporal
  patterns. The model treats sequences largely as collections of features without
  meaningful temporal ordering. This indicates a potential architecture or
  training issue that should be addressed.

Recommendations:
  1. Review temporal encoder architecture for potential information bottlenecks
  2. Consider adding explicit temporal losses (e.g., temporal coherence)
  3. Verify positional encodings are being used correctly
  4. Check for gradient flow issues in temporal layers
"""
        elif final_verdict == "REFUTED":
            summary += """
  The evidence suggests the model IS learning temporal patterns to some degree.
  The claim that it treats sequences as bags of features is NOT supported.
  However, there may still be room for improvement in temporal modeling.
"""
        else:
            summary += """
  The results are mixed and do not clearly confirm or refute the claim.
  Additional experiments or analysis may be needed to reach a definitive
  conclusion about the model's temporal learning capabilities.
"""

        print(summary)

        # Save report
        report = ValidationReport(
            experiment_results=self.results,
            final_verdict=final_verdict,
            confidence_level=confidence_level,
            summary=summary,
        )

        self._save_report(report)

        return report

    def _save_report(self, report: ValidationReport) -> None:
        """Save validation report to markdown file."""
        report_path = self.output_dir / 'C1_temporal_learning_report.md'

        content = f"""# Temporal Learning Validation Report (Claim C1)

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Checkpoint:** `{self.checkpoint_path}`

---

## Claim Under Investigation

> "The model ignores temporal structure and treats input sequences as bags of features."

---

## Final Verdict: **{report.final_verdict}**

**Confidence Level:** {report.confidence_level}

---

## Experiment Results Summary

| Experiment | Verdict | Confidence | Key Finding |
|------------|---------|------------|-------------|
"""
        for r in report.experiment_results:
            content += f"| {r.name} | {r.verdict} | {r.confidence:.2f} | {r.description[:50]}... |\n"

        content += """
---

## Detailed Experiment Results

"""
        for i, r in enumerate(report.experiment_results, 1):
            content += f"""### Experiment {i}: {r.name}

**Verdict:** {r.verdict}
**Confidence:** {r.confidence:.2f}

**Description:** {r.description}

**Key Metrics:**
```json
{json.dumps(r.metrics, indent=2)}
```

**Visualization:** See `exp{i}_*.png`

---

"""

        content += f"""## Summary

{report.summary}

---

## Visualizations

The following visualizations were generated:

1. `exp1_reverse_sequence.png` - Comparison of predictions for original vs reversed sequences
2. `exp2_trend_response.png` - Model response to synthetic trend injection
3. `exp3_ablation.png` - Prediction error vs temporal context length
4. `exp4_positional_encoding.png` - Analysis of positional encoding weights
5. `exp5_gradient_flow.png` - Gradient norms through temporal layers

---

## Methodology Notes

### Validation Thresholds

- **Reverse Sequence Test**: Correlation > 0.95 confirms C1, < 0.80 refutes
- **Synthetic Trend**: Trend difference < 0.1 confirms C1, > 0.3 refutes
- **Temporal Ablation**: Context-error correlation >= -0.3 confirms C1, < -0.6 refutes
- **Positional Encoding**: Variance < 0.1 confirms C1, > 0.5 refutes
- **Gradient Flow**: Temporal gradient < 1e-6 confirms C1, > 1e-4 refutes

### Interpretation Guide

- **CONFIRMED (3+ experiments)**: Strong evidence model ignores temporal structure
- **REFUTED (3+ experiments)**: Strong evidence model uses temporal structure
- **INCONCLUSIVE**: Mixed results require further investigation

---

*Report generated by Temporal Learning Validator v1.0*
"""

        with open(report_path, 'w') as f:
            f.write(content)

        print(f"\nReport saved to: {report_path}")

        # Also save as JSON for programmatic access
        json_path = self.output_dir / 'C1_validation_results.json'
        json_data = {
            'final_verdict': report.final_verdict,
            'confidence_level': report.confidence_level,
            'experiments': [
                {
                    'name': r.name,
                    'verdict': r.verdict,
                    'confidence': r.confidence,
                    'description': r.description,
                    'metrics': r.metrics,
                }
                for r in report.experiment_results
            ],
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': str(self.checkpoint_path),
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON results saved to: {json_path}")


def main():
    """Run temporal learning validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate temporal learning in Multi-Resolution HAN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/Users/daniel.tipton/ML_OSINT/analysis/training_runs/run_24-01-2026_20-22/stage3_han/best_checkpoint.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation',
        help='Output directory for report and visualizations',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for inference',
    )

    args = parser.parse_args()

    validator = TemporalLearningValidator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )

    report = validator.run_all_experiments()

    return 0 if report.final_verdict != "INCONCLUSIVE" else 1


if __name__ == '__main__':
    exit(main())
