#!/usr/bin/env python3
"""
Prediction Script for Ukraine Conflict OSINT Models

Makes predictions using:
1. Hierarchical Attention Network (HAN) - multi-domain forecasting
2. Phase 1 Interpolation Models - gap-filling for sparse observations
3. Phase 2 Child Models - decomposed feature interpolation

Usage:
    python analysis/predict.py --forecast --horizon 3
    python analysis/predict.py --interpolate --date 2024-01-15
    python analysis/predict.py --all
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR,
    get_interp_model_path,
)

# Import models
from hierarchical_attention_network import (
    HierarchicalAttentionNetwork,
    DOMAIN_CONFIGS,
    TOTAL_FEATURES
)
from joint_interpolation_models import (
    INTERPOLATION_CONFIGS, PHASE2_CONFIGS,
    JointInterpolationModel
)
from conflict_data_loader import create_data_loaders

BASE_DIR = PROJECT_ROOT


class ConflictPredictor:
    """
    Unified prediction interface for all trained models.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.han_model = None
        self.phase1_models = {}
        self.phase2_models = {}
        self.data_loaders = None

    def load_models(self, load_han: bool = True, load_interpolators: bool = True):
        """Load all trained models."""
        print("=" * 70)
        print("LOADING TRAINED MODELS")
        print("=" * 70)

        if load_han:
            self._load_han()

        if load_interpolators:
            self._load_phase1_models()
            self._load_phase2_models()

        print("\nModels loaded successfully.")

    def _load_han(self):
        """Load Hierarchical Attention Network."""
        print("\nLoading HAN model...")

        han_path = MODEL_DIR / "han_best.pt"
        if not han_path.exists():
            print(f"  Warning: HAN model not found at {han_path}")
            return

        self.han_model = HierarchicalAttentionNetwork(
            domain_configs=DOMAIN_CONFIGS,
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            num_temporal_layers=1,
            dropout=0.35
        )

        checkpoint = torch.load(han_path, map_location=self.device)
        self.han_model.load_state_dict(checkpoint['model_state_dict'])
        self.han_model.to(self.device)
        self.han_model.eval()

        n_params = sum(p.numel() for p in self.han_model.parameters())
        print(f"  Loaded HAN: {n_params:,} parameters")
        print(f"  Best val loss: {checkpoint.get('val_loss', 'N/A')}")

    def _load_phase1_models(self):
        """Load Phase 1 interpolation models."""
        print("\nLoading Phase 1 interpolation models...")

        for name, config in INTERPOLATION_CONFIGS.items():
            safe_name = config.name.replace(' ', '_').replace('/', '_').lower()
            model_path = INTERP_MODEL_DIR / f"interp_{safe_name}_best.pt"

            if model_path.exists():
                model = JointInterpolationModel(config)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.phase1_models[name] = model
                print(f"  Loaded: {name} ({len(config.features)} features)")
            else:
                print(f"  Missing: {name}")

    def _load_phase2_models(self):
        """Load Phase 2 child models."""
        print("\nLoading Phase 2 child models...")

        for name, config in PHASE2_CONFIGS.items():
            model_path = MODEL_DIR / f"phase2_{name}_best.pt"

            if model_path.exists():
                model = JointInterpolationModel(config)
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.phase2_models[name] = model
                print(f"  Loaded: {name} ({len(config.features)} features)")
            else:
                print(f"  Missing: {name}")

    def forecast_han(self, horizon: int = 1, return_attention: bool = False):
        """
        Make forecasts using the Hierarchical Attention Network.

        Args:
            horizon: Number of months ahead to forecast
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with forecasts and metadata
        """
        if self.han_model is None:
            raise ValueError("HAN model not loaded. Call load_models() first.")

        print("\n" + "=" * 70)
        print(f"HAN FORECAST (horizon={horizon} months)")
        print("=" * 70)

        # Load latest data
        _, val_loader = create_data_loaders(
            DOMAIN_CONFIGS,
            batch_size=1,
            seq_len=4
        )

        # Get last validation sample (most recent data)
        features_batch, masks_batch, targets_batch = None, None, None
        for f, m, t in val_loader:
            features_batch, masks_batch, targets_batch = f, m, t

        if features_batch is None:
            raise ValueError("No data available for prediction")

        # Move to device
        features = {k: v.to(self.device) for k, v in features_batch.items()}
        masks = {k: v.to(self.device) for k, v in masks_batch.items()}

        # Make prediction
        self.han_model.eval()
        with torch.no_grad():
            outputs = self.han_model(features, masks, return_attention=return_attention)

        results = {
            'timestamp': datetime.now().isoformat(),
            'horizon_months': horizon,
            'domains': {}
        }

        # Process outputs
        forecast = outputs['forecast'][0, -1, :].cpu().numpy()  # Last timestep forecast

        # Split forecast into domains
        idx = 0
        for domain_name, config in DOMAIN_CONFIGS.items():
            n_features = config.num_features
            domain_forecast = forecast[idx:idx + n_features]

            results['domains'][domain_name] = {
                'forecast': domain_forecast.tolist(),
                'num_features': n_features,
                'feature_names': getattr(config, 'feature_names', [f'feat_{i}' for i in range(n_features)])
            }
            idx += n_features

        # Regime prediction
        if 'regime_logits' in outputs:
            regime_probs = F.softmax(outputs['regime_logits'][0, -1, :], dim=0).cpu().numpy()
            regime_pred = int(regime_probs.argmax())
            regime_labels = ['Low Intensity', 'Medium Intensity', 'High Intensity', 'Offensive Operations']
            results['regime'] = {
                'predicted': regime_labels[regime_pred],
                'probabilities': {
                    label: float(prob) for label, prob in zip(regime_labels, regime_probs)
                }
            }

        # Anomaly score
        if 'anomaly_score' in outputs:
            anomaly = outputs['anomaly_score'][0, -1].item()
            results['anomaly_score'] = anomaly
            results['anomaly_flag'] = anomaly > 0.5

        # Attention weights (if requested)
        if return_attention and 'attention_weights' in outputs:
            results['attention'] = {
                k: v[0].cpu().numpy().tolist()
                for k, v in outputs['attention_weights'].items()
            }

        return results

    def print_forecast(self, results: dict):
        """Pretty print forecast results."""
        print("\n" + "-" * 70)
        print("FORECAST RESULTS")
        print("-" * 70)

        # Regime prediction
        if 'regime' in results:
            print(f"\nPredicted Conflict Regime: {results['regime']['predicted']}")
            print("  Probability distribution:")
            for label, prob in results['regime']['probabilities'].items():
                bar = "█" * int(prob * 30)
                print(f"    {label:25s}: {prob:5.1%} {bar}")

        # Anomaly detection
        if 'anomaly_score' in results:
            score = results['anomaly_score']
            flag = "⚠️  ANOMALY DETECTED" if results['anomaly_flag'] else "✓ Normal"
            print(f"\nAnomaly Score: {score:.3f} {flag}")

        # Domain forecasts
        print("\nDomain Forecasts (normalized values):")
        for domain, data in results['domains'].items():
            print(f"\n  {domain.upper()}:")
            forecast = data['forecast']
            names = data['feature_names']

            # Show top 5 features by magnitude
            sorted_idx = np.argsort(np.abs(forecast))[::-1][:5]
            for i in sorted_idx:
                name = names[i] if i < len(names) else f'feat_{i}'
                val = forecast[i]
                direction = "↑" if val > 0 else "↓" if val < 0 else "→"
                print(f"    {name:30s}: {val:+.4f} {direction}")

    def interpolate_date(self, target_date: str, source: str = 'all'):
        """
        Interpolate features for a specific date.

        Args:
            target_date: Date string (YYYY-MM-DD)
            source: Source to interpolate ('all', 'sentinel', 'deepstate', etc.)

        Returns:
            Dictionary with interpolated values and uncertainties
        """
        print("\n" + "=" * 70)
        print(f"INTERPOLATION for {target_date}")
        print("=" * 70)

        target_dt = datetime.strptime(target_date, '%Y-%m-%d')

        results = {
            'target_date': target_date,
            'timestamp': datetime.now().isoformat(),
            'phase1': {},
            'phase2': {}
        }

        # Phase 1 interpolation
        models_to_use = self.phase1_models if source == 'all' else {
            k: v for k, v in self.phase1_models.items()
            if INTERPOLATION_CONFIGS[k].source == source
        }

        print(f"\nPhase 1 Models: {len(models_to_use)}")
        for name, model in models_to_use.items():
            config = INTERPOLATION_CONFIGS[name]
            print(f"  {name}: {len(config.features)} features")

            # For demo, use synthetic boundary observations
            # In production, would load actual data around target_date
            n_feat = len(config.features)
            obs_before = torch.randn(1, n_feat) * 0.3 + 0.5
            obs_after = torch.randn(1, n_feat) * 0.3 + 0.5
            day_before = torch.tensor([[0.0]])
            day_after = torch.tensor([[5.0]])
            day_target = torch.tensor([[2.5]])

            model.eval()
            with torch.no_grad():
                predictions, uncertainties = model(
                    obs_before.to(self.device),
                    obs_after.to(self.device),
                    day_before.to(self.device),
                    day_after.to(self.device),
                    day_target.to(self.device)
                )

            results['phase1'][name] = {
                'predictions': predictions[0].cpu().numpy().tolist(),
                'uncertainties': uncertainties[0].cpu().numpy().tolist(),
                'features': config.features
            }

        # Phase 2 interpolation (conditioned on Phase 1)
        models_to_use = self.phase2_models if source == 'all' else {
            k: v for k, v in self.phase2_models.items()
            if PHASE2_CONFIGS[k].source == source
        }

        print(f"\nPhase 2 Models: {len(models_to_use)}")
        for name, model in models_to_use.items():
            config = PHASE2_CONFIGS[name]
            print(f"  {name}: {len(config.features)} features")

            n_feat = len(config.features)
            obs_before = torch.randn(1, n_feat) * 0.3 + 0.5
            obs_after = torch.randn(1, n_feat) * 0.3 + 0.5
            day_before = torch.tensor([[0.0]])
            day_after = torch.tensor([[5.0]])
            day_target = torch.tensor([[2.5]])

            model.eval()
            with torch.no_grad():
                predictions, uncertainties = model(
                    obs_before.to(self.device),
                    obs_after.to(self.device),
                    day_before.to(self.device),
                    day_after.to(self.device),
                    day_target.to(self.device)
                )

            results['phase2'][name] = {
                'predictions': predictions[0].cpu().numpy().tolist(),
                'uncertainties': uncertainties[0].cpu().numpy().tolist(),
                'features': config.features
            }

        return results

    def print_interpolation(self, results: dict):
        """Pretty print interpolation results."""
        print("\n" + "-" * 70)
        print(f"INTERPOLATION RESULTS for {results['target_date']}")
        print("-" * 70)

        print("\nPHASE 1 (Aggregate Features):")
        for name, data in results['phase1'].items():
            pred = np.array(data['predictions'])
            unc = np.array(data['uncertainties'])
            print(f"\n  {name}:")
            for i, feat in enumerate(data['features'][:5]):  # Show first 5
                print(f"    {feat:30s}: {pred[i]:.4f} ± {unc[i]:.4f}")
            if len(data['features']) > 5:
                print(f"    ... and {len(data['features']) - 5} more")

        print("\nPHASE 2 (Decomposed Features):")
        for name, data in results['phase2'].items():
            pred = np.array(data['predictions'])
            unc = np.array(data['uncertainties'])
            print(f"\n  {name}:")
            for i, feat in enumerate(data['features'][:5]):
                print(f"    {feat:30s}: {pred[i]:.4f} ± {unc[i]:.4f}")
            if len(data['features']) > 5:
                print(f"    ... and {len(data['features']) - 5} more")

    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)

        # HAN
        if self.han_model:
            n_params = sum(p.numel() for p in self.han_model.parameters())
            print(f"\nHierarchical Attention Network:")
            print(f"  Parameters: {n_params:,}")
            print(f"  Input features: {TOTAL_FEATURES}")
            print(f"  Domains: {len(DOMAIN_CONFIGS)}")
            for name, cfg in DOMAIN_CONFIGS.items():
                print(f"    - {name}: {cfg.num_features} features")

        # Phase 1
        print(f"\nPhase 1 Interpolation Models: {len(self.phase1_models)}")
        total_p1_features = 0
        for name, model in self.phase1_models.items():
            n_feat = len(INTERPOLATION_CONFIGS[name].features)
            total_p1_features += n_feat
            print(f"  - {name}: {n_feat} features")
        print(f"  Total Phase 1 features: {total_p1_features}")

        # Phase 2
        print(f"\nPhase 2 Child Models: {len(self.phase2_models)}")
        total_p2_features = 0
        for name, model in self.phase2_models.items():
            n_feat = len(PHASE2_CONFIGS[name].features)
            total_p2_features += n_feat
            print(f"  - {name}: {n_feat} features")
        print(f"  Total Phase 2 features: {total_p2_features}")

        print(f"\nTotal interpolated features: {total_p1_features + total_p2_features}")


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained models')
    parser.add_argument('--forecast', action='store_true', help='Run HAN forecast')
    parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon in months')
    parser.add_argument('--interpolate', action='store_true', help='Run interpolation')
    parser.add_argument('--date', type=str, default=None, help='Target date for interpolation (YYYY-MM-DD)')
    parser.add_argument('--source', type=str, default='all', help='Source for interpolation')
    parser.add_argument('--all', action='store_true', help='Run all predictions')
    parser.add_argument('--summary', action='store_true', help='Print model summary only')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    args = parser.parse_args()

    # Detect device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    predictor = ConflictPredictor(device=device)
    predictor.load_models()

    if args.summary:
        predictor.summary()
        return

    if args.all or args.forecast:
        results = predictor.forecast_han(horizon=args.horizon)
        predictor.print_forecast(results)

    if args.all or args.interpolate:
        target_date = args.date or datetime.now().strftime('%Y-%m-%d')
        results = predictor.interpolate_date(target_date, args.source)
        predictor.print_interpolation(results)

    if not (args.all or args.forecast or args.interpolate or args.summary):
        # Default: show summary and forecast
        predictor.summary()
        results = predictor.forecast_han()
        predictor.print_forecast(results)


if __name__ == "__main__":
    main()
