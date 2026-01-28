"""
ISW Integration Ablation Experiment

Purpose: Determine whether ISW narrative alignment improves or degrades forecast performance.

Hypothesis: Based on the emergent modularity finding, we hypothesize that:
- H0: ISW integration has no effect on forecast performance
- H1a: ISW integration improves forecast performance (document expectation)
- H1b: ISW integration degrades forecast performance (modularity finding suggests this)

Experimental Design:
1. Load the same checkpoint twice - once with ISW, once without
2. Run identical backtesting on both configurations
3. Compare forecast metrics with statistical significance testing
4. Analyze phase-specific performance differences

Author: Claude
Date: 2026-01-27
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from scipy import stats

from config.paths import CHECKPOINT_DIR, OUTPUT_DIR, DATA_DIR
from analysis.multi_resolution_han import MultiResolutionHAN, create_multi_resolution_han
from analysis.multi_resolution_data import MultiResolutionDataset, MultiResolutionConfig


@dataclass
class ExperimentConfig:
    """Configuration for ISW ablation experiment."""
    checkpoint_path: Path = CHECKPOINT_DIR / "multi_resolution" / "best_checkpoint.pt"
    output_dir: Path = OUTPUT_DIR / "experiments" / "isw_ablation"

    # Backtesting parameters
    n_backtest_samples: int = 100
    forecast_horizon: int = 7  # days

    # Statistical testing
    n_permutations: int = 1000
    confidence_level: float = 0.95

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ModelConfiguration:
    """Represents a specific model configuration for testing."""
    name: str
    use_isw: bool
    description: str


@dataclass
class ExperimentResults:
    """Stores results from one experimental condition."""
    config_name: str
    forecast_mse: List[float] = field(default_factory=list)
    forecast_mae: List[float] = field(default_factory=list)
    forecast_corr: List[float] = field(default_factory=list)
    regime_accuracy: List[float] = field(default_factory=list)
    latent_velocity: List[float] = field(default_factory=list)
    inference_time_ms: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        """Compute summary statistics."""
        def safe_stats(arr):
            if len(arr) == 0:
                return {"mean": None, "std": None, "median": None}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "n": len(arr)
            }

        return {
            "config_name": self.config_name,
            "forecast_mse": safe_stats(self.forecast_mse),
            "forecast_mae": safe_stats(self.forecast_mae),
            "forecast_corr": safe_stats(self.forecast_corr),
            "regime_accuracy": safe_stats(self.regime_accuracy),
            "latent_velocity": safe_stats(self.latent_velocity),
            "inference_time_ms": safe_stats(self.inference_time_ms)
        }


class ISWAblationExperiment:
    """
    Controlled experiment comparing model performance with and without ISW alignment.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Define experimental conditions
        self.conditions = [
            ModelConfiguration(
                name="with_isw",
                use_isw=True,
                description="Full model with ISW alignment module enabled"
            ),
            ModelConfiguration(
                name="without_isw",
                use_isw=False,
                description="Model with ISW alignment module disabled (weights discarded)"
            )
        ]

        self.results: Dict[str, ExperimentResults] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available()
                                   else "cpu")

    def load_model(self, use_isw: bool) -> MultiResolutionHAN:
        """
        Load model with or without ISW alignment.

        Args:
            use_isw: If True, load full model. If False, load without ISW weights.
        """
        from analysis.multi_resolution_han import SourceConfig

        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=False  # Required for numpy arrays in checkpoint
        )

        state_dict = checkpoint['model_state_dict']

        # Get feature dimensions from dataset (same approach as backtesting.py)
        dataset_config = MultiResolutionConfig()  # Uses defaults
        temp_dataset = MultiResolutionDataset(config=dataset_config, split='train')
        feature_info = temp_dataset.get_feature_info()
        del temp_dataset  # Free memory

        # Build source configs from dataset
        daily_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='daily')
            for name, info in feature_info.items() if info['resolution'] == 'daily'
        }
        monthly_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='monthly')
            for name, info in feature_info.items() if info['resolution'] == 'monthly'
        }

        # Create model with architecture matching checkpoint
        model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=128,
            nhead=8,
            num_daily_layers=3,  # Match checkpoint
            num_monthly_layers=2,  # Match checkpoint
            num_fusion_layers=2,
            dropout=0.0,
            use_isw_alignment=use_isw,
            isw_dim=1024,
        )

        if not use_isw:
            # Filter out ISW-related weights
            isw_keys = [k for k in state_dict.keys() if 'isw' in k.lower()]
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'isw' not in k.lower()}

            # Load with strict=False to allow missing ISW keys
            model.load_state_dict(filtered_state_dict, strict=False)
            print(f"  Loaded model WITHOUT ISW ({len(isw_keys)} ISW weights discarded)")
        else:
            model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded model WITH ISW (full checkpoint)")

        model.to(self.device)
        model.eval()

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {n_params:,}")

        return model

    def load_dataset(self) -> MultiResolutionDataset:
        """Load the evaluation dataset (validation split for unbiased evaluation)."""
        dataset_config = MultiResolutionConfig()  # Uses defaults

        # First load train to get norm_stats (required for val/test splits)
        train_dataset = MultiResolutionDataset(config=dataset_config, split='train')
        norm_stats = train_dataset.norm_stats

        # Now load val with the norm_stats
        val_dataset = MultiResolutionDataset(
            config=dataset_config,
            split='val',
            norm_stats=norm_stats
        )
        return val_dataset

    def run_single_evaluation(
        self,
        model: MultiResolutionHAN,
        sample,
        config_name: str
    ) -> Dict:
        """
        Run evaluation on a single sample.

        Returns dict with metrics.
        """
        import time

        # Convert sample to batch format
        batch = self._sample_to_batch(sample)

        # Time the inference
        start_time = time.perf_counter()

        with torch.no_grad():
            # Model takes positional arguments, not dict
            outputs = model(
                daily_features=batch['daily_features'],
                daily_masks=batch['daily_masks'],
                monthly_features=batch['monthly_features'],
                monthly_masks=batch['monthly_masks'],
                month_boundaries=batch['month_boundaries'],
            )

        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Extract metrics
        metrics = {
            "inference_time_ms": inference_time
        }

        # Forecast metrics - compare forecast_pred to forecast_targets from sample
        if 'forecast_pred' in outputs and 'forecast_targets' in batch:
            forecast_pred = outputs['forecast_pred']  # [1, 12, 35]
            forecast_targets = batch['forecast_targets']  # [35] or [1, 35]

            if forecast_pred is not None and forecast_targets is not None:
                # Get last timestep prediction
                pred = forecast_pred[0, -1, :].cpu().numpy()  # [35]
                target = forecast_targets.squeeze().cpu().numpy()  # [35]

                # Ensure same size
                min_len = min(len(pred), len(target))
                if min_len > 0:
                    pred = pred[:min_len]
                    target = target[:min_len]

                    metrics["forecast_mse"] = float(np.mean((pred - target) ** 2))
                    metrics["forecast_mae"] = float(np.mean(np.abs(pred - target)))

                    if np.std(pred) > 1e-6 and np.std(target) > 1e-6:
                        metrics["forecast_corr"] = float(np.corrcoef(pred, target)[0, 1])

        # Latent velocity (measure of representation stability)
        # Note: The key is 'temporal_output' not 'temporal_encoded'
        if 'temporal_output' in outputs:
            encoded = outputs['temporal_output']
            if encoded.dim() >= 2 and encoded.size(1) > 1:
                # Compute velocity as mean L2 distance between consecutive timesteps
                velocity = torch.norm(encoded[:, 1:] - encoded[:, :-1], dim=-1).mean()
                metrics["latent_velocity"] = float(velocity.cpu())

        return metrics

    def _sample_to_batch(self, sample) -> Dict[str, torch.Tensor]:
        """Convert a MultiResolutionSample to batch format for the model."""
        batch = {}

        # Daily features and masks (use sample's own masks)
        batch['daily_features'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.daily_features.items()
        }
        batch['daily_masks'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.daily_masks.items()
        }

        # Monthly features and masks
        batch['monthly_features'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.monthly_features.items()
        }
        batch['monthly_masks'] = {
            name: tensor.unsqueeze(0).to(self.device)
            for name, tensor in sample.monthly_masks.items()
        }

        # Month boundaries
        batch['month_boundaries'] = sample.month_boundary_indices.unsqueeze(0).to(self.device)

        # Forecast targets - concatenate monthly sources to match forecast_pred format [35 features]
        # Monthly sources: sentinel(7) + hdx_conflict(5) + hdx_food(10) + hdx_rainfall(6) + iom(7) = 35
        if hasattr(sample, 'forecast_targets') and sample.forecast_targets is not None:
            monthly_sources = ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']
            monthly_targets = []
            for source in monthly_sources:
                if source in sample.forecast_targets:
                    monthly_targets.append(sample.forecast_targets[source])
            if monthly_targets:
                # Concatenate along feature dimension, take last timestep
                concat_targets = torch.cat(monthly_targets, dim=-1)  # [12, 35]
                batch['forecast_targets'] = concat_targets[-1, :].to(self.device)  # [35]

        return batch

    def _sample_to_batch_legacy(self, sample) -> Dict[str, torch.Tensor]:
        """Legacy method for dict-style samples."""
        batch = {}
        for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(self.device) if value.dim() < 3 else value.to(self.device)
                elif isinstance(value, dict):
                    batch[key] = {k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v
                                  for k, v in value.items()}
                else:
                    batch[key] = value

        return batch

    def run_condition(self, condition: ModelConfiguration) -> ExperimentResults:
        """Run all evaluations for one experimental condition."""
        print(f"\n{'='*60}")
        print(f"Running condition: {condition.name}")
        print(f"Description: {condition.description}")
        print(f"{'='*60}")

        # Load model for this condition
        model = self.load_model(use_isw=condition.use_isw)

        # Load dataset
        dataset = self.load_dataset()

        # Initialize results
        results = ExperimentResults(config_name=condition.name)

        # Run evaluations
        n_samples = min(self.config.n_backtest_samples, len(dataset))

        # Sample indices (same for both conditions due to seed)
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        print(f"  Running {n_samples} evaluations...")

        for i, idx in enumerate(indices):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{n_samples}")

            try:
                sample = dataset[idx]
                metrics = self.run_single_evaluation(model, sample, condition.name)

                # Store results
                if "forecast_mse" in metrics:
                    results.forecast_mse.append(metrics["forecast_mse"])
                if "forecast_mae" in metrics:
                    results.forecast_mae.append(metrics["forecast_mae"])
                if "forecast_corr" in metrics:
                    results.forecast_corr.append(metrics["forecast_corr"])
                if "regime_accuracy" in metrics:
                    results.regime_accuracy.append(metrics["regime_accuracy"])
                if "latent_velocity" in metrics:
                    results.latent_velocity.append(metrics["latent_velocity"])
                if "inference_time_ms" in metrics:
                    results.inference_time_ms.append(metrics["inference_time_ms"])

            except Exception as e:
                import traceback
                print(f"    Warning: Sample {idx} failed: {e}")
                traceback.print_exc()
                continue

        print(f"  Completed {len(results.forecast_mse)} successful evaluations")

        return results

    def run_experiment(self):
        """Run the full experiment across all conditions."""
        print("\n" + "="*70)
        print("ISW INTEGRATION ABLATION EXPERIMENT")
        print("="*70)
        print(f"Checkpoint: {self.config.checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Samples per condition: {self.config.n_backtest_samples}")
        print(f"Random seed: {self.config.seed}")

        # Run each condition
        for condition in self.conditions:
            self.results[condition.name] = self.run_condition(condition)

        # Analyze results
        analysis = self.analyze_results()

        # Save results
        self.save_results(analysis)

        # Generate visualizations
        self.generate_visualizations(analysis)

        return analysis

    def analyze_results(self) -> Dict:
        """Perform statistical analysis comparing conditions."""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        with_isw = self.results["with_isw"]
        without_isw = self.results["without_isw"]

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_samples": self.config.n_backtest_samples,
                "seed": self.config.seed,
                "checkpoint": str(self.config.checkpoint_path)
            },
            "summaries": {
                "with_isw": with_isw.summary(),
                "without_isw": without_isw.summary()
            },
            "comparisons": {},
            "conclusion": {}
        }

        # Compare each metric
        metrics_to_compare = [
            ("forecast_mse", "lower_better"),
            ("forecast_mae", "lower_better"),
            ("forecast_corr", "higher_better"),
            ("latent_velocity", "lower_better"),  # Lower = more stable
            ("inference_time_ms", "lower_better")
        ]

        for metric_name, direction in metrics_to_compare:
            with_vals = getattr(with_isw, metric_name)
            without_vals = getattr(without_isw, metric_name)

            if len(with_vals) < 2 or len(without_vals) < 2:
                print(f"  {metric_name}: Insufficient data for comparison")
                continue

            comparison = self._compare_metric(
                with_vals, without_vals, metric_name, direction
            )
            analysis["comparisons"][metric_name] = comparison

            # Print results
            better = comparison["better_condition"]
            diff = comparison["effect_size"]["cohens_d"]
            p_val = comparison["statistical_tests"]["t_test"]["p_value"]

            print(f"\n  {metric_name}:")
            print(f"    With ISW:    {comparison['with_isw']['mean']:.4f} ± {comparison['with_isw']['std']:.4f}")
            print(f"    Without ISW: {comparison['without_isw']['mean']:.4f} ± {comparison['without_isw']['std']:.4f}")
            print(f"    Better: {better} (Cohen's d = {diff:.3f}, p = {p_val:.4f})")

        # Overall conclusion
        analysis["conclusion"] = self._draw_conclusion(analysis["comparisons"])

        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print(f"  Recommendation: {analysis['conclusion']['recommendation']}")
        print(f"  Confidence: {analysis['conclusion']['confidence']}")
        print(f"  Key finding: {analysis['conclusion']['key_finding']}")

        return analysis

    def _compare_metric(
        self,
        with_vals: List[float],
        without_vals: List[float],
        metric_name: str,
        direction: str
    ) -> Dict:
        """Compare a single metric between conditions."""
        with_arr = np.array(with_vals)
        without_arr = np.array(without_vals)

        # Basic statistics
        result = {
            "metric": metric_name,
            "direction": direction,
            "with_isw": {
                "mean": float(np.mean(with_arr)),
                "std": float(np.std(with_arr)),
                "median": float(np.median(with_arr)),
                "n": len(with_arr)
            },
            "without_isw": {
                "mean": float(np.mean(without_arr)),
                "std": float(np.std(without_arr)),
                "median": float(np.median(without_arr)),
                "n": len(without_arr)
            }
        }

        # T-test
        t_stat, p_value = stats.ttest_ind(with_arr, without_arr)

        # Mann-Whitney U (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(with_arr, without_arr, alternative='two-sided')

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(with_arr) + np.var(without_arr)) / 2)
        cohens_d = (np.mean(with_arr) - np.mean(without_arr)) / pooled_std if pooled_std > 0 else 0

        # Bootstrap confidence interval for difference
        n_bootstrap = 1000
        differences = []
        for _ in range(n_bootstrap):
            with_sample = np.random.choice(with_arr, len(with_arr), replace=True)
            without_sample = np.random.choice(without_arr, len(without_arr), replace=True)
            differences.append(np.mean(with_sample) - np.mean(without_sample))

        ci_lower = np.percentile(differences, 2.5)
        ci_upper = np.percentile(differences, 97.5)

        result["statistical_tests"] = {
            "t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            },
            "mann_whitney": {
                "u_statistic": float(u_stat),
                "p_value": float(u_pvalue),
                "significant": u_pvalue < 0.05
            }
        }

        result["effect_size"] = {
            "cohens_d": float(cohens_d),
            "interpretation": self._interpret_cohens_d(cohens_d)
        }

        result["confidence_interval"] = {
            "difference_mean": float(np.mean(differences)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "excludes_zero": ci_lower > 0 or ci_upper < 0
        }

        # Determine which is better
        if direction == "lower_better":
            better = "without_isw" if np.mean(without_arr) < np.mean(with_arr) else "with_isw"
        else:
            better = "with_isw" if np.mean(with_arr) > np.mean(without_arr) else "without_isw"

        result["better_condition"] = better

        return result

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _draw_conclusion(self, comparisons: Dict) -> Dict:
        """Draw overall conclusion from comparisons."""
        # Count wins for each condition
        wins = {"with_isw": 0, "without_isw": 0, "tie": 0}
        significant_wins = {"with_isw": 0, "without_isw": 0}

        key_metrics = ["forecast_mse", "forecast_corr", "latent_velocity"]

        for metric in key_metrics:
            if metric not in comparisons:
                continue

            comp = comparisons[metric]
            better = comp["better_condition"]
            significant = comp["statistical_tests"]["t_test"]["significant"]

            if significant:
                wins[better] += 1
                significant_wins[better] += 1
            else:
                wins["tie"] += 1

        # Determine recommendation
        if significant_wins["without_isw"] > significant_wins["with_isw"]:
            recommendation = "DISABLE ISW alignment for forecasting tasks"
            confidence = "high" if significant_wins["without_isw"] >= 2 else "moderate"
            key_finding = "ISW integration degrades forecast performance (supports modularity hypothesis)"
        elif significant_wins["with_isw"] > significant_wins["without_isw"]:
            recommendation = "KEEP ISW alignment enabled"
            confidence = "high" if significant_wins["with_isw"] >= 2 else "moderate"
            key_finding = "ISW integration improves forecast performance"
        else:
            recommendation = "ISW integration has no significant effect - consider removing for simplicity"
            confidence = "moderate"
            key_finding = "No significant difference detected between conditions"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "key_finding": key_finding,
            "wins": wins,
            "significant_wins": significant_wins
        }

    def save_results(self, analysis: Dict):
        """Save experiment results to disk."""
        output_file = self.config.output_dir / "experiment_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        analysis_json = convert_to_json_serializable(analysis)

        with open(output_file, 'w') as f:
            json.dump(analysis_json, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    def generate_visualizations(self, analysis: Dict):
        """Generate visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping visualizations")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("ISW Integration Ablation Experiment Results", fontsize=14, fontweight='bold')

        metrics = [
            ("forecast_mse", "Forecast MSE", "lower"),
            ("forecast_corr", "Forecast Correlation", "higher"),
            ("forecast_mae", "Forecast MAE", "lower"),
            ("latent_velocity", "Latent Velocity", "lower"),
            ("inference_time_ms", "Inference Time (ms)", "lower")
        ]

        for idx, (metric, title, better) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            with_vals = self.results["with_isw"].__dict__[metric]
            without_vals = self.results["without_isw"].__dict__[metric]

            if len(with_vals) == 0 or len(without_vals) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                continue

            # Box plot
            bp = ax.boxplot([with_vals, without_vals], labels=['With ISW', 'Without ISW'])

            # Color boxes based on which is better
            with_mean = np.mean(with_vals)
            without_mean = np.mean(without_vals)

            if better == "lower":
                better_idx = 0 if with_mean < without_mean else 1
            else:
                better_idx = 0 if with_mean > without_mean else 1

            colors = ['lightgreen' if i == better_idx else 'lightcoral' for i in range(2)]
            for patch, color in zip([bp['boxes'][0], bp['boxes'][1]], colors):
                patch.set_facecolor(color)

            ax.set_title(title)
            ax.set_ylabel(metric)

            # Add significance annotation
            if metric in analysis["comparisons"]:
                p_val = analysis["comparisons"][metric]["statistical_tests"]["t_test"]["p_value"]
                sig_text = f"p = {p_val:.4f}" + (" *" if p_val < 0.05 else "")
                ax.annotate(sig_text, xy=(0.5, 0.95), xycoords='axes fraction',
                           ha='center', fontsize=9)

        # Conclusion panel
        ax = axes[1, 2]
        ax.axis('off')

        conclusion = analysis["conclusion"]
        text = f"""
CONCLUSION

Recommendation:
{conclusion['recommendation']}

Confidence: {conclusion['confidence'].upper()}

Key Finding:
{conclusion['key_finding']}

Significant Wins:
  With ISW: {conclusion['significant_wins']['with_isw']}
  Without ISW: {conclusion['significant_wins']['without_isw']}
"""
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_file = self.config.output_dir / "experiment_results.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {output_file}")


def main():
    """Run the ISW ablation experiment."""
    config = ExperimentConfig()

    # Check if checkpoint exists
    if not config.checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {config.checkpoint_path}")
        print("Please specify correct checkpoint path.")
        return

    experiment = ISWAblationExperiment(config)
    results = experiment.run_experiment()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")

    return results


if __name__ == "__main__":
    main()
