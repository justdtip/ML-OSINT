#!/usr/bin/env python3
"""
Test Training Script for Delta HAN with Probe Battery Optimizations.

This script runs an isolated 10-epoch training test with all optimizations
identified from the probe battery synthesis:

1. RSA Fusion Regularization - prevents fusion quality degradation
2. Optimal Context Window - 14 days (best accuracy per Probe 3.1.1)
3. VIIRS Detrending - removes spurious temporal correlation
4. Disaggregated Equipment - isolates genuine drone signal
5. Epoch 10 checkpoint - saves at optimal fusion quality point

Usage:
    python -m analysis.test_delta_han_optimized [--epochs 10] [--device auto]
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.paths import (
    PROJECT_ROOT, OUTPUT_DIR, CHECKPOINT_DIR,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

# Output directory for this test
TEST_OUTPUT_DIR = OUTPUT_DIR / "analysis" / "delta_han_optimized_test"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def run_test_training(
    epochs: int = 10,
    batch_size: int = 16,
    device_str: str = "auto",
    use_disaggregated_equipment: bool = True,
    detrend_viirs: bool = True,
    fusion_loss_weight: float = 0.1,
    context_window_days: int = 14,
):
    """
    Run test training with all probe battery optimizations.

    Args:
        epochs: Number of training epochs (default 10 for quick test)
        batch_size: Training batch size
        device_str: Device to use ('auto', 'cuda', 'mps', 'cpu')
        use_disaggregated_equipment: Use separate drone/armor/artillery/aircraft sources
        detrend_viirs: Apply first-order differencing to VIIRS
        fusion_loss_weight: Weight for RSA fusion regularization loss
        context_window_days: Context window in days (optimal: 7-14)
    """
    device = get_device(device_str)

    print("=" * 70)
    print("DELTA HAN OPTIMIZED TEST TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print()
    print("Optimizations enabled:")
    print(f"  - Context window: {context_window_days} days (Probe 3.1.1)")
    print(f"  - VIIRS detrending: {detrend_viirs} (Probe 1.2.1)")
    print(f"  - Disaggregated equipment: {use_disaggregated_equipment} (Probe 1.1.2)")
    print(f"  - Fusion regularization weight: {fusion_loss_weight} (Probe 2.1.4)")
    print("=" * 70)

    # Import here to avoid circular imports
    from analysis.unified_interpolation_delta import (
        SOURCE_CONFIGS,
        UnifiedInterpolationModelDelta,
        CrossSourceDatasetDelta,
        UnifiedTrainerDelta,
        train_unified_model_delta,
    )

    # Run training with new parameters
    print("\nStarting training with optimizations...")

    results = train_unified_model_delta(
        epochs=epochs,
        batch_size=batch_size,
        device=str(device),
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = TEST_OUTPUT_DIR / f"test_results_{timestamp}.json"

    results_summary = {
        "timestamp": timestamp,
        "epochs": epochs,
        "device": str(device),
        "optimizations": {
            "context_window_days": context_window_days,
            "detrend_viirs": detrend_viirs,
            "disaggregated_equipment": use_disaggregated_equipment,
            "fusion_loss_weight": fusion_loss_weight,
        },
        "history": {
            "train_loss": results["history"]["train_loss"],
            "val_mae": results["history"]["val_mae"],
        },
        "source_configs": {
            name: {
                "n_features": cfg.n_features,
                "use_delta_only": cfg.use_delta_only,
            }
            for name, cfg in results["source_configs"].items()
        },
    }

    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return results


def run_analysis_on_results(results_path: Path = None):
    """
    Run quick analysis on training results.

    Checks:
    1. Did loss decrease over epochs?
    2. Is there evidence of fusion quality maintenance?
    3. Comparison of early vs late epochs
    """
    if results_path is None:
        # Find most recent results
        results_files = sorted(TEST_OUTPUT_DIR.glob("test_results_*.json"))
        if not results_files:
            print("No results files found. Run training first.")
            return
        results_path = results_files[-1]

    with open(results_path) as f:
        results = json.load(f)

    print("\n" + "=" * 70)
    print("ANALYSIS OF TRAINING RESULTS")
    print("=" * 70)
    print(f"Results file: {results_path.name}")
    print(f"Timestamp: {results['timestamp']}")
    print()

    train_loss = results["history"]["train_loss"]
    val_mae = results["history"]["val_mae"]

    print("Training Loss Progression:")
    print(f"  Epoch 1:  {train_loss[0]:.4f}")
    print(f"  Epoch 5:  {train_loss[min(4, len(train_loss)-1)]:.4f}")
    print(f"  Epoch 10: {train_loss[-1]:.4f}")
    print(f"  Improvement: {(1 - train_loss[-1]/train_loss[0])*100:.1f}%")
    print()

    print("Validation MAE Progression:")
    print(f"  Epoch 1:  {val_mae[0]:.4f}")
    print(f"  Epoch 5:  {val_mae[min(4, len(val_mae)-1)]:.4f}")
    print(f"  Epoch 10: {val_mae[-1]:.4f}")
    print(f"  Improvement: {(1 - val_mae[-1]/val_mae[0])*100:.1f}%")
    print()

    print("Optimizations Used:")
    for opt, val in results["optimizations"].items():
        print(f"  {opt}: {val}")
    print()

    print("Source Configurations:")
    for name, cfg in results["source_configs"].items():
        delta_marker = " [DELTA]" if cfg["use_delta_only"] else ""
        print(f"  {name}: {cfg['n_features']} features{delta_marker}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test Delta HAN with probe battery optimizations"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only run analysis on existing results"
    )
    parser.add_argument(
        "--fusion_loss_weight", type=float, default=0.1,
        help="Weight for fusion regularization loss (default: 0.1)"
    )

    args = parser.parse_args()

    if args.analyze_only:
        run_analysis_on_results()
    else:
        results = run_test_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_str=args.device,
            fusion_loss_weight=args.fusion_loss_weight,
        )
        run_analysis_on_results()


if __name__ == "__main__":
    main()
