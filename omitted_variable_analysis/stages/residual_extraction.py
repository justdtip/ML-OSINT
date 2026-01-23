#!/usr/bin/env python3
"""
Stage 1: Residual Extraction

Extracts prediction residuals from the trained hybrid model for each source
under both full-reconstruction and masked-reconstruction conditions.

The residuals encode information about omitted variables - systematic structure
in residuals reveals properties of missing information.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add parent paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from copy import deepcopy
from unified_interpolation_hybrid import (
    SOURCE_CONFIGS,
    CrossSourceDatasetHybrid,
    UnifiedInterpolationModelHybrid,
    MODEL_DIR
)


@dataclass
class ResidualExtractionResults:
    """Container for residual extraction outputs."""
    full_residuals: Dict[str, np.ndarray]  # source -> (n_samples, n_features)
    masked_residuals: Dict[str, np.ndarray]  # source -> (n_samples, n_features)
    timestamps: List[str]  # Date strings
    feature_names: Dict[str, List[str]]  # source -> feature names
    metadata: Dict = field(default_factory=dict)

    def save(self, output_dir: Path):
        """Save results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save residuals as npz
        np.savez(
            output_dir / 'full_residuals.npz',
            **{f'{k}': v for k, v in self.full_residuals.items()}
        )
        np.savez(
            output_dir / 'masked_residuals.npz',
            **{f'{k}': v for k, v in self.masked_residuals.items()}
        )

        # Save metadata
        metadata = {
            'timestamps': self.timestamps,
            'feature_names': self.feature_names,
            'extraction_info': self.metadata,
            'sources': list(self.full_residuals.keys()),
            'shapes': {k: list(v.shape) for k, v in self.full_residuals.items()}
        }
        with open(output_dir / 'residual_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved residuals to {output_dir}")

    @classmethod
    def load(cls, output_dir: Path) -> 'ResidualExtractionResults':
        """Load results from disk."""
        output_dir = Path(output_dir)

        full_data = np.load(output_dir / 'full_residuals.npz')
        full_residuals = {k: full_data[k] for k in full_data.files}

        masked_data = np.load(output_dir / 'masked_residuals.npz')
        masked_residuals = {k: masked_data[k] for k in masked_data.files}

        with open(output_dir / 'residual_metadata.json') as f:
            metadata = json.load(f)

        return cls(
            full_residuals=full_residuals,
            masked_residuals=masked_residuals,
            timestamps=metadata['timestamps'],
            feature_names=metadata['feature_names'],
            metadata=metadata.get('extraction_info', {})
        )


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS has compatibility issues with nested tensor operations in transformers
    # Fall back to CPU for reliable execution
    return torch.device('cpu')


def load_model_and_data(
    checkpoint_path: Optional[Path] = None,
    temporal_gap: int = 7,
    use_full_data: bool = True
) -> Tuple[UnifiedInterpolationModelHybrid, CrossSourceDatasetHybrid, torch.device]:
    """
    Load trained hybrid model and evaluation dataset.

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses default.
        temporal_gap: Days gap between train/val (for proper temporal split)
        use_full_data: If True, returns dataset covering all samples for residual extraction

    Returns:
        model, dataset, device
    """
    device = get_device()
    print(f"Using device: {device}")

    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = MODEL_DIR / 'unified_interpolation_hybrid_best.pt'

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create fresh configs and detect feature dimensions from saved weights
    configs = deepcopy(SOURCE_CONFIGS)
    for name in configs:
        encoder_key = f'encoders.{name}.feature_proj.0.weight'
        if encoder_key in state_dict:
            n_features = state_dict[encoder_key].shape[1]
            configs[name].n_features = n_features
            print(f"  {name}: {n_features} features (from checkpoint)")

    # Create model
    model = UnifiedInterpolationModelHybrid(
        source_configs=configs,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create dataset for evaluation
    # We need training stats for proper normalization
    print("\nLoading evaluation data...")
    train_dataset = CrossSourceDatasetHybrid(
        configs,
        train=True,
        exclude_clown_units=True,
        temporal_gap=temporal_gap
    )

    if use_full_data:
        # Create a dataset that covers ALL samples (for full residual extraction)
        # We use val_ratio=0.0 to get all data, but still use training norm stats
        eval_dataset = CrossSourceDatasetHybrid(
            configs,
            train=True,  # Gets all data before split
            val_ratio=0.0,  # No validation split
            exclude_clown_units=True,
            temporal_gap=0,
            norm_stats=train_dataset.norm_stats
        )
        # Override indices to include ALL samples
        eval_dataset.indices = list(range(eval_dataset.n_samples))
        print(f"  Using ALL {len(eval_dataset)} samples for residual extraction")
    else:
        # Just use validation set
        eval_dataset = CrossSourceDatasetHybrid(
            configs,
            train=False,
            exclude_clown_units=True,
            temporal_gap=temporal_gap,
            norm_stats=train_dataset.norm_stats
        )
        print(f"  Using {len(eval_dataset)} validation samples")

    return model, eval_dataset, device


def extract_residuals(
    model: UnifiedInterpolationModelHybrid,
    dataset: CrossSourceDatasetHybrid,
    device: torch.device,
    batch_size: int = 64
) -> ResidualExtractionResults:
    """
    Extract residuals for full and masked reconstruction.

    For full reconstruction: residual = target - reconstruction(all sources)
    For masked reconstruction: residual = target - reconstruction(other sources only)

    Args:
        model: Trained hybrid model
        dataset: Evaluation dataset
        device: Torch device
        batch_size: Batch size for inference

    Returns:
        ResidualExtractionResults containing full and masked residuals
    """
    source_names = model.source_names
    n_samples = len(dataset)

    print(f"\nExtracting residuals for {n_samples} samples...")
    print(f"  Sources: {source_names}")

    # Initialize storage
    full_residuals = {name: [] for name in source_names}
    masked_residuals = {name: [] for name in source_names}

    # Create dataloader
    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_size_actual = next(iter(batch.values())).size(0)

            # Move to device
            features = {k: v.to(device) for k, v in batch.items()}

            # --- Full Reconstruction ---
            # All sources visible, compute reconstruction
            outputs = model(features, mask=None, return_reconstructions=True)

            for source in source_names:
                target = features[source].cpu().numpy()
                pred = outputs['reconstructions'][source].cpu().numpy()
                residual = target - pred
                full_residuals[source].append(residual)

            # --- Masked Reconstruction ---
            # For each source, mask it out and predict from others
            for masked_source in source_names:
                # Create mask: 0 for masked source, 1 for others
                mask = {}
                for name in source_names:
                    if name == masked_source:
                        mask[name] = torch.zeros(batch_size_actual, device=device)
                    else:
                        mask[name] = torch.ones(batch_size_actual, device=device)

                # Forward with mask
                masked_outputs = model(features, mask=mask, return_reconstructions=True)

                # Compute residual for the masked source
                target = features[masked_source].cpu().numpy()
                pred = masked_outputs['reconstructions'][masked_source].cpu().numpy()
                residual = target - pred
                masked_residuals[masked_source].append(residual)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} / {n_samples} samples")

    # Concatenate batches
    full_residuals = {s: np.concatenate(v, axis=0) for s, v in full_residuals.items()}
    masked_residuals = {s: np.concatenate(v, axis=0) for s, v in masked_residuals.items()}

    # Get timestamps
    # Use first available source's dates
    timestamps = None
    for name in source_names:
        if name in dataset.source_dates and dataset.source_dates[name]:
            # Get timestamps for the indices we actually used
            all_dates = dataset.source_dates[name]
            timestamps = [all_dates[i] for i in dataset.indices]
            break

    if timestamps is None:
        print("  WARNING: No timestamps available, using sequential indices")
        timestamps = [f"day_{i}" for i in range(n_samples)]

    # Get feature names
    feature_names = dataset.feature_names

    # Compute metadata / validation checks
    metadata = {
        'extraction_timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'device': str(device),
        'validation_checks': {}
    }

    # Validation: residuals should have approximately zero mean
    print("\n--- Validation Checks ---")
    for source in source_names:
        full_mean = np.mean(full_residuals[source])
        full_var = np.var(full_residuals[source])
        masked_mean = np.mean(masked_residuals[source])
        masked_var = np.var(masked_residuals[source])

        # Input variance (from normalized data, should be ~1)
        # Residual variance should be less than input variance

        metadata['validation_checks'][source] = {
            'full_residual_mean': float(full_mean),
            'full_residual_var': float(full_var),
            'masked_residual_mean': float(masked_mean),
            'masked_residual_var': float(masked_var),
            'has_nan': bool(np.isnan(full_residuals[source]).any() or
                          np.isnan(masked_residuals[source]).any()),
            'has_inf': bool(np.isinf(full_residuals[source]).any() or
                          np.isinf(masked_residuals[source]).any())
        }

        print(f"  {source}:")
        print(f"    Full recon:   mean={full_mean:+.4f}, var={full_var:.4f}")
        print(f"    Masked recon: mean={masked_mean:+.4f}, var={masked_var:.4f}")

        if abs(full_mean) > 0.1:
            print(f"    WARNING: Full residual mean > 0.1 (potential bias)")
        if metadata['validation_checks'][source]['has_nan']:
            print(f"    WARNING: NaN values detected!")
        if metadata['validation_checks'][source]['has_inf']:
            print(f"    WARNING: Inf values detected!")

    return ResidualExtractionResults(
        full_residuals=full_residuals,
        masked_residuals=masked_residuals,
        timestamps=timestamps,
        feature_names=feature_names,
        metadata=metadata
    )


def main():
    """Run Stage 1: Residual Extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 1: Residual Extraction')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'results'),
                       help='Output directory for results')
    parser.add_argument('--temporal-gap', type=int, default=7,
                       help='Days gap between train/val')
    parser.add_argument('--val-only', action='store_true',
                       help='Extract residuals only from validation set')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 1: RESIDUAL EXTRACTION")
    print("=" * 70)
    print("\nExtracting prediction residuals from trained hybrid model.")
    print("Residuals encode information about omitted variables.\n")

    # Load model and data
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model, dataset, device = load_model_and_data(
        checkpoint_path=checkpoint_path,
        temporal_gap=args.temporal_gap,
        use_full_data=not args.val_only
    )

    # Extract residuals
    results = extract_residuals(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size
    )

    # Save results
    output_dir = Path(args.output_dir)
    results.save(output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("RESIDUAL EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nExtracted residuals for {results.metadata['n_samples']} samples")
    print(f"Sources: {list(results.full_residuals.keys())}")
    print(f"\nOutput saved to: {output_dir}")
    print("\nNext step: Run Stage 2 (Temporal Structure Analysis)")


if __name__ == '__main__':
    main()
