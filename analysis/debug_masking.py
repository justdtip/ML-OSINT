#!/usr/bin/env python3
"""Debug script to trace MISSING_VALUE handling through the pipeline."""

import sys
import torch
import warnings

# Disable torch anomaly detection to reduce noise
torch.autograd.set_detect_anomaly(False)

from train_multi_resolution import (
    create_multi_resolution_dataloaders,
    MultiResolutionConfig,
    MISSING_VALUE,
)
from multi_resolution_han import create_multi_resolution_han


def check_tensor_stats(tensor: torch.Tensor, name: str) -> dict:
    """Compute tensor statistics for debugging."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    has_missing = (tensor == MISSING_VALUE).any().item()
    extreme_count = (tensor.abs() > 100).sum().item()

    return {
        'name': name,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'has_missing': has_missing,
        'extreme_count': extreme_count,
        'min': tensor.min().item() if not has_nan else float('nan'),
        'max': tensor.max().item() if not has_nan else float('nan'),
        'mean': tensor.mean().item() if not has_nan else float('nan'),
    }


def main():
    print("=" * 80)
    print("DEBUG: MISSING_VALUE Handling Trace")
    print("=" * 80)

    # Create minimal config
    config = MultiResolutionConfig(
        daily_seq_len=180,
        monthly_seq_len=6,
        prediction_horizon=1,
    )

    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, test_loader, norm_stats = create_multi_resolution_dataloaders(
        config=config,
        batch_size=2,
        num_workers=0,
    )

    # Get one batch
    print("\n[2] Getting first batch...")
    batch = next(iter(train_loader))

    # Check raw batch features before device move
    print("\n[3] Raw batch features (before _move_batch_to_device):")
    for source_name, tensor in batch['daily_features'].items():
        stats = check_tensor_stats(tensor, f"daily_features.{source_name}")
        print(f"  {source_name}: shape={stats['shape']}, dtype={stats['dtype']}, "
              f"missing={stats['has_missing']}, extreme={stats['extreme_count']}, "
              f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")

    print("\n[4] Raw batch masks (before _move_batch_to_device):")
    for source_name, tensor in batch['daily_masks'].items():
        obs_rate = tensor.float().mean().item()
        print(f"  {source_name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"obs_rate={obs_rate:.2%}")

    # Simulate _move_batch_to_device
    print("\n[5] Simulating _move_batch_to_device...")
    device = 'cpu'
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    tensor = v.to(device)
                    if key in ('daily_features', 'monthly_features'):
                        mask_key = key.replace('features', 'masks')
                        if mask_key in batch and k in batch[mask_key]:
                            mask = batch[mask_key][k].to(device)
                            print(f"  Applying mask to {key}.{k}: mask.shape={tuple(mask.shape)}, "
                                  f"mask.dtype={mask.dtype}")
                            before_extreme = (tensor.abs() > 100).sum().item()
                            tensor = tensor.masked_fill(~mask, 0.0)
                            after_extreme = (tensor.abs() > 100).sum().item()
                            print(f"    Extreme values: {before_extreme} -> {after_extreme}")
                    moved[key][k] = tensor
                else:
                    moved[key][k] = v
        else:
            moved[key] = value

    print("\n[6] Features after _move_batch_to_device:")
    for source_name, tensor in moved['daily_features'].items():
        stats = check_tensor_stats(tensor, f"daily_features.{source_name}")
        print(f"  {source_name}: missing={stats['has_missing']}, extreme={stats['extreme_count']}, "
              f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")

    # Create model
    print("\n[7] Creating model...")
    model = create_multi_resolution_han(
        daily_sources=['equipment', 'personnel', 'deepstate', 'firms', 'viirs'],
        monthly_sources=['sentinel'],
        d_model=128,
        n_heads=4,
    )

    # Test the forward pass with detailed instrumentation
    print("\n[8] Testing forward pass with instrumentation...")

    # Patch DailySourceEncoder.forward to add debug output
    original_forward = model.daily_encoders['equipment'].forward

    def debug_forward(features, observation_mask, return_attention=False):
        print(f"\n  [DailySourceEncoder.forward] equipment:")
        print(f"    features: shape={tuple(features.shape)}, dtype={features.dtype}")
        print(f"    observation_mask: shape={tuple(observation_mask.shape)}, dtype={observation_mask.dtype}")

        # Check for MISSING_VALUE before masking
        missing_count = (features == MISSING_VALUE).sum().item()
        extreme_count = (features.abs() > 100).sum().item()
        print(f"    BEFORE masking: missing_count={missing_count}, extreme_count={extreme_count}")

        # The fix in the model
        features_clean = features.clone()
        features_clean = features_clean.masked_fill(~observation_mask, 0.0)

        missing_after = (features_clean == MISSING_VALUE).sum().item()
        extreme_after = (features_clean.abs() > 100).sum().item()
        print(f"    AFTER masking: missing_count={missing_after}, extreme_count={extreme_after}")

        # Check where the remaining extreme values are
        if extreme_after > 0:
            extreme_mask = features_clean.abs() > 100
            extreme_indices = torch.nonzero(extreme_mask, as_tuple=False)
            print(f"    Extreme value locations (first 5): {extreme_indices[:5].tolist()}")

            # Check if observation_mask is True at those locations
            for idx in extreme_indices[:5]:
                b, t, f = idx.tolist()
                mask_val = observation_mask[b, t, f].item()
                feat_val = features[b, t, f].item()
                print(f"      [{b},{t},{f}]: feat={feat_val:.1f}, mask={mask_val}")

        # Call original
        return original_forward(features, observation_mask, return_attention)

    model.daily_encoders['equipment'].forward = debug_forward

    # Run forward pass
    print("\n[9] Running forward pass...")
    try:
        with torch.no_grad():
            outputs = model(
                daily_features=moved['daily_features'],
                daily_masks=moved['daily_masks'],
                monthly_features=moved['monthly_features'],
                monthly_masks=moved['monthly_masks'],
                month_boundaries=moved['month_boundary_indices'],
            )
        print("\n[10] Forward pass SUCCEEDED!")
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                stats = check_tensor_stats(tensor, key)
                print(f"  {key}: shape={stats['shape']}, has_nan={stats['has_nan']}")
    except Exception as e:
        print(f"\n[10] Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
