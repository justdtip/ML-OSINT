#!/usr/bin/env python3
"""
Diagnostic Script: Debug NaN Loss in Multi-Resolution HAN Training

This script traces the data flow through the model layer-by-layer to identify
the exact location where NaN values first appear.

Key Finding: The MISSING_VALUE sentinel (-999.0) is being fed into the feature
projection layer BEFORE the mask is applied, causing extreme values that
propagate through the network and eventually produce NaN in softmax/attention.

Root Cause Analysis:
===================
1. In multi_resolution_data.py (lines 1418-1419):
   - NaN values are replaced with MISSING_VALUE (-999.0)
   - This happens AFTER normalization: `normalized = np.where(np.isnan(normalized), self.config.missing_value, normalized)`

2. In DailySourceEncoder.forward() (lines 302-319):
   - Line 306: `hidden = self.feature_projection(features)` - projects RAW features including -999.0 values
   - Line 312: `timestep_observed = observation_mask.any(dim=-1)` - creates timestep mask
   - Line 319: `hidden = torch.where(obs_mask_expanded, hidden, no_obs_expanded)` - replaces unobserved timesteps

   THE BUG: The feature projection at line 306 processes the -999.0 values BEFORE
   they are masked out at line 319. With a linear projection of ~128 features,
   -999.0 * weights can produce values on the order of tens of thousands,
   which cause numerical instability.

3. Even though line 319 replaces unobserved timesteps with no_observation_token,
   the OBSERVED timesteps may still have individual FEATURES with -999.0 values.
   The `timestep_observed = observation_mask.any(dim=-1)` marks a timestep as observed
   if ANY feature is observed, but individual features within that timestep may still
   be -999.0.

4. This propagates through:
   - Feature projection creates extreme values
   - These extreme values flow to attention scores
   - Softmax of extreme values -> NaN
"""

import os
import sys
from pathlib import Path

# Enable MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    multi_resolution_collate_fn,
    MISSING_VALUE,
)
from multi_resolution_han import (
    create_multi_resolution_han,
    DailySourceEncoder,
    SourceConfig,
)

def diagnose_batch(batch, device='cpu'):
    """
    Diagnose a single batch for NaN-causing values.

    Returns a detailed report of where problematic values occur.
    """
    report = []

    report.append("=" * 80)
    report.append("BATCH DIAGNOSTIC REPORT")
    report.append("=" * 80)

    # Check daily features for MISSING_VALUE
    report.append("\n[1] DAILY FEATURES - Checking for MISSING_VALUE (-999.0):")
    for source_name, features in batch['daily_features'].items():
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features

        n_missing = (features_np == MISSING_VALUE).sum()
        total = features_np.size
        pct_missing = 100 * n_missing / total

        # Check for extreme values
        features_clean = features_np[features_np != MISSING_VALUE]
        if len(features_clean) > 0:
            vmin, vmax = features_clean.min(), features_clean.max()
        else:
            vmin, vmax = np.nan, np.nan

        report.append(f"  {source_name}:")
        report.append(f"    Shape: {features_np.shape}")
        report.append(f"    Missing values: {n_missing}/{total} ({pct_missing:.1f}%)")
        report.append(f"    Value range (excluding missing): [{vmin:.2f}, {vmax:.2f}]")

        # Check mask alignment
        mask = batch['daily_masks'][source_name]
        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

        # Count positions where mask=True but value=-999
        bad_positions = ((mask_np == True) & (features_np == MISSING_VALUE)).sum()
        if bad_positions > 0:
            report.append(f"    *** WARNING: {bad_positions} positions have mask=True but value=-999.0")

    # Check monthly features
    report.append("\n[2] MONTHLY FEATURES - Checking for MISSING_VALUE (-999.0):")
    for source_name, features in batch['monthly_features'].items():
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features

        n_missing = (features_np == MISSING_VALUE).sum()
        total = features_np.size
        pct_missing = 100 * n_missing / total

        report.append(f"  {source_name}:")
        report.append(f"    Shape: {features_np.shape}")
        report.append(f"    Missing values: {n_missing}/{total} ({pct_missing:.1f}%)")

    return "\n".join(report)


def trace_forward_pass(model, batch, device='cpu'):
    """
    Trace the forward pass layer-by-layer to find where NaN appears.
    """
    report = []
    report.append("\n" + "=" * 80)
    report.append("FORWARD PASS TRACE")
    report.append("=" * 80)

    model = model.to(device)
    model.eval()

    # Move batch to device
    daily_features = {k: v.to(device) for k, v in batch['daily_features'].items()}
    daily_masks = {k: v.to(device) for k, v in batch['daily_masks'].items()}
    monthly_features = {k: v.to(device) for k, v in batch['monthly_features'].items()}
    monthly_masks = {k: v.to(device) for k, v in batch['monthly_masks'].items()}
    month_boundaries = batch['month_boundary_indices'].to(device)

    def check_tensor(name, tensor):
        """Check a tensor for NaN/Inf and report."""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        tmin = tensor.min().item() if not has_nan else float('nan')
        tmax = tensor.max().item() if not has_nan else float('nan')
        tmean = tensor.mean().item() if not has_nan else float('nan')

        status = "OK"
        if has_nan:
            status = "*** NaN DETECTED ***"
        elif has_inf:
            status = "*** Inf DETECTED ***"
        elif abs(tmin) > 1000 or abs(tmax) > 1000:
            status = "** EXTREME VALUES **"

        return f"  {name}: shape={list(tensor.shape)}, range=[{tmin:.4f}, {tmax:.4f}], mean={tmean:.4f} {status}", has_nan

    nan_found = False
    first_nan_location = None

    report.append("\n[STEP 1] Daily Source Encoding")
    report.append("-" * 40)

    for source_name, encoder in model.daily_encoders.items():
        if source_name not in daily_features:
            continue

        features = daily_features[source_name]
        mask = daily_masks[source_name]

        report.append(f"\n  Source: {source_name}")

        # Check input
        line, has_nan = check_tensor(f"    Input features", features)
        report.append(line)
        if has_nan and not nan_found:
            first_nan_location = f"Daily encoder {source_name} INPUT"
            nan_found = True

        # Check for -999.0 values
        n_sentinel = (features == MISSING_VALUE).sum().item()
        if n_sentinel > 0:
            report.append(f"    *** SENTINEL VALUES: {n_sentinel} positions have -999.0")

        # Step through encoder manually
        with torch.no_grad():
            # 1. Feature projection (THIS IS WHERE THE BUG IS)
            hidden = encoder.feature_projection(features)
            line, has_nan = check_tensor("    After feature_projection", hidden)
            report.append(line)
            if has_nan and not nan_found:
                first_nan_location = f"Daily encoder {source_name} FEATURE_PROJECTION"
                nan_found = True

            # Check specifically: what happens to -999.0 through projection?
            if n_sentinel > 0:
                # Find a position with -999.0
                sentinel_mask = (features == MISSING_VALUE)
                # Get first batch item with sentinel
                for b in range(features.shape[0]):
                    if sentinel_mask[b].any():
                        # Get indices where sentinel exists
                        t_idx, f_idx = torch.where(sentinel_mask[b])
                        if len(t_idx) > 0:
                            t, f = t_idx[0].item(), f_idx[0].item()
                            input_val = features[b, t, :].clone()
                            output_val = hidden[b, t, :]
                            report.append(f"    Debug: At position [batch={b}, t={t}]:")
                            report.append(f"      Input has {sentinel_mask[b, t].sum().item()} sentinel features")
                            report.append(f"      Projected output range: [{output_val.min().item():.2f}, {output_val.max().item():.2f}]")
                            break

            # 2. Timestep observed mask
            timestep_observed = mask.any(dim=-1)
            n_observed = timestep_observed.sum().item()
            n_total = timestep_observed.numel()
            report.append(f"    Timesteps observed: {n_observed}/{n_total} ({100*n_observed/n_total:.1f}%)")

            # 3. Replace unobserved with no_observation_token
            obs_mask_expanded = timestep_observed.unsqueeze(-1)
            no_obs_expanded = encoder.no_observation_token.expand(hidden.shape[0], hidden.shape[1], -1)
            hidden_masked = torch.where(obs_mask_expanded, hidden, no_obs_expanded)

            line, has_nan = check_tensor("    After no_obs masking", hidden_masked)
            report.append(line)
            if has_nan and not nan_found:
                first_nan_location = f"Daily encoder {source_name} AFTER_MASKING"
                nan_found = True

            # Check: Are there still extreme values in OBSERVED positions?
            observed_vals = hidden_masked[obs_mask_expanded.expand_as(hidden_masked)]
            if len(observed_vals) > 0:
                report.append(f"    Observed positions range: [{observed_vals.min().item():.2f}, {observed_vals.max().item():.2f}]")

            # 4. Add observation status embedding
            obs_status = timestep_observed.long()
            obs_status_emb = encoder.observation_status_embedding(obs_status)
            hidden_with_status = hidden_masked + obs_status_emb

            line, has_nan = check_tensor("    After obs_status_emb", hidden_with_status)
            report.append(line)

            # 5. Positional encoding
            hidden_pos = encoder.positional_encoding(hidden_with_status)
            line, has_nan = check_tensor("    After positional_encoding", hidden_pos)
            report.append(line)

            # 6. Transformer encoding (most likely to amplify issues)
            src_key_padding_mask = ~timestep_observed
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

            try:
                hidden_encoded = encoder.transformer_encoder(
                    hidden_pos,
                    src_key_padding_mask=src_key_padding_mask,
                )
                line, has_nan = check_tensor("    After transformer_encoder", hidden_encoded)
                report.append(line)
                if has_nan and not nan_found:
                    first_nan_location = f"Daily encoder {source_name} TRANSFORMER"
                    nan_found = True
            except Exception as e:
                report.append(f"    *** TRANSFORMER FAILED: {e}")
                nan_found = True
                first_nan_location = f"Daily encoder {source_name} TRANSFORMER (exception)"

    report.append("\n" + "=" * 80)
    report.append("DIAGNOSIS SUMMARY")
    report.append("=" * 80)

    if nan_found:
        report.append(f"\n*** NaN FIRST APPEARED AT: {first_nan_location}")
        report.append("\nROOT CAUSE:")
        report.append("  The MISSING_VALUE sentinel (-999.0) is being fed into the feature")
        report.append("  projection layer BEFORE the observation mask is applied.")
        report.append("")
        report.append("  At line 306 in DailySourceEncoder.forward():")
        report.append("    hidden = self.feature_projection(features)  # -999.0 values included!")
        report.append("")
        report.append("  The masking at line 319 only replaces ENTIRE TIMESTEPS where ALL")
        report.append("  features are missing. But individual features within observed timesteps")
        report.append("  may still contain -999.0 values, which produce extreme outputs.")
        report.append("")
        report.append("SOLUTION:")
        report.append("  1. Replace -999.0 values with 0.0 (or learned missing token) BEFORE projection")
        report.append("  2. Or use masked linear projection that ignores -999.0 positions")
        report.append("  3. Or clamp the input features to a reasonable range before projection")
    else:
        report.append("\nNo NaN detected in trace. Issue may be in later layers.")

    return "\n".join(report)


def main():
    """Run the diagnostic."""
    print("=" * 80)
    print("NaN LOSS DIAGNOSTIC SCRIPT")
    print("=" * 80)

    # Create config
    config = MultiResolutionConfig(
        daily_seq_len=180,
        monthly_seq_len=6,
        prediction_horizon=1,
    )

    print("\n[1] Creating dataset...")
    try:
        train_dataset = MultiResolutionDataset(config=config, split='train')
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    print(f"Dataset created with {len(train_dataset)} samples")

    # Get one sample
    print("\n[2] Loading sample...")
    sample = train_dataset[0]

    # Convert to batch
    batch = multi_resolution_collate_fn([sample])

    # Run batch diagnostic
    print("\n[3] Analyzing batch data...")
    batch_report = diagnose_batch(batch)
    print(batch_report)

    # Create model
    print("\n[4] Creating model...")
    feature_info = train_dataset.get_feature_info()

    daily_source_configs = {
        name: SourceConfig(
            name=name,
            n_features=info['n_features'],
            resolution='daily',
        )
        for name, info in feature_info.items()
        if info['resolution'] == 'daily'
    }
    monthly_source_configs = {
        name: SourceConfig(
            name=name,
            n_features=info['n_features'],
            resolution='monthly',
        )
        for name, info in feature_info.items()
        if info['resolution'] == 'monthly'
    }

    from multi_resolution_han import MultiResolutionHAN

    model = MultiResolutionHAN(
        daily_source_configs=daily_source_configs,
        monthly_source_configs=monthly_source_configs,
        d_model=128,
        nhead=8,
        num_daily_layers=2,
        num_monthly_layers=2,
        num_fusion_layers=1,
        dropout=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")

    # Trace forward pass
    print("\n[5] Tracing forward pass...")
    trace_report = trace_forward_pass(model, batch, device='cpu')
    print(trace_report)

    # Print final summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nSee above for detailed analysis of where NaN values originate.")


if __name__ == "__main__":
    main()
