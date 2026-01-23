#!/usr/bin/env python3
"""
Extended Diagnostic: Debug NaN Loss During Training Step

This script performs a complete training step to identify where NaN appears,
including loss computation and gradient calculation.
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
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from multi_resolution_data import (
    MultiResolutionDataset,
    MultiResolutionConfig,
    multi_resolution_collate_fn,
    MISSING_VALUE,
)
from multi_resolution_han import (
    MultiResolutionHAN,
    SourceConfig,
)

def check_tensor(name, tensor):
    """Check a tensor for NaN/Inf and report."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        tmin = tensor[~torch.isnan(tensor)].min().item() if not tensor.isnan().all() else float('nan')
        tmax = tensor[~torch.isnan(tensor)].max().item() if not tensor.isnan().all() else float('nan')
        return f"  {name}: NaN={n_nan}, Inf={n_inf}, range=[{tmin:.4f}, {tmax:.4f}]", True
    else:
        tmin = tensor.min().item()
        tmax = tensor.max().item()
        tmean = tensor.mean().item()
        return f"  {name}: range=[{tmin:.4f}, {tmax:.4f}], mean={tmean:.4f}", False


def run_training_step(model, batch, device='cpu'):
    """
    Run a complete training step and trace for NaN.
    """
    report = []
    report.append("=" * 80)
    report.append("TRAINING STEP TRACE")
    report.append("=" * 80)

    model = model.to(device)
    model.train()

    # Move batch to device
    daily_features = {k: v.to(device) for k, v in batch['daily_features'].items()}
    daily_masks = {k: v.to(device) for k, v in batch['daily_masks'].items()}
    monthly_features = {k: v.to(device) for k, v in batch['monthly_features'].items()}
    monthly_masks = {k: v.to(device) for k, v in batch['monthly_masks'].items()}
    month_boundaries = batch['month_boundary_indices'].to(device)

    # Check all inputs
    report.append("\n[1] INPUT DATA CHECK")
    report.append("-" * 40)

    nan_in_input = False
    for source_name, features in daily_features.items():
        line, has_nan = check_tensor(f"daily_features[{source_name}]", features)
        report.append(line)
        if has_nan:
            nan_in_input = True

        # Detailed check for -999.0
        n_sentinel = (features == MISSING_VALUE).sum().item()
        mask = daily_masks[source_name]
        n_observed = mask.any(dim=-1).sum().item()

        if n_sentinel > 0:
            report.append(f"    Sentinel values: {n_sentinel}, Observed timesteps: {n_observed}/{features.shape[1]}")

            # Critical check: sentinel values at observed positions
            # This means features[mask] contains -999.0
            observed_features = features[mask]
            n_sentinel_observed = (observed_features == MISSING_VALUE).sum().item()
            if n_sentinel_observed > 0:
                report.append(f"    *** PROBLEM: {n_sentinel_observed} sentinel values in OBSERVED positions!")

    for source_name, features in monthly_features.items():
        line, has_nan = check_tensor(f"monthly_features[{source_name}]", features)
        report.append(line)
        if has_nan:
            nan_in_input = True

        n_sentinel = (features == MISSING_VALUE).sum().item()
        if n_sentinel > 0:
            mask = monthly_masks[source_name]
            observed_features = features[mask]
            n_sentinel_observed = (observed_features == MISSING_VALUE).sum().item()
            if n_sentinel_observed > 0:
                report.append(f"    *** PROBLEM: {n_sentinel_observed} sentinel values in OBSERVED positions!")

    # Forward pass
    report.append("\n[2] FORWARD PASS")
    report.append("-" * 40)

    try:
        with torch.autograd.detect_anomaly():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

        nan_in_output = False
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                line, has_nan = check_tensor(f"outputs[{key}]", value)
                report.append(line)
                if has_nan:
                    nan_in_output = True
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    if isinstance(subval, torch.Tensor):
                        line, has_nan = check_tensor(f"outputs[{key}][{subkey}]", subval)
                        report.append(line)
                        if has_nan:
                            nan_in_output = True

    except RuntimeError as e:
        report.append(f"  *** FORWARD PASS FAILED: {e}")
        return "\n".join(report)

    # Loss computation
    report.append("\n[3] LOSS COMPUTATION")
    report.append("-" * 40)

    batch_size = batch['batch_size']

    # Simulate the loss computation from train_multi_resolution.py
    losses = {}

    # Regime loss
    regime_logits = outputs['regime_logits']  # [batch, seq, n_classes]
    regime_logits_last = regime_logits[:, -1, :]  # [batch, n_classes]
    regime_targets = torch.randint(0, regime_logits.size(-1), (batch_size,), device=device)

    line, has_nan = check_tensor("regime_logits_last", regime_logits_last)
    report.append(line)

    try:
        losses['regime'] = F.cross_entropy(regime_logits_last, regime_targets)
        report.append(f"  regime_loss: {losses['regime'].item():.6f}")
    except Exception as e:
        report.append(f"  *** regime_loss FAILED: {e}")

    # Casualty loss
    casualty_pred = outputs['casualty_pred']  # [batch, seq, 3]
    casualty_var = outputs['casualty_var']  # [batch, seq, 3]
    casualty_pred_last = casualty_pred[:, -1, :]
    casualty_var_last = casualty_var[:, -1, :]

    line, has_nan = check_tensor("casualty_pred_last", casualty_pred_last)
    report.append(line)
    line, has_nan = check_tensor("casualty_var_last", casualty_var_last)
    report.append(line)

    # Gaussian NLL loss
    try:
        # Clamp variance to prevent division by zero
        var_clamped = casualty_var_last.clamp(min=1e-6)
        target = torch.randn_like(casualty_pred_last)  # Synthetic target
        nll = 0.5 * (torch.log(var_clamped) + (casualty_pred_last - target).pow(2) / var_clamped)

        line, has_nan = check_tensor("casualty_nll (before mean)", nll)
        report.append(line)

        losses['casualty'] = nll.mean()
        report.append(f"  casualty_loss: {losses['casualty'].item():.6f}")
    except Exception as e:
        report.append(f"  *** casualty_loss FAILED: {e}")

    # Anomaly loss
    anomaly_score = outputs['anomaly_score']  # [batch, seq, 1]
    anomaly_score_last = anomaly_score[:, -1, 0]

    line, has_nan = check_tensor("anomaly_score_last", anomaly_score_last)
    report.append(line)

    try:
        losses['anomaly'] = anomaly_score_last.pow(2).mean()
        report.append(f"  anomaly_loss: {losses['anomaly'].item():.6f}")
    except Exception as e:
        report.append(f"  *** anomaly_loss FAILED: {e}")

    # Forecast loss
    forecast_pred = outputs['forecast_pred']  # [batch, seq, n_features]
    try:
        losses['forecast'] = forecast_pred.var(dim=1).mean() * 0.01
        report.append(f"  forecast_loss: {losses['forecast'].item():.6f}")
    except Exception as e:
        report.append(f"  *** forecast_loss FAILED: {e}")

    # Total loss
    report.append("\n[4] TOTAL LOSS AND BACKWARD")
    report.append("-" * 40)

    total_loss = sum(losses.values())
    report.append(f"  total_loss: {total_loss.item():.6f}")

    if torch.isnan(total_loss):
        report.append("  *** TOTAL LOSS IS NaN")
        # Find which loss is NaN
        for name, loss in losses.items():
            if torch.isnan(loss):
                report.append(f"    NaN source: {name}")
    else:
        # Try backward pass
        try:
            total_loss.backward()
            report.append("  backward() succeeded")

            # Check gradients
            grad_nan = False
            grad_inf = False
            max_grad = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        report.append(f"    NaN gradient in: {name}")
                        grad_nan = True
                    if torch.isinf(param.grad).any():
                        report.append(f"    Inf gradient in: {name}")
                        grad_inf = True
                    max_grad = max(max_grad, param.grad.abs().max().item())

            if not grad_nan and not grad_inf:
                report.append(f"  Gradients OK, max: {max_grad:.4f}")
        except RuntimeError as e:
            report.append(f"  *** BACKWARD FAILED: {e}")

    # Summary
    report.append("\n" + "=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    if nan_in_input:
        report.append("  - NaN detected in INPUT data")
    if nan_in_output:
        report.append("  - NaN detected in MODEL OUTPUT")
    if torch.isnan(total_loss):
        report.append("  - NaN detected in LOSS")
    else:
        report.append("  - Training step completed successfully")

    return "\n".join(report)


def test_multiple_batches():
    """Test multiple batches to find one that causes NaN."""
    print("=" * 80)
    print("TESTING MULTIPLE BATCHES FOR NaN")
    print("=" * 80)

    # Create config
    config = MultiResolutionConfig(
        daily_seq_len=180,
        monthly_seq_len=6,
        prediction_horizon=1,
    )

    print("\n[1] Creating dataset...")
    train_dataset = MultiResolutionDataset(config=config, split='train')
    print(f"Dataset created with {len(train_dataset)} samples")

    # Create model
    print("\n[2] Creating model...")
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

    # Test each sample
    print("\n[3] Testing each sample...")
    for i in range(min(len(train_dataset), 10)):
        sample = train_dataset[i]
        batch = multi_resolution_collate_fn([sample])

        model.zero_grad()

        # Quick check
        try:
            daily_features = {k: v for k, v in batch['daily_features'].items()}
            daily_masks = {k: v for k, v in batch['daily_masks'].items()}
            monthly_features = {k: v for k, v in batch['monthly_features'].items()}
            monthly_masks = {k: v for k, v in batch['monthly_masks'].items()}
            month_boundaries = batch['month_boundary_indices']

            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

            # Check outputs for NaN
            nan_found = False
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    nan_found = True
                    break

            if nan_found:
                print(f"  Sample {i}: NaN in output!")
                # Run detailed trace
                report = run_training_step(model, batch, device='cpu')
                print(report)
                break
            else:
                # Quick loss check
                regime_logits = outputs['regime_logits'][:, -1, :]
                regime_targets = torch.zeros(1, dtype=torch.long)
                loss = F.cross_entropy(regime_logits, regime_targets)

                if torch.isnan(loss):
                    print(f"  Sample {i}: NaN in loss!")
                    report = run_training_step(model, batch, device='cpu')
                    print(report)
                    break
                else:
                    print(f"  Sample {i}: OK (loss={loss.item():.4f})")

        except Exception as e:
            print(f"  Sample {i}: Exception - {e}")

    # Test with larger batch
    print("\n[4] Testing with batch size 4...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=multi_resolution_collate_fn,
    )

    for batch_idx, batch in enumerate(loader):
        model.zero_grad()

        try:
            daily_features = {k: v for k, v in batch['daily_features'].items()}
            daily_masks = {k: v for k, v in batch['daily_masks'].items()}
            monthly_features = {k: v for k, v in batch['monthly_features'].items()}
            monthly_masks = {k: v for k, v in batch['monthly_masks'].items()}
            month_boundaries = batch['month_boundary_indices']

            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
            )

            # Compute full loss
            regime_logits = outputs['regime_logits'][:, -1, :]
            batch_size = regime_logits.size(0)
            regime_targets = torch.zeros(batch_size, dtype=torch.long)
            regime_loss = F.cross_entropy(regime_logits, regime_targets)

            casualty_var = outputs['casualty_var'][:, -1, :]
            casualty_loss = casualty_var.mean()

            total_loss = regime_loss + casualty_loss

            if torch.isnan(total_loss):
                print(f"  Batch {batch_idx}: NaN in loss!")
                report = run_training_step(model, batch, device='cpu')
                print(report)
                break
            else:
                print(f"  Batch {batch_idx}: OK (loss={total_loss.item():.4f})")

            if batch_idx >= 5:
                break

        except Exception as e:
            print(f"  Batch {batch_idx}: Exception - {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    test_multiple_batches()
