# Root Cause Analysis: NaN Loss in Multi-Resolution HAN Training

## Executive Summary

The NaN loss during training is caused by the **MISSING_VALUE sentinel (-999.0)** being fed into neural network layers without proper masking. The core issue is a **mask granularity mismatch** between feature-level missing values and timestep-level observation masks.

## Root Cause

### The Bug Location

The bug exists in two locations:

1. **`/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`** lines 1527-1531
2. **`/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_modules.py`** lines 2150-2160

### Detailed Analysis

#### Step 1: Data Preparation (multi_resolution_data.py)

In `_apply_normalization()` (lines 1397-1442):
```python
normalized = (values - mean) / std
normalized = np.where(np.isnan(normalized), self.config.missing_value, normalized)
```

This correctly replaces NaN with -999.0 as a sentinel value. The mask is correctly created at the feature level in `_convert_to_tensors()` (lines 1444-1476):
```python
combined_mask = expanded_mask & data_mask  # [timesteps, features]
```

**At this point: Masks are feature-level (3D: batch x timesteps x features)**

#### Step 2: Mask Reduction (multi_resolution_han.py)

In `MultiResolutionHAN.forward()` lines 1527-1531:
```python
monthly_timestep_masks = {}
for name, mask in monthly_masks.items():
    if mask.dim() == 3:
        monthly_timestep_masks[name] = mask.any(dim=-1).float()  # PROBLEM!
    else:
        monthly_timestep_masks[name] = mask.float()
```

**BUG: The 3D feature-level mask is reduced to 2D timestep-level by taking `any(dim=-1)`**

This means a timestep is marked as "observed" if ANY feature is observed, but individual features may still be -999.0.

#### Step 3: Mask Expansion (multi_resolution_modules.py)

In `MonthlyEncoder._embed_features()` lines 2150-2160:
```python
# observation_mask is 2D: [batch, n_months]
feature_mask = observation_mask.unsqueeze(-1).expand(-1, -1, n_features)

values_clean = values.clone()
values_clean = values_clean.masked_fill(~feature_mask.bool(), 0.0)

# This assertion fails because -999.0 values at "observed" timesteps are not masked
assert not (values_clean.abs() > 100).any(), "Extreme values detected"
```

**BUG: The 2D timestep mask is expanded back to 3D by broadcasting - assuming all features in an observed timestep are valid**

### Visual Example

```
Original Data for one timestep (n_features=5):
  values: [1.2, -999.0, 0.5, -999.0, 2.1]
  feature_mask: [True, False, True, False, True]

After mask.any(dim=-1):
  timestep_observed: True (because some features observed)

After expand(-1, -1, n_features):
  expanded_mask: [True, True, True, True, True]  # WRONG!

After masked_fill:
  values_clean: [1.2, -999.0, 0.5, -999.0, 2.1]  # -999.0 NOT replaced!
```

### Impact

1. **Direct assertion failure**: The code at line 2160 asserts `values_clean.abs() <= 100`, which fails when -999.0 remains
2. **If assertion removed**: -999.0 flows through linear projections, producing extreme activations (potentially 10,000+)
3. **Attention collapse**: Extreme values in attention scores cause softmax to produce NaN
4. **Gradient explosion**: Even if forward pass succeeds, gradients explode during backprop

## Affected Data Sources

From diagnostic output:
- **daily equipment**: 125/1980 (6.3%) missing
- **daily personnel**: 126/540 (23.3%) missing
- **daily deepstate**: 615/900 (68.3%) missing
- **daily firms**: 208/2340 (8.9%) missing
- **daily viirs**: 120/1440 (8.3%) missing
- **monthly sentinel**: 14/42 (33.3%) missing

## Recommended Fixes

### Option 1: Preserve Feature-Level Masks (Recommended)

Pass the full 3D masks to the monthly encoder instead of reducing to 2D.

**File: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`**

Change lines 1527-1531 from:
```python
monthly_timestep_masks = {}
for name, mask in monthly_masks.items():
    if mask.dim() == 3:
        monthly_timestep_masks[name] = mask.any(dim=-1).float()
    else:
        monthly_timestep_masks[name] = mask.float()
```

To:
```python
monthly_timestep_masks = {}
for name, mask in monthly_masks.items():
    # Keep feature-level mask for proper missing value handling
    monthly_timestep_masks[name] = mask.float() if mask.dim() == 3 else mask.unsqueeze(-1).float()
```

Then update `MonthlyEncoder._embed_features()` to handle 3D masks:
```python
if observation_mask.dim() == 2:
    feature_mask = observation_mask.unsqueeze(-1).expand(-1, -1, n_features)
else:
    feature_mask = observation_mask  # Already 3D
```

### Option 2: Replace Sentinel Values Earlier

Replace -999.0 with 0.0 immediately after loading, before any processing.

**File: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`**

Add in `_move_batch_to_device()` (around line 735):
```python
def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    MISSING_VALUE = -999.0
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(self.device)
        elif isinstance(value, dict):
            moved[key] = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    tensor = v.to(self.device)
                    # Replace sentinel values with 0.0 for features
                    if 'features' in key:
                        tensor = tensor.masked_fill(tensor == MISSING_VALUE, 0.0)
                    moved[key][k] = tensor
                else:
                    moved[key][k] = v
        else:
            moved[key] = value
    return moved
```

### Option 3: Clamp Input Values

Add clamping in `DailySourceEncoder.forward()` and `MonthlyEncoder._embed_features()`.

**File: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`**

At line 304 (before feature projection):
```python
# Clamp extreme values before projection
features = features.clamp(min=-10.0, max=10.0)
```

## Verification

After applying fixes, run:
```bash
python3 /Users/daniel.tipton/ML_OSINT/analysis/debug_nan_training.py
```

Expected output: All samples should report "OK" with finite loss values.

## Files to Modify

1. `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`
   - Lines 1527-1531: Fix mask conversion
   - Lines 302-306: Add input clamping (defense in depth)

2. `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_modules.py`
   - Lines 2150-2160: Update `_embed_features()` to handle 3D masks

3. `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`
   - Lines 735-748: Add sentinel value replacement in `_move_batch_to_device()`

## Summary

| Issue | Location | Line | Impact |
|-------|----------|------|--------|
| Mask reduction | multi_resolution_han.py | 1529 | Feature-level info lost |
| Mask expansion | multi_resolution_modules.py | 2153 | Wrong features marked observed |
| No early clamping | DailySourceEncoder.forward() | 306 | Extreme values projected |
| No sentinel handling | _move_batch_to_device() | 735-748 | -999.0 reaches model |
