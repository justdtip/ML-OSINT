# Metadata Propagation Analysis

## Executive Summary

This analysis traces how training metadata propagates to probe runs and identifies critical bugs in the metadata propagation logic. The investigation reveals **three distinct bugs** causing probe runs to record incorrect training configuration values.

**Key Finding**: The probe run `run_24-01-2026_17-10` shows `use_disaggregated_equipment: false` despite being linked to training run `run_24-01-2026_15-13` which has `use_disaggregated_equipment: true`. This discrepancy can lead to incorrect model interpretation and misleading probe results.

---

## 1. Metadata Files Examined

### Training Run: `run_24-01-2026_15-13`

**config.json** (lines 17):
```json
{
  "use_disaggregated_equipment": true,
  "detrend_viirs": true,
  ...
}
```

**metadata.json** (lines 7-8):
```json
{
  "use_disaggregated_equipment": true,
  "detrend_viirs": true,
  "effective_daily_sources": [
    "drones", "armor", "artillery", "personnel",
    "deepstate", "firms", "viina", "viirs"
  ],
  ...
}
```

### Probe Run: `run_24-01-2026_17-10`

**metadata.json** (lines 28-29):
```json
{
  "use_disaggregated_equipment": false,   // BUG: Should be true!
  "detrend_viirs": true,
  "daily_sources": [
    "equipment", "personnel", "deepstate",
    "firms", "viina", "viirs"
  ],
  ...
}
```

**Critical Discrepancy**: The training run used disaggregated equipment (drones, armor, artillery) but the probe metadata incorrectly shows aggregated "equipment" source and `use_disaggregated_equipment: false`.

---

## 2. Code Path Analysis

### 2.1. Probe Metadata Creation Path

The metadata propagation follows this path:

```
run_probes.py::MasterProbeRunner.__init__()
    |
    v
output_manager.py::RunOutputManager.__init__()
    |-- Creates RunMetadata with DATACLASS DEFAULTS
    |   (use_disaggregated_equipment: bool = True)  # Default in dataclass
    |
    v
run_probes.py::MasterProbeRunner._load_model()
    |
    |-- Loads checkpoint
    |-- Loads config_dict from training_summary.json
    |-- ATTEMPTS to merge training run config.json
    |   (ONLY if config.training_run_id is set)
    |
    v
run_probes.py line 421-426:
    data_config = MultiResolutionConfig(
        ...
        use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),  # BUG #1
        detrend_viirs=config_dict.get('detrend_viirs', True),
    )
    |
    v
output_manager.py::extract_data_metadata(data_config)
    |-- Extracts from data_config which has WRONG default
```

### 2.2. Key Code Locations

**File: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`**

Lines 405-427 - Model loading with config extraction:
```python
# Also check training run's config.json (train_full_pipeline.py runs)
# This may have data config values like use_disaggregated_equipment
if self.config.training_run_id:
    from pathlib import Path
    training_run_dir = Path(self.config.checkpoint_dir).parent.parent
    training_run_config_path = training_run_dir / "config.json"
    if training_run_config_path.exists():
        with open(training_run_config_path, "r") as f:
            run_config = json.load(f)
        # Merge run config (lower priority than training_summary)
        for key in ['use_disaggregated_equipment', 'detrend_viirs']:
            if key in run_config and key not in config_dict:
                config_dict[key] = run_config[key]

# Create data configuration
# Note: defaults are conservative for older checkpoints without these settings
data_config = MultiResolutionConfig(
    daily_seq_len=config_dict.get('daily_seq_len', 365),
    monthly_seq_len=config_dict.get('monthly_seq_len', 12),
    prediction_horizon=config_dict.get('prediction_horizon', 1),
    use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),  # BUG!
    detrend_viirs=config_dict.get('detrend_viirs', True),
)
```

**File: `/Users/daniel.tipton/ML_OSINT/analysis/probes/output_manager.py`**

Lines 50-51 - RunMetadata dataclass defaults:
```python
@dataclass
class RunMetadata:
    ...
    detrend_viirs: bool = False
    use_disaggregated_equipment: bool = True  # Inconsistent with run_probes.py default!
```

Lines 286-293 - Data metadata extraction:
```python
def extract_data_metadata(self, data_config) -> None:
    if hasattr(data_config, 'daily_sources'):
        self.update_metadata(daily_sources=list(data_config.daily_sources))
    ...
    if hasattr(data_config, 'use_disaggregated_equipment'):
        self.update_metadata(use_disaggregated_equipment=data_config.use_disaggregated_equipment)
```

---

## 3. Identified Bugs

### Bug #1: Incorrect Default Value in MultiResolutionConfig Creation

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`, line 425

**Problem**: When creating `MultiResolutionConfig`, the code uses `False` as the default for `use_disaggregated_equipment`:
```python
use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),
```

However, the `MultiResolutionConfig` dataclass itself defaults to `True` (line 118 in `multi_resolution_data.py`):
```python
use_disaggregated_equipment: bool = True  # Set True for new runs with optimized source separation
```

**Impact**: If `config_dict` doesn't contain the key (which happens when training_summary.json lacks it), the probe creates data config with `False` even when the actual training used `True`.

### Bug #2: Missing Training Run ID in CLI Pass-Through

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`, lines 1599-1630

**Problem**: The `training_run_id` is only set when `--training-run` CLI argument is provided. However, the probe run's `metadata.json` shows `training_run_id: ""` even though it loaded a checkpoint from a training run:

```json
"model_checkpoint": ".../run_24-01-2026_15-13/stage3_han/best_checkpoint.pt",
"training_run_id": "",  // Empty even though checkpoint path reveals the training run!
```

The code at lines 1619-1625 only sets `training_run_id` when explicitly provided:
```python
# Link the probe run to the training run
config_kwargs["training_run_id"] = args.training_run
```

But if the user specifies `--checkpoint` directly to a training run's checkpoint, the `training_run_id` remains empty and the config merge at line 407 (`if self.config.training_run_id:`) never executes.

**Impact**: Training config values like `use_disaggregated_equipment` are not loaded from the training run's config.json when using `--checkpoint` directly.

### Bug #3: Incorrect `daily_sources` Field Extraction

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/output_manager.py`, lines 287-288

**Problem**: The `extract_data_metadata` method extracts `daily_sources` from the **config object's `daily_sources` field**, NOT from the **effective sources**:

```python
if hasattr(data_config, 'daily_sources'):
    self.update_metadata(daily_sources=list(data_config.daily_sources))
```

But `MultiResolutionConfig.daily_sources` is the **raw** list `["equipment", ...]`, not the effective list `["drones", "armor", "artillery", ...]` returned by `get_effective_daily_sources()`.

**Evidence from probe metadata**:
```json
"daily_sources": ["equipment", "personnel", "deepstate", "firms", "viina", "viirs"]
```

While training metadata shows the actual sources used:
```json
"effective_daily_sources": ["drones", "armor", "artillery", "personnel", ...]
```

**Impact**: The probe metadata records the wrong sources, making it impossible to accurately compare configurations between runs.

---

## 4. Additional Issues

### 4.1. Inconsistent Defaults Across Files

| File | Field | Default Value |
|------|-------|---------------|
| `run_probes.py` line 425 | `use_disaggregated_equipment` | `False` |
| `output_manager.py` line 51 | `use_disaggregated_equipment` | `True` |
| `multi_resolution_data.py` line 118 | `use_disaggregated_equipment` | `True` |
| `training_output_manager.py` line 107 | `use_disaggregated_equipment` | `None` |

This inconsistency means different components assume different defaults, causing confusion and silent data corruption.

### 4.2. `training_summary.json` May Lack Config Values

The code loads config from `training_summary.json` first (line 401), but this file may not contain all relevant config values. The training run's `config.json` is only consulted if `training_run_id` is explicitly set, creating a fragile dependency.

### 4.3. Stale Cached Values Not Updated

When `output_manager.extract_data_metadata()` is called, it extracts values from the `MultiResolutionConfig` object, but this object was already created with potentially wrong defaults. Even though `update_metadata()` later extracts `daily_sources` from the actual sample (line 542), the `use_disaggregated_equipment` flag was already set incorrectly.

---

## 5. Affected Comparisons

The following probe run comparisons are potentially invalid due to metadata discrepancies:

| Probe Run | Training Run | Expected disaggregated | Recorded disaggregated | Status |
|-----------|--------------|------------------------|------------------------|--------|
| run_24-01-2026_17-10 | run_24-01-2026_15-13 | true | false | **MISMATCH** |
| run_24-01-2026_14-21 | run_24-01-2026_11-57 | false | false | OK |

---

## 6. Recommended Fixes

### Fix 1: Use Consistent Defaults

All files should use the same default value. Since `MultiResolutionConfig` uses `True` as the modern default, update `run_probes.py` line 425:

```python
# Current (buggy):
use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),

# Fixed:
use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', True),
```

### Fix 2: Extract Training Run ID from Checkpoint Path

When `--training-run` is not specified but `--checkpoint` points to a training run directory, auto-detect the training run ID:

```python
# In run_probes.py, after setting checkpoint_path:
if args.training_run is None and args.checkpoint is not None:
    # Try to extract training run ID from checkpoint path
    checkpoint_path = Path(args.checkpoint)
    # Pattern: .../training_runs/run_DD-MM-YYYY_HH-MM/stage3_han/best_checkpoint.pt
    for parent in checkpoint_path.parents:
        if parent.parent.name == "training_runs" and parent.name.startswith("run_"):
            config_kwargs["training_run_id"] = parent.name
            break
```

### Fix 3: Extract Effective Sources Instead of Raw Config

In `output_manager.py`, use `get_effective_daily_sources()`:

```python
def extract_data_metadata(self, data_config) -> None:
    # Use effective sources (accounts for disaggregation)
    if hasattr(data_config, 'get_effective_daily_sources'):
        self.update_metadata(daily_sources=data_config.get_effective_daily_sources())
    elif hasattr(data_config, 'daily_sources'):
        self.update_metadata(daily_sources=list(data_config.daily_sources))
```

### Fix 4: Validate Metadata After Model Load

Add a validation step after model loading to verify that the recorded metadata matches what was actually loaded:

```python
# After extract_data_metadata()
sample = self.dataset[0]
actual_sources = list(sample.daily_features.keys())
if set(actual_sources) != set(self.output_manager.metadata.daily_sources):
    self.logger.warning(
        f"Metadata mismatch: recorded {self.output_manager.metadata.daily_sources}, "
        f"actual {actual_sources}"
    )
    self.output_manager.update_metadata(daily_sources=actual_sources)
```

---

## 7. Conclusion

The metadata propagation system has multiple bugs that cause probe runs to record incorrect configuration values:

1. **Hardcoded `False` default** for `use_disaggregated_equipment` overrides actual training config
2. **Missing training run ID detection** when using `--checkpoint` directly
3. **Wrong sources extracted** (raw config vs effective sources)

These bugs make it impossible to accurately compare probe results across model variations, as the metadata does not reflect the actual model configuration. The recommended fixes address each issue and add validation to catch future discrepancies.
