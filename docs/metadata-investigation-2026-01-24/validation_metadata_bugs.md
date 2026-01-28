# Metadata Bug Validation Report

**Report Date**: 2026-01-24
**Validator**: Code Review Agent
**Files Reviewed**:
- `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`
- `/Users/daniel.tipton/ML_OSINT/analysis/probes/output_manager.py`
- `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_data.py`
- `/Users/daniel.tipton/ML_OSINT/analysis/training_runs/run_24-01-2026_15-13/config.json`
- `/Users/daniel.tipton/ML_OSINT/analysis/training_runs/run_24-01-2026_15-13/metadata.json`
- `/Users/daniel.tipton/ML_OSINT/analysis/probes/runs/run_24-01-2026_17-10/metadata.json`

---

## Executive Summary

Validation confirmed **3 of 3 claimed bugs** and identified **2 additional issues** missed by the debugger agents. The root cause chain is accurate: a single default value bug at `run_probes.py:425` causes cascading discrepancies in equipment disaggregation and source counts.

---

## Confirmed Discrepancies

### 1. `use_disaggregated_equipment` Mismatch - CONFIRMED

| Source | Value |
|--------|-------|
| config.json (line 17) | `true` |
| training metadata.json (line 8) | `true` |
| probe metadata.json (line 29) | `false` |

**Evidence**: Direct comparison of the three files confirms the discrepancy exactly as reported.

### 2. Daily Sources Mismatch - CONFIRMED

| Source | Sources |
|--------|---------|
| Training effective_daily_sources | `["drones", "armor", "artillery", "personnel", "deepstate", "firms", "viina", "viirs"]` (8 sources) |
| Probe daily_sources | `["equipment", "personnel", "deepstate", "firms", "viina", "viirs"]` (6 sources) |

**Evidence**:
- Training metadata.json lines 9-18 show 8 disaggregated sources
- Probe metadata.json lines 13-20 show 6 aggregated sources with "equipment" instead of the three disaggregated types

**Root Cause**: This is a direct consequence of Bug #1 - when `use_disaggregated_equipment=False`, the `get_effective_daily_sources()` method (multi_resolution_data.py:120-148) does not expand "equipment" into ["drones", "armor", "artillery"].

### 3. Device Mismatch - CONFIRMED

| Source | Value |
|--------|-------|
| config.json (line 44) | `"auto"` |
| training metadata.json (line 102) | `"mps"` |
| probe metadata.json (line 45) | `"cpu"` |

**Evidence**: Direct file comparison confirms discrepancy.

**Analysis**: This is expected behavior, not a bug. The training run resolved "auto" to "mps" (Apple Silicon GPU). The probe run may have run on a different environment or the device resolution happened differently. The config stores the intent ("auto"), while runtime metadata stores the actual device used.

### 4. `best_epoch` Mismatch - CONFIRMED

| Source | Value |
|--------|-------|
| training stage3_metrics.best_epoch | `199` |
| training han_best_epoch | `0` |
| probe best_epoch | `0` |

**Evidence**:
- Training metadata.json line 66: `"best_epoch": 199` (in stage3_metrics)
- Training metadata.json line 93: `"han_best_epoch": 0`
- Probe metadata.json line 31: `"best_epoch": 0`

**Analysis**: This reveals an internal inconsistency in the training metadata. The probe correctly reads `best_epoch` from the checkpoint (run_probes.py:553-554), but the checkpoint itself contains `best_epoch: 0` which doesn't match the training summary.

### 5. Empty `training_run_id` - CONFIRMED

| Source | Value |
|--------|-------|
| Probe metadata.json (line 44) | `""` (empty string) |

**Evidence**: The probe metadata shows `"training_run_id": ""` despite being run against a specific training run.

---

## Confirmed Bugs

### Bug #1: Wrong Default for `use_disaggregated_equipment` - CONFIRMED

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`, line 425

**Code**:
```python
data_config = MultiResolutionConfig(
    daily_seq_len=config_dict.get('daily_seq_len', 365),
    monthly_seq_len=config_dict.get('monthly_seq_len', 12),
    prediction_horizon=config_dict.get('prediction_horizon', 1),
    use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', False),  # BUG: Default should be True
    detrend_viirs=config_dict.get('detrend_viirs', True),
)
```

**Problem**: The dataclass `MultiResolutionConfig` in `multi_resolution_data.py:118` defines:
```python
use_disaggregated_equipment: bool = True  # Set True for new runs with optimized source separation
```

But `run_probes.py:425` uses `False` as the fallback when the key is not found in `config_dict`. The training_summary.json does NOT contain this key (verified at lines 1-23), so the fallback is used.

**Severity**: HIGH - This causes the probe to use different data sources than training, invalidating probe results.

### Bug #2: Training Run ID Not Propagated to Output Manager - CONFIRMED

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`, lines 1625 and 337

**Code at line 1625**:
```python
config_kwargs["training_run_id"] = args.training_run
```

**Code at line 337**:
```python
# Update metadata with device info
self.output_manager.update_metadata(device=config.device)
# BUG: training_run_id is NOT passed to output_manager here
```

**Problem**: While `training_run_id` is correctly stored in `ProbeRunnerConfig` at line 132, and set from args at line 1625, it is never propagated to the `output_manager`. The `MasterProbeRunner.__init__()` only calls `update_metadata(device=config.device)` at line 337, missing the training_run_id.

**Severity**: MEDIUM - Breaks traceability between probe runs and training runs.

### Bug #3: `extract_data_metadata` Uses Raw Sources Instead of Effective Sources - CONFIRMED

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/output_manager.py`, lines 286-287

**Code**:
```python
if hasattr(data_config, 'daily_sources'):
    self.update_metadata(daily_sources=list(data_config.daily_sources))
```

**Problem**: This extracts `data_config.daily_sources` (the raw default list) instead of calling `data_config.get_effective_daily_sources()` which applies the equipment disaggregation transformation.

**Analysis**: However, examining run_probes.py:541-544:
```python
self.output_manager.extract_data_metadata(data_config)
self.output_manager.update_metadata(
    daily_sources=list(daily_source_configs.keys()),
    ...
)
```

The `extract_data_metadata` call is immediately followed by an explicit `update_metadata` call that overwrites `daily_sources` with the actual source configs. The bug in `extract_data_metadata` is present but **masked** by the subsequent override. The real issue is that `daily_source_configs` is built from the dataset sample (line 460-476), which uses the wrong `data_config` created with `use_disaggregated_equipment=False`.

**Severity**: LOW (as standalone bug) - The method has a design flaw but the actual source of the discrepancy is Bug #1.

---

## Additional Issues Discovered

### Additional Issue #1: Checkpoint Condition Race for Config Loading

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py`, lines 407-417

**Code**:
```python
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
```

**Problem**: This code DOES attempt to load `use_disaggregated_equipment` from the training run's config.json (which DOES contain it at line 17). However, since `self.config.training_run_id` is empty (due to Bug #2), this entire block is skipped.

**Impact**: Bug #2 causes Bug #1 to manifest. If training_run_id were properly set, the fallback to `False` would never be reached because the value would be loaded from config.json.

**Severity**: HIGH - This is the actual causal chain: Bug #2 -> config block skipped -> Bug #1 fallback used -> wrong data sources.

### Additional Issue #2: Internal Inconsistency in Training Metadata

**Location**: `/Users/daniel.tipton/ML_OSINT/analysis/training_runs/run_24-01-2026_15-13/metadata.json`

**Evidence**:
- Line 66: `"best_epoch": 199` (inside stage3_metrics)
- Line 93: `"han_best_epoch": 0`

**Problem**: These two fields should contain the same value. The HAN best epoch from stage 3 should be recorded consistently.

**Severity**: MEDIUM - Causes confusion and breaks traceability for checkpoint validation.

---

## Refuted/Clarified Findings

### Device Discrepancy

**Status**: NOT A BUG

The device difference (auto -> mps -> cpu) reflects:
1. config.json stores the configuration intent ("auto")
2. training metadata stores the resolved device at training time ("mps")
3. probe metadata stores the device used during probes ("cpu")

This is expected behavior - different runs may use different devices. The only concern would be if probe results depend on device-specific numerical precision.

---

## Bug Causality Chain

```
Bug #2 (training_run_id not propagated)
    |
    v
run_probes.py:407 condition `if self.config.training_run_id:` is FALSE
    |
    v
Config.json NOT loaded (lines 407-417 skipped)
    |
    v
config_dict does NOT contain 'use_disaggregated_equipment'
    |
    v
Bug #1 fallback triggered: `config_dict.get('use_disaggregated_equipment', False)`
    |
    v
MultiResolutionConfig created with use_disaggregated_equipment=False
    |
    v
MultiResolutionDataset uses aggregated 'equipment' instead of ['drones', 'armor', 'artillery']
    |
    v
Probe runs against DIFFERENT data sources than training
    |
    v
Probe results INVALID for validating the trained model
```

---

## Severity Assessment Summary

| Bug | Severity | Impact |
|-----|----------|--------|
| Bug #1: Wrong default for use_disaggregated_equipment | HIGH | Invalidates probe results |
| Bug #2: training_run_id not propagated | HIGH | Breaks config loading chain |
| Bug #3: extract_data_metadata uses raw sources | LOW | Masked by subsequent code |
| Additional #1: Config loading race condition | HIGH | Root cause of Bug #1 manifestation |
| Additional #2: han_best_epoch inconsistency | MEDIUM | Traceability issue |

---

## Recommended Fix Priority Order

### Priority 1 (Critical - Fix Immediately)

**Fix Bug #2**: Add training_run_id propagation to output_manager

Location: `run_probes.py` around line 337

```python
# Update metadata with device info AND training run linkage
self.output_manager.update_metadata(
    device=config.device,
    training_run_id=config.training_run_id or "",  # Add this
)
```

**Rationale**: This fix alone will enable the config.json loading path (lines 407-417), which will correctly set `use_disaggregated_equipment=True` from the training run's config.

### Priority 2 (High - Fix Soon)

**Fix Bug #1**: Change default from False to True

Location: `run_probes.py` line 425

```python
use_disaggregated_equipment=config_dict.get('use_disaggregated_equipment', True),  # Match dataclass default
```

**Rationale**: Even after fixing Bug #2, this ensures robustness for cases where training_run_id is not available.

### Priority 3 (Medium - Fix When Convenient)

**Fix Bug #3**: Use get_effective_daily_sources()

Location: `output_manager.py` lines 286-287

```python
if hasattr(data_config, 'get_effective_daily_sources'):
    self.update_metadata(daily_sources=list(data_config.get_effective_daily_sources()))
elif hasattr(data_config, 'daily_sources'):
    self.update_metadata(daily_sources=list(data_config.daily_sources))
```

**Rationale**: While currently masked, this is a latent bug that could manifest if the code structure changes.

### Priority 4 (Low - Track for Investigation)

**Investigate Additional Issue #2**: han_best_epoch vs stage3_metrics.best_epoch inconsistency

This appears to be a bug in the training pipeline's metadata recording, not the probe system. Should be investigated separately.

---

## Conclusion

The debugger agents correctly identified the core bugs and their manifestation as metadata discrepancies. The key insight missed was the causal relationship: Bug #2 is the ROOT CAUSE that prevents Bug #1's mitigation code from executing. Fixing Bug #2 alone would resolve the critical discrepancies, but fixing both provides defense in depth.

The probe results from `run_24-01-2026_17-10` should be considered INVALID as they were generated against incorrectly configured data sources.
