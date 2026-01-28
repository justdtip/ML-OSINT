# Metadata/Config Inconsistency Analysis Report

**Analysis Date:** 2026-01-24
**Analyst:** Debugging Agent

## Executive Summary

This report documents significant inconsistencies found between three configuration/metadata files from a training run and its derived probe run. Several **critical discrepancies** were identified that could impact reproducibility and validation integrity.

---

## Files Analyzed

| File | Path | Purpose |
|------|------|---------|
| **config.json** | `analysis/training_runs/run_24-01-2026_15-13/config.json` | Original training configuration |
| **training metadata.json** | `analysis/training_runs/run_24-01-2026_15-13/metadata.json` | Training run metadata (runtime state) |
| **probe metadata.json** | `analysis/probes/runs/run_24-01-2026_17-10/metadata.json` | Probe run metadata (derived from training) |

---

## Critical Discrepancies

### 1. `use_disaggregated_equipment` Mismatch (CRITICAL)

| File | Value |
|------|-------|
| config.json | `true` |
| training metadata.json | `true` |
| probe metadata.json | `false` |

**Impact:** The probe run is operating with a different equipment feature configuration than the model was trained with. This could lead to incorrect probe results or model evaluation on mismatched data representations.

### 2. Daily Sources Inconsistency (CRITICAL)

**Training metadata `effective_daily_sources`:**
```json
["drones", "armor", "artillery", "personnel", "deepstate", "firms", "viina", "viirs"]
```
- **Count:** 8 sources
- **Note:** Uses disaggregated equipment (drones, armor, artillery as separate sources)

**Probe metadata `daily_sources`:**
```json
["equipment", "personnel", "deepstate", "firms", "viina", "viirs"]
```
- **Count:** 6 sources
- **Note:** Uses aggregated "equipment" source

**Impact:** The probe is configured with 6 daily sources while the training used 8 disaggregated sources. This is directly related to the `use_disaggregated_equipment` mismatch above.

### 3. Device Inconsistency

| File | Device |
|------|--------|
| config.json | `"auto"` |
| training metadata.json | `"mps"` (Apple Silicon GPU) |
| probe metadata.json | `"cpu"` |

**Impact:** Probe run executed on CPU instead of MPS. While this shouldn't affect numerical results significantly, it could impact timing-based probe metrics and indicates the probe did not inherit the training device configuration.

### 4. `best_epoch` Inconsistency

| File | Value |
|------|-------|
| training metadata.json (stage3_metrics) | `199` |
| probe metadata.json | `0` |

**Impact:** Probe metadata shows `best_epoch: 0` which is inconsistent with the training metadata showing `best_epoch: 199`. This may indicate the probe failed to read the training history correctly.

### 5. `training_run_id` Empty in Probe

| Field | Value |
|-------|-------|
| probe metadata `training_run_id` | `""` (empty string) |
| Expected value | `"run_24-01-2026_15-13"` |

**Impact:** The probe metadata does not properly record which training run it derived from, making provenance tracking difficult.

---

## Field Presence Comparison

### Fields Present in config.json Only

| Field | Value | Notes |
|-------|-------|-------|
| `resolution` | `"weekly"` | Training resolution setting |
| `temporal_gap` | `14` | Gap for temporal prediction |
| `epochs_interpolation` | `100` | Stage-specific epochs |
| `epochs_unified` | `100` | Stage-specific epochs |
| `epochs_han` | `200` | Stage-specific epochs |
| `epochs_temporal` | `150` | Stage-specific epochs |
| `epochs_tactical` | `100` | Stage-specific epochs |
| `n_states` | `8` | Number of regime states |
| `skip_interpolation` | `true` | Pipeline skip flags |
| `skip_unified` | `true` | Pipeline skip flags |
| `skip_han` | `false` | Pipeline skip flags |
| `skip_temporal` | `false` | Pipeline skip flags |
| `skip_tactical` | `false` | Pipeline skip flags |
| `use_multi_resolution` | `true` | Multi-res flag |
| `checkpoint_dir` | (path) | Checkpoint directory |
| `resume_from_stage` | `null` | Resume control |
| `warmup_epochs` | `10` | LR warmup |
| `patience` | `30` | Early stopping patience |
| `early_stopping_min_epochs` | `50` | Min epochs before stopping |
| `early_stopping_min_delta` | `0.0001` | Minimum improvement |
| `early_stopping_smoothing` | `0.9` | Smoothing factor |
| `early_stopping_relative_threshold` | `0.1` | Relative threshold |
| `disable_early_stopping` | `false` | Disable flag |
| `swa_start_pct` | `0.75` | SWA start percentage |
| `swa_freq` | `5` | SWA update frequency |
| `use_snapshots` | `false` | Snapshot ensemble |
| `n_snapshots` | `5` | Number of snapshots |
| `use_label_smoothing` | `false` | Label smoothing |
| `label_smoothing` | `0.1` | Smoothing value |
| `use_mixup` | `false` | Mixup augmentation |
| `mixup_alpha` | `0.2` | Mixup alpha |
| `lr_schedule` | `"cosine"` | LR schedule type |
| `cosine_t0` | `10` | Cosine annealing T0 |
| `cosine_t_mult` | `2` | Cosine annealing mult |
| `verbose` | `true` | Verbosity flag |
| `save_history` | `true` | Save training history |

### Fields Present in Training metadata.json Only

| Field | Value | Notes |
|-------|-------|-------|
| `run_id` | `"run_24-01-2026_15-13"` | Unique run identifier |
| `run_timestamp` | `"2026-01-24T15:13:37.737522"` | Start timestamp |
| `run_dir` | (path) | Run directory path |
| `effective_daily_sources` | (array of 8) | Actual daily sources used |
| `effective_monthly_sources` | (array of 5) | Actual monthly sources used |
| `n_daily_sources` | `8` | Count of daily sources |
| `n_monthly_sources` | `5` | Count of monthly sources |
| `daily_seq_len` | `365` | Daily sequence length |
| `monthly_seq_len` | `12` | Monthly sequence length |
| `date_range_start` | `""` | Date range start |
| `date_range_end` | `""` | Date range end |
| `n_train_samples` | `636` | Training samples |
| `n_val_samples` | `10` | Validation samples |
| `n_test_samples` | `10` | Test samples |
| `feature_dims_per_source` | (object) | Per-source feature dimensions |
| `stage1_complete` - `stage5_complete` | (bool) | Stage completion flags |
| `stage1_duration` - `stage5_duration` | (float) | Stage durations |
| `stage1_metrics` - `stage5_metrics` | (object) | Per-stage metrics |
| `n_jim_models` | `0` | Number of JIM models |
| `n_unified_models` | `0` | Number of unified models |
| `han_best_epoch` | `0` | HAN best epoch (NOTE: inconsistent with stage3) |
| `han_best_val_loss` | `1.9123811721801758` | HAN best validation loss |
| `han_n_params` | `2267967` | HAN parameter count |
| `total_duration_seconds` | `6730.81` | Total training duration |
| `torch_version` | `"2.9.1"` | PyTorch version |
| `python_version` | `"3.13.7"` | Python version |
| `git_commit_hash` | `"af6dfe7..."` | Git commit |
| `random_seed` | `null` | Random seed (NOT SET) |
| `errors` | `[]` | Error list |

### Fields Present in Probe metadata.json Only

| Field | Value | Notes |
|-------|-------|-------|
| `run_id` | `"run_24-01-2026_17-10"` | Probe run ID (different from training) |
| `run_timestamp` | `"2026-01-24T17:10:34.849972"` | Probe start time |
| `model_checkpoint` | (path) | Checkpoint loaded |
| `nhead` | `4` | Number of attention heads |
| `num_daily_layers` | `3` | Daily encoder layers |
| `num_monthly_layers` | `2` | Monthly encoder layers |
| `num_fusion_layers` | `2` | Fusion layers |
| `num_params` | `2267967` | Parameter count (matches training) |
| `daily_sources` | (array of 6) | Sources (MISMATCHED) |
| `monthly_sources` | (array of 5) | Monthly sources |
| `training_epochs` | `199` | Training epochs recorded |
| `task_names` | (array) | Task head names |
| `task_priors` | `{}` | Task priors (empty) |
| `phase_name` | `"Training:run_24-01-2026_15-13"` | Phase reference |
| `phase_description` | `""` | Phase description |
| `optimizations_applied` | `[]` | Optimizations (empty) |
| `training_run_id` | `""` | Training run ID (EMPTY) |
| `probes_completed` | `57` | Probes completed |
| `probes_failed` | `0` | Probes failed |
| `total_duration_seconds` | `1346.51` | Probe run duration |

---

## Value Comparison for Shared Fields

| Field | config.json | training metadata | probe metadata | Status |
|-------|-------------|-------------------|----------------|--------|
| `d_model` | 64 | 64 | 64 | MATCH |
| `batch_size` | 32 | 32 | N/A | N/A |
| `learning_rate` | 0.0001 | 0.0001 | N/A | N/A |
| `weight_decay` | 0.01 | 0.01 | N/A | N/A |
| `use_swa` | true | true | N/A | N/A |
| `early_stopping_strategy` | "smoothed" | "smoothed" | N/A | N/A |
| `use_multi_resolution` | true | true | N/A | N/A |
| `detrend_viirs` | true | true | true | MATCH |
| `use_disaggregated_equipment` | true | true | **false** | **MISMATCH** |
| `device` | "auto" | "mps" | "cpu" | VARIES |
| `daily_seq_len` | N/A | 365 | 365 | MATCH |
| `monthly_seq_len` | N/A | 12 | 12 | MATCH |
| `best_val_loss` | N/A | 1.9123811721801758 | 1.9123811721801758 | MATCH |
| `num_params` / `han_n_params` | N/A | 2267967 | 2267967 | MATCH |
| `torch_version` | N/A | "2.9.1" | "2.9.1" | MATCH |
| `python_version` | N/A | "3.13.7" | "3.13.7" | MATCH |

---

## Timestamp Analysis

| Event | Timestamp | Duration Since Training Start |
|-------|-----------|-------------------------------|
| Training start | 2026-01-24T15:13:37 | 0 |
| Training end (estimated) | 2026-01-24T17:05:28 | ~1h 52m (6730.81 seconds) |
| Probe start | 2026-01-24T17:10:34 | ~1h 57m |
| Probe end (estimated) | 2026-01-24T17:33:01 | ~2h 19m (training + 1346.51s) |

The timeline is consistent - probes started approximately 5 minutes after training completed.

---

## Internal Inconsistency in Training Metadata

**Issue:** `han_best_epoch` vs `stage3_metrics.best_epoch`

| Field | Value |
|-------|-------|
| `han_best_epoch` (top-level) | `0` |
| `stage3_metrics.best_epoch` | `199` |

This internal inconsistency suggests the top-level `han_best_epoch` field was not properly updated after training completed.

---

## Recommendations

### Immediate Actions

1. **Fix `use_disaggregated_equipment` propagation:** Ensure probe runs inherit the correct equipment disaggregation setting from the training configuration.

2. **Populate `training_run_id` in probe metadata:** The probe should properly record which training run it is validating.

3. **Fix `han_best_epoch` update:** Ensure the top-level field is updated to match `stage3_metrics.best_epoch`.

4. **Standardize source naming:** Either use consistent naming between training and probe runs, or implement a mapping layer.

### Configuration Improvements

1. **Add random seed to config:** The `random_seed: null` in training metadata indicates non-deterministic training. Consider requiring a seed for reproducibility.

2. **Validate configuration inheritance:** Implement validation to ensure probe configurations match training configurations for critical fields.

3. **Add schema versioning:** Include a schema version field to track configuration format changes.

---

## Summary of Discrepancies

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 2 | `use_disaggregated_equipment` mismatch, daily sources mismatch |
| HIGH | 2 | Empty `training_run_id`, `best_epoch` inconsistency |
| MEDIUM | 2 | Device differences, internal `han_best_epoch` inconsistency |
| LOW | 1 | Empty `date_range_start/end`, `random_seed: null` |

**Total Issues Identified:** 7

---

*Report generated by Debugging Agent - ML_OSINT Project*
