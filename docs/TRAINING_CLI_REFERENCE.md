# ML_OSINT Training Pipeline CLI Reference

Complete command-line interface documentation for the tactical state prediction training pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Complete CLI Reference](#complete-cli-reference)
   - [Pipeline Control](#pipeline-control)
   - [Training Hyperparameters](#training-hyperparameters)
   - [Early Stopping Options](#early-stopping-options)
   - [Regularization](#regularization)
   - [Data Configuration](#data-configuration)
   - [Hardware](#hardware)
   - [Checkpointing](#checkpointing)
4. [Configuration Presets](#configuration-presets)
5. [Output Files](#output-files)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The `train_full_pipeline.py` script orchestrates end-to-end training for the tactical state prediction system. It trains a sequence of five interconnected models that progressively transform raw multi-source data into discrete tactical state predictions with uncertainty quantification.

### The 5-Stage Architecture

The pipeline follows a hierarchical approach where each stage builds upon the outputs of previous stages:

```
Raw Data Sources
       |
       v
+------------------+
| Stage 1:         |    Fills temporal gaps in each data source
| Interpolation    | -> Produces interpolated daily features
+------------------+
       |
       v
+------------------+
| Stage 2:         |    Learns cross-source relationships
| Unified Model    | -> Produces unified feature embeddings
+------------------+
       |
       v
+------------------+
| Stage 3:         |    Multi-domain attention for state encoding
| HAN              | -> Produces encoded state representations
+------------------+
       |
       v
+------------------+
| Stage 4:         |    Multi-horizon forecasting (T+1, T+3, T+7)
| Temporal         | -> Produces feature predictions with uncertainty
+------------------+
       |
       v
+------------------+
| Stage 5:         |    Discrete state classification
| Tactical         | -> Produces state predictions P(S(t+1)|S(t))
+------------------+
```

### Stage Descriptions

| Stage | Name | Description | Output |
|-------|------|-------------|--------|
| 1 | **Joint Interpolation** | Fills temporal gaps in irregularly-sampled data sources using transformer-based interpolation | Interpolated daily features for each source |
| 2 | **Unified Cross-Source** | Learns relationships between different data sources through self-supervised reconstruction | Unified latent embeddings combining all sources |
| 3 | **Hierarchical Attention Network (HAN)** | Applies multi-domain hierarchical attention to encode complex state representations | Domain-aware encoded representations |
| 4 | **Temporal Prediction** | Forecasts features at multiple horizons (T+1, T+3, T+7 days) with uncertainty estimates | Multi-horizon predictions with confidence intervals |
| 5 | **Tactical State Predictor** | Classifies discrete tactical states and models state transition probabilities | State predictions and transition matrices |

---

## Quick Start

### Basic Usage

```bash
# Run the complete pipeline with default settings
python train_full_pipeline.py

# Quick test run (10 epochs per stage, useful for verification)
python train_full_pipeline.py --quick

# Run with custom epoch count for all stages
python train_full_pipeline.py --epochs 50
```

### Minimal Test Command

For quickly verifying that the pipeline runs correctly:

```bash
python train_full_pipeline.py --quick --force-cpu
```

This runs 10 epochs per stage on CPU, completing in approximately 5-10 minutes depending on hardware.

### Full Production Training

```bash
python train_full_pipeline.py \
    --epochs-interpolation 100 \
    --epochs-unified 100 \
    --epochs-han 200 \
    --epochs-temporal 150 \
    --epochs-tactical 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --use-swa \
    --early-stop-strategy smoothed \
    --checkpoint-dir models/production_run
```

---

## Complete CLI Reference

### Pipeline Control

These arguments control which stages run and in what order.

#### `--config`

| Property | Value |
|----------|-------|
| Type | `str` |
| Default | `None` |
| Description | Path to a JSON configuration file. When provided, loads all settings from the file. Command-line arguments override file settings. |

**Example:**
```bash
python train_full_pipeline.py --config experiments/my_config.json
```

#### `--stage`

| Property | Value |
|----------|-------|
| Type | `int` |
| Choices | `1`, `2`, `3`, `4`, `5` |
| Default | `None` (run all stages) |
| Description | Run only a specific stage. All other stages are skipped. Useful for debugging or retraining individual components. |

**Example:**
```bash
# Train only the HAN model (Stage 3)
python train_full_pipeline.py --stage 3
```

#### `--resume`

| Property | Value |
|----------|-------|
| Type | `int` |
| Choices | `1`, `2`, `3`, `4`, `5` |
| Default | `None` |
| Description | Resume training from a specific stage. Loads checkpoint data from previous stages and continues from the specified stage. |

**Example:**
```bash
# Resume from Stage 3 after a failure
python train_full_pipeline.py --resume 3
```

#### `--quick`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Quick test run with 10 epochs per stage and patience of 5. Useful for testing pipeline connectivity and debugging. |

**Example:**
```bash
python train_full_pipeline.py --quick
```

#### `--skip-interpolation`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Skip Stage 1 (Joint Interpolation Models). |

#### `--skip-unified`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Skip Stage 2 (Unified Cross-Source Model). |

#### `--skip-han`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Skip Stage 3 (Hierarchical Attention Network). |

#### `--skip-temporal`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Skip Stage 4 (Temporal Prediction Model). |

#### `--skip-tactical`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Skip Stage 5 (Tactical State Predictor). |

**Example:**
```bash
# Run only Stages 3-5 (skip interpolation and unified)
python train_full_pipeline.py --skip-interpolation --skip-unified
```

---

### Training Hyperparameters

Core training parameters that affect learning dynamics.

#### `--epochs`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `None` |
| Description | Override the number of epochs for **all** stages. When set, applies uniformly across all five stages. |

**Example:**
```bash
python train_full_pipeline.py --epochs 50
```

#### `--epochs-interpolation`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `100` |
| Description | Number of training epochs for Stage 1 (Joint Interpolation). |

#### `--epochs-unified`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `100` |
| Description | Number of training epochs for Stage 2 (Unified Cross-Source). |

#### `--epochs-han`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `200` |
| Description | Number of training epochs for Stage 3 (HAN). This stage typically requires more epochs due to the complexity of hierarchical attention. |

#### `--epochs-temporal`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `150` |
| Description | Number of training epochs for Stage 4 (Temporal Prediction). |

#### `--epochs-tactical`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `100` |
| Description | Number of training epochs for Stage 5 (Tactical State Predictor). |

**Example:**
```bash
# Custom epochs per stage
python train_full_pipeline.py \
    --epochs-interpolation 50 \
    --epochs-unified 50 \
    --epochs-han 100 \
    --epochs-temporal 75 \
    --epochs-tactical 50
```

#### `--batch-size`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `32` |
| Description | Number of samples per gradient update. Reduce if running out of memory. |

**Example:**
```bash
python train_full_pipeline.py --batch-size 16
```

#### `--lr`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `1e-4` (0.0001) |
| Description | Initial learning rate for the optimizer. |

**Example:**
```bash
python train_full_pipeline.py --lr 3e-4
```

#### `--d-model`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `64` |
| Description | Model embedding dimension. Must be divisible by the number of attention heads (4). Higher values increase model capacity but require more memory. |

**Example:**
```bash
python train_full_pipeline.py --d-model 128
```

#### `--n-states`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `8` |
| Description | Number of discrete tactical states for classification in Stage 5. |

**Example:**
```bash
python train_full_pipeline.py --n-states 12
```

---

### Early Stopping Options

Fine-grained control over when training should stop to prevent overfitting.

#### `--patience`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `30` |
| Description | Number of epochs without improvement before early stopping triggers. The definition of "improvement" depends on the strategy. |

**Example:**
```bash
python train_full_pipeline.py --patience 50
```

#### `--min-epochs`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `50` |
| Description | Minimum number of epochs before early stopping can trigger, regardless of validation performance. Ensures sufficient training before evaluation. |

**Example:**
```bash
python train_full_pipeline.py --min-epochs 100
```

#### `--early-stop-strategy`

| Property | Value |
|----------|-------|
| Type | `str` |
| Choices | `standard`, `smoothed`, `relative`, `plateau`, `combined` |
| Default | `smoothed` |
| Description | Strategy for determining when to stop training early. |

**Strategy Descriptions:**

| Strategy | Description |
|----------|-------------|
| `standard` | Stop when validation loss has not improved for `patience` epochs |
| `smoothed` | Apply exponential moving average (EMA) to validation loss before checking improvement |
| `relative` | Consider improvement relative to best loss (e.g., 10% tolerance) |
| `plateau` | Detect when loss has plateaued using statistical tests |
| `combined` | Use multiple criteria together for robust stopping |

**Example:**
```bash
python train_full_pipeline.py --early-stop-strategy combined
```

#### `--smoothing-factor`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.9` |
| Description | EMA smoothing factor for the `smoothed` early stopping strategy. Higher values (closer to 1.0) give more weight to historical values. |

**Example:**
```bash
python train_full_pipeline.py --early-stop-strategy smoothed --smoothing-factor 0.95
```

#### `--relative-threshold`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.1` |
| Description | Relative tolerance for the `relative` early stopping strategy. A value of 0.1 means 10% tolerance from the best loss. |

**Example:**
```bash
python train_full_pipeline.py --early-stop-strategy relative --relative-threshold 0.15
```

#### `--min-delta`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `1e-4` |
| Description | Minimum improvement in validation loss required to reset the patience counter. Prevents stopping due to numerical noise. |

**Example:**
```bash
python train_full_pipeline.py --min-delta 1e-5
```

#### `--no-early-stop`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Completely disable early stopping. Training runs for the full number of epochs. Useful when using SWA or snapshot ensembles. |

**Example:**
```bash
python train_full_pipeline.py --no-early-stop --epochs 300
```

---

### Regularization

Advanced regularization techniques to improve generalization.

#### `--use-swa` / `--no-swa`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `--use-swa` (enabled) |
| Description | Enable or disable Stochastic Weight Averaging (SWA). SWA averages model weights during the final portion of training to find flatter minima with better generalization. |

**Example:**
```bash
# Disable SWA
python train_full_pipeline.py --no-swa
```

#### `--swa-start-pct`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.75` |
| Description | Start SWA at this percentage of total training. For example, 0.75 means SWA begins at 75% of the way through training. |

**Example:**
```bash
python train_full_pipeline.py --use-swa --swa-start-pct 0.6
```

#### `--swa-freq`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `5` |
| Description | Update the SWA model every N epochs during the SWA phase. |

**Example:**
```bash
python train_full_pipeline.py --use-swa --swa-freq 3
```

#### `--use-snapshots`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Enable snapshot ensemble collection. Saves model checkpoints at strategic points for ensemble predictions. |

**Example:**
```bash
python train_full_pipeline.py --use-snapshots --n-snapshots 5
```

#### `--n-snapshots`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `5` |
| Description | Maximum number of model snapshots to keep for ensemble. |

#### `--use-label-smoothing`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Enable label smoothing for classification tasks. Reduces overconfidence and improves calibration. |

**Example:**
```bash
python train_full_pipeline.py --use-label-smoothing --label-smoothing 0.1
```

#### `--label-smoothing`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.1` |
| Description | Label smoothing factor. A value of 0.1 means 10% of probability mass is distributed across non-target classes. |

#### `--use-mixup`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Enable mixup data augmentation. Interpolates between training samples to improve generalization. |

**Example:**
```bash
python train_full_pipeline.py --use-mixup --mixup-alpha 0.2
```

#### `--mixup-alpha`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.2` |
| Description | Mixup interpolation strength. Higher values create more aggressive mixing. |

---

### Learning Rate Scheduling

Control how the learning rate evolves during training.

#### `--lr-schedule`

| Property | Value |
|----------|-------|
| Type | `str` |
| Choices | `cosine`, `cosine_restarts`, `linear`, `constant` |
| Default | `cosine` |
| Description | Learning rate schedule type. |

| Schedule | Description |
|----------|-------------|
| `cosine` | Cosine annealing from initial LR to minimum LR |
| `cosine_restarts` | Cosine annealing with warm restarts (SGDR) |
| `linear` | Linear decay from initial LR to minimum LR |
| `constant` | Constant learning rate throughout training |

**Example:**
```bash
python train_full_pipeline.py --lr-schedule cosine_restarts --cosine-t0 20 --cosine-t-mult 2
```

#### `--cosine-t0`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `10` |
| Description | Initial cycle length (in epochs) for cosine annealing with restarts. |

#### `--cosine-t-mult`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `2` |
| Description | Cycle length multiplier for successive restart cycles. For example, with T0=10 and T_mult=2, cycles are 10, 20, 40, 80... epochs. |

---

### Data Configuration

#### `--temporal-gap`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `14` |
| Description | Temporal gap in days between training and validation/test splits. Prevents data leakage from temporal autocorrelation. |

**Example:**
```bash
python train_full_pipeline.py --temporal-gap 30
```

---

### Hardware

Control compute device selection.

#### `--device`

| Property | Value |
|----------|-------|
| Type | `str` |
| Choices | `auto`, `cuda`, `mps`, `cpu` |
| Default | `auto` |
| Description | Compute device for training. `auto` automatically selects the best available device (CUDA > MPS > CPU). |

**Example:**
```bash
python train_full_pipeline.py --device cuda
```

#### `--force-cpu`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Force CPU usage regardless of GPU availability. Recommended for transformer operations on Apple Silicon where MPS may cause issues. |

**Example:**
```bash
python train_full_pipeline.py --force-cpu
```

---

### Checkpointing

Control where models and results are saved.

#### `--checkpoint-dir`

| Property | Value |
|----------|-------|
| Type | `str` |
| Default | `models/pipeline` |
| Description | Directory for saving checkpoints and results. Creates the directory if it does not exist. |

**Example:**
```bash
python train_full_pipeline.py --checkpoint-dir models/experiment_001
```

#### `--quiet`

| Property | Value |
|----------|-------|
| Type | `flag` |
| Default | `False` |
| Description | Reduce output verbosity. Only errors and final summaries are printed. |

**Example:**
```bash
python train_full_pipeline.py --quiet
```

---

## Configuration Presets

The `training_config.py` module provides pre-defined configuration presets for common use cases. These can be used with the `--config` option or programmatically via `get_config()`.

### Available Presets

| Preset | Description | Epochs | Batch Size | Resolution | Key Features |
|--------|-------------|--------|------------|------------|--------------|
| `default` | Standard balanced configuration | 200 | 4 | weekly | SWA enabled, smoothed early stopping |
| `fast_debug` | Quick iterations for debugging | 10 | 8 | monthly | Minimal training, fast feedback |
| `production` | Full training for deployment | 500 | 4 | weekly | Extended patience, lower LR |
| `high_capacity` | Larger model for complex patterns | 300 | 2 | weekly | d_model=128, 4 encoder layers |
| `quick_validation` | Fast validation runs | 50 | 8 | weekly | Shorter sequences |
| `long_training` | Extended training with regularization | 500 | 4 | weekly | SWA, mixup, label smoothing |
| `no_early_stop` | Train for full epochs | 300 | 4 | weekly | Early stopping disabled, snapshots |
| `aggressive_regularization` | Heavy regularization | 400 | 4 | weekly | High dropout, mixup, cutmix |
| `cyclic_lr` | Cosine restarts with snapshots | 300 | 4 | weekly | SGDR schedule, snapshot ensemble |

### Ablation Presets

For systematic ablation studies:

| Preset | Disabled Component |
|--------|-------------------|
| `ablation_no_state_transition` | State transition modeling |
| `ablation_no_multi_scale` | Multi-scale temporal attention |
| `ablation_no_delta` | Delta prediction (absolute values instead) |

### Using Presets

**Programmatically:**
```python
from training_config import get_config

# Load a preset
config = get_config('production')

# Modify as needed
config.training.epochs = 300

# Save for reproducibility
config.save('my_experiment.json')
```

**From Command Line:**
```bash
# Create config file from preset, then use it
python -c "from training_config import get_config; get_config('production').save('prod_config.json')"
python train_full_pipeline.py --config prod_config.json
```

---

## Output Files

### Directory Structure

After training, the checkpoint directory contains:

```
models/pipeline/
    pipeline_config.json       # Saved configuration
    pipeline_results.json      # Final results summary
    stage1_checkpoint.pt       # Stage 1 checkpoint
    stage2_checkpoint.pt       # Stage 2 checkpoint
    stage3_checkpoint.pt       # Stage 3 checkpoint
    stage4_checkpoint.pt       # Stage 4 checkpoint
    stage5_checkpoint.pt       # Stage 5 checkpoint
    logs/
        pipeline_YYYYMMDD_HHMMSS.log  # Training log
```

### Checkpoint Files

Each stage checkpoint (`stageN_checkpoint.pt`) contains:

| Key | Type | Description |
|-----|------|-------------|
| `stage` | int | Stage number (1-5) |
| `timestamp` | str | ISO format timestamp |
| `results` | dict | Stage-specific metrics and history |
| `model_state` | dict | Model weights (if applicable) |

### Results JSON

The `pipeline_results.json` file contains:

```json
{
  "interpolation": {
    "metrics": {
      "best_val_mae": 0.0234,
      "final_val_mae": 0.0256,
      "epochs_trained": 100
    },
    "duration_seconds": 1234.5
  },
  "unified": {
    "metrics": {...},
    "duration_seconds": 567.8
  },
  "han": {...},
  "temporal": {...},
  "tactical": {
    "metrics": {
      "final_val_acc": 0.82,
      "final_trans_acc": 0.75,
      "epochs_trained": 100
    }
  },
  "pipeline": {
    "total_duration_seconds": 5678.9,
    "completed_at": "2024-01-15T14:30:00",
    "device": "cuda"
  }
}
```

### Interpreting Results

**Key Metrics by Stage:**

| Stage | Key Metric | Good Value | Description |
|-------|------------|------------|-------------|
| 1 | `best_val_mae` | < 0.05 | Mean absolute error for interpolation |
| 2 | `best_val_mae` | < 0.1 | Cross-source reconstruction error |
| 3 | `final_regime_acc` | > 0.7 | Regime classification accuracy |
| 4 | `overall_mean_corr` | > 0.5 | Prediction correlation across horizons |
| 5 | `final_val_acc` | > 0.75 | State classification accuracy |
| 5 | `final_trans_acc` | > 0.70 | State transition prediction accuracy |

---

## Examples

### Common Use Cases

#### 1. First-Time Training (Verification)

```bash
# Quick test to ensure everything works
python train_full_pipeline.py --quick --force-cpu

# Check the output
cat models/pipeline/pipeline_results.json
```

#### 2. Full Production Training

```bash
python train_full_pipeline.py \
    --epochs-interpolation 100 \
    --epochs-unified 100 \
    --epochs-han 200 \
    --epochs-temporal 150 \
    --epochs-tactical 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --patience 30 \
    --use-swa \
    --swa-start-pct 0.75 \
    --early-stop-strategy smoothed \
    --min-epochs 50 \
    --checkpoint-dir models/production_v1
```

#### 3. Resuming Failed Training

If training fails at Stage 4:

```bash
# Resume from Stage 4, using Stage 1-3 checkpoints
python train_full_pipeline.py --resume 4 --checkpoint-dir models/production_v1
```

#### 4. Retraining a Single Stage

```bash
# Retrain only the HAN model with different hyperparameters
python train_full_pipeline.py \
    --stage 3 \
    --epochs-han 300 \
    --d-model 128 \
    --lr 5e-5 \
    --checkpoint-dir models/han_experiment
```

#### 5. Long Training Without Early Stopping

```bash
python train_full_pipeline.py \
    --epochs 500 \
    --no-early-stop \
    --use-swa \
    --swa-start-pct 0.6 \
    --use-snapshots \
    --n-snapshots 5 \
    --checkpoint-dir models/long_run
```

#### 6. Memory-Constrained Training

```bash
# Reduce batch size and use CPU
python train_full_pipeline.py \
    --batch-size 8 \
    --d-model 32 \
    --force-cpu \
    --checkpoint-dir models/low_memory
```

#### 7. Hyperparameter Tuning Example

```bash
# Test different learning rates
for lr in 1e-3 5e-4 1e-4 5e-5; do
    python train_full_pipeline.py \
        --quick \
        --lr $lr \
        --checkpoint-dir models/lr_sweep/lr_$lr
done
```

#### 8. Ablation Study

```bash
# Run ablation experiments
python train_full_pipeline.py \
    --epochs 100 \
    --checkpoint-dir models/ablation/baseline

# Then use ablation presets from training_config.py
```

#### 9. Custom Early Stopping

```bash
# Combined strategy with aggressive parameters
python train_full_pipeline.py \
    --early-stop-strategy combined \
    --min-epochs 100 \
    --patience 40 \
    --smoothing-factor 0.95 \
    --relative-threshold 0.15 \
    --min-delta 1e-5
```

#### 10. Full Regularization Suite

```bash
python train_full_pipeline.py \
    --epochs 300 \
    --use-swa \
    --swa-start-pct 0.7 \
    --use-label-smoothing \
    --label-smoothing 0.1 \
    --use-mixup \
    --mixup-alpha 0.2 \
    --lr-schedule cosine_restarts \
    --cosine-t0 30 \
    --cosine-t-mult 2 \
    --checkpoint-dir models/full_regularization
```

---

## Troubleshooting

### Common Errors and Solutions

#### MPS (Apple Silicon) Issues

**Error:** `NotImplementedError: The operator 'aten::...' is not currently implemented for the MPS device`

**Solution:**
```bash
# Force CPU usage
python train_full_pipeline.py --force-cpu

# Or set environment variable before running
export PYTORCH_ENABLE_MPS_FALLBACK=1
python train_full_pipeline.py
```

**Note:** The script automatically sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, but some operations still fail. CPU is recommended for Apple Silicon.

---

#### CUDA Out of Memory

**Error:** `CUDA out of memory. Tried to allocate X MiB...`

**Solutions:**
```bash
# Reduce batch size
python train_full_pipeline.py --batch-size 8

# Reduce model dimension
python train_full_pipeline.py --d-model 32

# Use gradient accumulation (modify training_config.py)
# Set accumulation_steps higher to achieve effective batch size

# Force CPU if GPU memory is insufficient
python train_full_pipeline.py --force-cpu
```

---

#### Module Import Errors

**Error:** `Warning: Joint interpolation models not available: No module named '...'`

**Solution:**
Ensure all required files are in the analysis directory:
- `joint_interpolation_models.py`
- `unified_interpolation.py`
- `hierarchical_attention_network.py`
- `temporal_prediction.py`
- `tactical_state_prediction.py`
- `training_utils.py`
- `training_config.py`

The pipeline will skip stages with missing modules.

---

#### Checkpoint Loading Errors

**Error:** `RuntimeError: Error(s) in loading state_dict...`

**Cause:** Model architecture changed between runs.

**Solution:**
```bash
# Start fresh without loading old checkpoints
rm -rf models/pipeline/stage*_checkpoint.pt
python train_full_pipeline.py
```

---

#### Data Not Found

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: '...data/...'`

**Solution:**
1. Ensure data files are in the expected `data/` directory
2. Check file paths in source configuration modules
3. Verify data preprocessing has been completed

---

#### Training Stops Too Early

**Problem:** Early stopping triggers before the model has converged.

**Solutions:**
```bash
# Increase minimum epochs
python train_full_pipeline.py --min-epochs 100

# Increase patience
python train_full_pipeline.py --patience 50

# Use smoothed early stopping
python train_full_pipeline.py --early-stop-strategy smoothed --smoothing-factor 0.95

# Disable early stopping entirely
python train_full_pipeline.py --no-early-stop
```

---

#### Loss is NaN or Inf

**Problem:** Training produces NaN or infinite loss values.

**Solutions:**
```bash
# Reduce learning rate
python train_full_pipeline.py --lr 1e-5

# Enable gradient clipping (default max_grad_norm=1.0 in config)

# Check for data issues - NaN values in input data
```

---

#### Slow Training on MPS

**Problem:** MPS is slower than expected on Apple Silicon.

**Solution:**
For transformer models, CPU is often faster due to MPS fallback overhead:
```bash
python train_full_pipeline.py --force-cpu
```

---

### Getting Help

1. **Check Logs:** Review `models/pipeline/logs/pipeline_*.log` for detailed error messages
2. **Verify Configuration:** Check `models/pipeline/pipeline_config.json` to confirm settings
3. **Quick Test:** Run `--quick` to isolate issues quickly
4. **Component Test:** Use `--stage N` to test individual stages

---

## Appendix: Default Values Summary

| Parameter | Default | Category |
|-----------|---------|----------|
| `--epochs-interpolation` | 100 | Training |
| `--epochs-unified` | 100 | Training |
| `--epochs-han` | 200 | Training |
| `--epochs-temporal` | 150 | Training |
| `--epochs-tactical` | 100 | Training |
| `--batch-size` | 32 | Training |
| `--lr` | 1e-4 | Training |
| `--d-model` | 64 | Model |
| `--n-states` | 8 | Model |
| `--patience` | 30 | Early Stop |
| `--min-epochs` | 50 | Early Stop |
| `--early-stop-strategy` | smoothed | Early Stop |
| `--smoothing-factor` | 0.9 | Early Stop |
| `--relative-threshold` | 0.1 | Early Stop |
| `--min-delta` | 1e-4 | Early Stop |
| `--swa-start-pct` | 0.75 | Regularization |
| `--swa-freq` | 5 | Regularization |
| `--label-smoothing` | 0.1 | Regularization |
| `--mixup-alpha` | 0.2 | Regularization |
| `--lr-schedule` | cosine | LR Schedule |
| `--cosine-t0` | 10 | LR Schedule |
| `--cosine-t-mult` | 2 | LR Schedule |
| `--temporal-gap` | 14 | Data |
| `--device` | auto | Hardware |
| `--checkpoint-dir` | models/pipeline | Output |

---

*Documentation generated for ML_OSINT tactical state prediction training pipeline.*
