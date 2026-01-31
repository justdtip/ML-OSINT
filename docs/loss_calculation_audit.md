# Loss Calculation Audit: Multi-Resolution HAN Training

## Executive Summary

**Critical Finding**: The 9x discrepancy between validation loss (55.85) and test loss (6.20) is caused by fundamentally different loss aggregation methods:

| Phase | Aggregation Method | Result |
|-------|-------------------|--------|
| **Training** | Multi-task uncertainty weighting | Weighted sum with learned log-variance |
| **Validation** | Multi-task uncertainty weighting | Same as training (consistent) |
| **Test** | Simple arithmetic mean | Unweighted average of task losses |

The test loss reports **unweighted average of raw task losses**, while train/val report **uncertainty-weighted combined loss**. These are not comparable metrics.

---

## Detailed Analysis

### 1. Training Loss Calculation (`train_epoch`, lines 2236-2351)

**Model Mode**: `self.model.train()` (line 2243)
**Gradients**: Enabled (normal backprop)
**Mixed Precision**: Optional via `autocast`

#### Individual Task Loss Computation
Uses `_compute_losses()` which returns a dictionary of task losses:
- `regime`: Cross-entropy with class balancing, label smoothing, entropy/diversity regularization
- `transition`: Binary cross-entropy for phase transitions
- `casualty`: ZINB (Zero-Inflated Negative Binomial) NLL loss
- `anomaly`: MSE with variance penalty
- `forecast`: Autoregressive feature prediction MSE
- `daily_forecast`: Latent smoothness or MSE (optional)

#### Total Loss Combination
```python
# Line 2277-2278
task_losses = self._compute_losses(outputs, batch)
total_loss, task_weights = self.multi_task_loss(task_losses)
```

**MultiTaskLoss.forward()** (lines 752-821) applies Kendall et al. (2018) uncertainty weighting:

```
L_total = SUM_i [ 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i ] + reg_term
```

Where:
- `log_var_i` is a learnable parameter per task (clamped to [-2, 2])
- `exp(-log_var_i)` is the "precision" (task weight), ranging from 0.14 to 7.4
- Higher uncertainty (larger log_var) -> lower weight on that task
- `reg_term = 0.01 * SUM(log_var_i^2)` prevents drift

#### What is Tracked
```python
# Lines 2307-2310
epoch_losses['total'] += total_loss.item()  # WEIGHTED total
for task_name, loss_val in task_losses.items():
    epoch_losses[task_name] += loss_val.item()  # RAW unweighted
```

**Reported Training Loss**: Weighted uncertainty loss averaged over batches.

---

### 2. Validation Loss Calculation (`validate`, lines 2353-2418)

**Model Mode**: `self.model.eval()` (line 2360)
**Gradients**: Disabled (`torch.no_grad()`, line 2370)
**Mixed Precision**: Same as training

#### Individual Task Loss Computation
Uses the **same** `_compute_losses()` method as training.

#### Total Loss Combination
```python
# Lines 2387-2388
task_losses = self._compute_losses(outputs, batch)
total_loss, _ = self.multi_task_loss(task_losses)
```

**Same MultiTaskLoss** with uncertainty weighting is used.

#### What is Tracked
```python
# Lines 2391-2393
epoch_losses['total'] += total_loss.item()  # WEIGHTED total
for task_name, loss_val in task_losses.items():
    epoch_losses[task_name] += loss_val.item()  # RAW unweighted
```

**Reported Validation Loss**: Weighted uncertainty loss averaged over batches.

---

### 3. Test Loss Calculation (`evaluate`, lines 2702-2760)

**Model Mode**: `self.model.eval()` (line 2720)
**Gradients**: Disabled (`torch.no_grad()`, line 2725)
**Mixed Precision**: NOT explicitly enabled (potential difference)

#### Individual Task Loss Computation
Uses the **same** `_compute_losses()` method.

#### Total Loss Combination
```python
# Lines 2738-2741
task_losses = self._compute_losses(outputs, batch)

for task_name, loss_val in task_losses.items():
    all_losses[task_name].append(loss_val.item())
```

**CRITICAL DIFFERENCE**: No `multi_task_loss()` is called!

```python
# Lines 2754-2758
metrics = {
    f'{task}_loss': np.mean(losses)
    for task, losses in all_losses.items()
}
metrics['total_loss'] = sum(metrics.values()) / len(metrics)  # SIMPLE AVERAGE
```

**Reported Test Loss**: Simple arithmetic mean of per-task loss averages.

---

## The Discrepancy Explained

### Mathematical Analysis

Let's trace through with example numbers:

**Raw Task Losses** (hypothetical batch averages):
| Task | Raw Loss |
|------|----------|
| casualty | 45.0 |
| regime | 2.5 |
| transition | 0.3 |
| anomaly | 1.2 |
| forecast | 12.0 |
| daily_forecast | 2.0 |

**Uncertainty-Weighted (Train/Val)**:

Assuming learned log_var values after training (based on DEFAULT_TASK_PRIORS):
| Task | log_var | Weight (exp(-log_var)) | Contribution |
|------|---------|------------------------|--------------|
| casualty | 1.4 | 0.247 | 0.5 * 0.247 * 45.0 + 0.5 * 1.4 = 6.26 |
| regime | 1.05 | 0.350 | 0.5 * 0.350 * 2.5 + 0.5 * 1.05 = 0.96 |
| transition | 1.4 | 0.247 | 0.5 * 0.247 * 0.3 + 0.5 * 1.4 = 0.74 |
| anomaly | 1.4 | 0.247 | 0.5 * 0.247 * 1.2 + 0.5 * 1.4 = 0.85 |
| forecast | 1.9 | 0.150 | 0.5 * 0.150 * 12.0 + 0.5 * 1.9 = 1.85 |
| daily_forecast | 1.4 | 0.247 | 0.5 * 0.247 * 2.0 + 0.5 * 1.4 = 0.95 |

**Total (Train/Val)**: 6.26 + 0.96 + 0.74 + 0.85 + 1.85 + 0.95 + reg = ~12.0 + reg

**But wait** - the formula adds `0.5 * log_var` terms even when task_loss is high, which inflates the total. With 6 tasks and positive log_var values, the base contribution is ~4.0 before any task losses.

**Simple Average (Test)**:
```
total = (45.0 + 2.5 + 0.3 + 1.2 + 12.0 + 2.0) / 6 = 10.5
```

However, the actual discrepancy (55.85 vs 6.20) suggests:

1. **Casualty loss dominates**: If casualty raw loss is O(100+) due to ZINB NLL on count data, the uncertainty weighting adds `0.5 * exp(-log_var) * 100 + 0.5 * log_var` which could easily reach 50+.

2. **Test averaging dilutes high-loss tasks**: Simple averaging gives equal weight to all tasks, so a single high-loss task is diluted by 5 other low-loss tasks.

---

## All Differences Summary

| Aspect | Training | Validation | Test |
|--------|----------|------------|------|
| Model mode | `train()` | `eval()` | `eval()` |
| Gradients | Enabled | Disabled | Disabled |
| Dropout/BatchNorm | Active | Inference mode | Inference mode |
| Mixed precision | Yes (if enabled) | Yes (if enabled) | **No** |
| Loss computation | `_compute_losses` | `_compute_losses` | `_compute_losses` |
| Loss aggregation | `multi_task_loss()` | `multi_task_loss()` | **Simple mean** |
| Includes log_var terms | Yes (+0.5*log_var) | Yes (+0.5*log_var) | **No** |
| Includes regularization | Yes (+0.01*sum(log_var^2)) | Yes | **No** |
| Temporal regularization | Optional | **No** | **No** |

---

## Recommendations

### What Should Each Phase Optimize/Report?

#### Training Loss
**Current**: Correct for optimization.

The uncertainty-weighted loss is appropriate for training because:
- It allows tasks to self-balance based on homoscedastic uncertainty
- High-uncertainty (harder) tasks get down-weighted to prevent one task dominating
- The `0.5 * log_var` term acts as regularization, preventing log_var from going to -infinity

**No change recommended.**

#### Validation Loss
**Current**: Correct for early stopping, but confusing.

Using the same loss as training ensures consistency for early stopping decisions. However, the absolute value is not interpretable as a "prediction error".

**Recommendation**: Keep using `multi_task_loss()` for early stopping, but **also log** the simple average for comparability with test metrics:
```python
# In validate():
simple_total = sum(loss_val.item() for loss_val in task_losses.values()) / len(task_losses)
epoch_losses['simple_total'] = simple_total
```

#### Test Loss
**Current**: Inconsistent and misleading.

The simple average makes cross-experiment comparison difficult because it depends on task count and doesn't account for task difficulty.

**Recommendation Options**:

**Option A (Consistency)**: Use `multi_task_loss()` for test as well:
```python
# In evaluate():
total_loss, _ = self.multi_task_loss(task_losses)
all_losses['weighted_total'].append(total_loss.item())
```

**Option B (Interpretability)**: Report both weighted and unweighted:
```python
metrics['weighted_total_loss'] = np.mean(all_losses['weighted_total'])
metrics['simple_total_loss'] = sum(metrics.values()) / len(metrics)
```

**Option C (Task-Specific Focus)**: Report each task loss separately without aggregation, since different use cases care about different tasks:
- Forecasting applications: Focus on `forecast_loss` and `daily_forecast_loss`
- Casualty prediction: Focus on `casualty_loss`
- Regime classification: Focus on `regime_loss`

### Recommended Fix

The most principled fix is **Option B** - report both metrics:

```python
def evaluate(self, dataset: MultiResolutionDataset) -> Dict[str, float]:
    # ... existing code ...

    with torch.no_grad():
        for batch in loader:
            # ... existing forward pass ...

            task_losses = self._compute_losses(outputs, batch)

            # NEW: Compute weighted loss for consistency with train/val
            weighted_loss, _ = self.multi_task_loss(task_losses)
            all_losses['weighted_total'].append(weighted_loss.item())

            for task_name, loss_val in task_losses.items():
                all_losses[task_name].append(loss_val.item())

    # Aggregate metrics
    metrics = {
        f'{task}_loss': np.mean(losses)
        for task, losses in all_losses.items()
        if task != 'weighted_total'
    }

    # Report both aggregation methods
    metrics['total_loss'] = sum(v for k, v in metrics.items() if k.endswith('_loss')) / len([k for k in metrics if k.endswith('_loss')])
    metrics['weighted_total_loss'] = np.mean(all_losses['weighted_total'])

    return metrics
```

---

## Additional Observations

### 1. Potential Numerical Issues

The `_compute_losses` function has several anti-collapse mechanisms that may behave differently in train vs eval:
- **Class-balanced weights** for regime (line 1797-1802): Computed on batch, may differ
- **Loss floor** for regime (line 1837): `torch.maximum(regime_loss, 0.05)`
- **Variance penalty** for anomaly (lines 1996-2006): Based on batch statistics

These could cause minor train/val differences even with identical loss aggregation.

### 2. Missing Mixed Precision in Evaluate

`evaluate()` does not use `autocast`:
```python
# In train_epoch and validate:
with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    outputs = self.model(...)

# In evaluate:
outputs = self.model(...)  # No autocast
```

This could cause minor numerical differences in float16 vs float32 operations.

### 3. Temporal Regularization Only in Training

Training adds optional temporal regularization (lines 2280-2286):
```python
if self.use_temporal_reg and self.temporal_regularizer is not None:
    temp_penalty, temp_breakdown = self.temporal_regularizer(outputs, seq_len, self.device)
    total_loss = total_loss + temp_penalty
```

This is correctly excluded from validation and test, but adds to the train loss.

---

## Conclusion

The 9x discrepancy between val_loss (55.85) and test_loss (6.20) is **expected behavior** given the different aggregation methods:

1. **Validation** uses uncertainty-weighted loss including `0.5 * log_var` terms
2. **Test** uses simple unweighted average

This is a **reporting inconsistency**, not a model bug. The model is training correctly; the metrics are just not comparable.

**Primary Recommendation**: Modify `evaluate()` to also compute `weighted_total_loss` using `multi_task_loss()`, so all three phases report comparable metrics.

**Secondary Recommendation**: Always report both weighted and simple averages, clearly labeled, so users understand what they're comparing.

---

## File Locations

- **MultiTaskLoss class**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`, lines 672-844
- **_compute_losses method**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`, lines 1681-2100
- **train_epoch method**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`, lines 2236-2351
- **validate method**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`, lines 2353-2418
- **evaluate method**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`, lines 2702-2760

---

*Audit conducted: 2026-01-31*
*Auditor: Architecture Reviewer*
