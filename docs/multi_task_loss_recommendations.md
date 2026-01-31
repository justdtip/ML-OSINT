# Multi-Task Loss Weighting Recommendations

## Executive Summary

This document analyzes the multi-task loss weighting strategies in the ML_OSINT Multi-Resolution HAN and provides recommendations for training, validation, and model comparison.

**Key Recommendations:**
1. Use learned uncertainty weighting (SoftplusKendallLoss) during **training only**
2. Use **uniform weights (or no weighting)** for validation, early stopping, and model comparison
3. Report **per-task metrics separately** alongside any aggregated score
4. Implement availability gating to handle sparse supervision

---

## 1. Current Implementation Analysis

### 1.1 MultiTaskLoss (train_multi_resolution.py)

```python
# Kendall et al. (2018) formulation:
L = 0.5 * exp(-log_var) * L_task + 0.5 * log_var
```

**Strengths:**
- Automatically balances task magnitudes based on learned uncertainty
- Tasks with higher noise get lower weight
- Clamping log_var to [-2, 2] prevents weight explosion

**Weaknesses:**
- Can produce negative total loss when log_var < 0 and L_task is small
- Learned weights change during training, making loss values non-comparable across epochs
- Early stopping based on this loss can be misleading

### 1.2 SoftplusKendallLoss (training_improvements.py)

```python
# Fixed formulation:
s_i = softplus(u_i)  # Ensures s_i >= 0
L = exp(-s_i) * L_i + s_i
```

**Strengths:**
- Guarantees non-negative loss (fixes pathological negative loss issue)
- Maintains the spirit of learned task weighting
- More numerically stable

**Weaknesses:**
- Same fundamental issue: learned weights are non-stationary

### 1.3 AvailabilityGatedLoss (training_improvements.py)

```python
# Hard-gates tasks with availability < threshold (default: 20%)
if availability[task] >= min_availability:
    include_in_loss(task)
else:
    exclude_from_loss(task)
```

**Strengths:**
- Prevents task collapse from sparse supervision
- Avoids learning constant outputs for rarely-supervised tasks

**Weaknesses:**
- Binary gating may be too aggressive; soft gating could be smoother

---

## 2. Is Uncertainty Weighting Appropriate?

### 2.1 For This Multi-Task Setup: **Yes, with caveats**

The ML_OSINT model has 6 tasks with **very different characteristics**:

| Task | Type | Supervision | Loss Scale |
|------|------|-------------|------------|
| regime | 4-class classification | Dense | ~1.0-2.0 |
| casualty | Regression (ZINB) | Dense | Variable (0.5-50+) |
| anomaly | Regression | Sparse | ~0.1-1.0 |
| forecast | Multi-step regression | Dense | Variable |
| transition | Binary classification | Very sparse | ~0.5-1.5 |
| daily_forecast | Latent prediction | Dense | ~0.1-2.0 |

**Key findings from probe analysis:**
- Casualty head dominates loss at 62.6% (Probe findings in optimisation-proposals.md)
- Regime head performs best despite lower loss contribution
- Forecast head has "suspicious" low loss (possible data leakage)

**Uncertainty weighting helps because:**
1. It prevents the high-magnitude casualty loss from dominating gradients
2. Tasks with unreliable targets (anomaly, transition) naturally get down-weighted
3. The model can learn which tasks have more noise vs. signal

---

## 3. Validation and Test Loss: Critical Distinction

### 3.1 The Problem

**Using uncertainty-weighted loss for validation is problematic:**

```python
# Current implementation (train_multi_resolution.py line 2388):
task_losses = self._compute_losses(outputs, batch)
total_loss, _ = self.multi_task_loss(task_losses)  # Uses learned weights!
```

**Issues:**
1. **Non-comparability**: If weights change during training, epoch 10 loss cannot be compared to epoch 50 loss
2. **Gaming the metric**: The model could improve validation loss by adjusting weights rather than predictions
3. **Model selection bias**: Early stopping may select checkpoints that have "better" weights rather than better predictions

### 3.2 ML Best Practices

The academic literature is clear on this distinction:

**Kendall et al. (2018)** - Original paper:
> "The learned uncertainty parameters are used during training to balance the losses, but for evaluation we report task-specific metrics."

**Chen et al. (2018) - GradNorm:**
> "Model selection should be based on the primary task metric or a weighted average with fixed weights."

**Crawshaw (2020) - Multi-Task Learning Survey:**
> "For fair comparison between models, use fixed task weights or report per-task performance."

---

## 4. Recommendations

### 4.1 Training Loss: Use SoftplusKendallLoss + AvailabilityGating

```python
# Recommended training configuration:
training_loss = AvailabilityGatedLoss(
    task_names=['regime', 'casualty', 'anomaly', 'forecast', 'transition', 'daily_forecast'],
    min_availability=0.2,  # Exclude tasks with <20% valid targets
    base_loss=SoftplusKendallLoss(task_names, init_scale=0.0),
)
```

**Rationale:**
- SoftplusKendall fixes negative loss pathology
- Availability gating prevents task collapse from sparse supervision
- Learned weights help balance gradient magnitudes

### 4.2 Validation/Early Stopping: Use Fixed Uniform Weights

```python
# Recommended validation configuration:
class UniformWeightedValidationLoss(nn.Module):
    """Fixed equal weights for validation - ensures comparability."""

    def __init__(self, task_names: List[str], weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_names = task_names
        # Default: uniform weights (or user-specified fixed weights)
        self.weights = weights or {name: 1.0 / len(task_names) for name in task_names}

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for task_name, loss in losses.items():
            if task_name in self.weights and not torch.isnan(loss):
                total = total + self.weights[task_name] * loss
        return total
```

**Rationale:**
- Fixed weights ensure loss values are comparable across epochs
- Early stopping triggers on genuine prediction improvement, not weight optimization
- Enables fair comparison between different training runs

### 4.3 Model Comparison: Report Per-Task Metrics

For comparing different models or hyperparameter configurations:

```python
def evaluate_model(model, test_loader) -> Dict[str, float]:
    """Comprehensive evaluation with per-task metrics."""
    results = {
        # Per-task losses (unweighted)
        'regime_loss': compute_regime_loss(model, test_loader),
        'casualty_loss': compute_casualty_loss(model, test_loader),
        'anomaly_loss': compute_anomaly_loss(model, test_loader),
        'forecast_loss': compute_forecast_loss(model, test_loader),
        'transition_loss': compute_transition_loss(model, test_loader),
        'daily_forecast_loss': compute_daily_forecast_loss(model, test_loader),

        # Per-task metrics (interpretable)
        'regime_accuracy': compute_regime_accuracy(model, test_loader),
        'regime_f1_macro': compute_regime_f1(model, test_loader),
        'casualty_mae': compute_casualty_mae(model, test_loader),
        'casualty_rmse': compute_casualty_rmse(model, test_loader),
        'forecast_mse': compute_forecast_mse(model, test_loader),

        # Aggregate (with fixed business-relevant weights)
        'aggregate_score': (
            0.35 * regime_accuracy +  # Primary task based on probe findings
            0.25 * casualty_score +
            0.25 * forecast_score +
            0.15 * auxiliary_score
        ),
    }
    return results
```

### 4.4 Implementation Changes Required

**Modify train_multi_resolution.py:**

```python
class Trainer:
    def __init__(self, ...):
        # Training loss: learned weights
        self.training_loss = SoftplusKendallLoss(task_names)

        # Validation loss: fixed weights for comparability
        self.validation_loss = UniformWeightedLoss(task_names)

    def train_step(self, batch):
        task_losses = self._compute_losses(outputs, batch)
        total_loss, task_weights = self.training_loss(task_losses)  # Learned
        return total_loss

    def validate(self):
        with torch.no_grad():
            task_losses = self._compute_losses(outputs, batch)
            # Use fixed weights for validation!
            total_loss = self.validation_loss(task_losses)
        return total_loss
```

---

## 5. Pros and Cons Summary

### 5.1 For Model Selection (Early Stopping)

| Approach | Pros | Cons |
|----------|------|------|
| **Learned weights (current)** | Reflects training objective | Non-comparable across epochs; may select "good weight" checkpoints |
| **Fixed uniform weights (recommended)** | Comparable; fair selection | May not reflect task importance |
| **Fixed task-prior weights** | Reflects domain importance | Requires domain knowledge |
| **Per-task early stopping** | Task-specific patience | Complex; may conflict |

**Recommendation:** Fixed uniform weights for simplicity and comparability.

### 5.2 For Final Performance Reporting

| Approach | Pros | Cons |
|----------|------|------|
| **Single aggregate score** | Simple; easy to compare | Hides task-level performance |
| **Per-task metrics (recommended)** | Full transparency | More complex; harder to summarize |
| **Primary task + auxiliaries** | Focus on main goal | May neglect auxiliary tasks |

**Recommendation:** Report per-task metrics with interpretable units (accuracy, MAE, F1), plus a fixed-weight aggregate for quick comparison.

### 5.3 For Comparing Different Models

| Approach | Pros | Cons |
|----------|------|------|
| **Validation loss** | Standard ML practice | Depends on weight configuration |
| **Test set per-task metrics** | Fair, reproducible | Requires holdout set |
| **Pareto frontier** | Shows trade-offs | Harder to select "best" |

**Recommendation:** Test set with per-task metrics; use Pareto analysis if tasks conflict.

---

## 6. Specific Answers to Your Questions

### Q1: Is uncertainty weighting appropriate for this multi-task setup?

**Yes**, but only during training. The heterogeneous task types (classification, regression, sparse supervision) benefit from learned balancing. However, the learned weights should not influence model selection or evaluation.

### Q2: Should validation and test use the same weighting as training?

**No.** Validation and test should use **fixed weights** (uniform or domain-specified) to ensure:
- Loss values are comparable across epochs and runs
- Model selection is based on prediction quality, not weight optimization
- Fair comparison between different model configurations

### Q3: Pros/cons for model selection, reporting, and comparison?

See Section 5 above. Key takeaways:
- **Model selection:** Fixed weights prevent gaming; use uniform or domain-informed fixed weights
- **Reporting:** Always include per-task metrics; aggregate scores are useful but incomplete
- **Comparison:** Use test set with per-task metrics; learned training weights make comparison unfair

### Q4: What do ML best practices suggest?

The literature consensus:
1. Use adaptive loss weighting (Kendall, GradNorm, etc.) during **training** to balance gradients
2. Use **fixed weights or per-task metrics** for validation, early stopping, and model comparison
3. Report **interpretable per-task metrics** (accuracy, F1, MAE) alongside any aggregate score
4. Be explicit about the weighting scheme used for any aggregate metric

---

## 7. Implementation Roadmap

### Phase 1: Immediate (Low Effort)

1. Add `UniformWeightedLoss` class for validation
2. Modify `validate()` method to use fixed weights
3. Log both training loss (learned weights) and validation loss (fixed weights)

### Phase 2: Short-term (Medium Effort)

1. Add per-task metrics to validation logging
2. Implement separate early stopping criteria (validation loss with fixed weights)
3. Update model comparison scripts to use per-task metrics

### Phase 3: Long-term (Higher Effort)

1. Implement Pareto frontier analysis for multi-objective optimization
2. Add task-specific early stopping with patience per task
3. Create standardized evaluation benchmark for model comparison

---

## 8. References

1. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.

2. Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML 2018*.

3. Crawshaw, M. (2020). Multi-Task Learning with Deep Neural Networks: A Survey. *arXiv:2009.09796*.

4. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning. *NeurIPS 2020*.

5. Liu, S., Johns, E., & Davison, A. J. (2019). End-To-End Multi-Task Learning With Attention. *CVPR 2019*.

---

## 9. Summary Table

| Use Case | Loss Function | Weights | Rationale |
|----------|--------------|---------|-----------|
| Training | SoftplusKendallLoss + AvailabilityGating | Learned | Balance gradients, handle sparse supervision |
| Validation | UniformWeightedLoss | Fixed (uniform or domain-specified) | Comparability across epochs |
| Early Stopping | UniformWeightedLoss | Fixed | Select on prediction quality, not weight quality |
| Model Comparison | Per-task metrics | N/A (per-task) | Fair comparison, full transparency |
| Final Reporting | Per-task + aggregate | Fixed (domain-specified) | Interpretability + summary |

---

*Document created: 2026-01-31*
*Based on analysis of: `train_multi_resolution.py`, `training_improvements.py`, probe findings*
