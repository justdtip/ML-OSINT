# Hybrid Loss Recommendation: A³DRO + Spectral + Cycle Consistency

**Date:** 2026-01-31
**Status:** Implementation Complete
**Based on:** Analysis of gemini.md, gpt52.md, kimik2.md proposals

---

## Executive Summary

After thorough analysis of all three proposals against the current architecture, audit findings, and already-implemented improvements, I recommend a **hybrid approach** combining the best elements:

| Component | Source | Purpose | Status |
|-----------|--------|---------|--------|
| **A³DRO Loss** | gpt52.md | Training aggregation (no learned weights) | ✅ Implemented |
| **SpectralDriftPenalty** | gemini.md | Forecast regularization (FFT-based) | ✅ Implemented |
| **CrossResolutionCycleConsistency** | gpt52.md | Daily/monthly alignment | ✅ Already existed |
| **UniformValidationLoss** | Audit recommendation | Fixed weights for validation | ✅ Implemented |
| **LatentStatePredictor** | gpt52.md | Prevent forecast memorization | ✅ Already existed |
| **AvailabilityGatedLoss** | gpt52.md | Handle sparse supervision | ✅ Already existed |
| **PCGradSurgery** | kimik2/gemini | Gradient conflict resolution | ✅ Already existed |

---

## Why This Hybrid?

### 1. A³DRO vs SoftplusKendall

**Key insight from audit:** Learned weights make validation loss non-comparable across epochs.

| Feature | SoftplusKendall | A³DRO |
|---------|-----------------|-------|
| Learned weights | Yes (problematic) | **No (fixed after warmup)** |
| Validation comparable | No | **Yes** |
| Focuses on worst task | No | **Yes (robust optimization)** |
| Handles sparse supervision | Only via gating | **Built-in soft gating** |
| Prevents negative loss | Yes | **Yes** |

**Recommendation:** Use A³DRO for training, UniformValidationLoss for validation.

### 2. SpectralDriftPenalty (from gemini)

**Key insight:** The 140x Train/Val gap for forecasts is because the model memorizes high-frequency noise.

The spectral penalty:
- Transforms prediction error to frequency domain via FFT
- Weights low-frequency (trend) errors more heavily
- Ignores high-frequency (noise) that the model was memorizing

This directly addresses overfitting without changing the architecture.

### 3. Why NOT kimik2.md (Complete Architecture Rewrite)

While theoretically interesting, the AFNP-STGS + VTB-PATS approach requires:
- Complete architecture replacement with Mixture-of-Experts
- HyperNetwork for dynamic head generation
- Bilevel optimization
- Variational Bayesian inference

**Risk assessment:**
- High implementation complexity (~4x more code)
- Significant regression risk
- Current architecture already shows promise (regime head performs well)
- The core problems (overfitting, task dominance) can be solved with loss changes alone

---

## Implementation Guide

### Usage Pattern

```python
from analysis.training_improvements import (
    HybridLossConfig,
    create_training_losses,
    PCGradSurgery,
)

# Create loss functions
config = HybridLossConfig(
    use_a3dro=True,              # Recommended: robust aggregation
    use_spectral_penalty=True,   # Recommended: forecast regularization
    use_cycle_consistency=True,  # Recommended: daily/monthly alignment
    lambda_temp=0.5,             # Smaller = more focus on worst task
    spectral_weight=0.1,         # Weight for spectral penalty
    cycle_weight=0.2,            # Weight for cycle consistency
)

task_names = ['regime', 'casualty', 'forecast', 'daily_forecast', 'anomaly', 'transition']
losses = create_training_losses(task_names, config)

# In training loop:
training_loss = losses['training'](task_losses, targets=targets, epoch=epoch)
spectral_loss = losses['spectral'](daily_pred, daily_targets)
cycle_loss = losses['cycle'](daily_latents, monthly_teacher)

total_train_loss = training_loss + spectral_loss + cycle_loss

# In validation loop (IMPORTANT: use separate loss)
val_loss = losses['validation'](task_losses)  # Fixed weights!
```

### Integration with train_multi_resolution.py

The key changes needed:

1. **Replace MultiTaskLoss initialization:**
```python
# OLD:
self.multi_task_loss = MultiTaskLoss(task_names)

# NEW:
from analysis.training_improvements import HybridLossConfig, create_training_losses
config = HybridLossConfig()
self.loss_modules = create_training_losses(task_names, config)
self.training_loss = self.loss_modules['training']
self.validation_loss = self.loss_modules['validation']
```

2. **Modify train_epoch:**
```python
# OLD:
total_loss, task_weights = self.multi_task_loss(task_losses)

# NEW:
total_loss, task_weights = self.training_loss(
    task_losses,
    targets=batch_targets,
    epoch=epoch
)
# Add spectral penalty for forecast
if 'daily_forecast' in outputs:
    spectral_loss = self.loss_modules['spectral'](
        outputs['daily_forecast'],
        targets['daily_next_7_days']
    )
    total_loss = total_loss + spectral_loss
```

3. **Modify validate:**
```python
# OLD:
total_loss, _ = self.multi_task_loss(task_losses)

# NEW:
total_loss, _ = self.validation_loss(task_losses)  # Fixed weights!
```

---

## Expected Impact

| Problem | Solution | Expected Improvement |
|---------|----------|---------------------|
| Val/Test loss discrepancy | UniformValidationLoss | Direct comparability |
| 140x Forecast overfitting | SpectralDriftPenalty | Filters high-freq noise |
| Task dominance (casualty 62.6%) | A³DRO anchored regrets | Normalized task scales |
| Non-comparable validation | No learned weights | Meaningful early stopping |
| Daily/monthly divergence | CycleConsistency | Physical coupling enforced |

---

## Hyperparameter Recommendations

### A³DRO Parameters
- `lambda_temp`: Start at 1.0, reduce to 0.3-0.5 as training stabilizes
  - Higher = more uniform weighting
  - Lower = more focus on worst task
- `warmup_epochs`: 3 (freeze baselines after this)
- `a_min`: 0.2 (minimum availability threshold)
- `kappa`: 20 (soft gating steepness)

### Spectral Parameters
- `weight_decay`: 0.5 (exponential decay for frequency weights)
  - Higher = more focus on very low frequencies
- `weight`: 0.1 (contribution to total loss)

### Cycle Consistency
- `weight`: 0.2 (contribution to total loss)
- `loss_type`: 'cosine' (more robust than MSE for latent comparison)

---

## Files Modified

| File | Changes |
|------|---------|
| `analysis/training_improvements.py` | Added A3DROLoss, SpectralDriftPenalty, UniformValidationLoss, HybridLossConfig |

---

## Verification

All components tested:
```
============================================================
✓ All tests passed!
============================================================
```

Run with: `python -m analysis.training_improvements`

---

## References

1. **A³DRO**: Based on Distributionally Robust Optimization literature, adapted from gpt52.md proposal
2. **Spectral Loss**: Based on frequency-domain analysis, adapted from gemini.md proposal
3. **Audit Findings**: `docs/loss_calculation_audit.md`, `docs/multi_task_loss_recommendations.md`

---

*Document created: 2026-01-31*
*Implementation: analysis/training_improvements.py*
