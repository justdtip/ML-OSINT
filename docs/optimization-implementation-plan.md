# ML_OSINT Optimization Implementation Plan

**Generated:** 2026-01-23
**Status:** PENDING REVIEW

---

## Executive Summary

This document synthesizes findings from three specialized investigations (AI Architecture, ML Training, Data Engineering) to provide a comprehensive implementation plan for the optimization proposals. The recommendations are prioritized based on impact, complexity, and risk.

### Critical Findings

| Finding | Impact | Source |
|---------|--------|--------|
| Context window 14 days optimal (vs 365) | +27% accuracy | Probe 3.1.1 |
| RSA degrades 0.77 → 0.37 during training | Fusion quality loss | Probe 2.1.4 |
| Drones lead casualties by 7-27 days | Predictive signal | Probe 1.1.3/4 |
| VIIRS correlation is 300-8500% spurious | Remove confound | Probe 1.2.3 |
| Casualty head dominates loss at 62.6% | Training imbalance | Loss analysis |

---

## Implementation Phases

### Phase 0: Quick Wins (Week 1)
**Impact: Critical | Effort: Low | Risk: Low**

These changes require minimal code modifications and provide immediate benefits.

#### 0.1 Reduce Context Window to 14 Days

**Rationale:** Probe 3.1.1 shows 78.8% accuracy at 7 days vs 51.3% at full context.

**Files to modify:**
- `analysis/training_config.py`: Change `seq_len` default from 365 to 14
- `analysis/multi_resolution_data.py`: Update `daily_seq_len` default

**Code change:**
```python
# In MultiResolutionConfig
daily_seq_len: int = 14  # Changed from 365
context_window_days: int = 14
```

**Risk:** None - already validated by probes

---

#### 0.2 Enable VIIRS Detrending by Default

**Rationale:** VIIRS lags casualties by 10 days; correlation is trend-driven artifact.

**Files to modify:**
- `analysis/multi_resolution_data.py`: Change `detrend_viirs` default to `True`

**Code change:**
```python
# In MultiResolutionConfig
detrend_viirs: bool = True  # Changed from False
```

**Risk:** Low - first-differencing is well-understood

---

#### 0.3 Use Disaggregated Equipment Sources

**Rationale:** Drones have 4x higher mutual information (0.449) than other equipment.

**Files to modify:**
- `analysis/multi_resolution_data.py`: Update default `daily_sources`

**Code change:**
```python
# In MultiResolutionConfig
daily_sources: List[str] = field(default_factory=lambda: [
    "drones",      # MI=0.449, r=0.289 - HIGHEST
    "armor",       # APCs r=0.221
    "artillery",   # Mixed signal
    "personnel",
    "deepstate",
    "firms",
    "viina",
])
# Note: "equipment" (aggregated) and "aircraft" (negative correlation) removed
```

**Risk:** Low - disaggregated loaders already implemented

---

#### 0.4 Initialize Loss Weights with Task Priors

**Rationale:** Casualty head dominates at 62.6%; rebalancing improves multi-task learning.

**Files to modify:**
- `analysis/train_multi_resolution.py`: Update `MultiTaskLoss` initialization

**Code change:**
```python
# Initialize log_vars with priors instead of zeros
initial_log_vars = {
    'casualty': 1.4,    # exp(-1.4) ≈ 0.25 weight
    'regime': 1.05,     # exp(-1.05) ≈ 0.35 weight
    'anomaly': 1.4,     # exp(-1.4) ≈ 0.25 weight
    'forecast': 1.9,    # exp(-1.9) ≈ 0.15 weight
}
```

**Risk:** Medium - may need tuning; monitor task-specific metrics

---

### Phase 1: Training Improvements (Week 2-3)
**Impact: High | Effort: Medium | Risk: Medium**

#### 1.1 RSA-Based Early Stopping

**Rationale:** RSA peaks at epoch 10, then degrades. Stop before fusion quality collapses.

**Design Decision:** Compute RSA every 5 epochs (not every epoch) to reduce overhead.

**Implementation:**
```python
class RSAEarlyStopping:
    def __init__(
        self,
        min_rsa: float = 0.30,
        rsa_drop_threshold: float = 0.10,
        check_frequency: int = 5,
        patience: int = 10,
        warmup_epochs: int = 10,
    ):
        self.best_rsa = 0.0
        self.counter = 0

    def should_stop(self, epoch: int, current_rsa: float) -> bool:
        if epoch < self.warmup_epochs:
            return False
        if current_rsa < self.min_rsa or current_rsa < self.best_rsa - self.rsa_drop_threshold:
            self.counter += self.check_frequency
        else:
            self.best_rsa = max(self.best_rsa, current_rsa)
            self.counter = 0
        return self.counter >= self.patience
```

**Drawbacks:**
- ~20-25 seconds overhead per RSA computation
- May stop before task loss converges

**Mitigation:** Save best-RSA checkpoint separately from best-loss checkpoint

---

#### 1.2 Partial Gradient Isolation for Casualty Head

**Rationale:** Prevent casualty head from dominating shared representations.

**Design Decision:** Use 0.7 gradient scale (not full isolation) to maintain some learning signal.

**Implementation:**
```python
def forward(self, temporal_encoded, ...):
    # Casualty gets reduced gradient influence on shared layers
    casualty_input = temporal_encoded.detach() * 0.3 + temporal_encoded * 0.7
    casualty_pred = self.casualty_head(casualty_input)

    # Other heads use full gradient
    regime_pred = self.regime_head(temporal_encoded)
    anomaly_pred = self.anomaly_head(temporal_encoded)
```

**Drawbacks:**
- May slow casualty head convergence

**Mitigation:** Monitor casualty validation loss; adjust scale if needed

---

#### 1.3 Curriculum Learning for Context Window

**Rationale:** Start with optimal short context, gradually expand to capture longer patterns.

**Design Decision:** Smooth cosine annealing (not hard cutoffs) to prevent distribution shift.

**Implementation:**
```python
class CurriculumScheduler:
    def __init__(
        self,
        initial_context: int = 14,
        final_context: int = 60,
        warmup_epochs: int = 20,
        rampup_epochs: int = 30,
    ):
        pass

    def get_context_window(self, epoch: int) -> int:
        if epoch < self.warmup_epochs:
            return self.initial_context
        ramp_progress = min(1.0, (epoch - self.warmup_epochs) / self.rampup_epochs)
        context = self.initial_context + (
            (self.final_context - self.initial_context) *
            (1 - np.cos(np.pi * ramp_progress)) / 2
        )
        return int(context)
```

**Drawbacks:**
- Longer training time
- Requires DataLoader modification

**Mitigation:** Can fall back to fixed 14-day context if curriculum doesn't help

---

### Phase 2: Architecture Modifications (Week 4-5)
**Impact: High | Effort: Medium-High | Risk: Medium**

#### 2.1 RSA Fusion Regularization Loss

**Rationale:** Directly addresses RSA degradation by penalizing when fusion quality drops.

**Design Decision:** Soft hinge loss with target RSA=0.5, weight=0.05 relative to task losses.

**Implementation:**
```python
class FusionRegularizationLoss(nn.Module):
    def __init__(self, target_rsa: float = 0.5, margin: float = 0.1, loss_weight: float = 0.05):
        self.target_rsa = target_rsa
        self.margin = margin
        self.loss_weight = loss_weight

    def compute_rsa_loss(self, source_hidden: Dict[str, Tensor]) -> Tensor:
        # Compute pairwise RSA between source representations
        # Use differentiable correlation distance
        mean_rsa = self._compute_mean_rsa(source_hidden)
        violation = F.relu(self.target_rsa - mean_rsa - self.margin)
        return violation.pow(2) * self.loss_weight
```

**Drawbacks:**
- O(n^2) computation for RDM matrices
- May interfere with task optimization

**Mitigation:**
- Subsample to 32 timesteps for efficiency
- Compute every 5 batches, not every batch

---

#### 2.2 Equipment Encoder with Category Embeddings

**Rationale:** Disaggregating into 6 separate encoders creates inefficient narrow encoders. Category embeddings provide interpretability with fewer parameters.

**Design Decision:** Single encoder + 6 category embeddings (not 6 separate encoders).

**Implementation:**
```python
class EquipmentEncoder(nn.Module):
    def __init__(self, d_model: int = 128):
        self.category_embedding = nn.Embedding(6, d_model)
        # Categories: drone, tank, apc, artillery, aircraft, other
        self.drone_lag_projection = nn.Linear(1, d_model // 4)

    def forward(self, x, category_ids, drone_lag_features=None):
        cat_emb = self.category_embedding(category_ids)
        x = x + cat_emb
        if drone_lag_features is not None:
            lag_emb = self.drone_lag_projection(drone_lag_features)
            x = torch.cat([x, lag_emb], dim=-1)
        return self.encoder(x)
```

**Alternative Considered:** 6 separate sub-encoders
- Rejected: Too few features per encoder (1-3); inefficient

---

#### 2.3 Variable Encoder Depths

**Rationale:** Low-importance sources (equipment: 0.07) don't need full 4-layer encoders.

**Design Decision:** Reduce by 1 layer (not 2) as conservative compromise.

| Source | Current | Proposed | Importance |
|--------|---------|----------|------------|
| deepstate | 4 | 4 | 0.29 |
| firms | 4 | 4 | 0.30 |
| personnel | 4 | 3 | 0.06 |
| viina | 4 | 3 | 0.08 |
| drones | 4 | 3 | 0.07 (but high MI) |

**Drawbacks:**
- Reduced capacity for future data patterns

**Mitigation:** Keep architecture configurable via config

---

### Phase 3: Feature Engineering (Week 5-6)
**Impact: Medium | Effort: Medium | Risk: Low**

#### 3.1 Drone Multi-Lag Features

**Rationale:** Drones lead casualties by 7-27 days. Multiple lag features capture this range.

**Design Decision:** Multi-lag features at [7, 14, 21, 28] days (not single fixed lag).

**Implementation:**
```python
def add_drone_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_windows = [7, 14, 21, 28]
    for lag in lag_windows:
        df[f'drone_daily_lag{lag}'] = df['drone_daily'].shift(-lag)
    # Also add rolling averages
    df['drone_daily_7d_avg'] = df['drone_daily'].rolling(7, min_periods=1).mean()
    return df
```

**Drawbacks:**
- Loses 28 days at end of series (forward shift creates NaN)
- Feature expansion (4 new features)

**Mitigation:** Update observation masks; features are sparse anyway

---

#### 3.2 VIIRS PCA Reduction

**Rationale:** 7 highly correlated feature pairs with |r| > 0.8. Reduce redundancy.

**Design Decision:** PCA on radiance features only (3 components); keep quality features separate.

**Implementation:**
```python
radiance_features = ['viirs_radiance_mean', 'viirs_radiance_std',
                     'viirs_radiance_p50', 'viirs_radiance_p90']
quality_features = ['viirs_clear_sky_pct', 'viirs_moon_illumination']

# PCA on detrended radiance -> 3 components
# Final: 3 PCA + 2 quality = 5 VIIRS features (down from 8)
```

**Drawbacks:**
- PCA must be fit on training data only
- Less interpretable than raw features

**Mitigation:** Store PCA model in norm_stats for consistent transforms

---

#### 3.3 Recency Weighting for Aggregation

**Rationale:** More recent days should have higher weight in monthly aggregation.

**Design Decision:** Exponential decay with 14-day half-life (matches optimal context window).

**Implementation:**
```python
def aggregate_with_recency(daily_features, half_life_days=14):
    tau = half_life_days / np.log(2)
    days_from_end = np.arange(len(daily_features) - 1, -1, -1)
    weights = np.exp(-days_from_end / tau)
    weights = weights / weights.sum()
    return (daily_features * weights[:, np.newaxis]).sum(axis=0)
```

**Drawbacks:**
- Fixed weighting; not learned

**Mitigation:** Phase 4 can add learnable attention-based weighting

---

## Summary: Priority Matrix

| # | Optimization | Impact | Effort | Risk | Phase |
|---|--------------|--------|--------|------|-------|
| 0.1 | Context window 14 days | Critical | Low | Low | 0 |
| 0.2 | VIIRS detrending | High | Low | Low | 0 |
| 0.3 | Disaggregated equipment | High | Low | Low | 0 |
| 0.4 | Loss weight priors | High | Low | Medium | 0 |
| 1.1 | RSA early stopping | High | Medium | Medium | 1 |
| 1.2 | Gradient isolation | High | Low | Medium | 1 |
| 1.3 | Curriculum learning | Medium | Medium | Low | 1 |
| 2.1 | Fusion regularization | High | High | Medium | 2 |
| 2.2 | Category embeddings | Medium | Medium | Low | 2 |
| 2.3 | Variable encoder depths | Low | Low | Low | 2 |
| 3.1 | Drone lag features | Medium | Medium | Low | 3 |
| 3.2 | VIIRS PCA | Medium | Medium | Low | 3 |
| 3.3 | Recency weighting | Medium | Medium | Low | 3 |

---

## Rejected Proposals

| Proposal | Reason for Rejection |
|----------|---------------------|
| **Frozen epoch-10 fusion blending** | Training instability; doesn't address root cause; maintenance burden |
| **GradientReversal for heads** | Designed for domain adaptation; would harm auxiliary task learning |
| **6 separate equipment encoders** | Too few features per encoder; category embeddings more efficient |
| **Fixed lag (single value)** | Lag varies 7-27 days; multi-lag captures range better |

---

## Monitoring Requirements

After implementation, monitor these metrics:

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| RSA (cross-source similarity) | > 0.40 | < 0.30 |
| Regime accuracy | > 78% | < 70% |
| Casualty head loss proportion | < 40% | > 50% |
| Context window (curriculum) | Follows schedule | Stuck at initial |
| Training time per epoch | < 60s | > 120s |

---

## Next Steps

1. **Review this document** - Approve, modify, or reject proposals
2. **Phase 0 implementation** - Quick wins (estimated 1-2 hours)
3. **Baseline training run** - Validate Phase 0 changes
4. **Probe validation** - Rerun probe battery on new model
5. **Phase 1-3 implementation** - Based on Phase 0 results

---

## Appendix: Research References

- Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
- Bengio et al. (2009) "Curriculum Learning"
- Kriegeskorte et al. (2008) "Representational Similarity Analysis"
- Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
- Detrending comparison: https://link.springer.com/article/10.1007/s10614-024-10548-x
