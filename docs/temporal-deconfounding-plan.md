# Temporal Deconfounding Implementation Plan

**Created**: 2026-01-30
**Status**: Implementation Ready
**Goal**: Fix early overfitting caused by temporal trend confounding while preserving day-level tactical state prediction capability.

---

## Problem Statement

Training exhibits severe early overfitting:
- Best validation loss at epoch 2-3
- Validation loss increases continuously thereafter
- Train loss continues decreasing (model memorizes)

**Root Cause** (from probe analysis):
- 71% of feature correlations are spurious (driven by shared time trends)
- Non-stationary cumulative features enable "time cheating"
- Model learns "what time is it?" rather than "what do these features predict?"

---

## Approved Changes

### 1. Delta Encoding ✅ APPROVED - IMPLEMENTING FIRST

**Rationale**: Replace cumulative features with daily differences to remove monotonic time trends.

**Current State** (from agent analysis):
- **New system** (`multi_resolution_data.py`): Already implements delta-only for `personnel`, `drones`, `armor`, `artillery`
- **Legacy system** (`conflict_data_loader.py`): Still uses cumulative features
- Key helper functions exist: `_load_equipment_base()`, `_compute_delta_features()`

**Files to Modify**:

| File | Function | Lines | Current | Change |
|------|----------|-------|---------|--------|
| `multi_resolution_data.py` | `load_equipment_daily()` | 218-278 | Cumulative | Apply `_compute_delta_features()` |
| `conflict_data_loader.py` | `extract_domain_features()` | 532-569 | Cumulative | Deprecate or convert to delta |

**Missing Equipment Loaders to Create**:
- `load_naval_daily()` - naval ship losses
- `load_special_equipment_daily()` - special equipment
- `load_vehicles_daily()` - vehicles and fuel tanks

**Implementation Pattern** (existing, reusable):
```python
def _compute_delta_features(df, base_cols, prefix):
    """Convert cumulative to delta-only features."""
    for col in available_cols:
        result[f'{col}_daily'] = df[col].diff().fillna(0)
        result[f'{col}_7day_avg'] = result[f'{col}_daily'].rolling(7, min_periods=1).mean().fillna(0)
        result[f'{col}_volatility'] = result[f'{col}_daily'].rolling(7, min_periods=2).std().fillna(0)
    return result
```

**Edge Cases**:
- First observation: `.diff()` returns NaN → fill with 0
- Data gaps: Already handled by observation mask
- Negative deltas: Possible (equipment recovered), handle via log1p after clipping

---

### 2. Feature Detrending ✅ APPROVED - IMPLEMENTING SECOND

**Rationale**: Remove slow-moving trends while preserving high-frequency daily fluctuations.

**Current State**:
- VIIRS has optional detrending via first-order differencing
- Personnel has rolling means but as features, not detrending
- No systematic detrending across all sources

**Implementation Location**:
- Add detrending AFTER loading, BEFORE normalization
- Insert in `_load_all_sources()` method
- Create new `analysis/preprocessing_utils.py` module

**Recommended Approach**: Subtract N-day rolling mean
```python
def detrend_features_vectorized(features, config):
    for source_name, (df, mask) in features.items():
        window = config.detrending_window.get(source_name, 14)
        rolling_mean = df[feature_cols].rolling(window=window, min_periods=1).mean()
        df[feature_cols] = df[feature_cols] - rolling_mean
    return features
```

**Per-Source Window Sizes**:
| Source | Window | Rationale |
|--------|--------|-----------|
| personnel | 14 days | Casualties reported daily |
| equipment (drones, armor, artillery) | 14 days | Updates frequently |
| deepstate_raion | 7 days | Front line updates sporadically |
| firms_expanded_raion | 7 days | Satellite observations |
| default | 14 days | Balance noise vs trend |

**Columns to EXCLUDE from detrending** (already processed):
- `*_rolling*`, `*_volatility*`, `*_momentum*` patterns

**Performance**: ~1.5 seconds for 30K features × 1,426 timesteps (vectorized)

---

### 3. Temporal Regularization ✅ APPROVED - IMPLEMENTING THIRD

**Rationale**: Penalize the model for learning time-dependent shortcuts.

**Current State**:
- Loss computed at `train_multi_resolution.py:2161`
- `MultiTaskLoss` uses uncertainty-weighted combination
- No temporal regularization exists

**Implementation Location**:
```python
# Line 2158-2161 in train_multi_resolution.py
task_losses = self._compute_losses(outputs, batch)
# >>> INSERT TEMPORAL REGULARIZATION HERE <<<
total_loss, task_weights = self.multi_task_loss(task_losses)
```

**Recommended Approach**: Hybrid of two components

**Component 1: Correlation Penalty** (weight: 0.01)
```python
def compute_temporal_correlation_penalty(predictions, positions):
    """Predictions should not correlate with temporal position."""
    pos_normalized = (positions - positions.mean()) / positions.std()
    pred_normalized = (predictions - predictions.mean()) / predictions.std()
    correlations = (pred_normalized * pos_normalized).mean(dim=1)
    penalty = F.relu(correlations).mean()  # Only penalize positive correlation
    return penalty
```

**Component 2: Smoothness Penalty** (weight: 0.001)
```python
def compute_temporal_roughness_penalty(predictions):
    """Penalize overly smooth predictions (indicates position-based learning)."""
    deltas = predictions[:, 1:, :] - predictions[:, :-1, :]
    smoothness = deltas.pow(2).mean(dim=1)
    target_roughness = 0.1
    penalty = F.relu(target_roughness - smoothness).mean()
    return penalty
```

**Configuration** (add to `training_config.py`):
```python
@dataclass
class TemporalRegularizationConfig:
    enabled: bool = False
    correlation_weight: float = 0.01
    smoothness_weight: float = 0.001
```

---

## Under Consideration

### 4. Seasonal Decomposition ❌ NOT IMPLEMENTING

**Rationale**: Conflict data may not have strong seasonality like economic data.

**Concerns**:
- Risk of removing meaningful tactical patterns (offensive operations may cluster)
- Delta encoding + detrending should address the core issue
- More complex than necessary

**Decision**: Defer. Revisit if approved changes don't resolve overfitting.

---

## Implementation Order

### Phase 1: Delta Encoding
1. Verify existing delta helpers work correctly
2. Apply `_compute_delta_features()` to `load_equipment_daily()`
3. Create missing loaders (naval, special_equipment, vehicles)
4. Add deprecation warnings to legacy `conflict_data_loader.py`
5. Test: Run probe 1.1.1 to verify stationarity improves

### Phase 2: Detrending
1. Create `analysis/preprocessing_utils.py` with `detrend_features_vectorized()`
2. Add `apply_detrending` config option to `MultiResolutionConfig`
3. Integrate into `_load_all_sources()` pipeline
4. Test: Run probe 1.2.3 (Trend Confounding Test) to verify reduction

### Phase 3: Temporal Regularization
1. Add `TemporalRegularizer` class to `train_multi_resolution.py`
2. Implement correlation + smoothness penalties
3. Extend `MultiTaskLoss.forward()` to accept temporal reg term
4. Add config to `training_config.py`
5. Test: Train and verify val loss improves beyond epoch 3

---

## Success Criteria

After implementation:
1. ✅ Validation loss improves beyond epoch 3
2. ✅ Train/val loss curves more parallel (less divergence)
3. ✅ Probe 1.1.2: Correlation reduction < 30% (down from 71%)
4. ✅ Previously unused sources show non-zero impact
5. ✅ Probe 1.1.1: > 80% features stationary (up from 33%)

---

## Rollback Plan

All changes are configurable:
```python
# Delta encoding: Use existing cumulative loaders
use_delta_encoding: bool = True  # Set False to revert

# Detrending: Skip detrending step
apply_detrending: bool = True  # Set False to revert

# Temporal regularization: Set weights to 0
temporal_reg_config.enabled: bool = True  # Set False to revert
```

---

## Files Summary

| File | Changes |
|------|---------|
| `analysis/multi_resolution_data.py` | Delta encoding for equipment, detrending integration |
| `analysis/preprocessing_utils.py` | NEW - Detrending functions |
| `analysis/train_multi_resolution.py` | Temporal regularization in loss |
| `analysis/training_config.py` | New config options |
| `config/paths.py` | No changes |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial plan created | Claude |
| 2026-01-30 | Added detailed implementation analysis from agents | Claude |

