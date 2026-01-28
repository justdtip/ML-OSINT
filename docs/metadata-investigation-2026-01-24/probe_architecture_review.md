# Probe Architecture Review: Equipment Loss Disaggregation Support

**Version:** 1.0
**Date:** 2026-01-24
**Author:** Architecture Review Agent
**Purpose:** Deep architecture analysis of probes for equipment loss disaggregation support

---

## Executive Summary

This review analyzes the ML_OSINT probe system architecture to evaluate its readiness for equipment loss disaggregation. The system already includes partial support for disaggregation via the `use_disaggregated_equipment` configuration option, but several architectural bottlenecks limit flexibility for more granular category analysis.

### Key Findings

1. **Existing Disaggregation Support**: The `MultiResolutionConfig` already supports basic equipment disaggregation (drones/armor/artillery) via a configuration flag.

2. **Probe Data Pipeline**: Probes load data independently without sharing a common data loader, leading to redundant loading and potential inconsistency.

3. **Feature Abstraction Gap**: No unified feature registry exists that could support dynamic disaggregation levels.

4. **HAN Input Layer Flexibility**: The `DailySourceEncoder` uses per-source feature counts, requiring model reinitialization for new category configurations.

5. **Attention Compatibility**: The attention mechanisms are compatible with variable-length equipment categories since they operate at the source level, not individual feature level.

---

## 1. Data Flow Diagram

```
                          Raw Data Sources
                                |
    +---------------------------+---------------------------+
    |           |          |          |          |         |
russia_losses  personnel  deepstate  firms     viina     viirs
_equipment.json           /*.json    /*.csv             /*.csv
    |           |          |          |          |         |
    v           v          v          v          v         v

         +-------------------------------------------------+
         |       multi_resolution_data.py                  |
         |  +-------------------------------------------+  |
         |  | load_equipment_daily()                    |  |  <-- AGGREGATED PATH
         |  |  - 12 cumulative features                 |  |
         |  |  - daily delta computation                |  |
         |  +-------------------------------------------+  |
         |                      OR                         |
         |  +-------------------------------------------+  |
         |  | load_drones_daily()   [drone, cruise_missiles]   |  <-- DISAGGREGATED PATH
         |  | load_armor_daily()    [tank, APC]                |      (when use_disaggregated_equipment=True)
         |  | load_artillery_daily()[field_artillery, MRL]     |
         |  +-------------------------------------------+  |
         +-------------------------------------------------+
                                |
                                v
                   +------------------------+
                   | MultiResolutionDataset |
                   |  - Temporal alignment  |
                   |  - Observation masks   |
                   |  - Normalization       |
                   +------------------------+
                                |
                                v
            +-------------------+-------------------+
            |                                       |
            v                                       v
+----------------------+               +----------------------+
| DailySourceEncoders  |               | MonthlySourceEncoders|
| (per-source models)  |               | (per-source models)  |
+----------------------+               +----------------------+
            |                                       |
            v                                       v
     DailyFusion                            MonthlyFusion
            |                                       |
            +-------------------+-------------------+
                                |
                                v
                   CrossResolutionFusion
                                |
                                v
                        TemporalEncoder
                                |
            +-------------------+-------------------+
            |           |           |               |
            v           v           v               v
    CasualtyHead  RegimeHead  AnomalyHead    ForecastHead



                    PROBE DATA PIPELINE
                    ===================

+----------------------------------------------------------+
|                    data_artifact_probes.py               |
|  +----------------------------------------------------+  |
|  | load_equipment_raw()                               |  |
|  |  - Direct JSON load (bypasses multi_resolution)    |  |
|  |  - Returns raw cumulative DataFrame                |  |
|  +----------------------------------------------------+  |
|  | EQUIPMENT_CATEGORIES dict                          |  |
|  |  - tanks: ['tank']                                 |  |
|  |  - apcs: ['APC']                                   |  |
|  |  - artillery: ['field_artillery', 'MRL']           |  |
|  |  - aircraft: ['aircraft', 'helicopter']            |  |
|  |  - drones: ['drone']                               |  |
|  |  - ... (9 categories total)                        |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
              |
              | (used by)
              v
+----------------------------------------------------------+
|     Probe Classes (Section 1.1)                          |
|  +----------------------------------------------------+  |
|  | EncodingVarianceComparisonProbe (1.1.1)            |  |
|  |  - Compares cumulative vs delta encoding variance  |  |
|  +----------------------------------------------------+  |
|  | EquipmentPersonnelRedundancyProbe (1.1.2)          |  |
|  |  - Correlation/partial correlation analysis        |  |
|  +----------------------------------------------------+  |
|  | EquipmentCategoryDisaggregationProbe (1.1.3)       |  |
|  |  - Per-category importance analysis                |  |
|  |  - Gradient magnitude through casualty head        |  |
|  +----------------------------------------------------+  |
|  | TemporalLagAnalysisProbe (1.1.4)                   |  |
|  |  - Cross-correlation at multiple lags              |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
```

---

## 2. Current Equipment Feature Extraction

### 2.1 Aggregated Path (Default)

Location: `analysis/multi_resolution_data.py::load_equipment_daily()`

```python
feature_cols = [
    'aircraft', 'helicopter', 'tank', 'APC', 'field_artillery',
    'MRL', 'drone', 'naval_ship', 'anti_aircraft_warfare',
    'special_equipment', 'vehicles_and_fuel_tanks', 'cruise_missiles'
]
# Returns: DataFrame with 12 features + daily deltas = 24 potential features
```

### 2.2 Disaggregated Path (Conditional)

Location: `analysis/multi_resolution_data.py::MultiResolutionConfig.get_effective_daily_sources()`

When `use_disaggregated_equipment=True`:
- Replaces single `equipment` source with three separate sources:
  - `drones`: drone, cruise_missiles (4 features with deltas)
  - `armor`: tank, APC (4 features with deltas)
  - `artillery`: field_artillery, MRL (4 features with deltas)

**Note**: Aircraft is explicitly EXCLUDED per optimization plan due to negative correlation with casualties.

### 2.3 Probe-Level Category Mapping

Location: `analysis/probes/data_artifact_probes.py`

```python
EQUIPMENT_CATEGORIES = {
    'tanks': ['tank'],
    'apcs': ['APC'],
    'artillery': ['field_artillery', 'MRL'],
    'aircraft': ['aircraft', 'helicopter'],
    'air_defense': ['anti_aircraft_warfare'],
    'drones': ['drone'],
    'naval': ['naval_ship'],
    'vehicles': ['vehicles_and_fuel_tanks', 'special_equipment'],
    'missiles': ['cruise_missiles']
}
# 9 semantic categories mapping to 12 raw equipment types
```

---

## 3. Architectural Bottlenecks for Disaggregation

### 3.1 Bottleneck: No Shared Data Loader in Probes

**Issue**: Probes use `load_equipment_raw()` which directly loads JSON, bypassing the `MultiResolutionDataset` infrastructure. This creates:
- Redundant data loading across probes
- Inconsistent preprocessing (probes compute their own deltas)
- No access to trained normalization statistics

**Impact**: Probe results may not reflect how the model actually sees equipment data.

**Recommendation**: Create a shared `ProbeDataManager` class that provides:
```python
class ProbeDataManager:
    def __init__(self, data_config: MultiResolutionConfig):
        self.dataset = MultiResolutionDataset(data_config, split='train')
        self.norm_stats = self.dataset.norm_stats

    def get_equipment_features(self, disaggregated: bool = True) -> pd.DataFrame:
        """Get equipment features matching model input pipeline."""
        ...

    def get_aligned_targets(self) -> pd.DataFrame:
        """Get casualty targets aligned with equipment dates."""
        ...
```

### 3.2 Bottleneck: Static Source Configuration

**Issue**: `SourceConfig` requires `n_features` to be specified at model initialization. Changing the disaggregation level requires:
1. Modifying `MultiResolutionConfig`
2. Recreating the dataset
3. Reinitializing the model with new `daily_source_configs`

**Impact**: Cannot dynamically switch between aggregated and disaggregated equipment without model rebuild.

**Current Code Path**:
```
MultiResolutionConfig.use_disaggregated_equipment
    -> MultiResolutionConfig.get_effective_daily_sources()
        -> Returns ['drones', 'armor', 'artillery'] instead of ['equipment']
            -> MultiResolutionDataset creates separate tensors per source
                -> MultiResolutionHAN.daily_source_encoders has separate encoder per source
```

**Recommendation**: This is acceptable for training/evaluation cycles but limits interactive analysis. Consider:
- Checkpoint metadata should record `use_disaggregated_equipment` setting
- Probes should auto-detect configuration from checkpoint

### 3.3 Bottleneck: Feature Index Opacity

**Issue**: The `conflict_data_loader.py` and `hierarchical_attention_network.py` define equipment features as fixed-index arrays without runtime mapping.

Location: `analysis/conflict_data_loader.py::extract_domain_features()`
```python
# Equipment features (29)
equipment_features = np.zeros((n_months, 29))
if 'aircraft' in equipment.columns:
    equipment_features[:, 4] = equipment['aircraft']  # aircraft_total
if 'helicopter' in equipment.columns:
    equipment_features[:, 10] = equipment['helicopter']  # heli_total
# ... hardcoded indices
```

**Impact**:
- No programmatic way to query "which features correspond to tanks?"
- Gradient analysis requires manual index mapping
- Adding new equipment types requires code changes in multiple files

**Recommendation**: Create a feature registry:
```python
# config/feature_registry.py
EQUIPMENT_FEATURE_REGISTRY = {
    'aircraft': {'indices': [4], 'category': 'aviation'},
    'helicopter': {'indices': [10], 'category': 'aviation'},
    'tank': {'indices': [17], 'category': 'armor'},
    # ...
}
```

### 3.4 Bottleneck: Probe Specification vs Implementation Gap

**Issue**: The Probe-specs.md specifies equipment category disaggregation (Probe 1.1.3) with this interface:
```
Equipment categories: [tanks, APCs, artillery, MLRS, anti-aircraft, aircraft, helicopters, drones, vehicles, fuel_tanks, special_equipment, ships]

For each category:
  - Compute standalone correlation with daily casualties
  - Compute gradient magnitude through casualty head
  - Rank categories by predictive contribution
```

But `EquipmentCategoryDisaggregationProbe` uses a slightly different category mapping from `EQUIPMENT_CATEGORIES`.

**Recommendation**: Align probe category definitions with Probe-specs.md specification.

---

## 4. HAN Input Layer Analysis

### 4.1 Daily Source Encoder Architecture

Location: `analysis/multi_resolution_han.py::DailySourceEncoder`

```python
class DailySourceEncoder(nn.Module):
    def __init__(self, source_config: SourceConfig, d_model=128, ...):
        self.n_features = source_config.n_features
        self.feature_projection = nn.Sequential(
            nn.Linear(source_config.n_features, d_model),  # <-- Fixed input dimension
            ...
        )
        self.feature_embedding = nn.Embedding(source_config.n_features, d_model)
```

**Key Insight**: The `feature_projection` layer has a fixed input dimension set at initialization. This means:
- Model weights are tied to specific feature count
- Cannot add/remove equipment categories without retraining
- Checkpoint loading will fail if n_features changes

### 4.2 Attention Mechanism Compatibility

**Good News**: The attention mechanisms operate at the source level, not individual feature level:

```python
# From multi_resolution_han.py
class DailySourceEncoder:
    # Features are projected to d_model, then attention happens across timesteps
    # Individual equipment categories are NOT separate attention positions
```

This means:
- Fusing tanks+APCs into "armor" is architecturally sound
- Splitting "equipment" into drones/armor/artillery works without attention changes
- The cross-source attention in `DailyFusion` treats each source holistically

### 4.3 Required Changes for Variable Categories

To support runtime-variable equipment categories:

1. **Option A: Feature Masking** (Minimal changes)
   - Keep all 12 equipment features in model
   - Use attention masking to "disable" unwanted categories
   - Pro: No weight changes needed
   - Con: Wasted parameters for disabled categories

2. **Option B: Dynamic Projection** (Moderate changes)
   - Replace fixed `nn.Linear` with dynamically-sized projection
   - Store feature dimension mapping in checkpoint
   - Pro: Efficient parameter usage
   - Con: Requires checkpoint format changes

3. **Option C: Feature-Level Attention** (Significant changes)
   - Replace flat feature projection with per-feature embeddings
   - Apply attention across features within equipment source
   - Pro: Maximum flexibility, interpretable per-feature importance
   - Con: Significant architecture change, retraining required

**Recommendation**: Option A for probing existing models, Option C for future architecture.

---

## 5. Recommended Refactoring Approach

### Phase 1: Immediate (Probe Infrastructure)

1. **Create ProbeDataManager**
   ```python
   # analysis/probes/probe_data_manager.py
   class ProbeDataManager:
       """Unified data loading for all probes."""

       def __init__(self, checkpoint_path: Path):
           # Auto-detect config from checkpoint
           self.config = self._load_config_from_checkpoint(checkpoint_path)
           self.dataset = MultiResolutionDataset(self.config, split='train')

       def get_equipment_by_category(self) -> Dict[str, pd.DataFrame]:
           """Return equipment data grouped by EQUIPMENT_CATEGORIES."""
           ...
   ```

2. **Align EQUIPMENT_CATEGORIES with Probe-specs.md**
   - Update `data_artifact_probes.py` to match spec categories exactly
   - Add docstring noting category definitions

3. **Add Feature Registry**
   ```python
   # config/feature_registry.py
   from dataclasses import dataclass
   from typing import List, Dict

   @dataclass
   class FeatureSpec:
       name: str
       indices: List[int]
       category: str
       source: str

   EQUIPMENT_FEATURES: Dict[str, FeatureSpec] = {...}
   ```

### Phase 2: Short-term (Model Configuration)

1. **Store disaggregation config in checkpoints**
   - Add `use_disaggregated_equipment` to checkpoint metadata
   - Update `run_probes.py` to read this from checkpoint

2. **Create config validation**
   - Verify probe config matches model config
   - Warn if mismatch detected

### Phase 3: Medium-term (Architecture)

1. **Feature-Level Attention for Equipment**
   - Add `EquipmentAttentionEncoder` that operates at feature level
   - Each equipment type gets its own learnable embedding
   - Output per-category importance weights for interpretability

2. **Dynamic Source Configuration**
   - Allow runtime source configuration changes
   - Checkpoint stores source configuration as metadata
   - Model auto-configures based on loaded checkpoint

---

## 6. Compatibility Assessment

| Disaggregation Level | Current Support | HAN Compatible | Probe Compatible |
|---------------------|-----------------|----------------|------------------|
| Aggregated (1 source, 24 features) | Yes (default) | Yes | Yes |
| 3-way (drones/armor/artillery) | Yes (config flag) | Yes | Partial |
| 9-way (semantic categories) | No | Yes (with changes) | Yes (in probes only) |
| 12-way (raw equipment types) | No | Possible | Yes |

### Current State Summary

- **Training Pipeline**: Supports 2 modes via `use_disaggregated_equipment`
- **Probes**: Use independent loading with 9-category mapping
- **HAN**: Fixed per-source feature dimensions, flexible at source level

---

## 7. Conclusion

The current architecture provides a foundation for equipment disaggregation but has several gaps:

1. **Data Loading Inconsistency**: Probes bypass the training pipeline's data loading
2. **Configuration Opacity**: Checkpoint doesn't encode disaggregation settings
3. **Feature Index Hardcoding**: No programmatic feature-to-category mapping

The recommended phased approach prioritizes:
- Phase 1: Align probe data loading with training pipeline
- Phase 2: Store configuration in checkpoints for reproducibility
- Phase 3: Enable feature-level attention for maximum flexibility

The attention mechanisms are fundamentally compatible with variable-length equipment categories since they operate at the source level rather than individual feature level. This means the core transformer architecture can support any disaggregation scheme without fundamental redesign.

---

## Appendix A: File Locations

| Component | File Path |
|-----------|-----------|
| Multi-resolution data loading | `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_data.py` |
| HAN model architecture | `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py` |
| Original conflict data loader | `/Users/daniel.tipton/ML_OSINT/analysis/conflict_data_loader.py` |
| Probe infrastructure | `/Users/daniel.tipton/ML_OSINT/analysis/probes/__init__.py` |
| Data artifact probes | `/Users/daniel.tipton/ML_OSINT/analysis/probes/data_artifact_probes.py` |
| Probe runner | `/Users/daniel.tipton/ML_OSINT/analysis/probes/run_probes.py` |
| Feature selection utilities | `/Users/daniel.tipton/ML_OSINT/analysis/feature_selection.py` |
| Interpolation data loaders | `/Users/daniel.tipton/ML_OSINT/analysis/interpolation_data_loaders.py` |
| Probe specifications | `/Users/daniel.tipton/ML_OSINT/docs/Probe-specs.md` |

## Appendix B: Equipment Category Reference

### Raw Equipment Types (12)
```
aircraft, helicopter, tank, APC, field_artillery, MRL, drone,
naval_ship, anti_aircraft_warfare, special_equipment,
vehicles_and_fuel_tanks, cruise_missiles
```

### Semantic Categories (9 from data_artifact_probes.py)
```
tanks, apcs, artillery, aircraft, air_defense, drones, naval, vehicles, missiles
```

### Disaggregated Sources (3 when use_disaggregated_equipment=True)
```
drones: [drone, cruise_missiles]
armor: [tank, APC]
artillery: [field_artillery, MRL]
```

### Excluded from Disaggregation
```
aircraft: Negative correlation with casualties (per optimization plan)
```
