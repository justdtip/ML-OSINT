# Raion-Level Spatial Architecture Design

## Overview

This document describes the architecture for raion-level (district-level) conflict prediction in the Multi-Resolution HAN. The goal is fine-grained tactical state prediction at the smallest practical administrative scale.

## Data Foundation

### Raion Boundaries
- **Source**: GADM 4.1 Ukraine admin level 2
- **Location**: `data/boundaries/ukraine_raions.geojson`
- **Total raions**: 629 across 28 oblasts
- **Conflict-relevant**: 325 raions across 13 oblasts

### Active Raion Filtering
Not all 325 raions have meaningful conflict data. We filter to raions with:
- At least N fire detections (FIRMS)
- At least M unit position updates (DeepState)
- Any recorded conflict events (HDX)

This reduces to approximately 50-100 active raions.

## Architecture

### Level 1: Raion Encoders

Each active raion gets its own encoder with 8 attention heads.

```
RaionEncoder(raion_id):
    Input: [batch, seq_len, n_features_raion]
    - FIRMS features: fire_count, brightness, frp, day_ratio
    - DeepState features: unit_count, frontline_km, attack_directions
    - (Optional) conflict events if available at raion level

    Architecture:
    - Source embedding (raion-specific)
    - Positional encoding (temporal)
    - N transformer layers with 8 heads

    Output: [batch, seq_len, d_model]
```

### Level 2: Cross-Raion Attention

Captures spatial dependencies between raions.

```
CrossRaionAttention:
    "What features in raion A predict changes in raion B?"

    Inputs: Dict[raion_id, raion_repr]  # All raion representations

    Mechanism:
    - Geographic adjacency prior (nearby raions attend more strongly)
    - Learned attention on top of prior
    - Optional: frontline topology (raions on same front segment)

    Output: Dict[raion_id, cross_raion_repr]
```

**Geographic Adjacency Matrix:**
```python
# Pre-computed from raion polygons
adjacency[raion_i, raion_j] = 1.0 if neighbors else 0.0
distance_weight[raion_i, raion_j] = exp(-distance_km / scale)

# Attention with geographic prior
attn_logits = Q @ K.T + log(adjacency + epsilon)
```

### Level 3: Macro-Temporal Context

National-level signals that affect all raions.

```
MacroEncoder:
    Inputs:
    - National equipment losses (not regionalizable)
    - National personnel casualties (not regionalizable)
    - Monthly humanitarian indicators
    - ISW narrative embeddings (strategic context)

    Output: macro_context [batch, seq_len, d_model]
```

### Level 4: Temporo-Spatial Fusion

Connects macro patterns to regional predictions.

```
TemporoSpatialFusion:
    "What macro/temporal patterns predict this raion's state?"

    Inputs:
    - cross_raion_repr: Dict[raion_id, Tensor]
    - macro_context: Tensor

    Mechanism:
    - Each raion attends to macro context
    - Learns which national-level signals matter for each region
    - Frontline raions may weight military signals higher
    - Rear raions may weight logistics/humanitarian signals higher

    Output: Dict[raion_id, fused_repr]
```

### Level 5: Raion Prediction Heads

Per-raion predictions for the next horizon days.

```
RaionForecastHead(raion_id):
    Input: fused_repr for this raion
    Output: [batch, horizon, n_features_raion]

    Predicts:
    - Fire activity (count, intensity)
    - Territorial changes (if DeepState data available)
    - Conflict intensity proxy
```

## Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAION SOURCE ENCODERS                            │
│                                                                          │
│  Bakhmut    ──→ [RaionEncoder 8h] ──→ bakhmut_repr                      │
│  Avdiivka  ──→ [RaionEncoder 8h] ──→ avdiivka_repr                      │
│  Marinka   ──→ [RaionEncoder 8h] ──→ marinka_repr                       │
│  ... (N active raions)                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CROSS-RAION ATTENTION                              │
│                                                                          │
│  "What in raion A predicts raion B?"                                    │
│                                                                          │
│  All raion reprs ──→ [CrossRaionAttention] ──→ cross_raion_reprs        │
│                       (with geographic prior)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       MACRO-TEMPORAL CONTEXT                             │
│                                                                          │
│  National equipment  ──┐                                                 │
│  National personnel  ──┼──→ [MacroEncoder] ──→ macro_context            │
│  Monthly indicators  ──┤                                                 │
│  ISW narratives      ──┘                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEMPORO-SPATIAL FUSION                                │
│                                                                          │
│  "What macro patterns predict this raion?"                               │
│                                                                          │
│  cross_raion_reprs ──┬──→ [TemporoSpatialFusion] ──→ fused_reprs        │
│  macro_context     ──┘     (per-raion attention to macro)               │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RAION PREDICTION HEADS                              │
│                                                                          │
│  fused_reprs ──→ [BakhmutHead]   ──→ bakhmut_forecast   [horizon, feat] │
│             ──→ [AvdiivkaHead]  ──→ avdiivka_forecast  [horizon, feat] │
│             ──→ [MarinkaHead]   ──→ marinka_forecast   [horizon, feat] │
│             ──→ ...                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementation Tasks

### Task 1: Raion Spatial Loader
Create loader that assigns FIRMS/DeepState points to raions.

**Files to create:**
- `analysis/loaders/raion_spatial_loader.py`

**Functions:**
- `load_raion_boundaries()` - Load GADM polygons
- `assign_point_to_raion(lat, lon)` - Point-in-polygon lookup
- `RaionSpatialLoader.load_daily_features()` - Per-raion daily features
- `get_active_raions(min_observations)` - Filter to active raions

### Task 2: Cross-Raion Attention Module
Implement geographic-aware attention between raions.

**Files to create:**
- `analysis/cross_raion_attention.py`

**Classes:**
- `GeographicAdjacency` - Compute/cache raion adjacency matrix
- `CrossRaionAttention` - Attention with geographic prior

### Task 3: Raion HAN Architecture
Build the full raion-level model.

**Files to modify:**
- `analysis/multi_resolution_han.py` (or new file)

**Classes:**
- `RaionEncoder` - Per-raion transformer encoder
- `MacroEncoder` - National-level encoder
- `TemporoSpatialFusion` - Connects macro to raion
- `RaionHAN` - Full model combining all components

### Task 4: Training Integration
Modify training loop for raion-level predictions.

**Files to modify:**
- `analysis/train_multi_resolution.py`
- `analysis/multi_resolution_data.py`

**Changes:**
- Add raion forecast targets
- Per-raion loss computation
- Aggregate metrics across raions

## Computational Considerations

### Parameter Count
- N raions × encoder parameters per raion
- Shared cross-attention parameters
- Shared macro encoder
- N prediction heads

For 50 active raions with d_model=128:
- ~50 × 500K = 25M parameters in raion encoders
- ~2M in cross-attention
- ~2M in macro encoder
- ~50 × 100K = 5M in prediction heads
- **Total: ~35M parameters** (vs ~10M current)

### Memory Optimization
- Share encoder weights across raions (with raion embedding)
- Gradient checkpointing for raion encoders
- Sparse cross-attention (only adjacent raions)

## Evaluation Metrics

### Per-Raion Metrics
- MSE/MAE per raion
- Correlation per raion
- Frontline vs rear area comparison

### Aggregate Metrics
- Weighted average by raion activity level
- Detection of raion-level events (escalation, territorial change)

### Spatial Metrics
- Cross-raion prediction accuracy (does A help predict B?)
- Geographic coherence (nearby raions should correlate)
