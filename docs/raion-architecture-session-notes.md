# Raion-Level Architecture Session Notes

**Date**: 2026-01-27
**Context**: Designing fine-grained tactical prediction at raion (district) level

---

## Key Decisions Made

### 1. Spatial Resolution: Raion Level
- Rejected aggregated regions (6 regions) in favor of raion-level (district)
- Ukraine has 629 raions across 28 oblasts
- 325 conflict-relevant raions in 13 frontline oblasts
- Downloaded GADM admin level 2 boundaries to `data/boundaries/ukraine_raions.geojson`

### 2. Architecture Design

**Each raion = one data source with its own encoder (8 attention heads)**

Three-level hierarchy:
1. **RaionEncoder**: Temporal attention within each raion's feature sequence
2. **CrossRaionAttention**: "What in raion A predicts raion B?" with geographic adjacency prior
3. **TemporoSpatialFusion**: "What macro/national patterns predict this raion?"

Design doc: `docs/raion-architecture-design.md`

### 3. Data Sources for Raion Features

**Research document**: `docs/ukrain-region-proxy-sources.md` (47 sources identified)

**Primary sources (already have):**
- VIINA: 6-hour village-level events with ADM2 (raion) codes, BERT-classified event types
- FIRMS: GPS fire hotspots (assign to raion via point-in-polygon)
- DeepState: Territorial control GeoJSON (assign to raion)
- VIIRS: Nighttime lights (infrastructure proxy)

**To acquire:**
- WarSpotting API: GPS equipment losses
- Air Raid Alert API: Real-time alerts with historical archive
- GeoConfirmed: 5,455+ geolocated events

### 4. VIINA Dataset Details

**Location**: `data/viina/extracted/`

**Key files:**
- `event_info_latest_YYYY.csv`: Event details with ADM1_NAME, ADM2_NAME, ADM2_CODE, lat/lon
- `event_labels_latest_YYYY.csv`: BERT-classified event types (t_artillery, t_airstrike, t_uav, t_armor, t_control, etc.)

**Current loader** (`load_viina_daily`): Only uses ADM1 (oblast) level
**Needed**: New `load_viina_raion()` that uses ADM2 and includes event type classifications

### 5. Per-Raion Feature Vector

```
For each raion, daily features include:
├── VIINA event types: t_artillery, t_airstrike, t_uav, t_armor, t_control, t_firefight, t_milcas, t_civcas, ...
├── FIRMS: fire_count, brightness_mean, frp_sum
├── DeepState: unit_count, frontline_km (if applicable)
└── Total: ~20-30 features per raion
```

### 6. Implementation Tasks

1. ✅ Downloaded raion boundaries (GADM)
2. ✅ Identified 325 conflict-relevant raions
3. ✅ Updated `load_viina_daily()` to use ADM2 (raion) + event labels
4. ✅ Created raion spatial loader for FIRMS point-in-polygon (`analysis/loaders/raion_spatial_loader.py`)
   - RaionBoundaryManager: Loads 291 frontline raions, efficient point-in-polygon lookup
   - FIRMSRaionLoader: 192,520 hotspots assigned, 84 features (20 raions × 4 features)
5. ✅ Implemented CrossRaionAttention + RaionEncoder (`analysis/cross_raion_attention.py`)
   - GeographicAdjacency: Haversine distance-based prior, 629 centroids loaded
   - CrossRaionAttention: 8-head attention with geographic bias, temporal broadcast
   - RaionEncoder: Per-raion transformer encoder with positional encoding
6. ✅ Built full RaionHAN model (`analysis/raion_han.py`)
   - MacroEncoder: National-level feature encoding
   - TemporoSpatialFusion: Connects macro to raion with gated attention
   - RaionForecastHead: Shared weights + raion embedding for efficient prediction
   - 1.36M parameters total (shared weights architecture)
7. ✅ Integrated geographic prior into MultiResolutionHAN
   - `analysis/geographic_source_encoder.py`: GeographicSourceEncoder + GeographicDailyCrossSourceFusion
   - `analysis/loaders/viirs_spatial_loader.py`: Tile-based VIIRS with 6 regions (34 features)
   - Added `use_geographic_prior=True` option to MultiResolutionHAN
   - Spatial sources (viina, firms) now use cross-raion attention with geographic adjacency prior
8. ⬜ Training loop modifications for per-raion predictions

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAION SOURCE ENCODERS                            │
│  Bakhmut    ──→ [RaionEncoder 8h] ──→ bakhmut_repr                      │
│  Avdiivka  ──→ [RaionEncoder 8h] ──→ avdiivka_repr                      │
│  ... (N active raions)                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CROSS-RAION ATTENTION                              │
│  "What in raion A predicts raion B?" (with geographic adjacency prior)  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       MACRO-TEMPORAL CONTEXT                             │
│  National equipment/personnel + monthly indicators + ISW narratives     │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEMPORO-SPATIAL FUSION                                │
│  "What macro patterns predict this raion?"                               │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RAION PREDICTION HEADS                              │
│  Per-raion forecasts: [batch, horizon, n_features_raion]                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Files Created/Modified This Session

### Created:
- `scripts/download_raion_boundaries.py` - Downloads GADM Ukraine admin2
- `data/boundaries/ukraine_raions.geojson` - 629 raion polygons
- `data/boundaries/frontline_raions.json` - 325 conflict-relevant raions
- `docs/raion-architecture-design.md` - Full architecture design
- `docs/attention-architecture-deep-dive.md` - Attention mechanism explainer
- `docs/raion-architecture-session-notes.md` - This file
- `analysis/loaders/raion_spatial_loader.py` - Raion-level spatial loader with point-in-polygon
- `analysis/cross_raion_attention.py` - CrossRaionAttention + RaionEncoder modules
- `analysis/raion_han.py` - Full RaionHAN model (1.36M params)
- `analysis/geographic_source_encoder.py` - Geographic attention for spatial sources
- `analysis/loaders/viirs_spatial_loader.py` - Tile-based VIIRS spatial loader

### Modified:
- `analysis/multi_resolution_han.py` - Added DailyForecastingHead class
- `analysis/multi_resolution_data.py` - Added daily_forecast_targets, updated load_viina_daily for ADM2
- `analysis/train_multi_resolution.py` - Added daily forecast loss computation
- `analysis/backtesting.py` - Added daily resolution forecasting mode
- `analysis/loaders/__init__.py` - Export raion + VIIRS spatial loaders
- `analysis/multi_resolution_han.py` - Added use_geographic_prior option, geographic fusion integration

---

## Next Steps

1. ✅ Updated `load_viina_daily()` to use ADM2 (raion) + event labels (280 features, 100 raions)
2. ✅ Created raion spatial loader for FIRMS (84 features, 20 top raions)
3. ✅ Created VIIRS tile-based loader (34 features, 6 regions)
4. ✅ Implemented CrossRaionAttention with geographic adjacency prior
5. ✅ Implemented RaionEncoder (per-raion transformer encoder)
6. ✅ Built full RaionHAN model (MacroEncoder, TemporoSpatialFusion, RaionForecastHead)
7. ✅ Integrated geographic prior into MultiResolutionHAN (`use_geographic_prior=True`)
8. ⬜ Train MultiResolutionHAN with geographic prior enabled
9. ⬜ Evaluate cross-raion attention patterns (are nearby raions attending to each other?)
10. ⬜ Compare prediction accuracy with/without geographic prior
