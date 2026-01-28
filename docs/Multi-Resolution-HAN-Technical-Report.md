# Multi-Resolution Hierarchical Attention Network (HAN) Technical Report

## Comprehensive Architecture Analysis

**Model:** Multi-Resolution Hierarchical Attention Network
**Application:** Ukraine Conflict Dynamics Prediction from OSINT Sources
**Report Date:** 2026-01-28
**Status:** Raion-Ready Architecture with Full Feature Integration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Feature Decomposition Analysis](#3-feature-decomposition-analysis)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [Attention Mechanism Dynamics](#5-attention-mechanism-dynamics)
6. [Multi-Resolution Processing](#6-multi-resolution-processing)
7. [Raion-Level Spatial Architecture](#7-raion-level-spatial-architecture)
8. [Prediction Heads & Multi-Task Learning](#8-prediction-heads--multi-task-learning)
9. [Validation & Probe System](#9-validation--probe-system)
10. [Key Findings & Known Issues](#10-key-findings--known-issues)
11. [Parameter Counts & Efficiency](#11-parameter-counts--efficiency)
12. [File Reference](#12-file-reference)

---

## 1. Executive Summary

The Multi-Resolution HAN is a hierarchical attention network designed to predict Ukraine conflict dynamics by fusing heterogeneous OSINT data sources at their native temporal resolutions. The architecture processes:

- **Daily sources** (~1,426 timesteps): Equipment losses, personnel casualties, territorial control, fire hotspots, nighttime brightness
- **Monthly sources** (~48 timesteps): Satellite metadata, humanitarian indicators, displacement tracking
- **Narrative sources**: ISW assessment embeddings for strategic context

**Key Design Principles:**
1. **Never fabricate data** - Missing observations use learned `no_observation_token` embeddings
2. **Process at native resolution** - Daily and monthly data processed separately, then fused via attention
3. **Explicit observation masks** - Maintained throughout for proper attention masking
4. **Geographic prior integration** - Raion-level predictions use distance-based adjacency matrices

**Model Outputs:**
- Casualty prediction (regression with uncertainty)
- Regime classification (4-class: Initial Invasion, Stalemate, Counteroffensive, Attritional)
- Anomaly detection (binary outlier identification)
- Forecasting (next-period feature prediction)

---

## 2. Architecture Overview

### 2.1 Hierarchical Structure (6 Levels)

```
LEVEL 1: Daily Source Encoders (6 parallel encoders)
├── Equipment losses    → DailySourceEncoder(11 features)
├── Personnel casualties → DailySourceEncoder(3 features)
├── DeepState control   → DailySourceEncoder(5 features)
├── FIRMS fire hotspots → DailySourceEncoder(13 features)
├── VIINA territorial   → DailySourceEncoder(6 features)
└── VIIRS brightness    → DailySourceEncoder(variable)
    ↓ [each produces: batch × 1426 timesteps × 128-dim]

LEVEL 2: Daily Cross-Source Fusion
├── Cross-source attention between daily encoders
├── Source importance weighting (learned softmax gating)
└── Output: fused_daily [batch × 1426 × 128]
    ↓

LEVEL 3: Learnable Monthly Aggregation (Daily → Monthly)
├── Month query vectors attend to their constituent days
├── Cross-attention: months QUERY days
├── Causal mask: Each month only sees days before month end
└── Output: aggregated_daily [batch × n_months × 128]
    ↓

LEVEL 4: Monthly Source Encoders (5 parallel encoders)
├── Sentinel satellite  → MonthlyEncoder(7 features)
├── HDX conflict        → MonthlyEncoder(5 features)
├── HDX food prices     → MonthlyEncoder(10 features)
├── HDX rainfall        → MonthlyEncoder(6 features)
└── IOM displacement    → MonthlyEncoder(7 features)
    ↓ [each produces: batch × n_months × 128-dim]

LEVEL 5: Cross-Resolution Fusion
├── Bidirectional attention: aggregated_daily ↔ monthly
├── Gated fusion with learned mixing ratios
└── Output: fused_monthly [batch × n_months × 128]
    ↓

LEVEL 6: Temporal Encoder + Prediction Heads
├── Transformer encoder with positional encoding
├── Causal masking for autoregressive prediction
└── 4 prediction heads: casualty, regime, anomaly, forecast
```

### 2.2 Core Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `DailySourceEncoder` | multi_resolution_han.py:186 | Encode daily sources with observation masking |
| `MultiSourceDailyEncoder` | multi_resolution_modules.py | Fuse multiple daily sources |
| `LearnableMonthlyAggregator` | multi_resolution_modules.py | Attention-based daily→monthly aggregation |
| `MonthlyEncoder` | multi_resolution_modules.py | Encode monthly sources |
| `CrossResolutionFusion` | multi_resolution_modules.py | Bidirectional daily↔monthly attention |
| `TemporalEncoder` | multi_resolution_han.py | Long-range temporal patterns |

---

## 3. Feature Decomposition Analysis

### 3.1 Daily Features (Total: ~38 base features)

#### Equipment Losses (11 features)
```
Source: data/war-losses-data/.../russia_losses_equipment.json
Temporal Resolution: Daily
Observation Rate: ~95% (official updates most days)

Feature Categories:
├── Disaggregated (recommended):
│   ├── drones: UAV losses [Mutual Info: 0.449, HIGHEST]
│   ├── armor: APCs, tanks [r=0.221]
│   ├── artillery: Howitzers, MLRS
│   └── aircraft: Fixed-wing, helicopters [EXCLUDED: negative correlation]
│
└── Aggregated (legacy):
    └── total_equipment: Sum of all categories

Key Finding (Probe 1.1.2):
- Drone losses lead casualties by 7-27 days
- Disaggregated > aggregated for prediction
```

#### Personnel Casualties (3 features)
```
Source: data/war-losses-data/.../russia_losses_personnel.json
Temporal Resolution: Daily
Observation Rate: ~90%

Features:
├── personnel_total: Cumulative count
├── personnel_daily: Daily change (derived)
└── personnel_pow: POW count

Key Finding:
- Direct measure of target variable
- High collinearity risk with casualty prediction
```

#### DeepState Territorial Control (5 features)
```
Source: data/deepstate/wayback_snapshots/
Format: GeoJSON polygons/points
Temporal Resolution: Daily (snapshot frequency varies)
Observation Rate: ~60-70%

Features:
├── polygon_count: Number of controlled areas
├── multipolygon_count: Complex territorial shapes
├── point_count: Individual positions
├── linestring_count: Front lines
└── total_features: Aggregate count

Key Finding:
- Captures territorial state, not dynamics
- Forward-fill imputation appropriate (stability assumption)
```

#### FIRMS Fire Hotspots (13 features)
```
Source: data/firms/DL_FIRE_SV-C2_706038/
Temporal Resolution: Daily (satellite passes)
Observation Rate: Variable (cloud cover)

Base Features:
├── fires_total: Daily fire count
├── frp_mean/max/sum: Fire Radiative Power statistics
├── brightness_mean/max: Detection brightness
├── day_fire_fraction: Day vs night ratio
└── confidence_mean: Detection confidence

Spatial Decomposition (optional):
├── 5 regional tiles
├── 84 features from top 20 raions
└── Frontline concentration metrics

Key Finding:
- Strong spatial signal for tactical activity
- Cloud cover creates systematic gaps
```

#### VIINA Territorial Control (6 features)
```
Source: Alternative territorial data source
Temporal Resolution: Daily
Observation Rate: ~80%

Features:
├── control_state: Territorial status
├── change_events: Control transitions
└── intensity_metrics: Activity levels
```

#### VIIRS Nighttime Brightness (6 features)
```
Source: data/nasa/viirs_nightlights/viirs_daily_brightness_stats.csv
Temporal Resolution: Daily
Observation Rate: ~85%

Features:
├── radiance_mean: Average brightness
├── radiance_std: Brightness variability [ONLY NON-SPURIOUS]
├── radiance_p50/p90: Percentile statistics
└── coverage_fraction: Valid pixel ratio

Key Finding (Probe 1.2.1):
- VIIRS is a +10 day LAGGING indicator
- Captures "damage occurred" not "damage will occur"
- Detrending (first-order differencing) recommended
- radiance_std is only feature surviving detrend analysis
```

### 3.2 Monthly Features (Total: ~35 features)

#### Sentinel Satellite Metadata (7 features)
```
Source: data/sentinel/sentinel_timeseries_raw.json
Temporal Resolution: Monthly aggregation
Observation Rate: ~100%

Features:
├── acquisition_count: Images per month
├── cloud_cover_mean/std: Cloud metrics
├── processing_level: Data quality
└── coverage_area: Spatial extent

Purpose: Measures observation quality, not conflict state
```

#### HDX Conflict Events (5 features)
```
Source: data/hdx/ukraine/conflict_events_*.csv
Temporal Resolution: Monthly
Observation Rate: ~95%

Features:
├── event_count: Total conflict events
├── fatalities_best/low/high: Casualty estimates
└── incidents_by_type: Event categorization
```

#### HDX Food Security (10 features)
```
Source: data/hdx/ukraine/food_prices_*.csv
Temporal Resolution: Monthly
Observation Rate: ~90%

Features:
├── price_index: Food price tracking
├── availability_score: Supply metrics
└── regional_breakdown: Oblast-level prices
```

#### HDX Rainfall (6 features)
```
Source: data/hdx/ukraine/
Temporal Resolution: Monthly
Observation Rate: ~100%

Features:
├── precipitation_mm: Monthly rainfall
├── anomaly_percent: Deviation from normal
└── regional_coverage: Spatial distribution

Key Finding:
- Only source showing temporal sensitivity in shuffling test
- May encode seasonal/operational constraints
```

#### IOM Displacement (7 features)
```
Source: data/iom/*.csv
Temporal Resolution: Monthly surveys
Observation Rate: ~80%

Features:
├── displaced_total: IDP count
├── returnee_count: Return movements
├── mobility_index: Population dynamics
└── regional_flows: Inter-oblast movement
```

### 3.3 Narrative Features (Optional)

#### ISW Assessment Embeddings
```
Source: data/wayback_archives/isw_assessments/embeddings/
Embedding Model: Voyage AI (1024-dim → 128-dim PCA)
Temporal Resolution: Daily (publication schedule)
Observation Rate: ~95%

Integration:
├── Separate ISWAlignmentModule
├── Cross-attention to numerical representations
└── Semantic context for strategic interpretation

Key Finding (Probe 5.1.1):
- Mean cosine similarity with latents: 0.0087 (near-zero)
- ISW pathway currently contributes zero predictive signal
- Embedding-latent relationship is non-functional
```

---

## 4. Data Flow Architecture

### 4.1 Data Loading Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES (10+)                       │
│  Equipment, Personnel, DeepState, FIRMS, VIINA, VIIRS,         │
│  Sentinel, HDX (conflict, food, rainfall), IOM, ISW            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LOADER_REGISTRY                            │
│  load_equipment_daily(), load_firms_daily(), etc.               │
│  Each returns: (DataFrame, observation_mask)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               DOMAIN-SPECIFIC IMPUTATION                        │
│  UCDPImputation: Rolling median with decay                      │
│  FIRMSImputation: Linear interpolation + forward fill           │
│  DeepStateImputation: Forward fill (stability assumption)       │
│  EquipmentImputation: Forward fill with decay                   │
│  SentinelImputation: Cubic spline across months                 │
│                                                                 │
│  CRITICAL: Observation masks preserved (True=real, False=imp)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MultiResolutionDataset                             │
│  - Temporal alignment (daily ↔ monthly mapping)                 │
│  - Train/val/test split with temporal ordering                  │
│  - Normalization (train-only statistics)                        │
│  - Conversion to PyTorch tensors                                │
│  - ISW embedding loading (optional)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                MultiResolutionSample                            │
│  ├── daily_features: Dict[source_name, Tensor]                  │
│  ├── daily_masks: Dict[source_name, Tensor]                     │
│  ├── monthly_features: Dict[source_name, Tensor]                │
│  ├── monthly_masks: Dict[source_name, Tensor]                   │
│  ├── month_boundaries: List[(start_idx, end_idx)]               │
│  ├── isw_embeddings: Optional[Tensor]                           │
│  └── forecast_targets: Optional[Tensor]                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Temporal Alignment

```python
TemporalAlignment:
    daily_dates:       [2022-02-24, 2022-02-25, ..., 2026-01-27]  # ~1426 days
    monthly_dates:     [2022-02, 2022-03, ..., 2026-01]           # ~48 months
    month_boundaries:  [(0, 6), (7, 34), (35, 63), ...]           # day indices per month
    daily_to_monthly:  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...]        # day → month mapping
```

### 4.3 Observation Mask Protocol

```
Principle: NEVER fabricate data. Use observation masks throughout.

Mask Convention:
- True  = Real observation (use value)
- False = Missing/imputed (use no_observation_token)

Mask Flow:
1. Loader returns raw mask from data source
2. Imputer may modify values but preserves original mask
3. Dataset passes mask to model
4. DailySourceEncoder uses mask to select:
   - observed positions → feature projection
   - unobserved positions → no_observation_token
5. Attention mechanisms mask unobserved keys/values
6. Loss computation ignores unobserved targets
```

---

## 5. Attention Mechanism Dynamics

### 5.1 Attention Components

#### Scaled Dot-Product Attention
```python
# multi_resolution_modules.py:71
class ScaledDotProductAttention:
    """
    Core attention with observation mask handling.

    Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) * V

    Mask handling:
    - True positions: normal attention
    - False positions: set score to -inf before softmax
    - Edge case: if ALL positions masked, unmask first to prevent NaN
    """

    def forward(query, key, value, mask=None):
        scores = query @ key.T / sqrt(d_k)
        if mask is not None:
            # Handle all-masked edge case
            all_masked = ~mask.any(dim=-1, keepdim=True)
            if all_masked.any():
                mask[..., 0] = True  # Unmask first position
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = softmax(scores, dim=-1)
        return attention_weights @ value
```

#### Multi-Head Cross-Attention
```python
# multi_resolution_modules.py:155
class MultiHeadCrossAttention:
    """
    Cross-attention between two sequences (e.g., daily → monthly).

    Query sequence attends to key/value sequence.
    Used for:
    - Monthly aggregation (month queries attend to day keys/values)
    - Cross-resolution fusion (daily attends to monthly and vice versa)
    """

    def forward(query_seq, key_value_seq, key_value_mask):
        # Project to multi-head space
        Q = query_proj(query_seq)  # [batch, seq_q, d_model]
        K = key_proj(key_value_seq)
        V = value_proj(key_value_seq)

        # Reshape: [batch, seq, d] → [batch, heads, seq, head_dim]
        Q = Q.view(batch, seq_q, n_heads, head_dim).transpose(1, 2)
        K = K.view(batch, seq_kv, n_heads, head_dim).transpose(1, 2)
        V = V.view(batch, seq_kv, n_heads, head_dim).transpose(1, 2)

        # Attention with key_value_mask
        output = scaled_dot_product_attention(Q, K, V, mask=key_value_mask)
        return output_proj(output.reshape(batch, seq_q, d_model))
```

#### Bidirectional Cross-Attention
```python
# multi_resolution_modules.py:277
class BidirectionalCrossAttention:
    """
    Two-way information flow between sequences.

    A → B: Sequence A attends to B
    B → A: Sequence B attends to A

    Used for cross-resolution fusion (daily ↔ monthly).
    """

    def forward(seq_a, seq_b, mask_a, mask_b):
        # A attends to B
        a_to_b = cross_attn_a(query=seq_a, kv=seq_b, mask=mask_b)

        # B attends to A
        b_to_a = cross_attn_b(query=seq_b, kv=seq_a, mask=mask_a)

        # Gated fusion
        gate_a = sigmoid(gate_proj([seq_a, a_to_b]))
        gate_b = sigmoid(gate_proj([seq_b, b_to_a]))

        updated_a = gate_a * a_to_b + (1 - gate_a) * seq_a
        updated_b = gate_b * b_to_a + (1 - gate_b) * seq_b

        return updated_a, updated_b
```

### 5.2 Attention Patterns (Empirical Findings)

From Probe 6.2.2 (Attention Knockout):

```
Cross-Source Attention Flow (Expected: Structured, Actual: Uniform)

Flow Strength Matrix (excerpt):
              drones  armor  artillery  personnel  deepstate
drones          -     0.041    0.045      0.041      0.044
armor         0.041     -      0.038      0.036      0.038
artillery     0.045   0.038      -        0.039      0.041
personnel     0.041   0.036    0.039        -        0.040
deepstate     0.044   0.038    0.041      0.040        -

Uniform baseline: 1/20 sources = 0.05

Finding: All attention weights ~0.04 (near-uniform)
Critical pathways: 0 of 20 tested
Interpretation: Attention performs averaging, not selective routing
```

### 5.3 Causal Masking

```python
# DailySourceEncoder uses causal attention by default
class DailySourceEncoder:
    def __init__(self, ..., causal=True):
        if causal:
            # Upper triangular mask: position i cannot attend to j > i
            # Prevents future information leakage
            self.register_buffer(
                'causal_mask',
                torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            )
```

---

## 6. Multi-Resolution Processing

### 6.1 Daily-to-Monthly Aggregation

```python
# LearnableMonthlyAggregator
class LearnableMonthlyAggregator:
    """
    Learn which daily events matter for monthly representation.

    NOT simple averaging - uses attention to weight days.
    Month query vectors learn to focus on significant events.
    """

    def __init__(self, d_model, n_heads, n_months_max):
        # Learnable month query vectors
        self.month_queries = nn.Parameter(torch.randn(n_months_max, d_model))

    def forward(self, daily_repr, month_boundaries):
        """
        Args:
            daily_repr: [batch, n_days, d_model]
            month_boundaries: List[(start_idx, end_idx)] per month

        Returns:
            monthly_repr: [batch, n_months, d_model]
        """
        monthly_reprs = []

        for month_idx, (start, end) in enumerate(month_boundaries):
            # Get month's daily representations
            month_days = daily_repr[:, start:end, :]  # [batch, days_in_month, d_model]

            # Month query attends to its days
            query = self.month_queries[month_idx:month_idx+1]  # [1, d_model]

            # Cross-attention: 1 query, days_in_month keys/values
            month_repr = cross_attention(query, month_days, month_days)
            monthly_reprs.append(month_repr)

        return torch.cat(monthly_reprs, dim=1)  # [batch, n_months, d_model]
```

### 6.2 Cross-Resolution Fusion

```python
# CrossResolutionFusion
class CrossResolutionFusion:
    """
    Bidirectional fusion between aggregated daily and native monthly.

    Flow:
    1. aggregated_daily attends to monthly → enriched_daily
    2. monthly attends to aggregated_daily → enriched_monthly
    3. Gated combination of original + enriched
    """

    def forward(self, aggregated_daily, monthly, daily_mask, monthly_mask):
        # Bidirectional attention
        enriched_daily, enriched_monthly = self.bidirectional_attn(
            aggregated_daily, monthly, daily_mask, monthly_mask
        )

        # Residual connections with gating
        fused_daily = self.gate_daily(aggregated_daily, enriched_daily)
        fused_monthly = self.gate_monthly(monthly, enriched_monthly)

        return FusionOutput(
            fused_monthly=fused_monthly,
            fused_daily=fused_daily,
            monthly_mask=monthly_mask,
            daily_mask=daily_mask,
            attention_weights=attention_weights
        )
```

### 6.3 Resolution Summary

| Aspect | Daily Stream | Monthly Stream |
|--------|-------------|----------------|
| Timesteps | ~1,426 | ~48 |
| Sources | 6 (equipment, personnel, deepstate, firms, viina, viirs) | 5 (sentinel, hdx×3, iom) |
| Features | ~38 base, expandable to 231 with raion decomposition | ~35 |
| Primary Signal | Tactical activity, combat operations | Humanitarian, strategic trends |
| Aggregation | → Monthly via learnable attention | Native |
| Fusion | ← Attends to monthly context | Attends to daily details → |

---

## 7. Raion-Level Spatial Architecture

### 7.1 Design Overview

The raion (district) level architecture enables fine-grained tactical prediction at administrative level 2 (~629 raions total, 50-100 active).

```
RAION-LEVEL ARCHITECTURE (5 Levels):

LEVEL 1: Per-Raion Encoders
├── 50-100 active raions (filtered by observation count)
├── Each raion: RaionEncoder with 8 attention heads
├── Input per raion: [batch, seq_len, n_raion_features]
├── Shared encoder weights + per-raion embedding (efficiency)
└── Output: [batch, seq_len, d_model] per raion

LEVEL 2: Cross-Raion Attention
├── GeographicAdjacency: Learned from raion centroids
│   ├── Adjacent raions (sharing border): weight = 1.0
│   ├── Non-adjacent: weight = exp(-distance_km / 50)
├── CrossRaionAttention: "What in raion A predicts raion B?"
│   ├── Geographic prior as attention bias
│   ├── Learned attention on top of prior
│   └── Captures spillover effects (conflict spreading)
└── Output: [batch, n_raions, seq_len, d_model]

LEVEL 3: Macro-Temporal Context (National Level)
├── MacroEncoder processes non-regionalizable features:
│   ├── National equipment losses
│   ├── National personnel casualties
│   ├── Monthly humanitarian indicators
│   └── ISW narrative embeddings
└── Output: [batch, seq_len, d_model]

LEVEL 4: Temporo-Spatial Fusion
├── TemporoSpatialFusion: connects macro to regional
│   ├── Each raion attends to macro context
│   ├── Learns which national signals matter per region
│   ├── Frontline raions → weight military signals
│   ├── Rear raions → weight logistics signals
│   └── Gating mechanism for mixing ratio
└── Output: [batch, n_raions, seq_len, d_model]

LEVEL 5: Raion Prediction Heads
├── Shared weights + raion embedding (efficiency)
├── Per-raion forecasts: [batch, horizon, n_features_raion]
├── Predicts:
│   ├── Fire activity (count, intensity, FRP)
│   ├── Territorial changes (DeepState available)
│   └── Conflict intensity proxy
└── Total parameters: ~1.36M (shared architecture)
```

### 7.2 Geographic Adjacency Prior

```python
# cross_raion_attention.py:56
class GeographicAdjacency:
    """
    Computes soft adjacency matrix from raion centroids.

    Adjacent (sharing border): weight = 1.0
    Non-adjacent: weight = exp(-distance_km / scale)

    This matrix biases attention toward nearby raions,
    but model can learn to override geographic bias.
    """

    def compute_adjacency_matrix(self, raion_keys):
        n_raions = len(raion_keys)
        adjacency = torch.zeros(n_raions, n_raions)

        for i, key_i in enumerate(raion_keys):
            for j, key_j in enumerate(raion_keys):
                if i == j:
                    adjacency[i, j] = 1.0
                else:
                    dist = haversine_distance(centroid_i, centroid_j)
                    if dist < self.adjacency_threshold_km:
                        adjacency[i, j] = 1.0  # Neighbors
                    else:
                        adjacency[i, j] = exp(-dist / self.distance_scale_km)

        return adjacency

    # Usage in attention:
    # attn_logits = Q @ K.T + log(adjacency + epsilon)  # Geographic bias
```

### 7.3 Raion Data Sources

| Source | Features/Raion | Coverage |
|--------|---------------|----------|
| GeoconfirmedRaionLoader | 50 | ~300 raions |
| AirRaidSirensRaionLoader | 30 | Active alerts |
| UCDPRaionLoader | 35 | All raions |
| WarspottingRaionLoader | 33 | Sparse |
| DeepStateRaionLoader | 48 | Frontline |
| FIRMSExpandedRaionLoader | 35 | All raions |
| **CombinedRaionLoader** | **231** | Per-raion masking |

---

## 8. Prediction Heads & Multi-Task Learning

### 8.1 Prediction Tasks

#### Casualty Prediction (Regression)
```python
class CasualtyPredictionHead:
    """
    Predict personnel casualties with uncertainty estimation.

    Output: (prediction, uncertainty)
    - prediction: [batch, 3] for best/low/high estimates
    - uncertainty: [batch, 1] based on observation density
    """

    def forward(self, temporal_repr, observation_density):
        pred = self.regression_layers(temporal_repr)
        uncertainty = self.uncertainty_estimator(observation_density)
        return pred, uncertainty
```

#### Regime Classification (4-Class)
```python
class RegimeClassificationHead:
    """
    Classify current conflict regime.

    Classes:
    0: Initial Invasion (Feb-Apr 2022)
    1: Stalemate (Apr-Aug 2022)
    2: Counteroffensive (Sep 2022 - Oct 2023)
    3: Attritional Warfare (Nov 2023 - present)
    """

    def forward(self, temporal_repr):
        logits = self.classification_layers(temporal_repr)
        return F.softmax(logits, dim=-1)
```

#### Anomaly Detection (Binary)
```python
class AnomalyDetectionHead:
    """
    Detect unusual activity patterns.

    Trained on: VIIRS brightness anomalies, unusual event clusters
    Output: [batch, 1] anomaly probability
    """
```

#### Forecasting (Autoregressive)
```python
class ForecastingHead:
    """
    Predict next-period feature values.

    Output: [batch, n_features] for next timestep
    Used for: Self-supervised pretraining, trajectory prediction
    """
```

### 8.2 Multi-Task Loss

```python
class UncertaintyWeightedLoss:
    """
    Learn task-specific loss weights via uncertainty (Kendall et al. 2018).

    L_total = Σ (1/2σ²_i) * L_i + log(σ_i)

    Benefits:
    - Automatically balances task magnitudes
    - Tasks with higher uncertainty get lower weight
    - Learned during training
    """

    def __init__(self, n_tasks=4):
        # Learnable log-variance per task
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        """
        Args:
            losses: Dict[task_name, loss_value]
        Returns:
            weighted_total: Scalar total loss
        """
        total = 0
        for i, (task, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
```

---

## 9. Validation & Probe System

### 9.1 Probe Architecture

The project includes a comprehensive **50+ probe validation framework** organized into 9 sections:

```
analysis/probes/
├── run_probes.py              # Master runner with tier-based execution
├── output_manager.py          # Run organization & results tracking
├── task_key_mapping.py        # Task name resolution
│
├── data_artifact_probes.py    # Section 1: Equipment, VIIRS, personnel
├── cross_modal_fusion_probes.py   # Section 2: RSA, attention flow
├── temporal_dynamics_probes.py    # Section 3: Context windows
├── semantic_structure_probes.py   # Section 4: Operation clustering
├── semantic_association_probes.py # Section 5: ISW alignment
├── causal_importance_probes.py    # Section 6: Interventions
├── tactical_readiness_probes.py   # Section 7: Spatial decomposition
├── architecture_validation_probes.py # Section 9: Component validation
└── ...
```

### 9.2 Key Probes

| ID | Name | Question Answered |
|----|------|------------------|
| 1.1.2 | Equipment-Personnel Redundancy | Is equipment signal independent or redundant? |
| 1.2.1 | VIIRS-Casualty Temporal | Is VIIRS dominance real or temporal confound? |
| 2.1.1 | Representation Similarity | Do sources share geometric structure? |
| 2.2.1 | Leave-One-Out Ablation | Which sources are necessary? |
| 3.1.1 | Context Window Effects | What's minimum viable context? |
| 5.1.1 | ISW-Latent Correlation | How aligned are numerical/narrative? |
| 6.1.1 | Source Zeroing | Causal importance per source |
| 6.2.2 | Attention Knockout | Which attention pathways matter? |

### 9.3 Probe Execution

```bash
# Run all critical probes (Tier 1)
python -m analysis.probes.run_probes --tier 1

# Run specific section
python -m analysis.probes.run_probes --section 2

# Run single probe
python -m analysis.probes.run_probes --probe 1.2.1

# Data-only probes (no model needed)
python -m analysis.probes.run_probes --data-only
```

---

## 10. Key Findings & Known Issues

### 10.1 Validated Findings

| Finding | Evidence | Confidence |
|---------|----------|------------|
| **Context Window Paradox** | 7-day: 78.8% acc, 365-day: 51.3% | HIGH |
| **Equipment Disaggregation Value** | Drones MI=0.449 vs aggregated | HIGH |
| **VIIRS Lagging Indicator** | +10 day offset, detrend improves | MEDIUM-HIGH |
| **ISW Pathway Non-Functional** | Cosine sim 0.0087, R²=-88.6 | HIGH |
| **Attention Uniform** | All flows ~0.04, 0 critical paths | HIGH |

### 10.2 Central Thesis (from HAN-REPORT.md)

> The HAN model, despite its hierarchical temporal attention architecture, operates as a **static multi-input classifier** that ignores temporal dynamics, discards narrative content, and relies on uniform attention averaging across a small subset of quantitative sources.

### 10.3 Five Sub-Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Temporal structure not learned | 51/52 source-task pairs unaffected by shuffling |
| C2 | Attention non-functional | Uniform weights, zero critical pathways |
| C3 | ISW pathway dead | Zero correlation, bidirectional R² failure |
| C4 | VIIRS contaminates as lag | +10 day offset, dominates anomaly task |
| C5 | Static classification, not forecasting | Identical results across horizons |

### 10.4 Recommendations

1. **Shorter Context Windows**: Use 7-14 day windows instead of full history
2. **Disaggregated Equipment**: Use drones/armor/artillery instead of totals
3. **Detrend VIIRS**: Apply first-order differencing
4. **Fix Attention**: Investigate why attention is uniform (training objective?)
5. **ISW Integration**: Redesign narrative pathway or remove
6. **Temporal Supervision**: Add explicit temporal prediction objectives

---

## 11. Parameter Counts & Efficiency

### 11.1 Multi-Resolution HAN

| Component | Parameters |
|-----------|------------|
| Daily encoders (6×) | ~5M |
| Monthly encoders (5×) | ~2M |
| Cross-resolution fusion | ~1M |
| Temporal encoder | ~1M |
| Prediction heads (4×) | ~2M |
| **Total** | **~11M** |

### 11.2 RaionHAN (Spatial Extension)

| Component | Parameters |
|-----------|------------|
| Shared raion encoder | ~500K |
| Cross-raion attention | ~2M |
| Macro encoder | ~2M |
| Temporo-spatial fusion | ~1M |
| Shared prediction heads | ~100K |
| **Total** | **~1.36M** |

### 11.3 Memory Optimization

- Shared encoder weights across raions (with embedding)
- Gradient checkpointing for long sequences
- Sparse cross-attention (adjacent raions only)
- Mixed precision training (FP16)

---

## 12. File Reference

### Core Architecture
| File | Lines | Purpose |
|------|-------|---------|
| `multi_resolution_han.py` | 3,057 | Main model class |
| `multi_resolution_modules.py` | 4,053 | Encoder/fusion modules |
| `multi_resolution_data.py` | 3,057 | Dataset & loading |
| `raion_han.py` | ~400 | Raion-level model |
| `cross_raion_attention.py` | ~350 | Geographic attention |
| `geographic_source_encoder.py` | ~500 | Spatial source handling |

### Training
| File | Purpose |
|------|---------|
| `train_multi_resolution.py` | Multi-resolution training loop |
| `train_han.py` | HAN training script |
| `training_config.py` | Hyperparameter presets |
| `training_utils.py` | Schedulers, losses, utilities |

### Data Loaders
| File | Purpose |
|------|---------|
| `loaders/raion_spatial_loader.py` | Raion boundary management |
| `loaders/firms_spatial_loader.py` | FIRMS processing |
| `loaders/deepstate_spatial_loader.py` | Territorial data |
| `loaders/new_source_raion_loaders.py` | Comprehensive raion loaders |
| `loaders/raion_adapter.py` | LOADER_REGISTRY integration |

### Probes & Validation
| File | Purpose |
|------|---------|
| `probes/run_probes.py` | Master runner |
| `probes/causal_importance_probes.py` | Intervention analysis |
| `probes/temporal_dynamics_probes.py` | Temporal patterns |
| `probes/cross_modal_fusion_probes.py` | Fusion validation |

### Documentation
| File | Purpose |
|------|---------|
| `docs/HAN-REPORT.md` | Interpretability findings |
| `docs/raion-architecture-design.md` | Spatial design |
| `docs/Probe-specs.md` | Probe specifications |

---

## Appendix: Configuration Defaults

### MultiResolutionHANConfig
```python
@dataclass
class MultiResolutionHANConfig:
    d_model: int = 128
    nhead: int = 8
    num_daily_layers: int = 4
    num_monthly_layers: int = 3
    num_fusion_layers: int = 2
    num_temporal_layers: int = 2
    dropout: float = 0.1
    max_daily_len: int = 1500
    max_monthly_len: int = 60
    prediction_tasks: List[str] = ['casualty', 'regime', 'anomaly', 'forecast']
    causal: bool = True
    use_geographic_prior: bool = False
```

### RaionHANConfig
```python
@dataclass
class RaionHANConfig:
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 2
    dropout: float = 0.1
    max_raions: int = 50
    n_raion_features: int = 20
    n_macro_features: int = 30
    forecast_horizon: int = 7
    prior_weight: float = 1.0
    distance_scale_km: float = 50.0
```

---

**End of Technical Report**

*This document represents the current state of the Multi-Resolution HAN architecture as of 2026-01-28. Refer to the probe system and HAN-REPORT.md for ongoing validation findings.*
