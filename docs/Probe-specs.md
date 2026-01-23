# Multi-Resolution HAN Probe Specification
## Validation Test Battery for Tactical State Prediction

**Version:** 1.0  
**Date:** 2026-01-23  
**Purpose:** Systematic validation of learned representations, cross-modal fusion quality, and semantic-numerical associations in the Multi-Resolution Hierarchical Attention Network for Ukraine conflict dynamics.

---

## Executive Summary

The Multi-Resolution HAN demonstrates strong performance on regime classification and anomaly detection, with improved casualty prediction following ZINB integration. However, several findings require deeper investigation:

1. **VIIRS nightlight data dominates casualty prediction** while equipment losses contribute minimally—counterintuitive given the domain
2. **Cross-source correlations are surprisingly low** (most < 0.1), suggesting sources may flow through independently rather than fusing
3. **The relationship between semantic context and numerical signals is unexplored**, limiting our understanding of what the model has actually learned

This document specifies a comprehensive test battery to:
- Distinguish genuine learned relationships from data artifacts
- Quantify cross-modal fusion quality
- Probe semantic structure in numerical representations
- Assess whether tactical state prediction would benefit from richer semantic integration

---

## Table of Contents

1. [Data Artifact Investigation](#1-data-artifact-investigation)
2. [Cross-Modal Fusion Validation](#2-cross-modal-fusion-validation)
3. [Temporal Dynamics Analysis](#3-temporal-dynamics-analysis)
4. [Semantic Structure Probing](#4-semantic-structure-probing)
5. [Semantic-Numerical Association Tests](#5-semantic-numerical-association-tests)
6. [Causal Importance Validation](#6-causal-importance-validation)
7. [Tactical Prediction Readiness](#7-tactical-prediction-readiness)
8. [Implementation Notes](#8-implementation-notes)

---

## 1. Data Artifact Investigation

### 1.1 Equipment Signal Degradation Analysis

**Question:** Is equipment's low predictive importance a genuine finding or an artifact of data preprocessing?

**Hypothesis:** Equipment losses may be redundant with personnel data (same MOD source), or cumulative-to-delta encoding may destroy meaningful variance.

**Tests:**

#### 1.1.1 Encoding Variance Comparison
```
For each equipment feature:
  - Compute variance of raw cumulative values
  - Compute variance of delta (day-over-day change)
  - Compute variance of 7-day rolling delta
  - Compare signal-to-noise ratios across encodings
  
Output: Table of variance ratios by feature and encoding type
```

**Expected if artifact:** Delta encoding shows >50% variance reduction vs cumulative  
**Expected if genuine:** Variance comparable across encodings

#### 1.1.2 Equipment-Personnel Redundancy Test
```
For the casualty prediction target:
  - Compute correlation: equipment features → target
  - Compute partial correlation: equipment features → target | controlling for personnel
  - Compute mutual information: I(equipment; target) vs I(equipment; target | personnel)

Output: Redundancy scores, partial correlations
```

**Expected if redundant:** Partial correlation near zero; MI drops >70% when conditioning  
**Expected if independent signal:** Partial correlation >0.1; MI drops <30%

#### 1.1.3 Equipment Category Disaggregation
```
Equipment categories: [tanks, APCs, artillery, MLRS, anti-aircraft, aircraft, helicopters, drones, vehicles, fuel_tanks, special_equipment, ships]

For each category:
  - Compute standalone correlation with daily casualties
  - Compute gradient magnitude through casualty head
  - Rank categories by predictive contribution

Output: Per-category importance ranking
```

**Question answered:** Is "equipment" weak because aggregation destroys category-specific signals?

#### 1.1.4 Temporal Lag Analysis - Equipment
```
For equipment_total and each major category:
  - Compute cross-correlation with casualties at lags [-30, -20, -10, -7, -3, -1, 0, +1, +3, +7, +10, +20, +30] days
  - Identify lag with peak correlation
  - Test significance of peak vs zero-lag

Output: Optimal lag per category, correlation at optimal vs zero lag
```

**Expected if reporting lag:** Peak correlation at positive lag (equipment reported after event)  
**Expected if genuine:** Peak near zero or negative (equipment predicts casualties)

---

### 1.2 VIIRS Dominance Investigation

**Question:** Is VIIRS genuinely predictive of casualties, or exploiting a temporal confounder?

**Hypothesis:** VIIRS nightlight changes could be: (a) upstream signal of conflict intensity, (b) concurrent infrastructure damage, (c) spurious correlation with conflict trend.

**Tests:**

#### 1.2.1 VIIRS-Casualty Temporal Relationship
```
For each VIIRS feature:
  - Compute cross-correlation with casualties at lags [-30 to +30] days
  - Identify optimal lag
  - Classify as: leading (lag < -3), concurrent (-3 ≤ lag ≤ +3), lagging (lag > +3)

Output: Lag classification per feature, correlation strength at optimal lag
```

**Critical finding:** If VIIRS lags casualties, it cannot be genuinely predictive—the model is using it as a proxy for recent history, not future state.

#### 1.2.2 VIIRS Feature Decomposition
```
VIIRS features to analyze: [list all 9 features with semantic meaning]

For each feature:
  - Compute gradient magnitude w.r.t. casualty ZINB head
  - Compute standalone correlation with target
  - Describe physical meaning (radiance mean, variance, geographic distribution, etc.)

Output: Ranked feature importance with physical interpretation
```

**Question answered:** What specific aspect of nightlights is the model using?

#### 1.2.3 Trend Confounding Test
```
Detrend all time series:
  - Apply first-differencing to VIIRS features
  - Apply first-differencing to casualty target
  - Retrain casualty head on detrended data (or compute correlations)
  - Compare VIIRS importance before/after detrending

Output: VIIRS importance ratio (detrended / original)
```

**Expected if confounded:** Importance drops >50% after detrending  
**Expected if genuine:** Importance stable or increases

#### 1.2.4 Geographic VIIRS Decomposition
```
If VIIRS features include regional breakdown:
  - Separate front-line regions vs rear areas vs civilian centers
  - Compute casualty correlation per region type
  - Identify which geographic component drives signal

If national aggregate only:
  - Flag as limitation
  - Recommend sector-level VIIRS for future work

Output: Regional importance breakdown or limitation flag
```

---

### 1.3 Personnel Data Quality Check

**Question:** Is personnel the true driver, with VIIRS proxying for personnel-correlated variance?

#### 1.3.1 Personnel-VIIRS Mediation Analysis
```
Test mediation model:
  VIIRS → Personnel → Casualties
  vs
  VIIRS → Casualties (direct)

Compute:
  - Direct effect of VIIRS on casualties
  - Indirect effect mediated through personnel
  - Proportion mediated

Output: Mediation statistics, path coefficients
```

**Expected if VIIRS proxies personnel:** High indirect effect, low direct effect  
**Expected if independent:** Significant direct effect remains after controlling for personnel

---

## 2. Cross-Modal Fusion Validation

### 2.1 Fusion Quality Metrics

**Question:** Is the model learning genuine cross-modal relationships or treating sources independently?

#### 2.1.1 Representation Similarity Analysis (RSA)
```
For each source pair (equipment, personnel, deepstate, firms, viina, viirs):
  - Extract latent representations for all timesteps
  - Compute representational dissimilarity matrix (RDM) for each source
  - Compute correlation between RDMs (second-order similarity)
  
Output: RSA matrix showing which sources have similar representational geometry
```

**Expected if fusing:** Related sources (e.g., equipment-personnel, firms-viirs) show RSA > 0.3  
**Expected if independent:** RSA near zero for all pairs

#### 2.1.2 Cross-Source Information Flow
```
Using attention weights from cross-source fusion layer:
  - For each source, compute mean attention received from each other source
  - Identify dominant information flow patterns
  - Compare attention patterns across conflict phases

Output: Attention flow matrix, phase-specific patterns
```

**Question answered:** Which sources attend to which? Is attention uniform or structured?

#### 2.1.3 Fusion Layer Ablation
```
Compare model performance with:
  A) Full cross-source fusion (current)
  B) No cross-source fusion (sources concatenated directly)
  C) Pairwise fusion only (each source fuses with one other)

Metrics: Regime accuracy, transition F1, casualty NLL, anomaly MSE

Output: Performance delta attributable to fusion architecture
```

**Expected if fusion helps:** A > B on most metrics  
**Expected if fusion neutral:** A ≈ B (sources already independent)

#### 2.1.4 Checkpoint Comparison
```
If checkpoints available from different training stages:
  - Extract cross-source correlations at epoch 10, 25, 50, 75, 100
  - Track evolution of fusion quality over training
  - Identify if ZINB introduction degraded fusion

Output: Fusion quality trajectory, inflection points
```

---

### 2.2 Source Contribution Analysis

#### 2.2.1 Leave-One-Out Ablation
```
For each source S in [equipment, personnel, deepstate, firms, viina, viirs]:
  - Create modified input with S zeroed/masked
  - Run inference on full test set
  - Compute performance delta per task

Output: Source necessity matrix (source × task)
```

**Question answered:** Which sources are necessary vs redundant for each task?

#### 2.2.2 Source Sufficiency Test
```
For each source S:
  - Create modified input with ONLY S present (others masked)
  - Run inference on full test set
  - Compute performance (vs random baseline, vs full model)

Output: Source sufficiency scores—can any single source solve any task?
```

---

## 3. Temporal Dynamics Analysis

### 3.1 Context Window Effects

**Question:** How does the model's behavior change with different amounts of temporal context?

#### 3.1.1 Truncated Context Inference
```
Context lengths to test: [7, 14, 30, 60, 90, full]

For each context length:
  - Run inference with truncated history
  - Record: task performance, latent variance, cross-source correlations
  - Measure prediction confidence (ZINB variance parameter)

Output: Performance curves by context length, minimum viable context
```

**Expected:** Performance degrades gracefully; cross-modal correlations may require longer context to emerge

#### 3.1.2 Temporal Attention Patterns
```
From temporal encoder attention weights:
  - Compute mean attention distance (how far back does model look?)
  - Identify attention peaks (specific historical days attended to)
  - Compare patterns during different conflict phases

Output: Attention distance distribution, phase-specific patterns
```

#### 3.1.3 Predictive Horizon Analysis
```
Modify prediction targets to different horizons:
  - t+1 (next day) - current setup
  - t+3 (3 days ahead)
  - t+7 (1 week ahead)
  - t+14 (2 weeks ahead)

For each horizon:
  - Evaluate prediction performance
  - Identify which sources become more/less important
  - Measure uncertainty calibration

Output: Performance decay curve, horizon-dependent source importance
```

---

### 3.2 State Transition Dynamics

#### 3.2.1 Transition Boundary Analysis
```
Identify all regime transition dates in training data

For each transition:
  - Extract latent trajectory [-14 days, +14 days] around transition
  - Compute velocity in latent space (daily change magnitude)
  - Measure distance to source and destination regime centroids

Output: Transition signatures, early warning potential
```

**Question answered:** Can transitions be detected before they occur based on latent dynamics?

#### 3.2.2 Latent Velocity Prediction
```
Define latent velocity: v(t) = ||z(t+1) - z(t)||

Analyze:
  - Does high velocity predict transitions?
  - Which sources drive velocity spikes?
  - Correlation between velocity and next-day casualties

Output: Velocity-transition correlation, velocity-casualty correlation
```

---

## 4. Semantic Structure Probing

### 4.1 Implicit Semantic Categories

**Question:** Do numerical representations encode semantic categories without explicit supervision?

#### 4.1.1 Named Operation Clustering
```
Operations to analyze:
  - Kyiv offensive/retreat (Feb-Apr 2022)
  - Kharkiv counteroffensive (Sep 2022)
  - Kherson counteroffensive (Oct-Nov 2022)
  - Bakhmut offensive (Aug 2022 - May 2023)
  - 2023 Ukrainian counteroffensive (Jun-Oct 2023)
  - Avdiivka battle (Oct 2023 - Feb 2024)
  - Kursk incursion (Aug 2024 - present)

For each operation:
  - Extract latent states for operation date range
  - Compute operation centroid in latent space
  - Measure within-operation vs between-operation variance
  - Compute silhouette score for operation clustering

Output: Operation separability metrics, clustering visualization
```

**Expected if semantic structure:** Silhouette score > 0.3, distinct operation clusters  
**Expected if no structure:** Operations overlap significantly

#### 4.1.2 Day-Type Decoding Probe
```
Label days by type (from external sources):
  - Major missile/drone strike day
  - Ground assault day
  - Counterattack day
  - Relatively quiet day
  - Infrastructure attack day

Train linear probe: frozen_latent → day_type
Evaluate: accuracy, per-class F1, confusion matrix

Output: Probe accuracy, most confusable categories
```

**Question answered:** Can semantic day types be decoded from numerical representations?

#### 4.1.3 Intensity Level Decoding
```
Label days by intensity (from casualty distribution):
  - Low intensity (bottom quartile)
  - Medium intensity (middle 50%)
  - High intensity (top quartile)
  - Extreme (top 5%)

Train linear probe: frozen_latent → intensity_level
Compare: using fused latent vs individual source latents

Output: Intensity decoding accuracy by representation type
```

#### 4.1.4 Geographic Focus Decoding
```
If regional data available, label days by primary theater:
  - Eastern front dominant
  - Southern front dominant
  - Multi-front activity
  - Strategic depth attacks (rear area strikes)

Train probe to decode geographic focus from latent state

Output: Geographic decoding accuracy
```

---

### 4.2 Temporal Semantic Patterns

#### 4.2.1 Weekly Cycle Detection
```
Analyze latent states by day of week:
  - Compute mean latent vector per weekday
  - Test for significant weekday effects (ANOVA)
  - Identify features with strongest weekly patterns

Output: Weekly pattern significance, interpretable features
```

**Question answered:** Does the model capture operational tempo patterns (e.g., weekend lulls)?

#### 4.2.2 Seasonal Pattern Detection
```
Analyze latent states by month/season:
  - Compute mean latent vector per month
  - Test for seasonal effects
  - Correlate with known seasonal factors (weather, daylight, mud season)

Output: Seasonal pattern significance
```

#### 4.2.3 Event Anniversary Detection
```
Test if model encodes "time since major event":
  - Compute distance from current latent to major event latent (e.g., invasion start, Kherson liberation)
  - Correlate with calendar distance
  - Test if latent space has "temporal landmarks"

Output: Temporal landmark correlations
```

---

## 5. Semantic-Numerical Association Tests

### 5.1 ISW Alignment Validation

**Question:** How well do existing numerical representations align with semantic content from ISW reports?

#### 5.1.1 ISW-Latent Correlation
```
Using ISW embeddings (1024-dim Voyage) reduced to 128-dim:

For each day with ISW report:
  - Compute cosine similarity: ISW_embedding ↔ fused_latent
  - Compute correlation across time series
  - Identify days with highest/lowest alignment

Output: Alignment distribution, exemplar high/low alignment days
```

**Expected:** Alignment should be moderate (0.3-0.5) if numerical and semantic capture same dynamics

#### 5.1.2 ISW Topic-Source Correlation
```
Extract ISW topics (via clustering or LDA on embeddings):
  - Topic: Russian offensive operations
  - Topic: Ukrainian counterattacks
  - Topic: Infrastructure strikes
  - Topic: International developments
  - Topic: Logistics/supply
  - etc.

For each topic, compute correlation with each numerical source's latent

Output: Topic-source correlation matrix
```

**Question answered:** Which numerical sources correlate with which narrative themes?

#### 5.1.3 ISW Predictive Content Test
```
Test if ISW embeddings predict next-day numerical signals:
  - ISW(t) → Equipment_delta(t+1)
  - ISW(t) → FIRMS(t+1)
  - ISW(t) → Casualty(t+1)

Compare to baseline: Numerical(t) → Numerical(t+1)

Output: ISW predictive power vs autoregressive baseline
```

**Expected if ISW adds value:** ISW predictions competitive with or better than autoregressive

---

### 5.2 Cross-Modal Semantic Grounding

#### 5.2.1 Event-Triggered Response Analysis
```
Identify specific documented events with known dates:
  - Kerch Bridge attack (Oct 8, 2022)
  - Kherson withdrawal (Nov 11, 2022)
  - Prigozhin mutiny (Jun 23-24, 2023)
  - Dam collapse (Jun 6, 2023)
  - Avdiivka fall (Feb 17, 2024)

For each event:
  - Extract latent state trajectory [-7, +7] days
  - Measure anomaly score spike
  - Identify which sources respond most strongly
  - Compare numerical response to ISW narrative emphasis

Output: Event response signatures, source-specific sensitivities
```

#### 5.2.2 Narrative-Numerical Lag Analysis
```
Hypothesis: ISW narratives may lead or lag numerical signals

Compute cross-correlation at multiple lags:
  - ISW_embedding ↔ fused_latent at lags [-7 to +7] days
  - Per ISW topic ↔ per numerical source

Identify:
  - Which signals lead (predictive)
  - Which signals lag (reactive/confirmatory)

Output: Lead/lag classification per signal pair
```

#### 5.2.3 Semantic Anomaly Detection
```
Define semantic anomaly: ISW embedding significantly different from recent context

Compare:
  - Numerical anomaly days (from VIIRS-based detector)
  - Semantic anomaly days (ISW embedding outliers)

Compute overlap: Jaccard similarity of anomaly sets
Test: Do semantic and numerical anomalies co-occur?

Output: Anomaly set overlap, co-occurrence significance
```

---

### 5.3 Counterfactual Semantic Probing

#### 5.3.1 Semantic Perturbation Effects
```
For a sample of days with ISW reports:
  - Get baseline prediction using ISW(t) + Numerical(t)
  - Replace ISW(t) with ISW from different day/context
  - Measure prediction change

Perturbation types:
  - Random ISW from same regime
  - ISW from different regime
  - ISW from same weekday, different week
  - Negated sentiment ISW (if available)

Output: Prediction sensitivity to semantic context
```

#### 5.3.2 Missing Semantic Interpolation
```
For days without ISW reports:
  - Current approach: missing token or interpolation
  - Alternative: use numerical signals to "infer" semantic state

Train: Numerical_latent → ISW_embedding predictor
Evaluate: Can numerical signals recover semantic content?

Output: Numerical→semantic reconstruction quality
```

---

### 5.4 Semantic Enrichment Potential

#### 5.4.1 Telegram Narrative Integration (Specification)
```
Data source: Ukrainian/Russian Telegram channels (public military bloggers)

Features to extract:
  - Daily message volume per channel
  - Sentiment scores (ukr-roberta or similar)
  - Entity mentions (locations, unit names)
  - Claim types (advance, retreat, strike, defense)

Integration test:
  - Add Telegram features as additional source
  - Measure task performance delta
  - Analyze cross-modal correlations with existing sources

Output: Telegram value-add assessment
```

#### 5.4.2 Combat Footage Metadata (Specification)
```
Data source: Combat footage titles/metadata from aggregators

Features to extract:
  - Date (parsed from title)
  - Location (geocoded from title)
  - Attack type (drone, artillery, armor, infantry)
  - Attacker (UA/RU)
  - Target type
  - Time of day (from frame analysis or title)

Integration test:
  - Daily aggregated footage statistics as new source
  - Correlation with existing sources
  - Predictive contribution to casualty/equipment heads

Output: Combat footage metadata value assessment
```

#### 5.4.3 Official Statement Encoding (Specification)
```
Data sources:
  - Ukrainian General Staff daily briefings
  - Russian MOD statements
  - Presidential/ministerial statements

Features:
  - Statement embeddings (same Voyage model as ISW)
  - Claim extraction (numbers claimed, locations mentioned)
  - Sentiment/confidence scores

Test: Do official statements provide signal beyond ISW analysis?

Output: Official statement marginal value
```

---

## 6. Causal Importance Validation

### 6.1 Intervention-Based Importance

#### 6.1.1 Source Zeroing Interventions
```
For each source S and each test day:
  - Baseline: predict with all sources
  - Intervention: set S to zero vector
  - Measure: |prediction_baseline - prediction_intervention|

Aggregate across days:
  - Mean absolute change per source
  - Variance of change (consistency)
  - Task-specific importance

Output: Causal importance ranking per task
```

#### 6.1.2 Source Shuffling Interventions
```
For each source S:
  - Shuffle S values across days (destroy temporal structure)
  - Keep other sources intact
  - Measure performance degradation per task

Output: Temporal importance per source
```

**Question answered:** Which sources contribute via their values vs their temporal patterns?

#### 6.1.3 Source Mean Substitution
```
For each source S:
  - Replace S(t) with mean(S) for all t
  - Measure performance degradation

Compare to zeroing:
  - If mean substitution ≈ zeroing: model uses deviation from mean
  - If mean substitution << zeroing: model uses absolute values

Output: Value vs deviation importance per source
```

---

### 6.2 Gradient-Based Causal Analysis

#### 6.2.1 Integrated Gradients
```
For casualty prediction head:
  - Compute integrated gradients from zero baseline to actual input
  - Aggregate attribution per source
  - Compare to simple gradient magnitude (current approach)

Output: Integrated gradient importance ranking
```

#### 6.2.2 Attention Knockout
```
For cross-source attention layers:
  - Zero out attention from source A to source B
  - Measure task performance change
  - Map causal information flow

Output: Attention-based causal graph
```

---

## 7. Tactical Prediction Readiness

### 7.1 Spatial Decomposition Potential

#### 7.1.1 Regional Signal Availability
```
Audit data sources for regional granularity:

| Source | National | Oblast | Raion | Coordinate |
|--------|----------|--------|-------|------------|
| Equipment | ? | ? | ? | ? |
| Personnel | ? | ? | ? | ? |
| DeepState | ? | ? | ? | ✓ (polygons) |
| FIRMS | ✓ | ✓ | ✓ | ✓ |
| VIINA | ? | ? | ? | ? |
| VIIRS | ✓ | ✓ | ✓ | ✓ |

Output: Regional data availability matrix, limiting factors
```

#### 7.1.2 Front-Line Sector Definition
```
Define tactical sectors based on DeepState data:
  - Kharkiv sector
  - Luhansk sector (Svatove-Kreminna)
  - Donetsk north (Bakhmut-Siversk)
  - Donetsk central (Avdiivka-Marinka)
  - Donetsk south (Vuhledar)
  - Zaporizhzhia sector
  - Kherson sector
  - Kursk sector (new)

For each sector:
  - Define bounding polygon
  - Filter FIRMS, VIIRS, DeepState to sector
  - Compute sector-specific daily features

Output: Sector feature pipeline specification
```

#### 7.1.3 Sector Independence Test
```
Using sector-filtered data:
  - Compute correlation between sectors for each feature
  - Identify which signals are sector-specific vs national
  - Measure information leakage between sectors

Output: Sector independence assessment
```

---

### 7.2 Entity-Level Readiness

#### 7.2.1 Unit Tracking Data Availability
```
Audit sources for unit-level information:
  - Oryx (equipment by unit, sparse)
  - UA General Staff (some unit mentions)
  - Milbloggers (unit identifications)
  - DeepState (some unit annotations)

Assess: Is unit-level tracking feasible with available data?

Output: Unit tracking feasibility assessment
```

#### 7.2.2 Entity State Representation Design
```
Specify entity state vector for future implementation:

Unit entity:
  - Estimated strength (personnel)
  - Equipment inventory (by type)
  - Position (centroid + uncertainty)
  - Days in contact
  - Recent loss rate
  - Historical behavior features

Infrastructure entity:
  - Type (depot, HQ, air defense, bridge, etc.)
  - Status (operational, damaged, destroyed)
  - Last activity date
  - Strategic value score

Output: Entity schema specification
```

---

### 7.3 Prediction Resolution Requirements

#### 7.3.1 Temporal Resolution Analysis
```
Current: Daily predictions

Test requirements for:
  - 12-hour predictions (day/night cycle)
  - 6-hour predictions (operational tempo)
  - Hourly predictions (tactical)

Assess:
  - Data availability at each resolution
  - Expected performance degradation
  - Use case requirements

Output: Resolution-performance tradeoff analysis
```

#### 7.3.2 Spatial Resolution Analysis
```
Current: National

Test requirements for:
  - Oblast level (~25 units)
  - Sector level (~10 units)
  - Grid cell level (10km × 10km)
  - Point level (specific coordinates)

Assess: Data density per resolution, meaningful prediction granularity

Output: Spatial resolution recommendations
```

---

## 8. Implementation Notes

### 8.1 Priority Ranking

**Tier 1 - Critical (Run First):**
1. VIIRS-Casualty Temporal Relationship (1.2.1)
2. Equipment-Personnel Redundancy Test (1.1.2)
3. Source Zeroing Interventions (6.1.1)
4. Named Operation Clustering (4.1.1)
5. ISW-Latent Correlation (5.1.1)

**Tier 2 - Important:**
6. Trend Confounding Test (1.2.3)
7. Leave-One-Out Ablation (2.2.1)
8. Day-Type Decoding Probe (4.1.2)
9. Event-Triggered Response Analysis (5.2.1)
10. Truncated Context Inference (3.1.1)

**Tier 3 - Exploratory:**
11. All remaining tests

### 8.2 Data Requirements

```
Required inputs:
  - Trained model checkpoint
  - Full training/validation/test data
  - ISW embeddings (1024-dim, daily)
  - Event timeline (named operations, major events)
  - Day-type labels (if available, else construct from news)

Optional inputs:
  - Intermediate checkpoints (for training dynamics)
  - Telegram channel data
  - Combat footage metadata
```

### 8.3 Output Format

Each test should produce:

```yaml
test_id: "1.2.1"
test_name: "VIIRS-Casualty Temporal Relationship"
status: completed | failed | partial
findings:
  - key_result: "VIIRS feature X leads casualties by Y days"
  - significance: p-value or confidence interval
  - interpretation: "VIIRS is genuinely predictive | confounded | inconclusive"
artifacts:
  - figures: ["viirs_lag_correlation.png", ...]
  - tables: ["viirs_lag_table.csv", ...]
  - data: ["viirs_lag_raw.json", ...]
recommendations:
  - "Proceed with VIIRS as primary signal"
  - "Investigate alternative encoding"
  - etc.
```

### 8.4 Success Criteria

The test battery succeeds if it answers:

1. **Is VIIRS dominance real?** → Temporal lag + detrending tests
2. **Is equipment contribution salvageable?** → Disaggregation + encoding tests
3. **Is the model actually fusing?** → RSA + ablation tests
4. **Does the model encode semantics implicitly?** → Operation clustering + day-type probes
5. **Would explicit semantics help?** → ISW correlation + predictive tests
6. **Is sector-level prediction feasible?** → Data availability + independence tests

### 8.5 Agent Instructions

```
For each test:
1. Read specification carefully
2. Verify data availability
3. Implement test logic
4. Run on appropriate data split (test set unless specified)
5. Generate all specified outputs
6. Write interpretation in findings
7. Flag any anomalies or unexpected results
8. Suggest follow-up tests if warranted

Do not:
- Modify model weights
- Use training data for evaluation
- Skip tests without documenting why
- Interpret ambiguous results as conclusive
```

---

## Appendix A: Feature Index Mappings

*To be populated with actual feature names from model config*

```
Equipment features [0-11]: tank, apc, artillery, mlrs, anti_aircraft, ...
Personnel features [0-2]: killed, wounded, pow
DeepState features [0-4]: area_controlled, front_length, ...
FIRMS features [0-12]: fire_count, brightness_mean, ...
VIINA features [0-6]: events_by_region...
VIIRS features [0-8]: radiance_mean, radiance_std, ...
```

---

## Appendix B: Timeline Reference

*Key dates for event-based analyses*

| Date | Event |
|------|-------|
| 2022-02-24 | Invasion begins |
| 2022-04-02 | Kyiv withdrawal complete |
| 2022-09-06 | Kharkiv counteroffensive begins |
| 2022-10-08 | Kerch Bridge attack |
| 2022-11-11 | Kherson liberated |
| 2023-06-06 | Kakhovka dam collapse |
| 2023-06-08 | 2023 counteroffensive begins |
| 2023-06-23 | Prigozhin mutiny |
| 2024-02-17 | Avdiivka falls |
| 2024-08-06 | Kursk incursion begins |

---

## Appendix C: Semantic Data Sources

*For future integration*

| Source | Type | Availability | Integration Complexity |
|--------|------|--------------|----------------------|
| ISW Daily | Narrative | Available | Low (embeddings exist) |
| UA General Staff | Official claims | Public | Medium (parsing required) |
| Telegram milbloggers | Social/OSINT | API accessible | High (filtering, dedup) |
| Combat footage | Visual metadata | Aggregator sites | High (scraping, NLP) |
| Russian MOD | Adversary claims | Public | Medium (parsing, inversion) |
| FIRMS | Thermal | Available | Already integrated |
| Sentinel-2 | Optical | Available | Already integrated |

---

*End of specification*