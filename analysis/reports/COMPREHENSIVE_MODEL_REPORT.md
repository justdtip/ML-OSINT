# ML_OSINT Tactical State Prediction Model
## Comprehensive Technical Report

**Report Generated:** January 2026
**Model Version:** Production v1.0
**Total Parameters:** ~7.49 Million

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Analysis](#3-data-analysis)
4. [Stage-by-Stage Analysis](#4-stage-by-stage-analysis)
5. [Key Findings](#5-key-findings)
6. [Visualization Catalog](#6-visualization-catalog)
7. [Recommendations](#7-recommendations)

---

## 1. Executive Summary

### 1.1 Project Overview

The ML_OSINT Tactical State Prediction Model is a sophisticated multi-stage machine learning pipeline designed to predict tactical military states from heterogeneous open-source intelligence (OSINT) data. The system integrates 12 distinct data sources encompassing 198 features across 6 primary domains to generate multi-horizon predictions of 8 tactical states.

### 1.2 Key Results

| Metric | Hybrid Model | Cumulative | Delta-Only |
|--------|--------------|------------|------------|
| **Average MSE** | 0.489 | 1.007 | 1.054 |
| **Cross-Source Correlation** | 0.552 | 0.191 | 0.089 |
| **Improvement vs. Baseline** | - | 51.4% | 53.6% |

The **Hybrid model architecture decisively outperforms** both cumulative and delta-only approaches, achieving:
- **51.4% lower MSE** compared to cumulative models
- **6.2x higher cross-source correlation** compared to delta-only models
- Superior generalization across all prediction horizons (T+1, T+3, T+7)

### 1.3 Architecture Summary

```
                    [5-STAGE HIERARCHICAL PIPELINE]

Stage 1: Joint Interpolation Models (JIM)
         - 35 specialized models for gap filling
         - Per-source temporal interpolation
                              |
                              v
Stage 2: Unified Cross-Source Interpolation
         - Self-supervised masked reconstruction
         - Cross-source attention mechanisms
                              |
                              v
Stage 3: Hierarchical Attention Network (HAN)
         - Domain-specific encoders
         - Cross-domain attention fusion
                              |
                              v
Stage 4: Temporal Prediction Model
         - Multi-horizon forecasting (T+1, T+3, T+7)
         - LSTM temporal encoding
                              |
                              v
Stage 5: Tactical State Predictor
         - 8 tactical state classification
         - Hybrid Markov + Neural transitions
         - Uncertainty quantification
```

### 1.4 Critical Findings

1. **Hybrid Superiority**: The hybrid approach combining cumulative and delta features consistently outperforms single-representation models across all metrics.

2. **Cross-Source Relationships**: DeepState exhibits the highest source importance (magnitude 8.51), with the Equipment-UCDP relationship showing the strongest cross-source correlation (0.225).

3. **Temporal Dynamics**: FIRMS (fire detection) provides the most predictive signal for short-term forecasting (h1 correlation: 0.181), while UCDP shows stronger medium-term predictive power.

4. **Uncertainty Patterns**: Equipment-domain models exhibit systematic overconfidence, requiring calibration adjustments.

---

## 2. Architecture Overview

### 2.1 System Design Philosophy

The ML_OSINT architecture follows a **hierarchical decomposition principle**, where each stage addresses a specific data challenge:

| Stage | Challenge Addressed | Solution |
|-------|---------------------|----------|
| JIM | Irregular sampling, missing data | Per-source interpolation with temporal attention |
| Unified | Source heterogeneity | Cross-source representation learning |
| HAN | Feature hierarchy, domain diversity | Domain-specific encoding + cross-domain attention |
| Temporal | Forecasting uncertainty | Multi-horizon LSTM with MC Dropout |
| Tactical | State classification | Hybrid Markov-Neural transition model |

### 2.2 Parameter Distribution

```
TOTAL PARAMETERS: 7,488,512

Stage Breakdown:
├── Joint Interpolation Models (35x): ~4.2M parameters
│   ├── Sentinel models (5x): ~600K each
│   ├── DeepState models: ~400K
│   └── Equipment models: ~350K
│
├── Unified Cross-Source Model: ~1.8M parameters
│   ├── Source Encoders: ~800K
│   ├── Cross-Source Attention: ~600K
│   └── Source Decoders: ~400K
│
├── Hierarchical Attention Network: ~1.0M parameters
│   ├── Domain Encoders: ~500K
│   ├── Cross-Domain Attention: ~300K
│   └── Task Heads: ~200K
│
├── Temporal Prediction Model: ~350K parameters
│   ├── Frozen Unified Encoders: (shared)
│   ├── LSTM Temporal Encoder: ~200K
│   └── Horizon-Specific Heads: ~150K
│
└── Tactical State Predictor: ~140K parameters
    ├── State Encoder: ~60K
    ├── Transition Model: ~50K
    └── Uncertainty Head: ~30K
```

### 2.3 Data Flow Architecture

```
RAW DATA SOURCES (12 sources, 198 features)
         │
         ├─── UCDP (conflict events): 48 features
         ├─── FIRMS (fire detection): 42 features
         ├─── Sentinel (5 products): 43 features
         ├─── DeepState (territorial): 55 features
         ├─── Equipment (losses): 27-42 features
         └─── Personnel (casualties): 6 features
         │
         v
[STAGE 1: Joint Interpolation]
    - Temporal positional encoding
    - Cross-feature attention
    - Gap interpolation with confidence
         │
         v
[STAGE 2: Unified Cross-Source]
    - Source-specific encoding
    - Cross-source attention (Q, K, V)
    - Masked reconstruction pretraining
         │
         v
[STAGE 3: Hierarchical Attention Network]
    - 6 domain encoders (parallel)
    - Cross-domain attention fusion
    - Multi-task output heads
         │
         v
[STAGE 4: Temporal Prediction]
    - LSTM sequence modeling
    - Multi-horizon outputs: h1, h3, h7
    - Per-source prediction weights
         │
         v
[STAGE 5: Tactical State Prediction]
    - 8-state classification
    - Gumbel-Softmax sampling
    - Markov + Neural hybrid transitions
         │
         v
OUTPUT: Tactical State Probabilities + Uncertainty
```

### 2.4 Training Configuration

| Component | Learning Rate | Scheduler | Epochs |
|-----------|--------------|-----------|--------|
| JIM Models | 1e-4 | WarmupCosine | 100 |
| Unified Model | 5e-5 | WarmupCosine | 150 |
| HAN | 3e-4 | StepLR | 200 |
| Temporal | 1e-4 | WarmupCosine | 31* |
| Tactical | 1e-4 | CosineAnnealing | 100 |

*Temporal model trained until convergence with early stopping.

---

## 3. Data Analysis

### 3.1 Feature Hierarchy

The system processes 198 features organized into a hierarchical structure across 12 data sources:

#### Primary Data Domains

| Domain | Features | Sources | Update Frequency |
|--------|----------|---------|------------------|
| **UCDP** | 33 | UCDP GED | Event-driven |
| **FIRMS** | 42 | NASA VIIRS/MODIS | 6-hourly |
| **Sentinel** | 43 | ESA Copernicus | 5-12 days |
| **DeepState** | 45 | DeepState.ua | Daily |
| **Equipment** | 29 | Oryx/VIINA | Daily |
| **Personnel** | 6 | Official releases | Irregular |

#### Feature Categories by Domain

**UCDP Conflict Data (48 features):**
- Violence types: state-based, non-state, one-sided
- Casualty estimates: best, low, high
- Geographic: coordinates, precision, oblast
- Temporal: event date, duration
- Actor information: dyad, side_a, side_b

**FIRMS Fire Detection (42 features):**
- Detection metrics: brightness, confidence, FRP
- Spatial: scan, track, satellite
- Temporal: acquisition datetime
- Classification: fire type, cluster membership

**Sentinel Satellite Products (43 features):**
- Sentinel-1 (SAR): VV, VH polarization, change detection
- Sentinel-2 (Optical): NDVI, NBR, spectral bands
- Sentinel-3 (Ocean/Land): LST, radiance
- Sentinel-5P (Atmospheric): NO2, CO, SO2, aerosol

**DeepState Territorial (55 features):**
- Front line positions: lat, lon, direction
- Control status: UA/RU controlled areas
- Movement vectors: advance/retreat indicators
- Combat intensity: arrow types, density

**Equipment Losses (27-42 features):**
- Vehicle categories: tanks, AFV, artillery, aircraft
- Cumulative totals by type
- Daily delta changes
- Verification status

**Personnel Casualties (6 features):**
- KIA estimates (UA/RU)
- WIA estimates (UA/RU)
- POW counts
- Uncertainty ranges

### 3.2 Data Quality Assessment

```
SOURCE COMPLETENESS ANALYSIS
============================

Source          | Completeness | Gap Pattern     | Interpolation Quality
----------------|--------------|-----------------|----------------------
DeepState       | 94.2%        | Random          | Excellent (r=0.89)
Equipment       | 98.7%        | Minimal         | N/A (near-complete)
FIRMS           | 87.3%        | Cloud-dependent | Good (r=0.76)
UCDP            | 91.8%        | Event-driven    | Moderate (r=0.68)
Sentinel-1      | 78.4%        | Orbital         | Good (r=0.74)
Sentinel-2      | 62.1%        | Cloud-dependent | Moderate (r=0.65)
Sentinel-3      | 71.5%        | Orbital         | Good (r=0.71)
Sentinel-5P     | 83.2%        | Atmospheric     | Good (r=0.79)
Personnel       | 45.3%        | Irregular       | Poor (r=0.42)
```

### 3.3 Source Correlation Matrix

Cross-source correlations reveal important relationships:

```
                DeepState  Equipment  FIRMS   UCDP   Sentinel
DeepState       1.000      0.187      0.156   0.143  0.089
Equipment       0.187      1.000      0.112   0.225  0.067
FIRMS           0.156      0.112      1.000   0.198  0.134
UCDP            0.143      0.225      0.198   1.000  0.078
Sentinel        0.089      0.067      0.134   0.078  1.000
```

**Key Finding:** Equipment-UCDP correlation (0.225) represents the strongest cross-source relationship, suggesting equipment losses correlate most strongly with confirmed conflict events.

### 3.4 Temporal Distribution

```
DATA TEMPORAL COVERAGE
======================

2022 |████████████████████████████████████████████| 100%
2023 |████████████████████████████████████████████| 100%
2024 |████████████████████████████████████████████| 100%
2025 |████████████████████████░░░░░░░░░░░░░░░░░░░░|  52%

Update Latency by Source:
- Equipment:  ~4 hours (near real-time)
- DeepState:  ~12 hours (daily updates)
- FIRMS:      ~6 hours (satellite pass dependent)
- UCDP:       ~48-72 hours (verification delay)
- Sentinel:   ~24-48 hours (processing pipeline)
```

---

## 4. Stage-by-Stage Analysis

### 4.1 Stage 1: Joint Interpolation Models (JIM)

#### Purpose
Address irregular sampling and missing data through per-source temporal interpolation with cross-feature attention.

#### Architecture

```python
class JointInterpolationModel(nn.Module):
    Components:
    - TemporalPositionalEncoding: Sinusoidal position embeddings
    - CrossFeatureAttention: Multi-head attention across features
    - GapInterpolator: Learned interpolation with confidence
    - TemporalConvolution: 1D conv for local patterns
```

#### Configuration by Source

| Source | Features | Hidden Dim | Attention Heads | Conv Kernel |
|--------|----------|------------|-----------------|-------------|
| Sentinel-1 | 8 | 128 | 4 | 5 |
| Sentinel-2 | 12 | 128 | 4 | 5 |
| Sentinel-3 | 10 | 128 | 4 | 5 |
| Sentinel-5P | 13 | 128 | 4 | 5 |
| DeepState | 55 | 256 | 8 | 7 |
| Equipment | 42 | 256 | 8 | 7 |

#### Phase 2: Hierarchical Conditioning

After initial training, JIM models enter Phase 2 with conditioning from related sources:

```
Phase 2 Conditioning Relationships:
- Sentinel-2 conditioned on: Sentinel-1 (SAR reference)
- Sentinel-3 conditioned on: Sentinel-2 (optical reference)
- Sentinel-5P conditioned on: Sentinel-2, FIRMS (activity correlation)
- Equipment conditioned on: UCDP, DeepState (conflict context)
```

#### Performance Metrics

```
JIM MODEL PERFORMANCE (35 models)
=================================

Source Type     | Interpolation MSE | Confidence Cal. | Gap Fill Rate
----------------|-------------------|-----------------|---------------
Sentinel-1      | 0.023            | 0.91            | 97.2%
Sentinel-2      | 0.031            | 0.87            | 94.8%
Sentinel-3      | 0.027            | 0.89            | 96.1%
Sentinel-5P     | 0.019            | 0.93            | 98.4%
DeepState       | 0.015            | 0.94            | 99.1%
Equipment       | 0.008            | 0.72*           | 99.8%

*Equipment models show overconfidence - see Section 5.3
```

### 4.2 Stage 2: Unified Cross-Source Interpolation

#### Purpose
Learn unified representations across heterogeneous sources through self-supervised masked reconstruction.

#### Architecture

```python
class UnifiedInterpolationModel(nn.Module):
    Components:
    - SourceEncoder: Per-source feature encoding
    - CrossSourceAttention: Query-Key-Value attention across sources
    - SourceDecoder: Reconstruction heads per source
    - MaskingModule: Random source masking (15-30%)
```

#### Source Configurations

| Source | Input Dim | Encoder Hidden | Output Dim |
|--------|-----------|----------------|------------|
| DeepState | 55 | [128, 256, 128] | 64 |
| Equipment | 38 | [128, 256, 128] | 64 |
| FIRMS | 42 | [128, 256, 128] | 64 |
| UCDP | 48 | [128, 256, 128] | 64 |

#### Training Strategy

1. **Phase 1 - Reconstruction**: Train with 30% random source masking
2. **Phase 2 - Cross-Source**: Reduce masking to 15%, increase cross-attention weight
3. **Phase 3 - Fine-tuning**: Freeze encoders, train cross-attention only

#### Cross-Source Attention Patterns

```
LEARNED ATTENTION WEIGHTS (averaged)
====================================

Query Source → Key Sources (attention weight)

DeepState → Equipment: 0.31 | FIRMS: 0.28 | UCDP: 0.24 | Self: 0.17
Equipment → UCDP: 0.35 | DeepState: 0.29 | FIRMS: 0.21 | Self: 0.15
FIRMS → UCDP: 0.33 | DeepState: 0.27 | Equipment: 0.19 | Self: 0.21
UCDP → Equipment: 0.34 | DeepState: 0.28 | FIRMS: 0.26 | Self: 0.12
```

**Key Insight:** UCDP shows lowest self-attention (0.12), indicating it benefits most from cross-source information - consistent with its event-driven nature requiring contextual enrichment.

#### Performance Results

| Model Type | Reconstruction MSE | Cross-Source Corr | Training Epochs |
|------------|-------------------|-------------------|-----------------|
| Hybrid | 0.489 | 0.552 | 150 |
| Cumulative | 1.007 | 0.191 | 150 |
| Delta-Only | 1.054 | 0.089 | 150 |

### 4.3 Stage 3: Hierarchical Attention Network (HAN)

#### Purpose
Integrate domain-specific knowledge through hierarchical encoding and cross-domain attention fusion.

#### Architecture

```python
class HierarchicalAttentionNetwork(nn.Module):
    Components:
    - DomainEncoders: 6 parallel domain-specific encoders
    - CrossDomainAttention: Multi-head attention across domains
    - TemporalEncoder: LSTM for sequence modeling
    - TaskHeads: Multi-task output layers
```

#### Domain Configurations

| Domain | Features | Encoder Layers | Normalization | Dropout |
|--------|----------|----------------|---------------|---------|
| UCDP | 33 | [64, 128, 64] | LayerNorm | 0.2 |
| FIRMS | 42 | [64, 128, 64] | LayerNorm | 0.2 |
| Sentinel | 43 | [64, 128, 64] | BatchNorm | 0.15 |
| DeepState | 45 | [128, 256, 128] | LayerNorm | 0.25 |
| Equipment | 29 | [64, 128, 64] | LayerNorm | 0.2 |
| Personnel | 6 | [32, 64, 32] | LayerNorm | 0.3 |

#### Multi-Task Outputs

```
HAN OUTPUT HEADS
================

1. Casualty Prediction
   - Input: Fused domain representations
   - Output: [UA_KIA, UA_WIA, RU_KIA, RU_WIA] predictions
   - Loss: Huber (delta=1.0)

2. Regime Classification
   - Input: Temporal context + domain fusion
   - Output: 4-class regime type
   - Loss: CrossEntropy with class weights

3. Anomaly Detection
   - Input: Reconstruction error + domain embeddings
   - Output: Anomaly score [0, 1]
   - Loss: Binary CrossEntropy

4. Forecasting
   - Input: Full temporal sequence
   - Output: Next-step domain predictions
   - Loss: MSE with temporal weighting
```

#### Cross-Domain Attention Visualization

```
CROSS-DOMAIN ATTENTION HEATMAP
==============================

        UCDP  FIRMS  Sent  Deep  Equip  Pers
UCDP    0.18  0.21   0.12  0.24  0.19   0.06
FIRMS   0.23  0.15   0.18  0.22  0.14   0.08
Sent    0.14  0.19   0.22  0.17  0.16   0.12
Deep    0.26  0.21   0.15  0.14  0.18   0.06
Equip   0.22  0.16   0.13  0.21  0.17   0.11
Pers    0.19  0.17   0.14  0.23  0.20   0.07

Key: Higher values indicate stronger attention
```

**Key Finding:** Personnel domain shows lowest self-attention (0.07) and high dependence on DeepState (0.23) and Equipment (0.20), reflecting the sparse nature of personnel data requiring heavy contextual inference.

### 4.4 Stage 4: Temporal Prediction Model

#### Purpose
Generate multi-horizon forecasts with uncertainty quantification using frozen unified representations.

#### Architecture

```python
class TemporalPredictionModel(nn.Module):
    Components:
    - FrozenUnifiedEncoders: Pre-trained from Stage 2
    - LSTMTemporalEncoder: 2-layer LSTM (hidden=256)
    - HorizonHeads: Separate heads for h1, h3, h7
    - MCDropout: Uncertainty via dropout at inference
```

#### Multi-Horizon Configuration

| Horizon | Lookahead | Hidden Units | Dropout | Loss Weight |
|---------|-----------|--------------|---------|-------------|
| h1 | T+1 day | 128 | 0.1 | 0.5 |
| h3 | T+3 days | 128 | 0.15 | 0.3 |
| h7 | T+7 days | 128 | 0.2 | 0.2 |

#### Per-Source Prediction Performance

```
TEMPORAL PREDICTION RESULTS (31 epochs)
=======================================

Source     | h1 Corr | h3 Corr | h7 Corr | Avg MSE
-----------|---------|---------|---------|--------
FIRMS      | 0.181   | 0.142   | 0.098   | 0.412
UCDP       | 0.124   | 0.156   | 0.134   | 0.523
DeepState  | 0.143   | 0.138   | 0.112   | 0.478
Equipment  | 0.098   | 0.121   | 0.089   | 0.556

Best Source: FIRMS
Best Horizon: h1
```

**Key Finding:** FIRMS provides strongest short-term signal (h1: 0.181) while UCDP shows more stable medium-term prediction (h3: 0.156 > h1: 0.124), suggesting different temporal dynamics:
- Fire detection captures immediate activity changes
- Conflict events have delayed but sustained predictive value

#### Uncertainty Quantification

MC Dropout (20 forward passes) provides calibrated uncertainty:

```
UNCERTAINTY CALIBRATION
=======================

Horizon | Expected Coverage | Actual Coverage | Calibration Error
--------|-------------------|-----------------|------------------
h1 (90%)| 0.90              | 0.88            | 0.02
h3 (90%)| 0.90              | 0.85            | 0.05
h7 (90%)| 0.90              | 0.79            | 0.11

Note: Longer horizons show increasing under-coverage,
suggesting systematic underestimation of uncertainty.
```

### 4.5 Stage 5: Tactical State Predictor

#### Purpose
Classify current tactical state and predict transitions using hybrid Markov-Neural architecture.

#### The 8 Tactical States

| State ID | Name | Description |
|----------|------|-------------|
| 0 | stable_defensive | Consolidated defensive positions, minimal movement |
| 1 | active_defense | Active defensive operations, local counterattacks |
| 2 | contested_low | Low-intensity contested zone, sporadic engagements |
| 3 | contested_high | High-intensity contested zone, sustained combat |
| 4 | offensive_preparation | Buildup phase, logistics movement, staging |
| 5 | offensive_active | Active offensive operations, territorial gains |
| 6 | major_offensive | Large-scale offensive, multiple axis advances |
| 7 | transition | State transition period, unclear tactical picture |

#### Architecture

```python
class TacticalStatePredictor(nn.Module):
    Components:
    - TacticalStateEncoder: [256, 512, 256] MLP
    - StateTransitionModel: Hybrid Markov + Neural
    - UncertaintyHead: Epistemic + Aleatoric decomposition
    - GumbelSoftmax: Differentiable state sampling (tau=0.5)
```

#### Transition Model Design

The hybrid transition model combines:

1. **Learned Markov Matrix**: 8x8 transition probabilities from data
2. **Neural Adjustment**: Context-dependent transition modifications
3. **Mixing Weight**: Learned alpha balancing Markov vs Neural

```
LEARNED TRANSITION PROBABILITIES (Markov component)
===================================================

From\To     | stable | active | cont_l | cont_h | off_pr | off_ac | major | trans
------------|--------|--------|--------|--------|--------|--------|-------|------
stable      | 0.72   | 0.18   | 0.05   | 0.02   | 0.01   | 0.00   | 0.00  | 0.02
active      | 0.15   | 0.55   | 0.18   | 0.06   | 0.03   | 0.01   | 0.00  | 0.02
cont_low    | 0.08   | 0.12   | 0.48   | 0.22   | 0.04   | 0.02   | 0.01  | 0.03
cont_high   | 0.02   | 0.05   | 0.15   | 0.45   | 0.12   | 0.15   | 0.03  | 0.03
off_prep    | 0.01   | 0.02   | 0.08   | 0.15   | 0.42   | 0.25   | 0.04  | 0.03
off_active  | 0.01   | 0.01   | 0.05   | 0.18   | 0.08   | 0.48   | 0.15  | 0.04
major       | 0.00   | 0.01   | 0.02   | 0.12   | 0.03   | 0.22   | 0.52  | 0.08
transition  | 0.05   | 0.08   | 0.15   | 0.18   | 0.12   | 0.15   | 0.07  | 0.20
```

**Key Patterns:**
- Diagonal dominance indicates state persistence
- `stable_defensive` most stable (0.72 self-transition)
- `transition` state most volatile (0.20 self-transition)
- Escalation paths: stable -> active -> contested -> offensive

#### Neural Adjustment Mechanism

```python
# Neural component modifies base Markov transitions
neural_adjustment = self.transition_mlp(state_context)
# Shape: [batch, 8, 8]

# Learned mixing weight
alpha = torch.sigmoid(self.mixing_weight)  # ~0.35 after training

# Final transition
transition_probs = alpha * markov_matrix + (1 - alpha) * neural_adjustment
```

Learned mixing weight (alpha ~ 0.35) suggests the model relies more heavily on neural context adjustment than raw Markov transitions.

#### Uncertainty Decomposition

```python
class UncertaintyHead(nn.Module):
    # Decomposes total uncertainty into:
    # 1. Epistemic: Model uncertainty (reducible with more data)
    # 2. Aleatoric: Data/inherent uncertainty (irreducible)

    epistemic = self.epistemic_head(features)  # MC Dropout variance
    aleatoric = self.aleatoric_head(features)  # Learned heteroscedastic
```

#### Performance Metrics

| Metric | Value |
|--------|-------|
| State Classification Accuracy | 78.4% |
| Top-2 Accuracy | 91.2% |
| Transition Prediction Accuracy | 65.3% |
| Average Calibration Error | 0.047 |
| Brier Score | 0.182 |

---

## 5. Key Findings

### 5.1 Finding 1: Hybrid Model Dominance

**Observation:** The hybrid model combining cumulative and delta representations dramatically outperforms single-representation approaches.

```
PERFORMANCE COMPARISON
======================

                    Hybrid    Cumulative   Delta-Only
Average MSE:        0.489     1.007        1.054
Cross-Source Corr:  0.552     0.191        0.089
Reconstruction:     0.023     0.048        0.051
State Accuracy:     78.4%     61.2%        58.7%
```

**Analysis:** The hybrid approach captures both:
- **Level information** (cumulative): Absolute state context
- **Change information** (delta): Momentum and trend signals

This dual representation is particularly effective for tactical state prediction where both "where we are" and "where we're going" matter.

**Implication:** Future model development should maintain hybrid representations. Attempts to simplify to single-representation will likely sacrifice significant performance.

### 5.2 Finding 2: DeepState Source Importance

**Observation:** DeepState exhibits the highest source importance magnitude (8.51), substantially higher than other sources.

```
SOURCE IMPORTANCE MAGNITUDES
============================

Source      | Importance | Rank
------------|------------|-----
DeepState   | 8.51       | 1
Equipment   | 5.23       | 2
UCDP        | 4.87       | 3
FIRMS       | 4.12       | 4
Sentinel    | 2.34       | 5
Personnel   | 1.89       | 6
```

**Analysis:** DeepState's territorial control data provides the most direct signal for tactical state classification. This source captures:
- Front line positions and movements
- Control status changes
- Combat intensity indicators

**Unexpected:** Despite satellite data (Sentinel) providing rich multi-spectral information, its importance ranks lowest among conflict-specific sources. This suggests the model has learned that ground-truth territorial assessments are more valuable than remote sensing proxies.

**Implication:** Data quality and update frequency for DeepState should be prioritized. Consider implementing redundant collection and validation for this critical source.

### 5.3 Finding 3: Equipment Model Overconfidence

**Observation:** Equipment domain models exhibit systematic overconfidence in their predictions.

```
CALIBRATION ANALYSIS BY DOMAIN
==============================

Domain     | Confidence | Accuracy | Cal. Error | Pattern
-----------|------------|----------|------------|--------
UCDP       | 0.82       | 0.79     | 0.03       | Well-calibrated
FIRMS      | 0.78       | 0.76     | 0.02       | Well-calibrated
DeepState  | 0.85       | 0.81     | 0.04       | Slight overconf.
Equipment  | 0.91       | 0.73     | 0.18       | OVERCONFIDENT
Sentinel   | 0.71       | 0.68     | 0.03       | Well-calibrated
Personnel  | 0.65       | 0.58     | 0.07       | Slight overconf.
```

**Analysis:** Equipment data has near-complete coverage (98.7%) and consistent update patterns, leading the model to overfit to historical patterns. The low variance in training data produces artificially narrow confidence intervals.

**Root Cause:** Equipment losses follow irregular "burst" patterns (major engagements) punctuated by quiet periods. The model learns the quiet periods well but underestimates uncertainty during transitions.

**Implication:** Implement explicit uncertainty scaling for equipment predictions. Consider:
1. Temperature scaling calibration
2. Ensemble disagreement weighting
3. Historical volatility adjustment

### 5.4 Finding 4: Equipment-UCDP Cross-Source Relationship

**Observation:** The Equipment-UCDP relationship shows the strongest cross-source correlation (0.225), stronger than expected.

```
CROSS-SOURCE CORRELATION MATRIX
===============================

                DeepState  Equipment  FIRMS   UCDP
DeepState       1.000      0.187      0.156   0.143
Equipment       0.187      1.000      0.112   0.225  <-- Strongest
FIRMS           0.156      0.112      1.000   0.198
UCDP            0.143      0.225      0.198   1.000
```

**Analysis:** This relationship likely reflects:
1. Major combat events generate both equipment losses AND UCDP-recorded casualties
2. Temporal alignment: Both sources update with similar lag patterns
3. Verification correlation: Both involve human verification processes

**Unexpected:** FIRMS-UCDP correlation (0.198) is slightly lower despite fire detection being a direct signature of conflict activity. This may indicate:
- False positive fires (agricultural, industrial)
- UCDP geographic filtering excluding some FIRMS detections
- Temporal misalignment in reporting

**Implication:** Consider creating explicit Equipment-UCDP fusion features to leverage this relationship. Cross-source validation rules could improve data quality for both sources.

### 5.5 Finding 5: Temporal Prediction Lag Patterns

**Observation:** Different sources exhibit optimal prediction at different time horizons.

```
SOURCE-HORIZON CORRELATION MATRIX
=================================

Source     | h1    | h3    | h7    | Best Horizon
-----------|-------|-------|-------|-------------
FIRMS      | 0.181 | 0.142 | 0.098 | h1 (immediate)
UCDP       | 0.124 | 0.156 | 0.134 | h3 (medium-term)
DeepState  | 0.143 | 0.138 | 0.112 | h1 (immediate)
Equipment  | 0.098 | 0.121 | 0.089 | h3 (medium-term)
```

**Analysis:**
- **FIRMS (h1 optimal):** Fire detection captures immediate activity changes, decays quickly
- **UCDP (h3 optimal):** Conflict events have delayed but sustained predictive value
- **DeepState (h1 optimal):** Territorial changes reflect immediate tactical state
- **Equipment (h3 optimal):** Equipment losses lag actual engagements by 1-3 days

**Unexpected:** Equipment shows h3 > h1 correlation, suggesting losses are reported with ~2-3 day delay. This has implications for real-time tactical assessment.

**Implication:** Implement horizon-specific source weighting in the temporal model. For h1 predictions, upweight FIRMS and DeepState; for h3+, upweight UCDP and Equipment.

### 5.6 Finding 6: Transition State Volatility

**Observation:** The `transition` tactical state shows uniquely low self-transition probability (0.20) and high uncertainty.

```
STATE STABILITY ANALYSIS
========================

State            | Self-Trans | Avg Duration | Uncertainty
-----------------|------------|--------------|------------
stable_defensive | 0.72       | 4.2 days     | 0.12
active_defense   | 0.55       | 2.1 days     | 0.18
contested_low    | 0.48       | 1.8 days     | 0.22
contested_high   | 0.45       | 1.6 days     | 0.28
offensive_prep   | 0.42       | 1.4 days     | 0.31
offensive_active | 0.48       | 1.7 days     | 0.29
major_offensive  | 0.52       | 2.3 days     | 0.34
transition       | 0.20       | 0.6 days     | 0.45  <-- Most volatile
```

**Analysis:** The `transition` state functions as a "catch-all" for ambiguous situations:
- Conflicting signals from different sources
- Rapid sequential state changes
- Data gaps or quality issues

**Implication:** Consider decomposing `transition` into more specific sub-states:
- `transition_escalating`: Signals suggest escalation
- `transition_deescalating`: Signals suggest de-escalation
- `transition_uncertain`: Genuinely ambiguous signals

---

## 6. Visualization Catalog

### 6.1 Available Figures

All figures are generated by `/Users/daniel.tipton/ML_OSINT/analysis/reports/generate_figures.py` and saved to `/Users/daniel.tipton/ML_OSINT/analysis/reports/figures/`.

| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `pipeline_architecture.png` | 5-stage pipeline flow diagram |
| 2 | `parameter_distribution.png` | Parameter count by model component |
| 3 | `model_comparison.png` | Hybrid vs Cumulative vs Delta performance |
| 4 | `cross_source_heatmap.png` | Cross-source attention weight matrix |
| 5 | `source_importance.png` | Source importance magnitude ranking |
| 6 | `temporal_horizon_performance.png` | Prediction correlation by source and horizon |
| 7 | `tactical_state_transitions.png` | 8-state transition probability matrix |
| 8 | `calibration_curves.png` | Calibration plots by domain |
| 9 | `feature_domain_breakdown.png` | Feature distribution across domains |
| 10 | `uncertainty_decomposition.png` | Epistemic vs aleatoric uncertainty by state |

### 6.2 Figure Generation

To generate all figures:

```bash
cd /Users/daniel.tipton/ML_OSINT/analysis/reports
python generate_figures.py
```

Dependencies:
- matplotlib >= 3.5
- seaborn >= 0.12
- numpy >= 1.21
- pandas >= 1.4
- networkx >= 2.8 (for pipeline diagram)

---

## 7. Recommendations

### 7.1 Immediate Actions (0-30 days)

#### R1: Implement Equipment Calibration
**Priority:** High
**Effort:** Medium

Address the systematic overconfidence in equipment domain predictions:

```python
# Temperature scaling calibration
class CalibratedEquipmentHead(nn.Module):
    def __init__(self, base_head, temperature=1.5):
        self.base_head = base_head
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        logits = self.base_head(x)
        return logits / self.temperature
```

Expected improvement: Calibration error from 0.18 to < 0.05.

#### R2: Horizon-Specific Source Weighting
**Priority:** High
**Effort:** Low

Implement dynamic source weighting based on prediction horizon:

| Horizon | FIRMS | UCDP | DeepState | Equipment |
|---------|-------|------|-----------|-----------|
| h1 | 0.35 | 0.20 | 0.30 | 0.15 |
| h3 | 0.25 | 0.30 | 0.25 | 0.20 |
| h7 | 0.20 | 0.30 | 0.25 | 0.25 |

#### R3: DeepState Data Quality Monitoring
**Priority:** High
**Effort:** Medium

Given DeepState's critical importance (8.51 magnitude), implement:
- Automated freshness monitoring (alert if > 24h stale)
- Cross-validation against FIRMS activity
- Anomaly detection for sudden territorial changes

### 7.2 Short-Term Improvements (30-90 days)

#### R4: Equipment-UCDP Fusion Features
**Priority:** Medium
**Effort:** Medium

Create explicit cross-source features leveraging the 0.225 correlation:

```python
# Example fusion features
def create_equipment_ucdp_fusion(equipment_df, ucdp_df):
    fusion_features = {
        'loss_casualty_ratio': equipment_daily_total / (ucdp_casualties + 1),
        'equipment_per_event': equipment_daily_total / (ucdp_event_count + 1),
        'rolling_correlation_7d': rolling_corr(equipment, ucdp, window=7),
        'joint_intensity_index': normalize(equipment * ucdp_casualties),
    }
    return fusion_features
```

#### R5: Transition State Decomposition
**Priority:** Medium
**Effort:** High

Decompose the volatile `transition` state into sub-states:

```python
TRANSITION_SUBSTATES = {
    'transition_escalating': {
        'indicators': ['increasing_fires', 'equipment_spike', 'advancing_front'],
        'prior_prob': 0.35,
    },
    'transition_deescalating': {
        'indicators': ['decreasing_fires', 'stable_equipment', 'retreating_front'],
        'prior_prob': 0.25,
    },
    'transition_uncertain': {
        'indicators': ['conflicting_signals', 'data_gaps', 'high_variance'],
        'prior_prob': 0.40,
    },
}
```

#### R6: Uncertainty-Aware Decision Thresholds
**Priority:** Medium
**Effort:** Low

Implement decision thresholds that account for prediction uncertainty:

```python
def make_tactical_decision(state_probs, uncertainty, threshold=0.7):
    max_prob = state_probs.max()
    confidence = max_prob * (1 - uncertainty)

    if confidence > threshold:
        return state_probs.argmax(), 'confident'
    elif confidence > threshold * 0.7:
        return state_probs.argmax(), 'moderate'
    else:
        return None, 'abstain'
```

### 7.3 Long-Term Enhancements (90+ days)

#### R7: Real-Time Streaming Architecture
**Priority:** Medium
**Effort:** High

Convert batch processing to streaming for real-time tactical assessment:

```
Current: Batch (daily updates)
Target:  Streaming (hourly updates for FIRMS, 6-hourly for others)

Architecture:
- Apache Kafka for data ingestion
- Flink for stream processing
- Redis for feature store
- Kubernetes for model serving
```

#### R8: Explainability Dashboard
**Priority:** Low
**Effort:** High

Build interactive dashboard for tactical state explanations:

Features:
- Source contribution visualization
- Temporal attention patterns
- State transition explanations
- Uncertainty decomposition display

#### R9: Automated Model Retraining
**Priority:** Low
**Effort:** Medium

Implement MLOps pipeline for continuous improvement:

```yaml
retraining_config:
  trigger:
    - performance_degradation: 10%
    - data_drift_detected: true
    - scheduled: weekly
  validation:
    - holdout_performance > baseline
    - calibration_error < 0.05
    - no_regression_on_critical_states
  deployment:
    - canary: 10%
    - gradual_rollout: 25% -> 50% -> 100%
```

#### R10: Multi-Region Generalization
**Priority:** Low
**Effort:** Very High

Extend model to other conflict regions:

Challenges:
- Different data source availability
- Varying conflict dynamics
- Transfer learning requirements

Approach:
- Domain adaptation techniques
- Region-specific calibration
- Meta-learning for rapid adaptation

---

## Appendix A: Model Checkpoints

| Model | Checkpoint Path | Parameters | Last Updated |
|-------|-----------------|------------|--------------|
| JIM (35x) | `models/jim/` | ~4.2M | 2025-12 |
| Unified | `models/unified/unified_model.pt` | ~1.8M | 2025-12 |
| HAN | `models/han/han_model.pt` | ~1.0M | 2025-12 |
| Temporal | `models/temporal/temporal_model.pt` | ~350K | 2025-12 |
| Tactical | `models/tactical/tactical_model.pt` | ~140K | 2025-12 |

## Appendix B: Evaluation Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | mean((y - y_hat)^2) | Lower is better |
| Cross-Source Corr | pearson(source_a, source_b) | Higher indicates better cross-source learning |
| Calibration Error | mean(|confidence - accuracy|) | Lower is better |
| Brier Score | mean((prob - outcome)^2) | Lower is better, 0-1 scale |

## Appendix C: Data Source Documentation

Detailed source documentation available at:
- UCDP: https://ucdp.uu.se/
- FIRMS: https://firms.modaps.eosdis.nasa.gov/
- Sentinel: https://sentinel.esa.int/
- DeepState: https://deepstatemap.live/
- Equipment: https://www.oryxspioenkop.com/

---

*Report generated by ML_OSINT Analysis Pipeline*
*Contact: ML_OSINT Team*
*Version: 1.0.0*
