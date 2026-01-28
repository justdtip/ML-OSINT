# Attention Architecture Deep Dive

This document provides an in-depth explanation of how attention mechanisms work in transformer architectures, and specifically how they're applied in the Multi-Resolution HAN for conflict prediction.

---

## Table of Contents

1. [Attention Fundamentals](#attention-fundamentals)
2. [Multi-Head Attention](#multi-head-attention)
3. [What Attention Operates Over](#what-attention-operates-over)
4. [The Multi-Resolution HAN Architecture](#the-multi-resolution-han-architecture)
5. [Hierarchical Attention Structure](#hierarchical-attention-structure)
6. [Prediction Heads vs Attention Heads](#prediction-heads-vs-attention-heads)
7. [Why This Architecture is Powerful](#why-this-architecture-is-powerful)

---

## Attention Fundamentals

At its core, attention is a mechanism that allows a model to dynamically focus on different parts of its input when producing an output. Rather than treating all inputs equally, the model learns to weight certain inputs more heavily based on their relevance.

### The Query-Key-Value Framework

Attention uses three learned projections:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain that might be relevant?"
- **Value (V)**: "What information should I pass forward if I'm relevant?"

The attention computation:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Where:
- `QK^T` computes similarity scores between queries and keys
- `√d` is a scaling factor for numerical stability
- `softmax` normalizes scores to attention weights (sum to 1)
- The result is a weighted sum of value vectors

### Intuitive Example

Imagine predicting tomorrow's conflict intensity given 30 days of history:

```
Day 1:  [equipment_lost=5, personnel=10, fires=20, ...]
Day 2:  [equipment_lost=3, personnel=8,  fires=15, ...]
...
Day 28: [equipment_lost=45, personnel=200, fires=500, ...]  ← Major escalation
Day 29: [equipment_lost=40, personnel=180, fires=480, ...]
Day 30: [equipment_lost=38, personnel=170, fires=450, ...]  ← Today
```

Without attention, the model might weight all days equally or use fixed decay. With attention, the model can learn: "Day 28 was unusual - I should pay extra attention to what happened there when making my prediction."

---

## Multi-Head Attention

Rather than having a single attention mechanism, transformers use **multiple attention heads in parallel**. Each head has its own learned Q, K, V projection matrices.

### Why Multiple Heads?

Different heads can learn to focus on different aspects of the data:

```
Head 1: Learns to detect weekly cyclical patterns
Head 2: Learns to identify sudden spikes/anomalies
Head 3: Learns long-range dependencies (events 3 weeks ago)
Head 4: Learns local smoothing (nearby days)
Head 5: Learns correlations between equipment types
Head 6: Learns regional fire patterns
Head 7: Learns personnel casualty trends
Head 8: Learns territorial change momentum
```

### How Heads Combine

```
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_8) · W_O

where head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
```

Each head produces its own output, these are concatenated, and a final projection (`W_O`) mixes them together. This gives the model 8 parallel "perspectives" on the same data.

---

## What Attention Operates Over

**Critical distinction**: Attention operates over the **sequence dimension** (timesteps), not over individual features.

### Sequence vs Features

For a daily source with shape `[batch, 30 days, 38 features]`:

```
                    ┌─────────────────────────────────┐
                    │   38 features at each timestep  │
                    └─────────────────────────────────┘
                                    │
Day 1  ────────────[tank, artillery, apc, ..., drone]
Day 2  ────────────[tank, artillery, apc, ..., drone]
Day 3  ────────────[tank, artillery, apc, ..., drone]
  ▲         ...
  │
  │     Day 28 ────────────[tank, artillery, apc, ..., drone] ← attn weight: 0.31
Attention
operates    Day 29 ────────────[tank, artillery, apc, ..., drone] ← attn weight: 0.12
over this
dimension   Day 30 ────────────[tank, artillery, apc, ..., drone] ← attn weight: 0.08
```

### What This Means

- **Attention weights** determine which **timesteps** to focus on
- **Feature importance** is learned through the Q, K, V projection matrices
- When attending to a timestep, the **entire feature vector** at that timestep contributes

### How Feature Importance is Learned

The Q, K, V projections are learned linear transformations:

```python
Q = input @ W_Q  # W_Q is [n_features, d_model]
K = input @ W_K  # W_K is [n_features, d_model]
V = input @ W_V  # W_V is [n_features, d_model]
```

Through training, these weight matrices learn to:
- **Emphasize important features** by giving them larger weights
- **De-emphasize noise** by giving smaller weights
- **Create useful combinations** by mixing features

So while attention weights operate over timesteps, the model learns feature importance through these projection matrices.

---

## The Multi-Resolution HAN Architecture

The Multi-Resolution Hierarchical Attention Network applies attention at multiple levels to process conflict data from diverse sources.

### Data Sources

**Daily Resolution:**
- Equipment losses (38 features): tanks, APCs, artillery, drones, etc.
- Personnel casualties (6 features): killed, wounded, captured, etc.
- DeepState spatial (53 features): unit positions, frontline metrics by region
- FIRMS spatial (39 features): fire hotspots, brightness, FRP by region
- Drones (6 features): drone-specific losses
- VIIRS (24 features): nighttime brightness patterns

**Monthly Resolution:**
- Sentinel satellite (43 features): imagery metadata, cloud cover
- HDX humanitarian (54 features): conflict events, food security, rainfall
- IOM displacement (18 features): population movement tracking

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Equipment ──→ [Source Embedding + Positional Encoding]             │
│  Personnel ──→ [Source Embedding + Positional Encoding]             │
│  DeepState ──→ [Source Embedding + Positional Encoding]             │
│  FIRMS     ──→ [Source Embedding + Positional Encoding]             │
│  ...                                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVEL 1: WITHIN-SOURCE ATTENTION                  │
├─────────────────────────────────────────────────────────────────────┤
│  Equipment ──→ [DailySourceEncoder (8 heads)] ──→ equipment_repr    │
│  Personnel ──→ [DailySourceEncoder (8 heads)] ──→ personnel_repr    │
│  DeepState ──→ [DailySourceEncoder (8 heads)] ──→ deepstate_repr    │
│  FIRMS     ──→ [DailySourceEncoder (8 heads)] ──→ firms_repr        │
│                                                                      │
│  Sentinel  ──→ [MonthlySourceEncoder (8 heads)] ──→ sentinel_repr   │
│  HDX       ──→ [MonthlySourceEncoder (8 heads)] ──→ hdx_repr        │
│  IOM       ──→ [MonthlySourceEncoder (8 heads)] ──→ iom_repr        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LEVEL 2: CROSS-SOURCE ATTENTION                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  equipment_repr ──┐                                                  │
│  personnel_repr ──┼──→ [CrossSourceAttention] ──→ fused_daily_repr  │
│  deepstate_repr ──┤         │                                        │
│  firms_repr     ──┘         │                                        │
│                             │                                        │
│  sentinel_repr  ──┐         │                                        │
│  hdx_repr       ──┼─────────┴──→ [CrossSourceAttention] ──→ fused_monthly │
│  iom_repr       ──┘                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 LEVEL 3: CROSS-RESOLUTION FUSION                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  fused_daily_repr   ──┬──→ [TemporalFusionLayer] ──→ temporal_output │
│  fused_monthly_repr ──┘         │                                    │
│                                 │                                    │
│                    [Month boundary alignment]                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PREDICTION HEADS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  temporal_output ──→ [CasualtyPredictionHead]  ──→ casualty_pred    │
│                 ──→ [RegimeClassificationHead] ──→ regime_logits    │
│                 ──→ [AnomalyDetectionHead]     ──→ anomaly_score    │
│                 ──→ [ForecastingHead]          ──→ forecast_pred    │
│                 ──→ [DailyForecastingHead]     ──→ daily_forecast   │
│                 ──→ [UncertaintyEstimator]     ──→ uncertainty      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hierarchical Attention Structure

The "Hierarchical" in HAN refers to attention operating at multiple levels:

### Level 1: Within-Source Temporal Attention

Each source has its own encoder with 8 attention heads. These heads attend **only within that source's temporal sequence**.

```
Equipment Source (30 days × 38 features)
    └── DailySourceEncoder
            └── 8 attention heads
                 ├── Head 1: learns to detect tank loss spikes
                 ├── Head 2: learns weekly artillery patterns
                 ├── Head 3: learns correlation with APC losses
                 └── ... (all attending to equipment timesteps only)
```

**Purpose**: Learn temporal patterns within each data source independently.

### Level 2: Cross-Source Attention

After each source is encoded, `CrossSourceAttention` allows sources to attend to each other.

```
Equipment representation can now "see":
    ├── Personnel patterns (do casualties follow equipment losses?)
    ├── FIRMS fire data (do fires correlate with losses?)
    ├── DeepState positions (does frontline movement predict losses?)
    └── etc.
```

**Purpose**: Learn relationships and correlations between different data sources.

### Level 3: Cross-Resolution Fusion

Daily and monthly representations are fused, with alignment based on month boundaries.

```
Daily (30 days) ──┬──→ Fused representation
                  │    - Daily patterns inform monthly context
Monthly (12 mo) ──┘    - Monthly trends inform daily predictions
```

**Purpose**: Combine fine-grained daily signals with longer-term monthly trends.

---

## Prediction Heads vs Attention Heads

These are fundamentally different components:

### Attention Heads (Inside Transformers)

- **Location**: Inside encoder layers
- **Function**: Learn which timesteps to attend to
- **Mechanism**: Q, K, V projections → attention weights → weighted sums
- **Count**: 8 per encoder layer
- **Training**: Learn automatically through backpropagation

### Prediction Heads (Output Layers)

- **Location**: After all encoding is complete
- **Function**: Transform encoded representations into task-specific outputs
- **Mechanism**: MLP layers → task-specific activation
- **Count**: One per prediction task
- **Training**: Supervised by task-specific loss functions

### Summary Table

| Head Type | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Attention Head** | Focus on relevant timesteps | Sequence of vectors | Weighted combination |
| **CasualtyPredictionHead** | Predict casualties | temporal_output | [batch, seq, 1] + variance |
| **RegimeClassificationHead** | Classify conflict phase | temporal_output | [batch, seq, 4] logits |
| **AnomalyDetectionHead** | Detect unusual activity | temporal_output | [batch, seq, 1] score |
| **ForecastingHead** | Predict monthly features | temporal_output | [batch, seq, 35] |
| **DailyForecastingHead** | Predict daily features | temporal_output | [batch, 7, 165+] |
| **UncertaintyEstimator** | Quantify confidence | temporal_output + density | [batch, seq, 1] |

---

## Why This Architecture is Powerful

### 1. Domain Agnostic

The attention mechanism doesn't care what the features represent. You provide:
- A sequence of feature vectors
- A prediction target

The model learns:
- Which timesteps matter
- Which feature combinations are predictive
- How sources relate to each other

This is why transformers work for text, images, audio, proteins, and conflict prediction.

### 2. Handles Irregular Data

The observation mask mechanism allows the model to:
- Process sequences with missing observations
- Learn from sparse, irregular OSINT data
- Gracefully handle days with no reports

### 3. Multi-Scale Reasoning

By combining daily and monthly resolutions:
- Captures short-term tactical changes
- Incorporates long-term strategic trends
- Learns which timescale matters for which prediction

### 4. Interpretable Attention

Attention weights can be extracted and analyzed:
- Which days did the model focus on for this prediction?
- Which sources were weighted most heavily?
- Do attention patterns align with known conflict events?

### 5. Extensible

Adding new data sources is straightforward:
1. Add a new source encoder
2. Register it in the cross-source attention
3. Retrain

The spatial data integration (DeepState + FIRMS) is an example - 92 new features added without architectural changes to the core attention mechanism.

---

## Practical Implications

### For Training

- The model learns feature importance through the Q, K, V projections
- Different attention heads specialize in different patterns
- Cross-source attention learns correlations between data modalities

### For Inference

- The model dynamically focuses on relevant timesteps
- Predictions are informed by the most relevant historical patterns
- Uncertainty estimates reflect observation density

### For Interpretation

- Attention weights show "what the model looked at"
- Source importance scores show relative weighting
- Probes can extract learned representations for analysis

---

## References

- Vaswani et al., "Attention Is All You Need" (2017) - Original transformer paper
- Yang et al., "Hierarchical Attention Networks" (2016) - HAN for document classification
- Project files: `analysis/multi_resolution_han.py`, `analysis/multi_resolution_data.py`
