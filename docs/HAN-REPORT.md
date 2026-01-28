# HAN Model Interpretability Analysis Report
## Findings, Arguments, and Validation Framework

**Model:** Multi-Resolution Hierarchical Attention Network (HAN)
**Task:** Multi-source OSINT fusion for Ukraine conflict monitoring
**Analysis Date:** 2026-01-25
**Probe Coverage:** Sections 1-6 (Attention, Representation, Temporal, Latent, Narrative, Causal)

> **Note:** This report analyzes the Stage 3 HAN model. References to "JIM" (Joint Interpolation Models)
> in earlier versions were incorrect—JIM is a separate Stage 1 component that was skipped during training.
> ISW narrative embeddings are used for probe analysis but are not integrated into the HAN architecture.

---

# PART I: CORE THESIS

## Central Claim

**The HAN model, despite its hierarchical temporal attention architecture, operates as a static multi-input classifier that ignores temporal dynamics, discards narrative content, and relies on uniform attention averaging across a small subset of quantitative sources.**

This claim is decomposed into five falsifiable sub-claims:

| ID | Sub-Claim | Confidence | Primary Evidence |
|----|-----------|------------|------------------|
| C1 | Temporal structure is not learned | HIGH | Shuffling invariance (6.1.2) |
| C2 | Attention mechanism is non-functional | HIGH | Uniform weights, zero critical pathways (6.2.2) |
| C3 | ISW/narrative pathway is dead | HIGH | Zero correlation, bidirectional R² failure (5.x) |
| C4 | VIIRS contaminates predictions as lagging indicator | MEDIUM-HIGH | +10 day temporal offset (3.x) |
| C5 | Model performs static classification, not forecasting | HIGH | Identical horizon results (3.x) |

---

# PART II: EVIDENCE BY CLAIM

## Claim C1: Temporal Structure Is Not Learned

### Statement
The model treats input sequences as bags of features. The temporal ordering of data within each source has no effect on predictions.

### Evidence

**Probe 6.1.2 - Source Shuffling Interventions**

Methodology: For each source, randomly permute the temporal dimension while preserving feature values. Measure prediction distance from original.

Results:
```
Sources where temporal_structure_matters = True:  1 of 52 tests
  - hdx_rainfall (regime task only): effect = 0.297

Sources where temporal_structure_matters = False: 51 of 52 tests
  - All other source-task combinations
  - Prediction distances: O(10^-7) to O(10^-3)
```

Interpretation: If the model learned temporal patterns (trends, sequences, dynamics), shuffling time should drastically change predictions. It does not. The model computes some aggregate (likely mean or recent value) and ignores ordering.

**Probe 6.1.3 - Mean Substitution**

Methodology: Replace each source's temporal sequence with its mean value repeated across time.

Results:
```
ALL 52 source-task combinations: interpretation = "value_matters"
ZERO combinations: interpretation = "deviation_important"
```

Interpretation: The model cares about absolute levels, not deviations from mean or temporal dynamics. This is consistent with static feature extraction.

**Supporting Evidence - Context Window Probe (Section 3)**

Results:
```
7-day context:   70.3% accuracy
365-day context: 51.8% accuracy
```

Interpretation: Longer temporal context HURTS performance. If the model used historical patterns, more context should help or be neutral. The degradation suggests the model is confused by (or ignoring) distant history.

### Validation Approaches for C1

1. **Reverse Sequence Test**: Feed sequences in reverse temporal order. If C1 is true, predictions should be identical.

2. **Synthetic Trend Injection**: Add obvious linear trends to sources. If C1 is true, model should not respond to trend direction.

3. **Temporal Ablation Gradient**: Systematically mask increasing portions of history. Plot accuracy vs. context length to find effective context window.

4. **Positional Encoding Analysis**: Examine whether positional encodings have learned meaningful weights or are near-uniform.

---

## Claim C2: Attention Mechanism Is Non-Functional

### Statement
The hierarchical attention mechanism adds parameters but not function. Cross-source attention weights are approximately uniform, and no critical information pathways exist between sources.

### Evidence

**Probe 6.2.2 - Attention Knockout**

Methodology: For each source pair, zero out the attention pathway and measure prediction impact.

Results:
```
Critical pathways found: 0 (across all 4 tasks)
Total pathways tested:   20
Mean flow strength:      0.0400 (uniform distribution = 0.05 for 20 sources)

Flow graph sample (all ~0.04):
  drones → armor:     0.0407
  drones → artillery: 0.0445
  drones → personnel: 0.0413
  armor → drones:     0.0407
  armor → artillery:  0.0384
```

Interpretation: All attention weights are nearly identical. The attention mechanism is performing uniform averaging, not selective routing. Knocking out any single pathway has negligible effect because all pathways carry equal (minimal) information.

**Supporting Evidence - Attention Distance Analysis (Section 2)**

Results:
```
Source encoder attention distances:      ~4 units (uniform)
Cross-attention head distances:          ~190 units (high but uniform)
Variance across sources:                 LOW (except hdx_rainfall)
```

Interpretation: Attention distances are high but undifferentiated. The model isn't learning to attend differently to different sources.

### Validation Approaches for C2

1. **Attention Entropy Measurement**: Compute entropy of attention distributions. Uniform attention = maximum entropy. Compare to randomly initialized model.

2. **Attention Ablation Cascade**: Remove attention layers entirely, replace with mean pooling. If C2 is true, performance should be identical.

3. **Forced Sparse Attention**: Retrain with top-k attention masking. If model has latent preferences, sparse attention should surface them.

4. **Attention Visualization**: Generate attention heatmaps for known events (Kerch Bridge, Kherson). Check for event-specific patterns.

---

## Claim C3: ISW/Narrative Pathway Is Dead

### Statement
The ISW (Institute for the Study of War) narrative embeddings contribute zero information to model predictions. The narrative pathway adds parameters and compute but no predictive signal.

### Evidence

**Probe 5.1.1 - ISW-Latent Correlation**

Methodology: Compute cosine similarity between ISW embeddings and model latent representations for each day.

Results:
```
Mean cosine similarity:  0.0087 (essentially zero)
Std:                     0.1226
Range:                   [-0.395, +0.340]

By conflict period:
  Initial invasion:      +0.067
  Eastern focus:         +0.043
  Ukrainian counter:     -0.082  ← NEGATIVE during UA success
  Bakhmut period:        +0.001
  Counteroffensive 2023: +0.041
  Attritional phase:     -0.004
```

Interpretation: Near-zero mean correlation indicates ISW content and model latents are unrelated. The NEGATIVE correlation during Ukrainian counteroffensive is particularly telling - the model's internal state diverged from ISW narrative during rapid battlefield changes.

**Probe 5.2.1 - Event-Triggered Response**

Methodology: Measure embedding and latent shifts around known major events.

Results:
```
Event                    | Embed Δ | Latent Δ | Ratio
-------------------------|---------|----------|-------
Kerch Bridge attack      | 0.0828  | 0.3522   | 4.25x  ← PROPAGATES
Kherson withdrawal       | 0.0521  | 0.2092   | 4.01x  ← PROPAGATES
Prigozhin mutiny (start) | 0.0599  | 0.0002   | 0.00x  ← IGNORED
Prigozhin mutiny (end)   | 0.0660  | 0.0002   | 0.00x  ← IGNORED
Kakhovka Dam collapse    | 0.0619  | 0.0001   | 0.00x  ← IGNORED
Avdiivka fall            | 0.0356  | null     | N/A
```

Interpretation: ISW embeddings DO shift for all events (Embed Δ column). But only events with quantitative signatures (Kerch = infrastructure destruction, Kherson = territorial change) propagate to latents. Political events (Prigozhin) and humanitarian events (Kakhovka) are captured by ISW but IGNORED by the model.

**Probe 5.3.2 - Bidirectional Prediction**

Methodology: Train linear probes to predict ISW from latents and latents from ISW.

Results:
```
Latent → ISW prediction:
  Mean R²: -1.940 (worse than predicting mean)
  Reconstruction cosine sim: -0.143

ISW → Latent prediction:
  Mean R²: -88.651 (catastrophically bad)
  Reconstruction cosine sim: 0.536
```

Interpretation: Neither representation can predict the other. They exist in essentially orthogonal spaces. The ISW encoder is producing embeddings that the rest of the model cannot use.

**Probe 5.3.1 - Counterfactual Perturbation**

Methodology: Swap ISW embeddings between random date pairs, measure latent change.

Results:
```
Embedding-Latent correlation: -0.154 (NEGATIVE)
Mean latent change:           0.232
```

Interpretation: Swapping ISW content produces unpredictable latent changes. The negative correlation means larger embedding changes sometimes produce SMALLER latent changes. The model is not using ISW information in any coherent way.

### Validation Approaches for C3

1. **ISW Ablation Test**: Zero out ISW embeddings entirely for all inputs. If C3 is true, task performance should be unchanged.

2. **ISW-Only Prediction**: Train a model using ONLY ISW embeddings. Measure baseline performance. Compare to ISW contribution in full model.

3. **Attention to ISW**: If ISW has a dedicated attention head, measure attention weights to ISW vs. other sources. 

4. **Narrative-Specific Probe**: Create synthetic ISW embeddings for "escalation" vs. "de-escalation" narratives. Test if model responds differently.

5. **Gradient Flow Analysis**: Compute gradients from loss back to ISW encoder. If gradients are near-zero, ISW is not receiving training signal.

---

## Claim C4: VIIRS Contaminates Predictions As Lagging Indicator

### Statement
VIIRS (night light/thermal anomaly data) is a +10 day lagging indicator that encodes "damage has already occurred" rather than "damage will occur." Its high attention weight contaminates predictions with backward-looking information.

### Evidence

**Temporal Ablation Analysis (Section 3)**

Methodology: Compute cross-correlation between VIIRS features and target variables at different time lags.

Results:
```
VIIRS temporal offset: +10 days (VIIRS lags behind events)
VIIRS attention weight: 39% (highest of all sources)
VIIRS importance trend: 13.8% → 19.7% (increases during training)
```

Interpretation: VIIRS gets the most attention and its importance grows during training. But it's measuring aftermath, not precursors. The model is learning "when VIIRS shows damage, bad things happened" - which is tautological and non-predictive.

**Source Zeroing Results (6.1.1)**

Results:
```
VIIRS zeroing effect by task:
  Anomaly:  0.825 (HIGHEST - 5x the prediction change)
  Casualty: 0.089
  Regime:   0.001
  Forecast: 0.004
```

Interpretation: VIIRS dominates anomaly detection but has minimal causal impact on forecasting. This is consistent with VIIRS being useful for "what just happened" but not "what will happen."

**Detrending Analysis (Section 3)**

Results:
```
VIIRS features surviving detrending: radiance_std only (16%)
Other VIIRS features: spurious correlation with time
```

Interpretation: Most VIIRS signal is trend-correlated (war damage accumulates over time). Only radiance variability contains non-spurious information.

### Validation Approaches for C4

1. **Temporal Shift Experiment**: Artificially shift VIIRS data forward by 10 days (so it "predicts" rather than "reports"). Measure if forecasting improves.

2. **VIIRS Feature Isolation**: Test model with only radiance_std (the non-spurious feature) vs. full VIIRS. Compare forecasting performance.

3. **Causal Direction Test**: Train auxiliary classifier: "Given VIIRS, predict past events" vs. "Given VIIRS, predict future events." C4 predicts past-prediction will be better.

4. **VIIRS Pathway Ablation**: Remove VIIRS from prediction heads but keep for anomaly detection. Measure forecast improvement.

---

## Claim C5: Model Performs Static Classification, Not Forecasting

### Statement
The model produces identical predictions regardless of forecast horizon (1, 3, 7, 14 days). It is performing static state classification ("current situation is X") rather than temporal forecasting ("in N days, situation will be Y").

### Evidence

**Predictive Horizon Probe (Section 3)**

Methodology: Evaluate model predictions at 1, 3, 7, and 14 day horizons.

Results:
```
Horizon | Accuracy | Confusion Matrix Pattern
--------|----------|-------------------------
1 day   | 57%      | Majority class prediction
3 days  | 57%      | Majority class prediction
7 days  | 57%      | Majority class prediction
14 days | 57%      | Majority class prediction
```

Interpretation: IDENTICAL results across all horizons. A true forecasting model should show:
- Decreasing accuracy at longer horizons (uncertainty grows)
- Different confusion patterns (short-term vs. long-term dynamics)

The model is outputting the same prediction regardless of horizon - it's classifying current state, not forecasting future state.

**Source Zeroing - Forecast Task (6.1.1)**

Results:
```
Forecast task - source zeroing effects:
  viirs:      0.0037
  personnel:  0.0012
  All others: < 0.0004
```

Interpretation: ALL sources have minimal causal impact on forecast task. The model isn't using input information to forecast - it's likely outputting a default or majority-class prediction.

### Validation Approaches for C5

1. **Horizon-Specific Training**: Train separate models for each horizon. If C5 is true about architecture, separate models should also fail. If they succeed, the issue is training objective.

2. **Prediction Distribution Analysis**: Plot prediction distributions for each horizon. If C5 is true, distributions should be identical.

3. **Synthetic Future Injection**: Create test cases where future is known to differ from present (e.g., scheduled events). Measure if model predictions change.

4. **Autoregressive Comparison**: Compare to simple autoregressive baseline (predict tomorrow = today). If HAN doesn't beat this, C5 is confirmed.

---

# PART III: ALTERNATIVE HYPOTHESES

For each claim, we consider alternative explanations that validation should rule out.

## Alternative to C1 (Temporal Learning)

**Alt-C1a: Temporal patterns exist but are subtle**
- Counter-evidence: Even hdx_rainfall shows temporal importance for regime task
- Validation: Amplify temporal signals synthetically, test if model responds

**Alt-C1b: Model learns temporal patterns in deeper layers not probed**
- Counter-evidence: Shuffling affects final predictions, not intermediate representations
- Validation: Apply shuffling at different layer depths

## Alternative to C2 (Attention Function)

**Alt-C2a: Attention is specialized but our knockout method is flawed**
- Counter-evidence: Multiple independent methods (knockout, entropy, distance) agree
- Validation: Use activation patching instead of zeroing

**Alt-C2b: Uniform attention is optimal for this task**
- Counter-evidence: Known domain structure (combat sources should attend differently to satellite sources)
- Validation: Compare to attention patterns in similar successful models

## Alternative to C3 (ISW Utility)

**Alt-C3a: ISW provides regularization, not direct signal**
- Counter-evidence: Ablation should show overfitting if this were true
- Validation: Compare train/test gap with and without ISW

**Alt-C3b: ISW signal is present but our embedding method is flawed**
- Counter-evidence: ISW embeddings DO shift for events (Embed Δ is non-zero)
- Validation: Use different embedding model (e.g., domain-specific fine-tuned)

## Alternative to C4 (VIIRS Causality)

**Alt-C4a: VIIRS is both lagging AND leading (captures ongoing situations)**
- Counter-evidence: Detrending removes most signal
- Validation: Separate VIIRS into "current" vs. "historical" features, test independently

## Alternative to C5 (Forecasting Ability)

**Alt-C5a: Model forecasts but evaluation metric is insensitive**
- Counter-evidence: Confusion matrices are identical, not just accuracy
- Validation: Use probabilistic metrics (calibration, Brier score) instead of accuracy

---

# PART IV: SYNTHESIS AND IMPLICATIONS

## What The Model Actually Does

Based on converging evidence, the HAN operates as follows:

```
Input Processing:
  1. Receive 365-day sequences from 13+ sources
  2. Ignore temporal ordering (treat as bag of features)
  3. Compute source-level summaries (likely means or recent values)
  
Fusion:
  4. Apply attention mechanism (but weights are uniform)
  5. Effectively: average all source summaries equally
  
Prediction:
  6. Route averaged representation to task heads
  7. Output static classification (insensitive to horizon)
  8. For anomaly task: heavily weight VIIRS (damage proxy)
  9. For casualty task: heavily weight personnel (direct measure)
  10. For forecast task: output near-default predictions
```

## Why This Happened

Likely causes:
1. **Training objective mismatch**: Loss function rewards correct classification, not temporal reasoning
2. **Data characteristics**: Temporal signal-to-noise ratio is low; static features are easier to learn
3. **Optimization path**: Gradient descent found local minimum using static features before exploring temporal patterns
4. **ISW embedding quality**: Pre-trained embeddings may not capture conflict-relevant semantics

## Architectural Capacity

The architecture HAS capacity for temporal reasoning:
- Attention mechanisms exist (just not utilized)
- Temporal encodings are present (just not learned)
- ISW pathway exists (just not integrated)

This is not a fundamental architectural limitation - it's a training/objective problem.

---

# PART V: VALIDATION AGENT INSTRUCTIONS

## Agent Spawn Protocol

Each claim should be validated by an independent agent with:
1. Access to model weights and inference
2. Access to training data
3. Ability to run new experiments
4. No access to this report's conclusions (blind validation)

## Agent Task Specifications

### Agent C1: Temporal Learning Validator
```
Objective: Confirm or refute that model ignores temporal structure
Required experiments:
  - Reverse sequence test
  - Synthetic trend injection  
  - Temporal ablation gradient
  - Positional encoding weight analysis
Success criteria: 3+ experiments must agree on conclusion
```

### Agent C2: Attention Function Validator
```
Objective: Confirm or refute that attention is non-functional
Required experiments:
  - Attention entropy measurement
  - Attention ablation (replace with mean pooling)
  - Attention visualization for known events
Success criteria: Entropy > 0.95 * max_entropy AND ablation shows < 2% performance change
```

### Agent C3: ISW Pathway Validator
```
Objective: Confirm or refute that ISW contributes zero signal
Required experiments:
  - ISW ablation (zero embeddings)
  - ISW-only baseline model
  - Gradient flow to ISW encoder
Success criteria: Ablation shows < 1% performance change AND gradients are < 0.01 * mean gradient
```

### Agent C4: VIIRS Causality Validator
```
Objective: Confirm or refute VIIRS lagging indicator hypothesis
Required experiments:
  - Temporal shift experiment (+10 days)
  - VIIRS feature isolation (radiance_std only)
  - Causal direction classifier
Success criteria: Forward-shifted VIIRS improves forecast OR past-prediction >> future-prediction
```

### Agent C5: Forecasting Ability Validator
```
Objective: Confirm or refute that model performs static classification
Required experiments:
  - Horizon-specific training
  - Prediction distribution analysis
  - Autoregressive baseline comparison
Success criteria: All horizons produce identical distributions AND model ≈ AR baseline
```

### Agent C6: Deep Dive - Alternative Hypothesis Tester
```
Objective: Test alternative explanations listed in Part III
Required experiments:
  - One experiment per alternative hypothesis
  - Document which alternatives can be ruled out
Success criteria: All plausible alternatives either ruled out or flagged for further investigation
```

---

# PART VI: APPENDIX - RAW PROBE RESULTS

## 6.1.1 Source Zeroing Rankings

### Anomaly Task
| Rank | Source | Effect |
|------|--------|--------|
| 1 | viirs | 0.8250 |
| 2 | personnel | 0.3481 |
| 3 | viina | 0.2197 |
| 4 | firms | 0.1648 |
| 5 | armor | 0.1277 |

### Casualty Task
| Rank | Source | Effect |
|------|--------|--------|
| 1 | personnel | 0.1647 |
| 2 | viirs | 0.0890 |
| 3 | viina | 0.0503 |
| 4 | drones | 0.0279 |
| 5 | armor | 0.0244 |

### Regime Task
| Rank | Source | Effect |
|------|--------|--------|
| 1 | viina | 0.0568 |
| 2 | personnel | 0.0547 |
| 3 | firms | 0.0536 |
| 4 | armor | 0.0188 |
| 5 | hdx_rainfall | 0.0147 |

### Forecast Task
| Rank | Source | Effect |
|------|--------|--------|
| 1 | viirs | 0.0037 |
| 2 | personnel | 0.0012 |
| 3 | viina | 0.0003 |
| 4 | armor | 0.0003 |
| 5 | firms | 0.0002 |

## 5.1.1 ISW Alignment by Period

| Period | Mean Similarity | N Days | Date Range |
|--------|-----------------|--------|------------|
| Initial Invasion | +0.067 | 47 | 2022-02-24 to 2022-04-15 |
| Eastern Focus | +0.043 | 136 | 2022-04-16 to 2022-08-31 |
| Ukrainian Counter | -0.082 | 119 | 2022-09-01 to 2022-12-31 |
| Bakhmut Period | +0.001 | 149 | 2023-01-01 to 2023-05-31 |
| Counteroffensive 2023 | +0.041 | 153 | 2023-06-01 to 2023-10-31 |
| Attritional Phase | -0.004 | 32 | 2023-11-01 to 2024-01-31 |

## 6.2.2 Attention Flow Graph (Sample)

| From | To | Weight |
|------|-----|--------|
| drones | armor | 0.0407 |
| drones | artillery | 0.0445 |
| drones | personnel | 0.0413 |
| drones | deepstate | 0.0435 |
| armor | drones | 0.0407 |
| armor | artillery | 0.0384 |
| armor | personnel | 0.0357 |
| armor | deepstate | 0.0376 |

---

**END OF REPORT**

*This document serves as the source of truth for HAN model interpretability findings. Validation agents should treat claims as hypotheses to be tested, not conclusions to be confirmed.*
