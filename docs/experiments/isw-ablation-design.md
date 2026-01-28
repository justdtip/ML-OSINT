# ISW Integration Ablation Experiment

**Date:** 2026-01-27
**Status:** Findings documented
**Purpose:** Determine whether ISW narrative alignment helps or hurts forecast performance

---

## Key Finding: ISW is Inference-Neutral

**Critical Discovery:** The ISW alignment module does NOT affect model outputs during inference. When comparing a model loaded with ISW weights vs. without, the outputs are **identical**:

```
temporal_output: mean_abs_diff = 0.000000
forecast_pred: mean_abs_diff = 0.000000
regime_logits: mean_abs_diff = 0.000000
```

### Why This Happens

The ISW alignment module is an **auxiliary training component**:
1. During training, it computes a contrastive loss to align model representations with ISW narrative embeddings
2. This loss affects gradient updates, influencing how the core model learns
3. During inference, the ISW module's projection outputs are not used in the main forward path
4. The core model weights (which ARE affected by ISW during training) are identical in both configurations

### Implications

1. **ISW is truly modular** - can be safely dropped at inference with:
   - 757K fewer parameters to load
   - Slightly faster inference time
   - No change in predictions

2. **The modularity finding from probes was about training dynamics**:
   - ISW affects *how* representations are learned during training
   - But both models produce identical outputs from the same trained weights

3. **Proper evaluation requires**:
   - Training models from scratch with/without ISW
   - Or measuring representation quality (not predictions)

---

---

## Background

Two conflicting pieces of evidence exist regarding ISW integration:

### Evidence FOR ISW Integration (from `narrative-inregration.md`)
- Raw sensor → ISW CCA correlation: **0.9996**
- This suggests ISW contains highly relevant information
- Expected to improve forecast correlation from 0.54 → 0.6-0.7

### Evidence AGAINST ISW Integration (from `future-architecture-ideas.md`)
- Model exhibits emergent modularity
- Without ISW: latent velocity range 1.06-1.09, phase correlation r = -0.69 (strong)
- With ISW: latent velocity range 0.93-6.28, phase correlation r = -0.28 (weak)
- ISW alignment showed overfitting (train: 0.27, val: 0.52)
- Model latent → ISW R² = -1.00 (anti-correlated)

---

## Hypotheses

| Hypothesis | Description | Implication |
|------------|-------------|-------------|
| H0 | ISW has no effect on forecast performance | Remove ISW for simplicity |
| H1a | ISW improves forecast performance | Keep/enhance ISW integration |
| H1b | ISW degrades forecast performance | Disable ISW for temporal tasks |

---

## Experimental Design

### Independent Variable
- **ISW Module State**: Enabled vs. Disabled

### Dependent Variables
| Metric | Description | Better Direction |
|--------|-------------|------------------|
| Forecast MSE | Mean squared error of 7-day forecast | Lower |
| Forecast MAE | Mean absolute error of 7-day forecast | Lower |
| Forecast Correlation | Pearson r between predicted and actual | Higher |
| Latent Velocity | Mean L2 distance between consecutive timesteps | Lower (more stable) |
| Inference Time | Computational cost | Lower |

### Control Variables
- Same checkpoint used for both conditions
- Same random seed (42) for sample selection
- Same evaluation samples
- Same evaluation pipeline

---

## Methodology

### 1. Model Loading

```
Condition A: WITH ISW
- Load full checkpoint
- All 11.4M parameters active

Condition B: WITHOUT ISW
- Load checkpoint with ISW weights filtered out
- ~10.7M parameters active (757K ISW params discarded)
```

### 2. Evaluation Protocol

For each condition:
1. Load model in specified configuration
2. Sample N=100 evaluation points (same indices for both conditions)
3. For each sample:
   - Run forward pass
   - Record forecast predictions vs actuals
   - Record regime predictions
   - Record latent representations
   - Measure inference time
4. Compute aggregate statistics

### 3. Statistical Analysis

| Test | Purpose |
|------|---------|
| Independent t-test | Compare means between conditions |
| Mann-Whitney U | Non-parametric comparison |
| Cohen's d | Effect size magnitude |
| Bootstrap CI | 95% confidence interval on difference |

### 4. Decision Criteria

| Outcome | Recommendation |
|---------|----------------|
| Without ISW significantly better on ≥2 key metrics | Disable ISW for forecast tasks |
| With ISW significantly better on ≥2 key metrics | Keep ISW enabled |
| No significant difference | Remove ISW for simplicity |
| Mixed results | Consider task-specific loading |

---

## Running the Experiment

```bash
cd /Users/daniel.tipton/ML_OSINT
python -m analysis.experiments.isw_ablation_experiment
```

### Expected Runtime
- ~5-10 minutes depending on hardware
- 100 samples × 2 conditions × ~1-2 sec/sample

### Output Files
- `outputs/experiments/isw_ablation/experiment_results.json` - Full results
- `outputs/experiments/isw_ablation/experiment_results.png` - Visualization

---

## Interpreting Results

### Effect Size (Cohen's d)
| d value | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### Significance Threshold
- p < 0.05 considered significant
- Both parametric (t-test) and non-parametric (Mann-Whitney) reported

---

## Potential Outcomes

### Scenario 1: ISW Hurts Performance (Likely based on prior evidence)

**Expected findings:**
- Without ISW shows lower forecast MSE
- Without ISW shows higher forecast correlation
- Without ISW shows lower latent velocity (tighter trajectories)

**Implication:**
- Confirms modularity hypothesis
- ISW should be disabled for temporal prediction
- Consider adapter architecture for semantic tasks

### Scenario 2: ISW Helps Performance

**Expected findings:**
- With ISW shows higher forecast correlation
- Effect size is meaningful (d > 0.3)

**Implication:**
- Current integration is working
- Focus on fixing ISW overfitting
- Continue enhancement per `narrative-inregration.md`

### Scenario 3: No Significant Difference

**Expected findings:**
- p > 0.05 on all metrics
- Effect sizes negligible (d < 0.2)

**Implication:**
- ISW neither helps nor hurts
- Remove for architectural simplicity
- 757K fewer parameters to maintain

---

## Follow-up Experiments

Depending on results:

1. **If ISW hurts:** Test phase-specific loading (enable ISW only for certain phases)
2. **If ISW helps:** Test increasing ISW alignment weight
3. **If no difference:** Test on semantic tasks (narrative grounding, anomaly detection)

---

## References

- `docs/narrative-inregration.md` - ISW integration specification
- `docs/future-architecture-ideas.md` - Emergent modularity discovery
- `analysis/backtesting.py` - Backtest framework (r=0.54 baseline)
- `analysis/checkpoints/multi_resolution/best_checkpoint.pt` - Target checkpoint
