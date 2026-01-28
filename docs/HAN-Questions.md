# HAN Model Interpretability Validation
## Master Prompt for Agent-Based Deep Analysis

**Last Updated:** 2026-01-26
**Status:** VALIDATION COMPLETE - Solutions Proposed

---

## EXECUTIVE SUMMARY

Comprehensive validation of the Multi-Resolution Hierarchical Attention Network (HAN) has been completed across two phases. Phase 1 tested the original claims through rigorous experiments, and Phase 2 investigated alternative hypotheses for each finding.

### Verdict Summary

| Claim | Original Hypothesis | Phase 1 Verdict | Phase 2 Verdict | Final Status |
|-------|---------------------|-----------------|-----------------|--------------|
| **C1** | Model ignores temporal structure | REFUTED | REFINED | Model learns at MONTHLY resolution primarily |
| **C2** | Attention is non-functional | INCONCLUSIVE | CLARIFIED | Functional but ARCHITECTURALLY limited |
| **C3** | ISW uncorrelated with latents | CONFIRMED | **CHALLENGED** | Raw data IS correlated; model fails to learn it |
| **C4** | VIIRS lags by +10-13 days | CONFIRMED | CONFIRMED | Lag real but mechanism unclear |
| **C5** | Model does static classification | CONFIRMED | STRENGTHENED | Output collapse + no causal mask |

---

## CONTEXT

You are tasked with validating and extending an interpretability analysis of the **Multi-Resolution Hierarchical Attention Network (HAN)** for multi-source OSINT fusion in Ukraine conflict monitoring.

> **Note on terminology:** Earlier versions of this document incorrectly referred to the model as "JIM"
> (Joint Interpolation Model). JIM is actually a separate Stage 1 pipeline component that was skipped
> during training (`--skip-interpolation` flag). This document analyzes the Stage 3 HAN model.
>
> **Note on ISW:** ISW (Institute for the Study of War) narrative embeddings are used for probe
> analysis (Section 5 probes) but are NOT integrated into the HAN model architecture. They are
> loaded separately for interpretability testing.

A preliminary interpretability probe suite has been executed, producing findings that challenge fundamental assumptions about what this model has learned. Before these findings can inform architectural decisions, they require independent validation through targeted experiments.

**Your mission:** Spawn a series of specialized agents to rigorously test each claim, explore alternative hypotheses, and produce a validated source-of-truth document that can guide model revision.

---

## THE MODEL: MULTI-RESOLUTION HIERARCHICAL ATTENTION NETWORK (HAN)

### Architecture Summary (Current: Disaggregated Equipment Configuration)

```
Input Layer:
├── DAILY SOURCES (~1426 timesteps, 8 sources with disaggregated equipment)
│   ├── drones (4 features) - UAV losses [disaggregated from equipment]
│   ├── armor (4 features) - tanks, APCs, IFVs [disaggregated from equipment]
│   ├── artillery (8 features) - field artillery, MRL [disaggregated from equipment]
│   ├── personnel (3 features) - casualty counts
│   ├── deepstate (5 features) - territorial control changes
│   ├── firms (13 features) - NASA FIRMS thermal hotspots
│   ├── viina (6 features) - Oryx-verified equipment losses
│   └── viirs (8 features) - VIIRS nightlight radiance [detrended]
│
└── MONTHLY SOURCES (~48 timesteps, 5 sources)
    ├── sentinel (7 features) - Copernicus satellite imagery metadata
    ├── hdx_conflict (5 features) - ACLED conflict events
    ├── hdx_food (10 features) - food security indicators
    ├── hdx_rainfall (6 features) - weather/precipitation
    └── iom (7 features) - IOM displacement tracking

Processing Pipeline:
1. Daily Source Encoders (8 parallel transformers, 3 layers each)
   - Learned no_observation_token for missing data
   - Observation status embeddings
   - Sinusoidal positional encoding

2. Daily Cross-Source Fusion (2 attention layers)
   - Sources attend to each other per timestep
   - Learned source_gate produces per-timestep importance weights
   - Output: fused_daily [batch, seq_len, d_model]

3. Learnable Monthly Aggregation
   - Attention-based (NOT averaging)
   - Month queries attend to their constituent days
   - Output: aggregated_daily [batch, n_months, d_model]

4. Monthly Source Encoders (parallel to daily)
   - Same structure for 5 monthly sources
   - Output: monthly_encoded [batch, n_months, d_model]

5. Cross-Resolution Fusion
   - Bidirectional attention between aggregated daily and monthly
   - Output: fused_monthly [batch, n_months, d_model]

6. Temporal Encoder (2 transformer layers)
   - Processes fused monthly sequence
   - NO causal masking (sees full sequence) ⚠️ PROBLEM
   - Output: temporal_encoded [batch, n_months, d_model]

7. Prediction Heads:
   - CasualtyPredictionHead → casualty predictions with uncertainty
   - RegimeClassificationHead → conflict regime logits
   - AnomalyDetectionHead → anomaly scores
   - ForecastingHead → multi-feature predictions

8. Uncertainty Estimation
   - Based on observation density
   - Scaled by (1 + (1 - density))
```

### Model Configuration Notes

- **d_model**: 64 (hidden dimension)
- **Equipment disaggregation**: Enabled (drones/armor/artillery instead of aggregated equipment)
- **VIIRS detrending**: Enabled (first-order differencing to remove spurious trend correlation)
- **ISW integration**: None (ISW embeddings are for probe analysis only, not model input)

---

## VALIDATED FINDINGS

### Claim C1: Temporal Structure Learning

**Original Claim:** "The model ignores temporal structure and treats input sequences as bags of features."

**Confidence: REFUTED → REFINED**

#### Phase 1 Experiments

| Experiment | Key Metrics | Result |
|------------|-------------|--------|
| Reverse Sequence Test | Casualty correlation: 0.998, Latent cosine similarity: 0.966 | CONFIRMS C1 (conf: 0.90) |
| Synthetic Trend Injection | Trend difference: -0.009, Up/down mean shift identical | CONFIRMS C1 (conf: 0.85) |
| Temporal Ablation Gradient | Context-error correlation: -0.73, Clear benefit from 365-day context | **REFUTES C1** (conf: 0.85) |
| Positional Encoding Analysis | Position variance: 0.35, Dimension variance: 0.46 across encoders | **REFUTES C1** (conf: 0.75) |
| Gradient Flow Analysis | Temporal encoder grad mean: 16.8, Daily encoder grad mean: 1.7 | **REFUTES C1** (conf: 0.80) |

**Phase 1 Verdict:** REFUTED (3/5 experiments refute)

#### Phase 2 Alternative Hypotheses

| Hypothesis | Test | Verdict |
|------------|------|---------|
| H2: Aggregate Statistics Confound | Shuffled Context Ablation | **FALSE** - prediction correlation drops to 0.185 |
| H3: Resolution Hierarchy Dominance | Resolution-Specific Reversal | **TRUE** - monthly 2.9x more sensitive than daily |
| H5: Gradient Passthrough | Temporal Knockout | **FALSE** - 22x relative change when temporal weights zeroed |

#### FINAL ANSWER

**The model DOES learn temporal patterns, but primarily at MONTHLY resolution.**

- Daily-level temporal order has minimal impact (monthly/daily sensitivity ratio: 2.88)
- Temporal computations are active (not passthrough)
- Strong gradients flow through temporal encoder (16.8 mean gradient)
- Positional encodings are actively used (variance 0.35-0.46)

**Implication:** The architecture is capable of temporal learning, but the hierarchical design concentrates temporal processing at the monthly level. Daily patterns may be lost during aggregation.

---

### Claim C2: Attention Mechanism Functionality

**Original Claim:** "Cross-source attention weights are approximately uniform (~0.04 for all pairs). No critical information pathways exist between sources."

**Confidence: INCONCLUSIVE → CLARIFIED**

#### Phase 1 Experiments

| Experiment | Key Metrics | Result |
|------------|-------------|--------|
| Entropy Measurement | Mean entropy: 2.02 (97% of max 2.08), Near-uniform weights | CONFIRMS C2 |
| Ablation vs Mean Pooling | Cosine similarity: 0.011 (near-orthogonal), L2 distance: 8.49 | **REFUTES C2** |
| Event-Specific Attention | Temporal importance std: 0.033, Max deviation: 0.023 | CONFIRMS C2 |
| Source Gate Analysis | VIIRS weight: 0.251 (2x uniform), Max deviation: 0.126 | **REFUTES C2** |
| Weight Statistics | Std: 0.071 (>> 0.01 threshold), Chi-square p < 0.001 | **REFUTES C2** |

**Source Importance Hierarchy Learned:**
- VIIRS: 27%
- Personnel: 16%
- FIRMS: 13%
- DeepState: 11%
- Equipment sources: ~7% each

**Phase 1 Verdict:** INCONCLUSIVE (2/5 confirm, 3/5 refute)

#### Phase 2 Alternative Hypotheses

| Hypothesis | Test | Verdict |
|------------|------|---------|
| H2: Gradient Starvation | Gradient Flow Analysis | **REFUTED** - Weights evolved throughout training |
| H3: Architectural Bottleneck | Temporal Context Analysis | **CONFIRMED** - source_gate is purely pointwise |

#### FINAL ANSWER

**Attention IS functional but ARCHITECTURALLY limited.**

The `source_gate` in `DailyCrossSourceFusion` is purely pointwise:
```python
self.source_gate = nn.Sequential(
    nn.Linear(d_model * n_sources, d_model),
    nn.LayerNorm(d_model),
    nn.GELU(),
    nn.Linear(d_model, n_sources),
    nn.Softmax(dim=-1),
)
```

- No RNN, Attention, or Conv layers = no temporal context
- Each timestep computes source importance independently
- Architecture CANNOT learn temporally-varying attention regardless of training

**Implication:** The attention mechanism successfully discriminates sources (VIIRS 2x higher than uniform), but cannot adapt dynamically to temporal context. Static source weighting, not a training failure.

---

### Claim C3: ISW-Latent Alignment

**Original Claim:** "ISW narrative embeddings show near-zero correlation with model latent representations."

**Confidence: CONFIRMED → CHALLENGED**

#### Phase 1 Experiments

| Experiment | Key Metrics | Result |
|------------|-------------|--------|
| Correlation Analysis (proxy) | Max r: 0.097, PC1 correlation: -0.001 | CONFIRMED |
| Event Response (proxy) | Shift correlation: 0.16 (p = 0.79) | REFUTED |
| Topic Alignment (proxy) | ARI: 0.035, NMI: 0.077 | CONFIRMED |
| Bidirectional Prediction (REAL) | Latent→ISW R²: **-1.00**, ISW→Latent R²: **-67.47** | CONFIRMED |

**Phase 1 Verdict:** CONFIRMED (Real latent-ISW R² = -1.00)

#### Phase 2 Alternative Hypotheses

| Hypothesis | Test | Verdict |
|------------|------|---------|
| H1: Fundamental Information Orthogonality | Cross-Domain Mutual Information | **FALSE** - CCA correlation = 0.9996, MI = 0.186 |
| H3: Temporal Resolution Mismatch | Daily Resolution Correlation | **INCONCLUSIVE** - Mixed evidence |
| H5: Regime-Specific Correlation | Regime-Stratified Analysis | **TRUE** - correlation varies r=0.55-0.91 across phases |

**Phase-Specific Correlations:**
| Conflict Phase | Sample Size | Max PC Correlation | Prediction R² |
|----------------|-------------|-------------------|---------------|
| summer_offensive_23 | 122 | **0.912** | 0.262 |
| eastern_focus | 110 | 0.884 | 0.322 |
| attritional_phase | 121 | 0.871 | 0.239 |
| kharkiv_offensive | 33 | 0.774 | 0.426 |
| avdiivka_fall | 90 | 0.550 | 0.247 |

#### FINAL ANSWER

**Raw sensor data DOES correlate with ISW (CCA r=0.9996), but model fails to learn this.**

**CRITICAL DISCREPANCY:** Phase 2 challenges Phase 1 conclusion:
- Phase 1: Model latents decorrelated from ISW (R² = -1.00)
- Phase 2: Raw features highly correlated with ISW (CCA = 0.9996)

**Root Causes:**
1. Daily vs monthly resolution mismatch (33 months of overlap vs 958 daily samples)
2. Global pooling masks phase-specific patterns (r ranges from 0.55 to 0.91)
3. Model architecture lacks mechanism to align with narrative content

**Implication:** The model SHOULD learn ISW-correlated representations. The failure is a technical limitation (architecture/training), not fundamental information orthogonality.

---

### Claim C4: VIIRS Causality

**Original Claim:** "VIIRS nightlight data correlates with events +10 days in the future (i.e., it reflects past damage)."

**Confidence: CONFIRMED**

#### Phase 1 Experiments

| Experiment | Key Metrics | Result |
|------------|-------------|--------|
| Cross-Correlation Analysis | radiance_mean peak: +12 days, radiance_p90 peak: +13 days | SUPPORTS C4 |
| Temporal Shift Experiment | Shifted R² improvement: +3.8% | CONFIRMED |
| Feature Isolation | All features non-stationary (ADF p=1.0), detrended r near 0 | SUPPORTS C4 |
| Causal Direction | Past R²: -7.41, Future R²: -6.90 (both poor) | NEUTRAL |

**Phase 1 Verdict:** CONFIRMED (VIIRS lags by +10-13 days)

#### Phase 2 Alternative Hypotheses

| Hypothesis | Test | Verdict |
|------------|------|---------|
| H1: Infrastructure Damage Cascade | FIRMS-to-VIIRS Cascade | **NOT SUPPORTED** - peak at 4-5 days, not 8-16 |
| H4: Third-Variable Confounding | Conflict Intensity Mediator | **NOT SUPPORTED** - controlling increases correlation |

**FIRMS→VIIRS Cascade Analysis:**
- fire_count peak lag: 5 days
- frp_sum peak lag: 4 days
- frp_max peak lag: -3 days (fires FOLLOW VIIRS changes)

#### FINAL ANSWER

**VIIRS lag of +10-13 days is REAL but the causal mechanism is UNCLEAR.**

- Simple infrastructure cascade (FIRMS→power grid→VIIRS) doesn't explain timing
- Conflict intensity doesn't fully mediate the relationship
- May involve complex reporting delays or composite period effects

**Implication:** Keep VIIRS detrending enabled. Consider explicit lag features or temporal shift modeling. VIIRS valuable for retrospective damage assessment, not prediction.

---

### Claim C5: Forecasting Ability

**Original Claim:** "Predictions are identical across 1, 3, 7, and 14-day horizons. The model classifies current state rather than forecasting future state."

**Confidence: CONFIRMED → STRENGTHENED**

#### Phase 1 Experiments

| Experiment | Key Metrics | Result |
|------------|-------------|--------|
| Prediction Distribution | Only 1 unique regime class (class 3), normalized entropy: 0.0 | SUPPORTS C5 |
| Persistence Baseline | HAN MAE: 7439.7, Persistence MAE: 411.4, Skill score: **-17.08** | SUPPORTS C5 |
| Output Variance | Target std: 665.4, Prediction std: 0.018, Variance ratio: ~0 | SUPPORTS C5 |
| Temporal Consistency | 0 transitions, Transition rate: 0.0 | SUPPORTS C5 |

**Phase 1 Verdict:** CONFIRMED (model performs 17x WORSE than persistence baseline)

#### Phase 2 Alternative Hypotheses

| Hypothesis | Test | Verdict |
|------------|------|---------|
| H3: Training Convergence to Mean/Mode | Output Collapse Quantification | **SUPPORTED** - 100% class 3, CV=0.002 |
| H5: Causal Masking Absence | Architecture Audit | **SUPPORTED** - no causal mask, future leakage possible |

**Output Collapse Evidence:**
- Regime predictions: 100% to class 3
- Normalized entropy: 5.2e-09 (essentially zero)
- Casualty CV: 0.002 (near-constant output)
- Mean logit margin: 6.49 (extremely overconfident)

**Causal Masking Audit:**
```python
# Current TemporalEncoder.forward():
hidden = self.transformer_encoder(
    hidden,
    src_key_padding_mask=src_key_padding_mask,  # Only padding mask
    # NO causal mask passed!
)
```
- `uses_causal`: False
- `uses_triu`: False
- `uses_is_causal`: False
- `src_mask_passed_to_transformer`: False

#### FINAL ANSWER

**Model performs static classification due to OUTPUT COLLAPSE and MISSING CAUSAL MASK.**

**Root Causes:**
1. **No causal masking:** TemporalEncoder allows future information leakage
2. **Output collapse:** Model converged to predicting majority class with near-zero variance
3. **Training objective issue:** `prediction_horizon` not used for target shifting

**Implication:** This is fixable. Add causal masking and use real autoregressive targets.

---

## PROPOSED SOLUTIONS

Based on the validated findings, the following architectural solutions are proposed:

### Priority 1: CRITICAL (Do First)

#### Solution C5-1: Add Causal Masking to TemporalEncoder

**Problem:** Future information leakage allows model to "cheat"

**Fix:**
```python
# In TemporalEncoder.__init__:
self.register_buffer(
    'causal_mask',
    torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
)

# In TemporalEncoder.forward:
attn_mask = self.causal_mask[:seq_len, :seq_len] if self.causal else None
hidden = self.transformer_encoder(hidden, mask=attn_mask, src_key_padding_mask=...)
```

**Effort:** 1 day | **Impact:** Critical

#### Solution C5-2: Replace Synthetic Targets

**Problem:** Training uses `torch.randint()` for regime targets, providing no learning signal

**Fix:** Use real next-timestep autoregressive targets with proper temporal shift

**Effort:** 3-5 days | **Impact:** Critical

---

### Priority 2: HIGH (Do Next)

#### Solution C2-1: Temporal Source Gate

**Problem:** `source_gate` is purely pointwise, cannot learn temporal attention patterns

**Fix:** Replace with `TemporalSourceGate` using Conv1d + self-attention:
```python
class TemporalSourceGate(nn.Module):
    def __init__(self, d_model, n_sources, kernel_size=7, nhead=4):
        self.temporal_conv = nn.Conv1d(d_model * n_sources, d_model, kernel_size)
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead)
        self.gate_projection = nn.Linear(d_model, n_sources)
```

**Effort:** 3-5 days | **Impact:** High

#### Solution C1-1: Daily Temporal Encoder

**Problem:** Daily patterns lost before monthly aggregation

**Fix:** Add `DailyTemporalEncoder` with local attention and multi-scale convolutions:
- Multi-scale Conv1d (3, 7, 14, 28 day kernels)
- Windowed self-attention (31-day window)
- Process BEFORE monthly aggregation

**Effort:** 5-7 days | **Impact:** High

---

### Priority 3: MEDIUM (After Core Fixes)

#### Solution C3-1: ISW Alignment Module

**Problem:** Raw data correlates with ISW (r=0.9996) but model fails to learn this

**Fix:** Add `ISWAlignmentModule` with contrastive loss:
- Project model latents and ISW embeddings to shared space
- Contrastive loss to align same-timestep representations
- Phase conditioning via learned phase embeddings

**Effort:** 7-10 days | **Impact:** Medium

#### Solution C3-2: Phase Conditioning

**Problem:** Global pooling masks phase-specific patterns (r varies 0.55-0.91)

**Fix:** Add `PhaseConditionedAttention`:
- Learn phase-specific query/key modulations
- Different attention patterns for different conflict phases

**Effort:** 3-5 days | **Impact:** Medium

---

### Priority 4: LOWER (Consider Later)

#### Solution C4-1: Learnable Temporal Shifts

**Problem:** VIIRS lags casualties by 10-13 days, treating sources as synchronous

**Fix:** Add `LearnableTemporalShift`:
- Learn source-specific temporal offsets
- Soft attention over shifted versions
- Cascade kernels for cross-source dependencies

**Effort:** 5-7 days | **Impact:** Low-Medium

---

## IMPLEMENTATION TIMELINE

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Foundation Fixes | Causal masking, real autoregressive targets |
| 2 | Source Processing | TemporalSourceGate, DailyTemporalEncoder |
| 3 | Multi-Modal Alignment | ISWAlignmentModule, phase conditioning |
| 4 | Temporal Alignment | LearnableTemporalShift |
| 5 | Validation | Re-run all Phase 2 tests, benchmark |

---

## EXPECTED IMPROVEMENTS

| Metric | Current | Target After Fixes |
|--------|---------|-------------------|
| Regime prediction CV | 0.002 | > 0.1 |
| Daily vs monthly sensitivity | 2.9x | < 2x |
| Model-ISW correlation | ~0 | > 0.5 |
| Source gate temporal autocorrelation | ~0 | > 0.3 |
| Forecast skill score | -17.08 | > 0 |

---

## FILES REFERENCE

### Validation Results
- Phase 1 Reports: `/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation/C[1-5]_*_report.md`
- Phase 2 Results: `/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation/phase2/`
- Phase 2 State: `/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation/phase2/PHASE2_STATE.md`

### Architecture
- Model Code: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`
- Training Code: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`
- Solutions Proposal: `/Users/daniel.tipton/ML_OSINT/docs/architecture_solutions_proposal.md`

### Checkpoint
- Path: `/Users/daniel.tipton/ML_OSINT/analysis/training_runs/run_24-01-2026_20-22/stage3_han/best_checkpoint.pt`
- d_model: 64
- Daily layers: 3
- Monthly layers: 2
- Fusion layers: 2
- Temporal layers: 2

---

## CONCLUSION

The HAN model validation reveals a model that:

1. **Does learn** temporal patterns at the monthly level (C1 refuted)
2. **Has functional** but architecturally limited attention (C2 clarified)
3. **Fails to capture** ISW correlations that exist in raw data (C3 challenged)
4. **Uses** a lagging indicator (VIIRS) with unclear causality (C4 confirmed)
5. **Exhibits** output collapse due to missing causal masking (C5 strengthened)

The most critical fix is adding causal masking and real autoregressive targets (C5). This alone may resolve much of the observed pathological behavior. Subsequent fixes address architectural limitations in temporal source gating (C2) and daily temporal processing (C1).

The C3 discrepancy - where raw data correlates strongly with ISW but model fails to learn this - represents an opportunity for significant improvement through ISW alignment and phase conditioning.

**Recommendation:** Proceed with implementation in priority order, re-running validation tests after each major change to track improvement.

---

*Document generated: 2026-01-26*
*Validation completed by Opus 4.5 agent ensemble*
