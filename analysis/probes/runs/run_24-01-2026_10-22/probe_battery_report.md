# Multi-Resolution HAN Probe Battery Report

**Generated:** 2026-01-24 10:44:29

## Executive Summary

- **Total Probes:** 46
- **Completed:** 42
- **Failed:** 4
- **Skipped:** 0
- **Total Duration:** 1345.26s

## Key Findings by Tier

### Tier 1

| Probe ID | Name | Status | Duration |
|----------|------|--------|----------|
| 1.2.1 | VIIRS-Casualty Temporal Relationship | ✓ completed | 0.24s |
| 1.1.2 | Equipment-Personnel Redundancy Test | ✓ completed | 0.25s |
| 6.1.1 | Source Zeroing Interventions | ✓ completed | 8.52s |
| 4.1.1 | Named Operation Clustering | ✓ completed | 11.93s |
| 5.1.1 | ISW-Latent Correlation | ✓ completed | 11.39s |

### Tier 2

| Probe ID | Name | Status | Duration |
|----------|------|--------|----------|
| 1.2.3 | Trend Confounding Test | ✓ completed | 0.20s |
| 2.2.1 | Leave-One-Out Ablation | ✓ completed | 183.91s |
| 4.1.2 | Day-Type Decoding Probe | ✓ completed | 11.26s |
| 5.2.1 | Event-Triggered Response Analysis | ✓ completed | 11.17s |
| 3.1.1 | Truncated Context Inference | ✗ failed | 3.97s |

### Tier 3

| Probe ID | Name | Status | Duration |
|----------|------|--------|----------|
| 1.1.1 | Encoding Variance Comparison | ✓ completed | 0.26s |
| 1.1.3 | Equipment Category Disaggregation | ✓ completed | 0.17s |
| 1.1.4 | Temporal Lag Analysis - Equipment | ✓ completed | 0.19s |
| 1.2.2 | VIIRS Feature Decomposition | ✓ completed | 0.24s |
| 1.2.4 | Geographic VIIRS Decomposition | ✓ completed | 0.17s |
| 1.3.1 | Personnel-VIIRS Mediation Analysis | ✓ completed | 0.14s |
| 2.1.1 | Representation Similarity Analysis | ✓ completed | 7.65s |
| 2.1.2 | Cross-Source Information Flow | ✓ completed | 4.59s |
| 2.1.4 | Checkpoint Comparison | ✓ completed | 12.61s |
| 2.2.2 | Source Sufficiency Test | ✓ completed | 188.76s |
| 3.1.2 | Temporal Attention Patterns | ✗ failed | 2.14s |
| 3.1.3 | Predictive Horizon Analysis | ✗ failed | 7.31s |
| 3.2.1 | Transition Boundary Analysis | ✓ completed | 0.12s |
| 3.2.2 | Latent Velocity Prediction | ✗ failed | 1.90s |
| 4.1.3 | Intensity Level Decoding | ✓ completed | 11.32s |
| 4.1.4 | Geographic Focus Decoding | ✓ completed | 11.23s |
| 4.2.1 | Weekly Cycle Detection | ✓ completed | 11.08s |
| 4.2.2 | Seasonal Pattern Detection | ✓ completed | 11.19s |
| 4.2.3 | Event Anniversary Detection | ✓ completed | 11.45s |
| 5.1.2 | ISW Topic-Source Correlation | ✓ completed | 11.89s |
| 5.1.3 | ISW Predictive Content Test | ✓ completed | 11.44s |
| 5.2.2 | Narrative-Numerical Lag Analysis | ✓ completed | 11.16s |
| 5.2.3 | Semantic Anomaly Detection | ✓ completed | 11.07s |
| 5.3.1 | Semantic Perturbation Effects | ✓ completed | 11.03s |
| 5.3.2 | Missing Semantic Interpolation | ✓ completed | 11.24s |
| 6.1.2 | Source Shuffling Interventions | ✓ completed | 7.93s |
| 6.1.3 | Source Mean Substitution | ✓ completed | 1.75s |
| 6.2.1 | Integrated Gradients | ✓ completed | 665.85s |
| 6.2.2 | Attention Knockout | ✓ completed | 0.16s |
| 7.1.1 | Regional Signal Availability | ✓ completed | 26.12s |
| 7.1.2 | Front-Line Sector Definition | ✓ completed | 0.00s |
| 7.1.3 | Sector Independence Test | ✓ completed | 0.00s |
| 7.2.1 | Unit Tracking Data Availability | ✓ completed | 0.00s |
| 7.2.2 | Entity State Representation Design | ✓ completed | 0.00s |
| 7.3.1 | Temporal Resolution Analysis | ✓ completed | 25.18s |
| 7.3.2 | Spatial Resolution Analysis | ✓ completed | 25.06s |

## Detailed Results

### 1.1.1: Encoding Variance Comparison

**Status:** completed
**Duration:** 0.26s

**Findings:**
```json
[
  {
    "key_result": "Coefficient of Variation - Cumulative: 0.5655, Delta: 2.3887, Rolling-7: 1.4988",
    "significance": "Delta encoding has 4.22x higher relative variance",
    "interpretation": "Cumulative encoding has very low CV due to monotonic increase, potentially masking daily variation signal"
  },
  {
    "key_result": "Stationarity: Cumulative 1/3 stationary, Delta 3/3 stationary",
    "significance": "ADF test p < 0.05 indicates stationarity",
    "interpretation": "Non-stationary cumulative series may cause trend confounding in neural networks"
  }
]
```

**Recommendations:**
- Consider using delta encoding instead of cumulative to capture daily variation
- Apply 7-day rolling average to reduce noise while preserving signal
- Test model performance with different encodings in ablation study
- Consider log-transform for delta values to handle large spikes

---

### 1.1.2: Equipment-Personnel Redundancy Test

**Status:** completed
**Duration:** 0.25s

**Findings:**
```json
[
  {
    "key_result": "Mean Pearson correlation between equipment delta and personnel delta: 0.0300",
    "significance": "Highest correlation: drone (r=0.2891, p=8.48e-29)",
    "interpretation": "Positive correlations suggest equipment and personnel losses are temporally synchronized"
  },
  {
    "key_result": "Mean correlation reduction after controlling for time: -71.2%",
    "significance": "Partial correlation analysis with time as confound",
    "interpretation": "High reduction suggests time trend explains much of the correlation (redundancy may be spurious)"
  }
]
```

**Recommendations:**
- Consider removing highly correlated equipment types to reduce redundancy
- Use partial correlations to assess true predictive value beyond time trends
- Investigate whether equipment losses predict personnel losses at specific lags
- Consider PCA or similar dimensionality reduction for equipment features

---

### 1.1.3: Equipment Category Disaggregation

**Status:** completed
**Duration:** 0.17s

**Findings:**
```json
[
  {
    "key_result": "Highest personnel correlation: drones (r=0.2891)",
    "significance": "p-value = 8.48e-29",
    "interpretation": "This equipment category most closely tracks personnel casualties"
  },
  {
    "key_result": "Lowest personnel correlation: aircraft (r=-0.1827)",
    "significance": "p-value = 3.77e-12",
    "interpretation": "This equipment category may be less informative for casualty prediction"
  },
  {
    "key_result": "Optimal lag analysis: [{'category': 'drones', 'best_lag': -7, 'best_lag_corr': 0.31965553211714326, 'lag_interpretation': 'equipment leads'}, {'category': 'apcs', 'best_lag': 0, 'best_lag_corr': 0.2207397385368487, 'lag_interpretation': 'concurrent'}, {'category': 'tanks', 'best_lag': 14, 'best_lag_corr': -0.05876220037962495, 'lag_interpretation': 'equipment lags'}]",
    "significance": "Positive lag = equipment losses precede personnel casualties",
    "interpretation": "Lag structure indicates whether equipment provides leading signal"
  }
]
```

**Recommendations:**
- Focus model attention on drones - highest correlation with casualties
- Consider dropping or downweighting aircraft - minimal correlation
- Use lag structure in feature engineering for predictive modeling
- Investigate why some categories correlate more strongly (tactical significance)

---

### 1.1.4: Temporal Lag Analysis - Equipment

**Status:** completed
**Duration:** 0.19s

**Findings:**
```json
[
  {
    "key_result": "Leading equipment (precede casualties): ['tank', 'MRL', 'aircraft', 'drone', 'total_equipment']",
    "significance": "Average lead time: -21.6 days",
    "interpretation": "These equipment types may have predictive value for upcoming casualties"
  },
  {
    "key_result": "Lagging equipment (follow casualties): ['helicopter']",
    "significance": "Average lag time: 21.0 days",
    "interpretation": "These equipment losses may be consequence of casualty-causing events"
  },
  {
    "key_result": "Total equipment optimal lag: -27 days (r=0.3492)",
    "significance": "Concurrent correlation: 0.3205",
    "interpretation": "Overall equipment-personnel temporal relationship"
  }
]
```

**Recommendations:**
- Use equipment types with negative optimal lag as leading indicators
- Consider time-shifted features in model to capture predictive relationships
- Investigate causal mechanisms for leading vs lagging equipment types
- Total equipment shows optimal lag of -27 days - consider this in prediction horizon

---

### 1.2.1: VIIRS-Casualty Temporal Relationship

**Status:** completed
**Duration:** 0.24s

**Findings:**
```json
[
  {
    "key_result": "VIIRS-Casualty temporal relationship: LAGGING (VIIRS follows casualties)",
    "significance": "Optimal lag for radiance_mean: 10 days (r=0.0758)",
    "interpretation": "Leading indicator suggests VIIRS may have predictive value; lagging suggests confounding"
  },
  {
    "key_result": "Mean concurrent correlation - Raw: 0.0113, Delta: 0.0396",
    "significance": "Correlation increased by 0.0284 after differencing",
    "interpretation": "Lower delta correlation suggests trend confounding; higher suggests robust signal"
  }
]
```

**Recommendations:**
- VIIRS shows lagging (viirs follows casualties) - adjust model accordingly
- Consider both raw and differenced VIIRS features in model
- Investigate physical mechanism linking nightlights to conflict intensity
- Test VIIRS importance with and without trend removal

---

### 1.2.2: VIIRS Feature Decomposition

**Status:** completed
**Duration:** 0.24s

**Findings:**
```json
[
  {
    "key_result": "Most important VIIRS feature: radiance_max",
    "significance": "r=-0.0506, MI=0.0504",
    "interpretation": "This feature contributes most to casualty prediction signal"
  },
  {
    "key_result": "Found 7 highly correlated VIIRS feature pairs (|r| > 0.8)",
    "significance": "[{'feature_1': 'radiance_mean', 'feature_2': 'radiance_min', 'correlation': np.float64(0.8572229612411902)}, {'feature_1': 'radiance_mean', 'feature_2': 'radiance_p50', 'correlation': np.float64(0.994131775902834)}, {'feature_1': 'radiance_mean', 'feature_2': 'radiance_p90', 'correlation': np.float64(0.9906214322947711)}]",
    "interpretation": "High redundancy among VIIRS features - consider dimensionality reduction"
  }
]
```

**Recommendations:**
- Prioritize radiance_max in model feature selection
- Apply PCA to reduce VIIRS feature redundancy
- Investigate non-linear relationships for features with high MI but low r
- Consider creating composite VIIRS index from top features

---

### 1.2.3: Trend Confounding Test

**Status:** completed
**Duration:** 0.20s

**Findings:**
```json
[
  {
    "key_result": "MINIMAL trend confounding",
    "significance": "Mean correlation reduction after detrending: -8530.5%",
    "interpretation": "High reduction indicates VIIRS-casualty correlation is largely spurious (both driven by time)"
  },
  {
    "key_result": "radiance_mean: raw r=0.0079, detrended r=-0.0369",
    "significance": "Correlation reduction: -366.9%",
    "interpretation": "Large reduction = trend-driven; small reduction = robust signal"
  },
  {
    "key_result": "radiance_std: raw r=-0.0298, detrended r=-0.0346",
    "significance": "Correlation reduction: -16.2%",
    "interpretation": "Large reduction = trend-driven; small reduction = robust signal"
  },
  {
    "key_result": "pct_clear_sky: raw r=-0.0002, detrended r=0.0611",
    "significance": "Correlation reduction: -25208.4%",
    "interpretation": "Large reduction = trend-driven; small reduction = robust signal"
  }
]
```

**Recommendations:**
- Use first-differenced VIIRS features to remove trend confounding
- Consider seasonal decomposition for more sophisticated detrending
- Test model performance with and without detrended features
- If confounding is strong, VIIRS dominance may be an artifact - verify with ablation

---

### 1.2.4: Geographic VIIRS Decomposition

**Status:** completed
**Duration:** 0.17s

**Findings:**
```json
[
  {
    "key_result": "Found 6 geographic tiles: ['h19v03', 'h19v04', 'h20v03', 'h20v04', 'h21v03', 'h21v04']",
    "significance": "Regional breakdown is available",
    "interpretation": "Can analyze regional variation in nightlight patterns"
  },
  {
    "key_result": "Radiance variation across tiles: 1.35",
    "significance": "Range from 2.21 to 3.56",
    "interpretation": "Large variation suggests regional differences in conflict intensity or urbanization"
  },
  {
    "key_result": "Tile with highest casualty correlation: h20v04 (r=-0.0055)",
    "significance": "p-value: 8.63e-01",
    "interpretation": "This region may be most relevant for casualty prediction"
  }
]
```

**Recommendations:**
- Consider using tile-specific features for regional conflict modeling
- Weight tiles by their casualty correlation in ensemble
- Investigate why certain tiles show stronger correlation with casualties
- Create regional VIIRS composite index from most predictive tiles

---

### 1.3.1: Personnel-VIIRS Mediation Analysis

**Status:** completed
**Duration:** 0.14s

**Findings:**
```json
[
  {
    "key_result": "Total effect (VIIRS -> Casualties): 0.0220 (p=4.91e-01)",
    "significance": "Direct effect: 0.0197, Indirect (via Personnel): 0.0023",
    "interpretation": "Personnel mediates 10.6% of VIIRS effect on casualties"
  },
  {
    "key_result": "No significant mediation (Sobel z=0.26, p=7.94e-01)",
    "significance": "Personnel does not significantly mediate VIIRS-Casualty relationship",
    "interpretation": "VIIRS and personnel may have independent effects on casualties"
  }
]
```

**Recommendations:**
- Consider both VIIRS and Personnel as separate predictors (complementary information)
- If mediation is strong, Personnel may be redundant with VIIRS
- Test reverse mediation (Personnel -> VIIRS -> Casualties) for comparison
- Use causal inference methods for stronger claims about mediation

---

### 2.1.1: Representation Similarity Analysis

**Status:** completed
**Duration:** 7.65s

**Findings:**
```json
"\nRSA Analysis Results:\n====================\nFusion Quality: STRONG\n\nOverall Statistics:\n- Mean RSA correlation: 0.7017\n- Max RSA correlation: 0.9848\n- Min RSA correlation: 0.3407\n- Related pairs mean RSA: 0.8263\n\nInterpretation:\nSources show significant representational similarity, indicating effective cross-modal fusion.\n\nThreshold Reference:\n- Expected if fusing well: RSA > 0.3 for related sources\n- Expected if independent: RSA near zero\n"
```

**Recommendations:**
- Some sources have very high RSA - possible redundancy

---

### 2.1.2: Cross-Source Information Flow

**Status:** completed
**Duration:** 4.59s

**Findings:**
```json
"Attention Flow Analysis Results:\n===================================\n\nSource Importance Ranking:\n  1. deepstate: 0.5947 (+/- 0.0315)\n  2. equipment: 0.1278 (+/- 0.0147)\n  3. firms: 0.0851 (+/- 0.0127)\n  4. personnel: 0.0701 (+/- 0.0087)\n  5. source_5: 0.0635 (+/- 0.0072)\n  6. viina: 0.0588 (+/- 0.0091)\n\nMean Attention Entropy: 2.4250\n  -> High entropy indicates diffuse attention\n"
```

**Recommendations:**
- Source importance is highly imbalanced - some sources may be underutilized
- High attention sparsity in cross_attn_a_to_b_layer0_sparsity - consider reducing attention temperature
- High attention sparsity in cross_attn_a_to_b_layer1_sparsity - consider reducing attention temperature

---

### 2.1.4: Checkpoint Comparison

**Status:** completed
**Duration:** 12.61s

**Findings:**
```json
"Checkpoint Comparison Results:\n===================================\n\nEpochs analyzed: [10]\n\nEpoch 10:\n  mean_rsa: 0.7254\n  max_rsa: 0.9980\n  cross_attn_a_to_b_layer0_entropy: 2.4814\n  cross_attn_b_to_a_layer0_entropy: 2.2976\n  cross_attn_a_to_b_layer1_entropy: 2.4805\n  cross_attn_b_to_a_layer1_entropy: 2.3720\n  source_importance: {'equipment': 0.12368260324001312, 'personnel': 0.10156919062137604, 'deepstate': 0.5007690191268921, 'firms': 0.1425219029188156, 'viina': 0.06991428136825562, 'source_5': 0.061541687697172165}\n\nTrend Analysis:\n"
```

---

### 2.2.1: Leave-One-Out Ablation

**Status:** completed
**Duration:** 183.91s

**Findings:**
```json
"Ablation Analysis Results:\n===================================\n\nLeave-One-Out Analysis:\n  Positive delta = source is necessary (removal hurts)\n  Negative delta = source is harmful (removal helps)\n\n\nTask: casualty\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n\nTask: regime\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n\nTask: anomaly\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n"
```

---

### 2.2.2: Source Sufficiency Test

**Status:** completed
**Duration:** 188.76s

**Findings:**
```json
"Ablation Analysis Results:\n===================================\n\nLeave-One-Out Analysis:\n  Positive delta = source is necessary (removal hurts)\n  Negative delta = source is harmful (removal helps)\n\n\nTask: casualty\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n\nTask: regime\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n\nTask: anomaly\n  Baseline: N/A\n  Source importance (by performance delta):\n    equipment: delta = +0.0000\n    personnel: delta = +0.0000\n    deepstate: delta = +0.0000\n    firms: delta = +0.0000\n    viina: delta = +0.0000\n"
```

---

### 3.1.1: Truncated Context Inference

**Status:** failed
**Duration:** 3.97s

**Error:** 'ProbeRunnerConfig' object has no attribute 'output_dir'

---

### 3.1.2: Temporal Attention Patterns

**Status:** failed
**Duration:** 2.14s

**Error:** 'ProbeRunnerConfig' object has no attribute 'output_dir'

---

### 3.1.3: Predictive Horizon Analysis

**Status:** failed
**Duration:** 7.31s

**Error:** 'ProbeRunnerConfig' object has no attribute 'output_dir'

---

### 3.2.1: Transition Boundary Analysis

**Status:** completed
**Duration:** 0.12s

**Findings:**
```json
{
  "error": "index 23 is out of bounds for axis 1 with size 12",
  "probe": "TransitionDynamicsProbe"
}
```

**Recommendations:**
- Probe TransitionDynamicsProbe failed: index 23 is out of bounds for axis 1 with size 12.
- This may indicate insufficient data samples spanning transition periods.
- Consider increasing num_samples or checking dataset date coverage.

---

### 3.2.2: Latent Velocity Prediction

**Status:** failed
**Duration:** 1.90s

**Error:** 'ProbeRunnerConfig' object has no attribute 'output_dir'

---

### 4.1.1: Named Operation Clustering

**Status:** completed
**Duration:** 11.93s

**Findings:**
```json
{
  "silhouette_score": 0.0,
  "calinski_harabasz_score": 0.0,
  "davies_bouldin_score": Infinity,
  "variance_ratio": 0.0,
  "n_samples_per_cluster": {},
  "cluster_centroids": null,
  "tsne_embeddings": null,
  "operation_labels": null
}
```

---

### 4.1.2: Day-Type Decoding Probe

**Status:** completed
**Duration:** 11.26s

**Findings:**
```json
{
  "accuracy": 0.65625,
  "f1_macro": 0.5549441193771091,
  "f1_weighted": 0.5475752008451751,
  "cv_accuracy_mean": 0.6653853620656184,
  "cv_accuracy_std": 0.00966666111270684,
  "confusion_matrix": "[[42  0  0]\n [27  2 15]\n [ 0  2 40]]",
  "classification_report": "              precision    recall  f1-score   support\n\n      Type 0       0.61      1.00      0.76        42\n      Type 1       0.50      0.05      0.08        44\n      Type 2       0.73      0.95      0.82        42\n\n    accuracy                           0.66       128\n   macro avg       0.61      0.67      0.55       128\nweighted avg       0.61      0.66      0.55       128\n",
  "feature_importance": "[[ 0.3293717  -1.30021879  2.06780079]\n [-1.03410964  0.12721079 -0.71417558]\n [ 0.70473794  1.173008   -1.35362521]]",
  "class_names": [
    "Type 0",
    "Type 1",
    "Type 2"
  ]
}
```

---

### 4.1.3: Intensity Level Decoding

**Status:** completed
**Duration:** 11.32s

**Findings:**
```json
{
  "error": "Number of classes, 3, does not match size of target_names, 4. Try specifying the labels parameter",
  "probe": "IntensityProbe"
}
```

**Recommendations:**
- Probe failed: Number of classes, 3, does not match size of target_names, 4. Try specifying the labels parameter

---

### 4.1.4: Geographic Focus Decoding

**Status:** completed
**Duration:** 11.23s

**Findings:**
```json
{
  "error": "Number of classes, 3, does not match size of target_names, 4. Try specifying the labels parameter",
  "probe": "GeographicFocusProbe"
}
```

**Recommendations:**
- Probe failed: Number of classes, 3, does not match size of target_names, 4. Try specifying the labels parameter

---

### 4.2.1: Weekly Cycle Detection

**Status:** completed
**Duration:** 11.08s

**Findings:**
```json
{
  "weekly_cycle": {
    "test_name": "Weekly Cycle ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  },
  "seasonal_pattern": {
    "test_name": "Seasonal Pattern ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  }
}
```

---

### 4.2.2: Seasonal Pattern Detection

**Status:** completed
**Duration:** 11.19s

**Findings:**
```json
{
  "weekly_cycle": {
    "test_name": "Weekly Cycle ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  },
  "seasonal_pattern": {
    "test_name": "Seasonal Pattern ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  }
}
```

---

### 4.2.3: Event Anniversary Detection

**Status:** completed
**Duration:** 11.45s

**Findings:**
```json
{
  "weekly_cycle": {
    "test_name": "Weekly Cycle ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  },
  "seasonal_pattern": {
    "test_name": "Seasonal Pattern ANOVA",
    "statistic": 0.0,
    "p_value": 1.0,
    "is_significant": false,
    "effect_size": null,
    "group_means": null,
    "additional_info": null
  }
}
```

---

### 5.1.1: ISW-Latent Correlation

**Status:** completed
**Duration:** 11.39s

---

### 5.1.2: ISW Topic-Source Correlation

**Status:** completed
**Duration:** 11.89s

---

### 5.1.3: ISW Predictive Content Test

**Status:** completed
**Duration:** 11.44s

---

### 5.2.1: Event-Triggered Response Analysis

**Status:** completed
**Duration:** 11.17s

---

### 5.2.2: Narrative-Numerical Lag Analysis

**Status:** completed
**Duration:** 11.16s

---

### 5.2.3: Semantic Anomaly Detection

**Status:** completed
**Duration:** 11.07s

---

### 5.3.1: Semantic Perturbation Effects

**Status:** completed
**Duration:** 11.03s

---

### 5.3.2: Missing Semantic Interpolation

**Status:** completed
**Duration:** 11.24s

---

### 6.1.1: Source Zeroing Interventions

**Status:** completed
**Duration:** 8.52s

---

### 6.1.2: Source Shuffling Interventions

**Status:** completed
**Duration:** 7.93s

---

### 6.1.3: Source Mean Substitution

**Status:** completed
**Duration:** 1.75s

---

### 6.2.1: Integrated Gradients

**Status:** completed
**Duration:** 665.85s

---

### 6.2.2: Attention Knockout

**Status:** completed
**Duration:** 0.16s

---

### 7.1.1: Regional Signal Availability

**Status:** completed
**Duration:** 26.12s

**Findings:**
```json
{
  "deepstate": {
    "source": "DataSource.DEEPSTATE",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Aggregated front line metrics available')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Polygons can be filtered by oblast bounding box')",
      "sector": "GranularitySupport(granularity=<SpatialGranularity.SECTOR: 'sector'>, is_available=True, data_density=0.8, limiting_factors=[], notes='Front line polygons align well with tactical sectors')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Full coordinate data available for all features')"
    },
    "temporal_granularity": {
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Daily snapshots available since July 2024')",
      "12h": "GranularitySupport(granularity=<TemporalGranularity.TWELVE_HOUR: '12h'>, is_available=False, data_density=None, limiting_factors=['Single daily snapshot only', 'No intraday updates'], notes='Would require more frequent scraping')",
      "hourly": "GranularitySupport(granularity=<TemporalGranularity.HOURLY: 'hourly'>, is_available=False, data_density=None, limiting_factors=['Map updates ~1x daily', 'No sub-daily resolution'], notes='Not feasible without live API access')"
    },
    "native_resolution": "Daily snapshots with coordinate polygons",
    "total_records": 1660,
    "date_range": [
      "20240708",
      "20260118"
    ],
    "has_coordinates": true,
    "coordinate_precision": "Polygon vertices (high precision ~10m)",
    "recommended_use_level": "SECTOR-DAILY: Excellent for sector-level daily tracking"
  },
  "firms": {
    "source": "DataSource.FIRMS",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Aggregated fire counts available')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=True, data_density=0.9, limiting_factors=[], notes='Point data can be filtered by oblast')",
      "sector": "GranularitySupport(granularity=<SpatialGranularity.SECTOR: 'sector'>, is_available=True, data_density=0.7, limiting_factors=[], notes='Sector-level aggregation possible from coordinates')",
      "grid_10km": "GranularitySupport(granularity=<SpatialGranularity.GRID_10KM: 'grid_10km'>, is_available=True, data_density=0.5, limiting_factors=['Sparse coverage in non-combat areas'], notes='Grid cells may have zero observations')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Full lat/lon available for every detection')"
    },
    "temporal_granularity": {
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Date field available for all records')",
      "12h": "GranularitySupport(granularity=<TemporalGranularity.TWELVE_HOUR: '12h'>, is_available=True, data_density=0.8, limiting_factors=[], notes='Day/night classification (D/N) enables 12h resolution')",
      "6h": "GranularitySupport(granularity=<TemporalGranularity.SIX_HOUR: '6h'>, is_available=True, data_density=0.6, limiting_factors=['Satellite overpasses not evenly distributed'], notes='acq_time field provides ~4 hour precision')",
      "hourly": "GranularitySupport(granularity=<TemporalGranularity.HOURLY: 'hourly'>, is_available=False, data_density=None, limiting_factors=['Satellite revisit ~4x per day', 'Gaps between overpasses'], notes='acq_time gives hour but sparse temporal coverage')"
    },
    "native_resolution": "Individual fire detections with lat/lon",
    "total_records": 245456,
    "date_range": [
      "2022-02-24",
      "2025-09-30"
    ],
    "has_coordinates": true,
    "coordinate_precision": "Point coordinates (~4 decimal places, ~375m resolution)",
    "recommended_use_level": "SECTOR-12H: Good for sector-level sub-daily intensity"
  },
  "sentinel": {
    "source": "DataSource.SENTINEL",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Aggregated across all tiles')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=True, data_density=0.7, limiting_factors=[\"Tile boundaries don't align with oblasts\"], notes='6 tiles cover Ukraine, can approximate oblast coverage')",
      "sector": "GranularitySupport(granularity=<SpatialGranularity.SECTOR: 'sector'>, is_available=False, data_density=None, limiting_factors=['Current data is tile-aggregated', 'Would need raw scene downloads', 'Cloud cover limits usable observations'], notes='Requires scene-level processing for sector analysis')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=False, data_density=None, limiting_factors=['Only metadata currently available', 'Pixel-level analysis requires full scene download'], notes='Theoretically available at 10-60m resolution')"
    },
    "temporal_granularity": {
      "monthly": "GranularitySupport(granularity=<TemporalGranularity.MONTHLY: 'monthly'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Current aggregation level')",
      "weekly": "GranularitySupport(granularity=<TemporalGranularity.WEEKLY: 'weekly'>, is_available=True, data_density=0.8, limiting_factors=[], notes='Weekly available from raw dates')",
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=0.5, limiting_factors=['5-day revisit for Sentinel-2', 'Cloud cover gaps'], notes='~19 unique dates per month (every ~1.5 days)')",
      "hourly": "GranularitySupport(granularity=<TemporalGranularity.HOURLY: 'hourly'>, is_available=False, data_density=None, limiting_factors=['Polar orbit satellites', 'No hourly revisit capability'], notes='Not physically possible with current constellation')"
    },
    "native_resolution": "Monthly aggregated scene counts by collection",
    "total_records": 19771,
    "date_range": [
      "2022-05-01",
      "2024-12-31"
    ],
    "has_coordinates": true,
    "coordinate_precision": "Bounding box: [37.0, 47.5, 39.5, 49.5]",
    "recommended_use_level": "TILE-WEEKLY: Good for regional weekly trends"
  },
  "equipment": {
    "source": "DataSource.EQUIPMENT",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Official UA MOD reports at national level')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=False, data_density=None, limiting_factors=['No geographic attribution in official data', 'Would require Oryx photo geolocation'], notes='Oryx data has some unit annotations that could be geolocated')",
      "sector": "GranularitySupport(granularity=<SpatialGranularity.SECTOR: 'sector'>, is_available=False, data_density=None, limiting_factors=['No sector-level reporting'], notes='Infeasible with current data')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=False, data_density=None, limiting_factors=['No coordinate data'], notes='Would require manual geolocation of Oryx images')"
    },
    "temporal_granularity": {
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Daily reports from UA MOD')",
      "12h": "GranularitySupport(granularity=<TemporalGranularity.TWELVE_HOUR: '12h'>, is_available=False, data_density=None, limiting_factors=['Single daily report'], notes='Not available')",
      "hourly": "GranularitySupport(granularity=<TemporalGranularity.HOURLY: 'hourly'>, is_available=False, data_density=None, limiting_factors=['Single daily report'], notes='Not available')"
    },
    "native_resolution": "Daily cumulative equipment counts (national level)",
    "total_records": 1423,
    "date_range": [
      "2022-02-25",
      "2026-01-17"
    ],
    "has_coordinates": false,
    "coordinate_precision": "",
    "recommended_use_level": "NATIONAL-DAILY: Limited to national-level daily trends"
  },
  "personnel": {
    "source": "DataSource.PERSONNEL",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Official estimates at national level')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=False, data_density=None, limiting_factors=['No geographic breakdown available'], notes='Would require open-source investigation')",
      "sector": "GranularitySupport(granularity=<SpatialGranularity.SECTOR: 'sector'>, is_available=False, data_density=None, limiting_factors=['No sector-level data'], notes='Infeasible')"
    },
    "temporal_granularity": {
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Daily cumulative counts')",
      "12h": "GranularitySupport(granularity=<TemporalGranularity.TWELVE_HOUR: '12h'>, is_available=False, data_density=None, limiting_factors=['Single daily update'], notes='Not available')"
    },
    "native_resolution": "Daily cumulative personnel counts (national level)",
    "total_records": 1423,
    "date_range": [
      "2022-02-25",
      "2026-01-17"
    ],
    "has_coordinates": false,
    "coordinate_precision": "",
    "recommended_use_level": "NATIONAL-DAILY: Limited to national-level daily trends"
  },
  "ucdp": {
    "source": "DataSource.UCDP",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Aggregated event counts')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=True, data_density=0.8, limiting_factors=[], notes='Admin region data available (adm_1, adm_2)')",
      "raion": "GranularitySupport(granularity=<SpatialGranularity.RAION: 'raion'>, is_available=True, data_density=0.5, limiting_factors=['Some events have low precision'], notes='where_prec field indicates location precision')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=True, data_density=0.7, limiting_factors=['Precision varies (where_prec 1-7)'], notes='High precision for some, approximate for others')"
    },
    "temporal_granularity": {
      "monthly": "GranularitySupport(granularity=<TemporalGranularity.MONTHLY: 'monthly'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Standard aggregation level')",
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=0.8, limiting_factors=['Some multi-day events', 'Publication lag'], notes='date_start/date_end available')",
      "hourly": "GranularitySupport(granularity=<TemporalGranularity.HOURLY: 'hourly'>, is_available=False, data_density=None, limiting_factors=['No time-of-day data'], notes='Only date precision available')"
    },
    "native_resolution": "Individual conflict events with coordinates",
    "total_records": 33100,
    "date_range": [
      "2014-01-22",
      "2024-12-31"
    ],
    "has_coordinates": true,
    "coordinate_precision": "Event coordinates (precision varies)",
    "recommended_use_level": "OBLAST-DAILY: Good for oblast-level daily event analysis"
  },
  "viina": {
    "source": "DataSource.VIINA",
    "spatial_granularity": {
      "national": "GranularitySupport(granularity=<SpatialGranularity.NATIONAL: 'national'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Aggregated control percentages')",
      "oblast": "GranularitySupport(granularity=<SpatialGranularity.OBLAST: 'oblast'>, is_available=True, data_density=0.9, limiting_factors=[], notes='Locality data has oblast attribution')",
      "coordinate": "GranularitySupport(granularity=<SpatialGranularity.COORDINATE: 'coordinate'>, is_available=True, data_density=0.8, limiting_factors=[], notes='Locality coordinates available')"
    },
    "temporal_granularity": {
      "daily": "GranularitySupport(granularity=<TemporalGranularity.DAILY: 'daily'>, is_available=True, data_density=1.0, limiting_factors=[], notes='Daily control snapshots')",
      "12h": "GranularitySupport(granularity=<TemporalGranularity.TWELVE_HOUR: '12h'>, is_available=False, data_density=None, limiting_factors=['Single daily snapshot'], notes='Not available')"
    },
    "native_resolution": "Error: Error tokenizing data. C error: Expected 1 fields in line 42, saw 46\n",
    "total_records": 0,
    "date_range": [
      "",
      ""
    ],
    "has_coordinates": false,
    "coordinate_precision": "",
    "recommended_use_level": "OBLAST-DAILY: Good for oblast-level control tracking"
  }
}
```

---

### 7.1.2: Front-Line Sector Definition

**Status:** completed
**Duration:** 0.00s

**Findings:**
```json
{
  "sectors": {
    "kharkiv": "TacticalSector(name='Kharkiv Sector', sector_id='kharkiv', description='Northern Kharkiv oblast front including Vovchansk and border areas', bbox=(36.5, 49.5, 38.2, 50.4), polygon=<POLYGON ((36.5 50.4, 38.2 50.4, 38.2 49.5, 36.5 49.5, 36.5 50.4))>, key_locations=['Vovchansk', 'Kupiansk', 'Kharkiv city'], active_since='2022-02-24')",
    "luhansk_svatove_kreminna": "TacticalSector(name='Svatove-Kreminna Axis', sector_id='luhansk_svatove_kreminna', description='Luhansk oblast western front, Svatove-Kreminna line', bbox=(37.5, 48.8, 38.8, 49.8), polygon=<POLYGON ((37.5 49.8, 38.8 49.8, 38.8 48.8, 37.5 48.8, 37.5 49.8))>, key_locations=['Svatove', 'Kreminna', 'Starobilsk'], active_since='2022-09-01')",
    "donetsk_north": "TacticalSector(name='Bakhmut-Siversk Sector', sector_id='donetsk_north', description='Northern Donetsk oblast including Bakhmut salient and Siversk', bbox=(37.5, 48.3, 38.5, 49.0), polygon=<POLYGON ((37.5 49, 38.5 49, 38.5 48.3, 37.5 48.3, 37.5 49))>, key_locations=['Bakhmut', 'Siversk', 'Chasiv Yar', 'Soledar'], active_since='2022-05-01')",
    "donetsk_central": "TacticalSector(name='Avdiivka-Marinka Sector', sector_id='donetsk_central', description='Central Donetsk oblast around Avdiivka and western Donetsk city', bbox=(37.2, 47.7, 38.2, 48.3), polygon=<POLYGON ((37.2 48.3, 38.2 48.3, 38.2 47.7, 37.2 47.7, 37.2 48.3))>, key_locations=['Avdiivka', 'Marinka', 'Donetsk', 'Kurakhove'], active_since='2022-02-24')",
    "donetsk_south": "TacticalSector(name='Vuhledar Sector', sector_id='donetsk_south', description='Southern Donetsk oblast including Vuhledar and Velyka Novosilka', bbox=(36.5, 47.0, 37.8, 47.8), polygon=<POLYGON ((36.5 47.8, 37.8 47.8, 37.8 47, 36.5 47, 36.5 47.8))>, key_locations=['Vuhledar', 'Velyka Novosilka', 'Pavlivka'], active_since='2022-02-24')",
    "zaporizhzhia": "TacticalSector(name='Zaporizhzhia Sector', sector_id='zaporizhzhia', description='Zaporizhzhia oblast front including Orikhiv direction', bbox=(35.0, 46.8, 36.8, 48.0), polygon=<POLYGON ((35 48, 36.8 48, 36.8 46.8, 35 46.8, 35 48))>, key_locations=['Orikhiv', 'Tokmak', 'Robotyne', 'Melitopol'], active_since='2022-02-24')",
    "kherson": "TacticalSector(name='Kherson Sector', sector_id='kherson', description='Kherson oblast including Dnipro river line', bbox=(32.5, 46.0, 35.0, 47.5), polygon=<POLYGON ((32.5 47.5, 35 47.5, 35 46, 32.5 46, 32.5 47.5))>, key_locations=['Kherson', 'Nova Kakhovka', 'Kinburn Spit'], active_since='2022-02-24')",
    "kursk": "TacticalSector(name='Kursk Incursion Sector', sector_id='kursk', description='Kursk oblast (Russia) Ukrainian incursion zone', bbox=(34.5, 51.0, 36.5, 52.0), polygon=<POLYGON ((34.5 52, 36.5 52, 36.5 51, 34.5 51, 34.5 52))>, key_locations=['Sudzha', 'Korenevo', 'Lgov'], active_since='2024-08-06')"
  },
  "oblasts": {
    "donetsk": "TacticalSector(name='Donetsk', sector_id='oblast_donetsk', description='Donetsk oblast administrative boundary', bbox=(36.7, 46.8, 39.0, 49.3), polygon=<POLYGON ((39 46.8, 39 49.3, 36.7 49.3, 36.7 46.8, 39 46.8))>, key_locations=[], active_since='')",
    "luhansk": "TacticalSector(name='Luhansk', sector_id='oblast_luhansk', description='Luhansk oblast administrative boundary', bbox=(38.0, 48.0, 40.2, 50.1), polygon=<POLYGON ((40.2 48, 40.2 50.1, 38 50.1, 38 48, 40.2 48))>, key_locations=[], active_since='')",
    "kharkiv": "TacticalSector(name='Kharkiv', sector_id='oblast_kharkiv', description='Kharkiv oblast administrative boundary', bbox=(34.5, 48.4, 38.5, 50.4), polygon=<POLYGON ((38.5 48.4, 38.5 50.4, 34.5 50.4, 34.5 48.4, 38.5 48.4))>, key_locations=[], active_since='')",
    "zaporizhzhia": "TacticalSector(name='Zaporizhzhia', sector_id='oblast_zaporizhzhia', description='Zaporizhzhia oblast administrative boundary', bbox=(34.0, 46.5, 36.9, 48.2), polygon=<POLYGON ((36.9 46.5, 36.9 48.2, 34 48.2, 34 46.5, 36.9 46.5))>, key_locations=[], active_since='')",
    "kherson": "TacticalSector(name='Kherson', sector_id='oblast_kherson', description='Kherson oblast administrative boundary', bbox=(32.0, 45.8, 35.5, 47.7), polygon=<POLYGON ((35.5 45.8, 35.5 47.7, 32 47.7, 32 45.8, 35.5 45.8))>, key_locations=[], active_since='')",
    "dnipropetrovsk": "TacticalSector(name='Dnipropetrovsk', sector_id='oblast_dnipropetrovsk', description='Dnipropetrovsk oblast administrative boundary', bbox=(33.0, 47.8, 36.3, 49.5), polygon=<POLYGON ((36.3 47.8, 36.3 49.5, 33 49.5, 33 47.8, 36.3 47.8))>, key_locations=[], active_since='')",
    "mykolaiv": "TacticalSector(name='Mykolaiv', sector_id='oblast_mykolaiv', description='Mykolaiv oblast administrative boundary', bbox=(30.2, 46.2, 33.4, 48.0), polygon=<POLYGON ((33.4 46.2, 33.4 48, 30.2 48, 30.2 46.2, 33.4 46.2))>, key_locations=[], active_since='')",
    "sumy": "TacticalSector(name='Sumy', sector_id='oblast_sumy', description='Sumy oblast administrative boundary', bbox=(32.7, 50.0, 35.6, 52.4), polygon=<POLYGON ((35.6 50, 35.6 52.4, 32.7 52.4, 32.7 50, 35.6 50))>, key_locations=[], active_since='')",
    "chernihiv": "TacticalSector(name='Chernihiv', sector_id='oblast_chernihiv', description='Chernihiv oblast administrative boundary', bbox=(30.8, 50.6, 33.5, 52.4), polygon=<POLYGON ((33.5 50.6, 33.5 52.4, 30.8 52.4, 30.8 50.6, 33.5 50.6))>, key_locations=[], active_since='')",
    "kyiv_oblast": "TacticalSector(name='Kyiv_Oblast', sector_id='oblast_kyiv_oblast', description='Kyiv_Oblast oblast administrative boundary', bbox=(29.2, 49.4, 32.2, 51.6), polygon=<POLYGON ((32.2 49.4, 32.2 51.6, 29.2 51.6, 29.2 49.4, 32.2 49.4))>, key_locations=[], active_since='')",
    "crimea": "TacticalSector(name='Crimea', sector_id='oblast_crimea', description='Crimea oblast administrative boundary', bbox=(32.4, 44.4, 36.7, 46.2), polygon=<POLYGON ((36.7 44.4, 36.7 46.2, 32.4 46.2, 32.4 44.4, 36.7 44.4))>, key_locations=[], active_since='')"
  }
}
```

---

### 7.1.3: Sector Independence Test

**Status:** completed
**Duration:** 0.00s

**Findings:**
```json
{
  "probe_type": "SectorCorrelationProbe",
  "status": "initialized",
  "note": "SectorCorrelationProbe requires FIRMS data to compute correlations. Call compute_sector_correlations_firms() with a DataFrame.",
  "available_methods": [
    "compute_sector_correlations_firms",
    "generate_independence_report"
  ]
}
```

---

### 7.2.1: Unit Tracking Data Availability

**Status:** completed
**Duration:** 0.00s

**Findings:**
```json
{
  "unit_schema": "EntityStateVector(entity_type='military_unit', attributes={'strength': {'description': 'Estimated personnel strength', 'type': 'float', 'range': [0, 1], 'update_frequency': 'daily', 'source_availability': 'LOW', 'notes': 'Requires milblogger analysis, Telegram monitoring'}, 'equipment_count': {'description': 'Major equipment items remaining', 'type': 'int', 'update_frequency': 'weekly', 'source_availability': 'MEDIUM', 'notes': 'Oryx tracks some unit attributions'}, 'position': {'description': 'Estimated center of operations (lat, lon)', 'type': 'tuple[float, float]', 'update_frequency': 'daily', 'source_availability': 'MEDIUM', 'notes': 'DeepState unit markers, milblogger reports'}, 'days_in_contact': {'description': 'Days since unit entered active combat', 'type': 'int', 'update_frequency': 'daily', 'source_availability': 'LOW', 'notes': 'Requires tracking deployment history'}, 'loss_rate': {'description': '7-day rolling loss rate (equipment/day)', 'type': 'float', 'update_frequency': 'weekly', 'source_availability': 'LOW', 'notes': 'Oryx + milblogger correlation needed'}, 'unit_type': {'description': 'Unit classification (motorized_rifle, tank, VDV, etc)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented in open sources'}, 'echelon': {'description': 'Organizational level (brigade, regiment, battalion)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented'}, 'affiliation': {'description': 'Command structure (Western MD, Southern MD, etc)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented'}}, data_sources=['DeepState unit markers', 'Oryx (unit annotations on some losses)', 'UA General Staff briefings', 'Milblogger reports (Telegram)', 'ISW assessments'], feasibility_score=0.35, notes='Unit tracking is partially feasible. Static attributes (type, echelon) are well-documented. Dynamic attributes (strength, losses) require extensive OSINT integration and are often estimates.')",
  "infrastructure_schema": "EntityStateVector(entity_type='infrastructure', attributes={'type': {'description': 'Infrastructure type', 'type': 'categorical', 'values': ['airfield', 'depot', 'bridge', 'rail_junction', 'HQ', 'SAM_site'], 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-mapped in OSM and military sources'}, 'status': {'description': 'Operational status', 'type': 'categorical', 'values': ['operational', 'damaged', 'destroyed', 'unknown'], 'update_frequency': 'event-driven', 'source_availability': 'MEDIUM', 'notes': 'Strike reports, satellite imagery'}, 'last_activity': {'description': 'Last observed activity date', 'type': 'date', 'update_frequency': 'irregular', 'source_availability': 'LOW', 'notes': 'Requires regular satellite monitoring'}, 'strategic_value': {'description': 'Strategic importance score', 'type': 'float', 'range': [0, 1], 'update_frequency': 'monthly', 'source_availability': 'MEDIUM', 'notes': 'Can be derived from location, type, capacity'}, 'position': {'description': 'Location (lat, lon)', 'type': 'tuple[float, float]', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-mapped'}, 'capacity': {'description': 'Functional capacity (type-dependent metric)', 'type': 'float', 'update_frequency': 'event-driven', 'source_availability': 'LOW', 'notes': 'Difficult to assess remotely'}}, data_sources=['DeepState airfield markers', 'OpenStreetMap', 'Satellite imagery (Sentinel, commercial)', 'Strike reports', 'Social media geolocations'], feasibility_score=0.55, notes='Infrastructure tracking is more feasible than unit tracking. Static attributes are well-mapped. Status updates require satellite imagery analysis or strike report correlation.')",
  "data_source_audit": {
    "oryx": {
      "description": "Visual confirmation of equipment losses",
      "entity_coverage": "PARTIAL",
      "unit_annotations": true,
      "notes": "Some losses have unit attribution in description text",
      "update_frequency": "Daily",
      "feasibility": "MEDIUM - requires NLP extraction of unit names"
    },
    "ua_general_staff": {
      "description": "Official Ukrainian military briefings",
      "entity_coverage": "MINIMAL",
      "unit_annotations": false,
      "notes": "National-level aggregates only, no unit breakdown",
      "update_frequency": "Daily",
      "feasibility": "LOW - no entity-level data"
    },
    "deepstate": {
      "description": "Front line map with unit markers",
      "entity_coverage": "GOOD",
      "unit_annotations": true,
      "notes": "256+ unit markers with type/echelon information",
      "update_frequency": "Daily",
      "feasibility": "HIGH - structured unit marker data"
    },
    "milbloggers": {
      "description": "Telegram/social media reports",
      "entity_coverage": "VARIABLE",
      "unit_annotations": true,
      "notes": "Rich unit-level detail but unstructured",
      "update_frequency": "Real-time",
      "feasibility": "LOW - requires NLP + verification"
    },
    "isw": {
      "description": "ISW daily assessments",
      "entity_coverage": "PARTIAL",
      "unit_annotations": true,
      "notes": "Mentions specific units in context",
      "update_frequency": "Daily",
      "feasibility": "MEDIUM - semi-structured, NLP extractable"
    }
  }
}
```

---

### 7.2.2: Entity State Representation Design

**Status:** completed
**Duration:** 0.00s

**Findings:**
```json
{
  "unit_schema": "EntityStateVector(entity_type='military_unit', attributes={'strength': {'description': 'Estimated personnel strength', 'type': 'float', 'range': [0, 1], 'update_frequency': 'daily', 'source_availability': 'LOW', 'notes': 'Requires milblogger analysis, Telegram monitoring'}, 'equipment_count': {'description': 'Major equipment items remaining', 'type': 'int', 'update_frequency': 'weekly', 'source_availability': 'MEDIUM', 'notes': 'Oryx tracks some unit attributions'}, 'position': {'description': 'Estimated center of operations (lat, lon)', 'type': 'tuple[float, float]', 'update_frequency': 'daily', 'source_availability': 'MEDIUM', 'notes': 'DeepState unit markers, milblogger reports'}, 'days_in_contact': {'description': 'Days since unit entered active combat', 'type': 'int', 'update_frequency': 'daily', 'source_availability': 'LOW', 'notes': 'Requires tracking deployment history'}, 'loss_rate': {'description': '7-day rolling loss rate (equipment/day)', 'type': 'float', 'update_frequency': 'weekly', 'source_availability': 'LOW', 'notes': 'Oryx + milblogger correlation needed'}, 'unit_type': {'description': 'Unit classification (motorized_rifle, tank, VDV, etc)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented in open sources'}, 'echelon': {'description': 'Organizational level (brigade, regiment, battalion)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented'}, 'affiliation': {'description': 'Command structure (Western MD, Southern MD, etc)', 'type': 'categorical', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-documented'}}, data_sources=['DeepState unit markers', 'Oryx (unit annotations on some losses)', 'UA General Staff briefings', 'Milblogger reports (Telegram)', 'ISW assessments'], feasibility_score=0.35, notes='Unit tracking is partially feasible. Static attributes (type, echelon) are well-documented. Dynamic attributes (strength, losses) require extensive OSINT integration and are often estimates.')",
  "infrastructure_schema": "EntityStateVector(entity_type='infrastructure', attributes={'type': {'description': 'Infrastructure type', 'type': 'categorical', 'values': ['airfield', 'depot', 'bridge', 'rail_junction', 'HQ', 'SAM_site'], 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-mapped in OSM and military sources'}, 'status': {'description': 'Operational status', 'type': 'categorical', 'values': ['operational', 'damaged', 'destroyed', 'unknown'], 'update_frequency': 'event-driven', 'source_availability': 'MEDIUM', 'notes': 'Strike reports, satellite imagery'}, 'last_activity': {'description': 'Last observed activity date', 'type': 'date', 'update_frequency': 'irregular', 'source_availability': 'LOW', 'notes': 'Requires regular satellite monitoring'}, 'strategic_value': {'description': 'Strategic importance score', 'type': 'float', 'range': [0, 1], 'update_frequency': 'monthly', 'source_availability': 'MEDIUM', 'notes': 'Can be derived from location, type, capacity'}, 'position': {'description': 'Location (lat, lon)', 'type': 'tuple[float, float]', 'update_frequency': 'static', 'source_availability': 'HIGH', 'notes': 'Well-mapped'}, 'capacity': {'description': 'Functional capacity (type-dependent metric)', 'type': 'float', 'update_frequency': 'event-driven', 'source_availability': 'LOW', 'notes': 'Difficult to assess remotely'}}, data_sources=['DeepState airfield markers', 'OpenStreetMap', 'Satellite imagery (Sentinel, commercial)', 'Strike reports', 'Social media geolocations'], feasibility_score=0.55, notes='Infrastructure tracking is more feasible than unit tracking. Static attributes are well-mapped. Status updates require satellite imagery analysis or strike report correlation.')",
  "data_source_audit": {
    "oryx": {
      "description": "Visual confirmation of equipment losses",
      "entity_coverage": "PARTIAL",
      "unit_annotations": true,
      "notes": "Some losses have unit attribution in description text",
      "update_frequency": "Daily",
      "feasibility": "MEDIUM - requires NLP extraction of unit names"
    },
    "ua_general_staff": {
      "description": "Official Ukrainian military briefings",
      "entity_coverage": "MINIMAL",
      "unit_annotations": false,
      "notes": "National-level aggregates only, no unit breakdown",
      "update_frequency": "Daily",
      "feasibility": "LOW - no entity-level data"
    },
    "deepstate": {
      "description": "Front line map with unit markers",
      "entity_coverage": "GOOD",
      "unit_annotations": true,
      "notes": "256+ unit markers with type/echelon information",
      "update_frequency": "Daily",
      "feasibility": "HIGH - structured unit marker data"
    },
    "milbloggers": {
      "description": "Telegram/social media reports",
      "entity_coverage": "VARIABLE",
      "unit_annotations": true,
      "notes": "Rich unit-level detail but unstructured",
      "update_frequency": "Real-time",
      "feasibility": "LOW - requires NLP + verification"
    },
    "isw": {
      "description": "ISW daily assessments",
      "entity_coverage": "PARTIAL",
      "unit_annotations": true,
      "notes": "Mentions specific units in context",
      "update_frequency": "Daily",
      "feasibility": "MEDIUM - semi-structured, NLP extractable"
    }
  }
}
```

---

### 7.3.1: Temporal Resolution Analysis

**Status:** completed
**Duration:** 25.18s

**Findings:**
```json
{
  "temporal_resolution_analysis": {
    "12h": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density",
        "FIRMS day/night split enables 12h analysis"
      ]
    },
    "6h": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density"
      ]
    },
    "hourly": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density",
        "CRITICAL: Most sources lack hourly data",
        "Consider synthetic hourly interpolation with uncertainty"
      ]
    }
  },
  "spatial_resolution_analysis": {
    "oblast": {
      "feasibility": "HIGH",
      "recommendations": [
        "RECOMMENDED: Good coverage across most sources",
        "Use for: Regional trend analysis, operational planning"
      ]
    },
    "sector": {
      "feasibility": "LOW",
      "recommendations": [
        "CAUTION: Equipment/Personnel lack sector data"
      ]
    },
    "grid_10km": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "HIGH RESOLUTION: May have sparse coverage",
        "Consider aggregation strategies for sparse cells"
      ]
    },
    "coordinate": {
      "feasibility": "MEDIUM",
      "recommendations": [
        "POINT-LEVEL: Use for specific event analysis",
        "Not recommended for prediction at this granularity"
      ]
    }
  },
  "optimal_recommendation": {
    "recommended_temporal": "daily",
    "recommended_spatial": "oblast",
    "temporal_feasibility": "HIGH",
    "spatial_feasibility": "HIGH",
    "combined_recommendation": "SECTOR-DAILY predictions are recommended as the optimal balance of granularity and data availability. 12H resolution is achievable for FIRMS-based features.",
    "upgrade_path": [
      "Phase 1: Implement SECTOR-DAILY predictions",
      "Phase 2: Add 12H resolution for fire/thermal features",
      "Phase 3: Investigate entity-level tracking with DeepState unit data",
      "Phase 4: Consider grid-cell analysis for high-activity sectors"
    ]
  }
}
```

---

### 7.3.2: Spatial Resolution Analysis

**Status:** completed
**Duration:** 25.06s

**Findings:**
```json
{
  "temporal_resolution_analysis": {
    "12h": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density",
        "FIRMS day/night split enables 12h analysis"
      ]
    },
    "6h": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density"
      ]
    },
    "hourly": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "NOT RECOMMENDED: Insufficient data density",
        "CRITICAL: Most sources lack hourly data",
        "Consider synthetic hourly interpolation with uncertainty"
      ]
    }
  },
  "spatial_resolution_analysis": {
    "oblast": {
      "feasibility": "HIGH",
      "recommendations": [
        "RECOMMENDED: Good coverage across most sources",
        "Use for: Regional trend analysis, operational planning"
      ]
    },
    "sector": {
      "feasibility": "LOW",
      "recommendations": [
        "CAUTION: Equipment/Personnel lack sector data"
      ]
    },
    "grid_10km": {
      "feasibility": "NOT_FEASIBLE",
      "recommendations": [
        "HIGH RESOLUTION: May have sparse coverage",
        "Consider aggregation strategies for sparse cells"
      ]
    },
    "coordinate": {
      "feasibility": "MEDIUM",
      "recommendations": [
        "POINT-LEVEL: Use for specific event analysis",
        "Not recommended for prediction at this granularity"
      ]
    }
  },
  "optimal_recommendation": {
    "recommended_temporal": "daily",
    "recommended_spatial": "oblast",
    "temporal_feasibility": "HIGH",
    "spatial_feasibility": "HIGH",
    "combined_recommendation": "SECTOR-DAILY predictions are recommended as the optimal balance of granularity and data availability. 12H resolution is achievable for FIRMS-based features.",
    "upgrade_path": [
      "Phase 1: Implement SECTOR-DAILY predictions",
      "Phase 2: Add 12H resolution for fire/thermal features",
      "Phase 3: Investigate entity-level tracking with DeepState unit data",
      "Phase 4: Consider grid-cell analysis for high-activity sectors"
    ]
  }
}
```

---

