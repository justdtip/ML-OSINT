Current Implementation Overview
The Multi-Resolution HAN processes conflict monitoring data from 10 sources at their native temporal resolutions and produces multi-task predictions. Here's how it works:


┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-RESOLUTION HAN ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

INPUTS:
├── DAILY SOURCES (~1426 timesteps)          ├── MONTHLY SOURCES (~48 timesteps)
│   ├── equipment (11 features)              │   ├── sentinel (7 features)
│   ├── personnel (3 features)               │   ├── hdx_conflict (5 features)
│   ├── deepstate (5 features)               │   ├── hdx_food (10 features)
│   ├── firms (13 features)                  │   ├── hdx_rainfall (6 features)
│   └── viina (6 features)                   │   └── iom (7 features)
│                                            │
│   + Observation masks (True=observed)      │   + Observation masks
└────────────────────────────────────────────┴─────────────────────────────────

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DAILY SOURCE ENCODERS (5 parallel encoders)                         │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  For each daily source:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ raw_features [batch, 1426, n_features]                               │    │
│  │        ↓                                                             │    │
│  │ feature_projection: Linear → LayerNorm → GELU → Dropout              │    │
│  │        ↓ [batch, 1426, d_model=128]                                  │    │
│  │                                                                       │    │
│  │ CRITICAL: Handle missing values                                       │    │
│  │   - If timestep has NO observations → replace with no_observation_token│   │
│  │   - no_observation_token is LEARNED (not zero, not forward-fill)     │    │
│  │        ↓                                                              │    │
│  │ + observation_status_embedding (0=unobserved, 1=observed)            │    │
│  │ + sinusoidal_positional_encoding                                     │    │
│  │        ↓                                                              │    │
│  │ TransformerEncoder (4 layers, 8 heads)                               │    │
│  │   - src_key_padding_mask: unobserved positions CANNOT be keys/values │    │
│  │        ↓                                                              │    │
│  │ output_norm: LayerNorm                                               │    │
│  │        ↓ [batch, 1426, 128]                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: daily_encoded = {source_name: [batch, 1426, 128]} for 5 sources    │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DAILY CROSS-SOURCE FUSION                                           │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Learn dependencies between daily sources (equipment correlates with        │
│  personnel losses, FIRMS hotspots correlate with deepstate movements)       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Stack 5 sources: [batch, 1426, 5, 128]                               │    │
│  │        ↓                                                             │    │
│  │ Add source_type_embedding (5 learned embeddings)                     │    │
│  │        ↓                                                             │    │
│  │ Reshape: [batch*1426, 5, 128] (cross-attention per timestep)        │    │
│  │        ↓                                                             │    │
│  │ 2 layers of:                                                         │    │
│  │   - MultiheadAttention (sources attend to each other)               │    │
│  │   - LayerNorm + residual                                            │    │
│  │   - FFN (128→512→128) + LayerNorm + residual                        │    │
│  │        ↓                                                             │    │
│  │ Reshape: [batch, 1426, 5, 128]                                       │    │
│  │        ↓                                                             │    │
│  │ source_gate: Learn which sources matter at each timestep            │    │
│  │   - Linear(640→128) → LayerNorm → GELU → Linear(128→5) → Softmax    │    │
│  │        ↓                                                             │    │
│  │ Weighted combination + output_projection                            │    │
│  │        ↓ [batch, 1426, 128]                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: fused_daily [batch, 1426, 128]                                     │
│          combined_daily_mask [batch, 1426] (True if ANY source observed)    │
│          source_importance [batch, 1426, 5] (learned per-timestep weights)  │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: LEARNABLE MONTHLY AGGREGATION (daily → monthly)                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  CRITICAL: NOT simple averaging. Attention-based aggregation that learns    │
│  which daily events within a month are most important.                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ month_queries: Learnable [1, 60, 128] (60 possible months)          │    │
│  │ + month_position_embedding                                          │    │
│  │        ↓                                                             │    │
│  │ Cross-attention: months QUERY days                                   │    │
│  │   - Query: month_queries [batch, n_months, 128]                     │    │
│  │   - Key/Value: fused_daily [batch, 1426, 128]                       │    │
│  │   - attn_mask: Each month can ONLY attend to its own days           │    │
│  │     (enforced by month_boundaries [batch, n_months, 2])             │    │
│  │   - key_padding_mask: Unobserved days cannot be attended to         │    │
│  │        ↓                                                             │    │
│  │ attended = cross_attention(query, key, value)                       │    │
│  │        ↓                                                             │    │
│  │ Handle NaN (when all days in a month are masked): nan_to_num(0)     │    │
│  │        ↓                                                             │    │
│  │ residual + FFN (128→512→128)                                        │    │
│  │        ↓                                                             │    │
│  │ output_projection: Linear(128→128)                                  │    │
│  │        ↓ [batch, n_months, 128]                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: aggregated_daily [batch, n_months, 128]                            │
│          aggregated_daily_mask [batch, n_months]                            │
│          agg_attention [batch, nhead, n_months, 1426] (which days mattered) │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
┌─────────────────────────────────┐ ┌─────────────────────────────────────────┐
│ aggregated_daily               │ │ STEP 4: MONTHLY SOURCE ENCODERS          │
│ [batch, n_months, 128]         │ │ ─────────────────────────────────────── │
│                                │ │                                          │
│ (daily data, now at monthly    │ │ Uses MultiSourceMonthlyEncoder:          │
│  resolution via attention)     │ │   - 5 monthly sources                    │
│                                │ │   - Each has no_observation_token        │
│                                │ │   - 3-layer transformer per source       │
│                                │ │   - 2-layer cross-source fusion          │
│                                │ │                                          │
│                                │ │ Output: monthly_encoded                  │
│                                │ │         [batch, n_months, 128]           │
└───────────────┬────────────────┘ └───────────────┬───────────────────────────┘
                │                                   │
                └───────────────┬───────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: CROSS-RESOLUTION FUSION                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Bidirectional attention between aggregated daily and monthly data          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ aggregated_daily    ←────→    monthly_encoded                       │    │
│  │ [batch, n_mo, 128]           [batch, n_mo, 128]                     │    │
│  │                                                                      │    │
│  │ CrossResolutionFusion (2 layers):                                   │    │
│  │   - daily_to_monthly: daily queries monthly                         │    │
│  │   - monthly_to_daily: monthly queries daily                         │    │
│  │   - Gated combination of information                                │    │
│  │        ↓                                                             │    │
│  │ fused_monthly [batch, n_months, 128]                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: fused_monthly [batch, n_months, 128]                               │
│          cross_attention (attention weights for interpretability)           │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: TEMPORAL ENCODER                                                    │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Process the fused monthly sequence to capture long-range temporal patterns │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ fused_monthly [batch, n_months, 128]                                │    │
│  │        ↓                                                             │    │
│  │ + sinusoidal_positional_encoding                                    │    │
│  │        ↓                                                             │    │
│  │ TransformerEncoder (2 layers, 8 heads)                              │    │
│  │   - Can see full sequence (causal masking NOT applied currently)    │    │
│  │   - key_padding_mask for unobserved months                          │    │
│  │        ↓                                                             │    │
│  │ output_norm: LayerNorm                                              │    │
│  │        ↓ [batch, n_months, 128]                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: temporal_encoded [batch, n_months, 128]                            │
│                                                                              │
│  ⚠️ THIS IS THE LEARNED REPRESENTATION - should be exposed for downstream  │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: PREDICTION HEADS (Current - needs real targets for Option C)        │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  ┌───────────────────────┐  ┌───────────────────────┐                       │
│  │ CasualtyPredictionHead│  │RegimeClassificationHead│                       │
│  │ ───────────────────── │  │ ───────────────────── │                       │
│  │ temporal_encoded      │  │ temporal_encoded      │                       │
│  │    ↓                  │  │    ↓                  │                       │
│  │ Linear(128→256)       │  │ Linear(128→256)       │                       │
│  │ LayerNorm → GELU      │  │ LayerNorm → GELU      │                       │
│  │ Linear(256→128)       │  │ Linear(256→128)       │                       │
│  │ GELU → Dropout        │  │ GELU → Dropout        │                       │
│  │    ↓                  │  │    ↓                  │                       │
│  │ mean_head(128→3)      │  │ Linear(128→4)         │                       │
│  │ log_var_head(128→3)   │  │    ↓                  │                       │
│  │    ↓                  │  │ regime_logits         │                       │
│  │ casualty_pred [b,t,3] │  │ [batch, t, 4]         │                       │
│  │ casualty_var [b,t,3]  │  │                       │                       │
│  └───────────────────────┘  └───────────────────────┘                       │
│                                                                              │
│  ┌───────────────────────┐  ┌───────────────────────┐                       │
│  │ AnomalyDetectionHead  │  │ ForecastingHead       │                       │
│  │ ───────────────────── │  │ ───────────────────── │                       │
│  │ Similar MLP           │  │ temporal_encoded      │                       │
│  │    ↓                  │  │    ↓                  │                       │
│  │ anomaly_score         │  │ Linear(128→256)       │                       │
│  │ [batch, t, 1]         │  │ ... → Linear(256→35)  │                       │
│  │                       │  │    ↓                  │                       │
│  │                       │  │ forecast_pred         │                       │
│  │                       │  │ [batch, t, 35]        │                       │
│  │                       │  │ (sum of monthly       │                       │
│  │                       │  │  source features)     │                       │
│  └───────────────────────┘  └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: UNCERTAINTY ESTIMATION                                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Estimate prediction uncertainty based on observation density               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ observation_density = f(daily_masks, monthly_masks, month_boundaries)│    │
│  │   - Per month: what fraction of data sources were observed?         │    │
│  │   - Weighted: 0.6 * daily_density + 0.4 * monthly_density           │    │
│  │        ↓                                                             │    │
│  │ UncertaintyEstimator:                                               │    │
│  │   - MLP: Linear(128→64) → ReLU → Linear(64→1) → Softplus            │    │
│  │   - Scaled by (1 + (1 - density)): low density → high uncertainty   │    │
│  │        ↓                                                             │    │
│  │ uncertainty [batch, n_months, 1]                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

FINAL OUTPUTS (current):
{
    'casualty_pred': [batch, n_months, 3],      # deaths_best, deaths_low, deaths_high
    'casualty_var': [batch, n_months, 3],       # learned variance
    'regime_logits': [batch, n_months, 4],      # 4-class logits
    'anomaly_score': [batch, n_months, 1],      # anomaly probability
    'forecast_pred': [batch, n_months, 35],     # next-month features
    'uncertainty': [batch, n_months, 1],        # observation-based uncertainty
    'daily_attention': {...},                   # interpretability
    'monthly_attention': {...},                 # interpretability
    'cross_resolution_attention': {...},        # interpretability
    'source_importance': [batch, 1426, 5],      # which sources mattered
}
What Needs to Change for Option C
The Core Problem
Currently, the HAN is trained with synthetic/random targets (as we discovered):

casualty_target = torch.zeros_like(...) - always zeros
regime_targets = torch.randint(0, 4, ...) - random every batch!
anomaly_targets = torch.zeros(...) - always zeros
forecast_loss = variance * 0.01 - just regularization
Option C Requirements
For Option C, the HAN needs to:

Predict X(t+1) given X(t) - Given the current state, predict next month's state
Be evaluated against real targets - Compare predictions to actual observed data
Expose embeddings for downstream stages - temporal_encoded and fused_monthly should be primary outputs
Keep prediction heads for multi-task regularization - But with REAL targets
Key Questions for Enhancement
Before implementing, we should decide:

Forecasting Horizon: Predict t+1 month? Multiple horizons (t+1, t+3, t+6)?

What constitutes "state X"?

Daily source features (equipment, personnel, deepstate, firms, viina)?
Monthly source features (sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom)?
Both?
Learned embeddings from the model itself?
Causality: Should the temporal encoder use causal masking (can only see past, not future)?

Target Sources: Which features should we predict?

Self-supervised: Predict next month's raw features
All daily sources? All monthly sources?
Weighted by importance?
Additional Sophistication:

Probabilistic forecasting with calibrated uncertainty?
Multi-step rollout capability?
Contrastive learning for better representations?
Curriculum learning (easy → hard predictions)?
What are your thoughts on these design decisions before we proceed with the implementation?