# Multi-Resolution HAN Architecture Report

**Generated**: 2026-01-31
**Current Training Run**: `run_31-01-2026_06-51`
**Model Parameters**: 7.3M (reduced from 26.9M)

---

## Executive Summary

The Multi-Resolution Hierarchical Attention Network (HAN) is designed for conflict prediction using multi-source OSINT data. It processes data at native temporal resolutions (daily ~1400 timesteps, monthly ~48 timesteps) without fabricating missing values, using learned embeddings for unobserved data points.

### Current Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| nhead | 4 |
| Total parameters | 7,340,782 |
| Training samples | 644 |
| Params/sample ratio | 11,397 |
| Raion sources | 3 (geoconfirmed, deepstate, firms) |
| Raion PCA components | 8 |
| Correlation filter threshold | 0.85 |

### Key Challenges
1. **Overfitting**: Best val_loss at epoch 8 (11.57), then degrades
2. **Train/val gap**: ~3.2x difference (train ~3.6, val ~11.6)
3. **Limited samples**: 644 training samples for 7.3M parameters
4. **Sparse targets**: Many prediction targets have missing values

---

## 1. Architecture Overview

### 1.1 Main Model Class: MultiResolutionHAN

**Location**: `analysis/multi_resolution_han.py` (lines 2249+)

```python
class MultiResolutionHAN(nn.Module):
    """
    Processes conflict monitoring data at native temporal resolutions:
    - Daily sources: equipment, personnel, deepstate, firms, viina (~1426 timesteps)
    - Monthly sources: sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom (~48 timesteps)
    """
```

**Critical Design Principle**: The model NEVER fabricates or interpolates missing values. Instead:
- Each missing observation uses a **learned `no_observation_token`** embedding
- Explicit **observation masks** maintained throughout the forward pass
- Causal masking prevents future information leakage in all temporal encoders

---

## 2. Complete Architecture Pipeline

The model processes data through 9 sequential steps:

### STEP 1: Daily Source Encoders

```python
self.daily_encoders = nn.ModuleDict({
    name: DailySourceEncoder(
        source_config=config,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        causal=True,  # Autoregressive masking
    )
    for name in ['equipment', 'personnel', 'deepstate', 'firms', 'viina']
})
```

**DailySourceEncoder** (lines 187-449):
- Projects raw features to `d_model` dimension
- Learnable feature embeddings (one per feature)
- Observation status embeddings (unobserved vs observed)
- Sinusoidal positional encoding for temporal position
- **Multi-scale log compression** for extreme values (values > 100 compressed via log)
- Transformer encoder with **causal attention masking**
- Replaces missing values with learned `no_observation_token`

```python
# CRITICAL: Learned no_observation_token for missing values
self.no_observation_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

# Replace unobserved timesteps with learned token
obs_mask_expanded = timestep_observed.unsqueeze(-1)
hidden = torch.where(obs_mask_expanded, hidden, no_obs_expanded)

# Causal masking: prevents position i from attending to j > i
causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
```

### STEP 2: Daily Cross-Source Fusion

```python
self.daily_fusion = DailyCrossSourceFusion(
    d_model=128,
    nhead=8,
    num_layers=2,
    source_names=['equipment', 'personnel', 'deepstate', 'firms', 'viina']
)
```

**DailyCrossSourceFusion** (lines 737-929):
- **Source type embeddings**: Learnable embeddings per source
- **Multi-layer cross-attention**: Each source attends to all others
- **Temporal source gate**: Novel gating mechanism with:
  - Multi-scale convolutions (3, 7, 14-day kernels)
  - Temporal positional encoding (sinusoidal + day-of-week + month)
  - Self-attention for longer-range patterns
  - Temperature-scaled softmax

### STEP 2.5: Daily Temporal Encoder

```python
self.daily_temporal_encoder = DailyTemporalEncoder(
    d_model=128,
    nhead=4,
    window_size=31,  # ~1 month local window
    causal=True,
)
```

**DailyTemporalEncoder** (lines 936-1279):
- **Temporal positional encoding**: Combines sinusoidal + day-of-week + month embeddings
- **Multi-scale convolutions**: 3, 7, 14, 28-day kernels with GroupNorm
- **Temporal gate**: Controls blend between multi-scale features and original signal
- **Local windowed attention**: ~31 day window with causal masking

### STEP 3: Learnable Monthly Aggregation

```python
self.monthly_aggregation = EnhancedLearnableMonthlyAggregation(
    d_model=128,
    nhead=8,
    max_months=60,
    use_month_constraints=True,
)
```

**NOT simple averaging** - uses cross-attention to aggregate:
```python
# Cross-attention: month queries attend to days within month boundaries
queries = self.month_queries[:, :n_months, :]  # Learnable month representations
attended, attention_weights = self.cross_attention(
    query=queries_normed,
    key=daily_normed,
    value=daily_hidden,
    key_padding_mask=key_padding_mask,
    attn_mask=attn_mask,  # Constrains to month boundaries
)
```

### STEP 4: Monthly Source Encoders

```python
self.monthly_encoder = MultiSourceMonthlyEncoder(
    source_configs={
        'sentinel': MonthlySourceConfig(name='sentinel', n_features=43),
        'hdx_conflict': MonthlySourceConfig(name='hdx_conflict', n_features=18),
        # ... other monthly sources
    },
    d_model=128,
    nhead=8,
    num_encoder_layers=3,
)
```

### STEP 5: Cross-Resolution Fusion

```python
self.cross_resolution_fusion = CrossResolutionFusion(
    daily_dim=128,
    monthly_dim=128,
    hidden_dim=128,
    num_layers=2,
    num_heads=8,
    use_gating=True,
)
```

**Bidirectional cross-attention**:
- Aggregated daily attends to monthly (learns from coarser context)
- Monthly attends to aggregated daily (learns from finer details)
- Gating mechanism controls information blend

### STEP 6: Temporal Encoder

```python
self.temporal_encoder = TemporalEncoder(
    d_model=128,
    nhead=8,
    num_layers=2,
    causal=True,
    max_len=60,
)
```

### STEPS 7-9: Prediction Heads & Uncertainty

**Output Heads**:

1. **CasualtyPredictionHead**: Outputs [deaths_best, deaths_low, deaths_high] with variance
2. **RegimeClassificationHead**: 4-class [low_intensity, medium, high, major_offensive]
3. **AnomalyDetectionHead**: Binary anomaly scores (0-1)
4. **ForecastingHead**: Predicts next month's features
5. **DailyForecastingHead**: Predicts next 7 days

**UncertaintyEstimator**:
```python
class UncertaintyEstimator(nn.Module):
    """Scale uncertainty inversely with observation density"""
    density_scale = 1.0 + (1.0 - observation_density)
```

---

## 3. Spatial/Geographic Components

### Geographic Source Encoder

For raion-level (district) sources like `geoconfirmed_raion`:

```python
class GeographicSourceEncoder(nn.Module):
    """
    Encodes raion-level features with spatial attention.

    Input: [batch, seq_len, n_raions * n_features_per_raion]
    Output: [batch, seq_len, d_model]
    """
```

**Key Components**:
- **RaionEncoder**: Per-raion feature encoder with temporal attention
- **CrossRaionAttention**: Geographic-aware cross-raion attention
- **GeographicAdjacency**: Distance-based soft adjacency `exp(-distance/scale)`

---

## 4. Attention Mechanisms Summary

| Component | Type | Key Features |
|-----------|------|--------------|
| **Daily Encoders** | Self-attention | Causal masking, feature embeddings, no_observation_tokens |
| **Cross-Source Fusion** | Cross-attention | Multi-scale temporal convolutions, temporal gating |
| **Daily Temporal** | Multi-scale conv + windowed attention | 4 kernel sizes, temporal positional encoding |
| **Monthly Aggregation** | Cross-attention | Learnable month queries, month-boundary constraints |
| **Cross-Resolution Fusion** | Bidirectional cross-attention | Daily<->Monthly, gating mechanism |
| **Temporal Encoder** | Self-attention | Causal masking, full transformer stack |
| **Cross-Raion (Spatial)** | Multi-head attention | Geographic adjacency prior |

---

## 5. Data Pipeline

### 5.1 MultiResolutionDataset

**Location**: `analysis/multi_resolution_data.py`

**Initialization Pipeline**:
```python
def __init__(self, config, split='train', ...):
    # 1. Load all data sources (daily + monthly)
    self._load_all_sources()

    # 2. Compute temporal alignment (daily <-> monthly mapping)
    self._compute_alignment()

    # 3. CORRELATION FILTER (removes redundant raion features)
    # MUST run BEFORE normalization
    if self.config.use_correlation_filter:
        self.correlation_filter_info = self._fit_correlation_filter()
        self._apply_correlation_filter()

    # 4. NORMALIZATION (after correlation filter)
    self.norm_stats = self._compute_normalization_stats()
    self._apply_normalization()

    # 5. RAION PCA (reduces per-raion dimensionality)
    if self.config.use_raion_pca:
        self.raion_pca_transformer = self._fit_raion_pca()
        self._apply_raion_pca()

    # 6. Convert to tensors
    self._convert_to_tensors()
```

### 5.2 Correlation Filter

**Purpose**: Remove highly correlated feature pairs (|r| > 0.85)

```python
def _fit_correlation_filter(self) -> Dict[str, Any]:
    """
    For each raion source, identifies highly correlated feature pairs.
    Keeps higher-variance feature from each pair.
    """
    # Compute correlation matrix on first raion (same features across raions)
    corr = raion_df.corr()

    # Greedy selection: drop lower-variance feature from correlated pairs
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr.iloc[i, j]) > threshold:
                if variances[feature_names[i]] < variances[feature_names[j]]:
                    features_to_drop.add(feature_names[i])
                else:
                    features_to_drop.add(feature_names[j])
```

**Example Impact**:
- `geoconfirmed_raion`: 50 -> 42 features (dropped 8)
- `firms_expanded_raion`: 35 -> 20 features (dropped 15)
- `deepstate_raion`: 48 -> 45 features (dropped 3)

### 5.3 Raion PCA

**Purpose**: Reduce per-raion feature dimensionality while preserving spatial structure.

```python
def _fit_raion_pca(self) -> Dict[str, Any]:
    """
    Fit SINGLE PCA across all raions jointly.
    Features have same meaning across raions.

    Example: 50 features × 263 raions -> 8 components × 263 raions
    """
    # Reshape: [n_days, 263*50] -> [n_days*263, 50]
    values_2d = values_3d.reshape(n_days * n_raions, n_features)

    # Fit PCA
    pca = PCA(n_components=8)
    pca.fit(values_2d)

    # Transform and reshape back
    pca_values = pca.transform(values_2d).reshape(n_days, n_raions * n_components)
```

### 5.4 Sample Construction

Each sample includes:
- **Daily sequence**: Last 365 days ending at anchor day
- **Monthly sequence**: Last 12 months ending at anchor month
- **Forecast targets**: Next month + next 7 days
- **Observation masks**: Explicit True/False per feature per timestep
- **Month boundaries**: Where each month starts/ends within daily sequence
- **Raion masks**: Per-raion observation masks

---

## 6. Loss Functions

### 6.1 MultiTaskLoss (Uncertainty-Weighted)

```python
class MultiTaskLoss(nn.Module):
    """
    Kendall et al. formulation:
        L = 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i

    - High loss -> increase log_var (decrease weight)
    - Low loss -> decrease log_var (increase weight)
    """
    def forward(self, losses):
        for task_name in self.task_names:
            log_var = self.log_vars[task_name].clamp(-2, 2)
            precision = torch.exp(-log_var)  # Task weight
            weighted_loss = 0.5 * precision * task_loss + 0.5 * log_var
            total_loss += weighted_loss
```

### 6.2 Individual Task Losses

**Regime Classification**:
```python
# Cross-entropy with anti-collapse measures:
# - Class-balanced weights
# - Label smoothing (10%)
# - Entropy regularization
# - Prediction diversity penalty
# - Loss floor (0.05)
regime_loss = F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=0.1)
regime_loss += entropy_penalty * 0.3 + diversity_penalty * 0.3
regime_loss = torch.maximum(regime_loss, torch.tensor(0.05))
```

**Casualty Prediction (ZINB)**:
```python
def zinb_nll_loss(y_true, pi, mu, theta):
    """
    Zero-Inflated Negative Binomial NLL.

    P(Y=0) = pi + (1-pi) * NB(0|mu, theta)
    P(Y=k) = (1-pi) * NB(k|mu, theta)  for k > 0

    Handles overdispersed count data with excess zeros.
    """
```

**Anomaly Detection**:
```python
# MSE on VIIRS anomaly scores + variance penalty
anomaly_loss = F.mse_loss(pred, targets)
variance_penalty = F.relu(0.5 - pred_var / target_var)
losses['anomaly'] = anomaly_loss + variance_penalty * 0.1
```

**Forecast**:
```python
# MSE on next-month features
# Uses precomputed monthly aggregations from daily sources
forecast_loss = F.mse_loss(forecast_pred[valid_mask], forecast_targets[valid_mask])
```

---

## 7. Training Configuration

```python
@dataclass
class TrainerConfig:
    # Model
    d_model: int = 64
    nhead: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    accumulation_steps: int = 4  # Effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    patience: int = 30

    # Data
    daily_seq_len: int = 365
    monthly_seq_len: int = 12
    prediction_horizon: int = 1
```

---

## 8. Forward Pass Flow

```
INPUT: daily_features, daily_masks, monthly_features, monthly_masks, month_boundaries

STEP 1: Daily source encoders (separate encoder per source)
        -> DailySourceEncoder for equipment, personnel, deepstate, firms, viina

STEP 2: Cross-source fusion with temporal gating
        -> DailyCrossSourceFusion

STEP 2.5: Daily temporal encoding (multi-scale convolutions + windowed attention)
          -> DailyTemporalEncoder

STEP 3: Learnable monthly aggregation (cross-attention with month constraints)
        -> EnhancedLearnableMonthlyAggregation

STEP 4: Monthly source encoders
        -> MultiSourceMonthlyEncoder for sentinel, hdx_*, iom

STEP 5: Cross-resolution bidirectional fusion
        -> CrossResolutionFusion (daily <-> monthly)

STEP 6: Temporal encoding at monthly resolution
        -> TemporalEncoder with causal masking

STEP 7: Prediction heads
        -> CasualtyPredictionHead: [deaths_best, deaths_low, deaths_high]
        -> RegimeClassificationHead: 4-class
        -> AnomalyDetectionHead: binary scores
        -> ForecastingHead: next month
        -> DailyForecastingHead: 7-day forecast

STEP 8: Attention weights (for interpretability)

STEP 9: Uncertainty estimation

OUTPUT: {
  'temporal_output': [batch, seq, d_model],
  'casualty_pred': [batch, seq, 3],
  'casualty_var': [batch, seq, 3],
  'regime_logits': [batch, seq, 4],
  'anomaly_score': [batch, seq, 1],
  'forecast_pred': [batch, seq, monthly_features],
  'daily_forecast_pred': [batch, 7, daily_features],
  'uncertainty': [batch, seq, 1],
  'attention_weights': {...},
}
```

---

## 9. Current Training Results

### Run: 7.3M Parameters (3 raion sources + PCA)

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 0 | 3.74 | 11.61 | Initial |
| 8 | 3.68 | **11.57** | Best |
| 10 | 3.66 | 11.61 | LR peak |
| 20 | 3.59 | 11.78 | Overfitting |
| 30 | 3.55 | 11.93 | Continued degradation |

### Comparison: 26.9M Parameters (6 raion sources, no PCA)

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 1 | 3.72 | 11.66 | Best early |
| 10 | 3.66 | 11.73 | Peak overfitting |
| 28 | 3.59 | **11.61** | Recovered with cosine LR |

### Key Observations

1. **Smaller model achieves lower best val_loss** (11.57 vs 11.61)
2. **But overfits more severely** (11.93 vs 11.73 at worst)
3. **Cosine LR helps larger model recover** but not smaller model
4. **Train/val gap remains large** (~3.2x) regardless of model size

---

## 10. Ultimate Goals

1. **Predict conflict intensity** at multiple temporal scales (daily, monthly)
2. **Forecast regime transitions** (escalation, de-escalation)
3. **Detect anomalies** indicating significant events
4. **Provide uncertainty estimates** that scale with data sparsity
5. **Maintain interpretability** via attention weights
6. **Generalize to unseen time periods** (the current challenge)

---

## 11. Next Steps to Consider

1. **Data augmentation**: Generate synthetic training samples
2. **Stronger regularization**: Increase dropout, weight decay
3. **Simpler architecture**: Remove some attention layers
4. **Feature engineering**: Hand-crafted features instead of learned embeddings
5. **Transfer learning**: Pre-train on related tasks
6. **Ensemble methods**: Combine multiple smaller models

---

## 12. Training Algorithm Deep Dive

### 12.1 Learning Rate Schedule

**Configuration**: Warmup + Cosine Annealing
```python
warmup_epochs: int = 10
warmup_start_lr = learning_rate * 0.01  # Start at 1% of base LR
min_lr = 1e-6  # Minimum LR after decay
```

**Issue Identified**: Warmup starts too low (0.01x base LR = 1e-6 for lr=1e-4). First 10 epochs have negligible gradient steps.

### 12.2 Gradient Accumulation Issues

**Configuration**:
```python
batch_size: 8
accumulation_steps: 4  # Effective batch = 32
```

**Critical Issues Found**:
1. Loss scaling applied BEFORE backward, but gradients not averaged after accumulation
2. `total_loss` (unscaled) passed to accumulator, but backward used scaled loss
3. Unscaling happens BEFORE gradient clipping - potentially incorrect order with AMP

### 12.3 Early Stopping Mismatch

**Config defines sophisticated early stopping**:
```python
early_stopping_strategy: 'smoothed'
early_stopping_min_epochs: 50
early_stopping_min_delta: 1e-4
early_stopping_smoothing: 0.9
```

**But training loop uses simple counter**:
```python
if val_metrics['total'] < self.best_val_loss:
    self.patience_counter = 0
else:
    self.patience_counter += 1
```

No min_delta, no smoothing, no min_epochs enforcement.

### 12.4 Multi-Task Loss Weighting

**Kendall Uncertainty Weighting**:
```python
L = 0.5 * exp(-log_var) * L_i + 0.5 * log_var
```

**Learned Task Weights at Best Checkpoint (Epoch 8)**:
| Task | log_var | Weight | Interpretation |
|------|---------|--------|----------------|
| regime | 1.02 | 0.360 | Highest - most reliable |
| anomaly | 1.37 | 0.254 | Medium |
| casualty | 1.37 | 0.254 | Medium |
| transition | 1.37 | 0.254 | Medium |
| daily_forecast | 1.40 | 0.247 | Medium |
| forecast | 1.87 | **0.154** | Lowest - model knows it's unreliable |

**Critical Issue**: The `0.5 * log_var` term can produce negative losses when log_var < 0, breaking optimization semantics.

---

## 13. Loss Analysis & Failure Modes

### 13.1 Per-Task Loss Trajectory

From training summary at best epoch (8):

| Task | Train Loss | Val Loss | Ratio | Issue |
|------|------------|----------|-------|-------|
| total | 3.68 | 11.57 | 3.1x | Expected gap |
| regime | 0.069 | 0.068 | 1.0x | Healthy |
| forecast | 0.74 | **103.4** | **140x** | **MASSIVE OVERFITTING** |
| casualty | 4.7e-6 | 5.6e-7 | 0.1x | Collapsed to regularization |
| anomaly | 2.6e-5 | 8.9e-8 | ~0 | Collapsed to regularization |
| transition | 2.6e-9 | 8.9e-12 | ~0 | Collapsed to regularization |

### 13.2 Critical Finding: Forecast Task Dominance

The forecast task has:
- Training loss: 0.74 → 0.35 (decreasing normally)
- Validation loss: 103 → 105 (INCREASING!)
- Ratio: 140-300x

**This means**: The model is memorizing training forecast patterns but completely failing to generalize. The forecast head (4.4M params) is 60% of total model parameters and is overfitting catastrophically.

### 13.3 Task Collapse Pattern

Three tasks (casualty, anomaly, transition) have collapsed to near-zero loss:
- **Cause**: Missing real targets → falls back to regularization-only loss
- **Effect**: Model outputs ~0 for these tasks, satisfying regularization but not learning
- **Detection**: Losses < 1e-5 indicate collapse

**Fallback Behavior When Targets Missing**:
```python
# Casualty fallback - regularizes to constant output
pi_reg = (sigmoid(pred[:, 0]) - 0.5).pow(2).mean()
log_mu_reg = pred[:, 1].pow(2).mean() * 0.01
losses['casualty'] = (pi_reg + log_mu_reg + theta_reg) * 0.01

# Anomaly fallback - regularizes to zero
losses['anomaly'] = anomaly_score.pow(2).mean() * 0.01
```

### 13.4 Anti-Collapse Mechanisms (Inadequate)

Current mechanisms only **detect** collapse after it happens:
```python
collapse_threshold = 0.001
if task in results and results[task] < collapse_threshold:
    warnings.warn(f"COLLAPSE DETECTED...")  # Too late!
```

**Missing**: Proactive prevention (minimum loss floors, entropy regularization during loss computation).

---

## 14. Model Component Parameter Distribution

| Component | Parameters | % of Total | Purpose |
|-----------|------------|------------|---------|
| daily_encoders | 18.1M | 75.7% | Encode daily sources (includes raion spatial encoders) |
| daily_forecast_head | 4.4M | 18.3% | 7-day forecast prediction |
| monthly_encoder | 619K | 2.6% | Encode monthly sources |
| daily_fusion | 196K | 0.8% | Cross-source fusion |
| daily_temporal_encoder | 193K | 0.8% | Temporal pattern learning |
| cross_resolution_fusion | 188K | 0.8% | Daily↔Monthly fusion |
| temporal_encoder | 108K | 0.5% | Final temporal processing |
| prediction heads | ~70K | 0.3% | Casualty, regime, anomaly |

**Key Insight**: 94% of parameters are in encoders/forecast, only 6% in core fusion and prediction.

---

## 15. Identified Failure Modes

### 15.1 Forecast Task Overfitting (CRITICAL)

**Symptoms**:
- Val forecast loss 140-300x higher than train
- Forecast task weight drops to 0.15 (model learns it's unreliable)
- Daily forecast head has 4.4M params for 7-day prediction

**Root Cause**:
- Forecasting head predicts raw daily features (high-dimensional target)
- 4.4M params memorizes training patterns exactly
- No smoothing/regularization on forecast output

### 15.2 Target Data Missing (HIGH)

**Symptoms**:
- Casualty, anomaly, transition losses near zero
- These tasks using regularization-only fallback

**Root Cause**:
- `TargetLoader.load()` fails silently (warning only)
- Training continues with synthetic targets
- Model learns to output constants that minimize regularization

### 15.3 Gradient Flow Imbalance (MEDIUM)

**Symptoms**:
- Regime loss trains well (0.07 → 0.01)
- But validation regime loss is flat (0.068 → 0.068)

**Root Cause**:
- Large forecast loss gradients dominate
- Regime head gets attenuated gradients
- Effective learning rate for regime is much lower

### 15.4 Negative Loss Possibility (MEDIUM)

**The Kendall formulation allows**:
```python
weighted_loss = 0.5 * exp(-log_var) * task_loss + 0.5 * log_var
# If log_var = -2 (min clamp): 0.5 * (-2) = -1.0
# Net loss can be negative if task_loss is small
```

**Impact**: Creates optimization pathologies, saddle points.

---

## 16. Recommendations

### Immediate (address overfitting)

1. **Reduce daily_forecast_head complexity**: 4.4M → 500K params
2. **Add forecast output regularization**: L2 on predictions, temporal smoothness
3. **Cap forecast loss contribution**: `min(forecast_loss, 10.0)`

### Short-term (fix task collapse)

4. **Add target availability logging**: Track which targets are real vs synthetic
5. **Implement loss floors**: `loss = max(computed_loss, 0.1)` per task
6. **Remove tasks with missing targets**: If no real targets, exclude from loss

### Medium-term (architecture)

7. **Rebalance parameters**: Move params from encoders to fusion layers
8. **Implement SmartEarlyStopping**: Use the defined class, not simple counter
9. **Fix gradient accumulation**: Average gradients, not just scale loss

### Long-term (data)

10. **Acquire real target data**: Phase labels, casualty counts, VIIRS
11. **Increase training samples**: Data augmentation, synthetic scenarios
12. **Validate temporal split**: Ensure no data leakage via target loader

---

## 17. Training Run Comparison

| Metric | Run 1 (26.9M) | Run 2 (7.3M) | Notes |
|--------|---------------|--------------|-------|
| Best val_loss | 11.61 | **11.57** | Smaller is better |
| Best epoch | 28 | 8 | Smaller peaked earlier |
| Final val_loss | 11.61 | 11.94 | Smaller degraded more |
| Forecast train/val ratio | ~100x | ~140x | Both have forecast issue |
| Recovery with cosine LR | Yes | No | Larger recovered, smaller didn't |

**Conclusion**: The 7.3M model achieves better peak performance but overfits more severely and doesn't recover. The forecast task dominates both models.

---

## 18. Synthesized Solution: Training Improvements

Three external proposals were analyzed and synthesized into an implementable solution:

| Proposal | Core Approach | Key Innovation |
|----------|---------------|----------------|
| **kimik2 (AFNP-STGS)** | Sparse MoE + HyperNetwork | 85% parameter sparsity via routing |
| **gpt52 (SD-MRPF)** | Latent predictive coding | Predict future latents, not raw features |
| **gemini (HGS-CP)** | Hierarchical gradient surgery | Project noisy gradients onto stable anchor |

### 18.1 Root Cause Analysis

The forecast head memorization is the dominant failure mode:
- **DailyForecastingHead**: 4.4M params predicting ~165 features × 7 days
- **Training samples**: 644
- **Result**: Lookup table behavior (train loss ↓, val loss ↑)

### 18.2 Synthesized Solution

Implemented in `analysis/training_improvements.py` and `analysis/training_improvements_integration.py`:

#### A. LatentStatePredictor (from gpt52)
Replaces raw-feature forecasting with latent space prediction:
```python
# OLD: 4.4M params predicting high-dimensional features
self.daily_forecast_head = DailyForecastingHead(d_model=128, output_dim=165, horizon=7)

# NEW: ~65K params predicting latent states
self.daily_forecast_head = LatentStatePredictor(d_model=128, horizon=7)
```
**Impact**: Reduces forecast head from 4.4M → ~65K params (98% reduction)

#### B. SoftplusKendallLoss (from gpt52)
Fixes negative loss pathology in Kendall uncertainty weighting:
```python
# OLD: Can produce negative losses when log_var < 0
weighted_loss = 0.5 * exp(-log_var) * L_i + 0.5 * log_var

# NEW: Always positive with softplus
scale = softplus(raw_scale)  # Ensures scale >= 0
weighted_loss = exp(-scale) * L_i + scale
```
**Impact**: Eliminates negative loss values that confused optimization

#### C. AvailabilityGatedLoss (from gpt52)
Hard-gates tasks with insufficient supervision:
```python
# If availability < 20%, exclude task entirely (no regularization fallback)
if target_availability < min_availability:
    task_weight = 0  # Complete exclusion
```
**Impact**: Prevents task collapse (casualty, anomaly, transition → 0)

#### D. PCGradSurgery (from kimik2/gemini)
Projects conflicting gradients to prevent interference:
```python
# Groups: stable (regime, forecast) vs noisy (casualty, daily_forecast)
if dot(grad_noisy, grad_stable) < 0:
    # Remove conflicting component
    grad_noisy = grad_noisy - project(grad_noisy, grad_stable)
```
**Impact**: Prevents high-magnitude forecast gradients from destroying regime representations

#### E. CrossResolutionCycleConsistency (from gpt52/gemini)
Enforces daily→monthly aggregation consistency:
```python
# Aggregate predicted daily latents to monthly
aggregated_monthly = aggregate(daily_pred_latents)
# Should match monthly teacher latent
cycle_loss = 1 - cosine_sim(aggregated_monthly, monthly_teacher)
```
**Impact**: Prevents daily forecasts from diverging from monthly structure

### 18.3 Integration

Minimal integration (3 lines in train_multi_resolution.py):
```python
from analysis.training_improvements_integration import (
    apply_training_improvements, ImprovedTrainingConfig
)

# After creating trainer:
config = ImprovedTrainingConfig(
    use_pcgrad=True,
    use_softplus_kendall=True,
    use_availability_gating=True,
)
apply_training_improvements(trainer, config)
```

Full integration requires replacing DailyForecastingHead with ImprovedForecastModule.

### 18.4 Expected Impact

| Issue | Before | After (Expected) |
|-------|--------|------------------|
| Forecast train/val gap | 140x | <5x |
| Task collapse (casualty→0) | 1e-6 loss | >0.1 loss |
| Negative Kendall loss | Yes | No |
| Gradient interference | Uncontrolled | Projected |
| Daily/monthly divergence | High | Aligned |
| Params/sample ratio | 11,397 | ~1,500 (with latent predictor) |

### 18.5 Files Created

1. **`analysis/training_improvements.py`**: Core improvement modules
   - LatentStatePredictor
   - LowRankFeatureDecoder
   - PCGradSurgery
   - SoftplusKendallLoss
   - AvailabilityGatedLoss
   - CrossResolutionCycleConsistency
   - LatentPredictiveCodingLoss
   - PhysicalConsistencyConstraint

2. **`analysis/training_improvements_integration.py`**: Integration utilities
   - ImprovedForecastModule
   - ImprovedTrainingStep
   - apply_training_improvements()
   - create_improved_forecast_head()

### 18.6 Next Steps

1. **Immediate**: Apply minimal integration (SoftplusKendall + PCGrad)
2. **Short-term**: Replace DailyForecastingHead with LatentStatePredictor
3. **Medium-term**: Add cycle consistency and predictive coding loss
4. **Long-term**: Consider full AFNP-STGS architecture if improvements plateau

---

*Report generated by Claude Code architecture review agents*
*Last updated: 2026-01-31 (Section 18 added: synthesized solution)*
