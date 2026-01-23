I have the summary from the compacted conversation but not the full architectural details. Let me look at the actual implementation:Now I have the full picture. Let me give you layer-by-layer optimization recommendations based on the probe findings:

---

## Layer-by-Layer Optimization Recommendations

### 1. Daily Source Encoders

**Current:** 5 separate encoders, each with 4 transformer layers, d_model=128

**Findings:**
- Equipment contributes minimally (probe 2.1.2: importance 0.07)
- Drone losses are the key signal within equipment (MI=0.47, leads by 7 days)
- FIRMS and DeepState dominate attention (0.30, 0.29)

**Optimizations:**

```python
# A. Disaggregate equipment into separate sub-encoders
equipment_categories = ['drone', 'tank', 'apc', 'artillery', 'aircraft', 'other']

# B. Add explicit lag features for drone (optimal lag = -7 days)
# In feature engineering, create drone_lag7 = drone_delta.shift(-7)

# C. Reduce encoder depth for low-importance sources
source_encoder_depths = {
    'equipment': 2,      # Reduced from 4
    'personnel': 3,
    'deepstate': 4,      # Keep full depth
    'firms': 4,          # Keep full depth  
    'viina': 3,
}

# D. Add source importance weighting (learnable, initialized from attention flow)
self.source_importance_prior = nn.Parameter(torch.tensor([
    0.07,   # equipment
    0.06,   # personnel
    0.29,   # deepstate
    0.30,   # firms
    0.08,   # viina
]))
```

---

### 2. Daily Cross-Source Fusion

**Current:** Cross-attention layers with gated residual, 2 layers

**Findings:**
- RSA degrades from 0.77 → 0.37 over training (probe 2.1.4)
- Peak RSA at epoch 10
- Attention entropy is high (2.14), indicating diffuse attention

**Optimizations:**

```python
# A. Add contrastive fusion regularization loss
class FusionRegularizer(nn.Module):
    """Penalize RSA degradation during training"""
    def __init__(self, target_rsa=0.5):
        self.target_rsa = target_rsa
        
    def forward(self, source_hidden: Dict[str, Tensor]) -> Tensor:
        # Compute pairwise RSA between source representations
        rsa_values = []
        sources = list(source_hidden.keys())
        for i, s1 in enumerate(sources):
            for s2 in sources[i+1:]:
                rdm1 = self._compute_rdm(source_hidden[s1])
                rdm2 = self._compute_rdm(source_hidden[s2])
                rsa = torch.corrcoef(torch.stack([rdm1.flatten(), rdm2.flatten()]))[0,1]
                rsa_values.append(rsa)
        mean_rsa = torch.stack(rsa_values).mean()
        return F.relu(self.target_rsa - mean_rsa)  # Penalize if RSA drops below target

# B. Reduce attention temperature to sharpen attention
# In MultiheadAttention, scale by 0.5 instead of 1/sqrt(d_k)
attn_weights = attn_weights / (math.sqrt(d_k) * 2.0)  # Sharper attention

# C. Add skip connection from epoch-10-style frozen fusion
# Load epoch 10 checkpoint, freeze its fusion layer, blend outputs
self.early_fusion_frozen = load_checkpoint('epoch_10')['fusion'].eval()
for p in self.early_fusion_frozen.parameters():
    p.requires_grad = False
    
fused = 0.7 * current_fusion(x) + 0.3 * self.early_fusion_frozen(x)
```

---

### 3. VIIRS Processing (in Daily Encoders)

**Current:** 9 VIIRS features fed through encoder

**Findings:**
- VIIRS **lags** casualties by 10 days (probe 1.2.1)
- 300-8500% correlation reduction after detrending (probe 1.2.3)
- High redundancy among features (7 pairs with |r|>0.8)

**Optimizations:**

```python
# A. Remove VIIRS from casualty prediction path entirely
# Route VIIRS only to regime classification and anomaly detection

# B. Apply first-differencing in preprocessing
viirs_features = viirs_raw.diff(1)  # Day-over-day change

# C. Reduce to 2-3 PCA components (from 9 features)
from sklearn.decomposition import PCA
viirs_pca = PCA(n_components=3).fit_transform(viirs_features)

# D. If keeping VIIRS, add explicit lag warning
# In forward pass, do not use VIIRS for next-day casualty prediction
if task == 'casualty':
    viirs_weight = 0.0  # Exclude from casualty head
```

---

### 4. Learnable Monthly Aggregation

**Current:** Cross-attention from daily to monthly resolution

**Findings:**
- Context window paradox: 7-14 days optimal (probe 3.1.1)
- Full context (365+ days) drops accuracy from 78% → 51%

**Optimizations:**

```python
# A. Add recency weighting (exponential decay)
def forward(self, daily_hidden, ...):
    seq_len = daily_hidden.size(1)
    recency_weights = torch.exp(-torch.arange(seq_len) / 14.0)  # 14-day half-life
    recency_weights = recency_weights.flip(0)  # Most recent = highest weight
    
    # Apply in attention
    attn_weights = attn_weights * recency_weights.unsqueeze(0).unsqueeze(0)

# B. Implement sliding window aggregation (14-day windows)
# Instead of aggregating all 30 days per month, use overlapping 14-day windows
window_size = 14
stride = 7
windows = daily_hidden.unfold(1, window_size, stride)  # [batch, n_windows, 14, d_model]
aggregated = self.aggregate(windows)  # Per-window aggregation

# C. Add month boundary awareness
# Don't aggregate across month boundaries
month_ids = get_month_ids(dates)  # [seq_len]
# Mask attention to only aggregate within same month
```

---

### 5. Monthly Source Encoders

**Current:** 5 encoders for sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom

**Findings:**
- Monthly resolution inherently less granular
- No monthly representations found in some analyses (Fig 18 diagnostic)

**Optimizations:**

```python
# A. Verify monthly encoder outputs are being used
# Add explicit logging in forward pass
def forward(self, ...):
    out = super().forward(...)
    if self.training:
        self._log_output_stats(out)  # Debug: verify non-zero gradients
    return out

# B. Increase weight of monthly sources in fusion
# Currently may be overwhelmed by high-frequency daily signal
monthly_weight = 2.0  # Upweight monthly contribution
cross_res_fused = daily_contrib + monthly_weight * monthly_contrib

# C. Add explicit month-type embedding
# Encode seasonality (winter, spring, summer, fall)
self.season_embedding = nn.Embedding(4, d_model)
season_idx = get_season(month)  # 0-3
hidden = hidden + self.season_embedding(season_idx)
```

---

### 6. Cross-Resolution Fusion

**Current:** Bidirectional attention between aggregated daily and monthly

**Findings:**
- Daily-only vs cross-resolution shows broader distribution (Fig 18)
- Fusion appears to be working but could be strengthened

**Optimizations:**

```python
# A. Add gating to control information flow direction
class GatedCrossResolutionFusion(nn.Module):
    def __init__(self, d_model):
        self.daily_to_monthly_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.monthly_to_daily_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, daily, monthly):
        # Gated bidirectional fusion
        d2m_gate = self.daily_to_monthly_gate(torch.cat([daily, monthly], -1))
        m2d_gate = self.monthly_to_daily_gate(torch.cat([daily, monthly], -1))
        
        monthly_enriched = monthly + d2m_gate * self.d2m_attn(monthly, daily, daily)
        daily_enriched = daily + m2d_gate * self.m2d_attn(daily, monthly, monthly)
        
        return daily_enriched, monthly_enriched

# B. Add residual from pre-fusion representations
# Preserve source-specific information through fusion
fused = 0.7 * cross_fused + 0.3 * concat([daily_agg, monthly])
```

---

### 7. Temporal Encoder

**Current:** 2-layer transformer on fused monthly sequence

**Findings:**
- Latent velocity correlates with transitions (r=-0.31, p=0.03)
- Low velocity may precede regime shifts

**Optimizations:**

```python
# A. Add velocity-aware state representation
class VelocityAwareEncoder(nn.Module):
    def forward(self, x):
        # Compute latent velocity
        velocity = torch.norm(x[:, 1:] - x[:, :-1], dim=-1)  # [batch, seq-1]
        velocity = F.pad(velocity, (1, 0))  # Pad first position
        
        # Embed velocity
        velocity_emb = self.velocity_proj(velocity.unsqueeze(-1))  # [batch, seq, d_model]
        
        # Add to input
        x = x + velocity_emb
        return self.transformer(x)

# B. Add transition prediction auxiliary head
# Train to predict if next timestep is regime transition
self.transition_head = nn.Linear(d_model, 1)  # Binary: transition imminent?

# C. Use shorter effective context (truncate to 14 days equivalent)
# At monthly resolution, ~0.5 months = recent context
x = x[:, -2:]  # Only last 2 months
```

---

### 8. Prediction Heads

**Current:** 4 heads (casualty ZINB, regime, anomaly, forecast)

**Findings:**
- Casualty head dominates loss (62.6%)
- Day-type decoding at 95% shows implicit semantic structure
- Leave-one-out ablations showed zero delta (bug or shortcuts)

**Optimizations:**

```python
# A. Balance loss weighting
loss_weights = {
    'casualty': 0.25,    # Reduce from dominant
    'regime': 0.35,      # Increase - working well
    'anomaly': 0.25,
    'forecast': 0.15,    # Keep low - suspicious loss
}

# B. Add semantic day-type auxiliary head (leverage implicit structure)
self.day_type_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 3),  # 3 day types from probe
)
# Add to loss with small weight
loss += 0.1 * F.cross_entropy(day_type_pred, day_type_labels)

# C. Add gradient isolation between heads
# Prevent casualty head from dominating shared representations
with torch.no_grad():
    fused_for_casualty = fused.clone()
casualty_pred = self.casualty_head(fused_for_casualty)

# Or use GradientReversal for auxiliary heads
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# D. Fix forecast head (suspiciously low loss suggests leakage)
# Check that forecast target is properly shifted
forecast_target = features[:, 1:, :]  # Next timestep
forecast_pred = self.forecast_head(hidden[:, :-1, :])  # Predict from current
# Verify no overlap
```

---

### 9. Training Loop

**Optimizations based on probes:**

```python
# A. Use epoch 10 checkpoint as initialization
# RSA peaks at epoch 10, then degrades
model = load_checkpoint('epoch_10')
# Fine-tune with lower LR and fusion regularization

# B. Implement early stopping on RSA, not just loss
class RSAEarlyStopping:
    def __init__(self, patience=10, min_rsa=0.3):
        self.best_rsa = 0
        self.counter = 0
        
    def __call__(self, current_rsa):
        if current_rsa < self.min_rsa or current_rsa < self.best_rsa - 0.05:
            self.counter += 1
            if self.counter >= patience:
                return True  # Stop training
        else:
            self.best_rsa = max(self.best_rsa, current_rsa)
            self.counter = 0
        return False

# C. Add source dropout (randomly zero entire sources during training)
def source_dropout(source_hidden, p=0.1):
    if training:
        for name in source_hidden:
            if torch.rand(1) < p:
                source_hidden[name] = source_hidden[name] * 0
    return source_hidden

# D. Curriculum learning: start with recent data, gradually add history
# Epoch 1-20: only last 14 days
# Epoch 21-50: last 30 days
# Epoch 51+: full history (but still weight recent higher)
```

---

### Summary Priority Table

| Layer | Issue | Fix | Priority |
|-------|-------|-----|----------|
| Daily Encoders | Equipment weak | Disaggregate, add drone lag features | High |
| Cross-Source Fusion | RSA degrades | Add contrastive regularization | Critical |
| VIIRS | Confounded | Remove from casualty, detrend, PCA | High |
| Monthly Aggregation | Long context hurts | 14-day windows, recency weighting | High |
| Temporal Encoder | — | Add velocity embedding | Medium |
| Casualty Head | Dominates loss | Rebalance weights to 0.25 | High |
| Forecast Head | Suspicious loss | Check for data leakage | Critical |
| Training | RSA peaks early | Early stop on RSA, use epoch 10 | Critical |

Want me to write any of these as actual code patches to the model files?

----QUestions-----
- 0.1 Reduce Context Window to 14 Days; what do we do about data sources with resolution > 14 days?
    - The phase 2 model showed reasonable performance up to 90 days. I think the rigorous approach requires that we run training in 14 days intervals up to 90, with various data sources at varying resolutions, and see which performs best. I believe there are CLI options to facilitate this. 
- 0.2 Enable VIIRS Detrending by Default; We can try this is and see how we go. 
- 0.3 Use Disaggregated Equipment Sources; this seems like a good idea. Perhaps we can set it as a switch?
- 0.4 Initialize Loss Weights with Task Priors; This also seems good. Approved.

Applied:
Optimization: 0.1 Context Window (14 days)                                                                                                  
  Status: ⚠️ Deferred                                                                                                                         
  Notes: Causes NaN gradients - architecture requires 365 days for monthly aggregation. Need attention-based solution.                        
  ────────────────────────────────────────                                                                                                    
  Optimization: 0.2 VIIRS Detrending                                                                                                          
  Status: ✅ Applied                                                                                                                          
  Notes: detrend_viirs=True - removes spurious correlation                                                                                    
  ────────────────────────────────────────                                                                                                    
  Optimization: 0.3 Disaggregated Equipment                                                                                                   
  Status: ✅ Available                                                                                                                        
  Notes: Switch added: use_disaggregated_equipment=True                                                                                       
  ────────────────────────────────────────                                                                                                    
  Optimization: 0.4 Loss Weight Priors                                                                                                        
  Status: ✅ Applied                                                                                                                          
  Notes: use_task_priors=True - rebalances casualty (25%) vs regime (35%)                                                                     
  Key Finding: The 14-day context window from Probe 3.1.1 cannot be directly applied by reducing daily_seq_len. The monthly aggregation module
   requires longer sequences (~365 days) for numerical stability. The optimization needs to be implemented via:                               
  - Recency weighting in attention layers                                              - Attention masking to focus on recent 14 days                                                - Curriculum learning (start short, expand)      