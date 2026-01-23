# ISW Embedding Integration Research: Multi-Resolution HAN Enhancement

## Executive Summary

This document analyzes five integration strategies for incorporating pre-computed 1024-dimensional Voyage AI embeddings (voyage-4-large) from ~1,272 ISW daily assessment reports into the existing Multi-Resolution Hierarchical Attention Network (HAN) for conflict prediction.

**Key Constraint**: ~1,272 samples is a small dataset for deep learning, making overfitting the primary risk. All recommendations prioritize regularization and parameter efficiency.

**Architecture Context**: The existing HAN processes:
- Daily quantitative features (equipment, personnel, deepstate, firms, viina) at ~1426 timesteps
- Monthly aggregated features (sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom) at ~48 timesteps
- Uses `d_model=128` as the standard hidden dimension
- Employs learnable `no_observation_token` for missing data (never fabricates values)

---

## 1. Frozen Embedding Injection

### Concept
Concatenate ISW embeddings directly to daily feature vectors without training the embedding space. The pre-trained voyage-4-large embeddings capture semantic content; we use them as fixed features.

### Implementation Options

#### Option 1A: Input-Level Injection (Before Daily Encoder)
```python
class DailySourceEncoderWithISW(nn.Module):
    def __init__(self, source_config, isw_dim=1024, d_model=128, ...):
        # Project ISW embeddings to d_model (frozen projection)
        self.isw_projection = nn.Linear(isw_dim, d_model)
        self.isw_projection.requires_grad = False  # Keep frozen initially

        # Adjust input projection to handle concatenated features
        combined_dim = source_config.n_features + d_model
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
```

**Pros**:
- ISW context available from the start of processing
- Simple implementation
- Zero additional trainable parameters if projection is fixed

**Cons**:
- Dimensionality mismatch (1024 >> ~50 quantitative features) may dominate
- No separation of concerns between quantitative and narrative modalities

#### Option 1B: Post-Encoder Injection (After Daily Encoder, Before Fusion)
```python
class MultiResolutionHANWithISW(MultiResolutionHAN):
    def forward(self, ..., isw_embeddings, isw_mask):
        # Standard daily encoding
        daily_encoded = {}
        for name in self.daily_source_names:
            encoded, attn = self.daily_encoders[name](...)
            daily_encoded[name] = encoded

        # Fuse daily sources
        fused_daily, combined_daily_mask, _ = self.daily_fusion(...)

        # Inject ISW embeddings via simple gating
        isw_proj = self.frozen_isw_projection(isw_embeddings)  # [batch, seq, d_model]
        gate = torch.sigmoid(self.isw_gate(fused_daily))
        fused_daily = fused_daily + gate * isw_proj
```

**Pros**:
- Quantitative features processed first, then augmented with narrative context
- Cleaner separation of modalities
- Learnable gate controls ISW influence (only ~128*128 = 16K parameters)

**Cons**:
- ISW information not available during within-domain attention

#### Option 1C: Pre-Fusion Injection (Before Cross-Resolution Fusion)
```python
# Add ISW as an additional "source" at the monthly aggregation stage
aggregated_daily = self.monthly_aggregation(fused_daily, month_boundaries, ...)
isw_monthly = self.aggregate_isw_to_monthly(isw_embeddings, month_boundaries)

# Concatenate along feature dimension before fusion
combined_monthly = torch.cat([aggregated_daily, isw_monthly], dim=-1)
combined_monthly = self.combined_projection(combined_monthly)  # Project back to d_model
```

**Recommendation**: **Option 1B (Post-Encoder Injection) with Gating**

### Dimensionality Considerations

| Component | Dimension |
|-----------|-----------|
| ISW Embedding (voyage-4-large) | 1024 |
| Quantitative features (per source) | ~6-55 |
| Model hidden dimension (d_model) | 128 |

**Critical Issue**: The 1024-dim ISW embedding is 10-20x larger than typical quantitative feature sets. Direct concatenation would cause the model to over-rely on narrative features.

**Solutions**:
1. **Dimensionality reduction**: Project ISW to d_model (128) via frozen linear layer
2. **Feature scaling**: Normalize ISW projection to have similar magnitude as quantitative encodings
3. **Gated injection**: Use learned gates to control ISW contribution

```python
# Recommended projection with scaling
class FrozenISWProjection(nn.Module):
    def __init__(self, isw_dim=1024, d_model=128):
        super().__init__()
        self.projection = nn.Linear(isw_dim, d_model, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.projection.requires_grad = False

        # Learnable scale factor (single parameter)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, isw_embeddings):
        proj = self.projection(isw_embeddings)
        return self.scale * F.normalize(proj, dim=-1) * math.sqrt(self.d_model)
```

### Evaluation Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Implementation Complexity | **Low** | ~50-100 lines of code |
| Training Data Requirements | **Minimal** | Only need to train gate/scale (~130 params) |
| Overfitting Risk | **Very Low** | Almost no new parameters |
| Expected Benefit | **Moderate** | ISW provides context but limited adaptation |

---

## 2. Learned Projection Layer

### Concept
Add a trainable projection from the 1024-dim embedding space to a task-specific representation that can be jointly optimized with the prediction objectives.

### Implementation Options

#### Option 2A: Linear Projection
```python
class LearnedISWProjection(nn.Module):
    def __init__(self, isw_dim=1024, d_model=128, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(isw_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, isw_embeddings):
        return self.dropout(self.layer_norm(self.projection(isw_embeddings)))
```

**Parameters**: 1024 * 128 + 128 + 256 = **131,456 parameters**

#### Option 2B: MLP Projection with Bottleneck
```python
class MLPISWProjection(nn.Module):
    def __init__(self, isw_dim=1024, d_model=128, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(isw_dim, bottleneck_dim),  # Compress first
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, isw_embeddings):
        return self.projection(isw_embeddings)
```

**Parameters**: 1024 * 64 + 64 + 64 * 128 + 128 = **73,920 parameters**

The bottleneck architecture reduces parameters by ~44% while forcing the model to learn a compressed task-relevant representation.

### Training Strategies

#### Strategy A: Joint Training (End-to-End)
Train the projection jointly with the rest of the HAN from the start.

**Risk**: With only ~1,272 samples, the projection layer has enough capacity to memorize mappings. The ISW embedding space is rich (1024 dims) and a 131K parameter projection can easily overfit.

**Mitigation**:
- Heavy dropout (0.3-0.5)
- L2 regularization on projection weights (weight_decay=0.01)
- Early stopping based on validation loss

#### Strategy B: Pre-Training Then Fine-Tuning (Recommended)
1. **Phase 1**: Freeze HAN weights, train only ISW projection on auxiliary task
2. **Phase 2**: Fine-tune full model with lower learning rate on projection

```python
# Phase 1: Pre-train projection
for param in model.parameters():
    param.requires_grad = False
for param in model.isw_projection.parameters():
    param.requires_grad = True

optimizer = AdamW(model.isw_projection.parameters(), lr=1e-3, weight_decay=0.01)

# Train on auxiliary task: reconstruct key quantitative signals from ISW
# This forces projection to capture conflict-relevant information

# Phase 2: Fine-tune full model
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW([
    {'params': model.isw_projection.parameters(), 'lr': 1e-5},  # Lower LR
    {'params': other_params, 'lr': 1e-4},
], weight_decay=0.01)
```

#### Strategy C: Gradual Unfreezing
Start with frozen projection (like Option 1), then gradually unfreeze layers.

```python
class GradualUnfreezeScheduler:
    def __init__(self, model, unfreeze_epoch=10):
        self.model = model
        self.unfreeze_epoch = unfreeze_epoch
        # Start frozen
        for param in model.isw_projection.parameters():
            param.requires_grad = False

    def step(self, epoch):
        if epoch >= self.unfreeze_epoch:
            for param in self.model.isw_projection.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch}: Unfroze ISW projection")
```

### Evaluation Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Implementation Complexity | **Low-Medium** | ~100-200 lines including training logic |
| Training Data Requirements | **Medium** | ~1,272 samples borderline sufficient with regularization |
| Overfitting Risk | **Medium-High** | 74K-131K new parameters vs 1,272 samples |
| Expected Benefit | **Medium-High** | Task-specific adaptation can improve relevance |

**Recommendation**: Use **MLP with bottleneck (Option 2B)** and **Strategy B (Pre-train then fine-tune)**

---

## 3. Cross-Attention Mechanisms

### Concept
Use attention to dynamically weight the relevance of narrative embeddings to quantitative features, allowing the model to learn which aspects of ISW reports matter for specific predictions.

### Implementation: Quantitative-to-Narrative Cross-Attention

```python
class QuantitativeNarrativeCrossAttention(nn.Module):
    """
    Quantitative features (queries) attend to narrative embeddings (keys/values).
    This allows the model to retrieve relevant narrative context for each
    quantitative observation.
    """
    def __init__(
        self,
        quant_dim: int = 128,      # Dimension of quantitative encodings
        narrative_dim: int = 1024,  # Dimension of ISW embeddings
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project narrative to hidden dim (large reduction)
        self.narrative_proj_k = nn.Linear(narrative_dim, hidden_dim)
        self.narrative_proj_v = nn.Linear(narrative_dim, hidden_dim)

        # Queries come from quantitative encodings (already at hidden_dim)
        self.query_proj = nn.Linear(quant_dim, hidden_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        quant_encoded: Tensor,      # [batch, seq_len, quant_dim]
        narrative_embeddings: Tensor,  # [batch, seq_len, narrative_dim]
        narrative_mask: Tensor,     # [batch, seq_len] True=valid
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            enhanced_quant: Quantitative encodings enhanced with narrative context
            attention_weights: Optional [batch, num_heads, seq_q, seq_k]
        """
        batch_size, seq_len, _ = quant_encoded.shape

        # Project
        Q = self.query_proj(quant_encoded)
        K = self.narrative_proj_k(narrative_embeddings)
        V = self.narrative_proj_v(narrative_embeddings)

        # Cross-attention (quant queries, narrative keys/values)
        key_padding_mask = ~narrative_mask  # True = ignore

        attended, attn_weights = self.cross_attention(
            Q, K, V,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )

        # Gated residual connection
        gate = self.gate(torch.cat([quant_encoded, attended], dim=-1))
        enhanced = self.norm(quant_encoded + gate * attended)

        if return_attention:
            return enhanced, attn_weights
        return enhanced, None
```

**Parameters**: ~2 * (1024 * 128) + (128 * 128) + (128 * 2 * 128) = **311,296 parameters**

### Temporal Attention: Historical ISW Reports

For predictions at time `t`, which historical ISW reports (t-1, t-2, ..., t-k) are most relevant?

```python
class TemporalNarrativeAttention(nn.Module):
    """
    Attends over a window of past ISW reports to capture evolving narrative context.
    Uses causal masking to prevent information leakage from future reports.
    """
    def __init__(
        self,
        narrative_dim: int = 1024,
        hidden_dim: int = 128,
        num_heads: int = 8,
        window_size: int = 7,  # Look back 7 days
        dropout: float = 0.1,
    ):
        super().__init__()

        self.window_size = window_size

        # Project narrative embeddings
        self.narrative_proj = nn.Linear(narrative_dim, hidden_dim)

        # Learnable query for "what's relevant for prediction"
        self.prediction_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Temporal position encoding for window
        self.temporal_pos_encoding = nn.Embedding(window_size, hidden_dim)

        # Self-attention over temporal window
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        narrative_embeddings: Tensor,  # [batch, full_seq, narrative_dim]
        current_timestep: int,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            narrative_embeddings: Full sequence of ISW embeddings
            current_timestep: Index of current prediction timestep

        Returns:
            context: Temporally-aggregated narrative context [batch, hidden_dim]
            attention_weights: Optional temporal attention weights
        """
        batch_size = narrative_embeddings.shape[0]

        # Extract window: [t-window_size, ..., t-1]
        start_idx = max(0, current_timestep - self.window_size)
        end_idx = current_timestep

        if end_idx <= start_idx:
            # No history available, return zeros
            return torch.zeros(batch_size, self.hidden_dim, device=narrative_embeddings.device), None

        window = narrative_embeddings[:, start_idx:end_idx, :]  # [batch, window, narrative_dim]
        window_len = window.shape[1]

        # Project and add temporal position
        window_proj = self.narrative_proj(window)  # [batch, window, hidden_dim]
        positions = torch.arange(window_len, device=window.device)
        # Reverse so most recent is position 0
        positions = self.window_size - 1 - positions
        pos_encoding = self.temporal_pos_encoding(positions)
        window_proj = window_proj + pos_encoding

        # Expand prediction query
        query = self.prediction_query.expand(batch_size, -1, -1)

        # Attention: query attends to temporal window
        context, attn_weights = self.temporal_attention(
            query, window_proj, window_proj,
            need_weights=return_attention,
        )

        context = self.output_proj(context.squeeze(1))

        if return_attention:
            return context, attn_weights
        return context, None
```

### Integration into HAN

```python
class MultiResolutionHANWithCrossAttention(MultiResolutionHAN):
    def __init__(self, ..., enable_narrative_attention: bool = True):
        super().__init__(...)

        if enable_narrative_attention:
            self.quant_narrative_attention = QuantitativeNarrativeCrossAttention(
                quant_dim=self.d_model,
                narrative_dim=1024,
                hidden_dim=self.d_model,
                num_heads=8,
            )
            self.temporal_narrative_attention = TemporalNarrativeAttention(
                narrative_dim=1024,
                hidden_dim=self.d_model,
                window_size=7,
            )

    def forward(self, ..., isw_embeddings, isw_mask):
        # ... existing daily/monthly encoding ...

        # After daily fusion, before monthly aggregation
        fused_daily, daily_mask, _ = self.daily_fusion(...)

        # Enhance with narrative context
        fused_daily, _ = self.quant_narrative_attention(
            fused_daily, isw_embeddings, isw_mask
        )

        # Continue with monthly aggregation and fusion
        aggregated_daily, aggregated_daily_mask, _ = self.monthly_aggregation(...)
```

### Evaluation Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Implementation Complexity | **Medium-High** | ~300-400 lines, architectural changes |
| Training Data Requirements | **Medium-High** | ~311K new parameters need supervision |
| Overfitting Risk | **Medium-High** | Complex attention patterns can memorize |
| Expected Benefit | **High** | Dynamic, interpretable narrative integration |

**Recommendation**: Use cross-attention **after daily fusion** with heavy dropout (0.3) and attention regularization (entropy penalty to encourage spreading attention).

---

## 4. Contrastive Learning

### Concept
Train the model to align quantitative states with their corresponding narrative descriptions using InfoNCE or similar objectives. This creates a shared embedding space where similar conflict states cluster together.

### InfoNCE Implementation

```python
class ContrastiveISWLoss(nn.Module):
    """
    Contrastive loss to align quantitative state representations with
    corresponding ISW narrative embeddings.

    Uses InfoNCE objective:
        L = -log(exp(sim(q_i, n_i)/tau) / sum_j(exp(sim(q_i, n_j)/tau)))

    where q_i is quantitative encoding for day i, n_i is ISW embedding for day i,
    and the denominator sums over all samples in the batch (including negatives).
    """
    def __init__(
        self,
        quant_dim: int = 128,
        narrative_dim: int = 1024,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.temperature = temperature

        # Project both modalities to shared space
        self.quant_projection = nn.Sequential(
            nn.Linear(quant_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        self.narrative_projection = nn.Sequential(
            nn.Linear(narrative_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(
        self,
        quant_encodings: Tensor,      # [batch, seq_len, quant_dim]
        narrative_embeddings: Tensor,  # [batch, seq_len, narrative_dim]
        valid_mask: Tensor,           # [batch, seq_len] True where both modalities valid
    ) -> Tensor:
        """
        Compute contrastive loss over valid timesteps.

        Returns:
            loss: Scalar contrastive loss
        """
        # Flatten and filter to valid timesteps
        batch_size, seq_len, _ = quant_encodings.shape

        # Reshape: [batch * seq_len, dim]
        quant_flat = quant_encodings.view(-1, quant_encodings.shape[-1])
        narrative_flat = narrative_embeddings.view(-1, narrative_embeddings.shape[-1])
        mask_flat = valid_mask.view(-1)

        # Select valid samples
        valid_indices = mask_flat.nonzero(as_tuple=True)[0]
        if len(valid_indices) < 2:
            return torch.tensor(0.0, device=quant_encodings.device)

        quant_valid = quant_flat[valid_indices]
        narrative_valid = narrative_flat[valid_indices]
        n_samples = len(valid_indices)

        # Project to shared space
        quant_proj = F.normalize(self.quant_projection(quant_valid), dim=-1)
        narrative_proj = F.normalize(self.narrative_projection(narrative_valid), dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(quant_proj, narrative_proj.T) / self.temperature

        # Labels: diagonal elements are positives
        labels = torch.arange(n_samples, device=logits.device)

        # Cross-entropy loss (both directions)
        loss_q2n = F.cross_entropy(logits, labels)
        loss_n2q = F.cross_entropy(logits.T, labels)

        return (loss_q2n + loss_n2q) / 2
```

### Positive/Negative Pair Construction

The key challenge with contrastive learning is constructing meaningful positive/negative pairs.

#### Strategy 1: Same Day = Positive (Baseline)
```python
def create_pairs_same_day(quant_encodings, narrative_embeddings, masks):
    """
    Positive: (quant[t], narrative[t]) - same day
    Negative: (quant[t], narrative[t']) where t' != t
    """
    # The InfoNCE implementation above uses this naturally via the diagonal
    pass
```

**Problem**: This is too easy - the model may just learn to match by temporal features rather than semantic content.

#### Strategy 2: Same Context = Positive (Recommended)
```python
def create_pairs_same_context(
    quant_encodings: Tensor,
    narrative_embeddings: Tensor,
    regime_labels: Tensor,  # From existing regime classification
    window_size: int = 7,
):
    """
    Positive pairs:
    - Same day (baseline)
    - Days within same conflict regime/phase
    - Days within a rolling window (temporal smoothness)

    Hard negatives:
    - Different conflict regime
    - Temporally distant with different quantitative patterns
    """
    positives = []
    negatives = []

    for t in range(len(quant_encodings)):
        # Positive: same day
        positives.append((t, t, 1.0))

        # Positive: same regime (if available)
        same_regime = (regime_labels == regime_labels[t]).nonzero(as_tuple=True)[0]
        for t2 in same_regime:
            if t2 != t and abs(t2 - t) <= window_size:
                positives.append((t, t2.item(), 0.5))  # Lower weight

        # Hard negative: different regime, temporally close
        diff_regime = (regime_labels != regime_labels[t]).nonzero(as_tuple=True)[0]
        for t2 in diff_regime:
            if abs(t2 - t) <= window_size * 2:
                negatives.append((t, t2.item()))

    return positives, negatives
```

#### Strategy 3: Temporal Contrast (For State Transitions)
```python
class TemporalContrastiveLoss(nn.Module):
    """
    Learn representations that distinguish state transitions.

    Positive: (state[t], narrative[t]) and (delta[t->t+1], narrative[t+1])
    Negative: (state[t], narrative[t']) from different transition contexts
    """
    def __init__(self, ...):
        super().__init__()

        # Project state deltas
        self.delta_projection = nn.Sequential(
            nn.Linear(quant_dim * 2, projection_dim),  # [state_t, state_t+1]
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, states, narratives, masks):
        # Compute state deltas
        deltas = torch.cat([states[:, :-1], states[:, 1:]], dim=-1)

        # Project
        delta_proj = F.normalize(self.delta_projection(deltas), dim=-1)

        # Narratives at t+1 describe the transition from t to t+1
        narrative_proj = F.normalize(
            self.narrative_projection(narratives[:, 1:]), dim=-1
        )

        # InfoNCE on transitions
        # ...
```

### Integration with Main Loss

```python
class MultiResolutionHANWithContrastive(MultiResolutionHAN):
    def __init__(self, ..., contrastive_weight: float = 0.1):
        super().__init__(...)
        self.contrastive_loss = ContrastiveISWLoss(
            quant_dim=self.d_model,
            narrative_dim=1024,
            projection_dim=128,
        )
        self.contrastive_weight = contrastive_weight

    def compute_loss(self, outputs, targets, quant_encodings, narrative_embeddings, masks):
        # Main task losses
        casualty_loss = F.mse_loss(outputs['casualty_pred'], targets['casualties'])
        regime_loss = F.cross_entropy(outputs['regime_logits'], targets['regime'])

        # Contrastive loss
        contrastive_loss = self.contrastive_loss(
            quant_encodings, narrative_embeddings, masks['valid']
        )

        total_loss = casualty_loss + regime_loss + self.contrastive_weight * contrastive_loss

        return total_loss, {
            'casualty_loss': casualty_loss,
            'regime_loss': regime_loss,
            'contrastive_loss': contrastive_loss,
        }
```

### Evaluation Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Implementation Complexity | **Medium** | ~200-300 lines for loss + pair construction |
| Training Data Requirements | **Medium** | Works with limited data if pairs are well-constructed |
| Overfitting Risk | **Low-Medium** | Contrastive learning is regularizing by nature |
| Expected Benefit | **Medium-High** | Creates aligned semantic space |

**Recommendation**: Use contrastive learning as an **auxiliary loss** (weight ~0.1) with **same-regime positive pairs** to learn a semantic alignment without replacing the main objectives.

---

## 5. Multi-Task Learning

### Concept
Add auxiliary objectives that force the model to leverage narrative understanding, improving feature representations even if auxiliary tasks aren't the primary goal.

### Auxiliary Task 1: Narrative-to-State Prediction

Can we predict key quantitative signals from the ISW embedding alone?

```python
class NarrativeToStateHead(nn.Module):
    """
    Auxiliary task: Predict aggregated quantitative signals from narrative.

    If ISW embeddings contain conflict-relevant information, they should
    be able to predict (at least approximately):
    - Total equipment losses in the day
    - Personnel casualty rate
    - Front line activity level
    - Fire detection count
    """
    def __init__(
        self,
        narrative_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 10,  # Number of key quantitative signals
        dropout: float = 0.1,
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(narrative_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Output keys
        self.output_keys = [
            'equipment_total_losses',
            'personnel_daily_rate',
            'front_activity_index',
            'fire_count',
            'frp_total',
            'territorial_change',
            'unit_movements',
            'airstrike_indicator',
            'artillery_intensity',
            'supply_line_status',
        ]

    def forward(self, narrative_embeddings: Tensor) -> Tensor:
        """
        Args:
            narrative_embeddings: [batch, seq_len, narrative_dim]
        Returns:
            predictions: [batch, seq_len, output_dim]
        """
        return self.predictor(narrative_embeddings)

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Args:
            predictions: [batch, seq_len, output_dim]
            targets: [batch, seq_len, output_dim] actual quantitative values
            mask: [batch, seq_len] True where valid
        """
        # Masked MSE loss
        mask_expanded = mask.unsqueeze(-1)
        loss = F.mse_loss(
            predictions * mask_expanded,
            targets * mask_expanded,
            reduction='sum'
        ) / mask_expanded.sum().clamp(min=1)

        return loss
```

### Auxiliary Task 2: State-to-Narrative Similarity Scoring

Given a quantitative state, score how well different narrative embeddings describe it.

```python
class StateNarrativeSimilarityHead(nn.Module):
    """
    Auxiliary task: Score compatibility between quantitative states and narratives.

    This is a discriminative task:
    - Given state S and narrative N, predict if N accurately describes S
    - Training signal: same-day pairs are positive, shuffled pairs are negative

    Useful for:
    - Detecting narrative-state mismatches (potential anomalies)
    - Improving the joint embedding space
    """
    def __init__(
        self,
        state_dim: int = 128,
        narrative_dim: int = 1024,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Project both to common space
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.narrative_projection = nn.Linear(narrative_dim, hidden_dim)

        # Bilinear similarity + MLP
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state_encodings: Tensor,      # [batch, seq_len, state_dim]
        narrative_embeddings: Tensor,  # [batch, seq_len, narrative_dim]
    ) -> Tensor:
        """
        Returns:
            similarity_scores: [batch, seq_len] compatibility scores
        """
        state_proj = self.state_projection(state_encodings)
        narrative_proj = self.narrative_projection(narrative_embeddings)

        # Bilinear interaction
        interaction = self.bilinear(state_proj, narrative_proj)

        # Score
        scores = self.classifier(interaction).squeeze(-1)

        return scores

    def compute_loss(
        self,
        state_encodings: Tensor,
        narrative_embeddings: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        Create positive and negative pairs, compute BCE loss.
        """
        batch_size, seq_len = mask.shape
        device = state_encodings.device

        # Positive pairs: same timestep
        pos_scores = self(state_encodings, narrative_embeddings)

        # Negative pairs: shuffle narratives within batch
        shuffle_indices = torch.randperm(seq_len, device=device)
        shuffled_narratives = narrative_embeddings[:, shuffle_indices, :]
        neg_scores = self(state_encodings, shuffled_narratives)

        # BCE loss
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores[mask], pos_labels[mask]
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores[mask], neg_labels[mask]
        )

        return (pos_loss + neg_loss) / 2
```

### Auxiliary Task 3: Next-Day Narrative Prediction

Given current state and narrative, predict embedding of next day's narrative.

```python
class NextNarrativePredictionHead(nn.Module):
    """
    Auxiliary task: Predict tomorrow's ISW embedding from today's state + narrative.

    This forces the model to understand:
    - How quantitative states evolve
    - How narratives track with state evolution
    - Temporal dynamics of conflict reporting
    """
    def __init__(
        self,
        state_dim: int = 128,
        narrative_dim: int = 1024,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim + narrative_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Predict in a lower-dimensional space (not full 1024)
        self.predictor = nn.Linear(hidden_dim, 256)
        self.target_projection = nn.Linear(narrative_dim, 256)

    def forward(
        self,
        state_encodings: Tensor,      # [batch, seq_len, state_dim]
        narrative_embeddings: Tensor,  # [batch, seq_len, narrative_dim]
    ) -> Tensor:
        """
        Returns:
            predictions: [batch, seq_len-1, 256] predicted next-day narrative embeddings
        """
        # Concatenate state and narrative at each timestep
        combined = torch.cat([state_encodings, narrative_embeddings], dim=-1)

        # Encode
        encoded = self.encoder(combined)

        # Predict next-day narrative (shifted by 1)
        predictions = self.predictor(encoded[:, :-1])

        return predictions

    def compute_loss(
        self,
        state_encodings: Tensor,
        narrative_embeddings: Tensor,
        mask: Tensor,
    ) -> Tensor:
        predictions = self(state_encodings, narrative_embeddings)

        # Target: actual next-day narrative (projected)
        targets = self.target_projection(narrative_embeddings[:, 1:])

        # Shifted mask
        shifted_mask = mask[:, 1:] & mask[:, :-1]

        # Cosine similarity loss (1 - cos_sim)
        predictions_norm = F.normalize(predictions, dim=-1)
        targets_norm = F.normalize(targets, dim=-1)

        cos_sim = (predictions_norm * targets_norm).sum(dim=-1)
        loss = (1 - cos_sim)[shifted_mask].mean()

        return loss
```

### Combined Multi-Task Loss

```python
class MultiTaskLoss(nn.Module):
    """
    Combines main prediction objectives with auxiliary narrative tasks.
    """
    def __init__(
        self,
        main_weight: float = 1.0,
        narrative_to_state_weight: float = 0.1,
        state_narrative_sim_weight: float = 0.1,
        next_narrative_weight: float = 0.05,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.weights = {
            'main': main_weight,
            'narrative_to_state': narrative_to_state_weight,
            'state_narrative_sim': state_narrative_sim_weight,
            'next_narrative': next_narrative_weight,
            'contrastive': contrastive_weight,
        }

        # Auxiliary heads
        self.narrative_to_state = NarrativeToStateHead(1024, 256, 10)
        self.state_narrative_sim = StateNarrativeSimilarityHead(128, 1024, 128)
        self.next_narrative = NextNarrativePredictionHead(128, 1024, 256)
        self.contrastive = ContrastiveISWLoss(128, 1024, 128)

    def forward(
        self,
        model_outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        state_encodings: Tensor,
        narrative_embeddings: Tensor,
        masks: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute all losses and return weighted sum.
        """
        losses = {}

        # Main task losses
        if 'casualty_pred' in model_outputs:
            losses['casualty'] = F.mse_loss(
                model_outputs['casualty_pred'][masks['valid']],
                targets['casualties'][masks['valid']]
            )

        if 'regime_logits' in model_outputs:
            losses['regime'] = F.cross_entropy(
                model_outputs['regime_logits'].view(-1, 4),
                targets['regime'].view(-1)
            )

        # Combine main losses
        main_loss = sum(losses.values())
        losses['main'] = main_loss

        # Auxiliary losses
        valid_mask = masks['valid']

        # Narrative-to-state
        n2s_pred = self.narrative_to_state(narrative_embeddings)
        losses['narrative_to_state'] = self.narrative_to_state.compute_loss(
            n2s_pred, targets['key_quantitative'], valid_mask
        )

        # State-narrative similarity
        losses['state_narrative_sim'] = self.state_narrative_sim.compute_loss(
            state_encodings, narrative_embeddings, valid_mask
        )

        # Next-narrative prediction
        losses['next_narrative'] = self.next_narrative.compute_loss(
            state_encodings, narrative_embeddings, valid_mask
        )

        # Contrastive
        losses['contrastive'] = self.contrastive(
            state_encodings, narrative_embeddings, valid_mask
        )

        # Weighted total
        total_loss = sum(
            self.weights.get(name, 0.0) * loss
            for name, loss in losses.items()
        )

        return total_loss, losses
```

### Evaluation Summary

| Metric | Score | Notes |
|--------|-------|-------|
| Implementation Complexity | **High** | ~500+ lines, multiple auxiliary heads |
| Training Data Requirements | **Low-Medium** | Auxiliary tasks provide self-supervision |
| Overfitting Risk | **Low** | Multi-task learning is strongly regularizing |
| Expected Benefit | **High** | Forces meaningful narrative integration |

**Recommendation**: Implement **Narrative-to-State** and **Contrastive** as primary auxiliary tasks. They provide the strongest learning signal for the extractive use case.

---

## Comparative Analysis

### Summary Table

| Option | Complexity | Data Needs | Overfit Risk | Expected Benefit | Recommended Priority |
|--------|------------|------------|--------------|------------------|---------------------|
| 1. Frozen Injection | Low | Minimal | Very Low | Moderate | **Start here** |
| 2. Learned Projection | Low-Medium | Medium | Medium-High | Medium-High | Second |
| 3. Cross-Attention | Medium-High | Medium-High | Medium-High | High | Third |
| 4. Contrastive Learning | Medium | Medium | Low-Medium | Medium-High | **Auxiliary loss** |
| 5. Multi-Task Learning | High | Low-Medium | Low | High | **Combine with above** |

### Recommended Integration Path

Given the constraint of ~1,272 samples, I recommend a staged approach:

#### Stage 1: Baseline with Frozen Injection
```
Week 1-2: Implement Option 1B (Post-Encoder Injection with Gating)
- Add FrozenISWProjection to reduce 1024 -> 128 dims
- Add learnable gate (~16K parameters)
- Evaluate impact on prediction tasks
- Establish baseline performance
```

#### Stage 2: Add Contrastive Auxiliary Loss
```
Week 3-4: Implement Option 4 (Contrastive Learning)
- Add ContrastiveISWLoss as auxiliary objective (weight=0.1)
- Use same-day + same-regime positive pairs
- This regularizes the joint embedding space
- Monitor alignment metrics
```

#### Stage 3: Gradually Unfreeze Projection
```
Week 5-6: Implement Option 2B (MLP Projection with Gradual Unfreezing)
- Replace frozen projection with trainable MLP
- Use gradual unfreezing schedule
- Add narrative-to-state auxiliary task
- Apply aggressive regularization (dropout=0.4, weight_decay=0.01)
```

#### Stage 4 (If data allows): Cross-Attention
```
Week 7-8: Implement Option 3 (Cross-Attention) if Stage 3 shows improvement
- Add QuantitativeNarrativeCrossAttention after daily fusion
- Use attention regularization (entropy penalty)
- This is the most parameter-intensive option
```

### Key Implementation Considerations

1. **Data Alignment**: Ensure ISW embeddings are properly aligned with daily timestamps. Missing days should use `no_observation_token` (consistent with existing architecture).

2. **Evaluation Protocol**: Use k-fold cross-validation (k=5) with temporal ordering respected. Never leak future ISW reports into past predictions.

3. **Regularization is Critical**: With 1,272 samples:
   - Dropout: 0.3-0.5 on new components
   - Weight decay: 0.01-0.1
   - Early stopping with patience=10
   - Gradient clipping (max_norm=1.0)

4. **Monitoring**: Track:
   - Main task metrics (casualty RMSE, regime accuracy)
   - Auxiliary task metrics (contrastive loss, N2S correlation)
   - Attention entropy (should be moderate, not collapsed)
   - Embedding space properties (t-SNE visualization)

---

## Architectural Integration Point

Based on the existing `MultiResolutionHAN` architecture, the recommended integration point is:

```python
# In MultiResolutionHAN.forward()

# STEP 2: FUSE DAILY SOURCES
fused_daily, combined_daily_mask, daily_fusion_attn = self.daily_fusion(
    daily_encoded,
    daily_masks,
    return_attention=True,
)

# ===== NEW: ISW EMBEDDING INJECTION =====
if isw_embeddings is not None:
    # Project ISW embeddings (frozen or learned)
    isw_projected = self.isw_projection(isw_embeddings)

    # Gated fusion
    gate = torch.sigmoid(self.isw_gate(fused_daily))
    fused_daily = fused_daily + gate * isw_projected

    # Optional: Cross-attention for richer integration
    if self.use_narrative_cross_attention:
        fused_daily, _ = self.narrative_cross_attention(
            fused_daily, isw_embeddings, isw_mask
        )
# =========================================

# STEP 3: AGGREGATE DAILY TO MONTHLY
aggregated_daily, aggregated_daily_mask, agg_attention = self.monthly_aggregation(...)
```

This placement ensures:
1. Quantitative features are processed first (preserving their signal)
2. ISW context enhances the fused daily representation
3. The enhanced representation flows through monthly aggregation and fusion
4. Consistent with the architecture's principle of "information flows via attention, not fabrication"

---

## Conclusion

For a small dataset of ~1,272 samples, the recommended approach prioritizes:
1. **Parameter efficiency**: Frozen/low-parameter projections
2. **Regularization through multi-task learning**: Contrastive + narrative-to-state auxiliary losses
3. **Gradual complexity**: Start simple, add complexity only if validated

The ISW embeddings provide valuable expert narrative context that can complement the quantitative features. The key is extracting this value without overfitting to the limited sample size.
