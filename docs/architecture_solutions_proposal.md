# HAN Model Architecture Solutions Proposal

**Date:** 2026-01-26
**Based on:** Phase 2 Validation Findings
**Status:** Proposal for Review

---

## Executive Summary

Phase 2 validation identified five critical architectural issues with the Multi-Resolution HAN model. This document proposes concrete solutions for each problem, including code-level implementation details, expected improvements, and associated risks.

**Key Findings from Validation:**

| Problem | Severity | Effort | Quick Win? |
|---------|----------|--------|------------|
| C5: No Causal Masking | Critical | Low | Yes |
| C2: Pointwise Source Gate | High | Medium | No |
| C1: Daily Resolution Underutilized | High | Medium | No |
| C3: ISW Correlation Not Learned | Medium | High | No |
| C4: VIIRS Lag Modeling | Low | High | No |

---

## Problem C5: Static Classification / Future Leakage

### Problem Summary
TemporalEncoder uses bidirectional attention, allowing future information to leak into predictions. Output is collapsed (100% same regime, CV=0.002).

### Root Cause
The `TemporalEncoder` class (line 785-873 in `multi_resolution_han.py`) creates a standard `nn.TransformerEncoder` without any causal (autoregressive) mask. The encoder can "see" future timesteps when predicting current timesteps.

```python
# Current implementation - NO causal mask
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=num_layers,
    enable_nested_tensor=False,
)
```

Additionally, synthetic targets (`torch.randint`) during training provide no learning signal, causing output collapse.

### Proposed Solution
1. **Add causal attention mask** to TemporalEncoder
2. **Replace synthetic targets** with real next-timestep prediction targets
3. **Add teacher forcing** with scheduled sampling during training

### Implementation

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`

```python
class TemporalEncoder(nn.Module):
    """
    Causal transformer encoder for autoregressive temporal processing.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 60,
        causal: bool = True,  # NEW PARAMETER
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.causal = causal

        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(d_model)

        # Pre-compute causal mask (upper triangular)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Add positional encoding
        hidden = self.positional_encoding(x)

        # Prepare masks
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # True = ignore

            # Handle fully masked sequences
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

        # Causal mask for autoregressive processing
        attn_mask = None
        if self.causal:
            attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer encoding
        hidden = self.transformer_encoder(
            hidden,
            mask=attn_mask,  # Causal attention mask
            src_key_padding_mask=src_key_padding_mask,
        )

        return self.output_norm(hidden)
```

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

Replace synthetic targets with next-timestep autoregressive targets:

```python
def compute_autoregressive_loss(
    self,
    outputs: Dict[str, Tensor],
    batch: Dict[str, Any],
) -> Tensor:
    """
    Compute loss for predicting X(t+1) given X(t).

    Uses forecast_pred to predict next month's aggregated features.
    """
    forecast_pred = outputs['forecast_pred']  # [batch, seq, n_features]

    # Target: actual features at t+1
    # Shift features forward by 1 timestep
    monthly_features = batch['aggregated_monthly_features']  # [batch, seq, n_features]

    # Predictions for t predict actual values at t+1
    pred = forecast_pred[:, :-1, :]  # [batch, seq-1, features]
    target = monthly_features[:, 1:, :]  # [batch, seq-1, features]

    # Mask invalid positions
    mask = batch.get('monthly_mask', None)
    if mask is not None:
        # Both t and t+1 must be valid
        valid = mask[:, :-1] & mask[:, 1:]
        pred = pred[valid]
        target = target[valid]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return F.mse_loss(pred, target)
```

### Expected Impact
- **Prediction integrity:** Model can only use past information for predictions
- **Output diversity:** With real targets, regime predictions will vary appropriately
- **Forecasting ability:** Model learns true temporal dynamics, not shortcuts

### Risks/Trade-offs
- **Reduced context:** Causal masking limits information available at each timestep
- **Training complexity:** Autoregressive training may require longer training
- **Performance:** May need to increase model capacity to compensate for information restriction

### Validation
1. Re-run C5 tests: verify regime prediction diversity (expect CV > 0.1)
2. Run temporal ablation: shuffled future should not affect predictions
3. Measure forecast accuracy against held-out months

---

## Problem C2: Pointwise Source Gate (No Temporal Context)

### Problem Summary
The `source_gate` in `DailyCrossSourceFusion` is purely pointwise (Linear->LayerNorm->GELU->Linear->Softmax). It lacks any temporal awareness (no RNN, Conv, or Attention over time).

### Root Cause
Current implementation (lines 452-458):

```python
self.source_gate = nn.Sequential(
    nn.Linear(d_model * self.n_sources, d_model),
    nn.LayerNorm(d_model),
    nn.GELU(),
    nn.Linear(d_model, self.n_sources),
    nn.Softmax(dim=-1),
)
```

Each timestep independently computes source importance without considering:
- Recent source reliability patterns
- Temporal coherence of source weights
- Cross-source temporal dependencies

### Proposed Solution
Replace pointwise gating with **Temporal Attention Gate** that considers local temporal context.

### Implementation

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`

```python
class TemporalSourceGate(nn.Module):
    """
    Source gating with temporal context awareness.

    Uses local convolution and self-attention to compute source importance
    that considers temporal patterns, not just current timestep.
    """

    def __init__(
        self,
        d_model: int,
        n_sources: int,
        kernel_size: int = 7,
        nhead: int = 4,
        use_attention: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_sources = n_sources
        self.use_attention = use_attention

        # Local temporal context via 1D convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                d_model * n_sources,
                d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=1,  # Mix all sources
            ),
            nn.LayerNorm([d_model]),
            nn.GELU(),
        )

        # Optional self-attention for longer-range temporal patterns
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(d_model)

        # Final gate projection
        self.gate_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_sources),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        stacked_sources: Tensor,  # [batch, seq, n_sources * d_model]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute temporally-aware source importance weights.

        Args:
            stacked_sources: Concatenated source representations
            mask: Optional validity mask [batch, seq]

        Returns:
            Source importance [batch, seq, n_sources]
        """
        batch_size, seq_len, _ = stacked_sources.shape

        # Temporal convolution: capture local patterns
        # Conv1d expects [batch, channels, seq]
        x = stacked_sources.transpose(1, 2)
        x = self.temporal_conv(x)  # [batch, d_model, seq]
        x = x.transpose(1, 2)  # [batch, seq, d_model]

        # Optional attention for longer patterns
        if self.use_attention:
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = ~mask

            attended, _ = self.temporal_attention(
                x, x, x,
                key_padding_mask=key_padding_mask,
            )
            x = self.attn_norm(x + attended)

        # Compute gate weights
        gate_weights = self.gate_projection(x)

        return gate_weights
```

**Update `DailyCrossSourceFusion.__init__`:**

```python
# Replace:
# self.source_gate = nn.Sequential(...)

# With:
self.source_gate = TemporalSourceGate(
    d_model=d_model,
    n_sources=self.n_sources,
    kernel_size=7,  # ~1 week context window
    nhead=4,
    use_attention=True,
)
```

**Update `DailyCrossSourceFusion.forward`:**

```python
# Replace:
# source_importance = self.source_gate(concat_sources)

# With:
source_importance = self.source_gate(concat_sources, combined_mask)
```

### Expected Impact
- **Temporal coherence:** Source weights will vary smoothly over time
- **Context awareness:** Gate can learn "FIRMS becomes more important after equipment losses"
- **Pattern recognition:** Captures recurring weekly/monthly source reliability patterns

### Risks/Trade-offs
- **Increased parameters:** ~20% more parameters in fusion module
- **Training time:** Attention adds ~15% training time
- **Potential overfitting:** More expressive gate may overfit on small datasets

### Validation
1. Re-run C2-H3 test: verify temporal context in gate weights
2. Measure gate weight autocorrelation (expect > 0.3 at lag 1)
3. Ablate temporal components to verify contribution

---

## Problem C1: Daily Resolution Underutilized

### Problem Summary
Model learns temporal patterns at MONTHLY resolution but not daily. Monthly is 2.9x more sensitive to temporal perturbations.

### Root Cause
1. **LearnableMonthlyAggregation** compresses all daily information into single monthly vectors
2. Daily patterns are lost before reaching the main temporal encoder
3. The monthly aggregation attention may not preserve fine-grained daily dynamics

### Proposed Solution
**Hierarchical Temporal Processing** with separate daily-level temporal encoding before aggregation.

### Implementation

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`

Add a new `DailyTemporalEncoder` module:

```python
class DailyTemporalEncoder(nn.Module):
    """
    Process daily sequences with temporal awareness BEFORE monthly aggregation.

    Uses a lightweight causal transformer to capture daily dynamics,
    then passes the temporally-enriched representations to monthly aggregation.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 1,  # Lightweight
        dropout: float = 0.1,
        max_len: int = 1500,
        window_size: int = 31,  # ~1 month local attention
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size

        # Local attention pattern (not full O(n^2))
        self.local_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Temporal convolutions for multi-scale patterns
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(d_model, d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 14, 28]  # ~3 days, week, 2 weeks, month
        ])

        self.conv_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(d_model)

    def _create_local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create attention mask for local windowed attention."""
        # Allow attention only within window_size
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        return mask

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process daily sequence with local temporal patterns.

        Args:
            x: Daily representations [batch, n_days, d_model]
            mask: Observation mask [batch, n_days]

        Returns:
            Temporally-enriched daily representations [batch, n_days, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # 1. Multi-scale convolutions
        x_conv = x.transpose(1, 2)  # [batch, d_model, seq]
        conv_outputs = []
        for conv in self.multi_scale_conv:
            conv_out = conv(x_conv)
            conv_outputs.append(conv_out)

        multi_scale = torch.cat(conv_outputs, dim=1)  # [batch, d_model, seq]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, seq, d_model]
        multi_scale = self.conv_fusion(multi_scale)

        # 2. Local attention
        local_mask = self._create_local_mask(seq_len, device)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        attended, _ = self.local_attention(
            x, x, x,
            attn_mask=local_mask,
            key_padding_mask=key_padding_mask,
        )

        # 3. Combine
        output = self.norm(x + attended + multi_scale)

        return output
```

**Update `MultiResolutionHAN` to include daily temporal processing:**

```python
# In MultiResolutionHAN.__init__, after daily_fusion:
self.daily_temporal_encoder = DailyTemporalEncoder(
    d_model=d_model,
    nhead=nhead // 2,  # Lighter weight
    num_layers=1,
    dropout=dropout,
    max_len=max_daily_len,
    window_size=31,  # Monthly window
)

# In MultiResolutionHAN.forward, after daily_fusion, before monthly_aggregation:
# Apply daily temporal processing
fused_daily = self.daily_temporal_encoder(fused_daily, combined_daily_mask)
```

### Expected Impact
- **Daily pattern capture:** Weekly cycles, operational tempo preserved
- **Better aggregation:** Monthly aggregation receives temporally-enriched representations
- **Balanced resolution:** Daily and monthly contribute more equally

### Risks/Trade-offs
- **Memory usage:** Additional O(n) storage for daily sequences
- **Computational cost:** Extra attention operations on long sequences
- **Complexity:** More components to tune and maintain

### Validation
1. Re-run C1 resolution reversal test (expect daily sensitivity to increase)
2. Measure daily vs monthly contribution ratio (target: <2x difference)
3. Ablate daily encoder to verify contribution

---

## Problem C3: ISW Correlation Not Learned

### Problem Summary
Raw data DOES correlate with ISW embeddings (CCA r=0.9996, phase-specific correlations r=0.91), but the model fails to learn these patterns. Technical limitations mask phase-specific relationships.

### Root Cause
1. **Resolution mismatch:** ISW is daily narrative; model processes at monthly level
2. **No explicit ISW alignment:** Model has no mechanism to align with narrative embeddings
3. **Phase pooling:** Aggregating across conflict phases destroys phase-specific patterns
4. **Missing conditioning:** Model doesn't condition on conflict phase

### Proposed Solution
**Multi-Modal Contrastive Alignment** with phase-aware conditioning.

### Implementation

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`

Add ISW alignment components:

```python
class ISWAlignmentModule(nn.Module):
    """
    Aligns model representations with ISW narrative embeddings.

    Uses contrastive learning to ensure latent space captures
    information present in ISW assessments.
    """

    def __init__(
        self,
        d_model: int = 128,
        isw_dim: int = 1024,  # Voyage embedding dimension
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.temperature = temperature

        # Project model representations to shared space
        self.model_projection = nn.Sequential(
            nn.Linear(d_model, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )

        # Project ISW embeddings to shared space
        self.isw_projection = nn.Sequential(
            nn.Linear(isw_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Linear(projection_dim * 2, projection_dim),
        )

        # Phase conditioning
        self.phase_embedding = nn.Embedding(12, d_model)  # 11 phases + unknown

    def compute_alignment_loss(
        self,
        model_repr: Tensor,      # [batch, seq, d_model]
        isw_embeddings: Tensor,  # [batch, seq, isw_dim]
        phase_indices: Tensor,   # [batch, seq]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute contrastive alignment loss between model and ISW representations.

        Same timestep (positive pair): should be similar
        Different timesteps (negative pairs): should be dissimilar
        """
        batch_size, seq_len, _ = model_repr.shape

        # Add phase conditioning to model representations
        phase_emb = self.phase_embedding(phase_indices)
        model_repr = model_repr + phase_emb

        # Project to shared space
        model_proj = self.model_projection(model_repr)  # [batch, seq, proj_dim]
        isw_proj = self.isw_projection(isw_embeddings)  # [batch, seq, proj_dim]

        # Normalize
        model_proj = F.normalize(model_proj, dim=-1)
        isw_proj = F.normalize(isw_proj, dim=-1)

        # Flatten for contrastive computation
        if mask is not None:
            valid_positions = mask.view(-1)
            model_proj = model_proj.view(-1, model_proj.size(-1))[valid_positions]
            isw_proj = isw_proj.view(-1, isw_proj.size(-1))[valid_positions]
        else:
            model_proj = model_proj.view(-1, model_proj.size(-1))
            isw_proj = isw_proj.view(-1, isw_proj.size(-1))

        n_samples = model_proj.size(0)
        if n_samples < 2:
            return torch.tensor(0.0, device=model_repr.device)

        # Compute similarity matrix
        logits = torch.matmul(model_proj, isw_proj.T) / self.temperature

        # Labels: diagonal is positive
        labels = torch.arange(n_samples, device=logits.device)

        # Symmetric contrastive loss
        loss_model_to_isw = F.cross_entropy(logits, labels)
        loss_isw_to_model = F.cross_entropy(logits.T, labels)

        return (loss_model_to_isw + loss_isw_to_model) / 2


class PhaseConditionedAttention(nn.Module):
    """
    Attention mechanism conditioned on conflict phase.

    Different phases attend differently to sources and temporal patterns.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_phases: int = 12,
        nhead: int = 8,
    ):
        super().__init__()

        self.phase_query_mod = nn.Embedding(n_phases, d_model)
        self.phase_key_mod = nn.Embedding(n_phases, d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        phase_indices: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply phase-conditioned attention.

        Args:
            x: Input [batch, seq, d_model]
            phase_indices: Phase labels [batch, seq]
            mask: Optional mask [batch, seq]
        """
        # Modulate queries and keys based on phase
        q_mod = self.phase_query_mod(phase_indices)
        k_mod = self.phase_key_mod(phase_indices)

        q = x + q_mod
        k = x + k_mod

        key_padding_mask = ~mask if mask is not None else None

        attended, _ = self.attention(q, k, x, key_padding_mask=key_padding_mask)

        return self.norm(x + attended)
```

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

Add ISW alignment loss to training:

```python
# In training loop:
if 'isw_embeddings' in batch and batch['isw_embeddings'] is not None:
    alignment_loss = self.model.isw_alignment.compute_alignment_loss(
        model_repr=outputs['temporal_encoded'],
        isw_embeddings=batch['isw_embeddings'],
        phase_indices=batch.get('phase_indices', torch.zeros_like(mask).long()),
        mask=mask,
    )
    losses['isw_alignment'] = alignment_loss * 0.1  # Weight factor
```

### Expected Impact
- **ISW alignment:** Model learns to capture narrative-correlated patterns
- **Phase awareness:** Different behavior for different conflict phases
- **Better generalization:** Alignment with external knowledge improves robustness

### Risks/Trade-offs
- **Data dependency:** Requires ISW embeddings during training
- **Alignment dominance:** May overshadow other learning signals if weight too high
- **Embedding quality:** Dependent on ISW embedding quality

### Validation
1. Re-run C3 tests: expect correlation > 0.5 between model latents and ISW
2. Measure phase-specific clustering in latent space
3. Compare held-out ISW prediction accuracy

---

## Problem C4: VIIRS Lag Not Modeled

### Problem Summary
VIIRS lags casualties by +10-13 days. FIRMS->VIIRS cascade peaks at 4-5 days. Simple causal models don't explain this.

### Root Cause
1. **No explicit lag modeling:** Model treats all sources as synchronous
2. **Infrastructure damage cycle:** Physical damage (FIRMS) -> power infrastructure -> nighttime lights (VIIRS)
3. **Reporting delays:** VIIRS composite periods don't align with conflict events

### Proposed Solution
**Learnable Source-Specific Temporal Shifts** with cascade modeling.

### Implementation

**File:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`

```python
class LearnableTemporalShift(nn.Module):
    """
    Learn source-specific temporal offsets.

    Different sources have different delays from ground truth events.
    This module learns to align them temporally.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_sources: int = 6,
        max_shift: int = 14,  # Maximum shift in days
        use_soft_shift: bool = True,
    ):
        super().__init__()

        self.n_sources = n_sources
        self.max_shift = max_shift
        self.use_soft_shift = use_soft_shift

        if use_soft_shift:
            # Learn soft attention over possible shifts
            self.shift_attention = nn.Parameter(
                torch.zeros(n_sources, 2 * max_shift + 1)
            )
            # Initialize to center (no shift)
            nn.init.zeros_(self.shift_attention)
            self.shift_attention.data[:, max_shift] = 1.0
        else:
            # Learn discrete shifts
            self.shift_offsets = nn.Parameter(
                torch.zeros(n_sources)
            )

        # Source-specific temporal kernels for cascade modeling
        self.cascade_kernels = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)
            for _ in range(n_sources)
        ])

    def forward(
        self,
        source_features: Dict[str, Tensor],
        source_names: List[str],
    ) -> Dict[str, Tensor]:
        """
        Apply learned temporal shifts to each source.

        Args:
            source_features: Dict of [batch, seq, d_model] per source
            source_names: Ordered list of source names

        Returns:
            Temporally aligned source features
        """
        aligned_features = {}

        for i, name in enumerate(source_names):
            if name not in source_features:
                continue

            x = source_features[name]  # [batch, seq, d_model]
            batch_size, seq_len, d_model = x.shape

            if self.use_soft_shift:
                # Soft shift via weighted combination of shifted versions
                shift_weights = F.softmax(self.shift_attention[i], dim=-1)

                shifted_versions = []
                for s in range(-self.max_shift, self.max_shift + 1):
                    if s < 0:
                        shifted = F.pad(x[:, :s, :], (0, 0, -s, 0))
                    elif s > 0:
                        shifted = F.pad(x[:, s:, :], (0, 0, 0, s))
                    else:
                        shifted = x
                    shifted_versions.append(shifted)

                # Weighted combination
                shifted_stack = torch.stack(shifted_versions, dim=-1)  # [batch, seq, d, shifts]
                aligned = (shifted_stack * shift_weights).sum(dim=-1)
            else:
                # Hard shift (differentiable via straight-through estimator)
                shift = torch.round(self.shift_offsets[i]).int().item()
                shift = max(-self.max_shift, min(self.max_shift, shift))

                if shift < 0:
                    aligned = F.pad(x[:, :shift, :], (0, 0, -shift, 0))
                elif shift > 0:
                    aligned = F.pad(x[:, shift:, :], (0, 0, 0, shift))
                else:
                    aligned = x

            # Apply cascade kernel
            aligned = aligned.transpose(1, 2)  # [batch, d_model, seq]
            aligned = self.cascade_kernels[i](aligned)
            aligned = aligned.transpose(1, 2)  # [batch, seq, d_model]

            aligned_features[name] = aligned

        return aligned_features
```

**Integration into `DailyCrossSourceFusion`:**

```python
# Add to __init__:
self.temporal_shift = LearnableTemporalShift(
    d_model=d_model,
    n_sources=len(source_names),
    max_shift=14,
    use_soft_shift=True,
)

# Add to forward, before stacking:
source_hidden_aligned = self.temporal_shift(
    source_hidden_with_type,
    self.source_names,
)
# Use source_hidden_aligned instead of source_hidden_with_type for stacking
```

### Expected Impact
- **VIIRS alignment:** Learn optimal lag automatically
- **Cascade modeling:** Capture FIRMS->VIIRS dependency
- **Cross-source alignment:** All sources aligned to common temporal reference

### Risks/Trade-offs
- **Edge effects:** Shifting creates boundary artifacts
- **Overfitting:** May learn spurious alignments
- **Interpretability:** Learned shifts may be hard to interpret

### Validation
1. Inspect learned shifts (expect VIIRS ~10-13 days)
2. Measure cross-source correlation before/after alignment
3. Ablate shifts to verify improvement

---

## Priority Ranking

### Critical (Do First)

1. **C5: Add Causal Masking**
   - **Effort:** Low (1-2 days)
   - **Impact:** Critical - fixes fundamental prediction integrity
   - **Dependencies:** None
   - **Quick win:** Yes

2. **C5: Replace Synthetic Targets**
   - **Effort:** Medium (3-5 days)
   - **Impact:** Critical - enables meaningful learning
   - **Dependencies:** Requires target data preparation
   - **Quick win:** Partially

### High Priority (Do Next)

3. **C2: Temporal Source Gate**
   - **Effort:** Medium (3-5 days)
   - **Impact:** High - improves source weighting quality
   - **Dependencies:** None
   - **Quick win:** No

4. **C1: Daily Temporal Encoder**
   - **Effort:** Medium (5-7 days)
   - **Impact:** High - utilizes daily resolution properly
   - **Dependencies:** None
   - **Quick win:** No

### Medium Priority (Do After Core Fixes)

5. **C3: ISW Alignment**
   - **Effort:** High (7-10 days)
   - **Impact:** Medium - improves narrative grounding
   - **Dependencies:** Requires ISW embeddings in dataset
   - **Quick win:** No

6. **C3: Phase Conditioning**
   - **Effort:** Medium (3-5 days)
   - **Impact:** Medium - enables phase-specific behavior
   - **Dependencies:** Requires phase labels
   - **Quick win:** No

### Lower Priority (Consider Later)

7. **C4: Temporal Shifts**
   - **Effort:** Medium (5-7 days)
   - **Impact:** Low-Medium - addresses specific VIIRS issue
   - **Dependencies:** None
   - **Quick win:** No

---

## Quick Wins Summary

| Fix | Effort | Files to Modify | LOC Change |
|-----|--------|-----------------|------------|
| Add causal mask | 1 day | `multi_resolution_han.py` | ~30 lines |
| Add `causal` parameter | 1 hour | `multi_resolution_han.py` | ~5 lines |
| Config flag for causal | 1 hour | `MultiResolutionHANConfig` | ~3 lines |

---

## Major Refactors Summary

| Refactor | Effort | Complexity | Risk |
|----------|--------|------------|------|
| Temporal Source Gate | 3-5 days | Medium | Low |
| Daily Temporal Encoder | 5-7 days | High | Medium |
| ISW Alignment Module | 7-10 days | High | Medium |
| Phase Conditioning | 3-5 days | Medium | Low |
| Temporal Shift Module | 5-7 days | Medium | Medium |

---

## Implementation Order

### Phase 1: Foundation Fixes (Week 1)
1. Add causal masking to TemporalEncoder
2. Update config to support causal flag
3. Prepare real autoregressive targets
4. Replace synthetic targets in training

### Phase 2: Source Processing (Week 2)
1. Implement TemporalSourceGate
2. Replace pointwise gate in DailyCrossSourceFusion
3. Implement DailyTemporalEncoder
4. Integrate into model forward pass

### Phase 3: Multi-Modal Alignment (Week 3)
1. Implement ISWAlignmentModule
2. Add phase conditioning components
3. Update training loop with alignment loss
4. Integrate ISW embeddings into dataset

### Phase 4: Temporal Alignment (Week 4)
1. Implement LearnableTemporalShift
2. Integrate with source processing
3. Validate learned shifts
4. Tune shift hyperparameters

### Phase 5: Validation (Week 5)
1. Re-run all Phase 2 validation tests
2. Document improvements
3. Performance benchmarking
4. Final report

---

## References

- **Current Model:** `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_han.py`
- **Architecture Doc:** `/Users/daniel.tipton/ML_OSINT/docs/HAN.md`
- **Validation Results:** `/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation/phase2/PHASE2_STATE.md`
- **Training Code:** `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`
