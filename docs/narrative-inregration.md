# ISW Narrative Integration Module
## Implementation Specification for HAN Enhancement

**Context:** Post-validation architectural improvement  
**Priority:** High - addresses largest identified signal gap  
**Prerequisite:** Completed C1-C5 validation cycle; model now achieving r=0.54 forecast correlation

---

## Background

Phase 2 validation revealed a critical discrepancy in C3 (ISW-Latent Alignment):

| Analysis | Finding |
|----------|---------|
| Raw sensor → ISW CCA correlation | **0.9996** |
| Model latent → ISW R² | **-1.00** |
| Phase-specific correlations | 0.55 - 0.91 depending on conflict phase |

The raw input data is almost perfectly correlated with ISW narrative content in canonical space, yet the model completely fails to capture this relationship. This represents the largest identified opportunity for performance improvement.

The validation also revealed that correlation strength varies significantly by conflict phase:

| Conflict Phase | Max PC Correlation | Notes |
|----------------|-------------------|-------|
| summer_offensive_23 | 0.912 | Highest alignment |
| eastern_focus | 0.884 | Strong |
| attritional_phase | 0.871 | Strong |
| kharkiv_offensive | 0.774 | Moderate |
| avdiivka_fall | 0.550 | Weakest alignment |

This phase-dependent structure suggests that a global alignment approach will underperform compared to phase-conditioned alignment.

---

## Technical Objective

Implement an ISW alignment module that:

1. Projects model latents and ISW embeddings into a shared representational space
2. Applies contrastive learning to align same-timestep representations
3. Conditions alignment on conflict phase to handle the observed phase-dependent correlation structure
4. Integrates without disrupting the existing forecast performance (r=0.54 baseline)

---

## Proposed Architecture

### Component 1: ISWAlignmentModule

```python
class ISWAlignmentModule(nn.Module):
    """
    Aligns HAN latent representations with ISW narrative embeddings.
    
    The validation found CCA correlation of 0.9996 between raw features and ISW,
    but model latents show R² = -1.00. This module bridges that gap through
    learned projection and contrastive alignment.
    """
    
    def __init__(self, latent_dim, isw_dim, projection_dim, n_phases=7):
        super().__init__()
        
        # Project both representations to shared space
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.isw_projection = nn.Sequential(
            nn.Linear(isw_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Phase-specific modulation (addresses 0.55-0.91 phase variance)
        self.phase_embeddings = nn.Embedding(n_phases, projection_dim)
        self.phase_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid()
        )
        
    def forward(self, latents, isw_embeddings, phase_ids=None):
        # Project to shared space
        latent_proj = self.latent_projection(latents)
        isw_proj = self.isw_projection(isw_embeddings)
        
        # Apply phase conditioning if available
        if phase_ids is not None:
            phase_emb = self.phase_embeddings(phase_ids)
            gate_input = torch.cat([latent_proj, phase_emb], dim=-1)
            gate = self.phase_gate(gate_input)
            latent_proj = latent_proj * gate
        
        return latent_proj, isw_proj
```

### Component 2: Contrastive Alignment Loss

```python
class ISWContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for temporal alignment.
    
    Positive pairs: same timestep (latent_t, isw_t)
    Negative pairs: different timesteps within batch
    
    Temperature-scaled to handle the high correlation ceiling (0.9996 CCA).
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, latent_proj, isw_proj):
        # Normalize projections
        latent_norm = F.normalize(latent_proj, dim=-1)
        isw_norm = F.normalize(isw_proj, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(latent_norm, isw_norm.T) / self.temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Symmetric loss
        loss_l2i = F.cross_entropy(sim_matrix, labels)
        loss_i2l = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_l2i + loss_i2l) / 2
```

### Component 3: Phase-Conditioned Attention (Optional Enhancement)

```python
class PhaseConditionedAttention(nn.Module):
    """
    Modulates attention patterns based on detected conflict phase.
    
    Validation showed correlation varies from 0.55 (avdiivka_fall) to 
    0.91 (summer_offensive_23). This module learns phase-specific 
    query/key transformations to handle this heterogeneity.
    """
    
    def __init__(self, d_model, n_heads, n_phases=7):
        super().__init__()
        self.n_phases = n_phases
        
        # Phase-specific Q/K modulation
        self.phase_query_mod = nn.Embedding(n_phases, d_model)
        self.phase_key_mod = nn.Embedding(n_phases, d_model)
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
    def forward(self, x, phase_ids):
        # Get phase-specific modulations
        q_mod = self.phase_query_mod(phase_ids).unsqueeze(1)
        k_mod = self.phase_key_mod(phase_ids).unsqueeze(1)
        
        # Apply multiplicative modulation
        q = x * (1 + q_mod)
        k = x * (1 + k_mod)
        
        # Standard attention with modulated Q/K
        output, weights = self.attention(q, k, x)
        return output, weights
```

---

## Integration Points

### Data Pipeline

ISW embeddings need to be added to the batch. Current batch structure:

```python
# Current
batch = {
    'daily_sources': {...},
    'monthly_sources': {...},
    'masks': {...},
    'targets': {...}
}

# Required addition
batch['isw_embeddings'] = torch.Tensor  # [batch, seq_len, isw_dim]
batch['phase_ids'] = torch.LongTensor   # [batch, seq_len] or [batch]
```

Reference the ISW embedding files used in probe analysis (Section 5 probes) for the embedding format and dimensionality.

### Loss Integration

```python
# In training loop
outputs = model(batch)

# Existing losses
forecast_loss = compute_forecast_loss(outputs, targets)
regime_loss = compute_regime_loss(outputs, targets)

# New ISW alignment loss
latent_proj, isw_proj = model.isw_alignment(
    outputs['temporal_encoded'], 
    batch['isw_embeddings'],
    batch.get('phase_ids')
)
isw_loss = isw_contrastive_loss(latent_proj, isw_proj)

# Combined loss with weighting
total_loss = forecast_loss + regime_loss + lambda_isw * isw_loss
```

The `lambda_isw` weight should start small (0.1) and potentially be scheduled to increase as training stabilizes.

---

## Validation Criteria

After implementation, re-run the C3 validation probes:

| Metric | Current | Target |
|--------|---------|--------|
| Model latent → ISW R² | -1.00 | > 0.3 |
| Latent-ISW cosine similarity | ~0 | > 0.5 |
| Phase-stratified correlation (min) | N/A | > 0.4 |
| Forecast correlation (must not regress) | 0.54 | ≥ 0.54 |

The implementation succeeds if:
1. ISW alignment metrics improve substantially
2. Forecast performance does not degrade
3. Phase conditioning shows differentiated behavior across conflict phases

---

## Implementation Notes

1. **ISW Embedding Source:** The embeddings used in validation probes are pre-computed. Check `/Users/daniel.tipton/ML_OSINT/outputs/analysis/han_validation/` for the exact format and any preprocessing applied.

2. **Phase Labels:** The 7 conflict phases are defined in the validation code. These need to be mapped to timesteps in the training data. Consider whether phase should be:
   - Hard-coded based on dates
   - Predicted by the model (auxiliary task)
   - Soft/probabilistic based on learned regime predictions

3. **Gradient Isolation:** Consider using `stop_gradient` on the ISW projection during early training to prevent the alignment loss from disrupting the already-learned forecast representations.

4. **Temperature Tuning:** The 0.9996 CCA correlation suggests the alignment ceiling is very high. The contrastive temperature may need to be lower than typical (try 0.03-0.07 range) to provide sufficient gradient signal.

---

## Expected Outcome

If the 0.9996 CCA correlation represents learnable signal (as the validation suggests), successful ISW integration should:

1. Improve forecast correlation from 0.54 toward 0.6-0.7
2. Enable narrative-aware anomaly detection (political events that currently get ignored)
3. Provide interpretable alignment between model predictions and qualitative assessments
4. Potentially improve regime classification by incorporating narrative context

The phase conditioning specifically addresses the validation finding that correlation varies from 0.55 to 0.91 across conflict phases - a global alignment approach would be dragged down by the weaker phases.

---

## Files Reference

- Validation results: `outputs/analysis/han_validation/phase2/`
- ISW probe analysis: `probe_5_*.json`
- Model architecture: `analysis/multi_resolution_han.py`
- Current checkpoint: `analysis/training_runs/run_24-01-2026_20-22/stage3_han/best_checkpoint.pt`

Begin with the ISWAlignmentModule and contrastive loss. Phase conditioning can be added as a second iteration if the basic alignment shows promise.