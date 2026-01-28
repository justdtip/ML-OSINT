# Future Architecture Ideas

**Created:** 2026-01-26
**Status:** Ideas for future development phases

---

## Cross-Source Integration Forcing

### Problem Statement

The current multi-source fusion architecture *allows* integration but doesn't *force* it. The model may learn to do "soft source selection" (picking the best source per task) rather than true information integration across sources.

Evidence from probes:
- VIIRS gets 27% weight (2x uniform)
- Equipment sources get ~7% each
- This pattern suggests selection, not integration

### Proposed Solutions

#### 1. Cross-Source Reconstruction Loss

After fusion, require the model to reconstruct original source features from the fused representation.

```python
class CrossSourceReconstructionLoss(nn.Module):
    """
    Forces integration by requiring fused representation to
    contain information from all sources.
    """
    def __init__(self, d_model: int, source_dims: Dict[str, int]):
        super().__init__()
        self.reconstructors = nn.ModuleDict({
            name: nn.Linear(d_model, dim)
            for name, dim in source_dims.items()
        })

    def forward(self, fused: Tensor, original_sources: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            fused: [batch, seq, d_model] - fused representation
            original_sources: Dict of [batch, seq, source_dim] tensors
        """
        total_loss = 0
        for name, reconstructor in self.reconstructors.items():
            if name in original_sources:
                pred = reconstructor(fused)
                target = original_sources[name]
                total_loss += F.mse_loss(pred, target)
        return total_loss / len(self.reconstructors)
```

**Rationale:** If the model can reconstruct VIIRS from a fused representation that also encodes FIRMS, it must have learned cross-source relationships.

**Effort:** Medium (3-5 days)

---

#### 2. Source Dropout (Multimodal Masking)

Randomly zero entire sources during training, forcing the model to infer missing information from other sources.

```python
class SourceDropout(nn.Module):
    """
    Randomly masks entire sources during training.
    Forces cross-source information redundancy.
    """
    def __init__(self, p: float = 0.2, min_sources: int = 3):
        super().__init__()
        self.p = p
        self.min_sources = min_sources  # Keep at least this many

    def forward(self, source_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.training:
            return source_dict

        sources = list(source_dict.keys())
        n_to_keep = max(self.min_sources, int(len(sources) * (1 - self.p)))

        # Randomly select sources to keep
        keep_sources = random.sample(sources, n_to_keep)

        return {
            name: (tensor if name in keep_sources
                   else torch.zeros_like(tensor))
            for name, tensor in source_dict.items()
        }
```

**Rationale:** Like masked language modeling but for sources. Model must learn cross-source redundancies to handle missing sources.

**Effort:** Low (1-2 days)

---

#### 3. Cross-Source Contrastive Loss

Same timestep across different sources should have similar representations (they observe the same underlying reality).

```python
class CrossSourceContrastiveLoss(nn.Module):
    """
    Encourages representations from different sources at the
    same timestep to be similar.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        source_reprs: Dict[str, Tensor],  # {source: [batch, seq, d]}
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Positive pairs: same (batch, timestep), different sources
        Negative pairs: different timesteps
        """
        sources = list(source_reprs.keys())
        if len(sources) < 2:
            return torch.tensor(0.0)

        # Stack all source representations
        # [n_sources, batch, seq, d]
        stacked = torch.stack([source_reprs[s] for s in sources])

        # For each timestep, sources should be similar
        # Compute pairwise similarities across sources
        n_sources, batch, seq, d = stacked.shape

        # Normalize
        stacked_norm = F.normalize(stacked, dim=-1)

        # Compute cross-source similarity at each timestep
        # [batch, seq, n_sources, n_sources]
        sim_matrix = torch.einsum('sbti,sbtj->btsij',
                                   stacked_norm, stacked_norm)

        # Positive: off-diagonal elements (different sources, same time)
        # Negative: different timesteps
        # ... (full implementation would follow InfoNCE pattern)

        return contrastive_loss
```

**Rationale:** Since all sources observe the same conflict, their latent representations at the same timestep should align.

**Effort:** Medium (3-5 days)

---

#### 4. Mutual Information Maximization

Explicitly maximize mutual information between source representations.

```python
class MutualInformationLoss(nn.Module):
    """
    Maximize I(Z_source_a; Z_source_b | timestep)
    Uses MINE or InfoNCE estimator.
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        # Statistics network for MINE estimator
        self.stats_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """
        MINE lower bound on mutual information.
        Maximize this to encourage shared information.
        """
        # Joint samples (same timestep)
        joint = torch.cat([z_a, z_b], dim=-1)

        # Marginal samples (shuffled timesteps)
        z_b_shuffled = z_b[torch.randperm(z_b.size(0))]
        marginal = torch.cat([z_a, z_b_shuffled], dim=-1)

        # MINE estimator
        t_joint = self.stats_net(joint)
        t_marginal = self.stats_net(marginal)

        # MI lower bound (negative because we minimize loss)
        mi_estimate = t_joint.mean() - torch.log(torch.exp(t_marginal).mean())

        return -mi_estimate  # Negative to maximize MI
```

**Rationale:** Direct optimization objective for cross-source information sharing.

**Effort:** Medium-High (5-7 days)

---

#### 5. Bottleneck Forcing

Force all information through a narrow shared bottleneck.

```python
class IntegrationBottleneck(nn.Module):
    """
    All sources must compress through narrow bottleneck.
    Forces information integration rather than parallel pathways.
    """
    def __init__(
        self,
        d_model: int,
        n_sources: int,
        bottleneck_dim: int = 32,  # Much smaller than d_model
    ):
        super().__init__()
        self.compress = nn.Linear(d_model * n_sources, bottleneck_dim)
        self.expand = nn.Linear(bottleneck_dim, d_model)

    def forward(self, source_reprs: List[Tensor]) -> Tensor:
        # Concatenate all sources
        concat = torch.cat(source_reprs, dim=-1)

        # Force through bottleneck
        compressed = F.gelu(self.compress(concat))

        # Expand back
        integrated = self.expand(compressed)

        return integrated
```

**Rationale:** Information bottleneck forces the model to extract only the most important shared information, encouraging integration over parallel processing.

**Effort:** Low (1-2 days)

---

## Implementation Priority

For the current model (single conflict, limited data):
1. **Source Dropout** - Lowest effort, immediate benefit
2. **Cross-Source Reconstruction** - Best diagnostic of integration quality

For future multi-conflict expansion (Gaza, other conflicts):
1. **Cross-Source Contrastive** - Scales well to more sources
2. **Mutual Information** - Theoretically grounded
3. **Bottleneck Forcing** - Simplest architectural change

---

## Evaluation Metrics for Integration

How to measure if integration is actually happening:

1. **Leave-One-Source-Out Prediction**: Train with all sources, test with one removed. High degradation = good integration.

2. **Cross-Source RSA**: Representational similarity between sources at same timestep. Higher = better integration.

3. **Reconstruction Accuracy**: After fusion, how well can we reconstruct individual sources?

4. **Gradient Flow Analysis**: Do gradients flow between source encoders? Or are they isolated?

5. **Attention Pattern Analysis**: Is cross-source attention actually non-diagonal?

---

## Related Work

- **Multimodal Learning**: CLIP, ALIGN use contrastive objectives across modalities
- **Self-Supervised Learning**: Masked autoencoders force cross-patch integration
- **Information Bottleneck**: Tishby et al. - theoretical framework for compression
- **Multi-View Learning**: CCA, Deep CCA for cross-view alignment

---

## Emergent Modularity Discovery

**Date:** 2026-01-26
**Status:** Unexpected finding during probe validation

### Observation

When running latent velocity probes on the same checkpoint (`best_checkpoint.pt`), dramatically different results emerged depending on whether the ISW alignment module was loaded:

| Configuration | Parameters | Velocity Range | Transition Correlation |
|--------------|------------|----------------|------------------------|
| Without ISW module | 10.7M | 1.06 - 1.09 | r = -0.69 (strong) |
| With ISW module | 11.4M | 0.93 - 6.28 | r = -0.28 (weak) |

The checkpoint contains the full 11.4M model with ISW alignment. When loaded without the ISW module enabled, 16 ISW-related weights were discarded as "unexpected keys."

### Key Insight

The model exhibits **functional modularity** despite being a unified neural network:

1. **Core temporal dynamics** (10.7M params): Produces tight, structured latent trajectories with strong correlation to conflict phase transitions. This appears to be the "backbone" that learns fundamental conflict dynamics.

2. **ISW semantic alignment** (+757K params): Adds variance to the latent space to align with narrative embeddings. This may be operating as an independent "adapter" that enriches representations for semantic tasks but adds noise for pure temporal analysis.

### Implications

1. **Task-Specific Loading**: Different downstream tasks might benefit from loading different subsets of the model:
   - Temporal prediction / phase detection → Load without ISW
   - Narrative grounding / semantic analysis → Load with ISW
   - This is reminiscent of adapter-based fine-tuning in LLMs

2. **Modular Training**: Could train components separately:
   - Pre-train core temporal model
   - Add ISW adapter and train only that component
   - This might prevent ISW from interfering with temporal representations

3. **Probe Design**: Probes analyzing latent dynamics should perhaps disable optional modules to get cleaner signals, then compare with/without to measure each module's contribution.

4. **Architectural Validation**: The fact that removing ISW weights doesn't break the model (and actually improves some metrics) suggests the architecture has natural decomposition points.

### Source Checkpoint

These findings are from training run `run_26-01-2026_15-20` (epoch 55, val_loss=3.6168), with ISW alignment enabled during training. The checkpoint contains 31M total parameters (including optimizer state), with 757K ISW-specific parameters.

### Controlled Experiment Design

**Objective:** Validate that the observed modularity is real and not an artifact of probe configuration.

**Protocol:**

```bash
# 1. Run full probe battery with ISW disabled (baseline temporal analysis)
python -m analysis.probes.run_probes --all \
    --checkpoint analysis/checkpoints/multi_resolution/best_checkpoint.pt \
    --no-isw-alignment \
    --run-id modularity_exp_no_isw

# 2. Run full probe battery with ISW enabled
python -m analysis.probes.run_probes --all \
    --checkpoint analysis/checkpoints/multi_resolution/best_checkpoint.pt \
    --use-isw-alignment \
    --run-id modularity_exp_with_isw

# 3. Compare results
python -m analysis.probes.run_probes --compare-runs modularity_exp_no_isw modularity_exp_with_isw
```

**Metrics to Compare:**

| Category | Specific Metrics | Hypothesis |
|----------|-----------------|------------|
| Latent Dynamics | Velocity mean, std, range | No-ISW should have lower variance |
| Phase Detection | Transition correlation (r) | No-ISW should have stronger correlation |
| RSA | Cross-source similarity | Should be similar (ISW operates post-fusion) |
| Task Performance | Regime/casualty accuracy | With-ISW may be better on semantic tasks |
| Attention Patterns | Source weights, temporal focus | Should be similar for core attention |

**Implementation Requirements:**

1. Add `--no-isw-alignment` and `--use-isw-alignment` flags to `run_probes.py`
2. Ensure probe runner respects these flags rather than auto-detecting
3. Add comparison report generation for side-by-side analysis

**Expected Outcomes:**

- If modularity is real: Clear performance separation by task type
- If interference: With-ISW uniformly worse on temporal metrics
- If integration: Minimal difference (ISW learned to complement, not compete)

### Future Experiments

1. **Controlled Ablation**: Systematically enable/disable modules and measure:
   - Prediction accuracy on each task
   - Latent space structure (velocity, clustering, RSA)
   - Attention pattern differences

2. **Module Isolation Training**: Train ISW alignment while freezing core model, then compare to joint training.

3. **Adapter Architecture**: Formalize the ISW module as a true adapter with residual connections, allowing explicit interpolation between "with ISW" and "without ISW" behavior.

```python
class ModularHAN(nn.Module):
    """Potential refactoring to make modularity explicit."""

    def __init__(self, core_model, adapters: Dict[str, nn.Module]):
        self.core = core_model  # Frozen or jointly trained
        self.adapters = nn.ModuleDict(adapters)
        self.adapter_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(1.0))
            for name in adapters
        })

    def forward(self, x, active_adapters: List[str] = None):
        core_out = self.core(x)

        if active_adapters is None:
            active_adapters = list(self.adapters.keys())

        for name in active_adapters:
            adapter_out = self.adapters[name](core_out)
            core_out = core_out + self.adapter_weights[name] * adapter_out

        return core_out
```

### Connection to Integration Forcing

This finding is somewhat orthogonal to the cross-source integration problem. The ISW module operates on the *fused* representation, not individual sources. However, it suggests that:

- Integration might be happening in the core 10.7M model
- The ISW module might be learning a separate, parallel representation
- A bottleneck *before* ISW might force better integration

---

## Notes

- These ideas are for future development phases
- Current priority is fixing output collapse and basic temporal learning
- Integration forcing becomes more important with more data/conflicts
- Gaza conflict data could be a good test case for multi-conflict generalization
