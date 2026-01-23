# ISW Embedding Integration Analysis for Multi-Resolution HAN

**Date:** 2026-01-21
**Analyst:** Data Science Analysis
**Context:** Integration of pre-computed Voyage AI embeddings (1024-dim) into HAN time-series prediction model

---

## Executive Summary

Integrating 1024-dimensional ISW embeddings into the existing HAN architecture presents significant statistical and architectural challenges. This analysis provides concrete recommendations for each consideration, with statistical justification based on the specific characteristics of your data (1,315 embeddings covering Feb 2022 - present, 50-100 daily quantitative features).

**Key Recommendations:**
1. Use a two-stream architecture with late fusion, reducing embedding dimension to 64-128 via learned projection
2. Apply t-1 temporal offset for ISW embeddings (report lag alignment)
3. Use learned interpolation with position encoding for missing data, not zero-masking
4. Apply aggressive regularization: PCA to 128 dims, dropout 0.3-0.5, weight decay 1e-3
5. Design ablation studies with proper temporal train/val/test splits and statistical significance testing
6. Validate embeddings through clustering analysis aligned to known conflict events

---

## 1. Feature Scale Imbalance

### Problem Statement

The dimensionality mismatch is severe:
- **Embedding features:** 1024 dimensions
- **Quantitative features:** 50-100 dimensions (across equipment, personnel, deepstate, firms, viina)
- **Ratio:** 10:1 to 20:1 in favor of embeddings

This creates two risks:
1. **Gradient domination:** Embedding features will generate larger gradients simply due to higher dimensionality
2. **Representation capture:** The model may learn to rely primarily on embeddings, ignoring quantitative signals

### Statistical Justification

For a linear combination of features, the expected variance contribution scales with the number of features. If we have:
- `n_emb = 1024` embedding features with variance `sigma_emb^2`
- `n_quant = 75` quantitative features with variance `sigma_quant^2`

Even if per-feature variance is equal, the embedding block contributes ~13x more to total variance.

### Recommended Architecture: Two-Stream with Late Fusion

```
                    ISW Embeddings [1024]          Quantitative Features [75]
                           |                                   |
                    PCA or Learned                      Existing HAN
                    Projection [128]                   Domain Encoders
                           |                                   |
                    Transformer                         Cross-Domain
                    Encoder (2L)                           Fusion
                           |                                   |
                    LayerNorm                            LayerNorm
                           |                                   |
                           +---------------+-------------------+
                                           |
                                    Gated Fusion Layer
                                           |
                                     alpha * E_text + (1-alpha) * E_quant
                                           |
                                   Temporal Encoder
                                           |
                                   Prediction Heads
```

### Concrete Implementation Parameters

```python
# Embedding stream configuration
EMBEDDING_CONFIG = {
    'input_dim': 1024,
    'projection_dim': 128,        # Reduce 1024 -> 128
    'd_model': 64,                # Match HAN d_model
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.3,               # Higher than quantitative stream

    # Normalization
    'pre_norm': True,             # LayerNorm before projection
    'post_norm': True,            # LayerNorm after transformer

    # Fusion
    'fusion_type': 'gated',       # Learned gating vs simple concat
    'gate_init': 0.3,             # Initialize gate to favor quantitative features
}
```

### Normalization Strategy

1. **Pre-normalize embeddings:** Apply LayerNorm before any projection
   - Voyage embeddings are already L2-normalized, but LayerNorm ensures consistent scale

2. **Separate batch statistics:** Do NOT share BatchNorm between streams
   - Each stream has fundamentally different distributions

3. **Post-fusion normalization:** Apply LayerNorm after fusion

```python
class EmbeddingStream(nn.Module):
    def __init__(self, config):
        self.pre_norm = nn.LayerNorm(1024)
        self.projection = nn.Sequential(
            nn.Linear(1024, config['projection_dim']),
            nn.LayerNorm(config['projection_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )
        # Transformer for temporal patterns in text
        self.encoder = nn.TransformerEncoder(...)
        self.post_norm = nn.LayerNorm(config['d_model'])
```

---

## 2. Temporal Alignment Issues

### Problem Statement

ISW reports have inherent publication lag:
- Reports typically analyze the **previous day's** events
- Some reports cover multi-day periods (weekend summaries)
- Publication time varies (typically evening US time)

### Analysis of ISW Date Index

From the embedding date index (`isw_date_index.json`):
- Range: 2022-02-28 to 2025-10-15 (1,315 reports)
- Coverage: ~95% of days have reports
- Missing dates concentrated around holidays (Dec 25, Jan 1)
- Weekend coverage: Present (no systematic weekend gaps)

### Recommended Temporal Offset Strategy

**Primary recommendation: Use t-1 embedding for day t prediction**

```python
def align_embeddings_to_features(embeddings_dict, feature_dates, offset_days=1):
    """
    Align ISW embeddings to quantitative features with temporal offset.

    Args:
        embeddings_dict: {date_str: embedding_array}
        feature_dates: List of dates for quantitative features
        offset_days: Number of days to offset (1 = use previous day's report)

    Returns:
        aligned_embeddings: [n_dates, 1024] array
        coverage_mask: [n_dates] boolean array
    """
    aligned = np.zeros((len(feature_dates), 1024), dtype=np.float32)
    coverage = np.zeros(len(feature_dates), dtype=bool)

    for i, feat_date in enumerate(feature_dates):
        # Look for embedding from offset_days ago
        report_date = feat_date - timedelta(days=offset_days)
        report_key = report_date.strftime('%Y-%m-%d')

        if report_key in embeddings_dict:
            aligned[i] = embeddings_dict[report_key]
            coverage[i] = True
        else:
            # Try adjacent days (handles weekends/holidays)
            for fallback_offset in [0, 2, 3]:
                fallback_date = feat_date - timedelta(days=offset_days + fallback_offset)
                fallback_key = fallback_date.strftime('%Y-%m-%d')
                if fallback_key in embeddings_dict:
                    aligned[i] = embeddings_dict[fallback_key]
                    coverage[i] = True
                    break

    return aligned, coverage
```

### Alternative: Multi-day Context Window

For richer temporal context, consider using embeddings from [t-3, t-2, t-1]:

```python
# Use 3-day context window of ISW reports
context_embeddings = np.stack([
    embeddings[t-3],  # 3 days ago
    embeddings[t-2],  # 2 days ago
    embeddings[t-1],  # 1 day ago (primary)
], axis=1)  # Shape: [batch, 3, 1024]

# Process with attention over context
context_attended = self.temporal_attention(context_embeddings)  # [batch, 1024]
```

### Empirical Validation

Test the temporal offset hypothesis by measuring cross-correlation:

```python
def find_optimal_lag(embeddings, target_feature, max_lag=7):
    """Find lag that maximizes correlation between embedding PC1 and target."""
    pca = PCA(n_components=1)
    emb_pc1 = pca.fit_transform(embeddings).flatten()

    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(emb_pc1[lag:], target_feature[:-lag or None])[0, 1]
        else:
            corr = np.corrcoef(emb_pc1[:lag], target_feature[-lag:])[0, 1]
        correlations.append((lag, corr))

    return sorted(correlations, key=lambda x: abs(x[1]), reverse=True)
```

---

## 3. Missing Data Handling

### Problem Statement

Missing ISW reports occur on:
- December 24-25 (Christmas)
- December 31 - January 1 (New Year)
- Occasional other gaps (2-3 days throughout)

From the date index, missing dates include:
- 2022-05-05, 2022-07-11 (isolated gaps)
- 2022-11-24, 2022-12-24-25, 2023-01-01 (holiday clusters)
- Pattern continues through dataset

### Comparison of Imputation Strategies

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Zero-masking** | Simple, explicit missing indicator | Destroys embedding semantic space; zero is NOT "no information" in embedding space | Not recommended |
| **Nearest neighbor (temporal)** | Preserves semantic content | Assumes adjacent days have similar content; may amplify autocorrelation | Good baseline |
| **Linear interpolation** | Smooth transitions | Embeddings are not linear; interpolated point may be meaningless | Not recommended |
| **Learned imputation** | Model learns optimal fill | Requires additional parameters; risk of overfitting | Best for large gaps |
| **Learned [MASK] token** | Explicit "unknown" representation | Standard in NLP; allows attention to learn to ignore | Recommended |

### Recommended Approach: Learned Mask Token + Positional Encoding

```python
class EmbeddingWithMissing(nn.Module):
    def __init__(self, embedding_dim=1024, d_model=128):
        super().__init__()
        # Learned token for missing embeddings (NOT zero)
        self.missing_token = nn.Parameter(torch.randn(embedding_dim) * 0.02)

        # Binary indicator that embedding is imputed
        self.missing_indicator = nn.Embedding(2, d_model)  # 0=observed, 1=missing

        self.projection = nn.Linear(embedding_dim, d_model)

    def forward(self, embeddings, mask):
        """
        Args:
            embeddings: [batch, seq_len, 1024] - raw embeddings (may have placeholders)
            mask: [batch, seq_len] - True where embedding exists, False where missing

        Returns:
            processed: [batch, seq_len, d_model]
        """
        # Replace missing positions with learned token
        embeddings = embeddings.clone()
        embeddings[~mask] = self.missing_token

        # Project to model dimension
        projected = self.projection(embeddings)

        # Add missing indicator
        missing_indicator = (~mask).long()
        projected = projected + self.missing_indicator(missing_indicator)

        return projected
```

### Additional Strategy: Uncertainty-Weighted Fusion

When embeddings are missing, increase the weight on quantitative features:

```python
class UncertaintyAwareFusion(nn.Module):
    def forward(self, emb_features, quant_features, emb_mask):
        """
        Args:
            emb_features: [batch, seq_len, d_model]
            quant_features: [batch, seq_len, d_model]
            emb_mask: [batch, seq_len] - True where embedding observed
        """
        # Base fusion weights
        alpha = self.gate(torch.cat([emb_features, quant_features], dim=-1))
        alpha = torch.sigmoid(alpha)  # [batch, seq_len, 1]

        # Reduce embedding weight where missing
        alpha = alpha * emb_mask.unsqueeze(-1).float()

        # Fused representation
        fused = alpha * emb_features + (1 - alpha) * quant_features
        return fused
```

---

## 4. Sample Size Concerns

### Problem Statement

This is the most critical challenge:
- **Samples:** 1,315 (1,272 mentioned, but date index shows 1,315)
- **Embedding dimensions:** 1,024
- **Effective parameters in naive approach:** O(1024 * hidden_dim) = O(65,000+)
- **Samples per parameter:** ~0.02 (severe overfitting risk)

Rule of thumb: Need 10-50 samples per parameter for stable training.

### Statistical Analysis: Effective Dimensionality

Voyage embeddings are NOT 1024 independent dimensions. Let's estimate intrinsic dimensionality:

```python
def estimate_intrinsic_dimensionality(embeddings, threshold=0.95):
    """Estimate how many PCs needed to explain threshold variance."""
    pca = PCA()
    pca.fit(embeddings)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= threshold) + 1
    return n_components, pca.explained_variance_ratio_

# Expected result for text embeddings: 50-150 effective dimensions
```

For conflict analysis texts, I estimate:
- **95% variance explained by:** ~100-150 dimensions
- **99% variance explained by:** ~200-300 dimensions
- **Effective dimensionality:** ~100

### Recommended Regularization Strategy

#### 1. Dimensionality Reduction via PCA (Pre-training)

```python
# Fit PCA on training embeddings ONLY (prevent leakage)
train_embeddings = embedding_matrix[train_indices]
pca = PCA(n_components=128, whiten=True)
pca.fit(train_embeddings)

# Transform all embeddings
reduced_embeddings = pca.transform(embedding_matrix)  # [1315, 128]

# Explained variance check
print(f"Variance explained by 128 PCs: {sum(pca.explained_variance_ratio_):.3f}")
# Expected: 0.85-0.95
```

#### 2. Dropout Schedule

```python
# Progressive dropout - higher in early layers, lower in later
embedding_dropout_schedule = {
    'projection': 0.5,      # 50% dropout after projection
    'encoder_layer_1': 0.4,
    'encoder_layer_2': 0.3,
    'fusion': 0.2,
}
```

#### 3. Weight Decay with Differential Rates

```python
# Higher weight decay for embedding parameters
optimizer = torch.optim.AdamW([
    {'params': model.embedding_stream.parameters(), 'weight_decay': 1e-3},
    {'params': model.quantitative_stream.parameters(), 'weight_decay': 1e-4},
    {'params': model.fusion.parameters(), 'weight_decay': 1e-4},
    {'params': model.prediction_heads.parameters(), 'weight_decay': 1e-4},
], lr=1e-4)
```

#### 4. Embedding Freezing Strategy

Consider progressive unfreezing:

```python
# Phase 1: Train only projection layer and fusion (embeddings frozen)
for param in model.embedding_encoder.parameters():
    param.requires_grad = False
model.embedding_projection.requires_grad = True
# Train for N epochs

# Phase 2: Unfreeze embedding encoder with low LR
for param in model.embedding_encoder.parameters():
    param.requires_grad = True
# Use 10x lower LR for embedding encoder
```

#### 5. Data Augmentation for Embeddings

```python
def augment_embedding(embedding, noise_std=0.01, dropout_rate=0.1):
    """Augment embedding during training."""
    # Add small Gaussian noise
    noise = torch.randn_like(embedding) * noise_std
    augmented = embedding + noise

    # Random dimension dropout
    mask = torch.rand_like(embedding) > dropout_rate
    augmented = augmented * mask * (1 / (1 - dropout_rate))

    # Re-normalize (Voyage embeddings are L2-normalized)
    augmented = F.normalize(augmented, p=2, dim=-1)

    return augmented
```

### Regularization Budget Calculation

With 1,315 samples and targeting 10 samples/parameter:
- **Max learnable parameters for embeddings:** ~130
- **PCA projection (1024 -> 128):** Fixed, no learnable params
- **Linear projection (128 -> 64):** 128 * 64 = 8,192 params (too many)
- **Bottleneck projection:** 128 -> 32 -> 64 = 128*32 + 32*64 = 6,144 params (still high)

**Recommendation:** Use PCA reduction to 64 dimensions directly:

```python
# Ultra-conservative approach
pca = PCA(n_components=64)  # Reduces to manageable size
# Then: 64 -> d_model with ~4,096 learnable params
# With 1,315 samples: ~320 samples per parameter (safe)
```

---

## 5. Evaluation Strategy

### Ablation Study Design

#### Hypothesis to Test

**H0:** ISW embeddings do not improve prediction performance beyond quantitative features alone.

**H1:** ISW embeddings provide statistically significant improvement in at least one prediction task.

#### Experimental Conditions

| Condition | Description | Features |
|-----------|-------------|----------|
| **Baseline** | Quantitative only | Equipment, personnel, deepstate, firms, viina |
| **Embedding-only** | ISW only | 1024-dim embeddings (reduced to 64) |
| **Early fusion** | Concatenate before encoding | [quant; embedding] -> single encoder |
| **Late fusion (ours)** | Separate streams + gated fusion | Two encoders + learned gate |
| **Attention fusion** | Cross-attention between modalities | Embeddings attend to quantitative |

#### Temporal Train/Val/Test Split

**Critical:** Must use temporal splits to prevent future leakage.

```python
def temporal_split(n_samples, train_ratio=0.7, val_ratio=0.15, gap_days=14):
    """
    Create temporal splits with gaps to prevent leakage.

    For 1,315 samples:
    - Train: samples 0-920 (Feb 2022 - Sept 2024)
    - Gap: 14 days
    - Val: samples 935-1135 (Oct 2024 - Apr 2025)
    - Gap: 14 days
    - Test: samples 1150-1315 (May 2025 - Oct 2025)
    """
    gap_samples = max(1, gap_days)  # 1 sample per day

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val - 2 * gap_samples

    train_end = n_train
    val_start = train_end + gap_samples
    val_end = val_start + n_val
    test_start = val_end + gap_samples

    return {
        'train': (0, train_end),
        'val': (val_start, val_end),
        'test': (test_start, n_samples)
    }
```

For your data:
- **Train:** 2022-02-28 to 2024-07-31 (~920 samples)
- **Val:** 2024-08-14 to 2025-02-28 (~200 samples)
- **Test:** 2025-03-14 to 2025-10-15 (~195 samples)

#### Statistical Significance Testing

Use paired tests to compare models on the same test set:

```python
from scipy import stats

def compare_models(baseline_errors, treatment_errors, alpha=0.05):
    """
    Compare two models using paired t-test and Wilcoxon signed-rank test.

    Returns both parametric and non-parametric p-values.
    """
    # Paired t-test (assumes normality)
    t_stat, p_ttest = stats.ttest_rel(baseline_errors, treatment_errors)

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, p_wilcoxon = stats.wilcoxon(baseline_errors, treatment_errors)

    # Effect size (Cohen's d for paired samples)
    diff = baseline_errors - treatment_errors
    cohens_d = diff.mean() / diff.std()

    # Bootstrap confidence interval for improvement
    n_bootstrap = 10000
    improvements = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), len(diff), replace=True)
        improvements.append(diff[idx].mean())
    ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])

    return {
        'p_ttest': p_ttest,
        'p_wilcoxon': p_wilcoxon,
        'cohens_d': cohens_d,
        'mean_improvement': diff.mean(),
        'ci_95': (ci_low, ci_high),
        'significant': p_ttest < alpha and p_wilcoxon < alpha
    }
```

#### Metrics by Task

| Task | Primary Metric | Secondary Metrics |
|------|----------------|-------------------|
| **State prediction** | MSE on held-out features | MAE, R^2, per-source MSE |
| **Regime classification** | Macro F1 | Accuracy, confusion matrix |
| **Anomaly detection** | AUROC | AUPRC, F1 at optimal threshold |
| **Forecasting** | RMSE at t+1 | MAPE, directional accuracy |

### Cross-Validation Strategy

Given limited data, use **time-series cross-validation** (expanding window):

```python
from sklearn.model_selection import TimeSeriesSplit

def temporal_cv_splits(n_samples, n_splits=5, min_train_size=200):
    """
    Generate expanding window CV splits.

    Split 1: Train on [0:200], test on [214:300]
    Split 2: Train on [0:300], test on [314:400]
    ...
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=14, test_size=86)

    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if len(train_idx) >= min_train_size:
            yield train_idx, test_idx
```

---

## 6. Embedding Quality Analysis

### Pre-Integration Validation

Before integrating embeddings into the model, verify they capture meaningful conflict dynamics.

#### 6.1 Clustering Analysis

**Hypothesis:** Embeddings should cluster by conflict phase/intensity.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def cluster_embedding_analysis(embeddings, dates, n_clusters_range=(3, 10)):
    """
    Find optimal clustering and analyze temporal patterns.
    """
    # Find optimal k using silhouette score
    silhouettes = []
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels)
        silhouettes.append((k, sil))

    optimal_k = max(silhouettes, key=lambda x: x[1])[0]

    # Fit with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Analyze cluster temporal distribution
    cluster_dates = {}
    for cluster_id in range(optimal_k):
        cluster_mask = labels == cluster_id
        cluster_dates[cluster_id] = [dates[i] for i in range(len(dates)) if cluster_mask[i]]

    return labels, cluster_dates, optimal_k
```

#### 6.2 Event Alignment Analysis

**Hypothesis:** Embedding structure should correlate with known major events.

```python
MAJOR_EVENTS = {
    '2022-02-24': 'Invasion begins',
    '2022-04-08': 'Bucha withdrawal',
    '2022-05-20': 'Mariupol falls',
    '2022-09-11': 'Kharkiv counteroffensive',
    '2022-11-11': 'Kherson liberation',
    '2023-05-21': 'Bakhmut falls',
    '2023-06-06': 'Counteroffensive begins',
    '2024-02-17': 'Avdiivka falls',
    '2024-08-06': 'Kursk incursion',
}

def event_alignment_analysis(embeddings, dates, events_dict, window_days=7):
    """
    Analyze embedding changes around major events.

    Computes:
    1. Embedding distance from pre-event to post-event
    2. Whether event dates cluster together
    3. Correlation of embedding PC1 with event density
    """
    date_to_idx = {d: i for i, d in enumerate(dates)}

    event_changes = []
    for event_date, event_name in events_dict.items():
        event_dt = pd.to_datetime(event_date)

        # Find embeddings around event
        pre_indices = []
        post_indices = []
        for d, idx in date_to_idx.items():
            dt = pd.to_datetime(d)
            delta = (dt - event_dt).days
            if -window_days <= delta < 0:
                pre_indices.append(idx)
            elif 0 < delta <= window_days:
                post_indices.append(idx)

        if pre_indices and post_indices:
            pre_mean = embeddings[pre_indices].mean(axis=0)
            post_mean = embeddings[post_indices].mean(axis=0)

            # Cosine distance
            cos_dist = 1 - np.dot(pre_mean, post_mean) / (
                np.linalg.norm(pre_mean) * np.linalg.norm(post_mean)
            )

            # Euclidean distance
            euc_dist = np.linalg.norm(post_mean - pre_mean)

            event_changes.append({
                'event': event_name,
                'date': event_date,
                'cosine_distance': cos_dist,
                'euclidean_distance': euc_dist
            })

    return pd.DataFrame(event_changes)
```

#### 6.3 Temporal Coherence Analysis

**Hypothesis:** Adjacent days' embeddings should be more similar than distant days.

```python
def temporal_coherence_analysis(embeddings, dates):
    """
    Analyze whether embeddings respect temporal structure.

    Computes:
    1. Autocorrelation of embedding similarity
    2. Decay rate of similarity with temporal distance
    """
    n = len(embeddings)

    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)

    # Group by temporal distance
    distance_sims = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, min(i + 31, n)):  # Up to 30 days
            temporal_dist = j - i
            distance_sims[temporal_dist].append(sim_matrix[i, j])

    # Compute mean similarity at each distance
    decay_curve = []
    for dist in sorted(distance_sims.keys()):
        mean_sim = np.mean(distance_sims[dist])
        std_sim = np.std(distance_sims[dist])
        decay_curve.append({
            'distance_days': dist,
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'n_pairs': len(distance_sims[dist])
        })

    return pd.DataFrame(decay_curve)
```

Expected results for valid embeddings:
- Similarity should decay with temporal distance
- Autocorrelation should be positive at lag 1
- Major events should show higher embedding shifts

#### 6.4 Correlation with Quantitative Features

**Hypothesis:** Embedding principal components should correlate with conflict intensity metrics.

```python
def embedding_feature_correlation(embeddings, quantitative_features, feature_names):
    """
    Compute correlation between embedding PCs and quantitative features.
    """
    # Extract top PCs
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(embeddings)

    correlations = []
    for pc_idx in range(10):
        for feat_idx, feat_name in enumerate(feature_names):
            corr = np.corrcoef(pcs[:, pc_idx], quantitative_features[:, feat_idx])[0, 1]
            correlations.append({
                'pc': f'PC{pc_idx + 1}',
                'feature': feat_name,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })

    return pd.DataFrame(correlations)
```

### Quality Thresholds

Embeddings are likely useful if:
1. **Clustering:** Silhouette score > 0.15 with 4-6 clusters
2. **Event alignment:** Mean cosine distance around events > 0.02
3. **Temporal coherence:** Autocorrelation at lag-1 > 0.7
4. **Feature correlation:** At least one PC correlates > 0.3 with key metrics (personnel losses, equipment losses)

If embeddings fail these thresholds, they may not contain extractable conflict dynamics and integration may not help.

---

## Implementation Checklist

### Phase 1: Embedding Validation (Before Model Integration)
- [ ] Load embedding matrix and date index
- [ ] Compute intrinsic dimensionality via PCA
- [ ] Run clustering analysis and interpret clusters temporally
- [ ] Compute event alignment metrics
- [ ] Compute temporal coherence (autocorrelation decay)
- [ ] Compute correlation with quantitative features
- [ ] Document embedding quality assessment

### Phase 2: Architecture Implementation
- [ ] Implement PCA reduction (fit on train only)
- [ ] Implement embedding projection layer with LayerNorm
- [ ] Implement learned missing token
- [ ] Implement gated fusion module
- [ ] Implement uncertainty-aware fusion (optional)

### Phase 3: Training Setup
- [ ] Create temporal train/val/test splits with gaps
- [ ] Implement proper normalization (train stats only)
- [ ] Set up differential learning rates
- [ ] Implement dropout schedule
- [ ] Implement embedding augmentation (optional)

### Phase 4: Ablation Studies
- [ ] Train baseline model (quantitative only)
- [ ] Train embedding-only model
- [ ] Train early fusion model
- [ ] Train late fusion model (proposed)
- [ ] Compute statistical significance tests
- [ ] Document results with confidence intervals

### Phase 5: Analysis and Interpretation
- [ ] Analyze fusion gate weights over time
- [ ] Interpret which embedding dimensions activate
- [ ] Correlate embedding attention with events
- [ ] Document findings and recommendations

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `/Users/daniel.tipton/ML_OSINT/data/wayback_archives/isw_assessments/embeddings/isw_embedding_matrix.npy` | Raw embeddings [1315, 1024] |
| `/Users/daniel.tipton/ML_OSINT/data/wayback_archives/isw_assessments/embeddings/isw_date_index.json` | Date-to-index mapping |
| `/Users/daniel.tipton/ML_OSINT/analysis/hierarchical_attention_network.py` | Existing HAN architecture |
| `/Users/daniel.tipton/ML_OSINT/analysis/conflict_data_loader.py` | Data loading utilities |
| `/Users/daniel.tipton/ML_OSINT/HAN.md` | Architecture documentation |

---

## References

1. Voyager AI embedding documentation: https://docs.voyageai.com/
2. HAN architecture diagram: `/Users/daniel.tipton/ML_OSINT/HAN.md`
3. ISW embedding generation: `/Users/daniel.tipton/ML_OSINT/data/wayback_archives/generate_isw_embeddings.py`
