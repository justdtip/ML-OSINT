# Omitted Variable Analysis Specification

## Multi-Source Conflict Monitoring Platform
### Empirical Methodology for Identifying Missing Data Sources

**Version:** 1.0  
**Date:** 2026-01-20  
**Purpose:** Guide systematic identification and characterization of omitted variables in cross-source conflict prediction models

---

## 1. Executive Summary

### 1.1 Problem Statement

The hybrid cross-source model achieves r=0.552 mean correlation on masked source prediction, leaving approximately 70% of variance unexplained. When strict temporal controls are applied, same-day correlations disappear, confirming that relationships are lagged rather than contemporaneous. 

This specification defines a methodology to:
1. Characterize the statistical fingerprint of missing variables from model residuals
2. Identify candidate data sources that match those fingerprints
3. Rank candidates by marginal predictive value
4. Guide data acquisition priorities empirically rather than speculatively

### 1.2 Core Insight

Model residuals encode information about omitted variables. Systematic structure in residuals—temporal patterns, cross-source correlations, spectral signatures—reveals the *properties* of missing information even when the missing variables themselves are unknown.

---

## 2. Theoretical Foundation

### 2.1 Residuals as Projected Omitted Variables

Consider a true data-generating process:

```
Y_t = f(X_t, Z_t) + ε_t
```

Where:
- Y_t = target variable (e.g., Equipment losses)
- X_t = observed predictors (FIRMS, UCDP, DeepState features)
- Z_t = unobserved/omitted variable
- ε_t = irreducible noise

When we estimate:

```
Ŷ_t = f̂(X_t)
```

The residual r_t = Y_t - Ŷ_t contains:

```
r_t = g(Z_t) + ε_t + estimation_error
```

If Z_t has systematic structure (temporal autocorrelation, periodicity, correlation with other omitted factors), that structure appears in r_t. The residual is the omitted variable projected onto the outcome space.

### 2.2 Cross-Residual Correlation Theory

For a multi-source model predicting K sources, let r_t^(k) denote the residual when predicting source k. If an omitted variable Z_t affects multiple sources:

```
Cov(r_t^(i), r_t^(j)) ≈ β_i × β_j × Var(Z_t)
```

Where β_i is the effect of Z on source i. Cross-residual correlation reveals:
- Existence of shared omitted factors
- Relative sensitivity of each source to the omitted factor
- Temporal dynamics of the omitted factor (via lagged cross-correlation)

### 2.3 Spectral Characterization

The power spectral density of residuals reveals missing periodic components:

```
S_r(f) = |ℱ(r_t)|²
```

Peaks at frequency f indicate missing variables with period 1/f. Common interpretations:
- f ≈ 0.14 (7-day period): Weekly operational/reporting cycles
- f ≈ 0.033 (30-day period): Monthly reporting, lunar cycles
- f ≈ 0.0027 (365-day period): Seasonal effects

### 2.4 Sensitivity Analysis Framework

Following Cinelli & Hazlett (2020), for an observed relationship between X and Y with coefficient β̂, the bias from omitting Z is bounded by:

```
|bias| ≤ √(R²_{Y~Z|X} × R²_{X~Z}) × (σ_Y / σ_X)
```

Where:
- R²_{Y~Z|X} = partial R² of Z on Y controlling for X
- R²_{X~Z} = R² of Z on X

This bounds how strong an omitted variable must be to explain away observed relationships, providing a specification for the "missing variable" in terms of required effect sizes.

### 2.5 Partial Information Decomposition

For predictors X₁, X₂ and target Y, total mutual information I(Y; X₁, X₂) decomposes into:

```
I(Y; X₁, X₂) = Unique(X₁) + Unique(X₂) + Redundancy + Synergy
```

- **Unique(X₁)**: Information only X₁ provides
- **Unique(X₂)**: Information only X₂ provides  
- **Redundancy**: Information both provide (overlapping signal)
- **Synergy**: Information available only from combination

The gap between total variance and explained variance, decomposed this way, indicates whether missing information is:
- **Reducible**: Would correlate with existing sources (get better data of same type)
- **Orthogonal**: Requires genuinely new data domains

### 2.6 Granger Causality for Candidate Screening

Variable Z Granger-causes residual r if:

```
E[r_t | r_{t-1}, ..., r_{t-p}, Z_{t-1}, ..., Z_{t-p}] ≠ E[r_t | r_{t-1}, ..., r_{t-p}]
```

Operationally: regress r_t on its own lags plus lags of Z, test whether Z coefficients are jointly significant (F-test or likelihood ratio).

This directly tests: "Does knowing Z at time t-k help predict the part of Y that existing sources couldn't predict?"

---

## 3. Implementation Pipeline

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OMITTED VARIABLE ANALYSIS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Stage 1    │───▶│   Stage 2    │───▶│   Stage 3    │       │
│  │  Residual    │    │  Temporal    │    │   Factor     │       │
│  │  Extraction  │    │  Analysis    │    │  Extraction  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Stage 4    │◀───│   Stage 5    │◀───│   Stage 6    │       │
│  │  Candidate   │    │   Granger    │    │   Ranking &  │       │
│  │  Correlation │    │  Causality   │    │   Report     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage 1: Residual Extraction

#### 3.2.1 Objective
Extract prediction residuals from the trained hybrid model for each source under both full-reconstruction and masked-reconstruction conditions.

#### 3.2.2 Inputs
- Trained hybrid model checkpoint
- Aligned evaluation dataset (416 samples)
- Source names: ['deepstate', 'equipment', 'firms', 'ucdp']

#### 3.2.3 Procedure

```python
def extract_residuals(model, dataloader, source_names):
    """
    Extract residuals for full and masked reconstruction.
    
    Returns:
        full_residuals: dict[source_name] -> np.array shape (n_samples, n_features)
        masked_residuals: dict[source_name] -> np.array shape (n_samples, n_features)
        timestamps: np.array of datetime objects
    """
    
    full_residuals = {s: [] for s in source_names}
    masked_residuals = {s: [] for s in source_names}
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Full reconstruction
            inputs = {s: batch[s] for s in source_names}
            outputs = model(inputs)
            
            for source in source_names:
                residual = batch[source] - outputs[source]
                full_residuals[source].append(residual.numpy())
            
            # Masked reconstruction (one source at a time)
            for masked_source in source_names:
                masked_inputs = {s: batch[s] for s in source_names if s != masked_source}
                masked_inputs[masked_source] = torch.zeros_like(batch[masked_source])
                
                masked_outputs = model(masked_inputs, mask=[masked_source])
                residual = batch[masked_source] - masked_outputs[masked_source]
                masked_residuals[masked_source].append(residual.numpy())
    
    # Concatenate batches
    full_residuals = {s: np.concatenate(v, axis=0) for s, v in full_residuals.items()}
    masked_residuals = {s: np.concatenate(v, axis=0) for s, v in masked_residuals.items()}
    
    return full_residuals, masked_residuals
```

#### 3.2.4 Outputs
- `full_residuals.npz`: Per-source residuals from full reconstruction
- `masked_residuals.npz`: Per-source residuals from masked reconstruction
- `residual_metadata.json`: Feature names, timestamps, reconstruction MSE per source

#### 3.2.5 Validation Checks
- Residuals should have mean ≈ 0 (unbiased predictions)
- Residual variance should be less than input variance (model explains something)
- No NaN or Inf values

---

### 3.3 Stage 2: Temporal Structure Analysis

#### 3.3.1 Objective
Characterize temporal patterns in residuals: autocorrelation, cross-correlation, and spectral density.

#### 3.3.2 Inputs
- Residuals from Stage 1
- Max lag for correlation analysis (default: 60 days)
- Spectral analysis parameters

#### 3.3.3 Procedure

##### 3.3.3.1 Residual Autocorrelation

```python
def compute_residual_autocorrelation(residuals, max_lag=60):
    """
    Compute autocorrelation function for aggregated residuals.
    
    For each source, compute ACF of the mean residual magnitude
    (L2 norm across features per timestep).
    
    Returns:
        acf_results: dict[source] -> {
            'lags': np.array,
            'acf': np.array,
            'confidence_interval': tuple,
            'significant_lags': list of ints
        }
    """
    from statsmodels.tsa.stattools import acf
    
    acf_results = {}
    for source, res in residuals.items():
        # Aggregate to scalar time series: L2 norm of residual vector
        residual_magnitude = np.linalg.norm(res, axis=1)
        
        # Compute ACF with confidence intervals
        acf_values, confint = acf(residual_magnitude, nlags=max_lag, 
                                   alpha=0.05, fft=True)
        
        # Identify significant lags (outside 95% CI)
        ci_lower, ci_upper = confint[1:, 0] - acf_values[1:], confint[1:, 1] - acf_values[1:]
        significant = np.where(np.abs(acf_values[1:]) > np.abs(ci_upper))[0] + 1
        
        acf_results[source] = {
            'lags': np.arange(max_lag + 1),
            'acf': acf_values,
            'confidence_interval': (confint[:, 0], confint[:, 1]),
            'significant_lags': significant.tolist()
        }
    
    return acf_results
```

##### 3.3.3.2 Cross-Residual Correlation

```python
def compute_cross_residual_correlation(residuals, max_lag=60):
    """
    Compute lagged cross-correlation between residuals of different sources.
    
    Returns:
        xcorr_results: dict[(source_i, source_j)] -> {
            'lags': np.array,
            'correlation': np.array,
            'peak_lag': int,
            'peak_correlation': float,
            'zero_lag_correlation': float
        }
    """
    from scipy import signal
    
    sources = list(residuals.keys())
    xcorr_results = {}
    
    for i, src_i in enumerate(sources):
        for j, src_j in enumerate(sources):
            if i >= j:
                continue
            
            # Aggregate to scalar
            mag_i = np.linalg.norm(residuals[src_i], axis=1)
            mag_j = np.linalg.norm(residuals[src_j], axis=1)
            
            # Normalize
            mag_i = (mag_i - mag_i.mean()) / mag_i.std()
            mag_j = (mag_j - mag_j.mean()) / mag_j.std()
            
            # Cross-correlation
            xcorr = signal.correlate(mag_i, mag_j, mode='full') / len(mag_i)
            lags = signal.correlation_lags(len(mag_i), len(mag_j), mode='full')
            
            # Trim to max_lag
            center = len(lags) // 2
            trim_slice = slice(center - max_lag, center + max_lag + 1)
            lags = lags[trim_slice]
            xcorr = xcorr[trim_slice]
            
            peak_idx = np.argmax(np.abs(xcorr))
            
            xcorr_results[(src_i, src_j)] = {
                'lags': lags,
                'correlation': xcorr,
                'peak_lag': int(lags[peak_idx]),
                'peak_correlation': float(xcorr[peak_idx]),
                'zero_lag_correlation': float(xcorr[lags == 0][0])
            }
    
    return xcorr_results
```

##### 3.3.3.3 Spectral Analysis

```python
def compute_residual_spectra(residuals, sampling_rate=1.0):
    """
    Compute power spectral density of residuals using Welch's method.
    
    Args:
        residuals: dict of residual arrays
        sampling_rate: samples per day (1.0 for daily data)
    
    Returns:
        spectral_results: dict[source] -> {
            'frequencies': np.array (cycles per day),
            'psd': np.array,
            'dominant_periods': list of (period_days, power) tuples,
            'period_interpretation': dict mapping periods to likely causes
        }
    """
    from scipy import signal
    
    # Known periodicities to check
    KNOWN_PERIODS = {
        7: 'weekly_cycle',
        14: 'biweekly_cycle',
        30: 'monthly_cycle',
        90: 'quarterly_cycle',
        365: 'annual_cycle'
    }
    
    spectral_results = {}
    
    for source, res in residuals.items():
        # Aggregate to scalar
        residual_magnitude = np.linalg.norm(res, axis=1)
        
        # Welch's method for PSD estimation
        # nperseg chosen for reasonable frequency resolution with limited data
        nperseg = min(256, len(residual_magnitude) // 4)
        frequencies, psd = signal.welch(residual_magnitude, fs=sampling_rate,
                                         nperseg=nperseg, noverlap=nperseg//2)
        
        # Convert to periods (days)
        periods = 1 / (frequencies + 1e-10)
        
        # Find dominant periods (local maxima in PSD)
        peak_indices, properties = signal.find_peaks(psd, height=np.median(psd))
        dominant = sorted(
            [(periods[i], psd[i]) for i in peak_indices],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Match to known periodicities
        interpretations = {}
        for period, power in dominant:
            for known_period, label in KNOWN_PERIODS.items():
                if 0.8 * known_period <= period <= 1.2 * known_period:
                    interpretations[f'{period:.1f}_days'] = label
                    break
            else:
                interpretations[f'{period:.1f}_days'] = 'unknown'
        
        spectral_results[source] = {
            'frequencies': frequencies,
            'psd': psd,
            'dominant_periods': dominant,
            'period_interpretation': interpretations
        }
    
    return spectral_results
```

#### 3.3.4 Outputs
- `temporal_analysis.json`:
  - Autocorrelation results per source
  - Cross-residual correlation matrix with lags
  - Spectral analysis with dominant periods
- `temporal_analysis_plots/`:
  - ACF plots per source
  - Cross-correlation heatmaps
  - Power spectral density plots

#### 3.3.5 Interpretation Guidelines

| Pattern | Interpretation | Likely Missing Variable |
|---------|---------------|------------------------|
| Strong ACF at lag 7 | Weekly periodicity | Reporting cycle, operational tempo |
| Strong ACF at lag 30 | Monthly periodicity | Monthly aggregated source, lunar effects |
| Cross-residual correlation at lag 0 | Shared instantaneous factor | Weather, coordinated events |
| Cross-residual correlation at lag k | Lagged shared factor | Slow-propagating effects |
| 1/f spectral slope | Long-memory process | Cumulative/persistent factor |
| Flat spectrum | White noise | Irreducible randomness |

---

### 3.4 Stage 3: Latent Factor Extraction

#### 3.4.1 Objective
Extract latent factors from the cross-source residual covariance matrix. These factors represent empirical "omitted variables" that affect multiple sources.

#### 3.4.2 Inputs
- Residuals from Stage 1 (masked reconstruction preferred)
- Number of factors to extract (default: 5)

#### 3.4.3 Procedure

##### 3.4.3.1 Construct Cross-Source Residual Matrix

```python
def construct_residual_matrix(residuals):
    """
    Stack residuals from all sources into a single matrix.
    
    Returns:
        R: np.array shape (n_samples, sum of all feature dimensions)
        feature_info: list of (source, feature_idx, feature_name) tuples
    """
    sources = list(residuals.keys())
    
    # Stack horizontally
    R = np.hstack([residuals[s] for s in sources])
    
    # Track which features belong to which source
    feature_info = []
    col_idx = 0
    for source in sources:
        n_features = residuals[source].shape[1]
        for i in range(n_features):
            feature_info.append((source, i, f'{source}_feat_{i}'))
        col_idx += n_features
    
    return R, feature_info
```

##### 3.4.3.2 Factor Analysis

```python
def extract_residual_factors(residuals, n_factors=5, method='pca'):
    """
    Extract latent factors from residual covariance structure.
    
    Args:
        residuals: dict of residual arrays
        n_factors: number of factors to extract
        method: 'pca' or 'factor_analysis'
    
    Returns:
        factors: dict containing:
            'scores': np.array shape (n_samples, n_factors) - factor time series
            'loadings': np.array shape (n_features, n_factors) - factor loadings
            'variance_explained': np.array - variance explained per factor
            'source_loadings': dict[source] -> np.array of mean abs loading per factor
            'feature_info': list of (source, idx, name) tuples
    """
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    
    R, feature_info = construct_residual_matrix(residuals)
    
    # Standardize
    scaler = StandardScaler()
    R_scaled = scaler.fit_transform(R)
    
    # Extract factors
    if method == 'pca':
        model = PCA(n_components=n_factors)
        scores = model.fit_transform(R_scaled)
        loadings = model.components_.T  # (n_features, n_factors)
        variance_explained = model.explained_variance_ratio_
    else:
        model = FactorAnalysis(n_components=n_factors, random_state=42)
        scores = model.fit_transform(R_scaled)
        loadings = model.components_.T
        # Approximate variance explained
        total_var = R_scaled.var(axis=0).sum()
        factor_var = np.var(scores, axis=0)
        variance_explained = factor_var / total_var
    
    # Compute mean absolute loading per source per factor
    sources = list(residuals.keys())
    source_loadings = {s: np.zeros(n_factors) for s in sources}
    
    for idx, (source, feat_idx, feat_name) in enumerate(feature_info):
        for f in range(n_factors):
            source_loadings[source][f] += np.abs(loadings[idx, f])
    
    # Normalize by number of features per source
    feature_counts = {s: residuals[s].shape[1] for s in sources}
    for s in sources:
        source_loadings[s] /= feature_counts[s]
    
    return {
        'scores': scores,
        'loadings': loadings,
        'variance_explained': variance_explained,
        'source_loadings': source_loadings,
        'feature_info': feature_info,
        'method': method
    }
```

##### 3.4.3.3 Factor Characterization

```python
def characterize_factors(factor_results, residuals, timestamps):
    """
    Characterize each extracted factor:
    - Which sources it most affects
    - Temporal dynamics
    - Potential interpretations
    
    Returns:
        characterization: list of dicts, one per factor
    """
    from scipy import stats
    
    n_factors = factor_results['scores'].shape[1]
    sources = list(residuals.keys())
    characterizations = []
    
    for f in range(n_factors):
        factor_scores = factor_results['scores'][:, f]
        
        # Source sensitivity (which sources does this factor affect most?)
        source_sensitivity = {
            s: factor_results['source_loadings'][s][f] 
            for s in sources
        }
        dominant_source = max(source_sensitivity, key=source_sensitivity.get)
        
        # Temporal properties
        autocorr_lag1 = np.corrcoef(factor_scores[:-1], factor_scores[1:])[0, 1]
        
        # Trend (Spearman correlation with time)
        time_indices = np.arange(len(factor_scores))
        trend_corr, trend_pval = stats.spearmanr(time_indices, factor_scores)
        
        # Volatility clustering (autocorrelation of squared scores)
        squared_scores = factor_scores ** 2
        volatility_persistence = np.corrcoef(squared_scores[:-1], squared_scores[1:])[0, 1]
        
        characterizations.append({
            'factor_index': f,
            'variance_explained': float(factor_results['variance_explained'][f]),
            'source_sensitivity': source_sensitivity,
            'dominant_source': dominant_source,
            'autocorrelation_lag1': float(autocorr_lag1),
            'trend_correlation': float(trend_corr),
            'trend_pvalue': float(trend_pval),
            'volatility_persistence': float(volatility_persistence),
            'interpretation': interpret_factor(
                source_sensitivity, autocorr_lag1, trend_corr, volatility_persistence
            )
        })
    
    return characterizations


def interpret_factor(source_sensitivity, autocorr, trend, volatility):
    """
    Generate interpretation based on factor properties.
    """
    interpretations = []
    
    # Source pattern
    sensitivities = list(source_sensitivity.values())
    if max(sensitivities) > 2 * np.median(sensitivities):
        dominant = max(source_sensitivity, key=source_sensitivity.get)
        interpretations.append(f'primarily_affects_{dominant}')
    else:
        interpretations.append('affects_multiple_sources_equally')
    
    # Temporal pattern
    if autocorr > 0.7:
        interpretations.append('highly_persistent')
    elif autocorr > 0.3:
        interpretations.append('moderately_persistent')
    else:
        interpretations.append('transient')
    
    # Trend
    if abs(trend) > 0.3:
        direction = 'increasing' if trend > 0 else 'decreasing'
        interpretations.append(f'trending_{direction}')
    
    # Volatility
    if volatility > 0.5:
        interpretations.append('volatility_clustering')
    
    return interpretations
```

#### 3.4.4 Outputs
- `latent_factors.npz`:
  - Factor scores (time series of each factor)
  - Factor loadings (feature weights)
  - Variance explained
- `factor_characterization.json`:
  - Per-factor source sensitivity
  - Temporal properties
  - Interpretations
- `factor_plots/`:
  - Factor score time series
  - Loading heatmaps by source
  - Source sensitivity radar charts

---

### 3.5 Stage 4: Candidate Variable Correlation

#### 3.5.1 Objective
Correlate extracted factors and raw residuals against candidate external variables to identify what the omitted variables might represent.

#### 3.5.2 Inputs
- Factor scores from Stage 3
- Aggregated residuals from Stage 1
- Candidate variable datasets (see 3.5.3)

#### 3.5.3 Candidate Variable Categories

```python
CANDIDATE_VARIABLES = {
    'temporal': {
        'description': 'Time-based patterns',
        'variables': [
            'day_of_week',           # 0-6, Monday=0
            'day_of_month',          # 1-31
            'month',                 # 1-12
            'days_since_start',      # Trend proxy
            'is_weekend',            # Binary
            'is_month_start',        # First 5 days
            'is_month_end',          # Last 5 days
        ],
        'source': 'computed_from_timestamps'
    },
    
    'meteorological': {
        'description': 'Weather conditions affecting operations',
        'variables': [
            'temperature_mean',      # Daily mean temp (Celsius)
            'temperature_range',     # Daily max - min
            'precipitation',         # mm
            'cloud_cover',           # Fraction 0-1
            'visibility',            # km
            'wind_speed',            # m/s
            'snow_depth',            # cm
        ],
        'source': 'ERA5 reanalysis or Open-Meteo API',
        'spatial_aggregation': 'Mean over Ukraine bounding box or conflict regions'
    },
    
    'lunar': {
        'description': 'Night operations visibility',
        'variables': [
            'moon_phase',            # 0-1, 0=new, 0.5=full
            'moon_illumination',     # Fraction visible
            'moonrise_time',         # Hours after midnight
            'moonset_time',          # Hours after midnight
            'night_hours',           # Hours of darkness
        ],
        'source': 'Astronomical calculations (ephem or astropy)'
    },
    
    'news_volume': {
        'description': 'Media attention proxy',
        'variables': [
            'ukraine_news_count',    # Articles mentioning Ukraine
            'russia_news_count',     # Articles mentioning Russia
            'conflict_keywords',     # Count of conflict-related terms
            'region_mentions',       # Counts per oblast
        ],
        'source': 'GDELT or NewsAPI',
        'note': 'Lagged relationship expected (events -> news)'
    },
    
    'economic': {
        'description': 'Economic indicators',
        'variables': [
            'ruble_usd',             # Exchange rate
            'brent_crude',           # Oil price
            'wheat_futures',         # Agricultural proxy
            'natural_gas_price',     # Energy proxy
        ],
        'source': 'Yahoo Finance or FRED',
        'note': 'May have weekly/daily gaps, requires interpolation'
    },
    
    'connectivity': {
        'description': 'Infrastructure status',
        'variables': [
            'ukraine_bgp_prefixes',  # Announced routes
            'regional_latency',      # Ping times to Ukraine
            'cloudflare_traffic',    # Relative traffic volume
        ],
        'source': 'RIPE Atlas, Cloudflare Radar',
        'note': 'Near-real-time availability'
    }
}
```

#### 3.5.4 Procedure

```python
def correlate_with_candidates(factor_scores, residuals, candidates, timestamps):
    """
    Compute correlation between factors/residuals and candidate variables.
    
    Args:
        factor_scores: np.array (n_samples, n_factors)
        residuals: dict[source] -> np.array (n_samples, n_features)
        candidates: dict[category] -> pd.DataFrame with timestamp index
        timestamps: array of datetime objects
    
    Returns:
        correlation_results: dict containing:
            'factor_correlations': dict[candidate_var] -> list of correlations per factor
            'residual_correlations': dict[source] -> dict[candidate_var] -> correlation
            'significant_pairs': list of (factor/source, candidate, correlation, pvalue)
    """
    from scipy import stats
    
    results = {
        'factor_correlations': {},
        'residual_correlations': {s: {} for s in residuals.keys()},
        'significant_pairs': []
    }
    
    # Align candidate data with model timestamps
    for category, candidate_df in candidates.items():
        for var_name in candidate_df.columns:
            # Align by timestamp
            aligned_values = align_to_timestamps(candidate_df[var_name], timestamps)
            
            if aligned_values is None or np.isnan(aligned_values).sum() > len(aligned_values) * 0.2:
                continue  # Skip if too much missing data
            
            # Fill remaining NaNs with interpolation
            aligned_values = interpolate_gaps(aligned_values)
            
            # Correlate with factors
            factor_corrs = []
            for f in range(factor_scores.shape[1]):
                corr, pval = stats.pearsonr(factor_scores[:, f], aligned_values)
                factor_corrs.append({'correlation': corr, 'pvalue': pval})
                
                if pval < 0.05:
                    results['significant_pairs'].append({
                        'type': 'factor',
                        'index': f,
                        'candidate': f'{category}/{var_name}',
                        'correlation': float(corr),
                        'pvalue': float(pval)
                    })
            
            results['factor_correlations'][f'{category}/{var_name}'] = factor_corrs
            
            # Correlate with aggregated residuals
            for source, res in residuals.items():
                res_magnitude = np.linalg.norm(res, axis=1)
                corr, pval = stats.pearsonr(res_magnitude, aligned_values)
                results['residual_correlations'][source][f'{category}/{var_name}'] = {
                    'correlation': float(corr),
                    'pvalue': float(pval)
                }
                
                if pval < 0.05:
                    results['significant_pairs'].append({
                        'type': 'residual',
                        'source': source,
                        'candidate': f'{category}/{var_name}',
                        'correlation': float(corr),
                        'pvalue': float(pval)
                    })
    
    # Sort significant pairs by absolute correlation
    results['significant_pairs'].sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results
```

#### 3.5.5 Outputs
- `candidate_correlations.json`:
  - Factor-candidate correlation matrix
  - Residual-candidate correlation matrix
  - Ranked significant pairs
- `candidate_correlation_plots/`:
  - Heatmaps of correlation matrices
  - Scatter plots for top correlations

---

### 3.6 Stage 5: Granger Causality Testing

#### 3.6.1 Objective
For candidate variables showing significant correlation, test whether they Granger-cause the residuals (i.e., provide predictive information beyond the residuals' own history).

#### 3.6.2 Inputs
- Residuals from Stage 1
- Candidate variables from Stage 4
- Significant pairs from Stage 4
- Max lag for testing (default: 30 days)

#### 3.6.3 Procedure

```python
def granger_causality_test(residuals, candidates, significant_pairs, max_lag=30):
    """
    Test Granger causality from candidates to residuals.
    
    For each significant pair, test whether knowing the candidate variable's
    past helps predict the residual beyond the residual's own past.
    
    Returns:
        granger_results: list of dicts with test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    results = []
    
    for pair in significant_pairs:
        if pair['type'] == 'factor':
            continue  # Granger test on raw residuals, not factors
        
        source = pair['source']
        candidate_name = pair['candidate']
        
        # Get residual magnitude time series
        res_magnitude = np.linalg.norm(residuals[source], axis=1)
        
        # Get candidate time series
        category, var_name = candidate_name.split('/')
        candidate_values = candidates[category][var_name].values
        
        # Ensure same length and no NaNs
        if len(candidate_values) != len(res_magnitude):
            continue
        
        mask = ~(np.isnan(res_magnitude) | np.isnan(candidate_values))
        if mask.sum() < 50:  # Minimum samples
            continue
        
        # Stack for Granger test: [residual, candidate]
        data = np.column_stack([res_magnitude[mask], candidate_values[mask]])
        
        try:
            # Test multiple lags
            gc_results = grangercausalitytests(data, maxlag=min(max_lag, len(data)//5), 
                                                verbose=False)
            
            # Find best lag (lowest p-value)
            best_lag = None
            best_pvalue = 1.0
            f_stats = []
            
            for lag, result in gc_results.items():
                pval = result[0]['ssr_ftest'][1]
                f_stat = result[0]['ssr_ftest'][0]
                f_stats.append({'lag': lag, 'f_stat': f_stat, 'pvalue': pval})
                
                if pval < best_pvalue:
                    best_pvalue = pval
                    best_lag = lag
            
            results.append({
                'source': source,
                'candidate': candidate_name,
                'correlation': pair['correlation'],
                'granger_best_lag': best_lag,
                'granger_best_pvalue': best_pvalue,
                'granger_significant': best_pvalue < 0.05,
                'all_lags': f_stats
            })
            
        except Exception as e:
            results.append({
                'source': source,
                'candidate': candidate_name,
                'error': str(e)
            })
    
    # Sort by Granger significance
    results.sort(key=lambda x: x.get('granger_best_pvalue', 1.0))
    
    return results
```

#### 3.6.4 Outputs
- `granger_results.json`:
  - Per-pair Granger test results
  - Best lag for each significant pair
  - Ranked by Granger p-value
- `granger_plots/`:
  - F-statistic vs lag plots
  - Summary table of significant Granger relationships

---

### 3.7 Stage 6: Ranking and Reporting

#### 3.7.1 Objective
Synthesize all analyses into a ranked list of recommended data sources with justification.

#### 3.7.2 Procedure

```python
def generate_recommendations(
    temporal_analysis,
    factor_characterization,
    candidate_correlations,
    granger_results
):
    """
    Generate ranked recommendations for data source additions.
    
    Scoring criteria:
    1. Factor correlation strength (how much variance might it explain?)
    2. Granger causality (does it provide predictive information?)
    3. Interpretability (does the relationship make domain sense?)
    4. Data availability (can we actually get this data?)
    
    Returns:
        recommendations: list of dicts, ranked by composite score
    """
    
    # Build candidate scores
    candidate_scores = {}
    
    # Score from correlations
    for candidate, factor_corrs in candidate_correlations['factor_correlations'].items():
        max_corr = max(abs(fc['correlation']) for fc in factor_corrs)
        avg_corr = np.mean([abs(fc['correlation']) for fc in factor_corrs])
        
        if candidate not in candidate_scores:
            candidate_scores[candidate] = {
                'correlation_score': 0,
                'granger_score': 0,
                'details': {}
            }
        
        candidate_scores[candidate]['correlation_score'] = max_corr * 0.7 + avg_corr * 0.3
        candidate_scores[candidate]['details']['max_factor_correlation'] = max_corr
    
    # Score from Granger tests
    for gr in granger_results:
        if 'error' in gr:
            continue
        candidate = gr['candidate']
        if candidate in candidate_scores:
            if gr['granger_significant']:
                # Score based on F-stat magnitude and lag reasonableness
                f_stat = gr['all_lags'][gr['granger_best_lag'] - 1]['f_stat']
                lag_penalty = 1.0 / (1 + gr['granger_best_lag'] / 10)  # Prefer shorter lags
                candidate_scores[candidate]['granger_score'] = min(f_stat / 10, 1.0) * lag_penalty
                candidate_scores[candidate]['details']['granger_lag'] = gr['granger_best_lag']
                candidate_scores[candidate]['details']['granger_pvalue'] = gr['granger_best_pvalue']
    
    # Composite score
    for candidate in candidate_scores:
        cs = candidate_scores[candidate]
        cs['composite_score'] = cs['correlation_score'] * 0.5 + cs['granger_score'] * 0.5
    
    # Convert to ranked list
    recommendations = [
        {
            'candidate': candidate,
            'category': candidate.split('/')[0],
            'variable': candidate.split('/')[1],
            **scores
        }
        for candidate, scores in candidate_scores.items()
    ]
    
    recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Add data availability notes
    DATA_AVAILABILITY = {
        'temporal': {'difficulty': 'trivial', 'cost': 'free', 'latency': 'none'},
        'meteorological': {'difficulty': 'easy', 'cost': 'free', 'latency': 'hours'},
        'lunar': {'difficulty': 'trivial', 'cost': 'free', 'latency': 'none'},
        'news_volume': {'difficulty': 'moderate', 'cost': 'low', 'latency': 'hours'},
        'economic': {'difficulty': 'easy', 'cost': 'free', 'latency': 'daily'},
        'connectivity': {'difficulty': 'moderate', 'cost': 'free', 'latency': 'hourly'},
    }
    
    for rec in recommendations:
        category = rec['category']
        if category in DATA_AVAILABILITY:
            rec['availability'] = DATA_AVAILABILITY[category]
    
    return recommendations


def generate_final_report(
    recommendations,
    temporal_analysis,
    factor_characterization,
    output_path
):
    """
    Generate comprehensive markdown report.
    """
    
    report = [
        "# Omitted Variable Analysis Report\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        "## Executive Summary\n\n",
        "This analysis characterizes the structure of prediction residuals to identify ",
        "candidate data sources that would improve model performance.\n\n",
        "### Top Recommended Data Additions\n\n",
        "| Rank | Variable | Score | Correlation | Granger | Availability |\n",
        "|------|----------|-------|-------------|---------|-------------|\n"
    ]
    
    for i, rec in enumerate(recommendations[:10]):
        avail = rec.get('availability', {}).get('difficulty', 'unknown')
        report.append(
            f"| {i+1} | {rec['variable']} | {rec['composite_score']:.3f} | "
            f"{rec['details'].get('max_factor_correlation', 0):.3f} | "
            f"{rec['granger_score']:.3f} | {avail} |\n"
        )
    
    report.extend([
        "\n## Residual Temporal Structure\n\n",
        "### Significant Periodicities Detected\n\n"
    ])
    
    for source, spectral in temporal_analysis['spectral'].items():
        if spectral['dominant_periods']:
            report.append(f"**{source}:**\n")
            for period, power in spectral['dominant_periods'][:3]:
                interp = spectral['period_interpretation'].get(f'{period:.1f}_days', 'unknown')
                report.append(f"- {period:.1f} day period (power={power:.2f}, likely: {interp})\n")
            report.append("\n")
    
    report.extend([
        "\n## Latent Factor Characterization\n\n"
    ])
    
    for char in factor_characterization:
        report.append(f"### Factor {char['factor_index'] + 1}\n")
        report.append(f"- Variance explained: {char['variance_explained']*100:.1f}%\n")
        report.append(f"- Dominant source: {char['dominant_source']}\n")
        report.append(f"- Persistence: {char['autocorrelation_lag1']:.2f}\n")
        report.append(f"- Interpretation: {', '.join(char['interpretation'])}\n\n")
    
    report.extend([
        "\n## Methodology Notes\n\n",
        "This analysis follows the omitted variable characterization framework:\n",
        "1. Extract residuals from trained model\n",
        "2. Analyze temporal structure (ACF, spectral)\n",
        "3. Extract latent factors from cross-residual covariance\n",
        "4. Correlate with candidate variables\n",
        "5. Test Granger causality for predictive value\n",
        "6. Rank candidates by composite score\n"
    ])
    
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    return ''.join(report)
```

#### 3.7.3 Outputs
- `omitted_variable_report.md`: Human-readable summary report
- `recommendations.json`: Machine-readable ranked recommendations
- `full_analysis_results.json`: Complete results from all stages

---

## 4. Visualization Specifications

### 4.1 Required Figures

| Figure | Description | Purpose |
|--------|-------------|---------|
| `residual_acf_grid.png` | ACF plots for each source | Show temporal persistence in residuals |
| `cross_residual_heatmap.png` | Cross-correlation at multiple lags | Reveal shared omitted factors |
| `spectral_density.png` | PSD plots per source | Identify missing periodicities |
| `factor_loadings_heatmap.png` | Factor loadings by source | Show which sources each factor affects |
| `factor_timeseries.png` | Factor scores over time | Visualize omitted variable dynamics |
| `candidate_correlation_matrix.png` | Factor × Candidate correlation | Identify what factors might represent |
| `granger_summary.png` | Granger F-stats by lag | Show predictive relationships |
| `recommendation_ranking.png` | Bar chart of composite scores | Summarize recommendations |

### 4.2 Style Guidelines

```python
PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
}

COLOR_PALETTE = {
    'deepstate': '#2ecc71',
    'equipment': '#3498db', 
    'firms': '#e74c3c',
    'ucdp': '#9b59b6',
    'factor': '#34495e',
    'significant': '#e74c3c',
    'non_significant': '#bdc3c7'
}
```

---

## 5. Integration Notes

### 5.1 Dependencies

```
numpy>=1.21
scipy>=1.7
pandas>=1.3
statsmodels>=0.13
scikit-learn>=1.0
matplotlib>=3.5
seaborn>=0.11
torch>=1.10  # For model loading
```

### 5.2 File Structure

```
omitted_variable_analysis/
├── config.yaml                    # Analysis parameters
├── run_analysis.py               # Main entry point
├── stages/
│   ├── __init__.py
│   ├── residual_extraction.py    # Stage 1
│   ├── temporal_analysis.py      # Stage 2
│   ├── factor_extraction.py      # Stage 3
│   ├── candidate_correlation.py  # Stage 4
│   ├── granger_testing.py        # Stage 5
│   └── reporting.py              # Stage 6
├── data/
│   ├── candidates/               # External candidate data
│   └── model_outputs/            # Residuals and factors
├── outputs/
│   ├── figures/                  # All generated plots
│   ├── results/                  # JSON/NPZ results
│   └── reports/                  # Markdown reports
└── tests/
    └── test_stages.py            # Unit tests
```

### 5.3 Configuration Schema

```yaml
# config.yaml
model:
  checkpoint_path: "models/hybrid_model.pt"
  source_names: ["deepstate", "equipment", "firms", "ucdp"]

data:
  evaluation_data_path: "data/aligned_evaluation.npz"
  candidate_data_dir: "data/candidates/"

analysis:
  max_lag: 60
  n_factors: 5
  factor_method: "pca"  # or "factor_analysis"
  significance_threshold: 0.05
  granger_max_lag: 30

output:
  base_dir: "outputs/"
  save_intermediate: true
  generate_plots: true
```

### 5.4 Execution

```bash
# Run full analysis
python run_analysis.py --config config.yaml

# Run specific stage
python run_analysis.py --config config.yaml --stage temporal_analysis

# Generate report only (requires completed stages)
python run_analysis.py --config config.yaml --report-only
```

---

## 6. Validation Checklist

Before accepting results, verify:

- [ ] Residuals have approximately zero mean
- [ ] Residual variance is less than input variance
- [ ] ACF values are within [-1, 1] and ACF at lag 0 equals 1
- [ ] Factor loadings are interpretable (not all noise)
- [ ] Variance explained sums to ≤ 1 across factors
- [ ] Granger tests have sufficient sample size (n > 50)
- [ ] Significant correlations survive Bonferroni correction for multiple testing
- [ ] Recommendations include availability assessment

---

## Appendix A: Theoretical References

1. **Residual Diagnostics**: Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287-1294.

2. **Sensitivity Analysis**: Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39-67.

3. **Partial Information Decomposition**: Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. *arXiv:1004.2515*.

4. **Granger Causality**: Granger, C. W. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.

5. **Spectral Analysis**: Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Residual** | Difference between observed value and model prediction |
| **ACF** | Autocorrelation function - correlation of a signal with its lagged self |
| **PSD** | Power spectral density - distribution of signal power across frequencies |
| **Granger causality** | X Granger-causes Y if past X helps predict Y beyond past Y alone |
| **Factor loading** | Weight of a variable on a latent factor |
| **Silhouette score** | Measure of cluster separation (-1 to 1, higher is better) |

---

*End of specification*
