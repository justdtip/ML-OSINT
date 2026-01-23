# Omitted Variable Analysis - Final Report

*Generated: 2026-01-20 22:52:06*

## Executive Summary

This analysis identified **24** candidate variables that correlate 
with residual patterns in the prediction model. Of these:

- **24** are from independent data sources (low leakage risk)

- **0** show Granger-causal relationships (predictive power)


## Key Findings

### Top Independent Candidate Variables

These variables show significant correlation with residual factors and 
come from data sources independent of the model's training data:


| Rank | Variable | Source | Factor | Correlation | Granger Causal |

|------|----------|--------|--------|-------------|----------------|

| 1 | month_sin | Calendar | Factor_5 | 0.815 |  |

| 2 | month_cos | Calendar | Factor_1 | 0.605 |  |

| 3 | month_cos | Calendar | Factor_3 | 0.460 |  |

| 4 | month_cos | Calendar | Factor_2 | 0.404 |  |

| 5 | month_sin | Calendar | Factor_4 | 0.389 |  |

| 6 | fire_count_detrended | FIRMS | Factor_1 | 0.383 |  |

| 7 | fire_count_detrended | FIRMS | Factor_5 | 0.337 |  |

| 8 | month_cos | Calendar | Factor_4 | 0.308 |  |

| 9 | frp_sum_detrended | FIRMS | Factor_1 | 0.304 |  |

| 10 | frp_sum_detrended | FIRMS | Factor_5 | 0.300 |  |



### Factor Interpretations

Based on correlations with independent data sources:


**Factor_1** (explains 0.0% of residual variance):

- Interpretation: Seasonal/cyclical pattern in conflict dynamics

- Top correlates:

  - month_cos (Calendar): r=-0.605

  - fire_count_detrended (FIRMS): r=0.383

  - frp_sum_detrended (FIRMS): r=0.304



**Factor_2** (explains 0.0% of residual variance):

- Interpretation: Seasonal/cyclical pattern in conflict dynamics

- Top correlates:

  - month_cos (Calendar): r=-0.404

  - month_sin (Calendar): r=0.251

  - frp_sum_detrended (FIRMS): r=-0.144



**Factor_3** (explains 0.0% of residual variance):

- Interpretation: Seasonal/cyclical pattern in conflict dynamics

- Top correlates:

  - month_cos (Calendar): r=0.460

  - fire_count_detrended (FIRMS): r=-0.293

  - frp_sum_detrended (FIRMS): r=-0.249



**Factor_4** (explains 0.0% of residual variance):

- Interpretation: Seasonal/cyclical pattern in conflict dynamics

- Top correlates:

  - month_sin (Calendar): r=0.389

  - month_cos (Calendar): r=-0.308



**Factor_5** (explains 0.0% of residual variance):

- Interpretation: Seasonal/cyclical pattern in conflict dynamics

- Top correlates:

  - month_sin (Calendar): r=0.815

  - fire_count_detrended (FIRMS): r=-0.337

  - frp_sum_detrended (FIRMS): r=-0.300



### Leakage Analysis

## Recommendations

1. Consider adding these independent variables to the model: month_sin, month_cos, month_cos. These show significant correlation with residual factors and low leakage risk.


2. NASA FIRMS satellite fire data shows strong correlation with residual factors. This is truly independent (satellite-based) and captures military activity intensity not currently in the model.


3. Strong seasonal patterns detected in residuals. Consider adding cyclical month encoding (sin/cos) to capture seasonal effects on conflict dynamics.


## Methodology

This analysis followed a 6-stage pipeline:


1. **Residual Extraction**: Extract prediction residuals from all data sources

2. **Temporal Structure Analysis**: Identify autocorrelation and cross-correlation patterns

3. **Latent Factor Extraction**: Use PCA to identify common latent factors in residuals

4. **Candidate Variable Correlation**: Test correlations with external candidate variables

5. **Granger Causality Testing**: Test predictive relationships

6. **Ranking and Reporting**: Combine results and generate recommendations


## Data Sources Evaluated

### Independent Sources (Low Leakage Risk)

- **FIRMS**: NASA VIIRS satellite fire detection

- **UCDP**: Uppsala Conflict Data Program event tracking

- **Calendar**: Seasonal/cyclical temporal features

- **Lunar**: Lunar phase data

- **ERA5**: ECMWF weather reanalysis (pending)


### High Leakage Risk Sources

- War losses/equipment data (shares provenance with model features)

- Cumulative casualty counts (derived from same source)
