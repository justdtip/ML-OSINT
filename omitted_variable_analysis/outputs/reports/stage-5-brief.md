Stage 5: Granger Causality Testing - Complete
Implementation: stages/granger_causality.py

Outputs created:

- outputs/results/granger_causality.json (full test results)
- outputs/figures/granger_by_source.png
- outputs/figures/granger_top_candidates.png
- outputs/figures/granger_optimal_lags.png
- outputs/figures/granger_bidirectional.png

Test Configuration:

- Max lag tested: 15 days
- Significance level: p < 0.05
- Pairs tested: 53 (30 residual pairs + 10 factor-derived × 4 sources)
- Uses ADF test for stationarity, differences non-stationary series

Key Findings:

**26 significant Granger-causal relationships found (49% of tested pairs)**

Top Candidates that Granger-Cause Residuals:

| Candidate | Sources Caused | Avg Lag |
|-----------|----------------|---------|
| equip_aircraft_cumulative | 2/2 | TBD |
| days_since_start | 2/2 | TBD |
| equip_total_key | 2/2 | TBD |
| equip_tank_cumulative | 2/2 | TBD |
| equip_APC_cumulative | 2/2 | TBD |
| equip_helicopter_cumulative | 1/2 | TBD |
| equip_naval_ship_cumulative | 1/2 | TBD |
| equip_MRL_cumulative | 1/1 | TBD |
| equip_anti_aircraft_warfare_cumulative | 1/1 | TBD |
| equip_field_artillery_daily | 1/1 | TBD |

Bidirectional Causality Results:

| Direction | Count |
|-----------|-------|
| candidate_causes_residual | 3 |
| bidirectional_candidate_dominant | 6 |
| bidirectional_residual_dominant | 1 |

Interpretation:

1. **Equipment losses Granger-cause residuals** - Past values of cumulative equipment losses help predict future residuals beyond the residuals' own history. This confirms they are not just correlated but have predictive value.

2. **Bidirectional relationships exist** - Most significant pairs show bidirectional causality, with candidates being the dominant direction. This suggests:
   - Equipment losses → affect model residuals (missing information)
   - Model residuals → somewhat predict equipment losses (common underlying factors)

3. **Temporal trend (days_since_start) is causal** - The simple passage of time Granger-causes residuals, confirming the model is missing secular trend information.

4. **Multiple equipment categories provide information** - Aircraft, tanks, APCs, helicopters all independently Granger-cause residuals, suggesting the model would benefit from a composite attrition metric.

Recommendations:

1. **Add cumulative equipment losses** - Strongest Granger-causal signal
2. **Include time trend** - Either explicit days_since_start or detrend residuals
3. **Consider composite loss metric** - Sum of key equipment categories

Note: ERA5 weather download at 149/171 points (87%). Weather Granger tests pending completion.
