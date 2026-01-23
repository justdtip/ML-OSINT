Stage 4: Candidate Variable Correlation - Complete
Implementation: stages/candidate_correlation.py

Outputs created:

- outputs/results/candidate_correlations.json (full correlation results)
- outputs/figures/factor_candidate_heatmap.png
- outputs/figures/top_correlations.png
- outputs/figures/lagged_correlations.png
- outputs/figures/factor_candidate_timeseries.png

Data Sources Loaded:

| Source | Features | Description |
|--------|----------|-------------|
| Temporal | 14 | Day of week, month, trend, cyclic encodings |
| Lunar | 5 | Moon phase, illumination (simplified calculation) |
| War Losses | 23 | Daily equipment/personnel losses (cumulative + daily change) |
| VIINA | 5 | Daily event counts, 7-day rolling stats |
| ERA5 | ~8 | Weather (partial - 110/171 points downloaded) |
| HDX | 2 | Aggregated conflict events |

Total: 47 candidate features

Key Findings:

222 significant correlations found (p < 0.05, |r| > 0.1)

**Factor 1 (Territorial Dynamics, 13.5% variance):**
- Strongest correlate: Equipment losses (r = -0.95 for aircraft, -0.95 for helicopters)
- Interpretation: Factor 1 tracks cumulative war attrition - as more equipment is lost, territorial dynamics shift
- Also correlates with days_since_start (r = -0.87) and VIINA event count (r = +0.83)

**Factor 2 (Conflict Intensity, 9.2% variance):**
- Moderate correlations with daily loss rates
- Lower correlations than Factor 1 (more transient/volatile)

**Factor 3 (Spatial Concentration, 8.5% variance):**
- Correlates with equipment cumulative losses (r ~ -0.6)
- Weaker than Factor 1

**Factor 4 (Fire Activity, 7.3% variance):**
- Weak correlations with all candidates
- Suggests FIRMS residuals capture something distinct from losses

**Factor 5 (Movement Direction, 6.0% variance):**
- Strong seasonal signal: month_sin (r = 0.82), month_cos (r = 0.57)
- Suggests operational tempo varies by season

Top 10 Correlations (by |r|):

| Rank | Factor | Candidate | Correlation |
|------|--------|-----------|-------------|
| 1 | F1 | equip_aircraft_cumulative | -0.953 |
| 2 | F1 | equip_helicopter_cumulative | -0.946 |
| 3 | F1 | equip_tank_cumulative | -0.897 |
| 4 | F1 | equip_total_key | -0.895 |
| 5 | F1 | equip_APC_cumulative | -0.889 |
| 6 | F1 | days_since_start | -0.868 |
| 7 | F1 | equip_naval_ship_cumulative | -0.868 |
| 8 | F1 | viina_event_count_7d_mean | 0.834 |
| 9 | F1 | equip_MRL_cumulative | -0.832 |
| 10 | F5 | month_sin | 0.819 |

Interpretation:

The analysis reveals that **Factor 1** (the dominant residual pattern explaining 13.5% of variance) is almost perfectly correlated with cumulative equipment losses. This suggests:

1. **The model is missing war attrition information** - cumulative losses track long-term degradation of Russian capability that affects all sources
2. **Adding war losses data could substantially reduce residual variance** - the ~0.95 correlation means >90% of Factor 1's variance could be explained
3. **Factor 5 captures seasonality** - strong month correlations suggest operational tempo varies by season (mud season, winter)
4. **Factor 4 (FIRMS) is distinct** - fire activity residuals don't correlate with other candidates, suggesting unique omitted variables (weather? targeting patterns?)

Recommendations for model improvement:

1. **High priority**: Add cumulative equipment losses as a feature (would capture Factor 1)
2. **Medium priority**: Add VIINA daily event counts (captures conflict intensity)
3. **Medium priority**: Add month/season encoding (captures Factor 5)
4. **Lower priority**: Weather data when ERA5 download completes

Note: ERA5 download in progress (111/171 points complete). Re-run correlation analysis after completion for weather correlations.
