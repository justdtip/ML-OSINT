Stage 3: Latent Factor Extraction - Complete
Implementation: stages/factor_extraction.py

Outputs created:

outputs/results/latent_factors.npz (13KB) - Factor scores and loadings
outputs/results/factor_characterization.json (26KB) - Detailed characterizations
outputs/figures/factor_timeseries.png - Factor time series
outputs/figures/factor_loadings_heatmap.png - Source loadings
outputs/figures/source_sensitivity_radar.png - Radar charts
outputs/figures/variance_explained.png - Scree plot
outputs/figures/top_feature_loadings.png - Feature loadings
Key Findings
Variance Explained: 5 factors capture 44.4% of residual variance

Factor	Variance	Dominant Source	Key Features	Interpretation
F1	13.5%	DeepState	liberated_ratio, area_liberated, frontline	Highly persistent, strongly decreasing trend (territorial changes)
F2	9.2%	FIRMS	govt_side_a, event_count, russia_ukraine_events	UCDP-dominated, volatility clustering (conflict intensity)
F3	8.5%	DeepState	lon_spread, activity_intensity, units_per_polygon	Persistent, decreasing trend (spatial dynamics)
F4	7.3%	FIRMS	frp_total, frp_medium, type_0	Fire activity clustering
F5	6.0%	DeepState	arrows_N, arrows_westward, arrows_total	Movement direction patterns
Interpretation
The extracted factors represent distinct "omitted variables":

Factor 1 (Territorial Dynamics): A slow-trending factor dominated by territorial control features - suggests the model is missing information about strategic/operational planning that drives territorial changes

Factor 2 (Conflict Intensity): Cross-source factor loading on both UCDP events and fire activity - represents unmeasured overall conflict intensity

Factor 3 (Spatial Concentration): Geographic spread of operations - missing information about where fighting is concentrated

Factor 4 (Fire Activity): Pure FIRMS factor - thermal signature patterns not explained by other sources

Factor 5 (Movement Direction): Military movement patterns - strategic direction of operations

All factors show high persistence (lag-1 ACF > 0.77), indicating the omitted variables change slowly over time rather than being random noise.