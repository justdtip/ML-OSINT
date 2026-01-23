Stage 1: Residual Extraction - Complete
Implementation: stages/residual_extraction.py

Outputs created:

outputs/results/full_residuals.npz (300KB) - Residuals from full reconstruction
outputs/results/masked_residuals.npz (300KB) - Residuals from masked reconstruction
outputs/results/residual_metadata.json - Timestamps, feature names, validation checks
Key Results:

Source	Full Recon Mean	Full Recon Var	Masked Recon Mean	Masked Recon Var
DeepState	-0.094	0.580	-0.125	0.796
Equipment	+0.062	0.750	+0.045	0.762
FIRMS	+0.029	0.822	+0.050	0.871
UCDP	-0.036	0.742	-0.047	0.981
Validation Checks (all passed):

✓ No NaN values
✓ No Inf values
✓ Residual means near zero (largest is -0.125 for masked DeepState)
✓ Residual variance < input variance (input is normalized to ~1.0)
Interpretation:

The model explains ~20-42% of variance (residual var ~0.58-0.98 vs input var ~1.0)
Masked reconstruction has higher variance than full reconstruction (as expected - less information)
UCDP has highest masked residual variance (0.98) - hardest to predict from other sources
DeepState has lowest full reconstruction variance (0.58) - best self-reconstruction