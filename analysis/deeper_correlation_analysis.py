#!/usr/bin/env python3
"""
Deeper correlation analysis investigating relationships between:
- Cloud cover and conflict intensity
- Seasonal patterns in combat operations
- Potential tactical use of weather conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

# Load merged data
df = pd.read_csv(ANALYSIS_DIR / "sentinel_osint_merged.csv")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

print("=" * 80)
print("DEEPER CORRELATION ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. FULL CORRELATION MATRIX
# =============================================================================
print("\n[1] FULL CORRELATION MATRIX")
print("-" * 60)

corr_cols = ['s2_avg_cloud', 'ucdp_events', 'ucdp_deaths', 'firms_fires',
             'monthly_loss', 'ds_points', 's5p_no2', 's3_fire']
corr_matrix = df[corr_cols].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3).to_string())

# Highlight significant correlations with cloud cover
print("\n\nCorrelations with CLOUD COVER:")
cloud_corr = corr_matrix['s2_avg_cloud'].drop('s2_avg_cloud').sort_values()
for var, r in cloud_corr.items():
    sig = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.2 else ""
    print(f"  {var:<20} r = {r:+.3f} {sig}")

# =============================================================================
# 2. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================
print("\n\n[2] STATISTICAL SIGNIFICANCE (Pearson correlation p-values)")
print("-" * 60)

key_pairs = [
    ('s2_avg_cloud', 'ucdp_deaths'),
    ('s2_avg_cloud', 'monthly_loss'),
    ('s2_avg_cloud', 'ucdp_events'),
    ('s2_avg_cloud', 'firms_fires'),
    ('monthly_loss', 'ds_points'),
    ('ucdp_deaths', 'monthly_loss'),
]

print(f"\n{'Variable Pair':<45} {'r':>8} {'p-value':>12} {'Significant?':>15}")
print("-" * 85)

for var1, var2 in key_pairs:
    valid = df[[var1, var2]].dropna()
    r, p = stats.pearsonr(valid[var1], valid[var2])
    sig = "YES (p<0.001)" if p < 0.001 else "YES (p<0.01)" if p < 0.01 else "YES (p<0.05)" if p < 0.05 else "NO"
    print(f"{var1} vs {var2:<20} {r:>8.3f} {p:>12.6f} {sig:>15}")

# =============================================================================
# 3. SEASONAL ANALYSIS
# =============================================================================
print("\n\n[3] SEASONAL PATTERNS")
print("-" * 60)

seasonal_stats = df.groupby('season').agg({
    's2_avg_cloud': 'mean',
    'ucdp_events': 'mean',
    'ucdp_deaths': 'mean',
    'monthly_loss': 'mean',
    'firms_fires': 'mean'
}).round(1)

# Reorder seasons
seasonal_stats = seasonal_stats.reindex(['Winter', 'Spring', 'Summer', 'Fall'])

print("\nSeasonal Averages:")
print(seasonal_stats.to_string())

print("\n\nKey Observations:")
print(f"  - Winter avg cloud cover: {seasonal_stats.loc['Winter', 's2_avg_cloud']:.1f}%")
print(f"  - Summer avg cloud cover: {seasonal_stats.loc['Summer', 's2_avg_cloud']:.1f}%")
print(f"  - Winter avg monthly losses: {seasonal_stats.loc['Winter', 'monthly_loss']:,.0f}")
print(f"  - Summer avg monthly losses: {seasonal_stats.loc['Summer', 'monthly_loss']:,.0f}")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
print("\n\n[4] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Cloud Cover, Conflict Intensity & Tactical Patterns\nUkraine May 2022 - Dec 2024',
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Cloud cover vs Deaths - scatter with regression
ax1 = axes[0, 0]
ax1.scatter(df['s2_avg_cloud'], df['ucdp_deaths'], alpha=0.7, s=60, c='#E63946')
z = np.polyfit(df['s2_avg_cloud'], df['ucdp_deaths'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['s2_avg_cloud'].min(), df['s2_avg_cloud'].max(), 100)
ax1.plot(x_line, p(x_line), 'k--', linewidth=2)
r, pval = stats.pearsonr(df['s2_avg_cloud'], df['ucdp_deaths'])
ax1.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.4f}', transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('Cloud Cover (%)')
ax1.set_ylabel('UCDP Deaths (monthly)')
ax1.set_title('Cloud Cover vs Combat Deaths', fontweight='bold')

# Plot 2: Cloud cover vs Losses
ax2 = axes[0, 1]
ax2.scatter(df['s2_avg_cloud'], df['monthly_loss'], alpha=0.7, s=60, c='#264653')
z = np.polyfit(df['s2_avg_cloud'], df['monthly_loss'], 1)
p = np.poly1d(z)
ax2.plot(x_line, p(x_line), 'k--', linewidth=2)
r, pval = stats.pearsonr(df['s2_avg_cloud'], df['monthly_loss'])
ax2.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.4f}', transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.set_xlabel('Cloud Cover (%)')
ax2.set_ylabel('Personnel Losses (monthly)')
ax2.set_title('Cloud Cover vs Personnel Losses', fontweight='bold')

# Plot 3: Cloud cover vs Fire Detection (negative control)
ax3 = axes[0, 2]
ax3.scatter(df['s2_avg_cloud'], df['firms_fires'], alpha=0.7, s=60, c='#F4A261')
z = np.polyfit(df['s2_avg_cloud'], df['firms_fires'], 1)
p = np.poly1d(z)
ax3.plot(x_line, p(x_line), 'k--', linewidth=2)
r, pval = stats.pearsonr(df['s2_avg_cloud'], df['firms_fires'])
ax3.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.4f}', transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.set_xlabel('Cloud Cover (%)')
ax3.set_ylabel('FIRMS Fire Detections')
ax3.set_title('Cloud Cover vs Fire Detections\n(Expected negative - sensor limitation)', fontweight='bold')

# Plot 4: Seasonal boxplots - Deaths
ax4 = axes[1, 0]
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
sns.boxplot(data=df, x='season', y='ucdp_deaths', ax=ax4, palette='coolwarm', order=season_order)
ax4.set_xlabel('Season')
ax4.set_ylabel('UCDP Deaths (monthly)')
ax4.set_title('Combat Deaths by Season', fontweight='bold')

# Plot 5: Seasonal boxplots - Losses
ax5 = axes[1, 1]
sns.boxplot(data=df, x='season', y='monthly_loss', ax=ax5, palette='coolwarm', order=season_order)
ax5.set_xlabel('Season')
ax5.set_ylabel('Personnel Losses (monthly)')
ax5.set_title('Personnel Losses by Season', fontweight='bold')

# Plot 6: Time series with cloud overlay
ax6 = axes[1, 2]
ax6.fill_between(df['date'], 0, df['s2_avg_cloud'], alpha=0.3, color='gray', label='Cloud Cover %')
ax6b = ax6.twinx()
ax6b.plot(df['date'], df['monthly_loss']/1000, 'o-', color='#E63946', label='Monthly Losses (K)', linewidth=2)
ax6.set_xlabel('Date')
ax6.set_ylabel('Cloud Cover %', color='gray')
ax6b.set_ylabel('Monthly Losses (thousands)', color='#E63946')
ax6.set_title('Cloud Cover & Losses Over Time', fontweight='bold')
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / '07_cloud_conflict_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: 07_cloud_conflict_analysis.png")

# =============================================================================
# 5. INTERPRETATION
# =============================================================================
print("\n\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Get the actual correlations
r_cloud_deaths, p_cloud_deaths = stats.pearsonr(df['s2_avg_cloud'], df['ucdp_deaths'])
r_cloud_loss, p_cloud_loss = stats.pearsonr(df['s2_avg_cloud'], df['monthly_loss'])
r_cloud_fires, p_cloud_fires = stats.pearsonr(df['s2_avg_cloud'], df['firms_fires'])

print(f"""
FINDINGS:

1. CLOUD COVER vs COMBAT DEATHS:
   Correlation: r = {r_cloud_deaths:+.3f} (p = {p_cloud_deaths:.4f})
   {"STATISTICALLY SIGNIFICANT" if p_cloud_deaths < 0.05 else "NOT SIGNIFICANT"}

2. CLOUD COVER vs PERSONNEL LOSSES:
   Correlation: r = {r_cloud_loss:+.3f} (p = {p_cloud_loss:.4f})
   {"STATISTICALLY SIGNIFICANT" if p_cloud_loss < 0.05 else "NOT SIGNIFICANT"}

3. CLOUD COVER vs FIRE DETECTIONS:
   Correlation: r = {r_cloud_fires:+.3f} (p = {p_cloud_fires:.4f})
   {"STATISTICALLY SIGNIFICANT" if p_cloud_fires < 0.05 else "NOT SIGNIFICANT"}
   This negative correlation is EXPECTED - clouds block satellite fire detection.

POSSIBLE EXPLANATIONS FOR POSITIVE CLOUD-CONFLICT CORRELATIONS:

A) TACTICAL HYPOTHESIS:
   - Forces may conduct more intensive operations under cloud cover
   - Reduced satellite/aerial surveillance enables movement
   - Drone operations may be limited, changing combat dynamics

B) SEASONAL CONFOUNDING:
   - Winter months have both high cloud cover AND intense fighting
   - This could be coincidental timing (e.g., Bakhmut offensive winter 2022-23)
   - Need to control for temporal autocorrelation

C) OBSERVATION BIAS:
   - UCDP deaths may be reported differently in different seasons
   - Personnel loss reporting may have seasonal patterns

RECOMMENDATION:
   The correlations appear genuine (not random noise) but causation is unclear.
   Time-lagged analysis and controlling for offensive operations would help
   distinguish tactical behavior from seasonal confounding.
""")

# =============================================================================
# 6. PARTIAL CORRELATION (controlling for season/time)
# =============================================================================
print("\n[6] PARTIAL CORRELATION ANALYSIS")
print("-" * 60)

# Add month number as a control variable
df['month_num'] = (df['date'] - df['date'].min()).dt.days / 30

from scipy.stats import pearsonr

def partial_corr(df, x, y, control):
    """Calculate partial correlation controlling for a third variable."""
    # Residuals of x ~ control
    slope_x, intercept_x, _, _, _ = stats.linregress(df[control], df[x])
    resid_x = df[x] - (slope_x * df[control] + intercept_x)

    # Residuals of y ~ control
    slope_y, intercept_y, _, _, _ = stats.linregress(df[control], df[y])
    resid_y = df[y] - (slope_y * df[control] + intercept_y)

    # Correlation of residuals
    r, p = pearsonr(resid_x, resid_y)
    return r, p

print("\nPartial correlations (controlling for time trend):")
print(f"{'Relationship':<40} {'Raw r':>10} {'Partial r':>12} {'Change':>10}")
print("-" * 75)

for var in ['ucdp_deaths', 'monthly_loss', 'firms_fires']:
    raw_r, _ = pearsonr(df['s2_avg_cloud'], df[var])
    part_r, part_p = partial_corr(df, 's2_avg_cloud', var, 'month_num')
    change = part_r - raw_r
    print(f"Cloud vs {var:<30} {raw_r:>10.3f} {part_r:>12.3f} {change:>+10.3f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
