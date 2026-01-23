#!/usr/bin/env python3
"""
Correlation analysis with outlier removal using IQR method.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

df = pd.read_csv(ANALYSIS_DIR / "sentinel_osint_merged.csv")
df['date'] = pd.to_datetime(df['date'])

print("=" * 80)
print("CORRELATION ANALYSIS WITH OUTLIER REMOVAL")
print("=" * 80)

# =============================================================================
# 1. IDENTIFY OUTLIERS IN UCDP_DEATHS
# =============================================================================
print("\n[1] IDENTIFYING OUTLIERS IN UCDP_DEATHS")
print("-" * 60)

deaths = df['ucdp_deaths']
Q1 = deaths.quantile(0.25)
Q3 = deaths.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"  Q1: {Q1:,.0f}")
print(f"  Q3: {Q3:,.0f}")
print(f"  IQR: {IQR:,.0f}")
print(f"  Lower bound: {lower_bound:,.0f}")
print(f"  Upper bound: {upper_bound:,.0f}")

outliers = df[deaths > upper_bound][['date', 'ucdp_deaths', 's2_avg_cloud']]
print(f"\n  Outliers (>{upper_bound:,.0f} deaths):")
for _, row in outliers.iterrows():
    print(f"    {row['date'].strftime('%Y-%m')}: {row['ucdp_deaths']:,.0f} deaths (cloud: {row['s2_avg_cloud']:.1f}%)")

# =============================================================================
# 2. COMPARE CORRELATIONS WITH AND WITHOUT OUTLIERS
# =============================================================================
print("\n\n[2] CORRELATION COMPARISON")
print("-" * 60)

# Full data
r_full, p_full = stats.pearsonr(df['s2_avg_cloud'], df['ucdp_deaths'])

# Without outliers (IQR method)
df_no_outliers = df[(deaths >= lower_bound) & (deaths <= upper_bound)]
r_iqr, p_iqr = stats.pearsonr(df_no_outliers['s2_avg_cloud'], df_no_outliers['ucdp_deaths'])

# Without top 2 outliers only
df_no_top2 = df[deaths <= df['ucdp_deaths'].nlargest(3).min()]
r_no2, p_no2 = stats.pearsonr(df_no_top2['s2_avg_cloud'], df_no_top2['ucdp_deaths'])

# Spearman (rank-based, robust to outliers)
r_spearman, p_spearman = stats.spearmanr(df['s2_avg_cloud'], df['ucdp_deaths'])

print(f"\n{'Method':<40} {'r':>10} {'p-value':>12} {'n':>6} {'Significant?':>15}")
print("-" * 85)
print(f"{'Full data (Pearson)':<40} {r_full:>10.3f} {p_full:>12.4f} {len(df):>6} {'YES' if p_full < 0.05 else 'NO':>15}")
print(f"{'Without IQR outliers (Pearson)':<40} {r_iqr:>10.3f} {p_iqr:>12.4f} {len(df_no_outliers):>6} {'YES' if p_iqr < 0.05 else 'NO':>15}")
print(f"{'Without top 2 months (Pearson)':<40} {r_no2:>10.3f} {p_no2:>12.4f} {len(df_no_top2):>6} {'YES' if p_no2 < 0.05 else 'NO':>15}")
print(f"{'Full data (Spearman rank)':<40} {r_spearman:>10.3f} {p_spearman:>12.4f} {len(df):>6} {'YES' if p_spearman < 0.05 else 'NO':>15}")

# =============================================================================
# 3. SAME FOR MONTHLY_LOSS
# =============================================================================
print("\n\n[3] CLOUD vs MONTHLY_LOSS (same analysis)")
print("-" * 60)

losses = df['monthly_loss']
Q1_l = losses.quantile(0.25)
Q3_l = losses.quantile(0.75)
IQR_l = Q3_l - Q1_l
upper_l = Q3_l + 1.5 * IQR_l

outliers_loss = df[losses > upper_l][['date', 'monthly_loss', 's2_avg_cloud']]
print(f"  Outliers (>{upper_l:,.0f} losses):")
for _, row in outliers_loss.iterrows():
    print(f"    {row['date'].strftime('%Y-%m')}: {row['monthly_loss']:,.0f} losses (cloud: {row['s2_avg_cloud']:.1f}%)")

df_no_outliers_l = df[(losses >= Q1_l - 1.5*IQR_l) & (losses <= upper_l)]

r_full_l, p_full_l = stats.pearsonr(df['s2_avg_cloud'], df['monthly_loss'])
r_iqr_l, p_iqr_l = stats.pearsonr(df_no_outliers_l['s2_avg_cloud'], df_no_outliers_l['monthly_loss'])
r_spearman_l, p_spearman_l = stats.spearmanr(df['s2_avg_cloud'], df['monthly_loss'])

print(f"\n{'Method':<40} {'r':>10} {'p-value':>12} {'n':>6} {'Significant?':>15}")
print("-" * 85)
print(f"{'Full data (Pearson)':<40} {r_full_l:>10.3f} {p_full_l:>12.4f} {len(df):>6} {'YES' if p_full_l < 0.05 else 'NO':>15}")
print(f"{'Without IQR outliers (Pearson)':<40} {r_iqr_l:>10.3f} {p_iqr_l:>12.4f} {len(df_no_outliers_l):>6} {'YES' if p_iqr_l < 0.05 else 'NO':>15}")
print(f"{'Full data (Spearman rank)':<40} {r_spearman_l:>10.3f} {p_spearman_l:>12.4f} {len(df):>6} {'YES' if p_spearman_l < 0.05 else 'NO':>15}")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
print("\n\n[4] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Cloud Cover vs Combat Intensity: Outlier Analysis', fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Full data with outliers highlighted
ax1 = axes[0, 0]
outlier_mask = deaths > upper_bound
ax1.scatter(df.loc[~outlier_mask, 's2_avg_cloud'], df.loc[~outlier_mask, 'ucdp_deaths'],
            alpha=0.7, s=60, c='#2A9D8F', label='Normal')
ax1.scatter(df.loc[outlier_mask, 's2_avg_cloud'], df.loc[outlier_mask, 'ucdp_deaths'],
            alpha=0.9, s=100, c='#E63946', marker='X', label='Outliers')
for _, row in outliers.iterrows():
    ax1.annotate(row['date'].strftime('%Y-%m'),
                 (row['s2_avg_cloud'], row['ucdp_deaths']),
                 textcoords="offset points", xytext=(5,5), fontsize=8)
ax1.set_xlabel('Cloud Cover (%)')
ax1.set_ylabel('UCDP Deaths')
ax1.set_title(f'Full Data\nr={r_full:.3f}, p={p_full:.4f}', fontweight='bold')
ax1.legend()

# Plot 2: Without outliers
ax2 = axes[0, 1]
ax2.scatter(df_no_outliers['s2_avg_cloud'], df_no_outliers['ucdp_deaths'],
            alpha=0.7, s=60, c='#2A9D8F')
z = np.polyfit(df_no_outliers['s2_avg_cloud'], df_no_outliers['ucdp_deaths'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_no_outliers['s2_avg_cloud'].min(), df_no_outliers['s2_avg_cloud'].max(), 100)
ax2.plot(x_line, p(x_line), 'k--', linewidth=2)
ax2.set_xlabel('Cloud Cover (%)')
ax2.set_ylabel('UCDP Deaths')
ax2.set_title(f'Outliers Removed (IQR)\nr={r_iqr:.3f}, p={p_iqr:.4f}', fontweight='bold')

# Plot 3: Monthly loss - full data
ax3 = axes[1, 0]
outlier_mask_l = losses > upper_l
ax3.scatter(df.loc[~outlier_mask_l, 's2_avg_cloud'], df.loc[~outlier_mask_l, 'monthly_loss']/1000,
            alpha=0.7, s=60, c='#264653', label='Normal')
ax3.scatter(df.loc[outlier_mask_l, 's2_avg_cloud'], df.loc[outlier_mask_l, 'monthly_loss']/1000,
            alpha=0.9, s=100, c='#E63946', marker='X', label='Outliers')
ax3.set_xlabel('Cloud Cover (%)')
ax3.set_ylabel('Monthly Losses (thousands)')
ax3.set_title(f'Monthly Losses - Full Data\nr={r_full_l:.3f}, p={p_full_l:.4f}', fontweight='bold')
ax3.legend()

# Plot 4: Summary bar chart
ax4 = axes[1, 1]
methods = ['Full\n(Pearson)', 'No Outliers\n(Pearson)', 'Spearman\n(Rank)']
deaths_corr = [r_full, r_iqr, r_spearman]
loss_corr = [r_full_l, r_iqr_l, r_spearman_l]

x = np.arange(len(methods))
width = 0.35
bars1 = ax4.bar(x - width/2, deaths_corr, width, label='Cloud vs Deaths', color='#E63946', alpha=0.7)
bars2 = ax4.bar(x + width/2, loss_corr, width, label='Cloud vs Losses', color='#264653', alpha=0.7)

ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax4.set_ylabel('Correlation (r)')
ax4.set_xlabel('Method')
ax4.set_xticks(x)
ax4.set_xticklabels(methods)
ax4.set_title('Correlation Comparison', fontweight='bold')
ax4.legend()
ax4.set_ylim(-0.2, 0.5)

# Add significance markers
for i, (d, l) in enumerate(zip([p_full, p_iqr, p_spearman], [p_full_l, p_iqr_l, p_spearman_l])):
    if d < 0.05:
        ax4.annotate('*', (i - width/2, deaths_corr[i] + 0.02), ha='center', fontsize=14, fontweight='bold')
    if l < 0.05:
        ax4.annotate('*', (i + width/2, loss_corr[i] + 0.02), ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / '08_outlier_correlation_analysis.png', dpi=150, bbox_inches='tight')
print("  Saved: 08_outlier_correlation_analysis.png")

# =============================================================================
# 5. CONCLUSIONS
# =============================================================================
print("\n\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

print(f"""
CLOUD COVER vs UCDP DEATHS:
  - Full data:        r = {r_full:+.3f} (p = {p_full:.4f}) - {"SIGNIFICANT" if p_full < 0.05 else "NOT significant"}
  - Without outliers: r = {r_iqr:+.3f} (p = {p_iqr:.4f}) - {"SIGNIFICANT" if p_iqr < 0.05 else "NOT significant"}
  - Spearman rank:    r = {r_spearman:+.3f} (p = {p_spearman:.4f}) - {"SIGNIFICANT" if p_spearman < 0.05 else "NOT significant"}

CLOUD COVER vs MONTHLY LOSSES:
  - Full data:        r = {r_full_l:+.3f} (p = {p_full_l:.4f}) - {"SIGNIFICANT" if p_full_l < 0.05 else "NOT significant"}
  - Without outliers: r = {r_iqr_l:+.3f} (p = {p_iqr_l:.4f}) - {"SIGNIFICANT" if p_iqr_l < 0.05 else "NOT significant"}
  - Spearman rank:    r = {r_spearman_l:+.3f} (p = {p_spearman_l:.4f}) - {"SIGNIFICANT" if p_spearman_l < 0.05 else "NOT significant"}

INTERPRETATION:
""")

if r_iqr > 0.2 and p_iqr < 0.1:
    print("  The positive cloud-deaths correlation PERSISTS after outlier removal,")
    print("  suggesting a genuine relationship (though weaker).")
elif abs(r_iqr) < 0.15:
    print("  The correlation DISAPPEARS after outlier removal.")
    print("  The apparent relationship was driven by the extreme months")
    print("  (Jan 2023 Bakhmut, Jan 2024 Avdiivka offensives).")
else:
    print("  Relationship is ambiguous - more data needed.")

print(f"""
OUTLIER MONTHS IDENTIFIED:
  - Jan 2023: 18,152 deaths - Bakhmut offensive peak
  - Jan 2024: 14,699 deaths - Avdiivka offensive

These extreme months both occurred in winter (high cloud cover) which
creates apparent correlation. The question is whether this is:
  A) Coincidental timing of major offensives
  B) Tactical choice to launch offensives under cloud cover
  C) Winter conditions driving both variables independently
""")
