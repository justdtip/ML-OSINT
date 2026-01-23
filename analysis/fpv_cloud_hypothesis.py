#!/usr/bin/env python3
"""
Investigate whether cloud cover affects combat dynamics in ways consistent
with the FPV drone hypothesis:

Hypothesis: Cloud cover makes FPV drones harder to spot visually, increasing
their effectiveness and thus casualties.

Evidence we might expect:
1. Higher deaths per event under cloud cover (more lethal engagements)
2. Changes in casualty composition (military vs civilian)
3. Geographic patterns (front line vs rear areas)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

BASE_DIR = str(PROJECT_ROOT)

print("=" * 80)
print("FPV DRONE / CLOUD COVER HYPOTHESIS ANALYSIS")
print("=" * 80)

# Load data
merged = pd.read_csv(f"{BASE_DIR}/analysis/sentinel_osint_merged.csv")
merged['date'] = pd.to_datetime(merged['date'])

ucdp = pd.read_csv(f"{BASE_DIR}/data/ucdp/ged_events.csv", low_memory=False)
ucdp['date_start'] = pd.to_datetime(ucdp['date_start'], format='mixed')
ucdp = ucdp[ucdp['date_start'] >= '2022-02-24']

# Add month column for merging with cloud data
ucdp['month'] = ucdp['date_start'].dt.to_period('M').dt.to_timestamp()

# Merge cloud data with UCDP events
cloud_monthly = merged[['date', 's2_avg_cloud']].copy()
cloud_monthly.columns = ['month', 'cloud_cover']
ucdp = ucdp.merge(cloud_monthly, on='month', how='left')

print(f"\nUCDP events with cloud data: {ucdp['cloud_cover'].notna().sum():,}")

# =============================================================================
# 1. DEATHS PER EVENT ANALYSIS
# =============================================================================
print("\n[1] DEATHS PER EVENT vs CLOUD COVER")
print("-" * 60)

# Monthly aggregation
monthly = ucdp.groupby('month').agg({
    'id': 'count',
    'best_est': 'sum',
    'deaths_a': 'sum',
    'deaths_b': 'sum',
    'deaths_civilians': 'sum',
    'deaths_unknown': 'sum',
    'cloud_cover': 'first'
}).reset_index()

monthly['deaths_per_event'] = monthly['best_est'] / monthly['id']
monthly['military_deaths'] = monthly['deaths_a'] + monthly['deaths_b']
monthly['military_ratio'] = monthly['military_deaths'] / monthly['best_est']
monthly['civilian_ratio'] = monthly['deaths_civilians'] / monthly['best_est']

# Correlation: cloud cover vs deaths per event
valid = monthly.dropna(subset=['cloud_cover', 'deaths_per_event'])
r_dpe, p_dpe = stats.pearsonr(valid['cloud_cover'], valid['deaths_per_event'])
print(f"\nCloud cover vs Deaths per event:")
print(f"  r = {r_dpe:.3f}, p = {p_dpe:.4f}")

# High cloud vs low cloud comparison
high_cloud = monthly[monthly['cloud_cover'] > monthly['cloud_cover'].median()]
low_cloud = monthly[monthly['cloud_cover'] <= monthly['cloud_cover'].median()]

print(f"\nDeaths per event comparison:")
print(f"  High cloud months (>{monthly['cloud_cover'].median():.0f}%): {high_cloud['deaths_per_event'].mean():.2f} deaths/event")
print(f"  Low cloud months (≤{monthly['cloud_cover'].median():.0f}%):  {low_cloud['deaths_per_event'].mean():.2f} deaths/event")

t_stat, t_p = stats.ttest_ind(high_cloud['deaths_per_event'].dropna(),
                               low_cloud['deaths_per_event'].dropna())
print(f"  t-test: t = {t_stat:.2f}, p = {t_p:.4f}")

# =============================================================================
# 2. CASUALTY COMPOSITION
# =============================================================================
print("\n\n[2] CASUALTY COMPOSITION vs CLOUD COVER")
print("-" * 60)

# Does the ratio of military to civilian deaths change with cloud cover?
valid2 = monthly.dropna(subset=['cloud_cover', 'military_ratio'])
r_mil, p_mil = stats.pearsonr(valid2['cloud_cover'], valid2['military_ratio'])
print(f"\nCloud cover vs Military death ratio:")
print(f"  r = {r_mil:.3f}, p = {p_mil:.4f}")

print(f"\nMilitary death ratio comparison:")
print(f"  High cloud months: {high_cloud['military_ratio'].mean():.1%} military")
print(f"  Low cloud months:  {low_cloud['military_ratio'].mean():.1%} military")

# =============================================================================
# 3. EVENT FREQUENCY vs LETHALITY
# =============================================================================
print("\n\n[3] EVENT FREQUENCY vs LETHALITY")
print("-" * 60)

# Are there more events under cloud cover, or just more deadly events?
r_events, p_events = stats.pearsonr(valid['cloud_cover'], valid['id'])
print(f"\nCloud cover vs Number of events:")
print(f"  r = {r_events:.3f}, p = {p_events:.4f}")

print(f"\nEvent count comparison:")
print(f"  High cloud months: {high_cloud['id'].mean():.0f} events/month")
print(f"  Low cloud months:  {low_cloud['id'].mean():.0f} events/month")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
print("\n\n[4] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Cloud Cover & Combat Dynamics: FPV Hypothesis Test\nUkraine May 2022 - Dec 2024',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Deaths per event vs cloud cover
ax1 = axes[0, 0]
ax1.scatter(monthly['cloud_cover'], monthly['deaths_per_event'], alpha=0.7, s=60, c='#E63946')
z = np.polyfit(valid['cloud_cover'], valid['deaths_per_event'], 1)
p = np.poly1d(z)
x_line = np.linspace(valid['cloud_cover'].min(), valid['cloud_cover'].max(), 100)
ax1.plot(x_line, p(x_line), 'k--', linewidth=2)
ax1.text(0.05, 0.95, f'r = {r_dpe:.3f}\np = {p_dpe:.4f}', transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('Cloud Cover (%)')
ax1.set_ylabel('Deaths per Event')
ax1.set_title('Engagement Lethality vs Cloud Cover', fontweight='bold')

# Plot 2: Military death ratio vs cloud cover
ax2 = axes[0, 1]
ax2.scatter(monthly['cloud_cover'], monthly['military_ratio']*100, alpha=0.7, s=60, c='#264653')
ax2.text(0.05, 0.95, f'r = {r_mil:.3f}\np = {p_mil:.4f}', transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.set_xlabel('Cloud Cover (%)')
ax2.set_ylabel('Military Deaths (%)')
ax2.set_title('Casualty Composition vs Cloud Cover', fontweight='bold')
ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

# Plot 3: Box plots comparing high vs low cloud
ax3 = axes[1, 0]
box_data = [low_cloud['deaths_per_event'].dropna(), high_cloud['deaths_per_event'].dropna()]
bp = ax3.boxplot(box_data, labels=['Low Cloud\n(≤55%)', 'High Cloud\n(>55%)'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2A9D8F')
bp['boxes'][1].set_facecolor('#E63946')
ax3.set_ylabel('Deaths per Event')
ax3.set_title(f'Deaths per Event by Cloud Condition\nt-test p = {t_p:.4f}', fontweight='bold')

# Plot 4: Time series of deaths per event and cloud cover
ax4 = axes[1, 1]
ax4.fill_between(monthly['month'], 0, monthly['cloud_cover'], alpha=0.3, color='gray', label='Cloud %')
ax4b = ax4.twinx()
ax4b.plot(monthly['month'], monthly['deaths_per_event'], 'o-', color='#E63946',
          label='Deaths/Event', linewidth=2, markersize=4)
ax4.set_xlabel('Date')
ax4.set_ylabel('Cloud Cover %', color='gray')
ax4b.set_ylabel('Deaths per Event', color='#E63946')
ax4.set_title('Temporal Pattern', fontweight='bold')

import matplotlib.dates as mdates
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/analysis/09_fpv_cloud_hypothesis.png', dpi=150, bbox_inches='tight')
print("  Saved: 09_fpv_cloud_hypothesis.png")

# =============================================================================
# 5. SUMMARY
# =============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY: FPV/CLOUD COVER HYPOTHESIS")
print("=" * 80)

print(f"""
HYPOTHESIS: Cloud cover increases FPV drone effectiveness by reducing visual
detection, leading to higher casualties per engagement.

FINDINGS:

1. DEATHS PER EVENT vs CLOUD COVER:
   Correlation: r = {r_dpe:+.3f} (p = {p_dpe:.4f})
   {"SUPPORTS hypothesis" if r_dpe > 0.15 and p_dpe < 0.1 else "WEAK/NO support" if r_dpe > 0 else "CONTRADICTS hypothesis"}

   High cloud months: {high_cloud['deaths_per_event'].mean():.2f} deaths/event
   Low cloud months:  {low_cloud['deaths_per_event'].mean():.2f} deaths/event
   Difference: {((high_cloud['deaths_per_event'].mean() / low_cloud['deaths_per_event'].mean()) - 1) * 100:+.1f}%

2. CASUALTY COMPOSITION:
   Military death ratio correlation: r = {r_mil:+.3f} (p = {p_mil:.4f})
   {"Higher military % under clouds" if r_mil > 0.1 else "No clear pattern"}

3. EVENT FREQUENCY:
   Cloud-event correlation: r = {r_events:+.3f} (p = {p_events:.4f})
   {"Fewer events under clouds" if r_events < -0.1 else "More events under clouds" if r_events > 0.1 else "No clear pattern"}

INTERPRETATION:
""")

if r_dpe > 0.2:
    print("  The data shows HIGHER lethality per engagement under cloud cover.")
    print("  This is consistent with (but doesn't prove) the FPV hypothesis:")
    print("  - Drones harder to spot → more successful strikes")
    print("  - Reduced aerial ISR → more close-quarters combat")
    print("  - Weather forcing infantry vs armor engagements")
elif r_dpe < -0.1:
    print("  The data shows LOWER lethality under cloud cover.")
    print("  This CONTRADICTS the FPV hypothesis, suggesting:")
    print("  - FPV effectiveness may actually decrease (guidance issues?)")
    print("  - Or other factors dominate the relationship")
else:
    print("  No clear relationship between cloud cover and engagement lethality.")
    print("  The FPV hypothesis is neither supported nor refuted by this data.")
    print("  The overall deaths-cloud correlation may be driven by:")
    print("  - Number of engagements, not lethality per engagement")
    print("  - Strategic timing of offensives (winter campaigns)")
