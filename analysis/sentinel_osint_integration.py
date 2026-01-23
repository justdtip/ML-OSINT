#!/usr/bin/env python3
"""
Integrate Sentinel satellite data with existing OSINT datasets.

Correlates:
- Sentinel-2 optical imagery availability (cloud cover patterns)
- Sentinel-1 radar imagery (all-weather coverage)
- Sentinel-5P NO2/CO atmospheric data (industrial/military activity proxy)
- Sentinel-3 fire radiative power (cross-validate with FIRMS)

With existing datasets:
- UCDP conflict events
- NASA FIRMS fire hotspots
- DeepState territorial maps
- War losses data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import os
from datetime import datetime, timedelta

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

# Paths
BASE_DIR = str(PROJECT_ROOT)
OUTPUT_DIR = str(ANALYSIS_DIR)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("=" * 80)
print("SENTINEL + OSINT DATA INTEGRATION ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD SENTINEL DATA
# =============================================================================
print("\n[1/6] Loading Sentinel data...")

with open(f"{BASE_DIR}/data/sentinel/sentinel_timeseries_raw.json") as f:
    sentinel_raw = json.load(f)

# Create monthly dataframe
sentinel_monthly = []
for coll_id, coll_data in sentinel_raw['collections'].items():
    for month in coll_data['monthly']:
        sentinel_monthly.append({
            'collection': coll_id,
            'name': coll_data['name'],
            'year': month['year'],
            'month': month['month'],
            'date': f"{month['year']}-{month['month']:02d}-01",
            'count': month['count'],
            'avg_cloud': month.get('avg_cloud_cover'),
            'min_cloud': month.get('min_cloud_cover'),
            'cloud_free': month.get('cloud_free_count', 0)
        })

sentinel_df = pd.DataFrame(sentinel_monthly)
sentinel_df['date'] = pd.to_datetime(sentinel_df['date'])

# Pivot to wide format
sentinel_pivot = sentinel_df.pivot_table(
    index='date',
    columns='collection',
    values='count',
    aggfunc='sum'
).reset_index()

# Also get cloud cover for optical
s2_cloud = sentinel_df[sentinel_df['collection'] == 'sentinel-2-l2a'][['date', 'avg_cloud', 'cloud_free']].copy()
s2_cloud.columns = ['date', 's2_avg_cloud', 's2_cloud_free']

sentinel_pivot = sentinel_pivot.merge(s2_cloud, on='date', how='left')

print(f"  Sentinel monthly records: {len(sentinel_pivot)}")
print(f"  Date range: {sentinel_pivot['date'].min()} to {sentinel_pivot['date'].max()}")

# =============================================================================
# 2. LOAD EXISTING OSINT DATASETS
# =============================================================================
print("\n[2/6] Loading existing OSINT datasets...")

# UCDP
ucdp_df = pd.read_csv(f"{BASE_DIR}/data/ucdp/ged_events.csv", low_memory=False)
ucdp_df['date_start'] = pd.to_datetime(ucdp_df['date_start'], format='mixed')
ucdp_df = ucdp_df[ucdp_df['date_start'] >= '2022-02-24']
print(f"  UCDP: {len(ucdp_df)} events")

# FIRMS - combine archive and NRT data
firms_archive = pd.read_csv(f"{BASE_DIR}/data/firms/DL_FIRE_SV-C2_706038/fire_archive_SV-C2_706038.csv")
firms_nrt = pd.read_csv(f"{BASE_DIR}/data/firms/DL_FIRE_SV-C2_706038/fire_nrt_SV-C2_706038.csv")
firms_df = pd.concat([firms_archive, firms_nrt], ignore_index=True)
firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'])
print(f"  FIRMS: {len(firms_df)} fire hotspots")

# DeepState
deepstate_index = f"{BASE_DIR}/data/deepstate/wayback_snapshots/_index.json"
with open(deepstate_index) as f:
    ds_index = json.load(f)
deepstate_df = pd.DataFrame(ds_index)
deepstate_df['date'] = pd.to_datetime(deepstate_df['datetime'].str[:10])
print(f"  DeepState: {len(deepstate_df)} snapshots")

# War Losses (use personnel file which has cumulative totals)
losses_path = f"{BASE_DIR}/data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_personnel.json"
with open(losses_path) as f:
    losses_data = json.load(f)
losses_df = pd.DataFrame(losses_data)
losses_df['date'] = pd.to_datetime(losses_df['date'])
print(f"  War Losses: {len(losses_df)} daily records")

# =============================================================================
# 3. CREATE MONTHLY AGGREGATIONS
# =============================================================================
print("\n[3/6] Creating monthly aggregations...")

# UCDP monthly
ucdp_monthly = ucdp_df.set_index('date_start').resample('MS').agg({
    'id': 'count',
    'best_est': 'sum'
}).reset_index()
ucdp_monthly.columns = ['date', 'ucdp_events', 'ucdp_deaths']

# FIRMS monthly
firms_monthly = firms_df.set_index('acq_date').resample('MS').agg({
    'latitude': 'count',
    'frp': 'sum'
}).reset_index()
firms_monthly.columns = ['date', 'firms_fires', 'firms_frp']

# DeepState monthly (average points per snapshot in month)
deepstate_monthly = deepstate_df.set_index('date').resample('MS').agg({
    'points': 'mean',
    'polygons': 'mean',
    'total_features': 'mean'
}).reset_index()
deepstate_monthly.columns = ['date', 'ds_points', 'ds_polygons', 'ds_features']

# War losses monthly
losses_df['daily_loss'] = losses_df['personnel'].diff().fillna(0)
losses_monthly = losses_df.set_index('date').resample('MS').agg({
    'daily_loss': 'sum',
    'personnel': 'last'
}).reset_index()
losses_monthly.columns = ['date', 'monthly_loss', 'cumulative_personnel']

# Merge all
merged = sentinel_pivot.copy()
merged = merged.merge(ucdp_monthly, on='date', how='outer')
merged = merged.merge(firms_monthly, on='date', how='outer')
merged = merged.merge(deepstate_monthly, on='date', how='outer')
merged = merged.merge(losses_monthly, on='date', how='outer')

# Filter to overlap period
merged = merged[(merged['date'] >= '2022-05-01') & (merged['date'] <= '2024-12-31')]
merged = merged.sort_values('date')

print(f"  Merged monthly records: {len(merged)}")

# Rename Sentinel columns for clarity
merged = merged.rename(columns={
    'sentinel-1-grd': 's1_radar',
    'sentinel-2-l2a': 's2_optical',
    'sentinel-3-sl-2-frp-ntc': 's3_fire',
    'sentinel-5p-l2-co-offl': 's5p_co',
    'sentinel-5p-l2-no2-offl': 's5p_no2'
})

# =============================================================================
# 4. VISUALIZATION - TIME SERIES
# =============================================================================
print("\n[4/6] Creating time series visualizations...")

fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig.suptitle('Sentinel Satellite Data + OSINT Integration\nUkraine Conflict Zone (May 2022 - Dec 2024)',
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Sentinel product counts
ax1 = axes[0]
ax1.bar(merged['date'], merged['s2_optical'], label='Sentinel-2 Optical', alpha=0.7, color='#2A9D8F', width=20)
ax1.bar(merged['date'], merged['s1_radar'], label='Sentinel-1 Radar', alpha=0.7, color='#E76F51', width=20, bottom=merged['s2_optical'].fillna(0))
ax1.set_ylabel('Products per Month')
ax1.set_title('Sentinel-1/2 Imagery Availability', fontweight='bold')
ax1.legend(loc='upper right')

# Add cloud cover on secondary axis
ax1b = ax1.twinx()
ax1b.plot(merged['date'], merged['s2_avg_cloud'], 'k--', alpha=0.6, label='Avg Cloud Cover %')
ax1b.set_ylabel('Cloud Cover %', color='gray')
ax1b.tick_params(axis='y', labelcolor='gray')
ax1b.set_ylim(0, 100)

# Plot 2: Atmospheric data vs conflict
ax2 = axes[1]
ax2.plot(merged['date'], merged['s5p_no2'], 'o-', label='Sentinel-5P NO2 Products', color='#9B2335', alpha=0.7)
ax2.set_ylabel('S5P Products', color='#9B2335')
ax2.tick_params(axis='y', labelcolor='#9B2335')

ax2b = ax2.twinx()
ax2b.bar(merged['date'], merged['ucdp_events'], alpha=0.4, color='#264653', width=20, label='UCDP Events')
ax2b.set_ylabel('UCDP Events', color='#264653')
ax2b.tick_params(axis='y', labelcolor='#264653')
ax2.set_title('Atmospheric Monitoring (NO2) vs Conflict Events', fontweight='bold')

# Plot 3: Fire detection comparison
ax3 = axes[2]
ax3.bar(merged['date'], merged['firms_fires']/1000, alpha=0.7, color='#F4A261', width=20, label='FIRMS Fires (thousands)')
ax3.set_ylabel('FIRMS Fires (K)', color='#F4A261')
ax3.tick_params(axis='y', labelcolor='#F4A261')

ax3b = ax3.twinx()
ax3b.plot(merged['date'], merged['s3_fire'], 's-', color='#E63946', alpha=0.7, label='Sentinel-3 FRP Products')
ax3b.set_ylabel('S3 FRP Products', color='#E63946')
ax3b.tick_params(axis='y', labelcolor='#E63946')
ax3.set_title('Fire Detection: NASA FIRMS vs Sentinel-3 SLSTR', fontweight='bold')

# Plot 4: DeepState + Losses
ax4 = axes[3]
ax4.plot(merged['date'], merged['ds_points'], 'o-', color='#2A9D8F', label='DeepState Points (avg)')
ax4.set_ylabel('DeepState Points', color='#2A9D8F')
ax4.tick_params(axis='y', labelcolor='#2A9D8F')

ax4b = ax4.twinx()
ax4b.bar(merged['date'], merged['monthly_loss']/1000, alpha=0.5, color='#264653', width=20, label='Monthly Losses (K)')
ax4b.set_ylabel('Monthly Losses (K)', color='#264653')
ax4b.tick_params(axis='y', labelcolor='#264653')
ax4.set_title('DeepState Military Units vs Personnel Losses', fontweight='bold')

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_sentinel_osint_timeseries.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/05_sentinel_osint_timeseries.png")

# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================
print("\n[5/6] Analyzing correlations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Sentinel + OSINT Cross-Dataset Correlations', fontsize=16, fontweight='bold', y=1.02)

# Select correlation columns
corr_cols = ['s2_optical', 's1_radar', 's5p_no2', 's3_fire', 's2_avg_cloud',
             'ucdp_events', 'ucdp_deaths', 'firms_fires', 'ds_points', 'monthly_loss']
corr_data = merged[corr_cols].dropna()

print(f"  Correlation data points: {len(corr_data)} months")

# Plot 1: Full correlation matrix
ax1 = axes[0, 0]
if len(corr_data) > 5:
    corr_matrix = corr_data.corr()
    labels = ['S2 Optical', 'S1 Radar', 'S5P NO2', 'S3 Fire', 'Cloud %',
              'UCDP Events', 'UCDP Deaths', 'FIRMS Fires', 'DS Points', 'Losses']
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                xticklabels=labels, yticklabels=labels, ax=ax1, fmt='.2f',
                vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Matrix (Monthly)', fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)

# Plot 2: FIRMS vs Sentinel-3 Fire scatter
ax2 = axes[0, 1]
valid = merged.dropna(subset=['firms_fires', 's3_fire'])
if len(valid) > 5:
    ax2.scatter(valid['firms_fires']/1000, valid['s3_fire'], alpha=0.6, s=60, c='#E63946')

    # Add trend line
    z = np.polyfit(valid['firms_fires']/1000, valid['s3_fire'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['firms_fires'].min()/1000, valid['firms_fires'].max()/1000, 100)
    ax2.plot(x_line, p(x_line), 'k--', alpha=0.7)

    corr = valid['firms_fires'].corr(valid['s3_fire'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('FIRMS Fire Hotspots (thousands)')
ax2.set_ylabel('Sentinel-3 FRP Products')
ax2.set_title('Fire Detection Cross-Validation\nFIRMS vs Sentinel-3', fontweight='bold')

# Plot 3: NO2 vs Conflict intensity
ax3 = axes[1, 0]
valid = merged.dropna(subset=['s5p_no2', 'ucdp_events'])
if len(valid) > 5:
    scatter = ax3.scatter(valid['s5p_no2'], valid['ucdp_events'],
                          c=valid['monthly_loss']/1000 if 'monthly_loss' in valid else None,
                          cmap='YlOrRd', alpha=0.7, s=60)
    if 'monthly_loss' in valid:
        plt.colorbar(scatter, ax=ax3, label='Monthly Losses (K)', shrink=0.7)

    corr = valid['s5p_no2'].corr(valid['ucdp_events'])
    ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax3.set_xlabel('Sentinel-5P NO2 Products')
ax3.set_ylabel('UCDP Conflict Events')
ax3.set_title('Atmospheric Monitoring vs Conflict\n(colored by losses)', fontweight='bold')

# Plot 4: Cloud cover vs observable activity
ax4 = axes[1, 1]
valid = merged.dropna(subset=['s2_avg_cloud', 'firms_fires'])
if len(valid) > 5:
    ax4.scatter(valid['s2_avg_cloud'], valid['firms_fires']/1000, alpha=0.6, s=60, c='#2A9D8F')

    # Add trend line
    z = np.polyfit(valid['s2_avg_cloud'], valid['firms_fires']/1000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['s2_avg_cloud'].min(), valid['s2_avg_cloud'].max(), 100)
    ax4.plot(x_line, p(x_line), 'k--', alpha=0.7)

    corr = valid['s2_avg_cloud'].corr(valid['firms_fires'])
    ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_xlabel('Sentinel-2 Avg Cloud Cover %')
ax4.set_ylabel('FIRMS Fire Hotspots (thousands)')
ax4.set_title('Cloud Cover Impact on Fire Detection', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_sentinel_osint_correlations.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}/06_sentinel_osint_correlations.png")

# =============================================================================
# 6. SUMMARY STATISTICS
# =============================================================================
print("\n[6/6] Computing summary statistics...")

print("\n" + "=" * 80)
print("CORRELATION SUMMARY")
print("=" * 80)

# Key correlations
key_correlations = [
    ('FIRMS Fires', 'S3 Fire Products', 'firms_fires', 's3_fire'),
    ('UCDP Events', 'S5P NO2 Products', 'ucdp_events', 's5p_no2'),
    ('UCDP Deaths', 'Monthly Losses', 'ucdp_deaths', 'monthly_loss'),
    ('DeepState Points', 'Monthly Losses', 'ds_points', 'monthly_loss'),
    ('Cloud Cover', 'FIRMS Fires', 's2_avg_cloud', 'firms_fires'),
    ('S2 Optical Products', 'UCDP Events', 's2_optical', 'ucdp_events'),
]

print(f"\n{'Variable 1':<25} {'Variable 2':<25} {'Correlation':>12} {'Interpretation':<30}")
print("-" * 95)

for name1, name2, col1, col2 in key_correlations:
    valid = merged[[col1, col2]].dropna()
    if len(valid) > 5:
        r = valid[col1].corr(valid[col2])
        if abs(r) > 0.7:
            interp = "Strong relationship"
        elif abs(r) > 0.4:
            interp = "Moderate relationship"
        elif abs(r) > 0.2:
            interp = "Weak relationship"
        else:
            interp = "No clear relationship"
        print(f"{name1:<25} {name2:<25} {r:>12.3f} {interp:<30}")
    else:
        print(f"{name1:<25} {name2:<25} {'N/A':>12} {'Insufficient data':<30}")

# Data volume summary
print("\n" + "=" * 80)
print("DATA VOLUME SUMMARY (May 2022 - Dec 2024)")
print("=" * 80)

print(f"\n{'Dataset':<30} {'Total Records':>15} {'Monthly Avg':>15}")
print("-" * 65)
print(f"{'Sentinel-2 Optical Products':<30} {merged['s2_optical'].sum():>15,.0f} {merged['s2_optical'].mean():>15,.0f}")
print(f"{'Sentinel-1 Radar Products':<30} {merged['s1_radar'].sum():>15,.0f} {merged['s1_radar'].mean():>15,.0f}")
print(f"{'Sentinel-5P NO2 Products':<30} {merged['s5p_no2'].sum():>15,.0f} {merged['s5p_no2'].mean():>15,.0f}")
print(f"{'Sentinel-3 Fire Products':<30} {merged['s3_fire'].sum():>15,.0f} {merged['s3_fire'].mean():>15,.0f}")
print(f"{'UCDP Conflict Events':<30} {merged['ucdp_events'].sum():>15,.0f} {merged['ucdp_events'].mean():>15,.0f}")
print(f"{'FIRMS Fire Hotspots':<30} {merged['firms_fires'].sum():>15,.0f} {merged['firms_fires'].mean():>15,.0f}")
print(f"{'War Losses (personnel)':<30} {merged['monthly_loss'].sum():>15,.0f} {merged['monthly_loss'].mean():>15,.0f}")

# Save merged data
merged.to_csv(f"{OUTPUT_DIR}/sentinel_osint_merged.csv", index=False)
print(f"\nMerged data saved: {OUTPUT_DIR}/sentinel_osint_merged.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
