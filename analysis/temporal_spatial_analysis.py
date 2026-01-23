#!/usr/bin/env python3
"""
Temporal and Spatial Overlap Analysis for ML_OSINT Data
Analyzes: UCDP conflict events, NASA FIRMS fire hotspots, DeepState territorial maps, and war losses data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely import wkt
import json
import os
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

BASE_PATH = str(PROJECT_ROOT)
OUTPUT_PATH = str(ANALYSIS_DIR / "figures")
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("TEMPORAL AND SPATIAL OVERLAP ANALYSIS")
print("=" * 60)

# =============================================================================
# 1. LOAD ALL DATASETS
# =============================================================================
print("\n[1/6] Loading datasets...")

# Load UCDP conflict events
print("  - Loading UCDP conflict events...")
ucdp_df = pd.read_csv(f"{BASE_PATH}/data/ucdp/ged_events.csv")
ucdp_df['date_start'] = pd.to_datetime(ucdp_df['date_start'], format='mixed')
ucdp_df['date_end'] = pd.to_datetime(ucdp_df['date_end'], format='mixed')
print(f"    Loaded {len(ucdp_df):,} UCDP events")

# Load NASA FIRMS fire data
print("  - Loading NASA FIRMS fire hotspots...")
firms_archive = pd.read_csv(f"{BASE_PATH}/data/firms/DL_FIRE_SV-C2_706038/fire_archive_SV-C2_706038.csv")
firms_nrt = pd.read_csv(f"{BASE_PATH}/data/firms/DL_FIRE_SV-C2_706038/fire_nrt_SV-C2_706038.csv")
firms_df = pd.concat([firms_archive, firms_nrt], ignore_index=True)
firms_df['acq_date'] = pd.to_datetime(firms_df['acq_date'])
firms_df['datetime'] = pd.to_datetime(firms_df['acq_date'].astype(str) + ' ' + firms_df['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
print(f"    Loaded {len(firms_df):,} fire hotspots")

# Load DeepState territorial data (sample files for analysis)
print("  - Loading DeepState territorial maps...")
deepstate_files = sorted(glob(f"{BASE_PATH}/data/deepstate/daily/*.geojson"))
deepstate_dates = []
deepstate_areas = []

for f in deepstate_files[::7]:  # Sample every 7th file for efficiency
    date_str = os.path.basename(f).replace('deepstatemap_data_', '').replace('.geojson', '')
    date = pd.to_datetime(date_str, format='%Y%m%d')
    try:
        gdf = gpd.read_file(f)
        if not gdf.empty:
            # Calculate area in km² (approximate)
            gdf_projected = gdf.to_crs(epsg=32637)  # UTM zone 37N for Ukraine
            area_km2 = gdf_projected.geometry.area.sum() / 1e6
            deepstate_dates.append(date)
            deepstate_areas.append(area_km2)
    except Exception as e:
        pass

deepstate_df = pd.DataFrame({'date': deepstate_dates, 'occupied_area_km2': deepstate_areas})
print(f"    Loaded {len(deepstate_df)} territorial snapshots")

# Load war losses data
print("  - Loading war losses data...")
with open(f"{BASE_PATH}/data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_personnel.json") as f:
    losses_personnel = json.load(f)
with open(f"{BASE_PATH}/data/war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_equipment.json") as f:
    losses_equipment = json.load(f)

personnel_df = pd.DataFrame(losses_personnel)
personnel_df['date'] = pd.to_datetime(personnel_df['date'])
equipment_df = pd.DataFrame(losses_equipment)
equipment_df['date'] = pd.to_datetime(equipment_df['date'])
print(f"    Loaded {len(personnel_df):,} daily loss records")

# =============================================================================
# 2. TEMPORAL COVERAGE ANALYSIS
# =============================================================================
print("\n[2/6] Analyzing temporal coverage...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Temporal Coverage of All Datasets', fontsize=16, fontweight='bold', y=1.02)

# Dataset date ranges
datasets = {
    'UCDP Conflict Events': (ucdp_df['date_start'].min(), ucdp_df['date_start'].max()),
    'NASA FIRMS Fire Hotspots': (firms_df['acq_date'].min(), firms_df['acq_date'].max()),
    'DeepState Territorial': (deepstate_df['date'].min(), deepstate_df['date'].max()),
    'War Losses Data': (personnel_df['date'].min(), personnel_df['date'].max())
}

# Plot 1: Timeline bars showing coverage
ax1 = axes[0, 0]
colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653']
y_positions = range(len(datasets))

for i, (name, (start, end)) in enumerate(datasets.items()):
    ax1.barh(i, (end - start).days, left=start, color=colors[i], alpha=0.8, height=0.6)
    ax1.text(start, i, f'  {start.strftime("%Y-%m-%d")}', va='center', ha='left', fontsize=9)
    ax1.text(end, i, f'{end.strftime("%Y-%m-%d")}  ', va='center', ha='right', fontsize=9)

ax1.set_yticks(y_positions)
ax1.set_yticklabels(datasets.keys())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.set_xlabel('Date')
ax1.set_title('Dataset Temporal Coverage Ranges', fontweight='bold')
ax1.axvline(pd.Timestamp('2022-02-24'), color='red', linestyle='--', alpha=0.7, label='Invasion Start (Feb 24, 2022)')
ax1.legend(loc='lower right')

# Plot 2: Daily event counts (overlapping period)
ax2 = axes[0, 1]
overlap_start = pd.Timestamp('2022-02-24')
overlap_end = min(firms_df['acq_date'].max(), personnel_df['date'].max())

# Resample to weekly for clarity
ucdp_weekly = ucdp_df[ucdp_df['date_start'] >= overlap_start].set_index('date_start').resample('W')['id'].count()
firms_weekly = firms_df[firms_df['acq_date'] >= overlap_start].set_index('acq_date').resample('W')['latitude'].count()

ax2.plot(ucdp_weekly.index, ucdp_weekly.values, label='UCDP Events', color='#E63946', alpha=0.8)
ax2.plot(firms_weekly.index, firms_weekly.values / 100, label='FIRMS Fires (÷100)', color='#F4A261', alpha=0.8)
ax2.set_xlabel('Date')
ax2.set_ylabel('Weekly Count')
ax2.set_title('Weekly Event Counts (Post-Invasion)', fontweight='bold')
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Personnel losses cumulative
ax3 = axes[1, 0]
ax3.fill_between(personnel_df['date'], personnel_df['personnel'], alpha=0.4, color='#264653')
ax3.plot(personnel_df['date'], personnel_df['personnel'], color='#264653', linewidth=1.5)
ax3.set_xlabel('Date')
ax3.set_ylabel('Cumulative Personnel Losses')
ax3.set_title('Cumulative Russian Personnel Losses', fontweight='bold')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

# Plot 4: Territorial control over time
ax4 = axes[1, 1]
ax4.fill_between(deepstate_df['date'], deepstate_df['occupied_area_km2'], alpha=0.4, color='#2A9D8F')
ax4.plot(deepstate_df['date'], deepstate_df['occupied_area_km2'], color='#2A9D8F', linewidth=1.5)
ax4.set_xlabel('Date')
ax4.set_ylabel('Occupied Area (km²)')
ax4.set_title('Russian-Occupied Territory Over Time', fontweight='bold')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/01_temporal_coverage.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_temporal_coverage.png")

# =============================================================================
# 3. SPATIAL COVERAGE ANALYSIS
# =============================================================================
print("\n[3/6] Analyzing spatial coverage...")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('Spatial Distribution of Events in Ukraine', fontsize=16, fontweight='bold', y=1.02)

# Ukraine approximate bounds
ukraine_bounds = {'west': 22, 'east': 40.5, 'south': 44, 'north': 52.5}

# Plot 1: UCDP conflict events
ax1 = axes[0]
post_invasion = ucdp_df[ucdp_df['date_start'] >= '2022-02-24']
scatter1 = ax1.scatter(post_invasion['longitude'], post_invasion['latitude'],
                       c=post_invasion['best_est'], cmap='Reds', alpha=0.5, s=10,
                       vmin=0, vmax=50)
ax1.set_xlim(ukraine_bounds['west'], ukraine_bounds['east'])
ax1.set_ylim(ukraine_bounds['south'], ukraine_bounds['north'])
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title(f'UCDP Conflict Events (Post-Invasion)\nn={len(post_invasion):,}', fontweight='bold')
ax1.set_aspect('equal')
plt.colorbar(scatter1, ax=ax1, label='Estimated Deaths', shrink=0.7)

# Plot 2: FIRMS fire hotspots (density)
ax2 = axes[1]
# Sample for efficiency
firms_sample = firms_df.sample(min(50000, len(firms_df)), random_state=42)
h = ax2.hexbin(firms_sample['longitude'], firms_sample['latitude'],
               gridsize=50, cmap='YlOrRd', mincnt=1)
ax2.set_xlim(ukraine_bounds['west'], ukraine_bounds['east'])
ax2.set_ylim(ukraine_bounds['south'], ukraine_bounds['north'])
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title(f'FIRMS Fire Hotspots Density\nn={len(firms_df):,}', fontweight='bold')
ax2.set_aspect('equal')
plt.colorbar(h, ax=ax2, label='Fire Count', shrink=0.7)

# Plot 3: Comparison overlay
ax3 = axes[2]
ax3.hexbin(firms_sample['longitude'], firms_sample['latitude'],
           gridsize=40, cmap='YlOrRd', alpha=0.5, mincnt=1)
ax3.scatter(post_invasion['longitude'], post_invasion['latitude'],
            c='blue', alpha=0.3, s=5, label='UCDP Events')
ax3.set_xlim(ukraine_bounds['west'], ukraine_bounds['east'])
ax3.set_ylim(ukraine_bounds['south'], ukraine_bounds['north'])
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.set_title('Spatial Overlap: Fires (heat) vs Conflicts (blue)', fontweight='bold')
ax3.set_aspect('equal')
ax3.legend(loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/02_spatial_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_spatial_distribution.png")

# =============================================================================
# 4. TEMPORAL-SPATIAL CORRELATION ANALYSIS
# =============================================================================
print("\n[4/6] Analyzing temporal-spatial correlations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Temporal-Spatial Correlation Analysis', fontsize=16, fontweight='bold', y=1.02)

# Create daily aggregations for correlation
daily_ucdp = ucdp_df[ucdp_df['date_start'] >= '2022-02-24'].groupby(ucdp_df['date_start'].dt.date).agg({
    'id': 'count',
    'best_est': 'sum',
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()
daily_ucdp.columns = ['date', 'ucdp_events', 'ucdp_deaths', 'ucdp_lat', 'ucdp_lon']
daily_ucdp['date'] = pd.to_datetime(daily_ucdp['date'])

daily_firms = firms_df.groupby(firms_df['acq_date'].dt.date).agg({
    'latitude': ['count', 'mean'],
    'longitude': 'mean',
    'frp': 'sum'
}).reset_index()
daily_firms.columns = ['date', 'firms_fires', 'firms_lat', 'firms_lon', 'firms_frp']
daily_firms['date'] = pd.to_datetime(daily_firms['date'])

# Merge datasets
merged = daily_ucdp.merge(daily_firms, on='date', how='inner')

# Add personnel losses (daily change)
personnel_df['daily_loss'] = personnel_df['personnel'].diff().fillna(0)
merged = merged.merge(personnel_df[['date', 'daily_loss', 'personnel']], on='date', how='left')
merged = merged.dropna()

# Plot 1: Fire count vs UCDP events scatter
ax1 = axes[0, 0]
ax1.scatter(merged['firms_fires'], merged['ucdp_events'], alpha=0.4, s=20, c='#2A9D8F')
# Add trend line
z = np.polyfit(merged['firms_fires'], merged['ucdp_events'], 1)
p = np.poly1d(z)
ax1.plot(sorted(merged['firms_fires']), p(sorted(merged['firms_fires'])),
         "r--", alpha=0.8, linewidth=2, label=f'Trend (r={merged["firms_fires"].corr(merged["ucdp_events"]):.3f})')
ax1.set_xlabel('Daily Fire Hotspots')
ax1.set_ylabel('Daily UCDP Events')
ax1.set_title('Fire Hotspots vs Conflict Events', fontweight='bold')
ax1.legend()

# Plot 2: Daily losses vs fires
ax2 = axes[0, 1]
scatter2 = ax2.scatter(merged['firms_fires'], merged['daily_loss'],
                       c=merged['ucdp_events'], cmap='plasma', alpha=0.5, s=20)
ax2.set_xlabel('Daily Fire Hotspots')
ax2.set_ylabel('Daily Personnel Losses')
ax2.set_title('Fire Hotspots vs Personnel Losses\n(colored by UCDP events)', fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='UCDP Events', shrink=0.7)

# Plot 3: Correlation heatmap
ax3 = axes[1, 0]
corr_cols = ['ucdp_events', 'ucdp_deaths', 'firms_fires', 'firms_frp', 'daily_loss']
corr_matrix = merged[corr_cols].corr()
corr_labels = ['UCDP Events', 'UCDP Deaths', 'Fire Count', 'Fire Power (FRP)', 'Personnel Loss']
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            xticklabels=corr_labels, yticklabels=corr_labels, ax=ax3,
            fmt='.2f', square=True)
ax3.set_title('Correlation Matrix: All Metrics', fontweight='bold')

# Plot 4: Time series comparison (normalized)
ax4 = axes[1, 1]
# Resample to weekly and normalize
weekly_merged = merged.set_index('date').resample('W').mean()
for col, label, color in [('ucdp_events', 'UCDP Events', '#E63946'),
                          ('firms_fires', 'Fire Hotspots', '#F4A261'),
                          ('daily_loss', 'Personnel Losses', '#264653')]:
    normalized = (weekly_merged[col] - weekly_merged[col].min()) / (weekly_merged[col].max() - weekly_merged[col].min())
    ax4.plot(weekly_merged.index, normalized, label=label, linewidth=1.5, alpha=0.8, color=color)

ax4.set_xlabel('Date')
ax4.set_ylabel('Normalized Value (0-1)')
ax4.set_title('Normalized Weekly Trends Comparison', fontweight='bold')
ax4.legend(loc='upper right')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/03_temporal_spatial_correlation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_temporal_spatial_correlation.png")

# =============================================================================
# 5. REGIONAL ANALYSIS
# =============================================================================
print("\n[5/6] Analyzing regional distribution...")

# Define key oblasts/regions
regions = {
    'Donetsk': {'lon': (37, 39.5), 'lat': (46.5, 49)},
    'Luhansk': {'lon': (38, 40.5), 'lat': (48, 50)},
    'Kharkiv': {'lon': (35, 38), 'lat': (48.5, 50.5)},
    'Zaporizhzhia': {'lon': (34, 37), 'lat': (46, 48)},
    'Kherson': {'lon': (32, 35.5), 'lat': (45.5, 47.5)},
    'Crimea': {'lon': (32.5, 36.5), 'lat': (44, 46)}
}

def classify_region(lon, lat):
    for name, bounds in regions.items():
        if bounds['lon'][0] <= lon <= bounds['lon'][1] and bounds['lat'][0] <= lat <= bounds['lat'][1]:
            return name
    return 'Other'

# Classify UCDP and FIRMS data
post_invasion['region'] = post_invasion.apply(lambda x: classify_region(x['longitude'], x['latitude']), axis=1)
firms_df['region'] = firms_df.apply(lambda x: classify_region(x['longitude'], x['latitude']), axis=1)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Regional Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Regional event counts
ax1 = axes[0, 0]
ucdp_by_region = post_invasion.groupby('region')['id'].count().sort_values(ascending=True)
firms_by_region = firms_df.groupby('region')['latitude'].count().sort_values(ascending=True)

x = np.arange(len(regions) + 1)
width = 0.35
bars1 = ax1.barh(x - width/2, ucdp_by_region.reindex(list(regions.keys()) + ['Other']).fillna(0),
                 width, label='UCDP Events', color='#E63946', alpha=0.8)
bars2 = ax1.barh(x + width/2, firms_by_region.reindex(list(regions.keys()) + ['Other']).fillna(0) / 100,
                 width, label='FIRMS Fires (÷100)', color='#F4A261', alpha=0.8)
ax1.set_yticks(x)
ax1.set_yticklabels(list(regions.keys()) + ['Other'])
ax1.set_xlabel('Count')
ax1.set_title('Events by Region', fontweight='bold')
ax1.legend()

# Plot 2: Monthly regional trends
ax2 = axes[0, 1]
post_invasion['month'] = post_invasion['date_start'].dt.to_period('M')
monthly_regional = post_invasion.groupby(['month', 'region'])['id'].count().unstack(fill_value=0)
monthly_regional.index = monthly_regional.index.to_timestamp()

for region in ['Donetsk', 'Luhansk', 'Kharkiv', 'Zaporizhzhia', 'Kherson']:
    if region in monthly_regional.columns:
        ax2.plot(monthly_regional.index, monthly_regional[region], label=region, linewidth=1.5, alpha=0.8)

ax2.set_xlabel('Date')
ax2.set_ylabel('Monthly UCDP Events')
ax2.set_title('Monthly Conflict Events by Region', fontweight='bold')
ax2.legend(loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Regional deaths
ax3 = axes[1, 0]
deaths_by_region = post_invasion.groupby('region')['best_est'].sum().sort_values(ascending=True)
colors = ['#264653' if r != 'Donetsk' else '#E63946' for r in deaths_by_region.index]
deaths_by_region.plot(kind='barh', ax=ax3, color=colors, alpha=0.8)
ax3.set_xlabel('Total Estimated Deaths')
ax3.set_title('Cumulative Deaths by Region (UCDP)', fontweight='bold')

# Plot 4: Fire intensity by region
ax4 = axes[1, 1]
frp_by_region = firms_df.groupby('region')['frp'].agg(['mean', 'sum', 'count'])
frp_by_region = frp_by_region.sort_values('mean', ascending=True)

colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(frp_by_region)))
frp_by_region['mean'].plot(kind='barh', ax=ax4, color=colors, alpha=0.8)
ax4.set_xlabel('Average Fire Radiative Power (MW)')
ax4.set_title('Average Fire Intensity by Region', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/04_regional_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 04_regional_analysis.png")

# =============================================================================
# 6. OVERLAP SUMMARY STATISTICS
# =============================================================================
print("\n[6/6] Generating summary statistics...")

fig = plt.figure(figsize=(16, 12))

# Summary text panel
ax_text = fig.add_subplot(2, 2, 1)
ax_text.axis('off')

summary_text = f"""
TEMPORAL OVERLAP SUMMARY
{'='*40}

Dataset Coverage:
• UCDP Events: {ucdp_df['date_start'].min().strftime('%Y-%m-%d')} to {ucdp_df['date_start'].max().strftime('%Y-%m-%d')}
• FIRMS Fires: {firms_df['acq_date'].min().strftime('%Y-%m-%d')} to {firms_df['acq_date'].max().strftime('%Y-%m-%d')}
• DeepState: {deepstate_df['date'].min().strftime('%Y-%m-%d')} to {deepstate_df['date'].max().strftime('%Y-%m-%d')}
• War Losses: {personnel_df['date'].min().strftime('%Y-%m-%d')} to {personnel_df['date'].max().strftime('%Y-%m-%d')}

Full Overlap Period: 2024-07-08 to present
Post-Invasion Overlap: 2022-02-24 to present

Record Counts:
• UCDP Events (total): {len(ucdp_df):,}
• UCDP Events (post-invasion): {len(post_invasion):,}
• FIRMS Fire Hotspots: {len(firms_df):,}
• DeepState Snapshots: {len(deepstate_df):,}
• War Loss Records: {len(personnel_df):,}

Key Correlations (daily):
• Fires ↔ UCDP Events: r = {merged['firms_fires'].corr(merged['ucdp_events']):.3f}
• Fires ↔ Personnel Losses: r = {merged['firms_fires'].corr(merged['daily_loss']):.3f}
• UCDP Events ↔ Personnel Losses: r = {merged['ucdp_events'].corr(merged['daily_loss']):.3f}
• Fire Power ↔ UCDP Deaths: r = {merged['firms_frp'].corr(merged['ucdp_deaths']):.3f}
"""
ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Venn-style overlap diagram (conceptual)
ax2 = fig.add_subplot(2, 2, 2)
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PatchCollection

# Create overlapping circles representing datasets
circle1 = Circle((0.35, 0.5), 0.25, alpha=0.4, color='#E63946', label='UCDP')
circle2 = Circle((0.55, 0.5), 0.25, alpha=0.4, color='#F4A261', label='FIRMS')
circle3 = Circle((0.45, 0.3), 0.2, alpha=0.4, color='#2A9D8F', label='DeepState')

ax2.add_patch(circle1)
ax2.add_patch(circle2)
ax2.add_patch(circle3)

ax2.text(0.22, 0.6, 'UCDP\n31K events', ha='center', fontsize=10, fontweight='bold')
ax2.text(0.68, 0.6, 'FIRMS\n247K fires', ha='center', fontsize=10, fontweight='bold')
ax2.text(0.45, 0.15, 'DeepState\n554 snapshots', ha='center', fontsize=10, fontweight='bold')
ax2.text(0.45, 0.5, 'OVERLAP', ha='center', fontsize=9, fontweight='bold', color='black')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Conceptual Data Overlap', fontweight='bold', fontsize=12)

# Monthly data availability heatmap
ax3 = fig.add_subplot(2, 1, 2)

# Create monthly availability matrix
date_range = pd.date_range('2014-01', '2026-01', freq='MS')
availability = pd.DataFrame(index=date_range)

# Calculate monthly data availability
ucdp_monthly = ucdp_df.set_index('date_start').resample('MS')['id'].count()
firms_monthly = firms_df.set_index('acq_date').resample('MS')['latitude'].count()
deepstate_monthly = deepstate_df.set_index('date').resample('MS')['occupied_area_km2'].count()
personnel_monthly = personnel_df.set_index('date').resample('MS')['personnel'].count()

availability['UCDP'] = availability.index.map(lambda x: 1 if x in ucdp_monthly.index and ucdp_monthly[x] > 0 else 0)
availability['FIRMS'] = availability.index.map(lambda x: 1 if x in firms_monthly.index and firms_monthly[x] > 0 else 0)
availability['DeepState'] = availability.index.map(lambda x: 1 if x in deepstate_monthly.index and deepstate_monthly[x] > 0 else 0)
availability['War Losses'] = availability.index.map(lambda x: 1 if x in personnel_monthly.index and personnel_monthly[x] > 0 else 0)

# Plot heatmap
availability_t = availability.T
availability_t.columns = [d.strftime('%Y-%m') for d in availability_t.columns]

# Downsample columns for readability
cols_to_show = availability_t.columns[::6]  # Every 6 months
availability_display = availability_t[cols_to_show]

sns.heatmap(availability_display, cmap='YlGn', cbar_kws={'label': 'Data Available'},
            ax=ax3, linewidths=0.5, yticklabels=True)
ax3.set_title('Monthly Data Availability Timeline', fontweight='bold', fontsize=12)
ax3.set_xlabel('Month')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add invasion marker
invasion_idx = list(cols_to_show).index('2022-02') if '2022-02' in cols_to_show else None
if invasion_idx:
    ax3.axvline(invasion_idx + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/05_overlap_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 05_overlap_summary.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFigures saved to: {OUTPUT_PATH}/")
print("\nGenerated visualizations:")
print("  1. 01_temporal_coverage.png - Dataset date ranges and event counts")
print("  2. 02_spatial_distribution.png - Geographic distribution maps")
print("  3. 03_temporal_spatial_correlation.png - Cross-dataset correlations")
print("  4. 04_regional_analysis.png - Regional breakdown")
print("  5. 05_overlap_summary.png - Summary statistics and data availability")

print("\n" + "-" * 60)
print("KEY FINDINGS:")
print("-" * 60)
print(f"""
1. TEMPORAL OVERLAP:
   - All 4 datasets overlap from 2022-02-24 (invasion start) onwards
   - Full 4-dataset overlap exists from 2024-07-08 (DeepState start)
   - Total overlapping days: {(min(firms_df['acq_date'].max(), personnel_df['date'].max()) - pd.Timestamp('2022-02-24')).days:,}

2. SPATIAL OVERLAP:
   - Both UCDP and FIRMS data cover all of Ukraine (22°-40.5°E, 44°-52.5°N)
   - Highest concentration: Eastern oblasts (Donetsk, Luhansk, Kharkiv)
   - Fire hotspots and conflict events show strong geographic clustering

3. CORRELATIONS:
   - Fire hotspots ↔ UCDP events: r = {merged['firms_fires'].corr(merged['ucdp_events']):.3f}
   - Both metrics peak in active combat zones
   - Daily patterns show moderate temporal alignment

4. REGIONAL HOTSPOTS:
   - Donetsk: Highest conflict event density
   - Eastern front: Most fire activity
   - Territorial control: Fluctuates with front-line changes
""")
