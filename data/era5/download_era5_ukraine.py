#!/usr/bin/env python3
"""
Download ERA5-Land timeseries data for Ukraine's landmass.

Ukraine bounding box (approximate):
- Latitude: 44.3°N to 52.4°N
- Longitude: 22.1°E to 40.2°E

Using 1-degree grid spacing gives ~80 points covering Ukraine.
The API returns point-based timeseries data.

Setup:
1. Register at https://cds.climate.copernicus.eu/
2. Get your API key from https://cds.climate.copernicus.eu/how-to-api
3. Create ~/.cdsapirc with:
   url: https://cds.climate.copernicus.eu/api
   key: <your-uid>:<your-api-key>
"""

import cdsapi
import os
import json
import time
from pathlib import Path
from datetime import datetime
from itertools import product

# Ukraine bounding box
UKRAINE_BOUNDS = {
    'lat_min': 44.3,
    'lat_max': 52.4,
    'lon_min': 22.1,
    'lon_max': 40.2
}

# Grid spacing in degrees (1 degree ~ 111km at equator, ~70km at Ukraine's latitude)
GRID_SPACING = 1.0

# Date range matching our analysis period
DATE_RANGE = "2022-02-01/2024-01-01"

# Variables relevant for conflict/fire analysis
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "snow_cover",
    "skin_temperature",
    "surface_solar_radiation_downwards",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# Output directory
OUTPUT_DIR = Path(__file__).parent


def generate_ukraine_grid():
    """Generate a grid of points covering Ukraine."""
    lats = []
    lat = UKRAINE_BOUNDS['lat_min']
    while lat <= UKRAINE_BOUNDS['lat_max']:
        lats.append(round(lat, 1))
        lat += GRID_SPACING

    lons = []
    lon = UKRAINE_BOUNDS['lon_min']
    while lon <= UKRAINE_BOUNDS['lon_max']:
        lons.append(round(lon, 1))
        lon += GRID_SPACING

    # Generate all grid points
    grid_points = list(product(lats, lons))
    return grid_points


def download_point_timeseries(client, lat, lon, output_file):
    """Download ERA5-Land timeseries for a single point."""
    dataset = "reanalysis-era5-land-timeseries"

    request = {
        "variable": VARIABLES,
        "location": {"latitude": lat, "longitude": lon},
        "date": [DATE_RANGE],
        "data_format": "csv"
    }

    print(f"  Requesting data for ({lat}, {lon})...")

    try:
        client.retrieve(dataset, request).download(output_file)
        return True
    except Exception as e:
        print(f"  Error downloading ({lat}, {lon}): {e}")
        return False


def download_all_points():
    """Download ERA5 data for all grid points covering Ukraine."""

    # Check for credentials
    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        print("ERROR: No ~/.cdsapirc file found!")
        print("\nTo set up CDS API access:")
        print("1. Register at https://cds.climate.copernicus.eu/")
        print("2. Get your API key from https://cds.climate.copernicus.eu/how-to-api")
        print("3. Create ~/.cdsapirc with:")
        print("   url: https://cds.climate.copernicus.eu/api")
        print("   key: <your-uid>:<your-api-key>")
        return

    # Generate grid
    grid_points = generate_ukraine_grid()
    print(f"Generated {len(grid_points)} grid points covering Ukraine")
    print(f"Latitude range: {UKRAINE_BOUNDS['lat_min']}° to {UKRAINE_BOUNDS['lat_max']}°")
    print(f"Longitude range: {UKRAINE_BOUNDS['lon_min']}° to {UKRAINE_BOUNDS['lon_max']}°")
    print(f"Grid spacing: {GRID_SPACING}°")
    print(f"Date range: {DATE_RANGE}")
    print(f"Variables: {len(VARIABLES)}")
    print()

    # Create output directory
    points_dir = OUTPUT_DIR / "points"
    points_dir.mkdir(exist_ok=True)

    # Initialize client
    client = cdsapi.Client()

    # Track progress
    successful = []
    failed = []

    # Download each point
    for i, (lat, lon) in enumerate(grid_points):
        output_file = points_dir / f"era5_lat{lat}_lon{lon}.csv"

        # Skip if already downloaded
        if output_file.exists():
            print(f"[{i+1}/{len(grid_points)}] Skipping ({lat}, {lon}) - already exists")
            successful.append((lat, lon))
            continue

        print(f"[{i+1}/{len(grid_points)}] Downloading ({lat}, {lon})...")

        if download_point_timeseries(client, lat, lon, str(output_file)):
            successful.append((lat, lon))
            print(f"  Saved to {output_file}")
        else:
            failed.append((lat, lon))

        # Rate limiting - be nice to the API
        if i < len(grid_points) - 1:
            time.sleep(1)

    # Save manifest
    manifest = {
        "download_time": datetime.now().isoformat(),
        "grid_spacing_degrees": GRID_SPACING,
        "bounds": UKRAINE_BOUNDS,
        "date_range": DATE_RANGE,
        "variables": VARIABLES,
        "total_points": len(grid_points),
        "successful": len(successful),
        "failed": len(failed),
        "successful_points": successful,
        "failed_points": failed
    }

    manifest_file = OUTPUT_DIR / "download_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"Successful: {len(successful)}/{len(grid_points)}")
    print(f"Failed: {len(failed)}/{len(grid_points)}")
    print(f"Manifest saved to: {manifest_file}")

    if failed:
        print(f"\nFailed points:")
        for lat, lon in failed:
            print(f"  ({lat}, {lon})")


def aggregate_to_daily():
    """
    Aggregate downloaded point data into a single daily timeseries.
    Call this after download_all_points() completes.
    """
    import pandas as pd
    import numpy as np

    points_dir = OUTPUT_DIR / "points"
    if not points_dir.exists():
        print("No points directory found. Run download first.")
        return

    csv_files = list(points_dir.glob("era5_*.csv"))
    if not csv_files:
        print("No CSV files found in points directory.")
        return

    print(f"Aggregating {len(csv_files)} point files...")

    all_data = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Extract lat/lon from filename
            name = f.stem
            parts = name.replace("era5_lat", "").replace("_lon", ",").split(",")
            lat, lon = float(parts[0]), float(parts[1])
            df['lat'] = lat
            df['lon'] = lon
            all_data.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not all_data:
        print("No valid data to aggregate.")
        return

    # Combine all points
    combined = pd.concat(all_data, ignore_index=True)

    # Parse datetime
    if 'valid_time' in combined.columns:
        combined['date'] = pd.to_datetime(combined['valid_time']).dt.date
    elif 'time' in combined.columns:
        combined['date'] = pd.to_datetime(combined['time']).dt.date

    # Aggregate by date (mean across all Ukraine points)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['lat', 'lon']]

    daily_agg = combined.groupby('date')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
    daily_agg = daily_agg.reset_index()

    # Save aggregated data
    output_file = OUTPUT_DIR / "era5_ukraine_daily.csv"
    daily_agg.to_csv(output_file, index=False)
    print(f"Saved aggregated daily data to: {output_file}")
    print(f"Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
    print(f"Columns: {len(daily_agg.columns)}")

    return daily_agg


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        aggregate_to_daily()
    else:
        print("ERA5-Land Ukraine Download Script")
        print("="*50)
        download_all_points()
        print("\nTo aggregate after download completes:")
        print("  python download_era5_ukraine.py aggregate")
