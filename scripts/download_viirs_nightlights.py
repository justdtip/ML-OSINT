#!/usr/bin/env python3
"""
Download VIIRS Black Marble (VNP46A3) monthly nighttime lights data for Ukraine
and extract brightness statistics without storing full imagery.

Uses NASA Earthdata API with token authentication.
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import VIIRS_DIR, EARTHDATA_TOKEN_FILE, ensure_dir

# Configuration
DATA_DIR = ensure_dir(VIIRS_DIR)

# Ukraine bounding box
UKRAINE_BBOX = {
    'min_lon': 22.0,
    'max_lon': 40.5,
    'min_lat': 44.0,
    'max_lat': 52.5
}

# VIIRS sinusoidal tiles covering Ukraine
# Tile naming: hXXvYY where XX is horizontal, YY is vertical
UKRAINE_TILES = [
    'h19v03', 'h19v04',  # Western Ukraine
    'h20v03', 'h20v04',  # Central Ukraine
    'h21v03', 'h21v04',  # Eastern Ukraine (Donbas)
]

# Product info
PRODUCT = 'VNP46A3'  # Monthly composite
COLLECTION = '5000'   # Collection version

# Base URLs
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
LAADS_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"


def get_token():
    """Get the Earthdata token from environment or prompt."""
    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        if EARTHDATA_TOKEN_FILE.exists():
            token = EARTHDATA_TOKEN_FILE.read_text().strip()
    return token


def search_granules(product, start_date, end_date, bbox, token):
    """
    Search for granules using NASA CMR API.
    """
    params = {
        'short_name': product,
        'temporal': f'{start_date},{end_date}',
        'bounding_box': f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        'page_size': 2000,
        'sort_key': 'start_date'
    }

    headers = {
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(CMR_URL, params=params, headers=headers)

    if response.status_code != 200:
        print(f"Error searching CMR: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    granules = data.get('feed', {}).get('entry', [])

    return granules


def get_download_url(granule):
    """Extract download URL from granule metadata."""
    links = granule.get('links', [])
    for link in links:
        href = link.get('href', '')
        if href.endswith('.h5') or href.endswith('.hdf'):
            return href
    return None


def download_file(url, output_path, token):
    """Download a file with authentication."""
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # Handle redirects for Earthdata
    session = requests.Session()

    response = session.get(url, headers=headers, stream=True, allow_redirects=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        print(f"  Error downloading: {response.status_code}")
        return False


def extract_brightness_stats(h5_file):
    """
    Extract brightness statistics from HDF5 file.
    Returns dict with mean, sum, std, etc.
    """
    try:
        import h5py
        import numpy as np

        stats = {}

        with h5py.File(h5_file, 'r') as f:
            # List available datasets
            def list_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    stats['_datasets'] = stats.get('_datasets', []) + [name]

            f.visititems(list_datasets)

            # Try to find the main radiance layer
            radiance_paths = [
                'HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/NearNadir_Composite_Snow_Free',
                'HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/AllAngle_Composite_Snow_Free',
            ]

            for path in radiance_paths:
                try:
                    data = f[path][:]
                    # Mask fill values (typically 65535 or similar)
                    fill_value = f[path].attrs.get('_FillValue', 65535)
                    if hasattr(fill_value, '__iter__'):
                        fill_value = fill_value[0]

                    masked = np.ma.masked_equal(data, fill_value)

                    # Also mask zeros and negatives
                    masked = np.ma.masked_less_equal(masked, 0)

                    if masked.count() > 0:
                        layer_name = path.split('/')[-1]
                        stats[f'{layer_name}_mean'] = float(masked.mean())
                        stats[f'{layer_name}_std'] = float(masked.std())
                        stats[f'{layer_name}_min'] = float(masked.min())
                        stats[f'{layer_name}_max'] = float(masked.max())
                        stats[f'{layer_name}_sum'] = float(masked.sum())
                        stats[f'{layer_name}_count'] = int(masked.count())
                        stats[f'{layer_name}_pct_valid'] = float(masked.count() / data.size * 100)
                except KeyError:
                    continue

        return stats

    except ImportError:
        print("h5py not installed. Install with: pip install h5py")
        return None
    except Exception as e:
        print(f"Error extracting stats: {e}")
        return None


def build_direct_urls(year, month, tiles):
    """
    Build direct download URLs for VNP46A3 monthly files.
    URL pattern: https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A3/{year}/{doy}/
    """
    urls = []

    # VNP46A3 monthly files are stored at day 001 of each month's folder
    # Actually they're organized by the month's representative DOY
    # For monthly composites, we need to find the actual file

    # The files follow pattern: VNP46A3.A{year}{doy}.{tile}.{collection}.{timestamp}.h5
    # For monthly products, DOY is typically the first day of the month

    from datetime import date

    first_day = date(year, month, 1)
    doy = first_day.timetuple().tm_yday

    base_url = f"{LAADS_URL}/5000/VNP46A3/{year}/{doy:03d}/"

    for tile in tiles:
        urls.append({
            'year': year,
            'month': month,
            'tile': tile,
            'doy': doy,
            'base_url': base_url
        })

    return urls


def list_files_in_directory(url, token):
    """List HDF5 files in a LAADS directory."""
    headers = {'Authorization': f'Bearer {token}'}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []

    # Parse HTML to find .h5 files
    import re
    files = re.findall(r'href="([^"]+\.h5)"', response.text)

    return files


def main():
    print("=" * 70)
    print("VIIRS BLACK MARBLE (VNP46A3) DOWNLOAD FOR UKRAINE")
    print("=" * 70)

    # Get token
    token = get_token()
    if not token:
        print("\nNo Earthdata token found!")
        print("Please either:")
        print("  1. Set EARTHDATA_TOKEN environment variable")
        print("  2. Create .earthdata_token file in project root")
        print("\nProvide token as argument:")

        if len(sys.argv) > 1:
            token = sys.argv[1]
        else:
            return

    print(f"\nConfiguration:")
    print(f"  Product: {PRODUCT} (Monthly Black Marble)")
    print(f"  Tiles: {', '.join(UKRAINE_TILES)}")
    print(f"  Output: {DATA_DIR}")

    # Date range
    start_year, start_month = 2022, 2  # Feb 2022
    end_year, end_month = 2024, 12     # Dec 2024

    print(f"  Date range: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

    # Create output CSV for statistics
    stats_file = DATA_DIR / "viirs_brightness_stats.csv"
    stats_data = []

    # Iterate through months
    current_year, current_month = start_year, start_month

    while (current_year, current_month) <= (end_year, end_month):
        print(f"\n--- {current_year}-{current_month:02d} ---")

        # Build URLs for this month
        from datetime import date
        first_day = date(current_year, current_month, 1)
        doy = first_day.timetuple().tm_yday

        base_url = f"{LAADS_URL}/5000/VNP46A3/{current_year}/{doy:03d}/"

        print(f"  Checking: {base_url}")

        # List available files
        files = list_files_in_directory(base_url, token)

        if not files:
            print(f"  No files found for {current_year}-{current_month:02d}")
        else:
            # Filter for Ukraine tiles
            for tile in UKRAINE_TILES:
                matching = [f for f in files if tile in f]
                if matching:
                    filename = matching[0]
                    file_url = base_url + filename
                    local_path = DATA_DIR / filename

                    print(f"  Downloading {tile}: {filename}")

                    if local_path.exists():
                        print(f"    Already exists, extracting stats...")
                    else:
                        success = download_file(file_url, local_path, token)
                        if not success:
                            print(f"    Failed to download")
                            continue

                    # Extract statistics
                    stats = extract_brightness_stats(local_path)

                    if stats:
                        stats['year'] = current_year
                        stats['month'] = current_month
                        stats['tile'] = tile
                        stats['filename'] = filename
                        stats_data.append(stats)
                        print(f"    Mean brightness: {stats.get('NearNadir_Composite_Snow_Free_mean', 'N/A')}")

                        # Optionally delete the HDF5 file to save space
                        # local_path.unlink()
                    else:
                        print(f"    Failed to extract stats")

                else:
                    print(f"  No file for tile {tile}")

        # Next month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1

        # Rate limiting
        time.sleep(0.5)

    # Save statistics to CSV
    if stats_data:
        import csv

        # Get all unique keys
        all_keys = set()
        for s in stats_data:
            all_keys.update(s.keys())
        all_keys.discard('_datasets')
        all_keys = sorted(all_keys)

        with open(stats_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in stats_data:
                row_clean = {k: v for k, v in row.items() if k != '_datasets'}
                writer.writerow(row_clean)

        print(f"\n\nStatistics saved to: {stats_file}")
        print(f"Total records: {len(stats_data)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
