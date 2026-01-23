#!/usr/bin/env python3
"""
Download VIIRS Black Marble (VNP46A1) DAILY nighttime lights data for Ukraine,
extract brightness statistics, and DELETE the raw HDF5 files to save space.

Only keeps the extracted statistics CSV.
Uses curl for downloads (more reliable with NASA's auth).
"""

import os
import sys
import csv
import subprocess
import requests
from pathlib import Path
from datetime import datetime, date, timedelta
import time
import re
import tempfile

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import VIIRS_DIR, EARTHDATA_TOKEN_FILE, ensure_dir

# Configuration
DATA_DIR = ensure_dir(VIIRS_DIR)

# VIIRS sinusoidal tiles covering Ukraine
UKRAINE_TILES = [
    'h19v03', 'h19v04',  # Western Ukraine
    'h20v03', 'h20v04',  # Central Ukraine
    'h21v03', 'h21v04',  # Eastern Ukraine (Donbas)
]

# Product info - VNP46A1 is the daily product
PRODUCT = 'VNP46A1'
COLLECTION = '5200'

LAADS_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"


def get_token():
    """Get the Earthdata token."""
    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        if EARTHDATA_TOKEN_FILE.exists():
            token = EARTHDATA_TOKEN_FILE.read_text().strip()
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
    return token


def list_files_in_directory(url, token):
    """List HDF5 files in a LAADS directory."""
    headers = {'Authorization': f'Bearer {token}'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return []

        # Parse HTML to find .h5 files (full URLs)
        files = re.findall(r'href="(https://[^"]+\.h5)"', response.text)
        return files
    except Exception as e:
        print(f"    Error listing directory: {e}")
        return []


def download_file_curl(url, output_path, token):
    """Download a file using curl (more reliable with NASA auth)."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use curl with -L to follow redirects
        cmd = [
            'curl', '-s', '-L',
            '-H', f'Authorization: Bearer {token}',
            '-o', str(output_path),
            url
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=300)

        # Check if file was downloaded and has content
        if output_path.exists() and output_path.stat().st_size > 1000:
            return True
        else:
            if output_path.exists():
                output_path.unlink()
            return False

    except subprocess.TimeoutExpired:
        print(f"    Timeout downloading")
        return False
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def extract_brightness_stats(h5_file):
    """
    Extract brightness statistics from VNP46A1 HDF5 file.
    Returns dict with mean, sum, std, count for radiance data.
    """
    try:
        import h5py
        import numpy as np

        stats = {}

        with h5py.File(h5_file, 'r') as f:
            # VNP46A1 data paths (actual structure)
            radiance_path = 'HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/DNB_At_Sensor_Radiance'

            try:
                data = f[radiance_path][:]

                # Mask invalid values
                masked = np.ma.masked_less_equal(data, 0)
                masked = np.ma.masked_greater(masked, 1e10)

                if masked.count() > 0:
                    stats['radiance_mean'] = float(masked.mean())
                    stats['radiance_std'] = float(masked.std())
                    stats['radiance_min'] = float(masked.min())
                    stats['radiance_max'] = float(masked.max())
                    stats['radiance_sum'] = float(masked.sum())
                    stats['radiance_count'] = int(masked.count())
                    stats['radiance_total_pixels'] = int(data.size)
                    stats['radiance_pct_valid'] = float(masked.count() / data.size * 100)

                    # Percentiles
                    valid_data = masked.compressed()
                    if len(valid_data) > 100:
                        stats['radiance_p10'] = float(np.percentile(valid_data, 10))
                        stats['radiance_p25'] = float(np.percentile(valid_data, 25))
                        stats['radiance_p50'] = float(np.percentile(valid_data, 50))
                        stats['radiance_p75'] = float(np.percentile(valid_data, 75))
                        stats['radiance_p90'] = float(np.percentile(valid_data, 90))

            except KeyError:
                pass

            # Moon illumination
            try:
                moon_path = 'HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Moon_Illumination_Fraction'
                moon = f[moon_path][:]
                moon_masked = np.ma.masked_less(moon, 0)
                if moon_masked.count() > 0:
                    stats['moon_illumination_pct'] = float(moon_masked.mean() / 100)
            except KeyError:
                pass

            # Lunar zenith (affects visibility)
            try:
                lunar_zen_path = 'HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Lunar_Zenith'
                lunar = f[lunar_zen_path][:]
                lunar_masked = np.ma.masked_less(lunar, -900)
                if lunar_masked.count() > 0:
                    stats['lunar_zenith_mean'] = float(lunar_masked.mean() / 100)  # Scale factor
            except KeyError:
                pass

            # Cloud mask quality
            try:
                qf_path = 'HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/QF_Cloud_Mask'
                qf = f[qf_path][:]
                # Count clear pixels (typically bit 0-1 indicate cloud status)
                clear_mask = (qf & 0x03) == 0  # Clear sky
                stats['pct_clear_sky'] = float(clear_mask.sum() / qf.size * 100)
            except KeyError:
                pass

        return stats if stats else None

    except ImportError:
        print("h5py not installed. Run: pip install h5py")
        return None
    except Exception as e:
        print(f"      Error extracting stats: {e}")
        return None


def process_day(year, doy, token, stats_writer, temp_dir):
    """Process all Ukraine tiles for a single day."""
    base_url = f"{LAADS_URL}/{COLLECTION}/{PRODUCT}/{year}/{doy:03d}/"

    # List files for this day
    files = list_files_in_directory(base_url, token)

    if not files:
        return 0

    processed = 0

    for tile in UKRAINE_TILES:
        # Find file for this tile
        matching = [f for f in files if tile in f]

        if not matching:
            continue

        file_url = matching[0]
        filename = file_url.split('/')[-1]

        # Download to temp file
        temp_path = temp_dir / filename

        success = download_file_curl(file_url, temp_path, token)

        if not success:
            continue

        # Extract statistics
        stats = extract_brightness_stats(temp_path)

        # Delete the file immediately
        try:
            temp_path.unlink()
        except:
            pass

        if stats:
            # Convert DOY to date
            d = date(year, 1, 1) + timedelta(days=doy - 1)

            row = {
                'date': d.isoformat(),
                'year': year,
                'month': d.month,
                'day': d.day,
                'doy': doy,
                'tile': tile,
                'filename': filename,
                **stats
            }

            stats_writer.writerow(row)
            processed += 1

    return processed


def main():
    print("=" * 70)
    print("VIIRS VNP46A1 DAILY BRIGHTNESS STATISTICS EXTRACTOR")
    print("=" * 70)
    print("\nThis downloads daily VIIRS nighttime imagery, extracts statistics,")
    print("and DELETES the raw files - only keeping the statistics CSV.\n")

    # Get token
    token = get_token()
    if not token:
        print("ERROR: No Earthdata token found!")
        print("\nUsage: python download_viirs_daily_stats.py <TOKEN>")
        print("\nOr set EARTHDATA_TOKEN environment variable")
        return

    # Configuration
    start_date = date(2022, 2, 24)  # Invasion start
    end_date = date(2024, 12, 31)

    print(f"Configuration:")
    print(f"  Product: {PRODUCT} (Daily)")
    print(f"  Tiles: {', '.join(UKRAINE_TILES)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Output: {DATA_DIR}")

    # Calculate total days
    total_days = (end_date - start_date).days + 1
    print(f"  Total days to process: {total_days}")
    print(f"  Expected files: ~{total_days * len(UKRAINE_TILES)} (if all available)")

    # Output CSV
    stats_file = DATA_DIR / "viirs_daily_brightness_stats.csv"

    # Field names
    fieldnames = [
        'date', 'year', 'month', 'day', 'doy', 'tile', 'filename',
        'radiance_mean', 'radiance_std', 'radiance_min', 'radiance_max',
        'radiance_sum', 'radiance_count', 'radiance_total_pixels', 'radiance_pct_valid',
        'radiance_p10', 'radiance_p25', 'radiance_p50', 'radiance_p75', 'radiance_p90',
        'moon_illumination_pct', 'lunar_zenith_mean', 'pct_clear_sky'
    ]

    # Use temp directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Check if we're resuming
        existing_dates = set()
        if stats_file.exists():
            print(f"\n  Found existing stats file, will resume...")
            with open(stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_dates.add((row['date'], row['tile']))
            print(f"  Already processed: {len(existing_dates)} tile-days")

        # Open CSV for appending
        mode = 'a' if stats_file.exists() else 'w'
        with open(stats_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

            if mode == 'w':
                writer.writeheader()

            # Process each day
            current_date = start_date
            processed_total = 0
            errors = 0
            days_done = 0

            while current_date <= end_date:
                year = current_date.year
                doy = current_date.timetuple().tm_yday

                # Skip if already processed
                skip_day = all((current_date.isoformat(), tile) in existing_dates
                              for tile in UKRAINE_TILES)
                if skip_day:
                    current_date += timedelta(days=1)
                    days_done += 1
                    continue

                print(f"\r  [{days_done}/{total_days}] {current_date} (DOY {doy})...", end='', flush=True)

                try:
                    count = process_day(year, doy, token, writer, temp_path)
                    processed_total += count

                    if count > 0:
                        print(f" [{count} tiles]", end='')
                        f.flush()  # Flush after each day to save progress

                except Exception as e:
                    print(f" ERROR: {e}")
                    errors += 1

                current_date += timedelta(days=1)
                days_done += 1

                # Rate limiting (be nice to NASA servers)
                time.sleep(0.5)

                # Progress update every 30 days
                if days_done % 30 == 0:
                    print(f"\n    Progress: {processed_total} tile-days processed, {errors} errors")

    print(f"\n\n{'=' * 70}")
    print(f"COMPLETE!")
    print(f"{'=' * 70}")
    print(f"  Statistics saved to: {stats_file}")
    print(f"  Total tile-days processed: {processed_total}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
