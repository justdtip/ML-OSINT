#!/usr/bin/env python3
"""
Download VIIRS Black Marble (VNP46A2) gap-filled nighttime lights data for Ukraine,
extract brightness statistics, and DELETE the raw HDF5 files to save space.

VNP46A2 is the gap-filled, BRDF-corrected daily product - better for time series
analysis as it fills cloud gaps with recent clear-sky observations.

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

# Product info - VNP46A2 is the gap-filled daily product
PRODUCT = 'VNP46A2'
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
    Extract brightness statistics from VNP46A2 HDF5 file.
    Returns dict with mean, sum, std, count for NTL data.

    VNP46A2 contains:
    - DNB_BRDF-Corrected_NTL: BRDF-corrected nighttime lights
    - Gap_Filled_DNB_BRDF-Corrected_NTL: Gap-filled (cloud-free) version
    - Mandatory_Quality_Flag: Quality indicator
    - Snow_Flag: Snow detection
    - Latest_High_Quality_Retrieval: Days since last good observation
    """
    try:
        import h5py
        import numpy as np

        stats = {}

        with h5py.File(h5_file, 'r') as f:
            base_path = 'HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields'

            # Gap-filled NTL (primary - most useful for time series)
            try:
                ntl_path = f'{base_path}/Gap_Filled_DNB_BRDF-Corrected_NTL'
                data = f[ntl_path][:]

                # VNP46A2 uses fill value 65535 for invalid data
                # Valid range is typically 0-6553.4 nW/cm²/sr
                masked = np.ma.masked_equal(data, 65535.0)
                masked = np.ma.masked_less(masked, 0)
                masked = np.ma.masked_greater(masked, 6600)  # Allow slightly above max

                if masked.count() > 0:
                    stats['gap_filled_ntl_mean'] = float(masked.mean())
                    stats['gap_filled_ntl_std'] = float(masked.std())
                    stats['gap_filled_ntl_min'] = float(masked.min())
                    stats['gap_filled_ntl_max'] = float(masked.max())
                    stats['gap_filled_ntl_sum'] = float(masked.sum())
                    stats['gap_filled_ntl_count'] = int(masked.count())
                    stats['gap_filled_ntl_total_pixels'] = int(data.size)
                    stats['gap_filled_ntl_pct_valid'] = float(masked.count() / data.size * 100)

                    # Percentiles
                    valid_data = masked.compressed()
                    if len(valid_data) > 100:
                        stats['gap_filled_ntl_p10'] = float(np.percentile(valid_data, 10))
                        stats['gap_filled_ntl_p25'] = float(np.percentile(valid_data, 25))
                        stats['gap_filled_ntl_p50'] = float(np.percentile(valid_data, 50))
                        stats['gap_filled_ntl_p75'] = float(np.percentile(valid_data, 75))
                        stats['gap_filled_ntl_p90'] = float(np.percentile(valid_data, 90))
                        stats['gap_filled_ntl_p99'] = float(np.percentile(valid_data, 99))

            except KeyError:
                pass

            # Original BRDF-corrected NTL (before gap filling)
            try:
                brdf_path = f'{base_path}/DNB_BRDF-Corrected_NTL'
                brdf_data = f[brdf_path][:]

                brdf_masked = np.ma.masked_equal(brdf_data, 65535.0)
                brdf_masked = np.ma.masked_less(brdf_masked, 0)
                brdf_masked = np.ma.masked_greater(brdf_masked, 6600)

                if brdf_masked.count() > 0:
                    stats['brdf_ntl_mean'] = float(brdf_masked.mean())
                    stats['brdf_ntl_std'] = float(brdf_masked.std())
                    stats['brdf_ntl_count'] = int(brdf_masked.count())
                    stats['brdf_ntl_pct_valid'] = float(brdf_masked.count() / brdf_data.size * 100)

                    # Percentiles for BRDF
                    brdf_valid = brdf_masked.compressed()
                    if len(brdf_valid) > 100:
                        stats['brdf_ntl_p50'] = float(np.percentile(brdf_valid, 50))
                        stats['brdf_ntl_p90'] = float(np.percentile(brdf_valid, 90))

            except KeyError:
                pass

            # Quality flag analysis
            try:
                qf_path = f'{base_path}/Mandatory_Quality_Flag'
                qf = f[qf_path][:]

                # Quality flag values:
                # 0 = High quality
                # 1 = Good quality
                # 2 = Poor quality (likely cloud)
                # 255 = Fill value
                total_valid = np.sum(qf != 255)
                if total_valid > 0:
                    stats['qf_high_quality_pct'] = float(np.sum(qf == 0) / total_valid * 100)
                    stats['qf_good_quality_pct'] = float(np.sum(qf == 1) / total_valid * 100)
                    stats['qf_poor_quality_pct'] = float(np.sum(qf == 2) / total_valid * 100)

            except KeyError:
                pass

            # Latest high quality retrieval (days since last clear observation)
            try:
                lhq_path = f'{base_path}/Latest_High_Quality_Retrieval'
                lhq = f[lhq_path][:]

                # Value = days since last high quality retrieval
                # 255 = fill value
                lhq_masked = np.ma.masked_equal(lhq, 255)
                if lhq_masked.count() > 0:
                    stats['days_since_clear_mean'] = float(lhq_masked.mean())
                    stats['days_since_clear_max'] = float(lhq_masked.max())
                    stats['pct_same_day_obs'] = float(np.sum(lhq == 0) / lhq_masked.count() * 100)

            except KeyError:
                pass

            # Snow flag
            try:
                snow_path = f'{base_path}/Snow_Flag'
                snow = f[snow_path][:]

                # 0 = no snow, 1 = snow, 255 = fill
                snow_valid = snow[snow != 255]
                if len(snow_valid) > 0:
                    stats['snow_cover_pct'] = float(np.sum(snow_valid == 1) / len(snow_valid) * 100)

            except KeyError:
                pass

            # Cloud mask
            try:
                cloud_path = f'{base_path}/QF_Cloud_Mask'
                cloud = f[cloud_path][:]

                # Bit 0-1: Cloud detection
                # 00 = confident clear, 01 = probably clear, 10 = probably cloudy, 11 = confident cloudy
                cloud_bits = cloud & 0x03
                total = cloud.size
                stats['pct_confident_clear'] = float(np.sum(cloud_bits == 0) / total * 100)
                stats['pct_probably_clear'] = float(np.sum(cloud_bits == 1) / total * 100)
                stats['pct_probably_cloudy'] = float(np.sum(cloud_bits == 2) / total * 100)
                stats['pct_confident_cloudy'] = float(np.sum(cloud_bits == 3) / total * 100)

            except KeyError:
                pass

            # Lunar irradiance (for context)
            try:
                lunar_path = f'{base_path}/DNB_Lunar_Irradiance'
                lunar = f[lunar_path][:]

                lunar_masked = np.ma.masked_equal(lunar, 65535)
                if lunar_masked.count() > 0:
                    # Scale factor is 0.0001 for lunar irradiance
                    stats['lunar_irradiance_mean'] = float(lunar_masked.mean() * 0.0001)

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
    print("VIIRS VNP46A2 GAP-FILLED NTL STATISTICS EXTRACTOR")
    print("=" * 70)
    print("\nThis downloads gap-filled VIIRS nighttime lights (VNP46A2),")
    print("extracts statistics, and DELETES the raw files.\n")
    print("VNP46A2 provides BRDF-corrected, gap-filled NTL data - ideal for")
    print("time series analysis as cloud gaps are filled with recent clear obs.\n")

    # Get token
    token = get_token()
    if not token:
        print("ERROR: No Earthdata token found!")
        print("\nUsage: python download_viirs_a2_stats.py <TOKEN>")
        print("\nOr set EARTHDATA_TOKEN environment variable")
        return

    # Configuration - same date range as VNP46A1
    start_date = date(2022, 2, 24)  # Invasion start
    end_date = date(2024, 12, 31)

    print(f"Configuration:")
    print(f"  Product: {PRODUCT} (Gap-filled Daily)")
    print(f"  Tiles: {', '.join(UKRAINE_TILES)}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Output: {DATA_DIR}")

    # Calculate total days
    total_days = (end_date - start_date).days + 1
    print(f"  Total days to process: {total_days}")
    print(f"  Expected files: ~{total_days * len(UKRAINE_TILES)} (if all available)")

    # Output CSV
    stats_file = DATA_DIR / "viirs_a2_gap_filled_stats.csv"

    # Field names
    fieldnames = [
        'date', 'year', 'month', 'day', 'doy', 'tile', 'filename',
        # Gap-filled NTL stats
        'gap_filled_ntl_mean', 'gap_filled_ntl_std', 'gap_filled_ntl_min', 'gap_filled_ntl_max',
        'gap_filled_ntl_sum', 'gap_filled_ntl_count', 'gap_filled_ntl_total_pixels', 'gap_filled_ntl_pct_valid',
        'gap_filled_ntl_p10', 'gap_filled_ntl_p25', 'gap_filled_ntl_p50', 'gap_filled_ntl_p75', 'gap_filled_ntl_p90', 'gap_filled_ntl_p99',
        # BRDF NTL stats
        'brdf_ntl_mean', 'brdf_ntl_std', 'brdf_ntl_count', 'brdf_ntl_pct_valid', 'brdf_ntl_p50', 'brdf_ntl_p90',
        # Quality flags
        'qf_high_quality_pct', 'qf_good_quality_pct', 'qf_poor_quality_pct',
        # Temporal info
        'days_since_clear_mean', 'days_since_clear_max', 'pct_same_day_obs',
        # Environmental
        'snow_cover_pct',
        # Cloud mask
        'pct_confident_clear', 'pct_probably_clear', 'pct_probably_cloudy', 'pct_confident_cloudy',
        # Lunar
        'lunar_irradiance_mean'
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
