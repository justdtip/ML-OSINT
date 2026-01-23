#!/usr/bin/env python3
"""
Fetch ALL historical DeepState snapshots from the Internet Archive Wayback Machine.
"""

import requests
import json
import os
from datetime import datetime
import time

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/deepstate/wayback_snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Backoff configuration
BASE_DELAY = 3          # Base delay between requests (seconds)
MAX_DELAY = 300         # Maximum backoff delay (5 minutes)
BACKOFF_MULTIPLIER = 2  # Multiply delay by this on consecutive failures
FAILURE_THRESHOLD = 3   # Number of consecutive failures before backing off

def get_all_snapshots():
    """Get list of all available Wayback Machine snapshots"""
    print("Fetching list of all available snapshots from Wayback Machine...")

    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "deepstatemap.live/api/history/last",
        "output": "json",
        "fl": "timestamp,statuscode,digest",
        "filter": "statuscode:200"
    }

    r = requests.get(cdx_url, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    if len(data) <= 1:
        return []

    # Skip header row
    snapshots = [(row[0], row[1], row[2]) for row in data[1:]]
    return snapshots

def fetch_snapshot(ts_str):
    """Fetch a single snapshot from Wayback Machine"""
    url = f"http://web.archive.org/web/{ts_str}id_/https://deepstatemap.live/api/history/last"

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def analyze_snapshot(data):
    """Analyze a snapshot and return summary stats"""
    features = data.get('map', {}).get('features', [])

    geom_types = {}
    for f in features:
        gt = f.get('geometry', {}).get('type', 'Unknown')
        geom_types[gt] = geom_types.get(gt, 0) + 1

    return {
        'total_features': len(features),
        'points': geom_types.get('Point', 0),
        'polygons': geom_types.get('Polygon', 0) + geom_types.get('MultiPolygon', 0),
        'timestamp_id': data.get('id')
    }

def download_snapshot(ts_str, dt, idx, total):
    """Download and save a single snapshot. Returns (success, data)."""
    filename = f"deepstate_wayback_{ts_str}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Skip if exists
    if os.path.exists(filepath):
        return ('skipped', None)

    data = fetch_snapshot(ts_str)
    if data:
        stats = analyze_snapshot(data)

        # Add metadata
        data['_wayback_metadata'] = {
            'wayback_timestamp': ts_str,
            'fetched_at': datetime.now().isoformat(),
            'stats': stats
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"[{idx}/{total}] {dt.strftime('%Y-%m-%d %H:%M')} - {stats['total_features']} features ({stats['points']} pts)")
        return ('success', stats)
    else:
        print(f"[{idx}/{total}] {dt.strftime('%Y-%m-%d %H:%M')} - FAILED")
        return ('failed', None)

def main():
    print("="*70)
    print("DEEPSTATE WAYBACK MACHINE - FULL HISTORICAL DATA DOWNLOAD")
    print("="*70)

    # Get all snapshots
    snapshots = get_all_snapshots()
    print(f"\nFound {len(snapshots)} total snapshots")

    if not snapshots:
        print("No snapshots found!")
        return

    # Parse timestamps and dedupe by digest (content hash)
    seen_digests = set()
    unique_snapshots = []
    for ts_str, status, digest in snapshots:
        if digest not in seen_digests:
            seen_digests.add(digest)
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
            unique_snapshots.append((ts_str, dt))

    print(f"Unique snapshots (by content hash): {len(unique_snapshots)}")

    first_ts = unique_snapshots[0][1]
    last_ts = unique_snapshots[-1][1]
    print(f"Date range: {first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')}")

    # Check existing
    existing = set(os.listdir(OUTPUT_DIR))
    to_download = []
    for ts_str, dt in unique_snapshots:
        filename = f"deepstate_wayback_{ts_str}.json"
        if filename not in existing:
            to_download.append((ts_str, dt))

    print(f"Already downloaded: {len(unique_snapshots) - len(to_download)}")
    print(f"To download: {len(to_download)}")

    if not to_download:
        print("\nAll snapshots already downloaded!")
    else:
        print(f"\nStarting download of {len(to_download)} snapshots...")
        print(f"Using exponential backoff: base delay {BASE_DELAY}s, max delay {MAX_DELAY}s")
        print(f"Will back off after {FAILURE_THRESHOLD} consecutive failures\n")

        success = 0
        failed = 0
        skipped = 0
        consecutive_failures = 0
        current_delay = BASE_DELAY

        for i, (ts_str, dt) in enumerate(to_download):
            idx = i + 1
            total = len(to_download)

            status, stats = download_snapshot(ts_str, dt, idx, total)

            if status == 'success':
                success += 1
                consecutive_failures = 0
                # Gradually reduce delay back to base after success
                current_delay = max(BASE_DELAY, current_delay / BACKOFF_MULTIPLIER)
            elif status == 'skipped':
                skipped += 1
                consecutive_failures = 0
            else:  # failed
                failed += 1
                consecutive_failures += 1

                # Check if we need to back off
                if consecutive_failures >= FAILURE_THRESHOLD:
                    current_delay = min(current_delay * BACKOFF_MULTIPLIER, MAX_DELAY)
                    print(f"\n>>> {consecutive_failures} consecutive failures - backing off to {current_delay:.0f}s delay")
                    print(f">>> Waiting {current_delay:.0f}s before retrying...\n")
                    time.sleep(current_delay)
                    consecutive_failures = 0  # Reset after backoff wait

            # Regular delay between requests
            time.sleep(current_delay)

            # Progress update every 50 successful downloads
            if success > 0 and success % 50 == 0:
                print(f"\n--- Progress: {success} success, {failed} failed, {skipped} skipped ---\n")

        print(f"\n{'='*70}")
        print(f"DOWNLOAD COMPLETE: {success} success, {failed} failed, {skipped} skipped")

    # Create index of all files
    print("\nCreating index of all downloaded snapshots...")

    all_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json') and not f.startswith('_')])
    print(f"Total files: {len(all_files)}")

    index = []
    for filename in all_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            meta = data.get('_wayback_metadata', {})
            stats = meta.get('stats', analyze_snapshot(data))
            ts = meta.get('wayback_timestamp', filename.replace('deepstate_wayback_', '').replace('.json', ''))

            index.append({
                'filename': filename,
                'wayback_timestamp': ts,
                'datetime': datetime.strptime(ts, "%Y%m%d%H%M%S").isoformat() if len(ts) == 14 else None,
                'total_features': stats.get('total_features'),
                'points': stats.get('points'),
                'polygons': stats.get('polygons')
            })
        except Exception as e:
            print(f"Error indexing {filename}: {e}")

    index_path = os.path.join(OUTPUT_DIR, "_index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex saved: {index_path}")
    print(f"Total indexed: {len(index)} snapshots")

    if index:
        # Summary stats
        dates = [datetime.fromisoformat(i['datetime']) for i in index if i['datetime']]
        if dates:
            print(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

        total_points = sum(i.get('points', 0) for i in index)
        total_polys = sum(i.get('polygons', 0) for i in index)
        print(f"Total point records across all snapshots: {total_points:,}")
        print(f"Total polygon records across all snapshots: {total_polys:,}")

if __name__ == "__main__":
    main()
