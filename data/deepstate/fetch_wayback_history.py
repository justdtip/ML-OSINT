#!/usr/bin/env python3
"""
Fetch historical DeepState snapshots from the Internet Archive Wayback Machine.
This retrieves Point + Polygon data that the GitHub archive doesn't have.
"""

import requests
import json
import os
from datetime import datetime
import time

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/deepstate/wayback_snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_snapshots():
    """Get list of all available Wayback Machine snapshots"""
    print("Fetching list of available snapshots from Wayback Machine...")

    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "deepstatemap.live/api/history/last",
        "output": "json",
        "fl": "timestamp,statuscode",
        "filter": "statuscode:200"
    }

    r = requests.get(cdx_url, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()
    if len(data) <= 1:
        return []

    # Skip header row, return list of timestamps
    snapshots = [(row[0], row[1]) for row in data[1:]]
    return snapshots

def fetch_snapshot(timestamp):
    """Fetch a single snapshot from Wayback Machine"""
    # Use 'id_' modifier to get raw content without Wayback wrapper
    url = f"http://web.archive.org/web/{timestamp}id_/https://deepstatemap.live/api/history/last"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Error fetching {timestamp}: {e}")
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

def main():
    print("="*60)
    print("DEEPSTATE WAYBACK MACHINE HISTORICAL DATA FETCHER")
    print("="*60)

    # Get all available snapshots
    snapshots = get_all_snapshots()
    print(f"\nFound {len(snapshots)} available snapshots")

    if not snapshots:
        print("No snapshots found!")
        return

    # Show date range
    first_ts = datetime.strptime(snapshots[0][0], "%Y%m%d%H%M%S")
    last_ts = datetime.strptime(snapshots[-1][0], "%Y%m%d%H%M%S")
    print(f"Date range: {first_ts.strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')}")

    # Sample strategy: get one snapshot per week
    print("\nSampling one snapshot per week...")

    sampled = {}
    for ts_str, status in snapshots:
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        week_key = dt.strftime("%Y-W%W")
        if week_key not in sampled:
            sampled[week_key] = (ts_str, dt)

    print(f"Selected {len(sampled)} weekly snapshots to download")

    # Check what we already have
    existing = set()
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.json'):
            existing.add(f)

    # Download snapshots
    results = []
    timestamps_to_fetch = sorted(sampled.values(), key=lambda x: x[1])

    for i, (ts_str, dt) in enumerate(timestamps_to_fetch):
        filename = f"deepstate_wayback_{dt.strftime('%Y%m%d')}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if filename in existing:
            print(f"[{i+1}/{len(timestamps_to_fetch)}] {dt.strftime('%Y-%m-%d')} - already exists, skipping")
            continue

        print(f"[{i+1}/{len(timestamps_to_fetch)}] Fetching {dt.strftime('%Y-%m-%d %H:%M:%S')}...", end=" ")

        data = fetch_snapshot(ts_str)
        if data:
            stats = analyze_snapshot(data)
            print(f"OK - {stats['total_features']} features ({stats['points']} pts, {stats['polygons']} polys)")

            # Save with metadata
            data['_wayback_metadata'] = {
                'wayback_timestamp': ts_str,
                'fetched_at': datetime.now().isoformat(),
                'stats': stats
            }

            with open(filepath, 'w') as f:
                json.dump(data, f)

            results.append({
                'date': dt.strftime('%Y-%m-%d'),
                'file': filename,
                **stats
            })
        else:
            print("FAILED")

        # Be nice to the Wayback Machine
        time.sleep(1)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)

    all_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    print(f"\nTotal files in {OUTPUT_DIR}: {len(all_files)}")

    # Create index file
    index_path = os.path.join(OUTPUT_DIR, "_index.json")

    # Load all files and create index
    index = []
    for filename in sorted(all_files):
        if filename.startswith('_'):
            continue
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            meta = data.get('_wayback_metadata', {})
            stats = meta.get('stats', analyze_snapshot(data))

            # Extract date from filename
            date_str = filename.replace('deepstate_wayback_', '').replace('.json', '')

            index.append({
                'filename': filename,
                'date': date_str,
                'wayback_timestamp': meta.get('wayback_timestamp'),
                'total_features': stats.get('total_features'),
                'points': stats.get('points'),
                'polygons': stats.get('polygons')
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    print(f"Created index: {index_path}")

    if index:
        print(f"\nDate range in collection: {index[0]['date']} to {index[-1]['date']}")
        print(f"Total snapshots: {len(index)}")

if __name__ == "__main__":
    main()
