#!/usr/bin/env python3
"""
Check available date ranges for each Sentinel collection.
"""

import requests
from datetime import datetime

STAC_API = "https://catalogue.dataspace.copernicus.eu/stac"
UKRAINE_BBOX = [37.0, 47.5, 39.5, 49.5]

COLLECTIONS = [
    ("sentinel-1-grd", "Sentinel-1 GRD (Radar)"),
    ("sentinel-2-l1c", "Sentinel-2 L1C (Optical TOA)"),
    ("sentinel-2-l2a", "Sentinel-2 L2A (Optical BOA)"),
    ("sentinel-3-olci-2-lfr-ntc", "Sentinel-3 OLCI Land"),
    ("sentinel-3-sl-2-frp-ntc", "Sentinel-3 SLSTR Fire"),
    ("sentinel-5p-l2-no2-offl", "Sentinel-5P NO2"),
    ("sentinel-5p-l2-co-offl", "Sentinel-5P CO"),
]

def get_date_range(collection_id, bbox):
    """Get earliest and latest product dates for a collection."""
    search_url = f"{STAC_API}/search"

    # Get earliest
    earliest_body = {
        "collections": [collection_id],
        "bbox": bbox,
        "limit": 1,
        "sortby": [{"field": "datetime", "direction": "asc"}]
    }

    # Get latest
    latest_body = {
        "collections": [collection_id],
        "bbox": bbox,
        "limit": 1,
        "sortby": [{"field": "datetime", "direction": "desc"}]
    }

    earliest_date = None
    latest_date = None

    try:
        # Earliest
        r = requests.post(search_url, json=earliest_body, timeout=60)
        if r.status_code == 200:
            features = r.json().get('features', [])
            if features:
                earliest_date = features[0].get('properties', {}).get('datetime')

        # Latest
        r = requests.post(search_url, json=latest_body, timeout=60)
        if r.status_code == 200:
            features = r.json().get('features', [])
            if features:
                latest_date = features[0].get('properties', {}).get('datetime')

    except Exception as e:
        print(f"  Error: {e}")

    return earliest_date, latest_date

def main():
    print("=" * 80)
    print("SENTINEL DATA AVAILABILITY - DATE RANGES")
    print("=" * 80)
    print(f"Search area: Ukraine conflict zone {UKRAINE_BBOX}")
    print()

    results = []

    for coll_id, name in COLLECTIONS:
        print(f"Checking {name}...", end=" ", flush=True)
        earliest, latest = get_date_range(coll_id, UKRAINE_BBOX)

        if earliest and latest:
            # Parse dates
            try:
                e_dt = datetime.fromisoformat(earliest.replace('Z', '+00:00'))
                l_dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
                days = (l_dt - e_dt).days
                results.append({
                    'collection': coll_id,
                    'name': name,
                    'earliest': e_dt.strftime('%Y-%m-%d'),
                    'latest': l_dt.strftime('%Y-%m-%d'),
                    'days': days
                })
                print(f"{e_dt.strftime('%Y-%m-%d')} to {l_dt.strftime('%Y-%m-%d')} ({days} days)")
            except:
                print(f"{earliest} to {latest}")
        else:
            print("No data found")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Collection':<35} {'Earliest':<12} {'Latest':<12} {'Span':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<35} {r['earliest']:<12} {r['latest']:<12} {r['days']:>6} days")

    # Compare with existing datasets
    print("\n" + "=" * 80)
    print("COMPARISON WITH EXISTING OSINT DATASETS")
    print("=" * 80)
    print("""
Your existing datasets cover:
  UCDP Conflict Events:  2018-01-01 to 2024-12-31 (2,557 days)
  NASA FIRMS Fire:       2022-02-24 to 2024-12-27 (1,037 days)
  DeepState Maps:        2022-05-10 to 2026-01-15 (1,346 days)
  War Losses:            2022-02-25 to 2024-12-31 (1,040 days)

Overlap period (all 4): 2022-05-10 to 2024-12-27 (965 days)
""")

if __name__ == "__main__":
    main()
