#!/usr/bin/env python3
"""
GeoConfirmed API Downloader

Downloads geolocated conflict events from the GeoConfirmed API.
Based on the QGIS plugin client by Silverfish94 & Claude.

API Base: https://geoconfirmed.org/api
Key endpoints:
- /Conflict - List all conflicts
- /Placemark/v2/{conflict}/{skip}/{take} - Placemarks with pagination

Each placemark contains:
- id: Unique identifier
- name: Event name/title
- description: Detailed description
- lat, lng: Coordinates
- dateCreated: Timestamp
- gear: Equipment involved
- sources: Source URLs
- plusCode: Location text (city/region/country)

Author: ML Engineering Team
Date: 2026-01-27
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "geoconfirmed"

BASE_URL = "https://geoconfirmed.org/api"


def make_request(url: str, method: str = "GET", body: Optional[Dict] = None,
                 retries: int = 3, delay: float = 1.0) -> Optional[Dict]:
    """
    Make an HTTP request to the GeoConfirmed API.

    Args:
        url: Full URL to request
        method: HTTP method (GET or POST)
        body: Optional JSON body for POST requests
        retries: Number of retry attempts
        delay: Delay between retries

    Returns:
        Parsed JSON response or None
    """
    for attempt in range(retries):
        try:
            if body is not None:
                data = json.dumps(body).encode('utf-8')
            else:
                data = None

            req = urllib.request.Request(
                url,
                data=data,
                method=method,
                headers={
                    "User-Agent": "ML-OSINT-Research/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} for {url}, attempt {attempt + 1}/{retries}")
            if e.code == 404:
                return None
        except urllib.error.URLError as e:
            print(f"  URL error: {e.reason}, attempt {attempt + 1}/{retries}")
        except Exception as e:
            print(f"  Error: {e}, attempt {attempt + 1}/{retries}")

        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))

    return None


def get_conflicts() -> List[Dict]:
    """Fetch list of all available conflicts."""
    url = f"{BASE_URL}/Conflict"
    print(f"Fetching conflicts from {url}")
    data = make_request(url)
    if data and isinstance(data, list):
        return data
    return []


def get_conflict_details(conflict_name: str) -> Optional[Dict]:
    """Fetch details for a specific conflict."""
    url = f"{BASE_URL}/Conflict/{conflict_name}"
    return make_request(url)


def get_placemarks_batch(
    conflict: str,
    skip: int = 0,
    take: int = 500,
    filter_body: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Fetch a batch of placemarks using v2 API.

    Args:
        conflict: Conflict short name (e.g., 'ukraine')
        skip: Pagination offset
        take: Batch size
        filter_body: Optional filter for server-side search

    Returns:
        API response with 'items' and 'count'
    """
    url = f"{BASE_URL}/Placemark/v2/{conflict}/{skip}/{take}"
    body = filter_body or {}
    return make_request(url, method="POST", body=body)


def get_all_placemarks(
    conflict: str,
    batch_size: int = 500,
    delay: float = 0.5,
    filter_body: Optional[Dict] = None
) -> List[Dict]:
    """
    Fetch all placemarks for a conflict with pagination.

    Args:
        conflict: Conflict short name
        batch_size: Items per batch
        delay: Delay between requests
        filter_body: Optional filter

    Returns:
        List of all placemarks
    """
    all_items = []
    skip = 0

    # First request to get total count
    print(f"Fetching placemarks for '{conflict}'...")
    data = get_placemarks_batch(conflict, skip, batch_size, filter_body)

    if data is None:
        print("  Failed to fetch initial batch")
        return []

    total_count = data.get('count', 0)
    items = data.get('items', [])
    all_items.extend(items)

    print(f"  Total placemarks available: {total_count}")
    print(f"  Batch 1: {len(items)} items (total: {len(all_items)})")

    # Fetch remaining batches
    batch_num = 1
    while len(all_items) < total_count:
        skip = len(all_items)
        batch_num += 1

        time.sleep(delay)
        data = get_placemarks_batch(conflict, skip, batch_size, filter_body)

        if data is None:
            print(f"  Batch {batch_num}: FAILED")
            break

        items = data.get('items', [])
        if not items:
            break

        all_items.extend(items)
        print(f"  Batch {batch_num}: {len(items)} items (total: {len(all_items)}/{total_count})")

    print(f"  Fetched {len(all_items)} placemarks total")
    return all_items


def extract_coordinates(placemarks: List[Dict]) -> Dict[str, int]:
    """Extract coordinate statistics from placemarks."""
    stats = {
        "total": len(placemarks),
        "with_coords": 0,
        "with_pluscode": 0,
        "with_gear": 0,
        "with_sources": 0,
    }

    for p in placemarks:
        if p.get('lat') and p.get('lng'):
            stats["with_coords"] += 1
        if p.get('plusCode'):
            stats["with_pluscode"] += 1
        if p.get('gear'):
            stats["with_gear"] += 1
        if p.get('sources'):
            stats["with_sources"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download geolocated events from GeoConfirmed API"
    )
    parser.add_argument(
        "--conflict",
        type=str,
        default="ukraine",
        help="Conflict name to download (default: ukraine)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for pagination (default: 500)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--list-conflicts",
        action="store_true",
        help="List available conflicts and exit",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GeoConfirmed API Downloader")
    print("=" * 60)

    # List conflicts mode
    if args.list_conflicts:
        conflicts = get_conflicts()
        print(f"\nAvailable conflicts ({len(conflicts)}):")
        for c in conflicts:
            name = c.get('shortName', c.get('name', 'Unknown'))
            full_name = c.get('name', '')
            print(f"  - {name}: {full_name}")
        return

    # Get conflict details
    print(f"\nFetching details for '{args.conflict}'...")
    details = get_conflict_details(args.conflict)
    if details:
        print(f"  Name: {details.get('name', 'Unknown')}")
        factions = details.get('factions', [])
        print(f"  Factions: {len(factions)}")
        for f in factions:
            print(f"    - {f.get('name', 'Unknown')}")

    # Download all placemarks
    print("\n" + "-" * 60)
    placemarks = get_all_placemarks(
        conflict=args.conflict,
        batch_size=args.batch_size,
        delay=args.delay
    )

    if not placemarks:
        print("No placemarks downloaded")
        return

    # Statistics
    stats = extract_coordinates(placemarks)
    print("\n" + "-" * 60)
    print("Statistics:")
    print(f"  Total placemarks: {stats['total']}")
    print(f"  With coordinates: {stats['with_coords']} ({100*stats['with_coords']/stats['total']:.1f}%)")
    print(f"  With PlusCode: {stats['with_pluscode']} ({100*stats['with_pluscode']/stats['total']:.1f}%)")
    print(f"  With gear info: {stats['with_gear']} ({100*stats['with_gear']/stats['total']:.1f}%)")
    print(f"  With sources: {stats['with_sources']} ({100*stats['with_sources']/stats['total']:.1f}%)")

    # Get date range
    dates = [p.get('dateCreated', '') for p in placemarks if p.get('dateCreated')]
    if dates:
        dates.sort()
        print(f"  Date range: {dates[0][:10]} to {dates[-1][:10]}")

    # Save to file
    output_file = output_dir / f"geoconfirmed_{args.conflict}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "conflict": args.conflict,
            "downloaded_at": datetime.now().isoformat(),
            "count": len(placemarks),
            "statistics": stats,
            "placemarks": placemarks
        }, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
