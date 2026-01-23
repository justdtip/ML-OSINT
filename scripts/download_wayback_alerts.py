#!/usr/bin/env python3
"""
Download historical snapshots of alerts.com.ua from the Wayback Machine.

This script:
1. Queries the CDX API to get all available snapshots
2. Downloads each unique snapshot (skipping duplicates based on digest)
3. Saves the map images which show active air raid alerts
4. Extracts timestamps for correlation with other data
"""

import os
import sys
import json
import csv
import requests
import time
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import hashlib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR as PROJECT_DATA_DIR, ensure_dir

# Configuration
DATA_DIR = ensure_dir(PROJECT_DATA_DIR / "wayback" / "alerts_com_ua")

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_URL = "https://web.archive.org/web"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


def get_snapshot_list(url="alerts.com.ua", include_subpages=True):
    """
    Query the CDX API to get all available snapshots.
    Returns list of (timestamp, original_url, digest) tuples.
    """
    query_url = f"{url}/*" if include_subpages else url

    params = {
        'url': query_url,
        'output': 'json',
        'filter': 'statuscode:200',
        'collapse': 'digest',  # Collapse duplicate content
    }

    print(f"Querying CDX API for {query_url}...")
    response = requests.get(CDX_API, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if not data:
        return []

    # First row is header
    header = data[0]
    snapshots = []

    for row in data[1:]:
        record = dict(zip(header, row))
        snapshots.append({
            'timestamp': record['timestamp'],
            'url': record['original'],
            'digest': record['digest'],
            'mimetype': record.get('mimetype', ''),
            'length': record.get('length', 0)
        })

    return snapshots


def download_snapshot(timestamp, url):
    """
    Download a specific snapshot from the Wayback Machine.
    Returns the HTML content or None if failed.
    """
    # Use 'id_' modifier to get original content without Wayback toolbar
    wayback_url = f"{WAYBACK_URL}/{timestamp}id_/{url}"

    try:
        response = requests.get(wayback_url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"    Error downloading {timestamp}: {e}")
        return None


def parse_alert_data(html, timestamp):
    """
    Parse alert information from the HTML.
    Returns dict with extracted data.
    """
    soup = BeautifulSoup(html, 'html.parser')

    data = {
        'timestamp': timestamp,
        'datetime': datetime.strptime(timestamp, '%Y%m%d%H%M%S').isoformat(),
        'regions_with_alerts': [],
        'total_alerts': 0,
        'alert_types': {},
        'raw_text': ''
    }

    # Try to find alert regions - the exact structure may vary
    # Look for common patterns in Ukrainian alert sites

    # Method 1: Look for region elements with alert status
    regions = soup.find_all(class_=re.compile(r'(region|oblast|area)', re.I))
    for region in regions:
        text = region.get_text(strip=True)
        if text:
            # Check if region has active alert
            classes = region.get('class', [])
            if any('alert' in c.lower() or 'active' in c.lower() or 'danger' in c.lower()
                   for c in classes):
                data['regions_with_alerts'].append(text)

    # Method 2: Look for alert list items
    alert_items = soup.find_all(class_=re.compile(r'alert', re.I))
    for item in alert_items:
        text = item.get_text(strip=True)
        if text and len(text) < 200:  # Reasonable length for region name
            if text not in data['regions_with_alerts']:
                data['regions_with_alerts'].append(text)

    # Method 3: Look for SVG map with highlighted regions
    svg_regions = soup.find_all('path', {'data-name': True})
    for path in svg_regions:
        fill = path.get('fill', '').lower()
        # Red/orange fills typically indicate alerts
        if fill in ['#ff0000', '#f00', 'red', '#ff6600', '#ff3300', 'orange']:
            region_name = path.get('data-name', '')
            if region_name and region_name not in data['regions_with_alerts']:
                data['regions_with_alerts'].append(region_name)

    # Count total
    data['total_alerts'] = len(data['regions_with_alerts'])

    # Get page text for analysis
    body = soup.find('body')
    if body:
        data['raw_text'] = ' '.join(body.get_text().split())[:5000]  # First 5000 chars

    return data


def save_snapshot(html, timestamp, url):
    """Save the raw HTML snapshot."""
    # Create filename from timestamp and URL
    url_slug = re.sub(r'[^\w]', '_', url)[:50]
    filename = f"{timestamp}_{url_slug}.html"
    filepath = DATA_DIR / "html" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)

    return filepath


def download_map_image(timestamp, url):
    """
    Download a map image snapshot from the Wayback Machine.
    Returns the binary content or None if failed.
    """
    # Use 'id_' modifier to get original content
    wayback_url = f"{WAYBACK_URL}/{timestamp}id_/{url}"

    try:
        response = requests.get(wayback_url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"    Error downloading {timestamp}: {e}")
        return None


def save_map_image(data, timestamp, url):
    """Save the map image."""
    filename = f"{timestamp}_map.png"
    filepath = DATA_DIR / "maps" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        f.write(data)

    return filepath


def main():
    print("=" * 70)
    print("WAYBACK MACHINE SNAPSHOT DOWNLOADER - alerts.com.ua")
    print("=" * 70)

    # Get list of map image snapshots (these show actual alert status)
    print("\nFetching map.png snapshot list...")
    map_snapshots = get_snapshot_list("alerts.com.ua/map.png", include_subpages=False)
    print(f"Found {len(map_snapshots)} map snapshots")

    # Also get main page snapshots for documentation
    print("\nFetching main page snapshots...")
    page_snapshots = get_snapshot_list("alerts.com.ua", include_subpages=False)
    print(f"Found {len(page_snapshots)} page snapshots")

    # Process map images (the main data source)
    print("\n" + "-" * 70)
    print("DOWNLOADING MAP IMAGES")
    print("-" * 70)

    # Remove duplicates by digest
    seen_digests = set()
    unique_maps = []
    for s in map_snapshots:
        if s['digest'] not in seen_digests:
            seen_digests.add(s['digest'])
            unique_maps.append(s)

    print(f"Unique map images to download: {len(unique_maps)}")

    # Prepare CSV for map metadata
    csv_file = DATA_DIR / "alert_maps.csv"
    fieldnames = ['timestamp', 'datetime', 'url', 'digest', 'image_file']

    # Check existing
    existing_timestamps = set()
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_timestamps.add(row['timestamp'])
        print(f"Already downloaded: {len(existing_timestamps)} maps")

    # Download each map
    mode = 'a' if csv_file.exists() else 'w'
    with open(csv_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        downloaded = 0
        skipped = 0
        errors = 0

        for i, snapshot in enumerate(unique_maps):
            ts = snapshot['timestamp']
            url = snapshot['url']

            # Skip if already downloaded
            if ts in existing_timestamps:
                skipped += 1
                continue

            dt = datetime.strptime(ts, '%Y%m%d%H%M%S')
            print(f"[{i+1}/{len(unique_maps)}] {dt.strftime('%Y-%m-%d %H:%M:%S')}", end='')

            # Download
            data = download_map_image(ts, url)

            if data and len(data) > 1000:  # Valid image
                # Save image
                image_file = save_map_image(data, ts, url)

                # Write to CSV
                writer.writerow({
                    'timestamp': ts,
                    'datetime': dt.isoformat(),
                    'url': url,
                    'digest': snapshot['digest'],
                    'image_file': str(image_file.relative_to(DATA_DIR))
                })

                f.flush()
                downloaded += 1
                print(f" - saved ({len(data)/1024:.1f} KB)")
            else:
                errors += 1
                print(" - FAILED")

            # Rate limiting
            time.sleep(REQUEST_DELAY)

    # Also download page snapshots
    print("\n" + "-" * 70)
    print("DOWNLOADING HTML PAGES")
    print("-" * 70)

    seen_digests = set()
    unique_pages = []
    for s in page_snapshots:
        if s['digest'] not in seen_digests:
            seen_digests.add(s['digest'])
            unique_pages.append(s)

    print(f"Unique pages to download: {len(unique_pages)}")

    html_csv = DATA_DIR / "alerts_pages.csv"
    html_fieldnames = ['timestamp', 'datetime', 'url', 'digest', 'html_file']

    existing_page_ts = set()
    if html_csv.exists():
        with open(html_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_page_ts.add(row['timestamp'])

    mode = 'a' if html_csv.exists() else 'w'
    with open(html_csv, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=html_fieldnames)
        if mode == 'w':
            writer.writeheader()

        for i, snapshot in enumerate(unique_pages):
            ts = snapshot['timestamp']
            url = snapshot['url']

            if ts in existing_page_ts:
                continue

            dt = datetime.strptime(ts, '%Y%m%d%H%M%S')
            print(f"[{i+1}/{len(unique_pages)}] {dt.strftime('%Y-%m-%d %H:%M:%S')}", end='')

            html = download_snapshot(ts, url)

            if html:
                html_file = save_snapshot(html, ts, url)
                writer.writerow({
                    'timestamp': ts,
                    'datetime': dt.isoformat(),
                    'url': url,
                    'digest': snapshot['digest'],
                    'html_file': str(html_file.relative_to(DATA_DIR))
                })
                f.flush()
                print(f" - saved")
            else:
                print(" - FAILED")

            time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"  Map images downloaded: {downloaded}")
    print(f"  Skipped (already had): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Data saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
