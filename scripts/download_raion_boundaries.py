#!/usr/bin/env python3
"""
Download Ukraine Administrative Boundaries (Raion Level)

Downloads admin level 2 (raion) boundaries from GADM for Ukraine.
These are used for point-in-polygon assignment of FIRMS/DeepState data.

Usage:
    python scripts/download_raion_boundaries.py
"""

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR

# Output directory
BOUNDARIES_DIR = DATA_DIR / "boundaries"
BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)

# GADM Ukraine admin level 2 (raion) GeoJSON
# GADM 4.1 data
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_UKR_2.json.zip"
OUTPUT_FILE = BOUNDARIES_DIR / "ukraine_raions.geojson"


def download_gadm_boundaries():
    """Download GADM Ukraine admin level 2 boundaries."""
    print("Downloading Ukraine raion boundaries from GADM...")

    zip_path = BOUNDARIES_DIR / "gadm41_UKR_2.json.zip"

    # Download
    print(f"  URL: {GADM_URL}")
    urlretrieve(GADM_URL, zip_path)
    print(f"  Downloaded to: {zip_path}")

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the JSON file in the archive
        json_files = [f for f in zf.namelist() if f.endswith('.json')]
        if json_files:
            zf.extract(json_files[0], BOUNDARIES_DIR)
            extracted_path = BOUNDARIES_DIR / json_files[0]
            extracted_path.rename(OUTPUT_FILE)
            print(f"  Extracted to: {OUTPUT_FILE}")

    # Clean up zip
    zip_path.unlink()

    # Load and summarize
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get('features', [])
    print(f"\nLoaded {len(features)} raion boundaries")

    # List unique oblasts and raion counts
    oblasts = {}
    for feature in features:
        props = feature.get('properties', {})
        oblast = props.get('NAME_1', 'Unknown')
        raion = props.get('NAME_2', 'Unknown')
        if oblast not in oblasts:
            oblasts[oblast] = []
        oblasts[oblast].append(raion)

    print("\nRaions by oblast:")
    for oblast, raions in sorted(oblasts.items()):
        print(f"  {oblast}: {len(raions)} raions")

    print(f"\nTotal: {sum(len(r) for r in oblasts.values())} raions across {len(oblasts)} oblasts")

    return OUTPUT_FILE


def identify_frontline_raions():
    """Identify raions likely to be conflict-relevant based on geography."""

    # Oblasts with active frontline or significant conflict activity
    FRONTLINE_OBLASTS = [
        "Donets'k",       # Donetsk - main frontline
        "Luhans'k",       # Luhansk - main frontline
        "Kherson",        # Southern front
        "Zaporizhia",     # Southern front (Zaporizhzhia)
        "Kharkiv",        # Eastern front
        "Dnipropetrovs'k",  # Rear area, logistics
        "Mykolayiv",      # Southern support (Mykolaiv)
        "KievCity",       # Strategic target (Kyiv city)
        "Kiev",           # Kyiv oblast
        "Sumy",           # Northern border
        "Chernihiv",      # Northern border
        "Crimea",         # Occupied
        "Sevastopol'",    # Occupied (Sevastopol)
    ]

    if not OUTPUT_FILE.exists():
        print("Boundaries file not found. Run download first.")
        return

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frontline_raions = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        oblast = props.get('NAME_1', '')
        raion = props.get('NAME_2', '')
        gid = props.get('GID_2', '')

        if any(fo.lower() in oblast.lower() for fo in FRONTLINE_OBLASTS):
            frontline_raions.append({
                'gid': gid,
                'raion': raion,
                'oblast': oblast,
            })

    print(f"\nIdentified {len(frontline_raions)} conflict-relevant raions:")
    for oblast in sorted(set(r['oblast'] for r in frontline_raions)):
        count = len([r for r in frontline_raions if r['oblast'] == oblast])
        print(f"  {oblast}: {count} raions")

    # Save list
    frontline_path = BOUNDARIES_DIR / "frontline_raions.json"
    with open(frontline_path, 'w') as f:
        json.dump(frontline_raions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved frontline raion list to: {frontline_path}")

    return frontline_raions


if __name__ == '__main__':
    output = download_gadm_boundaries()
    if output and output.exists():
        identify_frontline_raions()
