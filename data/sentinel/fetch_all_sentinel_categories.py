#!/usr/bin/env python3
"""
Comprehensive Sentinel data discovery and sample download script.

Searches all major Sentinel mission categories and downloads thumbnail previews
(no authentication required for thumbnails).
"""

import requests
import json
from datetime import datetime
import os
import time

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/sentinel"
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

# Copernicus Data Space STAC API
STAC_API = "https://catalogue.dataspace.copernicus.eu/stac"

# Ukraine conflict zone bounding box (Donetsk/Luhansk region)
UKRAINE_BBOX = [37.0, 47.5, 39.5, 49.5]

# Key Sentinel collections to sample
COLLECTIONS = {
    # Sentinel-2: Optical (Land monitoring, vegetation)
    "sentinel-2-l1c": {
        "name": "Sentinel-2 Level-1C",
        "mission": "Sentinel-2",
        "category": "Optical Imagery",
        "resolution": "10-60m",
        "description": "Top-of-Atmosphere reflectance. Raw radiance before atmospheric correction. 13 spectral bands.",
        "use_cases": ["Land cover mapping", "Vegetation monitoring", "Change detection", "Urban mapping"]
    },
    "sentinel-2-l2a": {
        "name": "Sentinel-2 Level-2A",
        "mission": "Sentinel-2",
        "category": "Optical Imagery",
        "resolution": "10-60m",
        "description": "Surface reflectance (atmospherically corrected). Ready for analysis. Includes scene classification.",
        "use_cases": ["NDVI vegetation analysis", "Burn severity mapping", "Water body detection", "Damage assessment"]
    },

    # Sentinel-1: Radar (SAR - works through clouds)
    "sentinel-1-grd": {
        "name": "Sentinel-1 GRD",
        "mission": "Sentinel-1",
        "category": "Radar (SAR)",
        "resolution": "10-40m",
        "description": "Ground Range Detected radar imagery. Works through clouds/night. Detects surface changes.",
        "use_cases": ["Flood mapping", "Ship detection", "Ground deformation", "Infrastructure monitoring"]
    },

    # Sentinel-3: Ocean/Land/Atmosphere monitoring
    "sentinel-3-olci-2-lfr-ntc": {
        "name": "Sentinel-3 OLCI Land",
        "mission": "Sentinel-3",
        "category": "Land/Ocean Color",
        "resolution": "300m",
        "description": "Ocean and Land Color Instrument. Global vegetation, fire, water quality monitoring.",
        "use_cases": ["Global vegetation health", "Active fire detection", "Water quality", "Atmospheric correction"]
    },
    "sentinel-3-sl-2-frp-ntc": {
        "name": "Sentinel-3 SLSTR Fire",
        "mission": "Sentinel-3",
        "category": "Thermal/Fire",
        "resolution": "1km",
        "description": "Fire Radiative Power product. Detects active fires and measures fire intensity.",
        "use_cases": ["Active fire detection", "Fire intensity measurement", "Burn area mapping"]
    },

    # Sentinel-5P: Atmospheric composition
    "sentinel-5p-l2-no2-offl": {
        "name": "Sentinel-5P NO2",
        "mission": "Sentinel-5P",
        "category": "Atmospheric",
        "resolution": "5.5km",
        "description": "Nitrogen Dioxide concentration. Traces industrial emissions, explosions, fires.",
        "use_cases": ["Air quality monitoring", "Industrial activity detection", "Explosion/fire signatures"]
    },
    "sentinel-5p-l2-co-offl": {
        "name": "Sentinel-5P CO",
        "mission": "Sentinel-5P",
        "category": "Atmospheric",
        "resolution": "7km",
        "description": "Carbon Monoxide concentration. Indicates combustion, fires, industrial activity.",
        "use_cases": ["Fire smoke tracking", "Industrial emissions", "Air quality"]
    },
}


def search_collection(collection_id, bbox, start_date, end_date, limit=3):
    """Search a specific collection for products."""
    search_url = f"{STAC_API}/search"

    search_body = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit
    }

    try:
        response = requests.post(search_url, json=search_body, timeout=60)
        response.raise_for_status()
        return response.json().get('features', [])
    except Exception as e:
        print(f"    Error searching {collection_id}: {e}")
        return []


def download_thumbnail(feature, collection_id):
    """Download thumbnail/quicklook image if available."""
    assets = feature.get('assets', {})

    # Common thumbnail asset names
    thumb_keys = ['thumbnail', 'quicklook', 'preview', 'rendered_preview']

    for key in thumb_keys:
        if key in assets:
            thumb_url = assets[key].get('href')
            if thumb_url:
                try:
                    # Clean product ID for filename
                    product_id = feature.get('id', 'unknown')[:50].replace('/', '_')
                    ext = thumb_url.split('.')[-1][:4]
                    if ext not in ['png', 'jpg', 'jpeg', 'tif']:
                        ext = 'png'

                    filename = f"{collection_id}_{product_id}.{ext}"
                    filepath = os.path.join(THUMBNAILS_DIR, filename)

                    if os.path.exists(filepath):
                        return filepath

                    r = requests.get(thumb_url, timeout=30)
                    if r.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(r.content)
                        return filepath
                except Exception as e:
                    pass

    return None


def main():
    print("=" * 80)
    print("COPERNICUS SENTINEL MISSIONS - COMPREHENSIVE DATA CATEGORIES")
    print("=" * 80)
    print(f"\nSearch area: Ukraine conflict zone (Donetsk/Luhansk)")
    print(f"Bounding box: {UKRAINE_BBOX}")
    print(f"Date range: Sep 2022 (major offensive period)")

    results = {}
    all_samples = []

    # Search each collection
    print("\n" + "-" * 80)
    print("SEARCHING SENTINEL DATA COLLECTIONS")
    print("-" * 80)

    for coll_id, coll_info in COLLECTIONS.items():
        print(f"\n[{coll_info['mission']}] {coll_info['name']}")
        print(f"  Category: {coll_info['category']} | Resolution: {coll_info['resolution']}")

        # Search for products
        products = search_collection(
            coll_id,
            UKRAINE_BBOX,
            "2022-09-01",
            "2022-09-15",
            limit=3
        )

        results[coll_id] = {
            **coll_info,
            'products_found': len(products),
            'samples': []
        }

        if products:
            print(f"  Found {len(products)} products")

            for p in products[:2]:  # Sample 2 per collection
                props = p.get('properties', {})
                product_info = {
                    'id': p.get('id'),
                    'datetime': props.get('datetime'),
                    'cloud_cover': props.get('eo:cloud_cover'),
                    'platform': props.get('platform'),
                    'bbox': p.get('bbox'),
                    'assets': list(p.get('assets', {}).keys())
                }

                # Try to download thumbnail
                thumb = download_thumbnail(p, coll_id)
                if thumb:
                    product_info['thumbnail'] = thumb
                    print(f"    ✓ Downloaded thumbnail: {os.path.basename(thumb)}")

                results[coll_id]['samples'].append(product_info)
                all_samples.append({
                    'collection': coll_id,
                    **product_info
                })

                print(f"    - {p.get('id', 'N/A')[:60]}...")
                if props.get('eo:cloud_cover') is not None:
                    print(f"      Cloud: {props.get('eo:cloud_cover'):.1f}%")
        else:
            print(f"  No products found (may need different date range)")

        time.sleep(0.5)  # Rate limiting

    # Save results
    output = {
        'search_time': datetime.now().isoformat(),
        'bbox': UKRAINE_BBOX,
        'search_period': '2022-09-01 to 2022-09-15',
        'collections': results
    }

    output_path = os.path.join(OUTPUT_DIR, "sentinel_categories_overview.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("SENTINEL MISSION CATEGORIES SUMMARY")
    print("=" * 80)

    categories_table = """
┌────────────────────┬───────────────────────────────────────────────────────────────────────┐
│ MISSION            │ DESCRIPTION & USE CASES                                               │
├────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ SENTINEL-1         │ C-band SAR (Synthetic Aperture Radar)                                 │
│ (Radar)            │ • Works through clouds, day/night                                     │
│                    │ • 10-40m resolution, 6-day revisit                                    │
│                    │ • Detects: floods, ships, ground deformation, infrastructure changes  │
│                    │ • OSINT: Identify military movements, damaged structures, flooding    │
├────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ SENTINEL-2         │ Multi-spectral Optical Imagery                                        │
│ (Optical)          │ • 13 bands: visible, NIR, SWIR at 10-60m resolution                   │
│                    │ • 5-day revisit (with both S2A/S2B)                                   │
│                    │ • L1C: Top-of-atmosphere (raw)                                        │
│                    │ • L2A: Surface reflectance (atmospherically corrected) ← PREFERRED   │
│                    │ • OSINT: Damage assessment, burn scars, vegetation change, urban      │
├────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ SENTINEL-3         │ Land/Ocean/Atmosphere Monitoring                                      │
│ (Global)           │ • OLCI: Ocean/Land Color (300m) - vegetation, water quality           │
│                    │ • SLSTR: Thermal (1km) - fire detection, sea surface temp             │
│                    │ • SRAL: Altimetry - sea level, ice thickness                          │
│                    │ • OSINT: Active fire detection, large-scale environmental monitoring  │
├────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ SENTINEL-5P        │ Atmospheric Composition (TROPOMI instrument)                          │
│ (Atmosphere)       │ • Measures: NO2, CO, O3, CH4, SO2, aerosols                           │
│                    │ • 5.5-7km resolution, daily global coverage                           │
│                    │ • OSINT: Industrial activity, explosions, fire smoke plumes           │
│                    │ • NO2 spikes can indicate military/industrial activity                │
├────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ SENTINEL-6         │ Ocean Altimetry                                                       │
│ (Oceans)           │ • Precise sea surface height measurements                             │
│                    │ • Continuity of Jason missions                                        │
│                    │ • Less relevant for land-based OSINT                                  │
└────────────────────┴───────────────────────────────────────────────────────────────────────┘

PROCESSING LEVELS:
┌─────────┬────────────────────────────────────────────────────────────────────────────────┐
│ Level   │ Description                                                                    │
├─────────┼────────────────────────────────────────────────────────────────────────────────┤
│ L0      │ Raw instrument data (not publicly distributed)                                 │
│ L1      │ Radiometrically corrected, georeferenced (TOA for optical)                     │
│ L2      │ Derived geophysical products (surface reflectance, atmospheric variables)     │
│ L3      │ Temporally/spatially composited products (mosaics, time series)               │
└─────────┴────────────────────────────────────────────────────────────────────────────────┘

TIMELINESS CODES:
• NRT  = Near Real-Time (within 3 hours)
• STC  = Short Time Critical (within 48 hours)
• NTC  = Non Time Critical (within 1 month) - highest quality
• OFFL = Offline processing
• RPRO = Reprocessed data
"""
    print(categories_table)

    # Results summary
    print("\nSEARCH RESULTS SUMMARY:")
    print("-" * 50)
    total_thumbnails = len([f for f in os.listdir(THUMBNAILS_DIR) if not f.startswith('.')])
    print(f"Collections searched: {len(COLLECTIONS)}")
    print(f"Sample products found: {sum(r['products_found'] for r in results.values())}")
    print(f"Thumbnails downloaded: {total_thumbnails}")
    print(f"\nResults saved: {output_path}")
    print(f"Thumbnails saved: {THUMBNAILS_DIR}/")

    # For OSINT relevance
    print("\n" + "=" * 80)
    print("RECOMMENDED FOR UKRAINE OSINT ANALYSIS")
    print("=" * 80)
    print("""
1. SENTINEL-2 L2A (Primary)
   - High-resolution (10m) optical imagery
   - Calculate NDVI for vegetation damage
   - NBR for burn severity mapping
   - Visual damage assessment

2. SENTINEL-1 GRD (All-weather)
   - Works through clouds (radar)
   - Detect flooded areas
   - Monitor infrastructure changes
   - Ship/vehicle detection

3. SENTINEL-5P NO2/CO (Activity detection)
   - Industrial emissions tracking
   - Explosion/fire smoke signatures
   - Requires large-scale analysis

4. NASA FIRMS + Sentinel-3 SLSTR (Fire)
   - Cross-validate fire detections
   - Fire radiative power measurement
""")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
