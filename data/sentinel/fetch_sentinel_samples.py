#!/usr/bin/env python3
"""
Fetch sample Sentinel-2 data from Copernicus Data Space.

This script demonstrates how to search and download Sentinel-2 imagery
using the STAC API (no authentication required for search, but required for download).

For actual downloads, you need to register at:
https://dataspace.copernicus.eu/

Categories of Sentinel data available:
- SENTINEL-2 L1C: Top-of-atmosphere reflectance (raw radiance)
- SENTINEL-2 L2A: Bottom-of-atmosphere reflectance (atmospherically corrected)
"""

import requests
import json
from datetime import datetime
import os

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/sentinel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copernicus Data Space STAC API
STAC_API = "https://catalogue.dataspace.copernicus.eu/stac"

# Ukraine bounding box (approximate conflict zone: Donetsk/Luhansk region)
UKRAINE_BBOX = [37.0, 47.5, 39.5, 49.5]  # [west, south, east, north]

def search_sentinel2_products(bbox, start_date, end_date, max_cloud=30, limit=10, collection="sentinel-2-l2a"):
    """
    Search for Sentinel-2 products using STAC API.

    Args:
        bbox: [west, south, east, north] bounding box
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_cloud: Maximum cloud cover percentage
        limit: Maximum number of results
        collection: Collection ID (sentinel-2-l1c or sentinel-2-l2a)

    Returns:
        List of product metadata dictionaries
    """
    search_url = f"{STAC_API}/search"

    # STAC search request
    search_body = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud}
        }
    }

    print(f"Searching Sentinel-2 products...")
    print(f"  Bbox: {bbox}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Max cloud cover: {max_cloud}%")

    try:
        response = requests.post(search_url, json=search_body, timeout=60)
        response.raise_for_status()
        results = response.json()

        features = results.get('features', [])
        print(f"  Found {len(features)} products")

        return features
    except Exception as e:
        print(f"  Error: {e}")
        return []

def get_collection_info():
    """Get information about available Sentinel collections."""
    collections_url = f"{STAC_API}/collections"

    print("Fetching available collections...")

    try:
        response = requests.get(collections_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        collections = data.get('collections', [])

        # Filter for Sentinel collections
        sentinel_collections = [c for c in collections if 'SENTINEL' in c.get('id', '').upper()]

        return sentinel_collections
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return []

def analyze_product(feature):
    """Extract key metadata from a STAC feature."""
    props = feature.get('properties', {})

    return {
        'id': feature.get('id'),
        'datetime': props.get('datetime'),
        'platform': props.get('platform'),
        'instrument': props.get('instruments', []),
        'cloud_cover': props.get('eo:cloud_cover'),
        'processing_level': props.get('processing:level') or props.get('processingLevel'),
        'tile_id': props.get('tileId') or props.get('title', '')[:6],
        'size_mb': props.get('size', 0) / (1024*1024) if props.get('size') else None,
        'bbox': feature.get('bbox'),
        'assets': list(feature.get('assets', {}).keys())
    }

def main():
    print("=" * 70)
    print("COPERNICUS DATA SPACE - SENTINEL-2 SAMPLE SEARCH")
    print("=" * 70)

    # 1. Get available collections
    print("\n[1] AVAILABLE SENTINEL COLLECTIONS")
    print("-" * 50)

    collections = get_collection_info()

    sentinel_info = {}
    for coll in collections:
        coll_id = coll.get('id', 'Unknown')
        if 'SENTINEL' in coll_id.upper():
            desc = coll.get('description', 'No description')[:100]
            sentinel_info[coll_id] = {
                'description': desc,
                'title': coll.get('title', coll_id),
                'license': coll.get('license', 'Unknown')
            }
            print(f"\n  {coll_id}")
            print(f"    Title: {coll.get('title', 'N/A')}")
            print(f"    Desc: {desc}...")

    # 2. Search for sample products (recent imagery over Ukraine conflict zone)
    print("\n\n[2] SEARCHING FOR SAMPLE PRODUCTS")
    print("-" * 50)

    # Search different time periods
    searches = [
        ("Early conflict (Mar 2022)", "2022-03-01", "2022-03-15"),
        ("Major offensive (Sep 2022)", "2022-09-01", "2022-09-15"),
        ("Recent (Dec 2024)", "2024-12-01", "2024-12-15"),
    ]

    all_products = []

    for label, start, end in searches:
        print(f"\n  {label}:")
        products = search_sentinel2_products(UKRAINE_BBOX, start, end, max_cloud=20, limit=5)

        for p in products:
            meta = analyze_product(p)
            all_products.append(meta)
            print(f"    - {meta['id'][:50]}...")
            print(f"      Date: {meta['datetime']}, Cloud: {meta['cloud_cover']}%")
            print(f"      Level: {meta['processing_level']}")

    # 3. Save search results
    print("\n\n[3] SAVING SEARCH RESULTS")
    print("-" * 50)

    output = {
        'search_time': datetime.now().isoformat(),
        'bbox': UKRAINE_BBOX,
        'collections': sentinel_info,
        'sample_products': all_products
    }

    output_path = os.path.join(OUTPUT_DIR, "sentinel_search_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved: {output_path}")

    # 4. Summary of data categories
    print("\n\n[4] SENTINEL-2 DATA CATEGORIES")
    print("=" * 70)

    categories = """
┌─────────────────┬────────────────────────────────────────────────────────────┐
│ Category        │ Description                                                │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SENTINEL-2 L1C  │ Top-of-Atmosphere (TOA) reflectance. Raw radiance data    │
│                 │ before atmospheric correction. 13 spectral bands at       │
│                 │ 10m/20m/60m resolution. Good for time series consistency. │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SENTINEL-2 L2A  │ Bottom-of-Atmosphere (BOA) reflectance. Atmospherically   │
│                 │ corrected surface reflectance. Ready for analysis.        │
│                 │ Includes Scene Classification Map (SCL) for cloud/water.  │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ SPECTRAL BANDS  │ B02 (Blue, 10m), B03 (Green, 10m), B04 (Red, 10m),        │
│                 │ B08 (NIR, 10m), B05-B07 (Red Edge, 20m),                  │
│                 │ B8A (NIR narrow, 20m), B11-B12 (SWIR, 20m),               │
│                 │ B01 (Coastal, 60m), B09-B10 (Water vapor/Cirrus, 60m)     │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ COMMON INDICES  │ NDVI (vegetation), NDWI (water), NBR (burn severity),     │
│                 │ NDBI (built-up), BSI (bare soil)                          │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ USE CASES       │ Land cover mapping, vegetation health, damage assessment, │
│                 │ change detection, urban monitoring, fire/burn mapping     │
└─────────────────┴────────────────────────────────────────────────────────────┘

To download actual data, you need:
1. Register at https://dataspace.copernicus.eu/
2. Generate API credentials (OAuth2)
3. Use the OData API or S3 bucket access

For OSINT analysis:
- L2A is preferred (atmospherically corrected)
- 5-day revisit time (with both S2A and S2B satellites)
- Free and open data policy
"""
    print(categories)

    # 5. Show sample download command
    print("\n[5] EXAMPLE DOWNLOAD (requires authentication)")
    print("-" * 50)

    if all_products:
        sample = all_products[0]
        print(f"""
To download product: {sample['id']}

1. Get access token:
   curl -X POST 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' \\
     -d 'grant_type=password&username=YOUR_EMAIL&password=YOUR_PASSWORD&client_id=cdse-public'

2. Download via OData:
   curl -H "Authorization: Bearer $TOKEN" \\
     'https://catalogue.dataspace.copernicus.eu/odata/v1/Products({sample['id']})/$value' \\
     -o product.zip
""")

    print("\n" + "=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)
    print(f"\nTotal products found: {len(all_products)}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
