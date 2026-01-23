#!/usr/bin/env python3
"""
Fetch DeepState live API data including Point geometries (military units, airfields, etc.)
Based on: https://github.com/sgofferj/tak-feeder-deepstate
"""

import json
import requests
from datetime import datetime
import os

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/deepstate/snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_deepstate_live():
    """Fetch the latest snapshot from DeepState API"""

    url = "https://deepstatemap.live/api/history/last"

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Alt-Used": "deepstatemap.live",
        "User-Agent": "DeepState Data Collector (OSINT Research)",
        "Connection": "close",
        "Host": "deepstatemap.live",
        "Referer": "https://deepstatemap.live/",
        "X-Requested-With": "XMLHttpRequest"
    }

    print(f"Fetching data from {url}...")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract timestamp
        timestamp_id = data.get('id')
        timestamp_dt = datetime.fromtimestamp(timestamp_id) if timestamp_id else datetime.now()

        print(f"Snapshot timestamp: {timestamp_dt.isoformat()}")

        # Analyze the data
        features = data.get('map', {}).get('features', [])

        # Count geometry types
        geom_counts = {}
        point_categories = {}
        polygon_styles = {}

        for feature in features:
            geom_type = feature.get('geometry', {}).get('type', 'Unknown')
            geom_counts[geom_type] = geom_counts.get(geom_type, 0) + 1

            props = feature.get('properties', {})

            if geom_type == 'Point':
                # Categorize points by icon
                icon = props.get('icon', 'no-icon')
                point_categories[icon] = point_categories.get(icon, 0) + 1
            elif geom_type in ['Polygon', 'MultiPolygon']:
                # Categorize polygons by style
                style = props.get('fill', props.get('styleUrl', 'no-style'))
                polygon_styles[style] = polygon_styles.get(style, 0) + 1

        print(f"\nTotal features: {len(features)}")
        print("\nGeometry types:")
        for gtype, count in sorted(geom_counts.items(), key=lambda x: -x[1]):
            print(f"  {gtype}: {count}")

        print("\nPoint categories (by icon):")
        for icon, count in sorted(point_categories.items(), key=lambda x: -x[1]):
            print(f"  {icon}: {count}")

        print("\nPolygon styles (by fill color):")
        for style, count in sorted(polygon_styles.items(), key=lambda x: -x[1]):
            print(f"  {style}: {count}")

        # Save full data
        date_str = timestamp_dt.strftime("%Y%m%d_%H%M%S")

        # Save complete snapshot
        full_path = f"{OUTPUT_DIR}/deepstate_full_{date_str}.json"
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved full snapshot: {full_path}")

        # Save as GeoJSON (just the map portion)
        geojson_path = f"{OUTPUT_DIR}/deepstate_geojson_{date_str}.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(data['map'], f, indent=2)
        print(f"Saved GeoJSON: {geojson_path}")

        # Save Points only (for military unit analysis)
        points_only = {
            "type": "FeatureCollection",
            "features": [f for f in features if f.get('geometry', {}).get('type') == 'Point']
        }
        points_path = f"{OUTPUT_DIR}/deepstate_points_{date_str}.geojson"
        with open(points_path, 'w') as f:
            json.dump(points_only, f, indent=2)
        print(f"Saved Points only: {points_path}")

        # Save Polygons only (territorial control)
        polygons_only = {
            "type": "FeatureCollection",
            "features": [f for f in features if f.get('geometry', {}).get('type') in ['Polygon', 'MultiPolygon']]
        }
        polygons_path = f"{OUTPUT_DIR}/deepstate_polygons_{date_str}.geojson"
        with open(polygons_path, 'w') as f:
            json.dump(polygons_only, f, indent=2)
        print(f"Saved Polygons only: {polygons_path}")

        # Print sample Point features
        print("\n" + "="*60)
        print("SAMPLE POINT FEATURES (Military Units)")
        print("="*60)

        point_features = [f for f in features if f.get('geometry', {}).get('type') == 'Point']
        for i, feat in enumerate(point_features[:10]):
            props = feat.get('properties', {})
            coords = feat.get('geometry', {}).get('coordinates', [])
            name = props.get('name', 'Unnamed')
            # Parse bilingual name (Ukrainian///English format)
            if '///' in str(name):
                name_parts = name.split('///')
                name_en = name_parts[1].strip() if len(name_parts) > 1 else name_parts[0]
            else:
                name_en = name

            print(f"\n{i+1}. {name_en[:60]}")
            print(f"   Coords: [{coords[0]:.4f}, {coords[1]:.4f}]")
            print(f"   Icon: {props.get('icon', 'N/A')}")
            if props.get('description'):
                print(f"   Desc: {props.get('description')[:50]}")

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    fetch_deepstate_live()
