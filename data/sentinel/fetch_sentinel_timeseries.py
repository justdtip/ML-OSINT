#!/usr/bin/env python3
"""
Fetch Sentinel time series data for Ukraine conflict period.

Downloads metadata/counts for integration with existing OSINT datasets:
- UCDP conflict events
- NASA FIRMS fire hotspots
- DeepState territorial maps
- War losses data

Focus on metrics that can be correlated:
- Sentinel-2: Cloud-free imagery availability, could indicate observation gaps
- Sentinel-5P NO2/CO: Atmospheric pollution levels (industrial/military activity proxy)
- Sentinel-3 FRP: Fire radiative power (cross-validate with FIRMS)
"""

import requests
import json
from datetime import datetime, timedelta
import os
import time

OUTPUT_DIR = "/Users/daniel.tipton/ML_OSINT/data/sentinel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STAC_API = "https://catalogue.dataspace.copernicus.eu/stac"

# Ukraine conflict zone bounding box
UKRAINE_BBOX = [37.0, 47.5, 39.5, 49.5]

# Date range matching existing datasets overlap
START_DATE = "2022-05-01"  # DeepState starts May 2022
END_DATE = "2024-12-31"    # Match other datasets


def search_products_for_month(collection_id, year, month, bbox):
    """Search for products in a specific month."""
    search_url = f"{STAC_API}/search"

    # Calculate month boundaries
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"

    search_body = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": f"{start}T00:00:00Z/{end}T00:00:00Z",
        "limit": 1000  # Get all products for the month
    }

    try:
        r = requests.post(search_url, json=search_body, timeout=120)
        if r.status_code == 200:
            return r.json().get('features', [])
        else:
            return []
    except Exception as e:
        print(f"    Error: {e}")
        return []


def extract_product_stats(features, collection_id):
    """Extract relevant statistics from product features."""
    if not features:
        return {
            'count': 0,
            'avg_cloud_cover': None,
            'min_cloud_cover': None,
            'dates': []
        }

    cloud_covers = []
    dates = []

    for f in features:
        props = f.get('properties', {})

        # Get datetime
        dt = props.get('datetime')
        if dt:
            dates.append(dt[:10])  # Just date part

        # Get cloud cover (optical sensors)
        cc = props.get('eo:cloud_cover')
        if cc is not None:
            cloud_covers.append(cc)

    return {
        'count': len(features),
        'avg_cloud_cover': sum(cloud_covers) / len(cloud_covers) if cloud_covers else None,
        'min_cloud_cover': min(cloud_covers) if cloud_covers else None,
        'cloud_free_count': sum(1 for c in cloud_covers if c < 20) if cloud_covers else None,
        'unique_dates': len(set(dates)),
        'dates': sorted(set(dates))
    }


def main():
    print("=" * 80)
    print("SENTINEL TIME SERIES DATA COLLECTION")
    print("=" * 80)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Bounding box: {UKRAINE_BBOX}")
    print()

    # Collections to fetch
    collections = {
        'sentinel-2-l2a': 'Sentinel-2 L2A (Optical)',
        'sentinel-1-grd': 'Sentinel-1 GRD (Radar)',
        'sentinel-5p-l2-no2-offl': 'Sentinel-5P NO2',
        'sentinel-5p-l2-co-offl': 'Sentinel-5P CO',
        'sentinel-3-sl-2-frp-ntc': 'Sentinel-3 Fire (FRP)',
    }

    # Parse date range
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")

    # Generate months
    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print(f"Fetching data for {len(months)} months across {len(collections)} collections")
    print(f"Total API calls: {len(months) * len(collections)}")
    print()

    all_data = {}

    for coll_id, coll_name in collections.items():
        print(f"\n{'='*60}")
        print(f"[{coll_name}]")
        print('='*60)

        monthly_data = []

        for i, (year, month) in enumerate(months):
            month_str = f"{year}-{month:02d}"
            print(f"  {month_str}...", end=" ", flush=True)

            features = search_products_for_month(coll_id, year, month, UKRAINE_BBOX)
            stats = extract_product_stats(features, coll_id)

            monthly_data.append({
                'year': year,
                'month': month,
                'month_str': month_str,
                **stats
            })

            if stats['count'] > 0:
                if stats['avg_cloud_cover'] is not None:
                    print(f"{stats['count']} products, avg cloud: {stats['avg_cloud_cover']:.1f}%")
                else:
                    print(f"{stats['count']} products")
            else:
                print("no data")

            # Rate limiting
            time.sleep(0.3)

        all_data[coll_id] = {
            'name': coll_name,
            'monthly': monthly_data,
            'total_products': sum(m['count'] for m in monthly_data)
        }

        print(f"\n  Total: {all_data[coll_id]['total_products']} products")

    # Save raw data
    output_path = os.path.join(OUTPUT_DIR, "sentinel_timeseries_raw.json")
    with open(output_path, 'w') as f:
        json.dump({
            'bbox': UKRAINE_BBOX,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'fetched_at': datetime.now().isoformat(),
            'collections': all_data
        }, f, indent=2)
    print(f"\n\nRaw data saved: {output_path}")

    # Create weekly aggregation for correlation analysis
    print("\n" + "=" * 80)
    print("CREATING WEEKLY AGGREGATIONS")
    print("=" * 80)

    # For weekly data, we need to re-query with finer granularity
    # Use the dates we collected to build weekly counts

    weekly_data = []
    current_week = start_dt

    while current_week <= end_dt:
        week_end = current_week + timedelta(days=6)
        week_str = current_week.strftime("%Y-%m-%d")

        week_record = {
            'week_start': week_str,
            'week_end': week_end.strftime("%Y-%m-%d"),
        }

        # For each collection, count products in this week based on dates we collected
        for coll_id, coll_info in all_data.items():
            count = 0
            for month in coll_info['monthly']:
                for date_str in month.get('dates', []):
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        if current_week <= dt <= week_end:
                            count += 1
                    except:
                        pass

            # Use short column names
            col_prefix = coll_id.split('-')[1]  # e.g., '2', '1', '5p', '3'
            if '5p' in coll_id:
                if 'no2' in coll_id:
                    week_record['s5p_no2_count'] = count
                elif 'co' in coll_id:
                    week_record['s5p_co_count'] = count
            elif '3' in coll_id:
                week_record['s3_fire_count'] = count
            elif '2' in coll_id:
                week_record['s2_optical_count'] = count
            elif '1' in coll_id:
                week_record['s1_radar_count'] = count

        weekly_data.append(week_record)
        current_week += timedelta(days=7)

    # Save weekly data
    weekly_path = os.path.join(OUTPUT_DIR, "sentinel_weekly.json")
    with open(weekly_path, 'w') as f:
        json.dump(weekly_data, f, indent=2)
    print(f"Weekly data saved: {weekly_path}")

    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    print(f"{'Collection':<30} {'Total Products':>15} {'Months':>10}")
    print("-" * 60)
    for coll_id, info in all_data.items():
        months_with_data = sum(1 for m in info['monthly'] if m['count'] > 0)
        print(f"{info['name']:<30} {info['total_products']:>15,} {months_with_data:>10}")

    print(f"\nWeekly records: {len(weekly_data)}")
    print(f"Date range: {weekly_data[0]['week_start']} to {weekly_data[-1]['week_end']}")


if __name__ == "__main__":
    main()
