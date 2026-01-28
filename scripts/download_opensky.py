#!/usr/bin/env python3
"""
OpenSky Network Historical Data Downloader

Downloads ADS-B aircraft tracking data from OpenSky Network's Trino database.
Uses pyopensky library which handles authentication properly.

Requires:
- OpenSky account with data access approved
- pyopensky configured with credentials

First-time setup:
    python scripts/download_opensky.py --setup

Ukraine bounding box: lat 44-52.5, lon 22-40.5

Author: ML Engineering Team
Date: 2026-01-27
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "opensky"

# Ukraine bounding box (lon_min, lat_min, lon_max, lat_max for pyopensky)
UKRAINE_BOUNDS = (22.0, 44.0, 40.5, 52.5)  # (west, south, east, north)


def setup_credentials():
    """Set up pyopensky credentials from .env file."""
    from pyopensky.config import opensky_config_dir

    config_file = Path(opensky_config_dir) / "settings.conf"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Get credentials from environment
    username = os.getenv("OPENSKY_USERNAME") or os.getenv("clientId") or ""
    password = os.getenv("OPENSKY_PASSWORD") or os.getenv("clientSecret") or ""
    client_id = os.getenv("clientId") or ""
    client_secret = os.getenv("clientSecret") or ""

    config_content = f"""[default]
username = {username}
password = {password}
client_id = {client_id}
client_secret = {client_secret}
"""

    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"Credentials written to: {config_file}")
    print(f"Username: {username}")
    print(f"Client ID: {client_id[:10]}..." if client_id else "Client ID: (not set)")
    return config_file


def download_hour(
    trino,
    hour_start: datetime,
    bounds: tuple,
    output_file: Path,
) -> int:
    """
    Download state vectors for a single hour.

    Args:
        trino: pyopensky Trino instance
        hour_start: Start of the hour (UTC)
        bounds: (west, south, east, north) bounds tuple
        output_file: Output JSON file path

    Returns:
        Number of records downloaded
    """
    hour_end = hour_start + timedelta(hours=1)

    # Use history() method for state vectors within bounds
    df = trino.history(
        start=hour_start,
        stop=hour_end,
        bounds=bounds,
    )

    if df is None or len(df) == 0:
        # Save empty result
        with open(output_file, 'w') as f:
            json.dump({
                "hour_start": hour_start.isoformat(),
                "hour_end": hour_end.isoformat(),
                "bounds": bounds,
                "record_count": 0,
                "records": [],
            }, f)
        return 0

    # Convert to records
    records = df.to_dict(orient='records')

    # Convert timestamps to ISO format for JSON
    for r in records:
        for key in ['timestamp', 'last_position', 'hour']:
            if key in r and r[key] is not None:
                try:
                    if hasattr(r[key], 'isoformat'):
                        r[key] = r[key].isoformat()
                    elif isinstance(r[key], (int, float)):
                        r[key] = datetime.fromtimestamp(r[key], tz=timezone.utc).isoformat()
                except:
                    pass

    with open(output_file, 'w') as f:
        json.dump({
            "hour_start": hour_start.isoformat(),
            "hour_end": hour_end.isoformat(),
            "bounds": bounds,
            "record_count": len(records),
            "records": records,
        }, f, default=str)

    return len(records)


def download_date_range(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    bounds: tuple = UKRAINE_BOUNDS,
    delay: float = 2.0,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Download state vectors for a date range, hour by hour.
    """
    from pyopensky.trino import Trino

    output_dir.mkdir(parents=True, exist_ok=True)

    trino = Trino()

    stats = {
        "total_hours": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "total_records": 0,
    }

    # Generate hours
    current = start_date.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end = end_date.replace(tzinfo=timezone.utc)

    hours = []
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)

    print(f"Downloading OpenSky data from {start_date} to {end_date}")
    print(f"Total hours to process: {len(hours)}")
    print(f"Bounds: {bounds}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    import time

    for i, hour in enumerate(hours):
        stats["total_hours"] += 1
        date_str = hour.strftime("%Y-%m-%d_%H")
        output_file = output_dir / f"state_vectors_{date_str}.json"

        # Skip if exists
        if skip_existing and output_file.exists():
            stats["skipped"] += 1
            continue

        try:
            n_records = download_hour(trino, hour, bounds, output_file)
            stats["downloaded"] += 1
            stats["total_records"] += n_records
            print(f"  [{i+1}/{len(hours)}] {date_str}: {n_records} records")

        except Exception as e:
            stats["failed"] += 1
            print(f"  [{i+1}/{len(hours)}] {date_str}: FAILED - {e}")

        # Rate limiting
        time.sleep(delay)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download aircraft tracking data from OpenSky Network"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up pyopensky credentials from .env file",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD or YYYY-MM-DD_HH)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD or YYYY-MM-DD_HH). Default: today",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "hourly"),
        help="Output directory for hourly files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between queries in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection and query one hour only",
    )

    args = parser.parse_args()

    # Setup mode
    if args.setup:
        print("=" * 60)
        print("Setting up pyopensky credentials")
        print("=" * 60)
        setup_credentials()
        return

    # Require start date for download
    if not args.start_date:
        parser.error("--start-date is required (or use --setup)")

    # Parse dates
    try:
        if "_" in args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d_%H")
        else:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid start date format: {args.start_date}")
        return

    try:
        if "_" in args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d_%H")
        else:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            end_date = end_date.replace(hour=23)
    except ValueError:
        print(f"Invalid end date format: {args.end_date}")
        return

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("OpenSky Network Historical Data Downloader")
    print("=" * 60)

    # Test mode
    if args.test:
        print("\nTest mode: querying single hour...")
        from pyopensky.trino import Trino

        trino = Trino()
        test_hour = start_date.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        try:
            df = trino.history(
                start=test_hour,
                stop=test_hour + timedelta(hours=1),
                bounds=UKRAINE_BOUNDS,
            )
            if df is not None:
                print(f"Retrieved {len(df)} records")
                print(f"Columns: {list(df.columns)}")
                if len(df) > 0:
                    print(f"Sample:\n{df.head(1)}")
            else:
                print("No data returned (might be outside coverage or no flights)")
        except Exception as e:
            print(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
        return

    # Full download
    stats = download_date_range(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        delay=args.delay,
        skip_existing=not args.no_skip_existing,
    )

    print("-" * 60)
    print("Download complete!")
    print(f"  Total hours: {stats['total_hours']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total records: {stats['total_records']}")


if __name__ == "__main__":
    main()
