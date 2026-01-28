#!/usr/bin/env python3
"""
WarSpotting API Downloader

Downloads Russian equipment loss data from the WarSpotting API.
API endpoint: https://ukr.warspotting.net/api/losses/russia/YYYY-MM-DD

Each record contains:
- id: Unique identifier
- type: Equipment category (Tank, IFV, Artillery, etc.)
- model: Specific equipment model
- status: Destroyed/Damaged/Abandoned/Captured
- lost_by: Nation
- date: ISO date
- nearest_location: Geographic location name (often includes raion)
- geo: "lat,lon" coordinates
- unit: Military unit (often null)
- tags: Additional attributes

Author: ML Engineering Team
Date: 2026-01-27
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "warspotting"
OUTPUT_DIR = DATA_DIR / "daily"


def fetch_losses_for_date(date: datetime, retries: int = 3, delay: float = 1.0) -> Optional[Dict]:
    """
    Fetch losses for a specific date from the WarSpotting API.

    Args:
        date: Date to fetch
        retries: Number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        JSON response dict or None if failed
    """
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://ukr.warspotting.net/api/losses/russia/{date_str}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (research project)",
                    "Accept": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # No data for this date (might be before war or future)
                return {"losses": []}
            print(f"  HTTP {e.code} for {date_str}, attempt {attempt + 1}/{retries}")
        except urllib.error.URLError as e:
            print(f"  URL error for {date_str}: {e.reason}, attempt {attempt + 1}/{retries}")
        except Exception as e:
            print(f"  Error for {date_str}: {e}, attempt {attempt + 1}/{retries}")

        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))

    return None


def download_date_range(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    delay: float = 0.5,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Download losses for a range of dates.

    Args:
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        output_dir: Directory to save daily JSON files
        delay: Delay between requests in seconds
        skip_existing: Skip dates that already have files

    Returns:
        Stats dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_days": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "total_losses": 0,
    }

    current = start_date
    total_days = (end_date - start_date).days + 1

    print(f"Downloading WarSpotting data from {start_date.date()} to {end_date.date()}")
    print(f"Total days to process: {total_days}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    while current <= end_date:
        stats["total_days"] += 1
        date_str = current.strftime("%Y-%m-%d")
        output_file = output_dir / f"losses_{date_str}.json"

        # Skip if exists
        if skip_existing and output_file.exists():
            stats["skipped"] += 1
            current += timedelta(days=1)
            continue

        # Fetch data
        data = fetch_losses_for_date(current)

        if data is not None:
            # Save to file
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            n_losses = len(data.get("losses", []))
            stats["downloaded"] += 1
            stats["total_losses"] += n_losses

            # Progress output every 10 days or if there are losses
            if stats["downloaded"] % 10 == 0 or n_losses > 0:
                print(f"  {date_str}: {n_losses} losses (total downloaded: {stats['downloaded']})")
        else:
            stats["failed"] += 1
            print(f"  {date_str}: FAILED")

        # Rate limiting
        time.sleep(delay)
        current += timedelta(days=1)

    return stats


def aggregate_to_single_file(
    daily_dir: Path,
    output_file: Path,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """
    Aggregate daily JSON files into a single consolidated file.

    Args:
        daily_dir: Directory with daily JSON files
        output_file: Output path for aggregated file
        start_date: Optional filter start
        end_date: Optional filter end

    Returns:
        Total number of records
    """
    all_losses = []

    for json_file in sorted(daily_dir.glob("losses_*.json")):
        # Extract date from filename
        date_str = json_file.stem.replace("losses_", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Apply date filters
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue

        with open(json_file) as f:
            data = json.load(f)
            losses = data.get("losses", [])
            all_losses.extend(losses)

    # Sort by date and id
    all_losses.sort(key=lambda x: (x.get("date", ""), x.get("id", 0)))

    # Save aggregated file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"losses": all_losses, "count": len(all_losses)}, f, indent=2)

    print(f"Aggregated {len(all_losses)} records to {output_file}")
    return len(all_losses)


def main():
    parser = argparse.ArgumentParser(
        description="Download Russian equipment losses from WarSpotting API"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-06-09",  # Day after existing data ends
        help="Start date (YYYY-MM-DD). Default: 2025-06-09",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for daily JSON files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate all daily files into single JSON after download",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing files, don't download",
    )
    parser.add_argument(
        "--full-war",
        action="store_true",
        help="Download full war period (2022-02-24 to today)",
    )

    args = parser.parse_args()

    # Parse dates
    if args.full_war:
        start_date = datetime(2022, 2, 24)
    else:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    output_dir = Path(args.output_dir)

    if not args.aggregate_only:
        # Download
        print("=" * 60)
        print("WarSpotting API Downloader")
        print("=" * 60)

        stats = download_date_range(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            delay=args.delay,
            skip_existing=not args.no_skip_existing,
        )

        print("-" * 60)
        print("Download complete!")
        print(f"  Total days: {stats['total_days']}")
        print(f"  Downloaded: {stats['downloaded']}")
        print(f"  Skipped (existing): {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total losses: {stats['total_losses']}")

    # Aggregate if requested
    if args.aggregate or args.aggregate_only:
        print("\n" + "=" * 60)
        print("Aggregating daily files...")
        print("=" * 60)

        aggregate_file = DATA_DIR / "warspotting_losses_api.json"
        aggregate_to_single_file(output_dir, aggregate_file)


if __name__ == "__main__":
    main()
