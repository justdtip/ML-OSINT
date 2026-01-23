#!/usr/bin/env python3
"""
Command-line interface for Telegram scraper.

Usage:
    python -m telegram_scraper.cli --channel @channel_name --limit 1000
    python -m telegram_scraper.cli --channels channels.txt --limit 500
    python -m telegram_scraper.cli --setup
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .config import TelegramConfig, print_setup_instructions
from .pipeline import IngestionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Telegram OSINT Scraper - Extract and attribute text-image pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape a single channel
  python -m telegram_scraper.cli --channel @war_monitor --limit 1000

  # Scrape multiple channels from a file
  python -m telegram_scraper.cli --channels channels.txt --limit 500

  # Scrape last 30 days only
  python -m telegram_scraper.cli --channel @war_monitor --days 30

  # Show setup instructions
  python -m telegram_scraper.cli --setup
        """
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help="Show setup instructions for Telegram API"
    )

    parser.add_argument(
        "--channel",
        type=str,
        help="Single channel to scrape (e.g., @channel_name)"
    )

    parser.add_argument(
        "--channels",
        type=str,
        help="File containing list of channels (one per line)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum messages to fetch per channel (default: 1000)"
    )

    parser.add_argument(
        "--days",
        type=int,
        help="Only fetch messages from last N days"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/telegram",
        help="Output directory (default: ./data/telegram)"
    )

    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Skip downloading media files"
    )

    parser.add_argument(
        "--session",
        type=str,
        default="telegram_osint",
        help="Session name for Telegram client"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    if args.setup:
        print_setup_instructions()
        return 0

    if not args.channel and not args.channels:
        print("Error: Must specify --channel or --channels")
        print("Use --setup for configuration instructions")
        return 1

    # Load configuration
    try:
        config = TelegramConfig.from_env()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nRun with --setup for instructions")
        return 1

    # Override config with CLI args
    config.output_dir = Path(args.output)
    config.session_name = args.session
    config.download_media = not args.no_media

    # Determine channels to scrape
    channels = []
    if args.channel:
        channels.append(args.channel)
    if args.channels:
        with open(args.channels, 'r') as f:
            channels.extend([
                line.strip() for line in f
                if line.strip() and not line.startswith('#')
            ])

    if not channels:
        print("Error: No channels specified")
        return 1

    # Determine date range
    min_date = None
    if args.days:
        min_date = datetime.now() - timedelta(days=args.days)

    # Create pipeline
    pipeline = IngestionPipeline(
        api_id=config.api_id,
        api_hash=config.api_hash,
        output_dir=str(config.output_dir),
        session_name=config.session_name,
        download_media=config.download_media
    )

    # Run ingestion
    print(f"Starting ingestion for {len(channels)} channel(s)")
    print(f"Limit: {args.limit} messages per channel")
    if min_date:
        print(f"Date range: {min_date.strftime('%Y-%m-%d')} to present")
    print(f"Output: {config.output_dir}")
    print()

    if len(channels) == 1:
        await pipeline.ingest_channel(
            channel_name=channels[0],
            limit=args.limit,
            min_date=min_date
        )
    else:
        await pipeline.ingest_multiple_channels(
            channel_names=channels,
            limit_per_channel=args.limit,
            min_date=min_date
        )

    print("\nIngestion complete!")
    return 0


def run():
    """Entry point for CLI."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
