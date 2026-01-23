#!/usr/bin/env python3
"""
Example usage of the Telegram scraper module.

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Set environment variables:
   export TELEGRAM_API_ID="your_api_id"
   export TELEGRAM_API_HASH="your_api_hash"
3. Get credentials from: https://my.telegram.org/apps
"""

import asyncio
import os
from datetime import datetime, timedelta

# Import from local module
from telegram_scraper import TelegramScraper, TextImageAttributor, IngestionPipeline


async def basic_scrape_example():
    """Basic example: scrape a single channel."""

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]

    # Example public channel (replace with actual target)
    channel = "@telegram"  # Official Telegram channel

    scraper = TelegramScraper(
        api_id=api_id,
        api_hash=api_hash,
        media_dir="./data/telegram/media"
    )

    async with scraper:
        messages, metadata = await scraper.scrape_channel(
            channel_name=channel,
            limit=100,
            download_media=False
        )

        print(f"\nScraped {len(messages)} messages from {channel}")
        print(f"Messages with media: {sum(1 for m in messages if m.has_media)}")

        # Show sample messages
        for msg in messages[:5]:
            print(f"\n[{msg.timestamp}] ID: {msg.message_id}")
            if msg.text:
                print(f"  Text: {msg.text[:100]}...")
            if msg.has_media:
                print(f"  Media: {msg.media_type.value}")
            if msg.grouped_id:
                print(f"  Album: {msg.grouped_id}")
            if msg.reply_to_id:
                print(f"  Reply to: {msg.reply_to_id}")


async def attribution_example():
    """Example: scrape and apply text-image attribution."""

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]

    channel = "@telegram"

    scraper = TelegramScraper(api_id=api_id, api_hash=api_hash)
    attributor = TextImageAttributor()

    async with scraper:
        messages, _ = await scraper.scrape_channel(
            channel_name=channel,
            limit=200,
            download_media=False
        )

        # Get channel entity for reply resolution
        channel_entity = await scraper.client.get_entity(channel)

        # Apply attribution
        messages = await attributor.attribute_all(
            messages,
            client=scraper.client,
            channel=channel_entity
        )

        # Print statistics
        stats = attributor.get_attribution_stats(messages)
        print("\nAttribution Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Show messages with attribution
        print("\nMessages with text-image attribution:")
        for msg in messages:
            if msg.has_media and msg.associated_text:
                print(f"\n[{msg.message_id}] Media: {msg.media_type.value}")
                print(f"  Attribution: {[s.value for s in msg.attribution_sources]}")
                print(f"  Confidence: {msg.confidence}")
                for text in msg.associated_text:
                    print(f"  Text: {text[:100]}...")


async def pipeline_example():
    """Example: use the full ingestion pipeline."""

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]

    pipeline = IngestionPipeline(
        api_id=api_id,
        api_hash=api_hash,
        output_dir="./data/telegram",
        download_media=True
    )

    # Scrape last 7 days
    min_date = datetime.now() - timedelta(days=7)

    result = await pipeline.ingest_channel(
        channel_name="@telegram",
        limit=100,
        min_date=min_date
    )

    print(f"\nSaved {len(result['messages'])} messages")
    print(f"Channel: {result['channel']['channel_title']}")


async def multi_channel_example():
    """Example: scrape multiple channels with deduplication."""

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]

    # Example channels - replace with actual OSINT targets
    channels = [
        "@telegram",
        "@durov",  # Example channels
    ]

    pipeline = IngestionPipeline(
        api_id=api_id,
        api_hash=api_hash,
        output_dir="./data/telegram"
    )

    results = await pipeline.ingest_multiple_channels(
        channel_names=channels,
        limit_per_channel=50
    )

    total_messages = sum(len(r['messages']) for r in results)
    print(f"\nTotal messages across all channels: {total_messages}")


if __name__ == "__main__":
    # Check for credentials
    if not os.environ.get("TELEGRAM_API_ID"):
        print("Error: TELEGRAM_API_ID not set")
        print("Get credentials from: https://my.telegram.org/apps")
        print("\nSet environment variables:")
        print('  export TELEGRAM_API_ID="your_api_id"')
        print('  export TELEGRAM_API_HASH="your_api_hash"')
        exit(1)

    # Run example
    print("Running basic scrape example...")
    asyncio.run(basic_scrape_example())
