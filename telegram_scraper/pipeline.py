"""
Full ingestion pipeline for Telegram channels.

Orchestrates:
1. Message scraping
2. Text-image attribution
3. Media download
4. Data export
"""

import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict

from .models import ExtractedMessage, ScrapedChannel
from .scraper import TelegramScraper
from .attribution import TextImageAttributor


class IngestionPipeline:
    """
    Full pipeline for ingesting Telegram channel data.

    Usage:
        pipeline = IngestionPipeline(api_id, api_hash, output_dir="./data/telegram")
        results = await pipeline.ingest_channel("channel_name", limit=1000)
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        output_dir: str = "./data/telegram",
        session_name: str = "telegram_pipeline",
        download_media: bool = True,
        temporal_window_seconds: int = 60
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self.download_media = download_media

        # Create base directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.attributor = TextImageAttributor(
            temporal_window_seconds=temporal_window_seconds
        )

    def _get_channel_dir(self, channel_name: str) -> Path:
        """Get or create directory for a specific channel."""
        channel_safe = channel_name.replace('@', '').replace('/', '_')
        channel_dir = self.output_dir / channel_safe
        channel_dir.mkdir(parents=True, exist_ok=True)

        # Create media subdirectory for this channel
        media_dir = channel_dir / "media"
        media_dir.mkdir(exist_ok=True)

        return channel_dir

    async def ingest_channel(
        self,
        channel_name: str,
        limit: int = 1000,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Ingest messages from a single channel.

        Saves messages incrementally to JSONL file as they're fetched.

        Returns:
            Dictionary with messages, metadata, and statistics
        """
        # Create channel-specific directory
        channel_dir = self._get_channel_dir(channel_name)
        media_dir = channel_dir / "media"

        scraper = TelegramScraper(
            api_id=self.api_id,
            api_hash=self.api_hash,
            session_name=self.session_name,
            media_dir=str(media_dir)
        )

        # Prepare output files in channel directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_file = channel_dir / f"messages_{timestamp}.jsonl"
        json_file = channel_dir / f"messages_{timestamp}.json"

        messages = []
        channel_meta = None

        async with scraper:
            # Get channel entity for reply resolution
            try:
                channel = await scraper.client.get_entity(channel_name)
            except Exception as e:
                print(f"Error getting channel {channel_name}: {e}")
                raise

            # Scrape messages with incremental save
            print(f"Saving messages to: {jsonl_file}")
            messages, channel_meta = await self._scrape_with_incremental_save(
                scraper=scraper,
                channel=channel,
                channel_name=channel_name,
                limit=limit,
                min_date=min_date,
                max_date=max_date,
                jsonl_file=jsonl_file
            )

            # Apply attribution strategies
            if messages:
                print("Applying text-image attribution...")
                messages = await self.attributor.attribute_all(
                    messages,
                    client=scraper.client,
                    channel=channel
                )

        # Get attribution stats
        stats = self.attributor.get_attribution_stats(messages) if messages else {}

        # Export final results
        result = {
            "channel": channel_meta.to_dict() if channel_meta else {},
            "messages": [m.to_dict() for m in messages],
            "attribution_stats": stats,
            "ingestion_timestamp": datetime.now().isoformat()
        }

        # Save final JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {json_file}")
        print(f"Raw messages (JSONL): {jsonl_file}")
        if stats:
            self._print_stats(stats)

        return result

    async def _scrape_with_incremental_save(
        self,
        scraper: TelegramScraper,
        channel,
        channel_name: str,
        limit: int,
        min_date: Optional[datetime],
        max_date: Optional[datetime],
        jsonl_file: Path
    ) -> tuple[List[ExtractedMessage], Optional[ScrapedChannel]]:
        """Scrape messages and save each one immediately to JSONL."""
        from telethon.errors import FloodWaitError

        messages = []
        raw_messages = {}
        channel_id = channel.id if hasattr(channel, 'id') else 0
        channel_title = channel.title if hasattr(channel, 'title') else channel_name

        print(f"Scraping channel: {channel_title} ({channel_name})")

        with open(jsonl_file, 'w', encoding='utf-8') as f:
            try:
                async for message in scraper.client.iter_messages(
                    channel,
                    limit=limit,
                    offset_date=max_date,
                    reverse=False
                ):
                    # Check date bounds
                    if min_date and message.date.replace(tzinfo=None) < min_date:
                        continue

                    extracted = await scraper._extract_message(message, channel_name)
                    messages.append(extracted)
                    raw_messages[message.id] = message

                    # Save immediately to JSONL
                    f.write(json.dumps(extracted.to_dict(), ensure_ascii=False) + '\n')
                    f.flush()  # Ensure it's written to disk

                    # Download media if requested
                    if self.download_media and extracted.has_media:
                        try:
                            path, hash_val = await scraper._download_media(message)
                            extracted.media_path = path
                            extracted.media_hash = hash_val
                        except Exception as e:
                            print(f"  Error downloading media for message {extracted.message_id}: {e}")

                    # Rate limiting
                    await asyncio.sleep(scraper.rate_limit_delay)

                    if len(messages) % 50 == 0:
                        print(f"  Fetched {len(messages)} messages...")

            except FloodWaitError as e:
                print(f"Rate limited. Waiting {e.seconds} seconds...")
                await asyncio.sleep(e.seconds)

        # Create channel metadata
        date_range_start = min(m.timestamp for m in messages) if messages else None
        date_range_end = max(m.timestamp for m in messages) if messages else None

        channel_meta = ScrapedChannel(
            channel_id=channel_id,
            channel_name=channel_name,
            channel_title=channel_title,
            scrape_timestamp=datetime.now(),
            message_count=len(messages),
            media_count=len([m for m in messages if m.has_media]),
            date_range_start=date_range_start,
            date_range_end=date_range_end
        )

        print(f"Scraped {len(messages)} messages ({channel_meta.media_count} with media)")
        return messages, channel_meta

    async def ingest_multiple_channels(
        self,
        channel_names: List[str],
        limit_per_channel: int = 500,
        min_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest messages from multiple channels.

        Includes cross-channel deduplication based on media hashes.
        """
        all_results = []
        seen_media_hashes = set()

        for channel_name in channel_names:
            print(f"\n{'='*60}")
            print(f"Processing channel: {channel_name}")
            print('='*60)

            try:
                result = await self.ingest_channel(
                    channel_name=channel_name,
                    limit=limit_per_channel,
                    min_date=min_date
                )

                # Track media hashes for cross-channel dedup
                for msg in result["messages"]:
                    if msg.get("media_hash"):
                        if msg["media_hash"] in seen_media_hashes:
                            msg["is_duplicate"] = True
                        else:
                            seen_media_hashes.add(msg["media_hash"])
                            msg["is_duplicate"] = False

                all_results.append(result)

            except Exception as e:
                print(f"Error processing {channel_name}: {e}")
                continue

            # Delay between channels to avoid rate limiting
            await asyncio.sleep(2)

        # Save combined results
        combined_file = self.output_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\nCombined results saved to: {combined_file}")
        return all_results

    def _print_stats(self, stats: Dict[str, int]) -> None:
        """Print attribution statistics."""
        print("\n" + "="*60)
        print("ATTRIBUTION STATISTICS")
        print("="*60)
        print(f"Total messages:        {stats['total_messages']:,}")
        print(f"Messages with media:   {stats['total_with_media']:,}")
        print(f"Media with attribution:{stats['media_with_attribution']:,}")
        print(f"  - Direct caption:    {stats['direct_caption']:,}")
        print(f"  - Album group:       {stats['album_group']:,}")
        print(f"  - Reply chain:       {stats['reply_chain']:,}")
        print(f"  - Temporal proximity:{stats['temporal_proximity']:,}")
        print(f"Media without text:    {stats['no_attribution']:,}")

        if stats['total_with_media'] > 0:
            coverage = stats['media_with_attribution'] / stats['total_with_media'] * 100
            print(f"\nAttribution coverage:  {coverage:.1f}%")


def compute_event_signature(extraction: dict) -> str:
    """
    Create a signature for deduplication.
    Events with same signature within time window are likely duplicates.
    """
    components = []

    # Location (most important)
    location = extraction.get('location', {})
    if location.get('coordinates'):
        lat, lon = location['coordinates']
        components.append(f"{lat:.2f},{lon:.2f}")
    elif location.get('placename'):
        components.append(location['placename'].lower())

    # Equipment
    equipment = extraction.get('equipment', {})
    if equipment.get('model'):
        components.append(equipment['model'].lower())
    elif equipment.get('type'):
        components.append(equipment['type'])

    # Event type
    if extraction.get('event_type'):
        components.append(extraction['event_type'])

    return '|'.join(components) if components else hashlib.md5(
        json.dumps(extraction, sort_keys=True).encode()
    ).hexdigest()[:12]


def deduplicate_events(
    observations: List[dict],
    time_window_hours: int = 6
) -> List[dict]:
    """
    Group observations by signature within time window.

    Returns canonical events with corroboration counts.
    """
    from collections import defaultdict
    from datetime import timedelta

    # Sort by time
    observations.sort(key=lambda o: o['timestamp'])

    # Group by signature with time window
    groups = defaultdict(list)

    for obs in observations:
        sig = compute_event_signature(obs.get('extraction', {}))

        # Check for existing group within time window
        matched = False
        for existing_sig, group in groups.items():
            if existing_sig == sig and group:
                last_time = group[-1]['timestamp']
                if isinstance(last_time, str):
                    last_time = datetime.fromisoformat(last_time)
                obs_time = obs['timestamp']
                if isinstance(obs_time, str):
                    obs_time = datetime.fromisoformat(obs_time)

                if (obs_time - last_time) < timedelta(hours=time_window_hours):
                    group.append(obs)
                    matched = True
                    break

        if not matched:
            groups[sig].append(obs)

    # Merge groups into canonical events
    canonical = []
    for sig, group in groups.items():
        # Use highest-confidence observation as primary
        group.sort(key=lambda o: o.get('confidence', 0), reverse=True)
        primary = group[0].copy()
        primary['corroborating_sources'] = [o.get('message_id') for o in group[1:]]
        primary['corroboration_count'] = len(group)
        canonical.append(primary)

    return canonical


def format_for_gnn(messages: List[ExtractedMessage]) -> dict:
    """
    Format extracted data for GNN ingestion.

    Creates nodes and edges for a heterogeneous graph.
    """
    nodes = {
        "message": [],
        "media": [],
        "text_segment": []
    }

    edges = {
        "message_has_media": [],
        "message_has_text": [],
        "message_replies_to": [],
        "message_in_album": [],
        "message_forwards": []
    }

    for msg in messages:
        # Message node
        msg_node = {
            "id": f"msg_{msg.message_id}",
            "channel": msg.channel_name,
            "timestamp": msg.timestamp.isoformat(),
            "confidence": msg.confidence
        }
        nodes["message"].append(msg_node)

        # Media node
        if msg.has_media and msg.media_hash:
            media_node = {
                "id": f"media_{msg.media_hash}",
                "type": msg.media_type.value,
                "path": msg.media_path
            }
            nodes["media"].append(media_node)
            edges["message_has_media"].append({
                "source": msg_node["id"],
                "target": media_node["id"]
            })

        # Text segment nodes
        for i, text in enumerate(msg.associated_text):
            text_node = {
                "id": f"text_{msg.message_id}_{i}",
                "content": text[:500],  # Truncate for graph storage
                "source": msg.attribution_sources[i].value if i < len(msg.attribution_sources) else "unknown"
            }
            nodes["text_segment"].append(text_node)
            edges["message_has_text"].append({
                "source": msg_node["id"],
                "target": text_node["id"]
            })

        # Reply edges
        if msg.reply_to_id:
            edges["message_replies_to"].append({
                "source": msg_node["id"],
                "target": f"msg_{msg.reply_to_id}"
            })

        # Album edges
        if msg.grouped_id:
            edges["message_in_album"].append({
                "source": msg_node["id"],
                "target": f"album_{msg.grouped_id}"
            })

        # Forward edges
        if msg.forward_from_channel and msg.forward_from_message_id:
            edges["message_forwards"].append({
                "source": msg_node["id"],
                "target": f"msg_{msg.forward_from_channel}_{msg.forward_from_message_id}"
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "message_count": len(messages),
            "media_count": len(nodes["media"]),
            "text_segment_count": len(nodes["text_segment"])
        }
    }
