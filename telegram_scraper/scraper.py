"""
Telegram channel scraper using Telethon.

Handles:
- Message fetching with pagination
- Media type detection
- Rate limiting and error handling
"""

import asyncio
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto,
    MessageMediaDocument,
    DocumentAttributeVideo,
    DocumentAttributeFilename,
    DocumentAttributeAudio,
    DocumentAttributeAnimated
)
from telethon.errors import FloodWaitError, ChannelPrivateError

from .models import ExtractedMessage, ScrapedChannel, MediaType


class TelegramScraper:
    """
    Scrapes messages from Telegram channels using Telethon.

    Usage:
        scraper = TelegramScraper(api_id, api_hash, session_name)
        async with scraper:
            messages = await scraper.scrape_channel("channel_name", limit=1000)
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str = "telegram_scraper",
        media_dir: str = "./media",
        rate_limit_delay: float = 0.5
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.media_dir = Path(media_dir)
        self.rate_limit_delay = rate_limit_delay
        self.client: Optional[TelegramClient] = None

        # Create media directory
        self.media_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Start the Telegram client."""
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        await self.client.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Disconnect the client."""
        if self.client:
            await self.client.disconnect()

    async def scrape_channel(
        self,
        channel_name: str,
        limit: int = 1000,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        download_media: bool = False
    ) -> tuple[List[ExtractedMessage], ScrapedChannel]:
        """
        Scrape messages from a Telegram channel.

        Args:
            channel_name: Channel username or ID
            limit: Maximum messages to fetch
            min_date: Only fetch messages after this date
            max_date: Only fetch messages before this date
            download_media: Whether to download media files

        Returns:
            Tuple of (list of ExtractedMessage, ScrapedChannel metadata)
        """
        if not self.client:
            raise RuntimeError("Client not started. Use 'async with' context manager.")

        try:
            channel = await self.client.get_entity(channel_name)
        except ChannelPrivateError:
            raise ValueError(f"Channel {channel_name} is private or does not exist")

        channel_id = channel.id if hasattr(channel, 'id') else 0
        channel_title = channel.title if hasattr(channel, 'title') else channel_name

        print(f"Scraping channel: {channel_title} ({channel_name})")

        # Fetch messages with pagination
        messages = []
        raw_messages = {}
        offset_date = max_date

        try:
            async for message in self.client.iter_messages(
                channel,
                limit=limit,
                offset_date=offset_date,
                reverse=False  # Newest first
            ):
                # Check date bounds
                if min_date and message.date.replace(tzinfo=None) < min_date:
                    continue

                extracted = await self._extract_message(message, channel_name)
                messages.append(extracted)
                raw_messages[message.id] = message

                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)

                if len(messages) % 100 == 0:
                    print(f"  Fetched {len(messages)} messages...")

        except FloodWaitError as e:
            print(f"Rate limited. Waiting {e.seconds} seconds...")
            await asyncio.sleep(e.seconds)
            # Could retry here, but for now just return what we have

        # Download media if requested
        if download_media:
            print(f"Downloading media for {len([m for m in messages if m.has_media])} messages...")
            for msg in messages:
                if msg.has_media and msg.message_id in raw_messages:
                    try:
                        path, hash_val = await self._download_media(raw_messages[msg.message_id])
                        msg.media_path = path
                        msg.media_hash = hash_val
                    except Exception as e:
                        print(f"  Error downloading media for message {msg.message_id}: {e}")

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

    async def _extract_message(self, message, channel_name: str) -> ExtractedMessage:
        """Extract normalized data from a Telegram message."""

        # Get channel ID
        channel_id = 0
        if hasattr(message, 'peer_id'):
            if hasattr(message.peer_id, 'channel_id'):
                channel_id = message.peer_id.channel_id
            elif hasattr(message.peer_id, 'chat_id'):
                channel_id = message.peer_id.chat_id

        # Get text content
        text = message.text or message.message or None

        # Determine media type
        has_media = message.media is not None
        media_type = MediaType.NONE

        if message.media:
            if isinstance(message.media, MessageMediaPhoto):
                media_type = MediaType.PHOTO
            elif isinstance(message.media, MessageMediaDocument):
                doc = message.media.document
                if doc and doc.attributes:
                    for attr in doc.attributes:
                        if isinstance(attr, DocumentAttributeVideo):
                            media_type = MediaType.VIDEO
                            break
                        elif isinstance(attr, DocumentAttributeAnimated):
                            media_type = MediaType.VIDEO  # GIFs
                            break
                if media_type == MediaType.NONE:
                    media_type = MediaType.DOCUMENT

        # Handle timestamp timezone
        timestamp = message.date
        if timestamp.tzinfo:
            timestamp = timestamp.replace(tzinfo=None)

        extracted = ExtractedMessage(
            message_id=message.id,
            channel_id=channel_id,
            channel_name=channel_name,
            timestamp=timestamp,
            text=text,
            has_media=has_media,
            media_type=media_type,
            grouped_id=message.grouped_id
        )

        # Handle replies
        if message.reply_to:
            extracted.reply_to_id = message.reply_to.reply_to_msg_id

        # Handle forwards
        if message.fwd_from:
            if message.fwd_from.from_id:
                if hasattr(message.fwd_from.from_id, 'channel_id'):
                    extracted.forward_from_channel = str(message.fwd_from.from_id.channel_id)
                elif hasattr(message.fwd_from.from_id, 'user_id'):
                    extracted.forward_from_channel = f"user_{message.fwd_from.from_id.user_id}"
            if message.fwd_from.channel_post:
                extracted.forward_from_message_id = message.fwd_from.channel_post

        return extracted

    async def _download_media(self, message) -> tuple[Optional[str], Optional[str]]:
        """Download media and compute content hash for deduplication."""

        if not message.media:
            return None, None

        # Download to bytes first for hashing
        try:
            media_bytes = await self.client.download_media(message, bytes)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None, None

        if media_bytes is None:
            return None, None

        # Compute hash
        media_hash = hashlib.sha256(media_bytes).hexdigest()[:16]

        # Determine extension
        if isinstance(message.media, MessageMediaPhoto):
            ext = 'jpg'
        elif isinstance(message.media, MessageMediaDocument):
            # Try to get extension from filename attribute
            ext = 'bin'
            if message.media.document and message.media.document.attributes:
                for attr in message.media.document.attributes:
                    if isinstance(attr, DocumentAttributeFilename):
                        ext = attr.file_name.split('.')[-1] if '.' in attr.file_name else 'bin'
                        break
                    elif isinstance(attr, DocumentAttributeVideo):
                        ext = 'mp4'
                        break
        else:
            ext = 'bin'

        # Save with hash-based filename (automatic dedup)
        filename = f"{media_hash}.{ext}"
        filepath = self.media_dir / filename

        # Skip if already exists (deduplication)
        if not filepath.exists():
            with open(filepath, 'wb') as f:
                f.write(media_bytes)

        return str(filepath), media_hash

    async def get_message_by_id(self, channel_name: str, message_id: int) -> Optional[ExtractedMessage]:
        """Fetch a specific message by ID."""
        if not self.client:
            raise RuntimeError("Client not started")

        try:
            channel = await self.client.get_entity(channel_name)
            message = await self.client.get_messages(channel, ids=message_id)
            if message:
                return await self._extract_message(message, channel_name)
        except Exception as e:
            print(f"Error fetching message {message_id}: {e}")

        return None

    async def search_messages(
        self,
        channel_name: str,
        query: str,
        limit: int = 100
    ) -> List[ExtractedMessage]:
        """Search for messages containing specific text."""
        if not self.client:
            raise RuntimeError("Client not started")

        channel = await self.client.get_entity(channel_name)
        messages = []

        async for message in self.client.iter_messages(channel, search=query, limit=limit):
            extracted = await self._extract_message(message, channel_name)
            messages.append(extracted)

        return messages
