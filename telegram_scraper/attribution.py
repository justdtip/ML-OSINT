"""
Text-image attribution strategies for Telegram messages.

Handles:
- Direct captions (highest confidence)
- Album grouping (high confidence)
- Reply chain resolution (medium confidence)
- Temporal proximity (lowest confidence)
"""

from typing import List, Dict, Optional
from datetime import timedelta
import asyncio

from .models import ExtractedMessage, AttributionSource


class TextImageAttributor:
    """
    Associates text with images in Telegram messages using multiple strategies.

    The strategies are applied in order of confidence:
    1. Direct caption - text attached to the same message as media
    2. Album grouping - messages sharing the same grouped_id
    3. Reply chain - text message replying to media or vice versa
    4. Temporal proximity - messages posted within a time window
    """

    def __init__(
        self,
        temporal_window_seconds: int = 60,
        max_lookback_messages: int = 5
    ):
        self.temporal_window_seconds = temporal_window_seconds
        self.max_lookback_messages = max_lookback_messages

    async def attribute_all(
        self,
        messages: List[ExtractedMessage],
        client=None,
        channel=None
    ) -> List[ExtractedMessage]:
        """
        Apply all attribution strategies in order of confidence.

        Args:
            messages: List of extracted messages
            client: Optional Telethon client for fetching reply targets
            channel: Optional channel entity for fetching reply targets

        Returns:
            Messages with associated_text populated
        """
        # Build message cache for lookups
        message_cache: Dict[int, ExtractedMessage] = {
            m.message_id: m for m in messages
        }

        # Pass 1: Direct captions (highest confidence)
        for msg in messages:
            self._attribute_direct_caption(msg)

        # Pass 2: Album grouping (high confidence)
        messages = self._attribute_album_groups(messages)

        # Pass 3: Reply chain resolution (medium confidence)
        if client and channel:
            for msg in messages:
                await self._resolve_reply_chain(msg, message_cache, client, channel)
        else:
            # Local-only reply resolution
            for msg in messages:
                self._resolve_reply_chain_local(msg, message_cache)

        # Pass 4: Temporal proximity (lowest confidence, only for unattributed media)
        messages = self._attribute_temporal_proximity(messages)

        return messages

    def _attribute_direct_caption(self, msg: ExtractedMessage) -> ExtractedMessage:
        """
        Text directly attached to media as caption.
        Confidence: 1.0
        """
        if msg.has_media and msg.text:
            if msg.text not in msg.associated_text:
                msg.associated_text.append(msg.text)
            msg.attribution_sources.append(AttributionSource.DIRECT_CAPTION)
            msg.confidence = 1.0
        return msg

    def _attribute_album_groups(
        self,
        messages: List[ExtractedMessage]
    ) -> List[ExtractedMessage]:
        """
        Group messages by grouped_id and share text across album.

        Telegram albums: multiple photos share one grouped_id.
        Caption appears on ONE message (usually first or last).
        Confidence: 0.95
        """
        # Separate albums from standalone messages
        albums: Dict[int, List[ExtractedMessage]] = {}
        standalone: List[ExtractedMessage] = []

        for msg in messages:
            if msg.grouped_id:
                if msg.grouped_id not in albums:
                    albums[msg.grouped_id] = []
                albums[msg.grouped_id].append(msg)
            else:
                standalone.append(msg)

        # Process each album
        for grouped_id, album_msgs in albums.items():
            # Sort by message_id to get order
            album_msgs.sort(key=lambda m: m.message_id)

            # Find caption (check first and last, then all)
            caption = None
            caption_source = None

            # Check first message
            if album_msgs[0].text:
                caption = album_msgs[0].text
                caption_source = album_msgs[0]
            # Check last message
            elif album_msgs[-1].text:
                caption = album_msgs[-1].text
                caption_source = album_msgs[-1]
            # Check all others
            else:
                for msg in album_msgs:
                    if msg.text:
                        caption = msg.text
                        caption_source = msg
                        break

            # Propagate caption to all media in album
            if caption:
                for msg in album_msgs:
                    if msg.has_media and msg != caption_source:
                        if caption not in msg.associated_text:
                            msg.associated_text.append(caption)
                        if AttributionSource.ALBUM_GROUP not in msg.attribution_sources:
                            msg.attribution_sources.append(AttributionSource.ALBUM_GROUP)
                        msg.confidence = min(msg.confidence, 0.95)

        # Return all messages
        return standalone + [m for album in albums.values() for m in album]

    async def _resolve_reply_chain(
        self,
        msg: ExtractedMessage,
        message_cache: Dict[int, ExtractedMessage],
        client,
        channel
    ) -> ExtractedMessage:
        """
        Resolve reply chains - fetch target message if not in cache.
        Confidence: 0.75-0.85
        """
        if not msg.reply_to_id:
            return msg

        # Try to get from cache first
        replied_msg = message_cache.get(msg.reply_to_id)

        if not replied_msg:
            # Fetch from Telegram
            try:
                from .scraper import TelegramScraper
                raw_msg = await client.get_messages(channel, ids=msg.reply_to_id)
                if raw_msg:
                    # Extract the message
                    channel_name = msg.channel_name
                    replied_msg = ExtractedMessage(
                        message_id=raw_msg.id,
                        channel_id=msg.channel_id,
                        channel_name=channel_name,
                        timestamp=raw_msg.date.replace(tzinfo=None) if raw_msg.date.tzinfo else raw_msg.date,
                        text=raw_msg.text or raw_msg.message,
                        has_media=raw_msg.media is not None
                    )
                    message_cache[msg.reply_to_id] = replied_msg
            except Exception as e:
                print(f"Could not fetch reply target {msg.reply_to_id}: {e}")
                return msg

        if replied_msg:
            self._apply_reply_attribution(msg, replied_msg)

        return msg

    def _resolve_reply_chain_local(
        self,
        msg: ExtractedMessage,
        message_cache: Dict[int, ExtractedMessage]
    ) -> ExtractedMessage:
        """
        Resolve reply chains using only local cache.
        Confidence: 0.75-0.85
        """
        if not msg.reply_to_id:
            return msg

        replied_msg = message_cache.get(msg.reply_to_id)
        if replied_msg:
            self._apply_reply_attribution(msg, replied_msg)

        return msg

    def _apply_reply_attribution(
        self,
        msg: ExtractedMessage,
        replied_msg: ExtractedMessage
    ) -> None:
        """Apply attribution based on reply relationship."""

        # Case 1: Text message replying to media
        # The reply text describes the media
        if msg.text and not msg.has_media and replied_msg.has_media:
            if msg.text not in replied_msg.associated_text:
                replied_msg.associated_text.append(msg.text)
            if AttributionSource.REPLY_CHAIN not in replied_msg.attribution_sources:
                replied_msg.attribution_sources.append(AttributionSource.REPLY_CHAIN)
            replied_msg.confidence = min(replied_msg.confidence, 0.85)

        # Case 2: Media replying to text
        # The text provides context for the media
        if msg.has_media and not msg.text and replied_msg.text:
            if replied_msg.text not in msg.associated_text:
                msg.associated_text.append(replied_msg.text)
            if AttributionSource.REPLY_CHAIN not in msg.attribution_sources:
                msg.attribution_sources.append(AttributionSource.REPLY_CHAIN)
            msg.confidence = min(msg.confidence, 0.75)

    def _attribute_temporal_proximity(
        self,
        messages: List[ExtractedMessage]
    ) -> List[ExtractedMessage]:
        """
        Associate text with media posted within a time window.

        Use case: User posts text, then immediately posts image.
        This is LOWEST confidence - many false positives possible.
        Only applied to media without existing attribution.
        Confidence: 0.4-0.5
        """
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)

        for i, msg in enumerate(messages):
            # Only process media without text
            if not msg.has_media:
                continue
            if msg.associated_text:  # Already has attribution
                continue

            window = timedelta(seconds=self.temporal_window_seconds)

            # Look backwards for text
            for j in range(i - 1, max(0, i - self.max_lookback_messages) - 1, -1):
                prev = messages[j]
                time_diff = msg.timestamp - prev.timestamp

                if time_diff > window:
                    break

                # Found text-only message
                if prev.text and not prev.has_media:
                    # Skip if this text belongs to an album
                    if prev.grouped_id:
                        continue

                    msg.associated_text.append(prev.text)
                    msg.attribution_sources.append(AttributionSource.TEMPORAL_PROXIMITY)
                    msg.confidence = min(msg.confidence, 0.5)
                    break

            # If still no attribution, look forwards
            if not msg.associated_text:
                for j in range(i + 1, min(len(messages), i + self.max_lookback_messages)):
                    next_msg = messages[j]
                    time_diff = next_msg.timestamp - msg.timestamp

                    if time_diff > window:
                        break

                    if next_msg.text and not next_msg.has_media:
                        if next_msg.grouped_id:
                            continue

                        msg.associated_text.append(next_msg.text)
                        msg.attribution_sources.append(AttributionSource.TEMPORAL_PROXIMITY)
                        msg.confidence = min(msg.confidence, 0.4)
                        break

        return messages

    def get_attribution_stats(
        self,
        messages: List[ExtractedMessage]
    ) -> Dict[str, int]:
        """Get statistics on attribution sources."""
        stats = {
            "total_messages": len(messages),
            "total_with_media": 0,
            "media_with_attribution": 0,
            "direct_caption": 0,
            "album_group": 0,
            "reply_chain": 0,
            "temporal_proximity": 0,
            "no_attribution": 0
        }

        for msg in messages:
            if msg.has_media:
                stats["total_with_media"] += 1

                if msg.associated_text:
                    stats["media_with_attribution"] += 1

                    for source in msg.attribution_sources:
                        if source == AttributionSource.DIRECT_CAPTION:
                            stats["direct_caption"] += 1
                        elif source == AttributionSource.ALBUM_GROUP:
                            stats["album_group"] += 1
                        elif source == AttributionSource.REPLY_CHAIN:
                            stats["reply_chain"] += 1
                        elif source == AttributionSource.TEMPORAL_PROXIMITY:
                            stats["temporal_proximity"] += 1
                else:
                    stats["no_attribution"] += 1

        return stats
