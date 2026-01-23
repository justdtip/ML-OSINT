"""
Data models for Telegram message extraction and attribution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class MediaType(Enum):
    PHOTO = "photo"
    VIDEO = "video"
    DOCUMENT = "document"
    NONE = "none"


class AttributionSource(Enum):
    """How text was associated with media."""
    DIRECT_CAPTION = "direct_caption"      # Text attached to media (confidence: 1.0)
    ALBUM_GROUP = "album_group"            # Shared grouped_id (confidence: 0.95)
    REPLY_CHAIN = "reply_chain"            # Reply relationship (confidence: 0.75-0.85)
    TEMPORAL_PROXIMITY = "temporal_proximity"  # Time-based (confidence: 0.4-0.5)
    NONE = "none"


@dataclass
class ExtractedMessage:
    """Normalized message structure for downstream processing."""
    message_id: int
    channel_id: int
    channel_name: str
    timestamp: datetime

    # Text content
    text: Optional[str] = None

    # Media
    has_media: bool = False
    media_type: MediaType = MediaType.NONE
    media_path: Optional[str] = None
    media_hash: Optional[str] = None  # SHA256 for deduplication
    media_perceptual_hash: Optional[str] = None  # For image similarity

    # Attribution links
    grouped_id: Optional[int] = None  # Album grouping
    reply_to_id: Optional[int] = None  # Reply chain
    forward_from_channel: Optional[str] = None
    forward_from_message_id: Optional[int] = None

    # Computed associations
    associated_text: List[str] = field(default_factory=list)
    attribution_sources: List[AttributionSource] = field(default_factory=list)
    confidence: float = 1.0

    # Raw message data for debugging
    raw_message: Optional[dict] = None

    def get_combined_text(self) -> str:
        """Get all associated text combined."""
        texts = []
        if self.text:
            texts.append(self.text)
        texts.extend(self.associated_text)
        return "\n\n".join(texts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "message_id": self.message_id,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "has_media": self.has_media,
            "media_type": self.media_type.value,
            "media_path": self.media_path,
            "media_hash": self.media_hash,
            "media_perceptual_hash": self.media_perceptual_hash,
            "grouped_id": self.grouped_id,
            "reply_to_id": self.reply_to_id,
            "forward_from_channel": self.forward_from_channel,
            "forward_from_message_id": self.forward_from_message_id,
            "associated_text": self.associated_text,
            "attribution_sources": [s.value for s in self.attribution_sources],
            "confidence": self.confidence,
            "combined_text": self.get_combined_text()
        }


@dataclass
class ScrapedChannel:
    """Metadata about a scraped channel."""
    channel_id: int
    channel_name: str
    channel_title: str
    scrape_timestamp: datetime
    message_count: int
    media_count: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "channel_title": self.channel_title,
            "scrape_timestamp": self.scrape_timestamp.isoformat(),
            "message_count": self.message_count,
            "media_count": self.media_count,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None
        }


@dataclass
class Equipment:
    """Extracted equipment entity."""
    id: str  # hash of first observation
    type: str
    model: Optional[str]
    status: str
    first_seen: datetime
    last_seen: datetime
    observations: List[str] = field(default_factory=list)  # message IDs


@dataclass
class Location:
    """Extracted location entity."""
    id: str
    placename: str
    placename_en: Optional[str] = None
    coordinates: Optional[tuple] = None
    precision: str = "unknown"  # exact, approximate, regional
    oblast: Optional[str] = None


@dataclass
class Event:
    """Canonical conflict event."""
    id: str
    timestamp: datetime
    event_type: str  # strike, movement, equipment_loss, etc.
    location_id: Optional[str] = None
    equipment_ids: List[str] = field(default_factory=list)
    source_messages: List[str] = field(default_factory=list)
    confidence: float = 1.0
    corroboration_count: int = 1


@dataclass
class Observation:
    """Links message -> event -> equipment -> location."""
    message_id: str
    channel: str
    timestamp: datetime
    event_id: Optional[str] = None
    equipment_id: Optional[str] = None
    location_id: Optional[str] = None
    raw_text: str = ""
    image_hash: Optional[str] = None
    extraction_confidence: float = 1.0
