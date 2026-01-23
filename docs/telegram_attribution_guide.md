# Telegram Text-Image Attribution for Weapon-Location Analysis

## Problem Statement

Telegram messages with media attachments have a specific structure that must be understood to correctly associate text with images. When scraping via Telethon, the relationship between caption text, standalone text messages, and media can be lost if not handled carefully.

**Goal:** Build a pipeline that correctly attributes text to images, enabling downstream analysis of:
- Equipment type (from image classification + text extraction)
- Deployment location (from text geolocation + image EXIF/visual landmarks)
- Impact/outcome (from text sentiment + damage assessment in images)

---

## Telegram Message Structure

### Message Types

```
┌─────────────────────────────────────────────────────────────┐
│ Type 1: Text-only message                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.text = "Explosion reported in Kharkiv"          │ │
│ │ message.media = None                                    │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Type 2: Media with caption                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.text = "T-72B3 destroyed near Avdiivka"         │ │
│ │ message.media = MessageMediaPhoto(...)                  │ │
│ │ [IMAGE IS DIRECTLY ATTACHED TO THIS MESSAGE]            │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Type 3: Media without caption                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.text = "" or None                               │ │
│ │ message.media = MessageMediaPhoto(...)                  │ │
│ │ [CONTEXT MAY BE IN ADJACENT MESSAGES]                   │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Type 4: Grouped media (album)                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.grouped_id = 12345678901234                     │ │
│ │ [MULTIPLE MESSAGES SHARE SAME grouped_id]               │ │
│ │ [CAPTION ONLY ON FIRST OR LAST MESSAGE IN GROUP]        │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Type 5: Reply to media                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.text = "This is a Lancet strike"                │ │
│ │ message.reply_to = MessageReplyHeader(reply_to_msg_id)  │ │
│ │ [TEXT DESCRIBES MEDIA IN ANOTHER MESSAGE]               │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Type 6: Forward with added commentary                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ message.fwd_from = MessageFwdHeader(...)                │ │
│ │ [ORIGINAL CONTEXT FROM SOURCE CHANNEL]                  │ │
│ │ [FORWARDING CHANNEL MAY ADD CONTEXT BEFORE/AFTER]       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

**Telegram does NOT separate text and images into different messages when posted together.** If a user posts an image with a caption, both are in the same `Message` object. The attribution problem arises from:

1. **Albums** - Multiple images grouped, caption on only one
2. **Replies** - Text message replying to a media message
3. **Context messages** - Text posted immediately before/after media
4. **Forwards** - Original caption may be stripped or modified

---

## Telethon Data Extraction

### Basic Message Attributes

```python
from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto, 
    MessageMediaDocument,
    DocumentAttributeVideo,
    DocumentAttributeFilename
)
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List
import hashlib

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
    media_type: Optional[str] = None  # 'photo', 'video', 'document'
    media_path: Optional[str] = None  # Local path after download
    media_hash: Optional[str] = None  # For deduplication
    
    # Attribution links
    grouped_id: Optional[int] = None  # Album grouping
    reply_to_id: Optional[int] = None  # Reply chain
    forward_from_channel: Optional[str] = None
    forward_from_message_id: Optional[int] = None
    
    # Computed associations
    associated_text: List[str] = field(default_factory=list)
    confidence: float = 1.0


async def extract_message(client, message, channel_name: str) -> ExtractedMessage:
    """Extract normalized data from a Telegram message."""
    
    extracted = ExtractedMessage(
        message_id=message.id,
        channel_id=message.peer_id.channel_id if hasattr(message.peer_id, 'channel_id') else 0,
        channel_name=channel_name,
        timestamp=message.date,
        text=message.text or message.message,  # .message is the raw text
        grouped_id=message.grouped_id,
    )
    
    # Handle replies
    if message.reply_to:
        extracted.reply_to_id = message.reply_to.reply_to_msg_id
    
    # Handle forwards
    if message.fwd_from:
        if message.fwd_from.from_id:
            # Could be channel or user
            if hasattr(message.fwd_from.from_id, 'channel_id'):
                extracted.forward_from_channel = str(message.fwd_from.from_id.channel_id)
        extracted.forward_from_message_id = message.fwd_from.channel_post
    
    # Handle media
    if message.media:
        extracted.has_media = True
        
        if isinstance(message.media, MessageMediaPhoto):
            extracted.media_type = 'photo'
        elif isinstance(message.media, MessageMediaDocument):
            # Could be video, gif, or file
            doc = message.media.document
            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeVideo):
                    extracted.media_type = 'video'
                    break
            if not extracted.media_type:
                extracted.media_type = 'document'
    
    return extracted
```

### Downloading Media with Hash

```python
async def download_media_with_hash(client, message, output_dir: str) -> tuple[str, str]:
    """Download media and compute content hash for deduplication."""
    
    if not message.media:
        return None, None
    
    # Download to bytes first for hashing
    media_bytes = await client.download_media(message, bytes)
    
    if media_bytes is None:
        return None, None
    
    # Compute hash
    media_hash = hashlib.sha256(media_bytes).hexdigest()[:16]
    
    # Determine extension
    if isinstance(message.media, MessageMediaPhoto):
        ext = 'jpg'
    else:
        ext = 'mp4'  # Simplification; could inspect document attributes
    
    # Save with hash-based filename (automatic dedup)
    filename = f"{media_hash}.{ext}"
    filepath = f"{output_dir}/{filename}"
    
    with open(filepath, 'wb') as f:
        f.write(media_bytes)
    
    return filepath, media_hash
```

---

## Text-Image Attribution Strategies

### Strategy 1: Direct Caption (Highest Confidence)

```python
def attribute_direct_caption(extracted: ExtractedMessage) -> ExtractedMessage:
    """Text is directly attached to media as caption."""
    if extracted.has_media and extracted.text:
        extracted.associated_text.append(extracted.text)
        extracted.confidence = 1.0
    return extracted
```

### Strategy 2: Album Grouping

```python
async def attribute_album_group(
    client, 
    channel, 
    messages: List[ExtractedMessage]
) -> List[ExtractedMessage]:
    """
    Group messages by grouped_id and share text across album.
    
    Telegram albums: multiple photos share one grouped_id.
    Caption appears on ONE message (usually first or last).
    """
    
    # Group by grouped_id
    albums = {}
    standalone = []
    
    for msg in messages:
        if msg.grouped_id:
            if msg.grouped_id not in albums:
                albums[msg.grouped_id] = []
            albums[msg.grouped_id].append(msg)
        else:
            standalone.append(msg)
    
    # For each album, find the caption and propagate
    for grouped_id, album_msgs in albums.items():
        # Sort by message_id to get order
        album_msgs.sort(key=lambda m: m.message_id)
        
        # Find caption (usually on first or last)
        caption = None
        for msg in album_msgs:
            if msg.text:
                caption = msg.text
                break
        
        # Propagate to all media in album
        if caption:
            for msg in album_msgs:
                if msg.has_media:
                    msg.associated_text.append(caption)
                    msg.confidence = 0.95  # High but not certain
    
    return standalone + [m for album in albums.values() for m in album]
```

### Strategy 3: Reply Chain Resolution

```python
async def resolve_reply_chain(
    client, 
    channel,
    message: ExtractedMessage,
    message_cache: dict
) -> ExtractedMessage:
    """
    If a text message replies to a media message, associate them.
    Also handles media replying to text (less common but occurs).
    """
    
    if not message.reply_to_id:
        return message
    
    # Fetch the replied-to message
    if message.reply_to_id in message_cache:
        replied_msg = message_cache[message.reply_to_id]
    else:
        try:
            replied_msg = await client.get_messages(channel, ids=message.reply_to_id)
            replied_msg = await extract_message(client, replied_msg, channel.username)
            message_cache[message.reply_to_id] = replied_msg
        except Exception as e:
            print(f"Could not fetch reply target {message.reply_to_id}: {e}")
            return message
    
    # Case: Text message replying to media
    if message.text and not message.has_media:
        if replied_msg.has_media:
            replied_msg.associated_text.append(message.text)
            replied_msg.confidence = min(replied_msg.confidence, 0.85)
    
    # Case: Media replying to text (using text as context)
    if message.has_media and not message.text:
        if replied_msg.text:
            message.associated_text.append(replied_msg.text)
            message.confidence = min(message.confidence, 0.75)
    
    return message
```

### Strategy 4: Temporal Proximity (Lowest Confidence)

```python
def attribute_temporal_proximity(
    messages: List[ExtractedMessage],
    window_seconds: int = 60
) -> List[ExtractedMessage]:
    """
    Associate text with media posted within a time window.
    
    Use case: User posts text, then immediately posts image.
    This is LOWEST confidence - many false positives possible.
    """
    
    # Sort by timestamp
    messages.sort(key=lambda m: m.timestamp)
    
    for i, msg in enumerate(messages):
        if not msg.has_media:
            continue
        if msg.associated_text:  # Already has attribution
            continue
        
        # Look backwards for text
        for j in range(i - 1, max(0, i - 5) - 1, -1):
            prev = messages[j]
            time_diff = (msg.timestamp - prev.timestamp).total_seconds()
            
            if time_diff > window_seconds:
                break
            
            if prev.text and not prev.has_media:
                # Check it's not already used
                msg.associated_text.append(prev.text)
                msg.confidence = min(msg.confidence, 0.5)  # Low confidence
                break
        
        # Look forwards for text (image posted, then described)
        for j in range(i + 1, min(len(messages), i + 5)):
            next_msg = messages[j]
            time_diff = (next_msg.timestamp - msg.timestamp).total_seconds()
            
            if time_diff > window_seconds:
                break
            
            if next_msg.text and not next_msg.has_media:
                msg.associated_text.append(next_msg.text)
                msg.confidence = min(msg.confidence, 0.4)
                break
    
    return messages
```

---

## Full Ingestion Pipeline

```python
async def ingest_channel(
    client: TelegramClient,
    channel_name: str,
    output_dir: str,
    limit: int = 1000,
    min_date: datetime = None
) -> List[ExtractedMessage]:
    """
    Full pipeline: fetch, extract, attribute, download.
    """
    
    channel = await client.get_entity(channel_name)
    
    # Fetch messages
    messages = await client.get_messages(
        channel, 
        limit=limit,
        offset_date=min_date
    )
    
    # Extract normalized structure
    extracted = []
    for msg in messages:
        ext = await extract_message(client, msg, channel_name)
        extracted.append(ext)
    
    # Build message cache for reply resolution
    message_cache = {m.message_id: m for m in extracted}
    
    # Attribution passes (in order of confidence)
    
    # Pass 1: Direct captions
    for msg in extracted:
        attribute_direct_caption(msg)
    
    # Pass 2: Album grouping
    extracted = await attribute_album_group(client, channel, extracted)
    
    # Pass 3: Reply chains
    for msg in extracted:
        await resolve_reply_chain(client, channel, msg, message_cache)
    
    # Pass 4: Temporal proximity (only for unattributed media)
    extracted = attribute_temporal_proximity(extracted)
    
    # Download media for messages with attribution
    for msg in extracted:
        if msg.has_media and msg.associated_text:
            path, hash = await download_media_with_hash(client, messages_raw[msg.message_id], output_dir)
            msg.media_path = path
            msg.media_hash = hash
    
    return extracted
```

---

## LLM Processing for Entity Extraction

Once text is attributed to images, run LLM extraction on the combined context.

### Extraction Prompt Template

```python
EXTRACTION_PROMPT = """You are an OSINT analyst processing Telegram messages about the Ukraine conflict.

MESSAGE METADATA:
- Channel: {channel_name}
- Timestamp: {timestamp}
- Confidence of text-image association: {confidence}

ATTRIBUTED TEXT:
{combined_text}

IMAGE DESCRIPTION (from vision model):
{image_description}

Extract the following structured data. If a field cannot be determined, use null.

```json
{{
  "event_type": "strike|movement|equipment_loss|equipment_capture|fortification|claim|other",
  
  "equipment": {{
    "type": "tank|ifv|apc|artillery|mlrs|air_defense|aircraft|drone|truck|other",
    "model": "specific model if identifiable, e.g. T-72B3, M777, Leopard 2A6",
    "quantity": number or null,
    "status": "destroyed|damaged|captured|abandoned|operational"
  }},
  
  "location": {{
    "placename": "extracted location name in original language",
    "placename_en": "English transliteration/translation",
    "coordinates": [lat, lon] or null,
    "precision": "exact|approximate|regional",
    "oblast": "if determinable"
  }},
  
  "parties": {{
    "attacker": "UA|RU|unknown",
    "target": "UA|RU|unknown", 
    "unit_mentioned": "any military unit names"
  }},
  
  "munition": {{
    "type": "fpv_drone|lancet|artillery|glide_bomb|cruise_missile|other|unknown",
    "model": "specific if mentioned"
  }},
  
  "confidence_signals": {{
    "has_photo": boolean,
    "has_video": boolean,
    "has_geolocation_markers": boolean,
    "corroborated": boolean,
    "source_reliability": "high|medium|low|unknown"
  }},
  
  "raw_claims": ["list of specific claims made in text"]
}}
```

Respond ONLY with the JSON object, no other text.
"""
```

### Vision Model for Image Description

```python
async def describe_image_for_context(image_path: str, client) -> str:
    """
    Use vision model to generate description for LLM context.
    This provides the LLM with image content without multimodal input.
    """
    
    # Option 1: Claude vision
    # Option 2: GPT-4V
    # Option 3: Local model (LLaVA, etc.)
    
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": """Describe this image for military/OSINT analysis:
1. What equipment or vehicles are visible? Be specific about type/model if identifiable.
2. What is the condition/status (destroyed, damaged, operational)?
3. Are there any visible location markers (signs, landmarks, terrain)?
4. What is happening in the image (strike aftermath, movement, etc.)?
5. Any visible unit markings, flags, or identifying features?

Be factual and specific. Do not speculate beyond what is visible."""
                }
            ]
        }]
    )
    
    return response.content[0].text
```

---

## Data Model for Knowledge Graph

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Equipment:
    id: str  # hash of first observation
    type: str
    model: Optional[str]
    status: str
    first_seen: datetime
    last_seen: datetime
    observations: List[str]  # message IDs

@dataclass  
class Location:
    id: str
    placename: str
    coordinates: Optional[tuple]
    precision: str
    
@dataclass
class Event:
    id: str
    timestamp: datetime
    event_type: str
    location_id: str
    equipment_ids: List[str]
    source_messages: List[str]
    confidence: float
    
@dataclass
class Observation:
    """Links message -> event -> equipment -> location"""
    message_id: str
    channel: str
    timestamp: datetime
    event_id: Optional[str]
    equipment_id: Optional[str]
    location_id: Optional[str]
    raw_text: str
    image_hash: Optional[str]
    extraction_confidence: float
```

---

## Deduplication Strategy

The same event is often reported across multiple channels.

```python
def compute_event_signature(extraction: dict) -> str:
    """
    Create a signature for deduplication.
    Events with same signature within time window are likely duplicates.
    """
    
    components = []
    
    # Location (most important)
    if extraction.get('location', {}).get('coordinates'):
        # Round to ~1km precision
        lat, lon = extraction['location']['coordinates']
        components.append(f"{lat:.2f},{lon:.2f}")
    elif extraction.get('location', {}).get('placename'):
        components.append(extraction['location']['placename'].lower())
    
    # Equipment
    if extraction.get('equipment', {}).get('model'):
        components.append(extraction['equipment']['model'].lower())
    elif extraction.get('equipment', {}).get('type'):
        components.append(extraction['equipment']['type'])
    
    # Event type
    if extraction.get('event_type'):
        components.append(extraction['event_type'])
    
    return '|'.join(components)


def deduplicate_events(
    observations: List[dict],
    time_window_hours: int = 6
) -> List[dict]:
    """Group observations by signature within time window."""
    
    from collections import defaultdict
    
    # Sort by time
    observations.sort(key=lambda o: o['timestamp'])
    
    # Group by signature
    groups = defaultdict(list)
    
    for obs in observations:
        sig = compute_event_signature(obs['extraction'])
        
        # Check if we have a recent group with this signature
        matched = False
        for existing_sig, group in groups.items():
            if existing_sig == sig:
                # Check time window
                last_time = group[-1]['timestamp']
                if (obs['timestamp'] - last_time).total_seconds() < time_window_hours * 3600:
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
        primary = group[0]
        primary['corroborating_sources'] = [o['message_id'] for o in group[1:]]
        primary['corroboration_count'] = len(group)
        canonical.append(primary)
    
    return canonical
```

---

## Output Schema for GNN Ingestion

```python
def format_for_gnn(
    observations: List[Observation],
    events: List[Event],
    equipment: List[Equipment],
    locations: List[Location]
) -> dict:
    """
    Format extracted data for the heterogeneous GNN.
    
    Node types:
    - thermal_region (from FIRMS)
    - unit (from equipment tracking)
    - zone (from location clustering)
    - event (from this extraction pipeline)
    
    Edge types:
    - event_at_location
    - event_involves_equipment
    - equipment_observed_at
    - temporal_sequence
    """
    
    return {
        "nodes": {
            "event": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type,
                    "confidence": e.confidence,
                    "features": [...]  # Embedding from LLM
                }
                for e in events
            ],
            "equipment": [...],
            "location": [...]
        },
        "edges": {
            "event_at_location": [
                {"source": e.id, "target": e.location_id}
                for e in events if e.location_id
            ],
            "event_involves_equipment": [
                {"source": e.id, "target": eq_id}
                for e in events for eq_id in e.equipment_ids
            ],
            # ... more edge types
        }
    }
```

---

## Exploration Tasks for Claude Code

1. **Verify Telethon message structure**
   - Fetch sample messages from a public channel
   - Print full message object to understand actual attribute names
   - Confirm grouped_id behavior with albums

2. **Test attribution strategies**
   - Find examples of each message type (album, reply, etc.)
   - Measure false positive rate of temporal proximity method

3. **Benchmark LLM extraction**
   - Test extraction prompt on sample messages
   - Measure entity extraction accuracy against manual labels

4. **Image hashing for deduplication**
   - Test perceptual hashing (imagehash library) vs. content hashing
   - Same image re-encoded may have different SHA but same perceptual hash

5. **Geolocation pipeline**
   - Test Nominatim for Ukrainian placenames
   - Consider fallback to Google Geocoding API for better Cyrillic handling

6. **Rate limiting strategy**
   - Determine safe request rate for Telegram API
   - Implement exponential backoff

---

## Dependencies

```
telethon>=1.28.0
anthropic>=0.18.0  # For Claude API
imagehash>=4.3.0   # Perceptual hashing
langdetect>=1.0.9  # Language detection
httpx>=0.24.0      # Async HTTP for geocoding
```

---

## Security Notes

1. **API credentials** - Store Telegram API ID/hash and LLM API keys in environment variables, not code.

2. **Data retention** - Some channels may contain graphic content. Consider filtering or flagging.

3. **Rate limiting** - Aggressive scraping can get your Telegram account banned.

4. **Legal** - Ensure compliance with Telegram ToS and local laws regarding data collection.
