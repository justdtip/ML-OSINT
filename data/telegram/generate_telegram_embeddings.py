#!/usr/bin/env python3
"""
Telegram Multimodal Embedding Generator

Generates embeddings for Telegram messages using Voyage AI's voyage-multimodal-3.5
model. This embeds text + images together into a unified 1024-dim vector space.

Key features:
- Interleaved text + image embedding (not separate like CLIP)
- Video support via frame extraction
- Checkpoint support to avoid reprocessing
- Per-channel and aggregated output

Usage:
    python generate_telegram_embeddings.py [--channel CHANNEL] [--batch-size N]
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import voyageai
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('telegram_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TELEGRAM_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "embeddings"

# Voyage API limits for multimodal
MAX_INPUTS_PER_BATCH = 20  # Conservative batch size for multimodal
MAX_TOKENS_PER_BATCH = 320000  # Max tokens across all inputs
PIXELS_PER_TOKEN = 560  # 560 pixels = 1 token


def load_api_key() -> str:
    """Load Voyage API key from environment or .env file."""
    if 'VOYAGE_API_KEY' in os.environ:
        return os.environ['VOYAGE_API_KEY']

    # Try project root .env
    for env_path in [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent.parent.parent / '.env',
    ]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('VOYAGE_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"')
                        os.environ['VOYAGE_API_KEY'] = key
                        return key

    raise ValueError("VOYAGE_API_KEY not found in environment or .env file")


def extract_video_frame(video_path: Path, frame_position: float = 0.5) -> Optional[Image.Image]:
    """Extract a single frame from video for embedding.

    Args:
        video_path: Path to video file
        frame_position: Position in video (0.0 = start, 1.0 = end)

    Returns:
        PIL Image of extracted frame, or None if extraction fails
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(total_frames * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    except ImportError:
        logger.warning("OpenCV not available, skipping video frame extraction")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract frame from {video_path}: {e}")
        return None


def resize_image_for_tokens(img: Image.Image, max_tokens: int = 5000) -> Image.Image:
    """Resize image to fit within token budget.

    Args:
        img: PIL Image
        max_tokens: Maximum tokens for this image (560 pixels = 1 token)

    Returns:
        Resized image
    """
    max_pixels = max_tokens * PIXELS_PER_TOKEN
    current_pixels = img.width * img.height

    if current_pixels <= max_pixels:
        return img

    # Calculate scale factor
    scale = (max_pixels / current_pixels) ** 0.5
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


class TelegramEmbeddingGenerator:
    """Generates multimodal embeddings for Telegram messages."""

    def __init__(
        self,
        telegram_dir: Path,
        output_dir: Path,
        model: str = "voyage-multimodal-3.5",
        batch_size: int = 10,
        include_videos: bool = True,
        text_only_fallback: bool = True
    ):
        self.telegram_dir = telegram_dir
        self.output_dir = output_dir
        self.model = model
        self.batch_size = min(batch_size, MAX_INPUTS_PER_BATCH)
        self.include_videos = include_videos
        self.text_only_fallback = text_only_fallback

        # Initialize Voyage client
        load_api_key()
        self.client = voyageai.Client()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata = {
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "channels": {},
            "total_messages": 0,
            "total_with_media": 0,
            "total_text_tokens": 0,
            "total_image_pixels": 0,
            "embedding_dim": 1024
        }

    def find_channel_files(self) -> dict[str, Path]:
        """Find all channel JSON files."""
        channels = {}

        for subdir in self.telegram_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Look for messages_*.json files
            json_files = list(subdir.glob("messages_*.json"))
            if json_files:
                # Use most recent
                latest = max(json_files, key=lambda p: p.stat().st_mtime)
                channels[subdir.name] = latest

        logger.info(f"Found {len(channels)} channels with message files")
        return channels

    def load_existing_embeddings(self, channel_name: str) -> dict[int, np.ndarray]:
        """Load existing embeddings for a channel."""
        existing = {}
        by_id_dir = self.output_dir / channel_name / "by_message_id"

        if by_id_dir.exists():
            for npy_file in by_id_dir.glob("*.npy"):
                try:
                    msg_id = int(npy_file.stem)
                    existing[msg_id] = np.load(npy_file)
                except (ValueError, Exception) as e:
                    logger.warning(f"Could not load {npy_file}: {e}")

        if existing:
            logger.info(f"Loaded {len(existing)} existing embeddings for {channel_name}")

        return existing

    def load_messages(self, json_path: Path, skip_ids: set[int] = None) -> list[dict]:
        """Load messages from a channel JSON file."""
        skip_ids = skip_ids or set()

        with open(json_path) as f:
            data = json.load(f)

        messages = data.get('messages', [])
        filtered = []

        for msg in messages:
            msg_id = msg.get('message_id')
            if msg_id in skip_ids:
                continue

            # Must have text or media
            text = msg.get('text') or ''
            text = text.strip() if text else ''
            has_media = msg.get('has_media', False)

            if not text and not has_media:
                continue

            filtered.append(msg)

        logger.info(f"Loaded {len(filtered)} messages (skipped {len(skip_ids)} existing)")
        return filtered

    def prepare_multimodal_input(self, msg: dict) -> list:
        """Prepare a single message as multimodal input.

        Returns list of [text, image] or [text] for voyage multimodal_embed.
        """
        components = []

        # Add text (use combined_text which includes attributions)
        text = msg.get('text') or msg.get('combined_text') or ''
        if text:
            # Clean up markdown links but keep text
            import re
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000] + "..."
            components.append(text)

        # Add image if available
        media_type = msg.get('media_type', 'none')
        media_path = msg.get('media_path')

        if media_path:
            full_path = self.telegram_dir.parent.parent / media_path

            # Infer media type from extension if not specified
            if media_type == 'none' and full_path.exists():
                ext = full_path.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                    media_type = 'photo'
                elif ext in ['.mp4', '.mov', '.avi', '.webm']:
                    media_type = 'video'

            if media_type == 'photo' and full_path.exists():
                try:
                    img = Image.open(full_path)
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize to fit token budget
                    img = resize_image_for_tokens(img, max_tokens=3000)
                    components.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {full_path}: {e}")

            elif media_type == 'video' and self.include_videos and full_path.exists():
                # Extract frame from video
                frame = extract_video_frame(full_path)
                if frame:
                    frame = resize_image_for_tokens(frame, max_tokens=3000)
                    components.append(frame)

        # Fallback to text-only if no components
        if not components:
            return None

        return components

    def generate_embeddings(self, inputs: list[list]) -> tuple[list[list[float]], int, int]:
        """Generate multimodal embeddings for a batch of inputs."""
        result = self.client.multimodal_embed(
            inputs,
            model=self.model,
            input_type="document",
            truncation=True
        )
        return result.embeddings, result.text_tokens, result.image_pixels

    def process_channel(self, channel_name: str, json_path: Path) -> dict[int, np.ndarray]:
        """Process all messages from a single channel."""
        # Create channel output directory
        channel_output = self.output_dir / channel_name
        channel_output.mkdir(exist_ok=True)
        (channel_output / "by_message_id").mkdir(exist_ok=True)

        # Load existing embeddings
        existing = self.load_existing_embeddings(channel_name)

        # Load messages
        messages = self.load_messages(json_path, skip_ids=set(existing.keys()))

        if not messages:
            logger.info(f"No new messages to process for {channel_name}")
            return existing

        # Prepare inputs
        prepared = []
        msg_ids = []

        for msg in messages:
            multimodal_input = self.prepare_multimodal_input(msg)
            if multimodal_input:
                prepared.append(multimodal_input)
                msg_ids.append(msg['message_id'])

        if not prepared:
            logger.info(f"No valid inputs for {channel_name}")
            return existing

        logger.info(f"Processing {len(prepared)} messages for {channel_name}")

        # Process in batches
        new_embeddings = {}
        total_text_tokens = 0
        total_image_pixels = 0

        batches = [
            (prepared[i:i + self.batch_size], msg_ids[i:i + self.batch_size])
            for i in range(0, len(prepared), self.batch_size)
        ]

        for batch_inputs, batch_ids in tqdm(batches, desc=f"Embedding {channel_name}"):
            try:
                embeddings, text_tokens, image_pixels = self.generate_embeddings(batch_inputs)
                total_text_tokens += text_tokens
                total_image_pixels += image_pixels

                for msg_id, emb in zip(batch_ids, embeddings):
                    emb_array = np.array(emb, dtype=np.float32)
                    new_embeddings[msg_id] = emb_array

                    # Save individual embedding
                    np.save(channel_output / "by_message_id" / f"{msg_id}.npy", emb_array)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Try individual items
                for inp, msg_id in zip(batch_inputs, batch_ids):
                    try:
                        embs, tt, ip = self.generate_embeddings([inp])
                        total_text_tokens += tt
                        total_image_pixels += ip

                        emb_array = np.array(embs[0], dtype=np.float32)
                        new_embeddings[msg_id] = emb_array
                        np.save(channel_output / "by_message_id" / f"{msg_id}.npy", emb_array)
                        time.sleep(0.5)
                    except Exception as e2:
                        logger.error(f"Failed to embed message {msg_id}: {e2}")

        # Update metadata
        self.metadata['total_text_tokens'] += total_text_tokens
        self.metadata['total_image_pixels'] += total_image_pixels

        # Merge with existing
        all_embeddings = {**existing, **new_embeddings}

        # Save consolidated files
        self.save_channel_embeddings(channel_name, all_embeddings, channel_output)

        return all_embeddings

    def save_channel_embeddings(
        self,
        channel_name: str,
        embeddings: dict[int, np.ndarray],
        output_dir: Path
    ):
        """Save consolidated embedding files for a channel."""
        if not embeddings:
            return

        # Sort by message ID
        sorted_ids = sorted(embeddings.keys())

        # Save as NPZ
        npz_data = {str(msg_id): embeddings[msg_id] for msg_id in sorted_ids}
        np.savez_compressed(output_dir / "embeddings.npz", **npz_data)

        # Save as matrix with index
        matrix = np.stack([embeddings[msg_id] for msg_id in sorted_ids])
        np.save(output_dir / "embedding_matrix.npy", matrix)

        with open(output_dir / "message_index.json", 'w') as f:
            json.dump({
                "message_ids": sorted_ids,
                "shape": list(matrix.shape),
                "channel": channel_name
            }, f, indent=2)

        logger.info(f"Saved {len(embeddings)} embeddings for {channel_name}")

        # Update channel metadata
        self.metadata['channels'][channel_name] = {
            "total_embeddings": len(embeddings),
            "message_id_range": [min(sorted_ids), max(sorted_ids)]
        }

    def run(self, channels: list[str] = None):
        """Run the full embedding generation pipeline."""
        logger.info(f"Starting multimodal embedding generation with model: {self.model}")

        # Find all channel files
        channel_files = self.find_channel_files()

        if not channel_files:
            logger.error("No channel files found")
            return

        # Filter to specified channels if provided
        if channels:
            channel_files = {k: v for k, v in channel_files.items() if k in channels}

        logger.info(f"Processing {len(channel_files)} channels: {list(channel_files.keys())}")

        all_embeddings = {}

        for channel_name, json_path in channel_files.items():
            try:
                embeddings = self.process_channel(channel_name, json_path)
                all_embeddings[channel_name] = embeddings
                self.metadata['total_messages'] += len(embeddings)
            except Exception as e:
                logger.error(f"Failed to process channel {channel_name}: {e}")

        # Save global metadata
        self.metadata['completed_at'] = datetime.now().isoformat()
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("MULTIMODAL EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Channels processed: {len(all_embeddings)}")
        print(f"Total messages embedded: {self.metadata['total_messages']}")
        print(f"Total text tokens: {self.metadata['total_text_tokens']:,}")
        print(f"Total image pixels: {self.metadata['total_image_pixels']:,}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate multimodal embeddings for Telegram messages"
    )
    parser.add_argument(
        "--telegram-dir",
        type=Path,
        default=DEFAULT_TELEGRAM_DIR,
        help="Directory containing Telegram channel folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="voyage-multimodal-3.5",
        help="Voyage multimodal model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of messages to process per API call"
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Specific channels to process (default: all)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip video frame extraction"
    )

    args = parser.parse_args()

    generator = TelegramEmbeddingGenerator(
        telegram_dir=args.telegram_dir,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        include_videos=not args.no_videos
    )

    generator.run(channels=args.channels)


if __name__ == "__main__":
    main()
