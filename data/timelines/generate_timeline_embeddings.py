#!/usr/bin/env python3
"""
Timeline Embedding Generator

Generates embeddings for Ukraine war timeline events (battles, offensives)
and aligns them with ISW daily assessments to create temporal anchors.

Usage:
    python generate_timeline_embeddings.py [--batch-size N]
"""

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import voyageai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('timeline_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TIMELINE_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "embeddings"
ISW_EMBEDDING_DIR = Path(__file__).parent.parent / "wayback_archives" / "isw_assessments" / "embeddings"


def load_api_key() -> str:
    """Load Voyage API key from environment or .env file."""
    if 'VOYAGE_API_KEY' in os.environ:
        return os.environ['VOYAGE_API_KEY']

    for env_path in [
        Path(__file__).parent.parent.parent / '.env',
        Path(__file__).parent.parent.parent.parent / '.env',
    ]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith('VOYAGE_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"')
                        os.environ['VOYAGE_API_KEY'] = key
                        return key

    raise ValueError("VOYAGE_API_KEY not found in environment or .env file")


def parse_date_raw(date_raw: str) -> str:
    """Parse various date formats into ISO format."""
    if not date_raw:
        return None

    # Clean up common wiki artifacts
    date_raw = re.sub(r'\{\{[^}]*\}\}', '', date_raw)
    date_raw = re.sub(r'<[^>]+>', '', date_raw)
    date_raw = date_raw.strip()

    # Try various patterns
    patterns = [
        (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
         lambda m: f"{m.group(3)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"),
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})',
         lambda m: f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}-01"),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, date_raw, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except (KeyError, ValueError):
                continue

    return None


MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}


class TimelineEmbeddingGenerator:
    """Generates embeddings for timeline events."""

    def __init__(
        self,
        timeline_dir: Path,
        output_dir: Path,
        model: str = "voyage-4-large",
        batch_size: int = 20
    ):
        self.timeline_dir = timeline_dir
        self.output_dir = output_dir
        self.model = model
        self.batch_size = batch_size

        # Initialize Voyage client
        load_api_key()
        self.client = voyageai.Client()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        self.metadata = {
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "embedding_dim": None,
            "total_events": 0,
            "total_engagements": 0,
            "total_tokens": 0
        }

    def load_engagements(self) -> list[dict]:
        """Load military engagements from JSON."""
        engagements_file = self.timeline_dir / "wikipedia_engagements.json"

        if not engagements_file.exists():
            logger.error(f"Engagements file not found: {engagements_file}")
            return []

        with open(engagements_file) as f:
            engagements = json.load(f)

        # Clean and enhance data
        cleaned = []
        for eng in engagements:
            # Skip non-battle entries
            if not eng.get('name') or eng['name'] in ['Russia', 'Ukraine', 'Naval']:
                continue

            # Parse date
            date_raw = eng.get('date_raw', '')
            date_parsed = parse_date_raw(date_raw)

            # Clean location
            location = eng.get('location', '')
            location = re.sub(r'\[\[', '', location)
            location = re.sub(r'\]\]', '', location)
            location = location.strip()

            # Clean result
            result = eng.get('result', '')
            result = re.sub(r'<[^>]+>', '', result)
            result = re.sub(r'\{\{[^}]+\}\}', '', result)
            result = result.strip()

            # Create text for embedding
            text_parts = [eng['name']]
            if location:
                text_parts.append(f"Location: {location}")
            if result:
                text_parts.append(f"Result: {result}")
            if date_raw:
                text_parts.append(f"Date: {date_raw}")

            cleaned.append({
                'id': eng.get('wiki_title', eng['name']).replace(' ', '_'),
                'name': eng['name'],
                'date': date_parsed,
                'date_raw': date_raw,
                'location': location,
                'result': result,
                'text': '. '.join(text_parts),
                'type': 'military_engagement',
                'source': eng.get('source', 'Wikipedia')
            })

        logger.info(f"Loaded {len(cleaned)} military engagements")
        return cleaned

    def load_timeline_events(self) -> list[dict]:
        """Load timeline events from JSON."""
        events_file = self.timeline_dir / "wikipedia_timeline_events.json"

        if not events_file.exists():
            logger.warning(f"Timeline events file not found: {events_file}")
            return []

        with open(events_file) as f:
            events = json.load(f)

        logger.info(f"Loaded {len(events)} timeline events")
        return events

    def generate_embeddings(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Generate embeddings for a batch of texts."""
        result = self.client.embed(
            texts,
            model=self.model,
            input_type="document",
            truncation=True
        )
        return result.embeddings, result.total_tokens

    def embed_items(self, items: list[dict]) -> dict[str, np.ndarray]:
        """Generate embeddings for all items."""
        embeddings = {}
        total_tokens = 0

        # Prepare texts
        texts = [item['text'] for item in items]
        ids = [item['id'] for item in items]

        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]

            try:
                embs, tokens = self.generate_embeddings(batch_texts)
                total_tokens += tokens

                for item_id, emb in zip(batch_ids, embs):
                    embeddings[item_id] = np.array(emb, dtype=np.float32)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Try individual items
                for text, item_id in zip(batch_texts, batch_ids):
                    try:
                        emb, tokens = self.generate_embeddings([text])
                        total_tokens += tokens
                        embeddings[item_id] = np.array(emb[0], dtype=np.float32)
                        time.sleep(0.5)
                    except Exception as e2:
                        logger.error(f"Error processing {item_id}: {e2}")

        self.metadata['total_tokens'] = total_tokens
        return embeddings

    def save_embeddings(
        self,
        embeddings: dict[str, np.ndarray],
        items: list[dict],
        prefix: str = "timeline"
    ):
        """Save embeddings in multiple formats."""

        # Save individual embeddings
        individual_dir = self.output_dir / "by_id"
        individual_dir.mkdir(exist_ok=True)
        for item_id, emb in embeddings.items():
            np.save(individual_dir / f"{item_id}.npy", emb)

        # Save as NPZ
        npz_file = self.output_dir / f"{prefix}_embeddings.npz"
        np.savez_compressed(npz_file, **embeddings)
        logger.info(f"Saved embeddings to {npz_file}")

        # Save as matrix with index
        sorted_ids = sorted(embeddings.keys())
        if embeddings:
            matrix = np.stack([embeddings[i] for i in sorted_ids])
            np.save(self.output_dir / f"{prefix}_embedding_matrix.npy", matrix)

            self.metadata['embedding_dim'] = matrix.shape[1]

        # Save item metadata with embeddings index
        items_by_id = {item['id']: item for item in items}
        indexed_items = []
        for idx, item_id in enumerate(sorted_ids):
            item = items_by_id.get(item_id, {})
            item['embedding_index'] = idx
            indexed_items.append(item)

        with open(self.output_dir / f"{prefix}_index.json", 'w') as f:
            json.dump({
                "items": indexed_items,
                "ids": sorted_ids,
                "matrix_shape": list(matrix.shape) if embeddings else None
            }, f, indent=2, ensure_ascii=False)

        return sorted_ids

    def run(self):
        """Run the embedding generation pipeline."""
        logger.info(f"Starting timeline embedding generation with model: {self.model}")

        # Load data
        engagements = self.load_engagements()
        timeline_events = self.load_timeline_events()

        all_items = engagements + timeline_events
        self.metadata['total_engagements'] = len(engagements)
        self.metadata['total_events'] = len(timeline_events)

        if not all_items:
            logger.error("No items to embed")
            return

        # Generate embeddings
        embeddings = self.embed_items(all_items)

        if not embeddings:
            logger.error("No embeddings generated")
            return

        # Save embeddings
        self.save_embeddings(embeddings, all_items)

        # Save metadata
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("TIMELINE EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Military engagements: {self.metadata['total_engagements']}")
        print(f"Timeline events: {self.metadata['total_events']}")
        print(f"Total tokens: {self.metadata['total_tokens']:,}")
        print(f"Embedding dimension: {self.metadata['embedding_dim']}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


class ISWTimelineAligner:
    """Aligns timeline events with ISW daily assessments."""

    def __init__(
        self,
        timeline_embedding_dir: Path,
        isw_embedding_dir: Path,
        output_dir: Path
    ):
        self.timeline_embedding_dir = timeline_embedding_dir
        self.isw_embedding_dir = isw_embedding_dir
        self.output_dir = output_dir

    def load_timeline_embeddings(self) -> tuple[np.ndarray, list[dict]]:
        """Load timeline embeddings and metadata."""
        matrix = np.load(self.timeline_embedding_dir / "timeline_embedding_matrix.npy")

        with open(self.timeline_embedding_dir / "timeline_index.json") as f:
            index_data = json.load(f)

        return matrix, index_data['items']

    def load_isw_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Load ISW embeddings and date index."""
        matrix = np.load(self.isw_embedding_dir / "isw_embedding_matrix.npy")

        with open(self.isw_embedding_dir / "isw_date_index.json") as f:
            index_data = json.load(f)

        return matrix, index_data['dates']

    def compute_alignment(self, top_k: int = 5):
        """Compute alignment between timeline events and ISW assessments."""
        logger.info("Computing timeline-ISW alignment")

        # Load embeddings
        timeline_matrix, timeline_items = self.load_timeline_embeddings()
        isw_matrix, isw_dates = self.load_isw_embeddings()

        logger.info(f"Timeline: {timeline_matrix.shape}, ISW: {isw_matrix.shape}")

        # Normalize for cosine similarity
        timeline_norm = timeline_matrix / np.linalg.norm(timeline_matrix, axis=1, keepdims=True)
        isw_norm = isw_matrix / np.linalg.norm(isw_matrix, axis=1, keepdims=True)

        # Compute similarity matrix: timeline x isw
        similarity_matrix = timeline_norm @ isw_norm.T

        # For each timeline event, find most similar ISW assessments
        alignments = []
        for i, item in enumerate(timeline_items):
            sims = similarity_matrix[i]
            top_indices = np.argsort(sims)[-top_k:][::-1]

            matches = []
            for idx in top_indices:
                matches.append({
                    'date': isw_dates[idx],
                    'similarity': float(sims[idx])
                })

            alignments.append({
                'event_id': item.get('id', f'event_{i}'),
                'event_name': item.get('name', ''),
                'event_date': item.get('date'),
                'event_type': item.get('type'),
                'isw_matches': matches
            })

        # Save alignment results
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "timeline_isw_alignment.json", 'w') as f:
            json.dump({
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'timeline_count': len(timeline_items),
                    'isw_count': len(isw_dates),
                    'top_k': top_k
                },
                'alignments': alignments
            }, f, indent=2, ensure_ascii=False)

        # Also save the similarity matrix for further analysis
        np.save(self.output_dir / "similarity_matrix.npy", similarity_matrix)

        logger.info(f"Saved alignment results to {self.output_dir}")

        # Print sample alignments
        print("\n" + "="*60)
        print("SAMPLE TIMELINE-ISW ALIGNMENTS")
        print("="*60)
        for alignment in alignments[:5]:
            print(f"\n{alignment['event_name']} (date: {alignment['event_date']})")
            print(f"  Top ISW matches:")
            for match in alignment['isw_matches'][:3]:
                print(f"    {match['date']}: {match['similarity']:.4f}")
        print("="*60)

        return alignments


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for timeline events and align with ISW"
    )
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        default=DEFAULT_TIMELINE_DIR,
        help="Directory containing timeline JSON files"
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
        default="voyage-4-large",
        help="Voyage embedding model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for API calls"
    )
    parser.add_argument(
        "--skip-embed",
        action='store_true',
        help="Skip embedding generation (use existing)"
    )
    parser.add_argument(
        "--skip-align",
        action='store_true',
        help="Skip ISW alignment"
    )

    args = parser.parse_args()

    # Generate embeddings
    if not args.skip_embed:
        generator = TimelineEmbeddingGenerator(
            timeline_dir=args.timeline_dir,
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size
        )
        generator.run()

    # Align with ISW
    if not args.skip_align:
        if ISW_EMBEDDING_DIR.exists():
            aligner = ISWTimelineAligner(
                timeline_embedding_dir=args.output_dir,
                isw_embedding_dir=ISW_EMBEDDING_DIR,
                output_dir=args.output_dir
            )
            aligner.compute_alignment()
        else:
            logger.warning(f"ISW embeddings not found at {ISW_EMBEDDING_DIR}, skipping alignment")


if __name__ == "__main__":
    main()
