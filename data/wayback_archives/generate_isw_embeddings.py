#!/usr/bin/env python3
"""
ISW Assessment Embedding Generator

Generates embeddings for ISW daily assessment reports using Voyage AI's
voyage-4-large model. Embeddings are saved in a format compatible with
the Multi-Resolution HAN model.

Usage:
    python generate_isw_embeddings.py [--batch-size N] [--model MODEL]
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
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('isw_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TEXT_DIR = Path(__file__).parent / "isw_assessments" / "text"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "isw_assessments" / "embeddings"

# Voyage API limits
MAX_BATCH_SIZE = 128  # Max texts per request
MAX_TOKENS_PER_BATCH = 120000  # Max tokens per request for voyage-4-large


def load_api_key() -> str:
    """Load Voyage API key from environment or .env file."""
    if 'VOYAGE_API_KEY' in os.environ:
        return os.environ['VOYAGE_API_KEY']

    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith('VOYAGE_API_KEY='):
                    key = line.split('=', 1)[1].strip().strip('"')
                    os.environ['VOYAGE_API_KEY'] = key
                    return key

    raise ValueError("VOYAGE_API_KEY not found in environment or .env file")


class ISWEmbeddingGenerator:
    """Generates and manages embeddings for ISW assessment reports."""

    def __init__(
        self,
        text_dir: Path,
        output_dir: Path,
        model: str = "voyage-4-large",
        batch_size: int = 20
    ):
        self.text_dir = text_dir
        self.output_dir = output_dir
        self.model = model
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)

        # Initialize Voyage client
        load_api_key()
        self.client = voyageai.Client()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata = {
            "model": model,
            "generated_at": datetime.now().isoformat(),
            "total_reports": 0,
            "total_tokens": 0,
            "embedding_dim": None,
            "date_range": {"start": None, "end": None},
            "reports": {}
        }

    def load_existing_embeddings(self) -> dict[str, np.ndarray]:
        """Load any existing embeddings from the by_date directory."""
        existing = {}
        by_date_dir = self.output_dir / "by_date"

        if by_date_dir.exists():
            for npy_file in by_date_dir.glob("*.npy"):
                date_key = npy_file.stem
                try:
                    existing[date_key] = np.load(npy_file)
                except Exception as e:
                    logger.warning(f"Could not load existing embedding {npy_file}: {e}")

        if existing:
            logger.info(f"Loaded {len(existing)} existing embeddings (will skip these)")

        return existing

    def load_reports(self, skip_dates: set[str] = None) -> list[dict]:
        """Load all ISW reports from text directory.

        Args:
            skip_dates: Set of date strings to skip (already have embeddings)
        """
        skip_dates = skip_dates or set()
        reports = []
        skipped = 0

        for json_file in sorted(self.text_dir.glob("*.json")):
            date_key = json_file.stem

            # Skip if we already have an embedding for this date
            if date_key in skip_dates:
                skipped += 1
                continue

            try:
                with open(json_file) as f:
                    data = json.load(f)

                content = data.get('content', '').strip()
                if not content:
                    logger.warning(f"Empty content in {json_file.name}")
                    continue

                reports.append({
                    'date': data.get('date', json_file.stem),
                    'title': data.get('title', ''),
                    'content': content,
                    'file': json_file.name
                })
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        logger.info(f"Loaded {len(reports)} reports with content (skipped {skipped} already embedded)")
        return reports

    def generate_embeddings(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Generate embeddings for a batch of texts."""
        result = self.client.embed(
            texts,
            model=self.model,
            input_type="document",
            truncation=True
        )
        return result.embeddings, result.total_tokens

    def process_all(self, reports: list[dict]) -> dict[str, np.ndarray]:
        """Process all reports and generate embeddings."""
        all_embeddings = {}
        total_tokens = 0

        # Process in batches
        batches = [
            reports[i:i + self.batch_size]
            for i in range(0, len(reports), self.batch_size)
        ]

        logger.info(f"Processing {len(reports)} reports in {len(batches)} batches")

        for batch in tqdm(batches, desc="Generating embeddings"):
            texts = [r['content'] for r in batch]
            dates = [r['date'] for r in batch]

            try:
                embeddings, tokens = self.generate_embeddings(texts)
                total_tokens += tokens

                for date, emb, report in zip(dates, embeddings, batch):
                    all_embeddings[date] = np.array(emb, dtype=np.float32)
                    self.metadata['reports'][date] = {
                        'title': report['title'],
                        'content_length': len(report['content']),
                        'file': report['file']
                    }

                # Rate limiting - be respectful to the API
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Try individual items on batch failure
                for report in batch:
                    try:
                        emb, tokens = self.generate_embeddings([report['content']])
                        total_tokens += tokens
                        all_embeddings[report['date']] = np.array(emb[0], dtype=np.float32)
                        self.metadata['reports'][report['date']] = {
                            'title': report['title'],
                            'content_length': len(report['content']),
                            'file': report['file']
                        }
                        time.sleep(0.5)
                    except Exception as e2:
                        logger.error(f"Error processing {report['date']}: {e2}")

        self.metadata['total_tokens'] = total_tokens
        self.metadata['total_reports'] = len(all_embeddings)

        if all_embeddings:
            dates = sorted(all_embeddings.keys())
            self.metadata['date_range'] = {
                'start': dates[0],
                'end': dates[-1]
            }
            self.metadata['embedding_dim'] = len(next(iter(all_embeddings.values())))

        return all_embeddings

    def save_embeddings(self, embeddings: dict[str, np.ndarray]):
        """Save embeddings in multiple formats for flexibility."""

        # 1. Save as single NPZ file (efficient for loading all at once)
        npz_file = self.output_dir / "isw_embeddings.npz"
        np.savez_compressed(npz_file, **embeddings)
        logger.info(f"Saved compressed embeddings to {npz_file}")

        # 2. Save as individual npy files (efficient for loading by date)
        individual_dir = self.output_dir / "by_date"
        individual_dir.mkdir(exist_ok=True)
        for date, emb in embeddings.items():
            np.save(individual_dir / f"{date}.npy", emb)
        logger.info(f"Saved {len(embeddings)} individual embedding files")

        # 3. Save metadata
        metadata_file = self.output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # 4. Save as a single matrix with date index (for batch loading)
        dates = sorted(embeddings.keys())
        matrix = np.stack([embeddings[d] for d in dates])
        matrix_file = self.output_dir / "isw_embedding_matrix.npy"
        np.save(matrix_file, matrix)

        index_file = self.output_dir / "isw_date_index.json"
        with open(index_file, 'w') as f:
            json.dump({"dates": dates, "shape": list(matrix.shape)}, f, indent=2)
        logger.info(f"Saved embedding matrix {matrix.shape} to {matrix_file}")

    def run(self):
        """Run the full embedding generation pipeline."""
        logger.info(f"Starting embedding generation with model: {self.model}")

        # Load existing embeddings to avoid reprocessing
        existing_embeddings = self.load_existing_embeddings()

        # Load reports, skipping those we already have
        reports = self.load_reports(skip_dates=set(existing_embeddings.keys()))

        if not reports and not existing_embeddings:
            logger.error("No reports found to process")
            return

        if not reports:
            logger.info("All reports already have embeddings, nothing to do")
            # Still update metadata and consolidated files with existing
            embeddings = existing_embeddings
        else:
            # Generate embeddings for new reports
            new_embeddings = self.process_all(reports)

            if not new_embeddings and not existing_embeddings:
                logger.error("No embeddings generated")
                return

            # Merge existing and new embeddings
            embeddings = {**existing_embeddings, **new_embeddings}
            logger.info(f"Total embeddings: {len(embeddings)} ({len(existing_embeddings)} existing + {len(new_embeddings)} new)")

        # Save results (includes both existing and new)
        self.save_embeddings(embeddings)

        # Print summary
        print("\n" + "="*60)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Total reports: {self.metadata['total_reports']}")
        print(f"Embedding dimension: {self.metadata['embedding_dim']}")
        print(f"Total tokens: {self.metadata['total_tokens']:,}")
        print(f"Date range: {self.metadata['date_range']['start']} to {self.metadata['date_range']['end']}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for ISW assessment reports"
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=DEFAULT_TEXT_DIR,
        help="Directory containing ISW text JSON files"
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
        choices=["voyage-4-large", "voyage-4", "voyage-4-lite", "voyage-3-large", "voyage-3", "voyage-2"],
        help="Voyage embedding model to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of reports to process per API call (default: 8 to stay under 120K token limit)"
    )

    args = parser.parse_args()

    generator = ISWEmbeddingGenerator(
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size
    )

    generator.run()


if __name__ == "__main__":
    main()
