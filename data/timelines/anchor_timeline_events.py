#!/usr/bin/env python3
"""
Timeline Event Anchoring

Creates explicit mappings between timeline events (battles, offensives) and
the underlying data sources (ISW assessments, VIINA events, Telegram messages).

This enables the HAN model to:
1. Learn phase-aware representations
2. Use operational context as auxiliary features
3. Understand temporal boundaries of major events

Usage:
    python anchor_timeline_events.py
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
TIMELINE_DIR = Path(__file__).parent
EMBEDDING_DIR = TIMELINE_DIR / "embeddings"
ISW_DIR = Path(__file__).parent.parent / "wayback_archives" / "isw_assessments"
VIINA_DIR = Path(__file__).parent.parent / "viina" / "extracted"
OUTPUT_DIR = TIMELINE_DIR / "anchored"


# Major operations with known date ranges (manually curated for accuracy)
# These override the Wikipedia-scraped dates which can be incomplete
MAJOR_OPERATIONS = {
    "Battle_of_Kyiv_(2022)": {
        "name": "Battle of Kyiv",
        "start": "2022-02-24",
        "end": "2022-04-02",
        "phase": "northern_offensive",
        "result": "ukrainian_victory",
        "locations": ["Kyiv", "Irpin", "Bucha", "Hostomel", "Antonov Airport"]
    },
    "Battle_of_Kharkiv_(2022)": {
        "name": "Battle of Kharkiv",
        "start": "2022-02-24",
        "end": "2022-05-14",
        "phase": "northeastern_offensive",
        "result": "ukrainian_victory",
        "locations": ["Kharkiv", "Chuhuiv", "Derhachi"]
    },
    "Siege_of_Mariupol": {
        "name": "Siege of Mariupol",
        "start": "2022-02-24",
        "end": "2022-05-20",
        "phase": "southern_offensive",
        "result": "russian_victory",
        "locations": ["Mariupol", "Azovstal"]
    },
    "Battle_of_Sievierodonetsk_(2022)": {
        "name": "Battle of Sievierodonetsk",
        "start": "2022-05-06",
        "end": "2022-06-25",
        "phase": "donbas_offensive",
        "result": "russian_victory",
        "locations": ["Sievierodonetsk", "Lysychansk", "Luhansk"]
    },
    "2022_Kharkiv_counteroffensive": {
        "name": "Kharkiv Counteroffensive",
        "start": "2022-09-06",
        "end": "2022-10-02",
        "phase": "ukrainian_counteroffensive",
        "result": "ukrainian_victory",
        "locations": ["Kharkiv", "Izium", "Kupiansk", "Lyman"]
    },
    "2022_Kherson_counteroffensive": {
        "name": "Kherson Counteroffensive",
        "start": "2022-08-29",
        "end": "2022-11-11",
        "phase": "ukrainian_counteroffensive",
        "result": "ukrainian_victory",
        "locations": ["Kherson", "Nova Kakhovka"]
    },
    "Battle_of_Bakhmut": {
        "name": "Battle of Bakhmut",
        "start": "2022-08-01",
        "end": "2023-05-20",
        "phase": "bakhmut_campaign",
        "result": "russian_victory",
        "locations": ["Bakhmut", "Soledar", "Chasiv Yar"]
    },
    "2023_Ukrainian_counteroffensive": {
        "name": "2023 Ukrainian Counteroffensive",
        "start": "2023-06-04",
        "end": "2023-10-31",
        "phase": "2023_counteroffensive",
        "result": "inconclusive",
        "locations": ["Zaporizhzhia", "Robotyne", "Tokmak", "Melitopol"]
    },
    "Battle_of_Avdiivka_(2023–2024)": {
        "name": "Battle of Avdiivka",
        "start": "2023-10-10",
        "end": "2024-02-17",
        "phase": "avdiivka_campaign",
        "result": "russian_victory",
        "locations": ["Avdiivka", "Donetsk"]
    },
    "2024_Kharkiv_offensive": {
        "name": "2024 Kharkiv Offensive",
        "start": "2024-05-10",
        "end": "2024-06-30",
        "phase": "2024_russian_offensive",
        "result": "ongoing",
        "locations": ["Vovchansk", "Kharkiv"]
    },
    "Kursk_incursion": {
        "name": "Kursk Incursion",
        "start": "2024-08-06",
        "end": None,  # Ongoing
        "phase": "kursk_operation",
        "result": "ongoing",
        "locations": ["Sudzha", "Kursk"]
    }
}


class TimelineAnchor:
    """Anchors timeline events to underlying data sources."""

    def __init__(self):
        self.operations = MAJOR_OPERATIONS
        self.isw_dates = self._load_isw_dates()
        self.timeline_embeddings = self._load_timeline_embeddings()
        self.isw_embeddings = self._load_isw_embeddings()

    def _load_isw_dates(self) -> list[str]:
        """Load available ISW report dates."""
        text_dir = ISW_DIR / "text"
        if not text_dir.exists():
            return []
        dates = sorted([f.stem for f in text_dir.glob("*.json")])
        logger.info(f"Loaded {len(dates)} ISW report dates")
        return dates

    def _load_timeline_embeddings(self) -> tuple[np.ndarray, list[dict]]:
        """Load timeline embeddings and metadata."""
        matrix_file = EMBEDDING_DIR / "timeline_embedding_matrix.npy"
        index_file = EMBEDDING_DIR / "timeline_index.json"

        if not matrix_file.exists():
            return None, []

        matrix = np.load(matrix_file)
        with open(index_file) as f:
            index_data = json.load(f)
        return matrix, index_data.get('items', [])

    def _load_isw_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Load ISW embeddings and date index."""
        isw_emb_dir = ISW_DIR / "embeddings"
        matrix_file = isw_emb_dir / "isw_embedding_matrix.npy"
        index_file = isw_emb_dir / "isw_date_index.json"

        if not matrix_file.exists():
            return None, []

        matrix = np.load(matrix_file)
        with open(index_file) as f:
            index_data = json.load(f)
        return matrix, index_data.get('dates', [])

    def create_phase_labels(self) -> dict[str, dict]:
        """Create phase labels for each date in the ISW corpus.

        Returns:
            Dict mapping date -> {phase, operations, phase_day}
        """
        phase_labels = {}

        for date_str in self.isw_dates:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Find all active operations on this date
            active_ops = []
            primary_phase = "baseline"

            for op_id, op_info in self.operations.items():
                start = datetime.strptime(op_info['start'], "%Y-%m-%d")
                end_str = op_info.get('end')
                end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else datetime.now()

                if start <= date <= end:
                    active_ops.append({
                        'id': op_id,
                        'name': op_info['name'],
                        'phase': op_info['phase'],
                        'day_of_operation': (date - start).days + 1,
                        'locations': op_info['locations']
                    })
                    # Use most recent operation's phase as primary
                    primary_phase = op_info['phase']

            phase_labels[date_str] = {
                'date': date_str,
                'primary_phase': primary_phase,
                'active_operations': active_ops,
                'n_active_operations': len(active_ops)
            }

        # Add phase transition markers
        phases = [phase_labels[d]['primary_phase'] for d in sorted(phase_labels.keys())]
        dates_sorted = sorted(phase_labels.keys())

        for i, date_str in enumerate(dates_sorted):
            if i > 0:
                prev_phase = phases[i-1]
                curr_phase = phases[i]
                phase_labels[date_str]['is_phase_transition'] = (prev_phase != curr_phase)
            else:
                phase_labels[date_str]['is_phase_transition'] = True  # First day

        logger.info(f"Created phase labels for {len(phase_labels)} dates")
        return phase_labels

    def compute_soft_alignment(self, threshold: float = 0.4) -> dict[str, dict]:
        """Compute soft alignment scores between operations and ISW dates.

        Uses embedding similarity to create weighted associations.
        """
        if self.timeline_embeddings[0] is None or self.isw_embeddings[0] is None:
            logger.warning("Embeddings not available for soft alignment")
            return {}

        timeline_matrix, timeline_items = self.timeline_embeddings
        isw_matrix, isw_dates = self.isw_embeddings

        # Normalize
        timeline_norm = timeline_matrix / np.linalg.norm(timeline_matrix, axis=1, keepdims=True)
        isw_norm = isw_matrix / np.linalg.norm(isw_matrix, axis=1, keepdims=True)

        # Compute similarity
        similarity = timeline_norm @ isw_norm.T  # [n_events, n_dates]

        # Create alignment structure
        alignment = {}

        # For each ISW date, find relevant operations
        for j, date_str in enumerate(isw_dates):
            sims = similarity[:, j]

            # Get operations above threshold
            relevant = []
            for i, item in enumerate(timeline_items):
                if sims[i] >= threshold:
                    relevant.append({
                        'operation_id': item.get('id', ''),
                        'operation_name': item.get('name', ''),
                        'similarity': float(sims[i])
                    })

            # Sort by similarity
            relevant.sort(key=lambda x: x['similarity'], reverse=True)

            alignment[date_str] = {
                'date': date_str,
                'relevant_operations': relevant[:5],  # Top 5
                'max_similarity': float(sims.max()),
                'mean_similarity': float(sims.mean())
            }

        logger.info(f"Computed soft alignment for {len(alignment)} dates")
        return alignment

    def create_operation_features(self) -> dict[str, np.ndarray]:
        """Create operation-aware feature vectors for each date.

        Returns:
            Dict mapping date -> feature vector
        """
        phase_labels = self.create_phase_labels()

        # Define phase encoding
        phases = list(set(op['phase'] for op in self.operations.values()))
        phases.append('baseline')
        phase_to_idx = {p: i for i, p in enumerate(phases)}
        n_phases = len(phases)

        features = {}

        for date_str, label_info in phase_labels.items():
            # One-hot encode primary phase
            phase_vec = np.zeros(n_phases, dtype=np.float32)
            phase_idx = phase_to_idx.get(label_info['primary_phase'], phase_to_idx['baseline'])
            phase_vec[phase_idx] = 1.0

            # Add numerical features
            n_ops = label_info['n_active_operations']
            is_transition = 1.0 if label_info['is_phase_transition'] else 0.0

            # Day of operation (normalized, use first active op)
            day_of_op = 0.0
            if label_info['active_operations']:
                day_of_op = label_info['active_operations'][0]['day_of_operation'] / 100.0  # Normalize

            # Combine features
            feature_vec = np.concatenate([
                phase_vec,
                np.array([n_ops, is_transition, day_of_op], dtype=np.float32)
            ])

            features[date_str] = feature_vec

        logger.info(f"Created {len(features)} feature vectors (dim={len(feature_vec)})")
        return features

    def save_anchored_data(self):
        """Save all anchored data to files."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Phase labels
        phase_labels = self.create_phase_labels()
        with open(OUTPUT_DIR / "phase_labels.json", 'w') as f:
            json.dump(phase_labels, f, indent=2)

        # 2. Soft alignment
        soft_alignment = self.compute_soft_alignment()
        with open(OUTPUT_DIR / "soft_alignment.json", 'w') as f:
            json.dump(soft_alignment, f, indent=2)

        # 3. Operation features
        features = self.create_operation_features()

        # Save as numpy matrix with index
        dates = sorted(features.keys())
        matrix = np.stack([features[d] for d in dates])
        np.save(OUTPUT_DIR / "operation_features.npy", matrix)

        with open(OUTPUT_DIR / "operation_features_index.json", 'w') as f:
            json.dump({
                'dates': dates,
                'shape': list(matrix.shape),
                'phases': list(set(op['phase'] for op in self.operations.values())) + ['baseline'],
                'feature_names': [
                    *[f"phase_{p}" for p in list(set(op['phase'] for op in self.operations.values())) + ['baseline']],
                    'n_active_operations',
                    'is_phase_transition',
                    'day_of_operation_normalized'
                ]
            }, f, indent=2)

        # 4. Save curated operations list
        with open(OUTPUT_DIR / "major_operations.json", 'w') as f:
            json.dump(self.operations, f, indent=2)

        logger.info(f"Saved anchored data to {OUTPUT_DIR}")

        # Print summary
        print("\n" + "="*60)
        print("TIMELINE ANCHORING COMPLETE")
        print("="*60)
        print(f"Phase labels: {len(phase_labels)} dates")
        print(f"Soft alignments: {len(soft_alignment)} dates")
        print(f"Feature matrix: {matrix.shape}")
        print(f"Operations defined: {len(self.operations)}")
        print(f"\nPhases detected:")
        phase_counts = {}
        for d, info in phase_labels.items():
            p = info['primary_phase']
            phase_counts[p] = phase_counts.get(p, 0) + 1
        for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
            print(f"  {phase}: {count} days")
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("="*60)


def main():
    anchor = TimelineAnchor()
    anchor.save_anchored_data()


if __name__ == "__main__":
    main()
