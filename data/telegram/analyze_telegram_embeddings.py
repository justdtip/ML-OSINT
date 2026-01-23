#!/usr/bin/env python3
"""
Telegram Embedding Analysis

Analyzes multimodal embeddings from Telegram messages to discover:
- Semantic clusters (equipment types, event categories, locations)
- Temporal patterns and anomalies
- Cross-channel correlations
- Information signatures that could enhance the HAN model

Usage:
    python analyze_telegram_embeddings.py [--output-dir DIR]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DEFAULT_EMBEDDING_DIR = Path(__file__).parent / "embeddings"
DEFAULT_TELEGRAM_DIR = Path(__file__).parent


class TelegramEmbeddingAnalyzer:
    """Analyzes Telegram message embeddings for information patterns."""

    def __init__(self, embedding_dir: Path, telegram_dir: Path, output_dir: Path):
        self.embedding_dir = embedding_dir
        self.telegram_dir = telegram_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = {}  # channel -> {msg_id: embedding}
        self.messages = {}    # channel -> {msg_id: message_dict}
        self.combined_matrix = None
        self.combined_metadata = []

    def load_channel_data(self, channel_name: str):
        """Load embeddings and messages for a channel."""
        # Load embeddings
        emb_dir = self.embedding_dir / channel_name
        if not emb_dir.exists():
            logger.warning(f"No embeddings found for {channel_name}")
            return

        npz_file = emb_dir / "embeddings.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            self.embeddings[channel_name] = {
                int(k): v for k, v in data.items()
            }
            logger.info(f"Loaded {len(self.embeddings[channel_name])} embeddings for {channel_name}")

        # Load messages
        channel_dir = self.telegram_dir / channel_name
        json_files = list(channel_dir.glob("messages_*.json"))
        if json_files:
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                data = json.load(f)
            self.messages[channel_name] = {
                msg['message_id']: msg for msg in data.get('messages', [])
            }

    def load_all_channels(self):
        """Load data from all channels with embeddings."""
        for subdir in self.embedding_dir.iterdir():
            if subdir.is_dir() and (subdir / "embeddings.npz").exists():
                self.load_channel_data(subdir.name)

        # Create combined matrix
        all_embeddings = []
        for channel, embs in self.embeddings.items():
            for msg_id, emb in embs.items():
                all_embeddings.append(emb)
                msg = self.messages.get(channel, {}).get(msg_id, {})
                self.combined_metadata.append({
                    'channel': channel,
                    'message_id': msg_id,
                    'timestamp': msg.get('timestamp'),
                    'media_type': msg.get('media_type', 'none'),
                    'text_preview': (msg.get('text', '') or '')[:100]
                })

        if all_embeddings:
            self.combined_matrix = np.stack(all_embeddings)
            logger.info(f"Combined matrix shape: {self.combined_matrix.shape}")

    def analyze_dimensionality(self) -> dict:
        """Analyze intrinsic dimensionality of embedding space."""
        if self.combined_matrix is None:
            return {}

        logger.info("Analyzing dimensionality...")

        # PCA analysis
        pca = PCA()
        pca.fit(self.combined_matrix)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_50 = np.argmax(cumvar >= 0.50) + 1
        dim_75 = np.argmax(cumvar >= 0.75) + 1
        dim_90 = np.argmax(cumvar >= 0.90) + 1
        dim_95 = np.argmax(cumvar >= 0.95) + 1

        # Find elbow point
        second_deriv = np.diff(np.diff(pca.explained_variance_ratio_))
        elbow = np.argmax(second_deriv) + 1

        results = {
            'total_dimensions': self.combined_matrix.shape[1],
            'samples': self.combined_matrix.shape[0],
            'variance_explained': {
                '50%': int(dim_50),
                '75%': int(dim_75),
                '90%': int(dim_90),
                '95%': int(dim_95)
            },
            'elbow_point': int(elbow),
            'top_10_variance': [float(v) for v in pca.explained_variance_ratio_[:10]]
        }

        logger.info(f"Dimensionality: elbow at {elbow}, 90% variance at {dim_90} dims")
        return results

    def analyze_clustering(self, n_clusters_range: tuple = (5, 20)) -> dict:
        """Find optimal clustering and analyze cluster composition."""
        if self.combined_matrix is None:
            return {}

        logger.info("Analyzing clustering...")

        # Reduce dimensionality for clustering
        pca = PCA(n_components=64)
        reduced = pca.fit_transform(self.combined_matrix)

        # Find optimal k
        silhouettes = []
        for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(reduced)
            score = silhouette_score(reduced, labels)
            silhouettes.append((k, score))

        best_k, best_score = max(silhouettes, key=lambda x: x[1])

        # Cluster with optimal k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced)

        # Analyze cluster composition
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'count': 0,
                    'channels': {},
                    'media_types': {},
                    'sample_texts': []
                }

            clusters[label]['count'] += 1
            meta = self.combined_metadata[i]

            # Count by channel
            ch = meta['channel']
            clusters[label]['channels'][ch] = clusters[label]['channels'].get(ch, 0) + 1

            # Count by media type
            mt = meta['media_type']
            clusters[label]['media_types'][mt] = clusters[label]['media_types'].get(mt, 0) + 1

            # Sample texts
            if len(clusters[label]['sample_texts']) < 3 and meta['text_preview']:
                clusters[label]['sample_texts'].append(meta['text_preview'])

        results = {
            'optimal_k': best_k,
            'silhouette_score': float(best_score),
            'silhouette_curve': [(k, float(s)) for k, s in silhouettes],
            'clusters': {int(k): v for k, v in clusters.items()}
        }

        logger.info(f"Optimal clustering: k={best_k}, silhouette={best_score:.3f}")
        return results

    def analyze_temporal_patterns(self) -> dict:
        """Analyze how embeddings evolve over time."""
        if self.combined_matrix is None:
            return {}

        logger.info("Analyzing temporal patterns...")

        # Group by date
        date_embeddings = {}
        for i, meta in enumerate(self.combined_metadata):
            ts = meta.get('timestamp')
            if not ts:
                continue

            try:
                date = ts.split('T')[0]
                if date not in date_embeddings:
                    date_embeddings[date] = []
                date_embeddings[date].append(self.combined_matrix[i])
            except:
                continue

        if not date_embeddings:
            return {}

        # Compute daily centroids
        dates = sorted(date_embeddings.keys())
        centroids = []
        for date in dates:
            centroid = np.mean(date_embeddings[date], axis=0)
            centroids.append(centroid)

        centroids = np.stack(centroids)

        # Compute day-to-day similarity (cosine)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        normalized = centroids / (norms + 1e-8)
        similarities = []

        for i in range(1, len(normalized)):
            sim = np.dot(normalized[i], normalized[i-1])
            similarities.append(float(sim))

        # Find anomalies (large shifts)
        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            anomalies = []

            for i, sim in enumerate(similarities):
                if sim < mean_sim - 2 * std_sim:
                    anomalies.append({
                        'date': dates[i+1],
                        'previous_date': dates[i],
                        'similarity': float(sim),
                        'z_score': float((sim - mean_sim) / std_sim)
                    })

            # Autocorrelation
            autocorr = []
            for lag in [1, 7, 14, 30]:
                if lag < len(centroids):
                    corrs = []
                    for i in range(lag, len(centroids)):
                        c = np.dot(normalized[i], normalized[i-lag])
                        corrs.append(c)
                    autocorr.append({
                        'lag': lag,
                        'mean_correlation': float(np.mean(corrs))
                    })

            results = {
                'date_range': [dates[0], dates[-1]],
                'total_days': len(dates),
                'mean_daily_similarity': float(mean_sim),
                'std_daily_similarity': float(std_sim),
                'autocorrelation': autocorr,
                'anomalies': sorted(anomalies, key=lambda x: x['z_score'])[:10]
            }
        else:
            results = {'date_range': [dates[0], dates[-1]], 'total_days': len(dates)}

        logger.info(f"Temporal analysis: {len(dates)} days, mean similarity {results.get('mean_daily_similarity', 0):.3f}")
        return results

    def analyze_cross_channel(self) -> dict:
        """Analyze correlations between channels."""
        if len(self.embeddings) < 2:
            return {}

        logger.info("Analyzing cross-channel patterns...")

        # Compute channel centroids
        channel_centroids = {}
        for channel, embs in self.embeddings.items():
            if embs:
                centroid = np.mean(list(embs.values()), axis=0)
                channel_centroids[channel] = centroid

        # Compute pairwise similarities
        channels = list(channel_centroids.keys())
        similarities = {}

        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                c1 = channel_centroids[ch1]
                c2 = channel_centroids[ch2]
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                similarities[f"{ch1}_{ch2}"] = float(sim)

        # Find most/least similar pairs
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        results = {
            'channels': channels,
            'pairwise_similarities': similarities,
            'most_similar': sorted_pairs[:5] if sorted_pairs else [],
            'least_similar': sorted_pairs[-5:] if sorted_pairs else [],
            'channel_sizes': {ch: len(embs) for ch, embs in self.embeddings.items()}
        }

        logger.info(f"Cross-channel analysis: {len(channels)} channels")
        return results

    def find_information_signatures(self) -> dict:
        """Look for recurring semantic patterns that could be useful signals."""
        if self.combined_matrix is None:
            return {}

        logger.info("Finding information signatures...")

        # Use DBSCAN to find dense regions (potential recurring patterns)
        pca = PCA(n_components=32)
        reduced = pca.fit_transform(self.combined_matrix)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(reduced)

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(scaled)

        # Analyze dense clusters
        unique_labels = set(labels) - {-1}  # Exclude noise
        dense_patterns = []

        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]

            if len(indices) < 10:
                continue

            # Analyze cluster
            cluster_meta = [self.combined_metadata[i] for i in indices]

            # Check channel distribution
            channels = {}
            media_types = {}
            texts = []

            for meta in cluster_meta:
                ch = meta['channel']
                channels[ch] = channels.get(ch, 0) + 1
                mt = meta['media_type']
                media_types[mt] = media_types.get(mt, 0) + 1
                if meta['text_preview']:
                    texts.append(meta['text_preview'])

            # Multi-channel clusters are more interesting (corroboration)
            cross_channel = len(channels) > 1

            dense_patterns.append({
                'cluster_id': int(label),
                'size': int(len(indices)),
                'cross_channel': cross_channel,
                'channels': channels,
                'media_types': media_types,
                'sample_texts': texts[:5]
            })

        # Sort by size and cross-channel status
        dense_patterns.sort(key=lambda x: (x['cross_channel'], x['size']), reverse=True)

        results = {
            'total_patterns': len(dense_patterns),
            'cross_channel_patterns': sum(1 for p in dense_patterns if p['cross_channel']),
            'noise_ratio': float(np.sum(labels == -1) / len(labels)),
            'patterns': dense_patterns[:20]  # Top 20
        }

        logger.info(f"Found {len(dense_patterns)} dense patterns, {results['cross_channel_patterns']} cross-channel")
        return results

    def generate_visualizations(self):
        """Generate t-SNE visualization of embedding space."""
        if self.combined_matrix is None or len(self.combined_matrix) < 50:
            return

        logger.info("Generating visualizations...")

        try:
            import matplotlib.pyplot as plt

            # PCA first, then t-SNE
            pca = PCA(n_components=50)
            pca_result = pca.fit_transform(self.combined_matrix)

            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_result = tsne.fit_transform(pca_result)

            # Color by channel
            channels = [meta['channel'] for meta in self.combined_metadata]
            unique_channels = list(set(channels))
            colors = [unique_channels.index(ch) for ch in channels]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                tsne_result[:, 0], tsne_result[:, 1],
                c=colors, cmap='tab10', alpha=0.5, s=10
            )
            plt.colorbar(scatter, label='Channel')
            plt.title('t-SNE Visualization of Telegram Embeddings (colored by channel)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            # Add legend
            for i, ch in enumerate(unique_channels):
                plt.scatter([], [], c=[plt.cm.tab10(i / 10)], label=ch)
            plt.legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'tsne_by_channel.png', dpi=150)
            plt.close()

            # Color by media type
            media_types = [meta['media_type'] for meta in self.combined_metadata]
            unique_types = list(set(media_types))
            type_colors = [unique_types.index(mt) for mt in media_types]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                tsne_result[:, 0], tsne_result[:, 1],
                c=type_colors, cmap='Set1', alpha=0.5, s=10
            )
            plt.title('t-SNE Visualization of Telegram Embeddings (colored by media type)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')

            for i, mt in enumerate(unique_types):
                plt.scatter([], [], c=[plt.cm.Set1(i / len(unique_types))], label=mt)
            plt.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'tsne_by_media_type.png', dpi=150)
            plt.close()

            logger.info(f"Saved visualizations to {self.output_dir}")

        except ImportError:
            logger.warning("matplotlib not available, skipping visualizations")

    def run(self):
        """Run full analysis pipeline."""
        logger.info("Starting Telegram embedding analysis")

        # Load data
        self.load_all_channels()

        if self.combined_matrix is None:
            logger.error("No embeddings to analyze")
            return

        # Run analyses
        results = {
            'summary': {
                'total_embeddings': len(self.combined_matrix),
                'channels': list(self.embeddings.keys()),
                'embedding_dim': self.combined_matrix.shape[1]
            },
            'dimensionality': self.analyze_dimensionality(),
            'clustering': self.analyze_clustering(),
            'temporal': self.analyze_temporal_patterns(),
            'cross_channel': self.analyze_cross_channel(),
            'information_signatures': self.find_information_signatures()
        }

        # Generate visualizations
        self.generate_visualizations()

        # Save results
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Analysis complete. Results saved to {self.output_dir}")

        # Print summary
        print("\n" + "="*60)
        print("TELEGRAM EMBEDDING ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total embeddings: {results['summary']['total_embeddings']}")
        print(f"Channels: {', '.join(results['summary']['channels'])}")

        if results.get('dimensionality'):
            dim = results['dimensionality']
            print(f"\nDimensionality:")
            print(f"  90% variance in {dim['variance_explained']['90%']} dims")
            print(f"  Elbow point: {dim['elbow_point']}")

        if results.get('clustering'):
            clust = results['clustering']
            print(f"\nClustering:")
            print(f"  Optimal k: {clust['optimal_k']}")
            print(f"  Silhouette score: {clust['silhouette_score']:.3f}")

        if results.get('temporal'):
            temp = results['temporal']
            print(f"\nTemporal:")
            print(f"  Date range: {temp['date_range'][0]} to {temp['date_range'][1]}")
            print(f"  Mean daily similarity: {temp.get('mean_daily_similarity', 0):.3f}")
            if temp.get('anomalies'):
                print(f"  Anomalies detected: {len(temp['anomalies'])}")

        if results.get('information_signatures'):
            sigs = results['information_signatures']
            print(f"\nInformation Signatures:")
            print(f"  Dense patterns: {sigs['total_patterns']}")
            print(f"  Cross-channel patterns: {sigs['cross_channel_patterns']}")

        print("="*60)

        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Telegram message embeddings")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR,
        help="Directory containing channel embeddings"
    )
    parser.add_argument(
        "--telegram-dir",
        type=Path,
        default=DEFAULT_TELEGRAM_DIR,
        help="Directory containing Telegram message files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_DIR / "analysis",
        help="Output directory for analysis results"
    )

    args = parser.parse_args()

    analyzer = TelegramEmbeddingAnalyzer(
        embedding_dir=args.embedding_dir,
        telegram_dir=args.telegram_dir,
        output_dir=args.output_dir
    )

    analyzer.run()


if __name__ == "__main__":
    main()
