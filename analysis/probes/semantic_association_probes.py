"""
Semantic-Numerical Association Testing Probes for ISW Embeddings

This module implements comprehensive probes to assess how well numerical representations
(model latents, quantitative features) align with semantic content (ISW embeddings).

The core question: How well do numerical representations align with semantic content?

Probe Categories:
================

5.1 ISW Alignment Validation
    - 5.1.1 ISW-Latent Correlation: Cosine similarity between ISW (reduced) and fused latent
    - 5.1.2 ISW Topic-Source Correlation: Topic extraction via clustering/LDA, correlate with sources
    - 5.1.3 ISW Predictive Content Test: Can ISW(t) predict numerical deltas at t+1?

5.2 Cross-Modal Semantic Grounding
    - 5.2.1 Event-Triggered Response Analysis: Latent trajectory around known events
    - 5.2.2 Narrative-Numerical Lag Analysis: Cross-correlation at lags [-7, +7] days
    - 5.2.3 Semantic Anomaly Detection: Compare numerical vs embedding outliers

5.3 Counterfactual Semantic Probing
    - 5.3.1 Semantic Perturbation Effects: Replace ISW(t), measure prediction change
    - 5.3.2 Missing Semantic Interpolation: Train Numerical_latent -> ISW_embedding predictor

5.4 Semantic Enrichment Potential (Specification only)
    - Document requirements for Telegram, combat footage, official statements

Data Sources:
- ISW embeddings: 1024-dim Voyage embeddings, 1272 daily reports
- Location: data/wayback_archives/isw_assessments/embeddings/ (see config.paths.ISW_EMBEDDINGS_DIR)
- Model latent: 128-dim, matches temporal resolution

Author: NLP Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import correlate
from scipy.spatial.distance import cdist

# Optional imports for advanced analysis
try:
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_LDA = True
except ImportError:
    HAS_LDA = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    ISW_EMBEDDINGS_DIR,
    PROBE_OUTPUT_DIR,
)

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================

BASE_DIR = PROJECT_ROOT
EMBEDDING_DIR = ISW_EMBEDDINGS_DIR
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR
OUTPUT_DIR = PROBE_OUTPUT_DIR / "semantic_results"

# Key conflict events for event-triggered analysis
MAJOR_EVENTS = {
    '2022-10-08': {'name': 'Kerch Bridge attack', 'type': 'infrastructure'},
    '2022-11-11': {'name': 'Kherson withdrawal', 'type': 'territorial'},
    '2023-06-23': {'name': 'Prigozhin mutiny (start)', 'type': 'political'},
    '2023-06-24': {'name': 'Prigozhin mutiny (end)', 'type': 'political'},
    '2023-06-06': {'name': 'Kakhovka Dam collapse', 'type': 'infrastructure'},
    '2024-02-17': {'name': 'Avdiivka fall', 'type': 'territorial'},
}

# Additional events for comprehensive analysis
EXTENDED_EVENTS = {
    '2022-02-24': {'name': 'Invasion begins', 'type': 'major'},
    '2022-04-03': {'name': 'Bucha massacre revealed', 'type': 'political'},
    '2022-05-20': {'name': 'Mariupol falls', 'type': 'territorial'},
    '2022-07-03': {'name': 'Lysychansk falls', 'type': 'territorial'},
    '2022-08-29': {'name': 'Kherson counteroffensive starts', 'type': 'operational'},
    '2022-09-11': {'name': 'Kharkiv counteroffensive success', 'type': 'territorial'},
    '2023-01-11': {'name': 'Soledar falls', 'type': 'territorial'},
    '2023-05-21': {'name': 'Bakhmut falls', 'type': 'territorial'},
    '2023-08-24': {'name': 'Prigozhin death', 'type': 'political'},
    '2024-08-06': {'name': 'Kursk incursion begins', 'type': 'operational'},
    '2024-10-14': {'name': 'North Korean troops reported', 'type': 'political'},
}


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

@dataclass
class ISWEmbeddingData:
    """Container for ISW embedding data and metadata."""
    embeddings: np.ndarray  # [n_days, 1024]
    dates: List[str]
    date_to_idx: Dict[str, int]
    idx_to_date: Dict[int, str]
    metadata: Dict[str, Any]

    @classmethod
    def load(cls, embedding_dir: Path = EMBEDDING_DIR) -> 'ISWEmbeddingData':
        """Load ISW embeddings from disk."""
        matrix_path = embedding_dir / "isw_embedding_matrix.npy"
        index_path = embedding_dir / "isw_date_index.json"
        metadata_path = embedding_dir / "embedding_metadata.json"

        if not matrix_path.exists():
            raise FileNotFoundError(f"Embedding matrix not found at {matrix_path}")

        embeddings = np.load(matrix_path)

        with open(index_path) as f:
            index_data = json.load(f)
        dates = index_data['dates']

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        date_to_idx = {d: i for i, d in enumerate(dates)}
        idx_to_date = {i: d for i, d in enumerate(dates)}

        print(f"Loaded ISW embeddings: {embeddings.shape}")
        print(f"Date range: {dates[0]} to {dates[-1]}")

        return cls(
            embeddings=embeddings,
            dates=dates,
            date_to_idx=date_to_idx,
            idx_to_date=idx_to_date,
            metadata=metadata
        )

    def get_embedding(self, date: str) -> Optional[np.ndarray]:
        """Get embedding for a specific date."""
        if date in self.date_to_idx:
            return self.embeddings[self.date_to_idx[date]]
        return None

    def get_date_range(self, start: str, end: str) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings and dates for a date range."""
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        indices = []
        selected_dates = []
        for i, d in enumerate(self.dates):
            dt = pd.to_datetime(d)
            if start_dt <= dt <= end_dt:
                indices.append(i)
                selected_dates.append(d)

        return self.embeddings[indices], selected_dates


# =============================================================================
# 5.1 ISW ALIGNMENT VALIDATION
# =============================================================================

class ISWAlignmentProbe:
    """
    5.1.1 ISW-Latent Correlation Probe

    Computes cosine similarity between ISW embeddings (reduced to 128-dim)
    and fused latent representations per day.

    Methods:
    - fit_projection: Fit PCA or learned projection from 1024 -> 128 dim
    - compute_daily_alignment: Compute cosine similarity for each day
    - compute_temporal_correlation: Correlation of similarity over time
    - analyze_alignment_by_period: Break down by conflict phases
    """

    def __init__(
        self,
        target_dim: int = 128,
        projection_method: str = 'pca',  # 'pca' or 'learned'
    ):
        self.target_dim = target_dim
        self.projection_method = projection_method
        self.projection = None
        self.scaler = StandardScaler()

    def fit_projection(self, embeddings: np.ndarray) -> 'ISWAlignmentProbe':
        """
        Fit dimensionality reduction from 1024-dim ISW to 128-dim latent space.

        Args:
            embeddings: ISW embedding matrix [n_days, 1024]

        Returns:
            self for chaining
        """
        print(f"Fitting {self.projection_method} projection: 1024 -> {self.target_dim}")

        if self.projection_method == 'pca':
            self.projection = PCA(n_components=self.target_dim)
            self.projection.fit(embeddings)

            explained_var = sum(self.projection.explained_variance_ratio_)
            print(f"  PCA explains {explained_var*100:.1f}% variance with {self.target_dim} components")

        elif self.projection_method == 'learned':
            # For learned projection, we need paired data (ISW, latent)
            # This will be fitted during compute_daily_alignment if latents provided
            self.projection = None
            print("  Learned projection will be fitted when latents are provided")

        return self

    def project_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Project ISW embeddings to target dimension."""
        if self.projection is None:
            raise ValueError("Projection not fitted. Call fit_projection first.")
        return self.projection.transform(embeddings)

    def compute_daily_alignment(
        self,
        isw_embeddings: np.ndarray,
        latent_representations: np.ndarray,
        dates: List[str],
        fit_projection: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute cosine similarity between reduced ISW and latent per day.

        Args:
            isw_embeddings: [n_days, 1024] ISW embeddings
            latent_representations: [n_days, 128] model latent states
            dates: List of date strings
            fit_projection: Whether to fit projection (if not already done)

        Returns:
            Dict with alignment metrics and per-day similarities
        """
        n_days = len(dates)

        # Ensure same length
        min_len = min(len(isw_embeddings), len(latent_representations), n_days)
        isw_embeddings = isw_embeddings[:min_len]
        latent_representations = latent_representations[:min_len]
        dates = dates[:min_len]

        # Fit and project ISW embeddings
        if fit_projection and self.projection is None:
            self.fit_projection(isw_embeddings)

        isw_projected = self.project_embeddings(isw_embeddings)

        # Normalize for cosine similarity
        isw_norm = isw_projected / (np.linalg.norm(isw_projected, axis=1, keepdims=True) + 1e-8)
        latent_norm = latent_representations / (np.linalg.norm(latent_representations, axis=1, keepdims=True) + 1e-8)

        # Compute per-day cosine similarity
        daily_similarities = np.sum(isw_norm * latent_norm, axis=1)

        # Compute overall statistics
        results = {
            'daily_similarities': daily_similarities,
            'dates': dates,
            'mean_similarity': float(np.mean(daily_similarities)),
            'std_similarity': float(np.std(daily_similarities)),
            'min_similarity': float(np.min(daily_similarities)),
            'max_similarity': float(np.max(daily_similarities)),
            'median_similarity': float(np.median(daily_similarities)),
        }

        # Find top/bottom alignment days
        sorted_idx = np.argsort(daily_similarities)
        results['lowest_alignment_days'] = [
            {'date': dates[i], 'similarity': float(daily_similarities[i])}
            for i in sorted_idx[:5]
        ]
        results['highest_alignment_days'] = [
            {'date': dates[i], 'similarity': float(daily_similarities[i])}
            for i in sorted_idx[-5:][::-1]
        ]

        # Temporal autocorrelation of alignment
        autocorr_lags = [1, 7, 14, 30]
        results['alignment_autocorrelation'] = {}
        for lag in autocorr_lags:
            if len(daily_similarities) > lag:
                corr = np.corrcoef(daily_similarities[:-lag], daily_similarities[lag:])[0, 1]
                results['alignment_autocorrelation'][f'lag_{lag}'] = float(corr)

        print(f"\nISW-Latent Alignment Results:")
        print(f"  Mean cosine similarity: {results['mean_similarity']:.4f}")
        print(f"  Std deviation: {results['std_similarity']:.4f}")
        print(f"  Range: [{results['min_similarity']:.4f}, {results['max_similarity']:.4f}]")

        return results

    def analyze_alignment_by_period(
        self,
        daily_similarities: np.ndarray,
        dates: List[str],
        period_definitions: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Break down alignment by conflict phases/periods.

        Args:
            daily_similarities: Per-day cosine similarities
            dates: Date strings
            period_definitions: Dict mapping period name to (start_date, end_date)

        Returns:
            Dict with period-wise alignment statistics
        """
        if period_definitions is None:
            # Default conflict phases
            period_definitions = {
                'initial_invasion': ('2022-02-24', '2022-04-15'),
                'eastern_focus': ('2022-04-16', '2022-08-31'),
                'ukrainian_counter': ('2022-09-01', '2022-12-31'),
                'bakhmut_period': ('2023-01-01', '2023-05-31'),
                'counteroffensive_2023': ('2023-06-01', '2023-10-31'),
                'attritional_phase': ('2023-11-01', '2024-01-31'),
                'avdiivka_fall': ('2024-02-01', '2024-04-30'),
                'kursk_incursion': ('2024-08-01', '2024-10-31'),
            }

        dates_dt = pd.to_datetime(dates)
        period_stats = {}

        for period_name, (start, end) in period_definitions.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            mask = (dates_dt >= start_dt) & (dates_dt <= end_dt)
            if mask.sum() == 0:
                continue

            period_sims = daily_similarities[mask]
            period_stats[period_name] = {
                'mean': float(np.mean(period_sims)),
                'std': float(np.std(period_sims)),
                'n_days': int(mask.sum()),
                'date_range': f"{start} to {end}",
            }

        return period_stats


class TopicExtractionProbe:
    """
    5.1.2 ISW Topic-Source Correlation Probe

    Extract topics from ISW embeddings via clustering or LDA,
    then correlate with numerical data sources.

    Methods:
    - extract_topics_kmeans: Cluster embeddings to find topic groups
    - extract_topics_lda: LDA on projected embeddings (if applicable)
    - compute_topic_source_correlation: Correlate topics with numerical sources
    """

    def __init__(
        self,
        n_topics: int = 8,
        method: str = 'kmeans',  # 'kmeans' or 'lda'
        pca_components: int = 50,  # For dimensionality reduction before clustering
    ):
        self.n_topics = n_topics
        self.method = method
        self.pca_components = pca_components
        self.pca = None
        self.model = None
        self.topic_labels = None

    def extract_topics(
        self,
        embeddings: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Any]:
        """
        Extract topics from ISW embeddings.

        Args:
            embeddings: [n_days, 1024] ISW embeddings
            dates: List of date strings

        Returns:
            Dict with topic assignments and analysis
        """
        print(f"\nExtracting {self.n_topics} topics using {self.method}...")

        # Reduce dimensionality first for better clustering
        self.pca = PCA(n_components=self.pca_components)
        embeddings_reduced = self.pca.fit_transform(embeddings)
        print(f"  Reduced to {self.pca_components} dims, explains {sum(self.pca.explained_variance_ratio_)*100:.1f}% variance")

        if self.method == 'kmeans':
            return self._extract_topics_kmeans(embeddings_reduced, dates)
        elif self.method == 'lda' and HAS_LDA:
            return self._extract_topics_lda(embeddings_reduced, dates)
        else:
            print(f"  Falling back to kmeans (LDA not available or not selected)")
            return self._extract_topics_kmeans(embeddings_reduced, dates)

    def _extract_topics_kmeans(
        self,
        embeddings_reduced: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Any]:
        """Extract topics using K-Means clustering."""
        # Find optimal k using silhouette score
        k_range = range(max(3, self.n_topics - 3), min(15, self.n_topics + 5))
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_reduced)
            sil = silhouette_score(embeddings_reduced, labels)
            silhouettes.append((k, sil))

        best_k = max(silhouettes, key=lambda x: x[1])[0]
        print(f"  Best k={best_k} (silhouette={max(silhouettes, key=lambda x: x[1])[1]:.4f})")

        # Fit with optimal k (or requested n_topics)
        self.model = KMeans(n_clusters=self.n_topics, random_state=42, n_init=10)
        self.topic_labels = self.model.fit_predict(embeddings_reduced)

        # Analyze temporal distribution
        dates_dt = pd.to_datetime(dates)
        topic_stats = []

        for topic_id in range(self.n_topics):
            mask = self.topic_labels == topic_id
            topic_dates = dates_dt[mask]

            if len(topic_dates) > 0:
                # Compute median date as middle element of sorted dates
                sorted_dates = topic_dates.sort_values()
                median_date = sorted_dates[len(sorted_dates) // 2]
                topic_stats.append({
                    'topic_id': topic_id,
                    'count': int(mask.sum()),
                    'fraction': float(mask.sum() / len(self.topic_labels)),
                    'median_date': str(median_date.date()),
                    'date_range': f"{topic_dates.min().date()} to {topic_dates.max().date()}",
                })

        # Compute topic centroids in PCA space
        centroids = self.model.cluster_centers_

        results = {
            'topic_labels': self.topic_labels,
            'topic_stats': topic_stats,
            'silhouette_score': float(silhouette_score(embeddings_reduced, self.topic_labels)),
            'centroids_pca': centroids,
            'n_topics': self.n_topics,
            'method': 'kmeans',
        }

        print(f"\n  Topic distribution:")
        for ts in sorted(topic_stats, key=lambda x: x['count'], reverse=True)[:5]:
            print(f"    Topic {ts['topic_id']}: {ts['count']} days ({ts['fraction']*100:.1f}%)")

        return results

    def _extract_topics_lda(
        self,
        embeddings_reduced: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Any]:
        """Extract topics using LDA (on transformed positive space)."""
        # LDA requires non-negative values, transform embeddings
        embeddings_positive = embeddings_reduced - embeddings_reduced.min() + 0.01

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20,
        )
        topic_distributions = self.model.fit_transform(embeddings_positive)
        self.topic_labels = topic_distributions.argmax(axis=1)

        # Topic stats
        dates_dt = pd.to_datetime(dates)
        topic_stats = []

        for topic_id in range(self.n_topics):
            mask = self.topic_labels == topic_id
            topic_dates = dates_dt[mask]

            if len(topic_dates) > 0:
                sorted_dates = topic_dates.sort_values()
                median_date = sorted_dates[len(sorted_dates) // 2]
                topic_stats.append({
                    'topic_id': topic_id,
                    'count': int(mask.sum()),
                    'fraction': float(mask.sum() / len(self.topic_labels)),
                    'median_date': str(median_date.date()),
                })

        return {
            'topic_labels': self.topic_labels,
            'topic_distributions': topic_distributions,
            'topic_stats': topic_stats,
            'n_topics': self.n_topics,
            'method': 'lda',
        }

    def compute_topic_source_correlation(
        self,
        topic_results: Dict[str, Any],
        numerical_sources: Dict[str, np.ndarray],
        dates: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Correlate topic assignments with numerical data sources.

        Args:
            topic_results: Output from extract_topics
            numerical_sources: Dict mapping source name to [n_days, n_features] array
            dates: List of date strings

        Returns:
            Dict mapping source name to topic correlation metrics
        """
        topic_labels = topic_results['topic_labels']
        n_topics = topic_results['n_topics']

        correlations = {}

        for source_name, source_data in numerical_sources.items():
            # Ensure same length
            min_len = min(len(topic_labels), len(source_data))
            labels = topic_labels[:min_len]
            data = source_data[:min_len]

            # Aggregate source data per topic
            topic_means = np.zeros((n_topics, data.shape[1] if len(data.shape) > 1 else 1))
            topic_counts = np.zeros(n_topics)

            for topic_id in range(n_topics):
                mask = labels == topic_id
                if mask.sum() > 0:
                    topic_means[topic_id] = data[mask].mean(axis=0) if len(data.shape) > 1 else data[mask].mean()
                    topic_counts[topic_id] = mask.sum()

            # Compute ANOVA F-statistic (do topics explain variance in source?)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            f_stats = []
            p_values = []
            for feat_idx in range(data.shape[1]):
                groups = [data[labels == t, feat_idx] for t in range(n_topics)]
                groups = [g for g in groups if len(g) > 0]
                if len(groups) >= 2:
                    f, p = stats.f_oneway(*groups)
                    f_stats.append(f if not np.isnan(f) else 0)
                    p_values.append(p if not np.isnan(p) else 1)

            correlations[source_name] = {
                'mean_f_statistic': float(np.mean(f_stats)) if f_stats else 0.0,
                'n_significant_features': int(sum(p < 0.05 for p in p_values)),
                'total_features': len(f_stats),
                'topic_means_range': float(np.ptp(topic_means)),
            }

        print(f"\n  Topic-Source Correlations:")
        for source, corr in correlations.items():
            print(f"    {source}: F={corr['mean_f_statistic']:.2f}, "
                  f"{corr['n_significant_features']}/{corr['total_features']} sig. features")

        return correlations


class ISWPredictiveContentProbe:
    """
    5.1.3 ISW Predictive Content Test

    Test if ISW(t) can predict numerical deltas at t+1:
    - Equipment_delta(t+1)
    - FIRMS(t+1)
    - Casualty(t+1)

    Uses ridge regression with cross-validation.
    """

    def __init__(
        self,
        prediction_targets: List[str] = None,
        test_ratio: float = 0.2,
    ):
        self.prediction_targets = prediction_targets or [
            'equipment_delta', 'firms_count', 'casualty_delta'
        ]
        self.test_ratio = test_ratio
        self.models = {}
        self.results = {}

    def compute_predictive_power(
        self,
        isw_embeddings: np.ndarray,
        numerical_targets: Dict[str, np.ndarray],
        dates: List[str],
        reduce_dim: int = 64,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test if ISW(t) predicts numerical signals at t+1.

        Args:
            isw_embeddings: [n_days, 1024] ISW embeddings
            numerical_targets: Dict mapping target name to [n_days] or [n_days, n_feat] array
            dates: Date strings
            reduce_dim: PCA components for ISW before regression

        Returns:
            Dict with R2, MSE, and correlation for each target
        """
        print("\nISW Predictive Content Test:")

        # Reduce ISW dimensionality
        pca = PCA(n_components=reduce_dim)
        isw_reduced = pca.fit_transform(isw_embeddings)
        print(f"  Reduced ISW to {reduce_dim} dims ({sum(pca.explained_variance_ratio_)*100:.1f}% var)")

        results = {}

        for target_name, target_data in numerical_targets.items():
            if target_name not in self.prediction_targets and len(self.prediction_targets) > 0:
                continue

            # Prepare X (ISW at t) and y (target at t+1)
            X = isw_reduced[:-1]  # ISW at time t

            # Handle multi-dimensional targets
            if len(target_data.shape) > 1:
                y = target_data[1:].mean(axis=1)  # Average across features, shifted by 1
            else:
                y = target_data[1:]  # Target at time t+1

            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

            # Remove NaN
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 20:
                print(f"  {target_name}: Insufficient data ({len(X)} samples)")
                continue

            # Train/test split (temporal)
            split_idx = int(len(X) * (1 - self.test_ratio))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Fit ridge regression
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0.0

            self.models[target_name] = model
            results[target_name] = {
                'r2_score': float(r2),
                'mse': float(mse),
                'correlation': float(corr) if not np.isnan(corr) else 0.0,
                'n_train': len(X_train),
                'n_test': len(X_test),
            }

            print(f"  {target_name}: R2={r2:.4f}, MSE={mse:.4f}, r={corr:.4f}")

        self.results = results
        return results


# =============================================================================
# 5.2 CROSS-MODAL SEMANTIC GROUNDING
# =============================================================================

class EventResponseProbe:
    """
    5.2.1 Event-Triggered Response Analysis

    Extract latent trajectory around known events to understand
    how the model responds to major conflict developments.

    Events analyzed:
    - Kerch Bridge attack (Oct 8, 2022)
    - Kherson withdrawal (Nov 11, 2022)
    - Prigozhin mutiny (Jun 23-24, 2023)
    - Dam collapse (Jun 6, 2023)
    - Avdiivka fall (Feb 17, 2024)
    """

    def __init__(
        self,
        window_before: int = 7,
        window_after: int = 7,
        events: Dict[str, Dict[str, str]] = None,
    ):
        self.window_before = window_before
        self.window_after = window_after
        self.events = events or MAJOR_EVENTS

    def extract_event_trajectories(
        self,
        embeddings: np.ndarray,
        latents: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract embedding and latent trajectories around each event.

        Args:
            embeddings: [n_days, embed_dim] ISW embeddings
            latents: [n_days, latent_dim] model latent states
            dates: Date strings

        Returns:
            Dict mapping event date to trajectory data
        """
        print("\nEvent-Triggered Response Analysis:")

        date_to_idx = {d: i for i, d in enumerate(dates)}
        dates_dt = pd.to_datetime(dates)

        results = {}

        for event_date_str, event_info in self.events.items():
            event_name = event_info['name']
            event_dt = pd.to_datetime(event_date_str)

            # Check if event is in our date range
            if event_dt < dates_dt.min() or event_dt > dates_dt.max():
                print(f"  {event_name}: Outside date range")
                continue

            # Find closest date index
            date_diffs = np.abs((dates_dt - event_dt).days)
            center_idx = date_diffs.argmin()

            # Extract window
            start_idx = max(0, center_idx - self.window_before)
            end_idx = min(len(dates), center_idx + self.window_after + 1)

            window_dates = dates[start_idx:end_idx]
            window_embeddings = embeddings[start_idx:end_idx]
            window_latents = latents[start_idx:end_idx] if latents is not None else None

            # Compute metrics
            # 1. Embedding change (pre vs post)
            pre_embedding = embeddings[start_idx:center_idx].mean(axis=0)
            post_embedding = embeddings[center_idx:end_idx].mean(axis=0)

            # Cosine distance
            cos_dist = 1 - np.dot(pre_embedding, post_embedding) / (
                np.linalg.norm(pre_embedding) * np.linalg.norm(post_embedding) + 1e-8
            )

            # Euclidean distance (normalized)
            euc_dist = np.linalg.norm(post_embedding - pre_embedding) / (np.linalg.norm(pre_embedding) + 1e-8)

            # 2. Latent change (if available)
            latent_cos_dist = None
            latent_euc_dist = None
            if window_latents is not None and len(window_latents) > 0:
                relative_center = center_idx - start_idx
                pre_latent = window_latents[:relative_center].mean(axis=0) if relative_center > 0 else window_latents[0]
                post_latent = window_latents[relative_center:].mean(axis=0) if relative_center < len(window_latents) else window_latents[-1]

                latent_cos_dist = float(1 - np.dot(pre_latent, post_latent) / (
                    np.linalg.norm(pre_latent) * np.linalg.norm(post_latent) + 1e-8
                ))
                latent_euc_dist = float(np.linalg.norm(post_latent - pre_latent) / (np.linalg.norm(pre_latent) + 1e-8))

            # 3. Embedding velocity (rate of change around event)
            if len(window_embeddings) > 2:
                embedding_diffs = np.diff(window_embeddings, axis=0)
                embedding_velocities = np.linalg.norm(embedding_diffs, axis=1)
                peak_velocity_idx = embedding_velocities.argmax()
                peak_velocity = float(embedding_velocities.max())
            else:
                peak_velocity = 0.0
                peak_velocity_idx = 0

            results[event_date_str] = {
                'event_name': event_name,
                'event_type': event_info.get('type', 'unknown'),
                'window_dates': window_dates,
                'embedding_cosine_distance': float(cos_dist),
                'embedding_euclidean_distance': float(euc_dist),
                'latent_cosine_distance': latent_cos_dist,
                'latent_euclidean_distance': latent_euc_dist,
                'peak_velocity': peak_velocity,
                'peak_velocity_day': window_dates[peak_velocity_idx] if window_dates else None,
                'n_days_captured': len(window_dates),
            }

            print(f"  {event_name} ({event_date_str}):")
            print(f"    Embedding shift: cos={cos_dist:.4f}, euc={euc_dist:.4f}")
            if latent_cos_dist is not None:
                print(f"    Latent shift: cos={latent_cos_dist:.4f}, euc={latent_euc_dist:.4f}")

        return results

    def compute_event_response_strength(
        self,
        event_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Summarize event response patterns across all events.

        Returns:
            Summary statistics and rankings
        """
        if not event_results:
            return {'error': 'No event results available'}

        # Rank events by embedding shift
        ranked = sorted(
            event_results.items(),
            key=lambda x: x[1]['embedding_cosine_distance'],
            reverse=True
        )

        # Compute baseline (random shifts at same window size)
        shifts = [v['embedding_cosine_distance'] for v in event_results.values()]

        return {
            'strongest_response': {
                'date': ranked[0][0],
                'name': ranked[0][1]['event_name'],
                'cosine_distance': ranked[0][1]['embedding_cosine_distance'],
            },
            'weakest_response': {
                'date': ranked[-1][0],
                'name': ranked[-1][1]['event_name'],
                'cosine_distance': ranked[-1][1]['embedding_cosine_distance'],
            },
            'mean_response': float(np.mean(shifts)),
            'std_response': float(np.std(shifts)),
            'n_events_analyzed': len(event_results),
        }


class LagAnalysisProbe:
    """
    5.2.2 Narrative-Numerical Lag Analysis

    Cross-correlation at lags [-7 to +7] days to detect
    lead/lag relationships between ISW narrative and numerical data.

    Questions answered:
    - Does ISW lead numerical signals (narrative anticipates events)?
    - Does numerical data lead ISW (ISW reports on past events)?
    - What is the optimal lag for maximum correlation?
    """

    def __init__(
        self,
        max_lag: int = 7,
    ):
        self.max_lag = max_lag
        self.results = {}

    def compute_cross_correlation(
        self,
        isw_signal: np.ndarray,
        numerical_signal: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Any]:
        """
        Compute cross-correlation between ISW and numerical signals.

        Args:
            isw_signal: 1D ISW-derived signal (e.g., first PC of embeddings)
            numerical_signal: 1D numerical signal
            dates: Date strings

        Returns:
            Cross-correlation at each lag and optimal lag
        """
        # Ensure same length
        min_len = min(len(isw_signal), len(numerical_signal))
        isw_signal = isw_signal[:min_len]
        numerical_signal = numerical_signal[:min_len]

        # Standardize
        isw_std = (isw_signal - np.mean(isw_signal)) / (np.std(isw_signal) + 1e-8)
        num_std = (numerical_signal - np.mean(numerical_signal)) / (np.std(numerical_signal) + 1e-8)

        # Compute cross-correlation at each lag
        lags = range(-self.max_lag, self.max_lag + 1)
        correlations = []

        for lag in lags:
            if lag < 0:
                # Negative lag: ISW leads (ISW at t predicts numerical at t+|lag|)
                corr = np.corrcoef(isw_std[:lag], num_std[-lag:])[0, 1]
            elif lag > 0:
                # Positive lag: Numerical leads (numerical at t predicts ISW at t+lag)
                corr = np.corrcoef(isw_std[lag:], num_std[:-lag])[0, 1]
            else:
                # Zero lag
                corr = np.corrcoef(isw_std, num_std)[0, 1]

            correlations.append(corr if not np.isnan(corr) else 0.0)

        # Find optimal lag
        correlations = np.array(correlations)
        optimal_lag_idx = np.argmax(np.abs(correlations))
        optimal_lag = list(lags)[optimal_lag_idx]
        optimal_corr = correlations[optimal_lag_idx]

        return {
            'lags': list(lags),
            'correlations': correlations.tolist(),
            'optimal_lag': int(optimal_lag),
            'optimal_correlation': float(optimal_corr),
            'zero_lag_correlation': float(correlations[self.max_lag]),
            'interpretation': self._interpret_lag(optimal_lag),
        }

    def _interpret_lag(self, lag: int) -> str:
        """Interpret the meaning of the optimal lag."""
        if lag < 0:
            return f"ISW leads by {abs(lag)} days (narrative anticipates events)"
        elif lag > 0:
            return f"Numerical leads by {lag} days (ISW reports on past)"
        else:
            return "Synchronous (same-day correlation)"

    def analyze_all_sources(
        self,
        isw_embeddings: np.ndarray,
        numerical_sources: Dict[str, np.ndarray],
        dates: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute lag analysis for all numerical sources.

        Args:
            isw_embeddings: [n_days, 1024] ISW embeddings
            numerical_sources: Dict mapping source name to array
            dates: Date strings

        Returns:
            Dict with lag analysis for each source
        """
        print("\nNarrative-Numerical Lag Analysis:")

        # Use first PC of ISW as signal
        pca = PCA(n_components=1)
        isw_signal = pca.fit_transform(isw_embeddings).flatten()

        results = {}

        for source_name, source_data in numerical_sources.items():
            # Use mean across features if multi-dimensional
            if len(source_data.shape) > 1:
                num_signal = source_data.mean(axis=1)
            else:
                num_signal = source_data

            lag_result = self.compute_cross_correlation(isw_signal, num_signal, dates)
            results[source_name] = lag_result

            print(f"  {source_name}:")
            print(f"    Optimal lag: {lag_result['optimal_lag']} ({lag_result['interpretation']})")
            print(f"    Correlation at optimal lag: {lag_result['optimal_correlation']:.4f}")
            print(f"    Zero-lag correlation: {lag_result['zero_lag_correlation']:.4f}")

        self.results = results
        return results


class SemanticAnomalyProbe:
    """
    5.2.3 Semantic Anomaly Detection

    Compare numerical anomaly days vs ISW embedding outliers.

    Questions:
    - Do embedding outliers correspond to numerical anomalies?
    - Are there "quiet" numerical days with unusual narratives (and vice versa)?
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,  # Z-score threshold
    ):
        self.anomaly_threshold = anomaly_threshold

    def detect_embedding_anomalies(
        self,
        embeddings: np.ndarray,
        dates: List[str],
    ) -> Dict[str, Any]:
        """
        Detect anomalous embeddings using distance from rolling mean.

        Args:
            embeddings: [n_days, embed_dim]
            dates: Date strings

        Returns:
            Dict with anomaly detection results
        """
        n_days = len(embeddings)

        # Compute rolling mean embedding (7-day window)
        window = 7
        anomaly_scores = np.zeros(n_days)

        for i in range(n_days):
            start = max(0, i - window // 2)
            end = min(n_days, i + window // 2 + 1)

            # Exclude current day from mean
            indices = [j for j in range(start, end) if j != i]
            if indices:
                local_mean = embeddings[indices].mean(axis=0)

                # Cosine distance from local mean
                cos_dist = 1 - np.dot(embeddings[i], local_mean) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(local_mean) + 1e-8
                )
                anomaly_scores[i] = cos_dist

        # Standardize scores
        z_scores = (anomaly_scores - np.mean(anomaly_scores)) / (np.std(anomaly_scores) + 1e-8)

        # Identify anomalies
        anomaly_mask = np.abs(z_scores) > self.anomaly_threshold
        anomaly_dates = [dates[i] for i in range(n_days) if anomaly_mask[i]]

        return {
            'anomaly_scores': anomaly_scores,
            'z_scores': z_scores,
            'anomaly_dates': anomaly_dates,
            'n_anomalies': int(anomaly_mask.sum()),
            'anomaly_fraction': float(anomaly_mask.sum() / n_days),
        }

    def compare_anomalies(
        self,
        embedding_anomalies: Dict[str, Any],
        numerical_anomalies: Dict[str, List[str]],
        dates: List[str],
    ) -> Dict[str, Any]:
        """
        Compare embedding anomalies with numerical anomalies.

        Args:
            embedding_anomalies: Output from detect_embedding_anomalies
            numerical_anomalies: Dict mapping source to list of anomaly dates
            dates: All dates

        Returns:
            Comparison metrics
        """
        embed_anomaly_set = set(embedding_anomalies['anomaly_dates'])
        all_dates_set = set(dates)

        comparisons = {}

        for source_name, num_anomaly_dates in numerical_anomalies.items():
            num_anomaly_set = set(num_anomaly_dates)

            # Compute overlap
            overlap = embed_anomaly_set & num_anomaly_set
            embed_only = embed_anomaly_set - num_anomaly_set
            num_only = num_anomaly_set - embed_anomaly_set

            # Jaccard similarity
            if embed_anomaly_set | num_anomaly_set:
                jaccard = len(overlap) / len(embed_anomaly_set | num_anomaly_set)
            else:
                jaccard = 0.0

            comparisons[source_name] = {
                'overlap_count': len(overlap),
                'overlap_dates': list(overlap)[:10],  # Limit for readability
                'embedding_only_count': len(embed_only),
                'numerical_only_count': len(num_only),
                'jaccard_similarity': float(jaccard),
            }

        return {
            'source_comparisons': comparisons,
            'total_embedding_anomalies': len(embed_anomaly_set),
        }


# =============================================================================
# 5.3 COUNTERFACTUAL SEMANTIC PROBING
# =============================================================================

class CounterfactualProbe:
    """
    5.3.1 Semantic Perturbation Effects

    Replace ISW(t) with a different day's ISW embedding and measure
    how much model predictions change.

    This tests whether the model actually uses ISW content or just
    relies on numerical features.
    """

    def __init__(
        self,
        n_perturbations: int = 100,
    ):
        self.n_perturbations = n_perturbations

    def compute_perturbation_effects(
        self,
        isw_embeddings: np.ndarray,
        latent_representations: np.ndarray,
        dates: List[str],
        model_predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Measure prediction change when ISW embedding is swapped.

        Args:
            isw_embeddings: [n_days, embed_dim]
            latent_representations: [n_days, latent_dim]
            dates: Date strings
            model_predictions: [n_days] or [n_days, n_targets] model outputs

        Returns:
            Perturbation effect statistics
        """
        print("\nCounterfactual Semantic Perturbation Analysis:")

        n_days = len(isw_embeddings)

        # For each sample, swap ISW with a random different day
        perturbation_effects = []

        np.random.seed(42)
        sample_indices = np.random.choice(n_days, min(self.n_perturbations, n_days), replace=False)

        for idx in sample_indices:
            # Select a random different day
            other_idx = np.random.choice([i for i in range(n_days) if i != idx])

            # Compute embedding similarity before/after swap
            original_embedding = isw_embeddings[idx]
            swapped_embedding = isw_embeddings[other_idx]

            # Cosine distance between original and swapped
            cos_dist = 1 - np.dot(original_embedding, swapped_embedding) / (
                np.linalg.norm(original_embedding) * np.linalg.norm(swapped_embedding) + 1e-8
            )

            # If latents available, compute how much latent would change
            # (This is a proxy - full effect requires model forward pass)
            latent_change = 0.0
            if latent_representations is not None:
                # Estimate latent change based on ISW contribution
                # This is approximate without actual model forward pass
                original_latent = latent_representations[idx]
                swapped_latent = latent_representations[other_idx]
                latent_change = 1 - np.dot(original_latent, swapped_latent) / (
                    np.linalg.norm(original_latent) * np.linalg.norm(swapped_latent) + 1e-8
                )

            # Temporal distance between original and swap day
            orig_date = pd.to_datetime(dates[idx])
            swap_date = pd.to_datetime(dates[other_idx])
            temporal_dist = abs((orig_date - swap_date).days)

            perturbation_effects.append({
                'original_idx': int(idx),
                'swapped_idx': int(other_idx),
                'temporal_distance': int(temporal_dist),
                'embedding_cosine_distance': float(cos_dist),
                'latent_change_estimate': float(latent_change),
            })

        # Analyze effects
        cos_dists = [p['embedding_cosine_distance'] for p in perturbation_effects]
        latent_changes = [p['latent_change_estimate'] for p in perturbation_effects]
        temp_dists = [p['temporal_distance'] for p in perturbation_effects]

        # Correlation between embedding distance and latent change
        if latent_representations is not None:
            distance_latent_corr = np.corrcoef(cos_dists, latent_changes)[0, 1]
        else:
            distance_latent_corr = None

        results = {
            'n_perturbations': len(perturbation_effects),
            'mean_embedding_distance': float(np.mean(cos_dists)),
            'std_embedding_distance': float(np.std(cos_dists)),
            'mean_latent_change': float(np.mean(latent_changes)),
            'distance_latent_correlation': float(distance_latent_corr) if distance_latent_corr is not None else None,
            'mean_temporal_distance': float(np.mean(temp_dists)),
            'perturbation_details': perturbation_effects[:20],  # Sample for inspection
        }

        print(f"  Mean embedding distance on swap: {results['mean_embedding_distance']:.4f}")
        print(f"  Mean latent change estimate: {results['mean_latent_change']:.4f}")
        if distance_latent_corr is not None:
            print(f"  Embedding-Latent correlation: {distance_latent_corr:.4f}")

        return results


class SemanticPredictorProbe:
    """
    5.3.2 Missing Semantic Interpolation

    Train a Numerical_latent -> ISW_embedding predictor.

    This tests if numerical features contain enough information
    to reconstruct semantic content (and vice versa).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        test_ratio: float = 0.2,
    ):
        self.hidden_dim = hidden_dim
        self.test_ratio = test_ratio
        self.model = None

    def train_latent_to_isw_predictor(
        self,
        latent_representations: np.ndarray,
        isw_embeddings: np.ndarray,
        dates: List[str],
        target_dim: int = 128,  # Reduce ISW to this before prediction
    ) -> Dict[str, Any]:
        """
        Train predictor: Numerical_latent -> ISW_embedding.

        Args:
            latent_representations: [n_days, latent_dim] model latents
            isw_embeddings: [n_days, 1024] ISW embeddings
            dates: Date strings
            target_dim: Reduce ISW to this dimension

        Returns:
            Prediction metrics
        """
        print("\nSemantic Interpolation Predictor (Latent -> ISW):")

        # Reduce ISW target dimension
        pca = PCA(n_components=target_dim)
        isw_reduced = pca.fit_transform(isw_embeddings)
        print(f"  Reduced ISW to {target_dim} dims ({sum(pca.explained_variance_ratio_)*100:.1f}% var)")

        # Ensure same length
        min_len = min(len(latent_representations), len(isw_reduced))
        X = latent_representations[:min_len]
        y = isw_reduced[:min_len]

        # Temporal train/test split
        split_idx = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale inputs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train MLP regressor
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        # Per-dimension R2
        r2_per_dim = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(target_dim)]
        mean_r2 = float(np.mean(r2_per_dim))

        # Reconstruction cosine similarity
        cos_sims = []
        for i in range(len(y_test)):
            cos_sim = np.dot(y_test[i], y_pred[i]) / (
                np.linalg.norm(y_test[i]) * np.linalg.norm(y_pred[i]) + 1e-8
            )
            cos_sims.append(cos_sim)
        mean_cos_sim = float(np.mean(cos_sims))

        results = {
            'target_dim': target_dim,
            'mean_r2': mean_r2,
            'median_r2': float(np.median(r2_per_dim)),
            'mean_reconstruction_cosine_similarity': mean_cos_sim,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'best_r2_dimension': int(np.argmax(r2_per_dim)),
            'worst_r2_dimension': int(np.argmin(r2_per_dim)),
        }

        print(f"  Mean R2 across dimensions: {mean_r2:.4f}")
        print(f"  Mean cosine similarity: {mean_cos_sim:.4f}")
        print(f"  Best/worst dim R2: {max(r2_per_dim):.4f} / {min(r2_per_dim):.4f}")

        return results

    def train_isw_to_latent_predictor(
        self,
        isw_embeddings: np.ndarray,
        latent_representations: np.ndarray,
        dates: List[str],
        isw_reduced_dim: int = 128,
    ) -> Dict[str, Any]:
        """
        Train predictor: ISW_embedding -> Numerical_latent.

        This is the inverse direction - can we predict latent from narrative?
        """
        print("\nSemantic Interpolation Predictor (ISW -> Latent):")

        # Reduce ISW input dimension
        pca = PCA(n_components=isw_reduced_dim)
        isw_reduced = pca.fit_transform(isw_embeddings)

        # Setup
        min_len = min(len(isw_reduced), len(latent_representations))
        X = isw_reduced[:min_len]
        y = latent_representations[:min_len]

        split_idx = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        latent_dim = y_test.shape[1]
        r2_per_dim = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(latent_dim)]
        mean_r2 = float(np.mean(r2_per_dim))

        cos_sims = []
        for i in range(len(y_test)):
            cos_sim = np.dot(y_test[i], y_pred[i]) / (
                np.linalg.norm(y_test[i]) * np.linalg.norm(y_pred[i]) + 1e-8
            )
            cos_sims.append(cos_sim)

        results = {
            'isw_input_dim': isw_reduced_dim,
            'latent_output_dim': latent_dim,
            'mean_r2': mean_r2,
            'mean_reconstruction_cosine_similarity': float(np.mean(cos_sims)),
            'n_train': len(X_train),
            'n_test': len(X_test),
        }

        print(f"  Mean R2 across latent dims: {mean_r2:.4f}")
        print(f"  Mean cosine similarity: {np.mean(cos_sims):.4f}")

        return results


# =============================================================================
# 5.4 SEMANTIC ENRICHMENT POTENTIAL (SPECIFICATION)
# =============================================================================

SEMANTIC_ENRICHMENT_SPEC = """
5.4 SEMANTIC ENRICHMENT POTENTIAL

This section documents requirements for future semantic data integration.

===============================================================================
5.4.1 Telegram Channel Integration
===============================================================================

Purpose: Real-time tactical updates from verified channels.

Required Data:
- Channel IDs for verified military/OSINT channels
- Message content (text, captions)
- Timestamps (UTC)
- Media attachments (for geotagging)
- Message reactions/views (engagement proxy)

Processing Pipeline:
1. Channel selection based on verification status
2. Language detection and translation (Ukrainian, Russian -> English)
3. Named entity extraction (locations, equipment, units)
4. Sentiment/intensity scoring
5. Deduplication across channels
6. Temporal alignment with ISW dates

Embedding Strategy:
- Per-message embeddings (voyage-4-large or similar)
- Daily aggregation via attention pooling
- Separate channel-type embeddings (UA mil, RU mil, OSINT)

Integration Points:
- Concatenate with ISW embeddings before projection
- Use cross-attention between ISW and Telegram streams
- Weight by channel credibility score

Estimated Impact:
- Latency: ~6-12 hours faster than ISW
- Coverage: More tactical detail
- Risk: Misinformation, propaganda

===============================================================================
5.4.2 Combat Footage Metadata
===============================================================================

Purpose: Visual confirmation of events, equipment losses, tactical patterns.

Required Data:
- Video/image metadata from aggregators (Oryx, etc.)
- Geolocation (if extractable)
- Timestamp
- Equipment type labels
- Outcome labels (destroyed, damaged, captured, abandoned)
- Source attribution

Processing Pipeline:
1. Metadata collection from verified sources
2. Geolocation clustering by region
3. Equipment categorization alignment with Oryx taxonomy
4. Temporal aggregation to daily counts
5. Cross-reference with ISW mentions

Embedding Strategy:
- Categorical features (equipment types, outcomes)
- Geographic grid cell indicators
- No direct visual embedding (metadata only)

Integration Points:
- Use as additional numerical source in daily fusion
- Correlate with ISW equipment mentions
- Validate equipment loss reports

Estimated Impact:
- Ground truth for equipment claims
- Regional activity indicators
- Verification layer for reports

===============================================================================
5.4.3 Official Statement Encoding
===============================================================================

Purpose: Track official narratives from Ukrainian/Russian MoD, political leaders.

Required Data:
- Official press releases
- MoD briefings
- Presidential statements
- Social media from verified officials
- Diplomatic statements

Processing Pipeline:
1. Source attribution (UA gov, RU gov, third party)
2. Statement type classification (operational, political, diplomatic)
3. Claim extraction (territorial, casualty, equipment)
4. Sentiment/stance detection
5. Comparison with other sources (claim verification)

Embedding Strategy:
- Per-statement embeddings with source prefix
- Separate embedding streams per source type
- Contradiction detection between sources

Integration Points:
- Cross-attention with ISW for context
- Use as regime/escalation indicators
- Political event detection

Estimated Impact:
- Political context for military events
- Escalation/de-escalation signals
- Narrative divergence detection

===============================================================================
Implementation Priority:
===============================================================================

1. Official Statements (lowest effort, highest signal)
   - Structured sources
   - Clear attribution
   - Direct ISW complement

2. Combat Footage Metadata (medium effort, high signal)
   - Requires aggregator APIs
   - Ground truth value
   - Equipment focus

3. Telegram Channels (highest effort, variable signal)
   - Scale challenges
   - Misinformation filtering
   - Real-time value

===============================================================================
"""


# =============================================================================
# MAIN PROBE RUNNER
# =============================================================================

class SemanticAssociationProbeRunner:
    """
    Unified runner for all semantic-numerical association probes.

    Usage:
        runner = SemanticAssociationProbeRunner()
        results = runner.run_all_probes(
            isw_data=ISWEmbeddingData.load(),
            latent_data=np.load('latents.npy'),
            numerical_sources={'equipment': equip_data, ...},
        )
    """

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize probes
        self.alignment_probe = ISWAlignmentProbe()
        self.topic_probe = TopicExtractionProbe()
        self.predictive_probe = ISWPredictiveContentProbe()
        self.event_probe = EventResponseProbe()
        self.lag_probe = LagAnalysisProbe()
        self.anomaly_probe = SemanticAnomalyProbe()
        self.counterfactual_probe = CounterfactualProbe()
        self.semantic_predictor = SemanticPredictorProbe()

        self.results = {}

    def run_all_probes(
        self,
        isw_data: ISWEmbeddingData,
        latent_data: Optional[np.ndarray] = None,
        numerical_sources: Optional[Dict[str, np.ndarray]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all semantic association probes.

        Args:
            isw_data: Loaded ISW embedding data
            latent_data: [n_days, latent_dim] model latent representations
            numerical_sources: Dict mapping source name to array
            save_results: Whether to save results to disk

        Returns:
            Dict with all probe results
        """
        print("=" * 70)
        print("SEMANTIC-NUMERICAL ASSOCIATION PROBE SUITE")
        print("=" * 70)

        embeddings = isw_data.embeddings
        dates = isw_data.dates

        # 5.1.1 ISW-Latent Correlation
        print("\n" + "-" * 70)
        print("5.1.1 ISW-Latent Correlation")
        print("-" * 70)

        if latent_data is not None:
            self.alignment_probe.fit_projection(embeddings)
            alignment_results = self.alignment_probe.compute_daily_alignment(
                embeddings, latent_data, dates
            )
            period_results = self.alignment_probe.analyze_alignment_by_period(
                alignment_results['daily_similarities'], dates
            )
            self.results['isw_latent_correlation'] = {
                'alignment': alignment_results,
                'by_period': period_results,
            }
        else:
            print("  Skipped: No latent data provided")
            self.results['isw_latent_correlation'] = {'skipped': 'No latent data'}

        # 5.1.2 Topic-Source Correlation
        print("\n" + "-" * 70)
        print("5.1.2 ISW Topic-Source Correlation")
        print("-" * 70)

        topic_results = self.topic_probe.extract_topics(embeddings, dates)

        if numerical_sources:
            topic_source_corr = self.topic_probe.compute_topic_source_correlation(
                topic_results, numerical_sources, dates
            )
            self.results['topic_source_correlation'] = {
                'topics': topic_results,
                'source_correlations': topic_source_corr,
            }
        else:
            self.results['topic_source_correlation'] = {
                'topics': topic_results,
                'source_correlations': None,
            }

        # 5.1.3 ISW Predictive Content
        print("\n" + "-" * 70)
        print("5.1.3 ISW Predictive Content Test")
        print("-" * 70)

        if numerical_sources:
            predictive_results = self.predictive_probe.compute_predictive_power(
                embeddings, numerical_sources, dates
            )
            self.results['isw_predictive_content'] = predictive_results
        else:
            print("  Skipped: No numerical sources provided")
            self.results['isw_predictive_content'] = {'skipped': 'No numerical sources'}

        # 5.2.1 Event-Triggered Response
        print("\n" + "-" * 70)
        print("5.2.1 Event-Triggered Response Analysis")
        print("-" * 70)

        event_trajectories = self.event_probe.extract_event_trajectories(
            embeddings, latent_data, dates
        )
        event_summary = self.event_probe.compute_event_response_strength(event_trajectories)
        self.results['event_response'] = {
            'trajectories': event_trajectories,
            'summary': event_summary,
        }

        # 5.2.2 Lag Analysis
        print("\n" + "-" * 70)
        print("5.2.2 Narrative-Numerical Lag Analysis")
        print("-" * 70)

        if numerical_sources:
            lag_results = self.lag_probe.analyze_all_sources(
                embeddings, numerical_sources, dates
            )
            self.results['lag_analysis'] = lag_results
        else:
            print("  Skipped: No numerical sources provided")
            self.results['lag_analysis'] = {'skipped': 'No numerical sources'}

        # 5.2.3 Semantic Anomaly Detection
        print("\n" + "-" * 70)
        print("5.2.3 Semantic Anomaly Detection")
        print("-" * 70)

        embedding_anomalies = self.anomaly_probe.detect_embedding_anomalies(
            embeddings, dates
        )
        print(f"  Detected {embedding_anomalies['n_anomalies']} embedding anomalies "
              f"({embedding_anomalies['anomaly_fraction']*100:.1f}%)")
        self.results['semantic_anomalies'] = embedding_anomalies

        # 5.3.1 Counterfactual Perturbation
        print("\n" + "-" * 70)
        print("5.3.1 Counterfactual Semantic Perturbation")
        print("-" * 70)

        counterfactual_results = self.counterfactual_probe.compute_perturbation_effects(
            embeddings, latent_data, dates
        )
        self.results['counterfactual'] = counterfactual_results

        # 5.3.2 Semantic Interpolation
        print("\n" + "-" * 70)
        print("5.3.2 Missing Semantic Interpolation")
        print("-" * 70)

        if latent_data is not None:
            latent_to_isw = self.semantic_predictor.train_latent_to_isw_predictor(
                latent_data, embeddings, dates
            )
            isw_to_latent = self.semantic_predictor.train_isw_to_latent_predictor(
                embeddings, latent_data, dates
            )
            self.results['semantic_interpolation'] = {
                'latent_to_isw': latent_to_isw,
                'isw_to_latent': isw_to_latent,
            }
        else:
            print("  Skipped: No latent data provided")
            self.results['semantic_interpolation'] = {'skipped': 'No latent data'}

        # Save results
        if save_results:
            self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj

        output_path = self.output_dir / 'semantic_association_results.json'
        with open(output_path, 'w') as f:
            json.dump(make_serializable(self.results), f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def _print_summary(self):
        """Print summary of all probe results."""
        print("\n" + "=" * 70)
        print("SEMANTIC ASSOCIATION PROBE SUMMARY")
        print("=" * 70)

        # 5.1.1 Alignment
        if 'alignment' in self.results.get('isw_latent_correlation', {}):
            align = self.results['isw_latent_correlation']['alignment']
            print(f"\n5.1.1 ISW-Latent Alignment:")
            print(f"      Mean cosine similarity: {align['mean_similarity']:.4f}")

        # 5.1.2 Topics
        if 'topics' in self.results.get('topic_source_correlation', {}):
            topics = self.results['topic_source_correlation']['topics']
            print(f"\n5.1.2 Topic Extraction:")
            print(f"      Found {topics['n_topics']} topics (silhouette: {topics.get('silhouette_score', 0):.4f})")

        # 5.1.3 Predictive
        if not self.results.get('isw_predictive_content', {}).get('skipped'):
            pred = self.results['isw_predictive_content']
            if pred:
                best_target = max(pred.items(), key=lambda x: x[1].get('r2_score', 0))
                print(f"\n5.1.3 ISW Predictive Power:")
                print(f"      Best target: {best_target[0]} (R2={best_target[1]['r2_score']:.4f})")

        # 5.2.1 Events
        if 'summary' in self.results.get('event_response', {}):
            events = self.results['event_response']['summary']
            print(f"\n5.2.1 Event Response:")
            print(f"      Strongest: {events['strongest_response']['name']}")
            print(f"      Mean response: {events['mean_response']:.4f}")

        # 5.2.2 Lag
        if not self.results.get('lag_analysis', {}).get('skipped'):
            lag = self.results['lag_analysis']
            if lag:
                print(f"\n5.2.2 Lag Analysis:")
                for src, data in list(lag.items())[:3]:
                    print(f"      {src}: optimal lag={data['optimal_lag']} (r={data['optimal_correlation']:.4f})")

        # 5.3.2 Interpolation
        if 'latent_to_isw' in self.results.get('semantic_interpolation', {}):
            interp = self.results['semantic_interpolation']
            print(f"\n5.3.2 Semantic Interpolation:")
            print(f"      Latent->ISW R2: {interp['latent_to_isw']['mean_r2']:.4f}")
            print(f"      ISW->Latent R2: {interp['isw_to_latent']['mean_r2']:.4f}")

        print("\n" + "=" * 70)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SEMANTIC ASSOCIATION PROBES - STANDALONE TEST")
    print("=" * 70)

    # Load ISW embeddings
    try:
        isw_data = ISWEmbeddingData.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure ISW embeddings are available.")
        exit(1)

    # Create synthetic latent data for testing
    n_days = len(isw_data.dates)
    latent_dim = 128

    # Project ISW to latent space (simulate model latent)
    pca = PCA(n_components=latent_dim)
    synthetic_latent = pca.fit_transform(isw_data.embeddings)

    # Add some noise to make it different from direct projection
    synthetic_latent = synthetic_latent + np.random.randn(*synthetic_latent.shape) * 0.1

    print(f"\nCreated synthetic latent data: {synthetic_latent.shape}")

    # Create synthetic numerical sources
    numerical_sources = {
        'equipment_delta': np.random.randn(n_days) * 10 + 50,
        'firms_count': np.random.exponential(100, n_days),
        'casualty_delta': np.random.randn(n_days) * 100 + 500,
    }

    print(f"Created synthetic numerical sources: {list(numerical_sources.keys())}")

    # Run probes
    runner = SemanticAssociationProbeRunner()
    results = runner.run_all_probes(
        isw_data=isw_data,
        latent_data=synthetic_latent,
        numerical_sources=numerical_sources,
        save_results=True,
    )

    print("\n" + "=" * 70)
    print("PROBE SUITE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {runner.output_dir}")
    print(f"\nSemantic Enrichment Specification available in module:")
    print("  - semantic_association_probes.SEMANTIC_ENRICHMENT_SPEC")
