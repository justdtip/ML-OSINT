#!/usr/bin/env python3
"""
Semantic Structure Probing for Multi-Resolution HAN

This module implements probing classifiers and clustering analyses to investigate
whether the Multi-Resolution HAN model encodes implicit semantic structure in its
latent representations, despite being trained only on numerical data.

Probe Categories:
=================

4.1 Implicit Semantic Categories
--------------------------------
4.1.1 OperationClusteringProbe: Test if named military operations cluster separately
4.1.2 DayTypeDecodingProbe: Linear probe to classify day types from frozen latent
4.1.3 IntensityProbe: Decode intensity levels from latent representations
4.1.4 GeographicFocusProbe: Decode primary theater from latent representations

4.2 Temporal Semantic Patterns
------------------------------
4.2.1 WeeklyCycleProbe: ANOVA for weekday effects on latent states
4.2.2 SeasonalPatternProbe: Test for monthly/seasonal patterns
4.2.3 EventAnniversaryProbe: Test for "time since major event" encoding

Key Design Principles:
- All probes operate on FROZEN latent representations (no backprop to model)
- Linear probes ensure we're testing for linear separability (interpretable)
- Clustering uses silhouette score and within/between variance ratios
- Statistical tests (ANOVA, etc.) use scipy.stats with p < 0.05 threshold
- Label construction helpers derive labels from numerical data when needed

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR as CONFIG_ANALYSIS_DIR,
    PROBE_OUTPUT_DIR,
)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_DIR = PROJECT_ROOT
ANALYSIS_DIR = CONFIG_ANALYSIS_DIR
OUTPUT_DIR = PROBE_OUTPUT_DIR

# Military operations with date ranges (start, end inclusive)
MILITARY_OPERATIONS = {
    'kyiv_offensive': {
        'name': 'Kyiv Offensive',
        'start': '2022-02-24',
        'end': '2022-04-02',
        'description': 'Initial Russian advance on Kyiv and subsequent withdrawal',
    },
    'kharkiv_counteroffensive': {
        'name': 'Kharkiv Counteroffensive',
        'start': '2022-09-06',
        'end': '2022-09-16',
        'description': 'Ukrainian counteroffensive liberating Kharkiv Oblast',
    },
    'kherson_counteroffensive': {
        'name': 'Kherson Counteroffensive',
        'start': '2022-10-01',
        'end': '2022-11-11',
        'description': 'Ukrainian counteroffensive liberating Kherson',
    },
    'bakhmut_offensive': {
        'name': 'Bakhmut Offensive',
        'start': '2022-08-01',
        'end': '2023-05-20',
        'description': 'Russian Wagner Group assault on Bakhmut',
    },
    'counteroffensive_2023': {
        'name': '2023 Counteroffensive',
        'start': '2023-06-04',
        'end': '2023-10-31',
        'description': 'Ukrainian 2023 summer counteroffensive',
    },
    'avdiivka_battle': {
        'name': 'Avdiivka Battle',
        'start': '2023-10-10',
        'end': '2024-02-17',
        'description': 'Russian assault and capture of Avdiivka',
    },
    'kursk_incursion': {
        'name': 'Kursk Incursion',
        'start': '2024-08-06',
        'end': None,  # Ongoing as of data cutoff
        'description': 'Ukrainian incursion into Kursk Oblast',
    },
}

# Day type categories
DAY_TYPES = {
    'major_strike': 'Major Strike Day',
    'ground_assault': 'Ground Assault Day',
    'counterattack': 'Counterattack Day',
    'quiet': 'Quiet Day',
    'infrastructure': 'Infrastructure Attack Day',
}

# Intensity levels
INTENSITY_LEVELS = {
    0: 'Low',       # Bottom quartile
    1: 'Medium',    # Middle 50%
    2: 'High',      # Top quartile (75-95%)
    3: 'Extreme',   # Top 5%
}

# Geographic theaters
GEOGRAPHIC_THEATERS = {
    0: 'Eastern Front',
    1: 'Southern Front',
    2: 'Multi-front',
    3: 'Strategic Depth',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_output_dir() -> Path:
    """Create output directory for probe results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def date_to_operation(date: Union[str, datetime, np.datetime64]) -> Optional[str]:
    """
    Map a date to a military operation if it falls within an operation's date range.

    Args:
        date: Date to check

    Returns:
        Operation key if date is within an operation, None otherwise
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif isinstance(date, np.datetime64):
        date = pd.to_datetime(date)
    else:
        # Handle non-datetime inputs (e.g., integer indices used as fallback)
        try:
            date = pd.to_datetime(date)
        except (ValueError, TypeError):
            return None

    for op_key, op_info in MILITARY_OPERATIONS.items():
        start = pd.to_datetime(op_info['start'])
        end = pd.to_datetime(op_info['end']) if op_info['end'] else pd.Timestamp.now()

        if start <= date <= end:
            return op_key

    return None


def compute_variance_ratio(latents: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the within-group vs between-group variance ratio.

    Higher values indicate better clustering (more separation between groups).
    This is equivalent to the F-statistic in one-way ANOVA.

    Args:
        latents: Latent representations [n_samples, d_model]
        labels: Cluster/group labels [n_samples]

    Returns:
        Variance ratio (between-group variance / within-group variance)
    """
    unique_labels = np.unique(labels)
    n_groups = len(unique_labels)
    n_samples = len(labels)

    if n_groups < 2:
        return 0.0

    # Global mean
    global_mean = latents.mean(axis=0)

    # Between-group variance (weighted by group size)
    between_var = 0.0
    for label in unique_labels:
        mask = labels == label
        group_mean = latents[mask].mean(axis=0)
        n_group = mask.sum()
        between_var += n_group * np.sum((group_mean - global_mean) ** 2)
    between_var /= (n_groups - 1)

    # Within-group variance
    within_var = 0.0
    for label in unique_labels:
        mask = labels == label
        group_mean = latents[mask].mean(axis=0)
        within_var += np.sum((latents[mask] - group_mean) ** 2)
    within_var /= (n_samples - n_groups)

    if within_var < 1e-10:
        return float('inf')

    return between_var / within_var


# =============================================================================
# LABEL CONSTRUCTION HELPERS
# =============================================================================

class LabelConstructor:
    """
    Construct semantic labels from numerical data when external labels unavailable.

    This class derives labels for day types, intensity levels, and geographic focus
    from the raw numerical features (casualties, equipment losses, etc.).
    """

    def __init__(
        self,
        casualty_data: Optional[np.ndarray] = None,
        equipment_data: Optional[np.ndarray] = None,
        territorial_data: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
    ):
        """
        Initialize with available data arrays.

        Args:
            casualty_data: Daily casualty figures [n_days, n_casualty_features]
            equipment_data: Daily equipment losses [n_days, n_equipment_features]
            territorial_data: Daily territorial changes [n_days, n_territory_features]
            dates: Corresponding dates [n_days]
        """
        self.casualty_data = casualty_data
        self.equipment_data = equipment_data
        self.territorial_data = territorial_data
        self.dates = dates

        # Compute statistics if data is provided
        if casualty_data is not None:
            self._compute_casualty_stats()
        if equipment_data is not None:
            self._compute_equipment_stats()

    def _compute_casualty_stats(self):
        """Compute casualty distribution statistics for thresholding."""
        # Assume first column is total casualties or use sum
        if self.casualty_data.ndim == 1:
            total = self.casualty_data
        else:
            total = self.casualty_data.sum(axis=1)

        self.casualty_percentiles = {
            'p25': np.percentile(total, 25),
            'p50': np.percentile(total, 50),
            'p75': np.percentile(total, 75),
            'p95': np.percentile(total, 95),
        }

    def _compute_equipment_stats(self):
        """Compute equipment loss distribution statistics."""
        if self.equipment_data.ndim == 1:
            total = self.equipment_data
        else:
            total = self.equipment_data.sum(axis=1)

        self.equipment_percentiles = {
            'p25': np.percentile(total, 25),
            'p50': np.percentile(total, 50),
            'p75': np.percentile(total, 75),
            'p95': np.percentile(total, 95),
        }

    def construct_intensity_labels(
        self,
        combined_metric: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construct intensity level labels based on casualty distribution quartiles.

        Levels:
        - 0 (Low): Bottom quartile (<= p25)
        - 1 (Medium): Middle 50% (p25 < x <= p75)
        - 2 (High): Top quartile but not extreme (p75 < x <= p95)
        - 3 (Extreme): Top 5% (> p95)

        Args:
            combined_metric: Optional pre-computed combined intensity metric.
                            If None, derives from casualty_data.

        Returns:
            Array of intensity labels [n_samples]
        """
        if combined_metric is None:
            if self.casualty_data is None:
                raise ValueError("No data available for intensity label construction")

            if self.casualty_data.ndim == 1:
                combined_metric = self.casualty_data
            else:
                combined_metric = self.casualty_data.sum(axis=1)

        # Compute percentiles on the fly if not pre-computed
        p25 = np.percentile(combined_metric, 25)
        p75 = np.percentile(combined_metric, 75)
        p95 = np.percentile(combined_metric, 95)

        labels = np.zeros(len(combined_metric), dtype=np.int64)
        labels[combined_metric > p25] = 1  # Medium
        labels[combined_metric > p75] = 2  # High
        labels[combined_metric > p95] = 3  # Extreme

        return labels

    def construct_day_type_labels(
        self,
        casualty_threshold_high: Optional[float] = None,
        equipment_threshold_high: Optional[float] = None,
    ) -> np.ndarray:
        """
        Construct day type labels heuristically from numerical data.

        Categories:
        - 'quiet': Low casualties AND low equipment losses
        - 'major_strike': High equipment losses (esp. aircraft, missiles)
        - 'ground_assault': High casualties, high AFV/tank losses
        - 'counterattack': Territorial changes favorable to Ukraine
        - 'infrastructure': Missile/drone losses elevated

        Returns:
            Array of day type labels (encoded as integers) [n_samples]
        """
        if self.casualty_data is None or self.equipment_data is None:
            raise ValueError("Both casualty and equipment data needed for day type labels")

        n_samples = len(self.casualty_data)
        labels = np.zeros(n_samples, dtype=np.int64)  # Default: quiet (0)

        # Compute totals
        if self.casualty_data.ndim == 1:
            casualty_total = self.casualty_data
        else:
            casualty_total = self.casualty_data.sum(axis=1)

        if self.equipment_data.ndim == 1:
            equipment_total = self.equipment_data
        else:
            equipment_total = self.equipment_data.sum(axis=1)

        # Thresholds
        if casualty_threshold_high is None:
            casualty_threshold_high = np.percentile(casualty_total, 75)
        if equipment_threshold_high is None:
            equipment_threshold_high = np.percentile(equipment_total, 75)

        casualty_p25 = np.percentile(casualty_total, 25)
        equipment_p25 = np.percentile(equipment_total, 25)

        # Label assignment (priority order)
        # Quiet: both low
        quiet_mask = (casualty_total <= casualty_p25) & (equipment_total <= equipment_p25)
        labels[quiet_mask] = 3  # 'quiet'

        # Ground assault: high casualties
        ground_mask = casualty_total > casualty_threshold_high
        labels[ground_mask] = 1  # 'ground_assault'

        # Major strike: high equipment, lower casualties (relative)
        strike_mask = (equipment_total > equipment_threshold_high) & (casualty_total <= casualty_threshold_high)
        labels[strike_mask] = 0  # 'major_strike'

        # Everything else is medium/mixed - label as infrastructure (4) or counterattack (2)
        # For simplicity, use infrastructure for remaining high equipment
        infrastructure_mask = (labels == 0) & (equipment_total > np.percentile(equipment_total, 50))
        labels[infrastructure_mask] = 4  # 'infrastructure'

        return labels

    def construct_operation_labels(
        self,
        dates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Construct operation labels based on date ranges.

        Args:
            dates: Array of dates to label. Uses self.dates if None.

        Returns:
            labels: Integer-encoded operation labels [n_samples]
            label_map: Mapping from integer to operation name
        """
        if dates is None:
            dates = self.dates

        if dates is None:
            raise ValueError("No dates available for operation labeling")

        # Build label map (0 = no operation)
        label_map = {0: 'no_operation'}
        for i, (op_key, op_info) in enumerate(MILITARY_OPERATIONS.items(), start=1):
            label_map[i] = op_key

        reverse_map = {v: k for k, v in label_map.items()}

        labels = np.zeros(len(dates), dtype=np.int64)

        for i, date in enumerate(dates):
            op = date_to_operation(date)
            if op is not None:
                labels[i] = reverse_map[op]

        return labels, label_map


# =============================================================================
# 4.1.1 OPERATION CLUSTERING PROBE
# =============================================================================

@dataclass
class ClusteringResults:
    """Results from operation clustering analysis."""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    variance_ratio: float
    n_samples_per_cluster: Dict[str, int]
    cluster_centroids: Optional[np.ndarray] = None
    tsne_embeddings: Optional[np.ndarray] = None
    operation_labels: Optional[np.ndarray] = None


class OperationClusteringProbe:
    """
    Test if named military operations cluster separately in latent space.

    This probe analyzes whether the model's latent representations naturally
    group by military operation phase, even though the model was only trained
    on numerical data without explicit operation labels.

    Metrics:
    - Silhouette score: [-1, 1], higher is better
    - Calinski-Harabasz index: Higher is better (ratio of between/within cluster dispersion)
    - Davies-Bouldin index: Lower is better (average similarity between clusters)
    - Within-vs-between variance ratio: Higher is better (F-statistic analog)
    """

    def __init__(
        self,
        latents: np.ndarray,
        dates: np.ndarray,
        operation_labels: Optional[np.ndarray] = None,
    ):
        """
        Initialize the clustering probe.

        Args:
            latents: Latent representations [n_samples, d_model]
            dates: Corresponding dates [n_samples]
            operation_labels: Optional pre-computed operation labels.
                            If None, will be derived from dates.
        """
        self.latents = latents
        self.dates = dates

        if operation_labels is None:
            self.operation_labels, self.label_map = self._assign_operation_labels()
        else:
            self.operation_labels = operation_labels
            self.label_map = {i: MILITARY_OPERATIONS.get(i, f'operation_{i}')
                             for i in np.unique(operation_labels)}

        # Filter to only samples with assigned operations (non-zero)
        self.operation_mask = self.operation_labels > 0
        self.n_operations = len(np.unique(self.operation_labels[self.operation_mask]))

    def _assign_operation_labels(self) -> Tuple[np.ndarray, Dict[int, str]]:
        """Assign operation labels based on date ranges."""
        label_constructor = LabelConstructor(dates=self.dates)
        return label_constructor.construct_operation_labels()

    def compute_clustering_metrics(
        self,
        use_filtered: bool = True,
    ) -> ClusteringResults:
        """
        Compute clustering quality metrics.

        Args:
            use_filtered: If True, only use samples with assigned operations

        Returns:
            ClusteringResults with all computed metrics
        """
        if use_filtered:
            latents = self.latents[self.operation_mask]
            labels = self.operation_labels[self.operation_mask]
        else:
            latents = self.latents
            labels = self.operation_labels

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            warnings.warn("Need at least 2 clusters for clustering metrics")
            return ClusteringResults(
                silhouette_score=0.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=float('inf'),
                variance_ratio=0.0,
                n_samples_per_cluster={str(l): int((labels == l).sum()) for l in unique_labels},
            )

        # Compute metrics
        sil_score = silhouette_score(latents, labels)
        ch_score = calinski_harabasz_score(latents, labels)
        db_score = davies_bouldin_score(latents, labels)
        var_ratio = compute_variance_ratio(latents, labels)

        # Samples per cluster
        n_samples_per_cluster = {}
        for label in unique_labels:
            label_name = self.label_map.get(label, f'operation_{label}')
            n_samples_per_cluster[label_name] = int((labels == label).sum())

        # Compute centroids
        centroids = np.array([latents[labels == l].mean(axis=0) for l in unique_labels])

        return ClusteringResults(
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            variance_ratio=var_ratio,
            n_samples_per_cluster=n_samples_per_cluster,
            cluster_centroids=centroids,
            operation_labels=labels,
        )

    def compute_tsne_embedding(
        self,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Compute t-SNE embedding for visualization.

        Args:
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            random_state: Random seed for reproducibility

        Returns:
            2D t-SNE embedding [n_samples, 2]
        """
        latents = self.latents[self.operation_mask]

        # Adjust perplexity if needed
        n_samples = len(latents)
        effective_perplexity = min(perplexity, n_samples // 4)

        if effective_perplexity < 5:
            warnings.warn(f"Very few samples ({n_samples}), t-SNE may not be meaningful")
            effective_perplexity = max(5, n_samples // 4)

        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            max_iter=n_iter,  # sklearn >= 1.5 uses max_iter instead of n_iter
            random_state=random_state,
            init='pca',
        )

        return tsne.fit_transform(latents)

    def visualize_clusters(
        self,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Create visualization of operation clusters in latent space.

        Args:
            output_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Compute t-SNE
        tsne_embedding = self.compute_tsne_embedding()
        labels = self.operation_labels[self.operation_mask]
        unique_labels = np.unique(labels)

        # Color map
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        # Plot 1: t-SNE scatter
        ax = axes[0]
        for label in unique_labels:
            mask = labels == label
            label_name = self.label_map.get(label, f'Op {label}')
            ax.scatter(
                tsne_embedding[mask, 0],
                tsne_embedding[mask, 1],
                c=[color_map[label]],
                label=label_name,
                alpha=0.7,
                s=30,
            )

        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Operation Clustering in Latent Space (t-SNE)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Plot 2: Cluster metrics summary
        ax = axes[1]
        results = self.compute_clustering_metrics()

        metrics = {
            'Silhouette': results.silhouette_score,
            'Variance Ratio': min(results.variance_ratio, 50),  # Cap for visualization
            'Calinski-Harabasz\n(scaled)': results.calinski_harabasz_score / 100,
            '1 / Davies-Bouldin': 1 / max(results.davies_bouldin_score, 0.01),
        }

        bars = ax.bar(range(len(metrics)), list(metrics.values()), color='steelblue')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Score (higher = better clustering)')
        ax.set_title('Clustering Quality Metrics')

        # Add value labels
        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


# =============================================================================
# 4.1.2-4.1.4 LINEAR DECODER PROBE
# =============================================================================

@dataclass
class ProbeResults:
    """Results from linear probing analysis."""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None


class LinearDecoder(nn.Module):
    """
    Linear probe classifier for decoding semantic categories from frozen latents.

    This implements a simple linear layer (logistic regression equivalent) that
    is trained to classify semantic categories from the model's frozen latent
    representations. Linear probes test for LINEAR separability of concepts.

    Attributes:
        input_dim: Dimension of latent representations
        num_classes: Number of output classes
        probe: Linear layer mapping latents to class logits
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.probe = nn.Linear(input_dim, num_classes)

        # Initialize with small weights
        nn.init.xavier_uniform_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through linear probe.

        Args:
            x: Latent representations [batch, d_model]

        Returns:
            Class logits [batch, num_classes]
        """
        return self.probe(x)


class DayTypeDecodingProbe:
    """
    Train linear probe to classify day types from frozen latent representations.

    Day Types:
    - Major Strike Day: Large-scale missile/drone attacks
    - Ground Assault Day: High casualty ground combat
    - Counterattack Day: Successful territorial recovery
    - Quiet Day: Below-average activity across metrics
    - Infrastructure Attack Day: Civilian infrastructure targeted

    The probe is trained using sklearn's LogisticRegression with cross-validation.
    """

    def __init__(
        self,
        latents: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the day type decoding probe.

        Args:
            latents: Frozen latent representations [n_samples, d_model]
            labels: Day type labels (integer-encoded) [n_samples]
            class_names: Optional list of class names for reporting
        """
        self.latents = latents
        self.labels = labels
        self.class_names = class_names or [DAY_TYPES.get(i, f'Type {i}')
                                           for i in range(len(np.unique(labels)))]

        # Standardize latents
        self.scaler = StandardScaler()
        self.latents_scaled = self.scaler.fit_transform(latents)

        self.model = None

    def train_probe(
        self,
        test_size: float = 0.2,
        cv_folds: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> ProbeResults:
        """
        Train the linear probe with cross-validation.

        Args:
            test_size: Fraction of data for held-out test
            cv_folds: Number of cross-validation folds
            max_iter: Maximum iterations for convergence
            random_state: Random seed

        Returns:
            ProbeResults with all metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.latents_scaled, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )

        # Train logistic regression
        # Note: sklearn >= 1.5 removed multi_class parameter (always uses 'auto')
        self.model = LogisticRegression(
            solver='lbfgs',
            max_iter=max_iter,
            random_state=random_state,
            class_weight='balanced',  # Handle imbalanced classes
        )

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')

        # Final training and evaluation
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.class_names)

        return ProbeResults(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            confusion_matrix=cm,
            classification_report=report,
            feature_importance=self.model.coef_,
            class_names=self.class_names,
        )

    def visualize_results(
        self,
        results: ProbeResults,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Visualize probe results with confusion matrix and feature importance.

        Args:
            results: ProbeResults from train_probe()
            output_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot 1: Confusion matrix
        ax = axes[0]
        sns.heatmap(
            results.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=results.class_names,
            yticklabels=results.class_names,
            ax=ax,
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix\nAccuracy: {results.accuracy:.3f}')

        # Plot 2: Performance metrics
        ax = axes[1]
        metrics = {
            'Accuracy': results.accuracy,
            'F1 (Macro)': results.f1_macro,
            'F1 (Weighted)': results.f1_weighted,
            'CV Mean': results.cv_accuracy_mean,
        }
        bars = ax.bar(range(len(metrics)), list(metrics.values()), color='steelblue')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Probe Performance Metrics')
        ax.set_ylim(0, 1)

        # Add error bar for CV
        ax.errorbar(3, results.cv_accuracy_mean, yerr=results.cv_accuracy_std,
                   fmt='none', color='black', capsize=5)

        for bar, val in zip(bars, metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 3: Feature importance (top coefficients)
        ax = axes[2]
        if results.feature_importance is not None:
            # Average absolute importance across classes
            importance = np.abs(results.feature_importance).mean(axis=0)
            top_k = min(20, len(importance))
            top_indices = np.argsort(importance)[-top_k:][::-1]

            ax.barh(range(top_k), importance[top_indices], color='coral')
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([f'Dim {i}' for i in top_indices])
            ax.set_xlabel('Mean |Coefficient|')
            ax.set_title(f'Top {top_k} Important Latent Dimensions')
            ax.invert_yaxis()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


class IntensityProbe(DayTypeDecodingProbe):
    """
    Train probe for conflict intensity levels.

    Intensity Levels:
    - Low (0): Bottom quartile of activity
    - Medium (1): Middle 50% (25th-75th percentile)
    - High (2): 75th-95th percentile
    - Extreme (3): Top 5%

    Inherits from DayTypeDecodingProbe with intensity-specific defaults.
    """

    def __init__(
        self,
        latents: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Initialize intensity probe.

        Args:
            latents: Frozen latent representations [n_samples, d_model]
            labels: Intensity labels (0-3) [n_samples]
        """
        class_names = [INTENSITY_LEVELS[i] for i in range(4)]
        super().__init__(latents, labels, class_names)


class GeographicFocusProbe(DayTypeDecodingProbe):
    """
    Train probe to decode primary geographic theater from latent representations.

    Theaters:
    - Eastern Front: Donbas operations
    - Southern Front: Kherson/Zaporizhzhia direction
    - Multi-front: Simultaneous operations
    - Strategic Depth: Deep strikes into Russian territory

    Note: Geographic labels typically require external data (location of events).
    This probe assumes labels are pre-computed or derived from proxy indicators.
    """

    def __init__(
        self,
        latents: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Initialize geographic focus probe.

        Args:
            latents: Frozen latent representations [n_samples, d_model]
            labels: Geographic theater labels (0-3) [n_samples]
        """
        class_names = [GEOGRAPHIC_THEATERS[i] for i in range(4)]
        super().__init__(latents, labels, class_names)


# =============================================================================
# 4.2 TEMPORAL SEMANTIC PATTERN PROBES
# =============================================================================

@dataclass
class TemporalPatternResults:
    """Results from temporal pattern analysis."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    effect_size: Optional[float] = None
    group_means: Optional[Dict[str, float]] = None
    additional_info: Optional[Dict[str, Any]] = None


class TemporalPatternProbe:
    """
    Analyze temporal semantic patterns in latent representations.

    Tests for:
    - Weekly cycles (ANOVA across weekdays)
    - Seasonal/monthly patterns
    - Event anniversary encoding ("time since major event")
    """

    def __init__(
        self,
        latents: np.ndarray,
        dates: np.ndarray,
    ):
        """
        Initialize temporal pattern probe.

        Args:
            latents: Latent representations [n_samples, d_model]
            dates: Corresponding dates [n_samples]
        """
        self.latents = latents
        self.dates = pd.to_datetime(dates)
        self.d_model = latents.shape[1]

        # Precompute temporal features
        self._compute_temporal_features()

    def _compute_temporal_features(self):
        """Compute temporal grouping features from dates."""
        self.weekday = self.dates.dayofweek.values  # 0=Monday, 6=Sunday
        self.month = self.dates.month.values  # 1-12
        self.quarter = self.dates.quarter.values  # 1-4
        self.year = self.dates.year.values
        self.day_of_year = self.dates.dayofyear.values

        # Days since war start (Feb 24, 2022)
        war_start = pd.Timestamp('2022-02-24')
        self.days_since_war_start = (self.dates - war_start).days.values

    def test_weekly_cycle(self) -> TemporalPatternResults:
        """
        Test for weekday effects on latent states using ANOVA.

        Tests whether the mean latent representation differs significantly
        across weekdays (Monday-Sunday).

        Returns:
            TemporalPatternResults with ANOVA F-statistic and p-value
        """
        # Use latent magnitude as summary statistic
        latent_norms = np.linalg.norm(self.latents, axis=1)

        # Group by weekday
        groups = [latent_norms[self.weekday == i] for i in range(7)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return TemporalPatternResults(
                test_name='Weekly Cycle ANOVA',
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
            )

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        ss_total = np.sum((all_data - all_data.mean()) ** 2)
        ss_between = sum(len(g) * (g.mean() - all_data.mean()) ** 2 for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Group means
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        group_means = {weekday_names[i]: groups[i].mean() for i in range(len(groups))}

        return TemporalPatternResults(
            test_name='Weekly Cycle ANOVA',
            statistic=float(f_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            effect_size=float(eta_squared),
            group_means=group_means,
        )

    def test_seasonal_pattern(self) -> TemporalPatternResults:
        """
        Test for monthly/seasonal patterns using ANOVA.

        Returns:
            TemporalPatternResults with seasonal effect statistics
        """
        latent_norms = np.linalg.norm(self.latents, axis=1)

        # Group by month
        groups = [latent_norms[self.month == m] for m in range(1, 13)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return TemporalPatternResults(
                test_name='Seasonal Pattern ANOVA',
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
            )

        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size
        all_data = np.concatenate(groups)
        ss_total = np.sum((all_data - all_data.mean()) ** 2)
        ss_between = sum(len(g) * (g.mean() - all_data.mean()) ** 2 for g in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        group_means = {month_names[i]: groups[i].mean()
                      for i in range(len(groups)) if len(groups) > i}

        return TemporalPatternResults(
            test_name='Seasonal Pattern ANOVA',
            statistic=float(f_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            effect_size=float(eta_squared),
            group_means=group_means,
        )

    def test_event_anniversary_encoding(
        self,
        event_dates: Optional[List[str]] = None,
    ) -> TemporalPatternResults:
        """
        Test if latent space encodes "time since major event".

        Uses correlation between latent dimensions and days-since-event
        for major conflict milestones.

        Args:
            event_dates: List of event dates to test. If None, uses
                        war start and major operation dates.

        Returns:
            TemporalPatternResults with correlation statistics
        """
        if event_dates is None:
            # Use major event dates
            event_dates = [
                '2022-02-24',  # War start
                '2022-09-06',  # Kharkiv counteroffensive
                '2022-11-11',  # Kherson liberation
                '2023-05-20',  # Bakhmut fall
            ]

        # Compute days since each event for each sample
        results_per_event = []

        for event_date in event_dates:
            event_ts = pd.Timestamp(event_date)
            days_since = (self.dates - event_ts).days.values

            # Test correlation with each latent dimension
            correlations = []
            p_values = []

            for dim in range(self.d_model):
                r, p = stats.pearsonr(days_since, self.latents[:, dim])
                correlations.append(r)
                p_values.append(p)

            correlations = np.array(correlations)
            p_values = np.array(p_values)

            # Find max absolute correlation
            max_idx = np.argmax(np.abs(correlations))
            max_corr = correlations[max_idx]
            max_p = p_values[max_idx]

            # Count significant correlations (Bonferroni corrected)
            alpha_corrected = 0.05 / self.d_model
            n_significant = (p_values < alpha_corrected).sum()

            results_per_event.append({
                'event_date': event_date,
                'max_correlation': max_corr,
                'max_correlation_dim': max_idx,
                'max_p_value': max_p,
                'n_significant_dims': n_significant,
            })

        # Aggregate across events
        max_corrs = [r['max_correlation'] for r in results_per_event]
        best_idx = np.argmax(np.abs(max_corrs))
        best_result = results_per_event[best_idx]

        return TemporalPatternResults(
            test_name='Event Anniversary Encoding',
            statistic=float(best_result['max_correlation']),
            p_value=float(best_result['max_p_value']),
            is_significant=best_result['max_p_value'] < 0.05,
            effect_size=float(best_result['max_correlation'] ** 2),  # R-squared
            additional_info={
                'best_event': best_result['event_date'],
                'best_dimension': best_result['max_correlation_dim'],
                'all_events': results_per_event,
            },
        )

    def test_all_patterns(self) -> Dict[str, TemporalPatternResults]:
        """
        Run all temporal pattern tests.

        Returns:
            Dictionary mapping test name to results
        """
        return {
            'weekly_cycle': self.test_weekly_cycle(),
            'seasonal_pattern': self.test_seasonal_pattern(),
            'event_anniversary': self.test_event_anniversary_encoding(),
        }

    def visualize_patterns(
        self,
        results: Dict[str, TemporalPatternResults],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Visualize all temporal pattern results.

        Args:
            results: Dictionary of TemporalPatternResults
            output_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Weekly cycle
        ax = axes[0, 0]
        weekly = results['weekly_cycle']
        if weekly.group_means:
            days = list(weekly.group_means.keys())
            means = list(weekly.group_means.values())
            bars = ax.bar(days, means, color='steelblue')
            ax.axhline(y=np.mean(means), color='red', linestyle='--', label='Overall mean')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Mean Latent Norm')
            sig_str = '*' if weekly.is_significant else ''
            ax.set_title(f'Weekly Cycle{sig_str}\nF={weekly.statistic:.2f}, p={weekly.p_value:.4f}')
            ax.legend()

        # Plot 2: Seasonal pattern
        ax = axes[0, 1]
        seasonal = results['seasonal_pattern']
        if seasonal.group_means:
            months = list(seasonal.group_means.keys())
            means = list(seasonal.group_means.values())
            ax.plot(range(len(months)), means, 'o-', color='coral', linewidth=2, markersize=8)
            ax.axhline(y=np.mean(means), color='blue', linestyle='--', label='Overall mean')
            ax.set_xticks(range(len(months)))
            ax.set_xticklabels(months, rotation=45)
            ax.set_xlabel('Month')
            ax.set_ylabel('Mean Latent Norm')
            sig_str = '*' if seasonal.is_significant else ''
            ax.set_title(f'Seasonal Pattern{sig_str}\nF={seasonal.statistic:.2f}, p={seasonal.p_value:.4f}')
            ax.legend()

        # Plot 3: Event anniversary encoding
        ax = axes[1, 0]
        anniversary = results['event_anniversary']
        if anniversary.additional_info:
            events = anniversary.additional_info['all_events']
            event_names = [e['event_date'] for e in events]
            correlations = [e['max_correlation'] for e in events]

            colors = ['green' if abs(c) > 0.3 else 'gray' for c in correlations]
            bars = ax.barh(event_names, correlations, color=colors)
            ax.axvline(x=0, color='black', linestyle='-')
            ax.set_xlabel('Max Correlation with Latent Dimension')
            ax.set_title(f'Event Anniversary Encoding\nBest: {anniversary.additional_info["best_event"]}')

        # Plot 4: Summary of significance
        ax = axes[1, 1]
        test_names = ['Weekly\nCycle', 'Seasonal\nPattern', 'Event\nAnniversary']
        p_values = [
            results['weekly_cycle'].p_value,
            results['seasonal_pattern'].p_value,
            results['event_anniversary'].p_value,
        ]

        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        bars = ax.bar(test_names, [-np.log10(p) for p in p_values], color=colors)
        ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p=0.05 threshold')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Statistical Significance Summary\n(Higher = more significant)')
        ax.legend()

        # Add significance markers
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            marker = '*' if p < 0.05 else 'n.s.'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f'{marker}\np={p:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        return fig


# =============================================================================
# COMPREHENSIVE PROBE RUNNER
# =============================================================================

class SemanticStructureProbeRunner:
    """
    Orchestrates all semantic structure probes on model latent representations.

    This class coordinates:
    1. Latent extraction from trained model
    2. Label construction from numerical data
    3. Running all probes (clustering, linear decoding, temporal)
    4. Aggregating and visualizing results
    """

    def __init__(
        self,
        latents: np.ndarray,
        dates: np.ndarray,
        casualty_data: Optional[np.ndarray] = None,
        equipment_data: Optional[np.ndarray] = None,
        territorial_data: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize probe runner.

        Args:
            latents: Latent representations [n_samples, d_model]
            dates: Corresponding dates [n_samples]
            casualty_data: Optional casualty data for label construction
            equipment_data: Optional equipment loss data for label construction
            territorial_data: Optional territorial change data
            output_dir: Directory for saving results
        """
        self.latents = latents
        self.dates = pd.to_datetime(dates)
        self.casualty_data = casualty_data
        self.equipment_data = equipment_data
        self.territorial_data = territorial_data
        self.output_dir = output_dir or OUTPUT_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize label constructor
        self.label_constructor = LabelConstructor(
            casualty_data=casualty_data,
            equipment_data=equipment_data,
            territorial_data=territorial_data,
            dates=dates,
        )

        self.results = {}

    def run_operation_clustering(self) -> ClusteringResults:
        """Run operation clustering probe (4.1.1)."""
        print("Running Operation Clustering Probe (4.1.1)...")

        probe = OperationClusteringProbe(
            latents=self.latents,
            dates=self.dates,
        )

        results = probe.compute_clustering_metrics()

        # Visualize
        fig = probe.visualize_clusters(
            output_path=self.output_dir / 'operation_clustering.png'
        )
        plt.close(fig)

        self.results['operation_clustering'] = results

        print(f"  Silhouette Score: {results.silhouette_score:.4f}")
        print(f"  Variance Ratio: {results.variance_ratio:.4f}")

        return results

    def run_day_type_probe(self) -> Optional[ProbeResults]:
        """Run day type decoding probe (4.1.2)."""
        print("Running Day Type Decoding Probe (4.1.2)...")

        if self.casualty_data is None or self.equipment_data is None:
            print("  Skipping: casualty and equipment data required")
            return None

        # Construct labels
        labels = self.label_constructor.construct_day_type_labels()

        probe = DayTypeDecodingProbe(
            latents=self.latents,
            labels=labels,
        )

        results = probe.train_probe()

        # Visualize
        fig = probe.visualize_results(
            results,
            output_path=self.output_dir / 'day_type_probe.png'
        )
        plt.close(fig)

        self.results['day_type_probe'] = results

        print(f"  Accuracy: {results.accuracy:.4f}")
        print(f"  F1 (macro): {results.f1_macro:.4f}")

        return results

    def run_intensity_probe(self) -> Optional[ProbeResults]:
        """Run intensity level probe (4.1.3)."""
        print("Running Intensity Level Probe (4.1.3)...")

        if self.casualty_data is None:
            print("  Skipping: casualty data required")
            return None

        # Construct labels
        labels = self.label_constructor.construct_intensity_labels()

        probe = IntensityProbe(
            latents=self.latents,
            labels=labels,
        )

        results = probe.train_probe()

        # Visualize
        fig = probe.visualize_results(
            results,
            output_path=self.output_dir / 'intensity_probe.png'
        )
        plt.close(fig)

        self.results['intensity_probe'] = results

        print(f"  Accuracy: {results.accuracy:.4f}")
        print(f"  F1 (macro): {results.f1_macro:.4f}")

        return results

    def run_geographic_probe(
        self,
        geographic_labels: Optional[np.ndarray] = None,
    ) -> Optional[ProbeResults]:
        """
        Run geographic focus probe (4.1.4).

        Args:
            geographic_labels: Pre-computed geographic labels. Required since
                             these cannot be derived from numerical data alone.
        """
        print("Running Geographic Focus Probe (4.1.4)...")

        if geographic_labels is None:
            print("  Skipping: geographic labels required (cannot derive from numerical data)")
            return None

        probe = GeographicFocusProbe(
            latents=self.latents,
            labels=geographic_labels,
        )

        results = probe.train_probe()

        # Visualize
        fig = probe.visualize_results(
            results,
            output_path=self.output_dir / 'geographic_probe.png'
        )
        plt.close(fig)

        self.results['geographic_probe'] = results

        print(f"  Accuracy: {results.accuracy:.4f}")
        print(f"  F1 (macro): {results.f1_macro:.4f}")

        return results

    def run_temporal_probes(self) -> Dict[str, TemporalPatternResults]:
        """Run all temporal pattern probes (4.2.x)."""
        print("Running Temporal Pattern Probes (4.2)...")

        probe = TemporalPatternProbe(
            latents=self.latents,
            dates=self.dates,
        )

        results = probe.test_all_patterns()

        # Visualize
        fig = probe.visualize_patterns(
            results,
            output_path=self.output_dir / 'temporal_patterns.png'
        )
        plt.close(fig)

        self.results['temporal_patterns'] = results

        for name, result in results.items():
            sig_str = '*' if result.is_significant else ''
            print(f"  {name}: p={result.p_value:.4f}{sig_str}")

        return results

    def run_all_probes(
        self,
        geographic_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run all semantic structure probes.

        Args:
            geographic_labels: Optional pre-computed geographic labels

        Returns:
            Dictionary of all probe results
        """
        print("=" * 60)
        print("SEMANTIC STRUCTURE PROBE ANALYSIS")
        print("=" * 60)
        print(f"Latent shape: {self.latents.shape}")
        print(f"Date range: {self.dates.min()} to {self.dates.max()}")
        print()

        # Run all probes
        self.run_operation_clustering()
        print()

        self.run_day_type_probe()
        print()

        self.run_intensity_probe()
        print()

        self.run_geographic_probe(geographic_labels)
        print()

        self.run_temporal_probes()
        print()

        # Generate summary report
        self._generate_summary_report()

        return self.results

    def _generate_summary_report(self):
        """Generate a summary report of all probe results."""
        report_path = self.output_dir / 'probe_summary_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SEMANTIC STRUCTURE PROBE SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Latent Dimension: {self.latents.shape[1]}\n")
            f.write(f"Number of Samples: {self.latents.shape[0]}\n")
            f.write(f"Date Range: {self.dates.min()} to {self.dates.max()}\n\n")

            # 4.1.1 Operation Clustering
            f.write("-" * 40 + "\n")
            f.write("4.1.1 Operation Clustering\n")
            f.write("-" * 40 + "\n")
            if 'operation_clustering' in self.results:
                r = self.results['operation_clustering']
                f.write(f"Silhouette Score: {r.silhouette_score:.4f}\n")
                f.write(f"Calinski-Harabasz Score: {r.calinski_harabasz_score:.4f}\n")
                f.write(f"Davies-Bouldin Score: {r.davies_bouldin_score:.4f}\n")
                f.write(f"Variance Ratio: {r.variance_ratio:.4f}\n")
                f.write(f"Samples per cluster: {r.n_samples_per_cluster}\n")
            f.write("\n")

            # 4.1.2 Day Type Probe
            f.write("-" * 40 + "\n")
            f.write("4.1.2 Day Type Decoding Probe\n")
            f.write("-" * 40 + "\n")
            if 'day_type_probe' in self.results and self.results['day_type_probe']:
                r = self.results['day_type_probe']
                f.write(f"Accuracy: {r.accuracy:.4f}\n")
                f.write(f"F1 (macro): {r.f1_macro:.4f}\n")
                f.write(f"CV Accuracy: {r.cv_accuracy_mean:.4f} +/- {r.cv_accuracy_std:.4f}\n")
                f.write(f"\nClassification Report:\n{r.classification_report}\n")
            else:
                f.write("Not run (missing required data)\n")
            f.write("\n")

            # 4.1.3 Intensity Probe
            f.write("-" * 40 + "\n")
            f.write("4.1.3 Intensity Level Probe\n")
            f.write("-" * 40 + "\n")
            if 'intensity_probe' in self.results and self.results['intensity_probe']:
                r = self.results['intensity_probe']
                f.write(f"Accuracy: {r.accuracy:.4f}\n")
                f.write(f"F1 (macro): {r.f1_macro:.4f}\n")
                f.write(f"CV Accuracy: {r.cv_accuracy_mean:.4f} +/- {r.cv_accuracy_std:.4f}\n")
                f.write(f"\nClassification Report:\n{r.classification_report}\n")
            else:
                f.write("Not run (missing required data)\n")
            f.write("\n")

            # 4.1.4 Geographic Probe
            f.write("-" * 40 + "\n")
            f.write("4.1.4 Geographic Focus Probe\n")
            f.write("-" * 40 + "\n")
            if 'geographic_probe' in self.results and self.results['geographic_probe']:
                r = self.results['geographic_probe']
                f.write(f"Accuracy: {r.accuracy:.4f}\n")
                f.write(f"F1 (macro): {r.f1_macro:.4f}\n")
                f.write(f"CV Accuracy: {r.cv_accuracy_mean:.4f} +/- {r.cv_accuracy_std:.4f}\n")
            else:
                f.write("Not run (geographic labels required)\n")
            f.write("\n")

            # 4.2 Temporal Patterns
            f.write("-" * 40 + "\n")
            f.write("4.2 Temporal Semantic Patterns\n")
            f.write("-" * 40 + "\n")
            if 'temporal_patterns' in self.results:
                for name, r in self.results['temporal_patterns'].items():
                    sig = "SIGNIFICANT" if r.is_significant else "not significant"
                    f.write(f"\n{r.test_name}:\n")
                    f.write(f"  Statistic: {r.statistic:.4f}\n")
                    f.write(f"  P-value: {r.p_value:.4f} ({sig})\n")
                    if r.effect_size:
                        f.write(f"  Effect Size: {r.effect_size:.4f}\n")
            f.write("\n")

            # Conclusions
            f.write("=" * 60 + "\n")
            f.write("CONCLUSIONS\n")
            f.write("=" * 60 + "\n")
            f.write("\nKey findings from semantic structure probing:\n\n")

            # Auto-generate conclusions based on results
            conclusions = []

            if 'operation_clustering' in self.results:
                r = self.results['operation_clustering']
                if r.silhouette_score > 0.3:
                    conclusions.append(
                        "- STRONG: Military operations show clear clustering in latent space "
                        f"(silhouette={r.silhouette_score:.3f})"
                    )
                elif r.silhouette_score > 0.1:
                    conclusions.append(
                        "- MODERATE: Some operation clustering observed "
                        f"(silhouette={r.silhouette_score:.3f})"
                    )
                else:
                    conclusions.append(
                        "- WEAK: Operations do not cluster strongly "
                        f"(silhouette={r.silhouette_score:.3f})"
                    )

            if 'intensity_probe' in self.results and self.results['intensity_probe']:
                r = self.results['intensity_probe']
                if r.accuracy > 0.6:
                    conclusions.append(
                        f"- STRONG: Intensity levels are linearly decodable (acc={r.accuracy:.3f})"
                    )
                elif r.accuracy > 0.4:
                    conclusions.append(
                        f"- MODERATE: Some intensity encoding (acc={r.accuracy:.3f})"
                    )

            if 'temporal_patterns' in self.results:
                for name, r in self.results['temporal_patterns'].items():
                    if r.is_significant:
                        conclusions.append(
                            f"- {name.upper().replace('_', ' ')}: Significant encoding detected (p={r.p_value:.4f})"
                        )

            for c in conclusions:
                f.write(c + "\n")

            if not conclusions:
                f.write("- No strong semantic structure detected in latent representations\n")

        print(f"Summary report saved to: {report_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run semantic structure probes on sample data."""
    print("Semantic Structure Probe Module")
    print("=" * 60)

    # Create output directory
    setup_output_dir()

    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 500
    d_model = 128

    # Generate synthetic latents with some structure
    # Simulate 3 clusters (operations)
    latents = np.vstack([
        np.random.randn(150, d_model) + np.array([1, 0] + [0] * (d_model - 2)),
        np.random.randn(200, d_model) + np.array([0, 1] + [0] * (d_model - 2)),
        np.random.randn(150, d_model) + np.array([-1, -1] + [0] * (d_model - 2)),
    ])

    # Generate dates spanning the conflict
    dates = pd.date_range('2022-02-24', periods=n_samples, freq='D')

    # Generate synthetic casualty and equipment data
    casualty_data = np.abs(np.random.randn(n_samples, 1) * 100 + 200)
    equipment_data = np.abs(np.random.randn(n_samples, 5) * 10 + 20)

    print(f"Running probes on synthetic data:")
    print(f"  Latents: {latents.shape}")
    print(f"  Dates: {dates[0]} to {dates[-1]}")
    print()

    # Run all probes
    runner = SemanticStructureProbeRunner(
        latents=latents,
        dates=dates,
        casualty_data=casualty_data,
        equipment_data=equipment_data,
        output_dir=OUTPUT_DIR,
    )

    results = runner.run_all_probes()

    print()
    print("=" * 60)
    print("Probe analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
