#!/usr/bin/env python3
"""
ISW Embedding Quality Validation

Pre-integration analysis to verify that ISW text embeddings capture
meaningful conflict dynamics before incorporating into the HAN model.

Validates:
1. Intrinsic dimensionality (how many PCs needed?)
2. Temporal coherence (do adjacent days have similar embeddings?)
3. Cluster structure (do embeddings form meaningful groups?)
4. Event alignment (do major events show embedding shifts?)
5. Correlation with quantitative features

Author: Data Science Analysis
Date: 2026-01-21
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    ISW_EMBEDDINGS_DIR, EMBEDDING_OUTPUT_DIR,
)

# Paths
BASE_DIR = PROJECT_ROOT
EMBEDDING_DIR = ISW_EMBEDDINGS_DIR
OUTPUT_DIR = EMBEDDING_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAJOR CONFLICT EVENTS (for alignment analysis)
# =============================================================================

MAJOR_EVENTS = {
    '2022-02-24': 'Invasion begins',
    '2022-04-03': 'Bucha massacre revealed',
    '2022-05-20': 'Mariupol falls',
    '2022-07-03': 'Lysychansk falls',
    '2022-08-29': 'Kherson counteroffensive starts',
    '2022-09-11': 'Kharkiv counteroffensive success',
    '2022-10-08': 'Kerch Bridge attack',
    '2022-11-11': 'Kherson liberation',
    '2023-01-11': 'Soledar falls',
    '2023-05-21': 'Bakhmut falls',
    '2023-06-06': 'Dam destruction / counteroffensive',
    '2023-06-24': 'Prigozhin mutiny',
    '2023-08-24': 'Prigozhin death',
    '2024-02-17': 'Avdiivka falls',
    '2024-04-13': 'Iran drone/missile attack on Israel',  # Geopolitical context
    '2024-08-06': 'Kursk incursion begins',
    '2024-10-14': 'North Korean troops reported',
}


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    """Load embedding matrix and date index."""
    matrix_path = EMBEDDING_DIR / "isw_embedding_matrix.npy"
    index_path = EMBEDDING_DIR / "isw_date_index.json"

    if not matrix_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"Embedding files not found in {EMBEDDING_DIR}")

    embeddings = np.load(matrix_path)

    with open(index_path) as f:
        index_data = json.load(f)

    dates = index_data['dates']

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    return embeddings, dates


# =============================================================================
# 1. INTRINSIC DIMENSIONALITY ANALYSIS
# =============================================================================

def analyze_dimensionality(embeddings: np.ndarray) -> Dict:
    """
    Estimate intrinsic dimensionality of embedding space.

    Returns dict with PCA analysis results.
    """
    print("\n" + "=" * 60)
    print("1. INTRINSIC DIMENSIONALITY ANALYSIS")
    print("=" * 60)

    pca = PCA()
    pca.fit(embeddings)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    # Find components needed for various thresholds
    thresholds = [0.50, 0.75, 0.90, 0.95, 0.99]
    components_needed = {}
    for thresh in thresholds:
        n_comp = np.argmax(cumvar >= thresh) + 1
        components_needed[thresh] = n_comp
        print(f"  {thresh*100:.0f}% variance explained by {n_comp} components")

    # Estimate intrinsic dimensionality using elbow method
    # (second derivative of cumulative variance)
    second_deriv = np.diff(np.diff(cumvar))
    elbow_idx = np.argmin(second_deriv) + 2
    print(f"\n  Elbow point (intrinsic dimensionality estimate): {elbow_idx}")

    # Plot variance explained
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative variance
    ax1 = axes[0]
    ax1.plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=2)
    for thresh in [0.90, 0.95]:
        n = components_needed[thresh]
        ax1.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=n, color='red', linestyle='--', alpha=0.5)
        ax1.annotate(f'{n} PCs', xy=(n, thresh), xytext=(n+10, thresh-0.05),
                    fontsize=10)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Variance Explained')
    ax1.set_title('PCA Cumulative Variance')
    ax1.set_xlim(0, 200)
    ax1.grid(True, alpha=0.3)

    # Individual variance (log scale)
    ax2 = axes[1]
    ax2.semilogy(range(1, 101), pca.explained_variance_ratio_[:100], 'b-', linewidth=2)
    ax2.axvline(x=elbow_idx, color='red', linestyle='--', label=f'Elbow at {elbow_idx}')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Variance Explained (log scale)')
    ax2.set_title('Individual Component Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dimensionality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'components_needed': components_needed,
        'elbow_estimate': elbow_idx,
        'variance_ratios': pca.explained_variance_ratio_[:100].tolist(),
    }


# =============================================================================
# 2. TEMPORAL COHERENCE ANALYSIS
# =============================================================================

def analyze_temporal_coherence(embeddings: np.ndarray, dates: List[str]) -> Dict:
    """
    Analyze whether embeddings respect temporal structure.

    Adjacent days should have more similar embeddings than distant days.
    """
    print("\n" + "=" * 60)
    print("2. TEMPORAL COHERENCE ANALYSIS")
    print("=" * 60)

    n = len(embeddings)

    # Compute similarity decay with temporal distance
    max_lag = 30
    distance_sims = defaultdict(list)

    # Use cosine similarity
    for i in range(n):
        for j in range(i + 1, min(i + max_lag + 1, n)):
            temporal_dist = j - i
            # Cosine similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            distance_sims[temporal_dist].append(sim)

    # Compute statistics at each distance
    decay_data = []
    for dist in sorted(distance_sims.keys()):
        sims = distance_sims[dist]
        decay_data.append({
            'distance_days': dist,
            'mean_similarity': np.mean(sims),
            'std_similarity': np.std(sims),
            'n_pairs': len(sims)
        })

    decay_df = pd.DataFrame(decay_data)

    # Compute autocorrelation of first principal component
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(embeddings).flatten()
    autocorr = [np.corrcoef(pc1[:-lag], pc1[lag:])[0, 1] for lag in range(1, 31)]

    print(f"  Lag-1 autocorrelation of PC1: {autocorr[0]:.4f}")
    print(f"  Lag-7 autocorrelation of PC1: {autocorr[6]:.4f}")
    print(f"  Lag-30 autocorrelation of PC1: {autocorr[29]:.4f}")

    # Similarity at lag 1 vs lag 30
    sim_lag1 = decay_df[decay_df['distance_days'] == 1]['mean_similarity'].values[0]
    sim_lag30 = decay_df[decay_df['distance_days'] == 30]['mean_similarity'].values[0]
    decay_rate = (sim_lag1 - sim_lag30) / sim_lag1
    print(f"\n  Mean similarity at lag 1: {sim_lag1:.4f}")
    print(f"  Mean similarity at lag 30: {sim_lag30:.4f}")
    print(f"  Decay rate: {decay_rate*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Similarity decay
    ax1 = axes[0]
    ax1.errorbar(decay_df['distance_days'], decay_df['mean_similarity'],
                yerr=decay_df['std_similarity'], fmt='o-', capsize=3,
                markersize=4, alpha=0.7)
    ax1.set_xlabel('Temporal Distance (days)')
    ax1.set_ylabel('Mean Cosine Similarity')
    ax1.set_title('Embedding Similarity vs Temporal Distance')
    ax1.grid(True, alpha=0.3)

    # Autocorrelation
    ax2 = axes[1]
    ax2.bar(range(1, 31), autocorr, alpha=0.7, color='steelblue')
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Threshold (0.7)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Lag (days)')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('PC1 Autocorrelation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_coherence.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'lag1_autocorr': autocorr[0],
        'lag7_autocorr': autocorr[6],
        'lag30_autocorr': autocorr[29],
        'similarity_lag1': sim_lag1,
        'similarity_lag30': sim_lag30,
        'decay_rate': decay_rate,
    }


# =============================================================================
# 3. CLUSTERING ANALYSIS
# =============================================================================

def analyze_clusters(embeddings: np.ndarray, dates: List[str]) -> Dict:
    """
    Find optimal clustering and analyze temporal distribution of clusters.
    """
    print("\n" + "=" * 60)
    print("3. CLUSTERING ANALYSIS")
    print("=" * 60)

    # Reduce dimensionality for clustering (noise reduction)
    pca = PCA(n_components=50)
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"  Reduced to 50 PCs (explains {sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

    # Find optimal k using silhouette score
    k_range = range(3, 12)
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_reduced)
        sil = silhouette_score(embeddings_reduced, labels)
        silhouettes.append((k, sil))
        print(f"  k={k}: silhouette={sil:.4f}")

    optimal_k = max(silhouettes, key=lambda x: x[1])[0]
    optimal_sil = max(silhouettes, key=lambda x: x[1])[1]
    print(f"\n  Optimal k: {optimal_k} (silhouette: {optimal_sil:.4f})")

    # Fit with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_reduced)

    # Analyze temporal distribution of clusters
    dates_dt = pd.to_datetime(dates)
    cluster_stats = []

    for cluster_id in range(optimal_k):
        cluster_mask = labels == cluster_id
        cluster_dates = dates_dt[cluster_mask]

        if len(cluster_dates) > 0:
            cluster_stats.append({
                'cluster': cluster_id,
                'count': len(cluster_dates),
                'start': cluster_dates.min(),
                'end': cluster_dates.max(),
                'median': cluster_dates[len(cluster_dates)//2],
            })

    # Sort clusters by median date
    cluster_stats = sorted(cluster_stats, key=lambda x: x['median'])

    print("\n  Cluster temporal distribution:")
    for cs in cluster_stats:
        print(f"    Cluster {cs['cluster']}: {cs['count']} samples, "
              f"{cs['start'].date()} to {cs['end'].date()}")

    # Plot clustering results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Silhouette scores
    ax1 = axes[0, 0]
    ks, sils = zip(*silhouettes)
    ax1.plot(ks, sils, 'o-', markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs Number of Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Temporal distribution of clusters
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
    for cluster_id in range(optimal_k):
        cluster_mask = labels == cluster_id
        cluster_dates = dates_dt[cluster_mask]
        ax2.scatter(cluster_dates, [cluster_id] * len(cluster_dates),
                   c=[colors[cluster_id]], alpha=0.5, s=10,
                   label=f'Cluster {cluster_id}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cluster')
    ax2.set_title('Cluster Assignments Over Time')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # PCA projection colored by cluster
    ax3 = axes[1, 0]
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings)
    for cluster_id in range(optimal_k):
        mask = labels == cluster_id
        ax3.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[cluster_id]], alpha=0.5, s=10,
                   label=f'Cluster {cluster_id}')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_title('PCA Projection Colored by Cluster')
    ax3.legend(loc='upper right', fontsize=8)

    # Cluster sizes
    ax4 = axes[1, 1]
    cluster_sizes = [labels[labels == c].sum() for c in range(optimal_k)]
    # Actually count correctly
    cluster_counts = [sum(labels == c) for c in range(optimal_k)]
    ax4.bar(range(optimal_k), cluster_counts, color=colors)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Cluster Sizes')
    ax4.set_xticks(range(optimal_k))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clustering_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'optimal_k': optimal_k,
        'optimal_silhouette': optimal_sil,
        'silhouette_threshold_pass': optimal_sil > 0.15,
        'cluster_labels': labels.tolist(),
        'cluster_stats': cluster_stats,
    }


# =============================================================================
# 4. EVENT ALIGNMENT ANALYSIS
# =============================================================================

def analyze_event_alignment(embeddings: np.ndarray, dates: List[str],
                           events: Dict[str, str], window_days: int = 7) -> Dict:
    """
    Analyze embedding changes around major conflict events.
    """
    print("\n" + "=" * 60)
    print("4. EVENT ALIGNMENT ANALYSIS")
    print("=" * 60)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    dates_dt = pd.to_datetime(dates)

    event_changes = []

    for event_date_str, event_name in events.items():
        event_dt = pd.to_datetime(event_date_str)

        # Check if event is within our date range
        if event_dt < dates_dt.min() or event_dt > dates_dt.max():
            continue

        # Find embeddings around event
        pre_indices = []
        post_indices = []

        for d, idx in date_to_idx.items():
            dt = pd.to_datetime(d)
            delta = (dt - event_dt).days
            if -window_days <= delta < 0:
                pre_indices.append(idx)
            elif 0 < delta <= window_days:
                post_indices.append(idx)

        if len(pre_indices) >= 3 and len(post_indices) >= 3:
            pre_mean = embeddings[pre_indices].mean(axis=0)
            post_mean = embeddings[post_indices].mean(axis=0)

            # Cosine distance
            cos_sim = np.dot(pre_mean, post_mean) / (
                np.linalg.norm(pre_mean) * np.linalg.norm(post_mean)
            )
            cos_dist = 1 - cos_sim

            # Euclidean distance (normalized)
            euc_dist = np.linalg.norm(post_mean - pre_mean) / np.linalg.norm(pre_mean)

            event_changes.append({
                'event': event_name,
                'date': event_date_str,
                'cosine_distance': cos_dist,
                'euclidean_distance': euc_dist,
                'pre_samples': len(pre_indices),
                'post_samples': len(post_indices),
            })

    event_df = pd.DataFrame(event_changes)

    if len(event_df) > 0:
        event_df = event_df.sort_values('date')

        print(f"\n  Analyzed {len(event_df)} events within date range")
        print(f"  Mean cosine distance: {event_df['cosine_distance'].mean():.4f}")
        print(f"  Max cosine distance: {event_df['cosine_distance'].max():.4f}")

        # Top events by embedding shift
        print("\n  Top 5 events by embedding shift:")
        top_events = event_df.nlargest(5, 'cosine_distance')
        for _, row in top_events.iterrows():
            print(f"    {row['date']}: {row['event']} (cos_dist={row['cosine_distance']:.4f})")

        # Compute baseline (random pairs at same temporal distance)
        n_baseline = 1000
        baseline_distances = []
        for _ in range(n_baseline):
            i = np.random.randint(window_days, len(embeddings) - window_days)
            pre_idx = list(range(i - window_days, i))
            post_idx = list(range(i + 1, i + window_days + 1))
            pre_mean = embeddings[pre_idx].mean(axis=0)
            post_mean = embeddings[post_idx].mean(axis=0)
            cos_sim = np.dot(pre_mean, post_mean) / (
                np.linalg.norm(pre_mean) * np.linalg.norm(post_mean)
            )
            baseline_distances.append(1 - cos_sim)

        baseline_mean = np.mean(baseline_distances)
        baseline_std = np.std(baseline_distances)

        # Z-scores for each event
        event_df['z_score'] = (event_df['cosine_distance'] - baseline_mean) / baseline_std
        event_df['significant'] = event_df['z_score'] > 2.0

        print(f"\n  Baseline cosine distance: {baseline_mean:.4f} +/- {baseline_std:.4f}")
        print(f"  Events with significant shift (z > 2): {event_df['significant'].sum()}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Event distances
        ax1 = axes[0]
        colors = ['red' if sig else 'steelblue' for sig in event_df['significant']]
        ax1.barh(range(len(event_df)), event_df['cosine_distance'], color=colors, alpha=0.7)
        ax1.axvline(x=baseline_mean, color='green', linestyle='--',
                   label=f'Baseline mean: {baseline_mean:.4f}')
        ax1.axvline(x=baseline_mean + 2*baseline_std, color='orange', linestyle='--',
                   label=f'2 sigma: {baseline_mean + 2*baseline_std:.4f}')
        ax1.set_yticks(range(len(event_df)))
        ax1.set_yticklabels([f"{r['date'][:7]}: {r['event'][:25]}..."
                            if len(r['event']) > 25 else f"{r['date'][:7]}: {r['event']}"
                            for _, r in event_df.iterrows()], fontsize=8)
        ax1.set_xlabel('Cosine Distance (pre vs post event)')
        ax1.set_title('Embedding Shift Around Major Events')
        ax1.legend(loc='lower right', fontsize=8)

        # Z-scores
        ax2 = axes[1]
        colors = ['red' if sig else 'steelblue' for sig in event_df['significant']]
        ax2.barh(range(len(event_df)), event_df['z_score'], color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=2, color='orange', linestyle='--', label='Significance threshold (z=2)')
        ax2.set_yticks(range(len(event_df)))
        ax2.set_yticklabels(['' for _ in event_df.iterrows()])  # Hide for cleaner look
        ax2.set_xlabel('Z-Score')
        ax2.set_title('Statistical Significance of Embedding Shifts')
        ax2.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'event_alignment.png', dpi=150, bbox_inches='tight')
        plt.close()

        return {
            'n_events_analyzed': len(event_df),
            'mean_cosine_distance': event_df['cosine_distance'].mean(),
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'n_significant': int(event_df['significant'].sum()),
            'threshold_pass': event_df['cosine_distance'].mean() > 0.02,
            'event_details': event_df.to_dict('records'),
        }

    return {'n_events_analyzed': 0, 'threshold_pass': False}


# =============================================================================
# 5. SUMMARY AND RECOMMENDATIONS
# =============================================================================

def generate_summary(results: Dict) -> None:
    """Generate summary report and overall assessment."""
    print("\n" + "=" * 60)
    print("EMBEDDING QUALITY SUMMARY")
    print("=" * 60)

    checks = []

    # Dimensionality check
    dim_result = results.get('dimensionality', {})
    intrinsic_dim = dim_result.get('elbow_estimate', 0)
    dim_check = 50 <= intrinsic_dim <= 200
    checks.append(('Intrinsic dimensionality (50-200)', dim_check, intrinsic_dim))
    print(f"\n  1. Intrinsic dimensionality: {intrinsic_dim}")
    print(f"     {'PASS' if dim_check else 'WARN'}: Expected 50-200 dimensions")

    # Temporal coherence check
    temp_result = results.get('temporal_coherence', {})
    autocorr = temp_result.get('lag1_autocorr', 0)
    temp_check = autocorr > 0.7
    checks.append(('Temporal coherence (autocorr > 0.7)', temp_check, autocorr))
    print(f"\n  2. Lag-1 autocorrelation: {autocorr:.4f}")
    print(f"     {'PASS' if temp_check else 'FAIL'}: Expected > 0.7")

    # Clustering check
    cluster_result = results.get('clustering', {})
    silhouette = cluster_result.get('optimal_silhouette', 0)
    cluster_check = silhouette > 0.15
    checks.append(('Cluster quality (silhouette > 0.15)', cluster_check, silhouette))
    print(f"\n  3. Silhouette score: {silhouette:.4f}")
    print(f"     {'PASS' if cluster_check else 'FAIL'}: Expected > 0.15")

    # Event alignment check
    event_result = results.get('event_alignment', {})
    mean_shift = event_result.get('mean_cosine_distance', 0)
    event_check = mean_shift > 0.02
    checks.append(('Event alignment (cos_dist > 0.02)', event_check, mean_shift))
    print(f"\n  4. Mean event shift: {mean_shift:.4f}")
    print(f"     {'PASS' if event_check else 'FAIL'}: Expected > 0.02")

    # Overall assessment
    n_pass = sum(1 for _, passed, _ in checks if passed)
    overall_pass = n_pass >= 3

    print("\n" + "-" * 60)
    print(f"OVERALL ASSESSMENT: {n_pass}/4 checks passed")
    if overall_pass:
        print("RECOMMENDATION: Embeddings appear suitable for integration")
        print("  - Proceed with two-stream architecture")
        print(f"  - Reduce to {min(intrinsic_dim, 128)} dimensions via PCA")
        print("  - Use learned missing token for gaps")
    else:
        print("WARNING: Embeddings may not capture meaningful dynamics")
        print("  - Consider re-generating with different model")
        print("  - Or use simpler bag-of-words features")
        print("  - Or proceed with caution and extensive ablation")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'checks': [{'name': name, 'passed': bool(passed), 'value': float(val)}
                   for name, passed, val in checks],
        'overall_pass': bool(overall_pass),
        'recommendation': 'proceed' if overall_pass else 'caution',
    }

    with open(OUTPUT_DIR / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR}")


def main():
    """Run full embedding validation pipeline."""
    print("=" * 60)
    print("ISW EMBEDDING QUALITY VALIDATION")
    print("=" * 60)

    # Load embeddings
    embeddings, dates = load_embeddings()

    results = {}

    # 1. Dimensionality analysis
    results['dimensionality'] = analyze_dimensionality(embeddings)

    # 2. Temporal coherence
    results['temporal_coherence'] = analyze_temporal_coherence(embeddings, dates)

    # 3. Clustering
    results['clustering'] = analyze_clusters(embeddings, dates)

    # 4. Event alignment
    results['event_alignment'] = analyze_event_alignment(embeddings, dates, MAJOR_EVENTS)

    # 5. Summary
    generate_summary(results)

    # Save full results
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

    with open(OUTPUT_DIR / 'full_validation_results.json', 'w') as f:
        json.dump(make_serializable(results), f, indent=2)


if __name__ == "__main__":
    main()
