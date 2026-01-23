#!/usr/bin/env python3
"""
Comprehensive Analysis of Hierarchical Attention Network Feature Associations

This script loads the trained HAN model and analyzes:
1. Domain-level attention - which data sources matter most
2. Feature-level attention - which features within each domain are important
3. Temporal patterns - how attention changes over time
4. Cross-domain correlations - relationships between different data sources
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict

from hierarchical_attention_network import (
    HierarchicalAttentionNetwork,
    DOMAIN_CONFIGS,
    TOTAL_FEATURES
)
from conflict_data_loader import RealConflictDataset, create_data_loaders

ANALYSIS_DIR = Path(__file__).parent
MODEL_DIR = ANALYSIS_DIR / "models"


def load_trained_model(model_path: Path = None, device: str = 'cpu'):
    """Load the best trained model."""
    if model_path is None:
        model_path = MODEL_DIR / 'han_best.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create model with best configuration
    model = HierarchicalAttentionNetwork(
        domain_configs=DOMAIN_CONFIGS,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_temporal_layers=1,
        dropout=0.35
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss={checkpoint['val_loss']:.4f}")

    return model


def extract_attention_weights(model, data_loader, device='cpu'):
    """
    Run inference on data and extract all attention weights.

    Returns:
        domain_attentions: [n_samples, seq_len, n_domains] - importance of each domain
        feature_attentions: dict[domain] -> [n_samples, seq_len, n_features]
        raw_features: dict[domain] -> [n_samples, seq_len, n_features]
    """
    model.eval()

    all_domain_attn = []
    all_feature_attn = {name: [] for name in DOMAIN_CONFIGS.keys()}
    all_features = {name: [] for name in DOMAIN_CONFIGS.keys()}
    all_outputs = []

    with torch.no_grad():
        for features, masks, targets in data_loader:
            # Move to device
            features = {k: v.to(device) for k, v in features.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            # Forward pass with attention
            outputs = model(features, masks, return_attention=True)

            # Collect domain attention
            if 'domain_attention' in outputs:
                all_domain_attn.append(outputs['domain_attention'].cpu().numpy())

            # Collect feature attention per domain
            if 'feature_attention' in outputs:
                for domain_name, attn in outputs['feature_attention'].items():
                    all_feature_attn[domain_name].append(attn.cpu().numpy())

            # Collect raw features
            for domain_name, feat in features.items():
                all_features[domain_name].append(feat.cpu().numpy())

            all_outputs.append({
                'forecast': outputs['forecast'].cpu().numpy(),
                'regime_logits': outputs['regime_logits'].cpu().numpy(),
                'anomaly_score': outputs['anomaly_score'].cpu().numpy()
            })

    # Concatenate all batches
    domain_attentions = np.concatenate(all_domain_attn, axis=0) if all_domain_attn else None

    feature_attentions = {}
    for name, attn_list in all_feature_attn.items():
        if attn_list:
            feature_attentions[name] = np.concatenate(attn_list, axis=0)

    raw_features = {}
    for name, feat_list in all_features.items():
        if feat_list:
            raw_features[name] = np.concatenate(feat_list, axis=0)

    return domain_attentions, feature_attentions, raw_features, all_outputs


def analyze_domain_importance(domain_attentions, domain_names):
    """Analyze which domains the model focuses on most."""
    print("\n" + "=" * 80)
    print("DOMAIN-LEVEL ATTENTION ANALYSIS")
    print("=" * 80)

    # Average attention across all samples and timesteps
    mean_attn = domain_attentions.mean(axis=(0, 1))
    std_attn = domain_attentions.std(axis=(0, 1))

    # Sort by importance
    sorted_idx = np.argsort(mean_attn)[::-1]

    print("\nDomain Importance Ranking (averaged across all samples and timesteps):")
    print("-" * 60)

    results = []
    for rank, idx in enumerate(sorted_idx, 1):
        name = domain_names[idx]
        importance = mean_attn[idx]
        std = std_attn[idx]
        config = DOMAIN_CONFIGS[name]

        print(f"  {rank}. {config.name:<30} {importance:.4f} ± {std:.4f}")
        results.append({
            'rank': rank,
            'domain': name,
            'full_name': config.name,
            'importance': importance,
            'std': std,
            'num_features': config.num_features
        })

    # Temporal analysis
    print("\nTemporal Evolution of Domain Attention:")
    print("-" * 60)

    # Average across samples, keeping time dimension
    temporal_attn = domain_attentions.mean(axis=0)  # [seq_len, n_domains]

    print(f"  {'Timestep':<10}", end="")
    for name in domain_names:
        print(f"{name[:8]:<10}", end="")
    print()

    for t in range(temporal_attn.shape[0]):
        print(f"  t={t:<7}", end="")
        for d in range(temporal_attn.shape[1]):
            print(f"{temporal_attn[t, d]:.4f}    ", end="")
        print()

    return pd.DataFrame(results)


def analyze_feature_importance(feature_attentions, domain_configs):
    """Analyze which features within each domain are most important."""
    print("\n" + "=" * 80)
    print("FEATURE-LEVEL ATTENTION ANALYSIS")
    print("=" * 80)

    all_results = {}

    for domain_name, attn in feature_attentions.items():
        config = domain_configs[domain_name]

        print(f"\n{config.name} ({domain_name.upper()}):")
        print("-" * 60)

        # Average attention across samples and timesteps
        mean_attn = attn.mean(axis=(0, 1))
        std_attn = attn.std(axis=(0, 1))

        # Sort by importance
        sorted_idx = np.argsort(mean_attn)[::-1]

        # Show top 10 features
        print(f"  Top 10 Most Important Features:")
        results = []
        for rank, idx in enumerate(sorted_idx[:10], 1):
            feat_name = config.feature_names[idx] if idx < len(config.feature_names) else f"feature_{idx}"
            importance = mean_attn[idx]
            std = std_attn[idx]
            print(f"    {rank:2d}. {feat_name:<35} {importance:.4f} ± {std:.4f}")
            results.append({
                'rank': rank,
                'feature': feat_name,
                'importance': importance,
                'std': std
            })

        # Show bottom 5 (least important)
        print(f"\n  Bottom 5 Least Important Features:")
        for rank, idx in enumerate(sorted_idx[-5:], 1):
            feat_name = config.feature_names[idx] if idx < len(config.feature_names) else f"feature_{idx}"
            importance = mean_attn[idx]
            print(f"    {rank:2d}. {feat_name:<35} {importance:.4f}")

        all_results[domain_name] = pd.DataFrame(results)

    return all_results


def analyze_cross_domain_correlations(domain_attentions, raw_features, domain_names):
    """Analyze correlations between domain attention and feature values."""
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN CORRELATION ANALYSIS")
    print("=" * 80)

    # Flatten domain attention: [n_samples * seq_len, n_domains]
    n_samples, seq_len, n_domains = domain_attentions.shape
    flat_domain_attn = domain_attentions.reshape(-1, n_domains)

    print("\nDomain-to-Domain Attention Correlations:")
    print("-" * 60)
    print("  (Positive = domains attended together, Negative = trade-off)")
    print()

    # Compute correlation matrix between domains
    corr_matrix = np.corrcoef(flat_domain_attn.T)

    # Print correlation matrix
    print(f"  {'':>12}", end="")
    for name in domain_names:
        print(f"{name[:8]:>10}", end="")
    print()

    for i, name_i in enumerate(domain_names):
        print(f"  {name_i[:10]:>12}", end="")
        for j in range(len(domain_names)):
            corr = corr_matrix[i, j]
            if i == j:
                print(f"{'1.00':>10}", end="")
            else:
                print(f"{corr:>10.3f}", end="")
        print()

    # Find strongest correlations (excluding diagonal)
    print("\n  Strongest Domain Relationships:")
    correlations = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            correlations.append({
                'domain1': domain_names[i],
                'domain2': domain_names[j],
                'correlation': corr_matrix[i, j]
            })

    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    for item in correlations[:6]:
        d1, d2, corr = item['domain1'], item['domain2'], item['correlation']
        relationship = "positively correlated" if corr > 0 else "negatively correlated"
        print(f"    {d1} ↔ {d2}: {corr:+.3f} ({relationship})")

    return corr_matrix


def analyze_temporal_patterns(domain_attentions, feature_attentions, domain_names):
    """Analyze how attention patterns evolve over the sequence."""
    print("\n" + "=" * 80)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)

    n_samples, seq_len, n_domains = domain_attentions.shape

    # Calculate attention volatility (how much attention changes between timesteps)
    print("\nDomain Attention Volatility (std across timesteps):")
    print("-" * 60)

    # For each sample, compute std of attention across time
    temporal_std = domain_attentions.std(axis=1).mean(axis=0)  # [n_domains]

    sorted_idx = np.argsort(temporal_std)[::-1]
    for idx in sorted_idx:
        name = domain_names[idx]
        config = DOMAIN_CONFIGS[name]
        print(f"  {config.name:<30} volatility: {temporal_std[idx]:.4f}")

    # Early vs Late attention shift
    print("\nEarly vs Late Sequence Attention Shift:")
    print("-" * 60)

    mid = seq_len // 2
    early_attn = domain_attentions[:, :mid, :].mean(axis=(0, 1))
    late_attn = domain_attentions[:, mid:, :].mean(axis=(0, 1))
    shift = late_attn - early_attn

    print(f"  {'Domain':<30} {'Early':>10} {'Late':>10} {'Shift':>10}")
    for idx, name in enumerate(domain_names):
        config = DOMAIN_CONFIGS[name]
        direction = "↑" if shift[idx] > 0.01 else ("↓" if shift[idx] < -0.01 else "→")
        print(f"  {config.name:<30} {early_attn[idx]:>10.4f} {late_attn[idx]:>10.4f} {shift[idx]:>+9.4f} {direction}")


def analyze_regime_associations(outputs, domain_attentions, domain_names):
    """Analyze how attention patterns relate to predicted conflict regimes."""
    print("\n" + "=" * 80)
    print("REGIME CLASSIFICATION ANALYSIS")
    print("=" * 80)

    regime_names = ['Low Intensity', 'Medium', 'High', 'Major Offensive']

    # Get predicted regimes from logits
    all_logits = np.concatenate([o['regime_logits'] for o in outputs], axis=0)
    # Use last timestep for regime prediction
    final_logits = all_logits[:, -1, :]  # [n_samples, 4]
    predicted_regimes = final_logits.argmax(axis=-1)
    regime_probs = np.exp(final_logits) / np.exp(final_logits).sum(axis=-1, keepdims=True)

    print("\nPredicted Regime Distribution:")
    print("-" * 60)
    unique, counts = np.unique(predicted_regimes, return_counts=True)
    for regime_id, count in zip(unique, counts):
        pct = count / len(predicted_regimes) * 100
        print(f"  {regime_names[regime_id]:<20}: {count} samples ({pct:.1f}%)")

    # Average attention by predicted regime
    print("\nDomain Attention by Predicted Regime:")
    print("-" * 60)

    # Use attention from last timestep
    final_attn = domain_attentions[:, -1, :]  # [n_samples, n_domains]

    print(f"  {'Regime':<20}", end="")
    for name in domain_names:
        print(f"{name[:8]:>10}", end="")
    print()

    for regime_id in range(4):
        mask = predicted_regimes == regime_id
        if mask.sum() > 0:
            regime_attn = final_attn[mask].mean(axis=0)
            print(f"  {regime_names[regime_id]:<20}", end="")
            for val in regime_attn:
                print(f"{val:>10.4f}", end="")
            print()


def generate_summary_insights(domain_df, feature_results, corr_matrix, domain_names):
    """Generate high-level insights from the analysis."""
    print("\n" + "=" * 80)
    print("KEY INSIGHTS AND FINDINGS")
    print("=" * 80)

    print("\n1. MOST INFLUENTIAL DATA SOURCES:")
    print("-" * 60)
    top_domains = domain_df.head(3)
    for _, row in top_domains.iterrows():
        print(f"   • {row['full_name']}: {row['importance']:.1%} average attention")
        print(f"     ({row['num_features']} features, resolution: {DOMAIN_CONFIGS[row['domain']].native_resolution})")

    print("\n2. KEY PREDICTIVE FEATURES BY DOMAIN:")
    print("-" * 60)
    for domain_name, df in feature_results.items():
        config = DOMAIN_CONFIGS[domain_name]
        top_feat = df.iloc[0]
        print(f"   • {config.name}: '{top_feat['feature']}' ({top_feat['importance']:.1%})")

    print("\n3. CROSS-DOMAIN RELATIONSHIPS:")
    print("-" * 60)

    # Find strongest positive and negative correlations
    correlations = []
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            correlations.append((domain_names[i], domain_names[j], corr_matrix[i, j]))

    correlations.sort(key=lambda x: x[2], reverse=True)

    print("   Strongest positive (domains used together):")
    for d1, d2, corr in correlations[:2]:
        if corr > 0:
            print(f"     • {DOMAIN_CONFIGS[d1].name} ↔ {DOMAIN_CONFIGS[d2].name}: r={corr:.3f}")

    print("\n   Strongest negative (attention trade-offs):")
    for d1, d2, corr in sorted(correlations, key=lambda x: x[2])[:2]:
        if corr < 0:
            print(f"     • {DOMAIN_CONFIGS[d1].name} ↔ {DOMAIN_CONFIGS[d2].name}: r={corr:.3f}")

    print("\n4. INTERPRETATION:")
    print("-" * 60)

    # Get top domain
    top_domain = domain_df.iloc[0]['domain']
    top_domain_name = DOMAIN_CONFIGS[top_domain].name

    # Get most important feature in top domain
    if top_domain in feature_results:
        top_feature = feature_results[top_domain].iloc[0]['feature']
    else:
        top_feature = "N/A"

    print(f"""
   The model has learned to prioritize {top_domain_name} as the most
   informative data source for conflict prediction, with particular focus on
   the '{top_feature}' feature.

   This suggests that {top_domain.upper()} data provides the strongest signal
   for forecasting conflict dynamics in Ukraine.
""")


def main():
    print("=" * 80)
    print("HIERARCHICAL ATTENTION NETWORK - FEATURE ASSOCIATION ANALYSIS")
    print("=" * 80)

    # Load model
    print("\nLoading trained model...")
    try:
        model = load_trained_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python train_han.py")
        return

    # Create data loaders (use all data for analysis)
    print("\nLoading data...")
    train_loader, val_loader = create_data_loaders(
        DOMAIN_CONFIGS,
        batch_size=4,
        seq_len=4
    )

    # Combine train and val for comprehensive analysis
    from torch.utils.data import ConcatDataset, DataLoader

    # Use validation data for analysis (unseen during training)
    print("\nExtracting attention weights from validation data...")
    domain_attentions, feature_attentions, raw_features, outputs = extract_attention_weights(
        model, val_loader
    )

    if domain_attentions is None or len(domain_attentions) == 0:
        print("Error: No attention weights extracted. Check model output.")
        return

    domain_names = list(DOMAIN_CONFIGS.keys())

    print(f"\nAnalyzing {domain_attentions.shape[0]} samples, "
          f"{domain_attentions.shape[1]} timesteps, "
          f"{domain_attentions.shape[2]} domains")

    # Run all analyses
    domain_df = analyze_domain_importance(domain_attentions, domain_names)
    feature_results = analyze_feature_importance(feature_attentions, DOMAIN_CONFIGS)
    corr_matrix = analyze_cross_domain_correlations(domain_attentions, raw_features, domain_names)
    analyze_temporal_patterns(domain_attentions, feature_attentions, domain_names)
    analyze_regime_associations(outputs, domain_attentions, domain_names)
    generate_summary_insights(domain_df, feature_results, corr_matrix, domain_names)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
