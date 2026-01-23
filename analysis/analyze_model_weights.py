#!/usr/bin/env python3
"""
Ultra-Detailed Model Weight Analysis for Unified Cross-Source Interpolation Model

This script performs deep analysis of:
1. Source encoder weights - what features each source finds important
2. Cross-source attention patterns - how sources inform each other
3. Decoder weights - reconstruction patterns
4. Source embeddings - learned source representations
5. Unified projection - how sources combine for final output

Author: Analysis script for ML_OSINT project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from unified_interpolation import (
    UnifiedInterpolationModel,
    SOURCE_CONFIGS,
    MODEL_DIR
)

ANALYSIS_DIR = Path(__file__).parent


def load_unified_model(model_path: Path = None, device: str = 'cpu'):
    """Load the trained unified interpolation model."""
    if model_path is None:
        model_path = MODEL_DIR / 'unified_interpolation_best.pt'

    if not model_path.exists():
        # Try alternates
        for alt in ['unified_interpolation_hybrid_best.pt', 'unified_interpolation_delta_best.pt']:
            alt_path = MODEL_DIR / alt
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    # Load checkpoint first to infer feature counts
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Infer feature counts from checkpoint shapes
    from copy import deepcopy
    source_configs_adjusted = deepcopy(SOURCE_CONFIGS)

    for key in state_dict.keys():
        if 'encoder' in key and 'feature_proj.0.weight' in key:
            # Extract source name from key like "encoders.equipment.feature_proj.0.weight"
            parts = key.split('.')
            source_name = parts[1]
            n_features = state_dict[key].shape[1]  # [d_embed*2, n_features]
            if source_name in source_configs_adjusted:
                source_configs_adjusted[source_name].n_features = n_features
                print(f"  {source_name}: {n_features} features")

    # Create model architecture with inferred feature counts
    model = UnifiedInterpolationModel(
        source_configs=source_configs_adjusted,
        d_embed=64,
        nhead=4,
        num_fusion_layers=2,
        dropout=0.1
    )

    # Load weights
    model.load_state_dict(state_dict)
    if 'model_state_dict' in checkpoint:
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'unknown'))
        print(f"Loaded model from epoch {epoch} with val_loss={val_loss}")
    else:
        print("Loaded model (no metadata)")

    model.eval()
    return model, source_configs_adjusted


def get_feature_names():
    """Get feature names for each source."""
    feature_names = {}

    # DeepState features (55)
    feature_names['deepstate'] = [
        'arrows_count', 'arrows_north', 'arrows_south', 'arrows_east', 'arrows_west',
        'polygons_count', 'polygons_area', 'polygons_new', 'polygons_removed',
        'units_count', 'units_infantry', 'units_armor', 'units_artillery',
        'airfields_active', 'airfields_damaged',
    ] + [f'ds_feat_{i}' for i in range(15, 55)]

    # Equipment features (38)
    feature_names['equipment'] = [
        'tanks_destroyed', 'tanks_captured', 'tanks_damaged',
        'afv_destroyed', 'afv_captured', 'afv_damaged',
        'ifv_destroyed', 'ifv_captured',
        'apc_destroyed', 'apc_captured',
        'aircraft_destroyed', 'helicopters_destroyed',
        'drones_destroyed', 'naval_destroyed',
        'artillery_destroyed', 'mlrs_destroyed',
        'air_defense_destroyed', 'vehicles_destroyed',
        'fuel_tankers', 'engineering_vehicles',
    ] + [f'equip_feat_{i}' for i in range(20, 38)]

    # FIRMS features (42)
    feature_names['firms'] = [
        'fire_count', 'frp_mean', 'frp_max', 'frp_sum',
        'brightness_t4', 'brightness_t5', 'brightness_mean',
        'confidence_high', 'confidence_medium', 'confidence_low',
        'day_fires', 'night_fires', 'scan_angle_mean', 'fire_density',
    ] + [f'firms_feat_{i}' for i in range(14, 42)]

    # UCDP features (48)
    feature_names['ucdp'] = [
        'events_total', 'events_state_based', 'events_non_state', 'events_one_sided',
        'deaths_best', 'deaths_low', 'deaths_high',
        'deaths_civilians', 'deaths_side_a', 'deaths_side_b',
        'events_donetsk', 'events_luhansk', 'events_kharkiv', 'events_kherson',
        'events_zaporizhzhia', 'events_mykolaiv',
        'precision_high', 'precision_medium', 'precision_low',
    ] + [f'ucdp_feat_{i}' for i in range(19, 48)]

    # VIINA features (24)
    feature_names['viina'] = [
        'localities_ua_control', 'localities_ru_control',
        'localities_contested', 'localities_unknown',
        'pct_ua_control', 'pct_ru_control', 'pct_contested',
        'localities_gained_ua', 'localities_lost_ua',
        'localities_gained_ru', 'localities_lost_ru',
        'sources_agree_pct', 'wiki_dsm_agree', 'wiki_isw_agree', 'dsm_isw_agree',
        'ua_control_7day_avg', 'ru_control_7day_avg',
        'daily_change_7day_avg', 'control_volatility',
        'front_activity_index', 'control_momentum',
        'days_since_war_start', 'total_localities', 'data_completeness',
    ]

    # HDX Conflict features (18)
    feature_names['hdx_conflict'] = [
        'events_total', 'events_civilian_targeting', 'events_battles',
        'events_explosions', 'events_protests', 'events_other',
        'events_donetsk', 'events_kharkiv', 'events_kherson',
        'events_zaporizhzhia', 'events_other_regions',
        'fatalities_total', 'fatalities_per_event', 'fatalities_max_event',
        'intensity_index', 'regional_spread', 'month_idx', 'days_in_period',
    ]

    # HDX Food features (20)
    feature_names['hdx_food'] = [
        'avg_price', 'median_price', 'price_std', 'price_range',
        'cereals_avg', 'vegetables_avg', 'meat_avg', 'dairy_avg', 'oils_avg',
        'price_change_pct', 'price_7day_trend', 'price_volatility',
        'inflation_proxy', 'food_security_index', 'price_anomaly_score',
        'n_commodities', 'n_markets', 'month_idx', 'days_in_period', 'data_coverage',
    ]

    # HDX Rainfall features (16)
    feature_names['hdx_rainfall'] = [
        'rainfall_mean', 'rainfall_median', 'rainfall_max', 'rainfall_std',
        'anomaly_pct_mean', 'above_normal_pct', 'below_normal_pct',
        'drought_risk_index', 'flood_risk_index', 'rainfall_vs_lta_ratio',
        'n_admin_regions', 'dekad_idx', 'days_in_period', 'season_idx',
        'data_coverage', 'anomaly_severity',
    ]

    # IOM Displacement features (18)
    feature_names['iom'] = [
        'total_idps', 'idps_male', 'idps_female',
        'idps_kyiv', 'idps_lviv', 'idps_dnipro', 'idps_kharkiv',
        'idps_zaporizhzhia', 'idps_other',
        'from_donetsk', 'from_luhansk', 'from_other_conflict',
        'displacement_intensity', 'gender_ratio', 'avg_per_region',
        'round_idx', 'days_since_start', 'data_completeness',
    ]

    return feature_names


def analyze_encoder_weights(model, feature_names):
    """Analyze source encoder weights to understand feature importance."""
    print("\n" + "="*80)
    print("1. SOURCE ENCODER ANALYSIS")
    print("="*80)

    encoder_analysis = {}

    for source_name, encoder in model.encoders.items():
        print(f"\n{'─'*60}")
        print(f"Source: {source_name.upper()}")
        print(f"{'─'*60}")

        # Get the feature projection weights
        if hasattr(encoder, 'feature_proj'):
            # First layer of feature_proj: [n_features] -> [d_embed*2]
            first_layer = encoder.feature_proj[0]
            weights = first_layer.weight.data.cpu().numpy()  # [d_embed*2, n_features]
            bias = first_layer.bias.data.cpu().numpy() if first_layer.bias is not None else None

            # Feature importance: L2 norm of weights for each input feature
            feature_importance = np.linalg.norm(weights, axis=0)
            feature_importance = feature_importance / feature_importance.sum()  # Normalize

            # Get feature names for this source
            feat_names = feature_names.get(source_name, [f'feat_{i}' for i in range(len(feature_importance))])
            if len(feat_names) < len(feature_importance):
                feat_names = feat_names + [f'feat_{i}' for i in range(len(feat_names), len(feature_importance))]

            # Sort by importance
            sorted_indices = np.argsort(feature_importance)[::-1]

            print(f"\n  Feature Importance (by encoder weight magnitude):")
            print(f"  Top 10 most important features:")
            for i, idx in enumerate(sorted_indices[:10]):
                name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
                print(f"    {i+1:2d}. {name:<35} importance: {feature_importance[idx]:.4f}")

            print(f"\n  Bottom 5 least important features:")
            for i, idx in enumerate(sorted_indices[-5:]):
                name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
                print(f"    {i+1:2d}. {name:<35} importance: {feature_importance[idx]:.4f}")

            # Analyze weight statistics
            print(f"\n  Weight Statistics:")
            print(f"    Shape: {weights.shape}")
            print(f"    Mean: {weights.mean():.6f}")
            print(f"    Std:  {weights.std():.6f}")
            print(f"    Min:  {weights.min():.6f}")
            print(f"    Max:  {weights.max():.6f}")

            encoder_analysis[source_name] = {
                'feature_importance': dict(zip(feat_names[:len(feature_importance)], feature_importance.tolist())),
                'weights_stats': {
                    'mean': float(weights.mean()),
                    'std': float(weights.std()),
                    'min': float(weights.min()),
                    'max': float(weights.max())
                }
            }

        # Output projection analysis
        output_weights = encoder.output_proj.weight.data.cpu().numpy()
        print(f"\n  Output Projection:")
        print(f"    Shape: {output_weights.shape}")
        print(f"    Effective rank: {np.linalg.norm(output_weights, 'nuc') / np.linalg.norm(output_weights, 'fro'):.4f}")

    return encoder_analysis


def analyze_source_embeddings(model):
    """Analyze learned source type embeddings."""
    print("\n" + "="*80)
    print("2. SOURCE TYPE EMBEDDINGS ANALYSIS")
    print("="*80)

    # Get source embeddings
    source_emb = model.fusion.source_embeddings.weight.data.cpu().numpy()
    source_names = model.source_names

    print(f"\nSource embedding shape: {source_emb.shape}")
    print(f"Number of sources: {len(source_names)}")

    # Compute pairwise cosine similarities
    norms = np.linalg.norm(source_emb, axis=1, keepdims=True)
    normalized = source_emb / (norms + 1e-8)
    similarity_matrix = normalized @ normalized.T

    print(f"\n  Source-to-Source Cosine Similarity Matrix:")
    print(f"  " + " " * 15 + "".join(f"{s[:8]:>10}" for s in source_names))
    for i, s1 in enumerate(source_names):
        row = f"  {s1[:14]:<14} "
        for j, s2 in enumerate(source_names):
            sim = similarity_matrix[i, j]
            row += f"{sim:>10.3f}"
        print(row)

    # Find most similar and dissimilar pairs
    print(f"\n  Most Similar Source Pairs:")
    pairs = []
    for i in range(len(source_names)):
        for j in range(i+1, len(source_names)):
            pairs.append((source_names[i], source_names[j], similarity_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for s1, s2, sim in pairs[:5]:
        print(f"    {s1} <-> {s2}: {sim:.4f}")

    print(f"\n  Most Dissimilar Source Pairs:")
    for s1, s2, sim in pairs[-5:]:
        print(f"    {s1} <-> {s2}: {sim:.4f}")

    # Analyze embedding dimensions
    print(f"\n  Embedding Dimension Analysis:")
    for i, name in enumerate(source_names):
        emb = source_emb[i]
        print(f"    {name:<15} L2 norm: {np.linalg.norm(emb):.4f}, "
              f"mean: {emb.mean():.4f}, std: {emb.std():.4f}")

    # Principal components analysis
    U, S, Vh = np.linalg.svd(source_emb, full_matrices=False)
    explained_var = (S ** 2) / (S ** 2).sum()

    print(f"\n  SVD Analysis of Source Embeddings:")
    print(f"    Singular values: {S[:5]}")
    print(f"    Explained variance (top 5): {explained_var[:5]}")
    print(f"    Cumulative variance (5 components): {explained_var[:5].sum():.4f}")

    return {
        'similarity_matrix': similarity_matrix,
        'source_names': source_names,
        'embeddings': source_emb,
        'singular_values': S
    }


def analyze_cross_source_attention(model):
    """Analyze cross-source attention patterns in the transformer."""
    print("\n" + "="*80)
    print("3. CROSS-SOURCE ATTENTION ANALYSIS")
    print("="*80)

    fusion = model.fusion
    source_names = model.source_names

    # Analyze transformer encoder layers
    for layer_idx, layer in enumerate(fusion.transformer.layers):
        print(f"\n{'─'*60}")
        print(f"Transformer Layer {layer_idx}")
        print(f"{'─'*60}")

        # Self-attention weights
        self_attn = layer.self_attn

        # Get Q, K, V projection weights
        in_proj = self_attn.in_proj_weight.data.cpu().numpy()
        d_embed = in_proj.shape[1]

        # Split into Q, K, V
        Wq = in_proj[:d_embed, :]
        Wk = in_proj[d_embed:2*d_embed, :]
        Wv = in_proj[2*d_embed:, :]

        print(f"\n  Query Projection (Wq):")
        print(f"    Shape: {Wq.shape}")
        print(f"    Frobenius norm: {np.linalg.norm(Wq, 'fro'):.4f}")
        print(f"    Spectral norm: {np.linalg.norm(Wq, 2):.4f}")

        print(f"\n  Key Projection (Wk):")
        print(f"    Shape: {Wk.shape}")
        print(f"    Frobenius norm: {np.linalg.norm(Wk, 'fro'):.4f}")
        print(f"    Spectral norm: {np.linalg.norm(Wk, 2):.4f}")

        print(f"\n  Value Projection (Wv):")
        print(f"    Shape: {Wv.shape}")
        print(f"    Frobenius norm: {np.linalg.norm(Wv, 'fro'):.4f}")
        print(f"    Spectral norm: {np.linalg.norm(Wv, 2):.4f}")

        # Analyze QK^T pattern (what queries attend to what keys)
        source_emb = model.fusion.source_embeddings.weight.data.cpu().numpy()

        # Compute attention scores for source embeddings
        Q_sources = source_emb @ Wq.T  # [n_sources, d_embed]
        K_sources = source_emb @ Wk.T  # [n_sources, d_embed]

        # Attention scores (before softmax)
        attn_scores = Q_sources @ K_sources.T / np.sqrt(d_embed)
        attn_probs = np.exp(attn_scores - attn_scores.max()) / np.exp(attn_scores - attn_scores.max()).sum(axis=1, keepdims=True)

        print(f"\n  Learned Attention Pattern (which source attends to which):")
        print(f"  Query\\Key    " + "".join(f"{s[:8]:>10}" for s in source_names))
        for i, s1 in enumerate(source_names):
            row = f"  {s1[:12]:<12} "
            for j, s2 in enumerate(source_names):
                prob = attn_probs[i, j]
                row += f"{prob:>10.3f}"
            print(row)

        # Identify strongest cross-source attention patterns
        print(f"\n  Strongest Cross-Source Attention Patterns:")
        patterns = []
        for i, s1 in enumerate(source_names):
            for j, s2 in enumerate(source_names):
                if i != j:  # Exclude self-attention
                    patterns.append((s1, s2, attn_probs[i, j]))
        patterns.sort(key=lambda x: x[2], reverse=True)

        for query, key, prob in patterns[:10]:
            print(f"    {query} attends to {key}: {prob:.4f}")

        # Feedforward analysis
        ff1 = layer.linear1.weight.data.cpu().numpy()
        ff2 = layer.linear2.weight.data.cpu().numpy()
        print(f"\n  Feedforward Network:")
        print(f"    FF1 shape: {ff1.shape}, norm: {np.linalg.norm(ff1, 'fro'):.4f}")
        print(f"    FF2 shape: {ff2.shape}, norm: {np.linalg.norm(ff2, 'fro'):.4f}")


def analyze_decoder_weights(model, feature_names):
    """Analyze source decoder weights to understand reconstruction patterns."""
    print("\n" + "="*80)
    print("4. SOURCE DECODER ANALYSIS")
    print("="*80)

    decoder_analysis = {}

    for source_name, decoder in model.decoders.items():
        print(f"\n{'─'*60}")
        print(f"Decoder for: {source_name.upper()}")
        print(f"{'─'*60}")

        # Get decoder layers
        layers = list(decoder.decoder.children())

        # Analyze final output layer (maps to features)
        final_layer = layers[-1]  # Last Linear layer
        weights = final_layer.weight.data.cpu().numpy()  # [n_features, hidden]
        bias = final_layer.bias.data.cpu().numpy() if final_layer.bias is not None else None

        # Feature reconstruction importance
        output_importance = np.linalg.norm(weights, axis=1)
        output_importance = output_importance / output_importance.sum()

        feat_names = feature_names.get(source_name, [f'feat_{i}' for i in range(len(output_importance))])
        if len(feat_names) < len(output_importance):
            feat_names = feat_names + [f'feat_{i}' for i in range(len(feat_names), len(output_importance))]

        sorted_indices = np.argsort(output_importance)[::-1]

        print(f"\n  Feature Reconstruction Weight (by decoder output magnitude):")
        print(f"  Top 10 features with highest reconstruction weights:")
        for i, idx in enumerate(sorted_indices[:10]):
            name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
            print(f"    {i+1:2d}. {name:<35} weight: {output_importance[idx]:.4f}")

        # Bias analysis (default predictions)
        if bias is not None:
            print(f"\n  Bias Analysis (default predictions when input is zero):")
            bias_sorted = np.argsort(np.abs(bias))[::-1]
            print(f"  Features with strongest bias:")
            for i, idx in enumerate(bias_sorted[:5]):
                name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
                print(f"    {name:<35} bias: {bias[idx]:.4f}")

        # Layer-wise analysis
        print(f"\n  Layer-wise Weight Statistics:")
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                w = layer.weight.data.cpu().numpy()
                print(f"    Layer {i} (Linear): shape {w.shape}, "
                      f"mean={w.mean():.4f}, std={w.std():.4f}")

        decoder_analysis[source_name] = {
            'output_importance': dict(zip(feat_names[:len(output_importance)], output_importance.tolist())),
            'bias': bias.tolist() if bias is not None else None
        }

    return decoder_analysis


def analyze_unified_projection(model, feature_names):
    """Analyze the unified output projection that combines all sources."""
    print("\n" + "="*80)
    print("5. UNIFIED PROJECTION ANALYSIS")
    print("="*80)

    proj = model.unified_proj
    source_names = model.source_names
    d_embed = model.d_embed

    # First layer: [n_sources * d_embed] -> [d_embed * 2]
    first_linear = proj[0]
    weights = first_linear.weight.data.cpu().numpy()

    print(f"\nUnified projection input: {len(source_names)} sources × {d_embed} dimensions = {len(source_names) * d_embed}")
    print(f"First layer shape: {weights.shape}")

    # Analyze which sources contribute most to the unified representation
    print(f"\n  Source Contribution to Unified Representation:")
    for i, source_name in enumerate(source_names):
        start_idx = i * d_embed
        end_idx = (i + 1) * d_embed
        source_weights = weights[:, start_idx:end_idx]
        contribution = np.linalg.norm(source_weights, 'fro')
        print(f"    {source_name:<20} contribution: {contribution:.4f}")

    # Final layer analysis
    final_linear = proj[-1]
    final_weights = final_linear.weight.data.cpu().numpy()
    print(f"\n  Final projection layer: {final_weights.shape}")
    print(f"  Maps to {final_weights.shape[0]} total features across all sources")


def analyze_cross_source_relationships(model):
    """Synthesize cross-source relationships from all weight analyses."""
    print("\n" + "="*80)
    print("6. SYNTHESIZED CROSS-SOURCE RELATIONSHIPS")
    print("="*80)

    source_names = model.source_names

    # Get source embeddings similarity
    source_emb = model.fusion.source_embeddings.weight.data.cpu().numpy()
    norms = np.linalg.norm(source_emb, axis=1, keepdims=True)
    normalized = source_emb / (norms + 1e-8)
    similarity = normalized @ normalized.T

    # Get attention patterns from first transformer layer
    layer = model.fusion.transformer.layers[0]
    in_proj = layer.self_attn.in_proj_weight.data.cpu().numpy()
    d_embed = in_proj.shape[1]
    Wq = in_proj[:d_embed, :]
    Wk = in_proj[d_embed:2*d_embed, :]

    Q_sources = source_emb @ Wq.T
    K_sources = source_emb @ Wk.T
    attn_scores = Q_sources @ K_sources.T / np.sqrt(d_embed)
    attn_probs = np.exp(attn_scores - attn_scores.max()) / np.exp(attn_scores - attn_scores.max()).sum(axis=1, keepdims=True)

    print("\n  KEY FINDINGS:")
    print("\n  1. Source Clusters (by embedding similarity):")

    # Hierarchical clustering based on similarity
    threshold = 0.5
    clusters = []
    assigned = set()
    for i, s1 in enumerate(source_names):
        if i in assigned:
            continue
        cluster = [s1]
        assigned.add(i)
        for j, s2 in enumerate(source_names):
            if j > i and j not in assigned and similarity[i, j] > threshold:
                cluster.append(s2)
                assigned.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)

    for i, cluster in enumerate(clusters):
        print(f"    Cluster {i+1}: {', '.join(cluster)}")

    print("\n  2. Information Flow Patterns (who informs whom):")

    # Identify dominant information flows
    flows = []
    for i, s1 in enumerate(source_names):
        for j, s2 in enumerate(source_names):
            if i != j and attn_probs[i, j] > 0.15:  # Significant attention
                flows.append((s2, s1, attn_probs[i, j]))  # s2 informs s1

    flows.sort(key=lambda x: x[2], reverse=True)
    for src, dst, weight in flows[:15]:
        print(f"    {src} → {dst}: {weight:.3f}")

    print("\n  3. Domain Groupings (inferred from learned representations):")

    # Group sources by their primary attention targets
    groups = defaultdict(list)
    for i, source in enumerate(source_names):
        primary_target = source_names[np.argmax(attn_probs[i])]
        groups[primary_target].append(source)

    for target, sources in groups.items():
        if len(sources) > 1:
            print(f"    Sources focused on {target}: {sources}")

    print("\n  4. Interpretable Relationships:")

    interpretations = {
        ('viina', 'deepstate'): "Territorial control data mutually reinforcing",
        ('firms', 'ucdp'): "Fire detections correlate with conflict events",
        ('hdx_conflict', 'ucdp'): "Dual conflict event sources cross-validate",
        ('equipment', 'firms'): "Equipment losses coincide with fire activity",
        ('iom', 'hdx_conflict'): "Displacement driven by conflict intensity",
        ('hdx_food', 'iom'): "Food prices affected by displacement patterns",
        ('hdx_rainfall', 'hdx_food'): "Weather impacts food prices",
    }

    for (s1, s2), interpretation in interpretations.items():
        if s1 in source_names and s2 in source_names:
            i1, i2 = source_names.index(s1), source_names.index(s2)
            sim = similarity[i1, i2]
            attn = attn_probs[i1, i2]
            print(f"    {s1} <-> {s2}:")
            print(f"      Similarity: {sim:.3f}, Attention: {attn:.3f}")
            print(f"      Interpretation: {interpretation}")


def compute_gradient_importance(model, feature_names):
    """Compute gradient-based feature importance using synthetic inputs."""
    print("\n" + "="*80)
    print("7. GRADIENT-BASED FEATURE IMPORTANCE")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device

    for source_name in model.source_names:
        config = model.source_configs[source_name]
        n_features = config.n_features

        # Create input requiring gradients
        x = torch.randn(1, n_features, device=device, requires_grad=True)

        # Forward pass through encoder
        embedding = model.encoders[source_name](x)

        # Compute gradient of embedding norm w.r.t. input
        embedding_norm = embedding.norm()
        embedding_norm.backward()

        # Gradient magnitude indicates feature importance
        importance = x.grad.abs().squeeze().cpu().numpy()
        importance = importance / importance.sum()

        feat_names = feature_names.get(source_name, [f'feat_{i}' for i in range(n_features)])

        print(f"\n  {source_name.upper()} - Gradient-based importance:")
        sorted_idx = np.argsort(importance)[::-1]
        for i, idx in enumerate(sorted_idx[:5]):
            name = feat_names[idx] if idx < len(feat_names) else f'feat_{idx}'
            print(f"    {i+1}. {name:<30} {importance[idx]:.4f}")


def main():
    """Main analysis function."""
    print("="*80)
    print("UNIFIED INTERPOLATION MODEL - WEIGHT ANALYSIS")
    print("="*80)

    # Load model
    print("\nLoading model...")
    try:
        model, source_configs = load_unified_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Get feature names
    feature_names = get_feature_names()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Sources: {model.source_names}")

    # Run analyses
    encoder_analysis = analyze_encoder_weights(model, feature_names)
    embedding_analysis = analyze_source_embeddings(model)
    analyze_cross_source_attention(model)
    decoder_analysis = analyze_decoder_weights(model, feature_names)
    analyze_unified_projection(model, feature_names)
    analyze_cross_source_relationships(model)
    compute_gradient_importance(model, feature_names)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
