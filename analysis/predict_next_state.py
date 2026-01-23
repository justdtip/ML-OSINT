#!/usr/bin/env python3
"""
Conflict State Prediction using Hierarchical Attention Network

Uses the trained model to predict the next month's conflict state based on
the most recent available data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hierarchical_attention_network import (
    HierarchicalAttentionNetwork,
    DOMAIN_CONFIGS,
    TOTAL_FEATURES
)
from conflict_data_loader import (
    RealConflictDataset,
    load_ucdp_data,
    load_firms_data,
    load_sentinel_data,
    load_deepstate_data,
    load_equipment_data,
    load_personnel_data,
    extract_domain_features,
    normalize_features
)

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
    FIGURES_DIR, REPORTS_DIR, ANALYSIS_FIGURES_DIR,
)


def load_model(device='cpu'):
    """Load the trained model."""
    model_path = MODEL_DIR / 'han_best.pt'

    model = HierarchicalAttentionNetwork(
        domain_configs=DOMAIN_CONFIGS,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_temporal_layers=1,
        dropout=0.35
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def load_current_state(seq_len=4):
    """
    Load the most recent data and prepare it for prediction.
    Returns the last seq_len months of data.
    """
    print("Loading all data sources...")

    # Load raw data
    ucdp = load_ucdp_data()
    firms = load_firms_data()
    sentinel = load_sentinel_data()
    deepstate = load_deepstate_data()
    equipment = load_equipment_data()
    personnel = load_personnel_data()

    # Extract and align features
    features, date_range = extract_domain_features(
        ucdp, firms, sentinel, deepstate, equipment, personnel
    )

    if not features:
        raise ValueError("No features extracted")

    # Normalize
    features_norm, norm_stats = normalize_features(features, DOMAIN_CONFIGS)

    # Get the last seq_len months
    n_months = len(date_range)
    start_idx = n_months - seq_len

    print(f"\nData range: {date_range[0].strftime('%Y-%m')} to {date_range[-1].strftime('%Y-%m')}")
    print(f"Using last {seq_len} months for prediction: {date_range[start_idx].strftime('%Y-%m')} to {date_range[-1].strftime('%Y-%m')}")

    # Prepare tensors
    input_features = {}
    input_masks = {}

    for domain_name, data in features_norm.items():
        # Get last seq_len months
        domain_data = data[start_idx:]
        input_features[domain_name] = torch.tensor(domain_data, dtype=torch.float32).unsqueeze(0)
        # Create mask (1 for valid data)
        input_masks[domain_name] = torch.ones_like(input_features[domain_name])

    # Also keep raw features for interpretation
    raw_features = {name: data[start_idx:] for name, data in features.items()}

    return input_features, input_masks, raw_features, norm_stats, date_range, start_idx


def denormalize_predictions(predictions, norm_stats, domain_configs):
    """Convert normalized predictions back to original scale."""
    denorm = {}
    idx = 0

    for domain_name, config in domain_configs.items():
        n_feat = config.num_features
        domain_pred = predictions[idx:idx + n_feat]

        if domain_name in norm_stats:
            stats = norm_stats[domain_name]
            if stats['type'] == 'log':
                # Reverse: denorm = pred * std + mean, then exp - 1
                denorm_log = domain_pred * stats['std'] + stats['mean']
                domain_pred = np.expm1(denorm_log)
            elif stats['type'] == 'standard':
                domain_pred = domain_pred * stats['std'] + stats['mean']
            elif stats['type'] == 'minmax':
                domain_pred = domain_pred * (stats['max'] - stats['min']) + stats['min']

        denorm[domain_name] = domain_pred
        idx += n_feat

    return denorm


def interpret_regime(regime_probs):
    """Interpret the regime classification."""
    regime_names = ['Low Intensity', 'Medium Intensity', 'High Intensity', 'Major Offensive']
    regime_descriptions = [
        'Reduced combat activity, potential ceasefire or operational pause',
        'Sustained but limited engagements, positional warfare',
        'Intense combat across multiple fronts, significant casualties',
        'Large-scale offensive operations, rapid territorial changes'
    ]

    pred_regime = np.argmax(regime_probs)
    confidence = regime_probs[pred_regime]

    return {
        'predicted_regime': regime_names[pred_regime],
        'confidence': confidence,
        'description': regime_descriptions[pred_regime],
        'all_probs': dict(zip(regime_names, regime_probs))
    }


def interpret_anomaly(anomaly_score):
    """Interpret the anomaly detection score."""
    if anomaly_score < 0.3:
        return "Normal - Conflict dynamics within expected patterns"
    elif anomaly_score < 0.5:
        return "Slightly Unusual - Some deviation from typical patterns"
    elif anomaly_score < 0.7:
        return "Anomalous - Significant departure from normal dynamics"
    else:
        return "Highly Anomalous - Major disruption or unprecedented situation"


def format_predictions(denorm_predictions, current_values, domain_configs):
    """Format predictions with comparisons to current values."""
    results = {}

    for domain_name, config in domain_configs.items():
        if domain_name not in denorm_predictions:
            continue

        pred = denorm_predictions[domain_name]
        curr = current_values.get(domain_name, np.zeros_like(pred))

        # Get the last month's values for comparison
        if len(curr.shape) > 1:
            curr = curr[-1]  # Last month

        domain_results = []
        for i, feat_name in enumerate(config.feature_names):
            if i < len(pred) and i < len(curr):
                pred_val = pred[i]
                curr_val = curr[i]

                # Calculate change
                if abs(curr_val) > 1e-6:
                    pct_change = ((pred_val - curr_val) / abs(curr_val)) * 100
                else:
                    pct_change = 0 if abs(pred_val) < 1e-6 else float('inf')

                domain_results.append({
                    'feature': feat_name,
                    'current': curr_val,
                    'predicted': pred_val,
                    'change': pred_val - curr_val,
                    'pct_change': pct_change
                })

        results[domain_name] = domain_results

    return results


def print_prediction_report(predictions, regime_info, anomaly_score, date_range, formatted_results):
    """Print a comprehensive prediction report."""
    next_month = date_range[-1] + pd.DateOffset(months=1)

    print("\n" + "=" * 80)
    print(f"CONFLICT STATE PREDICTION FOR {next_month.strftime('%B %Y').upper()}")
    print("=" * 80)

    print(f"\nBased on data from: {date_range[-4].strftime('%B %Y')} to {date_range[-1].strftime('%B %Y')}")

    # Regime prediction
    print("\n" + "-" * 80)
    print("PREDICTED CONFLICT REGIME")
    print("-" * 80)
    print(f"\n  Prediction: {regime_info['predicted_regime']}")
    print(f"  Confidence: {regime_info['confidence']:.1%}")
    print(f"\n  Interpretation: {regime_info['description']}")

    print("\n  Probability Distribution:")
    for regime, prob in regime_info['all_probs'].items():
        bar = "█" * int(prob * 40)
        print(f"    {regime:<20} {prob:>6.1%} {bar}")

    # Anomaly score
    print("\n" + "-" * 80)
    print("ANOMALY DETECTION")
    print("-" * 80)
    print(f"\n  Anomaly Score: {anomaly_score:.3f}")
    print(f"  Assessment: {interpret_anomaly(anomaly_score)}")

    # Key predictions by domain
    print("\n" + "-" * 80)
    print("KEY PREDICTIONS BY DOMAIN")
    print("-" * 80)

    domain_summaries = {
        'ucdp': ('UCDP Conflict Events', ['deaths_best', 'deaths_side_a', 'deaths_side_b', 'events_state_based']),
        'firms': ('FIRMS Fire Detections', ['fires_total', 'frp_total', 'day_night_ratio']),
        'equipment': ('Equipment Losses', ['tank_total', 'aircraft_total', 'heli_total', 'drones_total']),
        'personnel': ('Personnel Losses', ['personnel_monthly', 'personnel_daily_avg']),
        'deepstate': ('DeepState Front Line', ['poly_occupied_count', 'poly_liberated_count', 'arrows_total']),
        'sentinel': ('Sentinel Satellite', ['s2_count', 's5p_no2_mean', 's3_frp_count'])
    }

    for domain_name, (display_name, key_features) in domain_summaries.items():
        if domain_name not in formatted_results:
            continue

        print(f"\n  {display_name}:")

        results = formatted_results[domain_name]
        results_dict = {r['feature']: r for r in results}

        for feat in key_features:
            if feat in results_dict:
                r = results_dict[feat]
                direction = "↑" if r['change'] > 0 else ("↓" if r['change'] < 0 else "→")

                # Format based on magnitude
                if abs(r['current']) > 1000:
                    curr_str = f"{r['current']:,.0f}"
                    pred_str = f"{r['predicted']:,.0f}"
                elif abs(r['current']) > 1:
                    curr_str = f"{r['current']:.1f}"
                    pred_str = f"{r['predicted']:.1f}"
                else:
                    curr_str = f"{r['current']:.3f}"
                    pred_str = f"{r['predicted']:.3f}"

                pct_str = f"{r['pct_change']:+.1f}%" if abs(r['pct_change']) < 1000 else "N/A"

                feat_display = feat.replace('_', ' ').title()
                print(f"    {feat_display:<25} {curr_str:>12} → {pred_str:>12} {direction} ({pct_str})")

    # Significant changes
    print("\n" + "-" * 80)
    print("MOST SIGNIFICANT PREDICTED CHANGES")
    print("-" * 80)

    all_changes = []
    for domain_name, results in formatted_results.items():
        for r in results:
            if abs(r['pct_change']) < 1000 and abs(r['current']) > 0.1:  # Filter noise
                all_changes.append({
                    'domain': domain_name,
                    'feature': r['feature'],
                    'current': r['current'],
                    'predicted': r['predicted'],
                    'pct_change': r['pct_change']
                })

    # Top increases
    increases = sorted([c for c in all_changes if c['pct_change'] > 0],
                       key=lambda x: x['pct_change'], reverse=True)[:5]

    if increases:
        print("\n  Largest Predicted Increases:")
        for c in increases:
            feat_display = c['feature'].replace('_', ' ').title()
            print(f"    ↑ {c['domain'].upper()}: {feat_display:<30} +{c['pct_change']:.1f}%")

    # Top decreases
    decreases = sorted([c for c in all_changes if c['pct_change'] < 0],
                       key=lambda x: x['pct_change'])[:5]

    if decreases:
        print("\n  Largest Predicted Decreases:")
        for c in decreases:
            feat_display = c['feature'].replace('_', ' ').title()
            print(f"    ↓ {c['domain'].upper()}: {feat_display:<30} {c['pct_change']:.1f}%")

    # Summary interpretation
    print("\n" + "-" * 80)
    print("SUMMARY INTERPRETATION")
    print("-" * 80)

    # Generate summary based on predictions
    regime = regime_info['predicted_regime']
    conf = regime_info['confidence']

    summary_lines = []

    # Regime-based summary
    if 'Major Offensive' in regime:
        summary_lines.append("• Model predicts escalation to major offensive operations")
    elif 'High' in regime:
        summary_lines.append("• Model predicts continued high-intensity combat")
    elif 'Medium' in regime:
        summary_lines.append("• Model predicts sustained medium-intensity operations")
    else:
        summary_lines.append("• Model predicts reduced combat intensity")

    # Equipment-based insights
    if 'equipment' in formatted_results:
        equip_results = {r['feature']: r for r in formatted_results['equipment']}
        if 'tank_total' in equip_results:
            tank_change = equip_results['tank_total']['pct_change']
            if tank_change > 10:
                summary_lines.append(f"• Tank losses expected to increase significantly (+{tank_change:.0f}%)")
            elif tank_change < -10:
                summary_lines.append(f"• Tank losses expected to decrease ({tank_change:.0f}%)")

    # Personnel insights
    if 'personnel' in formatted_results:
        pers_results = {r['feature']: r for r in formatted_results['personnel']}
        if 'personnel_monthly' in pers_results:
            pers_change = pers_results['personnel_monthly']['pct_change']
            if pers_change > 15:
                summary_lines.append(f"• Monthly personnel losses predicted to rise (+{pers_change:.0f}%)")
            elif pers_change < -15:
                summary_lines.append(f"• Monthly personnel losses predicted to decline ({pers_change:.0f}%)")

    # Fire activity insights
    if 'firms' in formatted_results:
        firms_results = {r['feature']: r for r in formatted_results['firms']}
        if 'fires_total' in firms_results:
            fire_change = firms_results['fires_total']['pct_change']
            if fire_change > 20:
                summary_lines.append(f"• Fire activity (combat indicator) expected to increase (+{fire_change:.0f}%)")

    # Anomaly insight
    if anomaly_score > 0.5:
        summary_lines.append(f"• ⚠️  Anomalous conditions detected - predictions may be less reliable")

    # Confidence caveat
    if conf < 0.4:
        summary_lines.append("• Note: Regime prediction confidence is low - situation may be transitional")

    print()
    for line in summary_lines:
        print(f"  {line}")

    print("\n" + "=" * 80)
    print("DISCLAIMER: These predictions are based on a machine learning model trained on")
    print("limited historical data. They should not be used for operational decisions.")
    print("=" * 80)


def main():
    print("=" * 80)
    print("HIERARCHICAL ATTENTION NETWORK - CONFLICT STATE PREDICTION")
    print("=" * 80)

    # Load model
    print("\nLoading trained model...")
    model, checkpoint = load_model()
    print(f"  Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    # Load current state
    print("\nLoading current battlefield state...")
    seq_len = 4
    input_features, input_masks, raw_features, norm_stats, date_range, start_idx = load_current_state(seq_len)

    # Make prediction
    print("\nGenerating predictions...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_features, input_masks, return_attention=True)

    # Extract predictions
    forecast = outputs['forecast'][0, -1, :].numpy()  # Last timestep prediction
    regime_logits = outputs['regime_logits'][0, -1, :].numpy()
    anomaly_score = outputs['anomaly_score'][0, -1, 0].item()

    # Convert regime logits to probabilities
    regime_probs = np.exp(regime_logits) / np.exp(regime_logits).sum()

    # Denormalize predictions
    denorm_predictions = denormalize_predictions(forecast, norm_stats, DOMAIN_CONFIGS)

    # Interpret regime
    regime_info = interpret_regime(regime_probs)

    # Format predictions with comparisons
    formatted_results = format_predictions(denorm_predictions, raw_features, DOMAIN_CONFIGS)

    # Print comprehensive report
    print_prediction_report(
        denorm_predictions,
        regime_info,
        anomaly_score,
        date_range,
        formatted_results
    )

    # Return data for further analysis if needed
    return {
        'predictions': denorm_predictions,
        'regime_info': regime_info,
        'anomaly_score': anomaly_score,
        'formatted_results': formatted_results,
        'date_range': date_range
    }


if __name__ == "__main__":
    results = main()
