#!/usr/bin/env python3
"""
ML_OSINT Comprehensive Report Figure Generator

Generates 10 publication-quality figures for the ML_OSINT Tactical State Prediction Model report.

Usage:
    python generate_figures.py

Output:
    All figures saved to ./figures/ directory

Dependencies:
    - matplotlib >= 3.5
    - seaborn >= 0.12
    - numpy >= 1.21
    - pandas >= 1.4
    - networkx >= 2.8

Author: ML_OSINT Team
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# Optional: networkx for pipeline diagram
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Pipeline diagram will use fallback rendering.")

# Set style for all figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================================
# Data for figures (extracted from model analysis)
# =============================================================================

# Model parameters by component
PARAMETER_DATA = {
    'Joint Interpolation (35x)': 4200000,
    'Unified Cross-Source': 1800000,
    'Hierarchical Attention': 1000000,
    'Temporal Prediction': 350000,
    'Tactical State': 140000,
}

# Model comparison data
MODEL_COMPARISON = {
    'Model': ['Hybrid', 'Cumulative', 'Delta-Only'],
    'MSE': [0.489, 1.007, 1.054],
    'Cross-Source Correlation': [0.552, 0.191, 0.089],
    'State Accuracy': [0.784, 0.612, 0.587],
}

# Cross-source attention weights
CROSS_SOURCE_ATTENTION = pd.DataFrame({
    'DeepState': [0.17, 0.29, 0.27, 0.28],
    'Equipment': [0.31, 0.15, 0.19, 0.34],
    'FIRMS': [0.28, 0.21, 0.21, 0.26],
    'UCDP': [0.24, 0.35, 0.33, 0.12],
}, index=['DeepState', 'Equipment', 'FIRMS', 'UCDP'])

# Source importance magnitudes
SOURCE_IMPORTANCE = {
    'DeepState': 8.51,
    'Equipment': 5.23,
    'UCDP': 4.87,
    'FIRMS': 4.12,
    'Sentinel': 2.34,
    'Personnel': 1.89,
}

# Temporal horizon performance
TEMPORAL_PERFORMANCE = pd.DataFrame({
    'h1': [0.181, 0.124, 0.143, 0.098],
    'h3': [0.142, 0.156, 0.138, 0.121],
    'h7': [0.098, 0.134, 0.112, 0.089],
}, index=['FIRMS', 'UCDP', 'DeepState', 'Equipment'])

# Tactical state transitions
STATE_NAMES = ['stable', 'active', 'cont_l', 'cont_h', 'off_pr', 'off_ac', 'major', 'trans']
TRANSITION_MATRIX = np.array([
    [0.72, 0.18, 0.05, 0.02, 0.01, 0.00, 0.00, 0.02],
    [0.15, 0.55, 0.18, 0.06, 0.03, 0.01, 0.00, 0.02],
    [0.08, 0.12, 0.48, 0.22, 0.04, 0.02, 0.01, 0.03],
    [0.02, 0.05, 0.15, 0.45, 0.12, 0.15, 0.03, 0.03],
    [0.01, 0.02, 0.08, 0.15, 0.42, 0.25, 0.04, 0.03],
    [0.01, 0.01, 0.05, 0.18, 0.08, 0.48, 0.15, 0.04],
    [0.00, 0.01, 0.02, 0.12, 0.03, 0.22, 0.52, 0.08],
    [0.05, 0.08, 0.15, 0.18, 0.12, 0.15, 0.07, 0.20],
])

# Calibration data by domain
CALIBRATION_DATA = {
    'Domain': ['UCDP', 'FIRMS', 'DeepState', 'Equipment', 'Sentinel', 'Personnel'],
    'Confidence': [0.82, 0.78, 0.85, 0.91, 0.71, 0.65],
    'Accuracy': [0.79, 0.76, 0.81, 0.73, 0.68, 0.58],
}

# Feature domain breakdown
FEATURE_DOMAINS = {
    'UCDP': 33,
    'FIRMS': 42,
    'Sentinel': 43,
    'DeepState': 45,
    'Equipment': 29,
    'Personnel': 6,
}

# Uncertainty decomposition by state
UNCERTAINTY_DATA = {
    'State': ['stable', 'active', 'cont_l', 'cont_h', 'off_pr', 'off_ac', 'major', 'trans'],
    'Epistemic': [0.05, 0.08, 0.10, 0.14, 0.16, 0.15, 0.18, 0.28],
    'Aleatoric': [0.07, 0.10, 0.12, 0.14, 0.15, 0.14, 0.16, 0.17],
}


# =============================================================================
# Figure 1: Pipeline Architecture Diagram
# =============================================================================

def create_pipeline_architecture():
    """Create 5-stage pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('ML_OSINT 5-Stage Hierarchical Pipeline Architecture', fontsize=16, fontweight='bold', pad=20)

    # Colors for each stage
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    # Stage boxes
    stages = [
        (7, 9, 'Stage 1: Joint Interpolation Models', '35 specialized models\nTemporal interpolation\nGap filling', colors[0]),
        (7, 7, 'Stage 2: Unified Cross-Source', 'Self-supervised learning\nCross-source attention\nMasked reconstruction', colors[1]),
        (7, 5, 'Stage 3: Hierarchical Attention Network', '6 domain encoders\nCross-domain attention\nMulti-task heads', colors[2]),
        (7, 3, 'Stage 4: Temporal Prediction', 'LSTM encoding\nMulti-horizon (T+1,3,7)\nMC Dropout uncertainty', colors[3]),
        (7, 1, 'Stage 5: Tactical State Predictor', '8 tactical states\nHybrid Markov-Neural\nUncertainty quantification', colors[4]),
    ]

    for x, y, title, desc, color in stages:
        # Main box
        box = FancyBboxPatch((x-3.5, y-0.6), 7, 1.2,
                            boxstyle="round,pad=0.05,rounding_size=0.2",
                            facecolor=color, edgecolor='black', alpha=0.8, linewidth=2)
        ax.add_patch(box)

        # Title
        ax.text(x, y+0.2, title, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Description
        ax.text(x, y-0.25, desc, ha='center', va='center', fontsize=8, color='white', alpha=0.9)

    # Arrows between stages
    for i in range(4):
        y_start = 9 - i*2 - 0.6
        y_end = 9 - (i+1)*2 + 0.6
        arrow = FancyArrowPatch((7, y_start), (7, y_end),
                               connectionstyle="arc3,rad=0",
                               arrowstyle="-|>",
                               mutation_scale=20,
                               lw=3, color='#2c3e50')
        ax.add_patch(arrow)

    # Input data sources (left side)
    ax.text(1.5, 9, 'DATA SOURCES', ha='center', va='center', fontsize=10, fontweight='bold')
    sources = ['UCDP (48)', 'FIRMS (42)', 'Sentinel (43)', 'DeepState (55)', 'Equipment (29)', 'Personnel (6)']
    for i, src in enumerate(sources):
        ax.text(1.5, 8.3 - i*0.5, src, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Arrow from sources to Stage 1
    ax.annotate('', xy=(3.5, 9), xytext=(2.8, 8),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    # Output (right side)
    ax.text(12.5, 1, 'OUTPUT', ha='center', va='center', fontsize=10, fontweight='bold')
    outputs = ['Tactical State', 'Probabilities', 'Uncertainty', 'Confidence']
    for i, out in enumerate(outputs):
        ax.text(12.5, 0.5 - i*0.4, out, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))

    # Arrow from Stage 5 to output
    ax.annotate('', xy=(11.2, 1), xytext=(10.5, 1),
               arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    # Parameter counts (annotations)
    params = ['~4.2M params', '~1.8M params', '~1.0M params', '~350K params', '~140K params']
    for i, p in enumerate(params):
        ax.text(11, 9 - i*2, p, ha='left', va='center', fontsize=8, style='italic', color='#666')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pipeline_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: pipeline_architecture.png")


# =============================================================================
# Figure 2: Parameter Distribution
# =============================================================================

def create_parameter_distribution():
    """Create parameter count by model component."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    components = list(PARAMETER_DATA.keys())
    params = [v/1e6 for v in PARAMETER_DATA.values()]
    colors = sns.color_palette("husl", len(components))

    bars = ax1.barh(components, params, color=colors, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Parameters (Millions)', fontsize=12)
    ax1.set_title('Parameter Distribution by Model Component', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 5)

    # Add value labels
    for bar, val in zip(bars, params):
        ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}M', va='center', fontsize=10)

    # Pie chart
    ax2.pie(params, labels=components, autopct='%1.1f%%', colors=colors,
           explode=[0.05]*len(components), shadow=True, startangle=90)
    ax2.set_title('Relative Parameter Share', fontsize=14, fontweight='bold')

    # Total annotation
    total = sum(PARAMETER_DATA.values())
    fig.text(0.5, 0.02, f'Total Parameters: {total/1e6:.2f}M', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: parameter_distribution.png")


# =============================================================================
# Figure 3: Model Comparison
# =============================================================================

def create_model_comparison():
    """Create Hybrid vs Cumulative vs Delta performance comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = MODEL_COMPARISON['Model']
    colors = ['#27ae60', '#3498db', '#e74c3c']

    # MSE comparison
    ax1 = axes[0]
    bars1 = ax1.bar(models, MODEL_COMPARISON['MSE'], color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Reconstruction MSE\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.2)
    for bar, val in zip(bars1, MODEL_COMPARISON['MSE']):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.3f}',
                ha='center', fontsize=11, fontweight='bold')

    # Cross-source correlation
    ax2 = axes[1]
    bars2 = ax2.bar(models, MODEL_COMPARISON['Cross-Source Correlation'], color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.set_title('Cross-Source Correlation\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 0.7)
    for bar, val in zip(bars2, MODEL_COMPARISON['Cross-Source Correlation']):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=11, fontweight='bold')

    # State accuracy
    ax3 = axes[2]
    bars3 = ax3.bar(models, [v*100 for v in MODEL_COMPARISON['State Accuracy']], color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('State Classification Accuracy\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    for bar, val in zip(bars3, MODEL_COMPARISON['State Accuracy']):
        ax3.text(bar.get_x() + bar.get_width()/2, val*100 + 2, f'{val*100:.1f}%',
                ha='center', fontsize=11, fontweight='bold')

    # Add improvement annotations
    fig.text(0.5, -0.02, 'Hybrid model achieves 51.4% lower MSE and 6.2x higher cross-source correlation than alternatives',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: model_comparison.png")


# =============================================================================
# Figure 4: Cross-Source Attention Heatmap
# =============================================================================

def create_cross_source_heatmap():
    """Create cross-source attention weight matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(CROSS_SOURCE_ATTENTION, annot=True, fmt='.2f', cmap='YlOrRd',
               linewidths=2, linecolor='white', cbar_kws={'label': 'Attention Weight'},
               ax=ax, vmin=0, vmax=0.4)

    ax.set_title('Cross-Source Attention Weights\n(Query Source -> Key Sources)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Source', fontsize=12)
    ax.set_ylabel('Query Source', fontsize=12)

    # Highlight diagonal (self-attention)
    for i in range(len(CROSS_SOURCE_ATTENTION)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Add annotation
    fig.text(0.5, -0.02, 'Blue boxes indicate self-attention. UCDP shows lowest self-attention (0.12), benefiting most from cross-source information.',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_source_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: cross_source_heatmap.png")


# =============================================================================
# Figure 5: Source Importance
# =============================================================================

def create_source_importance():
    """Create source importance magnitude ranking."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sources = list(SOURCE_IMPORTANCE.keys())
    importance = list(SOURCE_IMPORTANCE.values())

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    sources = [sources[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]

    # Color gradient based on importance
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sources)))

    bars = ax.barh(sources, importance, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Importance Magnitude', fontsize=12)
    ax.set_title('Source Importance Ranking\n(Higher magnitude = Greater contribution to predictions)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 10)

    # Add vertical reference line at mean
    mean_imp = np.mean(importance)
    ax.axvline(x=mean_imp, color='#2c3e50', linestyle='--', linewidth=2, label=f'Mean: {mean_imp:.2f}')
    ax.legend(loc='lower right')

    # Highlight DeepState
    ax.annotate('DeepState dominates\nwith 8.51 magnitude', xy=(8.51, 5), xytext=(6, 4.5),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'source_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: source_importance.png")


# =============================================================================
# Figure 6: Temporal Horizon Performance
# =============================================================================

def create_temporal_horizon_performance():
    """Create prediction correlation by source and horizon."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Grouped bar chart
    x = np.arange(len(TEMPORAL_PERFORMANCE.index))
    width = 0.25

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars1 = ax1.bar(x - width, TEMPORAL_PERFORMANCE['h1'], width, label='T+1 (h1)', color=colors[0], edgecolor='black')
    bars2 = ax1.bar(x, TEMPORAL_PERFORMANCE['h3'], width, label='T+3 (h3)', color=colors[1], edgecolor='black')
    bars3 = ax1.bar(x + width, TEMPORAL_PERFORMANCE['h7'], width, label='T+7 (h7)', color=colors[2], edgecolor='black')

    ax1.set_ylabel('Correlation', fontsize=12)
    ax1.set_title('Prediction Correlation by Source and Horizon', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(TEMPORAL_PERFORMANCE.index)
    ax1.legend()
    ax1.set_ylim(0, 0.25)

    # Heatmap
    sns.heatmap(TEMPORAL_PERFORMANCE, annot=True, fmt='.3f', cmap='RdYlGn',
               linewidths=1, linecolor='white', cbar_kws={'label': 'Correlation'},
               ax=ax2, vmin=0, vmax=0.2)
    ax2.set_title('Source-Horizon Correlation Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Prediction Horizon', fontsize=12)
    ax2.set_ylabel('Source', fontsize=12)

    # Highlight best performers
    # FIRMS h1 (best short-term)
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', lw=3))
    # UCDP h3 (best medium-term)
    ax2.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor='green', lw=3))

    fig.text(0.5, -0.02, 'FIRMS provides best short-term signal (h1=0.181), UCDP shows stronger medium-term prediction (h3=0.156)',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_horizon_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: temporal_horizon_performance.png")


# =============================================================================
# Figure 7: Tactical State Transitions
# =============================================================================

def create_tactical_state_transitions():
    """Create 8-state transition probability matrix visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap of transitions
    full_names = ['Stable\nDefensive', 'Active\nDefense', 'Contested\nLow', 'Contested\nHigh',
                  'Offensive\nPrep', 'Offensive\nActive', 'Major\nOffensive', 'Transition']

    sns.heatmap(TRANSITION_MATRIX, annot=True, fmt='.2f', cmap='Blues',
               linewidths=1, linecolor='white', cbar_kws={'label': 'Transition Probability'},
               ax=ax1, vmin=0, vmax=0.8,
               xticklabels=STATE_NAMES, yticklabels=STATE_NAMES)
    ax1.set_title('Tactical State Transition Probabilities', fontsize=14, fontweight='bold')
    ax1.set_xlabel('To State', fontsize=12)
    ax1.set_ylabel('From State', fontsize=12)

    # Highlight diagonal (self-transitions)
    for i in range(8):
        ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

    # Bar chart of self-transition probabilities
    self_trans = np.diag(TRANSITION_MATRIX)
    colors = plt.cm.RdYlGn(self_trans)  # Green for stable, red for volatile

    bars = ax2.bar(STATE_NAMES, self_trans, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Self-Transition Probability', fontsize=12)
    ax2.set_title('State Stability (Self-Transition Probability)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='Stability threshold')
    ax2.legend()

    # Add value labels
    for bar, val in zip(bars, self_trans):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', fontsize=9, fontweight='bold')

    # Rotate x labels
    ax2.set_xticklabels(STATE_NAMES, rotation=45, ha='right')

    fig.text(0.5, -0.02, 'Red boxes on heatmap = self-transitions. "Transition" state is most volatile (0.20), "Stable" most persistent (0.72)',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tactical_state_transitions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: tactical_state_transitions.png")


# =============================================================================
# Figure 8: Calibration Curves
# =============================================================================

def create_calibration_curves():
    """Create calibration plots by domain."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    domains = CALIBRATION_DATA['Domain']
    confidence = CALIBRATION_DATA['Confidence']
    accuracy = CALIBRATION_DATA['Accuracy']

    # Calibration scatter plot
    colors = sns.color_palette("husl", len(domains))

    for i, (dom, conf, acc) in enumerate(zip(domains, confidence, accuracy)):
        ax1.scatter(conf, acc, s=200, c=[colors[i]], edgecolors='black', linewidth=2, label=dom, zorder=5)

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Overconfidence region
    ax1.fill_between([0, 1], [0, 1], [0, 0], alpha=0.1, color='red', label='Overconfident region')

    ax1.set_xlabel('Mean Confidence', fontsize=12)
    ax1.set_ylabel('Actual Accuracy', fontsize=12)
    ax1.set_title('Calibration by Domain', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.5, 1)
    ax1.set_ylim(0.5, 1)
    ax1.legend(loc='lower right')
    ax1.set_aspect('equal')

    # Highlight Equipment overconfidence
    equip_idx = domains.index('Equipment')
    ax1.annotate('Equipment\nOverconfident', xy=(confidence[equip_idx], accuracy[equip_idx]),
                xytext=(0.85, 0.6), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Calibration error bar chart
    cal_errors = [abs(c - a) for c, a in zip(confidence, accuracy)]

    bar_colors = ['#27ae60' if e < 0.05 else '#f39c12' if e < 0.1 else '#e74c3c' for e in cal_errors]
    bars = ax2.bar(domains, cal_errors, color=bar_colors, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Calibration Error', fontsize=12)
    ax2.set_title('Calibration Error by Domain\n(|Confidence - Accuracy|)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0.05, color='#27ae60', linestyle='--', linewidth=2, label='Good (<0.05)')
    ax2.axhline(y=0.10, color='#f39c12', linestyle='--', linewidth=2, label='Moderate (<0.10)')
    ax2.legend()
    ax2.set_xticklabels(domains, rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars, cal_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.2f}',
                ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'calibration_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: calibration_curves.png")


# =============================================================================
# Figure 9: Feature Domain Breakdown
# =============================================================================

def create_feature_domain_breakdown():
    """Create feature distribution across domains."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    domains = list(FEATURE_DOMAINS.keys())
    features = list(FEATURE_DOMAINS.values())
    total = sum(features)

    colors = sns.color_palette("husl", len(domains))

    # Horizontal bar chart
    bars = ax1.barh(domains, features, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_title('Feature Count by Domain', fontsize=14, fontweight='bold')

    # Add value labels and percentages
    for bar, val in zip(bars, features):
        pct = val/total*100
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val} ({pct:.1f}%)', va='center', fontsize=10)

    # Pie chart with explode
    explode = [0.02] * len(domains)
    explode[domains.index('DeepState')] = 0.1  # Emphasize DeepState

    wedges, texts, autotexts = ax2.pie(features, labels=domains, autopct='%1.1f%%',
                                        colors=colors, explode=explode, shadow=True,
                                        startangle=90, pctdistance=0.75)

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    ax2.set_title('Feature Distribution', fontsize=14, fontweight='bold')

    # Center hole for total
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax2.add_patch(centre_circle)
    ax2.text(0, 0, f'Total\n{total}', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_domain_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: feature_domain_breakdown.png")


# =============================================================================
# Figure 10: Uncertainty Decomposition
# =============================================================================

def create_uncertainty_decomposition():
    """Create epistemic vs aleatoric uncertainty by state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    states = UNCERTAINTY_DATA['State']
    epistemic = UNCERTAINTY_DATA['Epistemic']
    aleatoric = UNCERTAINTY_DATA['Aleatoric']
    total = [e + a for e, a in zip(epistemic, aleatoric)]

    x = np.arange(len(states))
    width = 0.35

    # Stacked bar chart
    bars1 = ax1.bar(x, epistemic, width, label='Epistemic (Reducible)', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x, aleatoric, width, bottom=epistemic, label='Aleatoric (Irreducible)', color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('Uncertainty', fontsize=12)
    ax1.set_title('Uncertainty Decomposition by Tactical State', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(states, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 0.5)

    # Add total labels
    for i, t in enumerate(total):
        ax1.text(i, t + 0.01, f'{t:.2f}', ha='center', fontsize=9, fontweight='bold')

    # Ratio plot
    ratios = [e/(e+a) for e, a in zip(epistemic, aleatoric)]

    colors = plt.cm.RdYlGn_r(np.array(ratios))
    bars3 = ax2.bar(states, ratios, color=colors, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Epistemic / Total Uncertainty', fontsize=12)
    ax2.set_title('Epistemic Uncertainty Ratio\n(Higher = More reducible with more data)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(states, rotation=45, ha='right')
    ax2.axhline(y=0.5, color='#2c3e50', linestyle='--', linewidth=2, label='Equal split')
    ax2.legend()
    ax2.set_ylim(0, 0.7)

    # Add value labels
    for bar, val in zip(bars3, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
                ha='center', fontsize=9, fontweight='bold')

    # Highlight transition state
    ax2.annotate('Transition state has\nhighest uncertainty', xy=(7, ratios[-1]), xytext=(5.5, 0.6),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    fig.text(0.5, -0.02, 'Epistemic uncertainty (model uncertainty) can be reduced with more training data; Aleatoric (data noise) cannot.',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'uncertainty_decomposition.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: uncertainty_decomposition.png")


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Generate all figures for the comprehensive report."""
    print("=" * 60)
    print("ML_OSINT Comprehensive Report Figure Generator")
    print("=" * 60)
    print(f"\nOutput directory: {FIGURES_DIR}\n")

    # Generate all figures
    print("Generating figures...\n")

    create_pipeline_architecture()
    create_parameter_distribution()
    create_model_comparison()
    create_cross_source_heatmap()
    create_source_importance()
    create_temporal_horizon_performance()
    create_tactical_state_transitions()
    create_calibration_curves()
    create_feature_domain_breakdown()
    create_uncertainty_decomposition()

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"Total figures: 10")
    print(f"Output location: {FIGURES_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            filepath = os.path.join(FIGURES_DIR, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {f} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
