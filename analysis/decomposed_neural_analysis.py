#!/usr/bin/env python3
"""
Decomposed Neural Network Analysis for Ukraine Conflict OSINT Data

This script decomposes the aggregate data into granular components:
1. UCDP: By region (oblast), violence type, party (side_a vs side_b casualties)
2. Equipment: By category (tanks, aircraft, artillery, etc.)
3. FIRMS: By day/night, confidence level, fire intensity
4. DeepState: By icon type (attack direction indicators)

Then uses attention-based neural networks to find which decomposed
features predict outcomes and how they interact.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

# Centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR,
)

BASE_DIR = PROJECT_ROOT

# =============================================================================
# DATA DECOMPOSITION
# =============================================================================

def decompose_ucdp():
    """
    Decompose UCDP events by:
    - Oblast (adm_1)
    - Violence type
    - Party casualties (deaths_a, deaths_b, deaths_civilians)
    - Event clarity/precision
    """
    print("  Loading UCDP data...")
    df = pd.read_csv(DATA_DIR / "ucdp/ged_events.csv", low_memory=False)
    df['date_start'] = pd.to_datetime(df['date_start'], format='mixed')
    df = df[df['date_start'] >= '2022-02-24']
    df['month'] = df['date_start'].dt.to_period('M').dt.to_timestamp()

    monthly = defaultdict(lambda: defaultdict(float))

    # Key oblasts (frontline regions)
    key_oblasts = ['Donetsk', 'Luhansk', 'Kherson', 'Zaporizhzhia', 'Kharkiv', 'Kiev']

    for _, row in df.iterrows():
        month = row['month']

        # By oblast
        oblast = row['adm_1'] if pd.notna(row['adm_1']) else 'Unknown'
        for key_obl in key_oblasts:
            if key_obl.lower() in oblast.lower():
                monthly[month][f'deaths_{key_obl.lower()}'] += row['best_est']
                break
        else:
            monthly[month]['deaths_other'] += row['best_est']

        # By party
        monthly[month]['deaths_side_a'] += row['deaths_a'] if pd.notna(row['deaths_a']) else 0
        monthly[month]['deaths_side_b'] += row['deaths_b'] if pd.notna(row['deaths_b']) else 0
        monthly[month]['deaths_civilian'] += row['deaths_civilians'] if pd.notna(row['deaths_civilians']) else 0
        monthly[month]['deaths_unknown'] += row['deaths_unknown'] if pd.notna(row['deaths_unknown']) else 0

        # Event count by precision
        precision = row['where_prec'] if pd.notna(row['where_prec']) else 0
        monthly[month][f'events_precision_{int(precision)}'] += 1

    result = pd.DataFrame.from_dict(monthly, orient='index')
    result.index.name = 'date'
    result = result.reset_index()
    result = result.sort_values('date')
    result = result.fillna(0)

    print(f"    Created {len(result.columns)-1} UCDP features across {len(result)} months")
    return result


def decompose_equipment():
    """
    Decompose equipment losses by category and compute rates of change.
    """
    print("  Loading equipment data...")
    with open(DATA_DIR / "war-losses-data/2022-Ukraine-Russia-War-Dataset/data/russia_losses_equipment.json") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    # Equipment categories
    categories = ['aircraft', 'helicopter', 'tank', 'APC', 'field artillery',
                  'MRL', 'drone', 'naval ship', 'anti-aircraft warfare']

    # Get monthly values (use last value of each month as cumulative)
    monthly = df.groupby('month').last().reset_index()

    # Convert cumulative to monthly increments
    result = pd.DataFrame({'date': monthly['month']})

    for cat in categories:
        if cat in monthly.columns:
            col_name = cat.replace(' ', '_').lower()
            # Monthly increment
            result[f'equip_{col_name}'] = monthly[cat].diff().fillna(monthly[cat])
            # Keep cumulative too
            result[f'equip_{col_name}_cum'] = monthly[cat]

    # Compute ratios
    if 'equip_tank' in result.columns and 'equip_apc' in result.columns:
        result['equip_armor_ratio'] = result['equip_tank'] / (result['equip_apc'] + 1)

    if 'equip_aircraft' in result.columns and 'equip_helicopter' in result.columns:
        result['equip_air_total'] = result['equip_aircraft'] + result['equip_helicopter']

    print(f"    Created {len(result.columns)-1} equipment features across {len(result)} months")
    return result


def decompose_firms():
    """
    Decompose FIRMS fires by:
    - Day vs night
    - Confidence level
    - Fire radiative power intensity bands
    """
    print("  Loading FIRMS data...")
    archive = pd.read_csv(DATA_DIR / "firms/DL_FIRE_SV-C2_706038/fire_archive_SV-C2_706038.csv")
    nrt = pd.read_csv(DATA_DIR / "firms/DL_FIRE_SV-C2_706038/fire_nrt_SV-C2_706038.csv")
    df = pd.concat([archive, nrt], ignore_index=True)

    df['date'] = pd.to_datetime(df['acq_date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    monthly = defaultdict(lambda: defaultdict(float))

    for _, row in df.iterrows():
        month = row['month']

        # Day vs night
        if row['daynight'] == 'D':
            monthly[month]['fires_day'] += 1
            monthly[month]['frp_day'] += row['frp'] if pd.notna(row['frp']) else 0
        else:
            monthly[month]['fires_night'] += 1
            monthly[month]['frp_night'] += row['frp'] if pd.notna(row['frp']) else 0

        # By confidence
        conf = row['confidence']
        if conf in ['h', 'high']:
            monthly[month]['fires_high_conf'] += 1
        elif conf in ['n', 'nominal']:
            monthly[month]['fires_nominal_conf'] += 1
        else:
            monthly[month]['fires_low_conf'] += 1

        # FRP intensity bands
        frp = row['frp'] if pd.notna(row['frp']) else 0
        if frp < 10:
            monthly[month]['fires_low_frp'] += 1
        elif frp < 50:
            monthly[month]['fires_med_frp'] += 1
        else:
            monthly[month]['fires_high_frp'] += 1

    result = pd.DataFrame.from_dict(monthly, orient='index')
    result.index.name = 'date'
    result = result.reset_index()
    result = result.sort_values('date')
    result = result.fillna(0)

    # Compute ratios
    result['fires_day_ratio'] = result['fires_day'] / (result['fires_day'] + result['fires_night'] + 1)
    result['fires_high_conf_ratio'] = result['fires_high_conf'] / (result['fires_high_conf'] + result['fires_nominal_conf'] + result['fires_low_conf'] + 1)
    result['frp_per_fire'] = (result['frp_day'] + result['frp_night']) / (result['fires_day'] + result['fires_night'] + 1)

    print(f"    Created {len(result.columns)-1} FIRMS features across {len(result)} months")
    return result


def load_base_data():
    """Load the merged sentinel-osint data as base."""
    df = pd.read_csv(ANALYSIS_DIR / "sentinel_osint_merged.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df


def merge_decomposed_data():
    """Merge all decomposed datasets."""
    print("\nDecomposing data sources...")

    base = load_base_data()
    ucdp = decompose_ucdp()
    equip = decompose_equipment()
    firms = decompose_firms()

    # Merge all on date
    merged = base.copy()
    merged = merged.merge(ucdp, on='date', how='left')
    merged = merged.merge(equip, on='date', how='left')
    merged = merged.merge(firms, on='date', how='left')

    merged = merged.fillna(0)

    print(f"\nTotal features after decomposition: {len(merged.columns)}")
    return merged


# =============================================================================
# ATTENTION-BASED NEURAL NETWORK
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Self-attention to find feature interactions."""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, features) -> reshape to (batch, seq=features, embed=1)
        # But we need proper embedding first
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + attn_out), attn_weights


class DecomposedAttentionNetwork(nn.Module):
    """
    Network that uses attention to discover which decomposed features
    are most important for predicting outcomes.

    Structure:
    1. Feature group embeddings (UCDP, Equipment, FIRMS, Satellite)
    2. Cross-group attention
    3. Prediction heads for multiple targets
    """
    def __init__(self, feature_groups, embed_dim=32, num_heads=4):
        super().__init__()

        self.feature_groups = feature_groups
        self.embed_dim = embed_dim

        # Embedding layer for each feature group
        self.group_embeddings = nn.ModuleDict()
        for group_name, num_features in feature_groups.items():
            self.group_embeddings[group_name] = nn.Sequential(
                nn.Linear(num_features, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # Cross-group attention
        num_groups = len(feature_groups)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(embed_dim)

        # Feature importance (learnable per group)
        self.group_importance = nn.Parameter(torch.ones(num_groups))

        # Prediction heads
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * num_groups, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predict: deaths, losses, fires
        )

    def forward(self, x_groups):
        """
        x_groups: dict of {group_name: tensor of shape (batch, num_features)}
        """
        # Embed each group
        embeddings = []
        for group_name in self.feature_groups.keys():
            emb = self.group_embeddings[group_name](x_groups[group_name])
            embeddings.append(emb)

        # Stack for attention: (batch, num_groups, embed_dim)
        stacked = torch.stack(embeddings, dim=1)

        # Cross-group attention
        attended, attn_weights = self.cross_attention(stacked, stacked, stacked)
        attended = self.attention_norm(stacked + attended)

        # Weight by learned importance
        importance = torch.softmax(self.group_importance, dim=0)
        weighted = attended * importance.unsqueeze(0).unsqueeze(-1)

        # Flatten and predict
        flat = weighted.reshape(weighted.size(0), -1)
        predictions = self.predictor(flat)

        return predictions, attn_weights, importance

    def get_group_importance(self):
        return torch.softmax(self.group_importance, dim=0).detach().numpy()


class FeatureInteractionDiscovery(nn.Module):
    """
    Discovers which specific features interact across groups.
    Uses bilinear layers to model pairwise interactions.
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        self.input_dim = input_dim

        # Individual feature embeddings
        self.feature_embed = nn.Linear(input_dim, hidden_dim)

        # Pairwise interaction modeling
        self.interaction = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # Feature-wise attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Attention weights show feature importance
        attn = self.attention(x)
        weighted = x * attn

        # Embed
        emb = self.feature_embed(weighted)

        # Self-interaction
        interaction = self.interaction(emb, emb)

        # Combine
        combined = torch.cat([interaction, weighted], dim=-1)
        pred = self.predictor(combined)

        return pred, attn


# =============================================================================
# TRAINING AND ANALYSIS
# =============================================================================

def identify_outliers(df, columns, threshold=1.5):
    """Identify outliers using IQR method."""
    outlier_mask = pd.Series(False, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = outlier_mask | (df[col] > Q3 + threshold * IQR)
    return outlier_mask


def prepare_feature_groups(df, outlier_mask):
    """Organize features into logical groups."""

    # Remove outliers
    df_clean = df[~outlier_mask].copy()

    groups = {
        'satellite': ['s1_radar', 's2_optical', 's3_fire', 's5p_co', 's5p_no2',
                      's2_avg_cloud', 's2_cloud_free'],
        'ucdp_regional': [c for c in df_clean.columns if c.startswith('deaths_') and
                          any(x in c for x in ['donetsk', 'luhansk', 'kherson', 'zaporizhzhia', 'kharkiv', 'kiev', 'other'])],
        'ucdp_party': ['deaths_side_a', 'deaths_side_b', 'deaths_civilian', 'deaths_unknown'],
        'equipment': [c for c in df_clean.columns if c.startswith('equip_') and not c.endswith('_cum')],
        'fires': [c for c in df_clean.columns if c.startswith('fires_') or c.startswith('frp_')]
    }

    # Filter to columns that exist
    for group_name in groups:
        groups[group_name] = [c for c in groups[group_name] if c in df_clean.columns]

    return groups, df_clean


def train_attention_network(df, feature_groups, epochs=500):
    """Train the attention-based network."""
    print("\nTraining attention-based network...")

    # Prepare data
    scaler = StandardScaler()

    # Get all features
    all_features = []
    for group_features in feature_groups.values():
        all_features.extend(group_features)

    X = df[all_features].values
    X_scaled = scaler.fit_transform(X)

    # Targets
    targets = ['ucdp_deaths', 'monthly_loss', 'firms_fires']
    y = df[targets].values
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Split into groups
    X_groups = {}
    idx = 0
    for group_name, group_features in feature_groups.items():
        n_feat = len(group_features)
        X_groups[group_name] = torch.FloatTensor(X_scaled[:, idx:idx+n_feat])
        idx += n_feat

    y_tensor = torch.FloatTensor(y_scaled)

    # Model
    group_sizes = {name: len(feats) for name, feats in feature_groups.items()}
    model = DecomposedAttentionNetwork(group_sizes, embed_dim=16, num_heads=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, attn_weights, importance = model(X_groups)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return model, losses, scaler


def train_feature_discovery(df, feature_names, target, epochs=500):
    """Train feature interaction discovery network."""

    scaler = StandardScaler()
    X = df[feature_names].values
    X_scaled = scaler.fit_transform(X)

    y = df[target].values
    y_scaled = (y - y.mean()) / (y.std() + 1e-8)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled.reshape(-1, 1))

    model = FeatureInteractionDiscovery(len(feature_names))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, attn = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    # Get average attention weights
    model.eval()
    with torch.no_grad():
        _, attn = model(X_tensor)
        avg_attn = attn.mean(dim=0).numpy()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'attention': avg_attn
    }).sort_values('attention', ascending=False)

    return importance_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_decomposed_analysis(df, model, feature_groups, losses):
    """Visualize the decomposed analysis results."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Decomposed Neural Network Analysis\n'
                 'Attention-Based Discovery of Cross-Domain Patterns',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Training loss
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Convergence')
    ax1.set_yscale('log')

    # 2. Group importance
    ax2 = fig.add_subplot(2, 3, 2)
    importance = model.get_group_importance()
    group_names = list(feature_groups.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(group_names)))
    bars = ax2.bar(group_names, importance, color=colors)
    ax2.set_ylabel('Learned Importance')
    ax2.set_title('Feature Group Importance\n(Which data sources matter most)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, imp in zip(bars, importance):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{imp:.2f}', ha='center', fontsize=10)

    # 3. Regional death distribution over time
    ax3 = fig.add_subplot(2, 3, 3)
    regional_cols = [c for c in df.columns if c.startswith('deaths_') and
                     any(x in c for x in ['donetsk', 'luhansk', 'kherson', 'zaporizhzhia', 'kharkiv'])]
    if regional_cols:
        for col in regional_cols:
            region = col.replace('deaths_', '').title()
            ax3.plot(df['date'], df[col], label=region, alpha=0.7)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Deaths')
        ax3.set_title('Deaths by Oblast (Decomposed)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.tick_params(axis='x', rotation=45)

    # 4. Equipment breakdown
    ax4 = fig.add_subplot(2, 3, 4)
    equip_cols = ['equip_tank', 'equip_apc', 'equip_aircraft', 'equip_artillery']
    equip_cols = [c for c in equip_cols if c in df.columns]
    if equip_cols:
        for col in equip_cols:
            label = col.replace('equip_', '').replace('field_', '').title()
            ax4.plot(df['date'], df[col], label=label, alpha=0.7)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Monthly Losses')
        ax4.set_title('Equipment Losses by Category')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.tick_params(axis='x', rotation=45)

    # 5. Fire patterns (day vs night)
    ax5 = fig.add_subplot(2, 3, 5)
    if 'fires_day' in df.columns and 'fires_night' in df.columns:
        ax5.fill_between(df['date'], 0, df['fires_day'], alpha=0.5, label='Day fires')
        ax5.fill_between(df['date'], 0, df['fires_night'], alpha=0.5, label='Night fires')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Fire Count')
        ax5.set_title('Day vs Night Fire Detections')
        ax5.legend()
        ax5.tick_params(axis='x', rotation=45)

    # 6. Cross-attention heatmap
    ax6 = fig.add_subplot(2, 3, 6)

    # Get attention weights from model
    X_groups = {}
    all_features = []
    for group_features in feature_groups.values():
        all_features.extend(group_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[all_features].fillna(0).values)

    idx = 0
    for group_name, group_features in feature_groups.items():
        n_feat = len(group_features)
        X_groups[group_name] = torch.FloatTensor(X_scaled[:, idx:idx+n_feat])
        idx += n_feat

    model.eval()
    with torch.no_grad():
        _, attn_weights, _ = model(X_groups)
        avg_attn = attn_weights.mean(dim=0).numpy()

    im = ax6.imshow(avg_attn, cmap='Blues', aspect='auto')
    ax6.set_xticks(range(len(group_names)))
    ax6.set_xticklabels(group_names, rotation=45, ha='right')
    ax6.set_yticks(range(len(group_names)))
    ax6.set_yticklabels(group_names)
    ax6.set_title('Cross-Group Attention\n(Which groups attend to which)')
    plt.colorbar(im, ax=ax6, label='Attention Weight')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / '12_decomposed_neural_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: 12_decomposed_neural_analysis.png")


def plot_feature_importance(importance_results, top_n=15):
    """Plot discovered feature importance for each target."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Decomposed Feature Importance (Attention-Based Discovery)\n'
                 'Which specific features predict each outcome?',
                 fontsize=14, fontweight='bold')

    for ax, (target, imp_df) in zip(axes, importance_results.items()):
        imp_top = imp_df.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(imp_top)))
        ax.barh(range(len(imp_top)), imp_top['attention'].values, color=colors)
        ax.set_yticks(range(len(imp_top)))
        ax.set_yticklabels(imp_top['feature'].values)
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Predicting: {target}')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / '13_decomposed_feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: 13_decomposed_feature_importance.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("DECOMPOSED NEURAL NETWORK ANALYSIS")
    print("Finding patterns in granular feature breakdowns")
    print("=" * 80)

    # Merge all decomposed data
    df = merge_decomposed_data()

    # Identify outliers
    outlier_mask = identify_outliers(df, ['ucdp_deaths', 'monthly_loss'])
    outlier_months = df.loc[outlier_mask, 'date'].dt.strftime('%Y-%m').tolist()
    print(f"\nExcluding {len(outlier_months)} outlier months: {outlier_months}")

    # Prepare feature groups
    feature_groups, df_clean = prepare_feature_groups(df, outlier_mask)

    print("\nFeature groups:")
    for group_name, features in feature_groups.items():
        print(f"  {group_name}: {len(features)} features")
        if len(features) <= 10:
            print(f"    {features}")

    # Train attention network
    model, losses, scaler = train_attention_network(df_clean, feature_groups, epochs=500)

    # Get group importance
    importance = model.get_group_importance()
    print("\n" + "="*60)
    print("LEARNED GROUP IMPORTANCE:")
    print("="*60)
    for name, imp in zip(feature_groups.keys(), importance):
        print(f"  {name}: {imp:.3f}")

    # Feature-level importance for each target
    print("\n" + "="*60)
    print("FEATURE-LEVEL IMPORTANCE ANALYSIS:")
    print("="*60)

    all_features = []
    for feats in feature_groups.values():
        all_features.extend(feats)

    importance_results = {}
    for target in ['ucdp_deaths', 'monthly_loss', 'firms_fires']:
        print(f"\nAnalyzing predictors for {target}...")
        imp_df = train_feature_discovery(df_clean, all_features, target, epochs=300)
        importance_results[target] = imp_df

        print(f"  Top 10 predictors:")
        for _, row in imp_df.head(10).iterrows():
            print(f"    {row['feature']}: {row['attention']:.4f}")

    # Visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS...")
    print("="*60)
    plot_decomposed_analysis(df_clean, model, feature_groups, losses)
    plot_feature_importance(importance_results)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: KEY FINDINGS FROM DECOMPOSED ANALYSIS")
    print("="*80)

    print(f"""
DATA DECOMPOSITION:
  - UCDP events decomposed by: oblast, party (side A/B/civilian), precision
  - Equipment losses decomposed by: category (tank, aircraft, artillery, etc.)
  - FIRMS fires decomposed by: day/night, confidence level, intensity bands

MOST IMPORTANT FEATURE GROUPS (from attention network):
""")
    sorted_groups = sorted(zip(feature_groups.keys(), importance), key=lambda x: -x[1])
    for name, imp in sorted_groups:
        bar = "█" * int(imp * 30)
        print(f"  {name:20s} {bar} {imp:.3f}")

    print("""
TOP CROSS-DOMAIN FINDINGS:
""")

    # Find top predictors across targets
    for target, imp_df in importance_results.items():
        top = imp_df.iloc[0]
        print(f"  Best predictor for {target}: {top['feature']} (attention: {top['attention']:.4f})")

    print("""
INTERPRETATION:
  - Decomposed features reveal which SPECIFIC aspects drive outcomes
  - Regional breakdown shows which oblasts dominate casualty patterns
  - Equipment categories show which platforms are most heavily engaged
  - Day/night fire patterns may indicate different operational tempos
  - Cross-group attention reveals which data sources inform each other
""")


if __name__ == "__main__":
    main()
