"""
Statistical Analysis Probes for Multi-Resolution HAN Pipeline

This module wraps standalone statistical analysis scripts into the probe system:

Section 1.4: Statistical Correlation Analysis
    - 1.4.1: Multi-Variable Correlation Analysis (deeper_correlation_analysis)
    - 1.4.2: Seasonal Pattern Analysis

Section 1.5: Neural Pattern Discovery
    - 1.5.1: Autoencoder Latent Pattern Mining (neural_pattern_analysis)
    - 1.5.2: Decomposed Feature Importance (decomposed_neural_analysis)

Section 7.2: Temporal-Spatial Coverage (enhanced)
    - 7.2.1: Multi-Source Temporal-Spatial Analysis (temporal_spatial_analysis_v2)

Author: Data Science Team
Date: 2026-01-24
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    ANALYSIS_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# Import base probe infrastructure from data_artifact_probes
from .data_artifact_probes import Probe, ProbeResult


# =============================================================================
# SECTION 1.4: STATISTICAL CORRELATION ANALYSIS
# =============================================================================

class MultiVariableCorrelationProbe(Probe):
    """
    Probe 1.4.1: Multi-Variable Correlation Analysis

    Analyzes correlations between cloud cover, conflict intensity, and other
    OSINT variables. Tests for statistical significance, seasonal patterns,
    and potential confounding.

    Based on: deeper_correlation_analysis.py
    """

    @property
    def test_id(self) -> str:
        return "1.4.1"

    @property
    def test_name(self) -> str:
        return "Multi-Variable Correlation Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute correlation analysis."""
        self.log("Starting multi-variable correlation analysis...")

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Load merged data
        merged_path = ANALYSIS_DIR / "sentinel_osint_merged.csv"
        if not merged_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{
                    'category': 'ERROR',
                    'description': f'Merged data file not found: {merged_path}',
                    'severity': 'high'
                }],
                recommendations=['Run sentinel_osint_integration.py first to create merged dataset']
            )

        df = pd.read_csv(merged_path)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # 1. Full correlation matrix
        self.log("Computing correlation matrix...")
        corr_cols = ['s2_avg_cloud', 'ucdp_events', 'ucdp_deaths', 'firms_fires',
                     'monthly_loss', 'ds_points', 's5p_no2', 's3_fire']
        available_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[available_cols].corr()

        # Save correlation matrix
        table_path = self.save_table(corr_matrix, 'correlation_matrix')
        artifacts['tables'].append(table_path)

        # Cloud cover correlations
        if 's2_avg_cloud' in available_cols:
            cloud_corr = corr_matrix['s2_avg_cloud'].drop('s2_avg_cloud', errors='ignore')
            for var, r in cloud_corr.items():
                if abs(r) > 0.3:
                    findings.append({
                        'category': 'CORRELATION',
                        'variable_pair': f's2_avg_cloud vs {var}',
                        'correlation': float(r),
                        'strength': 'strong' if abs(r) > 0.5 else 'moderate',
                        'interpretation': f"Cloud cover {'positively' if r > 0 else 'negatively'} correlated with {var}"
                    })

        # 2. Statistical significance testing
        self.log("Testing statistical significance...")
        key_pairs = [
            ('s2_avg_cloud', 'ucdp_deaths'),
            ('s2_avg_cloud', 'monthly_loss'),
            ('s2_avg_cloud', 'firms_fires'),
            ('monthly_loss', 'ds_points'),
            ('ucdp_deaths', 'monthly_loss'),
        ]

        sig_results = []
        for var1, var2 in key_pairs:
            if var1 in df.columns and var2 in df.columns:
                valid = df[[var1, var2]].dropna()
                if len(valid) > 10:
                    r, p = stats.pearsonr(valid[var1], valid[var2])
                    sig_results.append({
                        'var1': var1,
                        'var2': var2,
                        'r': float(r),
                        'p_value': float(p),
                        'significant': p < 0.05
                    })

                    if p < 0.05:
                        findings.append({
                            'category': 'SIGNIFICANCE',
                            'variable_pair': f'{var1} vs {var2}',
                            'r': float(r),
                            'p_value': float(p),
                            'description': f"Statistically significant correlation (p={p:.4f})"
                        })

        sig_df = pd.DataFrame(sig_results)
        if not sig_df.empty:
            table_path = self.save_table(sig_df, 'significance_tests')
            artifacts['tables'].append(table_path)

        # 3. Seasonal analysis
        self.log("Analyzing seasonal patterns...")
        if 'season' in df.columns:
            seasonal_stats = df.groupby('season').agg({
                col: 'mean' for col in available_cols if col in df.columns
            }).round(2)

            if not seasonal_stats.empty:
                seasonal_stats = seasonal_stats.reindex(['Winter', 'Spring', 'Summer', 'Fall'])
                table_path = self.save_table(seasonal_stats, 'seasonal_statistics')
                artifacts['tables'].append(table_path)

                # Check for seasonal variation in cloud cover
                if 's2_avg_cloud' in seasonal_stats.columns:
                    cloud_range = seasonal_stats['s2_avg_cloud'].max() - seasonal_stats['s2_avg_cloud'].min()
                    if cloud_range > 10:
                        findings.append({
                            'category': 'SEASONAL',
                            'description': f'Significant seasonal cloud variation: {cloud_range:.1f}% range',
                            'winter_avg': float(seasonal_stats.loc['Winter', 's2_avg_cloud']),
                            'summer_avg': float(seasonal_stats.loc['Summer', 's2_avg_cloud']),
                        })

        # 4. Create visualization
        self.log("Creating visualization...")
        fig = self._create_correlation_figure(df, corr_matrix, available_cols)
        fig_path = self.save_figure(fig, 'correlation_analysis')
        artifacts['figures'].append(fig_path)

        # 5. Partial correlation (controlling for time)
        self.log("Computing partial correlations...")
        if 's2_avg_cloud' in df.columns and 'ucdp_deaths' in df.columns:
            df['day_index'] = (df['date'] - df['date'].min()).dt.days

            valid = df[['s2_avg_cloud', 'ucdp_deaths', 'day_index']].dropna()
            if len(valid) > 10:
                partial_r, partial_p = self._partial_correlation(
                    valid['s2_avg_cloud'].values,
                    valid['ucdp_deaths'].values,
                    valid['day_index'].values
                )

                findings.append({
                    'category': 'PARTIAL_CORRELATION',
                    'description': 'Cloud-Deaths correlation controlling for time trend',
                    'partial_r': float(partial_r) if not np.isnan(partial_r) else None,
                    'raw_r': float(corr_matrix.loc['s2_avg_cloud', 'ucdp_deaths']) if 'ucdp_deaths' in corr_matrix.columns else None,
                    'interpretation': 'Partial correlation removes spurious time-trend effects'
                })

                # If partial correlation is much lower, flag potential confound
                if 's2_avg_cloud' in corr_matrix.columns and 'ucdp_deaths' in corr_matrix.columns:
                    raw_r = abs(corr_matrix.loc['s2_avg_cloud', 'ucdp_deaths'])
                    if not np.isnan(partial_r) and abs(partial_r) < raw_r * 0.5:
                        recommendations.append(
                            f"Cloud-deaths correlation may be confounded by time trend "
                            f"(raw r={raw_r:.3f}, partial r={partial_r:.3f})"
                        )

        # Final recommendations
        if not recommendations:
            recommendations.append("Correlation analysis complete - review findings for actionable insights")

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={
                'n_samples': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'variables_analyzed': available_cols
            }
        )

    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
        """Compute partial correlation between x and y, controlling for z."""
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[valid], y[valid], z[valid]

        if len(x) < 4:
            return np.nan, 1.0

        r_xy, _ = pearsonr(x, y)
        r_xz, _ = pearsonr(x, z)
        r_yz, _ = pearsonr(y, z)

        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        if denominator < 1e-10:
            return np.nan, 1.0

        partial_r = (r_xy - r_xz * r_yz) / denominator

        n = len(x)
        df = n - 3
        if df <= 0:
            return partial_r, 1.0

        t_stat = partial_r * np.sqrt(df / (1 - partial_r**2 + 1e-10))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return partial_r, p_value

    def _create_correlation_figure(self, df: pd.DataFrame, corr_matrix: pd.DataFrame,
                                   cols: List[str]) -> plt.Figure:
        """Create correlation analysis visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Multi-Variable Correlation Analysis', fontsize=14, fontweight='bold')

        # 1. Correlation heatmap
        ax = axes[0, 0]
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax, square=True,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Matrix')

        # 2. Cloud vs Deaths scatter
        ax = axes[0, 1]
        if 's2_avg_cloud' in df.columns and 'ucdp_deaths' in df.columns:
            valid = df[['s2_avg_cloud', 'ucdp_deaths']].dropna()
            ax.scatter(valid['s2_avg_cloud'], valid['ucdp_deaths'], alpha=0.5, s=20)
            ax.set_xlabel('Cloud Cover (%)')
            ax.set_ylabel('UCDP Deaths')
            ax.set_title('Cloud Cover vs Deaths')

            # Add regression line
            if len(valid) > 2:
                z = np.polyfit(valid['s2_avg_cloud'], valid['ucdp_deaths'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid['s2_avg_cloud'].min(), valid['s2_avg_cloud'].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'r={corr_matrix.loc["s2_avg_cloud", "ucdp_deaths"]:.2f}')
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

        # 3. Cloud vs Losses scatter
        ax = axes[0, 2]
        if 's2_avg_cloud' in df.columns and 'monthly_loss' in df.columns:
            valid = df[['s2_avg_cloud', 'monthly_loss']].dropna()
            ax.scatter(valid['s2_avg_cloud'], valid['monthly_loss'], alpha=0.5, s=20)
            ax.set_xlabel('Cloud Cover (%)')
            ax.set_ylabel('Monthly Equipment Loss')
            ax.set_title('Cloud Cover vs Losses')
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

        # 4. Seasonal deaths boxplot
        ax = axes[1, 0]
        if 'season' in df.columns and 'ucdp_deaths' in df.columns:
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            df_plot = df[df['season'].isin(season_order)]
            sns.boxplot(data=df_plot, x='season', y='ucdp_deaths', ax=ax,
                        order=season_order, hue='season', palette='coolwarm', legend=False)
            ax.set_title('Deaths by Season')
            ax.set_xlabel('')
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

        # 5. Seasonal losses boxplot
        ax = axes[1, 1]
        if 'season' in df.columns and 'monthly_loss' in df.columns:
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            df_plot = df[df['season'].isin(season_order)]
            sns.boxplot(data=df_plot, x='season', y='monthly_loss', ax=ax,
                        order=season_order, hue='season', palette='coolwarm', legend=False)
            ax.set_title('Equipment Loss by Season')
            ax.set_xlabel('')
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

        # 6. Time series
        ax = axes[1, 2]
        if 's2_avg_cloud' in df.columns and 'ucdp_deaths' in df.columns:
            ax2 = ax.twinx()

            ax.plot(df['date'], df['s2_avg_cloud'], 'b-', alpha=0.7, label='Cloud Cover')
            ax2.plot(df['date'], df['ucdp_deaths'], 'r-', alpha=0.7, label='Deaths')

            ax.set_xlabel('Date')
            ax.set_ylabel('Cloud Cover (%)', color='b')
            ax2.set_ylabel('Deaths', color='r')
            ax.set_title('Time Series Comparison')

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        return fig


class SeasonalPatternProbe(Probe):
    """
    Probe 1.4.2: Seasonal Pattern Analysis

    Detailed analysis of seasonal patterns in conflict data to identify
    operational tempo variations, weather effects, and potential tactical patterns.
    """

    @property
    def test_id(self) -> str:
        return "1.4.2"

    @property
    def test_name(self) -> str:
        return "Seasonal Pattern Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute seasonal pattern analysis."""
        self.log("Starting seasonal pattern analysis...")

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Load data
        merged_path = ANALYSIS_DIR / "sentinel_osint_merged.csv"
        if not merged_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'Data not found'}],
                recommendations=['Run sentinel_osint_integration.py first']
            )

        df = pd.read_csv(merged_path)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter

        # Monthly patterns
        self.log("Analyzing monthly patterns...")
        monthly_cols = ['ucdp_deaths', 'monthly_loss', 'firms_fires', 's2_avg_cloud']
        available_cols = [c for c in monthly_cols if c in df.columns]

        monthly_stats = df.groupby('month')[available_cols].agg(['mean', 'std', 'count'])

        # Find peak months for each variable
        for col in available_cols:
            if col in monthly_stats.columns.get_level_values(0):
                peak_month = monthly_stats[col]['mean'].idxmax()
                trough_month = monthly_stats[col]['mean'].idxmin()

                findings.append({
                    'category': 'MONTHLY_PATTERN',
                    'variable': col,
                    'peak_month': int(peak_month),
                    'trough_month': int(trough_month),
                    'peak_value': float(monthly_stats[col]['mean'].max()),
                    'trough_value': float(monthly_stats[col]['mean'].min()),
                    'range_pct': float((monthly_stats[col]['mean'].max() - monthly_stats[col]['mean'].min())
                                      / (monthly_stats[col]['mean'].mean() + 1e-6) * 100)
                })

        # Day of week patterns (operational tempo)
        if 'ucdp_deaths' in df.columns:
            dow_stats = df.groupby('day_of_week')['ucdp_deaths'].agg(['mean', 'std'])

            # ANOVA test for day-of-week effect
            dow_groups = [group['ucdp_deaths'].dropna().values for _, group in df.groupby('day_of_week')]
            dow_groups = [g for g in dow_groups if len(g) > 0]

            if len(dow_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*dow_groups)

                findings.append({
                    'category': 'DAY_OF_WEEK',
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': 'Significant weekly operational pattern detected' if p_value < 0.05
                                     else 'No significant weekly pattern'
                })

        # Create visualization
        fig = self._create_seasonal_figure(df, available_cols)
        fig_path = self.save_figure(fig, 'seasonal_patterns')
        artifacts['figures'].append(fig_path)

        # Recommendations
        for finding in findings:
            if finding['category'] == 'MONTHLY_PATTERN' and finding.get('range_pct', 0) > 50:
                recommendations.append(
                    f"Strong seasonal variation in {finding['variable']} ({finding['range_pct']:.0f}% range) - "
                    f"consider seasonal adjustment or detrending"
                )

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'n_samples': len(df)}
        )

    def _create_seasonal_figure(self, df: pd.DataFrame, cols: List[str]) -> plt.Figure:
        """Create seasonal pattern visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Seasonal Pattern Analysis', fontsize=14, fontweight='bold')

        # 1. Monthly means
        ax = axes[0, 0]
        monthly_means = df.groupby('month')[cols].mean()
        for col in cols:
            if col in monthly_means.columns:
                ax.plot(monthly_means.index, monthly_means[col] / monthly_means[col].max(),
                       marker='o', label=col)
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Monthly Patterns (Normalized)')
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 13))

        # 2. Day of week patterns
        ax = axes[0, 1]
        if 'ucdp_deaths' in df.columns:
            dow_stats = df.groupby('day_of_week')['ucdp_deaths'].agg(['mean', 'std'])
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax.bar(days, dow_stats['mean'], yerr=dow_stats['std'], capsize=3, alpha=0.7)
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Mean Deaths')
            ax.set_title('Weekly Operational Tempo')

        # 3. Quarterly comparison
        ax = axes[1, 0]
        if 'ucdp_deaths' in df.columns:
            quarterly = df.groupby('quarter')['ucdp_deaths'].agg(['mean', 'std'])
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            ax.bar(quarters, quarterly['mean'], yerr=quarterly['std'], capsize=3, alpha=0.7, color='coral')
            ax.set_xlabel('Quarter')
            ax.set_ylabel('Mean Deaths')
            ax.set_title('Quarterly Patterns')

        # 4. Cloud-conflict seasonal relationship
        ax = axes[1, 1]
        if 's2_avg_cloud' in df.columns and 'ucdp_deaths' in df.columns:
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'orange', 'Fall': 'brown'}
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                subset = df[df['season'] == season]
                ax.scatter(subset['s2_avg_cloud'], subset['ucdp_deaths'],
                          alpha=0.5, label=season, color=colors.get(season, 'gray'), s=20)
            ax.set_xlabel('Cloud Cover (%)')
            ax.set_ylabel('Deaths')
            ax.set_title('Cloud-Conflict by Season')
            ax.legend()

        plt.tight_layout()
        return fig


# =============================================================================
# SECTION 1.5: NEURAL PATTERN DISCOVERY
# =============================================================================

class NeuralPatternMiningProbe(Probe):
    """
    Probe 1.5.1: Neural Pattern Mining

    Uses autoencoder-based compression to discover latent patterns in
    multi-source OSINT data. Identifies clusters and non-linear relationships.

    Based on: neural_pattern_analysis.py
    """

    @property
    def test_id(self) -> str:
        return "1.5.1"

    @property
    def test_name(self) -> str:
        return "Neural Pattern Mining"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute neural pattern mining analysis."""
        self.log("Starting neural pattern mining...")

        # Import torch here to avoid import errors if not available
        try:
            import torch
            import torch.nn as nn
            from sklearn.cluster import KMeans
            from sklearn.manifold import TSNE
        except ImportError as e:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': f'Missing dependency: {e}'}],
                recommendations=['Install torch and sklearn']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Load data
        merged_path = ANALYSIS_DIR / "sentinel_osint_merged.csv"
        if not merged_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'Data not found'}],
                recommendations=['Run sentinel_osint_integration.py first']
            )

        df = pd.read_csv(merged_path)
        df['date'] = pd.to_datetime(df['date'])

        # Select features for autoencoder
        feature_cols = ['s2_avg_cloud', 'ucdp_events', 'ucdp_deaths', 'firms_fires',
                        'monthly_loss', 'ds_points', 's5p_no2', 's3_fire']
        available_cols = [c for c in feature_cols if c in df.columns]

        if len(available_cols) < 4:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'Insufficient features available'}],
                recommendations=['Ensure merged dataset has required features']
            )

        # Prepare data
        X = df[available_cols].dropna()
        dates = df.loc[X.index, 'date']

        # Normalize
        X_mean = X.mean()
        X_std = X.std() + 1e-8
        X_norm = (X - X_mean) / X_std
        X_tensor = torch.tensor(X_norm.values, dtype=torch.float32)

        self.log(f"Training autoencoder on {len(X)} samples with {len(available_cols)} features...")

        # Simple autoencoder
        latent_dim = 4

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        model = Autoencoder(len(available_cols), latent_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Train
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            recon, _ = model(X_tensor)
            loss = criterion(recon, X_tensor)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        self.log(f"Final reconstruction loss: {final_loss:.4f}")

        # Get latent representations
        model.eval()
        with torch.no_grad():
            _, latent = model(X_tensor)
            latent_np = latent.numpy()

        # Cluster analysis
        self.log("Performing cluster analysis...")
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(latent_np)

        # Cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            mask = clusters == i
            stats = {
                'cluster': i,
                'n_samples': int(mask.sum()),
                'pct_samples': float(mask.sum() / len(clusters) * 100)
            }
            for col in available_cols:
                stats[f'{col}_mean'] = float(X.iloc[mask][col].mean())
            cluster_stats.append(stats)

        findings.append({
            'category': 'CLUSTERING',
            'n_clusters': n_clusters,
            'cluster_sizes': [s['n_samples'] for s in cluster_stats],
            'reconstruction_loss': float(final_loss)
        })

        # Feature-latent correlations
        latent_df = pd.DataFrame(latent_np, columns=[f'z{i}' for i in range(latent_dim)])
        for i, col in enumerate(available_cols):
            for j in range(latent_dim):
                r, _ = pearsonr(X[col].values, latent_np[:, j])
                if abs(r) > 0.5:
                    findings.append({
                        'category': 'LATENT_CORRELATION',
                        'feature': col,
                        'latent_dim': j,
                        'correlation': float(r)
                    })

        # Create visualization
        fig = self._create_pattern_figure(X, latent_np, clusters, dates, available_cols)
        fig_path = self.save_figure(fig, 'neural_patterns')
        artifacts['figures'].append(fig_path)

        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'date': dates.values,
            'cluster': clusters
        })
        table_path = self.save_table(cluster_df, 'cluster_assignments')
        artifacts['tables'].append(table_path)

        recommendations.append(
            f"Identified {n_clusters} distinct operational patterns in the data"
        )

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={
                'n_samples': len(X),
                'n_features': len(available_cols),
                'latent_dim': latent_dim,
                'n_clusters': n_clusters
            }
        )

    def _create_pattern_figure(self, X: pd.DataFrame, latent: np.ndarray,
                               clusters: np.ndarray, dates: pd.Series,
                               feature_cols: List[str]) -> plt.Figure:
        """Create neural pattern visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Neural Pattern Mining Results', fontsize=14, fontweight='bold')

        # 1. t-SNE of latent space
        ax = axes[0, 0]
        if len(latent) > 30:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent)-1))
                latent_2d = tsne.fit_transform(latent)
                scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters,
                                    cmap='viridis', alpha=0.6, s=20)
                ax.set_title('t-SNE of Latent Space')
                plt.colorbar(scatter, ax=ax, label='Cluster')
            except Exception:
                ax.text(0.5, 0.5, 't-SNE failed', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Insufficient samples for t-SNE', ha='center', va='center', transform=ax.transAxes)

        # 2. Cluster profiles
        ax = axes[0, 1]
        cluster_means = []
        for i in range(len(np.unique(clusters))):
            mask = clusters == i
            means = X.iloc[mask].mean()
            # Normalize for comparison
            means_norm = (means - X.mean()) / (X.std() + 1e-8)
            cluster_means.append(means_norm)

        if cluster_means:
            cluster_df = pd.DataFrame(cluster_means, columns=feature_cols)
            cluster_df.T.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Cluster Feature Profiles (Z-scored)')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Cluster', fontsize=8)

        # 3. Clusters over time
        ax = axes[0, 2]
        ax.scatter(dates, clusters, c=clusters, cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cluster')
        ax.set_title('Cluster Assignments Over Time')

        # 4-6. Latent dimension correlations
        for idx, dim in enumerate(range(min(3, latent.shape[1]))):
            ax = axes[1, idx]
            correlations = []
            for col in feature_cols:
                r, _ = pearsonr(X[col].values, latent[:, dim])
                correlations.append(r)

            colors = ['green' if r > 0 else 'red' for r in correlations]
            ax.barh(feature_cols, correlations, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Correlation')
            ax.set_title(f'Latent Dim {dim} Correlations')

        plt.tight_layout()
        return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MultiVariableCorrelationProbe',
    'SeasonalPatternProbe',
    'NeuralPatternMiningProbe',
]
