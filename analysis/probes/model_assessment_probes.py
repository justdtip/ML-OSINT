"""
Model Assessment Probes for Multi-Resolution HAN Pipeline

This module provides comprehensive model assessment capabilities:

Section 8.1: Cross-Model Comparison
    - 8.1.1: Model Architecture Comparison
    - 8.1.2: Reconstruction Performance Comparison

Section 8.2: HAN Assessment
    - 8.2.1: Multi-Task Performance Assessment
    - 8.2.2: Training Dynamics Analysis

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
from scipy.stats import pearsonr

# Centralized path configuration
from config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    ANALYSIS_DIR,
    MODEL_DIR,
    MULTI_RES_CHECKPOINT_DIR,
    get_probe_figures_dir,
    get_probe_metrics_dir,
)

# Import base probe infrastructure
from .data_artifact_probes import Probe, ProbeResult

# Check for torch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - model probes will be limited")


# =============================================================================
# SECTION 8.1: CROSS-MODEL COMPARISON
# =============================================================================

class ModelArchitectureComparisonProbe(Probe):
    """
    Probe 8.1.1: Model Architecture Comparison

    Compares architectures of different unified model variants
    (cumulative vs delta) in terms of parameters, features, and structure.

    Based on: comprehensive_model_report.py
    """

    @property
    def test_id(self) -> str:
        return "8.1.1"

    @property
    def test_name(self) -> str:
        return "Model Architecture Comparison"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute model architecture comparison."""
        self.log("Starting model architecture comparison...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Find available models
        model_paths = {
            'cumulative': MODEL_DIR / 'unified_interpolation_best.pt',
            'delta': MODEL_DIR / 'unified_interpolation_delta_best.pt',
        }

        available_models = {k: v for k, v in model_paths.items() if v.exists()}

        if not available_models:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No unified models found'}],
                recommendations=['Train unified interpolation models first']
            )

        model_analyses = []

        for model_name, model_path in available_models.items():
            self.log(f"Analyzing {model_name} model...")

            try:
                state = torch.load(model_path, map_location='cpu', weights_only=False)

                # Analyze state dict structure
                total_params = 0
                encoder_params = 0
                fusion_params = 0
                decoder_params = 0

                source_features = {}
                source_encoder_params = {}

                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        params = value.numel()
                        total_params += params

                        if 'encoder' in key:
                            encoder_params += params
                            # Extract source name
                            parts = key.split('.')
                            if len(parts) >= 2:
                                source = parts[1]
                                source_encoder_params[source] = source_encoder_params.get(source, 0) + params

                                # Extract feature count from weight shape
                                if 'feature_proj.0.weight' in key:
                                    source_features[source] = value.shape[1]

                        elif 'fusion' in key:
                            fusion_params += params
                        elif 'decoder' in key:
                            decoder_params += params

                analysis = {
                    'model_name': model_name,
                    'total_params': total_params,
                    'encoder_params': encoder_params,
                    'fusion_params': fusion_params,
                    'decoder_params': decoder_params,
                    'source_features': source_features,
                    'source_encoder_params': source_encoder_params,
                    'n_sources': len(source_features),
                    'total_features': sum(source_features.values())
                }
                model_analyses.append(analysis)

                findings.append({
                    'category': 'ARCHITECTURE',
                    'model': model_name,
                    'total_params': total_params,
                    'n_sources': len(source_features),
                    'total_features': sum(source_features.values()),
                    'encoder_pct': encoder_params / total_params * 100 if total_params > 0 else 0,
                    'fusion_pct': fusion_params / total_params * 100 if total_params > 0 else 0,
                    'decoder_pct': decoder_params / total_params * 100 if total_params > 0 else 0,
                })

            except Exception as e:
                findings.append({
                    'category': 'ERROR',
                    'model': model_name,
                    'description': str(e)
                })

        # Create comparison table
        if model_analyses:
            comparison_df = pd.DataFrame([{
                'model': a['model_name'],
                'total_params': a['total_params'],
                'encoder_params': a['encoder_params'],
                'fusion_params': a['fusion_params'],
                'decoder_params': a['decoder_params'],
                'n_sources': a['n_sources'],
                'total_features': a['total_features']
            } for a in model_analyses])

            table_path = self.save_table(comparison_df, 'architecture_comparison')
            artifacts['tables'].append(table_path)

            # Create visualization
            fig = self._create_comparison_figure(model_analyses)
            fig_path = self.save_figure(fig, 'architecture_comparison')
            artifacts['figures'].append(fig_path)

            # Recommendations
            if len(model_analyses) >= 2:
                param_diff = abs(model_analyses[0]['total_params'] - model_analyses[1]['total_params'])
                if param_diff > model_analyses[0]['total_params'] * 0.1:
                    recommendations.append(
                        f"Significant parameter difference between models: {param_diff:,} params"
                    )

        self.log("Comparison complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'models_analyzed': list(available_models.keys())}
        )

    def _create_comparison_figure(self, analyses: List[Dict]) -> plt.Figure:
        """Create architecture comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Architecture Comparison', fontsize=14, fontweight='bold')

        model_names = [a['model_name'] for a in analyses]
        colors = {'cumulative': 'steelblue', 'delta': 'coral', 'hybrid': 'green'}

        # 1. Total parameters
        ax = axes[0, 0]
        params = [a['total_params'] for a in analyses]
        bars = ax.bar(model_names, params, color=[colors.get(m, 'gray') for m in model_names])
        ax.set_ylabel('Total Parameters')
        ax.set_title('Total Model Parameters')
        for bar, p in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                   f'{p:,}', ha='center', va='bottom', fontsize=9)

        # 2. Parameter distribution
        ax = axes[0, 1]
        x = np.arange(len(model_names))
        width = 0.25
        encoder = [a['encoder_params'] for a in analyses]
        fusion = [a['fusion_params'] for a in analyses]
        decoder = [a['decoder_params'] for a in analyses]

        ax.bar(x - width, encoder, width, label='Encoder', color='steelblue')
        ax.bar(x, fusion, width, label='Fusion', color='coral')
        ax.bar(x + width, decoder, width, label='Decoder', color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Parameters')
        ax.set_title('Parameter Distribution by Component')
        ax.legend()

        # 3. Features by source
        ax = axes[1, 0]
        if analyses:
            all_sources = set()
            for a in analyses:
                all_sources.update(a['source_features'].keys())
            all_sources = sorted(all_sources)

            x = np.arange(len(all_sources))
            width = 0.8 / len(analyses)

            for i, a in enumerate(analyses):
                features = [a['source_features'].get(s, 0) for s in all_sources]
                offset = (i - len(analyses)/2 + 0.5) * width
                ax.bar(x + offset, features, width, label=a['model_name'],
                      color=colors.get(a['model_name'], 'gray'))

            ax.set_xticks(x)
            ax.set_xticklabels(all_sources, rotation=45, ha='right')
            ax.set_ylabel('Number of Features')
            ax.set_title('Features by Source')
            ax.legend()

        # 4. Summary metrics
        ax = axes[1, 1]
        metrics = ['Total Features', 'N Sources', 'Params per Feature']
        x = np.arange(len(metrics))
        width = 0.8 / len(analyses)

        for i, a in enumerate(analyses):
            values = [
                a['total_features'],
                a['n_sources'],
                a['total_params'] / max(a['total_features'], 1)
            ]
            # Normalize for comparison
            values_norm = [v / max(1, max(aa[k] for aa in analyses)) for v, k in
                          zip(values, ['total_features', 'n_sources', 'total_params'])]
            offset = (i - len(analyses)/2 + 0.5) * width
            ax.bar(x + offset, values_norm, width, label=a['model_name'],
                  color=colors.get(a['model_name'], 'gray'))

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Summary Metrics (Normalized)')
        ax.legend()

        plt.tight_layout()
        return fig


class ReconstructionPerformanceProbe(Probe):
    """
    Probe 8.1.2: Reconstruction Performance Comparison

    Compares reconstruction performance (MSE, correlation) between
    different model variants on the same test data.
    """

    @property
    def test_id(self) -> str:
        return "8.1.2"

    @property
    def test_name(self) -> str:
        return "Reconstruction Performance Comparison"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute reconstruction performance comparison."""
        self.log("Starting reconstruction performance comparison...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # This probe requires loading models and data - provide placeholder
        findings.append({
            'category': 'INFO',
            'description': 'Full reconstruction comparison requires running comprehensive_model_report.py',
            'recommendation': 'Run: python -m analysis.comprehensive_model_report'
        })

        # Check if previous results exist
        results_path = ANALYSIS_DIR / 'comprehensive_model_comparison.json'
        if results_path.exists():
            self.log("Loading previous comparison results...")
            with open(results_path) as f:
                results = json.load(f)

            for model_name, model_results in results.items():
                if isinstance(model_results, dict):
                    findings.append({
                        'category': 'PREVIOUS_RESULTS',
                        'model': model_name,
                        'metrics': model_results
                    })

        recommendations.append(
            "For detailed reconstruction analysis, run: python -m analysis.comprehensive_model_report"
        )

        self.log("Probe complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'note': 'This probe checks for existing results'}
        )


# =============================================================================
# SECTION 8.2: HAN ASSESSMENT
# =============================================================================

class MultiTaskPerformanceProbe(Probe):
    """
    Probe 8.2.1: Multi-Task Performance Assessment

    Assesses performance across all HAN prediction tasks:
    regime classification, casualty prediction, anomaly detection, and forecasting.

    Based on: comprehensive_model_assessment.py
    """

    @property
    def test_id(self) -> str:
        return "8.2.1"

    @property
    def test_name(self) -> str:
        return "Multi-Task Performance Assessment"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute multi-task performance assessment."""
        self.log("Starting multi-task performance assessment...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Check for HAN checkpoint
        checkpoint_path = MULTI_RES_CHECKPOINT_DIR / 'best_checkpoint.pt'
        if not checkpoint_path.exists():
            # Try alternate paths
            alt_paths = [
                MULTI_RES_CHECKPOINT_DIR / 'checkpoint_epoch_99.pt',
                MODEL_DIR / 'multi_resolution' / 'best_checkpoint.pt',
            ]
            for alt in alt_paths:
                if alt.exists():
                    checkpoint_path = alt
                    break

        if not checkpoint_path.exists():
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No HAN checkpoint found'}],
                recommendations=['Train Multi-Resolution HAN model first']
            )

        self.log(f"Found checkpoint: {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Extract training history if available
            if 'history' in checkpoint:
                history = checkpoint['history']

                # Analyze loss curves
                for loss_name in ['train_loss', 'val_loss', 'regime_loss', 'casualty_loss',
                                  'anomaly_loss', 'forecast_loss']:
                    if loss_name in history:
                        losses = history[loss_name]
                        if losses:
                            findings.append({
                                'category': 'TRAINING_HISTORY',
                                'metric': loss_name,
                                'initial': float(losses[0]) if losses else None,
                                'final': float(losses[-1]) if losses else None,
                                'min': float(min(losses)) if losses else None,
                                'improvement_pct': float((losses[0] - losses[-1]) / losses[0] * 100) if losses and losses[0] > 0 else 0
                            })

            # Extract best metrics
            if 'best_val_loss' in checkpoint:
                findings.append({
                    'category': 'BEST_PERFORMANCE',
                    'best_val_loss': float(checkpoint['best_val_loss']),
                    'best_epoch': int(checkpoint.get('epoch', 0))
                })

            # Model config
            if 'config' in checkpoint:
                config = checkpoint['config']
                findings.append({
                    'category': 'MODEL_CONFIG',
                    'config': {k: v for k, v in config.items()
                              if isinstance(v, (int, float, str, bool))}
                })

            # Create visualization
            if 'history' in checkpoint:
                fig = self._create_training_figure(checkpoint['history'])
                fig_path = self.save_figure(fig, 'training_dynamics')
                artifacts['figures'].append(fig_path)

        except Exception as e:
            findings.append({
                'category': 'ERROR',
                'description': f'Failed to analyze checkpoint: {str(e)}'
            })

        recommendations.append(
            "For detailed task-specific analysis, run: python -m analysis.comprehensive_model_assessment"
        )

        self.log("Assessment complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'checkpoint_path': str(checkpoint_path)}
        )

    def _create_training_figure(self, history: Dict) -> plt.Figure:
        """Create training dynamics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Multi-Task Training Dynamics', fontsize=14, fontweight='bold')

        # 1. Overall loss
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train', alpha=0.7)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Overall Loss')
        ax.legend()
        ax.set_yscale('log')

        # 2. Task-specific losses
        ax = axes[0, 1]
        task_losses = ['regime_loss', 'casualty_loss', 'anomaly_loss', 'forecast_loss']
        for loss_name in task_losses:
            if loss_name in history and history[loss_name]:
                ax.plot(history[loss_name], label=loss_name.replace('_loss', ''), alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Task-Specific Losses')
        ax.legend()

        # 3. Learning rate (if available)
        ax = axes[1, 0]
        if 'lr' in history and history['lr']:
            ax.plot(history['lr'], color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
        else:
            ax.text(0.5, 0.5, 'LR history not available', ha='center', va='center',
                   transform=ax.transAxes)

        # 4. Validation metrics (if available)
        ax = axes[1, 1]
        val_metrics = ['val_regime_acc', 'val_casualty_r2', 'val_anomaly_auc']
        found_metrics = False
        for metric_name in val_metrics:
            if metric_name in history and history[metric_name]:
                ax.plot(history[metric_name], label=metric_name.replace('val_', ''), alpha=0.7)
                found_metrics = True
        if found_metrics:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metric Value')
            ax.set_title('Validation Metrics')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Validation metrics not available', ha='center', va='center',
                   transform=ax.transAxes)

        plt.tight_layout()
        return fig


class TrainingDynamicsProbe(Probe):
    """
    Probe 8.2.2: Training Dynamics Analysis

    Analyzes training dynamics including convergence, learning rate effects,
    and potential issues like overfitting or instability.
    """

    @property
    def test_id(self) -> str:
        return "8.2.2"

    @property
    def test_name(self) -> str:
        return "Training Dynamics Analysis"

    def run(self, data: Dict[str, Any] = None) -> ProbeResult:
        """Execute training dynamics analysis."""
        self.log("Starting training dynamics analysis...")

        if not HAS_TORCH:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'PyTorch not available'}],
                recommendations=['Install PyTorch']
            )

        findings = []
        artifacts = {'figures': [], 'tables': []}
        recommendations = []

        # Find all checkpoints
        checkpoint_files = list(MULTI_RES_CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt'))
        checkpoint_files.extend(MULTI_RES_CHECKPOINT_DIR.glob('best_checkpoint.pt'))

        if not checkpoint_files:
            return ProbeResult(
                test_id=self.test_id,
                test_name=self.test_name,
                findings=[{'category': 'ERROR', 'description': 'No checkpoints found'}],
                recommendations=['Train Multi-Resolution HAN model first']
            )

        self.log(f"Found {len(checkpoint_files)} checkpoint files")

        # Analyze convergence from best checkpoint
        best_checkpoint = MULTI_RES_CHECKPOINT_DIR / 'best_checkpoint.pt'
        if best_checkpoint.exists():
            try:
                checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)

                if 'history' in checkpoint:
                    history = checkpoint['history']

                    # Check for overfitting
                    if 'train_loss' in history and 'val_loss' in history:
                        train_loss = history['train_loss']
                        val_loss = history['val_loss']

                        if len(train_loss) > 10 and len(val_loss) > 10:
                            # Compare last 10 epochs
                            train_trend = np.mean(train_loss[-10:]) - np.mean(train_loss[-20:-10])
                            val_trend = np.mean(val_loss[-10:]) - np.mean(val_loss[-20:-10])

                            if train_trend < 0 and val_trend > 0:
                                findings.append({
                                    'category': 'OVERFITTING',
                                    'severity': 'warning',
                                    'description': 'Training loss decreasing while validation loss increasing',
                                    'train_trend': float(train_trend),
                                    'val_trend': float(val_trend)
                                })
                                recommendations.append(
                                    "Potential overfitting detected - consider early stopping or regularization"
                                )
                            else:
                                findings.append({
                                    'category': 'CONVERGENCE',
                                    'status': 'healthy',
                                    'description': 'No overfitting detected in loss trends'
                                })

                    # Check for NaN/Inf in training
                    for loss_name, losses in history.items():
                        if isinstance(losses, list) and losses:
                            if any(np.isnan(l) or np.isinf(l) for l in losses if l is not None):
                                findings.append({
                                    'category': 'NUMERICAL_STABILITY',
                                    'severity': 'error',
                                    'description': f'NaN/Inf detected in {loss_name}',
                                    'metric': loss_name
                                })
                                recommendations.append(
                                    f"Numerical instability in {loss_name} - check gradients and learning rate"
                                )

                    # Convergence rate analysis
                    if 'val_loss' in history and len(history['val_loss']) > 5:
                        val_loss = [l for l in history['val_loss'] if l is not None]
                        if len(val_loss) > 5:
                            # Estimate epochs to convergence
                            initial = val_loss[0]
                            final = val_loss[-1]
                            mid_point = (initial + final) / 2

                            epochs_to_mid = None
                            for i, l in enumerate(val_loss):
                                if l <= mid_point:
                                    epochs_to_mid = i
                                    break

                            findings.append({
                                'category': 'CONVERGENCE_RATE',
                                'initial_val_loss': float(initial),
                                'final_val_loss': float(final),
                                'improvement_pct': float((initial - final) / initial * 100),
                                'epochs_to_midpoint': epochs_to_mid,
                                'total_epochs': len(val_loss)
                            })

            except Exception as e:
                findings.append({
                    'category': 'ERROR',
                    'description': f'Failed to analyze checkpoint: {str(e)}'
                })

        self.log("Analysis complete!")

        return ProbeResult(
            test_id=self.test_id,
            test_name=self.test_name,
            findings=findings,
            artifacts=artifacts,
            recommendations=recommendations,
            metadata={'n_checkpoints': len(checkpoint_files)}
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Section 8.1: Cross-Model Comparison
    'ModelArchitectureComparisonProbe',
    'ReconstructionPerformanceProbe',

    # Section 8.2: HAN Assessment
    'MultiTaskPerformanceProbe',
    'TrainingDynamicsProbe',
]
