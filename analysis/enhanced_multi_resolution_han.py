"""
Enhanced Multi-Resolution HAN with Timeline and ISW Integration

This module extends the base MultiResolutionHAN to incorporate:
1. ISW narrative embeddings (voyage-4-large, 1024-dim)
2. Timeline/operation phase features (14-dim)
3. Operation context from major campaign events

Architecture Enhancement:
========================

    DAILY DATA (5 sources)          ISW EMBEDDINGS (1024-dim)
         |                                    |
    DailySourceEncoders              ISW Projection (128-dim)
         |                                    |
    DailyCrossSourceFusion ←──── ISWGatedFusion
         |
    [Timeline Integration] ←──── Operation Features (14-dim)
         |                        Phase-Aware Attention
    LearnableMonthlyAggregation   Operation Context
         |
    CrossResolutionFusion ←──── Monthly Sources
         |
    TemporalEncoder
         |
    Multi-Task Heads + Phase Classification

New Features:
- Phase-aware regime classification (uses curated operation phases)
- Operation context injection (major campaign embeddings)
- ISW narrative fusion with gating
- Contrastive alignment between quantitative and narrative modalities

Usage:
    from enhanced_multi_resolution_han import create_enhanced_han

    model = create_enhanced_han(
        d_model=128,
        enable_isw=True,
        enable_timeline=True,
        timeline_dir="data/timelines",
    )

    outputs = model(
        daily_features=daily_features,
        daily_masks=daily_masks,
        monthly_features=monthly_features,
        monthly_masks=monthly_masks,
        month_boundaries=month_boundaries,
        isw_embeddings=isw_embeddings,  # Optional
        isw_mask=isw_mask,              # Optional
        dates=date_list,                # Optional
    )

Author: AI Engineering Team
Date: 2026-01-21
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import base components
try:
    from multi_resolution_han import (
        MultiResolutionHAN,
        SourceConfig,
        create_multi_resolution_han,
        DailySourceEncoder,
        DailyCrossSourceFusion,
        EnhancedLearnableMonthlyAggregation,
        TemporalEncoder,
        CasualtyPredictionHead,
        RegimeClassificationHead,
        AnomalyDetectionHead,
        ForecastingHead,
        UncertaintyEstimator,
    )
    from multi_resolution_modules import (
        MonthlySourceConfig,
        MultiSourceMonthlyEncoder,
        CrossResolutionFusion,
        SinusoidalPositionalEncoding,
    )
    from isw_embedding_integration import (
        ISWIntegrationModule,
        ISWIntegrationConfig,
        prepare_key_quantitative_targets,
    )
    from timeline_integration import (
        TimelineIntegrationModule,
        TimelineIntegrationConfig,
    )
except ImportError:
    from analysis.multi_resolution_han import (
        MultiResolutionHAN,
        SourceConfig,
        create_multi_resolution_han,
        DailySourceEncoder,
        DailyCrossSourceFusion,
        EnhancedLearnableMonthlyAggregation,
        TemporalEncoder,
        CasualtyPredictionHead,
        RegimeClassificationHead,
        AnomalyDetectionHead,
        ForecastingHead,
        UncertaintyEstimator,
    )
    from analysis.multi_resolution_modules import (
        MonthlySourceConfig,
        MultiSourceMonthlyEncoder,
        CrossResolutionFusion,
        SinusoidalPositionalEncoding,
    )
    from analysis.isw_embedding_integration import (
        ISWIntegrationModule,
        ISWIntegrationConfig,
        prepare_key_quantitative_targets,
    )
    from analysis.timeline_integration import (
        TimelineIntegrationModule,
        TimelineIntegrationConfig,
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnhancedHANConfig:
    """Configuration for the enhanced multi-resolution HAN."""

    # Base model configuration
    d_model: int = 128
    nhead: int = 8
    num_daily_layers: int = 4
    num_monthly_layers: int = 3
    num_fusion_layers: int = 2
    num_temporal_layers: int = 2
    dropout: float = 0.1

    # ISW integration
    enable_isw: bool = True
    isw_dim: int = 1024
    isw_use_frozen_projection: bool = True
    isw_use_cross_attention: bool = False
    isw_use_contrastive: bool = True
    isw_contrastive_weight: float = 0.1

    # Timeline integration
    enable_timeline: bool = True
    timeline_dir: Optional[str] = None
    n_phases: int = 11
    operation_feature_dim: int = 14
    use_phase_attention: bool = True
    use_operation_context: bool = True

    # Prediction tasks
    prediction_tasks: List[str] = field(
        default_factory=lambda: ['casualty', 'regime', 'anomaly', 'forecast']
    )

    # Phase-aware regime classification
    enable_phase_regime: bool = True


# =============================================================================
# PHASE-AWARE REGIME HEAD
# =============================================================================

class PhaseAwareRegimeHead(nn.Module):
    """
    Regime classification head that uses curated operation phases.

    Instead of learning regime classes from scratch, this head is supervised
    using the known operation phases from the timeline data. This provides:
    1. Better interpretability (phases have real-world meaning)
    2. Semi-supervised signal from curated phase labels
    3. Phase transition detection as an auxiliary task
    """

    def __init__(
        self,
        d_model: int = 128,
        n_phases: int = 11,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_phases = n_phases

        # Phase classification head
        self.phase_classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_phases + 1),  # +1 for 'baseline'
        )

        # Transition detection head (binary: is this a phase transition?)
        self.transition_detector = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Intensity estimation (how active is the current phase?)
        self.intensity_estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
        return_all: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Predict phase, transition, and intensity.

        Args:
            x: Input representations [batch, seq_len, d_model]
            return_all: Whether to return all outputs

        Returns:
            Dict with:
                'phase_logits': [batch, seq_len, n_phases+1]
                'transition_score': [batch, seq_len, 1]
                'intensity': [batch, seq_len, 1]
        """
        outputs = {
            'phase_logits': self.phase_classifier(x),
        }

        if return_all:
            outputs['transition_score'] = self.transition_detector(x)
            outputs['intensity'] = self.intensity_estimator(x)

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, Tensor],
        phase_targets: Tensor,           # [batch, seq_len] phase indices
        transition_targets: Tensor,      # [batch, seq_len] binary
        phase_mask: Tensor,              # [batch, seq_len] True=valid
    ) -> Tensor:
        """Compute combined phase prediction loss."""
        # Phase classification loss
        phase_logits = predictions['phase_logits']
        batch_size, seq_len, n_classes = phase_logits.shape

        # Reshape for cross-entropy
        phase_logits_flat = phase_logits.view(-1, n_classes)
        phase_targets_flat = phase_targets.view(-1)
        mask_flat = phase_mask.view(-1)

        # Only compute loss on valid positions
        valid_indices = mask_flat.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=phase_logits.device, requires_grad=True)

        phase_loss = F.cross_entropy(
            phase_logits_flat[valid_indices],
            phase_targets_flat[valid_indices],
        )

        # Transition detection loss
        if 'transition_score' in predictions:
            trans_pred = predictions['transition_score'].view(-1)
            trans_target = transition_targets.float().view(-1)

            trans_loss = F.binary_cross_entropy_with_logits(
                trans_pred[valid_indices],
                trans_target[valid_indices],
            )
            phase_loss = phase_loss + 0.5 * trans_loss

        return phase_loss


# =============================================================================
# ENHANCED MULTI-RESOLUTION HAN
# =============================================================================

class EnhancedMultiResolutionHAN(nn.Module):
    """
    Multi-Resolution HAN with ISW narrative and timeline/operation integration.

    This model extends the base architecture to incorporate:
    1. Pre-computed ISW narrative embeddings via gated fusion
    2. Curated operation phase features
    3. Major campaign context from timeline embeddings
    4. Phase-supervised regime classification

    The integration is designed to be modular - ISW and timeline features
    can be enabled/disabled independently based on available data.
    """

    def __init__(
        self,
        config: EnhancedHANConfig,
        daily_source_configs: Optional[Dict[str, SourceConfig]] = None,
        monthly_source_configs: Optional[Dict[str, SourceConfig]] = None,
    ) -> None:
        super().__init__()

        self.config = config

        # Use default configs if not provided
        if daily_source_configs is None:
            daily_source_configs = {
                'equipment': SourceConfig('equipment', 11, 'daily'),
                'personnel': SourceConfig('personnel', 3, 'daily'),
                'deepstate': SourceConfig('deepstate', 5, 'daily'),
                'firms': SourceConfig('firms', 13, 'daily'),
                'viina': SourceConfig('viina', 6, 'daily'),
            }

        if monthly_source_configs is None:
            monthly_source_configs = {
                'sentinel': SourceConfig('sentinel', 7, 'monthly'),
                'hdx_conflict': SourceConfig('hdx_conflict', 5, 'monthly'),
                'hdx_food': SourceConfig('hdx_food', 10, 'monthly'),
                'hdx_rainfall': SourceConfig('hdx_rainfall', 6, 'monthly'),
                'iom': SourceConfig('iom', 7, 'monthly'),
            }

        self.daily_source_configs = daily_source_configs
        self.monthly_source_configs = monthly_source_configs
        self.daily_source_names = list(daily_source_configs.keys())
        self.monthly_source_names = list(monthly_source_configs.keys())

        # =====================================================================
        # DAILY ENCODERS (same as base)
        # =====================================================================
        self.daily_encoders = nn.ModuleDict({
            name: DailySourceEncoder(
                source_config=cfg,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_daily_layers,
                dropout=config.dropout,
            )
            for name, cfg in daily_source_configs.items()
        })

        # =====================================================================
        # DAILY CROSS-SOURCE FUSION
        # =====================================================================
        self.daily_fusion = DailyCrossSourceFusion(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_fusion_layers,
            dropout=config.dropout,
            source_names=self.daily_source_names,
        )

        # =====================================================================
        # ISW INTEGRATION (if enabled)
        # =====================================================================
        self.isw_integration = None
        if config.enable_isw:
            isw_config = ISWIntegrationConfig(
                isw_dim=config.isw_dim,
                projection_dim=config.d_model,
                dropout=config.dropout,
                use_frozen_projection=config.isw_use_frozen_projection,
                use_cross_attention=config.isw_use_cross_attention,
                use_contrastive=config.isw_use_contrastive,
                contrastive_weight=config.isw_contrastive_weight,
            )
            self.isw_integration = ISWIntegrationModule(isw_config)

        # =====================================================================
        # TIMELINE INTEGRATION (if enabled)
        # =====================================================================
        self.timeline_integration = None
        if config.enable_timeline:
            timeline_config = TimelineIntegrationConfig(
                d_model=config.d_model,
                n_phases=config.n_phases,
                operation_feature_dim=config.operation_feature_dim,
                dropout=config.dropout,
                use_phase_attention=config.use_phase_attention,
                use_operation_context=config.use_operation_context,
            )
            self.timeline_integration = TimelineIntegrationModule(timeline_config)

            # Load timeline data if path provided
            if config.timeline_dir:
                self.timeline_integration.load_timeline_data(config.timeline_dir)

        # =====================================================================
        # LEARNABLE MONTHLY AGGREGATION
        # =====================================================================
        self.monthly_aggregation = EnhancedLearnableMonthlyAggregation(
            d_model=config.d_model,
            nhead=config.nhead,
            max_months=60,
            dropout=config.dropout,
        )

        # =====================================================================
        # MONTHLY ENCODERS
        # =====================================================================
        monthly_configs = {
            name: MonthlySourceConfig(
                name=name,
                n_features=cfg.n_features,
                description=cfg.description,
                typical_observations=40,  # Default estimate
            )
            for name, cfg in monthly_source_configs.items()
        }

        self.monthly_encoder = MultiSourceMonthlyEncoder(
            source_configs=monthly_configs,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_monthly_layers,
            num_fusion_layers=config.num_fusion_layers,
            dropout=config.dropout,
        )

        # =====================================================================
        # CROSS-RESOLUTION FUSION
        # =====================================================================
        self.cross_resolution_fusion = CrossResolutionFusion(
            daily_dim=config.d_model,
            monthly_dim=config.d_model,
            hidden_dim=config.d_model,
            num_layers=config.num_fusion_layers,
            num_heads=config.nhead,
            dropout=config.dropout,
            use_gating=True,
            output_daily=False,
        )

        # =====================================================================
        # TEMPORAL ENCODER
        # =====================================================================
        self.temporal_encoder = TemporalEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_temporal_layers,
            dropout=config.dropout,
        )

        # =====================================================================
        # PREDICTION HEADS
        # =====================================================================
        if 'casualty' in config.prediction_tasks:
            self.casualty_head = CasualtyPredictionHead(
                d_model=config.d_model,
                hidden_dim=config.d_model * 2,
                dropout=config.dropout,
            )

        if 'regime' in config.prediction_tasks:
            if config.enable_phase_regime and config.enable_timeline:
                # Use phase-aware head with curated phases
                self.regime_head = PhaseAwareRegimeHead(
                    d_model=config.d_model,
                    n_phases=config.n_phases,
                    hidden_dim=config.d_model * 2,
                    dropout=config.dropout,
                )
            else:
                # Standard 4-class regime head
                self.regime_head = RegimeClassificationHead(
                    d_model=config.d_model,
                    num_classes=4,
                    hidden_dim=config.d_model * 2,
                    dropout=config.dropout,
                )

        if 'anomaly' in config.prediction_tasks:
            self.anomaly_head = AnomalyDetectionHead(
                d_model=config.d_model,
                hidden_dim=config.d_model * 2,
                dropout=config.dropout,
            )

        if 'forecast' in config.prediction_tasks:
            forecast_output_dim = sum(
                cfg.n_features for cfg in monthly_source_configs.values()
            )
            self.forecast_head = ForecastingHead(
                d_model=config.d_model,
                output_dim=forecast_output_dim,
                hidden_dim=config.d_model * 2,
                dropout=config.dropout,
            )

        # =====================================================================
        # UNCERTAINTY ESTIMATION
        # =====================================================================
        self.uncertainty_estimator = UncertaintyEstimator(
            d_model=config.d_model,
            hidden_dim=config.d_model // 2,
        )

    def forward(
        self,
        daily_features: Dict[str, Tensor],
        daily_masks: Dict[str, Tensor],
        monthly_features: Dict[str, Tensor],
        monthly_masks: Dict[str, Tensor],
        month_boundaries: Tensor,
        isw_embeddings: Optional[Tensor] = None,
        isw_mask: Optional[Tensor] = None,
        dates: Optional[List[List[str]]] = None,
        targets: Optional[Dict[str, Tensor]] = None,
        compute_aux_losses: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the enhanced multi-resolution HAN.

        Args:
            daily_features: Dict[source_name, Tensor[batch, daily_seq, features]]
            daily_masks: Dict[source_name, Tensor[batch, daily_seq, features]]
            monthly_features: Dict[source_name, Tensor[batch, monthly_seq, features]]
            monthly_masks: Dict[source_name, Tensor[batch, monthly_seq, features]]
            month_boundaries: Tensor[batch, n_months, 2]
            isw_embeddings: Optional ISW embeddings [batch, daily_seq, 1024]
            isw_mask: Optional ISW mask [batch, daily_seq]
            dates: Optional list of date strings per batch item
            targets: Optional dict of target tensors
            compute_aux_losses: Whether to compute auxiliary losses

        Returns:
            Dict containing predictions, attention weights, and auxiliary losses
        """
        outputs = {}
        aux_losses = {}

        # =====================================================================
        # STEP 1: ENCODE DAILY SOURCES
        # =====================================================================
        daily_encoded = {}
        for name in self.daily_source_names:
            if name in daily_features:
                encoded, _ = self.daily_encoders[name](
                    daily_features[name],
                    daily_masks[name],
                )
                daily_encoded[name] = encoded

        # =====================================================================
        # STEP 2: FUSE DAILY SOURCES
        # =====================================================================
        fused_daily, combined_daily_mask, _ = self.daily_fusion(
            daily_encoded, daily_masks
        )

        # =====================================================================
        # STEP 3: ISW INTEGRATION (if enabled and data provided)
        # =====================================================================
        if self.isw_integration is not None and isw_embeddings is not None:
            # Prepare key quantitative targets for N2S auxiliary task
            key_targets = None
            if compute_aux_losses:
                key_targets = prepare_key_quantitative_targets(
                    daily_features, daily_masks
                )

            fused_daily, isw_aux = self.isw_integration(
                quant_encoded=fused_daily,
                isw_embeddings=isw_embeddings,
                isw_mask=isw_mask,
                key_quant_targets=key_targets,
                compute_aux_losses=compute_aux_losses,
            )

            # Collect auxiliary losses
            if compute_aux_losses:
                if 'contrastive_loss' in isw_aux:
                    aux_losses['isw_contrastive_loss'] = isw_aux['contrastive_loss']
                if 'n2s_loss' in isw_aux:
                    aux_losses['isw_n2s_loss'] = isw_aux['n2s_loss']

        # =====================================================================
        # STEP 4: TIMELINE INTEGRATION (if enabled and dates provided)
        # =====================================================================
        phase_info = None
        if self.timeline_integration is not None and dates is not None:
            fused_daily, phase_info = self.timeline_integration(
                encoded=fused_daily,
                dates=dates,
                mask=combined_daily_mask,
                return_phase_info=True,
            )
            outputs['phase_info'] = phase_info

        # =====================================================================
        # STEP 5: AGGREGATE DAILY TO MONTHLY
        # =====================================================================
        n_months = month_boundaries.shape[1]

        aggregated_daily, aggregated_daily_mask, _ = self.monthly_aggregation(
            fused_daily,
            month_boundaries,
            combined_daily_mask,
        )

        # =====================================================================
        # STEP 6: ENCODE MONTHLY SOURCES
        # =====================================================================
        batch_size, monthly_seq_len = list(monthly_features.values())[0].shape[:2]
        device = aggregated_daily.device

        month_indices = torch.arange(monthly_seq_len, device=device).unsqueeze(0)
        month_indices = month_indices.expand(batch_size, -1)

        monthly_timestep_masks = {}
        for name, mask in monthly_masks.items():
            if mask.dim() == 3:
                monthly_timestep_masks[name] = mask.any(dim=-1).float()
            else:
                monthly_timestep_masks[name] = mask.float()

        monthly_encoder_output = self.monthly_encoder(
            source_features=monthly_features,
            source_masks=monthly_timestep_masks,
            month_indices=month_indices,
        )

        monthly_encoded = monthly_encoder_output['hidden']

        # =====================================================================
        # STEP 7: CROSS-RESOLUTION FUSION
        # =====================================================================
        min_len = min(aggregated_daily.shape[1], monthly_encoded.shape[1])

        aggregated_daily_aligned = aggregated_daily[:, :min_len]
        monthly_encoded_aligned = monthly_encoded[:, :min_len]
        aggregated_daily_mask_aligned = aggregated_daily_mask[:, :min_len]

        monthly_combined_mask = torch.zeros(
            batch_size, min_len, dtype=torch.bool, device=device
        )
        for mask in monthly_timestep_masks.values():
            if mask.shape[1] >= min_len:
                monthly_combined_mask = monthly_combined_mask | (mask[:, :min_len] > 0.5)
            else:
                monthly_combined_mask[:, :mask.shape[1]] = (
                    monthly_combined_mask[:, :mask.shape[1]] | (mask > 0.5)
                )

        fusion_boundaries = torch.stack([
            torch.arange(min_len, device=device),
            torch.arange(1, min_len + 1, device=device),
        ], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        fusion_output = self.cross_resolution_fusion(
            daily_repr=aggregated_daily_aligned,
            monthly_repr=monthly_encoded_aligned,
            daily_mask=aggregated_daily_mask_aligned,
            monthly_mask=monthly_combined_mask,
            month_boundaries=fusion_boundaries,
        )

        fused_monthly = fusion_output.fused_monthly

        # =====================================================================
        # STEP 8: TEMPORAL ENCODING
        # =====================================================================
        temporal_encoded = self.temporal_encoder(
            fused_monthly,
            monthly_combined_mask,
        )

        # =====================================================================
        # STEP 9: PREDICTION HEADS
        # =====================================================================
        if hasattr(self, 'casualty_head'):
            casualty_pred, casualty_var = self.casualty_head(
                temporal_encoded, return_variance=True
            )
            outputs['casualty_pred'] = casualty_pred
            outputs['casualty_var'] = casualty_var

        if hasattr(self, 'regime_head'):
            if isinstance(self.regime_head, PhaseAwareRegimeHead):
                regime_outputs = self.regime_head(temporal_encoded)
                outputs['phase_logits'] = regime_outputs['phase_logits']
                outputs['transition_score'] = regime_outputs.get('transition_score')
                outputs['intensity'] = regime_outputs.get('intensity')
            else:
                outputs['regime_logits'] = self.regime_head(temporal_encoded)

        if hasattr(self, 'anomaly_head'):
            outputs['anomaly_score'] = self.anomaly_head(temporal_encoded)

        if hasattr(self, 'forecast_head'):
            outputs['forecast_pred'] = self.forecast_head(temporal_encoded)

        # =====================================================================
        # STEP 10: UNCERTAINTY ESTIMATION
        # =====================================================================
        outputs['uncertainty'] = self.uncertainty_estimator(temporal_encoded)

        # =====================================================================
        # STEP 11: AUXILIARY LOSSES
        # =====================================================================
        outputs['aux_losses'] = aux_losses

        return outputs

    def load_timeline_data(self, timeline_dir: Union[str, Path]) -> bool:
        """Load timeline data for the timeline integration module."""
        if self.timeline_integration is not None:
            return self.timeline_integration.load_timeline_data(timeline_dir)
        return False

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'd_model': self.config.d_model,
            'nhead': self.config.nhead,
            'enable_isw': self.config.enable_isw,
            'enable_timeline': self.config.enable_timeline,
            'prediction_tasks': self.config.prediction_tasks,
            'daily_sources': list(self.daily_source_configs.keys()),
            'monthly_sources': list(self.monthly_source_configs.keys()),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_han(
    d_model: int = 128,
    nhead: int = 8,
    num_daily_layers: int = 4,
    num_monthly_layers: int = 3,
    num_fusion_layers: int = 2,
    num_temporal_layers: int = 2,
    dropout: float = 0.1,
    enable_isw: bool = True,
    enable_timeline: bool = True,
    timeline_dir: Optional[str] = None,
    prediction_tasks: Optional[List[str]] = None,
) -> EnhancedMultiResolutionHAN:
    """
    Factory function to create an enhanced HAN with sensible defaults.

    Args:
        d_model: Hidden dimension
        nhead: Number of attention heads
        num_daily_layers: Layers per daily encoder
        num_monthly_layers: Layers per monthly encoder
        num_fusion_layers: Cross-resolution fusion layers
        num_temporal_layers: Final temporal encoder layers
        dropout: Dropout probability
        enable_isw: Enable ISW narrative integration
        enable_timeline: Enable timeline/operation integration
        timeline_dir: Path to timeline data directory
        prediction_tasks: List of prediction tasks

    Returns:
        Configured EnhancedMultiResolutionHAN instance
    """
    config = EnhancedHANConfig(
        d_model=d_model,
        nhead=nhead,
        num_daily_layers=num_daily_layers,
        num_monthly_layers=num_monthly_layers,
        num_fusion_layers=num_fusion_layers,
        num_temporal_layers=num_temporal_layers,
        dropout=dropout,
        enable_isw=enable_isw,
        enable_timeline=enable_timeline,
        timeline_dir=timeline_dir,
        prediction_tasks=prediction_tasks or ['casualty', 'regime', 'anomaly', 'forecast'],
    )

    return EnhancedMultiResolutionHAN(config)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Enhanced Multi-Resolution HAN - Tests")
    print("=" * 80)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test configuration
    batch_size = 2
    daily_seq_len = 500
    monthly_seq_len = 20

    # =========================================================================
    # TEST 1: Create Model
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 1: Create Enhanced HAN")
    print("-" * 40)

    try:
        model = create_enhanced_han(
            d_model=128,
            nhead=8,
            num_daily_layers=2,
            num_monthly_layers=2,
            num_fusion_layers=1,
            num_temporal_layers=1,
            dropout=0.1,
            enable_isw=True,
            enable_timeline=True,
        )
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model created!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # =========================================================================
    # TEST 2: Create Dummy Data
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 2: Create Dummy Data")
    print("-" * 40)

    try:
        # Daily features
        daily_features = {
            'equipment': torch.randn(batch_size, daily_seq_len, 11, device=device),
            'personnel': torch.randn(batch_size, daily_seq_len, 3, device=device),
            'deepstate': torch.randn(batch_size, daily_seq_len, 5, device=device),
            'firms': torch.randn(batch_size, daily_seq_len, 13, device=device),
            'viina': torch.randn(batch_size, daily_seq_len, 6, device=device),
        }

        daily_masks = {
            name: torch.rand(batch_size, daily_seq_len, feat.shape[-1], device=device) > 0.2
            for name, feat in daily_features.items()
        }

        # Monthly features
        monthly_features = {
            'sentinel': torch.randn(batch_size, monthly_seq_len, 7, device=device),
            'hdx_conflict': torch.randn(batch_size, monthly_seq_len, 5, device=device),
            'hdx_food': torch.randn(batch_size, monthly_seq_len, 10, device=device),
            'hdx_rainfall': torch.randn(batch_size, monthly_seq_len, 6, device=device),
            'iom': torch.randn(batch_size, monthly_seq_len, 7, device=device),
        }

        monthly_masks = {
            name: torch.rand(batch_size, monthly_seq_len, feat.shape[-1], device=device) > 0.3
            for name, feat in monthly_features.items()
        }

        # Month boundaries
        days_per_month = daily_seq_len // monthly_seq_len
        month_boundaries = torch.zeros(
            batch_size, monthly_seq_len, 2, dtype=torch.long, device=device
        )
        for m in range(monthly_seq_len):
            month_boundaries[:, m, 0] = m * days_per_month
            month_boundaries[:, m, 1] = min((m + 1) * days_per_month, daily_seq_len)

        # ISW embeddings
        isw_embeddings = torch.randn(batch_size, daily_seq_len, 1024, device=device)
        isw_mask = torch.rand(batch_size, daily_seq_len, device=device) > 0.2

        # Dates
        from datetime import datetime, timedelta
        base_date = datetime(2022, 3, 1)
        dates = [
            [(base_date + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(daily_seq_len)]
            for _ in range(batch_size)
        ]

        print(f"Daily features: {list(daily_features.keys())}")
        print(f"Monthly features: {list(monthly_features.keys())}")
        print(f"ISW embeddings: {isw_embeddings.shape}")
        print(f"Dates: {len(dates[0])} per batch")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

    # =========================================================================
    # TEST 3: Forward Pass (Full)
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 3: Forward Pass (Full Integration)")
    print("-" * 40)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
                isw_embeddings=isw_embeddings,
                isw_mask=isw_mask,
                dates=dates,
                compute_aux_losses=True,
            )

        print(f"Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # =========================================================================
    # TEST 4: Forward Pass (ISW only)
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 4: Forward Pass (ISW only)")
    print("-" * 40)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
                isw_embeddings=isw_embeddings,
                isw_mask=isw_mask,
                dates=None,  # No dates
            )

        print(f"Output keys: {list(outputs.keys())}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

    # =========================================================================
    # TEST 5: Forward Pass (Timeline only)
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 5: Forward Pass (Timeline only)")
    print("-" * 40)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
                isw_embeddings=None,  # No ISW
                isw_mask=None,
                dates=dates,
            )

        print(f"Output keys: {list(outputs.keys())}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

    # =========================================================================
    # TEST 6: Forward Pass (No enhancements)
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 6: Forward Pass (No enhancements)")
    print("-" * 40)

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
                isw_embeddings=None,
                isw_mask=None,
                dates=None,
            )

        print(f"Output keys: {list(outputs.keys())}")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

    # =========================================================================
    # TEST 7: Gradient Flow
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 7: Gradient Flow")
    print("-" * 40)

    try:
        model.train()
        outputs = model(
            daily_features=daily_features,
            daily_masks=daily_masks,
            monthly_features=monthly_features,
            monthly_masks=monthly_masks,
            month_boundaries=month_boundaries,
            isw_embeddings=isw_embeddings,
            isw_mask=isw_mask,
            dates=dates,
        )

        # Compute loss
        loss = outputs['casualty_pred'].mean()
        if 'phase_logits' in outputs:
            loss = loss + outputs['phase_logits'].mean()
        if 'aux_losses' in outputs:
            for name, aux_loss in outputs['aux_losses'].items():
                loss = loss + aux_loss

        loss.backward()

        grad_exists = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.parameters() if p.requires_grad)

        print(f"Loss: {loss.item():.4f}")
        print(f"Gradients computed: {grad_exists}")
        print("PASSED" if grad_exists else "FAILED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # =========================================================================
    # TEST 8: Model Config
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 8: Model Configuration")
    print("-" * 40)

    config = model.get_config()
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("PASSED")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nEnhanced Multi-Resolution HAN Features:")
    print("- ISW narrative integration via gated fusion")
    print("- Timeline/operation phase features")
    print("- Phase-aware attention mechanism")
    print("- Operation context retrieval")
    print("- Phase-supervised regime classification")
    print("- Contrastive alignment loss")
    print("- Modular design (features can be enabled/disabled)")
