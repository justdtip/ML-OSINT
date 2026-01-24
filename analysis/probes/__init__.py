"""
Multi-Resolution HAN Model Probe Battery
=========================================

A comprehensive test battery for validating learned representations,
cross-modal fusion quality, and semantic-numerical associations in
the Multi-Resolution Hierarchical Attention Network for Ukraine conflict dynamics.

Test Sections:
--------------
1. Data Artifact Investigation (Sections 1.1-1.3)
   - Equipment signal degradation analysis
   - VIIRS dominance investigation
   - Personnel data quality check

2. Cross-Modal Fusion Validation (Sections 2.1-2.2)
   - Representation similarity analysis (RSA)
   - Cross-source information flow
   - Fusion layer ablation
   - Source contribution analysis

3. Temporal Dynamics Analysis (Sections 3.1-3.2)
   - Context window effects
   - Temporal attention patterns
   - State transition dynamics

4. Semantic Structure Probing (Sections 4.1-4.2)
   - Implicit semantic categories
   - Temporal semantic patterns

5. Semantic-Numerical Association Tests (Sections 5.1-5.4)
   - ISW alignment validation
   - Cross-modal semantic grounding
   - Counterfactual semantic probing

6. Causal Importance Validation (Sections 6.1-6.2)
   - Intervention-based importance
   - Gradient-based causal analysis

7. Tactical Prediction Readiness (Sections 7.1-7.3)
   - Spatial decomposition potential
   - Entity-level readiness
   - Prediction resolution requirements

Priority Tiers:
---------------
Tier 1 (Critical - Run First):
    1. VIIRS-Casualty Temporal Relationship (1.2.1)
    2. Equipment-Personnel Redundancy Test (1.1.2)
    3. Source Zeroing Interventions (6.1.1)
    4. Named Operation Clustering (4.1.1)
    5. ISW-Latent Correlation (5.1.1)

Tier 2 (Important):
    6. Trend Confounding Test (1.2.3)
    7. Leave-One-Out Ablation (2.2.1)
    8. Day-Type Decoding Probe (4.1.2)
    9. Event-Triggered Response Analysis (5.2.1)
    10. Truncated Context Inference (3.1.1)

Tier 3 (Exploratory):
    All remaining tests

Usage:
------
    # Run all probes
    from analysis.probes import MasterProbeRunner

    runner = MasterProbeRunner(
        model_path="checkpoints/multi_resolution/checkpoint_epoch_99.pt",
        data_config=config,
        output_dir="probes/outputs"
    )
    results = runner.run_all()
    runner.generate_report()

    # Run specific tier
    results = runner.run_tier(1)  # Critical probes only

    # Run specific section
    from analysis.probes import DataArtifactProbeSuite
    suite = DataArtifactProbeSuite()
    results = suite.run_all()

Author: ML Engineering Team
Date: 2026-01-23
Version: 1.0
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# ============================================================================
# Section 1: Data Artifact Investigation
# ============================================================================
try:
    from .data_artifact_probes import (
        # Base classes
        Probe as DataArtifactProbe,
        ProbeResult as DataArtifactResult,

        # Section 1.1: Equipment Signal Degradation
        EncodingVarianceComparisonProbe as EncodingVarianceProbe,
        EquipmentPersonnelRedundancyProbe,
        EquipmentCategoryDisaggregationProbe,
        TemporalLagAnalysisProbe as EquipmentTemporalLagProbe,

        # Section 1.2: VIIRS Dominance
        VIIRSCasualtyTemporalProbe,
        VIIRSFeatureDecompositionProbe,
        TrendConfoundingProbe,
        GeographicVIIRSDecompositionProbe as GeographicVIIRSProbe,

        # Section 1.3: Personnel Quality
        PersonnelVIIRSMediationProbe,

        # Suite
        DataArtifactProbeSuite,
    )
    _DATA_ARTIFACT_AVAILABLE = True
except ImportError as e:
    _DATA_ARTIFACT_AVAILABLE = False
    warnings.warn(f"Data artifact probes not available: {e}")

# ============================================================================
# Section 1.4-1.5: Statistical Analysis Probes
# ============================================================================
try:
    from .statistical_analysis_probes import (
        # Section 1.4: Statistical Correlation Analysis
        MultiVariableCorrelationProbe,
        SeasonalPatternProbe,

        # Section 1.5: Neural Pattern Discovery
        NeuralPatternMiningProbe,
    )
    _STATISTICAL_AVAILABLE = True
except ImportError as e:
    _STATISTICAL_AVAILABLE = False
    warnings.warn(f"Statistical analysis probes not available: {e}")

# ============================================================================
# Section 2: Cross-Modal Fusion Validation
# ============================================================================
try:
    from .cross_modal_fusion_probes import (
        # Configs
        RSAProbeConfig,
        AttentionFlowProbeConfig,
        AblationProbeConfig,
        CheckpointProbeConfig,

        # Section 2.1: Fusion Quality Metrics
        RSAProbe,
        AttentionFlowProbe,
        AblationProbe,
        CheckpointComparisonProbe,

        # Utilities
        IntermediateRepresentationHook,
        SourceRepresentationExtractor,

        # Runner
        CrossModalFusionProbeRunner,
        run_fusion_quality_probes,
    )
    _CROSS_MODAL_AVAILABLE = True
except ImportError as e:
    _CROSS_MODAL_AVAILABLE = False
    warnings.warn(f"Cross-modal fusion probes not available: {e}")

# ============================================================================
# Section 3: Temporal Dynamics Analysis
# ============================================================================
try:
    from .temporal_dynamics_probes import (
        # Base classes
        BaseProbe,
        ProbeResult,

        # Section 3.1: Context Window Effects
        ContextWindowProbe,
        AttentionDistanceProbe,
        PredictiveHorizonProbe,

        # Section 3.2: State Transition Dynamics
        TransitionDynamicsProbe,
        LatentVelocityProbe,

        # Runner
        TemporalDynamicsProbeRunner,

        # Utilities
        date_to_phase,
        get_days_to_transition,
        create_truncated_batch,
        extract_latent_representations,
        extract_attention_weights,

        # Constants
        REGIME_TRANSITIONS,
        PHASE_LABELS,
        PHASE_COLORS,
        CONTEXT_LENGTHS,
        PREDICTION_HORIZONS,
        TRANSITION_WINDOW_DAYS,
    )
    _TEMPORAL_AVAILABLE = True
except ImportError as e:
    _TEMPORAL_AVAILABLE = False
    warnings.warn(f"Temporal dynamics probes not available: {e}")

# ============================================================================
# Section 4: Semantic Structure Probing
# ============================================================================
try:
    from .semantic_structure_probes import (
        # Section 4.1: Implicit Semantic Categories
        OperationClusteringProbe,
        LinearDecoder,
        DayTypeDecodingProbe,
        IntensityProbe,
        GeographicFocusProbe,

        # Section 4.2: Temporal Patterns
        TemporalPatternProbe,

        # Label construction
        LabelConstructor,

        # Runner
        SemanticStructureProbeRunner,

        # Constants
        MILITARY_OPERATIONS,
    )
    _SEMANTIC_STRUCTURE_AVAILABLE = True
except ImportError as e:
    _SEMANTIC_STRUCTURE_AVAILABLE = False
    warnings.warn(f"Semantic structure probes not available: {e}")

# ============================================================================
# Section 5: Semantic-Numerical Association Tests
# ============================================================================
try:
    from .semantic_association_probes import (
        # Data
        ISWEmbeddingData,

        # Section 5.1: ISW Alignment
        ISWAlignmentProbe,
        TopicExtractionProbe,
        ISWPredictiveContentProbe,

        # Section 5.2: Cross-Modal Grounding
        EventResponseProbe,
        LagAnalysisProbe,
        SemanticAnomalyProbe,

        # Section 5.3: Counterfactual
        CounterfactualProbe,
        SemanticPredictorProbe,

        # Runner
        SemanticAssociationProbeRunner,

        # Constants
        SEMANTIC_ENRICHMENT_SPEC,
        MAJOR_EVENTS as KEY_EVENTS,
    )
    _SEMANTIC_ASSOC_AVAILABLE = True
except ImportError as e:
    _SEMANTIC_ASSOC_AVAILABLE = False
    warnings.warn(f"Semantic association probes not available: {e}")

# ============================================================================
# Section 6: Causal Importance Validation
# ============================================================================
try:
    from .causal_importance_probes import (
        # Results
        InterventionResult,
        GradientAttributionResult,
        AttentionFlowResult,
        CausalRanking,

        # Section 6.1: Intervention-Based
        ZeroingInterventionProbe,
        ShufflingInterventionProbe,
        MeanSubstitutionProbe,

        # Section 6.2: Gradient-Based
        IntegratedGradientsProbe,
        AttentionKnockoutProbe,

        # Report
        CausalImportanceReport,
    )
    _CAUSAL_AVAILABLE = True
except ImportError as e:
    _CAUSAL_AVAILABLE = False
    warnings.warn(f"Causal importance probes not available: {e}")

# ============================================================================
# Section 7: Tactical Prediction Readiness
# ============================================================================
try:
    from .tactical_readiness_probes import (
        # Section 7.1: Spatial Decomposition
        DataAvailabilityAudit,
        SectorDefinition,
        SectorCorrelationProbe,

        # Section 7.2: Entity-Level
        EntitySchemaSpec,

        # Section 7.3: Resolution Requirements
        ResolutionAnalysisProbe,

        # Runner (alias TacticalReadinessProbe)
        TacticalReadinessProbe as TacticalReadinessProbeRunner,

        # Constants
        OBLAST_BBOXES as UKRAINIAN_OBLASTS,
        TACTICAL_SECTORS,
    )
    _TACTICAL_AVAILABLE = True
except ImportError as e:
    _TACTICAL_AVAILABLE = False
    warnings.warn(f"Tactical readiness probes not available: {e}")


# ============================================================================
# Module Availability Check
# ============================================================================
def get_available_modules() -> Dict[str, bool]:
    """Check which probe modules are available."""
    return {
        "data_artifact": _DATA_ARTIFACT_AVAILABLE,
        "statistical_analysis": _STATISTICAL_AVAILABLE,
        "model_interpretability": _MODEL_INTERP_AVAILABLE,
        "model_assessment": _MODEL_ASSESSMENT_AVAILABLE,
        "cross_modal_fusion": _CROSS_MODAL_AVAILABLE,
        "temporal_dynamics": _TEMPORAL_AVAILABLE,
        "semantic_structure": _SEMANTIC_STRUCTURE_AVAILABLE,
        "semantic_association": _SEMANTIC_ASSOC_AVAILABLE,
        "causal_importance": _CAUSAL_AVAILABLE,
        "tactical_readiness": _TACTICAL_AVAILABLE,
    }


def print_availability_report():
    """Print availability status of all probe modules."""
    modules = get_available_modules()
    print("\n" + "=" * 60)
    print("Multi-Resolution HAN Probe Battery - Module Status")
    print("=" * 60)
    for module, available in modules.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {module:25s}: {status}")
    print("=" * 60 + "\n")


# ============================================================================
# Tier Definitions
# ============================================================================
TIER_1_PROBES = [
    ("1.2.1", "VIIRS-Casualty Temporal Relationship", "VIIRSCasualtyTemporalProbe"),
    ("1.1.2", "Equipment-Personnel Redundancy Test", "EquipmentPersonnelRedundancyProbe"),
    ("6.1.1", "Source Zeroing Interventions", "ZeroingInterventionProbe"),
    ("4.1.1", "Named Operation Clustering", "OperationClusteringProbe"),
    ("5.1.1", "ISW-Latent Correlation", "ISWAlignmentProbe"),
]

TIER_2_PROBES = [
    ("1.2.3", "Trend Confounding Test", "TrendConfoundingProbe"),
    ("2.2.1", "Leave-One-Out Ablation", "AblationProbe"),
    ("4.1.2", "Day-Type Decoding Probe", "DayTypeDecodingProbe"),
    ("5.2.1", "Event-Triggered Response Analysis", "EventResponseProbe"),
    ("3.1.1", "Truncated Context Inference", "ContextWindowProbe"),
]

TIER_3_PROBES = [
    # Remaining data artifact probes
    ("1.1.1", "Encoding Variance Comparison", "EncodingVarianceProbe"),
    ("1.1.3", "Equipment Category Disaggregation", "EquipmentCategoryDisaggregationProbe"),
    ("1.1.4", "Temporal Lag Analysis - Equipment", "EquipmentTemporalLagProbe"),
    ("1.2.2", "VIIRS Feature Decomposition", "VIIRSFeatureDecompositionProbe"),
    ("1.2.4", "Geographic VIIRS Decomposition", "GeographicVIIRSProbe"),
    ("1.3.1", "Personnel-VIIRS Mediation Analysis", "PersonnelVIIRSMediationProbe"),
    # Fusion probes
    ("2.1.1", "Representation Similarity Analysis", "RSAProbe"),
    ("2.1.2", "Cross-Source Information Flow", "AttentionFlowProbe"),
    ("2.1.4", "Checkpoint Comparison", "CheckpointComparisonProbe"),
    ("2.2.2", "Source Sufficiency Test", "AblationProbe"),
    # Temporal probes
    ("3.1.2", "Temporal Attention Patterns", "AttentionDistanceProbe"),
    ("3.1.3", "Predictive Horizon Analysis", "PredictiveHorizonProbe"),
    ("3.2.1", "Transition Boundary Analysis", "TransitionDynamicsProbe"),
    ("3.2.2", "Latent Velocity Prediction", "LatentVelocityProbe"),
    # Semantic structure probes
    ("4.1.3", "Intensity Level Decoding", "IntensityProbe"),
    ("4.1.4", "Geographic Focus Decoding", "GeographicFocusProbe"),
    ("4.2.1", "Weekly Cycle Detection", "TemporalPatternProbe"),
    ("4.2.2", "Seasonal Pattern Detection", "TemporalPatternProbe"),
    ("4.2.3", "Event Anniversary Detection", "TemporalPatternProbe"),
    # Semantic association probes
    ("5.1.2", "ISW Topic-Source Correlation", "TopicExtractionProbe"),
    ("5.1.3", "ISW Predictive Content Test", "ISWPredictiveContentProbe"),
    ("5.2.2", "Narrative-Numerical Lag Analysis", "LagAnalysisProbe"),
    ("5.2.3", "Semantic Anomaly Detection", "SemanticAnomalyProbe"),
    ("5.3.1", "Semantic Perturbation Effects", "CounterfactualProbe"),
    ("5.3.2", "Missing Semantic Interpolation", "SemanticPredictorProbe"),
    # Causal probes
    ("6.1.2", "Source Shuffling Interventions", "ShufflingInterventionProbe"),
    ("6.1.3", "Source Mean Substitution", "MeanSubstitutionProbe"),
    ("6.2.1", "Integrated Gradients", "IntegratedGradientsProbe"),
    ("6.2.2", "Attention Knockout", "AttentionKnockoutProbe"),
    # Tactical probes
    ("7.1.1", "Regional Signal Availability", "DataAvailabilityAudit"),
    ("7.1.2", "Front-Line Sector Definition", "SectorDefinition"),
    ("7.1.3", "Sector Independence Test", "SectorCorrelationProbe"),
    ("7.2.1", "Unit Tracking Data Availability", "EntitySchemaSpec"),
    ("7.2.2", "Entity State Representation Design", "EntitySchemaSpec"),
    ("7.3.1", "Temporal Resolution Analysis", "ResolutionAnalysisProbe"),
    ("7.3.2", "Spatial Resolution Analysis", "ResolutionAnalysisProbe"),
    # Statistical analysis probes (Section 1.4-1.5)
    ("1.4.1", "Multi-Variable Correlation Analysis", "MultiVariableCorrelationProbe"),
    ("1.4.2", "Seasonal Pattern Analysis", "SeasonalPatternProbe"),
    ("1.5.1", "Neural Pattern Mining", "NeuralPatternMiningProbe"),
    # Model interpretability probes (Section 2.3-2.4)
    ("2.3.1", "JIM Module I/O Analysis", "JIMModuleIOProbe"),
    ("2.3.2", "JIM Attention Pattern Analysis", "JIMAttentionAnalysisProbe"),
    ("2.4.1", "Cross-Source Latent Analysis", "CrossSourceLatentProbe"),
    ("2.4.2", "Delta Model Validation", "DeltaModelValidationProbe"),
    # Model assessment probes (Section 8)
    ("8.1.1", "Model Architecture Comparison", "ModelArchitectureComparisonProbe"),
    ("8.1.2", "Reconstruction Performance Comparison", "ReconstructionPerformanceProbe"),
    ("8.2.1", "Multi-Task Performance Assessment", "MultiTaskPerformanceProbe"),
    ("8.2.2", "Training Dynamics Analysis", "TrainingDynamicsProbe"),
]


# ============================================================================
# __all__ Export List
# ============================================================================
__all__ = [
    # Availability utilities
    "get_available_modules",
    "print_availability_report",

    # Tier definitions
    "TIER_1_PROBES",
    "TIER_2_PROBES",
    "TIER_3_PROBES",
]

# Add exports conditionally based on module availability
if _DATA_ARTIFACT_AVAILABLE:
    __all__.extend([
        "DataArtifactProbe",
        "DataArtifactResult",
        "EncodingVarianceProbe",
        "EquipmentPersonnelRedundancyProbe",
        "EquipmentCategoryDisaggregationProbe",
        "EquipmentTemporalLagProbe",
        "VIIRSCasualtyTemporalProbe",
        "VIIRSFeatureDecompositionProbe",
        "TrendConfoundingProbe",
        "GeographicVIIRSProbe",
        "PersonnelVIIRSMediationProbe",
        "DataArtifactProbeSuite",
    ])

if _STATISTICAL_AVAILABLE:
    __all__.extend([
        "MultiVariableCorrelationProbe",
        "SeasonalPatternProbe",
        "NeuralPatternMiningProbe",
    ])

# ============================================================================
# Section 2.3-2.4: Model Interpretability Probes
# ============================================================================
try:
    from .model_interpretability_probes import (
        # Section 2.3: JIM Interpretability
        JIMModuleIOProbe,
        JIMAttentionAnalysisProbe,

        # Section 2.4: Unified Model Validation
        CrossSourceLatentProbe,
        DeltaModelValidationProbe,
    )
    _MODEL_INTERP_AVAILABLE = True
except ImportError as e:
    _MODEL_INTERP_AVAILABLE = False
    warnings.warn(f"Model interpretability probes not available: {e}")

if _MODEL_INTERP_AVAILABLE:
    __all__.extend([
        "JIMModuleIOProbe",
        "JIMAttentionAnalysisProbe",
        "CrossSourceLatentProbe",
        "DeltaModelValidationProbe",
    ])

# ============================================================================
# Section 8: Model Assessment Probes
# ============================================================================
try:
    from .model_assessment_probes import (
        # Section 8.1: Cross-Model Comparison
        ModelArchitectureComparisonProbe,
        ReconstructionPerformanceProbe,

        # Section 8.2: HAN Assessment
        MultiTaskPerformanceProbe,
        TrainingDynamicsProbe,
    )
    _MODEL_ASSESSMENT_AVAILABLE = True
except ImportError as e:
    _MODEL_ASSESSMENT_AVAILABLE = False
    warnings.warn(f"Model assessment probes not available: {e}")

if _MODEL_ASSESSMENT_AVAILABLE:
    __all__.extend([
        "ModelArchitectureComparisonProbe",
        "ReconstructionPerformanceProbe",
        "MultiTaskPerformanceProbe",
        "TrainingDynamicsProbe",
    ])

if _CROSS_MODAL_AVAILABLE:
    __all__.extend([
        "RSAProbeConfig",
        "AttentionFlowProbeConfig",
        "AblationProbeConfig",
        "CheckpointProbeConfig",
        "RSAProbe",
        "AttentionFlowProbe",
        "AblationProbe",
        "CheckpointComparisonProbe",
        "IntermediateRepresentationHook",
        "SourceRepresentationExtractor",
        "CrossModalFusionProbeRunner",
        "run_fusion_quality_probes",
    ])

if _TEMPORAL_AVAILABLE:
    __all__.extend([
        "BaseProbe",
        "ProbeResult",
        "ContextWindowProbe",
        "AttentionDistanceProbe",
        "PredictiveHorizonProbe",
        "TransitionDynamicsProbe",
        "LatentVelocityProbe",
        "TemporalDynamicsProbeRunner",
        "date_to_phase",
        "get_days_to_transition",
        "create_truncated_batch",
        "extract_latent_representations",
        "extract_attention_weights",
        "REGIME_TRANSITIONS",
        "PHASE_LABELS",
        "PHASE_COLORS",
        "CONTEXT_LENGTHS",
        "PREDICTION_HORIZONS",
        "TRANSITION_WINDOW_DAYS",
    ])

if _SEMANTIC_STRUCTURE_AVAILABLE:
    __all__.extend([
        "OperationClusteringProbe",
        "LinearDecoder",
        "DayTypeDecodingProbe",
        "IntensityProbe",
        "GeographicFocusProbe",
        "TemporalPatternProbe",
        "LabelConstructor",
        "SemanticStructureProbeRunner",
        "MILITARY_OPERATIONS",
    ])

if _SEMANTIC_ASSOC_AVAILABLE:
    __all__.extend([
        "ISWEmbeddingData",
        "ISWAlignmentProbe",
        "TopicExtractionProbe",
        "ISWPredictiveContentProbe",
        "EventResponseProbe",
        "LagAnalysisProbe",
        "SemanticAnomalyProbe",
        "CounterfactualProbe",
        "SemanticPredictorProbe",
        "SemanticAssociationProbeRunner",
        "SEMANTIC_ENRICHMENT_SPEC",
        "KEY_EVENTS",
    ])

if _CAUSAL_AVAILABLE:
    __all__.extend([
        "InterventionResult",
        "GradientAttributionResult",
        "AttentionFlowResult",
        "CausalRanking",
        "ZeroingInterventionProbe",
        "ShufflingInterventionProbe",
        "MeanSubstitutionProbe",
        "IntegratedGradientsProbe",
        "AttentionKnockoutProbe",
        "CausalImportanceReport",
    ])

if _TACTICAL_AVAILABLE:
    __all__.extend([
        "DataAvailabilityAudit",
        "SectorDefinition",
        "SectorCorrelationProbe",
        "EntitySchemaSpec",
        "ResolutionAnalysisProbe",
        "TacticalReadinessProbeRunner",
        "UKRAINIAN_OBLASTS",
        "TACTICAL_SECTORS",
    ])
