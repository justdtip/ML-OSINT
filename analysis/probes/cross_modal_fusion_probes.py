"""
Cross-Modal Fusion Quality Probes for Multi-Resolution HAN

This module implements a comprehensive test battery for validating cross-modal fusion
quality in the MultiResolutionHAN model. It addresses concerns about surprisingly low
cross-source correlations (most < 0.1) by providing tools to:

1. Measure representation similarity via RSA (Representational Similarity Analysis)
2. Analyze cross-source attention flow patterns
3. Perform ablation studies (leave-one-out, sufficiency tests)
4. Track fusion quality evolution across training checkpoints

Expected Results:
- If model is fusing: RSA > 0.3 for related sources (e.g., equipment-personnel)
- If sources independent: RSA near zero for all pairs

Architecture Context:
- 11 sources: 5 daily (equipment, personnel, deepstate, firms, viina) +
              5 monthly (sentinel, hdx_conflict, hdx_food, hdx_rainfall, iom) + isw
- DailyCrossSourceFusion: cross-source attention with source importance gating
- CrossResolutionFusion: bidirectional attention between daily and monthly
- Latent dimension: 128

Author: ML Engineering Team
Date: 2026-01-23
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

# Centralized path configuration
from config.paths import get_probe_figures_dir, get_probe_metrics_dir

# Task key mapping for consistent task/output key resolution
from .task_key_mapping import (
    TASK_OUTPUT_KEYS,
    get_output_key,
    extract_task_output,
    has_task_output,
)

# Import centralized batch preparation that filters training-only keys
from . import (
    prepare_batch_for_model,
    TRAINING_ONLY_KEYS,
    MODEL_FORWARD_EXCLUDE_KEYS,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_output_dir():
    """Get current output directory for figures."""
    return get_probe_figures_dir()


# Default output directory (set dynamically)
DEFAULT_OUTPUT_DIR = None  # Use get_output_dir() instead


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ProbeConfig:
    """Base configuration for all probes."""
    model: nn.Module
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = None  # Set dynamically in __post_init__
    random_seed: int = 42

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = get_output_dir()
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


@dataclass
class RSAProbeConfig(ProbeConfig):
    """Configuration for Representational Similarity Analysis probe."""
    distance_metric: str = "correlation"  # 'correlation', 'cosine', 'euclidean'
    comparison_metric: str = "spearman"   # 'spearman', 'pearson'
    min_samples: int = 100  # Minimum samples for reliable RSA
    subsample_size: Optional[int] = 1000  # Subsample for efficiency


@dataclass
class AttentionFlowProbeConfig(ProbeConfig):
    """Configuration for attention flow analysis probe."""
    attention_threshold: float = 0.1  # Minimum attention weight to consider
    normalize_by_source: bool = True  # Normalize attention per source
    temporal_windows: List[str] = field(
        default_factory=lambda: ["early", "mid", "late"]
    )


@dataclass
class AblationProbeConfig(ProbeConfig):
    """Configuration for ablation studies.

    Task names can be either simple names ('casualty', 'regime', 'anomaly', 'forecast')
    or model output keys ('casualty_pred', 'regime_logits', 'anomaly_score', 'forecast_pred').
    The probe automatically resolves to the correct output key using the centralized
    task_key_mapping module.

    The MultiResolutionHAN model outputs:
      - 'casualty_pred': Casualty prediction tensor
      - 'regime_logits': Regime classification logits
      - 'anomaly_score': Anomaly detection scores
      - 'forecast_pred': Forecast prediction tensor
    """
    ablation_mode: str = "zero"  # 'zero', 'mean', 'noise', 'shuffle'
    tasks: List[str] = field(
        # Use simple task names - they'll be resolved to output keys automatically
        default_factory=lambda: ["casualty", "regime", "anomaly"]
    )
    num_samples: int = 500  # Number of test samples


@dataclass
class CheckpointProbeConfig(ProbeConfig):
    """Configuration for checkpoint comparison probe."""
    checkpoint_epochs: List[int] = field(
        default_factory=lambda: [10, 25, 50, 75, 100]
    )
    checkpoint_dir: Optional[Path] = None
    metrics_to_track: List[str] = field(
        default_factory=lambda: ["rsa", "attention_entropy", "source_importance"]
    )


@dataclass
class ProbeResult:
    """Container for probe results."""
    probe_name: str
    timestamp: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, Path]  # Paths to generated figures/files
    interpretation: str
    recommendations: List[str]


# =============================================================================
# HOOK UTILITIES FOR INTERMEDIATE REPRESENTATION EXTRACTION
# =============================================================================

class IntermediateRepresentationHook:
    """Hook for extracting intermediate representations from model layers.

    Uses PyTorch forward hooks to capture layer outputs during inference.
    Thread-safe and supports multiple simultaneous extractions.
    """

    def __init__(self):
        self.representations: Dict[str, Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def register_hook(
        self,
        module: nn.Module,
        name: str,
        transform: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:
        """Register a forward hook on a module.

        Args:
            module: PyTorch module to hook
            name: Key for storing the representation
            transform: Optional transform to apply to output
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]  # Take first element if tuple
            if transform is not None:
                output = transform(output)
            self.representations[name] = output.detach().clone()

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def register_attention_hook(
        self,
        attention_module: nn.Module,
        name: str
    ) -> None:
        """Register hook specifically for attention weights.

        Args:
            attention_module: Attention module (e.g., nn.MultiheadAttention)
            name: Key for storing attention weights
        """
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # MultiheadAttention returns (attn_output, attn_weights)
                attn_weights = output[1]
            else:
                attn_weights = output
            self.representations[name] = attn_weights.detach().clone()

        handle = attention_module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def get_representation(self, name: str) -> Optional[Tensor]:
        """Get stored representation by name."""
        return self.representations.get(name)

    def clear(self) -> None:
        """Clear stored representations."""
        self.representations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        self.clear()


class SourceRepresentationExtractor:
    """Extracts per-source representations from MultiResolutionHAN model.

    This extractor registers hooks on the appropriate encoder layers to capture
    individual source representations before and after fusion.

    Source names are detected dynamically from the model's encoder dictionaries,
    supporting both aggregated (6 daily sources) and disaggregated (8 daily sources)
    configurations.
    """

    # Default source names (used when model encoders are not available)
    # These are overridden by actual encoder keys when the model is available
    DEFAULT_DAILY_SOURCES = ['equipment', 'personnel', 'deepstate', 'firms', 'viina', 'viirs']
    DEFAULT_MONTHLY_SOURCES = ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']

    # Class-level attributes for backwards compatibility (updated by instances)
    DAILY_SOURCES = DEFAULT_DAILY_SOURCES.copy()
    MONTHLY_SOURCES = DEFAULT_MONTHLY_SOURCES.copy()
    ALL_SOURCES = DAILY_SOURCES + MONTHLY_SOURCES

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_manager = IntermediateRepresentationHook()

        # Dynamically detect sources from model encoders
        self._daily_sources = self._detect_daily_sources()
        self._monthly_sources = self._detect_monthly_sources()
        self._all_sources = self._daily_sources + self._monthly_sources

        # Update class-level attributes for backwards compatibility
        SourceRepresentationExtractor.DAILY_SOURCES = self._daily_sources
        SourceRepresentationExtractor.MONTHLY_SOURCES = self._monthly_sources
        SourceRepresentationExtractor.ALL_SOURCES = self._all_sources

        logger.info(f"Detected daily sources: {self._daily_sources}")
        logger.info(f"Detected monthly sources: {self._monthly_sources}")

        self._setup_hooks()

    def _detect_daily_sources(self) -> List[str]:
        """Detect daily source names from model's encoder dictionaries."""
        if hasattr(self.model, 'daily_encoders') and self.model.daily_encoders:
            return list(self.model.daily_encoders.keys())
        elif hasattr(self.model, 'daily_source_configs') and self.model.daily_source_configs:
            return list(self.model.daily_source_configs.keys())
        else:
            logger.warning("Could not detect daily sources from model, using defaults")
            return self.DEFAULT_DAILY_SOURCES.copy()

    def _detect_monthly_sources(self) -> List[str]:
        """Detect monthly source names from model's encoder dictionaries."""
        if hasattr(self.model, 'monthly_encoders') and self.model.monthly_encoders:
            return list(self.model.monthly_encoders.keys())
        elif hasattr(self.model, 'monthly_source_configs') and self.model.monthly_source_configs:
            return list(self.model.monthly_source_configs.keys())
        else:
            logger.warning("Could not detect monthly sources from model, using defaults")
            return self.DEFAULT_MONTHLY_SOURCES.copy()

    @classmethod
    def get_sources_from_batch(cls, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract source names from a batch dictionary.

        This is useful when you don't have access to the model but have a data batch.

        Args:
            batch: Batch dictionary with 'daily_features' and 'monthly_features' keys

        Returns:
            Tuple of (daily_sources, monthly_sources) lists
        """
        daily_sources = list(batch.get('daily_features', {}).keys())
        monthly_sources = list(batch.get('monthly_features', {}).keys())
        return daily_sources, monthly_sources

    @classmethod
    def update_sources_from_batch(cls, batch: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Update class-level source lists from a batch and return them.

        Args:
            batch: Batch dictionary with 'daily_features' and 'monthly_features' keys

        Returns:
            Tuple of (daily_sources, monthly_sources) lists
        """
        daily_sources, monthly_sources = cls.get_sources_from_batch(batch)
        if daily_sources:
            cls.DAILY_SOURCES = daily_sources
        if monthly_sources:
            cls.MONTHLY_SOURCES = monthly_sources
        cls.ALL_SOURCES = cls.DAILY_SOURCES + cls.MONTHLY_SOURCES
        logger.info(f"Updated sources from batch - daily: {cls.DAILY_SOURCES}, monthly: {cls.MONTHLY_SOURCES}")
        return daily_sources, monthly_sources

    def _setup_hooks(self) -> None:
        """Set up hooks for all relevant model components."""
        # Hook daily source encoders
        if hasattr(self.model, 'daily_encoders'):
            for name, encoder in self.model.daily_encoders.items():
                self.hook_manager.register_hook(
                    encoder,
                    f"daily_{name}_pre_fusion",
                )

        # Hook monthly source encoders
        if hasattr(self.model, 'monthly_encoders'):
            for name, encoder in self.model.monthly_encoders.items():
                self.hook_manager.register_hook(
                    encoder,
                    f"monthly_{name}_pre_fusion",
                )

        # Hook daily cross-source fusion
        if hasattr(self.model, 'daily_fusion'):
            self.hook_manager.register_hook(
                self.model.daily_fusion,
                "daily_fused",
            )

        # Hook cross-resolution fusion
        if hasattr(self.model, 'cross_resolution_fusion'):
            self.hook_manager.register_hook(
                self.model.cross_resolution_fusion,
                "cross_resolution_fused",
            )

    def extract(
        self,
        batch: Dict[str, Tensor],
        return_attention: bool = True
    ) -> Dict[str, Tensor]:
        """Run forward pass and extract all registered representations.

        Args:
            batch: Input batch dictionary
            return_attention: Whether to request attention weights from model

        Returns:
            Dictionary mapping representation names to tensors
        """
        self.hook_manager.clear()

        # Prepare batch for model (map collate keys to model parameter names)
        model_batch = prepare_batch_for_model(batch)

        # Run forward pass
        with torch.no_grad():
            if return_attention and hasattr(self.model, 'forward'):
                try:
                    _ = self.model(**model_batch, return_attention=return_attention)
                except TypeError:
                    _ = self.model(**model_batch)
            else:
                _ = self.model(**model_batch)

        return self.hook_manager.representations.copy()

    def cleanup(self) -> None:
        """Remove all hooks and clear representations."""
        self.hook_manager.remove_hooks()
        self.hook_manager.clear()


# =============================================================================
# BASE PROBE CLASS
# =============================================================================

class FusionProbe(ABC):
    """Abstract base class for fusion quality probes."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model = config.model.to(config.device)
        self.model.eval()
        self.results: Optional[ProbeResult] = None

    @abstractmethod
    def run(self, data_loader: Any) -> ProbeResult:
        """Run the probe on provided data."""
        pass

    @abstractmethod
    def visualize(self) -> Dict[str, Path]:
        """Generate visualizations from probe results."""
        pass

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save probe results to JSON."""
        import json

        if self.results is None:
            raise ValueError("No results to save. Run the probe first.")

        if filename is None:
            filename = f"{self.results.probe_name}_{self.results.timestamp}.json"

        filepath = self.config.output_dir / filename

        # Convert results to serializable format
        results_dict = {
            "probe_name": self.results.probe_name,
            "timestamp": self.results.timestamp,
            "metrics": self._convert_to_serializable(self.results.metrics),
            "artifacts": {k: str(v) for k, v in self.results.artifacts.items()},
            "interpretation": self.results.interpretation,
            "recommendations": self.results.recommendations,
        }

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert numpy/torch types to JSON-serializable types."""
        if isinstance(obj, (np.ndarray, Tensor)):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: FusionProbe._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [FusionProbe._convert_to_serializable(v) for v in obj]
        return obj


# =============================================================================
# RSA PROBE: Representational Similarity Analysis
# =============================================================================

class RSAProbe(FusionProbe):
    """
    Representational Similarity Analysis (RSA) Probe for fusion quality.

    RSA measures second-order similarity between representations by:
    1. Computing Representational Dissimilarity Matrices (RDMs) per source
    2. Measuring correlation between RDMs across sources

    High RSA between sources indicates they have similar representational geometry,
    suggesting successful information fusion. Low RSA suggests independent processing.

    Reference: Kriegeskorte et al. (2008) "Representational similarity analysis"

    Usage:
        >>> config = RSAProbeConfig(model=model)
        >>> probe = RSAProbe(config)
        >>> results = probe.run(data_loader)
        >>> probe.visualize()
    """

    def __init__(self, config: RSAProbeConfig):
        super().__init__(config)
        self.config: RSAProbeConfig = config
        self.extractor = SourceRepresentationExtractor(self.model)
        self.rdms: Dict[str, np.ndarray] = {}
        self.rsa_matrix: Optional[np.ndarray] = None

    def compute_rdm(
        self,
        representations: Tensor,
        metric: str = "correlation"
    ) -> np.ndarray:
        """Compute Representational Dissimilarity Matrix.

        Args:
            representations: Tensor of shape (n_samples, latent_dim)
            metric: Distance metric ('correlation', 'cosine', 'euclidean')

        Returns:
            RDM as symmetric matrix of shape (n_samples, n_samples)
        """
        if isinstance(representations, Tensor):
            representations = representations.cpu().numpy()

        # Handle batched representations
        if representations.ndim == 3:
            # (batch, seq, dim) -> (batch * seq, dim)
            representations = representations.reshape(-1, representations.shape[-1])

        # Guard: Check for zero-size array
        n_samples = representations.shape[0]
        if n_samples < 2:
            logger.warning(f"Cannot compute RDM with {n_samples} samples (need at least 2)")
            return np.zeros((max(1, n_samples), max(1, n_samples)))

        # Subsample for efficiency if needed
        if self.config.subsample_size and n_samples > self.config.subsample_size:
            indices = np.random.choice(n_samples, self.config.subsample_size, replace=False)
            representations = representations[indices]
            n_samples = representations.shape[0]

        # Compute pairwise distances
        try:
            if metric == "correlation":
                # Correlation distance = 1 - correlation
                distances = pdist(representations, metric='correlation')
            elif metric == "cosine":
                distances = pdist(representations, metric='cosine')
            elif metric == "euclidean":
                distances = pdist(representations, metric='euclidean')
                # Normalize to [0, 1]
                distances = distances / (distances.max() + 1e-8)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Convert to square matrix
            rdm = squareform(distances)
        except ValueError as e:
            # Handle edge cases in distance computation
            logger.warning(f"Error computing RDM: {e}. Returning zeros.")
            rdm = np.zeros((n_samples, n_samples))

        return rdm

    def compute_rsa_correlation(
        self,
        rdm1: np.ndarray,
        rdm2: np.ndarray,
        method: str = "spearman"
    ) -> Tuple[float, float]:
        """Compute RSA correlation between two RDMs.

        Args:
            rdm1: First RDM
            rdm2: Second RDM
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Tuple of (correlation, p-value)
        """
        # Extract upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(rdm1.shape[0], k=1)
        vec1 = rdm1[triu_idx]
        vec2 = rdm2[triu_idx]

        if method == "spearman":
            corr, pval = spearmanr(vec1, vec2)
        elif method == "pearson":
            corr, pval = pearsonr(vec1, vec2)
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr, pval

    def run(self, data_loader: Any) -> ProbeResult:
        """Run RSA probe on data.

        Args:
            data_loader: PyTorch DataLoader providing batches

        Returns:
            ProbeResult with RSA metrics and interpretation
        """
        logger.info("Running RSA Probe...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collect representations from all batches
        all_representations: Dict[str, List[Tensor]] = {}

        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

            # Extract representations
            representations = self.extractor.extract(batch)

            for name, repr_tensor in representations.items():
                if name not in all_representations:
                    all_representations[name] = []
                all_representations[name].append(repr_tensor.cpu())

            if batch_idx >= 50:  # Limit for efficiency
                break

        # Concatenate representations
        source_representations = {}
        for name, repr_list in all_representations.items():
            concatenated = torch.cat(repr_list, dim=0)
            # Flatten to (n_samples, dim)
            if concatenated.ndim == 3:
                concatenated = concatenated.mean(dim=1)  # Pool over sequence
            source_representations[name] = concatenated

        # Compute RDMs
        logger.info("Computing RDMs...")
        for name, repr_tensor in source_representations.items():
            if repr_tensor.shape[0] >= self.config.min_samples:
                self.rdms[name] = self.compute_rdm(
                    repr_tensor,
                    metric=self.config.distance_metric
                )
                logger.info(f"  Computed RDM for {name}: shape {self.rdms[name].shape}")

        # Compute RSA matrix
        logger.info("Computing RSA correlations...")
        source_names = list(self.rdms.keys())
        n_sources = len(source_names)

        # Guard: Check if we have enough sources for RSA analysis
        if n_sources < 2:
            logger.warning(f"Insufficient sources for RSA analysis: found {n_sources} sources, need at least 2")
            self.rsa_matrix = np.zeros((max(1, n_sources), max(1, n_sources)))
            metrics = {
                "rsa_matrix": self.rsa_matrix,
                "source_names": source_names,
                "pvalue_matrix": np.zeros_like(self.rsa_matrix),
                "mean_rsa": 0.0,
                "max_rsa": 0.0,
                "min_rsa": 0.0,
                "error": f"Insufficient sources for RSA analysis: {n_sources} sources found, need at least 2",
            }
            interpretation = self._interpret_rsa_results(metrics)
            recommendations = ["Ensure model has multiple source representations available for RSA analysis"]

            self.results = ProbeResult(
                probe_name="RSAProbe",
                timestamp=timestamp,
                metrics=metrics,
                artifacts={},
                interpretation=interpretation,
                recommendations=recommendations,
            )
            return self.results

        self.rsa_matrix = np.zeros((n_sources, n_sources))
        pvalue_matrix = np.zeros((n_sources, n_sources))

        for i, name_i in enumerate(source_names):
            for j, name_j in enumerate(source_names):
                if i <= j:
                    corr, pval = self.compute_rsa_correlation(
                        self.rdms[name_i],
                        self.rdms[name_j],
                        method=self.config.comparison_metric
                    )
                    self.rsa_matrix[i, j] = corr
                    self.rsa_matrix[j, i] = corr
                    pvalue_matrix[i, j] = pval
                    pvalue_matrix[j, i] = pval

        # Analyze results - compute upper triangle statistics safely
        triu_indices = np.triu_indices(n_sources, k=1)
        rsa_upper_triangle = self.rsa_matrix[triu_indices]

        # Guard: Ensure we have values to compute statistics on
        if rsa_upper_triangle.size == 0:
            logger.warning("No off-diagonal RSA values to compute statistics")
            mean_rsa = 0.0
            max_rsa = 0.0
            min_rsa = 0.0
        else:
            mean_rsa = float(np.mean(rsa_upper_triangle))
            max_rsa = float(np.max(rsa_upper_triangle))
            min_rsa = float(np.min(rsa_upper_triangle))

        metrics = {
            "rsa_matrix": self.rsa_matrix,
            "source_names": source_names,
            "pvalue_matrix": pvalue_matrix,
            "mean_rsa": mean_rsa,
            "max_rsa": max_rsa,
            "min_rsa": min_rsa,
        }

        # Identify related source pairs
        related_pairs = self._identify_related_pairs(source_names)
        related_rsa_values = []
        for (s1, s2) in related_pairs:
            if s1 in source_names and s2 in source_names:
                i, j = source_names.index(s1), source_names.index(s2)
                related_rsa_values.append(self.rsa_matrix[i, j])
        metrics["related_pairs_mean_rsa"] = np.mean(related_rsa_values) if related_rsa_values else 0.0

        # Interpretation
        interpretation = self._interpret_rsa_results(metrics)
        recommendations = self._generate_recommendations(metrics)

        self.results = ProbeResult(
            probe_name="RSAProbe",
            timestamp=timestamp,
            metrics=metrics,
            artifacts={},
            interpretation=interpretation,
            recommendations=recommendations,
        )

        # Generate visualizations
        self.results.artifacts = self.visualize()

        return self.results

    def _identify_related_pairs(self, source_names: List[str]) -> List[Tuple[str, str]]:
        """Identify pairs of sources expected to be related."""
        related = [
            ("daily_equipment_pre_fusion", "daily_personnel_pre_fusion"),
            ("daily_firms_pre_fusion", "monthly_sentinel_pre_fusion"),
            ("daily_deepstate_pre_fusion", "daily_viina_pre_fusion"),
        ]
        return related

    def _interpret_rsa_results(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of RSA results."""
        mean_rsa = metrics["mean_rsa"]
        related_rsa = metrics.get("related_pairs_mean_rsa", 0)

        if mean_rsa > 0.3:
            fusion_quality = "STRONG"
            description = "Sources show significant representational similarity, indicating effective cross-modal fusion."
        elif mean_rsa > 0.15:
            fusion_quality = "MODERATE"
            description = "Sources show some representational similarity, but fusion could be improved."
        else:
            fusion_quality = "WEAK"
            description = "Sources show minimal representational similarity, suggesting independent processing."

        interpretation = f"""
RSA Analysis Results:
====================
Fusion Quality: {fusion_quality}

Overall Statistics:
- Mean RSA correlation: {mean_rsa:.4f}
- Max RSA correlation: {metrics['max_rsa']:.4f}
- Min RSA correlation: {metrics['min_rsa']:.4f}
- Related pairs mean RSA: {related_rsa:.4f}

Interpretation:
{description}

Threshold Reference:
- Expected if fusing well: RSA > 0.3 for related sources
- Expected if independent: RSA near zero
"""
        return interpretation

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on RSA results."""
        recommendations = []

        if metrics["mean_rsa"] < 0.15:
            recommendations.append(
                "Consider increasing fusion layer depth or attention heads"
            )
            recommendations.append(
                "Investigate if source-specific encoders are too strong"
            )
            recommendations.append(
                "Check if cross-source attention is being properly utilized"
            )

        if metrics.get("related_pairs_mean_rsa", 0) < 0.2:
            recommendations.append(
                "Related source pairs show low RSA - review source pairing logic"
            )

        if metrics["max_rsa"] > 0.8:
            recommendations.append(
                "Some sources have very high RSA - possible redundancy"
            )

        return recommendations

    def visualize(self) -> Dict[str, Path]:
        """Generate RSA visualizations."""
        if self.rsa_matrix is None:
            raise ValueError("No RSA results to visualize. Run probe first.")

        artifacts = {}

        # RSA heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        source_names = self.results.metrics["source_names"]

        # Create pretty labels
        pretty_names = [name.replace("_pre_fusion", "").replace("daily_", "D:").replace("monthly_", "M:")
                       for name in source_names]

        mask = np.triu(np.ones_like(self.rsa_matrix, dtype=bool), k=1)
        sns.heatmap(
            self.rsa_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            xticklabels=pretty_names,
            yticklabels=pretty_names,
            ax=ax,
            mask=mask,
            square=True,
        )
        ax.set_title("RSA Correlation Matrix: Cross-Modal Representation Similarity", fontsize=14)
        ax.set_xlabel("Source", fontsize=12)
        ax.set_ylabel("Source", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_path = self.config.output_dir / f"rsa_heatmap_{self.results.timestamp}.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts["rsa_heatmap"] = heatmap_path

        # RSA distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        rsa_values = self.rsa_matrix[np.triu_indices(len(source_names), k=1)]
        ax.hist(rsa_values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=0.3, color='g', linestyle='--', label='Expected threshold (0.3)')
        ax.axvline(x=np.mean(rsa_values), color='r', linestyle='-', label=f'Mean ({np.mean(rsa_values):.3f})')
        ax.set_xlabel("RSA Correlation", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of RSA Correlations Across Source Pairs", fontsize=14)
        ax.legend()
        plt.tight_layout()

        dist_path = self.config.output_dir / f"rsa_distribution_{self.results.timestamp}.png"
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts["rsa_distribution"] = dist_path

        logger.info(f"RSA visualizations saved to {self.config.output_dir}")
        return artifacts


# =============================================================================
# ATTENTION FLOW PROBE
# =============================================================================

class AttentionFlowProbe(FusionProbe):
    """
    Probe for analyzing cross-source attention flow patterns.

    This probe extracts and analyzes attention weights from:
    1. DailyCrossSourceFusion layer (source_importance gating)
    2. CrossResolutionFusion layer (bidirectional attention)

    It identifies:
    - Which sources attend to which
    - Dominant information flow patterns
    - Phase-specific attention changes

    Usage:
        >>> config = AttentionFlowProbeConfig(model=model)
        >>> probe = AttentionFlowProbe(config)
        >>> results = probe.run(data_loader)
        >>> probe.visualize()
    """

    def __init__(self, config: AttentionFlowProbeConfig):
        super().__init__(config)
        self.config: AttentionFlowProbeConfig = config
        self.attention_matrices: Dict[str, np.ndarray] = {}
        self.source_importance: Optional[np.ndarray] = None

    def run(self, data_loader: Any) -> ProbeResult:
        """Run attention flow analysis."""
        logger.info("Running Attention Flow Probe...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Storage for attention patterns
        source_importance_list = []
        cross_attn_a_to_b_list = []
        cross_attn_b_to_a_list = []

        # Hook for extracting attention weights
        hook = IntermediateRepresentationHook()

        # Register hooks on fusion layers
        if hasattr(self.model, 'daily_fusion') and hasattr(self.model.daily_fusion, 'source_gate'):
            # Hook the output of source_gate to get importance weights
            def source_gate_hook(module, input, output):
                hook.representations['source_importance'] = output.detach().clone()
            self.model.daily_fusion.source_gate.register_forward_hook(source_gate_hook)

        if hasattr(self.model, 'cross_resolution_fusion'):
            fusion = self.model.cross_resolution_fusion
            # Hook bidirectional attention layers if available
            if hasattr(fusion, 'fusion_layers'):
                for idx, layer in enumerate(fusion.fusion_layers):
                    if hasattr(layer, 'cross_attn_a_to_b'):
                        def make_hook(name):
                            def attn_hook(module, input, output):
                                if isinstance(output, tuple) and len(output) > 1:
                                    hook.representations[name] = output[1].detach().clone()
                            return attn_hook
                        layer.cross_attn_a_to_b.register_forward_hook(
                            make_hook(f'cross_attn_a_to_b_layer{idx}')
                        )
                        layer.cross_attn_b_to_a.register_forward_hook(
                            make_hook(f'cross_attn_b_to_a_layer{idx}')
                        )

        # Collect attention patterns
        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(self.config.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

            hook.clear()

            # Prepare batch for model
            model_batch = prepare_batch_for_model(batch)

            with torch.no_grad():
                try:
                    _ = self.model(**model_batch, return_attention=True)
                except TypeError:
                    _ = self.model(**model_batch)

            # Extract source importance
            if 'source_importance' in hook.representations:
                source_importance_list.append(
                    hook.representations['source_importance'].cpu().numpy()
                )

            # Extract cross-resolution attention
            for key in hook.representations:
                if 'cross_attn' in key:
                    if key not in self.attention_matrices:
                        self.attention_matrices[key] = []
                    self.attention_matrices[key].append(
                        hook.representations[key].cpu().numpy()
                    )

            if batch_idx >= 30:  # Limit for efficiency
                break

        # Aggregate results
        metrics = {}

        # Analyze source importance
        if source_importance_list:
            source_importance_all = np.concatenate(source_importance_list, axis=0)
            # Average over batch and sequence
            mean_importance = source_importance_all.mean(axis=(0, 1))
            std_importance = source_importance_all.std(axis=(0, 1))

            source_names = SourceRepresentationExtractor.DAILY_SOURCES

            # Guard: Ensure mean_importance size matches source_names
            if mean_importance.size == 0:
                logger.warning("Empty source importance array - no attention data extracted")
                metrics["source_importance"] = {
                    "mean": {},
                    "std": {},
                    "ranking": [],
                    "error": "No source importance data extracted from model",
                }
                self.source_importance = None
            elif mean_importance.size != len(source_names):
                logger.warning(
                    f"Source importance size mismatch: got {mean_importance.size} values, "
                    f"expected {len(source_names)} sources. Using available data."
                )
                # Use available data with numeric indices as fallback
                available_names = [source_names[i] if i < len(source_names) else f"source_{i}"
                                   for i in range(mean_importance.size)]
                sorted_indices = np.argsort(-mean_importance)
                metrics["source_importance"] = {
                    "mean": dict(zip(available_names, mean_importance.tolist())),
                    "std": dict(zip(available_names, std_importance.tolist())),
                    "ranking": [available_names[i] for i in sorted_indices if i < len(available_names)],
                }
                self.source_importance = mean_importance
            else:
                sorted_indices = np.argsort(-mean_importance)
                metrics["source_importance"] = {
                    "mean": dict(zip(source_names, mean_importance.tolist())),
                    "std": dict(zip(source_names, std_importance.tolist())),
                    "ranking": [source_names[i] for i in sorted_indices],
                }
                self.source_importance = mean_importance
        else:
            logger.warning("No source importance data collected - model may not have source gating")
            metrics["source_importance"] = {
                "mean": {},
                "std": {},
                "ranking": [],
                "error": "No source importance data collected from model",
            }

        # Analyze cross-resolution attention
        for key, attn_list in self.attention_matrices.items():
            if attn_list:
                attn_all = np.concatenate(attn_list, axis=0)
                mean_attn = attn_all.mean(axis=0)  # Average over samples

                # Compute attention entropy
                entropy = -np.sum(mean_attn * np.log(mean_attn + 1e-10), axis=-1).mean()

                metrics[f"{key}_entropy"] = float(entropy)
                metrics[f"{key}_sparsity"] = float((mean_attn < self.config.attention_threshold).mean())

        # Compute attention flow summary
        metrics["attention_flow_summary"] = self._compute_flow_summary()

        # Interpretation
        interpretation = self._interpret_attention_results(metrics)
        recommendations = self._generate_recommendations(metrics)

        self.results = ProbeResult(
            probe_name="AttentionFlowProbe",
            timestamp=timestamp,
            metrics=metrics,
            artifacts={},
            interpretation=interpretation,
            recommendations=recommendations,
        )

        self.results.artifacts = self.visualize()
        hook.remove_hooks()

        return self.results

    def _compute_flow_summary(self) -> Dict[str, Any]:
        """Compute summary of attention flow patterns."""
        summary = {
            "dominant_source": None,
            "attention_concentration": 0.0,
            "bidirectional_balance": 0.0,
        }

        if self.source_importance is not None and self.source_importance.size > 0:
            source_names = SourceRepresentationExtractor.DAILY_SOURCES
            max_idx = int(np.argmax(self.source_importance))

            # Guard: Ensure max_idx is within bounds
            if max_idx < len(source_names):
                summary["dominant_source"] = source_names[max_idx]
            else:
                summary["dominant_source"] = f"source_{max_idx}"

            # Guard: Avoid division by zero
            importance_sum = np.sum(self.source_importance)
            if importance_sum > 0:
                summary["attention_concentration"] = float(np.max(self.source_importance) / importance_sum)

        return summary

    def _interpret_attention_results(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of attention flow results."""
        interpretation = "Attention Flow Analysis Results:\n" + "=" * 35 + "\n\n"

        if "source_importance" in metrics:
            ranking = metrics["source_importance"].get("ranking", [])
            mean_vals = metrics["source_importance"].get("mean", {})
            std_vals = metrics["source_importance"].get("std", {})

            # Guard: Handle empty or error cases
            if not ranking:
                if "error" in metrics["source_importance"]:
                    interpretation += f"Source Importance: {metrics['source_importance']['error']}\n"
                else:
                    interpretation += "Source Importance: No ranking data available\n"
            else:
                interpretation += "Source Importance Ranking:\n"
                for i, source in enumerate(ranking, 1):
                    mean_val = mean_vals.get(source, 0.0)
                    std_val = std_vals.get(source, 0.0)
                    interpretation += f"  {i}. {source}: {mean_val:.4f} (+/- {std_val:.4f})\n"

        interpretation += "\n"

        # Analyze entropy
        entropy_values = [v for k, v in metrics.items() if "entropy" in k]
        if entropy_values:
            mean_entropy = np.mean(entropy_values)
            interpretation += f"Mean Attention Entropy: {mean_entropy:.4f}\n"
            if mean_entropy < 1.0:
                interpretation += "  -> Low entropy indicates focused attention\n"
            else:
                interpretation += "  -> High entropy indicates diffuse attention\n"

        return interpretation

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on attention analysis."""
        recommendations = []

        if "source_importance" in metrics:
            importance = metrics["source_importance"].get("mean", {})

            # Guard: Only compute if we have importance values
            if importance:
                max_imp = max(importance.values())
                min_imp = min(importance.values())

                if max_imp / (min_imp + 1e-8) > 5:
                    recommendations.append(
                        "Source importance is highly imbalanced - some sources may be underutilized"
                    )
            elif "error" in metrics["source_importance"]:
                recommendations.append(
                    "Unable to analyze source importance - check model architecture for source gating"
                )

        for key, val in metrics.items():
            if "sparsity" in key and val > 0.8:
                recommendations.append(
                    f"High attention sparsity in {key} - consider reducing attention temperature"
                )

        return recommendations

    def visualize(self) -> Dict[str, Path]:
        """Generate attention flow visualizations."""
        if self.results is None:
            raise ValueError("No results to visualize. Run probe first.")

        artifacts = {}

        # Source importance bar chart
        if "source_importance" in self.results.metrics:
            importance = self.results.metrics["source_importance"].get("mean", {})
            std = self.results.metrics["source_importance"].get("std", {})

            # Guard: Only create chart if we have data
            if importance:
                fig, ax = plt.subplots(figsize=(10, 6))

                sources = list(importance.keys())
                values = [importance[s] for s in sources]
                errors = [std.get(s, 0.0) for s in sources]

                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sources)))
                bars = ax.bar(sources, values, yerr=errors, capsize=5, color=colors, edgecolor='black')

                ax.set_xlabel("Source", fontsize=12)
                ax.set_ylabel("Importance Weight", fontsize=12)
                ax.set_title("Source Importance in DailyCrossSourceFusion", fontsize=14)
                plt.xticks(rotation=45, ha='right')

                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

                plt.tight_layout()

                bar_path = self.config.output_dir / f"source_importance_{self.results.timestamp}.png"
                plt.savefig(bar_path, dpi=150, bbox_inches='tight')
                plt.close()
                artifacts["source_importance_bar"] = bar_path
            else:
                logger.warning("No source importance data available for visualization")

        # Attention flow Sankey diagram (simplified as heatmap)
        if self.attention_matrices:
            fig, axes = plt.subplots(1, min(2, len(self.attention_matrices)),
                                    figsize=(14, 6), squeeze=False)

            for idx, (key, attn_list) in enumerate(list(self.attention_matrices.items())[:2]):
                if attn_list:
                    attn_all = np.concatenate(attn_list, axis=0)
                    mean_attn = attn_all.mean(axis=0)

                    # If multi-head, average over heads
                    if mean_attn.ndim > 2:
                        mean_attn = mean_attn.mean(axis=0)

                    # Visualize as heatmap
                    ax = axes[0, idx]
                    sns.heatmap(mean_attn[:20, :20], cmap='Blues', ax=ax)
                    ax.set_title(f"{key}\n(first 20x20)", fontsize=10)
                    ax.set_xlabel("Key Position")
                    ax.set_ylabel("Query Position")

            plt.tight_layout()
            flow_path = self.config.output_dir / f"attention_flow_{self.results.timestamp}.png"
            plt.savefig(flow_path, dpi=150, bbox_inches='tight')
            plt.close()
            artifacts["attention_flow"] = flow_path

        logger.info(f"Attention flow visualizations saved to {self.config.output_dir}")
        return artifacts


# =============================================================================
# ABLATION PROBE
# =============================================================================

class AblationProbe(FusionProbe):
    """
    Ablation probe for leave-one-out and source sufficiency tests.

    This probe measures:
    1. Leave-One-Out: Performance delta when each source is removed
    2. Source Sufficiency: Performance when using only a single source

    These tests quantify the necessity and sufficiency of each source for
    different prediction tasks.

    Usage:
        >>> config = AblationProbeConfig(model=model)
        >>> probe = AblationProbe(config)
        >>> results = probe.run(data_loader, task_evaluators)
    """

    def __init__(self, config: AblationProbeConfig):
        super().__init__(config)
        self.config: AblationProbeConfig = config
        self.baseline_performance: Dict[str, float] = {}
        self.loo_performance: Dict[str, Dict[str, float]] = {}
        self.sufficiency_performance: Dict[str, Dict[str, float]] = {}

    def _create_masked_batch(
        self,
        batch: Dict[str, Any],
        source_to_mask: str,
        mode: str = "zero"
    ) -> Dict[str, Any]:
        """Create batch with specified source masked.

        Args:
            batch: Original batch (may have nested dict structure like daily_features/monthly_features)
            source_to_mask: Name of source to mask (e.g., 'equipment', 'personnel', 'sentinel')
            mode: Masking mode ('zero', 'mean', 'noise', 'shuffle')

        Returns:
            Modified batch with source masked

        Note: The batch uses nested dicts (daily_features[source], monthly_features[source])
        rather than flat keys like 'equipment_features'. This method handles both structures.
        """
        # Deep copy the nested structure
        masked_batch = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                masked_batch[k] = {kk: vv.clone() if isinstance(vv, Tensor) else vv
                                   for kk, vv in v.items()}
            elif isinstance(v, Tensor):
                masked_batch[k] = v.clone()
            else:
                masked_batch[k] = v

        # Check daily_features nested dict
        if source_to_mask in masked_batch.get('daily_features', {}):
            features = masked_batch['daily_features'][source_to_mask]
            if mode == "zero":
                masked_batch['daily_features'][source_to_mask] = torch.zeros_like(features)
            elif mode == "mean":
                mean_val = features.mean(dim=1, keepdim=True).expand_as(features)
                masked_batch['daily_features'][source_to_mask] = mean_val
            elif mode == "noise":
                masked_batch['daily_features'][source_to_mask] = torch.randn_like(features)
            elif mode == "shuffle":
                idx = torch.randperm(features.size(0))
                masked_batch['daily_features'][source_to_mask] = features[idx]
            # Keep mask True so model sees masked values as "observed"
            # (DO NOT zero the mask - that would make intervention ineffective)
            logger.debug(f"Masked daily source {source_to_mask} with mode={mode}, mask kept True")

        # Check monthly_features nested dict
        elif source_to_mask in masked_batch.get('monthly_features', {}):
            features = masked_batch['monthly_features'][source_to_mask]
            if mode == "zero":
                masked_batch['monthly_features'][source_to_mask] = torch.zeros_like(features)
            elif mode == "mean":
                mean_val = features.mean(dim=1, keepdim=True).expand_as(features)
                masked_batch['monthly_features'][source_to_mask] = mean_val
            elif mode == "noise":
                masked_batch['monthly_features'][source_to_mask] = torch.randn_like(features)
            elif mode == "shuffle":
                idx = torch.randperm(features.size(0))
                masked_batch['monthly_features'][source_to_mask] = features[idx]
            # Keep mask True so model sees masked values as "observed"
            logger.debug(f"Masked monthly source {source_to_mask} with mode={mode}, mask kept True")

        # Fallback: Check for flat key structure (legacy compatibility)
        else:
            feature_key = f"{source_to_mask}_features"
            if feature_key in masked_batch and isinstance(masked_batch[feature_key], Tensor):
                features = masked_batch[feature_key]
                if mode == "zero":
                    masked_batch[feature_key] = torch.zeros_like(features)
                elif mode == "mean":
                    mean_val = features.mean()
                    masked_batch[feature_key] = torch.full_like(features, mean_val)
                elif mode == "noise":
                    masked_batch[feature_key] = torch.randn_like(features)
                elif mode == "shuffle":
                    idx = torch.randperm(features.size(0))
                    masked_batch[feature_key] = features[idx]
                logger.debug(f"Masked source {source_to_mask} (flat key) with mode={mode}")
            else:
                logger.warning(
                    f"Source '{source_to_mask}' not found in batch. "
                    f"Available daily: {list(masked_batch.get('daily_features', {}).keys())}, "
                    f"monthly: {list(masked_batch.get('monthly_features', {}).keys())}"
                )

        return masked_batch

    def _create_single_source_batch(
        self,
        batch: Dict[str, Tensor],
        source_to_keep: str
    ) -> Dict[str, Tensor]:
        """Create batch with only one source active.

        Args:
            batch: Original batch
            source_to_keep: Name of source to keep

        Returns:
            Modified batch with only specified source
        """
        masked_batch = {}

        for key, val in batch.items():
            if isinstance(val, Tensor):
                # Keep only the specified source, zero others
                if source_to_keep in key:
                    masked_batch[key] = val.clone()
                elif "_features" in key or "_mask" in key:
                    masked_batch[key] = torch.zeros_like(val)
                else:
                    masked_batch[key] = val.clone()
            else:
                masked_batch[key] = val

        return masked_batch

    def run(
        self,
        data_loader: Any,
        task_evaluators: Optional[Dict[str, Callable]] = None
    ) -> ProbeResult:
        """Run ablation studies.

        This probe measures source importance by comparing predictions with and without
        each source. When explicit targets are unavailable (common with MultiResolutionDataset),
        it uses baseline predictions as pseudo-targets to measure prediction deviation.

        The key insight: if removing a source causes large prediction changes, that source
        is important for the model's current behavior.

        Metrics interpretation:
        - prediction_deviation: How much predictions change when source is ablated (higher = more important)
        - baseline: Reference performance (0.0 when using self-comparison mode)
        - leave_one_out: Performance after ablation (negative MSE; closer to 0 = less change)
        - performance_delta: baseline - loo (positive = source is necessary)

        Args:
            data_loader: PyTorch DataLoader
            task_evaluators: Dict mapping task name to evaluation function
                Each evaluator takes (predictions, targets) and returns a metric.
                If None or empty, uses negative MSE for comparison.

        Returns:
            ProbeResult with ablation metrics
        """
        logger.info("Running Ablation Probe...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle both None and empty dict {} - both should trigger default evaluators
        if task_evaluators is None or len(task_evaluators) == 0:
            if task_evaluators is not None:
                logger.warning(
                    "Empty task_evaluators dict passed to AblationProbe.run(). "
                    "Using default evaluators. Pass None instead of {} for default behavior."
                )
            # Default evaluators (negative MSE - higher is better, 0 means identical)
            task_evaluators = {
                task: lambda pred, tgt: -F.mse_loss(pred, tgt).item()
                for task in self.config.tasks
            }

        # Dynamically detect sources from the first batch instead of using hardcoded list
        # This supports both aggregated (6 sources) and disaggregated (8 sources) models
        first_batch = next(iter(data_loader))
        daily_sources, monthly_sources = SourceRepresentationExtractor.get_sources_from_batch(first_batch)
        all_sources = daily_sources + monthly_sources
        logger.info(f"Detected sources for ablation: daily={daily_sources}, monthly={monthly_sources}")

        # Update class-level attributes for other components that might use them
        SourceRepresentationExtractor.update_sources_from_batch(first_batch)

        # Collect baseline predictions
        logger.info("Computing baseline predictions...")
        baseline_preds = {task: [] for task in self.config.tasks}
        baseline_targets = {task: [] for task in self.config.tasks}
        has_explicit_targets = {task: False for task in self.config.tasks}

        for batch_idx, batch in enumerate(data_loader):
            batch = {k: v.to(self.config.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

            # Prepare batch for model
            model_batch = prepare_batch_for_model(batch)

            with torch.no_grad():
                outputs = self.model(**model_batch)

            # Extract predictions and targets per task
            # Uses centralized task_key_mapping to resolve task names to output keys
            if isinstance(outputs, dict):
                for task in self.config.tasks:
                    # Try both the task name and mapped output key
                    task_output = extract_task_output(outputs, task)
                    if task_output is not None:
                        baseline_preds[task].append(task_output.cpu())
                    # Check for targets using both task name and output key
                    output_key = get_output_key(task)
                    target_keys = [f"{task}_target", f"{output_key}_target"]
                    for target_key in target_keys:
                        if target_key in batch:
                            baseline_targets[task].append(batch[target_key].cpu())
                            has_explicit_targets[task] = True
                            break

            if batch_idx >= self.config.num_samples // data_loader.batch_size:
                break

        # Concatenate baseline predictions for use as pseudo-targets
        baseline_preds_concat = {}
        for task in self.config.tasks:
            if baseline_preds[task]:
                baseline_preds_concat[task] = torch.cat(baseline_preds[task])

        # Check if we have explicit targets; if not, use self-supervised mode
        using_pseudo_targets = not any(has_explicit_targets.values())
        if using_pseudo_targets:
            logger.info(
                "No explicit targets found in dataset. Using self-supervised ablation mode: "
                "measuring prediction deviation from baseline (higher deviation = source is more important)."
            )
            # In self-supervised mode, baseline performance is 0 (predictions vs themselves)
            for task in self.config.tasks:
                if task in baseline_preds_concat:
                    self.baseline_performance[task] = 0.0
                    logger.info(f"  Baseline {task}: 0.0000 (self-comparison reference)")
        else:
            # Compute baseline metrics with explicit targets
            for task in self.config.tasks:
                if baseline_preds[task] and baseline_targets[task]:
                    preds = baseline_preds_concat[task]
                    targets = torch.cat(baseline_targets[task])
                    self.baseline_performance[task] = task_evaluators[task](preds, targets)
                    logger.info(f"  Baseline {task}: {self.baseline_performance[task]:.4f}")

        # Leave-One-Out ablation
        logger.info("Running Leave-One-Out ablation...")
        self.loo_performance = {source: {} for source in all_sources}
        # Store raw prediction deviations for additional metrics
        self._prediction_deviations = {source: {} for source in all_sources}

        for source in all_sources:
            logger.info(f"  Ablating {source}...")
            loo_preds = {task: [] for task in self.config.tasks}
            loo_targets = {task: [] for task in self.config.tasks}

            for batch_idx, batch in enumerate(data_loader):
                batch = {k: v.to(self.config.device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()}

                # Create masked batch and prepare for model
                masked_batch = self._create_masked_batch(
                    batch, source, self.config.ablation_mode
                )
                model_batch = prepare_batch_for_model(masked_batch)

                with torch.no_grad():
                    outputs = self.model(**model_batch)

                if isinstance(outputs, dict):
                    for task in self.config.tasks:
                        # Use centralized task key mapping
                        task_output = extract_task_output(outputs, task)
                        if task_output is not None:
                            loo_preds[task].append(task_output.cpu())
                        # Check for targets
                        output_key = get_output_key(task)
                        for target_key in [f"{task}_target", f"{output_key}_target"]:
                            if target_key in batch:
                                loo_targets[task].append(batch[target_key].cpu())
                                break

                if batch_idx >= self.config.num_samples // data_loader.batch_size:
                    break

            # Compute LOO metrics
            for task in self.config.tasks:
                if loo_preds[task]:
                    preds = torch.cat(loo_preds[task])

                    if using_pseudo_targets and task in baseline_preds_concat:
                        # Use baseline predictions as pseudo-targets
                        # Truncate to match lengths (in case of batching differences)
                        min_len = min(preds.shape[0], baseline_preds_concat[task].shape[0])
                        preds_aligned = preds[:min_len]
                        baseline_aligned = baseline_preds_concat[task][:min_len]

                        # Compute deviation: negative MSE (closer to 0 = less deviation)
                        self.loo_performance[source][task] = task_evaluators[task](
                            preds_aligned, baseline_aligned
                        )
                        # Also store raw deviation (positive MSE) for interpretation
                        deviation = F.mse_loss(preds_aligned, baseline_aligned).item()
                        self._prediction_deviations[source][task] = deviation

                    elif loo_targets[task]:
                        # Use explicit targets
                        targets = torch.cat(loo_targets[task])
                        self.loo_performance[source][task] = task_evaluators[task](preds, targets)
                        self._prediction_deviations[source][task] = 0.0  # N/A in this mode

        # Source Sufficiency tests
        logger.info("Running Source Sufficiency tests...")
        self.sufficiency_performance = {source: {} for source in all_sources}

        for source in all_sources:
            logger.info(f"  Testing sufficiency of {source}...")
            suff_preds = {task: [] for task in self.config.tasks}
            suff_targets = {task: [] for task in self.config.tasks}

            for batch_idx, batch in enumerate(data_loader):
                batch = {k: v.to(self.config.device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()}

                # Create single-source batch and prepare for model
                single_batch = self._create_single_source_batch(batch, source)
                model_batch = prepare_batch_for_model(single_batch)

                with torch.no_grad():
                    try:
                        outputs = self.model(**model_batch)
                    except Exception:
                        # Some sources may not work alone
                        break

                if isinstance(outputs, dict):
                    for task in self.config.tasks:
                        # Use centralized task key mapping
                        task_output = extract_task_output(outputs, task)
                        if task_output is not None:
                            suff_preds[task].append(task_output.cpu())
                        # Check for targets
                        output_key = get_output_key(task)
                        for target_key in [f"{task}_target", f"{output_key}_target"]:
                            if target_key in batch:
                                suff_targets[task].append(batch[target_key].cpu())
                                break

                if batch_idx >= self.config.num_samples // data_loader.batch_size:
                    break

            # Compute sufficiency metrics
            for task in self.config.tasks:
                if suff_preds[task]:
                    preds = torch.cat(suff_preds[task])

                    if using_pseudo_targets and task in baseline_preds_concat:
                        # Compare single-source predictions to baseline
                        min_len = min(preds.shape[0], baseline_preds_concat[task].shape[0])
                        preds_aligned = preds[:min_len]
                        baseline_aligned = baseline_preds_concat[task][:min_len]
                        self.sufficiency_performance[source][task] = task_evaluators[task](
                            preds_aligned, baseline_aligned
                        )
                    elif suff_targets[task]:
                        targets = torch.cat(suff_targets[task])
                        self.sufficiency_performance[source][task] = task_evaluators[task](preds, targets)

        # Compile metrics
        metrics = {
            "baseline": self.baseline_performance,
            "leave_one_out": self.loo_performance,
            "sufficiency": self.sufficiency_performance,
            "performance_delta": self._compute_performance_delta(),
            "prediction_deviation": self._prediction_deviations,
            "using_pseudo_targets": using_pseudo_targets,
        }

        interpretation = self._interpret_ablation_results(metrics)
        recommendations = self._generate_recommendations(metrics)

        self.results = ProbeResult(
            probe_name="AblationProbe",
            timestamp=timestamp,
            metrics=metrics,
            artifacts={},
            interpretation=interpretation,
            recommendations=recommendations,
        )

        self.results.artifacts = self.visualize()
        return self.results

    def _compute_performance_delta(self) -> Dict[str, Dict[str, float]]:
        """Compute performance delta for each source and task."""
        deltas = {}

        for source in self.loo_performance:
            deltas[source] = {}
            for task in self.config.tasks:
                if task in self.baseline_performance and task in self.loo_performance[source]:
                    deltas[source][task] = (
                        self.baseline_performance[task] - self.loo_performance[source][task]
                    )

        return deltas

    def _interpret_ablation_results(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of ablation results."""
        interpretation = "Ablation Analysis Results:\n" + "=" * 35 + "\n\n"

        using_pseudo_targets = metrics.get("using_pseudo_targets", False)

        if using_pseudo_targets:
            interpretation += "Mode: Self-Supervised (no explicit targets available)\n"
            interpretation += "  Metric: Prediction deviation from baseline\n"
            interpretation += "  Higher deviation = source is more important for predictions\n"
            interpretation += "  Positive delta = source is necessary (removal changes predictions)\n\n"
        else:
            interpretation += "Mode: Supervised (using explicit targets)\n"
            interpretation += "  Positive delta = source is necessary (removal hurts)\n"
            interpretation += "  Negative delta = source is harmful (removal helps)\n\n"

        for task in self.config.tasks:
            interpretation += f"\nTask: {task}\n"
            baseline_val = metrics['baseline'].get(task, 'N/A')
            if isinstance(baseline_val, (int, float)):
                interpretation += f"  Baseline: {baseline_val:.4f}\n"
            else:
                interpretation += f"  Baseline: {baseline_val}\n"

            # In self-supervised mode, also show prediction deviations
            if using_pseudo_targets and "prediction_deviation" in metrics:
                interpretation += "  Source importance (by prediction deviation):\n"
                deviations = [
                    (s, metrics["prediction_deviation"].get(s, {}).get(task, 0))
                    for s in self.loo_performance
                ]
                deviations.sort(key=lambda x: x[1], reverse=True)
                for source, dev in deviations[:5]:  # Top 5
                    interpretation += f"    {source}: deviation = {dev:.6f}\n"
            else:
                # Sort sources by importance (delta)
                deltas = [(s, metrics['performance_delta'].get(s, {}).get(task, 0))
                         for s in self.loo_performance]
                deltas.sort(key=lambda x: x[1], reverse=True)

                interpretation += "  Source importance (by performance delta):\n"
                for source, delta in deltas[:5]:  # Top 5
                    interpretation += f"    {source}: delta = {delta:+.4f}\n"

        return interpretation

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations from ablation results."""
        recommendations = []
        using_pseudo_targets = metrics.get("using_pseudo_targets", False)

        if using_pseudo_targets:
            # In self-supervised mode, use prediction deviation to determine importance
            # Higher deviation means the source is more important
            essential_sources = set()
            low_impact_sources = set()

            deviations = metrics.get("prediction_deviation", {})
            for source, task_devs in deviations.items():
                if not task_devs:
                    continue
                valid_devs = [d for d in task_devs.values() if d is not None and d > 0]
                if not valid_devs:
                    continue
                avg_deviation = np.mean(valid_devs)
                if avg_deviation > 0.01:  # Significant deviation
                    essential_sources.add(source)
                elif avg_deviation < 0.0001:  # Minimal deviation
                    low_impact_sources.add(source)

            if essential_sources:
                recommendations.append(
                    f"Important sources (high prediction deviation): {', '.join(sorted(essential_sources))}"
                )
            if low_impact_sources:
                recommendations.append(
                    f"Low-impact sources (minimal prediction change): {', '.join(sorted(low_impact_sources))}"
                )

            # Check sufficiency: sources that alone produce similar predictions to baseline
            for source, perfs in metrics.get("sufficiency", {}).items():
                for task, perf in perfs.items():
                    # In pseudo-target mode, perf is negative MSE; closer to 0 means more similar
                    if perf is not None and perf > -0.01:  # Very similar to baseline
                        recommendations.append(
                            f"Source '{source}' alone produces near-baseline predictions for {task}"
                        )
        else:
            # Original supervised mode logic
            essential_sources = set()
            redundant_sources = set()

            for source, deltas in metrics.get("performance_delta", {}).items():
                valid_deltas = [d for d in deltas.values() if d is not None]
                if not valid_deltas:
                    continue
                avg_delta = np.mean(valid_deltas)
                if avg_delta > 0.1:
                    essential_sources.add(source)
                elif avg_delta < -0.05:
                    redundant_sources.add(source)

            if essential_sources:
                recommendations.append(
                    f"Essential sources (removal hurts): {', '.join(essential_sources)}"
                )
            if redundant_sources:
                recommendations.append(
                    f"Potentially redundant sources (removal helps): {', '.join(redundant_sources)}"
                )

            # Check sufficiency
            for source, perfs in metrics.get("sufficiency", {}).items():
                for task, perf in perfs.items():
                    baseline = metrics["baseline"].get(task, 0)
                    if baseline > 0 and perf / baseline > 0.9:
                        recommendations.append(
                            f"Source '{source}' alone achieves >90% of baseline for {task}"
                        )

        return recommendations

    def visualize(self) -> Dict[str, Path]:
        """Generate ablation visualizations."""
        if self.results is None:
            raise ValueError("No results to visualize. Run probe first.")

        artifacts = {}
        using_pseudo_targets = self.results.metrics.get("using_pseudo_targets", False)

        sources = list(self.loo_performance.keys())
        tasks = self.config.tasks

        # Leave-One-Out heatmap
        fig, ax = plt.subplots(figsize=(14, 8))

        # In self-supervised mode, show prediction deviation (positive MSE)
        # In supervised mode, show performance delta
        if using_pseudo_targets and "prediction_deviation" in self.results.metrics:
            # Use prediction deviation for heatmap (higher = more important)
            delta_matrix = np.zeros((len(sources), len(tasks)))
            for i, source in enumerate(sources):
                for j, task in enumerate(tasks):
                    delta = self.results.metrics["prediction_deviation"].get(source, {}).get(task, 0)
                    delta_matrix[i, j] = delta

            sns.heatmap(
                delta_matrix,
                annot=True,
                fmt=".4f",
                cmap="YlOrRd",  # Yellow to Red: higher deviation = more important
                xticklabels=tasks,
                yticklabels=[s.replace("_pre_fusion", "") for s in sources],
                ax=ax,
            )
            ax.set_title(
                "Prediction Deviation When Source Ablated\n"
                "(Higher = source is more important for predictions)",
                fontsize=14
            )
        else:
            delta_matrix = np.zeros((len(sources), len(tasks)))
            for i, source in enumerate(sources):
                for j, task in enumerate(tasks):
                    delta = self.results.metrics["performance_delta"].get(source, {}).get(task, 0)
                    delta_matrix[i, j] = delta

            sns.heatmap(
                delta_matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=0,
                xticklabels=tasks,
                yticklabels=[s.replace("_pre_fusion", "") for s in sources],
                ax=ax,
            )
            ax.set_title("Leave-One-Out Performance Delta\n(Positive = source is necessary)", fontsize=14)

        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel("Ablated Source", fontsize=12)
        plt.tight_layout()

        loo_path = self.config.output_dir / f"loo_ablation_{self.results.timestamp}.png"
        plt.savefig(loo_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts["loo_heatmap"] = loo_path

        # Sufficiency bar chart
        fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 6), squeeze=False)

        for j, task in enumerate(tasks):
            ax = axes[0, j]

            baseline = self.results.metrics["baseline"].get(task, 0)

            sufficiency_values = []
            source_labels = []
            for source in sources:
                suff = self.results.metrics["sufficiency"].get(source, {}).get(task)
                if suff is not None:
                    sufficiency_values.append(suff)
                    source_labels.append(source.replace("_pre_fusion", ""))

            if sufficiency_values:
                if using_pseudo_targets:
                    # In pseudo-target mode, values are negative MSE; closer to 0 = better
                    # Convert to positive scale for visualization
                    display_values = [-v for v in sufficiency_values]  # Now higher = more deviation
                    colors = ['green' if v < 0.01 else 'orange' if v < 0.1 else 'red'
                             for v in display_values]
                    ax.barh(source_labels, display_values, color=colors, edgecolor='black')
                    ax.axvline(x=0, color='blue', linestyle='--', label='Baseline (0 deviation)')
                    ax.set_xlabel("Deviation from Baseline (MSE)")
                    ax.set_title(f"Source Sufficiency: {task}\n(Lower = source alone approximates baseline)")
                else:
                    colors = ['green' if v > baseline * 0.8 else 'orange' if v > baseline * 0.5 else 'red'
                             for v in sufficiency_values]
                    ax.barh(source_labels, sufficiency_values, color=colors, edgecolor='black')
                    ax.axvline(x=baseline, color='blue', linestyle='--', label=f'Baseline ({baseline:.3f})')
                    ax.axvline(x=baseline * 0.8, color='green', linestyle=':', alpha=0.5)
                    ax.set_xlabel("Performance")
                    ax.set_title(f"Source Sufficiency: {task}")
                ax.legend()

        plt.tight_layout()
        suff_path = self.config.output_dir / f"sufficiency_{self.results.timestamp}.png"
        plt.savefig(suff_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts["sufficiency_chart"] = suff_path

        logger.info(f"Ablation visualizations saved to {self.config.output_dir}")
        return artifacts


# =============================================================================
# CHECKPOINT COMPARISON PROBE
# =============================================================================

class CheckpointComparisonProbe(FusionProbe):
    """
    Probe for tracking fusion quality evolution across training epochs.

    This probe loads checkpoints from different training stages and computes:
    1. RSA evolution over epochs
    2. Source importance changes
    3. Attention entropy trends

    This helps identify:
    - When fusion quality plateaus or degrades
    - Impact of training interventions (e.g., ZINB loss introduction)
    - Optimal stopping points for fusion learning

    Usage:
        >>> config = CheckpointProbeConfig(model=model, checkpoint_dir=Path("./checkpoints"))
        >>> probe = CheckpointComparisonProbe(config)
        >>> results = probe.run(data_loader)
    """

    def __init__(self, config: CheckpointProbeConfig):
        super().__init__(config)
        self.config: CheckpointProbeConfig = config
        self.epoch_metrics: Dict[int, Dict[str, Any]] = {}

    def _load_checkpoint(self, epoch: int) -> bool:
        """Load model checkpoint for specified epoch.

        Args:
            epoch: Training epoch number

        Returns:
            True if checkpoint loaded successfully
        """
        if self.config.checkpoint_dir is None:
            logger.warning("No checkpoint directory specified")
            return False

        if not self.config.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory does not exist: {self.config.checkpoint_dir}")
            return False

        # Try common naming patterns
        naming_patterns = [
            f"epoch_{epoch}.pt",
            f"checkpoint_epoch{epoch}.pt",
            f"checkpoint_epoch_{epoch}.pt",
            f"model_{epoch}.pt",
            f"model_epoch_{epoch}.pt",
        ]

        checkpoint_path = None
        for pattern in naming_patterns:
            candidate = self.config.checkpoint_dir / pattern
            if candidate.exists():
                checkpoint_path = candidate
                break

        if checkpoint_path is None:
            logger.warning(
                f"Checkpoint not found for epoch {epoch}. "
                f"Searched patterns: {naming_patterns} in {self.config.checkpoint_dir}"
            )
            return False

        try:
            # Use weights_only=False to handle checkpoints saved with numpy arrays
            state_dict = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint for epoch {epoch}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint for epoch {epoch}: {e}")
            return False

    def run(self, data_loader: Any) -> ProbeResult:
        """Run checkpoint comparison across training epochs.

        Args:
            data_loader: PyTorch DataLoader

        Returns:
            ProbeResult with epoch-wise metrics
        """
        logger.info("Running Checkpoint Comparison Probe...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Will analyze epochs: {self.config.checkpoint_epochs}")
        if self.config.checkpoint_dir:
            logger.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        else:
            logger.warning("No checkpoint directory configured")

        epochs_loaded = 0
        for epoch in self.config.checkpoint_epochs:
            logger.info(f"Analyzing epoch {epoch}...")

            if not self._load_checkpoint(epoch):
                continue

            epochs_loaded += 1
            epoch_metrics = {}

            # Run RSA probe for this checkpoint
            if "rsa" in self.config.metrics_to_track:
                rsa_config = RSAProbeConfig(
                    model=self.model,
                    device=self.config.device,
                    output_dir=self.config.output_dir,
                )
                rsa_probe = RSAProbe(rsa_config)
                rsa_results = rsa_probe.run(data_loader)
                epoch_metrics["mean_rsa"] = rsa_results.metrics["mean_rsa"]
                epoch_metrics["max_rsa"] = rsa_results.metrics["max_rsa"]

            # Run attention flow probe
            if "attention_entropy" in self.config.metrics_to_track:
                attn_config = AttentionFlowProbeConfig(
                    model=self.model,
                    device=self.config.device,
                    output_dir=self.config.output_dir,
                )
                attn_probe = AttentionFlowProbe(attn_config)
                attn_results = attn_probe.run(data_loader)

                # Extract entropy metrics
                for key, val in attn_results.metrics.items():
                    if "entropy" in key:
                        epoch_metrics[key] = val

            # Extract source importance
            if "source_importance" in self.config.metrics_to_track:
                if "source_importance" in attn_results.metrics:
                    epoch_metrics["source_importance"] = attn_results.metrics["source_importance"]["mean"]

            self.epoch_metrics[epoch] = epoch_metrics

        logger.info(f"Successfully loaded and analyzed {epochs_loaded}/{len(self.config.checkpoint_epochs)} checkpoints")

        if epochs_loaded == 0:
            logger.warning(
                "No checkpoints were found. Verify checkpoint_dir path and that checkpoint files exist."
            )

        # Compile trend analysis
        metrics = {
            "epoch_metrics": self.epoch_metrics,
            "trends": self._analyze_trends(),
        }

        interpretation = self._interpret_trends(metrics)
        recommendations = self._generate_recommendations(metrics)

        self.results = ProbeResult(
            probe_name="CheckpointComparisonProbe",
            timestamp=timestamp,
            metrics=metrics,
            artifacts={},
            interpretation=interpretation,
            recommendations=recommendations,
        )

        # Only visualize if we have epoch metrics to display
        if self.epoch_metrics:
            self.results.artifacts = self.visualize()
        else:
            logger.warning("No checkpoints were successfully loaded. Skipping visualization.")
            self.results.artifacts = {}

        return self.results

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends across epochs."""
        trends = {}

        epochs = sorted(self.epoch_metrics.keys())
        if len(epochs) < 2:
            return trends

        # RSA trend
        rsa_values = [self.epoch_metrics[e].get("mean_rsa", 0) for e in epochs]
        if all(rsa_values):
            trends["rsa_trend"] = {
                "direction": "increasing" if rsa_values[-1] > rsa_values[0] else "decreasing",
                "change": rsa_values[-1] - rsa_values[0],
                "peak_epoch": epochs[np.argmax(rsa_values)],
            }

        # Check for degradation
        if len(rsa_values) >= 3:
            mid_idx = len(rsa_values) // 2
            if rsa_values[-1] < rsa_values[mid_idx] * 0.9:
                trends["degradation_detected"] = True
                trends["degradation_start_epoch"] = epochs[mid_idx]

        return trends

    def _interpret_trends(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of checkpoint comparison."""
        interpretation = "Checkpoint Comparison Results:\n" + "=" * 35 + "\n\n"

        epochs = sorted(self.epoch_metrics.keys())
        interpretation += f"Epochs analyzed: {epochs}\n\n"

        for epoch in epochs:
            interpretation += f"Epoch {epoch}:\n"
            for metric, value in self.epoch_metrics[epoch].items():
                if isinstance(value, float):
                    interpretation += f"  {metric}: {value:.4f}\n"
                elif isinstance(value, dict):
                    interpretation += f"  {metric}: {value}\n"

        if "trends" in metrics:
            interpretation += "\nTrend Analysis:\n"
            for trend_name, trend_data in metrics["trends"].items():
                interpretation += f"  {trend_name}: {trend_data}\n"

        return interpretation

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations from checkpoint analysis."""
        recommendations = []

        trends = metrics.get("trends", {})

        if trends.get("degradation_detected"):
            recommendations.append(
                f"Fusion quality degradation detected after epoch {trends['degradation_start_epoch']}. "
                "Consider early stopping or reducing learning rate."
            )

        if "rsa_trend" in trends:
            if trends["rsa_trend"]["direction"] == "decreasing":
                recommendations.append(
                    "RSA is decreasing over training - fusion may be getting worse. "
                    "Review loss function weighting."
                )
            peak_epoch = trends["rsa_trend"]["peak_epoch"]
            recommendations.append(
                f"Peak RSA observed at epoch {peak_epoch} - consider using this checkpoint."
            )

        return recommendations

    def visualize(self) -> Dict[str, Path]:
        """Generate checkpoint comparison visualizations."""
        if self.results is None:
            raise ValueError("No results to visualize. Run the probe first using probe.run(data_loader).")
        if not self.epoch_metrics:
            raise ValueError(
                "No checkpoint data available for visualization. "
                "Ensure checkpoint files exist in the checkpoint_dir and match expected naming patterns "
                "(epoch_N.pt, checkpoint_epochN.pt, or model_N.pt)."
            )

        artifacts = {}
        epochs = sorted(self.epoch_metrics.keys())

        # Fusion quality over epochs
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # RSA trend
        ax = axes[0]
        rsa_values = [self.epoch_metrics[e].get("mean_rsa", np.nan) for e in epochs]
        ax.plot(epochs, rsa_values, 'o-', linewidth=2, markersize=8, label='Mean RSA')
        ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.7, label='Expected threshold (0.3)')
        ax.fill_between(epochs, 0, rsa_values, alpha=0.3)
        ax.set_ylabel("RSA Correlation", fontsize=12)
        ax.set_title("Fusion Quality Evolution Across Training", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Entropy trend
        ax = axes[1]
        entropy_keys = [k for k in self.epoch_metrics[epochs[0]].keys() if "entropy" in k]
        for key in entropy_keys:
            values = [self.epoch_metrics[e].get(key, np.nan) for e in epochs]
            ax.plot(epochs, values, 'o-', linewidth=2, markersize=8, label=key)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Attention Entropy", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        trend_path = self.config.output_dir / f"checkpoint_trends_{self.results.timestamp}.png"
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        artifacts["checkpoint_trends"] = trend_path

        # Source importance evolution (if available)
        if any("source_importance" in self.epoch_metrics[e] for e in epochs):
            fig, ax = plt.subplots(figsize=(12, 8))

            # Collect all sources
            all_sources = set()
            for e in epochs:
                if "source_importance" in self.epoch_metrics[e]:
                    all_sources.update(self.epoch_metrics[e]["source_importance"].keys())

            for source in sorted(all_sources):
                values = []
                for e in epochs:
                    imp = self.epoch_metrics[e].get("source_importance", {}).get(source, np.nan)
                    values.append(imp)
                ax.plot(epochs, values, 'o-', label=source, linewidth=2)

            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Source Importance", fontsize=12)
            ax.set_title("Source Importance Evolution Over Training", fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            importance_path = self.config.output_dir / f"source_importance_evolution_{self.results.timestamp}.png"
            plt.savefig(importance_path, dpi=150, bbox_inches='tight')
            plt.close()
            artifacts["source_importance_evolution"] = importance_path

        logger.info(f"Checkpoint comparison visualizations saved to {self.config.output_dir}")
        return artifacts


# =============================================================================
# COMPREHENSIVE PROBE RUNNER
# =============================================================================

class CrossModalFusionProbeRunner:
    """
    Orchestrates running all fusion quality probes and generates a comprehensive report.

    Usage:
        >>> runner = CrossModalFusionProbeRunner(model, data_loader)
        >>> report = runner.run_all()
        >>> runner.save_report("fusion_quality_report.md")
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Any,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = checkpoint_dir
        self.results: Dict[str, ProbeResult] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_rsa_probe(self) -> ProbeResult:
        """Run RSA probe."""
        config = RSAProbeConfig(
            model=self.model,
            output_dir=self.output_dir,
        )
        probe = RSAProbe(config)
        result = probe.run(self.data_loader)
        probe.save_results()
        self.results["rsa"] = result
        return result

    def run_attention_flow_probe(self) -> ProbeResult:
        """Run attention flow probe."""
        config = AttentionFlowProbeConfig(
            model=self.model,
            output_dir=self.output_dir,
        )
        probe = AttentionFlowProbe(config)
        result = probe.run(self.data_loader)
        probe.save_results()
        self.results["attention_flow"] = result
        return result

    def run_ablation_probe(
        self,
        task_evaluators: Optional[Dict[str, Callable]] = None
    ) -> ProbeResult:
        """Run ablation probe."""
        config = AblationProbeConfig(
            model=self.model,
            output_dir=self.output_dir,
        )
        probe = AblationProbe(config)
        result = probe.run(self.data_loader, task_evaluators)
        probe.save_results()
        self.results["ablation"] = result
        return result

    def run_checkpoint_probe(self) -> Optional[ProbeResult]:
        """Run checkpoint comparison probe."""
        if self.checkpoint_dir is None:
            logger.warning("No checkpoint directory specified, skipping checkpoint probe")
            return None

        config = CheckpointProbeConfig(
            model=self.model,
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        probe = CheckpointComparisonProbe(config)
        result = probe.run(self.data_loader)
        probe.save_results()
        self.results["checkpoint"] = result
        return result

    def run_all(
        self,
        task_evaluators: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, ProbeResult]:
        """Run all probes and compile results."""
        logger.info("Running comprehensive fusion quality probe battery...")

        self.run_rsa_probe()
        self.run_attention_flow_probe()
        self.run_ablation_probe(task_evaluators)

        if self.checkpoint_dir:
            self.run_checkpoint_probe()

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive markdown report."""
        report = "# Cross-Modal Fusion Quality Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"

        # Executive Summary
        report += "## Executive Summary\n\n"

        if "rsa" in self.results:
            mean_rsa = self.results["rsa"].metrics.get("mean_rsa", 0)
            fusion_status = "EFFECTIVE" if mean_rsa > 0.3 else "WEAK" if mean_rsa > 0.15 else "MINIMAL"
            report += f"**Fusion Quality Status: {fusion_status}**\n\n"
            report += f"- Mean RSA Correlation: {mean_rsa:.4f}\n"

        if "attention_flow" in self.results:
            if "source_importance" in self.results["attention_flow"].metrics:
                ranking = self.results["attention_flow"].metrics["source_importance"].get("ranking", [])
                report += f"- Top Sources by Importance: {', '.join(ranking[:3])}\n"

        report += "\n---\n\n"

        # Individual probe reports
        for probe_name, result in self.results.items():
            report += f"## {result.probe_name} Results\n\n"
            report += result.interpretation + "\n\n"

            if result.recommendations:
                report += "### Recommendations\n\n"
                for rec in result.recommendations:
                    report += f"- {rec}\n"

            if result.artifacts:
                report += "\n### Generated Artifacts\n\n"
                for name, path in result.artifacts.items():
                    report += f"- {name}: `{path}`\n"

            report += "\n---\n\n"

        return report

    def save_report(self, filename: str = "fusion_quality_report.md") -> Path:
        """Save report to markdown file."""
        report = self.generate_report()
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {filepath}")
        return filepath


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_fusion_quality_probes(
    model: nn.Module,
    data_loader: Any,
    output_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    task_evaluators: Optional[Dict[str, Callable]] = None,
) -> Dict[str, ProbeResult]:
    """
    Convenience function to run all fusion quality probes.

    Args:
        model: Trained MultiResolutionHAN model
        data_loader: PyTorch DataLoader with test data
        output_dir: Directory for outputs (default: ./probes/outputs)
        checkpoint_dir: Directory with training checkpoints (optional)
        task_evaluators: Dict of task evaluation functions (optional)

    Returns:
        Dictionary of probe results

    Example:
        >>> from cross_modal_fusion_probes import run_fusion_quality_probes
        >>> results = run_fusion_quality_probes(
        ...     model=trained_model,
        ...     data_loader=test_loader,
        ...     checkpoint_dir=Path("./checkpoints"),
        ... )
        >>> print(results["rsa"].interpretation)
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    runner = CrossModalFusionProbeRunner(
        model=model,
        data_loader=data_loader,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )

    results = runner.run_all(task_evaluators)
    runner.save_report()

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Cross-Modal Fusion Quality Probes")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory for comparison")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--probe", type=str, default="all",
                       choices=["all", "rsa", "attention", "ablation", "checkpoint"],
                       help="Which probe to run")

    args = parser.parse_args()

    print("Cross-Modal Fusion Quality Probes")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Probe: {args.probe}")
    print("=" * 50)

    # Load model and data
    print("\nLoading model and data...")

    # Placeholder for actual model/data loading
    # In practice, this would be:
    # from multi_resolution_han import MultiResolutionHAN
    # from multi_resolution_data import create_dataloaders
    # model = MultiResolutionHAN.load(args.model_path)
    # data_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    print("\nNote: This script requires a trained model and data loader.")
    print("Example usage in Python:")
    print("""
    >>> from analysis.probes.cross_modal_fusion_probes import run_fusion_quality_probes
    >>> results = run_fusion_quality_probes(
    ...     model=your_trained_model,
    ...     data_loader=your_test_loader,
    ... )
    """)
