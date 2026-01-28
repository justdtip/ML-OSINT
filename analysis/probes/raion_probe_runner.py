#!/usr/bin/env python3
"""
Raion-Level Probe Runner for MultiResolutionHAN with Geographic Encoders

This module provides a specialized probe runner for raion-level checkpoints that use
GeographicDailyCrossSourceFusion for spatial attention. It handles:

1. Loading raion checkpoints with geoconfirmed_raion, personnel, sentinel sources
2. Creating the correct MultiResolutionConfig for these sources
3. Configuring GeographicSourceEncoder with custom_spatial_configs
4. Supporting the GeographicDailyCrossSourceFusion model architecture

Usage:
    from analysis.probes.raion_probe_runner import RaionProbeRunner

    runner = RaionProbeRunner(
        checkpoint_path="analysis/checkpoints/raion_training/run_20260128_001955/best_model.pt",
        device="cpu",
    )

    # Access model and dataloader
    model = runner.model
    dataloader = runner.dataloader

    # Run forward pass
    batch = next(iter(dataloader))
    outputs = runner.run_forward_pass(batch)

    # Get attention weights from geographic encoder
    attention_weights = runner.get_attention_weights(batch)

Author: ML Engineering Team
Date: 2026-01-28
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

# Centralized paths
from config.paths import (
    PROJECT_ROOT,
    ANALYSIS_DIR,
    CHECKPOINT_DIR,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_missing_values(
    features_dict: Dict[str, Tensor],
    missing_value: float = -999.0,
) -> Dict[str, Tensor]:
    """
    Replace missing value sentinel (-999.0) with 0.0 in feature tensors.

    The raion data uses -999.0 to indicate missing/unobserved values. This function
    replaces those sentinels with 0.0 to prevent extreme values from flowing
    through the model. The observation masks should be used to properly handle
    the missing data semantics.

    Args:
        features_dict: Dictionary mapping source names to feature tensors.
            Tensors have shape [batch, seq_len, n_features] or
            [batch, seq_len, n_raions, features_per_raion]
        missing_value: The sentinel value to replace (default: -999.0)

    Returns:
        Dictionary with cleaned feature tensors (sentinels replaced with 0.0)

    Example:
        >>> features = {'geoconfirmed_raion': torch.tensor([[[1.0, -999.0, 2.0]]])}
        >>> cleaned = clean_missing_values(features)
        >>> cleaned['geoconfirmed_raion']
        tensor([[[1., 0., 2.]]])
    """
    cleaned = {}
    for name, tensor in features_dict.items():
        # Clone to avoid modifying original
        cleaned_tensor = tensor.clone()
        # Replace missing values with 0.0 using torch.where for robustness
        # Use tolerance for floating point comparison
        is_missing = torch.abs(cleaned_tensor - missing_value) < 1.0
        cleaned_tensor = torch.where(is_missing, torch.zeros_like(cleaned_tensor), cleaned_tensor)
        # Also clamp any remaining extreme values
        cleaned_tensor = torch.clamp(cleaned_tensor, min=-100.0, max=100.0)
        cleaned[name] = cleaned_tensor
    return cleaned


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RaionProbeConfig:
    """Configuration for the raion probe runner."""

    # Checkpoint path
    checkpoint_path: Path = field(default_factory=lambda: Path(
        CHECKPOINT_DIR / "raion_training/run_20260128_001955/best_model.pt"
    ))

    # Device
    device: str = "cpu"

    # Data configuration
    batch_size: int = 4
    num_workers: int = 0

    # Sources expected in the checkpoint
    daily_sources: List[str] = field(default_factory=lambda: ["geoconfirmed_raion", "personnel"])
    monthly_sources: List[str] = field(default_factory=lambda: ["sentinel"])

    # Geoconfirmed raion configuration
    geoconfirmed_n_raions: int = 171
    geoconfirmed_features_per_raion: int = 50

    # Model architecture (from checkpoint)
    d_model: int = 64
    nhead: int = 4

    # Geographic prior
    use_geographic_prior: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs" / "probes" / "raion")

    def __post_init__(self):
        self.checkpoint_path = Path(self.checkpoint_path)
        self.output_dir = Path(self.output_dir)


# =============================================================================
# RAION PROBE RUNNER
# =============================================================================

class RaionProbeRunner:
    """
    Probe runner for raion-level MultiResolutionHAN checkpoints.

    Handles loading checkpoints that were trained with:
    - GeographicDailyCrossSourceFusion for daily sources
    - Custom spatial configs for geoconfirmed_raion (171 raions x 50 features)
    - Personnel and Sentinel sources

    Provides methods for:
    - Running forward passes through the model
    - Extracting daily encoder outputs
    - Extracting attention weights from geographic encoders

    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load model on ('cpu' or 'cuda')
        config: Optional RaionProbeConfig for additional settings

    Example:
        >>> runner = RaionProbeRunner(
        ...     checkpoint_path="analysis/checkpoints/raion_training/run_20260128_001955/best_model.pt"
        ... )
        >>> batch = next(iter(runner.dataloader))
        >>> outputs = runner.run_forward_pass(batch)
        >>> print(outputs['temporal_output'].shape)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        config: Optional[RaionProbeConfig] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Create or update config
        if config is None:
            self.config = RaionProbeConfig(
                checkpoint_path=self.checkpoint_path,
                device=device,
            )
        else:
            self.config = config
            self.config.checkpoint_path = self.checkpoint_path
            self.config.device = device

        # Validate checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Initialize model and data
        self._model: Optional[nn.Module] = None
        self._dataloader: Optional[DataLoader] = None
        self._dataset = None
        self._checkpoint_info: Dict[str, Any] = {}

        # Extracted model components (populated during setup)
        self._daily_fusion = None
        self._geographic_encoders: Dict[str, nn.Module] = {}
        self._raion_keys: List[str] = []

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint and create model
        self._load_checkpoint_and_model()

    @property
    def model(self) -> nn.Module:
        """Get the loaded MultiResolutionHAN model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_checkpoint_and_model() first.")
        return self._model

    @property
    def dataloader(self) -> DataLoader:
        """Get the DataLoader for the dataset."""
        if self._dataloader is None:
            raise RuntimeError("DataLoader not created. Call _load_checkpoint_and_model() first.")
        return self._dataloader

    @property
    def dataset(self):
        """Get the underlying dataset."""
        return self._dataset

    @property
    def checkpoint_info(self) -> Dict[str, Any]:
        """Get information extracted from the checkpoint."""
        return self._checkpoint_info

    def _load_checkpoint_and_model(self) -> None:
        """Load checkpoint and create model with correct configuration."""
        logger.info(f"Loading checkpoint from {self.checkpoint_path}")

        # Import required modules
        from analysis.multi_resolution_data import (
            MultiResolutionConfig,
            MultiResolutionDataset,
            multi_resolution_collate_fn,
        )
        from analysis.multi_resolution_han import MultiResolutionHAN, SourceConfig
        from analysis.geographic_source_encoder import SpatialSourceConfig

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Store checkpoint info
        self._checkpoint_info = {
            'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            'has_geographic_encoders': any('geographic_encoders' in k for k in state_dict.keys()),
        }

        # Detect model configuration from checkpoint
        config_dict = self._infer_config_from_checkpoint(state_dict)

        logger.info(f"Inferred config: d_model={config_dict['d_model']}, "
                   f"nhead={config_dict['nhead']}, "
                   f"n_daily_sources={config_dict['n_daily_sources']}")

        # Create data configuration
        data_config = MultiResolutionConfig(
            daily_sources=self.config.daily_sources,
            monthly_sources=self.config.monthly_sources,
            use_disaggregated_equipment=False,  # We use geoconfirmed_raion instead
            detrend_viirs=True,
        )

        # Create training dataset for normalization stats
        logger.info("Creating training dataset for normalization stats...")
        train_dataset = MultiResolutionDataset(data_config, split='train')
        norm_stats = train_dataset.norm_stats

        # Create test dataset for probing
        logger.info("Creating test dataset for probing...")
        self._dataset = MultiResolutionDataset(data_config, split='test', norm_stats=norm_stats)

        # Get feature dimensions from dataset sample
        sample = self._dataset[0]

        # Build source configs from actual data
        daily_source_configs = {}
        monthly_source_configs = {}

        for source_name in self.config.daily_sources:
            if source_name in sample.daily_features:
                n_features = sample.daily_features[source_name].shape[-1]
                daily_source_configs[source_name] = SourceConfig(
                    name=source_name,
                    n_features=n_features,
                    resolution='daily',
                )
                logger.info(f"  Daily source '{source_name}': {n_features} features")

        for source_name in self.config.monthly_sources:
            if source_name in sample.monthly_features:
                n_features = sample.monthly_features[source_name].shape[-1]
                monthly_source_configs[source_name] = SourceConfig(
                    name=source_name,
                    n_features=n_features,
                    resolution='monthly',
                )
                logger.info(f"  Monthly source '{source_name}': {n_features} features")

        # Create custom spatial config for geoconfirmed_raion
        custom_spatial_configs = {}
        if 'geoconfirmed_raion' in daily_source_configs:
            n_features = daily_source_configs['geoconfirmed_raion'].n_features
            # Calculate n_raions and features_per_raion
            # Total features = n_raions * features_per_raion
            # We know: 171 raions * 50 features = 8550
            n_raions = self.config.geoconfirmed_n_raions
            features_per_raion = self.config.geoconfirmed_features_per_raion

            # Verify feature count matches
            expected_features = n_raions * features_per_raion
            if n_features != expected_features:
                logger.warning(
                    f"geoconfirmed_raion features mismatch: got {n_features}, "
                    f"expected {expected_features} ({n_raions} raions x {features_per_raion} features). "
                    f"Adjusting n_raions..."
                )
                # Recalculate n_raions based on actual features
                n_raions = n_features // features_per_raion
                if n_features % features_per_raion != 0:
                    logger.warning(f"Features don't divide evenly. Using n_raions={n_raions}")

            custom_spatial_configs['geoconfirmed_raion'] = SpatialSourceConfig(
                name='geoconfirmed_raion',
                n_raions=n_raions,
                features_per_raion=features_per_raion,
                use_geographic_prior=self.config.use_geographic_prior,
            )
            logger.info(f"  Created SpatialSourceConfig: {n_raions} raions x {features_per_raion} features")

        # Create model
        logger.info("Creating MultiResolutionHAN model...")
        self._model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=config_dict['d_model'],
            nhead=config_dict['nhead'],
            num_daily_layers=config_dict.get('num_daily_layers', 1),  # Match training script
            num_monthly_layers=config_dict.get('num_monthly_layers', 1),
            num_fusion_layers=config_dict.get('num_fusion_layers', 1),
            num_temporal_layers=2,
            dropout=0.0,  # No dropout for inference
            use_geographic_prior=self.config.use_geographic_prior,
            custom_spatial_configs=custom_spatial_configs,
        )

        # Load state dict
        load_result = self._model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            logger.warning(f"Missing keys: {len(load_result.missing_keys)}")
            if len(load_result.missing_keys) <= 10:
                logger.warning(f"  {load_result.missing_keys}")
            else:
                logger.warning(f"  First 10: {load_result.missing_keys[:10]}")

        if load_result.unexpected_keys:
            logger.warning(f"Unexpected keys: {len(load_result.unexpected_keys)}")
            if len(load_result.unexpected_keys) <= 10:
                logger.warning(f"  {load_result.unexpected_keys}")
            else:
                logger.warning(f"  First 10: {load_result.unexpected_keys[:10]}")

        self._model.to(self.device)
        self._model.eval()

        # Extract model components for probing
        self._extract_components()

        # Create dataloader
        self._dataloader = DataLoader(
            self._dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=multi_resolution_collate_fn,
        )

        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"Model loaded: {n_params:,} parameters")
        logger.info(f"DataLoader created: {len(self._dataloader)} batches")

    def _infer_config_from_checkpoint(self, state_dict: Dict[str, Tensor]) -> Dict[str, Any]:
        """
        Infer model configuration from checkpoint tensor shapes.

        Args:
            state_dict: The model state dictionary

        Returns:
            Dictionary with inferred configuration values
        """
        config = {
            'd_model': self.config.d_model,
            'nhead': self.config.nhead,
            'n_daily_sources': 2,  # Default for geoconfirmed_raion + personnel
        }

        # Infer d_model from source_type_embedding if available
        if 'daily_fusion.source_type_embedding.weight' in state_dict:
            embedding_shape = state_dict['daily_fusion.source_type_embedding.weight'].shape
            config['n_daily_sources'] = embedding_shape[0]
            config['d_model'] = embedding_shape[1]
            logger.info(f"Inferred from daily_fusion.source_type_embedding: "
                       f"{config['n_daily_sources']} sources, d_model={config['d_model']}")

        # Infer nhead from d_model (common ratios)
        if config['d_model'] == 128:
            config['nhead'] = 8
        elif config['d_model'] == 64:
            config['nhead'] = 4

        # Check for geographic encoders
        has_geo_encoders = any('geographic_encoders' in k for k in state_dict.keys())
        config['has_geographic_encoders'] = has_geo_encoders

        if has_geo_encoders:
            # Find which sources have geographic encoders
            geo_sources = set()
            for key in state_dict.keys():
                if 'geographic_encoders.' in key:
                    parts = key.split('.')
                    idx = parts.index('geographic_encoders') + 1
                    if idx < len(parts):
                        geo_sources.add(parts[idx])
            config['geographic_encoder_sources'] = list(geo_sources)
            logger.info(f"Found geographic encoders for: {geo_sources}")

        return config

    def _extract_components(self) -> None:
        """Extract key components from model for probing."""
        if self._model is None:
            return

        # Import for type checking
        from analysis.geographic_source_encoder import GeographicDailyCrossSourceFusion

        # Find daily_fusion (GeographicDailyCrossSourceFusion)
        for name, module in self._model.named_modules():
            if isinstance(module, GeographicDailyCrossSourceFusion):
                self._daily_fusion = module

                # Extract geographic encoders
                if hasattr(module, 'geographic_encoders'):
                    for enc_name, encoder in module.geographic_encoders.items():
                        self._geographic_encoders[enc_name] = encoder

                        # Get raion info from config
                        if hasattr(encoder, 'config') and encoder.config.raion_keys:
                            self._raion_keys = encoder.config.raion_keys
                break

    def run_forward_pass(
        self,
        batch: Any,
        return_attention: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Run a forward pass through the model.

        Args:
            batch: A batch from the dataloader (MultiResolutionBatch)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing model outputs:
                - 'temporal_output': Latent representation [batch, seq_len, d_model]
                - 'casualty_pred': Casualty predictions if enabled
                - 'regime_logits': Regime classification logits if enabled
                - 'anomaly_score': Anomaly scores if enabled
                - 'forecast_pred': Monthly forecast predictions if enabled
                - 'daily_forecast_pred': Daily forecast predictions if enabled
                - 'source_importance': Source importance weights
        """
        self._model.eval()

        with torch.no_grad():
            # Handle both dict and namedtuple batch formats
            if isinstance(batch, dict):
                daily_features_raw = batch['daily_features']
                daily_masks_raw = batch['daily_masks']
                monthly_features_raw = batch['monthly_features']
                monthly_masks_raw = batch['monthly_masks']
                month_boundaries_raw = batch.get('month_boundary_indices', batch.get('month_boundaries'))
                raion_masks_raw = batch.get('raion_masks')
            else:
                daily_features_raw = batch.daily_features
                daily_masks_raw = batch.daily_masks
                monthly_features_raw = batch.monthly_features
                monthly_masks_raw = batch.monthly_masks
                month_boundaries_raw = getattr(batch, 'month_boundaries', getattr(batch, 'month_boundary_indices', None))
                raion_masks_raw = getattr(batch, 'raion_masks', None)

            # Move batch data to device
            daily_features = {
                k: v.to(self.device) for k, v in daily_features_raw.items()
            }
            daily_masks = {
                k: v.to(self.device) for k, v in daily_masks_raw.items()
            }
            monthly_features = {
                k: v.to(self.device) for k, v in monthly_features_raw.items()
            }
            monthly_masks = {
                k: v.to(self.device) for k, v in monthly_masks_raw.items()
            }
            month_boundaries = month_boundaries_raw.to(self.device)
            raion_masks = None
            if raion_masks_raw is not None:
                raion_masks = {k: v.to(self.device) for k, v in raion_masks_raw.items()}

            # Clean missing values
            daily_features = clean_missing_values(daily_features)
            monthly_features = clean_missing_values(monthly_features)

            # Forward pass
            outputs = self._model(
                daily_features=daily_features,
                daily_masks=daily_masks,
                monthly_features=monthly_features,
                monthly_masks=monthly_masks,
                month_boundaries=month_boundaries,
                raion_masks=raion_masks,
            )

        return outputs

    def get_daily_encoded(
        self,
        batch: Any,
    ) -> Dict[str, Tensor]:
        """
        Get the daily encoder outputs for each source before fusion.

        This is useful for probing what each source contributes independently.

        Args:
            batch: A batch from the dataloader

        Returns:
            Dictionary mapping source names to encoded tensors [batch, seq_len, d_model]
        """
        self._model.eval()

        with torch.no_grad():
            # Handle both dict and namedtuple batch formats
            if isinstance(batch, dict):
                daily_features_raw = batch['daily_features']
                daily_masks_raw = batch['daily_masks']
            else:
                daily_features_raw = batch.daily_features
                daily_masks_raw = batch.daily_masks

            # Move batch data to device
            daily_features = {
                k: v.to(self.device) for k, v in daily_features_raw.items()
            }
            daily_masks = {
                k: v.to(self.device) for k, v in daily_masks_raw.items()
            }

            # Clean missing values
            daily_features = clean_missing_values(daily_features)

            # Encode each source
            daily_encoded = {}
            for name in self._model.daily_source_names:
                if name in daily_features:
                    encoded, _ = self._model.daily_encoders[name](
                        daily_features[name],
                        daily_masks[name],
                        return_attention=False,
                    )
                    daily_encoded[name] = encoded

        return daily_encoded

    def get_attention_weights(
        self,
        batch: Any,
    ) -> Dict[str, Tensor]:
        """
        Get attention weights from the geographic encoder and cross-source fusion.

        This probes how the model attends across raions and sources.

        Args:
            batch: A batch from the dataloader

        Returns:
            Dictionary containing attention weights:
                - 'source_importance': [batch, seq_len, n_sources] source weights
                - 'geographic_cross_raion': Per-source cross-raion attention (if available)
        """
        self._model.eval()

        with torch.no_grad():
            # Handle both dict and namedtuple batch formats
            if isinstance(batch, dict):
                daily_features_raw = batch['daily_features']
                daily_masks_raw = batch['daily_masks']
            else:
                daily_features_raw = batch.daily_features
                daily_masks_raw = batch.daily_masks

            # Move batch data to device
            daily_features = {
                k: v.to(self.device) for k, v in daily_features_raw.items()
            }
            daily_masks = {
                k: v.to(self.device) for k, v in daily_masks_raw.items()
            }

            # Clean missing values
            daily_features = clean_missing_values(daily_features)

            attention_weights = {}

            # Get raion masks if available
            raion_masks = None
            if isinstance(batch, dict):
                raion_masks_raw = batch.get('raion_masks')
            else:
                raion_masks_raw = getattr(batch, 'raion_masks', None)
            if raion_masks_raw is not None:
                raion_masks = {k: v.to(self.device) for k, v in raion_masks_raw.items()}

            # Check if we have geographic fusion
            if hasattr(self._model.daily_fusion, 'geographic_encoders'):
                geo_encoder_names = set(self._model.daily_fusion.geographic_encoders.keys())

                # For geographic fusion, we need to:
                # 1. Pre-encode non-geographic sources through daily_encoders (model's design)
                # 2. Pass raw features for geographic sources (so geographic_encoders in fusion run)
                features_for_fusion = {}
                for name in self._model.daily_source_names:
                    if name not in daily_features:
                        continue

                    if name in geo_encoder_names:
                        # Pass raw features for geographic sources
                        features_for_fusion[name] = daily_features[name]
                    else:
                        # Pre-encode non-geographic sources through daily_encoders
                        encoded, attn = self._model.daily_encoders[name](
                            daily_features[name],
                            daily_masks[name],
                            return_attention=True,
                        )
                        features_for_fusion[name] = encoded
                        if attn is not None:
                            attention_weights[f'{name}_encoder_attention'] = attn

                # Now call fusion with mixed features
                fused, _, fusion_attn = self._model.daily_fusion(
                    features_for_fusion,
                    daily_masks,
                    return_attention=True,
                    raion_masks=raion_masks,
                )
                # Collect all attention weights from fusion
                attention_weights.update(fusion_attn)
            else:
                # Standard fusion - encode first then fuse
                daily_encoded = {}
                for name in self._model.daily_source_names:
                    if name in daily_features:
                        encoded, attn = self._model.daily_encoders[name](
                            daily_features[name],
                            daily_masks[name],
                            return_attention=True,
                        )
                        daily_encoded[name] = encoded
                        if attn is not None:
                            attention_weights[f'{name}_encoder_attention'] = attn

                fused, _, fusion_attn = self._model.daily_fusion(
                    daily_encoded,
                    daily_masks,
                    return_attention=True,
                )
                if 'source_importance' in fusion_attn:
                    attention_weights['source_importance'] = fusion_attn['source_importance']

        return attention_weights

    def get_geographic_encoder_info(self) -> Dict[str, Any]:
        """
        Get information about the geographic encoders in the model.

        Returns:
            Dictionary with geographic encoder configuration:
                - 'has_geographic_encoders': Whether the model uses geographic fusion
                - 'sources': List of sources with geographic encoders
                - 'configs': Per-source configuration details
        """
        info = {
            'has_geographic_encoders': False,
            'sources': [],
            'configs': {},
        }

        if hasattr(self._model, 'daily_fusion'):
            fusion = self._model.daily_fusion
            if hasattr(fusion, 'geographic_encoders'):
                info['has_geographic_encoders'] = True
                info['sources'] = list(fusion.geographic_encoders.keys())

                for name, encoder in fusion.geographic_encoders.items():
                    info['configs'][name] = {
                        'n_raions': encoder.n_raions,
                        'features_per_raion': encoder.features_per_raion,
                        'd_model': encoder.d_model,
                        'use_geographic_prior': encoder.config.use_geographic_prior,
                    }

        return info

    def save_results(
        self,
        results: Dict[str, Any],
        probe_name: str,
    ) -> Path:
        """
        Save probe results to output directory.

        Args:
            results: Results dictionary
            probe_name: Name of the probe

        Returns:
            Path to saved results
        """
        output_path = self.config.output_dir / f"{probe_name}_results.json"

        # Convert numpy arrays and tensors to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        serializable = convert(results)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Saved results to {output_path}")
        return output_path


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import sys

    print("Raion Probe Runner Test")
    print("=" * 60)

    # Default checkpoint path
    default_checkpoint = CHECKPOINT_DIR / "raion_training/run_20260128_001955/best_model.pt"

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else str(default_checkpoint)

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please provide a valid checkpoint path as argument")
        sys.exit(1)

    print(f"\nLoading checkpoint: {checkpoint_path}")

    try:
        runner = RaionProbeRunner(
            checkpoint_path=checkpoint_path,
            device="cpu",
        )

        print(f"\nModel loaded successfully")
        print(f"  Daily sources: {runner.model.daily_source_names}")
        print(f"  Monthly sources: {runner.model.monthly_source_names}")

        # Test geographic encoder info
        geo_info = runner.get_geographic_encoder_info()
        print(f"\nGeographic encoder info:")
        print(f"  Has geographic encoders: {geo_info['has_geographic_encoders']}")
        if geo_info['has_geographic_encoders']:
            print(f"  Sources: {geo_info['sources']}")
            for name, config in geo_info['configs'].items():
                print(f"    {name}: {config['n_raions']} raions x {config['features_per_raion']} features")

        # Test forward pass
        print(f"\nTesting forward pass...")
        batch = next(iter(runner.dataloader))
        outputs = runner.run_forward_pass(batch)

        print(f"  temporal_output shape: {outputs['temporal_output'].shape}")
        if 'casualty_pred' in outputs:
            casualty = outputs['casualty_pred']
            if isinstance(casualty, tuple):
                print(f"  casualty_pred shape: {casualty[0].shape}")
            else:
                print(f"  casualty_pred shape: {casualty.shape}")
        if 'source_importance' in outputs:
            print(f"  source_importance shape: {outputs['source_importance'].shape}")

        # Test daily encoded
        print(f"\nTesting daily encoder outputs...")
        daily_encoded = runner.get_daily_encoded(batch)
        for name, tensor in daily_encoded.items():
            print(f"  {name}: {tensor.shape}")

        # Test attention weights
        print(f"\nTesting attention weight extraction...")
        attention = runner.get_attention_weights(batch)
        for name, tensor in attention.items():
            print(f"  {name}: {tensor.shape}")

        print("\n" + "=" * 60)
        print("All tests passed!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
