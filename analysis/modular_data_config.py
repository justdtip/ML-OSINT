"""
Modular Data Configuration for Multi-Resolution HAN Training Pipeline.

This module provides a flexible configuration system for selectively enabling/disabling
data sources and configuring spatial feature modes for the Multi-Resolution HAN model.

Key Features:
- DataSourceConfig: Per-source configuration with enable/disable, spatial mode, temporal resolution
- Spatial modes: 'aggregated' (current behavior), 'tiled' (per-region), 'full' (raw lat/lon)
- Presets: 'baseline' (current behavior), 'spatial_rich' (all spatial features), ablations
- Backward compatible: default config reproduces current behavior

Usage:
    from modular_data_config import get_data_source_config, ModularDataConfig

    # Use a preset
    config = get_data_source_config('baseline')

    # Create custom config
    config = ModularDataConfig(
        viirs=DataSourceEntry(enabled=True, spatial_mode='tiled'),
        firms=DataSourceEntry(enabled=True, spatial_mode='binned'),
    )

    # Pass to dataset
    dataset = MultiResolutionDataset(
        config=multi_res_config,
        data_source_config=config,
    )

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Union

# Import centralized paths
from config.paths import (
    PROJECT_ROOT, DATA_DIR, VIIRS_DIR, FIRMS_DIR, DEEPSTATE_DIR,
    OUTPUT_DIR, ensure_dir,
)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class SpatialMode(str, Enum):
    """Spatial feature aggregation modes.

    Attributes:
        AGGREGATED: Aggregate all spatial data into a single feature vector (current behavior).
            Best for: Baseline models, computational efficiency, when spatial resolution
            doesn't matter.

        TILED: Divide spatial data into regional tiles (e.g., 6 VIIRS tiles, 5 FIRMS grid cells).
            Best for: Capturing regional variation, multi-resolution spatial analysis,
            attention-based spatial learning.

        FULL: Use raw lat/lon coordinates as features (high-dimensional).
            Best for: Fine-grained spatial analysis, GNN integration, spatial attention
            mechanisms. Warning: Significantly increases feature dimensionality.

        DISABLED: Do not load any spatial features from this source.
            Best for: Ablation studies, computational efficiency.
    """
    AGGREGATED = 'aggregated'
    TILED = 'tiled'
    FULL = 'full'
    DISABLED = 'disabled'


class TemporalResolution(str, Enum):
    """Temporal resolution for data sources."""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'


# =============================================================================
# SPATIAL REGION DEFINITIONS
# =============================================================================

# VIIRS tile definitions (6 regions based on Ukrainian oblasts)
# These correspond to major conflict zones for spatial disaggregation
VIIRS_TILES: Dict[str, Dict[str, Any]] = {
    'east': {
        'name': 'Eastern Ukraine (Donbas)',
        'oblasts': ['Donetsk', 'Luhansk'],
        'lat_range': (47.5, 50.0),
        'lon_range': (37.0, 40.5),
    },
    'south': {
        'name': 'Southern Ukraine',
        'oblasts': ['Kherson', 'Zaporizhzhia'],
        'lat_range': (45.5, 48.0),
        'lon_range': (33.5, 37.0),
    },
    'northeast': {
        'name': 'Northeastern Ukraine',
        'oblasts': ['Kharkiv', 'Sumy'],
        'lat_range': (49.0, 52.0),
        'lon_range': (34.0, 38.5),
    },
    'central': {
        'name': 'Central Ukraine',
        'oblasts': ['Dnipropetrovsk', 'Poltava', 'Kirovohrad'],
        'lat_range': (48.0, 50.0),
        'lon_range': (32.0, 36.0),
    },
    'west': {
        'name': 'Western Ukraine',
        'oblasts': ['Lviv', 'Ivano-Frankivsk', 'Ternopil', 'Volyn', 'Rivne'],
        'lat_range': (48.0, 52.0),
        'lon_range': (22.0, 28.0),
    },
    'kyiv_region': {
        'name': 'Kyiv Region',
        'oblasts': ['Kyiv', 'Kyiv Oblast', 'Chernihiv', 'Zhytomyr'],
        'lat_range': (49.5, 52.5),
        'lon_range': (28.0, 33.0),
    },
}

# FIRMS grid definitions (5x5 grid cells for spatial binning)
FIRMS_GRID_CELLS: int = 25  # 5x5 grid

# DeepState territorial control categories
DEEPSTATE_CONTROL_TYPES: List[str] = [
    'russian_controlled',
    'ukrainian_controlled',
    'contested',
    'liberated',
    'occupied_since_2014',
]


# =============================================================================
# DATA SOURCE ENTRY CONFIGURATION
# =============================================================================

@dataclass
class DataSourceEntry:
    """Configuration for a single data source.

    Attributes:
        enabled: Whether this data source is enabled for training.
        spatial_mode: How to handle spatial features ('aggregated', 'tiled', 'full', 'disabled').
        temporal_resolution: Override temporal resolution (None = use source default).
        weight: Relative importance weight for this source in loss calculation (default 1.0).
        features: Optional list of specific features to include (None = all features).
        detrend: Apply first-order differencing to remove trends (useful for VIIRS).
        lag_days: Apply temporal lag in days (useful for leading indicators).
        normalize_method: Normalization method ('standard', 'minmax', 'robust', None).

    Example:
        >>> viirs_config = DataSourceEntry(
        ...     enabled=True,
        ...     spatial_mode='tiled',
        ...     detrend=True,
        ... )
    """
    enabled: bool = True
    spatial_mode: SpatialMode = SpatialMode.AGGREGATED
    temporal_resolution: Optional[TemporalResolution] = None
    weight: float = 1.0
    features: Optional[List[str]] = None
    detrend: bool = False
    lag_days: int = 0
    normalize_method: Optional[str] = 'standard'

    def __post_init__(self) -> None:
        """Validate and convert string enums."""
        # Convert string to enum if needed
        if isinstance(self.spatial_mode, str):
            self.spatial_mode = SpatialMode(self.spatial_mode)
        if isinstance(self.temporal_resolution, str):
            self.temporal_resolution = TemporalResolution(self.temporal_resolution)

        # Validate weight
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")

        # Validate lag_days
        if self.lag_days < 0:
            raise ValueError(f"lag_days must be non-negative, got {self.lag_days}")

    @property
    def is_spatial_enabled(self) -> bool:
        """Check if spatial features are enabled (not disabled or aggregated)."""
        return self.enabled and self.spatial_mode not in (SpatialMode.DISABLED, SpatialMode.AGGREGATED)

    @property
    def n_spatial_tiles(self) -> int:
        """Get number of spatial tiles based on mode."""
        if self.spatial_mode == SpatialMode.TILED:
            return len(VIIRS_TILES)  # 6 tiles
        elif self.spatial_mode == SpatialMode.FULL:
            return -1  # Variable, depends on data
        return 1  # Aggregated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['spatial_mode'] = self.spatial_mode.value
        if self.temporal_resolution:
            result['temporal_resolution'] = self.temporal_resolution.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSourceEntry':
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# MODULAR DATA CONFIGURATION
# =============================================================================

@dataclass
class ModularDataConfig:
    """Complete modular data configuration for Multi-Resolution HAN.

    This configuration allows fine-grained control over which data sources
    are enabled and how their spatial features are processed.

    Attributes:
        # Daily sources (with equipment disaggregation)
        equipment: Aggregated equipment losses (legacy, disabled by default when disaggregated)
        drones: Drone/UAV losses (highest predictive signal per probe analysis)
        armor: Tank and APC losses
        artillery: Field artillery, MRL, AA, SRBM losses
        aircraft: Fixed-wing and helicopter losses
        personnel: Personnel/POW losses
        deepstate: DeepState territorial control
        firms: NASA FIRMS fire hotspots
        viina: VIINA conflict events
        viirs: VIIRS nighttime brightness

        # Monthly sources
        sentinel: Sentinel satellite observations
        hdx_conflict: HDX conflict events
        hdx_food: HDX food prices
        hdx_rainfall: HDX rainfall data
        iom: IOM displacement tracking

        # Global settings
        use_disaggregated_equipment: Use separate drones/armor/artillery instead of aggregated
        spatial_features_enabled: Global toggle for spatial features
        config_name: Name identifier for this configuration
        description: Human-readable description

    Example:
        >>> config = ModularDataConfig(
        ...     viirs=DataSourceEntry(enabled=True, spatial_mode='tiled', detrend=True),
        ...     firms=DataSourceEntry(enabled=True, spatial_mode='tiled'),
        ...     config_name='spatial_rich',
        ... )
    """
    # Daily sources - Equipment (disaggregated by default per probe findings)
    equipment: DataSourceEntry = field(default_factory=lambda: DataSourceEntry(enabled=False))
    drones: DataSourceEntry = field(default_factory=DataSourceEntry)
    armor: DataSourceEntry = field(default_factory=DataSourceEntry)
    artillery: DataSourceEntry = field(default_factory=DataSourceEntry)
    aircraft: DataSourceEntry = field(default_factory=lambda: DataSourceEntry(enabled=False))  # Negative correlation

    # Daily sources - Other
    personnel: DataSourceEntry = field(default_factory=DataSourceEntry)
    deepstate: DataSourceEntry = field(default_factory=DataSourceEntry)
    firms: DataSourceEntry = field(default_factory=DataSourceEntry)
    viina: DataSourceEntry = field(default_factory=DataSourceEntry)
    viirs: DataSourceEntry = field(default_factory=lambda: DataSourceEntry(detrend=True))

    # Spatial sources (daily resolution, rich spatial features)
    deepstate_spatial: DataSourceEntry = field(default_factory=lambda: DataSourceEntry(
        enabled=False,  # Off by default, enable for spatial experiments
        spatial_mode=SpatialMode.TILED,
    ))
    firms_spatial: DataSourceEntry = field(default_factory=lambda: DataSourceEntry(
        enabled=False,  # Off by default, enable for spatial experiments
        spatial_mode=SpatialMode.TILED,
    ))

    # Monthly sources
    sentinel: DataSourceEntry = field(default_factory=DataSourceEntry)
    hdx_conflict: DataSourceEntry = field(default_factory=DataSourceEntry)
    hdx_food: DataSourceEntry = field(default_factory=DataSourceEntry)
    hdx_rainfall: DataSourceEntry = field(default_factory=DataSourceEntry)
    iom: DataSourceEntry = field(default_factory=DataSourceEntry)

    # Global settings
    use_disaggregated_equipment: bool = True
    spatial_features_enabled: bool = True
    config_name: str = 'default'
    description: str = ''

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # If using disaggregated equipment, ensure equipment is disabled
        if self.use_disaggregated_equipment and self.equipment.enabled:
            import warnings
            warnings.warn(
                "use_disaggregated_equipment=True but equipment is enabled. "
                "Disabling aggregated equipment to avoid duplication."
            )
            self.equipment = DataSourceEntry(enabled=False)

    @property
    def daily_sources(self) -> Dict[str, DataSourceEntry]:
        """Get all daily source configurations."""
        sources = {
            'personnel': self.personnel,
            'deepstate': self.deepstate,
            'firms': self.firms,
            'viina': self.viina,
            'viirs': self.viirs,
        }

        # Add equipment sources based on disaggregation setting
        if self.use_disaggregated_equipment:
            sources.update({
                'drones': self.drones,
                'armor': self.armor,
                'artillery': self.artillery,
                'aircraft': self.aircraft,
            })
        else:
            sources['equipment'] = self.equipment

        return sources

    @property
    def spatial_daily_sources(self) -> Dict[str, DataSourceEntry]:
        """Get spatial-enriched daily source configurations (unit tracking, fire hotspots)."""
        return {
            'deepstate_spatial': self.deepstate_spatial,
            'firms_spatial': self.firms_spatial,
        }

    @property
    def monthly_sources(self) -> Dict[str, DataSourceEntry]:
        """Get all monthly source configurations."""
        return {
            'sentinel': self.sentinel,
            'hdx_conflict': self.hdx_conflict,
            'hdx_food': self.hdx_food,
            'hdx_rainfall': self.hdx_rainfall,
            'iom': self.iom,
        }

    @property
    def enabled_daily_sources(self) -> List[str]:
        """Get list of enabled daily source names."""
        return [name for name, cfg in self.daily_sources.items() if cfg.enabled]

    @property
    def enabled_spatial_sources(self) -> List[str]:
        """Get list of enabled spatial source names."""
        return [name for name, cfg in self.spatial_daily_sources.items() if cfg.enabled]

    @property
    def enabled_monthly_sources(self) -> List[str]:
        """Get list of enabled monthly source names."""
        return [name for name, cfg in self.monthly_sources.items() if cfg.enabled]

    @property
    def all_sources(self) -> Dict[str, DataSourceEntry]:
        """Get all source configurations (daily + monthly + spatial)."""
        return {**self.daily_sources, **self.monthly_sources, **self.spatial_daily_sources}

    @property
    def spatial_sources(self) -> Dict[str, DataSourceEntry]:
        """Get sources with spatial features enabled (tiled or full mode)."""
        if not self.spatial_features_enabled:
            return {}
        sources = {}
        # Check regular sources for spatial mode
        for name, cfg in {**self.daily_sources, **self.monthly_sources}.items():
            if cfg.enabled and cfg.is_spatial_enabled:
                sources[name] = cfg
        # Add dedicated spatial sources
        for name, cfg in self.spatial_daily_sources.items():
            if cfg.enabled:
                sources[name] = cfg
        return sources

    def get_source_config(self, source_name: str) -> Optional[DataSourceEntry]:
        """Get configuration for a specific source."""
        return self.all_sources.get(source_name)

    def is_source_enabled(self, source_name: str) -> bool:
        """Check if a source is enabled."""
        cfg = self.get_source_config(source_name)
        return cfg.enabled if cfg else False

    def get_spatial_mode(self, source_name: str) -> SpatialMode:
        """Get spatial mode for a source."""
        cfg = self.get_source_config(source_name)
        if cfg and self.spatial_features_enabled:
            return cfg.spatial_mode
        return SpatialMode.AGGREGATED

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        result = {
            'config_name': self.config_name,
            'description': self.description,
            'use_disaggregated_equipment': self.use_disaggregated_equipment,
            'spatial_features_enabled': self.spatial_features_enabled,
            'daily_sources': {
                name: cfg.to_dict() for name, cfg in self.daily_sources.items()
            },
            'monthly_sources': {
                name: cfg.to_dict() for name, cfg in self.monthly_sources.items()
            },
            'spatial_sources': {
                name: cfg.to_dict() for name, cfg in self.spatial_daily_sources.items()
            },
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModularDataConfig':
        """Create configuration from dictionary."""
        # Extract global settings
        config_name = data.get('config_name', 'loaded')
        description = data.get('description', '')
        use_disaggregated_equipment = data.get('use_disaggregated_equipment', True)
        spatial_features_enabled = data.get('spatial_features_enabled', True)

        # Build source configs
        kwargs = {
            'config_name': config_name,
            'description': description,
            'use_disaggregated_equipment': use_disaggregated_equipment,
            'spatial_features_enabled': spatial_features_enabled,
        }

        # Parse daily sources
        daily_sources = data.get('daily_sources', {})
        for name, cfg_dict in daily_sources.items():
            if name in ['equipment', 'drones', 'armor', 'artillery', 'aircraft',
                        'personnel', 'deepstate', 'firms', 'viina', 'viirs']:
                kwargs[name] = DataSourceEntry.from_dict(cfg_dict)

        # Parse monthly sources
        monthly_sources = data.get('monthly_sources', {})
        for name, cfg_dict in monthly_sources.items():
            if name in ['sentinel', 'hdx_conflict', 'hdx_food', 'hdx_rainfall', 'iom']:
                kwargs[name] = DataSourceEntry.from_dict(cfg_dict)

        # Parse spatial sources
        spatial_sources = data.get('spatial_sources', {})
        for name, cfg_dict in spatial_sources.items():
            if name in ['deepstate_spatial', 'firms_spatial']:
                kwargs[name] = DataSourceEntry.from_dict(cfg_dict)

        return cls(**kwargs)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModularDataConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Return formatted string representation."""
        lines = [
            f"ModularDataConfig: {self.config_name}",
            f"  Description: {self.description or '(none)'}",
            f"  Disaggregated Equipment: {self.use_disaggregated_equipment}",
            f"  Spatial Features: {self.spatial_features_enabled}",
            "",
            "  Enabled Daily Sources:",
        ]

        for name in self.enabled_daily_sources:
            cfg = self.daily_sources[name]
            spatial = cfg.spatial_mode.value if cfg.is_spatial_enabled else 'none'
            extras = []
            if cfg.detrend:
                extras.append('detrend')
            if cfg.lag_days > 0:
                extras.append(f'lag={cfg.lag_days}d')
            extra_str = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"    - {name}: spatial={spatial}{extra_str}")

        lines.append("")
        lines.append("  Enabled Monthly Sources:")
        for name in self.enabled_monthly_sources:
            cfg = self.monthly_sources[name]
            lines.append(f"    - {name}")

        if self.spatial_sources:
            lines.append("")
            lines.append("  Sources with Spatial Features:")
            for name, cfg in self.spatial_sources.items():
                lines.append(f"    - {name}: {cfg.spatial_mode.value}")

        return "\n".join(lines)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def _create_baseline_config() -> ModularDataConfig:
    """Create baseline configuration matching current behavior."""
    return ModularDataConfig(
        config_name='baseline',
        description='Baseline configuration matching current multi_resolution_data.py behavior',
        use_disaggregated_equipment=True,
        spatial_features_enabled=False,
        # All defaults except explicitly disabled sources
        drones=DataSourceEntry(enabled=True),
        armor=DataSourceEntry(enabled=True),
        artillery=DataSourceEntry(enabled=True),
        aircraft=DataSourceEntry(enabled=False),  # Negative correlation
        personnel=DataSourceEntry(enabled=True),
        deepstate=DataSourceEntry(enabled=True),
        firms=DataSourceEntry(enabled=True),
        viina=DataSourceEntry(enabled=True),
        viirs=DataSourceEntry(enabled=True, detrend=True),
        sentinel=DataSourceEntry(enabled=True),
        hdx_conflict=DataSourceEntry(enabled=True),
        hdx_food=DataSourceEntry(enabled=True),
        hdx_rainfall=DataSourceEntry(enabled=True),
        iom=DataSourceEntry(enabled=True),
    )


def _create_spatial_rich_config() -> ModularDataConfig:
    """Create configuration with all spatial features enabled including unit tracking."""
    return ModularDataConfig(
        config_name='spatial_rich',
        description='Full spatial features: unit positions, frontlines, fire hotspots per region',
        use_disaggregated_equipment=True,
        spatial_features_enabled=True,
        # Enable spatial modes for basic sources
        viirs=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,  # 6 regional tiles
            detrend=True,
        ),
        firms=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,  # Regional binning
        ),
        deepstate=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,  # Territorial polygons per region
        ),
        # Enable dedicated spatial sources (unit tracking, detailed fire analysis)
        deepstate_spatial=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,  # Unit positions, frontline metrics per region
        ),
        firms_spatial=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,  # Fire hotspot features per region
        ),
        # Other sources with defaults
        drones=DataSourceEntry(enabled=True),
        armor=DataSourceEntry(enabled=True),
        artillery=DataSourceEntry(enabled=True),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=True),
        viina=DataSourceEntry(enabled=True),
        sentinel=DataSourceEntry(enabled=True),
        hdx_conflict=DataSourceEntry(enabled=True),
        hdx_food=DataSourceEntry(enabled=True),
        hdx_rainfall=DataSourceEntry(enabled=True),
        iom=DataSourceEntry(enabled=True),
    )


def _create_viirs_only_config() -> ModularDataConfig:
    """Create ablation config with only VIIRS enabled."""
    return ModularDataConfig(
        config_name='ablation_viirs_only',
        description='Ablation: Only VIIRS data source enabled',
        use_disaggregated_equipment=False,
        spatial_features_enabled=False,
        equipment=DataSourceEntry(enabled=False),
        drones=DataSourceEntry(enabled=False),
        armor=DataSourceEntry(enabled=False),
        artillery=DataSourceEntry(enabled=False),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=False),
        deepstate=DataSourceEntry(enabled=False),
        firms=DataSourceEntry(enabled=False),
        viina=DataSourceEntry(enabled=False),
        viirs=DataSourceEntry(enabled=True, detrend=True),
        sentinel=DataSourceEntry(enabled=False),
        hdx_conflict=DataSourceEntry(enabled=False),
        hdx_food=DataSourceEntry(enabled=False),
        hdx_rainfall=DataSourceEntry(enabled=False),
        iom=DataSourceEntry(enabled=False),
    )


def _create_no_viirs_config() -> ModularDataConfig:
    """Create ablation config with VIIRS disabled."""
    config = _create_baseline_config()
    config.config_name = 'ablation_no_viirs'
    config.description = 'Ablation: VIIRS data source disabled'
    config.viirs = DataSourceEntry(enabled=False)
    return config


def _create_no_spatial_config() -> ModularDataConfig:
    """Create ablation config with all spatial features disabled."""
    config = _create_baseline_config()
    config.config_name = 'ablation_no_spatial'
    config.description = 'Ablation: All spatial features disabled'
    config.spatial_features_enabled = False
    return config


def _create_equipment_only_config() -> ModularDataConfig:
    """Create ablation config with only equipment sources."""
    return ModularDataConfig(
        config_name='ablation_equipment_only',
        description='Ablation: Only equipment loss sources enabled (drones, armor, artillery)',
        use_disaggregated_equipment=True,
        spatial_features_enabled=False,
        drones=DataSourceEntry(enabled=True),
        armor=DataSourceEntry(enabled=True),
        artillery=DataSourceEntry(enabled=True),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=False),
        deepstate=DataSourceEntry(enabled=False),
        firms=DataSourceEntry(enabled=False),
        viina=DataSourceEntry(enabled=False),
        viirs=DataSourceEntry(enabled=False),
        sentinel=DataSourceEntry(enabled=False),
        hdx_conflict=DataSourceEntry(enabled=False),
        hdx_food=DataSourceEntry(enabled=False),
        hdx_rainfall=DataSourceEntry(enabled=False),
        iom=DataSourceEntry(enabled=False),
    )


def _create_daily_only_config() -> ModularDataConfig:
    """Create ablation config with only daily sources."""
    config = _create_baseline_config()
    config.config_name = 'ablation_daily_only'
    config.description = 'Ablation: Only daily resolution sources enabled'
    config.sentinel = DataSourceEntry(enabled=False)
    config.hdx_conflict = DataSourceEntry(enabled=False)
    config.hdx_food = DataSourceEntry(enabled=False)
    config.hdx_rainfall = DataSourceEntry(enabled=False)
    config.iom = DataSourceEntry(enabled=False)
    return config


def _create_monthly_only_config() -> ModularDataConfig:
    """Create ablation config with only monthly sources."""
    return ModularDataConfig(
        config_name='ablation_monthly_only',
        description='Ablation: Only monthly resolution sources enabled',
        use_disaggregated_equipment=False,
        spatial_features_enabled=False,
        equipment=DataSourceEntry(enabled=False),
        drones=DataSourceEntry(enabled=False),
        armor=DataSourceEntry(enabled=False),
        artillery=DataSourceEntry(enabled=False),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=False),
        deepstate=DataSourceEntry(enabled=False),
        firms=DataSourceEntry(enabled=False),
        viina=DataSourceEntry(enabled=False),
        viirs=DataSourceEntry(enabled=False),
        sentinel=DataSourceEntry(enabled=True),
        hdx_conflict=DataSourceEntry(enabled=True),
        hdx_food=DataSourceEntry(enabled=True),
        hdx_rainfall=DataSourceEntry(enabled=True),
        iom=DataSourceEntry(enabled=True),
    )


def _create_minimal_config() -> ModularDataConfig:
    """Create minimal configuration for fast debugging."""
    return ModularDataConfig(
        config_name='minimal',
        description='Minimal configuration with only essential sources for fast debugging',
        use_disaggregated_equipment=True,
        spatial_features_enabled=False,
        drones=DataSourceEntry(enabled=True),
        armor=DataSourceEntry(enabled=False),
        artillery=DataSourceEntry(enabled=False),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=True),
        deepstate=DataSourceEntry(enabled=False),
        firms=DataSourceEntry(enabled=False),
        viina=DataSourceEntry(enabled=False),
        viirs=DataSourceEntry(enabled=False),
        sentinel=DataSourceEntry(enabled=True),
        hdx_conflict=DataSourceEntry(enabled=False),
        hdx_food=DataSourceEntry(enabled=False),
        hdx_rainfall=DataSourceEntry(enabled=False),
        iom=DataSourceEntry(enabled=False),
    )


def _create_spatial_viirs_tiled_config() -> ModularDataConfig:
    """Create configuration with VIIRS tiled spatial features."""
    config = _create_baseline_config()
    config.config_name = 'spatial_viirs_tiled'
    config.description = 'Baseline + VIIRS with 6 regional tiles'
    config.spatial_features_enabled = True
    config.viirs = DataSourceEntry(
        enabled=True,
        spatial_mode=SpatialMode.TILED,
        detrend=True,
    )
    return config


def _create_spatial_firms_tiled_config() -> ModularDataConfig:
    """Create configuration with FIRMS tiled spatial features."""
    config = _create_baseline_config()
    config.config_name = 'spatial_firms_tiled'
    config.description = 'Baseline + FIRMS with regional spatial binning'
    config.spatial_features_enabled = True
    config.firms = DataSourceEntry(
        enabled=True,
        spatial_mode=SpatialMode.TILED,
    )
    return config


def _create_unit_tracking_config() -> ModularDataConfig:
    """Create configuration focused on military unit tracking from DeepState."""
    config = _create_baseline_config()
    config.config_name = 'unit_tracking'
    config.description = 'Baseline + DeepState unit positions, frontline metrics per region'
    config.spatial_features_enabled = True
    config.deepstate_spatial = DataSourceEntry(
        enabled=True,
        spatial_mode=SpatialMode.TILED,
    )
    return config


def _create_spatial_full_config() -> ModularDataConfig:
    """Create configuration with ALL spatial sources at maximum resolution."""
    config = _create_spatial_rich_config()
    config.config_name = 'spatial_full'
    config.description = 'Maximum spatial resolution: all tiled features + unit tracking + fire hotspots'
    # Already has spatial_rich features, just update description
    return config


def _create_frontline_only_config() -> ModularDataConfig:
    """Create configuration with only frontline-related spatial features."""
    return ModularDataConfig(
        config_name='frontline_only',
        description='Frontline analysis: DeepState spatial only (units, frontlines, attacks)',
        use_disaggregated_equipment=True,
        spatial_features_enabled=True,
        # Only spatial source
        deepstate_spatial=DataSourceEntry(
            enabled=True,
            spatial_mode=SpatialMode.TILED,
        ),
        # Disable other spatial
        firms_spatial=DataSourceEntry(enabled=False),
        viirs=DataSourceEntry(enabled=True, detrend=True),  # Keep VIIRS but not tiled
        # Keep equipment for correlation
        drones=DataSourceEntry(enabled=True),
        armor=DataSourceEntry(enabled=True),
        artillery=DataSourceEntry(enabled=True),
        aircraft=DataSourceEntry(enabled=False),
        personnel=DataSourceEntry(enabled=True),
        deepstate=DataSourceEntry(enabled=True),
        firms=DataSourceEntry(enabled=True),
        viina=DataSourceEntry(enabled=True),
        sentinel=DataSourceEntry(enabled=True),
        hdx_conflict=DataSourceEntry(enabled=True),
        hdx_food=DataSourceEntry(enabled=True),
        hdx_rainfall=DataSourceEntry(enabled=True),
        iom=DataSourceEntry(enabled=True),
    )


# Preset registry
PRESET_CONFIGS: Dict[str, ModularDataConfig] = {}


def _initialize_presets() -> None:
    """Initialize preset configurations (called on module import)."""
    global PRESET_CONFIGS
    PRESET_CONFIGS = {
        # Core presets
        'baseline': _create_baseline_config(),
        'spatial_rich': _create_spatial_rich_config(),
        'spatial_full': _create_spatial_full_config(),
        # Unit and frontline tracking
        'unit_tracking': _create_unit_tracking_config(),
        'frontline_only': _create_frontline_only_config(),
        # Individual spatial source presets
        'spatial_viirs_tiled': _create_spatial_viirs_tiled_config(),
        'spatial_firms_tiled': _create_spatial_firms_tiled_config(),
        # Ablation presets
        'ablation_viirs_only': _create_viirs_only_config(),
        'ablation_no_viirs': _create_no_viirs_config(),
        'ablation_no_spatial': _create_no_spatial_config(),
        'ablation_equipment_only': _create_equipment_only_config(),
        'ablation_daily_only': _create_daily_only_config(),
        'ablation_monthly_only': _create_monthly_only_config(),
        # Debug
        'minimal': _create_minimal_config(),
    }


# Initialize presets on module load
_initialize_presets()


# =============================================================================
# PUBLIC API
# =============================================================================

def get_data_source_config(preset: str = 'baseline') -> ModularDataConfig:
    """Get a preset data source configuration by name.

    Args:
        preset: Name of the preset configuration. Available presets:
            - 'baseline': Current behavior (disaggregated equipment, no spatial)
            - 'spatial_rich': All spatial features enabled (VIIRS/FIRMS/DeepState tiled)
            - 'ablation_viirs_only': Only VIIRS enabled
            - 'ablation_no_viirs': VIIRS disabled
            - 'ablation_no_spatial': All spatial features disabled
            - 'ablation_equipment_only': Only equipment loss sources
            - 'ablation_daily_only': Only daily resolution sources
            - 'ablation_monthly_only': Only monthly resolution sources
            - 'minimal': Minimal sources for fast debugging
            - 'spatial_viirs_tiled': Baseline + VIIRS tiled
            - 'spatial_firms_tiled': Baseline + FIRMS tiled

    Returns:
        ModularDataConfig instance.

    Raises:
        ValueError: If preset name is not recognized.

    Example:
        >>> config = get_data_source_config('spatial_rich')
        >>> print(config.enabled_daily_sources)
        ['drones', 'armor', 'artillery', 'personnel', 'deepstate', 'firms', 'viina', 'viirs']
    """
    if preset not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets: {available}"
        )

    # Return a fresh instance to prevent mutation
    base_config = PRESET_CONFIGS[preset]
    return ModularDataConfig.from_dict(base_config.to_dict())


def list_presets() -> Dict[str, str]:
    """List all available preset configurations with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions.

    Example:
        >>> presets = list_presets()
        >>> for name, desc in presets.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: config.description for name, config in PRESET_CONFIGS.items()}


def create_ablation_config(
    disabled_sources: List[str],
    base_preset: str = 'baseline',
    name: Optional[str] = None,
) -> ModularDataConfig:
    """Create an ablation configuration by disabling specific sources.

    Args:
        disabled_sources: List of source names to disable.
        base_preset: Base preset to start from.
        name: Optional name for the new config.

    Returns:
        ModularDataConfig with specified sources disabled.

    Example:
        >>> config = create_ablation_config(['viirs', 'firms'])
        >>> print(config.is_source_enabled('viirs'))
        False
    """
    config = get_data_source_config(base_preset)

    for source_name in disabled_sources:
        source_cfg = config.get_source_config(source_name)
        if source_cfg:
            # Create a new disabled entry
            new_entry = DataSourceEntry(
                enabled=False,
                spatial_mode=source_cfg.spatial_mode,
            )
            setattr(config, source_name, new_entry)

    config.config_name = name or f"ablation_no_{'_'.join(disabled_sources)}"
    config.description = f"Ablation: {', '.join(disabled_sources)} disabled"

    return config


# =============================================================================
# MAIN (for demonstration)
# =============================================================================

if __name__ == '__main__':
    print("Available Data Source Presets")
    print("=" * 60)

    for preset_name, description in list_presets().items():
        print(f"\n{preset_name}:")
        print(f"  {description}")

    print("\n" + "=" * 60)
    print("\nExample: baseline configuration")
    print("=" * 60)
    config = get_data_source_config('baseline')
    print(config)

    print("\n" + "=" * 60)
    print("\nExample: spatial_rich configuration")
    print("=" * 60)
    config = get_data_source_config('spatial_rich')
    print(config)

    print("\n" + "=" * 60)
    print("\nExample: custom ablation configuration")
    print("=" * 60)
    config = create_ablation_config(['viirs', 'firms', 'viina'])
    print(config)
