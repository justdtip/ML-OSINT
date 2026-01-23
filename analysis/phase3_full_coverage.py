#!/usr/bin/env python3
"""
Phase 3: Full 198-Feature Coverage with Gap Analysis

This module extends the interpolation system to cover all 198 leaf features
defined in FEATURE_HIERARCHY.py, with emphasis on:

1. Data gap analysis - understanding where observations are sparse
2. Prediction accuracy tracking - which features interpolate well vs poorly
3. Cross-source fusion readiness - unified interface for multi-modal analysis

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         PHASE 3: FULL COVERAGE                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
    │  │    UCDP     │   │    FIRMS    │   │  SENTINEL   │   │  DEEPSTATE  │ │
    │  │  28 feats   │   │  24 feats   │   │  55 feats   │   │  52 feats   │ │
    │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘ │
    │         │                 │                 │                 │         │
    │         └────────────┬────┴────────┬────────┴─────────┬───────┘         │
    │                      │             │                  │                 │
    │                      ▼             ▼                  ▼                 │
    │               ┌──────────────────────────────────────────┐              │
    │               │         GAP ANALYZER & QUALITY           │              │
    │               │   - Observation density per feature      │              │
    │               │   - Interpolation accuracy metrics       │              │
    │               │   - Data freshness tracking              │              │
    │               └──────────────────────────────────────────┘              │
    │                                    │                                    │
    │                                    ▼                                    │
    │               ┌──────────────────────────────────────────┐              │
    │               │         CROSS-SOURCE FUSION HUB          │              │
    │               │   - Unified 198-dim feature vector       │              │
    │               │   - Source-aware attention masks         │              │
    │               │   - Temporal alignment                   │              │
    │               └──────────────────────────────────────────┘              │
    │                                                                         │
    │  ┌─────────────┐   ┌─────────────┐                                     │
    │  │  EQUIPMENT  │   │  PERSONNEL  │                                     │
    │  │  36 feats   │   │   3 feats   │                                     │
    │  └─────────────┘   └─────────────┘                                     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    python analysis/phase3_full_coverage.py --analyze-gaps
    python analysis/phase3_full_coverage.py --train-remaining
    python analysis/phase3_full_coverage.py --fuse --date 2024-01-15
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import json

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")

# Import existing components
from joint_interpolation_models import (
    INTERPOLATION_CONFIGS, PHASE2_CONFIGS,
    JointInterpolationModel, InterpolationConfig,
    InterpolationDataset
)
from FEATURE_HIERARCHY import FEATURE_HIERARCHY, get_all_leaf_features

# Import data loaders
try:
    from interpolation_data_loaders import (
        SentinelDataLoader, DeepStateDataLoader,
        EquipmentDataLoader, FIRMSDataLoader, UCDPDataLoader
    )
    HAS_DATA_LOADERS = True
except ImportError:
    HAS_DATA_LOADERS = False

from config.paths import (
    PROJECT_ROOT, DATA_DIR, ANALYSIS_DIR, MODEL_DIR, INTERP_MODEL_DIR,
    UNIFIED_INTERP_MODEL, UNIFIED_DELTA_MODEL, UNIFIED_HYBRID_MODEL,
    get_interp_model_path, get_phase_model_path,
)

# For backward compatibility, keep BASE_DIR as an alias to PROJECT_ROOT
BASE_DIR = PROJECT_ROOT


# =============================================================================
# PHASE 3 CONFIGURATIONS - REMAINING FEATURES
# =============================================================================

# Features already covered by Phase 1 + Phase 2: ~107
# Remaining to reach 198: ~91 features

PHASE3_CONFIGS = {
    # =========================================================================
    # UCDP Additional Decompositions (currently ~12 features, need ~16 more)
    # =========================================================================
    'ucdp_violence_type': InterpolationConfig(
        name='UCDP Violence Type Breakdown',
        source='ucdp',
        features=[
            'violence_state_based', 'violence_non_state', 'violence_one_sided'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['total_events'],
        conditioning_dim=64
    ),
    'ucdp_deaths_party': InterpolationConfig(
        name='UCDP Deaths by Party',
        source='ucdp',
        features=[
            'deaths_side_a', 'deaths_side_b', 'deaths_civilians', 'deaths_unknown'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['deaths_best'],
        conditioning_dim=64
    ),
    'ucdp_deaths_estimate': InterpolationConfig(
        name='UCDP Death Estimates',
        source='ucdp',
        features=[
            'deaths_best_estimate', 'deaths_high_estimate', 'deaths_low_estimate'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'ucdp_precision': InterpolationConfig(
        name='UCDP Event Precision',
        source='ucdp',
        features=[
            'precision_exact', 'precision_approximate', 'precision_regional'
        ],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'ucdp_fronts': InterpolationConfig(
        name='UCDP Events by Front',
        source='ucdp',
        features=[
            'front_eastern', 'front_southern', 'front_northeastern',
            'front_northern', 'front_rear'
        ],
        native_resolution_days=1.0,
        hierarchy_level=2,
        parent_features=['geo_donetsk', 'geo_luhansk', 'geo_kharkiv'],
        conditioning_dim=64
    ),

    # =========================================================================
    # FIRMS Additional Decompositions (currently ~10 features, need ~14 more)
    # =========================================================================
    'firms_daynight': InterpolationConfig(
        name='FIRMS DayNight Fires',
        source='firms',
        features=['fires_day', 'fires_night'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['fire_count'],
        conditioning_dim=64
    ),
    'firms_confidence': InterpolationConfig(
        name='FIRMS Confidence Levels',
        source='firms',
        features=['conf_high', 'conf_nominal', 'conf_low'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'firms_type': InterpolationConfig(
        name='FIRMS Fire Types',
        source='firms',
        features=['type_vegetation', 'type_active_fire', 'type_static'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'firms_frp_daynight': InterpolationConfig(
        name='FIRMS FRP by DayNight',
        source='firms',
        features=['frp_day_mean', 'frp_night_mean', 'frp_day_max', 'frp_night_max'],
        native_resolution_days=1.0,
        hierarchy_level=2,
        conditioning_dim=64
    ),
    'firms_brightness': InterpolationConfig(
        name='FIRMS Brightness Channels',
        source='firms',
        features=['brightness_t21', 'brightness_t31', 'brightness_ratio'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),

    # =========================================================================
    # SENTINEL Additional (currently ~26 features, need ~29 more)
    # =========================================================================
    'sentinel1_polarization': InterpolationConfig(
        name='Sentinel-1 Polarization Details',
        source='sentinel',
        features=['s1_vv_std', 's1_vh_std', 's1_vv_min', 's1_vh_min'],
        native_resolution_days=6.0,
        hierarchy_level=2,
        parent_features=['s1_vv_mean', 's1_vh_mean'],
        conditioning_dim=64
    ),
    'sentinel2_bands_detail': InterpolationConfig(
        name='Sentinel-2 Additional Bands',
        source='sentinel',
        features=[
            's2_b01_coastal', 's2_b05_veg_edge1', 's2_b06_veg_edge2',
            's2_b07_veg_edge3', 's2_b8a_veg_edge4', 's2_b09_water_vapor'
        ],
        native_resolution_days=5.0,
        hierarchy_level=2,
        conditioning_dim=64
    ),
    'sentinel2_indices_detail': InterpolationConfig(
        name='Sentinel-2 Derived Indices',
        source='sentinel',
        features=[
            's2_ndvi_std', 's2_ndwi_std', 's2_nbr_std', 's2_ndbi_mean'
        ],
        native_resolution_days=5.0,
        hierarchy_level=2,
        parent_features=['s2_ndvi_mean', 's2_ndwi_mean', 's2_nbr_mean'],
        conditioning_dim=64
    ),
    'sentinel2_cloud': InterpolationConfig(
        name='Sentinel-2 Cloud Metrics',
        source='sentinel',
        features=['s2_cloud_pct', 's2_cloud_free_count', 's2_avg_cloud'],
        native_resolution_days=5.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'sentinel3_products': InterpolationConfig(
        name='Sentinel-3 Product Details',
        source='sentinel',
        features=['s3_slstr_frp', 's3_olci_lfr', 's3_olci_wfr'],
        native_resolution_days=1.5,
        hierarchy_level=2,
        conditioning_dim=64
    ),
    'sentinel5p_gases_detail': InterpolationConfig(
        name='Sentinel-5P Additional Gases',
        source='sentinel',
        features=['s5p_so2', 's5p_ch4', 's5p_o3'],
        native_resolution_days=1.0,
        hierarchy_level=2,
        conditioning_dim=64
    ),
    'sentinel5p_aerosols': InterpolationConfig(
        name='Sentinel-5P Aerosols',
        source='sentinel',
        features=['s5p_aai', 's5p_aod'],
        native_resolution_days=1.0,
        hierarchy_level=2,
        conditioning_dim=64
    ),

    # =========================================================================
    # DEEPSTATE Additional (currently ~30 features, need ~22 more)
    # =========================================================================
    'deepstate_arrows_cardinal': InterpolationConfig(
        name='DeepState Cardinal Directions',
        source='deepstate',
        features=[
            'arrows_n', 'arrows_e', 'arrows_s', 'arrows_w'
        ],
        native_resolution_days=2.5,
        hierarchy_level=2,
        parent_features=['arrows_north', 'arrows_east', 'arrows_south', 'arrows_west'],
        conditioning_dim=64
    ),
    'deepstate_polygons_region': InterpolationConfig(
        name='DeepState Polygons by Region',
        source='deepstate',
        features=['poly_crimea', 'poly_ordlo', 'poly_other_occupied'],
        native_resolution_days=2.5,
        hierarchy_level=2,
        conditioning_dim=64
    ),
    'deepstate_polygons_metrics': InterpolationConfig(
        name='DeepState Polygon Metrics',
        source='deepstate',
        features=['poly_count_total', 'poly_area_total', 'boundary_length'],
        native_resolution_days=2.5,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'deepstate_airfields': InterpolationConfig(
        name='DeepState Airfields',
        source='deepstate',
        features=[
            'airfields_crimea', 'airfields_eastern', 'airfields_northern',
            'airfields_western', 'airfields_operational', 'airfields_damaged'
        ],
        native_resolution_days=2.5,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'deepstate_units_type': InterpolationConfig(
        name='DeepState Unit Types Extended',
        source='deepstate',
        features=[
            'units_special', 'units_naval', 'units_logistics', 'units_other'
        ],
        native_resolution_days=2.5,
        hierarchy_level=2,
        parent_features=['units_total'],
        conditioning_dim=64
    ),

    # =========================================================================
    # EQUIPMENT Additional (currently ~22 features, need ~14 more)
    # =========================================================================
    'equipment_helicopters': InterpolationConfig(
        name='Helicopter Losses by Type',
        source='equipment',
        features=['heli_ka52', 'heli_mi28', 'heli_mi24_35', 'heli_mi8', 'heli_other'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['heli_total'],
        conditioning_dim=64
    ),
    'equipment_artillery': InterpolationConfig(
        name='Artillery Losses by Type',
        source='equipment',
        features=['arty_field', 'arty_self_propelled', 'arty_mrl'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['arty_total'],
        conditioning_dim=64
    ),
    'equipment_drones': InterpolationConfig(
        name='Drone Losses by Role',
        source='equipment',
        features=['drone_recon', 'drone_combat', 'drone_loitering'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['drones_total'],
        conditioning_dim=64
    ),
    'equipment_air_defense': InterpolationConfig(
        name='Air Defense Losses by Type',
        source='equipment',
        features=['ad_sam_long', 'ad_sam_medium', 'ad_sam_short', 'ad_spaa'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        parent_features=['air_defense_total'],
        conditioning_dim=64
    ),
    'equipment_naval': InterpolationConfig(
        name='Naval Losses by Class',
        source='equipment',
        features=['naval_cruiser', 'naval_landing', 'naval_patrol', 'naval_other'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),
    'equipment_vehicles': InterpolationConfig(
        name='Vehicle Losses',
        source='equipment',
        features=['vehicles_trucks', 'vehicles_fuel', 'vehicles_command', 'vehicles_engineering'],
        native_resolution_days=1.0,
        hierarchy_level=1,
        conditioning_dim=64
    ),

    # =========================================================================
    # PERSONNEL (currently 0 interpolation, need 3)
    # =========================================================================
    'personnel_rates': InterpolationConfig(
        name='Personnel Casualty Rates',
        source='personnel',
        features=['personnel_cumulative', 'personnel_daily', 'personnel_monthly'],
        native_resolution_days=1.0,
        hierarchy_level=0,
        conditioning_dim=0
    ),
}


# =============================================================================
# GAP ANALYZER
# =============================================================================

@dataclass
class FeatureGapMetrics:
    """Metrics for a single feature's data gaps and quality."""
    feature_name: str
    source: str
    total_observations: int
    date_range_days: int
    observation_density: float  # obs per day
    avg_gap_days: float
    max_gap_days: float
    missing_pct: float
    interpolation_mae: Optional[float] = None
    interpolation_uncertainty: Optional[float] = None
    last_observation: Optional[str] = None
    freshness_days: Optional[int] = None


class GapAnalyzer:
    """
    Analyzes data gaps across all features to understand:
    1. Where observations are sparse
    2. Which features have reliable interpolation
    3. Data freshness and staleness
    """

    def __init__(self):
        self.metrics: Dict[str, FeatureGapMetrics] = {}
        self.source_summaries: Dict[str, Dict] = {}

    def analyze_source(self, source: str, data: np.ndarray, dates: List[str],
                       feature_names: List[str]) -> Dict[str, FeatureGapMetrics]:
        """
        Analyze gap metrics for all features in a source.

        Args:
            source: Source name (ucdp, firms, sentinel, etc.)
            data: [n_obs, n_features] array
            dates: List of date strings
            feature_names: List of feature names

        Returns:
            Dict mapping feature name to gap metrics
        """
        metrics = {}

        # Parse dates
        parsed_dates = []
        for d in dates:
            try:
                if len(d) == 10:  # YYYY-MM-DD
                    parsed_dates.append(datetime.strptime(d, '%Y-%m-%d'))
                else:  # YYYY-MM
                    parsed_dates.append(datetime.strptime(d[:7], '%Y-%m'))
            except:
                continue

        if len(parsed_dates) < 2:
            return metrics

        date_range_days = (parsed_dates[-1] - parsed_dates[0]).days
        today = datetime.now()

        for i, feat_name in enumerate(feature_names):
            if i >= data.shape[1]:
                continue

            col = data[:, i]

            # Find non-missing observations
            valid_mask = ~np.isnan(col) & (col != 0)
            n_valid = valid_mask.sum()

            # Calculate gaps between valid observations
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 1:
                gaps = []
                for j in range(1, len(valid_indices)):
                    gap = (parsed_dates[valid_indices[j]] - parsed_dates[valid_indices[j-1]]).days
                    gaps.append(gap)
                avg_gap = np.mean(gaps)
                max_gap = np.max(gaps)
            else:
                avg_gap = date_range_days
                max_gap = date_range_days

            # Last observation freshness
            if len(valid_indices) > 0:
                last_obs_date = parsed_dates[valid_indices[-1]]
                freshness = (today - last_obs_date).days
                last_obs_str = last_obs_date.strftime('%Y-%m-%d')
            else:
                freshness = None
                last_obs_str = None

            metrics[feat_name] = FeatureGapMetrics(
                feature_name=feat_name,
                source=source,
                total_observations=int(n_valid),
                date_range_days=date_range_days,
                observation_density=n_valid / max(date_range_days, 1),
                avg_gap_days=avg_gap,
                max_gap_days=max_gap,
                missing_pct=100 * (1 - n_valid / len(col)),
                last_observation=last_obs_str,
                freshness_days=freshness
            )

        self.metrics.update(metrics)
        return metrics

    def analyze_all_sources(self) -> Dict[str, Dict]:
        """Analyze gaps across all data sources."""
        if not HAS_DATA_LOADERS:
            print("Data loaders not available")
            return {}

        summaries = {}

        # UCDP
        try:
            loader = UCDPDataLoader().load().process()
            metrics = self.analyze_source(
                'ucdp', loader.processed_data, loader.dates, loader.feature_names
            )
            summaries['ucdp'] = self._summarize_source(metrics)
            print(f"UCDP: {len(metrics)} features analyzed")
        except Exception as e:
            print(f"Error analyzing UCDP: {e}")

        # FIRMS
        try:
            loader = FIRMSDataLoader().load().process()
            metrics = self.analyze_source(
                'firms', loader.processed_data, loader.dates, loader.feature_names
            )
            summaries['firms'] = self._summarize_source(metrics)
            print(f"FIRMS: {len(metrics)} features analyzed")
        except Exception as e:
            print(f"Error analyzing FIRMS: {e}")

        # Sentinel
        try:
            loader = SentinelDataLoader().load().process()
            data, dates = loader.get_daily_observations()
            metrics = self.analyze_source(
                'sentinel', data, dates, loader.feature_names
            )
            summaries['sentinel'] = self._summarize_source(metrics)
            print(f"Sentinel: {len(metrics)} features analyzed")
        except Exception as e:
            print(f"Error analyzing Sentinel: {e}")

        # DeepState
        try:
            loader = DeepStateDataLoader().load().process()
            metrics = self.analyze_source(
                'deepstate', loader.processed_data, loader.dates, loader.feature_names
            )
            summaries['deepstate'] = self._summarize_source(metrics)
            print(f"DeepState: {len(metrics)} features analyzed")
        except Exception as e:
            print(f"Error analyzing DeepState: {e}")

        # Equipment
        try:
            loader = EquipmentDataLoader().load().process()
            data, dates = loader.get_daily_changes()
            metrics = self.analyze_source(
                'equipment', data, dates, loader.feature_names
            )
            summaries['equipment'] = self._summarize_source(metrics)
            print(f"Equipment: {len(metrics)} features analyzed")
        except Exception as e:
            print(f"Error analyzing Equipment: {e}")

        self.source_summaries = summaries
        return summaries

    def _summarize_source(self, metrics: Dict[str, FeatureGapMetrics]) -> Dict:
        """Create summary statistics for a source."""
        if not metrics:
            return {}

        densities = [m.observation_density for m in metrics.values()]
        avg_gaps = [m.avg_gap_days for m in metrics.values()]
        max_gaps = [m.max_gap_days for m in metrics.values()]
        missing_pcts = [m.missing_pct for m in metrics.values()]

        return {
            'n_features': len(metrics),
            'avg_density': np.mean(densities),
            'min_density': np.min(densities),
            'max_density': np.max(densities),
            'avg_gap_days': np.mean(avg_gaps),
            'max_gap_days': np.max(max_gaps),
            'avg_missing_pct': np.mean(missing_pcts),
            'sparsest_features': sorted(
                metrics.keys(),
                key=lambda k: metrics[k].observation_density
            )[:5]
        }

    def get_interpolation_priority(self) -> List[Tuple[str, float]]:
        """
        Rank features by interpolation priority (sparse + important).

        Returns list of (feature_name, priority_score) sorted by priority.
        """
        priorities = []

        for name, m in self.metrics.items():
            # Higher priority for:
            # - More gaps (lower density)
            # - Longer max gaps
            # - Higher missing %
            priority = (
                (1 - m.observation_density) * 0.4 +
                min(m.max_gap_days / 30, 1.0) * 0.3 +
                (m.missing_pct / 100) * 0.3
            )
            priorities.append((name, priority))

        return sorted(priorities, key=lambda x: -x[1])

    def print_report(self):
        """Print comprehensive gap analysis report."""
        print("\n" + "=" * 80)
        print("DATA GAP ANALYSIS REPORT")
        print("=" * 80)

        for source, summary in self.source_summaries.items():
            print(f"\n{source.upper()}")
            print("-" * 40)
            print(f"  Features: {summary['n_features']}")
            print(f"  Avg observation density: {summary['avg_density']:.3f} obs/day")
            print(f"  Density range: [{summary['min_density']:.3f}, {summary['max_density']:.3f}]")
            print(f"  Avg gap between obs: {summary['avg_gap_days']:.1f} days")
            print(f"  Max gap: {summary['max_gap_days']:.0f} days")
            print(f"  Avg missing: {summary['avg_missing_pct']:.1f}%")
            print(f"  Sparsest features: {', '.join(summary['sparsest_features'][:3])}")

        # Overall priority
        print("\n" + "=" * 80)
        print("INTERPOLATION PRIORITY (top 20 sparsest features)")
        print("=" * 80)

        priorities = self.get_interpolation_priority()[:20]
        for i, (name, score) in enumerate(priorities):
            m = self.metrics[name]
            print(f"  {i+1:2}. {name:40s} "
                  f"density={m.observation_density:.3f}, "
                  f"max_gap={m.max_gap_days:.0f}d, "
                  f"missing={m.missing_pct:.0f}%")


# =============================================================================
# CROSS-SOURCE FUSION HUB
# =============================================================================

class CrossSourceFusionHub:
    """
    Unified interface for accessing all 198 features with:
    1. Automatic interpolation for missing dates
    2. Source-aware quality flags
    3. Aligned temporal representation
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.all_features = get_all_leaf_features()
        self.n_features = len(self.all_features)

        # Feature index mapping
        self.feature_to_idx = {f: i for i, f in enumerate(self.all_features)}

        # Source boundaries
        self.source_ranges = self._compute_source_ranges()

        # Loaded models (lazy loading)
        self.interpolators: Dict[str, JointInterpolationModel] = {}
        self.gap_analyzer = GapAnalyzer()

    def _compute_source_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Compute index ranges for each source in the unified vector."""
        ranges = {}
        current_idx = 0

        for source in ['ucdp', 'firms', 'sentinel', 'deepstate', 'equipment', 'personnel']:
            source_features = [f for f in self.all_features if f.startswith(source)]
            if source_features:
                start_idx = current_idx
                end_idx = current_idx + len(source_features)
                ranges[source] = (start_idx, end_idx)
                current_idx = end_idx

        return ranges

    def load_interpolators(self):
        """Load all available interpolation models."""
        print("Loading interpolation models...")

        # Phase 1 models
        for name, config in INTERPOLATION_CONFIGS.items():
            model_path = INTERP_MODEL_DIR / f"interp_{config.name.replace(' ', '_').lower()}_best.pt"
            if model_path.exists():
                model = JointInterpolationModel(config)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.interpolators[name] = model

        # Phase 2 models
        for name, config in PHASE2_CONFIGS.items():
            model_path = MODEL_DIR / f"phase2_{name}_best.pt"
            if model_path.exists():
                model = JointInterpolationModel(config)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.interpolators[name] = model

        # Phase 3 models (as they become available)
        for name, config in PHASE3_CONFIGS.items():
            model_path = MODEL_DIR / f"phase3_{name}_best.pt"
            if model_path.exists():
                model = JointInterpolationModel(config)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.interpolators[name] = model

        print(f"  Loaded {len(self.interpolators)} interpolation models")

    def get_unified_vector(
        self,
        target_date: datetime,
        return_quality: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get unified 198-dimensional feature vector for a specific date.

        Args:
            target_date: Date to get features for
            return_quality: Whether to return quality metrics

        Returns:
            features: [198] array of feature values
            uncertainties: [198] array of uncertainty estimates (if return_quality)
            observation_mask: [198] binary mask (1=observed, 0=interpolated)
        """
        features = np.zeros(self.n_features)
        uncertainties = np.zeros(self.n_features) if return_quality else None
        obs_mask = np.zeros(self.n_features) if return_quality else None

        # TODO: Implement actual data loading and interpolation
        # This would:
        # 1. Load actual observations near target_date
        # 2. For missing features, use interpolators
        # 3. Track which features are observed vs interpolated

        return features, uncertainties, obs_mask

    def get_source_mask(self, sources: List[str]) -> np.ndarray:
        """
        Get boolean mask for specific sources in the unified vector.

        Args:
            sources: List of source names to include

        Returns:
            [198] boolean mask
        """
        mask = np.zeros(self.n_features, dtype=bool)

        for source in sources:
            if source in self.source_ranges:
                start, end = self.source_ranges[source]
                mask[start:end] = True

        return mask

    def summary(self):
        """Print summary of the fusion hub."""
        print("\n" + "=" * 80)
        print("CROSS-SOURCE FUSION HUB")
        print("=" * 80)
        print(f"\nTotal features: {self.n_features}")
        print(f"Interpolators loaded: {len(self.interpolators)}")

        print("\nSource breakdown:")
        for source, (start, end) in self.source_ranges.items():
            n_feats = end - start
            print(f"  {source:12s}: {n_feats:3d} features (idx {start}-{end})")

        # Coverage analysis
        all_configs = {**INTERPOLATION_CONFIGS, **PHASE2_CONFIGS, **PHASE3_CONFIGS}
        covered_features = set()
        for config in all_configs.values():
            covered_features.update(config.features)

        print(f"\nInterpolation coverage:")
        print(f"  Configured: {len(covered_features)} features")
        print(f"  Target: {self.n_features} features")
        print(f"  Gap: {self.n_features - len(covered_features)} features")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_phase3_models(epochs: int = 100, batch_size: int = 32, device: str = None):
    """Train all Phase 3 interpolation models."""
    if not HAS_TORCH:
        print("PyTorch required")
        return

    from joint_interpolation_models import InterpolationTrainer

    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print("\n" + "=" * 80)
    print("TRAINING PHASE 3 MODELS")
    print(f"Device: {device}")
    print("=" * 80)

    results = {}

    for name, config in PHASE3_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Training: {config.name}")
        print(f"Config features: {len(config.features)}")
        print(f"{'='*60}")

        try:
            # Create datasets first to get actual feature count
            train_dataset = InterpolationDataset(
                config=config, data_path=DATA_DIR, train=True
            )
            val_dataset = InterpolationDataset(
                config=config, data_path=DATA_DIR, train=False
            )

            # Check if dataset loaded actual features (uses all available)
            actual_n_features = getattr(train_dataset, 'actual_num_features', len(config.features))
            actual_feature_names = getattr(train_dataset, 'actual_feature_names', config.features)

            # Create config with actual feature count
            actual_config = InterpolationConfig(
                name=config.name,
                source=config.source,
                features=actual_feature_names,  # Use actual features from data
                native_resolution_days=config.native_resolution_days,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                max_gap_days=config.max_gap_days,
                dropout=config.dropout,
                parent_features=config.parent_features,
                child_groups=config.child_groups,
                hierarchy_level=config.hierarchy_level,
                conditioning_dim=config.conditioning_dim
            )

            print(f"Actual features: {actual_n_features}")

            # Create model with actual feature count
            model = JointInterpolationModel(actual_config)

            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")

            if len(train_dataset) == 0:
                print(f"  Skipping {name} - no training samples")
                continue

            # Create loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Train with specified device
            trainer = InterpolationTrainer(
                model, train_loader, val_loader,
                lr=1e-4, weight_decay=0.01,
                device=device
            )
            history = trainer.train(epochs=epochs, patience=20, verbose=True)

            # Save - sanitize name for filesystem
            safe_name = name.replace('/', '_').replace(' ', '_')
            model_path = MODEL_DIR / f"phase3_{safe_name}_best.pt"
            torch.save(model.state_dict(), model_path)

            best_mae = min(history['val_mae'])
            results[name] = {
                'best_mae': best_mae,
                'n_features': actual_n_features
            }
            print(f"  Best MAE: {best_mae:.4f}")

        except Exception as e:
            print(f"  Error training {name}: {e}")
            results[name] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 TRAINING SUMMARY")
    print("=" * 80)

    for name, res in results.items():
        if 'error' in res:
            print(f"  {name}: ERROR - {res['error']}")
        else:
            print(f"  {name}: MAE={res['best_mae']:.4f} ({res['n_features']} features)")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3: Full 198-Feature Coverage')
    parser.add_argument('--analyze-gaps', action='store_true', help='Run gap analysis')
    parser.add_argument('--train', action='store_true', help='Train Phase 3 models')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for training (auto-detects MPS on Apple Silicon)')
    parser.add_argument('--summary', action='store_true', help='Print coverage summary')
    args = parser.parse_args()

    if args.analyze_gaps:
        analyzer = GapAnalyzer()
        analyzer.analyze_all_sources()
        analyzer.print_report()

    elif args.train:
        train_phase3_models(epochs=args.epochs, device=args.device)

    elif args.summary:
        hub = CrossSourceFusionHub()
        hub.summary()

    else:
        # Default: show summary and gap analysis
        print("=" * 80)
        print("PHASE 3: FULL 198-FEATURE COVERAGE")
        print("=" * 80)

        hub = CrossSourceFusionHub()
        hub.summary()

        print("\n" + "=" * 80)
        print("PHASE 3 CONFIGURATIONS")
        print("=" * 80)

        total_new = 0
        for name, config in PHASE3_CONFIGS.items():
            print(f"\n  {name}:")
            print(f"    Source: {config.source}")
            print(f"    Features: {len(config.features)}")
            print(f"    Level: {config.hierarchy_level}")
            total_new += len(config.features)

        print(f"\n  Total new Phase 3 features: {total_new}")

        # Run gap analysis
        print("\nRunning gap analysis...")
        analyzer = GapAnalyzer()
        analyzer.analyze_all_sources()
        analyzer.print_report()


if __name__ == "__main__":
    main()
