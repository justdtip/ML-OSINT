"""
Spatio-Temporal Representation Analysis for Multi-Resolution HAN

This module provides fine-grained analysis of the model's spatio-temporal learning,
including:
1. Latent trajectory analysis - how representations evolve through time
2. Attention pattern decomposition - what the model attends to at each timestep
3. Source importance dynamics - how source weighting changes over conflict phases
4. Geographic signal recovery - extracting spatial patterns from raw data
5. Temporal-spatial correlation structure - relating model representations to geography

Author: Claude
Date: 2026-01-27
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.paths import DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR
from analysis.multi_resolution_han import MultiResolutionHAN, SourceConfig
from analysis.multi_resolution_data import MultiResolutionDataset, MultiResolutionConfig

# Output directory
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis" / "spatio_temporal"
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFLICT PHASES FOR TEMPORAL CONTEXT
# =============================================================================

CONFLICT_PHASES = [
    {"name": "Initial Invasion", "start": "2022-02-24", "end": "2022-04-07"},
    {"name": "Donbas Focus", "start": "2022-04-08", "end": "2022-08-28"},
    {"name": "Kharkiv Counteroffensive", "start": "2022-08-29", "end": "2022-11-10"},
    {"name": "Winter Stalemate", "start": "2022-11-11", "end": "2023-06-03"},
    {"name": "2023 Counteroffensive", "start": "2023-06-04", "end": "2023-11-30"},
    {"name": "Attritional Warfare", "start": "2023-12-01", "end": "2024-08-05"},
    {"name": "Kursk Incursion", "start": "2024-08-06", "end": "2025-12-31"},
]

# Geographic tiles for VIIRS
VIIRS_TILES = {
    "h19v03": {"region": "Western (Lviv)", "center_lat": 50.0, "center_lon": 24.0},
    "h19v04": {"region": "Western-Central", "center_lat": 48.5, "center_lon": 24.0},
    "h20v03": {"region": "Central (Kyiv)", "center_lat": 50.0, "center_lon": 30.0},
    "h20v04": {"region": "Central-South", "center_lat": 48.5, "center_lon": 30.0},
    "h21v03": {"region": "Eastern (Kharkiv)", "center_lat": 50.0, "center_lon": 36.0},
    "h21v04": {"region": "Eastern (Donetsk)", "center_lat": 48.5, "center_lon": 36.0},
}


@dataclass
class SpatioTemporalConfig:
    """Configuration for spatio-temporal analysis."""
    checkpoint_path: Path = CHECKPOINT_DIR / "multi_resolution" / "best_checkpoint.pt"
    output_dir: Path = ANALYSIS_OUTPUT_DIR

    # Analysis parameters
    n_samples: int = 100  # Number of samples for trajectory analysis
    trajectory_timesteps: int = 365  # Days to analyze

    # Visualization parameters
    pca_components: int = 3
    tsne_perplexity: int = 30

    # Clustering
    n_temporal_clusters: int = 7  # Match conflict phases

    # Device
    device: str = "auto"


class SpatioTemporalAnalyzer:
    """
    Comprehensive analyzer for spatio-temporal representations in Multi-Resolution HAN.
    """

    def __init__(self, config: SpatioTemporalConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(config.device)

        print(f"Device: {self.device}")

        # Load model and dataset
        self.model, self.model_info = self._load_model()
        self.dataset, self.start_date = self._load_dataset()

        # Storage for analysis results
        self.results = {}

    def _load_model(self) -> Tuple[MultiResolutionHAN, Dict]:
        """Load the trained model."""
        print(f"Loading checkpoint: {self.config.checkpoint_path}")

        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        state_dict = checkpoint['model_state_dict']

        # Get feature info from dataset
        dataset_config = MultiResolutionConfig()
        temp_dataset = MultiResolutionDataset(config=dataset_config, split='train')
        feature_info = temp_dataset.get_feature_info()

        # Build source configs
        daily_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='daily')
            for name, info in feature_info.items() if info['resolution'] == 'daily'
        }
        monthly_source_configs = {
            name: SourceConfig(name=name, n_features=info['n_features'], resolution='monthly')
            for name, info in feature_info.items() if info['resolution'] == 'monthly'
        }

        # Create model
        model = MultiResolutionHAN(
            daily_source_configs=daily_source_configs,
            monthly_source_configs=monthly_source_configs,
            d_model=128,
            nhead=8,
            num_daily_layers=3,
            num_monthly_layers=2,
            num_fusion_layers=2,
            dropout=0.0,
            use_isw_alignment=True,
            isw_dim=1024,
        )

        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {n_params:,} parameters")

        del temp_dataset

        return model, {
            'n_params': n_params,
            'daily_sources': list(daily_source_configs.keys()),
            'monthly_sources': list(monthly_source_configs.keys()),
        }

    def _load_dataset(self) -> Tuple[MultiResolutionDataset, datetime]:
        """Load the dataset for analysis."""
        dataset_config = MultiResolutionConfig()
        dataset = MultiResolutionDataset(config=dataset_config, split='train')

        start_date = dataset.start_date
        print(f"Dataset: {len(dataset)} samples, start date: {start_date}")

        return dataset, start_date

    def _sample_to_batch(self, sample) -> Dict[str, torch.Tensor]:
        """Convert sample to batch format."""
        batch = {
            'daily_features': {
                name: tensor.unsqueeze(0).to(self.device)
                for name, tensor in sample.daily_features.items()
            },
            'daily_masks': {
                name: tensor.unsqueeze(0).to(self.device)
                for name, tensor in sample.daily_masks.items()
            },
            'monthly_features': {
                name: tensor.unsqueeze(0).to(self.device)
                for name, tensor in sample.monthly_features.items()
            },
            'monthly_masks': {
                name: tensor.unsqueeze(0).to(self.device)
                for name, tensor in sample.monthly_masks.items()
            },
            'month_boundaries': sample.month_boundary_indices.unsqueeze(0).to(self.device),
        }
        return batch

    def _get_phase_for_date(self, date: datetime) -> Optional[str]:
        """Get conflict phase for a given date."""
        date_str = date.strftime('%Y-%m-%d')
        for phase in CONFLICT_PHASES:
            if phase['start'] <= date_str <= phase['end']:
                return phase['name']
        return None

    # =========================================================================
    # ANALYSIS 1: LATENT TRAJECTORY ANALYSIS
    # =========================================================================

    def analyze_latent_trajectories(self) -> Dict:
        """
        Analyze how the model's latent representations evolve through time.

        Returns:
            Dict with trajectory statistics, velocity, and clustering results.
        """
        print("\n" + "="*60)
        print("ANALYSIS 1: LATENT TRAJECTORY ANALYSIS")
        print("="*60)

        # Collect latent representations across time
        n_samples = min(self.config.n_samples, len(self.dataset))

        all_latents = []
        all_dates = []
        all_phases = []
        all_sample_indices = []

        indices = np.linspace(0, len(self.dataset) - 1, n_samples, dtype=int)

        print(f"Extracting latents from {n_samples} samples...")

        for i, idx in enumerate(indices):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{n_samples}")

            sample = self.dataset[idx]
            batch = self._sample_to_batch(sample)

            with torch.no_grad():
                outputs = self.model(**batch)

            # Get temporal output [1, n_months, d_model]
            temporal_output = outputs['temporal_output'].cpu().numpy()[0]  # [n_months, 128]

            # Store each monthly representation
            for month_idx in range(temporal_output.shape[0]):
                all_latents.append(temporal_output[month_idx])

                # Calculate date for this month
                sample_start_idx = int(sample.sample_idx)
                month_offset = int(month_idx * 30)  # Approximate
                date = self.start_date + timedelta(days=sample_start_idx + month_offset)
                all_dates.append(date)
                all_phases.append(self._get_phase_for_date(date))
                all_sample_indices.append(idx)

        all_latents = np.array(all_latents)  # [n_total, 128]
        print(f"Collected {len(all_latents)} latent vectors")

        # Compute trajectory statistics
        results = {
            'n_latents': len(all_latents),
            'd_model': all_latents.shape[1],
            'dates': [d.isoformat() for d in all_dates],
            'phases': all_phases,
        }

        # 1. Latent velocity (rate of change)
        velocities = np.linalg.norm(np.diff(all_latents, axis=0), axis=1)
        results['velocity'] = {
            'mean': float(np.mean(velocities)),
            'std': float(np.std(velocities)),
            'min': float(np.min(velocities)),
            'max': float(np.max(velocities)),
            'values': velocities.tolist()
        }
        print(f"  Latent velocity: {results['velocity']['mean']:.4f} ± {results['velocity']['std']:.4f}")

        # 2. Phase-specific velocity
        phase_velocities = {}
        for phase_name in set(p for p in all_phases if p):
            phase_mask = np.array([p == phase_name for p in all_phases[:-1]])
            if phase_mask.sum() > 0:
                phase_vel = velocities[phase_mask]
                phase_velocities[phase_name] = {
                    'mean': float(np.mean(phase_vel)),
                    'std': float(np.std(phase_vel)),
                    'n': int(phase_mask.sum())
                }
        results['phase_velocities'] = phase_velocities

        # 3. PCA dimensionality reduction
        print("  Computing PCA...")
        pca = PCA(n_components=self.config.pca_components)
        latents_pca = pca.fit_transform(all_latents)

        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance_ratio_)),
            'components': latents_pca.tolist(),
        }
        print(f"  PCA variance explained: {results['pca']['total_variance_explained']:.2%}")

        # 4. Temporal clustering
        print("  Computing temporal clusters...")
        linkage_matrix = linkage(all_latents, method='ward')
        cluster_labels = fcluster(linkage_matrix, t=self.config.n_temporal_clusters, criterion='maxclust')

        # Analyze cluster-phase correspondence
        cluster_phase_matrix = np.zeros((self.config.n_temporal_clusters, len(set(p for p in all_phases if p))))
        phase_names = sorted(set(p for p in all_phases if p))
        for i, (cluster, phase) in enumerate(zip(cluster_labels, all_phases)):
            if phase:
                phase_idx = phase_names.index(phase)
                cluster_phase_matrix[cluster - 1, phase_idx] += 1

        results['clustering'] = {
            'n_clusters': self.config.n_temporal_clusters,
            'labels': cluster_labels.tolist(),
            'cluster_phase_matrix': cluster_phase_matrix.tolist(),
            'phase_names': phase_names,
        }

        # 5. Compute trajectory curvature (second derivative)
        if len(all_latents) > 2:
            second_diff = np.diff(all_latents, n=2, axis=0)
            curvature = np.linalg.norm(second_diff, axis=1)
            results['curvature'] = {
                'mean': float(np.mean(curvature)),
                'std': float(np.std(curvature)),
                'values': curvature.tolist()
            }
            print(f"  Trajectory curvature: {results['curvature']['mean']:.4f} ± {results['curvature']['std']:.4f}")

        self.results['latent_trajectories'] = results
        return results

    # =========================================================================
    # ANALYSIS 2: ATTENTION PATTERN DECOMPOSITION
    # =========================================================================

    def analyze_attention_patterns(self) -> Dict:
        """
        Analyze the model's attention patterns across sources and time.

        Returns:
            Dict with attention weight statistics and dynamics.
        """
        print("\n" + "="*60)
        print("ANALYSIS 2: ATTENTION PATTERN DECOMPOSITION")
        print("="*60)

        n_samples = min(self.config.n_samples, len(self.dataset))
        indices = np.linspace(0, len(self.dataset) - 1, n_samples, dtype=int)

        all_source_importance = []
        all_dates = []
        all_phases = []

        print(f"Extracting attention patterns from {n_samples} samples...")

        for i, idx in enumerate(indices):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{n_samples}")

            sample = self.dataset[idx]
            batch = self._sample_to_batch(sample)

            with torch.no_grad():
                outputs = self.model(**batch)

            # Source importance weights [1, n_months, n_sources]
            if 'source_importance' in outputs:
                importance = outputs['source_importance'].cpu().numpy()[0]  # [n_months, 5]

                for month_idx in range(importance.shape[0]):
                    all_source_importance.append(importance[month_idx])

                    sample_start_idx = int(sample.sample_idx)
                    month_offset = int(month_idx * 30)
                    date = self.start_date + timedelta(days=sample_start_idx + month_offset)
                    all_dates.append(date)
                    all_phases.append(self._get_phase_for_date(date))

        all_source_importance = np.array(all_source_importance)  # [n_total, 5]

        # Get source names
        source_names = self.model_info['monthly_sources']

        results = {
            'n_observations': len(all_source_importance),
            'n_sources': all_source_importance.shape[1],
            'source_names': source_names,
            'dates': [d.isoformat() for d in all_dates],
            'phases': all_phases,
        }

        # 1. Overall source importance
        mean_importance = np.mean(all_source_importance, axis=0)
        std_importance = np.std(all_source_importance, axis=0)

        results['overall_importance'] = {
            source: {
                'mean': float(mean_importance[i]),
                'std': float(std_importance[i]),
                'relative_weight': float(mean_importance[i] / mean_importance.sum())
            }
            for i, source in enumerate(source_names)
        }

        print("\n  Source Importance (mean ± std):")
        for source, stats in results['overall_importance'].items():
            print(f"    {source}: {stats['mean']:.4f} ± {stats['std']:.4f} ({stats['relative_weight']:.1%})")

        # 2. Phase-specific importance
        results['phase_importance'] = {}
        for phase_name in set(p for p in all_phases if p):
            phase_mask = np.array([p == phase_name for p in all_phases])
            if phase_mask.sum() > 0:
                phase_importance = all_source_importance[phase_mask]
                phase_mean = np.mean(phase_importance, axis=0)

                results['phase_importance'][phase_name] = {
                    source: float(phase_mean[i])
                    for i, source in enumerate(source_names)
                }

        # 3. Attention entropy (how spread out is attention?)
        # Higher entropy = more uniform attention across sources
        eps = 1e-8
        attention_entropy = -np.sum(
            all_source_importance * np.log(all_source_importance + eps),
            axis=1
        )

        results['attention_entropy'] = {
            'mean': float(np.mean(attention_entropy)),
            'std': float(np.std(attention_entropy)),
            'max_possible': float(np.log(all_source_importance.shape[1])),  # Uniform distribution
            'values': attention_entropy.tolist()
        }
        print(f"\n  Attention entropy: {results['attention_entropy']['mean']:.4f} (max: {results['attention_entropy']['max_possible']:.4f})")

        # 4. Source dominance analysis
        dominant_source = np.argmax(all_source_importance, axis=1)
        source_dominance_counts = {
            source: int((dominant_source == i).sum())
            for i, source in enumerate(source_names)
        }
        results['source_dominance'] = source_dominance_counts

        print("\n  Source dominance frequency:")
        for source, count in source_dominance_counts.items():
            print(f"    {source}: {count} ({count/len(dominant_source):.1%})")

        # 5. Attention dynamics (rate of change in attention)
        attention_velocity = np.linalg.norm(np.diff(all_source_importance, axis=0), axis=1)
        results['attention_dynamics'] = {
            'velocity_mean': float(np.mean(attention_velocity)),
            'velocity_std': float(np.std(attention_velocity)),
        }

        self.results['attention_patterns'] = results
        return results

    # =========================================================================
    # ANALYSIS 3: GEOGRAPHIC SIGNAL RECOVERY
    # =========================================================================

    def analyze_geographic_signals(self) -> Dict:
        """
        Recover and analyze geographic signals from raw data sources.

        Returns:
            Dict with geographic analysis results including VIIRS tiles, FIRMS hotspots, etc.
        """
        print("\n" + "="*60)
        print("ANALYSIS 3: GEOGRAPHIC SIGNAL RECOVERY")
        print("="*60)

        results = {}

        # 1. Load VIIRS tile-level data
        viirs_path = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"
        if viirs_path.exists():
            print(f"\nLoading VIIRS tile data from {viirs_path}...")
            viirs_df = pd.read_csv(viirs_path)
            viirs_df['date'] = pd.to_datetime(viirs_df['date'])

            # Analyze by tile
            tile_stats = {}
            for tile in VIIRS_TILES.keys():
                tile_data = viirs_df[viirs_df['tile'] == tile]
                if len(tile_data) > 0:
                    tile_stats[tile] = {
                        'region': VIIRS_TILES[tile]['region'],
                        'center_lat': VIIRS_TILES[tile]['center_lat'],
                        'center_lon': VIIRS_TILES[tile]['center_lon'],
                        'n_observations': len(tile_data),
                        'radiance_mean': float(tile_data['radiance_mean'].mean()),
                        'radiance_std': float(tile_data['radiance_mean'].std()),
                        'date_range': [
                            tile_data['date'].min().isoformat(),
                            tile_data['date'].max().isoformat()
                        ]
                    }

            results['viirs_tiles'] = tile_stats
            print(f"  Loaded {len(tile_stats)} VIIRS tiles")

            # Compute tile correlations
            tile_pivot = viirs_df.pivot_table(
                index='date',
                columns='tile',
                values='radiance_mean',
                aggfunc='mean'
            ).dropna()

            if len(tile_pivot) > 10:
                tile_corr = tile_pivot.corr()
                results['viirs_tile_correlations'] = tile_corr.to_dict()
                print(f"  Computed {len(tile_corr)} x {len(tile_corr)} tile correlation matrix")

        # 2. Load FIRMS hotspot data with coordinates
        firms_path = DATA_DIR / "firms" / "DL_FIRE_SV-C2_706038" / "fire_archive_SV-C2_706038.csv"
        if firms_path.exists():
            print(f"\nLoading FIRMS hotspot data from {firms_path}...")
            firms_df = pd.read_csv(firms_path)

            if 'latitude' in firms_df.columns and 'longitude' in firms_df.columns:
                # Filter to Ukraine bounds
                ukraine_bounds = {
                    'lat_min': 44.0, 'lat_max': 52.5,
                    'lon_min': 22.0, 'lon_max': 40.5
                }

                firms_df = firms_df[
                    (firms_df['latitude'] >= ukraine_bounds['lat_min']) &
                    (firms_df['latitude'] <= ukraine_bounds['lat_max']) &
                    (firms_df['longitude'] >= ukraine_bounds['lon_min']) &
                    (firms_df['longitude'] <= ukraine_bounds['lon_max'])
                ]

                print(f"  {len(firms_df)} fire hotspots within Ukraine bounds")

                # Spatial statistics
                results['firms_spatial'] = {
                    'n_hotspots': len(firms_df),
                    'lat_range': [float(firms_df['latitude'].min()), float(firms_df['latitude'].max())],
                    'lon_range': [float(firms_df['longitude'].min()), float(firms_df['longitude'].max())],
                    'lat_mean': float(firms_df['latitude'].mean()),
                    'lon_mean': float(firms_df['longitude'].mean()),
                    'lat_std': float(firms_df['latitude'].std()),
                    'lon_std': float(firms_df['longitude'].std()),
                }

                # Grid-based hotspot density
                lat_bins = np.linspace(ukraine_bounds['lat_min'], ukraine_bounds['lat_max'], 9)
                lon_bins = np.linspace(ukraine_bounds['lon_min'], ukraine_bounds['lon_max'], 9)

                hotspot_grid, _, _ = np.histogram2d(
                    firms_df['latitude'].values,
                    firms_df['longitude'].values,
                    bins=[lat_bins, lon_bins]
                )

                results['firms_spatial']['hotspot_grid'] = {
                    'counts': hotspot_grid.tolist(),
                    'lat_edges': lat_bins.tolist(),
                    'lon_edges': lon_bins.tolist(),
                }

                # Find hotspot concentration
                max_idx = np.unravel_index(np.argmax(hotspot_grid), hotspot_grid.shape)
                hotspot_center = {
                    'lat': float((lat_bins[max_idx[0]] + lat_bins[max_idx[0]+1]) / 2),
                    'lon': float((lon_bins[max_idx[1]] + lon_bins[max_idx[1]+1]) / 2),
                    'count': int(hotspot_grid[max_idx])
                }
                results['firms_spatial']['concentration_center'] = hotspot_center
                print(f"  Hotspot concentration center: ({hotspot_center['lat']:.2f}, {hotspot_center['lon']:.2f})")

        # 3. Load DeepState territorial data
        deepstate_dir = DATA_DIR / "deepstate" / "daily"
        if deepstate_dir.exists():
            print(f"\nAnalyzing DeepState territorial data...")
            geojson_files = sorted(deepstate_dir.glob("*.geojson"))[:100]  # Sample first 100

            territorial_stats = []
            for gj_file in geojson_files:
                try:
                    with open(gj_file) as f:
                        data = json.load(f)

                    features = data.get('features', [])

                    # Extract date from filename
                    date_str = gj_file.stem.split('_')[-1] if '_' in gj_file.stem else None

                    # Count geometry types
                    geom_counts = {'Polygon': 0, 'MultiPolygon': 0, 'Point': 0, 'LineString': 0}
                    for feat in features:
                        geom_type = feat.get('geometry', {}).get('type', '')
                        if geom_type in geom_counts:
                            geom_counts[geom_type] += 1

                    territorial_stats.append({
                        'file': gj_file.name,
                        'date': date_str,
                        'n_features': len(features),
                        **geom_counts
                    })
                except Exception as e:
                    continue

            if territorial_stats:
                results['deepstate_territorial'] = {
                    'n_files_analyzed': len(territorial_stats),
                    'total_features': sum(s['n_features'] for s in territorial_stats),
                    'avg_features_per_file': np.mean([s['n_features'] for s in territorial_stats]),
                    'geometry_totals': {
                        'Polygon': sum(s['Polygon'] for s in territorial_stats),
                        'MultiPolygon': sum(s['MultiPolygon'] for s in territorial_stats),
                        'Point': sum(s['Point'] for s in territorial_stats),
                        'LineString': sum(s['LineString'] for s in territorial_stats),
                    }
                }
                print(f"  Analyzed {len(territorial_stats)} territorial files")

        self.results['geographic_signals'] = results
        return results

    # =========================================================================
    # ANALYSIS 4: TEMPORAL-SPATIAL CORRELATION STRUCTURE
    # =========================================================================

    def analyze_temporal_spatial_correlations(self) -> Dict:
        """
        Analyze correlations between model representations and spatial features.

        Returns:
            Dict with correlation analysis between latents and geographic signals.
        """
        print("\n" + "="*60)
        print("ANALYSIS 4: TEMPORAL-SPATIAL CORRELATION STRUCTURE")
        print("="*60)

        # This analysis correlates model latent representations with
        # geographic features recovered from raw data

        results = {}

        # Need latent trajectories first
        if 'latent_trajectories' not in self.results:
            self.analyze_latent_trajectories()

        latent_data = self.results['latent_trajectories']

        # Load VIIRS tile data for correlation
        viirs_path = DATA_DIR / "nasa" / "viirs_nightlights" / "viirs_daily_brightness_stats.csv"
        if viirs_path.exists():
            viirs_df = pd.read_csv(viirs_path)
            viirs_df['date'] = pd.to_datetime(viirs_df['date'])

            # Create tile-level time series
            tile_series = {}
            for tile in VIIRS_TILES.keys():
                tile_data = viirs_df[viirs_df['tile'] == tile].set_index('date')['radiance_mean']
                tile_series[tile] = tile_data

            # Correlate latent PCA components with tile radiance
            pca_components = np.array(latent_data['pca']['components'])
            latent_dates = [datetime.fromisoformat(d) for d in latent_data['dates']]

            tile_latent_correlations = {}
            for tile, tile_data in tile_series.items():
                correlations = []
                for pc_idx in range(pca_components.shape[1]):
                    pc_values = pca_components[:, pc_idx]

                    # Align dates
                    aligned_tile = []
                    aligned_pc = []
                    for i, date in enumerate(latent_dates):
                        if date in tile_data.index:
                            aligned_tile.append(tile_data[date])
                            aligned_pc.append(pc_values[i])

                    if len(aligned_tile) > 10:
                        r, p = stats.pearsonr(aligned_tile, aligned_pc)
                        correlations.append({
                            'pc': pc_idx + 1,
                            'r': float(r),
                            'p': float(p),
                            'n': len(aligned_tile)
                        })

                tile_latent_correlations[tile] = correlations

            results['tile_latent_correlations'] = tile_latent_correlations

            # Find strongest correlations
            strongest = []
            for tile, corrs in tile_latent_correlations.items():
                for c in corrs:
                    if c['p'] < 0.05:
                        strongest.append({
                            'tile': tile,
                            'region': VIIRS_TILES[tile]['region'],
                            **c
                        })

            strongest.sort(key=lambda x: abs(x['r']), reverse=True)
            results['strongest_correlations'] = strongest[:10]

            print("\n  Top tile-latent correlations (p < 0.05):")
            for s in strongest[:5]:
                print(f"    {s['tile']} ({s['region']}) PC{s['pc']}: r={s['r']:.3f}, p={s['p']:.4f}")

        # Compute representational similarity analysis (RSA)
        print("\n  Computing RSA (latent space structure)...")

        # Sample latent distances
        sample_size = min(500, len(pca_components))
        sample_idx = np.random.choice(len(pca_components), sample_size, replace=False)
        sampled_latents = pca_components[sample_idx]

        # Latent distance matrix
        latent_distances = squareform(pdist(sampled_latents, metric='euclidean'))

        # Temporal distance matrix (days between observations)
        sampled_dates = [latent_dates[i] for i in sample_idx]
        temporal_distances = np.zeros((sample_size, sample_size))
        for i in range(sample_size):
            for j in range(sample_size):
                temporal_distances[i, j] = abs((sampled_dates[i] - sampled_dates[j]).days)

        # RSA: correlation between distance matrices
        latent_flat = latent_distances[np.triu_indices(sample_size, k=1)]
        temporal_flat = temporal_distances[np.triu_indices(sample_size, k=1)]

        rsa_r, rsa_p = stats.spearmanr(latent_flat, temporal_flat)

        results['rsa'] = {
            'latent_temporal_correlation': float(rsa_r),
            'p_value': float(rsa_p),
            'interpretation': 'Positive = similar timepoints have similar representations'
        }
        print(f"\n  RSA (latent vs temporal distance): r={rsa_r:.3f}, p={rsa_p:.4f}")

        self.results['temporal_spatial_correlations'] = results
        return results

    # =========================================================================
    # ANALYSIS 5: FINE-GRAINED TEMPORAL DECOMPOSITION
    # =========================================================================

    def analyze_fine_grained_temporal(self) -> Dict:
        """
        Analyze temporal patterns at fine granularity (daily level).

        Returns:
            Dict with daily-level analysis results.
        """
        print("\n" + "="*60)
        print("ANALYSIS 5: FINE-GRAINED TEMPORAL DECOMPOSITION")
        print("="*60)

        results = {}

        # Sample a few sequences and analyze daily patterns
        n_sequences = min(10, len(self.dataset))
        indices = np.linspace(0, len(self.dataset) - 1, n_sequences, dtype=int)

        daily_patterns = []

        print(f"Analyzing {n_sequences} sequences at daily granularity...")

        for idx in indices:
            sample = self.dataset[idx]

            # Analyze daily features directly (before model processing)
            daily_features = {}
            for source_name, features in sample.daily_features.items():
                feat_np = features.numpy() if isinstance(features, torch.Tensor) else features
                daily_features[source_name] = {
                    'mean_per_day': np.mean(feat_np, axis=1).tolist(),  # [n_days]
                    'std_per_day': np.std(feat_np, axis=1).tolist(),
                    'n_days': feat_np.shape[0],
                    'n_features': feat_np.shape[1],
                }

            daily_patterns.append({
                'sample_idx': int(idx),
                'sources': daily_features
            })

        results['daily_patterns'] = daily_patterns

        # Aggregate statistics across sequences
        aggregated = {}
        for source_name in self.model_info['daily_sources']:
            source_means = []
            for pattern in daily_patterns:
                if source_name in pattern['sources']:
                    source_means.extend(pattern['sources'][source_name]['mean_per_day'])

            if source_means:
                aggregated[source_name] = {
                    'mean': float(np.mean(source_means)),
                    'std': float(np.std(source_means)),
                    'min': float(np.min(source_means)),
                    'max': float(np.max(source_means)),
                }

        results['aggregated_daily_stats'] = aggregated

        print("\n  Daily source statistics:")
        for source, source_stats in aggregated.items():
            print(f"    {source}: {source_stats['mean']:.4f} ± {source_stats['std']:.4f}")

        # Analyze temporal autocorrelation
        print("\n  Computing temporal autocorrelation...")

        autocorr_lags = [1, 7, 30, 90]  # 1 day, 1 week, 1 month, 3 months
        autocorr_results = {}

        for source_name in self.model_info['daily_sources'][:3]:  # First 3 sources
            source_series = []
            for pattern in daily_patterns:
                if source_name in pattern['sources']:
                    source_series.extend(pattern['sources'][source_name]['mean_per_day'])

            if len(source_series) > max(autocorr_lags) + 10:
                series = np.array(source_series)
                autocorrs = {}
                for lag in autocorr_lags:
                    if lag < len(series) - 1:
                        r, p = stats.pearsonr(series[:-lag], series[lag:])
                        autocorrs[f'lag_{lag}'] = {'r': float(r), 'p': float(p)}
                autocorr_results[source_name] = autocorrs

        results['temporal_autocorrelation'] = autocorr_results

        for source, lags in autocorr_results.items():
            print(f"\n    {source} autocorrelation:")
            for lag_name, stats_dict in lags.items():
                print(f"      {lag_name}: r={stats_dict['r']:.3f}")

        self.results['fine_grained_temporal'] = results
        return results

    # =========================================================================
    # RUN ALL ANALYSES
    # =========================================================================

    def run_all_analyses(self) -> Dict:
        """Run all spatio-temporal analyses."""
        print("\n" + "="*70)
        print("SPATIO-TEMPORAL REPRESENTATION ANALYSIS")
        print("="*70)
        print(f"Model: {self.config.checkpoint_path.name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.config.output_dir}")

        # Run each analysis
        self.analyze_latent_trajectories()
        self.analyze_attention_patterns()
        self.analyze_geographic_signals()
        self.analyze_temporal_spatial_correlations()
        self.analyze_fine_grained_temporal()

        # Generate visualizations
        self.generate_visualizations()

        # Save results
        self.save_results()

        return self.results

    def generate_visualizations(self):
        """Generate visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("Warning: matplotlib not available, skipping visualizations")
            return

        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        # 1. Latent trajectory in PCA space
        if 'latent_trajectories' in self.results:
            print("  Creating latent trajectory plot...")

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            pca_data = np.array(self.results['latent_trajectories']['pca']['components'])
            phases = self.results['latent_trajectories']['phases']

            # Color by phase
            phase_colors = {
                'Initial Invasion': 'red',
                'Donbas Focus': 'orange',
                'Kharkiv Counteroffensive': 'green',
                'Winter Stalemate': 'blue',
                '2023 Counteroffensive': 'purple',
                'Attritional Warfare': 'brown',
                'Kursk Incursion': 'pink',
            }
            colors = [phase_colors.get(p, 'gray') for p in phases]

            # PC1 vs PC2
            axes[0, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=colors, alpha=0.5, s=10)
            axes[0, 0].set_xlabel('PC1')
            axes[0, 0].set_ylabel('PC2')
            axes[0, 0].set_title('Latent Trajectory (PC1 vs PC2)')

            # PC1 vs PC3
            axes[0, 1].scatter(pca_data[:, 0], pca_data[:, 2], c=colors, alpha=0.5, s=10)
            axes[0, 1].set_xlabel('PC1')
            axes[0, 1].set_ylabel('PC3')
            axes[0, 1].set_title('Latent Trajectory (PC1 vs PC3)')

            # Velocity over time
            velocities = self.results['latent_trajectories']['velocity']['values']
            axes[1, 0].plot(velocities, alpha=0.7)
            axes[1, 0].set_xlabel('Time (samples)')
            axes[1, 0].set_ylabel('Latent Velocity')
            axes[1, 0].set_title('Representation Change Rate Over Time')

            # Explained variance
            var_ratio = self.results['latent_trajectories']['pca']['explained_variance_ratio']
            axes[1, 1].bar(range(1, len(var_ratio) + 1), var_ratio)
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Variance Explained')
            axes[1, 1].set_title('PCA Explained Variance')

            # Add legend
            legend_patches = [plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=c, markersize=8, label=p)
                            for p, c in phase_colors.items()]
            fig.legend(handles=legend_patches, loc='center right', bbox_to_anchor=(1.12, 0.5))

            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'latent_trajectories.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 2. Attention patterns
        if 'attention_patterns' in self.results:
            print("  Creating attention pattern plot...")

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Overall source importance
            importance = self.results['attention_patterns']['overall_importance']
            sources = list(importance.keys())
            means = [importance[s]['mean'] for s in sources]
            stds = [importance[s]['std'] for s in sources]

            axes[0, 0].bar(sources, means, yerr=stds, capsize=5)
            axes[0, 0].set_ylabel('Importance Weight')
            axes[0, 0].set_title('Overall Source Importance')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Source dominance
            dominance = self.results['attention_patterns']['source_dominance']
            axes[0, 1].pie(dominance.values(), labels=dominance.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Source Dominance Frequency')

            # Phase-specific importance
            phase_importance = self.results['attention_patterns']['phase_importance']
            phase_names = list(phase_importance.keys())[:5]  # Limit to 5 phases

            x = np.arange(len(sources))
            width = 0.15

            for i, phase in enumerate(phase_names):
                phase_vals = [phase_importance[phase].get(s, 0) for s in sources]
                axes[1, 0].bar(x + i * width, phase_vals, width, label=phase[:15])

            axes[1, 0].set_xticks(x + width * len(phase_names) / 2)
            axes[1, 0].set_xticklabels(sources, rotation=45)
            axes[1, 0].set_ylabel('Importance')
            axes[1, 0].set_title('Source Importance by Conflict Phase')
            axes[1, 0].legend(fontsize=8)

            # Attention entropy over time
            entropy = self.results['attention_patterns']['attention_entropy']['values']
            axes[1, 1].plot(entropy, alpha=0.7)
            axes[1, 1].axhline(self.results['attention_patterns']['attention_entropy']['mean'],
                              color='red', linestyle='--', label='Mean')
            axes[1, 1].axhline(self.results['attention_patterns']['attention_entropy']['max_possible'],
                              color='green', linestyle=':', label='Max (uniform)')
            axes[1, 1].set_xlabel('Time (samples)')
            axes[1, 1].set_ylabel('Attention Entropy')
            axes[1, 1].set_title('Attention Entropy Over Time')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'attention_patterns.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Geographic signals
        if 'geographic_signals' in self.results and 'firms_spatial' in self.results['geographic_signals']:
            print("  Creating geographic analysis plot...")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # FIRMS hotspot grid
            firms = self.results['geographic_signals']['firms_spatial']
            grid = np.array(firms['hotspot_grid']['counts'])

            im = axes[0].imshow(grid, cmap='YlOrRd', aspect='auto', origin='lower')
            axes[0].set_xlabel('Longitude bins (W → E)')
            axes[0].set_ylabel('Latitude bins (S → N)')
            axes[0].set_title('FIRMS Fire Hotspot Density')
            plt.colorbar(im, ax=axes[0], label='Count')

            # Mark concentration center
            center = firms['concentration_center']
            # Convert to grid coordinates (approximate)
            lat_bins = np.array(firms['hotspot_grid']['lat_edges'])
            lon_bins = np.array(firms['hotspot_grid']['lon_edges'])
            center_x = np.argmin(np.abs(lon_bins - center['lon']))
            center_y = np.argmin(np.abs(lat_bins - center['lat']))
            axes[0].scatter([center_x], [center_y], marker='*', s=200, c='blue',
                           label=f"Peak: {center['count']} fires")
            axes[0].legend()

            # VIIRS tile comparison
            if 'viirs_tiles' in self.results['geographic_signals']:
                tiles = self.results['geographic_signals']['viirs_tiles']
                tile_names = list(tiles.keys())
                radiances = [tiles[t]['radiance_mean'] for t in tile_names]
                regions = [tiles[t]['region'] for t in tile_names]

                bars = axes[1].bar(range(len(tile_names)), radiances, color='navy', alpha=0.7)
                axes[1].set_xticks(range(len(tile_names)))
                axes[1].set_xticklabels([f"{t}\n({r})" for t, r in zip(tile_names, regions)],
                                       rotation=45, ha='right', fontsize=8)
                axes[1].set_ylabel('Mean Radiance')
                axes[1].set_title('VIIRS Tile Radiance (Regional Brightness)')

            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'geographic_signals.png', dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Visualizations saved to {self.config.output_dir}")

    def save_results(self):
        """Save analysis results to JSON."""
        output_file = self.config.output_dir / 'spatio_temporal_analysis.json'

        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj

        results_json = convert(self.results)

        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    """Run the spatio-temporal analysis."""
    config = SpatioTemporalConfig()

    if not config.checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {config.checkpoint_path}")
        return

    analyzer = SpatioTemporalAnalyzer(config)
    results = analyzer.run_all_analyses()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")

    return results


if __name__ == "__main__":
    main()
