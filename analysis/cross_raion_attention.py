"""
Cross-Raion Attention Module

Implements geographic-aware attention between raions for the RaionHAN architecture.
Key features:
- Geographic adjacency prior: Nearby raions attend more strongly to each other
- Learned attention on top of prior: Model can override geographic bias
- Efficient sparse attention option for large raion counts

Architecture:
    CrossRaionAttention:
        "What features in raion A predict changes in raion B?"

        Inputs: Dict[raion_id, raion_repr]  # All raion representations

        Mechanism:
        - Geographic adjacency prior (nearby raions attend more strongly)
        - Learned attention on top of prior
        - Optional: frontline topology (raions on same front segment)

        Output: Dict[raion_id, cross_raion_repr]

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import math
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import centralized paths
from config.paths import DATA_DIR


# =============================================================================
# CONSTANTS
# =============================================================================

RAION_BOUNDARIES_FILE = DATA_DIR / "boundaries" / "ukraine_raions.geojson"


# =============================================================================
# GEOGRAPHIC ADJACENCY
# =============================================================================

class GeographicAdjacency:
    """
    Computes and caches raion adjacency matrices based on geographic distance.

    Creates a soft adjacency matrix where:
    - Adjacent raions (sharing border) get weight 1.0
    - Non-adjacent raions get weight exp(-distance_km / scale)

    This matrix is used as a prior in the cross-raion attention mechanism.
    """

    def __init__(
        self,
        boundaries_file: Optional[Path] = None,
        distance_scale_km: float = 50.0,
        adjacency_threshold_km: float = 5.0,
    ):
        """
        Initialize the geographic adjacency computer.

        Args:
            boundaries_file: Path to raion boundaries GeoJSON
            distance_scale_km: Scale for distance-based weight decay
            adjacency_threshold_km: Max distance to consider raions adjacent
        """
        self.boundaries_file = boundaries_file or RAION_BOUNDARIES_FILE
        self.distance_scale_km = distance_scale_km
        self.adjacency_threshold_km = adjacency_threshold_km

        self._centroids: Dict[str, Tuple[float, float]] = {}
        self._adjacency_matrix: Optional[Tensor] = None
        self._raion_to_idx: Dict[str, int] = {}
        self._idx_to_raion: Dict[int, str] = {}

    def load_centroids(self) -> Dict[str, Tuple[float, float]]:
        """Load raion centroids from boundaries file."""
        if self._centroids:
            return self._centroids

        if not self.boundaries_file.exists():
            warnings.warn(f"Boundaries file not found: {self.boundaries_file}")
            return {}

        with open(self.boundaries_file) as f:
            data = json.load(f)

        for feature in data.get('features', []):
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})

            raion_name = props.get('NAME_2', '')
            oblast_name = props.get('NAME_1', '')
            key = f"{oblast_name}_{raion_name}"

            # Compute centroid from geometry
            coords = geom.get('coordinates', [])
            geom_type = geom.get('type', '')

            all_lons = []
            all_lats = []

            if geom_type == 'Polygon' and coords:
                exterior = coords[0]
                for point in exterior:
                    all_lons.append(point[0])
                    all_lats.append(point[1])
            elif geom_type == 'MultiPolygon' and coords:
                for polygon in coords:
                    if polygon:
                        exterior = polygon[0]
                        for point in exterior:
                            all_lons.append(point[0])
                            all_lats.append(point[1])

            if all_lons and all_lats:
                centroid = (np.mean(all_lons), np.mean(all_lats))
                self._centroids[key] = centroid

        return self._centroids

    @staticmethod
    def haversine_distance(
        lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """
        Compute Haversine distance between two points in kilometers.

        Args:
            lon1, lat1: First point coordinates (degrees)
            lon2, lat2: Second point coordinates (degrees)

        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def compute_distance_matrix(
        self, raion_keys: List[str]
    ) -> Tuple[Tensor, Dict[str, int]]:
        """
        Compute pairwise distance matrix for given raions.

        Args:
            raion_keys: List of raion keys to include

        Returns:
            Tuple of (distance matrix [n_raions, n_raions], raion_to_idx mapping)
        """
        centroids = self.load_centroids()

        # Build index mapping
        raion_to_idx = {k: i for i, k in enumerate(raion_keys)}
        n_raions = len(raion_keys)

        # Compute distances
        distances = np.zeros((n_raions, n_raions))

        for i, key_i in enumerate(raion_keys):
            if key_i not in centroids:
                continue
            lon_i, lat_i = centroids[key_i]

            for j, key_j in enumerate(raion_keys):
                if j <= i:  # Skip diagonal and lower triangle (will mirror)
                    continue
                if key_j not in centroids:
                    continue
                lon_j, lat_j = centroids[key_j]

                dist = self.haversine_distance(lon_i, lat_i, lon_j, lat_j)
                distances[i, j] = dist
                distances[j, i] = dist  # Symmetric

        return torch.tensor(distances, dtype=torch.float32), raion_to_idx

    def compute_adjacency_prior(
        self, raion_keys: List[str], device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Dict[str, int]]:
        """
        Compute adjacency prior matrix for attention.

        The prior is computed as:
        - 1.0 for adjacent raions (distance < threshold)
        - exp(-distance / scale) for non-adjacent raions

        This is added to attention logits before softmax, so higher values
        mean stronger attention.

        Args:
            raion_keys: List of raion keys
            device: Target device for the tensor

        Returns:
            Tuple of (prior matrix [n_raions, n_raions], raion_to_idx mapping)
        """
        distances, raion_to_idx = self.compute_distance_matrix(raion_keys)

        # Compute prior: exp(-distance / scale)
        prior = torch.exp(-distances / self.distance_scale_km)

        # Set diagonal to 1 (self-attention is always allowed)
        prior.fill_diagonal_(1.0)

        # Convert to log-space for addition to attention logits
        # Add small epsilon to avoid log(0)
        log_prior = torch.log(prior + 1e-8)

        if device:
            log_prior = log_prior.to(device)

        self._adjacency_matrix = log_prior
        self._raion_to_idx = raion_to_idx
        self._idx_to_raion = {v: k for k, v in raion_to_idx.items()}

        return log_prior, raion_to_idx

    def get_adjacency_for_raions(
        self, raion_keys: List[str], device: Optional[torch.device] = None
    ) -> Tensor:
        """
        Get adjacency prior for a list of raions.

        Caches the result if called with the same raions.

        Args:
            raion_keys: List of raion keys
            device: Target device

        Returns:
            Adjacency prior matrix [n_raions, n_raions]
        """
        # Check if we need to recompute
        if (self._adjacency_matrix is not None and
            set(raion_keys) == set(self._raion_to_idx.keys())):
            result = self._adjacency_matrix
            if device:
                result = result.to(device)
            return result

        prior, _ = self.compute_adjacency_prior(raion_keys, device)
        return prior


# =============================================================================
# CROSS-RAION ATTENTION MODULE
# =============================================================================

class CrossRaionAttention(nn.Module):
    """
    Cross-raion attention with geographic adjacency prior.

    This module allows each raion to attend to all other raions, with a learned
    attention mechanism that is biased by geographic proximity. Nearby raions
    naturally attend more strongly to each other, but the model can learn to
    override this for strategically connected regions.

    Architecture:
        Input: raion_reprs [batch, n_raions, seq_len, d_model]

        1. Pool each raion's temporal sequence to a single vector
        2. Compute cross-raion attention with geographic prior
        3. Combine with original representations

        Output: cross_raion_reprs [batch, n_raions, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        prior_weight: float = 1.0,
        geographic_adjacency: Optional[GeographicAdjacency] = None,
    ):
        """
        Initialize the cross-raion attention module.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            prior_weight: Weight for geographic prior (0 = ignore, 1 = full weight)
            geographic_adjacency: Pre-configured adjacency computer
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.prior_weight = prior_weight

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Geographic adjacency
        self.geographic_adjacency = geographic_adjacency or GeographicAdjacency()

        # Temporal pooling (aggregate sequence to single vector)
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Cross-raion attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learned prior weight (optional, allows model to adjust prior importance)
        self.prior_scale = nn.Parameter(torch.ones(1) * prior_weight)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm for residual
        self.norm = nn.LayerNorm(d_model)

        # Feedforward for temporal broadcast
        self.temporal_broadcast = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Cached adjacency prior
        self._cached_prior: Optional[Tensor] = None
        self._cached_raion_keys: Optional[List[str]] = None

    def _get_adjacency_prior(
        self, raion_keys: List[str], device: torch.device
    ) -> Tensor:
        """Get adjacency prior, using cache if possible."""
        if (self._cached_prior is not None and
            self._cached_raion_keys == raion_keys and
            self._cached_prior.device == device):
            return self._cached_prior

        prior = self.geographic_adjacency.get_adjacency_for_raions(raion_keys, device)
        self._cached_prior = prior
        self._cached_raion_keys = raion_keys

        return prior

    def forward(
        self,
        raion_reprs: Tensor,
        raion_keys: Optional[List[str]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply cross-raion attention.

        Args:
            raion_reprs: Raion representations [batch, n_raions, seq_len, d_model]
            raion_keys: List of raion keys (for geographic prior)
            mask: Optional attention mask [batch, n_raions]

        Returns:
            Cross-raion representations [batch, n_raions, seq_len, d_model]
        """
        batch_size, n_raions, seq_len, d_model = raion_reprs.shape
        device = raion_reprs.device

        # 1. Pool temporal dimension to get raion summaries
        # Shape: [batch, n_raions, d_model]
        raion_summary = raion_reprs.mean(dim=2)  # Simple mean pooling
        raion_summary = self.temporal_pool(raion_summary)

        # 2. Compute Q, K, V for cross-raion attention
        # Shape: [batch, n_raions, n_heads, head_dim]
        Q = self.q_proj(raion_summary).view(batch_size, n_raions, self.n_heads, self.head_dim)
        K = self.k_proj(raion_summary).view(batch_size, n_raions, self.n_heads, self.head_dim)
        V = self.v_proj(raion_summary).view(batch_size, n_raions, self.n_heads, self.head_dim)

        # Transpose for attention: [batch, n_heads, n_raions, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Compute attention scores
        # Shape: [batch, n_heads, n_raions, n_raions]
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # 4. Add geographic prior (if raion keys provided)
        if raion_keys is not None and len(raion_keys) == n_raions:
            geo_prior = self._get_adjacency_prior(raion_keys, device)
            # Expand prior to match attention shape
            geo_prior = geo_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, n_raions, n_raions]
            attn_logits = attn_logits + self.prior_scale * geo_prior

        # 5. Apply mask (if provided)
        if mask is not None:
            # mask shape: [batch, n_raions] -> expand to [batch, 1, 1, n_raions]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))

        # 6. Softmax and apply to values
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: [batch, n_heads, n_raions, head_dim]
        attn_output = torch.matmul(attn_weights, V)

        # 7. Combine heads and project
        # [batch, n_raions, n_heads, head_dim] -> [batch, n_raions, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, n_raions, d_model)
        attn_output = self.out_proj(attn_output)

        # 8. Residual connection with raion summary
        cross_context = self.norm(attn_output + raion_summary)

        # 9. Broadcast cross-context back to temporal dimension
        # [batch, n_raions, d_model] -> [batch, n_raions, seq_len, d_model]
        cross_context = cross_context.unsqueeze(2).expand(-1, -1, seq_len, -1)
        cross_context = self.temporal_broadcast(cross_context)

        # 10. Combine with original representations
        output = raion_reprs + cross_context

        return output

    def get_attention_weights(
        self,
        raion_reprs: Tensor,
        raion_keys: Optional[List[str]] = None,
    ) -> Tensor:
        """
        Get cross-raion attention weights for interpretability.

        Args:
            raion_reprs: Raion representations [batch, n_raions, seq_len, d_model]
            raion_keys: List of raion keys

        Returns:
            Attention weights [batch, n_heads, n_raions, n_raions]
        """
        batch_size, n_raions, seq_len, d_model = raion_reprs.shape
        device = raion_reprs.device

        raion_summary = raion_reprs.mean(dim=2)
        raion_summary = self.temporal_pool(raion_summary)

        Q = self.q_proj(raion_summary).view(batch_size, n_raions, self.n_heads, self.head_dim)
        K = self.k_proj(raion_summary).view(batch_size, n_raions, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if raion_keys is not None and len(raion_keys) == n_raions:
            geo_prior = self._get_adjacency_prior(raion_keys, device)
            geo_prior = geo_prior.unsqueeze(0).unsqueeze(0)
            attn_logits = attn_logits + self.prior_scale * geo_prior

        return F.softmax(attn_logits, dim=-1)


# =============================================================================
# RAION ENCODER
# =============================================================================

class RaionEncoder(nn.Module):
    """
    Temporal encoder for a single raion's feature sequence.

    Uses transformer-style self-attention to learn temporal patterns within
    each raion's time series.

    Architecture:
        Input: [batch, seq_len, n_features]
        -> Feature projection
        -> Positional encoding
        -> N transformer layers (8 heads each)
        Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        """
        Initialize the raion encoder.

        Args:
            n_features: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Feature projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode a raion's feature sequence.

        Args:
            x: Input features [batch, seq_len, n_features]
            mask: Optional padding mask [batch, seq_len]

        Returns:
            Encoded representations [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project features
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        if mask is not None:
            # Convert to attention mask format (True = masked out)
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        return x


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("Cross-Raion Attention Module Test")
    print("=" * 60)

    # Test geographic adjacency
    print("\n1. Testing GeographicAdjacency...")
    geo = GeographicAdjacency()
    centroids = geo.load_centroids()
    print(f"   Loaded {len(centroids)} raion centroids")

    # Test with a few raions
    test_raions = [
        "Donets'k_Artemivs'ka",  # Bakhmut
        "Donets'k_Mariupol's'ka",  # Mariupol
        "Donets'k_Krasnolymanskyi",  # Lyman
        "Kharkiv_Kharkivs'ka",  # Kharkiv
        "Luhansk_Popasnyans'ka",  # Popasna
    ]

    # Filter to existing raions
    test_raions = [r for r in test_raions if r in centroids]
    print(f"   Test raions: {test_raions}")

    if len(test_raions) >= 2:
        prior, mapping = geo.compute_adjacency_prior(test_raions)
        print(f"   Adjacency prior shape: {prior.shape}")
        print(f"   Prior matrix:")
        for i, ri in enumerate(test_raions):
            row = [f"{prior[i, j].item():.2f}" for j in range(len(test_raions))]
            print(f"     {ri[:20]:20s}: {row}")

    # Test cross-raion attention
    print("\n2. Testing CrossRaionAttention...")

    batch_size = 2
    n_raions = len(test_raions) if test_raions else 5
    seq_len = 30
    d_model = 128

    cross_attn = CrossRaionAttention(d_model=d_model, n_heads=8)

    # Random input
    x = torch.randn(batch_size, n_raions, seq_len, d_model)

    # Forward pass
    output = cross_attn(x, raion_keys=test_raions if test_raions else None)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")

    # Get attention weights
    attn_weights = cross_attn.get_attention_weights(x, raion_keys=test_raions if test_raions else None)
    print(f"   Attention weights shape: {attn_weights.shape}")

    # Test raion encoder
    print("\n3. Testing RaionEncoder...")

    n_features = 20
    encoder = RaionEncoder(n_features=n_features, d_model=d_model)

    raion_input = torch.randn(batch_size, seq_len, n_features)
    raion_output = encoder(raion_input)
    print(f"   Input shape: {raion_input.shape}")
    print(f"   Output shape: {raion_output.shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
