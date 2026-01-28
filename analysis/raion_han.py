"""
Raion-Level Hierarchical Attention Network (RaionHAN)

Complete model for fine-grained tactical prediction at the raion (district) level.
Combines multi-source raion encoders, cross-raion attention with geographic prior,
macro-temporal context, and per-raion prediction heads.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAION SOURCE ENCODERS                            │
│  [RaionEncoder] for each active raion (temporal attention within raion)  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CROSS-RAION ATTENTION                              │
│  "What in raion A predicts raion B?" (with geographic adjacency prior)  │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       MACRO-TEMPORAL CONTEXT                             │
│  National equipment/personnel + monthly indicators + narratives          │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TEMPORO-SPATIAL FUSION                                │
│  "What macro patterns predict this raion?"                               │
└─────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RAION PREDICTION HEADS                              │
│  Per-raion forecasts: [batch, horizon, n_features_raion]                │
└─────────────────────────────────────────────────────────────────────────┘

Author: ML Engineering Team
Date: 2026-01-27
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import components
from analysis.cross_raion_attention import (
    CrossRaionAttention,
    RaionEncoder,
    GeographicAdjacency,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RaionHANConfig:
    """Configuration for RaionHAN model."""
    # Model dimensions
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 2
    dropout: float = 0.1

    # Raion configuration
    max_raions: int = 50
    n_raion_features: int = 20  # Features per raion

    # Macro encoder
    n_macro_features: int = 30  # National-level features
    n_macro_layers: int = 2

    # Prediction
    forecast_horizon: int = 7  # Days to predict

    # Sequence lengths
    max_seq_len: int = 365  # Maximum temporal sequence

    # Geographic prior
    prior_weight: float = 1.0
    distance_scale_km: float = 50.0


# =============================================================================
# MACRO ENCODER
# =============================================================================

class MacroEncoder(nn.Module):
    """
    Encoder for national-level (macro) features.

    Processes aggregated national data that affects all raions:
    - Equipment losses (not regionalizable)
    - Personnel casualties
    - Monthly humanitarian indicators
    - ISW narrative embeddings (strategic context)

    Input: [batch, seq_len, n_macro_features]
    Output: [batch, seq_len, d_model]
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 365,
    ):
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
        Encode macro features.

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
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        return x


# =============================================================================
# TEMPORO-SPATIAL FUSION
# =============================================================================

class TemporoSpatialFusion(nn.Module):
    """
    Fuses macro-temporal context with raion representations.

    Each raion attends to the macro context to learn which national-level
    signals are most relevant for predicting that raion's state.

    - Frontline raions may weight military signals higher
    - Rear raions may weight logistics/humanitarian signals higher

    Input:
        - raion_reprs: [batch, n_raions, seq_len, d_model]
        - macro_context: [batch, seq_len, d_model]

    Output:
        - fused_reprs: [batch, n_raions, seq_len, d_model]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Cross-attention: raion attends to macro
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raion_reprs: Tensor,
        macro_context: Tensor,
        macro_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse macro context with raion representations.

        Args:
            raion_reprs: [batch, n_raions, seq_len, d_model]
            macro_context: [batch, seq_len, d_model]
            macro_mask: [batch, seq_len]

        Returns:
            Fused representations [batch, n_raions, seq_len, d_model]
        """
        batch_size, n_raions, seq_len, d_model = raion_reprs.shape

        # Process each raion separately (could be parallelized with reshape)
        fused_list = []

        for r in range(n_raions):
            raion_r = raion_reprs[:, r, :, :]  # [batch, seq_len, d_model]

            # Cross-attention: raion attends to macro
            attended, _ = self.cross_attn(
                query=raion_r,
                key=macro_context,
                value=macro_context,
                key_padding_mask=~macro_mask if macro_mask is not None else None,
            )

            # Add & norm
            x = self.norm1(raion_r + attended)

            # Feedforward
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

            # Gating: learn how much to mix macro info with original
            gate_input = torch.cat([x, raion_r], dim=-1)
            gate = self.gate(gate_input)
            x = gate * x + (1 - gate) * raion_r

            fused_list.append(x)

        # Stack back to [batch, n_raions, seq_len, d_model]
        fused_reprs = torch.stack(fused_list, dim=1)

        return fused_reprs


# =============================================================================
# RAION PREDICTION HEAD
# =============================================================================

class RaionForecastHead(nn.Module):
    """
    Prediction head for forecasting raion-level features.

    Takes the fused representation for a raion and predicts the next
    `horizon` days of that raion's features.

    Input: [batch, seq_len, d_model]
    Output: [batch, horizon, n_features]
    """

    def __init__(
        self,
        d_model: int = 128,
        n_features: int = 20,
        horizon: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.n_features = n_features

        # Context aggregation
        self.context_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Forecast network
        self.forecast_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * n_features),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Generate forecast for this raion.

        Args:
            x: Raion representation [batch, seq_len, d_model]
            mask: Optional mask [batch, seq_len]

        Returns:
            Forecast [batch, horizon, n_features]
        """
        batch_size = x.shape[0]

        # Self-attention to aggregate context
        attended, _ = self.context_attn(x, x, x)

        # Use last timestep for prediction
        last_repr = attended[:, -1, :]

        # Generate forecast
        flat_pred = self.forecast_net(last_repr)
        forecast = flat_pred.view(batch_size, self.horizon, self.n_features)

        return forecast


# =============================================================================
# FULL RAION HAN MODEL
# =============================================================================

class RaionHAN(nn.Module):
    """
    Raion-Level Hierarchical Attention Network.

    Complete model for fine-grained tactical prediction at raion level.
    Combines:
    1. Per-raion encoders (shared weights + raion embedding)
    2. Cross-raion attention with geographic prior
    3. Macro encoder for national-level features
    4. Temporo-spatial fusion
    5. Per-raion forecast heads (shared weights + raion embedding)
    """

    def __init__(self, config: RaionHANConfig):
        super().__init__()

        self.config = config
        self.d_model = config.d_model

        # Raion embedding (for shared encoder)
        self.raion_embedding = nn.Embedding(config.max_raions, config.d_model)

        # Shared raion encoder
        self.raion_encoder = RaionEncoder(
            n_features=config.n_raion_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Cross-raion attention
        self.geographic_adjacency = GeographicAdjacency(
            distance_scale_km=config.distance_scale_km,
        )
        self.cross_raion_attention = CrossRaionAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            prior_weight=config.prior_weight,
            geographic_adjacency=self.geographic_adjacency,
        )

        # Macro encoder
        self.macro_encoder = MacroEncoder(
            n_features=config.n_macro_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_macro_layers,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )

        # Temporo-spatial fusion
        self.fusion = TemporoSpatialFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Shared forecast head with raion-specific output projection
        self.shared_forecast = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
        )

        self.forecast_output = nn.Linear(
            config.d_model,
            config.forecast_horizon * config.n_raion_features,
        )

        # Raion key to index mapping (set during forward)
        self._raion_to_idx: Dict[str, int] = {}

    def set_raion_mapping(self, raion_keys: List[str]) -> None:
        """Set the raion key to index mapping."""
        self._raion_to_idx = {k: i for i, k in enumerate(raion_keys)}

    def forward(
        self,
        raion_features: Tensor,
        macro_features: Tensor,
        raion_keys: Optional[List[str]] = None,
        raion_mask: Optional[Tensor] = None,
        macro_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through RaionHAN.

        Args:
            raion_features: Per-raion features [batch, n_raions, seq_len, n_raion_features]
            macro_features: Macro features [batch, seq_len, n_macro_features]
            raion_keys: Optional list of raion keys for geographic prior
            raion_mask: Optional mask [batch, n_raions, seq_len]
            macro_mask: Optional mask [batch, seq_len]

        Returns:
            Dictionary with:
                - 'forecasts': [batch, n_raions, horizon, n_raion_features]
                - 'raion_reprs': [batch, n_raions, seq_len, d_model]
                - 'macro_repr': [batch, seq_len, d_model]
        """
        batch_size, n_raions, seq_len, _ = raion_features.shape

        # 1. Encode each raion (with shared encoder + raion embedding)
        raion_reprs = []
        for r in range(n_raions):
            raion_r = raion_features[:, r, :, :]  # [batch, seq_len, n_features]

            # Add raion embedding
            raion_idx = torch.full((batch_size,), r, dtype=torch.long, device=raion_features.device)
            raion_emb = self.raion_embedding(raion_idx).unsqueeze(1)  # [batch, 1, d_model]

            # Encode
            encoded = self.raion_encoder(raion_r)  # [batch, seq_len, d_model]
            encoded = encoded + raion_emb  # Add raion identity

            raion_reprs.append(encoded)

        # Stack: [batch, n_raions, seq_len, d_model]
        raion_reprs = torch.stack(raion_reprs, dim=1)

        # 2. Cross-raion attention
        cross_raion_reprs = self.cross_raion_attention(
            raion_reprs,
            raion_keys=raion_keys,
        )

        # 3. Encode macro features
        macro_repr = self.macro_encoder(macro_features, mask=macro_mask)

        # 4. Temporo-spatial fusion
        fused_reprs = self.fusion(
            cross_raion_reprs,
            macro_repr,
            macro_mask=macro_mask,
        )

        # 5. Generate forecasts for each raion
        forecasts = []
        for r in range(n_raions):
            raion_fused = fused_reprs[:, r, :, :]  # [batch, seq_len, d_model]

            # Aggregate and predict
            last_repr = raion_fused[:, -1, :]  # [batch, d_model]

            # Add raion embedding for prediction
            raion_idx = torch.full((batch_size,), r, dtype=torch.long, device=raion_features.device)
            raion_emb = self.raion_embedding(raion_idx)  # [batch, d_model]

            combined = last_repr + raion_emb

            # Shared forecast layers
            hidden = self.shared_forecast(combined)
            flat_pred = self.forecast_output(hidden)

            # Reshape to [batch, horizon, n_features]
            forecast = flat_pred.view(
                batch_size,
                self.config.forecast_horizon,
                self.config.n_raion_features,
            )
            forecasts.append(forecast)

        # Stack: [batch, n_raions, horizon, n_features]
        forecasts = torch.stack(forecasts, dim=1)

        return {
            'forecasts': forecasts,
            'raion_reprs': fused_reprs,
            'macro_repr': macro_repr,
        }

    def get_cross_raion_attention(
        self,
        raion_features: Tensor,
        raion_keys: Optional[List[str]] = None,
    ) -> Tensor:
        """
        Get cross-raion attention weights for interpretability.

        Args:
            raion_features: [batch, n_raions, seq_len, n_raion_features]
            raion_keys: Optional list of raion keys

        Returns:
            Attention weights [batch, n_heads, n_raions, n_raions]
        """
        batch_size, n_raions, seq_len, _ = raion_features.shape

        # Encode raions
        raion_reprs = []
        for r in range(n_raions):
            raion_r = raion_features[:, r, :, :]
            raion_idx = torch.full((batch_size,), r, dtype=torch.long, device=raion_features.device)
            raion_emb = self.raion_embedding(raion_idx).unsqueeze(1)
            encoded = self.raion_encoder(raion_r)
            encoded = encoded + raion_emb
            raion_reprs.append(encoded)

        raion_reprs = torch.stack(raion_reprs, dim=1)

        return self.cross_raion_attention.get_attention_weights(raion_reprs, raion_keys)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_raion_han(
    n_raion_features: int = 20,
    n_macro_features: int = 30,
    max_raions: int = 50,
    d_model: int = 128,
    forecast_horizon: int = 7,
    **kwargs,
) -> RaionHAN:
    """
    Factory function to create RaionHAN with common configurations.

    Args:
        n_raion_features: Number of features per raion
        n_macro_features: Number of macro (national) features
        max_raions: Maximum number of raions
        d_model: Model dimension
        forecast_horizon: Days to predict ahead
        **kwargs: Additional config options

    Returns:
        Configured RaionHAN model
    """
    config = RaionHANConfig(
        n_raion_features=n_raion_features,
        n_macro_features=n_macro_features,
        max_raions=max_raions,
        d_model=d_model,
        forecast_horizon=forecast_horizon,
        **kwargs,
    )
    return RaionHAN(config)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("RaionHAN Model Test")
    print("=" * 60)

    # Create model
    config = RaionHANConfig(
        n_raion_features=20,
        n_macro_features=30,
        max_raions=50,
        d_model=128,
        forecast_horizon=7,
    )

    model = RaionHAN(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Test forward pass
    batch_size = 2
    n_raions = 10
    seq_len = 30

    raion_features = torch.randn(batch_size, n_raions, seq_len, config.n_raion_features)
    macro_features = torch.randn(batch_size, seq_len, config.n_macro_features)

    print(f"\nInput shapes:")
    print(f"  raion_features: {raion_features.shape}")
    print(f"  macro_features: {macro_features.shape}")

    # Forward pass
    outputs = model(raion_features, macro_features)

    print(f"\nOutput shapes:")
    print(f"  forecasts: {outputs['forecasts'].shape}")
    print(f"  raion_reprs: {outputs['raion_reprs'].shape}")
    print(f"  macro_repr: {outputs['macro_repr'].shape}")

    # Test with raion keys
    test_raions = [f"Raion_{i}" for i in range(n_raions)]
    outputs_with_keys = model(raion_features, macro_features, raion_keys=test_raions)
    print(f"\nWith raion keys - forecasts: {outputs_with_keys['forecasts'].shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
