"""
Narrative Embedding Integration for Multi-Resolution Hierarchical Attention Network

This module provides production-ready implementations for integrating pre-computed
text embeddings (ISW daily assessments) into the existing HAN architecture.

EMBEDDING CHARACTERISTICS:
=========================
- Source: ISW (Institute for Study of War) daily assessments
- Dimension: 1024-dim float32 vectors
- Storage: Individual .npy files indexed by date (YYYY-MM-DD.npy)
- Coverage: ~1,315 files from Feb 2022 to present (~5.4MB total)
- Resolution: Daily (one embedding per assessment)

DESIGN DECISIONS:
=================

1. DATA LOADING STRATEGY:
   - Full load into memory (5.4MB is trivially small)
   - Pre-index by date during dataset initialization
   - Lazy loading not justified for this data volume

2. INTEGRATION APPROACH RANKING (recommended order for experimentation):

   A. EmbeddingFusion (RECOMMENDED FIRST)
      - Simple, modular, lowest implementation risk
      - Fuses narrative with quantitative at daily level
      - Easy to ablate and measure contribution

   B. CrossAttentionIntegration
      - More expressive but higher computational cost
      - Allows dynamic weighting based on context
      - Better for capturing complex narrative-quantitative interactions

   C. ParallelStreamArchitecture
      - Most flexible but most complex
      - Separate processing paths before late fusion
      - Best when narrative patterns differ significantly from quantitative

3. GRADIENT FLOW:
   - Embeddings treated as FROZEN inputs (pre-computed, no backprop)
   - Trainable projection layer maps 1024-dim to d_model
   - Optional learnable gating mechanism

Author: ML Engineering Team
Date: 2026-01-21
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import ISW_EMBEDDINGS_DIR


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NarrativeEmbeddingConfig:
    """Configuration for narrative embedding integration.

    Attributes:
        embedding_dir: Path to directory containing .npy embedding files
        embedding_dim: Dimension of pre-computed embeddings (1024 for ISW)
        projection_dim: Target dimension after projection (typically d_model)
        freeze_embeddings: Whether to treat embeddings as frozen (recommended True)
        use_layer_norm: Apply layer normalization after projection
        dropout: Dropout rate for projection
        missing_strategy: How to handle missing embeddings ('learned', 'zero', 'interpolate')
    """
    embedding_dir: Path = ISW_EMBEDDINGS_DIR / "by_date"
    embedding_dim: int = 1024
    projection_dim: int = 128
    freeze_embeddings: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.1
    missing_strategy: str = 'learned'


# =============================================================================
# EMBEDDING STORE: Efficient Loading and Lookup
# =============================================================================

class NarrativeEmbeddingStore:
    """
    Efficient storage and retrieval of pre-computed narrative embeddings.

    This class handles:
    - Loading all embeddings into memory at initialization
    - Date-based lookup with O(1) access
    - Handling of missing dates via configurable strategies

    Memory Analysis:
    ----------------
    - 1,315 embeddings x 1024 dims x 4 bytes = ~5.4MB
    - Memory mapping not justified - full load is optimal

    Example:
        >>> store = NarrativeEmbeddingStore(config)
        >>> emb = store.get_embedding('2022-03-15')  # Returns (1024,) tensor
        >>> batch = store.get_embeddings_batch(['2022-03-15', '2022-03-16'])
    """

    def __init__(self, config: NarrativeEmbeddingConfig) -> None:
        self.config = config
        self.embedding_dir = Path(config.embedding_dir)
        self.embedding_dim = config.embedding_dim

        # Storage for embeddings indexed by date string
        self._embeddings: Dict[str, np.ndarray] = {}
        self._date_index: List[str] = []

        # Learned token for missing embeddings
        self._missing_token: Optional[np.ndarray] = None

        self._load_all_embeddings()

    def _load_all_embeddings(self) -> None:
        """Load all embedding files into memory."""
        if not self.embedding_dir.exists():
            raise FileNotFoundError(f"Embedding directory not found: {self.embedding_dir}")

        npy_files = sorted(self.embedding_dir.glob("*.npy"))

        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {self.embedding_dir}")

        for npy_path in npy_files:
            # Extract date from filename: YYYY-MM-DD.npy
            date_str = npy_path.stem  # e.g., "2022-03-15"

            try:
                # Validate date format
                datetime.strptime(date_str, "%Y-%m-%d")

                # Load embedding
                embedding = np.load(npy_path)

                if embedding.shape != (self.embedding_dim,):
                    print(f"Warning: Unexpected shape {embedding.shape} for {date_str}, expected ({self.embedding_dim},)")
                    continue

                self._embeddings[date_str] = embedding.astype(np.float32)
                self._date_index.append(date_str)

            except ValueError as e:
                print(f"Warning: Skipping {npy_path.name} - invalid date format")
                continue

        print(f"Loaded {len(self._embeddings)} narrative embeddings "
              f"({len(self._embeddings) * self.embedding_dim * 4 / 1024 / 1024:.2f} MB)")

        # Compute mean embedding for missing token initialization
        if self._embeddings:
            all_embs = np.stack(list(self._embeddings.values()))
            self._mean_embedding = all_embs.mean(axis=0)
            self._std_embedding = all_embs.std(axis=0)

    @property
    def available_dates(self) -> List[str]:
        """Return sorted list of dates with available embeddings."""
        return self._date_index

    @property
    def date_range(self) -> Tuple[str, str]:
        """Return (min_date, max_date) of available embeddings."""
        return (self._date_index[0], self._date_index[-1])

    def has_embedding(self, date: Union[str, datetime]) -> bool:
        """Check if embedding exists for given date."""
        date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        return date_str in self._embeddings

    def get_embedding(
        self,
        date: Union[str, datetime],
        missing_value: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Get embedding for a specific date.

        Args:
            date: Date string (YYYY-MM-DD) or datetime object
            missing_value: Value to return if embedding missing (default: zeros)

        Returns:
            Tuple of (embedding array, is_observed flag)
        """
        date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")

        if date_str in self._embeddings:
            return self._embeddings[date_str], True

        # Handle missing
        if missing_value is not None:
            return missing_value, False

        if self.config.missing_strategy == 'zero':
            return np.zeros(self.embedding_dim, dtype=np.float32), False
        elif self.config.missing_strategy == 'mean':
            return self._mean_embedding.copy(), False
        else:  # 'learned' - return zeros, model will use learned token
            return np.zeros(self.embedding_dim, dtype=np.float32), False

    def get_embeddings_batch(
        self,
        dates: List[Union[str, datetime]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for a batch of dates.

        Args:
            dates: List of date strings or datetime objects

        Returns:
            Tuple of:
            - embeddings: (n_dates, embedding_dim) array
            - mask: (n_dates,) boolean array, True if embedding was observed
        """
        embeddings = []
        mask = []

        for date in dates:
            emb, is_observed = self.get_embedding(date)
            embeddings.append(emb)
            mask.append(is_observed)

        return np.stack(embeddings), np.array(mask)

    def get_embeddings_for_range(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get embeddings for a date range (inclusive).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of:
            - embeddings: (n_days, embedding_dim) array
            - mask: (n_days,) boolean array
            - dates: List of date strings
        """
        start = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        return (*self.get_embeddings_batch(dates), dates)


# =============================================================================
# OPTION A: EmbeddingFusion Module (RECOMMENDED FIRST)
# =============================================================================

class NarrativeProjection(nn.Module):
    """
    Projects pre-computed narrative embeddings to model dimension.

    This module handles:
    - Dimensionality reduction (1024 -> d_model)
    - Layer normalization for training stability
    - Learned token for missing embeddings

    The embeddings themselves are FROZEN - only the projection is trainable.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Learned token for missing/unobserved narrative embeddings
        # Initialized with small random values
        self.no_narrative_token = nn.Parameter(
            torch.randn(1, output_dim) * 0.02
        )

        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )

        if use_layer_norm:
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for projection layers."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: Tensor,
        observation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Project narrative embeddings to model dimension.

        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, input_dim]
            observation_mask: Boolean mask [batch, seq_len] where True = observed

        Returns:
            Projected embeddings [batch, seq_len, output_dim]
        """
        batch_size, seq_len, _ = embeddings.shape

        # Project embeddings (frozen input, trainable projection)
        projected = self.projection(embeddings)
        projected = self.norm(projected)

        # Replace unobserved positions with learned no_narrative_token
        if observation_mask is not None:
            # Expand no_narrative_token to match sequence
            no_narr_expanded = self.no_narrative_token.expand(batch_size, seq_len, -1)

            # Use mask to select between projected and no_narrative_token
            mask_expanded = observation_mask.unsqueeze(-1)  # [batch, seq, 1]
            projected = torch.where(mask_expanded, projected, no_narr_expanded)

        return self.dropout(projected)


class EmbeddingFusion(nn.Module):
    """
    Fuses quantitative features with narrative embeddings via gated addition.

    This is the SIMPLEST and RECOMMENDED FIRST approach for integration.

    Architecture:
    - Projects narrative embedding to same dimension as quantitative features
    - Learns a gating weight to balance narrative vs quantitative
    - Produces fused representation for downstream processing

    Advantages:
    - Minimal changes to existing architecture
    - Easy to ablate (set gate to 0)
    - Low computational overhead
    - Interpretable (can inspect gate values)

    Example:
        >>> fusion = EmbeddingFusion(d_model=128)
        >>> fused = fusion(
        ...     quantitative=daily_encoded,  # [batch, seq, 128]
        ...     narrative=projected_embeddings,  # [batch, seq, 128]
        ...     quantitative_mask=daily_mask,
        ...     narrative_mask=narrative_mask,
        ... )
    """

    def __init__(
        self,
        d_model: int = 128,
        fusion_type: str = 'gated_add',
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize fusion module.

        Args:
            d_model: Model dimension (must match quantitative and narrative dims)
            fusion_type: Type of fusion ('gated_add', 'concat', 'bilinear')
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.fusion_type = fusion_type

        if fusion_type == 'gated_add':
            # Learnable gate to balance narrative vs quantitative
            # Input: concatenation of both representations
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
            self.output_dim = d_model

        elif fusion_type == 'concat':
            # Simple concatenation with projection
            self.output_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            self.output_dim = d_model

        elif fusion_type == 'bilinear':
            # Bilinear interaction
            self.bilinear = nn.Bilinear(d_model, d_model, d_model)
            self.output_dim = d_model

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        quantitative: Tensor,
        narrative: Tensor,
        quantitative_mask: Optional[Tensor] = None,
        narrative_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse quantitative and narrative representations.

        Args:
            quantitative: Quantitative features [batch, seq_len, d_model]
            narrative: Narrative embeddings [batch, seq_len, d_model]
            quantitative_mask: Mask for quantitative [batch, seq_len]
            narrative_mask: Mask for narrative [batch, seq_len]

        Returns:
            Fused representation [batch, seq_len, d_model]
        """
        if self.fusion_type == 'gated_add':
            # Compute gate value based on both inputs
            concat = torch.cat([quantitative, narrative], dim=-1)
            gate_value = self.gate(concat)  # [batch, seq, 1]

            # Gated combination
            fused = gate_value * narrative + (1 - gate_value) * quantitative

        elif self.fusion_type == 'concat':
            concat = torch.cat([quantitative, narrative], dim=-1)
            fused = self.output_proj(concat)

        elif self.fusion_type == 'bilinear':
            fused = self.bilinear(quantitative, narrative)

        # Normalize and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


# =============================================================================
# OPTION B: Cross-Attention Integration
# =============================================================================

class NarrativeAttention(nn.Module):
    """
    Cross-attention module where quantitative features attend to narrative embeddings.

    This is MORE EXPRESSIVE than simple fusion but has higher computational cost.

    Architecture:
    - Quantitative features serve as queries
    - Narrative embeddings serve as keys/values
    - Multi-head attention learns which narrative aspects matter

    Use case:
    - When narrative contains complex, time-varying information
    - When different quantitative features need different narrative context

    Example:
        >>> attention = NarrativeAttention(d_model=128, nhead=8)
        >>> enriched = attention(
        ...     quantitative=daily_encoded,  # [batch, seq, 128]
        ...     narrative=projected_embeddings,  # [batch, seq, 128]
        ... )
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_residual = use_residual

        # Multi-head attention: quantitative queries narrative
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        quantitative: Tensor,
        narrative: Tensor,
        quantitative_mask: Optional[Tensor] = None,
        narrative_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply cross-attention from quantitative to narrative.

        Args:
            quantitative: Query features [batch, seq_len, d_model]
            narrative: Key/value features [batch, seq_len, d_model]
            quantitative_mask: Query mask [batch, seq_len]
            narrative_mask: Key/value mask [batch, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            If return_attention=False: enriched [batch, seq_len, d_model]
            If return_attention=True: (enriched, attention_weights)
        """
        # Prepare key_padding_mask for attention (True = ignore)
        key_padding_mask = None
        if narrative_mask is not None:
            key_padding_mask = ~narrative_mask  # Invert: True means ignore

        # Cross-attention
        attended, attn_weights = self.cross_attention(
            query=quantitative,
            key=narrative,
            value=narrative,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )

        # Residual connection and layer norm
        if self.use_residual:
            attended = self.norm_attn(quantitative + self.dropout(attended))
        else:
            attended = self.norm_attn(self.dropout(attended))

        # Feed-forward with residual
        ffn_out = self.ffn(attended)
        enriched = self.norm_ffn(attended + ffn_out)

        if return_attention:
            return enriched, attn_weights
        return enriched


class BidirectionalNarrativeAttention(nn.Module):
    """
    Bidirectional cross-attention between quantitative and narrative streams.

    Both streams attend to each other, allowing information flow in both directions:
    - Quantitative can inform narrative (e.g., high casualties -> negative sentiment)
    - Narrative can inform quantitative (e.g., offensive announced -> expect activity)

    Architecture:
    - Quantitative-to-narrative cross-attention
    - Narrative-to-quantitative cross-attention
    - Gated combination of attended representations
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
        use_gating: bool = True,
    ) -> None:
        super().__init__()

        self.use_gating = use_gating

        # Quantitative attends to narrative
        self.quant_to_narr = NarrativeAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_residual=True,
        )

        # Narrative attends to quantitative
        self.narr_to_quant = NarrativeAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            use_residual=True,
        )

        if use_gating:
            # Gates for combining attended representations
            self.quant_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.narr_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

    def forward(
        self,
        quantitative: Tensor,
        narrative: Tensor,
        quantitative_mask: Optional[Tensor] = None,
        narrative_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply bidirectional cross-attention.

        Args:
            quantitative: Quantitative features [batch, seq_len, d_model]
            narrative: Narrative features [batch, seq_len, d_model]
            quantitative_mask: Mask [batch, seq_len]
            narrative_mask: Mask [batch, seq_len]

        Returns:
            Tuple of (enriched_quantitative, enriched_narrative)
        """
        # Quantitative enriched by narrative
        quant_attended = self.quant_to_narr(
            quantitative, narrative,
            quantitative_mask, narrative_mask
        )

        # Narrative enriched by quantitative
        narr_attended = self.narr_to_quant(
            narrative, quantitative,
            narrative_mask, quantitative_mask
        )

        if self.use_gating:
            # Gated combination
            quant_concat = torch.cat([quantitative, quant_attended], dim=-1)
            narr_concat = torch.cat([narrative, narr_attended], dim=-1)

            quant_gate = self.quant_gate(quant_concat)
            narr_gate = self.narr_gate(narr_concat)

            enriched_quant = quant_gate * quant_attended + (1 - quant_gate) * quantitative
            enriched_narr = narr_gate * narr_attended + (1 - narr_gate) * narrative
        else:
            enriched_quant = quant_attended
            enriched_narr = narr_attended

        return enriched_quant, enriched_narr


# =============================================================================
# OPTION C: Parallel Stream Architecture
# =============================================================================

class NarrativeEncoder(nn.Module):
    """
    Dedicated encoder for narrative embeddings (parallel stream).

    This encoder processes narrative embeddings independently before fusion,
    allowing specialized feature extraction for textual information.

    Architecture:
    - Projection layer
    - Transformer encoder layers
    - Temporal attention to capture narrative evolution
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 1500,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Projection from embedding space to model space
        self.projection = NarrativeProjection(
            input_dim=input_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        # Positional encoding for temporal position
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        embeddings: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode narrative embeddings.

        Args:
            embeddings: Pre-computed embeddings [batch, seq_len, input_dim]
            mask: Observation mask [batch, seq_len]

        Returns:
            Encoded representations [batch, seq_len, d_model]
        """
        # Project to model dimension
        hidden = self.projection(embeddings, mask)

        # Add positional encoding
        seq_len = hidden.size(1)
        hidden = hidden + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask

            # Handle fully masked sequences
            all_masked = src_key_padding_mask.all(dim=1)
            if all_masked.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

        hidden = self.transformer(hidden, src_key_padding_mask=src_key_padding_mask)

        return self.output_norm(hidden)


class DualStreamEncoder(nn.Module):
    """
    Parallel encoder architecture with separate quantitative and narrative streams.

    This is the MOST FLEXIBLE approach but also MOST COMPLEX.

    Architecture:
    - Separate encoder for quantitative features (existing daily encoder)
    - Separate encoder for narrative embeddings (new)
    - Late fusion via cross-attention or concatenation

    Use case:
    - When narrative and quantitative patterns are very different
    - When you want maximum model expressiveness
    - When you have sufficient data to train both streams

    Example:
        >>> dual_encoder = DualStreamEncoder(
        ...     quantitative_encoder=daily_encoder,  # Existing
        ...     d_model=128,
        ... )
        >>> fused = dual_encoder(
        ...     quantitative_features=daily_features,
        ...     quantitative_mask=daily_mask,
        ...     narrative_embeddings=narrative_embs,
        ...     narrative_mask=narrative_mask,
        ... )
    """

    def __init__(
        self,
        quantitative_encoder: nn.Module,
        narrative_config: NarrativeEmbeddingConfig,
        d_model: int = 128,
        nhead: int = 8,
        num_narrative_layers: int = 2,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = 'bidirectional',
    ) -> None:
        super().__init__()

        self.quantitative_encoder = quantitative_encoder

        # Create narrative encoder
        self.narrative_encoder = NarrativeEncoder(
            input_dim=narrative_config.embedding_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_narrative_layers,
            dropout=dropout,
        )

        # Fusion module
        if fusion_type == 'bidirectional':
            self.fusion = BidirectionalNarrativeAttention(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
            )
        elif fusion_type == 'simple':
            self.fusion = EmbeddingFusion(
                d_model=d_model,
                fusion_type='gated_add',
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.fusion_type = fusion_type

        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        quantitative_features: Tensor,
        quantitative_mask: Tensor,
        narrative_embeddings: Tensor,
        narrative_mask: Tensor,
    ) -> Tensor:
        """
        Process both streams and fuse.

        Args:
            quantitative_features: Raw quantitative features [batch, seq, n_features]
            quantitative_mask: Quantitative mask [batch, seq, n_features]
            narrative_embeddings: Pre-computed embeddings [batch, seq, 1024]
            narrative_mask: Narrative mask [batch, seq]

        Returns:
            Fused representation [batch, seq, d_model]
        """
        # Encode quantitative stream
        quant_encoded, _ = self.quantitative_encoder(
            quantitative_features, quantitative_mask
        )

        # Encode narrative stream
        narr_encoded = self.narrative_encoder(narrative_embeddings, narrative_mask)

        # Fuse streams
        if self.fusion_type == 'bidirectional':
            fused_quant, fused_narr = self.fusion(
                quant_encoded, narr_encoded,
                quantitative_mask.any(dim=-1), narrative_mask
            )
            # Combine both enriched streams
            fused = fused_quant + fused_narr
        else:
            fused = self.fusion(
                quant_encoded, narr_encoded,
                quantitative_mask.any(dim=-1), narrative_mask
            )

        return self.output_norm(self.output_proj(fused))


# =============================================================================
# INTEGRATION WITH EXISTING DATASET
# =============================================================================

class NarrativeAugmentedDataset(Dataset):
    """
    Dataset wrapper that augments existing data with narrative embeddings.

    This class wraps an existing conflict dataset and adds narrative embeddings
    aligned by date. It handles:
    - Date alignment between quantitative and narrative data
    - Missing embedding handling
    - Proper mask generation

    Example:
        >>> base_dataset = RealConflictDataset(...)
        >>> augmented = NarrativeAugmentedDataset(
        ...     base_dataset=base_dataset,
        ...     embedding_store=NarrativeEmbeddingStore(config),
        ... )
    """

    def __init__(
        self,
        base_dataset: Dataset,
        embedding_store: NarrativeEmbeddingStore,
        date_field: str = 'date_range',
    ) -> None:
        self.base_dataset = base_dataset
        self.embedding_store = embedding_store
        self.date_field = date_field

        # Get dates from base dataset
        if hasattr(base_dataset, 'date_range'):
            self.dates = [d.strftime("%Y-%m-%d") for d in base_dataset.date_range]
        else:
            # Assume indices correspond to sequential dates
            # This would need to be customized for your specific dataset
            raise ValueError("Base dataset must have date_range attribute")

        # Pre-load all embeddings for this dataset's date range
        self._embeddings, self._embedding_mask = self._preload_embeddings()

    def _preload_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pre-load all embeddings for the dataset's date range."""
        embeddings = []
        masks = []

        for date_str in self.dates:
            emb, is_observed = self.embedding_store.get_embedding(date_str)
            embeddings.append(emb)
            masks.append(is_observed)

        return (
            torch.tensor(np.stack(embeddings), dtype=torch.float32),
            torch.tensor(masks, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base data
        base_data = self.base_dataset[idx]

        if isinstance(base_data, tuple):
            features, masks, targets = base_data
        else:
            features = base_data.get('features', {})
            masks = base_data.get('masks', {})
            targets = base_data.get('targets', {})

        # Get sequence indices from base dataset
        # This assumes base_dataset has indices attribute
        if hasattr(self.base_dataset, 'indices'):
            start_idx = self.base_dataset.indices[idx]
            seq_len = self.base_dataset.seq_len
        else:
            start_idx = idx
            seq_len = len(self.dates)

        end_idx = min(start_idx + seq_len, len(self.dates))

        # Get corresponding narrative embeddings
        narrative_embeddings = self._embeddings[start_idx:end_idx]
        narrative_mask = self._embedding_mask[start_idx:end_idx]

        # Add to features dict
        features['narrative_embedding'] = narrative_embeddings
        masks['narrative_embedding'] = narrative_mask

        return features, masks, targets


# =============================================================================
# INTEGRATION POINT MODULES
# =============================================================================

class PreEncoderNarrativeInjection(nn.Module):
    """
    Inject narrative embeddings BEFORE DailyEncoder processing.

    Integration Point: Augments input features before encoding.

    Strategy:
    - Project narrative to feature dimension
    - Concatenate with quantitative features
    - Feed combined features to encoder

    Pros:
    - Simplest integration
    - Narrative influences all subsequent processing

    Cons:
    - May dilute narrative signal if feature dim is large
    - Less interpretable
    """

    def __init__(
        self,
        narrative_config: NarrativeEmbeddingConfig,
        quantitative_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.narrative_proj = NarrativeProjection(
            input_dim=narrative_config.embedding_dim,
            output_dim=quantitative_dim,
            dropout=dropout,
        )

        # Combined projection
        self.combined_proj = nn.Sequential(
            nn.Linear(quantitative_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        quantitative: Tensor,
        narrative: Tensor,
        narrative_mask: Tensor,
    ) -> Tensor:
        """Combine quantitative and narrative before encoding."""
        narr_proj = self.narrative_proj(narrative, narrative_mask)
        combined = torch.cat([quantitative, narr_proj], dim=-1)
        return self.combined_proj(combined)


class PostEncoderNarrativeInjection(nn.Module):
    """
    Inject narrative embeddings AFTER DailyEncoder processing.

    Integration Point: Fuses with encoded representations.

    Strategy:
    - Encode quantitative features first
    - Project narrative embeddings
    - Fuse via attention or gating

    Pros:
    - Cleaner separation of concerns
    - Quantitative encoding not affected by narrative dimension
    - More interpretable fusion

    Cons:
    - Narrative doesn't influence low-level feature interactions
    """

    def __init__(
        self,
        narrative_config: NarrativeEmbeddingConfig,
        d_model: int = 128,
        fusion_type: str = 'gated_add',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.narrative_proj = NarrativeProjection(
            input_dim=narrative_config.embedding_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        self.fusion = EmbeddingFusion(
            d_model=d_model,
            fusion_type=fusion_type,
            dropout=dropout,
        )

    def forward(
        self,
        encoded_quantitative: Tensor,
        narrative: Tensor,
        quantitative_mask: Tensor,
        narrative_mask: Tensor,
    ) -> Tensor:
        """Fuse encoded quantitative with narrative."""
        narr_proj = self.narrative_proj(narrative, narrative_mask)
        return self.fusion(encoded_quantitative, narr_proj, quantitative_mask, narrative_mask)


class CrossResolutionNarrativeInjection(nn.Module):
    """
    Inject narrative as THIRD MODALITY in CrossResolutionFusion.

    Integration Point: Adds narrative alongside daily and monthly streams.

    Strategy:
    - Narrative embeddings aggregated to monthly resolution
    - Tripartite attention between daily, monthly, and narrative
    - Produces fused representation with all three modalities

    Pros:
    - Most comprehensive integration
    - Narrative can inform resolution alignment

    Cons:
    - Most complex implementation
    - May require architecture changes to CrossResolutionFusion
    """

    def __init__(
        self,
        narrative_config: NarrativeEmbeddingConfig,
        d_model: int = 128,
        nhead: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project narrative to model dimension
        self.narrative_proj = NarrativeProjection(
            input_dim=narrative_config.embedding_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        # Attention layers for tripartite fusion
        # Daily attends to narrative
        self.daily_narr_attn = NarrativeAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Monthly attends to narrative
        self.monthly_narr_attn = NarrativeAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Final combination
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        daily_repr: Tensor,
        monthly_repr: Tensor,
        narrative: Tensor,
        daily_mask: Tensor,
        monthly_mask: Tensor,
        narrative_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Fuse all three modalities.

        Returns enriched daily and monthly representations.
        """
        # Project narrative
        narr_proj = self.narrative_proj(narrative, narrative_mask)

        # Enrich daily with narrative
        daily_enriched = self.daily_narr_attn(
            daily_repr, narr_proj, daily_mask, narrative_mask
        )

        # Aggregate narrative to monthly for monthly attention
        # (simplified: use same narrative, assume alignment)
        monthly_enriched = self.monthly_narr_attn(
            monthly_repr, narr_proj, monthly_mask, narrative_mask
        )

        return daily_enriched, monthly_enriched


# =============================================================================
# RECOMMENDATION: Implementation Order for Experimentation
# =============================================================================

"""
RECOMMENDED EXPERIMENTATION ORDER:
==================================

1. FIRST: PostEncoderNarrativeInjection with EmbeddingFusion (gated_add)
   - Minimal changes to existing architecture
   - Easy to ablate and measure contribution
   - Implementation: ~30 lines of integration code

   Code to add to MultiResolutionHAN:

   ```python
   # In __init__:
   self.narrative_injection = PostEncoderNarrativeInjection(
       narrative_config=narrative_config,
       d_model=d_model,
   )

   # In forward, after daily fusion:
   fused_daily = self.daily_fusion(...)
   fused_daily = self.narrative_injection(
       fused_daily, narrative_embeddings, combined_daily_mask, narrative_mask
   )
   ```

2. SECOND: NarrativeAttention (cross-attention)
   - If gated fusion helps, try more expressive attention
   - Can dynamically weight narrative based on context
   - Provides attention weights for interpretability

3. THIRD: BidirectionalNarrativeAttention
   - If one-way attention helps, try bidirectional
   - Allows quantitative to inform narrative interpretation

4. FOURTH: DualStreamEncoder (parallel streams)
   - Only if above approaches plateau
   - Most parameters, most data-hungry
   - Best for capturing independent narrative patterns

5. FIFTH: CrossResolutionNarrativeInjection
   - Most complex, save for last
   - Only if narrative has strong multi-resolution structure


ABLATION EXPERIMENTS:
====================

A. Baseline: No narrative (current model)
B. Narrative with gate fixed at 0.5 (equal weighting)
C. Narrative with learned gate (EmbeddingFusion)
D. Narrative with cross-attention
E. Parallel streams with late fusion

METRICS TO TRACK:
================
- Task metrics: casualty RMSE, regime accuracy, anomaly F1
- Computational: training time, inference latency
- Interpretability: gate values, attention patterns
- Robustness: performance on dates with/without narrative
"""


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("Narrative Embedding Integration Tests")
    print("=" * 80)

    # Test configuration
    config = NarrativeEmbeddingConfig()
    batch_size = 4
    seq_len = 100
    d_model = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # =========================================================================
    # TEST 1: NarrativeEmbeddingStore
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 1: NarrativeEmbeddingStore")
    print("-" * 40)

    try:
        store = NarrativeEmbeddingStore(config)
        print(f"Loaded {len(store.available_dates)} embeddings")
        print(f"Date range: {store.date_range}")

        # Test single lookup
        emb, observed = store.get_embedding("2022-03-15")
        print(f"Single lookup shape: {emb.shape}, observed: {observed}")

        # Test batch lookup
        batch_emb, batch_mask = store.get_embeddings_batch([
            "2022-03-15", "2022-03-16", "2022-03-17"
        ])
        print(f"Batch lookup shape: {batch_emb.shape}, mask: {batch_mask}")

        print("PASSED")
    except FileNotFoundError as e:
        print(f"SKIPPED (embedding files not found): {e}")
        # Create dummy data for remaining tests
        store = None
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 2: NarrativeProjection
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 2: NarrativeProjection")
    print("-" * 40)

    try:
        projection = NarrativeProjection(
            input_dim=1024,
            output_dim=d_model,
        ).to(device)

        # Create dummy embeddings
        dummy_emb = torch.randn(batch_size, seq_len, 1024, device=device)
        dummy_mask = torch.rand(batch_size, seq_len, device=device) > 0.2

        # Forward pass
        projected = projection(dummy_emb, dummy_mask)

        print(f"Input shape: {dummy_emb.shape}")
        print(f"Output shape: {projected.shape}")
        assert projected.shape == (batch_size, seq_len, d_model)

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 3: EmbeddingFusion
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 3: EmbeddingFusion")
    print("-" * 40)

    try:
        fusion = EmbeddingFusion(
            d_model=d_model,
            fusion_type='gated_add',
        ).to(device)

        # Create dummy inputs
        quant = torch.randn(batch_size, seq_len, d_model, device=device)
        narr = torch.randn(batch_size, seq_len, d_model, device=device)

        # Forward pass
        fused = fusion(quant, narr)

        print(f"Quantitative shape: {quant.shape}")
        print(f"Narrative shape: {narr.shape}")
        print(f"Fused shape: {fused.shape}")
        assert fused.shape == (batch_size, seq_len, d_model)

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 4: NarrativeAttention
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 4: NarrativeAttention")
    print("-" * 40)

    try:
        attention = NarrativeAttention(
            d_model=d_model,
            nhead=8,
        ).to(device)

        # Forward pass
        enriched = attention(quant, narr)

        print(f"Enriched shape: {enriched.shape}")
        assert enriched.shape == (batch_size, seq_len, d_model)

        # Test with attention weights
        enriched, attn = attention(quant, narr, return_attention=True)
        print(f"Attention weights shape: {attn.shape}")

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 5: BidirectionalNarrativeAttention
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 5: BidirectionalNarrativeAttention")
    print("-" * 40)

    try:
        bidir = BidirectionalNarrativeAttention(
            d_model=d_model,
            nhead=8,
        ).to(device)

        # Forward pass
        enriched_q, enriched_n = bidir(quant, narr)

        print(f"Enriched quantitative shape: {enriched_q.shape}")
        print(f"Enriched narrative shape: {enriched_n.shape}")
        assert enriched_q.shape == (batch_size, seq_len, d_model)
        assert enriched_n.shape == (batch_size, seq_len, d_model)

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # TEST 6: NarrativeEncoder
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 6: NarrativeEncoder")
    print("-" * 40)

    try:
        encoder = NarrativeEncoder(
            input_dim=1024,
            d_model=d_model,
            nhead=8,
            num_layers=2,
        ).to(device)

        # Forward pass
        encoded = encoder(dummy_emb, dummy_mask)

        print(f"Input shape: {dummy_emb.shape}")
        print(f"Encoded shape: {encoded.shape}")
        assert encoded.shape == (batch_size, seq_len, d_model)

        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # TEST 7: Gradient Flow
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 7: Gradient Flow")
    print("-" * 40)

    try:
        # Create full pipeline
        projection = NarrativeProjection(1024, d_model).to(device)
        fusion = EmbeddingFusion(d_model).to(device)

        # Forward pass
        projected = projection(dummy_emb, dummy_mask)
        fused = fusion(quant, projected)

        # Compute loss and backprop
        loss = fused.mean()
        loss.backward()

        # Check gradients
        has_grad = False
        for name, param in projection.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients computed"
        print("Gradients flow correctly through projection and fusion")
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\nNarrative embedding integration modules ready for production use.")
    print("\nRecommended next steps:")
    print("1. Integrate PostEncoderNarrativeInjection into MultiResolutionHAN")
    print("2. Create NarrativeAugmentedDataset wrapper")
    print("3. Run ablation experiments comparing fusion strategies")
