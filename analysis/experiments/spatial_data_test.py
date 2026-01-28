"""
Spatial Data Integration Test

This script tests the full spatial data loading pipeline and runs 5 training steps
to verify the spatial loaders (DeepState, FIRMS) work correctly with the training system.

Usage:
    python -m analysis.experiments.spatial_data_test
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import PROJECT_ROOT, OUTPUT_DIR
from analysis.loaders import (
    load_deepstate_spatial,
    load_firms_tiled,
    load_firms_aggregated,
    DeepStateSpatialLoader,
    FIRMSSpatialLoader,
    UKRAINE_REGIONS,
)
from analysis.modular_data_config import (
    ModularDataConfig,
    get_data_source_config,
    SpatialMode,
)


# =============================================================================
# SIMPLE SPATIAL MODEL FOR TESTING
# =============================================================================

class SpatialTestModel(nn.Module):
    """
    Simple model for testing spatial data flow.

    Architecture:
    - Separate encoders for each spatial source
    - Temporal attention over sequence
    - Fusion layer for combining sources
    - Prediction head
    """

    def __init__(
        self,
        deepstate_features: int,
        firms_features: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        seq_len: int = 30,
    ):
        super().__init__()

        self.seq_len = seq_len

        # DeepState encoder
        self.deepstate_encoder = nn.Sequential(
            nn.Linear(deepstate_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # FIRMS encoder
        self.firms_encoder = nn.Sequential(
            nn.Linear(firms_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Concatenated sources
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        deepstate_features: torch.Tensor,  # [batch, seq, features]
        firms_features: torch.Tensor,       # [batch, seq, features]
        deepstate_mask: Optional[torch.Tensor] = None,
        firms_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode sources
        ds_encoded = self.deepstate_encoder(deepstate_features)  # [batch, seq, hidden]
        firms_encoded = self.firms_encoder(firms_features)        # [batch, seq, hidden]

        # Concatenate sources
        combined = torch.cat([ds_encoded, firms_encoded], dim=-1)  # [batch, seq, hidden*2]

        # Apply temporal attention
        attended, _ = self.temporal_attention(combined, combined, combined)

        # Pool over sequence (mean pooling)
        pooled = attended.mean(dim=1)  # [batch, hidden*2]

        # Predict
        output = self.prediction_head(pooled)  # [batch, 1]

        return output


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_full_spatial_data(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load full spatial data from all sources.

    Returns:
        Tuple of (deepstate_df, firms_df, masks_dict)
    """
    print("\n" + "=" * 60)
    print("Loading Full Spatial Data")
    print("=" * 60)

    # Load DeepState spatial features
    print("\n1. Loading DeepState Spatial Features...")
    ds_df, ds_mask = load_deepstate_spatial(
        start_date=start_date,
        end_date=end_date,
        spatial_mode='tiled',
    )
    print(f"   Shape: {ds_df.shape}")
    print(f"   Date range: {ds_df['date'].min()} to {ds_df['date'].max()}")
    print(f"   Observation coverage: {ds_mask.mean()*100:.1f}%")

    # List DeepState features
    ds_features = [c for c in ds_df.columns if c != 'date']
    print(f"   Features ({len(ds_features)}):")
    for i, feat in enumerate(ds_features[:8]):
        print(f"      {feat}")
    if len(ds_features) > 8:
        print(f"      ... and {len(ds_features) - 8} more")

    # Load FIRMS spatial features
    print("\n2. Loading FIRMS Spatial Features...")
    firms_df, firms_mask = load_firms_tiled(
        start_date=start_date,
        end_date=end_date,
    )
    print(f"   Shape: {firms_df.shape}")
    print(f"   Date range: {firms_df['date'].min()} to {firms_df['date'].max()}")
    print(f"   Observation coverage: {firms_mask.mean()*100:.1f}%")

    # List FIRMS features
    firms_features = [c for c in firms_df.columns if c != 'date']
    print(f"   Features ({len(firms_features)}):")
    for i, feat in enumerate(firms_features[:8]):
        print(f"      {feat}")
    if len(firms_features) > 8:
        print(f"      ... and {len(firms_features) - 8} more")

    return ds_df, firms_df, {'deepstate': ds_mask, 'firms': firms_mask}


def align_dataframes(
    ds_df: pd.DataFrame,
    firms_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align dataframes on common dates."""
    ds_df = ds_df.copy()
    firms_df = firms_df.copy()

    ds_df['date'] = pd.to_datetime(ds_df['date'])
    firms_df['date'] = pd.to_datetime(firms_df['date'])

    # Find common date range
    start_date = max(ds_df['date'].min(), firms_df['date'].min())
    end_date = min(ds_df['date'].max(), firms_df['date'].max())

    print(f"\n3. Aligning data to common range: {start_date.date()} to {end_date.date()}")

    ds_df = ds_df[(ds_df['date'] >= start_date) & (ds_df['date'] <= end_date)]
    firms_df = firms_df[(firms_df['date'] >= start_date) & (firms_df['date'] <= end_date)]

    # Merge on date
    ds_df = ds_df.set_index('date').sort_index()
    firms_df = firms_df.set_index('date').sort_index()

    # Fill missing dates
    all_dates = pd.date_range(start_date, end_date, freq='D')
    ds_df = ds_df.reindex(all_dates).ffill().fillna(0)
    firms_df = firms_df.reindex(all_dates).ffill().fillna(0)

    print(f"   DeepState aligned: {len(ds_df)} days")
    print(f"   FIRMS aligned: {len(firms_df)} days")

    return ds_df, firms_df


def create_sequences(
    ds_df: pd.DataFrame,
    firms_df: pd.DataFrame,
    seq_len: int = 30,
    stride: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training sequences from aligned dataframes.

    Returns:
        Tuple of (deepstate_sequences, firms_sequences, targets)
    """
    print(f"\n4. Creating sequences (seq_len={seq_len}, stride={stride})...")

    # Get feature arrays
    ds_values = ds_df.values.astype(np.float32)
    firms_values = firms_df.values.astype(np.float32)

    # Normalize
    ds_mean, ds_std = ds_values.mean(axis=0), ds_values.std(axis=0) + 1e-8
    firms_mean, firms_std = firms_values.mean(axis=0), firms_values.std(axis=0) + 1e-8

    ds_values = (ds_values - ds_mean) / ds_std
    firms_values = (firms_values - firms_mean) / firms_std

    # Create sequences
    ds_sequences = []
    firms_sequences = []
    targets = []

    n_samples = len(ds_df)
    for i in range(0, n_samples - seq_len - 1, stride):
        ds_seq = ds_values[i:i+seq_len]
        firms_seq = firms_values[i:i+seq_len]

        # Target: total fire count on next day (as proxy for conflict intensity)
        if 'fire_count_total' in firms_df.columns:
            target_col = list(firms_df.columns).index('fire_count_total')
            target = firms_values[i+seq_len, target_col]
        else:
            target = firms_values[i+seq_len].sum()

        ds_sequences.append(ds_seq)
        firms_sequences.append(firms_seq)
        targets.append(target)

    ds_tensor = torch.tensor(np.array(ds_sequences), dtype=torch.float32)
    firms_tensor = torch.tensor(np.array(firms_sequences), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(-1)

    print(f"   Created {len(ds_sequences)} sequences")
    print(f"   DeepState tensor shape: {ds_tensor.shape}")
    print(f"   FIRMS tensor shape: {firms_tensor.shape}")
    print(f"   Targets shape: {target_tensor.shape}")

    return ds_tensor, firms_tensor, target_tensor


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_training_steps(
    model: nn.Module,
    ds_data: torch.Tensor,
    firms_data: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
) -> List[float]:
    """
    Run training steps and return losses.
    """
    print(f"\n" + "=" * 60)
    print(f"Running {n_steps} Training Steps")
    print("=" * 60)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Move data to device
    ds_data = ds_data.to(device)
    firms_data = firms_data.to(device)
    targets = targets.to(device)

    n_samples = len(ds_data)
    losses = []

    print(f"\n   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training samples: {n_samples}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print()

    for step in range(n_steps):
        # Random batch sampling
        indices = torch.randperm(n_samples)[:batch_size]

        ds_batch = ds_data[indices]
        firms_batch = firms_data[indices]
        target_batch = targets[indices]

        # Forward pass
        optimizer.zero_grad()
        predictions = model(ds_batch, firms_batch)
        loss = criterion(predictions, target_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"   Step {step + 1}/{n_steps}: Loss = {loss.item():.6f}")

    return losses


# =============================================================================
# REGIONAL STATISTICS
# =============================================================================

def print_regional_statistics(ds_df: pd.DataFrame, firms_df: pd.DataFrame):
    """Print statistics by region."""
    print("\n" + "=" * 60)
    print("Regional Statistics")
    print("=" * 60)

    print("\nDeepState Unit Counts by Region:")
    for region in UKRAINE_REGIONS.keys():
        col = f'unit_count_{region}'
        if col in ds_df.columns:
            total = ds_df[col].sum()
            mean = ds_df[col].mean()
            max_val = ds_df[col].max()
            print(f"   {region:12s}: total={total:>8,.0f}, mean={mean:>6.1f}, max={max_val:>4.0f}")

    print("\nFIRMS Fire Counts by Region:")
    for region in UKRAINE_REGIONS.keys():
        col = f'fire_count_{region}'
        if col in firms_df.columns:
            total = firms_df[col].sum()
            mean = firms_df[col].mean()
            max_val = firms_df[col].max()
            print(f"   {region:12s}: total={total:>8,.0f}, mean={mean:>6.1f}, max={max_val:>4.0f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("SPATIAL DATA INTEGRATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    # Load full spatial data (full extent)
    ds_df, firms_df, masks = load_full_spatial_data()

    # Align dataframes
    ds_aligned, firms_aligned = align_dataframes(ds_df, firms_df)

    # Print regional statistics
    print_regional_statistics(ds_aligned.reset_index(), firms_aligned.reset_index())

    # Create sequences
    seq_len = 30
    ds_sequences, firms_sequences, targets = create_sequences(
        ds_aligned, firms_aligned,
        seq_len=seq_len,
        stride=7,
    )

    # Get feature dimensions
    deepstate_features = ds_sequences.shape[-1]
    firms_features = firms_sequences.shape[-1]

    print(f"\n5. Creating model...")
    print(f"   DeepState features: {deepstate_features}")
    print(f"   FIRMS features: {firms_features}")

    # Create model
    model = SpatialTestModel(
        deepstate_features=deepstate_features,
        firms_features=firms_features,
        hidden_dim=64,
        n_heads=4,
        seq_len=seq_len,
    )

    # Run 5 training steps
    losses = run_training_steps(
        model=model,
        ds_data=ds_sequences,
        firms_data=firms_sequences,
        targets=targets,
        n_steps=5,
        batch_size=16,
        learning_rate=1e-3,
        device=device,
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ DeepState Spatial: {deepstate_features} features loaded")
    print(f"✓ FIRMS Spatial: {firms_features} features loaded")
    print(f"✓ Total sequences: {len(ds_sequences)}")
    print(f"✓ Training steps completed: 5")
    print(f"✓ Loss progression: {' → '.join([f'{l:.4f}' for l in losses])}")
    print(f"✓ Final loss: {losses[-1]:.6f}")

    # Data coverage summary
    print("\nData Coverage:")
    print(f"   Date range: {ds_aligned.index.min().date()} to {ds_aligned.index.max().date()}")
    print(f"   Total days: {len(ds_aligned)}")
    print(f"   DeepState coverage: {masks['deepstate'].mean()*100:.1f}%")
    print(f"   FIRMS coverage: {masks['firms'].mean()*100:.1f}%")

    print("\n" + "=" * 60)
    print("SPATIAL DATA TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return {
        'deepstate_features': deepstate_features,
        'firms_features': firms_features,
        'n_sequences': len(ds_sequences),
        'losses': losses,
        'date_range': (ds_aligned.index.min(), ds_aligned.index.max()),
    }


if __name__ == '__main__':
    results = main()
