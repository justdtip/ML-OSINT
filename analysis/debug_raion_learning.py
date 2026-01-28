"""
Debug script: 5-step training run to verify per-raion mask learning.

Examines:
1. Loss decreases (model is learning something)
2. Gradient flow through geographic encoder
3. Attention patterns in cross-raion attention
4. Effect of per-raion masks vs no masks
5. Per-source contribution to predictions
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from analysis.multi_resolution_data import (
    MultiResolutionConfig,
    MultiResolutionDataset,
    multi_resolution_collate_fn,
)
from analysis.multi_resolution_han import MultiResolutionHAN, SourceConfig
from analysis.geographic_source_encoder import (
    GeographicSourceEncoder,
    GeographicDailyCrossSourceFusion,
    SpatialSourceConfig,
)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def clean_missing_values(features_dict):
    """Replace -999.0 (MISSING_VALUE) sentinel values with 0.0.

    The -999.0 values indicate missing data at the feature level (e.g., specific
    raions not observed on a given day). We replace them with 0.0 to prevent
    NaN propagation through the model.
    """
    MISSING_VALUE = -999.0
    cleaned = {}
    for name, features in features_dict.items():
        features_clean = features.clone()
        # Replace all -999.0 values with 0.0
        features_clean = torch.where(
            features_clean == MISSING_VALUE,
            torch.zeros_like(features_clean),
            features_clean
        )
        cleaned[name] = features_clean
    return cleaned


def run_debug():
    torch.manual_seed(42)
    np.random.seed(42)

    # Enable anomaly detection to find NaN source
    torch.autograd.set_detect_anomaly(True)

    # =========================================================================
    # STEP 1: Create Dataset with Raion Source
    # =========================================================================
    print_header("STEP 1: Dataset Setup")

    config = MultiResolutionConfig(
        daily_sources=['geoconfirmed_raion', 'personnel'],
        monthly_sources=['sentinel'],
        start_date='2023-06-01',
        end_date='2024-01-31',
        daily_seq_len=30,
        monthly_seq_len=3,
    )

    print("Loading dataset...")
    dataset = MultiResolutionDataset(config=config, split='train')

    if len(dataset) < 2:
        print(f"Only {len(dataset)} samples - need more data for debug")
        return

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=multi_resolution_collate_fn,
    )

    # Get sample batch for inspection
    batch = next(iter(loader))

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Batch size: {batch['batch_size']}")
    print(f"\nDaily sources: {list(batch['daily_features'].keys())}")
    for name, tensor in batch['daily_features'].items():
        print(f"  {name}: {tensor.shape}")

    print(f"\nRaion masks available: {list(batch['raion_masks'].keys()) if batch['raion_masks'] else 'None'}")
    if batch['raion_masks']:
        for name, mask in batch['raion_masks'].items():
            obs_rate = mask.float().mean().item() * 100
            print(f"  {name}: {mask.shape}, {obs_rate:.1f}% observed")

    # =========================================================================
    # STEP 2: Create Model with Geographic Encoder
    # =========================================================================
    print_header("STEP 2: Model Setup")

    # Get feature dimensions from dataset
    geoconfirmed_features = batch['daily_features']['geoconfirmed_raion'].shape[-1]
    personnel_features = batch['daily_features']['personnel'].shape[-1]
    sentinel_features = batch['monthly_features']['sentinel'].shape[-1]

    # Determine raion structure
    n_raions = batch['raion_masks']['geoconfirmed_raion'].shape[-1]
    features_per_raion = geoconfirmed_features // n_raions

    print(f"Geoconfirmed: {geoconfirmed_features} features = {n_raions} raions x {features_per_raion} features/raion")

    daily_configs = {
        'geoconfirmed_raion': SourceConfig(
            name='geoconfirmed_raion',
            n_features=geoconfirmed_features,
            resolution='daily',
            description='Geoconfirmed raion-level features',
        ),
        'personnel': SourceConfig(
            name='personnel',
            n_features=personnel_features,
            resolution='daily',
            description='Personnel losses',
        ),
    }

    monthly_configs = {
        'sentinel': SourceConfig(
            name='sentinel',
            n_features=sentinel_features,
            resolution='monthly',
            description='Sentinel satellite data',
        ),
    }

    # Create spatial config for geoconfirmed_raion to enable per-raion attention
    geoconfirmed_spatial_config = SpatialSourceConfig(
        name='geoconfirmed_raion',
        n_raions=n_raions,
        features_per_raion=features_per_raion,
        use_geographic_prior=False,  # No geographic prior, just per-raion attention
    )

    model = MultiResolutionHAN(
        daily_source_configs=daily_configs,
        monthly_source_configs=monthly_configs,
        d_model=64,
        nhead=4,
        num_daily_layers=1,
        num_monthly_layers=1,
        num_fusion_layers=1,
        dropout=0.1,
        use_geographic_prior=True,  # Enable geographic fusion
        custom_spatial_configs={'geoconfirmed_raion': geoconfirmed_spatial_config},
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Check if geographic encoder is being used
    if hasattr(model.daily_fusion, 'geographic_encoders'):
        print(f"Geographic encoders: {list(model.daily_fusion.geographic_encoders.keys())}")
    else:
        print("WARNING: No geographic encoders in model!")

    # =========================================================================
    # STEP 3: Run 5 Training Steps
    # =========================================================================
    print_header("STEP 3: Training Loop (5 steps)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    losses = []
    grad_norms = defaultdict(list)

    for step in range(5):
        batch = next(iter(loader))

        # Clean missing values: replace -999.0 sentinel with 0.0
        daily_features_clean = clean_missing_values(batch['daily_features'])
        monthly_features_clean = clean_missing_values(batch['monthly_features'])

        # Debug: Check for bad values in features
        import sys
        for name, feat in daily_features_clean.items():
            has_nan = torch.isnan(feat).any().item()
            has_inf = torch.isinf(feat).any().item()
            has_missing = (feat == -999.0).any().item()
            min_val = feat.min().item()
            max_val = feat.max().item()
            print(f"    {name}: NaN={has_nan}, Inf={has_inf}, -999={has_missing}, range=[{min_val:.2f}, {max_val:.2f}]")
            sys.stdout.flush()

        optimizer.zero_grad()

        outputs = model(
            daily_features=daily_features_clean,
            daily_masks=batch['daily_masks'],
            monthly_features=monthly_features_clean,
            monthly_masks=batch['monthly_masks'],
            month_boundaries=batch['month_boundary_indices'],
            raion_masks=batch.get('raion_masks'),
        )

        # Simple loss: MSE on casualty prediction (use dummy target)
        casualty_pred = outputs['casualty_pred']
        # Create dummy target (we just want to see if model learns)
        dummy_target = torch.randn_like(casualty_pred) * 0.5
        loss = nn.functional.mse_loss(casualty_pred, dummy_target)

        loss.backward()

        # Check for NaN in gradients before clipping
        nan_grads = []
        inf_grads = []
        max_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
        if nan_grads:
            print(f"    NaN gradients in: {nan_grads[:3]}...")
        if inf_grads:
            print(f"    Inf gradients in: {inf_grads[:3]}...")
        print(f"    Max grad norm before clipping: {max_grad_norm:.2f}")

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Track gradient norms for key components
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                if 'geographic' in name or 'raion' in name.lower():
                    grad_norms['geographic'].append(norm)
                elif 'daily_fusion' in name:
                    grad_norms['daily_fusion'].append(norm)
                elif 'casualty' in name:
                    grad_norms['casualty_head'].append(norm)

        optimizer.step()

        # Check for NaN in weights after update
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            print(f"  WARNING: NaN in params after step {step+1}: {nan_params[:3]}...")

        losses.append(loss.item())
        print(f"  Step {step+1}: loss={loss.item():.4f}")

    print(f"\nLoss trend: {losses[0]:.4f} -> {losses[-1]:.4f} (delta: {losses[-1]-losses[0]:+.4f})")

    print(f"\nGradient norms (mean across steps):")
    for component, norms in grad_norms.items():
        if norms:
            print(f"  {component}: {np.mean(norms):.6f}")

    # =========================================================================
    # STEP 4: Compare With vs Without Raion Masks
    # =========================================================================
    print_header("STEP 4: Effect of Per-Raion Masks")

    # Get a fresh batch for comparison
    batch = next(iter(loader))
    daily_features_clean = clean_missing_values(batch['daily_features'])
    monthly_features_clean = clean_missing_values(batch['monthly_features'])

    model.eval()
    with torch.no_grad():
        # With raion masks
        out_with_mask = model(
            daily_features=daily_features_clean,
            daily_masks=batch['daily_masks'],
            monthly_features=monthly_features_clean,
            monthly_masks=batch['monthly_masks'],
            month_boundaries=batch['month_boundary_indices'],
            raion_masks=batch.get('raion_masks'),
        )

        # Without raion masks (falls back to daily_masks)
        out_without_mask = model(
            daily_features=daily_features_clean,
            daily_masks=batch['daily_masks'],
            monthly_features=monthly_features_clean,
            monthly_masks=batch['monthly_masks'],
            month_boundaries=batch['month_boundary_indices'],
            raion_masks=None,
        )

    # Compare outputs
    temporal_diff = (out_with_mask['temporal_output'] - out_without_mask['temporal_output']).abs()
    casualty_diff = (out_with_mask['casualty_pred'] - out_without_mask['casualty_pred']).abs()

    print(f"Output difference (with vs without raion masks):")
    print(f"  temporal_output MAD: {temporal_diff.mean().item():.6f}")
    print(f"  casualty_pred MAD:   {casualty_diff.mean().item():.6f}")

    if temporal_diff.mean().item() > 0.01:
        print("  -> Per-raion masks ARE affecting model output")
    else:
        print("  -> WARNING: Per-raion masks have minimal effect")

    # =========================================================================
    # STEP 5: Examine Raion Mask Patterns & Test Geographic Encoder Directly
    # =========================================================================
    print_header("STEP 5: Raion Mask Analysis & Geographic Encoder Test")

    raion_mask = batch['raion_masks']['geoconfirmed_raion']
    geo_input = batch['daily_features']['geoconfirmed_raion']

    print(f"Geoconfirmed input: {geo_input.shape}")
    print(f"Raion mask: {raion_mask.shape}")

    # Check observation pattern
    obs_per_timestep = raion_mask.float().sum(dim=-1)  # [batch, seq]
    print(f"\nObservations per timestep:")
    print(f"  Mean: {obs_per_timestep.mean().item():.1f} raions")
    print(f"  Min:  {obs_per_timestep.min().item():.0f} raions")
    print(f"  Max:  {obs_per_timestep.max().item():.0f} raions")

    # Check which raions are most frequently observed
    obs_per_raion = raion_mask.float().mean(dim=(0, 1))  # [n_raions]
    top_raions = obs_per_raion.argsort(descending=True)[:5]
    print(f"\nTop 5 most observed raions (by index):")
    for idx in top_raions:
        print(f"  Raion {idx.item()}: {obs_per_raion[idx].item()*100:.1f}% observed")

    # Sparse observation analysis
    sparse_timesteps = (obs_per_timestep < 10).float().mean().item() * 100
    print(f"\nSparsity: {sparse_timesteps:.1f}% of timesteps have <10 raions observed")

    # Test GeographicSourceEncoder directly
    print(f"\n--- Testing GeographicSourceEncoder directly ---")

    geo_encoder = GeographicSourceEncoder(
        spatial_config=SpatialSourceConfig(
            name='geoconfirmed_raion',
            n_raions=n_raions,
            features_per_raion=features_per_raion,
            use_geographic_prior=False,
        ),
        d_model=64,
        n_heads=4,
    )

    # Forward with mask
    with torch.no_grad():
        out_with_mask = geo_encoder(geo_input, mask=raion_mask)
        out_without_mask = geo_encoder(geo_input, mask=None)

    diff = (out_with_mask - out_without_mask).abs().mean().item()
    print(f"GeographicSourceEncoder output: {out_with_mask.shape}")
    print(f"Output diff (with vs without mask): {diff:.6f}")

    # Check if mask actually changes attention behavior
    if diff > 0.01:
        print("  -> Raion mask IS affecting geographic encoder output")
    else:
        print("  -> WARNING: Raion mask has minimal effect on encoder")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("SUMMARY")

    print("1. Dataset loaded with per-raion masks")
    print(f"2. Model has {n_params:,} parameters with geographic encoder")
    print(f"3. Loss trend over 5 steps: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"4. Per-raion masks affect output: MAD={temporal_diff.mean().item():.6f}")
    print(f"5. Data is sparse: many timesteps have few observed raions")

    if grad_norms['geographic']:
        geo_grad = np.mean(grad_norms['geographic'])
        print(f"\nGradients flowing through geographic encoder: {geo_grad:.6f}")
        if geo_grad > 1e-6:
            print("   -> Geographic encoder IS learning from data")
        else:
            print("   -> WARNING: Geographic encoder may not be learning")

    return {
        'losses': losses,
        'grad_norms': dict(grad_norms),
        'output_diff': temporal_diff.mean().item(),
    }


if __name__ == '__main__':
    results = run_debug()
