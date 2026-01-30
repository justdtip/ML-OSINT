# Temporal Deconfounding Implementation Guide

**Created**: 2026-01-30
**Purpose**: Step-by-step implementation instructions for agents
**Project Root**: `/Users/daniel.tipton/ML_OSINT`

---

## CONTEXT SUMMARY (READ THIS FIRST)

### The Problem
The ML_OSINT conflict prediction model overfits extremely early:
- Best validation loss at epoch 2-3
- Val loss increases while train loss decreases
- Root cause: Model learns TIME TRENDS instead of feature relationships
- 71% of feature correlations are spurious (disappear when controlling for time)

### The Solution (3 Parts)
1. **Delta Encoding**: Convert cumulative features to daily changes
2. **Detrending**: Subtract rolling mean to remove slow trends
3. **Temporal Regularization**: Penalize predictions that correlate with time

### Key Files
```
/Users/daniel.tipton/ML_OSINT/
├── analysis/
│   ├── multi_resolution_data.py      # Main data loader (3289 lines)
│   ├── train_multi_resolution.py     # Training script (3438 lines)
│   ├── training_config.py            # Configuration
│   └── preprocessing_utils.py        # TO BE CREATED
├── docs/
│   └── temporal-deconfounding-plan.md  # Design document
└── CLAUDE.md                         # Project conventions
```

---

## PHASE 1: DELTA ENCODING

### Goal
Convert cumulative equipment features to daily deltas so the model can't infer time from monotonic growth.

### What Already Exists (REUSE THIS)

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_data.py`

**Helper Function 1** (lines 1282-1333): `_load_equipment_base()`
```python
def _load_equipment_base(
    base_cols: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load equipment JSON, create date range, return (df, observation_mask)."""
```

**Helper Function 2** (lines 1336-1391): `_compute_delta_features()`
```python
def _compute_delta_features(
    df: pd.DataFrame,
    base_cols: List[str],
    prefix: str = '',
) -> pd.DataFrame:
    """Convert cumulative to delta-only features."""
    # Creates: {col}_daily, {col}_7day_avg, {col}_volatility
    # Plus aggregates: {prefix}_total_daily, {prefix}_total_7day_avg, {prefix}_momentum
```

**Working Examples** (already delta-encoded):
- `load_drones_daily()` at lines 1394-1433
- `load_armor_daily()` at lines 1444-1477
- `load_artillery_daily()` at lines 1488-1529

### Task 1.1: Fix load_equipment_daily()

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_data.py`
**Function**: `load_equipment_daily()` at lines 218-278
**Current Problem**: Returns raw cumulative values

**What to do**:
1. Read the function at lines 218-278
2. After loading the base data, call `_compute_delta_features()`
3. Return delta features instead of cumulative
4. Update the EQUIPMENT_FEATURE_NAMES list to match new features

**Expected Pattern** (copy from load_drones_daily):
```python
def load_equipment_daily(...) -> Tuple[pd.DataFrame, np.ndarray]:
    base_cols = ['aircraft', 'helicopter', 'tank', 'APC', ...]  # All equipment types
    df, observation_mask = _load_equipment_base(base_cols, start_date, end_date)

    if df.empty:
        return df, observation_mask

    features_df = _compute_delta_features(df, base_cols, prefix='equipment')
    return features_df, observation_mask
```

### Task 1.2: Create Missing Loaders

Three equipment categories need new loaders. Create them in `multi_resolution_data.py` after line 1568 (after `load_aircraft_daily`).

**Loader 1: load_naval_daily()**
```python
def load_naval_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load naval ship losses at daily resolution with delta features."""
    base_cols = ['naval ship']
    df, observation_mask = _load_equipment_base(base_cols, start_date, end_date)
    if df.empty:
        return df, observation_mask
    features_df = _compute_delta_features(df, base_cols, prefix='naval')
    return features_df, observation_mask

NAVAL_FEATURE_NAMES = [
    'naval ship_daily', 'naval ship_7day_avg', 'naval ship_volatility',
    'naval_total_daily', 'naval_total_7day_avg', 'naval_momentum',
]
```

**Loader 2: load_vehicles_daily()**
```python
def load_vehicles_daily(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load vehicles and fuel tanks losses at daily resolution with delta features."""
    base_cols = ['vehicles and fuel tanks', 'special equipment']
    df, observation_mask = _load_equipment_base(base_cols, start_date, end_date)
    if df.empty:
        return df, observation_mask
    features_df = _compute_delta_features(df, base_cols, prefix='vehicles')
    return features_df, observation_mask

VEHICLES_FEATURE_NAMES = [
    'vehicles and fuel tanks_daily', 'vehicles and fuel tanks_7day_avg', 'vehicles and fuel tanks_volatility',
    'special equipment_daily', 'special equipment_7day_avg', 'special equipment_volatility',
    'vehicles_total_daily', 'vehicles_total_7day_avg', 'vehicles_momentum',
]
```

### Task 1.3: Update Feature Name Constants

After creating loaders, update the EQUIPMENT_FEATURE_NAMES constant (search for it in the file) to reflect delta features instead of cumulative.

### Verification
After Phase 1, run:
```bash
cd /Users/daniel.tipton/ML_OSINT
python -c "from analysis.multi_resolution_data import load_equipment_daily; df, mask = load_equipment_daily(); print(df.columns.tolist())"
```
Should show `*_daily`, `*_7day_avg`, `*_volatility` columns, NOT raw equipment names.

---

## PHASE 2: DETRENDING

### Goal
Subtract rolling mean from features to remove slow-moving trends while keeping daily fluctuations.

### Task 2.1: Create preprocessing_utils.py

**Create new file**: `/Users/daniel.tipton/ML_OSINT/analysis/preprocessing_utils.py`

```python
"""
Preprocessing utilities for temporal deconfounding.

This module provides detrending functions to remove slow-moving trends
from time series features while preserving daily fluctuations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class DetrendingConfig:
    """Configuration for feature detrending."""
    enabled: bool = True
    method: str = 'rolling_mean'  # 'rolling_mean' or 'diff'
    default_window: int = 14
    min_periods: int = 1

    # Per-source window sizes (days)
    source_windows: Dict[str, int] = field(default_factory=lambda: {
        'personnel': 14,
        'drones': 14,
        'armor': 14,
        'artillery': 14,
        'equipment': 14,
        'deepstate_raion': 7,
        'firms_expanded_raion': 7,
        'geoconfirmed_raion': 7,
        'ucdp_raion': 7,
        'air_raid_sirens_raion': 7,
        'warspotting_raion': 7,
    })

    # Column patterns to EXCLUDE from detrending (already processed)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        'rolling', 'volatility', 'momentum', 'avg', 'std'
    ])


def detrend_features(
    features: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    config: DetrendingConfig,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Apply detrending to all feature DataFrames.

    Subtracts rolling mean to remove slow trends while preserving
    daily fluctuations needed for tactical prediction.

    Args:
        features: Dict mapping source name to (DataFrame, observation_mask)
        config: DetrendingConfig with window sizes and options

    Returns:
        Dict with detrended features (masks unchanged for rolling_mean method)
    """
    if not config.enabled:
        return features

    detrended = {}

    for source_name, (df, mask) in features.items():
        if df.empty:
            detrended[source_name] = (df, mask)
            continue

        feature_cols = [c for c in df.columns if c != 'date']

        # Identify columns to exclude from detrending
        exclude_cols = set()
        for pattern in config.exclude_patterns:
            exclude_cols.update([c for c in feature_cols if pattern.lower() in c.lower()])

        detrend_cols = [c for c in feature_cols if c not in exclude_cols]

        if not detrend_cols:
            detrended[source_name] = (df, mask)
            continue

        # Get window size for this source
        window = config.source_windows.get(source_name, config.default_window)

        # Apply detrending
        df_copy = df.copy()

        if config.method == 'rolling_mean':
            # Subtract rolling mean (keeps daily fluctuations)
            rolling_mean = df_copy[detrend_cols].rolling(
                window=window,
                min_periods=config.min_periods,
                center=False
            ).mean()
            df_copy[detrend_cols] = df_copy[detrend_cols] - rolling_mean
            # Fill NaN from early rows with 0 (no trend to remove yet)
            df_copy[detrend_cols] = df_copy[detrend_cols].fillna(0)
            new_mask = mask  # Mask unchanged

        elif config.method == 'diff':
            # First-order differencing
            df_copy[detrend_cols] = df_copy[detrend_cols].diff()
            df_copy[detrend_cols] = df_copy[detrend_cols].fillna(0)
            # Update mask: need both current and previous observation
            prev_mask = np.roll(mask, 1)
            prev_mask[0] = False
            new_mask = mask & prev_mask
        else:
            raise ValueError(f"Unknown detrending method: {config.method}")

        detrended[source_name] = (df_copy, new_mask)

    return detrended
```

### Task 2.2: Integrate Detrending into Data Loader

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/multi_resolution_data.py`

**Step 1**: Add import at top of file (after line 31):
```python
from analysis.preprocessing_utils import DetrendingConfig, detrend_features
```

**Step 2**: Add config option to `MultiResolutionConfig` dataclass (around line 98-150):
```python
# Add this field to the dataclass
apply_detrending: bool = False
detrending_config: Optional[DetrendingConfig] = None
```

**Step 3**: Add detrending call in `_load_all_sources()` method.

Find the method `_load_all_sources()` at line 1965. At the END of the method, before the `print` statements, add:
```python
# Apply detrending if configured
if self.config.apply_detrending:
    detrend_config = self.config.detrending_config or DetrendingConfig()
    print(f"  Applying detrending (window={detrend_config.default_window})...")
    self.daily_data = detrend_features(self.daily_data, detrend_config)
    # Note: Don't detrend monthly data (too few observations)
```

### Task 2.3: Add Command Line Flag

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

Find the argument parser section (around line 3034-3100). Add:
```python
parser.add_argument('--apply-detrending', action='store_true', default=False,
                    help='Apply rolling mean detrending to remove temporal trends')
parser.add_argument('--detrend-window', type=int, default=14,
                    help='Window size for detrending rolling mean (days)')
```

Then where the dataset is created (around line 3111-3150), pass the config:
```python
from analysis.preprocessing_utils import DetrendingConfig

detrend_config = None
if args.apply_detrending:
    detrend_config = DetrendingConfig(
        enabled=True,
        default_window=args.detrend_window,
    )

# Pass to MultiResolutionConfig
config = MultiResolutionConfig(
    # ... existing args ...
    apply_detrending=args.apply_detrending,
    detrending_config=detrend_config,
)
```

### Verification
After Phase 2, run:
```bash
cd /Users/daniel.tipton/ML_OSINT
python -m analysis.train_multi_resolution --apply-detrending --test
```
Should show "Applying detrending" in output without errors.

---

## PHASE 3: TEMPORAL REGULARIZATION

### Goal
Add loss term that penalizes predictions correlated with time position.

### Task 3.1: Add TemporalRegularizer Class

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

Add this class BEFORE the `MultiTaskLoss` class (around line 500):

```python
class TemporalRegularizer(nn.Module):
    """
    Regularization to prevent model from exploiting temporal position.

    Two components:
    1. Correlation penalty: predictions shouldn't correlate with time index
    2. Smoothness penalty: predictions shouldn't be too smooth over time
    """

    def __init__(
        self,
        correlation_weight: float = 0.01,
        smoothness_weight: float = 0.001,
    ):
        super().__init__()
        self.correlation_weight = correlation_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seq_len: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute temporal regularization loss.

        Args:
            outputs: Dict with model predictions
            seq_len: Sequence length
            batch_size: Batch size
            device: Torch device

        Returns:
            Scalar regularization loss
        """
        total_reg = torch.tensor(0.0, device=device)

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]

        # Normalize positions to mean=0, std=1
        pos_mean = positions.mean()
        pos_std = positions.std() + 1e-8
        pos_norm = (positions - pos_mean) / pos_std

        # Apply correlation penalty to relevant outputs
        for key in ['regime_logits', 'anomaly_score', 'forecast']:
            if key not in outputs:
                continue

            pred = outputs[key]
            if pred.dim() < 2:
                continue

            # Get sequence dimension
            if pred.shape[1] != seq_len:
                continue  # Skip if shape doesn't match

            # Flatten if needed for correlation
            if pred.dim() == 3:
                pred_flat = pred.mean(dim=-1)  # [batch, seq_len]
            else:
                pred_flat = pred

            # Normalize predictions
            pred_mean = pred_flat.mean(dim=1, keepdim=True)
            pred_std = pred_flat.std(dim=1, keepdim=True) + 1e-8
            pred_norm = (pred_flat - pred_mean) / pred_std

            # Compute correlation: mean(pred_norm * pos_norm)
            correlation = (pred_norm * pos_norm).mean()

            # Penalize positive correlation (using position for prediction)
            total_reg = total_reg + self.correlation_weight * torch.relu(correlation)

        # Smoothness penalty on anomaly predictions
        if 'anomaly_score' in outputs and self.smoothness_weight > 0:
            anomaly = outputs['anomaly_score']
            if anomaly.dim() >= 2 and anomaly.shape[1] > 1:
                # Compute temporal differences
                deltas = anomaly[:, 1:] - anomaly[:, :-1]
                smoothness = deltas.pow(2).mean()

                # Penalize if too smooth (want roughness > threshold)
                target_roughness = 0.1
                total_reg = total_reg + self.smoothness_weight * torch.relu(target_roughness - smoothness)

        return total_reg
```

### Task 3.2: Integrate into Training Loop

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

**Step 1**: Add regularizer initialization in `TrainerBase.__init__()` (around line 750-850):
```python
# After self.multi_task_loss initialization, add:
self.temporal_regularizer = TemporalRegularizer(
    correlation_weight=0.01,
    smoothness_weight=0.001,
) if getattr(self, 'use_temporal_reg', False) else None
```

**Step 2**: Modify loss computation in training loop (around line 2158-2165):

Find these lines:
```python
task_losses = self._compute_losses(outputs, batch)
total_loss, task_weights = self.multi_task_loss(task_losses)
```

Change to:
```python
task_losses = self._compute_losses(outputs, batch)
total_loss, task_weights = self.multi_task_loss(task_losses)

# Add temporal regularization
if self.temporal_regularizer is not None:
    seq_len = batch['daily_features'][next(iter(batch['daily_features']))].shape[1]
    batch_size = batch['batch_size']
    temporal_reg_loss = self.temporal_regularizer(
        outputs, seq_len, batch_size, self.device
    )
    total_loss = total_loss + temporal_reg_loss
```

### Task 3.3: Add Command Line Flag

**File**: `/Users/daniel.tipton/ML_OSINT/analysis/train_multi_resolution.py`

In argument parser (around line 3034-3100), add:
```python
parser.add_argument('--use-temporal-reg', action='store_true', default=False,
                    help='Enable temporal regularization to prevent time-based shortcuts')
parser.add_argument('--temporal-corr-weight', type=float, default=0.01,
                    help='Weight for temporal correlation penalty')
parser.add_argument('--temporal-smooth-weight', type=float, default=0.001,
                    help='Weight for temporal smoothness penalty')
```

Then pass to trainer initialization (find where TrainerBase is instantiated):
```python
trainer.use_temporal_reg = args.use_temporal_reg
if args.use_temporal_reg:
    trainer.temporal_regularizer = TemporalRegularizer(
        correlation_weight=args.temporal_corr_weight,
        smoothness_weight=args.temporal_smooth_weight,
    )
```

### Verification
After Phase 3, run:
```bash
cd /Users/daniel.tipton/ML_OSINT
python -m analysis.train_multi_resolution --use-temporal-reg --test
```
Should complete without errors and show temporal regularization being applied.

---

## FULL TEST COMMAND

After all phases, test with:
```bash
cd /Users/daniel.tipton/ML_OSINT
python -m analysis.train_multi_resolution \
    --apply-detrending \
    --use-temporal-reg \
    --epochs 50 \
    --batch_size 4
```

Expected improvements:
- Val loss should improve beyond epoch 3
- Train/val curves should be more parallel
- Previously unused sources (air_raid_sirens_raion, warspotting_raion) may show non-zero impact

---

## CONTEXT FOR CONTINUATION

If context runs out, here's what matters:

1. **Project**: `/Users/daniel.tipton/ML_OSINT` - conflict prediction ML pipeline
2. **Problem**: Model overfits at epoch 2-3 due to temporal trend confounding
3. **Solution**: Delta encoding + detrending + temporal regularization
4. **Key files**:
   - `analysis/multi_resolution_data.py` - data loading
   - `analysis/train_multi_resolution.py` - training
   - `analysis/preprocessing_utils.py` - NEW detrending module
5. **Design doc**: `docs/temporal-deconfounding-plan.md`
6. **Implementation doc**: `docs/temporal-deconfounding-implementation.md` (this file)

