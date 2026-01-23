"""
Domain-specific missing data imputation strategies for ML_OSINT.

This module provides imputation strategies tailored to the characteristics
of different data sources in the conflict analysis domain. Each strategy
accounts for the semantic meaning of missing data in its respective domain.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from scipy.interpolate import interp1d


class DomainImputationStrategy(ABC):
    """
    Base class for domain-specific imputation strategies.

    Each domain has different semantics for missing data, requiring
    tailored imputation approaches that preserve data meaning.
    """

    @abstractmethod
    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute missing values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with DatetimeIndex containing observed data.
        date_range : pd.DatetimeIndex
            Complete date range to fill (may extend beyond observed data).
        feature_cols : List[str]
            Column names to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            - Imputed DataFrame with complete date_range index
            - Observation mask (1=observed, 0=imputed)
        """
        pass

    def _create_observation_mask(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create a mask indicating which values were observed vs imputed.

        Parameters
        ----------
        original_df : pd.DataFrame
            Original DataFrame before imputation.
        imputed_df : pd.DataFrame
            DataFrame after imputation.
        feature_cols : List[str]
            Columns to create mask for.

        Returns
        -------
        pd.DataFrame
            Mask with 1 for observed values, 0 for imputed values.
        """
        mask = pd.DataFrame(
            index=imputed_df.index,
            columns=feature_cols,
            dtype=np.int8
        )
        mask[:] = 0

        # Mark observed values
        common_idx = original_df.index.intersection(imputed_df.index)
        for col in feature_cols:
            if col in original_df.columns:
                observed_idx = original_df.loc[common_idx, col].dropna().index
                mask.loc[observed_idx, col] = 1

        return mask


class UCDPImputation(DomainImputationStrategy):
    """
    Imputation strategy for UCDP (Uppsala Conflict Data Program) event data.

    For conflict event data, missing observations typically indicate periods
    of low or no activity rather than data collection failures. This strategy
    uses rolling median with a decay factor to account for uncertainty in
    imputed values.

    Attributes
    ----------
    decay_factor : float
        Decay factor applied to rolling median for uncertainty (default: 0.5).
    window_size : int
        Window size for rolling median calculation (default: 7 days).
    """

    def __init__(self, decay_factor: float = 0.5, window_size: int = 7):
        """
        Initialize UCDP imputation strategy.

        Parameters
        ----------
        decay_factor : float
            Decay factor for uncertainty in imputed values.
        window_size : int
            Rolling window size in days.
        """
        self.decay_factor = decay_factor
        self.window_size = window_size

    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute UCDP conflict data using rolling median with decay.

        Missing conflict data is assumed to indicate low/no activity.
        Imputed values use rolling median multiplied by decay factor
        to reflect uncertainty.

        Parameters
        ----------
        df : pd.DataFrame
            Observed conflict event data.
        date_range : pd.DatetimeIndex
            Complete date range to fill.
        feature_cols : List[str]
            Conflict metric columns to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Imputed DataFrame and observation mask.
        """
        # Reindex to complete date range
        imputed_df = df.reindex(date_range)

        # Store original for mask creation
        original_df = df.copy()

        for col in feature_cols:
            if col not in imputed_df.columns:
                imputed_df[col] = np.nan

            # Calculate rolling median from observed values
            rolling_median = imputed_df[col].rolling(
                window=self.window_size,
                min_periods=1,
                center=True
            ).median()

            # Apply decay factor to imputed values (uncertainty adjustment)
            missing_mask = imputed_df[col].isna()
            imputed_values = rolling_median * self.decay_factor

            # Fill missing with decayed rolling median
            imputed_df.loc[missing_mask, col] = imputed_values.loc[missing_mask]

            # For completely missing periods, use global median with decay
            still_missing = imputed_df[col].isna()
            if still_missing.any():
                global_median = df[col].median() if col in df.columns else 0
                imputed_df.loc[still_missing, col] = global_median * self.decay_factor

            # Floor at zero (no negative events)
            imputed_df[col] = imputed_df[col].clip(lower=0)

        mask = self._create_observation_mask(original_df, imputed_df, feature_cols)

        return imputed_df, mask


class FIRMSImputation(DomainImputationStrategy):
    """
    Imputation strategy for FIRMS (Fire Information for Resource Management) data.

    Missing fire detection data is typically due to satellite pass timing or
    cloud cover rather than absence of fire activity. This strategy uses
    linear interpolation for short gaps and forward/backward fill for longer gaps.

    Attributes
    ----------
    short_gap_limit : int
        Maximum gap length for linear interpolation (default: 3 days).
    long_gap_limit : int
        Maximum gap length for forward/backward fill (default: 7 days).
    """

    def __init__(self, short_gap_limit: int = 3, long_gap_limit: int = 7):
        """
        Initialize FIRMS imputation strategy.

        Parameters
        ----------
        short_gap_limit : int
            Limit for linear interpolation.
        long_gap_limit : int
            Limit for forward/backward fill.
        """
        self.short_gap_limit = short_gap_limit
        self.long_gap_limit = long_gap_limit

    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute FIRMS fire data using interpolation and fill strategies.

        Short gaps use linear interpolation to estimate fire activity.
        Longer gaps use forward/backward fill as activity likely persists.

        Parameters
        ----------
        df : pd.DataFrame
            Observed fire detection data.
        date_range : pd.DatetimeIndex
            Complete date range to fill.
        feature_cols : List[str]
            Fire metric columns to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Imputed DataFrame and observation mask.
        """
        # Reindex to complete date range
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col not in imputed_df.columns:
                imputed_df[col] = np.nan
                continue

            # Step 1: Linear interpolation for short gaps
            imputed_df[col] = imputed_df[col].interpolate(
                method='linear',
                limit=self.short_gap_limit,
                limit_direction='both'
            )

            # Step 2: Forward fill for longer gaps
            imputed_df[col] = imputed_df[col].ffill(limit=self.long_gap_limit)

            # Step 3: Backward fill for remaining gaps
            imputed_df[col] = imputed_df[col].bfill(limit=self.long_gap_limit)

            # Floor at zero (no negative detections)
            imputed_df[col] = imputed_df[col].clip(lower=0)

        mask = self._create_observation_mask(original_df, imputed_df, feature_cols)

        return imputed_df, mask


class DeepStateImputation(DomainImputationStrategy):
    """
    Imputation strategy for DeepState territorial control data.

    Territorial control is cumulative and persistent - territory remains
    under control until explicitly observed otherwise. This strategy uses
    forward fill as the primary method, with backward fill for initial periods.

    This is semantically correct because territorial changes are discrete
    events, and control persists between observations.
    """

    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute territorial control data using forward/backward fill.

        Territory persists until next observation, making forward fill
        the semantically correct approach. Backward fill handles initial
        periods before first observation.

        Parameters
        ----------
        df : pd.DataFrame
            Observed territorial control data.
        date_range : pd.DatetimeIndex
            Complete date range to fill.
        feature_cols : List[str]
            Territory metric columns to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Imputed DataFrame and observation mask.
        """
        # Reindex to complete date range
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col not in imputed_df.columns:
                imputed_df[col] = np.nan
                continue

            # Primary: Forward fill (territory persists)
            imputed_df[col] = imputed_df[col].ffill()

            # Secondary: Backward fill for initial periods
            imputed_df[col] = imputed_df[col].bfill()

        mask = self._create_observation_mask(original_df, imputed_df, feature_cols)

        return imputed_df, mask


class EquipmentImputation(DomainImputationStrategy):
    """
    Imputation strategy for equipment (and personnel) loss data.

    Equipment and personnel losses are cumulative counts - losses don't
    un-happen. This strategy uses forward fill to maintain running totals,
    with initial missing values filled with zero (no losses recorded yet).

    This strategy is also appropriate for personnel loss data which follows
    the same cumulative semantics.
    """

    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute cumulative loss data using forward fill with zero initialization.

        Cumulative counts are maintained via forward fill. Initial missing
        values are filled with zero as no losses have been recorded yet.

        Parameters
        ----------
        df : pd.DataFrame
            Observed equipment/personnel loss data.
        date_range : pd.DatetimeIndex
            Complete date range to fill.
        feature_cols : List[str]
            Loss count columns to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Imputed DataFrame and observation mask.
        """
        # Reindex to complete date range
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col not in imputed_df.columns:
                imputed_df[col] = np.nan
                continue

            # Forward fill (cumulative counts persist)
            imputed_df[col] = imputed_df[col].ffill()

            # Fill initial missing values with zero
            imputed_df[col] = imputed_df[col].fillna(0)

            # Ensure non-negative (cumulative counts can't be negative)
            imputed_df[col] = imputed_df[col].clip(lower=0)

        mask = self._create_observation_mask(original_df, imputed_df, feature_cols)

        return imputed_df, mask


class SentinelImputation(DomainImputationStrategy):
    """
    Imputation strategy for Sentinel satellite imagery data.

    Missing satellite data is typically due to cloud cover or orbit timing
    rather than absence of ground activity. This strategy uses linear
    interpolation for medium gaps and forward/backward fill for longer gaps.

    Attributes
    ----------
    interp_limit : int
        Maximum gap length for linear interpolation (default: 4 days).
    fill_limit : int
        Maximum gap length for forward/backward fill (default: 8 days).
    """

    def __init__(self, interp_limit: int = 4, fill_limit: int = 8):
        """
        Initialize Sentinel imputation strategy.

        Parameters
        ----------
        interp_limit : int
            Limit for linear interpolation.
        fill_limit : int
            Limit for forward/backward fill.
        """
        self.interp_limit = interp_limit
        self.fill_limit = fill_limit

    def impute(
        self,
        df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Impute satellite imagery data using interpolation and fill strategies.

        Linear interpolation handles medium gaps from cloud cover.
        Forward/backward fill handles longer gaps from orbit timing.

        Parameters
        ----------
        df : pd.DataFrame
            Observed satellite imagery metrics.
        date_range : pd.DatetimeIndex
            Complete date range to fill.
        feature_cols : List[str]
            Imagery metric columns to impute.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Imputed DataFrame and observation mask.
        """
        # Reindex to complete date range
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col not in imputed_df.columns:
                imputed_df[col] = np.nan
                continue

            # Step 1: Linear interpolation for medium gaps
            imputed_df[col] = imputed_df[col].interpolate(
                method='linear',
                limit=self.interp_limit,
                limit_direction='both'
            )

            # Step 2: Forward fill for longer gaps
            imputed_df[col] = imputed_df[col].ffill(limit=self.fill_limit)

            # Step 3: Backward fill for remaining gaps
            imputed_df[col] = imputed_df[col].bfill(limit=self.fill_limit)

        mask = self._create_observation_mask(original_df, imputed_df, feature_cols)

        return imputed_df, mask


# Registry mapping domain names to strategy instances
IMPUTATION_STRATEGIES: Dict[str, DomainImputationStrategy] = {
    'ucdp': UCDPImputation(),
    'firms': FIRMSImputation(),
    'deepstate': DeepStateImputation(),
    'equipment': EquipmentImputation(),
    'personnel': EquipmentImputation(),  # Same cumulative semantics as equipment
    'sentinel': SentinelImputation(),
}


def impute_domain_data(
    domain_name: str,
    df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    feature_cols: List[str],
    strategy: str = 'domain_specific'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply domain-specific imputation strategy to a DataFrame.

    This is the main entry point for imputing missing data. It selects
    the appropriate strategy based on the domain name and applies it
    to fill missing values while tracking which values were observed
    vs imputed.

    Parameters
    ----------
    domain_name : str
        Name of the data domain (e.g., 'ucdp', 'firms', 'deepstate',
        'equipment', 'personnel', 'sentinel').
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex containing observed data.
    date_range : pd.DatetimeIndex
        Complete date range to fill (may extend beyond observed data).
    feature_cols : List[str]
        Column names to impute.
    strategy : str, optional
        Imputation strategy to use. Options:
        - 'domain_specific': Use the registered domain strategy (default)
        - 'forward_fill': Simple forward fill
        - 'linear': Linear interpolation
        - 'zero': Fill with zeros

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - imputed_df: DataFrame with imputed values and complete date_range index
        - mask: Observation mask where 1=observed, 0=imputed

    Raises
    ------
    ValueError
        If domain_name is not found in IMPUTATION_STRATEGIES and
        strategy is 'domain_specific'.

    Examples
    --------
    >>> date_range = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    >>> df = pd.DataFrame({'events': [1, np.nan, 3]},
    ...                   index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
    >>> imputed, mask = impute_domain_data('ucdp', df, date_range, ['events'])
    >>> imputed.shape[0]
    31
    """
    domain_key = domain_name.lower()

    if strategy == 'domain_specific':
        if domain_key not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"Unknown domain '{domain_name}'. Available domains: "
                f"{list(IMPUTATION_STRATEGIES.keys())}"
            )
        imputation_strategy = IMPUTATION_STRATEGIES[domain_key]
        return imputation_strategy.impute(df, date_range, feature_cols)

    elif strategy == 'forward_fill':
        # Simple forward fill strategy
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col in imputed_df.columns:
                imputed_df[col] = imputed_df[col].ffill().bfill()

        # Create mask
        mask = pd.DataFrame(
            index=imputed_df.index,
            columns=feature_cols,
            dtype=np.int8
        )
        mask[:] = 0
        common_idx = original_df.index.intersection(imputed_df.index)
        for col in feature_cols:
            if col in original_df.columns:
                observed_idx = original_df.loc[common_idx, col].dropna().index
                mask.loc[observed_idx, col] = 1

        return imputed_df, mask

    elif strategy == 'linear':
        # Linear interpolation strategy
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col in imputed_df.columns:
                imputed_df[col] = imputed_df[col].interpolate(
                    method='linear',
                    limit_direction='both'
                ).ffill().bfill()

        # Create mask
        mask = pd.DataFrame(
            index=imputed_df.index,
            columns=feature_cols,
            dtype=np.int8
        )
        mask[:] = 0
        common_idx = original_df.index.intersection(imputed_df.index)
        for col in feature_cols:
            if col in original_df.columns:
                observed_idx = original_df.loc[common_idx, col].dropna().index
                mask.loc[observed_idx, col] = 1

        return imputed_df, mask

    elif strategy == 'zero':
        # Fill with zeros strategy
        imputed_df = df.reindex(date_range)
        original_df = df.copy()

        for col in feature_cols:
            if col in imputed_df.columns:
                imputed_df[col] = imputed_df[col].fillna(0)
            else:
                imputed_df[col] = 0

        # Create mask
        mask = pd.DataFrame(
            index=imputed_df.index,
            columns=feature_cols,
            dtype=np.int8
        )
        mask[:] = 0
        common_idx = original_df.index.intersection(imputed_df.index)
        for col in feature_cols:
            if col in original_df.columns:
                observed_idx = original_df.loc[common_idx, col].dropna().index
                mask.loc[observed_idx, col] = 1

        return imputed_df, mask

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available strategies: "
            f"'domain_specific', 'forward_fill', 'linear', 'zero'"
        )
