#!/usr/bin/env python3
"""
Stage 2: Temporal Structure Analysis

Characterizes temporal patterns in residuals:
- Autocorrelation (ACF) - persistence of residuals over time
- Cross-residual correlation - shared omitted factors between sources
- Spectral analysis (PSD) - periodic components in residuals

These patterns reveal the properties of missing information even when
the missing variables themselves are unknown.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

import numpy as np
from scipy import signal, stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add parent paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from .residual_extraction import ResidualExtractionResults


@dataclass
class TemporalAnalysisResults:
    """Container for temporal analysis outputs."""
    acf_results: Dict[str, Dict]  # source -> ACF analysis
    cross_residual_correlations: Dict[Tuple[str, str], Dict]  # (src_i, src_j) -> correlation
    spectral_results: Dict[str, Dict]  # source -> spectral analysis
    metadata: Dict = field(default_factory=dict)

    def save(self, output_dir: Path):
        """Save results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings for JSON serialization
        xcorr_serializable = {
            f"{k[0]}___{k[1]}": v
            for k, v in self.cross_residual_correlations.items()
        }

        # Convert numpy arrays to lists for JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        results = {
            'acf_results': convert_for_json(self.acf_results),
            'cross_residual_correlations': convert_for_json(xcorr_serializable),
            'spectral_results': convert_for_json(self.spectral_results),
            'metadata': convert_for_json(self.metadata)
        }

        with open(output_dir / 'temporal_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved temporal analysis to {output_dir}")

    @classmethod
    def load(cls, output_dir: Path) -> 'TemporalAnalysisResults':
        """Load results from disk."""
        output_dir = Path(output_dir)

        with open(output_dir / 'temporal_analysis.json') as f:
            data = json.load(f)

        # Convert string keys back to tuples
        xcorr = {}
        for k, v in data['cross_residual_correlations'].items():
            parts = k.split('___')
            xcorr[(parts[0], parts[1])] = v

        return cls(
            acf_results=data['acf_results'],
            cross_residual_correlations=xcorr,
            spectral_results=data['spectral_results'],
            metadata=data.get('metadata', {})
        )


def compute_residual_autocorrelation(
    residuals: Dict[str, np.ndarray],
    max_lag: int = 60
) -> Dict[str, Dict]:
    """
    Compute autocorrelation function for aggregated residuals.

    For each source, compute ACF of the mean residual magnitude
    (L2 norm across features per timestep).

    Args:
        residuals: Dict mapping source names to residual arrays (n_samples, n_features)
        max_lag: Maximum lag to compute ACF for

    Returns:
        Dict mapping source names to ACF analysis results
    """
    try:
        from statsmodels.tsa.stattools import acf
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False
        print("  WARNING: statsmodels not available, using manual ACF computation")

    acf_results = {}

    for source, res in residuals.items():
        # Aggregate to scalar time series: L2 norm of residual vector
        residual_magnitude = np.linalg.norm(res, axis=1)
        n_samples = len(residual_magnitude)

        # Adjust max_lag if needed
        effective_max_lag = min(max_lag, n_samples // 3)

        if HAS_STATSMODELS:
            # Compute ACF with confidence intervals
            acf_values, confint = acf(
                residual_magnitude,
                nlags=effective_max_lag,
                alpha=0.05,
                fft=True
            )

            # Identify significant lags (outside 95% CI)
            # confint shape is (nlags+1, 2) with [lower, upper] bounds
            significant_lags = []
            for lag in range(1, len(acf_values)):
                if lag < len(confint):
                    lower, upper = confint[lag]
                    if acf_values[lag] < lower or acf_values[lag] > upper:
                        significant_lags.append(lag)
        else:
            # Manual ACF computation
            acf_values = np.zeros(effective_max_lag + 1)
            acf_values[0] = 1.0
            centered = residual_magnitude - np.mean(residual_magnitude)
            var = np.var(residual_magnitude)

            for lag in range(1, effective_max_lag + 1):
                if var > 1e-10:
                    acf_values[lag] = np.mean(centered[:-lag] * centered[lag:]) / var
                else:
                    acf_values[lag] = 0.0

            # Approximate 95% CI as ±1.96/sqrt(n)
            ci_bound = 1.96 / np.sqrt(n_samples)
            confint = np.array([[-ci_bound, ci_bound]] * (effective_max_lag + 1))
            significant_lags = [
                lag for lag in range(1, len(acf_values))
                if abs(acf_values[lag]) > ci_bound
            ]

        # Find decay rate (first lag where ACF drops below 0.5)
        decay_lag = None
        for lag in range(1, len(acf_values)):
            if acf_values[lag] < 0.5:
                decay_lag = lag
                break

        # Compute Ljung-Box test for residual autocorrelation
        # High Q-stat with low p-value indicates significant autocorrelation
        ljung_box_q = None
        ljung_box_p = None
        if HAS_STATSMODELS:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_result = acorr_ljungbox(residual_magnitude, lags=[10], return_df=False)
                ljung_box_q = float(lb_result[0][0])
                ljung_box_p = float(lb_result[1][0])
            except Exception:
                pass

        acf_results[source] = {
            'lags': list(range(effective_max_lag + 1)),
            'acf': acf_values.tolist(),
            'confidence_interval_lower': confint[:, 0].tolist() if len(confint.shape) > 1 else [-ci_bound] * len(acf_values),
            'confidence_interval_upper': confint[:, 1].tolist() if len(confint.shape) > 1 else [ci_bound] * len(acf_values),
            'significant_lags': significant_lags,
            'n_significant_lags': len(significant_lags),
            'decay_lag': decay_lag,
            'ljung_box_q': ljung_box_q,
            'ljung_box_p': ljung_box_p,
            'interpretation': interpret_acf(acf_values, significant_lags, decay_lag)
        }

    return acf_results


def interpret_acf(acf_values: np.ndarray, significant_lags: List[int], decay_lag: Optional[int]) -> List[str]:
    """Generate interpretation of ACF results."""
    interpretations = []

    if len(significant_lags) == 0:
        interpretations.append('no_significant_autocorrelation')
    elif len(significant_lags) > 10:
        interpretations.append('strong_persistent_autocorrelation')
    else:
        interpretations.append('moderate_autocorrelation')

    if decay_lag is not None:
        if decay_lag <= 3:
            interpretations.append('fast_decay')
        elif decay_lag <= 10:
            interpretations.append('moderate_decay')
        else:
            interpretations.append('slow_decay_long_memory')

    # Check for periodic patterns
    if 7 in significant_lags:
        interpretations.append('weekly_periodicity')
    if 14 in significant_lags:
        interpretations.append('biweekly_periodicity')
    if any(lag in significant_lags for lag in [28, 29, 30, 31]):
        interpretations.append('monthly_periodicity')

    return interpretations


def compute_cross_residual_correlation(
    residuals: Dict[str, np.ndarray],
    max_lag: int = 60
) -> Dict[Tuple[str, str], Dict]:
    """
    Compute lagged cross-correlation between residuals of different sources.

    Cross-residual correlation reveals shared omitted factors affecting
    multiple sources with potentially different lags.

    Args:
        residuals: Dict mapping source names to residual arrays
        max_lag: Maximum lag to compute cross-correlation for

    Returns:
        Dict mapping (source_i, source_j) tuples to correlation analysis
    """
    sources = list(residuals.keys())
    xcorr_results = {}

    for i, src_i in enumerate(sources):
        for j, src_j in enumerate(sources):
            if i >= j:
                continue  # Only compute upper triangle

            # Aggregate to scalar
            mag_i = np.linalg.norm(residuals[src_i], axis=1)
            mag_j = np.linalg.norm(residuals[src_j], axis=1)

            # Normalize (z-score)
            if np.std(mag_i) > 1e-10:
                mag_i = (mag_i - np.mean(mag_i)) / np.std(mag_i)
            if np.std(mag_j) > 1e-10:
                mag_j = (mag_j - np.mean(mag_j)) / np.std(mag_j)

            # Cross-correlation using scipy
            xcorr_full = signal.correlate(mag_i, mag_j, mode='full') / len(mag_i)
            lags_full = signal.correlation_lags(len(mag_i), len(mag_j), mode='full')

            # Trim to max_lag
            center = len(lags_full) // 2
            start_idx = max(0, center - max_lag)
            end_idx = min(len(lags_full), center + max_lag + 1)

            lags = lags_full[start_idx:end_idx]
            xcorr = xcorr_full[start_idx:end_idx]

            # Find peak correlation and lag
            peak_idx = np.argmax(np.abs(xcorr))
            peak_lag = int(lags[peak_idx])
            peak_correlation = float(xcorr[peak_idx])

            # Zero-lag correlation
            zero_idx = np.where(lags == 0)[0]
            zero_lag_corr = float(xcorr[zero_idx[0]]) if len(zero_idx) > 0 else 0.0

            # Statistical significance (approximate)
            n = len(mag_i)
            se = 1.0 / np.sqrt(n)
            is_significant = abs(peak_correlation) > 1.96 * se

            xcorr_results[(src_i, src_j)] = {
                'lags': lags.tolist(),
                'correlation': xcorr.tolist(),
                'peak_lag': peak_lag,
                'peak_correlation': peak_correlation,
                'zero_lag_correlation': zero_lag_corr,
                'is_significant': is_significant,
                'interpretation': interpret_xcorr(peak_lag, peak_correlation, zero_lag_corr, src_i, src_j)
            }

    return xcorr_results


def interpret_xcorr(
    peak_lag: int,
    peak_corr: float,
    zero_lag_corr: float,
    src_i: str,
    src_j: str
) -> List[str]:
    """Generate interpretation of cross-correlation results."""
    interpretations = []

    if abs(peak_corr) < 0.1:
        interpretations.append('weak_relationship')
    elif abs(peak_corr) < 0.3:
        interpretations.append('moderate_relationship')
    else:
        interpretations.append('strong_relationship')

    if abs(zero_lag_corr) > 0.2:
        interpretations.append('shared_instantaneous_factor')

    if peak_lag == 0:
        interpretations.append('synchronous')
    elif peak_lag > 0:
        interpretations.append(f'{src_i}_leads_{src_j}_by_{peak_lag}_days')
    else:
        interpretations.append(f'{src_j}_leads_{src_i}_by_{-peak_lag}_days')

    return interpretations


def compute_residual_spectra(
    residuals: Dict[str, np.ndarray],
    sampling_rate: float = 1.0
) -> Dict[str, Dict]:
    """
    Compute power spectral density of residuals using Welch's method.

    Peaks in the PSD reveal missing periodic components (weekly cycles,
    monthly reporting, seasonal effects, etc.).

    Args:
        residuals: Dict of residual arrays
        sampling_rate: Samples per day (1.0 for daily data)

    Returns:
        Dict mapping source names to spectral analysis results
    """
    # Known periodicities to check
    KNOWN_PERIODS = {
        7: 'weekly_cycle',
        14: 'biweekly_cycle',
        30: 'monthly_cycle',
        90: 'quarterly_cycle',
        365: 'annual_cycle'
    }

    spectral_results = {}

    for source, res in residuals.items():
        # Aggregate to scalar
        residual_magnitude = np.linalg.norm(res, axis=1)
        n_samples = len(residual_magnitude)

        # Welch's method for PSD estimation
        # nperseg chosen for reasonable frequency resolution with limited data
        nperseg = min(128, n_samples // 4)
        if nperseg < 16:
            nperseg = min(n_samples, 16)

        try:
            frequencies, psd = signal.welch(
                residual_magnitude,
                fs=sampling_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
        except Exception as e:
            print(f"  WARNING: Spectral analysis failed for {source}: {e}")
            spectral_results[source] = {
                'error': str(e),
                'dominant_periods': [],
                'period_interpretation': {}
            }
            continue

        # Convert to periods (days), avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            periods = np.where(frequencies > 1e-10, 1 / frequencies, np.inf)

        # Find dominant periods (local maxima in PSD)
        try:
            peak_indices, properties = signal.find_peaks(psd, height=np.median(psd))
            dominant = sorted(
                [(float(periods[i]), float(psd[i])) for i in peak_indices if np.isfinite(periods[i])],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        except Exception:
            dominant = []

        # Match to known periodicities
        interpretations = {}
        for period, power in dominant:
            matched = False
            for known_period, label in KNOWN_PERIODS.items():
                if 0.8 * known_period <= period <= 1.2 * known_period:
                    interpretations[f'{period:.1f}_days'] = label
                    matched = True
                    break
            if not matched:
                interpretations[f'{period:.1f}_days'] = 'unknown_periodicity'

        # Compute spectral slope (1/f characteristics)
        # Linear regression of log(PSD) vs log(freq) for frequencies > 0
        valid_mask = (frequencies > 0.01) & np.isfinite(psd) & (psd > 0)
        if valid_mask.sum() > 5:
            log_freq = np.log10(frequencies[valid_mask])
            log_psd = np.log10(psd[valid_mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_freq, log_psd)
            spectral_slope = float(slope)
            spectral_slope_r2 = float(r_value ** 2)
        else:
            spectral_slope = None
            spectral_slope_r2 = None

        spectral_results[source] = {
            'frequencies': frequencies.tolist(),
            'psd': psd.tolist(),
            'dominant_periods': dominant,
            'period_interpretation': interpretations,
            'spectral_slope': spectral_slope,
            'spectral_slope_r2': spectral_slope_r2,
            'interpretation': interpret_spectrum(dominant, spectral_slope)
        }

    return spectral_results


def interpret_spectrum(dominant_periods: List[Tuple[float, float]], spectral_slope: Optional[float]) -> List[str]:
    """Generate interpretation of spectral analysis."""
    interpretations = []

    if not dominant_periods:
        interpretations.append('no_clear_periodicity')
    else:
        # Check for common periodicities
        periods = [p[0] for p in dominant_periods]
        if any(6 <= p <= 8 for p in periods):
            interpretations.append('weekly_component')
        if any(13 <= p <= 16 for p in periods):
            interpretations.append('biweekly_component')
        if any(25 <= p <= 35 for p in periods):
            interpretations.append('monthly_component')

    if spectral_slope is not None:
        if spectral_slope < -1.5:
            interpretations.append('strong_1/f_noise_long_memory')
        elif spectral_slope < -0.5:
            interpretations.append('moderate_1/f_characteristics')
        elif spectral_slope > -0.5:
            interpretations.append('approximately_white_noise')

    return interpretations


def run_temporal_analysis(
    residual_results: ResidualExtractionResults,
    max_lag: int = 60,
    use_masked: bool = True
) -> TemporalAnalysisResults:
    """
    Run complete temporal structure analysis on residuals.

    Args:
        residual_results: Results from Stage 1 (residual extraction)
        max_lag: Maximum lag for correlation analysis
        use_masked: If True, analyze masked reconstruction residuals (recommended)
                   If False, analyze full reconstruction residuals

    Returns:
        TemporalAnalysisResults containing all temporal analyses
    """
    print("\n" + "=" * 70)
    print("STAGE 2: TEMPORAL STRUCTURE ANALYSIS")
    print("=" * 70)

    residuals = residual_results.masked_residuals if use_masked else residual_results.full_residuals
    residual_type = "masked" if use_masked else "full"
    print(f"\nAnalyzing {residual_type} reconstruction residuals")
    print(f"Sources: {list(residuals.keys())}")
    print(f"Max lag: {max_lag} days")

    # 1. Autocorrelation Analysis
    print("\n--- Computing Autocorrelation Functions ---")
    acf_results = compute_residual_autocorrelation(residuals, max_lag=max_lag)

    for source, result in acf_results.items():
        print(f"  {source}:")
        print(f"    Significant lags: {result['n_significant_lags']}")
        print(f"    Decay lag (ACF < 0.5): {result['decay_lag']}")
        print(f"    Interpretation: {', '.join(result['interpretation'])}")

    # 2. Cross-Residual Correlation
    print("\n--- Computing Cross-Residual Correlations ---")
    xcorr_results = compute_cross_residual_correlation(residuals, max_lag=max_lag)

    for (src_i, src_j), result in xcorr_results.items():
        print(f"  {src_i} ↔ {src_j}:")
        print(f"    Peak: r={result['peak_correlation']:.3f} at lag={result['peak_lag']}")
        print(f"    Zero-lag: r={result['zero_lag_correlation']:.3f}")
        print(f"    Interpretation: {', '.join(result['interpretation'])}")

    # 3. Spectral Analysis
    print("\n--- Computing Power Spectral Density ---")
    spectral_results = compute_residual_spectra(residuals)

    for source, result in spectral_results.items():
        if 'error' in result:
            print(f"  {source}: ERROR - {result['error']}")
            continue
        print(f"  {source}:")
        if result['dominant_periods']:
            top_periods = result['dominant_periods'][:3]
            periods_str = ', '.join([f"{p:.1f}d" for p, _ in top_periods])
            print(f"    Dominant periods: {periods_str}")
        if result['spectral_slope'] is not None:
            print(f"    Spectral slope: {result['spectral_slope']:.2f} (R²={result['spectral_slope_r2']:.2f})")
        print(f"    Interpretation: {', '.join(result['interpretation'])}")

    # Compile metadata
    metadata = {
        'analysis_timestamp': datetime.now().isoformat(),
        'residual_type': residual_type,
        'max_lag': max_lag,
        'n_samples': len(residual_results.timestamps),
        'sources': list(residuals.keys())
    }

    return TemporalAnalysisResults(
        acf_results=acf_results,
        cross_residual_correlations=xcorr_results,
        spectral_results=spectral_results,
        metadata=metadata
    )


def generate_temporal_plots(
    results: TemporalAnalysisResults,
    output_dir: Path
):
    """Generate visualization plots for temporal analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(results.acf_results.keys())
    n_sources = len(sources)

    # Color palette
    colors = {
        'deepstate': '#2ecc71',
        'equipment': '#3498db',
        'firms': '#e74c3c',
        'ucdp': '#9b59b6'
    }

    # 1. ACF Grid Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, source in enumerate(sources):
        ax = axes[idx]
        acf_data = results.acf_results[source]
        lags = acf_data['lags']
        acf_values = acf_data['acf']

        color = colors.get(source, '#34495e')
        ax.bar(lags, acf_values, color=color, alpha=0.7, width=0.8)

        # Confidence interval
        ci_upper = acf_data['confidence_interval_upper']
        if isinstance(ci_upper, list) and len(ci_upper) > 0:
            ci_val = ci_upper[1] if len(ci_upper) > 1 else 0.1
            ax.axhline(y=ci_val, color='red', linestyle='--', alpha=0.5, label='95% CI')
            ax.axhline(y=-ci_val, color='red', linestyle='--', alpha=0.5)

        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('ACF')
        ax.set_title(f'{source.upper()} Residual Autocorrelation')
        ax.set_xlim(-1, max(lags) + 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_acf_grid.png', dpi=150)
    plt.close()
    print(f"  Saved: residual_acf_grid.png")

    # 2. Cross-Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create matrix of zero-lag correlations
    corr_matrix = np.zeros((n_sources, n_sources))
    for i, src_i in enumerate(sources):
        corr_matrix[i, i] = 1.0  # Self-correlation
        for j, src_j in enumerate(sources):
            if i < j:
                key = (src_i, src_j)
                if key in results.cross_residual_correlations:
                    corr = results.cross_residual_correlations[key]['zero_lag_correlation']
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_sources))
    ax.set_yticks(range(n_sources))
    ax.set_xticklabels([s.upper() for s in sources], rotation=45, ha='right')
    ax.set_yticklabels([s.upper() for s in sources])

    # Add correlation values as text
    for i in range(n_sources):
        for j in range(n_sources):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=10,
                          color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    plt.colorbar(im, label='Correlation')
    ax.set_title('Cross-Residual Correlation (Zero Lag)')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_residual_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: cross_residual_heatmap.png")

    # 3. Spectral Density Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, source in enumerate(sources):
        ax = axes[idx]
        spec_data = results.spectral_results[source]

        if 'error' in spec_data:
            ax.text(0.5, 0.5, f'Error: {spec_data["error"]}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        freqs = np.array(spec_data['frequencies'])
        psd = np.array(spec_data['psd'])

        color = colors.get(source, '#34495e')
        ax.semilogy(freqs, psd, color=color, linewidth=1.5)

        # Mark dominant periods
        for period, power in spec_data['dominant_periods'][:3]:
            if period < 100:  # Only show periods less than 100 days
                freq = 1.0 / period
                ax.axvline(x=freq, color='gray', linestyle='--', alpha=0.5)
                ax.text(freq, ax.get_ylim()[1], f'{period:.0f}d',
                       ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Frequency (cycles/day)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'{source.upper()} Residual Spectrum')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_density.png', dpi=150)
    plt.close()
    print(f"  Saved: spectral_density.png")

    # 4. Cross-correlation lag plots (detailed)
    n_pairs = len(results.cross_residual_correlations)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ((src_i, src_j), xcorr_data) in enumerate(results.cross_residual_correlations.items()):
        if idx >= 6:
            break
        ax = axes[idx]

        lags = np.array(xcorr_data['lags'])
        corr = np.array(xcorr_data['correlation'])

        ax.plot(lags, corr, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Mark peak
        peak_lag = xcorr_data['peak_lag']
        peak_corr = xcorr_data['peak_correlation']
        ax.scatter([peak_lag], [peak_corr], color='red', s=100, zorder=5)
        ax.annotate(f'Peak: {peak_corr:.2f}\n@ lag {peak_lag}',
                   xy=(peak_lag, peak_corr), xytext=(10, 10),
                   textcoords='offset points', fontsize=9)

        ax.set_xlabel('Lag (days)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'{src_i.upper()} ↔ {src_j.upper()}')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(results.cross_residual_correlations), 6):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_correlation_lags.png', dpi=150)
    plt.close()
    print(f"  Saved: cross_correlation_lags.png")


def main():
    """Run Stage 2: Temporal Structure Analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 2: Temporal Structure Analysis')
    parser.add_argument('--input-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'results'),
                       help='Directory containing Stage 1 results')
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'results'),
                       help='Output directory for results')
    parser.add_argument('--figures-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'figures'),
                       help='Output directory for figures')
    parser.add_argument('--max-lag', type=int, default=60,
                       help='Maximum lag for correlation analysis')
    parser.add_argument('--use-full', action='store_true',
                       help='Use full reconstruction residuals instead of masked')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 2: TEMPORAL STRUCTURE ANALYSIS")
    print("=" * 70)
    print("\nAnalyzing temporal patterns in model residuals.")
    print("Autocorrelation, cross-correlation, and spectral structure")
    print("reveal properties of missing variables.\n")

    # Load Stage 1 results
    print("Loading residuals from Stage 1...")
    input_dir = Path(args.input_dir)
    residual_results = ResidualExtractionResults.load(input_dir)
    print(f"  Loaded {residual_results.metadata.get('n_samples', 'unknown')} samples")

    # Run temporal analysis
    results = run_temporal_analysis(
        residual_results,
        max_lag=args.max_lag,
        use_masked=not args.use_full
    )

    # Save results
    output_dir = Path(args.output_dir)
    results.save(output_dir)

    # Generate plots
    if not args.skip_plots:
        print("\n--- Generating Plots ---")
        figures_dir = Path(args.figures_dir)
        generate_temporal_plots(results, figures_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 70)

    # Summary interpretation
    print("\n--- Key Findings ---")

    # ACF summary
    persistent_sources = [
        s for s, r in results.acf_results.items()
        if r['n_significant_lags'] > 5
    ]
    if persistent_sources:
        print(f"\nPersistent autocorrelation in: {', '.join(persistent_sources)}")
        print("  → Suggests slowly-varying omitted factors")

    # Cross-correlation summary
    strong_pairs = [
        (k, v) for k, v in results.cross_residual_correlations.items()
        if abs(v['peak_correlation']) > 0.3
    ]
    if strong_pairs:
        print(f"\nStrong cross-residual correlations:")
        for (src_i, src_j), data in strong_pairs:
            print(f"  {src_i} ↔ {src_j}: r={data['peak_correlation']:.2f} at lag={data['peak_lag']}")
        print("  → Suggests shared omitted factors affecting multiple sources")

    # Spectral summary
    periodic_sources = [
        s for s, r in results.spectral_results.items()
        if 'error' not in r and r['dominant_periods']
    ]
    if periodic_sources:
        print(f"\nPeriodic components detected in: {', '.join(periodic_sources)}")
        print("  → Missing variables may have weekly/monthly cycles")

    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {args.figures_dir}")
    print("\nNext step: Run Stage 3 (Latent Factor Extraction)")


if __name__ == '__main__':
    main()
