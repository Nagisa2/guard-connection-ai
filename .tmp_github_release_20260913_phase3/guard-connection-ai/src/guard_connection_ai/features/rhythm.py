from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

FEATURE_NAMES = (
    "valid_fraction",
    "peak_count",
    "peak_prominence_median",
    "rr_retained_fraction",
    "heart_rate_median",
    "rr_mean",
    "rr_std",
    "rr_cv",
    "rr_mad",
    "rr_iqr",
    "rmssd",
    "pnn50",
    "successive_rr_correlation",
    "sample_entropy",
)


@dataclass(frozen=True)
class RhythmFeatures:
    values: np.ndarray
    peak_indices: np.ndarray
    is_usable: bool
    rejection_reason: str | None


def _robust_standardize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        scale = float(np.std(values))
    if scale < 1e-8:
        return np.zeros_like(values)
    return (values - median) / scale


def _fill_missing(signal: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array.")
    finite = np.isfinite(values)
    valid_fraction = float(np.mean(finite))
    if not np.any(finite):
        return np.zeros_like(values), valid_fraction
    indices = np.arange(values.size)
    return np.interp(indices, indices[finite], values[finite]), valid_fraction


def _causal_bandpass(signal: np.ndarray, sample_rate_hz: float, modality: str) -> np.ndarray:
    low_hz, high_hz = (5.0, 25.0) if modality == "ecg" else (0.5, 8.0)
    nyquist = sample_rate_hz / 2.0
    if high_hz >= nyquist:
        high_hz = 0.9 * nyquist
    if low_hz >= high_hz:
        raise ValueError("sample rate is too low for the requested bandpass.")
    sos = scipy_signal.butter(
        3, (low_hz, high_hz), btype="bandpass", fs=sample_rate_hz, output="sos"
    )
    initial_state = scipy_signal.sosfilt_zi(sos) * signal[0]
    filtered, _ = scipy_signal.sosfilt(sos, signal, zi=initial_state)
    return filtered


def _detect_peaks(signal: np.ndarray, sample_rate_hz: float, modality: str) -> tuple[np.ndarray, np.ndarray]:
    filtered = _causal_bandpass(signal, sample_rate_hz, modality)
    if modality == "ecg":
        derivative = np.diff(filtered, prepend=filtered[0])
        evidence = derivative * derivative
        integration_samples = max(round(0.12 * sample_rate_hz), 1)
        evidence = scipy_signal.lfilter(
            np.ones(integration_samples) / integration_samples, [1.0], evidence
        )
        evidence = _robust_standardize(evidence)
        height = max(float(np.quantile(evidence, 0.75)), 0.2)
        prominence = max(0.25 * float(np.std(evidence)), 0.1)
        distance = max(round(0.25 * sample_rate_hz), 1)
    else:
        evidence = _robust_standardize(filtered)
        height = float(np.quantile(evidence, 0.55))
        prominence = max(0.2 * float(np.std(evidence)), 0.15)
        distance = max(round(0.3 * sample_rate_hz), 1)
    peaks, properties = scipy_signal.find_peaks(
        evidence,
        distance=distance,
        height=height,
        prominence=prominence,
    )
    startup_samples = round(sample_rate_hz)
    keep = peaks >= startup_samples
    return peaks[keep], properties["prominences"][keep]


def detect_rhythm_peaks(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    modality: str,
) -> np.ndarray:
    """Return causally filtered ECG R-peak or PPG pulse-peak candidates."""

    filled, _ = _fill_missing(signal)
    peaks, _ = _detect_peaks(_robust_standardize(filled), sample_rate_hz, modality)
    return peaks


def _sample_entropy(intervals: np.ndarray) -> float:
    if intervals.size < 5:
        return np.nan
    tolerance = 0.2 * float(np.std(intervals))
    if tolerance < 1e-8:
        return 0.0

    def match_count(length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(intervals, length)
        count = 0
        for index in range(len(templates) - 1):
            distances = np.max(np.abs(templates[index + 1 :] - templates[index]), axis=1)
            count += int(np.sum(distances <= tolerance))
        return count

    matches_two = match_count(2)
    matches_three = match_count(3)
    if matches_two == 0:
        return np.nan
    if matches_three == 0:
        return float(-np.log(1.0 / (matches_two + 1.0)))
    return float(-np.log(matches_three / matches_two))


def interval_features(intervals_seconds: np.ndarray) -> np.ndarray:
    """Compute rhythm-irregularity features from physiologically plausible intervals."""

    intervals = np.asarray(intervals_seconds, dtype=np.float64)
    if intervals.ndim != 1 or intervals.size < 2:
        return np.full(10, np.nan, dtype=np.float64)
    differences = np.diff(intervals)
    correlation = (
        float(np.corrcoef(intervals[:-1], intervals[1:])[0, 1])
        if intervals.size >= 4
        and np.std(intervals[:-1]) > 1e-8
        and np.std(intervals[1:]) > 1e-8
        else 0.0
    )
    return np.asarray(
        [
            60.0 / np.median(intervals),
            np.mean(intervals),
            np.std(intervals),
            np.std(intervals) / max(np.mean(intervals), 1e-8),
            np.median(np.abs(intervals - np.median(intervals))),
            np.quantile(intervals, 0.75) - np.quantile(intervals, 0.25),
            np.sqrt(np.mean(differences * differences)),
            np.mean(np.abs(differences) > 0.05),
            correlation,
            _sample_entropy(intervals),
        ],
        dtype=np.float64,
    )


def extract_rhythm_features(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    modality: str,
    minimum_valid_fraction: float = 0.8,
    minimum_peaks: int = 8,
) -> RhythmFeatures:
    """Extract causal-filter rhythm features and an explicit rejection decision."""

    if modality not in {"ecg", "ppg"}:
        raise ValueError("modality must be 'ecg' or 'ppg'.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    filled, valid_fraction = _fill_missing(signal)
    standardized = _robust_standardize(filled)
    peaks, prominences = _detect_peaks(standardized, sample_rate_hz, modality)
    all_intervals = np.diff(peaks) / sample_rate_hz
    plausible = (all_intervals >= 0.25) & (all_intervals <= 2.0)
    intervals = all_intervals[plausible]
    retained_fraction = float(np.mean(plausible)) if all_intervals.size else 0.0
    values = np.concatenate(
        (
            np.asarray(
                [
                    valid_fraction,
                    float(peaks.size),
                    float(np.median(prominences)) if prominences.size else np.nan,
                    retained_fraction,
                ]
            ),
            interval_features(intervals),
        )
    )
    reason = None
    if valid_fraction < minimum_valid_fraction:
        reason = "insufficient_valid_samples"
    elif peaks.size < minimum_peaks or intervals.size < minimum_peaks - 1:
        reason = "insufficient_peaks"
    elif retained_fraction < 0.8:
        reason = "implausible_intervals"
    return RhythmFeatures(values, peaks, reason is None, reason)
