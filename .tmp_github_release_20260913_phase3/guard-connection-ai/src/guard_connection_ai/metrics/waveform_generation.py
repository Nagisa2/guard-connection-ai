from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy import signal as scipy_signal


def match_peak_indices(
    reference_peaks: Sequence[int] | np.ndarray,
    predicted_peaks: Sequence[int] | np.ndarray,
    *,
    tolerance_samples: int,
) -> list[tuple[int, int]]:
    """Greedily match ordered peaks one-to-one within a declared tolerance."""

    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative.")
    reference = np.asarray(reference_peaks, dtype=np.int64)
    predicted = np.asarray(predicted_peaks, dtype=np.int64)
    if reference.ndim != 1 or predicted.ndim != 1:
        raise ValueError("peak arrays must be one-dimensional.")
    matches: list[tuple[int, int]] = []
    reference_index = 0
    predicted_index = 0
    while reference_index < reference.size and predicted_index < predicted.size:
        difference = int(predicted[predicted_index] - reference[reference_index])
        if abs(difference) <= tolerance_samples:
            matches.append((reference_index, predicted_index))
            reference_index += 1
            predicted_index += 1
        elif difference < 0:
            predicted_index += 1
        else:
            reference_index += 1
    return matches


def peak_detection_metrics(
    reference_peaks: Sequence[int] | np.ndarray,
    predicted_peaks: Sequence[int] | np.ndarray,
    *,
    tolerance_samples: int,
    sample_rate_hz: float,
) -> dict[str, float]:
    reference = np.asarray(reference_peaks, dtype=np.int64)
    predicted = np.asarray(predicted_peaks, dtype=np.int64)
    matches = match_peak_indices(reference, predicted, tolerance_samples=tolerance_samples)
    true_positive = len(matches)
    precision = true_positive / max(predicted.size, 1)
    recall = true_positive / max(reference.size, 1)
    f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
    timing_errors = np.asarray(
        [predicted[predicted_index] - reference[reference_index] for reference_index, predicted_index in matches]
    )
    return {
        "r_peak_precision": float(precision),
        "r_peak_recall": float(recall),
        "r_peak_f1": float(f1),
        "r_peak_timing_mae_ms": (
            float(np.mean(np.abs(timing_errors)) / sample_rate_hz * 1000)
            if timing_errors.size
            else np.nan
        ),
    }


def rr_interval_metrics(
    reference_peaks: Sequence[int] | np.ndarray,
    predicted_peaks: Sequence[int] | np.ndarray,
    *,
    tolerance_samples: int,
    sample_rate_hz: float,
) -> dict[str, float]:
    reference = np.asarray(reference_peaks, dtype=np.int64)
    predicted = np.asarray(predicted_peaks, dtype=np.int64)
    matches = match_peak_indices(reference, predicted, tolerance_samples=tolerance_samples)
    rr_errors = []
    for (left_reference, left_predicted), (right_reference, right_predicted) in pairwise(matches):
        if right_reference != left_reference + 1 or right_predicted != left_predicted + 1:
            continue
        reference_rr = reference[right_reference] - reference[left_reference]
        predicted_rr = predicted[right_predicted] - predicted[left_predicted]
        rr_errors.append((predicted_rr - reference_rr) / sample_rate_hz * 1000)
    errors = np.asarray(rr_errors, dtype=np.float64)
    return {
        "rr_mae_ms": float(np.mean(np.abs(errors))) if errors.size else np.nan,
        "rr_rmse_ms": float(np.sqrt(np.mean(errors * errors))) if errors.size else np.nan,
        "matched_rr_count": float(errors.size),
    }


def _qrs_width_proxy_seconds(
    waveform: np.ndarray,
    peaks: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    absolute = np.abs(waveform - np.median(waveform))
    candidate_peaks, properties = scipy_signal.find_peaks(
        absolute,
        distance=max(round(0.02 * sample_rate_hz), 1),
        prominence=max(0.05 * float(np.std(absolute)), 1e-8),
    )
    if not candidate_peaks.size:
        return np.asarray([], dtype=np.float64)
    positive_prominence = properties["prominences"] > 0
    candidate_peaks = candidate_peaks[positive_prominence]
    localized = []
    radius = max(round(0.12 * sample_rate_hz), 1)
    for peak in peaks:
        candidates = candidate_peaks[np.abs(candidate_peaks - peak) <= radius]
        if not candidates.size:
            continue
        localized.append(int(candidates[np.argmax(absolute[candidates])]))
    if not localized:
        return np.asarray([], dtype=np.float64)
    widths, _, _, _ = scipy_signal.peak_widths(
        absolute,
        np.unique(localized),
        rel_height=0.5,
    )
    plausible = widths[(widths >= 0.02 * sample_rate_hz) & (widths <= 0.2 * sample_rate_hz)]
    return plausible / sample_rate_hz


def waveform_similarity_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    reference_values = np.asarray(reference, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    if reference_values.shape != prediction_values.shape or reference_values.ndim != 1:
        raise ValueError("reference and prediction must share one-dimensional shape.")
    mask = (
        np.ones(reference_values.size, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    mask &= np.isfinite(reference_values) & np.isfinite(prediction_values)
    if mask.shape != reference_values.shape or np.sum(mask) < 2:
        raise ValueError("At least two valid paired samples are required.")
    reference_valid = reference_values[mask]
    prediction_valid = prediction_values[mask]
    errors = prediction_valid - reference_valid
    correlation = (
        float(np.corrcoef(reference_valid, prediction_valid)[0, 1])
        if np.std(reference_valid) > 1e-8 and np.std(prediction_valid) > 1e-8
        else np.nan
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "pearson_correlation": correlation,
    }


def qrs_width_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    reference_peaks: np.ndarray,
    predicted_peaks: np.ndarray,
    *,
    sample_rate_hz: float,
) -> dict[str, float]:
    reference_widths = _qrs_width_proxy_seconds(reference, reference_peaks, sample_rate_hz)
    predicted_widths = _qrs_width_proxy_seconds(prediction, predicted_peaks, sample_rate_hz)
    return {
        "reference_qrs_width_proxy_median_ms": (
            float(np.median(reference_widths) * 1000) if reference_widths.size else np.nan
        ),
        "predicted_qrs_width_proxy_median_ms": (
            float(np.median(predicted_widths) * 1000) if predicted_widths.size else np.nan
        ),
        "qrs_width_proxy_median_absolute_error_ms": (
            float(abs(np.median(reference_widths) - np.median(predicted_widths)) * 1000)
            if reference_widths.size and predicted_widths.size
            else np.nan
        ),
    }
