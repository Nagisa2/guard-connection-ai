from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guard_connection_ai.features.rhythm import detect_rhythm_peaks


@dataclass(frozen=True)
class PulseArrivalEstimate:
    median_seconds: float
    upper_quantile_seconds: float
    interquartile_range_seconds: float
    matched_beats: int
    recording_estimates: int
    total_recordings: int
    recording_coverage: float


def pulse_arrival_delays_from_peaks(
    ecg_peaks: np.ndarray,
    ppg_peaks: np.ndarray,
    *,
    sample_rate_hz: float,
    minimum_seconds: float = 0.08,
    maximum_seconds: float = 0.5,
) -> np.ndarray:
    """Match each ECG peak to the first physiologically plausible following PPG peak."""

    if sample_rate_hz <= 0 or not 0 <= minimum_seconds < maximum_seconds:
        raise ValueError("invalid sample rate or pulse-arrival bounds.")
    ecg = np.asarray(ecg_peaks, dtype=np.int64)
    ppg = np.asarray(ppg_peaks, dtype=np.int64)
    if ecg.ndim != 1 or ppg.ndim != 1:
        raise ValueError("peak arrays must be one-dimensional.")
    minimum_samples = round(minimum_seconds * sample_rate_hz)
    maximum_samples = round(maximum_seconds * sample_rate_hz)
    delays = []
    previous_ppg_index = -1
    for ecg_peak in ecg:
        candidate_index = int(np.searchsorted(ppg, ecg_peak + minimum_samples, side="left"))
        if candidate_index <= previous_ppg_index or candidate_index >= ppg.size:
            continue
        delay = int(ppg[candidate_index] - ecg_peak)
        if delay <= maximum_samples:
            delays.append(delay / sample_rate_hz)
            previous_ppg_index = candidate_index
    return np.asarray(delays, dtype=np.float64)


def estimate_pulse_arrival_delay(
    paired_signals: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    minimum_seconds: float = 0.08,
    maximum_seconds: float = 0.5,
    minimum_beats_per_recording: int = 20,
    upper_quantile: float = 0.95,
    minimum_recording_coverage: float = 0.0,
) -> PulseArrivalEstimate:
    """Estimate a fixed delay from training recordings without using validation/test data."""

    if not 0.5 <= upper_quantile < 1:
        raise ValueError("upper_quantile must be between 0.5 inclusive and 1 exclusive.")
    if not 0 <= minimum_recording_coverage <= 1:
        raise ValueError("minimum_recording_coverage must be between zero and one.")
    recording_medians = []
    recording_upper_quantiles = []
    matched_beats = 0
    for ppg, ecg, sample_rate_hz in paired_signals:
        ecg_peaks = detect_rhythm_peaks(
            ecg, sample_rate_hz=sample_rate_hz, modality="ecg"
        )
        ppg_peaks = detect_rhythm_peaks(
            ppg, sample_rate_hz=sample_rate_hz, modality="ppg"
        )
        delays = pulse_arrival_delays_from_peaks(
            ecg_peaks,
            ppg_peaks,
            sample_rate_hz=sample_rate_hz,
            minimum_seconds=minimum_seconds,
            maximum_seconds=maximum_seconds,
        )
        if delays.size >= minimum_beats_per_recording:
            recording_medians.append(float(np.median(delays)))
            recording_upper_quantiles.append(float(np.quantile(delays, upper_quantile)))
            matched_beats += int(delays.size)
    if not recording_medians:
        raise ValueError("No training recording produced enough pulse-arrival matches.")
    total_recordings = len(paired_signals)
    recording_coverage = len(recording_medians) / max(total_recordings, 1)
    if recording_coverage < minimum_recording_coverage:
        raise ValueError(
            "Pulse-arrival recording coverage is below the configured minimum: "
            f"{recording_coverage:.3f} < {minimum_recording_coverage:.3f}."
        )
    values = np.asarray(recording_medians)
    return PulseArrivalEstimate(
        median_seconds=float(np.median(values)),
        upper_quantile_seconds=float(np.median(recording_upper_quantiles)),
        interquartile_range_seconds=float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        matched_beats=matched_beats,
        recording_estimates=int(values.size),
        total_recordings=total_recordings,
        recording_coverage=recording_coverage,
    )
