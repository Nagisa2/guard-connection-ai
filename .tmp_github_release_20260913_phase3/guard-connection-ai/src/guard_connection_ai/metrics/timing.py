from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guard_connection_ai.streaming.timing_decoder import StatefulTimingDecoder


@dataclass(frozen=True)
class EventTimingMetrics:
    precision: float
    recall: float
    f1: float
    timing_mae_seconds: float
    rr_mae_seconds: float
    true_events: int
    predicted_events: int
    matched_events: int


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    metrics: EventTimingMetrics


def evaluate_event_timing(
    reference_indices: np.ndarray,
    predicted_indices: np.ndarray,
    *,
    sample_rate_hz: float,
    tolerance_seconds: float = 0.1,
) -> EventTimingMetrics:
    """Greedily match event times without allowing duplicate matches."""

    reference = np.sort(np.asarray(reference_indices, dtype=np.int64))
    predicted = np.sort(np.asarray(predicted_indices, dtype=np.int64))
    if reference.ndim != 1 or predicted.ndim != 1 or sample_rate_hz <= 0:
        raise ValueError("event arrays must be 1D and sample_rate_hz positive.")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")
    tolerance = round(tolerance_seconds * sample_rate_hz)
    matches: list[tuple[int, int]] = []
    used: set[int] = set()
    for reference_value in reference:
        candidates = np.flatnonzero(np.abs(predicted - reference_value) <= tolerance)
        candidates = np.asarray(
            [index for index in candidates if int(index) not in used]
        )
        if candidates.size:
            selected = int(
                candidates[np.argmin(np.abs(predicted[candidates] - reference_value))]
            )
            used.add(selected)
            matches.append((int(reference_value), int(predicted[selected])))
    matched = len(matches)
    precision = matched / predicted.size if predicted.size else 0.0
    recall = matched / reference.size if reference.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing_mae = (
        float(np.mean([abs(left - right) for left, right in matches]) / sample_rate_hz)
        if matches
        else float("nan")
    )
    matched_reference = np.asarray([left for left, _ in matches], dtype=np.float64)
    matched_prediction = np.asarray([right for _, right in matches], dtype=np.float64)
    rr_mae = (
        float(
            np.mean(np.abs(np.diff(matched_reference) - np.diff(matched_prediction)))
            / sample_rate_hz
        )
        if matched >= 2
        else float("nan")
    )
    return EventTimingMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        timing_mae_seconds=timing_mae,
        rr_mae_seconds=rr_mae,
        true_events=int(reference.size),
        predicted_events=int(predicted.size),
        matched_events=matched,
    )


def _aggregate_event_metrics(metrics: list[EventTimingMetrics]) -> EventTimingMetrics:
    true_events = sum(value.true_events for value in metrics)
    predicted_events = sum(value.predicted_events for value in metrics)
    matched_events = sum(value.matched_events for value in metrics)
    precision = matched_events / predicted_events if predicted_events else 0.0
    recall = matched_events / true_events if true_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def weighted(name: str, weight_name: str) -> float:
        values = []
        weights = []
        for metric in metrics:
            value = getattr(metric, name)
            weight = getattr(metric, weight_name)
            if np.isfinite(value) and weight > 0:
                values.append(value)
                weights.append(weight)
        return float(np.average(values, weights=weights)) if weights else float("nan")

    return EventTimingMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        timing_mae_seconds=weighted("timing_mae_seconds", "matched_events"),
        rr_mae_seconds=weighted("rr_mae_seconds", "matched_events"),
        true_events=true_events,
        predicted_events=predicted_events,
        matched_events=matched_events,
    )


def evaluate_probability_windows(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
    *,
    sample_rate_hz: float,
    threshold: float,
    refractory_seconds: float = 0.25,
    tolerance_seconds: float = 0.1,
) -> EventTimingMetrics:
    """Evaluate independent windows with the same decoder used at runtime."""

    if not (len(probabilities) == len(targets) == len(valid_masks)):
        raise ValueError("probability, target, and mask window counts must match.")
    window_metrics = []
    for probability, target, valid in zip(
        probabilities, targets, valid_masks, strict=True
    ):
        decoder = StatefulTimingDecoder(
            sample_rate_hz=sample_rate_hz,
            threshold=threshold,
            refractory_seconds=refractory_seconds,
        )
        events = decoder.process(probability, valid)
        reference = np.flatnonzero((target >= 1.0 - 1e-6) & valid)
        predicted = np.asarray([event.sample_index for event in events], dtype=np.int64)
        window_metrics.append(
            evaluate_event_timing(
                reference,
                predicted,
                sample_rate_hz=sample_rate_hz,
                tolerance_seconds=tolerance_seconds,
            )
        )
    return _aggregate_event_metrics(window_metrics)


def calibrate_event_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
    *,
    sample_rate_hz: float,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    refractory_seconds: float = 0.25,
    tolerance_seconds: float = 0.1,
) -> ThresholdCalibration:
    """Select an event threshold on validation patients only."""

    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError(
            "thresholds must contain values strictly between zero and one."
        )
    candidates = [
        ThresholdCalibration(
            threshold=threshold,
            metrics=evaluate_probability_windows(
                probabilities,
                targets,
                valid_masks,
                sample_rate_hz=sample_rate_hz,
                threshold=threshold,
                refractory_seconds=refractory_seconds,
                tolerance_seconds=tolerance_seconds,
            ),
        )
        for threshold in thresholds
    ]
    return max(candidates, key=lambda value: (value.metrics.f1, -value.threshold))
