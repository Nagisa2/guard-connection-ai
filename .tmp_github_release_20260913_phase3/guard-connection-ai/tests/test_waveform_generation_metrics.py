from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.metrics.waveform_generation import (
    match_peak_indices,
    peak_detection_metrics,
    qrs_width_metrics,
    rr_interval_metrics,
    waveform_similarity_metrics,
)


def test_peak_matching_is_one_to_one_and_reports_timing() -> None:
    reference = np.asarray([100, 200, 300, 400])
    predicted = np.asarray([98, 203, 250, 399])

    matches = match_peak_indices(reference, predicted, tolerance_samples=5)
    metrics = peak_detection_metrics(
        reference,
        predicted,
        tolerance_samples=5,
        sample_rate_hz=100,
    )

    assert matches == [(0, 0), (1, 1), (3, 3)]
    assert metrics["r_peak_precision"] == pytest.approx(0.75)
    assert metrics["r_peak_recall"] == pytest.approx(0.75)
    assert metrics["r_peak_timing_mae_ms"] == pytest.approx(20.0)


def test_rr_metrics_only_compare_consecutive_matched_beats() -> None:
    metrics = rr_interval_metrics(
        [100, 200, 300, 400],
        [102, 203, 304, 450],
        tolerance_samples=5,
        sample_rate_hz=100,
    )

    assert metrics["matched_rr_count"] == 2
    assert metrics["rr_mae_ms"] == pytest.approx(10.0)


def test_waveform_metrics_respect_valid_mask() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 100.0])
    prediction = np.asarray([0.0, 1.0, 1.0, -100.0])
    metrics = waveform_similarity_metrics(
        reference,
        prediction,
        valid_mask=np.asarray([1, 1, 1, 0]),
    )

    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))


def test_qrs_width_proxy_is_explicitly_bounded() -> None:
    sample_rate = 250
    samples = np.arange(1000)
    reference = np.exp(-0.5 * ((samples - 300) / 10) ** 2)
    prediction = np.exp(-0.5 * ((samples - 302) / 12) ** 2)
    metrics = qrs_width_metrics(
        reference,
        prediction,
        np.asarray([300]),
        np.asarray([302]),
        sample_rate_hz=sample_rate,
    )

    assert 80 < metrics["reference_qrs_width_proxy_median_ms"] < 110
    assert (
        metrics["predicted_qrs_width_proxy_median_ms"]
        > metrics["reference_qrs_width_proxy_median_ms"]
    )
