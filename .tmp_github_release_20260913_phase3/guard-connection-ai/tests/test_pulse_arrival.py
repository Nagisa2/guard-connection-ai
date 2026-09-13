from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.metrics.pulse_arrival import (
    estimate_pulse_arrival_delay,
    pulse_arrival_delays_from_peaks,
)


def test_pulse_arrival_matches_first_following_ppg_peak_once() -> None:
    delays = pulse_arrival_delays_from_peaks(
        np.asarray([100, 200, 300, 400]),
        np.asarray([125, 226, 327, 700]),
        sample_rate_hz=100,
        minimum_seconds=0.08,
        maximum_seconds=0.5,
    )

    assert delays.tolist() == pytest.approx([0.25, 0.26, 0.27])


def test_pulse_arrival_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="invalid sample rate"):
        pulse_arrival_delays_from_peaks(
            np.asarray([1]),
            np.asarray([2]),
            sample_rate_hz=0,
        )


def test_pulse_arrival_estimator_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="upper_quantile"):
        estimate_pulse_arrival_delay([], upper_quantile=1.0)


def test_pulse_arrival_estimator_rejects_low_recording_coverage(monkeypatch) -> None:
    peaks = iter(
        (
            np.arange(10, 110, 10),
            np.arange(12, 112, 10),
            np.array([]),
            np.array([]),
        )
    )
    monkeypatch.setattr(
        "guard_connection_ai.metrics.pulse_arrival.detect_rhythm_peaks",
        lambda *_args, **_kwargs: next(peaks),
    )
    paired = [
        (np.zeros(120), np.zeros(120), 10.0),
        (np.zeros(120), np.zeros(120), 10.0),
    ]

    with pytest.raises(ValueError, match="recording coverage"):
        estimate_pulse_arrival_delay(
            paired,
            minimum_beats_per_recording=5,
            minimum_recording_coverage=0.75,
        )
