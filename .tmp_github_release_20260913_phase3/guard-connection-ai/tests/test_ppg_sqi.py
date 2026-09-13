from __future__ import annotations

import numpy as np

from guard_connection_ai.quality.ppg_sqi import StatefulPPGQuality
from guard_connection_ai.streaming.pulse_tracker import PulseEvent


def test_technical_sqi_does_not_multiply_interval_plausibility() -> None:
    quality = StatefulPPGQuality(sample_rate_hz=100, window_seconds=1.0)
    decision = quality.update(
        np.ones(100, dtype=bool),
        [PulseEvent(index, 1.0) for index in (0, 20, 80)],
    )

    assert decision.coverage == 1.0
    assert decision.technical_score == 1.0
    assert decision.score == 1.0
    assert decision.plausible_interval_fraction == 0.5
    assert decision.accepted


def test_missingness_still_rejects_technical_quality() -> None:
    quality = StatefulPPGQuality(sample_rate_hz=100, window_seconds=1.0)
    valid = np.ones(100, dtype=bool)
    valid[20:50] = False
    decision = quality.update(valid, [PulseEvent(10, 1.0), PulseEvent(70, 1.0)])

    assert decision.technical_score == 0.7
    assert not decision.accepted
    assert decision.rejection_reason == "insufficient_valid_samples"
