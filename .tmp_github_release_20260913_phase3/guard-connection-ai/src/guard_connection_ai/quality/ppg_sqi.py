from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from guard_connection_ai.streaming.pulse_tracker import PulseEvent


@dataclass(frozen=True)
class SignalQualityDecision:
    score: float
    technical_score: float
    coverage: float
    plausible_interval_fraction: float
    accepted: bool
    rejection_reason: str | None


class StatefulPPGQuality:
    """Rolling PPG coverage and pulse-plausibility quality gate."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        window_seconds: float = 10.0,
        minimum_coverage: float = 0.8,
        minimum_score: float = 0.65,
    ) -> None:
        if sample_rate_hz <= 0 or window_seconds <= 0:
            raise ValueError("sample rate and quality window must be positive.")
        if not 0 <= minimum_coverage <= 1 or not 0 <= minimum_score <= 1:
            raise ValueError("quality thresholds must be within [0, 1].")
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = max(round(window_seconds * sample_rate_hz), 1)
        self.minimum_coverage = minimum_coverage
        self.minimum_score = minimum_score
        self.reset()

    def reset(self) -> None:
        self._valid: deque[bool] = deque(maxlen=self.window_samples)
        self._event_indices: deque[int] = deque()
        self._latest_sample = -1

    def update(
        self, valid_mask: np.ndarray, events: list[PulseEvent]
    ) -> SignalQualityDecision:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim != 1 or valid.size == 0:
            raise ValueError("valid_mask must be a non-empty 1D array.")
        self._valid.extend(bool(value) for value in valid)
        self._latest_sample += valid.size
        self._event_indices.extend(event.sample_index for event in events)
        earliest = self._latest_sample - self.window_samples + 1
        while self._event_indices and self._event_indices[0] < earliest:
            self._event_indices.popleft()
        coverage = float(np.mean(self._valid))
        intervals = np.diff(np.asarray(self._event_indices, dtype=np.int64))
        if intervals.size:
            plausible = (intervals >= 0.25 * self.sample_rate_hz) & (
                intervals <= 2.0 * self.sample_rate_hz
            )
            plausible_fraction = float(np.mean(plausible))
        else:
            plausible_fraction = 0.0
        technical_score = coverage
        reason = None
        if len(self._valid) < self.window_samples:
            reason = "quality_warmup"
        elif coverage < self.minimum_coverage:
            reason = "insufficient_valid_samples"
        elif len(self._event_indices) < 2:
            reason = "insufficient_pulse_events"
        elif technical_score < self.minimum_score:
            reason = "insufficient_technical_quality"
        return SignalQualityDecision(
            technical_score,
            technical_score,
            coverage,
            plausible_fraction,
            reason is None,
            reason,
        )
