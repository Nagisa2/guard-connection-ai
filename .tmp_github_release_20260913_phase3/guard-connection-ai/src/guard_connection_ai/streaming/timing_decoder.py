from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingEvent:
    sample_index: int
    confidence: float


class StatefulTimingDecoder:
    """Decode causal event probabilities once across arbitrary chunk boundaries."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        threshold: float = 0.5,
        refractory_seconds: float = 0.25,
    ) -> None:
        if sample_rate_hz <= 0 or not 0 < threshold < 1 or refractory_seconds <= 0:
            raise ValueError("invalid decoder configuration.")
        self.threshold = threshold
        self.refractory_samples = max(round(refractory_seconds * sample_rate_hz), 1)
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._previous: float | None = None
        self._last_event = -self.refractory_samples

    def process(
        self, probabilities: np.ndarray, valid_mask: np.ndarray
    ) -> list[TimingEvent]:
        values = np.asarray(probabilities, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if values.ndim != 1 or values.size == 0 or valid.shape != values.shape:
            raise ValueError(
                "probabilities and valid_mask must share a non-empty 1D shape."
            )
        events: list[TimingEvent] = []
        for value, is_valid in zip(values, valid, strict=True):
            index = self._sample_index
            if not is_valid or not np.isfinite(value):
                self._previous = None
            elif self._previous is not None:
                if (
                    self._previous < self.threshold <= value
                    and index - self._last_event >= self.refractory_samples
                ):
                    events.append(TimingEvent(index, float(value)))
                    self._last_event = index
                self._previous = float(value)
            else:
                self._previous = float(value)
            self._sample_index += 1
        return events
