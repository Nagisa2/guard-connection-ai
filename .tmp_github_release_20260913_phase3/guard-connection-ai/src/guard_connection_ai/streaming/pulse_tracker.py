from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PulseEvent:
    sample_index: int
    amplitude: float


class StatefulPulseTracker:
    """Causal local-maximum detector with adaptive amplitude and refractory gates."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        refractory_seconds: float = 0.25,
        amplitude_ema_seconds: float = 8.0,
        threshold_scale: float = 0.35,
    ) -> None:
        if sample_rate_hz <= 0 or refractory_seconds <= 0 or amplitude_ema_seconds <= 0:
            raise ValueError("sample rate and tracker time constants must be positive.")
        if threshold_scale < 0:
            raise ValueError("threshold_scale must be non-negative.")
        self.sample_rate_hz = sample_rate_hz
        self.refractory_samples = max(round(refractory_seconds * sample_rate_hz), 1)
        self.ema_alpha = 1.0 - np.exp(-1.0 / (amplitude_ema_seconds * sample_rate_hz))
        self.threshold_scale = threshold_scale
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._previous_value: float | None = None
        self._previous_slope: float | None = None
        self._mean = 0.0
        self._deviation = 1.0
        self._last_event = -self.refractory_samples

    def process(self, signal: np.ndarray, valid_mask: np.ndarray) -> list[PulseEvent]:
        values = np.asarray(signal, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if values.ndim != 1 or values.size == 0 or valid.shape != values.shape:
            raise ValueError("signal and valid_mask must share a non-empty 1D shape.")
        events: list[PulseEvent] = []
        for value, is_valid in zip(values, valid, strict=True):
            current_index = self._sample_index
            if is_valid and np.isfinite(value):
                residual = value - self._mean
                self._mean += self.ema_alpha * residual
                self._deviation += self.ema_alpha * (abs(residual) - self._deviation)
                if self._previous_value is not None:
                    slope = value - self._previous_value
                    threshold = self._mean + self.threshold_scale * max(self._deviation, 1e-6)
                    if (
                        self._previous_slope is not None
                        and self._previous_slope > 0
                        and slope <= 0
                        and self._previous_value >= threshold
                        and current_index - 1 - self._last_event >= self.refractory_samples
                    ):
                        event_index = current_index - 1
                        events.append(PulseEvent(event_index, self._previous_value))
                        self._last_event = event_index
                    self._previous_slope = slope
                self._previous_value = float(value)
            else:
                self._previous_value = None
                self._previous_slope = None
            self._sample_index += 1
        return events
