from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from guard_connection_ai.streaming.pulse_tracker import PulseEvent


@dataclass(frozen=True)
class PulseIntervalRepresentation:
    latest_pulse_interval_ms: float | None
    heart_rate_bpm: float | None
    interval_cv: float | None
    rmssd_ms: float | None
    interval_window_count: int
    pulse_event_count_total: int
    rejected_interval_count_total: int
    # Deprecated compatibility alias. This is a rolling-window size, not a total.
    interval_count: int


class StatefulPulseIntervalRepresentation:
    """Maintain rolling pulse intervals without producing a rhythm diagnosis."""

    def __init__(self, *, sample_rate_hz: float, maximum_intervals: int = 64) -> None:
        if sample_rate_hz <= 0 or maximum_intervals < 2:
            raise ValueError("sample rate must be positive and interval capacity at least two.")
        self.sample_rate_hz = sample_rate_hz
        self._intervals: deque[float] = deque(maxlen=maximum_intervals)
        self._last_event: int | None = None
        self._pulse_event_count_total = 0
        self._rejected_interval_count_total = 0

    def update(self, events: list[PulseEvent]) -> PulseIntervalRepresentation:
        self._pulse_event_count_total += len(events)
        for event in events:
            if self._last_event is not None:
                interval = (event.sample_index - self._last_event) / self.sample_rate_hz
                if 0.25 <= interval <= 2.0:
                    self._intervals.append(interval)
                else:
                    self._rejected_interval_count_total += 1
            self._last_event = event.sample_index
        values = np.asarray(self._intervals, dtype=np.float64)
        if values.size == 0:
            return PulseIntervalRepresentation(
                None,
                None,
                None,
                None,
                0,
                self._pulse_event_count_total,
                self._rejected_interval_count_total,
                0,
            )
        latest_ms = 1000 * float(values[-1])
        heart_rate = 60.0 / float(np.median(values))
        interval_cv = float(np.std(values) / np.mean(values)) if values.size >= 2 else None
        rmssd = (
            1000 * float(np.sqrt(np.mean(np.diff(values) ** 2)))
            if values.size >= 2
            else None
        )
        return PulseIntervalRepresentation(
            latest_ms,
            heart_rate,
            interval_cv,
            rmssd,
            int(values.size),
            self._pulse_event_count_total,
            self._rejected_interval_count_total,
            int(values.size),
        )
