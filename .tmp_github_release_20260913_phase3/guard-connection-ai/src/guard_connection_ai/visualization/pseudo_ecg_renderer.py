from __future__ import annotations

import numpy as np

from guard_connection_ai.streaming.pulse_tracker import PulseEvent


def qrs_t_template(sample_rate_hz: float) -> np.ndarray:
    """Return a display-only QRS/T template without a synthetic P wave."""

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    duration_seconds = 0.48
    time = np.arange(max(round(duration_seconds * sample_rate_hz), 1)) / sample_rate_hz

    def gaussian(center: float, width: float, amplitude: float) -> np.ndarray:
        return amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)

    return (
        gaussian(0.025, 0.009, -0.18)
        + gaussian(0.055, 0.010, 1.0)
        + gaussian(0.085, 0.012, -0.28)
        + gaussian(0.30, 0.055, 0.24)
    ).astype(np.float32)


class StatefulPseudoECGRenderer:
    """Render an ECG-like rhythm trace on a deliberately delayed display timeline."""

    waveform_type = "ppg_derived_pseudo_ecg"
    diagnostic_ecg = False

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        pulse_arrival_seconds: float,
        display_delay_seconds: float,
    ) -> None:
        if sample_rate_hz <= 0 or pulse_arrival_seconds < 0:
            raise ValueError("sample rate must be positive and PAT non-negative.")
        if display_delay_seconds < pulse_arrival_seconds:
            raise ValueError("display delay must be at least the pulse-arrival delay.")
        self.sample_rate_hz = sample_rate_hz
        self.pat_samples = round(pulse_arrival_seconds * sample_rate_hz)
        self.delay_samples = round(display_delay_seconds * sample_rate_hz)
        self.template = qrs_t_template(sample_rate_hz)
        self.reset()

    @property
    def display_delay_seconds(self) -> float:
        return self.delay_samples / self.sample_rate_hz

    def reset(self) -> None:
        self._input_index = 0
        self._scheduled: dict[int, float] = {}

    def process(
        self,
        sample_count: int,
        events: list[PulseEvent],
        *,
        render_enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        chunk_start = self._input_index
        chunk_end = chunk_start + sample_count
        if render_enabled:
            for event in events:
                if not chunk_start - 1 <= event.sample_index < chunk_end:
                    raise ValueError("pulse event does not belong to the current chunk.")
                emission_start = event.sample_index - self.pat_samples + self.delay_samples
                for offset, value in enumerate(self.template):
                    target = emission_start + offset
                    self._scheduled[target] = self._scheduled.get(target, 0.0) + float(value)
        output = np.zeros(sample_count, dtype=np.float32)
        valid = np.zeros(sample_count, dtype=bool)
        display_indices = np.arange(chunk_start, chunk_end, dtype=np.int64) - self.delay_samples
        for local_index, input_index in enumerate(range(chunk_start, chunk_end)):
            if render_enabled and input_index >= self.delay_samples:
                output[local_index] = self._scheduled.pop(input_index, 0.0)
                valid[local_index] = True
            else:
                self._scheduled.pop(input_index, None)
        self._input_index = chunk_end
        return output, valid, display_indices
