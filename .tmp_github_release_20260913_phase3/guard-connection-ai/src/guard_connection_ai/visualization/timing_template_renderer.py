from __future__ import annotations

import numpy as np

from guard_connection_ai.streaming.timing_decoder import TimingEvent
from guard_connection_ai.visualization.pseudo_ecg_renderer import qrs_t_template


class StatefulTimingTemplateRenderer:
    """Render non-diagnostic QRS/T morphology from Timing Head events.

    Invalid samples are explicitly blanked. This class never synthesizes a P
    wave and must not be used for interval or morphology measurement.
    """

    waveform_type = "ppg_derived_pseudo_ecg"
    diagnostic_ecg = False

    def __init__(
        self, *, sample_rate_hz: float, template: np.ndarray | None = None
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        selected = (
            qrs_t_template(sample_rate_hz) if template is None else np.asarray(template)
        )
        if (
            selected.ndim != 1
            or selected.size == 0
            or not np.all(np.isfinite(selected))
        ):
            raise ValueError("template must be a finite non-empty 1D array.")
        self.template = selected.astype(np.float32)
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._scheduled: dict[int, float] = {}

    def process(
        self,
        sample_count: int,
        events: list[TimingEvent],
        *,
        valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = np.asarray(valid_mask, dtype=bool)
        if sample_count <= 0 or valid.shape != (sample_count,):
            raise ValueError("valid_mask must match the positive sample_count.")
        start = self._sample_index
        end = start + sample_count
        for event in events:
            if not start <= event.sample_index < end:
                raise ValueError("timing event does not belong to the current chunk.")
            for offset, value in enumerate(self.template):
                index = event.sample_index + offset
                self._scheduled[index] = self._scheduled.get(index, 0.0) + float(value)
        output = np.zeros(sample_count, dtype=np.float32)
        output_valid = np.zeros(sample_count, dtype=bool)
        for local, index in enumerate(range(start, end)):
            if valid[local]:
                output[local] = self._scheduled.pop(index, 0.0)
                output_valid[local] = True
            else:
                self._scheduled.pop(index, None)
        self._sample_index = end
        return output, output_valid
