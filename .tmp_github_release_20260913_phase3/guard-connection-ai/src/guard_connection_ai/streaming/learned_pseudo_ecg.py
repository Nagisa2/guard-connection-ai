from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from guard_connection_ai.models.timing_head import (
    CausalTimingHead,
    StatefulCausalTimingHead,
)
from guard_connection_ai.streaming.timing_decoder import StatefulTimingDecoder
from guard_connection_ai.visualization.timing_template_renderer import (
    StatefulTimingTemplateRenderer,
)


@dataclass(frozen=True)
class LearnedTimingOutput:
    event_probability: np.ndarray
    pseudo_ecg: np.ndarray
    pseudo_ecg_valid: np.ndarray
    event_sample_indices: np.ndarray


class StatefulLearnedTimingPseudoECG:
    """Session-local Timing Head to non-diagnostic template rendering pipeline."""

    waveform_type = "ppg_derived_pseudo_ecg"
    diagnostic_ecg = False

    def __init__(
        self,
        model: CausalTimingHead,
        *,
        sample_rate_hz: float,
        output_delay_seconds: float,
        event_threshold: float = 0.5,
        refractory_seconds: float = 0.25,
    ) -> None:
        if output_delay_seconds < 0:
            raise ValueError("output_delay_seconds must be non-negative.")
        self.model = model.eval()
        self.timing_head = StatefulCausalTimingHead(self.model)
        self.decoder = StatefulTimingDecoder(
            sample_rate_hz=sample_rate_hz,
            threshold=event_threshold,
            refractory_seconds=refractory_seconds,
        )
        self.renderer = StatefulTimingTemplateRenderer(sample_rate_hz=sample_rate_hz)
        self.output_delay_seconds = output_delay_seconds
        self.delay_samples = round(output_delay_seconds * sample_rate_hz)
        self._processed_samples = 0

    def reset(self) -> None:
        self.timing_head.reset()
        self.decoder.reset()
        self.renderer.reset()
        self._processed_samples = 0

    def process(
        self, inputs: np.ndarray, *, render_enabled: bool
    ) -> LearnedTimingOutput:
        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
            raise ValueError("inputs must have shape [2, positive samples].")
        tensor = torch.from_numpy(values[None, ...])
        logits = self.timing_head.process(tensor)[0, 0]
        probabilities = torch.sigmoid(logits).cpu().numpy()
        valid = values[1].astype(bool) & np.isfinite(values[0])
        absolute = np.arange(values.shape[1]) + self._processed_samples
        valid &= absolute >= self.delay_samples
        if not render_enabled:
            valid[:] = False
        events = self.decoder.process(probabilities, valid)
        waveform, waveform_valid = self.renderer.process(
            values.shape[1], events, valid_mask=valid
        )
        self._processed_samples += values.shape[1]
        return LearnedTimingOutput(
            event_probability=probabilities.astype(np.float32),
            pseudo_ecg=waveform,
            pseudo_ecg_valid=waveform_valid,
            event_sample_indices=np.asarray(
                [event.sample_index for event in events], dtype=np.int64
            ),
        )
