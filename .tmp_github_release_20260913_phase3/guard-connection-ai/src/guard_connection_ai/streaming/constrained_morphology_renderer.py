from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.models.ppg_morphology_encoder import (
    PPGMorphologyEncoder,
    morphology_confidence,
)
from guard_connection_ai.streaming.timing_decoder import TimingEvent


@dataclass(frozen=True)
class MorphologyEventResult:
    sample_index: int
    source: str
    confidence: float
    repolarization_duration_prior_ms: float


@dataclass(frozen=True)
class ConstrainedMorphologyOutput:
    pseudo_ecg: np.ndarray
    pseudo_ecg_valid: np.ndarray
    events: tuple[MorphologyEventResult, ...]


def stabilize_display_morphology(
    beat: np.ndarray,
    *,
    r_index: int,
    sample_rate_hz: float,
    previous_rr_seconds: float,
) -> tuple[np.ndarray, float]:
    """Apply a bounded, non-diagnostic QRS/T display prior.

    ``previous_rr_seconds`` only changes the display prior's repolarization tail.
    The returned duration is not a measured or reconstructed QT/QTc interval.
    """

    values = np.nan_to_num(np.asarray(beat, dtype=np.float32)).copy()
    if values.ndim != 1 or values.size < 4:
        raise ValueError("beat must be a one-dimensional waveform.")
    if not 0 <= r_index < values.size or sample_rate_hz <= 0:
        raise ValueError("invalid R index or sample rate.")
    if not np.isfinite(previous_rr_seconds) or previous_rr_seconds <= 0:
        raise ValueError("previous_rr_seconds must be finite and positive.")

    qrs_start = max(r_index - round(0.04 * sample_rate_hz), 0)
    qrs_end = min(r_index + round(0.10 * sample_rate_hz), values.size - 1)
    baseline_region = values[: max(qrs_start, 1)]
    values -= float(np.median(baseline_region))

    qrs = values[qrs_start : qrs_end + 1]
    dominant_index = int(np.argmax(np.abs(qrs)))
    if qrs[dominant_index] < 0:
        values *= -1
    qrs_scale = float(np.max(np.abs(values[qrs_start : qrs_end + 1])))
    if qrs_scale > 1e-6:
        values /= qrs_scale

    # No P-wave-like content is drawn. P/PQ/PR are not observable from PPG.
    values[:qrs_start] = 0.0

    # Fridericia-like RR scaling is used only as a bounded visual duration prior.
    duration_seconds = float(
        np.clip(0.40 * previous_rr_seconds ** (1.0 / 3.0), 0.30, 0.48)
    )
    target_end = min(
        r_index + round((duration_seconds - 0.04) * sample_rate_hz),
        values.size - 1,
    )
    t_start = min(r_index + round(0.16 * sample_rate_hz), target_end)
    if t_start > qrs_end:
        values[qrs_end:t_start] = np.linspace(
            values[qrs_end], 0.0, t_start - qrs_end, endpoint=False
        )
    source = values[t_start:].copy()
    target_count = target_end - t_start + 1
    if source.size >= 2 and target_count >= 2:
        smoothing_samples = min(max(round(0.08 * sample_rate_hz), 3), source.size)
        kernel = np.ones(smoothing_samples, dtype=np.float32) / smoothing_samples
        source = np.convolve(source, kernel, mode="same")
        dominant = source[int(np.argmax(np.abs(source)))]
        if dominant < 0:
            source *= -1
        source = np.maximum(source, 0.0)
        amplitude = float(np.clip(np.max(source), 0.25, 0.45))
        source_axis = np.linspace(0.0, 1.0, source.size)
        target_axis = np.linspace(0.0, 1.0, target_count)
        t_wave = np.interp(target_axis, source_axis, source)
        if np.max(t_wave) <= 1e-6:
            t_wave = np.sin(np.pi * target_axis) ** 1.5
        t_wave *= np.sin(np.pi * target_axis)
        peak = float(np.max(t_wave))
        if peak > 1e-6:
            t_wave *= amplitude / peak
        values[t_start : target_end + 1] = t_wave
    values[target_end + 1 :] = 0.0
    values[target_end] = 0.0
    return values.astype(np.float32), 1000.0 * duration_seconds


class StatefulConstrainedMorphologyRenderer:
    """Place one frozen morphology beat per Timing Head event."""

    waveform_type = "ppg_derived_pseudo_ecg"
    diagnostic_ecg = False

    def __init__(
        self,
        *,
        morphology_decoder: MorphologyAutoencoder1D,
        population_latent: torch.Tensor,
        sample_rate_hz: float,
        timing_output_delay_seconds: float,
        seconds_before_r: float,
        ppg_encoder: PPGMorphologyEncoder | None = None,
        context_seconds: float = 1.5,
        confidence_threshold: float = 0.5,
        allow_learned_morphology: bool = False,
    ) -> None:
        if (
            sample_rate_hz <= 0
            or min(timing_output_delay_seconds, seconds_before_r) < 0
        ):
            raise ValueError("invalid sample rate or delay.")
        if context_seconds <= 0 or not 0 <= confidence_threshold <= 1:
            raise ValueError("invalid context or confidence threshold.")
        if population_latent.ndim != 1:
            raise ValueError("population_latent must be one-dimensional.")
        self.decoder = morphology_decoder.eval()
        self.ppg_encoder = ppg_encoder.eval() if ppg_encoder is not None else None
        self.population_latent = population_latent.detach().cpu()
        self.sample_rate_hz = sample_rate_hz
        self.context_samples = round(context_seconds * sample_rate_hz)
        self.confidence_threshold = confidence_threshold
        self.allow_learned_morphology = allow_learned_morphology
        self.r_index = round(seconds_before_r * sample_rate_hz)
        self.display_delay_seconds = timing_output_delay_seconds + seconds_before_r
        self.reset()

    def reset(self) -> None:
        self._sample_index = 0
        self._ppg_history: np.ndarray | None = None
        self._scheduled: dict[int, float] = {}
        self._last_event_sample_index: int | None = None

    def _select_beat(self, context: np.ndarray) -> tuple[np.ndarray, str, float]:
        source = "population_mean_latent"
        confidence = 0.0
        latent = self.population_latent
        if (
            self.allow_learned_morphology
            and self.ppg_encoder is not None
            and context.shape[-1] == self.context_samples
            and np.all(context[1] > 0)
        ):
            with torch.no_grad():
                mean, log_variance = self.ppg_encoder(
                    torch.from_numpy(context[None, ...].astype(np.float32))
                )
                confidence = float(morphology_confidence(log_variance)[0])
                if confidence >= self.confidence_threshold:
                    latent = mean[0].cpu()
                    source = "learned_ppg_latent"
        with torch.no_grad():
            beat = self.decoder.decode(latent[None, :])[0, 0].cpu().numpy()
        return beat.astype(np.float32), source, confidence

    def process(
        self,
        ppg_chunk: np.ndarray,
        timing_events: list[TimingEvent],
        *,
        render_enabled: bool,
    ) -> ConstrainedMorphologyOutput:
        values = np.asarray(ppg_chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
            raise ValueError("ppg_chunk must have shape [2, positive samples].")
        start = self._sample_index
        end = start + values.shape[-1]
        if any(not start <= event.sample_index < end for event in timing_events):
            raise ValueError("timing events must belong to the current chunk.")
        combined = (
            values
            if self._ppg_history is None
            else np.concatenate((self._ppg_history, values), axis=-1)
        )
        history_start = start - (combined.shape[-1] - values.shape[-1])
        event_results = []
        for event in sorted(timing_events, key=lambda value: value.sample_index):
            previous_rr_seconds = (
                (event.sample_index - self._last_event_sample_index)
                / self.sample_rate_hz
                if self._last_event_sample_index is not None
                else 0.8
            )
            self._last_event_sample_index = event.sample_index
            if render_enabled:
                context_end = event.sample_index - history_start + 1
                context_start = context_end - self.context_samples
                context = (
                    combined[:, context_start:context_end]
                    if context_start >= 0
                    else np.empty((2, 0), dtype=np.float32)
                )
                beat, source, confidence = self._select_beat(context)
                beat, duration_prior_ms = stabilize_display_morphology(
                    beat,
                    r_index=self.r_index,
                    sample_rate_hz=self.sample_rate_hz,
                    previous_rr_seconds=previous_rr_seconds,
                )
                for scheduled_index in tuple(self._scheduled):
                    if scheduled_index >= event.sample_index:
                        self._scheduled.pop(scheduled_index)
                for offset, amplitude in enumerate(beat):
                    self._scheduled[event.sample_index + offset] = float(amplitude)
                event_results.append(
                    MorphologyEventResult(
                        event.sample_index, source, confidence, duration_prior_ms
                    )
                )
        output = np.zeros(values.shape[-1], dtype=np.float32)
        valid = np.zeros(values.shape[-1], dtype=bool)
        input_valid = values[1].astype(bool) & np.isfinite(values[0])
        for local_index, sample_index in enumerate(range(start, end)):
            if render_enabled and input_valid[local_index]:
                output[local_index] = self._scheduled.pop(sample_index, 0.0)
                valid[local_index] = True
            else:
                self._scheduled.pop(sample_index, None)
        self._ppg_history = combined[:, -self.context_samples :].copy()
        self._sample_index = end
        return ConstrainedMorphologyOutput(output, valid, tuple(event_results))
