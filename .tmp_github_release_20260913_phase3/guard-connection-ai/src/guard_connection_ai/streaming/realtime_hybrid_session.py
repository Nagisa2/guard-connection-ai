from __future__ import annotations

from typing import Protocol

import numpy as np

from guard_connection_ai.data.causal_preprocessing import (
    FixedRobustScaler,
    StatefulCausalPPGPreprocessor,
)
from guard_connection_ai.features.pulse_interval_representation import (
    StatefulPulseIntervalRepresentation,
)
from guard_connection_ai.quality.ppg_sqi import StatefulPPGQuality
from guard_connection_ai.schemas.realtime_waveform import (
    BeatEvent,
    RealtimeWaveformFrame,
)
from guard_connection_ai.streaming.constrained_morphology_renderer import (
    ConstrainedMorphologyOutput,
)
from guard_connection_ai.streaming.learned_pseudo_ecg import LearnedTimingOutput
from guard_connection_ai.streaming.timing_decoder import TimingEvent


class TimingSession(Protocol):
    output_delay_seconds: float

    def process(
        self, inputs: np.ndarray, *, render_enabled: bool
    ) -> LearnedTimingOutput: ...


class MorphologySession(Protocol):
    display_delay_seconds: float
    waveform_type: str
    diagnostic_ecg: bool

    def process(
        self,
        ppg_chunk: np.ndarray,
        timing_events: list[TimingEvent],
        *,
        render_enabled: bool,
    ) -> ConstrainedMorphologyOutput: ...


class RealtimeHybridPPGSession:
    """Session-local Timing Head and constrained morphology display pipeline."""

    def __init__(
        self,
        *,
        session_id: str,
        sample_rate_hz: float,
        ppg_scaler: FixedRobustScaler,
        timing_session: TimingSession,
        morphology_session: MorphologySession,
        morphology_source: str,
        quality_window_seconds: float = 10.0,
    ) -> None:
        if not session_id or sample_rate_hz <= 0:
            raise ValueError("session_id and sample_rate_hz are required.")
        self.session_id = session_id
        self.sample_rate_hz = sample_rate_hz
        self.preprocessor = StatefulCausalPPGPreprocessor(
            sample_rate_hz=sample_rate_hz, scaler=ppg_scaler
        )
        self.timing_session = timing_session
        self.morphology_session = morphology_session
        self.morphology_source = morphology_source
        self.quality = StatefulPPGQuality(
            sample_rate_hz=sample_rate_hz, window_seconds=quality_window_seconds
        )
        self.intervals = StatefulPulseIntervalRepresentation(
            sample_rate_hz=sample_rate_hz
        )
        self._input_samples = 0
        self._sequence_number = 0
        self._timeline_origin_seconds: float | None = None
        self._latest_timing_confidence: float | None = None
        self._latest_morphology_confidence: float | None = None
        self._latest_repolarization_prior_ms: float | None = None

    def process(
        self,
        ppg_chunk: np.ndarray,
        *,
        input_timestamp_start_seconds: float | None = None,
    ) -> RealtimeWaveformFrame:
        raw = np.asarray(ppg_chunk, dtype=np.float64)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("ppg_chunk must be a non-empty 1D array.")
        processed = self.preprocessor.process(raw)
        timing = self.timing_session.process(processed, render_enabled=True)
        start = self._input_samples
        events = [
            TimingEvent(
                int(index), float(timing.event_probability[int(index) - start])
            )
            for index in timing.event_sample_indices
        ]
        representation = self.intervals.update(events)
        decision = self.quality.update(processed[1], events)
        visualization_accepted = decision.accepted or (
            decision.rejection_reason == "quality_warmup"
            and decision.coverage >= self.quality.minimum_coverage
            and decision.score >= self.quality.minimum_score
        )
        morphology = self.morphology_session.process(
            processed, events, render_enabled=visualization_accepted
        )
        if events:
            self._latest_timing_confidence = float(
                max(event.confidence for event in events)
            )
        if morphology.events:
            latest = morphology.events[-1]
            self._latest_morphology_confidence = latest.confidence
            self._latest_repolarization_prior_ms = (
                latest.repolarization_duration_prior_ms
            )

        relative_input_start = start / self.sample_rate_hz
        if input_timestamp_start_seconds is not None:
            if not np.isfinite(input_timestamp_start_seconds):
                raise ValueError("input_timestamp_start_seconds must be finite.")
            if self._timeline_origin_seconds is None:
                self._timeline_origin_seconds = (
                    float(input_timestamp_start_seconds) - relative_input_start
                )
            expected = self._timeline_origin_seconds + relative_input_start
            if abs(float(input_timestamp_start_seconds) - expected) > (
                0.5 / self.sample_rate_hz
            ):
                raise ValueError("input timestamp is not contiguous with session state.")
        origin = self._timeline_origin_seconds or 0.0
        input_start = origin + relative_input_start
        display_start = input_start - self.morphology_session.display_delay_seconds
        beat_events = [
            BeatEvent(
                event_sample_index=event.sample_index,
                detected_at_input_timestamp_seconds=(
                    origin + event.sample_index / self.sample_rate_hz
                ),
                estimated_r_timestamp_seconds=(
                    origin
                    + event.sample_index / self.sample_rate_hz
                    - self.timing_session.output_delay_seconds
                ),
                event_type="estimated_ecg_r_event",
                source="timing_head",
                confidence=float(event.confidence),
            )
            for event in events
        ]
        frame = RealtimeWaveformFrame(
            schema_version="1.1",
            session_id=self.session_id,
            sequence_number=self._sequence_number,
            sample_rate_hz=self.sample_rate_hz,
            input_timestamp_start_seconds=input_start,
            display_timestamp_start_seconds=display_start,
            ppg=np.nan_to_num(raw, nan=0.0).astype(np.float32).tolist(),
            ppg_valid=np.isfinite(raw).tolist(),
            pseudo_ecg=morphology.pseudo_ecg.tolist(),
            pseudo_ecg_valid=morphology.pseudo_ecg_valid.tolist(),
            waveform_type=self.morphology_session.waveform_type,
            diagnostic_ecg=self.morphology_session.diagnostic_ecg,
            ppg_sqi=decision.score,
            technical_sqi=decision.technical_score,
            signal_coverage=decision.coverage,
            pulse_interval_plausibility=decision.plausible_interval_fraction,
            accepted=decision.accepted,
            abstention_reason=decision.rejection_reason,
            visualization_accepted=visualization_accepted,
            visualization_abstention_reason=(
                None if visualization_accepted else decision.rejection_reason
            ),
            estimated_pat_ms=None,
            display_delay_ms=1000 * self.morphology_session.display_delay_seconds,
            latest_pulse_interval_ms=representation.latest_pulse_interval_ms,
            heart_rate_bpm=representation.heart_rate_bpm,
            interval_cv=representation.interval_cv,
            rmssd_ms=representation.rmssd_ms,
            beat_events=beat_events,
            interval_window_count=representation.interval_window_count,
            pulse_event_count_total=representation.pulse_event_count_total,
            rejected_interval_count_total=(
                representation.rejected_interval_count_total
            ),
            interval_count=representation.interval_count,
            generation_mode=(
                (
                    "learned_morphology"
                    if self.morphology_source == "learned_ppg_latent"
                    else "population_prior"
                )
                if visualization_accepted
                else "blank"
            ),
            morphology_source=self.morphology_source,
            timing_confidence=self._latest_timing_confidence,
            morphology_confidence=self._latest_morphology_confidence,
            repolarization_duration_prior_ms=(
                self._latest_repolarization_prior_ms
            ),
        )
        self._input_samples += raw.size
        self._sequence_number += 1
        return frame
