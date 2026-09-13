from __future__ import annotations

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
from guard_connection_ai.streaming.pulse_tracker import PulseEvent, StatefulPulseTracker
from guard_connection_ai.visualization.pseudo_ecg_renderer import (
    StatefulPseudoECGRenderer,
)


class RealtimePPGSession:
    """Session-scoped causal PPG, quality, and display-only pseudo-ECG pipeline."""

    def __init__(
        self,
        *,
        session_id: str,
        sample_rate_hz: float,
        ppg_scaler: FixedRobustScaler,
        pulse_arrival_seconds: float = 0.2,
        display_delay_seconds: float = 0.5,
        quality_window_seconds: float = 10.0,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty.")
        self.session_id = session_id
        self.sample_rate_hz = sample_rate_hz
        self.preprocessor = StatefulCausalPPGPreprocessor(
            sample_rate_hz=sample_rate_hz, scaler=ppg_scaler
        )
        self.tracker = StatefulPulseTracker(sample_rate_hz=sample_rate_hz)
        self.quality = StatefulPPGQuality(
            sample_rate_hz=sample_rate_hz, window_seconds=quality_window_seconds
        )
        self.pulse_interval_representation = StatefulPulseIntervalRepresentation(
            sample_rate_hz=sample_rate_hz
        )
        self.renderer = StatefulPseudoECGRenderer(
            sample_rate_hz=sample_rate_hz,
            pulse_arrival_seconds=pulse_arrival_seconds,
            display_delay_seconds=display_delay_seconds,
        )
        self.pulse_arrival_seconds = pulse_arrival_seconds
        self._input_samples = 0
        self._sequence_number = 0
        self._timeline_origin_seconds: float | None = None

    def process(
        self,
        ppg_chunk: np.ndarray,
        *,
        input_timestamp_start_seconds: float | None = None,
    ) -> RealtimeWaveformFrame:
        raw = np.asarray(ppg_chunk, dtype=np.float64)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("ppg_chunk must be a non-empty 1D array.")
        pseudo_parts = []
        valid_parts = []
        display_parts = []
        decision = None
        representation = None
        raw_valid = np.isfinite(raw)
        chunk_events: list[PulseEvent] = []
        for value in raw:
            processed = self.preprocessor.process(np.asarray([value]))
            events = self.tracker.process(processed[0], processed[1])
            chunk_events.extend(events)
            representation = self.pulse_interval_representation.update(events)
            decision = self.quality.update(processed[1], events)
            # 後段AIは品質窓全体を要求する一方、医師画面はウォームアップ中でも
            # coverageとpulse intervalが妥当なら暫定的に描画できる。
            visualization_accepted = decision.accepted or (
                decision.rejection_reason == "quality_warmup"
                and decision.coverage >= self.quality.minimum_coverage
                and decision.score >= self.quality.minimum_score
            )
            pseudo_ecg, pseudo_valid, display_indices = self.renderer.process(
                1, events, render_enabled=visualization_accepted
            )
            pseudo_parts.append(pseudo_ecg)
            valid_parts.append(pseudo_valid)
            display_parts.append(display_indices)
        if decision is None or representation is None:
            raise RuntimeError("streaming decisions were not produced.")
        pseudo_ecg = np.concatenate(pseudo_parts)
        pseudo_valid = np.concatenate(valid_parts)
        display_indices = np.concatenate(display_parts)
        relative_input_start = self._input_samples / self.sample_rate_hz
        if input_timestamp_start_seconds is not None:
            if not np.isfinite(input_timestamp_start_seconds):
                raise ValueError("input_timestamp_start_seconds must be finite.")
            if self._timeline_origin_seconds is None:
                self._timeline_origin_seconds = (
                    float(input_timestamp_start_seconds) - relative_input_start
                )
            expected = self._timeline_origin_seconds + relative_input_start
            tolerance = 0.5 / self.sample_rate_hz
            if abs(float(input_timestamp_start_seconds) - expected) > tolerance:
                raise ValueError("input timestamp is not contiguous with session state.")
        origin = self._timeline_origin_seconds or 0.0
        input_start = origin + relative_input_start
        display_start = origin + float(display_indices[0] / self.sample_rate_hz)
        beat_events = [
            BeatEvent(
                event_sample_index=event.sample_index,
                detected_at_input_timestamp_seconds=(
                    origin + event.sample_index / self.sample_rate_hz
                ),
                estimated_r_timestamp_seconds=(
                    origin
                    + event.sample_index / self.sample_rate_hz
                    - self.pulse_arrival_seconds
                ),
                event_type="ppg_pulse_peak",
                source="pulse_tracker_fallback",
                amplitude=float(event.amplitude),
            )
            for event in chunk_events
        ]
        frame = RealtimeWaveformFrame(
            schema_version="1.1",
            session_id=self.session_id,
            sequence_number=self._sequence_number,
            sample_rate_hz=self.sample_rate_hz,
            input_timestamp_start_seconds=input_start,
            display_timestamp_start_seconds=display_start,
            ppg=np.nan_to_num(raw, nan=0.0).astype(np.float32).tolist(),
            ppg_valid=raw_valid.tolist(),
            pseudo_ecg=pseudo_ecg.tolist(),
            pseudo_ecg_valid=pseudo_valid.tolist(),
            waveform_type=self.renderer.waveform_type,
            diagnostic_ecg=self.renderer.diagnostic_ecg,
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
            estimated_pat_ms=1000 * self.pulse_arrival_seconds,
            display_delay_ms=1000 * self.renderer.display_delay_seconds,
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
                "interval_template" if visualization_accepted else "blank"
            ),
            morphology_source="fixed_qrs_t_template",
        )
        self._input_samples += raw.size
        self._sequence_number += 1
        return frame
