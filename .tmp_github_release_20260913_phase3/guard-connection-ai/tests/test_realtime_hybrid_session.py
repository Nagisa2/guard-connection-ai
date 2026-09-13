from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.streaming.constrained_morphology_renderer import (
    ConstrainedMorphologyOutput,
    MorphologyEventResult,
)
from guard_connection_ai.streaming.learned_pseudo_ecg import LearnedTimingOutput
from guard_connection_ai.streaming.realtime_api import (
    RealtimeSessionConfig,
    RealtimeSessionRegistry,
)
from guard_connection_ai.streaming.realtime_hybrid_session import (
    RealtimeHybridPPGSession,
)


class _Timing:
    def __init__(self, sample_rate_hz: float) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.output_delay_seconds = 0.5
        self.index = 0

    def process(
        self, inputs: np.ndarray, *, render_enabled: bool
    ) -> LearnedTimingOutput:
        del render_enabled
        count = inputs.shape[-1]
        absolute = np.arange(self.index, self.index + count)
        event_indices = absolute[(absolute > 0) & (absolute % 25 == 0)]
        probability = np.full(count, 0.1, dtype=np.float32)
        probability[event_indices - self.index] = 0.9
        self.index += count
        return LearnedTimingOutput(
            event_probability=probability,
            pseudo_ecg=np.zeros(count, dtype=np.float32),
            pseudo_ecg_valid=np.ones(count, dtype=bool),
            event_sample_indices=event_indices,
        )


class _Morphology:
    display_delay_seconds = 0.66
    waveform_type = "ppg_derived_pseudo_ecg"
    diagnostic_ecg = False

    def process(self, ppg_chunk, timing_events, *, render_enabled):
        count = ppg_chunk.shape[-1]
        events = tuple(
            MorphologyEventResult(
                event.sample_index,
                "population_mean_latent",
                0.0,
                380.0,
            )
            for event in timing_events
            if render_enabled
        )
        return ConstrainedMorphologyOutput(
            pseudo_ecg=np.full(count, 0.25, dtype=np.float32),
            pseudo_ecg_valid=np.full(count, render_enabled, dtype=bool),
            events=events,
        )


def _session(session_id: str) -> RealtimeHybridPPGSession:
    return RealtimeHybridPPGSession(
        session_id=session_id,
        sample_rate_hz=50,
        ppg_scaler=FixedRobustScaler(0.0, 1.0),
        timing_session=_Timing(50),
        morphology_session=_Morphology(),
        morphology_source="population_mean_latent",
        quality_window_seconds=1.0,
    )


def test_hybrid_frame_exposes_non_diagnostic_morphology_metadata() -> None:
    frame = _session("hybrid").process(
        np.sin(2 * np.pi * 1.0 * np.arange(100) / 50),
        input_timestamp_start_seconds=100.0,
    )
    assert frame.generation_mode == "population_prior"
    assert frame.morphology_source == "population_mean_latent"
    assert frame.timing_confidence == pytest.approx(0.9)
    assert frame.morphology_confidence == 0.0
    assert frame.repolarization_duration_prior_ms == 380.0
    assert frame.estimated_pat_ms is None
    assert frame.display_timestamp_start_seconds == 99.34
    assert frame.p_wave_generated is False
    assert frame.qt_measurement_supported is False
    assert frame.pq_pr_measurement_supported is False
    assert frame.schema_version == "1.1"
    assert [event.event_sample_index for event in frame.beat_events] == [25, 50, 75]
    assert all(event.source == "timing_head" for event in frame.beat_events)
    assert frame.beat_events[0].detected_at_input_timestamp_seconds == 100.5
    assert frame.beat_events[0].estimated_r_timestamp_seconds == 100.0
    assert frame.pulse_event_count_total == 3
    assert frame.interval_window_count == 2
    assert frame.interval_count == frame.interval_window_count


def test_registry_factory_creates_isolated_hybrid_state() -> None:
    registry = RealtimeSessionRegistry(
        pipeline_factories={
            "hybrid": lambda config, session_id: _session(session_id)
        }
    )
    for session_id in ("first", "second"):
        registry.create(
            RealtimeSessionConfig(
                user_id=session_id,
                session_id=session_id,
                sample_rate_hz=50,
                scaler_center=0,
                scaler_scale=1,
                generation_profile_id="hybrid",
            )
        )
    values = np.sin(2 * np.pi * np.arange(50) / 50).tolist()
    first = registry.process_chunk(
        "first", sequence_number=0, input_timestamp_start_seconds=0, ppg=values
    )
    second = registry.process_chunk(
        "second", sequence_number=0, input_timestamp_start_seconds=10, ppg=values
    )
    assert first.sequence_number == second.sequence_number == 0
    assert first.timing_confidence == second.timing_confidence
    assert first.pulse_event_count_total == second.pulse_event_count_total


def test_hybrid_missing_ppg_blanks_pseudo_ecg() -> None:
    frame = _session("missing").process(
        np.full(100, np.nan), input_timestamp_start_seconds=0
    )
    assert frame.generation_mode == "blank"
    assert not any(frame.pseudo_ecg_valid)
    assert not frame.visualization_accepted
