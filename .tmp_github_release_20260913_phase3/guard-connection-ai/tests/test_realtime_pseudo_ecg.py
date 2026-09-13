from __future__ import annotations

import numpy as np

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.features.pulse_interval_representation import (
    StatefulPulseIntervalRepresentation,
)
from guard_connection_ai.streaming.pulse_tracker import PulseEvent, StatefulPulseTracker
from guard_connection_ai.streaming.realtime_session import RealtimePPGSession
from guard_connection_ai.visualization.pseudo_ecg_renderer import (
    StatefulPseudoECGRenderer,
    qrs_t_template,
)


def test_pulse_tracker_events_do_not_depend_on_chunk_boundaries() -> None:
    sample_rate = 50
    time = np.arange(500) / sample_rate
    signal = np.sin(2 * np.pi * 1.2 * time)
    valid = np.ones(signal.size, dtype=bool)
    single = StatefulPulseTracker(sample_rate_hz=sample_rate)
    chunked = StatefulPulseTracker(sample_rate_hz=sample_rate)

    expected = single.process(signal, valid)
    actual = []
    for start, end in ((0, 73), (73, 211), (211, 500)):
        actual.extend(chunked.process(signal[start:end], valid[start:end]))

    assert [event.sample_index for event in actual] == [
        event.sample_index for event in expected
    ]


def test_renderer_has_no_p_wave_and_uses_a_delayed_timeline() -> None:
    sample_rate = 100
    template = qrs_t_template(sample_rate)
    renderer = StatefulPseudoECGRenderer(
        sample_rate_hz=sample_rate,
        pulse_arrival_seconds=0.2,
        display_delay_seconds=0.5,
    )
    output, valid, display_indices = renderer.process(
        100, [PulseEvent(30, 1.0)], render_enabled=True
    )

    assert renderer.diagnostic_ecg is False
    assert renderer.waveform_type == "ppg_derived_pseudo_ecg"
    assert np.allclose(template[:10], output[60:70])
    assert not np.any(valid[:50])
    assert display_indices[0] == -50


def test_realtime_session_is_chunk_invariant_and_serializes_provenance() -> None:
    sample_rate = 50
    time = np.arange(500) / sample_rate
    ppg = np.sin(2 * np.pi * 1.2 * time)
    scaler = FixedRobustScaler(center=0.0, scale=1.0)
    single = RealtimePPGSession(
        session_id="single",
        sample_rate_hz=sample_rate,
        ppg_scaler=scaler,
        quality_window_seconds=2.0,
    )
    chunked = RealtimePPGSession(
        session_id="chunked",
        sample_rate_hz=sample_rate,
        ppg_scaler=scaler,
        quality_window_seconds=2.0,
    )

    expected = single.process(ppg)
    frames = [chunked.process(ppg[:73]), chunked.process(ppg[73:211]), chunked.process(ppg[211:])]
    actual_waveform = np.concatenate([frame.pseudo_ecg for frame in frames])
    actual_valid = np.concatenate([frame.pseudo_ecg_valid for frame in frames])

    np.testing.assert_allclose(actual_waveform, expected.pseudo_ecg)
    np.testing.assert_array_equal(actual_valid, expected.pseudo_ecg_valid)
    assert frames[-1].accepted
    assert frames[-1].diagnostic_ecg is False
    assert frames[-1].waveform_type == "ppg_derived_pseudo_ecg"
    assert frames[-1].sequence_number == 2
    assert frames[-1].to_dict()["schema_version"] == "1.1"
    emitted = [event for frame in frames for event in frame.beat_events]
    assert len({event.event_sample_index for event in emitted}) == len(emitted)
    assert all(event.source == "pulse_tracker_fallback" for event in emitted)
    assert all(
        event.estimated_r_timestamp_seconds
        == event.detected_at_input_timestamp_seconds - 0.2
        for event in emitted
    )


def test_qrs_template_is_neither_duplicated_nor_lost_at_chunk_boundaries() -> None:
    sample_rate = 100
    events = [PulseEvent(40, 1.0), PulseEvent(140, 1.0), PulseEvent(240, 1.0)]
    single = StatefulPseudoECGRenderer(
        sample_rate_hz=sample_rate,
        pulse_arrival_seconds=0.2,
        display_delay_seconds=0.5,
    )
    chunked = StatefulPseudoECGRenderer(
        sample_rate_hz=sample_rate,
        pulse_arrival_seconds=0.2,
        display_delay_seconds=0.5,
    )
    expected, expected_valid, _ = single.process(350, events, render_enabled=True)
    outputs = []
    masks = []
    for start, end in ((0, 73), (73, 151), (151, 267), (267, 350)):
        local_events = [event for event in events if start - 1 <= event.sample_index < end]
        output, valid, _ = chunked.process(end - start, local_events, render_enabled=True)
        outputs.append(output)
        masks.append(valid)
    np.testing.assert_allclose(np.concatenate(outputs), expected)
    np.testing.assert_array_equal(np.concatenate(masks), expected_valid)


def test_realtime_session_abstains_after_sustained_missing_ppg() -> None:
    sample_rate = 50
    ppg = np.sin(2 * np.pi * 1.2 * np.arange(200) / sample_rate)
    session = RealtimePPGSession(
        session_id="missing",
        sample_rate_hz=sample_rate,
        ppg_scaler=FixedRobustScaler(center=0.0, scale=1.0),
        quality_window_seconds=2.0,
    )
    session.process(ppg)
    frame = session.process(np.full(100, np.nan))

    assert not frame.accepted
    assert not frame.visualization_accepted
    assert frame.abstention_reason == "insufficient_valid_samples"
    assert not any(frame.pseudo_ecg_valid[-50:])
    assert not any(frame.ppg_valid)


def test_pulse_interval_representation_does_not_call_intervals_ecg_rr() -> None:
    representation = StatefulPulseIntervalRepresentation(sample_rate_hz=100)
    summary = representation.update(
        [PulseEvent(index, 1.0) for index in (0, 80, 160, 245, 315)]
    )

    assert summary.interval_count == 4
    assert summary.interval_window_count == 4
    assert summary.pulse_event_count_total == 5
    assert summary.rejected_interval_count_total == 0
    assert summary.latest_pulse_interval_ms == 700
    assert summary.heart_rate_bpm is not None
    assert summary.interval_cv is not None and summary.interval_cv > 0
    assert summary.rmssd_ms is not None and summary.rmssd_ms > 0


def test_interval_count_is_rolling_but_total_event_count_is_monotonic() -> None:
    representation = StatefulPulseIntervalRepresentation(
        sample_rate_hz=100, maximum_intervals=4
    )
    summary = representation.update(
        [PulseEvent(index, 1.0) for index in range(0, 801, 80)]
    )

    assert summary.interval_window_count == 4
    assert summary.interval_count == 4
    assert summary.pulse_event_count_total == 11


def test_rejected_interval_count_does_not_hide_raw_event_total() -> None:
    representation = StatefulPulseIntervalRepresentation(sample_rate_hz=100)
    summary = representation.update(
        [PulseEvent(index, 1.0) for index in (0, 20, 100)]
    )

    assert summary.pulse_event_count_total == 3
    assert summary.rejected_interval_count_total == 1
    assert summary.interval_window_count == 1
