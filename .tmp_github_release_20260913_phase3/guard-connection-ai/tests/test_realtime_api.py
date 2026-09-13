from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.schemas.device_ppg import DevicePPGPacket
from guard_connection_ai.streaming.realtime_api import (
    RealtimeProtocolError,
    RealtimeSessionConfig,
    RealtimeSessionRegistry,
)


def _config(session_id: str) -> RealtimeSessionConfig:
    return RealtimeSessionConfig(
        session_id=session_id,
        user_id="test_user_01",
        sample_rate_hz=50,
        scaler_center=0.0,
        scaler_scale=1.0,
        quality_window_seconds=2.0,
    )


def _ppg(count: int = 300) -> np.ndarray:
    return np.sin(2 * np.pi * 1.2 * np.arange(count) / 50)


def test_registry_chunking_matches_batch_and_preserves_absolute_timestamps() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("batch"))
    registry.create(_config("chunks"))
    values = _ppg()
    batch = registry.process_chunk(
        "batch", sequence_number=0, input_timestamp_start_seconds=1_000.0, ppg=values.tolist()
    )
    chunks = []
    for sequence, (start, end) in enumerate(((0, 73), (73, 211), (211, 300))):
        chunks.append(
            registry.process_chunk(
                "chunks",
                sequence_number=sequence,
                input_timestamp_start_seconds=1_000.0 + start / 50,
                ppg=values[start:end].tolist(),
            )
        )

    np.testing.assert_allclose(
        np.concatenate([frame.pseudo_ecg for frame in chunks]), batch.pseudo_ecg
    )
    np.testing.assert_array_equal(
        np.concatenate([frame.pseudo_ecg_valid for frame in chunks]), batch.pseudo_ecg_valid
    )
    assert batch.input_timestamp_start_seconds == 1_000.0
    assert batch.display_timestamp_start_seconds == 999.5
    assert chunks[-1].input_timestamp_start_seconds == pytest.approx(1_004.22)


@pytest.mark.parametrize(
    ("sequence", "code"), ((0, "duplicate_packet"), (-1, "out_of_order"), (3, "sequence_gap"))
)
def test_registry_rejects_duplicate_out_of_order_and_sequence_gap(sequence: int, code: str) -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("ordered"))
    registry.process_chunk(
        "ordered", sequence_number=0, input_timestamp_start_seconds=10.0, ppg=_ppg(20).tolist()
    )
    with pytest.raises(RealtimeProtocolError, match="packet") as error:
        registry.process_chunk(
            "ordered", sequence_number=sequence, input_timestamp_start_seconds=10.4, ppg=[0.0]
        )
    assert error.value.code == code
    assert error.value.expected_sequence_number == 1


def test_registry_reconnect_reset_and_multiple_sessions_are_isolated() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("first"))
    registry.create(_config("other"))
    first = registry.process_chunk(
        "first", sequence_number=0, input_timestamp_start_seconds=100.0, ppg=_ppg(100).tolist()
    )
    other = registry.process_chunk(
        "other", sequence_number=0, input_timestamp_start_seconds=500.0, ppg=_ppg(20).tolist()
    )
    assert first.sequence_number == other.sequence_number == 0
    assert other.input_timestamp_start_seconds == 500.0

    registry.end("first")
    registry.create(_config("first"))
    reset = registry.process_chunk(
        "first", sequence_number=0, input_timestamp_start_seconds=900.0, ppg=_ppg(20).tolist()
    )
    assert reset.sequence_number == 0
    assert reset.input_timestamp_start_seconds == 900.0


def test_registry_status_snapshot_reports_activity_without_waveforms() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("status"))
    registry.process_chunk(
        "status",
        sequence_number=0,
        input_timestamp_start_seconds=100.0,
        ppg=_ppg(20).tolist(),
    )
    status = registry.status_snapshot()
    assert status["active_session_count"] == 1
    session = status["sessions"][0]
    assert session["session_id"] == "status"
    assert session["accepted_packet_count"] == 1
    assert session["accepted_sample_count"] == 20
    assert session["last_input_age_seconds"] >= 0
    assert "ppg" not in session


def test_missing_ppg_stops_pseudo_ecg_instead_of_repeating_history() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("missing"))
    registry.process_chunk(
        "missing", sequence_number=0, input_timestamp_start_seconds=0.0, ppg=_ppg(150).tolist()
    )
    frame = registry.process_chunk(
        "missing", sequence_number=1, input_timestamp_start_seconds=3.0, ppg=[None] * 100
    )
    assert not frame.accepted
    assert not any(frame.ppg_valid)
    assert not any(frame.pseudo_ecg_valid[-50:])


def test_lost_packet_can_be_resent_without_mutating_session_state() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("resend"))
    first = _ppg(20).tolist()
    registry.process_chunk(
        "resend", sequence_number=0, input_timestamp_start_seconds=0.0, ppg=first
    )
    with pytest.raises(RealtimeProtocolError) as gap:
        registry.process_chunk(
            "resend", sequence_number=2, input_timestamp_start_seconds=0.8, ppg=first
        )
    assert gap.value.code == "sequence_gap"
    resent = registry.process_chunk(
        "resend", sequence_number=1, input_timestamp_start_seconds=0.4, ppg=first
    )
    assert resent.sequence_number == 1
    next_frame = registry.process_chunk(
        "resend", sequence_number=2, input_timestamp_start_seconds=0.8, ppg=first
    )
    assert next_frame.sequence_number == 2


def test_two_users_remain_isolated_during_interleaved_processing() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("user-a"))
    other = _config("user-b")
    registry.create(
        RealtimeSessionConfig(**{**other.__dict__, "user_id": "test_user_02"})
    )
    values = _ppg(40).tolist()
    a0 = registry.process_chunk(
        "user-a", sequence_number=0, input_timestamp_start_seconds=10.0, ppg=values
    )
    b0 = registry.process_chunk(
        "user-b", sequence_number=0, input_timestamp_start_seconds=500.0, ppg=values
    )
    a1 = registry.process_chunk(
        "user-a", sequence_number=1, input_timestamp_start_seconds=10.8, ppg=values
    )
    assert (a0.sequence_number, a1.sequence_number, b0.sequence_number) == (0, 1, 0)
    assert b0.input_timestamp_start_seconds == 500.0


def test_device_packet_runs_135_hz_input_through_125_hz_frontend() -> None:
    registry = RealtimeSessionRegistry()
    registry.create(_config("device"))
    count = 135
    values = np.rint(10_000 * np.sin(2 * np.pi * 1.2 * np.arange(count) / 135)).astype(int)
    packet = DevicePPGPacket(
        schema_version="1.0",
        stream_id="device-stream",
        sequence_number=0,
        configured_sample_rate_hz=135,
        device_timestamp_end_ns=round((1000 + (count - 1) / 135) * 1_000_000_000),
        ppg0=values.tolist(),
        ppg1=(values + 2).tolist(),
        ppg2=(values - 2).tolist(),
        ambient0=[500] * count,
    )

    result = registry.process_device_packet("device", packet)

    assert result.adapted.ppg.size == 50
    assert result.waveform_frame is not None
    assert result.waveform_frame.sample_rate_hz == 50
    payload = result.downstream_payload()
    assert payload["source_device_packet"]["ambient0"] == [500] * count
    assert payload["derived"]["sample_count"] == 50


def test_session_provenance_is_fixed_and_packet_mismatch_is_rejected() -> None:
    registry = RealtimeSessionRegistry()
    config = _config("replay")
    created = registry.create(
        RealtimeSessionConfig(
            **{
                **config.__dict__,
                "source_mode": "recorded_demo_replay",
                "demo_scenario_id": "held-out-af-01",
            }
        )
    )
    assert created["source_mode"] == "recorded_demo_replay"
    assert registry.source_provenance_for("replay") == {
        "source_mode": "recorded_demo_replay",
        "demo_scenario_id": "held-out-af-01",
    }

    packet = DevicePPGPacket(
        schema_version="1.0",
        stream_id="device-stream",
        sequence_number=0,
        configured_sample_rate_hz=50,
        device_timestamp_end_ns=1_000_000_000,
        ppg0=[1, 2],
        ppg1=[1, 2],
        ppg2=[1, 2],
        ambient0=[0, 0],
    )
    with pytest.raises(RealtimeProtocolError) as error:
        registry.process_device_packet("replay", packet)
    assert error.value.code == "source_provenance_mismatch"
