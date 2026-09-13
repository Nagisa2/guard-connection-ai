from __future__ import annotations

import numpy as np

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.schemas.front_ai_outputs import (
    to_doctor_visualization_frame,
    to_downstream_ai_frame,
)
from guard_connection_ai.streaming.realtime_session import RealtimePPGSession


def _frame():
    session = RealtimePPGSession(
        session_id="handoff-sample",
        sample_rate_hz=50,
        ppg_scaler=FixedRobustScaler(center=0.0, scale=1.0),
        quality_window_seconds=2.0,
    )
    ppg = np.sin(2 * np.pi * 1.2 * np.arange(150) / 50)
    ppg[80:90] = np.nan
    return session.process(ppg, input_timestamp_start_seconds=1_000.0)


def test_downstream_output_keeps_source_ppg_and_contains_no_af_result() -> None:
    output = to_downstream_ai_frame(_frame())
    payload = output.to_dict()

    assert output.payload_type == "front_ai_downstream_input"
    assert output.source_signal_type == "measured_ppg"
    assert output.sample_count == len(output.ppg) == len(output.ppg_valid)
    assert not all(output.ppg_valid)
    assert output.schema_version == "1.1"
    assert output.interval_count == output.interval_window_count
    assert output.pulse_event_count_total >= len(output.beat_events)
    assert output.technical_sqi == output.ppg_sqi
    assert "af_probability" not in payload
    assert "uncertainty" not in payload
    assert "prediction" not in payload


def test_doctor_output_is_explicitly_non_diagnostic_and_drawable() -> None:
    output = to_doctor_visualization_frame(_frame())

    assert output.payload_type == "doctor_waveform_visualization"
    assert output.waveform_type == "ppg_derived_pseudo_ecg"
    assert output.diagnostic_ecg is False
    assert output.measurement_ui_enabled is False
    assert "診断用ではない" in output.display_label
    assert len(output.ppg) == len(output.pseudo_ecg) == len(output.pseudo_ecg_valid)
    assert output.display_timestamp_start_seconds < output.input_timestamp_start_seconds
    assert output.generation_mode in {"interval_template", "blank"}
    assert output.p_wave_generated is False
    assert output.qt_measurement_supported is False
    assert output.pq_pr_measurement_supported is False
    assert output.technical_sqi == output.ppg_sqi
    assert 0 <= output.pulse_interval_plausibility <= 1


def test_outputs_separate_generation_status_from_downstream_diagnosis() -> None:
    frame = _frame()
    downstream = to_downstream_ai_frame(frame)
    doctor = to_doctor_visualization_frame(frame)

    assert downstream.generation_accepted is frame.accepted
    assert doctor.generation_accepted is frame.visualization_accepted
    assert not hasattr(downstream, "accepted")


def test_doctor_can_draw_provisionally_before_downstream_quality_window() -> None:
    session = RealtimePPGSession(
        session_id="provisional",
        sample_rate_hz=50,
        ppg_scaler=FixedRobustScaler(center=0.0, scale=1.0),
        quality_window_seconds=10.0,
    )
    frame = session.process(np.sin(2 * np.pi * 1.2 * np.arange(150) / 50))

    downstream = to_downstream_ai_frame(frame)
    doctor = to_doctor_visualization_frame(frame)
    assert downstream.generation_accepted is False
    assert downstream.generation_abstention_reason == "quality_warmup"
    assert doctor.generation_accepted is True
    assert any(doctor.pseudo_ecg_valid)
