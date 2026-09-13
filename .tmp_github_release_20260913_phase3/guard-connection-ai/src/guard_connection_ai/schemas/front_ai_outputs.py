from __future__ import annotations

from dataclasses import asdict, dataclass

from guard_connection_ai.schemas.realtime_waveform import (
    BeatEvent,
    RealtimeWaveformFrame,
)

PSEUDO_ECG_LIMITATIONS = (
    "PPG由来疑似ECG・診断用ではない",
    "P波、細動波、PR、QRS幅、QT、STの測定には使用できない",
    "実測ECGまたは確定診断として扱わない",
)


@dataclass(frozen=True)
class DownstreamAIInputFrame:
    """前段AIから後段AIへ渡す、診断結果を含まない時系列データ。"""

    schema_version: str
    payload_type: str
    session_id: str
    sequence_number: int
    sample_rate_hz: float
    input_timestamp_start_seconds: float
    sample_count: int
    ppg: list[float]
    ppg_valid: list[bool]
    pseudo_ecg: list[float]
    pseudo_ecg_valid: list[bool]
    pseudo_ecg_timestamp_start_seconds: float
    ppg_sqi: float
    technical_sqi: float
    signal_coverage: float
    pulse_interval_plausibility: float
    generation_accepted: bool
    generation_abstention_reason: str | None
    latest_pulse_interval_ms: float | None
    heart_rate_bpm: float | None
    interval_cv: float | None
    rmssd_ms: float | None
    beat_events: list[BeatEvent]
    interval_window_count: int
    pulse_event_count_total: int
    rejected_interval_count_total: int
    interval_count: int
    estimated_pat_ms: float | None
    display_delay_ms: float
    generation_mode: str
    morphology_source: str
    timing_confidence: float | None
    morphology_confidence: float | None
    repolarization_duration_prior_ms: float | None
    p_wave_generated: bool
    qt_measurement_supported: bool
    pq_pr_measurement_supported: bool
    source_signal_type: str = "measured_ppg"
    pseudo_ecg_type: str = "ppg_derived_pseudo_ecg"
    diagnostic_ecg: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorVisualizationFrame:
    """医師用アプリで同期描画するための表示専用フレーム。"""

    schema_version: str
    payload_type: str
    session_id: str
    sequence_number: int
    sample_rate_hz: float
    input_timestamp_start_seconds: float
    display_timestamp_start_seconds: float
    ppg: list[float]
    ppg_valid: list[bool]
    pseudo_ecg: list[float]
    pseudo_ecg_valid: list[bool]
    waveform_type: str
    diagnostic_ecg: bool
    display_label: str
    measurement_ui_enabled: bool
    limitations: tuple[str, ...]
    ppg_sqi: float
    technical_sqi: float
    signal_coverage: float
    pulse_interval_plausibility: float
    generation_accepted: bool
    generation_abstention_reason: str | None
    latest_pulse_interval_ms: float | None
    heart_rate_bpm: float | None
    interval_cv: float | None
    rmssd_ms: float | None
    estimated_pat_ms: float | None
    display_delay_ms: float
    generation_mode: str
    morphology_source: str
    timing_confidence: float | None
    morphology_confidence: float | None
    repolarization_duration_prior_ms: float | None
    p_wave_generated: bool
    qt_measurement_supported: bool
    pq_pr_measurement_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def to_downstream_ai_frame(frame: RealtimeWaveformFrame) -> DownstreamAIInputFrame:
    return DownstreamAIInputFrame(
        schema_version=frame.schema_version,
        payload_type="front_ai_downstream_input",
        session_id=frame.session_id,
        sequence_number=frame.sequence_number,
        sample_rate_hz=frame.sample_rate_hz,
        input_timestamp_start_seconds=frame.input_timestamp_start_seconds,
        sample_count=len(frame.ppg),
        ppg=frame.ppg,
        ppg_valid=frame.ppg_valid,
        pseudo_ecg=frame.pseudo_ecg,
        pseudo_ecg_valid=frame.pseudo_ecg_valid,
        pseudo_ecg_timestamp_start_seconds=frame.display_timestamp_start_seconds,
        ppg_sqi=frame.ppg_sqi,
        technical_sqi=frame.technical_sqi,
        signal_coverage=frame.signal_coverage,
        pulse_interval_plausibility=frame.pulse_interval_plausibility,
        generation_accepted=frame.accepted,
        generation_abstention_reason=frame.abstention_reason,
        latest_pulse_interval_ms=frame.latest_pulse_interval_ms,
        heart_rate_bpm=frame.heart_rate_bpm,
        interval_cv=frame.interval_cv,
        rmssd_ms=frame.rmssd_ms,
        beat_events=frame.beat_events,
        interval_window_count=frame.interval_window_count,
        pulse_event_count_total=frame.pulse_event_count_total,
        rejected_interval_count_total=frame.rejected_interval_count_total,
        interval_count=frame.interval_count,
        estimated_pat_ms=frame.estimated_pat_ms,
        display_delay_ms=frame.display_delay_ms,
        generation_mode=frame.generation_mode,
        morphology_source=frame.morphology_source,
        timing_confidence=frame.timing_confidence,
        morphology_confidence=frame.morphology_confidence,
        repolarization_duration_prior_ms=frame.repolarization_duration_prior_ms,
        p_wave_generated=frame.p_wave_generated,
        qt_measurement_supported=frame.qt_measurement_supported,
        pq_pr_measurement_supported=frame.pq_pr_measurement_supported,
    )


def to_doctor_visualization_frame(
    frame: RealtimeWaveformFrame,
) -> DoctorVisualizationFrame:
    return DoctorVisualizationFrame(
        schema_version=frame.schema_version,
        payload_type="doctor_waveform_visualization",
        session_id=frame.session_id,
        sequence_number=frame.sequence_number,
        sample_rate_hz=frame.sample_rate_hz,
        input_timestamp_start_seconds=frame.input_timestamp_start_seconds,
        display_timestamp_start_seconds=frame.display_timestamp_start_seconds,
        ppg=frame.ppg,
        ppg_valid=frame.ppg_valid,
        pseudo_ecg=frame.pseudo_ecg,
        pseudo_ecg_valid=frame.pseudo_ecg_valid,
        waveform_type=frame.waveform_type,
        diagnostic_ecg=frame.diagnostic_ecg,
        display_label=PSEUDO_ECG_LIMITATIONS[0],
        measurement_ui_enabled=False,
        limitations=PSEUDO_ECG_LIMITATIONS,
        ppg_sqi=frame.ppg_sqi,
        technical_sqi=frame.technical_sqi,
        signal_coverage=frame.signal_coverage,
        pulse_interval_plausibility=frame.pulse_interval_plausibility,
        generation_accepted=frame.visualization_accepted,
        generation_abstention_reason=frame.visualization_abstention_reason,
        latest_pulse_interval_ms=frame.latest_pulse_interval_ms,
        heart_rate_bpm=frame.heart_rate_bpm,
        interval_cv=frame.interval_cv,
        rmssd_ms=frame.rmssd_ms,
        estimated_pat_ms=frame.estimated_pat_ms,
        display_delay_ms=frame.display_delay_ms,
        generation_mode=frame.generation_mode,
        morphology_source=frame.morphology_source,
        timing_confidence=frame.timing_confidence,
        morphology_confidence=frame.morphology_confidence,
        repolarization_duration_prior_ms=frame.repolarization_duration_prior_ms,
        p_wave_generated=frame.p_wave_generated,
        qt_measurement_supported=frame.qt_measurement_supported,
        pq_pr_measurement_supported=frame.pq_pr_measurement_supported,
    )
