from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BeatEvent:
    """One newly emitted, non-diagnostic timing event in a realtime frame."""

    event_sample_index: int
    detected_at_input_timestamp_seconds: float
    estimated_r_timestamp_seconds: float
    event_type: str
    source: str
    confidence: float | None = None
    amplitude: float | None = None


@dataclass(frozen=True)
class RealtimeWaveformFrame:
    schema_version: str
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
    ppg_sqi: float
    technical_sqi: float
    signal_coverage: float
    pulse_interval_plausibility: float
    accepted: bool
    abstention_reason: str | None
    visualization_accepted: bool
    visualization_abstention_reason: str | None
    estimated_pat_ms: float | None
    display_delay_ms: float
    latest_pulse_interval_ms: float | None
    heart_rate_bpm: float | None
    interval_cv: float | None
    rmssd_ms: float | None
    beat_events: list[BeatEvent]
    interval_window_count: int
    pulse_event_count_total: int
    rejected_interval_count_total: int
    interval_count: int
    generation_mode: str = "interval_template"
    morphology_source: str = "fixed_qrs_t_template"
    timing_confidence: float | None = None
    morphology_confidence: float | None = None
    repolarization_duration_prior_ms: float | None = None
    p_wave_generated: bool = False
    qt_measurement_supported: bool = False
    pq_pr_measurement_supported: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
