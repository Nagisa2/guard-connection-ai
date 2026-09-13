from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import RLock
from time import time
from typing import Protocol
from uuid import uuid4

import numpy as np

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.schemas.device_ppg import DevicePPGPacket
from guard_connection_ai.schemas.front_ai_outputs import to_downstream_ai_frame
from guard_connection_ai.schemas.realtime_waveform import RealtimeWaveformFrame
from guard_connection_ai.streaming.device_ppg_ingress import (
    AdaptedPPGChunk,
    DevicePPGIngressError,
    StatefulCausalADCNormalizer,
    StatefulDevicePPGIngress,
)
from guard_connection_ai.streaming.realtime_session import RealtimePPGSession


class RealtimeProtocolError(ValueError):
    """A stable, transport-independent realtime protocol error."""

    def __init__(self, code: str, message: str, *, expected_sequence_number: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.expected_sequence_number = expected_sequence_number

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"code": self.code, "message": self.message}
        if self.expected_sequence_number is not None:
            result["expected_sequence_number"] = self.expected_sequence_number
        return result


@dataclass(frozen=True)
class RealtimeSessionConfig:
    user_id: str
    sample_rate_hz: float
    scaler_center: float
    scaler_scale: float
    session_id: str | None = None
    pulse_arrival_seconds: float = 0.2
    display_delay_seconds: float = 0.5
    quality_window_seconds: float = 10.0
    generation_profile_id: str = "interval_template"
    normalize_device_adc: bool = False
    source_mode: str = "live_device"
    demo_scenario_id: str | None = None


class RealtimePipeline(Protocol):
    sample_rate_hz: float

    def process(
        self,
        ppg_chunk: np.ndarray,
        *,
        input_timestamp_start_seconds: float | None = None,
    ) -> RealtimeWaveformFrame: ...


PipelineFactory = Callable[[RealtimeSessionConfig, str], RealtimePipeline]


@dataclass
class _ManagedSession:
    user_id: str
    pipeline: RealtimePipeline
    source_mode: str = "live_device"
    demo_scenario_id: str | None = None
    expected_sequence_number: int = 0
    sample_count: int = 0
    timeline_origin_seconds: float | None = None
    input_mode: str | None = None
    device_ingress: StatefulDevicePPGIngress | None = None
    device_normalizer: StatefulCausalADCNormalizer | None = None
    created_at_seconds: float = 0.0
    last_input_at_seconds: float | None = None
    accepted_packet_count: int = 0


@dataclass(frozen=True)
class DevicePacketProcessingResult:
    packet: DevicePPGPacket
    adapted: AdaptedPPGChunk
    waveform_frame: RealtimeWaveformFrame | None

    def downstream_payload(self) -> dict[str, object]:
        derived = (
            to_downstream_ai_frame(self.waveform_frame).to_dict()
            if self.waveform_frame is not None
            else {
                "schema_version": "1.1",
                "payload_type": "front_ai_downstream_input",
                "session_id": self.packet.stream_id,
                "sequence_number": self.packet.sequence_number,
                "sample_rate_hz": self.adapted.target_sample_rate_hz,
                "input_timestamp_start_seconds": (
                    self.adapted.input_timestamp_start_seconds
                ),
                "sample_count": 0,
                "ppg": [],
                "ppg_valid": [],
                "pseudo_ecg": [],
                "pseudo_ecg_valid": [],
                "technical_sqi": 0.0,
                "signal_coverage": 0.0,
                "pulse_interval_plausibility": 0.0,
                "beat_events": [],
                "interval_window_count": 0,
                "pulse_event_count_total": 0,
                "rejected_interval_count_total": 0,
                "generation_accepted": False,
                "generation_abstention_reason": "waiting_for_resampled_sample",
            }
        )
        return {
            "schema_version": "1.0",
            "payload_type": "front_ai_device_handoff",
            "source_device_packet": asdict(self.packet),
            "derived": derived,
        }


class RealtimeSessionRegistry:
    """Own independent state per session and validate packet order before processing."""

    schema_version = "1.0"

    def __init__(
        self, *, pipeline_factories: dict[str, PipelineFactory] | None = None
    ) -> None:
        self._sessions: dict[str, _ManagedSession] = {}
        self._lock = RLock()
        self._pipeline_factories = dict(pipeline_factories or {})

    @staticmethod
    def _interval_template_pipeline(
        config: RealtimeSessionConfig, session_id: str
    ) -> RealtimePPGSession:
        return RealtimePPGSession(
            session_id=session_id,
            sample_rate_hz=config.sample_rate_hz,
            ppg_scaler=FixedRobustScaler(
                center=config.scaler_center, scale=config.scaler_scale
            ),
            pulse_arrival_seconds=config.pulse_arrival_seconds,
            display_delay_seconds=config.display_delay_seconds,
            quality_window_seconds=config.quality_window_seconds,
        )

    def create(self, config: RealtimeSessionConfig) -> dict[str, object]:
        session_id = config.session_id or str(uuid4())
        if not config.user_id:
            raise RealtimeProtocolError("invalid_session", "user_id must not be empty.")
        if config.source_mode not in {
            "live_device",
            "recorded_demo_replay",
            "synthetic_demo",
        }:
            raise RealtimeProtocolError("invalid_session", "unsupported source_mode.")
        if config.source_mode in {"recorded_demo_replay", "synthetic_demo"} and not config.demo_scenario_id:
            raise RealtimeProtocolError(
                "invalid_session", "recorded demo replay requires demo_scenario_id."
            )
        if config.source_mode == "live_device" and config.demo_scenario_id is not None:
            raise RealtimeProtocolError(
                "invalid_session", "live device input must not declare a demo scenario."
            )
        with self._lock:
            if session_id in self._sessions:
                raise RealtimeProtocolError("session_exists", "session_id already exists.")
            try:
                if config.generation_profile_id == "interval_template":
                    factory = self._interval_template_pipeline
                else:
                    factory = self._pipeline_factories.get(
                        config.generation_profile_id
                    )
                    if factory is None:
                        raise ValueError(
                            "unknown generation_profile_id: "
                            f"{config.generation_profile_id}"
                        )
                pipeline = factory(config, session_id)
            except ValueError as exc:
                raise RealtimeProtocolError("invalid_session", str(exc)) from exc
            self._sessions[session_id] = _ManagedSession(
                config.user_id,
                pipeline,
                source_mode=config.source_mode,
                demo_scenario_id=config.demo_scenario_id,
                device_ingress=StatefulDevicePPGIngress(
                    target_sample_rate_hz=config.sample_rate_hz
                ),
                device_normalizer=(
                    StatefulCausalADCNormalizer(
                        sample_rate_hz=config.sample_rate_hz
                    )
                    if config.normalize_device_adc
                    else None
                ),
                created_at_seconds=time(),
            )
        return {
            "schema_version": self.schema_version,
            "session_id": session_id,
            "user_id": config.user_id,
            "expected_sequence_number": 0,
            "generation_profile_id": config.generation_profile_id,
            "source_mode": config.source_mode,
            "demo_scenario_id": config.demo_scenario_id,
        }

    def end(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise RealtimeProtocolError("session_not_found", "session_id was not found.")

    def user_id_for(self, session_id: str) -> str:
        with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                raise RealtimeProtocolError("session_not_found", "session_id was not found.")
            return managed.user_id

    def source_provenance_for(self, session_id: str) -> dict[str, object]:
        with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                raise RealtimeProtocolError("session_not_found", "session_id was not found.")
            return {
                "source_mode": managed.source_mode,
                "demo_scenario_id": managed.demo_scenario_id,
            }

    def process_chunk(
        self,
        session_id: str,
        *,
        sequence_number: int,
        input_timestamp_start_seconds: float,
        ppg: list[float | None],
    ) -> RealtimeWaveformFrame:
        with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                raise RealtimeProtocolError("session_not_found", "session_id was not found.")
            if managed.input_mode not in {None, "single_ppg"}:
                raise RealtimeProtocolError(
                    "input_mode_conflict",
                    "this session already receives device PPG packets.",
                    expected_sequence_number=managed.expected_sequence_number,
                )
            managed.input_mode = "single_ppg"
            expected_sequence = managed.expected_sequence_number
            if sequence_number < expected_sequence:
                code = "duplicate_packet" if sequence_number == expected_sequence - 1 else "out_of_order"
                raise RealtimeProtocolError(
                    code,
                    "packet sequence is older than the current session state.",
                    expected_sequence_number=expected_sequence,
                )
            if sequence_number > expected_sequence:
                raise RealtimeProtocolError(
                    "sequence_gap",
                    "packet loss detected; recreate the session or resend the missing packet.",
                    expected_sequence_number=expected_sequence,
                )
            if not ppg:
                raise RealtimeProtocolError("invalid_chunk", "ppg must not be empty.")
            values = np.asarray([np.nan if value is None else value for value in ppg], dtype=np.float64)
            if values.ndim != 1:
                raise RealtimeProtocolError("invalid_chunk", "ppg must be a one-dimensional array.")
            timestamp = float(input_timestamp_start_seconds)
            if not np.isfinite(timestamp):
                raise RealtimeProtocolError("invalid_timestamp", "input timestamp must be finite.")
            if managed.timeline_origin_seconds is None:
                managed.timeline_origin_seconds = timestamp
            expected_timestamp = (
                managed.timeline_origin_seconds
                + managed.sample_count / managed.pipeline.sample_rate_hz
            )
            tolerance = 0.5 / managed.pipeline.sample_rate_hz
            if abs(timestamp - expected_timestamp) > tolerance:
                raise RealtimeProtocolError(
                    "timestamp_gap",
                    "input timestamp is not contiguous with accepted samples.",
                    expected_sequence_number=expected_sequence,
                )
            try:
                frame = managed.pipeline.process(
                    values,
                    input_timestamp_start_seconds=timestamp,
                )
            except ValueError as exc:
                raise RealtimeProtocolError("invalid_chunk", str(exc)) from exc
            managed.expected_sequence_number += 1
            managed.sample_count += values.size
            managed.accepted_packet_count += 1
            managed.last_input_at_seconds = time()
            return frame

    def process_device_packet(
        self, session_id: str, packet: DevicePPGPacket
    ) -> DevicePacketProcessingResult:
        with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                raise RealtimeProtocolError("session_not_found", "session_id was not found.")
            if managed.input_mode not in {None, "device_ppg"}:
                raise RealtimeProtocolError(
                    "input_mode_conflict",
                    "this session already receives single-channel PPG chunks.",
                )
            managed.input_mode = "device_ppg"
            if (
                packet.source_mode != managed.source_mode
                or packet.demo_scenario_id != managed.demo_scenario_id
            ):
                raise RealtimeProtocolError(
                    "source_provenance_mismatch",
                    "packet provenance does not match the session provenance.",
                )
            if managed.device_ingress is None:
                raise RuntimeError("device ingress was not initialized.")
            try:
                adapted = managed.device_ingress.process(packet)
            except DevicePPGIngressError as error:
                raise RealtimeProtocolError(
                    error.code,
                    str(error),
                    expected_sequence_number=error.expected_sequence_number,
                ) from error
            waveform_frame = None
            if adapted.ppg.size:
                pipeline_ppg = (
                    managed.device_normalizer.process(adapted.ppg)
                    if managed.device_normalizer is not None
                    else adapted.ppg
                )
                waveform_frame = managed.pipeline.process(
                    pipeline_ppg,
                    input_timestamp_start_seconds=adapted.input_timestamp_start_seconds,
                )
            managed.sample_count += int(adapted.ppg.size)
            managed.accepted_packet_count += 1
            managed.last_input_at_seconds = time()
            return DevicePacketProcessingResult(packet, adapted, waveform_frame)

    def status_snapshot(self) -> dict[str, object]:
        """Return operational metadata without exposing waveform samples."""
        now = time()
        with self._lock:
            sessions = []
            for session_id, managed in sorted(self._sessions.items()):
                age = (
                    None
                    if managed.last_input_at_seconds is None
                    else max(0.0, now - managed.last_input_at_seconds)
                )
                sessions.append({
                    "session_id": session_id,
                    "user_id": managed.user_id,
                    "source_mode": managed.source_mode,
                    "demo_scenario_id": managed.demo_scenario_id,
                    "input_mode": managed.input_mode,
                    "created_at_seconds": managed.created_at_seconds,
                    "last_input_at_seconds": managed.last_input_at_seconds,
                    "last_input_age_seconds": age,
                    "accepted_packet_count": managed.accepted_packet_count,
                    "accepted_sample_count": managed.sample_count,
                    "expected_sequence_number": managed.expected_sequence_number,
                })
            return {
                "schema_version": self.schema_version,
                "active_session_count": len(sessions),
                "sessions": sessions,
            }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
