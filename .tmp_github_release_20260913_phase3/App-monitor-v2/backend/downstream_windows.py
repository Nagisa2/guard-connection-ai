"""前段AIの逐次出力を、後段AI向けの未ラベル窓へ集約する。"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from threading import RLock

class DownstreamWindowError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

@dataclass(frozen=True)
class DownstreamWindowConfig:
    window_seconds: float = 30.0
    stride_seconds: float = 5.0
    max_pending_windows: int = 24

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window and stride seconds must be positive")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride_seconds cannot exceed window_seconds")
        if self.max_pending_windows <= 0:
            raise ValueError("max_pending_windows must be positive")

@dataclass
class _MotionPacket:
    stream_id: str
    sensor_type: str
    units: str
    sample_rate_hz: float
    start_seconds: float
    end_seconds: float
    clock_domain: str
    x: list[float | None]
    y: list[float | None]
    z: list[float | None]

@dataclass
class _AssemblerState:
    session_id: str
    config: DownstreamWindowConfig
    sample_rate_hz: float | None = None
    clock_domain: str | None = None
    buffer_start_seconds: float | None = None
    expected_packet_sequence: int | None = None
    next_window_sequence: int = 0
    ppg: list[float | None] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    pseudo_ecg: list[float | None] = field(default_factory=list)
    pseudo_valid: list[bool] = field(default_factory=list)
    technical_sqi: list[float] = field(default_factory=list)
    plausibility: list[float] = field(default_factory=list)
    source_sequences: list[int] = field(default_factory=list)
    beats: dict[int, dict] = field(default_factory=dict)
    motion: dict[str, deque[_MotionPacket]] = field(default_factory=dict)

    @staticmethod
    def _safe_values(values: list, valid: list[bool]) -> list[float | None]:
        result: list[float | None] = []
        for value, ok in zip(values, valid):
            numeric = None if value is None else float(value)
            result.append(numeric if ok and numeric is not None and isfinite(numeric) else None)
        return result

    def ingest_ppg(self, handoff: dict) -> list[dict]:
        if handoff.get("payload_type") != "front_ai_device_handoff":
            raise DownstreamWindowError("invalid_payload_type", "device handoff payload is required")
        source = handoff["source_device_packet"]
        derived = handoff["derived"]
        packet_sequence = int(source["sequence_number"])
        if self.expected_packet_sequence is None:
            self.expected_packet_sequence = packet_sequence
        if packet_sequence != self.expected_packet_sequence:
            code = "packet_sequence_gap" if packet_sequence > self.expected_packet_sequence else "packet_sequence_regression"
            raise DownstreamWindowError(code, "source packet sequence is not contiguous")
        self.expected_packet_sequence += 1

        count = int(derived.get("sample_count", 0))
        if count == 0:
            return []
        if str(derived.get("session_id")) != self.session_id:
            raise DownstreamWindowError("session_mismatch", "derived session_id does not match route session")
        rate = float(derived["sample_rate_hz"])
        clock_domain = str(source["clock_domain"])
        if self.sample_rate_hz is None:
            self.sample_rate_hz = rate
            self.clock_domain = clock_domain
        elif abs(rate - self.sample_rate_hz) > 1e-9:
            raise DownstreamWindowError("sample_rate_changed", "front AI sample rate changed within a session")
        elif clock_domain != self.clock_domain:
            raise DownstreamWindowError("clock_domain_changed", "PPG clock domain changed within a session")

        ppg = list(derived["ppg"])
        valid = [bool(value) for value in derived["ppg_valid"]]
        pseudo = list(derived.get("pseudo_ecg", [None] * count))
        pseudo_valid = [bool(value) for value in derived.get("pseudo_ecg_valid", [False] * count)]
        if not (len(ppg) == len(valid) == len(pseudo) == len(pseudo_valid) == count):
            raise DownstreamWindowError("sample_count_mismatch", "derived waveform arrays must match sample_count")
        start = float(derived["input_timestamp_start_seconds"])
        if self.buffer_start_seconds is None:
            self.buffer_start_seconds = start
        else:
            expected_start = self.buffer_start_seconds + len(self.ppg) / rate
            if abs(start - expected_start) > 0.5 / rate:
                raise DownstreamWindowError("timestamp_gap", "derived PPG timestamp is not contiguous")

        self.ppg.extend(self._safe_values(ppg, valid))
        self.valid.extend(valid)
        self.pseudo_ecg.extend(self._safe_values(pseudo, pseudo_valid))
        self.pseudo_valid.extend(pseudo_valid)
        self.technical_sqi.extend([float(derived.get("technical_sqi", 0.0))] * count)
        self.plausibility.extend([float(derived.get("pulse_interval_plausibility", 0.0))] * count)
        self.source_sequences.extend([packet_sequence] * count)
        for event in derived.get("beat_events", []):
            event_index = int(event["event_sample_index"])
            self.beats[event_index] = deepcopy(event)
        return self._emit_ready_windows()

    def ingest_motion(self, payload: dict) -> None:
        if payload.get("message_type") != "front_ai_auxiliary_stream":
            raise DownstreamWindowError("invalid_payload_type", "auxiliary stream payload is required")
        if str(payload["session_id"]) != self.session_id:
            raise DownstreamWindowError("session_mismatch", "motion session_id does not match route session")
        packet = _MotionPacket(
            stream_id=str(payload["stream_id"]),
            sensor_type=str(payload["sensor_type"]),
            units=str(payload["units"]),
            sample_rate_hz=float(payload["configured_sample_rate_hz"]),
            start_seconds=int(payload["device_timestamp_start_ns"]) / 1_000_000_000,
            end_seconds=int(payload["device_timestamp_end_ns"]) / 1_000_000_000,
            clock_domain=str(payload["clock_domain"]),
            x=list(payload["x"]),
            y=list(payload["y"]),
            z=list(payload["z"]),
        )
        packets = self.motion.setdefault(packet.stream_id, deque())
        packets.append(packet)
        newest = packet.end_seconds
        retention = max(2 * self.config.window_seconds, 60.0)
        while packets and packets[0].end_seconds < newest - retention:
            packets.popleft()

    def _motion_for_window(self, start: float, end: float) -> tuple[list[dict], list[dict]]:
        aligned: list[dict] = []
        unmapped: list[dict] = []
        for stream_id, packets in sorted(self.motion.items()):
            if not packets:
                continue
            latest = packets[-1]
            if latest.clock_domain != self.clock_domain:
                unmapped.append({"stream_id": stream_id, "clock_domain": latest.clock_domain, "reason": "clock_domain_mismatch"})
                continue
            values_x: list[float | None] = []
            values_y: list[float | None] = []
            values_z: list[float | None] = []
            first_time: float | None = None
            for packet in packets:
                for index, (x, y, z) in enumerate(zip(packet.x, packet.y, packet.z)):
                    timestamp = packet.start_seconds + index / packet.sample_rate_hz
                    if start <= timestamp < end:
                        if first_time is None:
                            first_time = timestamp
                        values_x.append(x)
                        values_y.append(y)
                        values_z.append(z)
            if values_x:
                aligned.append({
                    "stream_id": stream_id,
                    "sensor_type": latest.sensor_type,
                    "units": latest.units,
                    "fs_hz": latest.sample_rate_hz,
                    "timestamp_start_seconds": first_time,
                    "x": values_x,
                    "y": values_y,
                    "z": values_z,
                    "alignment": "same_clock_domain",
                    "coverage": min(1.0, len(values_x) / (self.config.window_seconds * latest.sample_rate_hz)),
                })
        return aligned, unmapped

    def _emit_ready_windows(self) -> list[dict]:
        assert self.sample_rate_hz is not None
        window_count = round(self.config.window_seconds * self.sample_rate_hz)
        stride_count = round(self.config.stride_seconds * self.sample_rate_hz)
        emitted: list[dict] = []
        while len(self.ppg) >= window_count:
            assert self.buffer_start_seconds is not None
            start = self.buffer_start_seconds
            end = start + self.config.window_seconds
            events = sorted(
                (
                    event for event in self.beats.values()
                    if start <= float(event["estimated_r_timestamp_seconds"]) < end
                ),
                key=lambda item: float(item["estimated_r_timestamp_seconds"]),
            )
            motion, unmapped = self._motion_for_window(start, end)
            valid = self.valid[:window_count]
            pseudo_valid = self.pseudo_valid[:window_count]
            emitted.append({
                "message_type": "front_ai_inference_window",
                "schema_version": "front_ai_window_v1",
                "session_id": self.session_id,
                "input_sequence_id": self.next_window_sequence,
                "start_seconds": start,
                "end_seconds": end,
                "window_seconds": self.config.window_seconds,
                "stride_seconds": self.config.stride_seconds,
                "primary_signal": "ppg",
                "ppg": {
                    "name": "PPG",
                    "fs_hz": self.sample_rate_hz,
                    "unit": "normalized_front_ai_input",
                    "values": self.ppg[:window_count],
                    "valid_mask": valid,
                },
                "beats": {
                    "source": "ppg",
                    "timing_reference": "estimated_r_timestamp_seconds",
                    "times_sec": [float(event["estimated_r_timestamp_seconds"]) for event in events],
                    "amplitudes": [float(event["amplitude"]) for event in events],
                    "event_sample_indices": [int(event["event_sample_index"]) for event in events],
                },
                "quality": {
                    "technical_sqi": sum(self.technical_sqi[:window_count]) / window_count,
                    "signal_coverage": sum(valid) / window_count,
                    "pulse_interval_plausibility": sum(self.plausibility[:window_count]) / window_count,
                },
                "auxiliary_pseudo_ecg": {
                    "waveform_type": "ppg_derived_pseudo_ecg",
                    "diagnostic_ecg": False,
                    "measurement_ui_enabled": False,
                    "values": self.pseudo_ecg[:window_count],
                    "valid_mask": pseudo_valid,
                },
                "motion_streams": motion,
                "unmapped_motion_streams": unmapped,
                "source_packet_sequence_start": self.source_sequences[0],
                "source_packet_sequence_end": self.source_sequences[window_count - 1],
            })
            self.next_window_sequence += 1
            for values in (
                self.ppg,
                self.valid,
                self.pseudo_ecg,
                self.pseudo_valid,
                self.technical_sqi,
                self.plausibility,
                self.source_sequences,
            ):
                del values[:stride_count]
            self.buffer_start_seconds += self.config.stride_seconds
            self.beats = {
                index: event for index, event in self.beats.items()
                if float(event["estimated_r_timestamp_seconds"]) >= self.buffer_start_seconds
            }
        return emitted

@dataclass
class _BrokerSession:
    assembler: _AssemblerState
    pending: deque[dict] = field(default_factory=deque)
    last_acked_sequence: int = -1
    dropped_window_count: int = 0

class DownstreamWindowBroker:
    """ACKされるまで窓を保持し、再接続時に同じ窓を再配信する。"""

    def __init__(self, config: DownstreamWindowConfig | None = None) -> None:
        self.config = config or DownstreamWindowConfig()
        self._sessions: dict[str, _BrokerSession] = {}
        self._lock = RLock()

    def _state(self, session_id: str) -> _BrokerSession:
        return self._sessions.setdefault(
            session_id,
            _BrokerSession(_AssemblerState(session_id=session_id, config=self.config)),
        )

    def ingest_ppg(self, session_id: str, handoff: dict) -> list[dict]:
        with self._lock:
            state = self._state(session_id)
            windows = state.assembler.ingest_ppg(handoff)
            for window in windows:
                if len(state.pending) >= self.config.max_pending_windows:
                    state.pending.popleft()
                    state.dropped_window_count += 1
                state.pending.append(window)
            return deepcopy(windows)

    def ingest_motion(self, session_id: str, payload: dict) -> None:
        with self._lock:
            self._state(session_id).assembler.ingest_motion(payload)

    def next_window(self, session_id: str) -> dict | None:
        with self._lock:
            state = self._sessions.get(session_id)
            return None if state is None or not state.pending else deepcopy(state.pending[0])

    def acknowledge(self, session_id: str, sequence: int) -> bool:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise DownstreamWindowError("window_session_not_found", "window session was not found")
            if sequence <= state.last_acked_sequence:
                return False
            if not state.pending:
                raise DownstreamWindowError("window_not_pending", "no window is pending acknowledgement")
            expected = int(state.pending[0]["input_sequence_id"])
            if sequence != expected:
                raise DownstreamWindowError("window_ack_out_of_order", f"expected acknowledgement for window {expected}")
            state.pending.popleft()
            state.last_acked_sequence = sequence
            return True

    def end_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def status_snapshot(self) -> dict:
        with self._lock:
            sessions = []
            for session_id, state in sorted(self._sessions.items()):
                sessions.append({
                    "session_id": session_id,
                    "pending_window_count": len(state.pending),
                    "next_pending_sequence": None if not state.pending else state.pending[0]["input_sequence_id"],
                    "last_acked_sequence": state.last_acked_sequence,
                    "dropped_window_count": state.dropped_window_count,
                })
            return {"active_session_count": len(sessions), "sessions": sessions}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

registry = DownstreamWindowBroker()
