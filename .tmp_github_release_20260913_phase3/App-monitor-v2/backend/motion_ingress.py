"""PPGと異なる周波数の加速度・ジャイロを独立状態で受け入れる。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import time

class MotionIngressError(ValueError):
    def __init__(self, code: str, message: str, *, expected_sequence_number: int):
        super().__init__(message)
        self.code = code
        self.expected_sequence_number = expected_sequence_number

@dataclass
class _StreamState:
    expected_sequence_number: int = 0
    last_timestamp_end_ns: int | None = None
    sample_rate_hz: float | None = None
    clock_domain: str | None = None
    accepted_sample_count: int = 0
    last_input_at_seconds: float | None = None

class MotionIngressRegistry:
    def __init__(self) -> None:
        self._streams: dict[tuple[str, str], _StreamState] = {}
        self._lock = RLock()

    def accept(self, session_id: str, packet: dict) -> dict:
        stream_id = str(packet["stream_id"])
        key = (session_id, stream_id)
        sequence = int(packet["sequence_number"])
        sample_rate = float(packet["configured_sample_rate_hz"])
        timestamp_end_ns = int(packet["device_timestamp_end_ns"])
        sample_count = len(packet["x"])
        clock_domain = str(packet["clock_domain"])
        with self._lock:
            state = self._streams.setdefault(key, _StreamState())
            expected = state.expected_sequence_number
            if sequence < expected:
                code = "duplicate_packet" if sequence == expected - 1 else "out_of_order"
                raise MotionIngressError(
                    code,
                    "motion packet sequence is older than accepted state",
                    expected_sequence_number=expected,
                )
            if sequence > expected:
                raise MotionIngressError(
                    "sequence_gap",
                    "one or more motion packets are missing",
                    expected_sequence_number=expected,
                )
            if state.clock_domain is not None and state.clock_domain != clock_domain:
                raise MotionIngressError(
                    "clock_domain_changed",
                    "clock domain cannot change within a motion stream",
                    expected_sequence_number=expected,
                )
            if state.sample_rate_hz is not None and abs(state.sample_rate_hz - sample_rate) > 1e-9:
                raise MotionIngressError(
                    "sample_rate_changed",
                    "configured sample rate cannot change within a motion stream",
                    expected_sequence_number=expected,
                )
            timestamp_start_ns = round(
                timestamp_end_ns - (sample_count - 1) * 1_000_000_000 / sample_rate
            )
            if state.last_timestamp_end_ns is not None:
                expected_start_ns = round(
                    state.last_timestamp_end_ns + 1_000_000_000 / sample_rate
                )
                tolerance_ns = 0.5 * 1_000_000_000 / sample_rate
                if abs(timestamp_start_ns - expected_start_ns) > tolerance_ns:
                    raise MotionIngressError(
                        "timestamp_gap",
                        "motion timestamp is not contiguous with accepted state",
                        expected_sequence_number=expected,
                    )
            state.expected_sequence_number += 1
            state.last_timestamp_end_ns = timestamp_end_ns
            state.sample_rate_hz = sample_rate
            state.clock_domain = clock_domain
            state.accepted_sample_count += sample_count
            state.last_input_at_seconds = time()
            return {
                "message_type": "front_ai_auxiliary_stream",
                "schema_version": "1.0",
                "session_id": session_id,
                "stream_id": stream_id,
                "sequence_number": sequence,
                "sensor_type": packet["sensor_type"],
                "units": packet["units"],
                "configured_sample_rate_hz": sample_rate,
                "device_timestamp_start_ns": timestamp_start_ns,
                "device_timestamp_end_ns": timestamp_end_ns,
                "clock_domain": clock_domain,
                "sample_count": sample_count,
                "x": packet["x"],
                "y": packet["y"],
                "z": packet["z"],
                "sensor_disconnected": packet["sensor_disconnected"],
                "device_id_hash": packet.get("device_id_hash"),
                "device_model": packet.get("device_model"),
            }

    def end_session(self, session_id: str) -> None:
        with self._lock:
            for key in [key for key in self._streams if key[0] == session_id]:
                self._streams.pop(key, None)

    def status_snapshot(self) -> dict:
        now = time()
        with self._lock:
            items = []
            for (session_id, stream_id), state in sorted(self._streams.items()):
                items.append({
                    "session_id": session_id,
                    "stream_id": stream_id,
                    "clock_domain": state.clock_domain,
                    "sample_rate_hz": state.sample_rate_hz,
                    "expected_sequence_number": state.expected_sequence_number,
                    "accepted_sample_count": state.accepted_sample_count,
                    "last_input_age_seconds": (
                        None
                        if state.last_input_at_seconds is None
                        else max(0.0, now - state.last_input_at_seconds)
                    ),
                })
            return {"active_stream_count": len(items), "streams": items}

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()

registry = MotionIngressRegistry()
