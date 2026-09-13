"""後段AIの窓単位出力を順序検証して保持する。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from time import time

class DownstreamInferenceError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        expected_sequence_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.expected_sequence_number = expected_sequence_number

@dataclass
class _SessionState:
    expected_sequence_number: int = 0
    latest_payload: dict | None = None
    latest_digest: str | None = None
    latest_input_sequence_id: int | None = None
    latest_window_end_seconds: float | None = None
    observed_seconds: float = 0.0
    valid_decision_seconds: float = 0.0
    af_suspected_seconds: float = 0.0
    consecutive_undecidable_seconds: float = 0.0
    undecidable_seconds_by_reason: dict[str, float] | None = None
    last_accepted_at_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.undecidable_seconds_by_reason is None:
            self.undecidable_seconds_by_reason = {}

def _payload_digest(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()

class DownstreamInferenceRegistry:
    """後段AIの状態をsessionごとに分離するインメモリ受信境界。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, _SessionState] = {}

    def accept(self, payload: dict) -> tuple[dict, bool]:
        session_id = str(payload["session_id"])
        sequence = int(payload["inference_sequence_number"])
        digest = _payload_digest(payload)
        with self._lock:
            state = self._sessions.setdefault(session_id, _SessionState())
            expected = state.expected_sequence_number
            if sequence < expected:
                if sequence == expected - 1 and digest == state.latest_digest:
                    return deepcopy(state.latest_payload), False
                code = "duplicate_inference" if sequence == expected - 1 else "out_of_order"
                raise DownstreamInferenceError(
                    code,
                    "inference sequence is older than the accepted state",
                    expected_sequence_number=expected,
                )
            if sequence > expected:
                raise DownstreamInferenceError(
                    "sequence_gap",
                    "one or more downstream inference results are missing",
                    expected_sequence_number=expected,
                )
            window = payload["window"]
            input_sequence_id = int(window["input_sequence_id"])
            window_end = float(window["end_seconds"])
            previous_window_end = state.latest_window_end_seconds
            if (
                state.latest_input_sequence_id is not None
                and input_sequence_id <= state.latest_input_sequence_id
            ):
                raise DownstreamInferenceError(
                    "window_regression",
                    "input window sequence must increase within a session",
                    expected_sequence_number=expected,
                )
            if (
                previous_window_end is not None
                and window_end <= previous_window_end
            ):
                raise DownstreamInferenceError(
                    "window_regression",
                    "input window time must increase within a session",
                    expected_sequence_number=expected,
                )
            contribution = (
                float(window["window_seconds"])
                if previous_window_end is None
                else window_end - previous_window_end
            )
            if (
                previous_window_end is not None
                and abs(contribution - float(window["stride_seconds"])) > 1e-6
            ):
                raise DownstreamInferenceError(
                    "window_gap",
                    "window time advance does not match stride_seconds",
                    expected_sequence_number=expected,
                )
            state.latest_payload = deepcopy(payload)
            state.latest_digest = digest
            state.latest_input_sequence_id = input_sequence_id
            state.latest_window_end_seconds = window_end
            state.expected_sequence_number += 1
            state.last_accepted_at_seconds = time()
            state.observed_seconds += contribution
            if payload["decidable"]:
                state.valid_decision_seconds += contribution
                state.consecutive_undecidable_seconds = 0.0
                if payload["decision"] == "af_suspected":
                    state.af_suspected_seconds += contribution
            else:
                state.consecutive_undecidable_seconds += contribution
                reason = str(payload["abstention_reason"])
                state.undecidable_seconds_by_reason[reason] = (
                    state.undecidable_seconds_by_reason.get(reason, 0.0)
                    + contribution
                )
            return deepcopy(payload), True

    def latest(self, session_id: str) -> dict:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.latest_payload is None:
                raise DownstreamInferenceError(
                    "inference_not_found", "no downstream inference has been accepted"
                )
            return deepcopy(state.latest_payload)

    def summary(self, session_id: str) -> dict:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.latest_payload is None:
                raise DownstreamInferenceError(
                    "inference_not_found", "no downstream inference has been accepted"
                )
            latest = state.latest_payload
            if latest["decision"] == "undecidable":
                analysis_state = "undecidable"
            elif latest["decision"] == "af_suspected":
                analysis_state = "af_suspected"
            else:
                analysis_state = "monitoring"
            valid_ratio = (
                state.valid_decision_seconds / state.observed_seconds
                if state.observed_seconds > 0
                else 0.0
            )
            af_ratio = (
                state.af_suspected_seconds / state.valid_decision_seconds
                if state.valid_decision_seconds > 0
                else None
            )
            return {
                "analysis_state": analysis_state,
                "latest_decision": latest["decision"],
                "decidable": latest["decidable"],
                "last_abstention_reason": latest["abstention_reason"],
                "observed_seconds": state.observed_seconds,
                "valid_decision_seconds": state.valid_decision_seconds,
                "valid_ratio": valid_ratio,
                "af_suspected_seconds": state.af_suspected_seconds,
                "af_suspected_ratio_over_valid_decisions": af_ratio,
                "consecutive_undecidable_seconds": (
                    state.consecutive_undecidable_seconds
                ),
                "undecidable_seconds_by_reason": deepcopy(
                    state.undecidable_seconds_by_reason
                ),
                "expected_inference_sequence_number": (
                    state.expected_sequence_number
                ),
            }

    def end_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def status_snapshot(self) -> dict:
        now = time()
        with self._lock:
            sessions = []
            for session_id, state in sorted(self._sessions.items()):
                latest = state.latest_payload
                sessions.append({
                    "session_id": session_id,
                    "model_version": latest.get("model_version") if latest else None,
                    "inference_mode": latest.get("inference_mode") if latest else None,
                    "latest_decision": latest.get("decision") if latest else None,
                    "expected_inference_sequence_number": state.expected_sequence_number,
                    "last_accepted_at_seconds": state.last_accepted_at_seconds,
                    "last_accepted_age_seconds": (
                        None
                        if state.last_accepted_at_seconds is None
                        else max(0.0, now - state.last_accepted_at_seconds)
                    ),
                })
            return {"active_session_count": len(sessions), "sessions": sessions}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

registry = DownstreamInferenceRegistry()
