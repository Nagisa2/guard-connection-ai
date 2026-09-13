"""AF疑い後の患者・家族・医師連携を管理する状態機械。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from time import time
from uuid import uuid4

class CareEventError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

_ACTION_ROLES = {
    "patient_ok": "patient",
    "patient_unwell": "patient",
    "patient_help": "patient",
    "fall_detected": "system",
    "patient_no_response": "system",
    "family_acknowledged": "family",
    "doctor_acknowledged": "doctor",
    "update_location": None,
    "resolve": "doctor",
}

def _validate_source(source_mode: str, demo_scenario_id: str | None) -> None:
    if source_mode not in {"live_device", "recorded_demo_replay", "synthetic_demo"}:
        raise CareEventError("invalid_source", "unsupported source_mode")
    if source_mode in {"recorded_demo_replay", "synthetic_demo"} and not demo_scenario_id:
        raise CareEventError(
            "invalid_source", "recorded demo replay requires demo_scenario_id"
        )
    if source_mode == "live_device" and demo_scenario_id is not None:
        raise CareEventError(
            "invalid_source", "live device input must not declare a demo scenario"
        )

def _validate_location(location: dict | None) -> None:
    if location is None:
        return
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    accuracy = location.get("accuracy_m")
    if not isinstance(latitude, (int, float)) or not -90 <= latitude <= 90:
        raise CareEventError("invalid_location", "latitude is outside [-90, 90]")
    if not isinstance(longitude, (int, float)) or not -180 <= longitude <= 180:
        raise CareEventError("invalid_location", "longitude is outside [-180, 180]")
    if not isinstance(accuracy, (int, float)) or accuracy < 0:
        raise CareEventError("invalid_location", "accuracy_m must be non-negative")

class CareEventRegistry:
    """イベント状態を分離し、任意のSQLiteへ永続化する。"""

    schema_version = "1.0"

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._events: dict[str, dict] = {}
        self._create_keys: dict[str, str] = {}
        self._action_keys: set[tuple[str, str]] = set()
        self._lock = RLock()
        self._storage_path = Path(storage_path) if storage_path is not None else None
        if self._storage_path is not None:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_storage()
            self._load_storage()

    def _connect(self) -> sqlite3.Connection:
        if self._storage_path is None:
            raise RuntimeError("care event persistence is not configured")
        connection = sqlite3.connect(self._storage_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_storage(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS care_events (
                    event_id TEXT PRIMARY KEY,
                    create_key TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    updated_at_seconds REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS care_event_action_keys (
                    event_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    PRIMARY KEY (event_id, idempotency_key),
                    FOREIGN KEY (event_id) REFERENCES care_events(event_id)
                        ON DELETE CASCADE
                )
                """
            )

    def _load_storage(self) -> None:
        with self._connection() as connection:
            for event_id, create_key, payload_json in connection.execute(
                "SELECT event_id, create_key, payload_json FROM care_events"
            ):
                event = json.loads(payload_json)
                self._events[event_id] = event
                self._create_keys[create_key] = event_id
            self._action_keys = set(
                connection.execute(
                    "SELECT event_id, idempotency_key FROM care_event_action_keys"
                ).fetchall()
            )

    def _persist_created_event(self, event: dict, create_key: str) -> None:
        if self._storage_path is None:
            return
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO care_events (
                    event_id, create_key, user_id, updated_at_seconds, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    create_key,
                    event["user_id"],
                    event["updated_at_seconds"],
                    json.dumps(event, ensure_ascii=False),
                ),
            )

    def _persist_action(
        self, event: dict, *, idempotency_key: str
    ) -> None:
        if self._storage_path is None:
            return
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO care_event_action_keys (event_id, idempotency_key)
                VALUES (?, ?)
                """,
                (event["event_id"], idempotency_key),
            )
            connection.execute(
                """
                UPDATE care_events
                SET updated_at_seconds = ?, payload_json = ?
                WHERE event_id = ?
                """,
                (
                    event["updated_at_seconds"],
                    json.dumps(event, ensure_ascii=False),
                    event["event_id"],
                ),
            )

    def create_af_suspicion(
        self,
        *,
        idempotency_key: str,
        user_id: str,
        patient_id: int | None,
        session_id: str,
        source_mode: str,
        demo_scenario_id: str | None,
        af_probability: float,
        uncertainty: float | None,
        model_version: str,
        inference_mode: str,
        analysis_window_start_seconds: float,
        analysis_window_end_seconds: float,
        occurred_at_seconds: float | None = None,
    ) -> tuple[dict, bool]:
        if not idempotency_key or not user_id or not session_id or not model_version:
            raise CareEventError(
                "invalid_event", "idempotency key, user, session, and model are required"
            )
        if not 0 <= af_probability <= 1:
            raise CareEventError(
                "invalid_event", "af_probability must be within [0, 1]"
            )
        if uncertainty is not None and not 0 <= uncertainty <= 1:
            raise CareEventError(
                "invalid_event", "uncertainty must be within [0, 1]"
            )
        if analysis_window_end_seconds < analysis_window_start_seconds:
            raise CareEventError(
                "invalid_event", "analysis window end precedes its start"
            )
        _validate_source(source_mode, demo_scenario_id)
        if inference_mode not in {"model", "demo_stub"}:
            raise CareEventError("invalid_event", "unsupported inference_mode")
        if inference_mode == "demo_stub" and source_mode == "live_device":
            raise CareEventError(
                "invalid_event", "demo stub inference cannot use live device input"
            )
        with self._lock:
            existing_id = self._create_keys.get(idempotency_key)
            if existing_id is not None:
                return deepcopy(self._events[existing_id]), False
            now = float(time() if occurred_at_seconds is None else occurred_at_seconds)
            event_id = str(uuid4())
            event = {
                "schema_version": self.schema_version,
                "event_id": event_id,
                "event_type": "af_suspected",
                "state": "awaiting_patient_response",
                "severity": "warning",
                "user_id": user_id,
                "patient_id": patient_id,
                "session_id": session_id,
                "source_mode": source_mode,
                "demo_mode": source_mode != "live_device",
                "demo_scenario_id": demo_scenario_id,
                "diagnostic_result": False,
                "ai_result": {
                    "result_source": "downstream_ai",
                    "af_probability": af_probability,
                    "uncertainty": uncertainty,
                    "model_version": model_version,
                    "inference_mode": inference_mode,
                    "analysis_window_start_seconds": analysis_window_start_seconds,
                    "analysis_window_end_seconds": analysis_window_end_seconds,
                },
                "location": None,
                "family_acknowledged": False,
                "doctor_acknowledged": False,
                "created_at_seconds": now,
                "updated_at_seconds": now,
                "revision": 0,
                "action_history": [],
            }
            self._events[event_id] = event
            self._create_keys[idempotency_key] = event_id
            self._persist_created_event(event, idempotency_key)
            return deepcopy(event), True

    def apply_action(
        self,
        event_id: str,
        *,
        idempotency_key: str,
        action: str,
        actor_role: str,
        occurred_at_seconds: float | None = None,
        location: dict | None = None,
    ) -> tuple[dict, bool]:
        if action not in _ACTION_ROLES:
            raise CareEventError("invalid_action", "unsupported care event action")
        required_role = _ACTION_ROLES[action]
        if required_role is not None and actor_role != required_role:
            raise CareEventError(
                "invalid_actor", f"{action} must be sent by {required_role}"
            )
        if actor_role not in {"patient", "family", "doctor", "system"}:
            raise CareEventError("invalid_actor", "unsupported actor role")
        if not idempotency_key:
            raise CareEventError("invalid_action", "idempotency_key is required")
        _validate_location(location)
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise CareEventError("event_not_found", "care event was not found")
            action_key = (event_id, idempotency_key)
            if action_key in self._action_keys:
                return deepcopy(event), False
            if event["state"] == "resolved":
                raise CareEventError("event_resolved", "resolved events cannot be changed")
            if action == "patient_ok":
                event["state"] = "monitoring"
                event["severity"] = "info"
            elif action == "patient_unwell":
                event["state"] = "clinician_review"
                event["severity"] = "warning"
            elif action in {"patient_help", "fall_detected", "patient_no_response"}:
                event["state"] = "emergency_escalated"
                event["severity"] = "emergency"
            elif action == "family_acknowledged":
                event["family_acknowledged"] = True
            elif action == "doctor_acknowledged":
                event["doctor_acknowledged"] = True
            elif action == "update_location":
                if location is None:
                    raise CareEventError(
                        "invalid_location", "update_location requires a location"
                    )
                event["location"] = dict(location)
            elif action == "resolve":
                event["state"] = "resolved"
                event["severity"] = "info"
            now = float(time() if occurred_at_seconds is None else occurred_at_seconds)
            event["updated_at_seconds"] = now
            event["revision"] += 1
            event["action_history"].append(
                {
                    "action": action,
                    "actor_role": actor_role,
                    "occurred_at_seconds": now,
                }
            )
            self._persist_action(event, idempotency_key=idempotency_key)
            self._action_keys.add(action_key)
            return deepcopy(event), True

    def get(self, event_id: str) -> dict:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise CareEventError("event_not_found", "care event was not found")
            return deepcopy(event)

    def list_for_user(self, user_id: str) -> list[dict]:
        with self._lock:
            events = [
                deepcopy(event)
                for event in self._events.values()
                if event["user_id"] == user_id
            ]
        return sorted(events, key=lambda item: item["updated_at_seconds"], reverse=True)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._create_keys.clear()
            self._action_keys.clear()
            if self._storage_path is not None:
                with self._connection() as connection:
                    connection.execute("DELETE FROM care_event_action_keys")
                    connection.execute("DELETE FROM care_events")

registry = CareEventRegistry()
