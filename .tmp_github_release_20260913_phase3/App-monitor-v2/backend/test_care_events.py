from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import care_event_routes
import schemas
from care_events import CareEventError, CareEventRegistry
from pydantic import ValidationError

def _af_request(**overrides) -> schemas.AFSuspicionCreate:
    values = {
        "idempotency_key": "session-a:100:af-v1",
        "user_id": "test_user_01",
        "patient_id": 1,
        "session_id": "session-a",
        "source_mode": "recorded_demo_replay",
        "demo_scenario_id": "af-held-out-01",
        "af_probability": 0.91,
        "uncertainty": 0.08,
        "model_version": "af-v1",
        "inference_mode": "model",
        "analysis_window_start_seconds": 100.0,
        "analysis_window_end_seconds": 130.0,
        "occurred_at_seconds": 131.0,
    }
    values.update(overrides)
    return schemas.AFSuspicionCreate(**values)

class CareEventRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CareEventRegistry()

    def _create(self, **overrides):
        return self.registry.create_af_suspicion(
            **_af_request(**overrides).model_dump()
        )

    def test_af_suspicion_starts_patient_confirmation_not_emergency(self) -> None:
        event, created = self._create()

        self.assertTrue(created)
        self.assertEqual(event["state"], "awaiting_patient_response")
        self.assertEqual(event["severity"], "warning")
        self.assertFalse(event["diagnostic_result"])
        self.assertEqual(event["ai_result"]["result_source"], "downstream_ai")
        self.assertEqual(event["ai_result"]["inference_mode"], "model")
        self.assertTrue(event["demo_mode"])

    def test_create_and_action_idempotency_do_not_advance_revision(self) -> None:
        first, _ = self._create()
        duplicate, created = self._create()
        self.assertFalse(created)
        self.assertEqual(first["event_id"], duplicate["event_id"])

        updated, changed = self.registry.apply_action(
            first["event_id"],
            idempotency_key="patient-response-1",
            action="patient_ok",
            actor_role="patient",
            occurred_at_seconds=132.0,
        )
        repeated, repeated_changed = self.registry.apply_action(
            first["event_id"],
            idempotency_key="patient-response-1",
            action="patient_ok",
            actor_role="patient",
            occurred_at_seconds=133.0,
        )

        self.assertTrue(changed)
        self.assertFalse(repeated_changed)
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(repeated["revision"], 1)

    def test_fall_or_no_response_escalates_but_af_alone_does_not(self) -> None:
        event, _ = self._create()
        escalated, _ = self.registry.apply_action(
            event["event_id"],
            idempotency_key="fall-1",
            action="fall_detected",
            actor_role="system",
            occurred_at_seconds=135.0,
        )

        self.assertEqual(escalated["state"], "emergency_escalated")
        self.assertEqual(escalated["severity"], "emergency")

    def test_location_keeps_accuracy_timestamp_and_provider(self) -> None:
        event, _ = self._create()
        location = {
            "latitude": 34.0,
            "longitude": 131.0,
            "accuracy_m": 12.5,
            "captured_at_seconds": 136.0,
            "provider": "fused",
        }
        updated, _ = self.registry.apply_action(
            event["event_id"],
            idempotency_key="location-1",
            action="update_location",
            actor_role="patient",
            occurred_at_seconds=136.1,
            location=location,
        )

        self.assertEqual(updated["location"], location)

    def test_actor_role_and_session_state_are_isolated(self) -> None:
        first, _ = self._create()
        second, _ = self._create(
            idempotency_key="session-b:200:af-v1",
            user_id="test_user_02",
            patient_id=2,
            session_id="session-b",
        )
        with self.assertRaises(CareEventError) as error:
            self.registry.apply_action(
                first["event_id"],
                idempotency_key="bad-role",
                action="doctor_acknowledged",
                actor_role="patient",
            )
        self.assertEqual(error.exception.code, "invalid_actor")
        self.assertEqual(self.registry.get(second["event_id"])["revision"], 0)
        self.assertEqual(len(self.registry.list_for_user("test_user_01")), 1)
        self.assertEqual(len(self.registry.list_for_user("test_user_02")), 1)

    def test_reference_label_cannot_be_submitted_as_ai_result(self) -> None:
        with self.assertRaises(ValidationError):
            _af_request(reference_rhythm="AF")

    def test_demo_stub_cannot_be_presented_as_live_device_inference(self) -> None:
        with self.assertRaises(ValidationError):
            _af_request(
                source_mode="live_device",
                demo_scenario_id=None,
                inference_mode="demo_stub",
            )

    def test_sqlite_restores_event_and_action_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "care-events.db"
            first_registry = CareEventRegistry(database)
            event, _ = first_registry.create_af_suspicion(
                **_af_request().model_dump()
            )
            updated, _ = first_registry.apply_action(
                event["event_id"],
                idempotency_key="patient-response-persisted",
                action="patient_unwell",
                actor_role="patient",
                occurred_at_seconds=140.0,
            )

            restored = CareEventRegistry(database)
            restored_event = restored.get(event["event_id"])
            duplicate, created = restored.create_af_suspicion(
                **_af_request().model_dump()
            )
            repeated, changed = restored.apply_action(
                event["event_id"],
                idempotency_key="patient-response-persisted",
                action="patient_unwell",
                actor_role="patient",
                occurred_at_seconds=150.0,
            )

            self.assertEqual(restored_event, updated)
            self.assertFalse(created)
            self.assertEqual(duplicate["event_id"], event["event_id"])
            self.assertFalse(changed)
            self.assertEqual(repeated["revision"], 1)

class CareEventRoutesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        care_event_routes.registry.clear()

    async def test_created_event_is_broadcast_once(self) -> None:
        broadcaster = AsyncMock()
        with patch.object(care_event_routes, "_broadcast", broadcaster):
            first = await care_event_routes.create_af_suspicion(_af_request())
            second = await care_event_routes.create_af_suspicion(_af_request())

        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        broadcaster.assert_awaited_once()
        self.assertEqual(
            broadcaster.await_args.args[1]["event"]["state"],
            "awaiting_patient_response",
        )

if __name__ == "__main__":
    unittest.main()
