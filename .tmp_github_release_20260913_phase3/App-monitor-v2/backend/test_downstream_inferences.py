from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

import care_event_routes
import downstream_inference_routes as routes
import realtime_routes
import schemas
from downstream_inferences import DownstreamInferenceRegistry

def _request(**overrides) -> schemas.DownstreamInferenceCreate:
    values = {
        "schema_version": "downstream_v1",
        "session_id": "downstream-test",
        "inference_sequence_number": 0,
        "model_version": "af-v1",
        "inference_mode": "model",
        "frontend_schema_version": "1.1",
        "window": {
            "input_sequence_id": 120,
            "start_seconds": 100.0,
            "end_seconds": 130.0,
            "window_seconds": 30.0,
            "stride_seconds": 5.0,
        },
        "decidable": True,
        "abstention_reason": None,
        "af_probability": 0.91,
        "probability_is_calibrated": False,
        "decision": "af_suspected",
        "episode": {
            "state": "candidate",
            "episode_id": "episode-1",
            "start_seconds": 100.0,
            "duration_seconds": 30.0,
        },
        "context": {
            "valid_ratio": 0.94,
            "n_valid_beats": 36,
            "sqi_window": 0.88,
            "gate_value": 0.76,
            "used_morphology": False,
            "frontend_model_version": "front-stage4-v1",
        },
        "inference_timestamp_utc": datetime(2026, 9, 13, tzinfo=timezone.utc),
    }
    values.update(overrides)
    if "inference_sequence_number" in overrides and "window" not in overrides:
        sequence = int(overrides["inference_sequence_number"])
        values["window"] = {
            "input_sequence_id": 120 + 20 * sequence,
            "start_seconds": 100.0 + 5.0 * sequence,
            "end_seconds": 130.0 + 5.0 * sequence,
            "window_seconds": 30.0,
            "stride_seconds": 5.0,
        }
    return schemas.DownstreamInferenceCreate(**values)

class DownstreamInferenceSchemaTest(unittest.TestCase):
    def test_undecidable_requires_null_probability_and_reason(self) -> None:
        request = _request(
            decidable=False,
            decision="undecidable",
            af_probability=None,
            abstention_reason="low_signal_quality",
        )
        self.assertIsNone(request.af_probability)

        with self.assertRaises(ValidationError):
            _request(decidable=False, decision="undecidable", af_probability=0.91)
        with self.assertRaises(ValidationError):
            _request(decidable=False, decision="no_af_suspected", af_probability=None)

    def test_decidable_requires_probability_and_no_abstention_reason(self) -> None:
        with self.assertRaises(ValidationError):
            _request(af_probability=None)
        with self.assertRaises(ValidationError):
            _request(abstention_reason="stale")

    def test_window_duration_and_timezone_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            _request(window={"input_sequence_id": 1, "start_seconds": 0, "end_seconds": 29})
        with self.assertRaises(ValidationError):
            _request(inference_timestamp_utc=datetime.fromisoformat("2026-09-13T00:00:00"))

class DownstreamInferenceRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = DownstreamInferenceRegistry()

    def test_status_snapshot_exposes_model_mode_and_activity(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        status = self.registry.status_snapshot()
        self.assertEqual(status["active_session_count"], 1)
        session = status["sessions"][0]
        self.assertEqual(session["inference_mode"], "model")
        self.assertEqual(session["model_version"], "af-v1")
        self.assertGreaterEqual(session["last_accepted_age_seconds"], 0)

    def test_sequence_duplicate_gap_and_out_of_order(self) -> None:
        first = _request().model_dump(mode="json")
        accepted, changed = self.registry.accept(first)
        duplicate, duplicate_changed = self.registry.accept(first)
        self.assertTrue(changed)
        self.assertFalse(duplicate_changed)
        self.assertEqual(accepted, duplicate)

        gap = _request(inference_sequence_number=2).model_dump(mode="json")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.registry.accept(gap)

        second = _request(inference_sequence_number=1).model_dump(mode="json")
        self.registry.accept(second)
        with self.assertRaisesRegex(ValueError, "older"):
            self.registry.accept(first)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        conflicting = _request(af_probability=0.2).model_dump(mode="json")
        with self.assertRaisesRegex(ValueError, "older"):
            self.registry.accept(conflicting)

    def test_sessions_are_isolated_and_resettable(self) -> None:
        self.registry.accept(_request(session_id="a").model_dump(mode="json"))
        self.registry.accept(_request(session_id="b").model_dump(mode="json"))
        self.registry.end_session("a")
        with self.assertRaisesRegex(ValueError, "no downstream"):
            self.registry.latest("a")
        self.assertEqual(self.registry.latest("b")["session_id"], "b")
        _, changed = self.registry.accept(_request(session_id="a").model_dump(mode="json"))
        self.assertTrue(changed)

    def test_undecidable_replaces_previous_probability(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        undecidable = _request(
            inference_sequence_number=1,
            decidable=False,
            decision="undecidable",
            af_probability=None,
            abstention_reason="missing_data",
        ).model_dump(mode="json")
        self.registry.accept(undecidable)
        self.assertIsNone(self.registry.latest("downstream-test")["af_probability"])
        summary = self.registry.summary("downstream-test")
        self.assertEqual(summary["analysis_state"], "undecidable")
        self.assertAlmostEqual(summary["valid_ratio"], 30 / 35)
        self.assertEqual(summary["consecutive_undecidable_seconds"], 5.0)
        self.assertEqual(summary["undecidable_seconds_by_reason"], {"missing_data": 5.0})

    def test_monitoring_summary_uses_stride_not_overlapping_window_length(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        second = _request(
            inference_sequence_number=1,
            window={
                "input_sequence_id": 140,
                "start_seconds": 105.0,
                "end_seconds": 135.0,
                "window_seconds": 30.0,
                "stride_seconds": 5.0,
            },
            decision="no_af_suspected",
            af_probability=0.1,
        ).model_dump(mode="json")
        self.registry.accept(second)
        summary = self.registry.summary("downstream-test")
        self.assertEqual(summary["observed_seconds"], 35.0)
        self.assertEqual(summary["valid_decision_seconds"], 35.0)
        self.assertEqual(summary["af_suspected_seconds"], 30.0)
        self.assertAlmostEqual(
            summary["af_suspected_ratio_over_valid_decisions"], 30 / 35
        )
        self.assertEqual(summary["analysis_state"], "monitoring")

    def test_window_sequence_and_time_cannot_regress(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        regressed = _request(
            inference_sequence_number=1,
            window={
                "input_sequence_id": 120,
                "start_seconds": 105.0,
                "end_seconds": 135.0,
            },
        ).model_dump(mode="json")
        with self.assertRaisesRegex(ValueError, "window sequence"):
            self.registry.accept(regressed)

    def test_window_advance_must_match_stride(self) -> None:
        self.registry.accept(_request().model_dump(mode="json"))
        gap = _request(
            inference_sequence_number=1,
            window={
                "input_sequence_id": 160,
                "start_seconds": 110.0,
                "end_seconds": 140.0,
                "stride_seconds": 5.0,
            },
        ).model_dump(mode="json")
        with self.assertRaisesRegex(ValueError, "stride_seconds"):
            self.registry.accept(gap)

class DownstreamInferenceRouteTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        realtime_routes.registry.clear()
        routes.registry.clear()
        care_event_routes.registry.clear()
        realtime_routes.create_realtime_session(
            schemas.RealtimeSessionCreate(
                session_id="downstream-test",
                user_id="test_user_01",
                sample_rate_hz=125,
                deployment_profile_id="bidmc_train43_ppg_v1",
            )
        )

    async def test_route_accepts_and_returns_source_provenance(self) -> None:
        response = await routes.create_downstream_inference(_request())
        self.assertTrue(response["changed"])
        self.assertEqual(response["user_id"], "test_user_01")
        self.assertEqual(response["source_provenance"]["source_mode"], "live_device")
        latest = routes.get_latest_downstream_inference("downstream-test")
        self.assertFalse(latest["changed"])

    async def test_unknown_session_is_rejected_without_mutating_registry(self) -> None:
        response = await routes.create_downstream_inference(
            _request(session_id="unknown")
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(json.loads(response.body)["error"]["code"], "session_not_found")
        with self.assertRaisesRegex(ValueError, "no downstream"):
            routes.registry.latest("unknown")

    async def test_ending_realtime_session_resets_inference_sequence(self) -> None:
        await routes.create_downstream_inference(_request())
        await realtime_routes.end_realtime_session("downstream-test")
        realtime_routes.create_realtime_session(
            schemas.RealtimeSessionCreate(
                session_id="downstream-test",
                user_id="test_user_01",
                sample_rate_hz=125,
                deployment_profile_id="bidmc_train43_ppg_v1",
            )
        )
        response = await routes.create_downstream_inference(_request())
        self.assertTrue(response["changed"])

    async def test_only_confirmed_episode_creates_one_care_event(self) -> None:
        broadcaster = AsyncMock()
        with patch.object(care_event_routes, "_broadcast", broadcaster):
            candidate = await routes.create_downstream_inference(_request())
            self.assertIsNone(candidate["care_event"])
            confirmed_request = _request(
                inference_sequence_number=1,
                window={
                    "input_sequence_id": 140,
                    "start_seconds": 105.0,
                    "end_seconds": 135.0,
                },
                episode={
                    "state": "confirmed",
                    "episode_id": "episode-1",
                    "start_seconds": 100.0,
                    "duration_seconds": 35.0,
                },
            )
            confirmed = await routes.create_downstream_inference(confirmed_request)
            duplicate = await routes.create_downstream_inference(confirmed_request)

        self.assertEqual(
            confirmed["care_event"]["event"]["state"],
            "awaiting_patient_response",
        )
        self.assertTrue(confirmed["care_event"]["changed"])
        self.assertFalse(duplicate["care_event"]["changed"])
        self.assertEqual(len(care_event_routes.registry.list_for_user("test_user_01")), 1)
        broadcaster.assert_awaited_once()

    async def test_demo_stub_provenance_is_preserved_in_care_event(self) -> None:
        await realtime_routes.end_realtime_session("downstream-test")
        realtime_routes.create_realtime_session(
            schemas.RealtimeSessionCreate(
                session_id="downstream-test",
                user_id="test_user_01",
                sample_rate_hz=125,
                deployment_profile_id="bidmc_train43_ppg_v1",
                source_mode="synthetic_demo",
                demo_scenario_id="demo-af-1",
            )
        )
        response = await routes.create_downstream_inference(
            _request(
                inference_mode="demo_stub",
                episode={
                    "state": "confirmed",
                    "episode_id": "demo-episode-1",
                    "start_seconds": 100.0,
                    "duration_seconds": 30.0,
                },
            )
        )
        self.assertEqual(
            response["care_event"]["event"]["ai_result"]["inference_mode"],
            "demo_stub",
        )

    async def test_demo_stub_is_rejected_for_live_device_session(self) -> None:
        response = await routes.create_downstream_inference(
            _request(inference_mode="demo_stub")
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            json.loads(response.body)["error"]["code"],
            "invalid_inference_mode",
        )

    def test_routes_are_registered(self) -> None:
        paths = {route.path for route in routes.router.routes}
        self.assertIn("/downstream-inferences", paths)
        self.assertIn("/downstream-inferences/{session_id}/latest", paths)

if __name__ == "__main__":
    unittest.main()
