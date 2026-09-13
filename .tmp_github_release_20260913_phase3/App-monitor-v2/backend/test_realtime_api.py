from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import realtime_routes as realtime_api
import schemas
from deployment_profiles import _default_artifact_path, get_profile, load_profile
from device_ppg_demo_sender import _end_sessions, _packet
from fastapi import WebSocketDisconnect

class _FakeWebSocket:
    def __init__(self, requests):
        self.requests = iter(requests)
        self.sent = []
        self.accepted = False
        self.headers = {}

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        try:
            return next(self.requests)
        except StopIteration as error:
            raise WebSocketDisconnect() from error

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code):
        self.close_code = code

class RealtimeAPITest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        realtime_api.registry.clear()
        realtime_api.motion_registry.clear()
        realtime_api.downstream_window_registry.clear()

    def _create(self, session_id: str = "api-test") -> None:
        response = realtime_api.create_realtime_session(
            schemas.RealtimeSessionCreate(
                schema_version="1.0",
                session_id=session_id,
                user_id="test_user_01",
                sample_rate_hz=125,
                deployment_profile_id="bidmc_train43_ppg_v1",
            )
        )
        self.assertEqual(response["session_id"], session_id)
        self.assertFalse(response["deployment_profile"]["clinically_validated"])

    async def test_http_session_chunk_explicit_sequence_error_and_end(self) -> None:
        self._create()
        chunk = schemas.RealtimeChunkRequest(
            schema_version="1.0",
            sequence_number=0,
            input_timestamp_start_seconds=100.0,
            ppg=[0.0, 0.2, None, -0.1],
        )
        response = await realtime_api.process_realtime_chunk("api-test", chunk)
        self.assertEqual(response["frame"]["sequence_number"], 0)
        duplicate = await realtime_api.process_realtime_chunk("api-test", chunk)
        self.assertEqual(duplicate.status_code, 409)
        duplicate_body = json.loads(duplicate.body)
        self.assertEqual(duplicate_body["error"]["code"], "duplicate_packet")
        ended = await realtime_api.end_realtime_session("api-test")
        self.assertTrue(ended["ended"])

    async def test_websocket_returns_waveform_and_structured_gap_error(self) -> None:
        self._create("ws-test")
        websocket = _FakeWebSocket(
            [
                {
                "schema_version": "1.0",
                "sequence_number": 1,
                "input_timestamp_start_seconds": 100.0,
                "ppg": [0.0, 0.1],
                },
                {
                "schema_version": "1.0",
                "sequence_number": 0,
                "input_timestamp_start_seconds": 100.0,
                "ppg": [0.0, 0.1],
                },
            ]
        )
        await realtime_api.realtime_processing_ws(websocket, "ws-test")
        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent[0]["message_type"], "error")
        self.assertEqual(websocket.sent[0]["error"]["code"], "sequence_gap")
        self.assertEqual(websocket.sent[1]["message_type"], "waveform_frame")
        self.assertFalse(websocket.sent[1]["frame"]["diagnostic_ecg"])

    def test_routes_are_registered(self) -> None:
        paths = {route.path for route in realtime_api.router.routes}
        self.assertIn("/realtime/sessions", paths)
        self.assertIn("/realtime/sessions/{session_id}/chunks", paths)
        self.assertIn("/ws/realtime/{session_id}", paths)
        self.assertIn(
            "/realtime/sessions/{session_id}/device-ppg-chunks", paths
        )
        self.assertIn("/ws/front-ai/downstream/{session_id}", paths)
        self.assertIn(
            "/realtime/sessions/{session_id}/device-motion-chunks", paths
        )
        self.assertIn("/front-ai/windows/{session_id}/next", paths)
        self.assertIn("/front-ai/windows/{session_id}/ack", paths)

    async def test_motion_chunk_is_forwarded_as_separate_clocked_stream(self) -> None:
        self._create("motion-api-test")
        request = schemas.DeviceMotionChunkRequest(
            stream_id="verity-gyro",
            sequence_number=0,
            sensor_type="gyroscope",
            units="deg/s",
            configured_sample_rate_hz=52,
            device_timestamp_end_ns=1_000_000_000,
            clock_domain="verity-device-clock",
            x=[1.0, 2.0],
            y=[3.0, 4.0],
            z=[5.0, 6.0],
        )
        with patch.object(realtime_api, "_broadcast_downstream", new_callable=AsyncMock) as broadcast:
            response = await realtime_api.process_device_motion_chunk(
                "motion-api-test", request
            )
        self.assertEqual(response["message_type"], "front_ai_auxiliary_stream")
        self.assertEqual(response["sensor_type"], "gyroscope")
        broadcast.assert_awaited_once()

    async def test_downstream_output_websocket_requires_api_key(self) -> None:
        websocket = _FakeWebSocket([])
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GUARD_DOWNSTREAM_API_KEY", None)
            await realtime_api.downstream_ai_output_ws(websocket, "missing")
        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent[0]["error"]["code"], "downstream_authentication_failed")
        self.assertEqual(websocket.close_code, 4401)

    def test_window_pull_requires_key_and_ack_is_idempotent(self) -> None:
        self._create("window-api-test")
        count = 3750
        handoff = {
            "schema_version": "1.0",
            "payload_type": "front_ai_device_handoff",
            "source_device_packet": {
                "sequence_number": 0,
                "clock_domain": "verity-clock",
            },
            "derived": {
                "session_id": "window-api-test",
                "sample_count": count,
                "sample_rate_hz": 125.0,
                "input_timestamp_start_seconds": 0.0,
                "ppg": [0.0] * count,
                "ppg_valid": [True] * count,
                "pseudo_ecg": [0.0] * count,
                "pseudo_ecg_valid": [False] * count,
                "technical_sqi": 1.0,
                "pulse_interval_plausibility": 1.0,
                "beat_events": [],
            },
        }
        realtime_api.downstream_window_registry.ingest_ppg(
            "window-api-test", handoff
        )
        with patch.dict(os.environ, {"GUARD_DOWNSTREAM_API_KEY": "secret"}):
            denied = realtime_api.get_next_downstream_window(
                "window-api-test", "wrong"
            )
            self.assertEqual(denied.status_code, 401)
            window = realtime_api.get_next_downstream_window(
                "window-api-test", "secret"
            )
            self.assertEqual(window["schema_version"], "front_ai_window_v1")
            ack = realtime_api.acknowledge_downstream_window(
                "window-api-test",
                schemas.DownstreamWindowAck(input_sequence_id=0),
                "secret",
            )
            repeated = realtime_api.acknowledge_downstream_window(
                "window-api-test",
                schemas.DownstreamWindowAck(input_sequence_id=0),
                "secret",
            )
        self.assertTrue(ack["changed"])
        self.assertFalse(repeated["changed"])

    def test_demo_packet_supports_half_second_updates_and_distinct_patients(self) -> None:
        first = _packet(
            0,
            rate=135,
            seconds=0.5,
            origin_ns=700000000000000000,
            stream_id="first",
            patient_index=0,
        )
        second = _packet(
            0,
            rate=135,
            seconds=0.5,
            origin_ns=700000000000000000,
            stream_id="second",
            patient_index=1,
        )
        self.assertEqual(len(first["ppg0"]), 68)
        self.assertEqual(len(second["ppg0"]), 68)
        self.assertNotEqual(first["ppg0"], second["ppg0"])

    def test_demo_sender_ends_every_created_session(self) -> None:
        sessions = [
            (0, "test_user_01", "demo-test_user_01"),
            (1, "test_user_02", "demo-test_user_02"),
        ]
        with patch("device_ppg_demo_sender._delete") as delete:
            _end_sessions("http://127.0.0.1:8000", sessions)

        self.assertEqual(delete.call_count, 2)
        delete.assert_any_call(
            "http://127.0.0.1:8000/realtime/sessions/demo-test_user_01"
        )
        delete.assert_any_call(
            "http://127.0.0.1:8000/realtime/sessions/demo-test_user_02"
        )

    async def test_device_ppg_chunk_returns_downstream_and_visualization_outputs(self) -> None:
        self._create("device-api-test")
        count = 270
        values = [500000 + index for index in range(count)]
        request = schemas.DevicePPGChunkRequest(
            stream_id="verity-demo",
            sequence_number=0,
            configured_sample_rate_hz=135,
            device_timestamp_end_ns=700000000000000000,
            ppg0=values,
            ppg1=[value + 2 for value in values],
            ppg2=[value - 2 for value in values],
            ambient0=[12000] * count,
        )

        response = await realtime_api.process_device_ppg_chunk(
            "device-api-test", request
        )

        self.assertEqual(response["message_type"], "device_ppg_result")
        self.assertEqual(
            response["downstream_ai"]["payload_type"],
            "front_ai_device_handoff",
        )
        self.assertEqual(response["source_provenance"]["source_mode"], "live_device")
        self.assertEqual(
            response["visualization"]["payload_type"],
            "doctor_waveform_visualization",
        )
        self.assertFalse(response["visualization"]["diagnostic_ecg"])
        self.assertEqual(
            response["downstream_ai"]["source_device_packet"]["ambient0"],
            [12000] * count,
        )

    def test_demo_packet_is_explicitly_marked_as_synthetic(self) -> None:
        packet = _packet(
            0,
            rate=135,
            seconds=0.25,
            origin_ns=700000000000000000,
            stream_id="demo",
            patient_index=0,
            scenario_id="demo-normal-01",
        )

        self.assertEqual(packet["source_mode"], "synthetic_demo")
        self.assertEqual(packet["demo_scenario_id"], "demo-normal-01")

    def test_sender_cannot_inject_scaler_or_af_probability(self) -> None:
        with self.assertRaises(ValueError):
            schemas.RealtimeSessionCreate(
                user_id="test_user_01",
                sample_rate_hz=125,
                scaler_center=999,
            )
        with self.assertRaises(ValueError):
            schemas.RealtimeChunkRequest(
                sequence_number=0,
                input_timestamp_start_seconds=0.0,
                ppg=[0.0],
                af_probability=0.99,
            )

    def test_profile_sample_rate_mismatch_is_rejected(self) -> None:
        response = realtime_api.create_realtime_session(
            schemas.RealtimeSessionCreate(
                user_id="test_user_01",
                sample_rate_hz=100,
                deployment_profile_id="bidmc_train43_ppg_v1",
            )
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)["error"]["code"], "invalid_profile")

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "Stage 4 profile requires the optional PyTorch runtime.",
    )
    async def test_stage4_profile_exposes_safe_morphology_metadata(self) -> None:
        profile = get_profile("mimic_fold0_stage4_st_prior_v1")
        self.assertEqual(
            profile.generation_profile_id, "timing_constrained_morphology_v1"
        )
        created = realtime_api.create_realtime_session(
            schemas.RealtimeSessionCreate(
                session_id="stage4-api-test",
                user_id="test_user_01",
                sample_rate_hz=125,
                deployment_profile_id="mimic_fold0_stage4_st_prior_v1",
            )
        )
        self.assertEqual(
            created["generation_profile_id"], "timing_constrained_morphology_v1"
        )
        count = 270
        values = [500000 + (index % 70) * 800 for index in range(count)]
        response = await realtime_api.process_device_ppg_chunk(
            "stage4-api-test",
            schemas.DevicePPGChunkRequest(
                stream_id="stage4-device",
                sequence_number=0,
                configured_sample_rate_hz=135,
                device_timestamp_end_ns=700000000000000000,
                ppg0=values,
                ppg1=[value + 100 for value in values],
                ppg2=[value - 100 for value in values],
                ambient0=[12000] * count,
            ),
        )
        frame = response["visualization"]
        self.assertIn(frame["generation_mode"], {"population_prior", "blank"})
        self.assertEqual(frame["morphology_source"], "population_mean_latent")
        self.assertFalse(frame["p_wave_generated"])
        self.assertFalse(frame["qt_measurement_supported"])
        self.assertFalse(frame["pq_pr_measurement_supported"])
        self.assertIsNone(frame["estimated_pat_ms"])

    def test_tampered_profile_hash_is_rejected(self) -> None:
        payload = json.loads(_default_artifact_path().read_text(encoding="utf-8"))
        payload["scaler"]["center"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_profile(path)

if __name__ == "__main__":
    unittest.main()
