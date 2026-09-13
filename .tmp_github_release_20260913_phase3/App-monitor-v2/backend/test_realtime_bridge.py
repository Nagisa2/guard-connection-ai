from __future__ import annotations

import unittest

from realtime_bridge import (
    RealtimeProtocolError,
    RealtimeSessionConfig,
    RealtimeSessionRegistry,
    waveform_envelope,
)
from schemas import RealtimeChunkRequest, RealtimeSessionCreate

class RealtimeBridgeTest(unittest.TestCase):
    def setUp(self):
        self.registry = RealtimeSessionRegistry()
        self.registry.create(
            RealtimeSessionConfig(
                session_id="doctor-app-test",
                user_id="test_user_01",
                sample_rate_hz=50,
                scaler_center=0.0,
                scaler_scale=1.0,
                quality_window_seconds=1.0,
            )
        )

    def test_doctor_app_request_schema_and_response_envelope(self):
        create = RealtimeSessionCreate(
            user_id="test_user_01", sample_rate_hz=125
        )
        self.assertEqual(create.schema_version, "1.0")
        request = RealtimeChunkRequest(
            sequence_number=0,
            input_timestamp_start_seconds=100.0,
            ppg=[0.0, 0.2, None, -0.1],
        )
        frame = self.registry.process_chunk(
            "doctor-app-test",
            sequence_number=request.sequence_number,
            input_timestamp_start_seconds=request.input_timestamp_start_seconds,
            ppg=request.ppg,
        )
        payload = waveform_envelope(frame)
        self.assertEqual(payload["message_type"], "waveform_frame")
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["frame"]["schema_version"], "1.1")
        self.assertEqual(payload["frame"]["waveform_type"], "ppg_derived_pseudo_ecg")
        self.assertFalse(payload["frame"]["diagnostic_ecg"])
        self.assertEqual(payload["frame"]["ppg_valid"], [True, True, False, True])

    def test_packet_gap_is_explicit(self):
        with self.assertRaises(RealtimeProtocolError) as raised:
            self.registry.process_chunk(
                "doctor-app-test",
                sequence_number=1,
                input_timestamp_start_seconds=100.0,
                ppg=[0.0],
            )
        self.assertEqual(raised.exception.code, "sequence_gap")
        self.assertEqual(raised.exception.expected_sequence_number, 0)

if __name__ == "__main__":
    unittest.main()
