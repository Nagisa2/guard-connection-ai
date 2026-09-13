from __future__ import annotations

import unittest

from downstream_ai_client_example import validate_front_ai_message
from validate_device_payload import validate_payload

class IntegrationClientValidationTest(unittest.TestCase):
    def test_device_ppg_and_motion_payloads_match_server_schema(self) -> None:
        ppg = {
            "schema_version": "1.0",
            "stream_id": "ppg",
            "sequence_number": 0,
            "configured_sample_rate_hz": 135,
            "device_timestamp_end_ns": 1_000_000_000,
            "ppg0": [1, 2],
            "ppg1": [2, 3],
            "ppg2": [3, 4],
            "ambient0": [0, 0],
        }
        motion = {
            "schema_version": "1.0",
            "stream_id": "acc",
            "sequence_number": 0,
            "sensor_type": "accelerometer",
            "units": "mG",
            "configured_sample_rate_hz": 52,
            "device_timestamp_end_ns": 1_000_000_000,
            "clock_domain": "verity-clock",
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
        self.assertEqual(validate_payload(ppg), ("ppg", 2))
        self.assertEqual(validate_payload(motion), ("motion", 1))

    def test_front_ai_output_validator_keeps_ppg_and_motion_separate(self) -> None:
        ppg = {
            "payload_type": "front_ai_device_handoff",
            "derived": {
                "session_id": "s",
                "sequence_number": 0,
                "ppg": [0.1],
                "ppg_valid": [True],
                "beat_events": [],
                "technical_sqi": 0.8,
                "signal_coverage": 1.0,
                "generation_accepted": False,
            },
        }
        motion = {
            "message_type": "front_ai_auxiliary_stream",
            "sensor_type": "accelerometer",
        }
        self.assertEqual(validate_front_ai_message(ppg), "ppg")
        self.assertEqual(validate_front_ai_message(motion), "motion")

if __name__ == "__main__":
    unittest.main()
