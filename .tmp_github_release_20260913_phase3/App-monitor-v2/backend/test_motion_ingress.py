from __future__ import annotations

import unittest

from motion_ingress import MotionIngressError, MotionIngressRegistry

def _packet(sequence: int = 0, *, timestamp_end_ns: int = 1_000_000_000) -> dict:
    return {
        "stream_id": "verity-accelerometer",
        "sequence_number": sequence,
        "sensor_type": "accelerometer",
        "units": "mG",
        "configured_sample_rate_hz": 52.0,
        "device_timestamp_end_ns": timestamp_end_ns,
        "clock_domain": "verity-device-clock",
        "x": [1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0],
        "z": [7.0, 8.0, 9.0],
        "sensor_disconnected": False,
    }

class MotionIngressRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = MotionIngressRegistry()

    def test_accepts_independent_motion_stream(self) -> None:
        result = self.registry.accept("session-1", _packet())
        self.assertEqual(result["message_type"], "front_ai_auxiliary_stream")
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["clock_domain"], "verity-device-clock")
        self.assertEqual(self.registry.status_snapshot()["active_stream_count"], 1)

    def test_duplicate_gap_and_clock_change_are_rejected(self) -> None:
        first = _packet()
        self.registry.accept("session-1", first)
        with self.assertRaises(MotionIngressError) as duplicate:
            self.registry.accept("session-1", first)
        self.assertEqual(duplicate.exception.code, "duplicate_packet")

        gap_packet = _packet(2, timestamp_end_ns=1_115_384_615)
        with self.assertRaises(MotionIngressError) as gap:
            self.registry.accept("session-1", gap_packet)
        self.assertEqual(gap.exception.code, "sequence_gap")

        second = _packet(1, timestamp_end_ns=1_057_692_308)
        second["clock_domain"] = "another-clock"
        with self.assertRaises(MotionIngressError) as clock:
            self.registry.accept("session-1", second)
        self.assertEqual(clock.exception.code, "clock_domain_changed")

    def test_session_end_does_not_remove_other_session(self) -> None:
        self.registry.accept("first", _packet())
        self.registry.accept("second", _packet())
        self.registry.end_session("first")
        status = self.registry.status_snapshot()
        self.assertEqual(status["active_stream_count"], 1)
        self.assertEqual(status["streams"][0]["session_id"], "second")

if __name__ == "__main__":
    unittest.main()
