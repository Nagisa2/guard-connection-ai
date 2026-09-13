from __future__ import annotations

import unittest
from unittest.mock import patch

import main

class _Query:
    @staticmethod
    def count() -> int:
        return 8

class _Database:
    @staticmethod
    def query(_model):
        return _Query()

class DemoSystemStatusTest(unittest.TestCase):
    def test_status_reports_interfaces_and_stale_sessions(self) -> None:
        realtime = {
            "active_session_count": 1,
            "sessions": [{"last_input_age_seconds": 3.0}],
        }
        downstream = {
            "active_session_count": 1,
            "sessions": [{"inference_mode": "model"}],
        }
        with (
            patch.object(main.realtime_session_registry, "status_snapshot", return_value=realtime),
            patch.object(main.downstream_inference_registry, "status_snapshot", return_value=downstream),
            patch.object(
                main.downstream_window_registry,
                "status_snapshot",
                return_value={
                    "active_session_count": 1,
                    "sessions": [{"pending_window_count": 2}],
                },
            ),
        ):
            result = main.demo_system_status(_Database())

        self.assertTrue(result["ready"])
        self.assertEqual(result["patient_count"], 8)
        self.assertEqual(result["stale_session_count"], 1)
        self.assertEqual(result["inference_modes"], ["model"])
        self.assertEqual(result["downstream_operating_mode"], "model")
        self.assertEqual(
            result["downstream_windows"]["sessions"][0]["pending_window_count"],
            2,
        )
        self.assertTrue(result["interfaces"]["device_ppg_ingress"]["ready"])
        self.assertEqual(
            result["interfaces"]["downstream_ai_result_ingress"]["schema_version"],
            "downstream_v1",
        )
        self.assertEqual(
            result["interfaces"]["front_ai_downstream_output"]["authentication"],
            "x-downstream-api-key",
        )

if __name__ == "__main__":
    unittest.main()
