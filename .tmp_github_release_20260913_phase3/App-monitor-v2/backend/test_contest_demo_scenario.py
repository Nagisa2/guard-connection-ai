from __future__ import annotations

import unittest
from unittest.mock import patch

from contest_demo_scenario import (
    _apply_demo_action,
    _create_demo_downstream_inference,
    _normal_ppg,
    _send_demo_location,
    _synthetic_irregular_rows,
)

class ContestDemoScenarioTest(unittest.TestCase):
    def test_normal_ppg_has_requested_length_and_plausible_range(self) -> None:
        values = _normal_ppg(0, 125, 125.0)
        self.assertEqual(len(values), 125)
        self.assertGreater(max(values) - min(values), 0.1)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_synthetic_fallback_is_explicit_and_contiguous(self) -> None:
        rows = _synthetic_irregular_rows()
        self.assertEqual([row["sequence_number"] for row in rows], list(range(len(rows))))
        self.assertTrue(all("reference_rhythm" not in row for row in rows))
        self.assertEqual(sum(len(row["ppg"]) for row in rows), 1875)

    @patch("contest_demo_scenario._post")
    def test_demo_inference_is_explicitly_stubbed(self, post) -> None:
        post.return_value = {"care_event": None}

        _create_demo_downstream_inference(
            "http://127.0.0.1:8000",
            session_id="session-af",
            inference_sequence_number=0,
            input_sequence_id=20,
            window_start_seconds=0.0,
            window_end_seconds=5.0,
            stride_seconds=5.0,
            decision="af_suspected",
            episode_state="candidate",
            episode_start_seconds=0.0,
        )

        payload = post.call_args.args[1]
        self.assertEqual(payload["inference_mode"], "demo_stub")
        self.assertEqual(payload["decision"], "af_suspected")
        self.assertEqual(payload["episode"]["state"], "candidate")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8000/downstream-inferences")
        self.assertNotIn("reference_rhythm", payload)

    @patch("contest_demo_scenario._post")
    def test_patient_action_uses_patient_actor(self, post) -> None:
        post.return_value = {"event": {"event_id": "event-1"}}

        _apply_demo_action("http://127.0.0.1:8000", "event-1", "patient_unwell")

        self.assertEqual(post.call_args.args[1]["actor_role"], "patient")

    @patch("contest_demo_scenario._post")
    def test_demo_location_is_explicitly_non_live(self, post) -> None:
        post.return_value = {"event": {"event_id": "event-1"}}

        _send_demo_location(
            "http://127.0.0.1:8000",
            "event-1",
            latitude=34.18824,
            longitude=132.43761,
        )

        payload = post.call_args.args[1]
        self.assertEqual(payload["action"], "update_location")
        self.assertEqual(payload["actor_role"], "system")
        self.assertEqual(payload["location"]["provider"], "demo_fixed_coordinate")

if __name__ == "__main__":
    unittest.main()
