from __future__ import annotations

import unittest

from downstream_window_client_example import validate_window

def _window() -> dict:
    return {
        "schema_version": "front_ai_window_v1",
        "primary_signal": "ppg",
        "ppg": {"values": [0.1, None], "valid_mask": [True, False]},
        "beats": {"source": "ppg", "times_sec": [1.0]},
        "auxiliary_pseudo_ecg": {"diagnostic_ecg": False},
    }

class DownstreamWindowClientExampleTest(unittest.TestCase):
    def test_accepts_unlabeled_ppg_primary_window(self) -> None:
        validate_window(_window())

    def test_rejects_training_label_and_diagnostic_pseudo_ecg(self) -> None:
        labeled = _window()
        labeled["label"] = 1
        with self.assertRaises(ValueError):
            validate_window(labeled)
        diagnostic = _window()
        diagnostic["auxiliary_pseudo_ecg"]["diagnostic_ecg"] = True
        with self.assertRaises(ValueError):
            validate_window(diagnostic)

if __name__ == "__main__":
    unittest.main()
