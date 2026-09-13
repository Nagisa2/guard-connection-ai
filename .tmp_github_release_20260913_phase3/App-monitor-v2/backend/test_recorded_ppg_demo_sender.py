from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from recorded_ppg_demo_sender import _chunk_payload, _load_rows

class RecordedPPGDemoSenderTest(unittest.TestCase):
    def test_replay_uses_only_ppg_and_never_transmits_reference_label(self) -> None:
        row = {
            "schema_version": "1.1",
            "sequence_number": 0,
            "sample_rate_hz": 125.0,
            "ppg": [0.1, 0.2, 0.3],
            "ppg_valid": [True, False, True],
            "reference_rhythm": "AF",
            "pseudo_ecg": [9.0, 9.0, 9.0],
        }

        payload = _chunk_payload(row, sequence_number=0, start_seconds=10.0)

        self.assertEqual(payload["ppg"], [0.1, None, 0.3])
        self.assertNotIn("reference_rhythm", payload)
        self.assertNotIn("pseudo_ecg", payload)

    def test_loader_rejects_non_contiguous_frames(self) -> None:
        rows = [
            {
                "schema_version": "1.1",
                "sequence_number": 2,
                "sample_rate_hz": 125.0,
                "ppg": [0.1],
                "ppg_valid": [True],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                _load_rows(path)

if __name__ == "__main__":
    unittest.main()
