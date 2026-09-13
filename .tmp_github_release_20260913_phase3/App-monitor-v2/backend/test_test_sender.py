from __future__ import annotations

import unittest

from test_sender import build_payloads, generate_ecg_values

class TestSenderPayloadTest(unittest.TestCase):
    def test_ecg_only_mode_is_explicitly_supplemental_and_non_diagnostic(self) -> None:
        payloads = build_payloads(
            user_id="test_user_01",
            sequence_number=3,
            base_time=1000,
            time_deltas=[0, 10],
            ecg_values=[0.1, 0.2],
            ppg_values=[0.3, 0.4],
            hr_value=70.0,
            signal_types={"ECG"},
        )

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["signal_type"], "ECG")
        self.assertEqual(payload["waveform_type"], "simulated_ecg")
        self.assertFalse(payload["diagnostic_ecg"])
        self.assertEqual(payload["display_role"], "supplemental_simulated_ecg")

    def test_ecg_beat_period_does_not_depend_on_packet_length(self) -> None:
        short = generate_ecg_values(1, 25, 0)
        combined = generate_ecg_values(1, 100, 0)
        self.assertEqual(short, combined[:25])

if __name__ == "__main__":
    unittest.main()
