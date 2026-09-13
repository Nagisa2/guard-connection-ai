from __future__ import annotations

import unittest

from downstream_windows import (
    DownstreamWindowBroker,
    DownstreamWindowConfig,
    DownstreamWindowError,
)

def _handoff(
    sequence: int,
    start: float,
    values: list[float],
    *,
    session_id: str = "session-1",
    rate: float = 4.0,
    valid: list[bool] | None = None,
    events: list[dict] | None = None,
    clock_domain: str = "verity-clock",
) -> dict:
    mask = valid or [True] * len(values)
    return {
        "schema_version": "1.0",
        "payload_type": "front_ai_device_handoff",
        "source_device_packet": {
            "sequence_number": sequence,
            "clock_domain": clock_domain,
        },
        "derived": {
            "session_id": session_id,
            "sample_count": len(values),
            "sample_rate_hz": rate,
            "input_timestamp_start_seconds": start,
            "ppg": values,
            "ppg_valid": mask,
            "pseudo_ecg": [value * 2 for value in values],
            "pseudo_ecg_valid": mask,
            "technical_sqi": 0.8,
            "pulse_interval_plausibility": 0.9,
            "beat_events": events or [],
        },
    }

def _event(sample_index: int, timestamp: float) -> dict:
    return {
        "event_sample_index": sample_index,
        "estimated_r_timestamp_seconds": timestamp,
        "amplitude": 1.0,
    }

class DownstreamWindowBrokerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DownstreamWindowConfig(
            window_seconds=4.0,
            stride_seconds=2.0,
            max_pending_windows=3,
        )

    def test_chunked_and_single_packet_have_equal_signal_window(self) -> None:
        chunked = DownstreamWindowBroker(self.config)
        chunked.ingest_ppg("session-1", _handoff(0, 10.0, list(range(8))))
        windows = chunked.ingest_ppg(
            "session-1", _handoff(1, 12.0, list(range(8, 16)))
        )

        single = DownstreamWindowBroker(self.config)
        expected = single.ingest_ppg(
            "session-1", _handoff(0, 10.0, list(range(16)))
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["ppg"], expected[0]["ppg"])
        self.assertEqual(windows[0]["start_seconds"], 10.0)
        self.assertEqual(windows[0]["end_seconds"], 14.0)
        self.assertNotIn("label", windows[0])

    def test_stride_beat_dedup_and_invalid_pseudo_samples(self) -> None:
        broker = DownstreamWindowBroker(self.config)
        events = [_event(2, 0.5), _event(2, 0.5), _event(10, 2.5)]
        windows = broker.ingest_ppg(
            "session-1",
            _handoff(
                0,
                0.0,
                list(range(24)),
                valid=[True] * 15 + [False] + [True] * 8,
                events=events,
            ),
        )

        self.assertEqual([item["input_sequence_id"] for item in windows], [0, 1])
        self.assertEqual(windows[0]["beats"]["event_sample_indices"], [2, 10])
        self.assertEqual(windows[1]["beats"]["event_sample_indices"], [10])
        self.assertIsNone(windows[0]["auxiliary_pseudo_ecg"]["values"][15])
        self.assertFalse(windows[0]["auxiliary_pseudo_ecg"]["diagnostic_ecg"])

    def test_motion_is_aligned_only_for_the_same_clock_domain(self) -> None:
        broker = DownstreamWindowBroker(self.config)
        for stream, clock in (("verity-acc", "verity-clock"), ("h10-acc", "h10-clock")):
            broker.ingest_motion(
                "session-1",
                {
                    "message_type": "front_ai_auxiliary_stream",
                    "session_id": "session-1",
                    "stream_id": stream,
                    "sensor_type": "accelerometer",
                    "units": "mG",
                    "configured_sample_rate_hz": 1.0,
                    "device_timestamp_start_ns": 0,
                    "device_timestamp_end_ns": 3_000_000_000,
                    "clock_domain": clock,
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "y": [1.0, 2.0, 3.0, 4.0],
                    "z": [1.0, 2.0, 3.0, 4.0],
                },
            )
        window = broker.ingest_ppg(
            "session-1", _handoff(0, 0.0, list(range(16)))
        )[0]

        self.assertEqual([item["stream_id"] for item in window["motion_streams"]], ["verity-acc"])
        self.assertEqual(window["motion_streams"][0]["coverage"], 1.0)
        self.assertEqual(window["unmapped_motion_streams"][0]["stream_id"], "h10-acc")

    def test_ack_is_idempotent_and_reconnect_repeats_pending_window(self) -> None:
        broker = DownstreamWindowBroker(self.config)
        broker.ingest_ppg("session-1", _handoff(0, 0.0, list(range(16))))
        first = broker.next_window("session-1")
        repeated = broker.next_window("session-1")
        self.assertEqual(first, repeated)
        self.assertTrue(broker.acknowledge("session-1", 0))
        self.assertFalse(broker.acknowledge("session-1", 0))
        self.assertIsNone(broker.next_window("session-1"))

    def test_gap_and_cross_session_state_are_explicit(self) -> None:
        broker = DownstreamWindowBroker(self.config)
        broker.ingest_ppg("first", _handoff(0, 0.0, [0.0] * 4, session_id="first"))
        broker.ingest_ppg("second", _handoff(0, 20.0, [0.0] * 4, session_id="second"))
        with self.assertRaises(DownstreamWindowError) as error:
            broker.ingest_ppg("first", _handoff(2, 1.0, [0.0] * 4, session_id="first"))
        self.assertEqual(error.exception.code, "packet_sequence_gap")
        broker.end_session("first")
        status = broker.status_snapshot()
        self.assertEqual(status["active_session_count"], 1)
        self.assertEqual(status["sessions"][0]["session_id"], "second")

if __name__ == "__main__":
    unittest.main()
