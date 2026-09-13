"""正常例と記録済みAF例を同期再生し、デモ用AF疑いイベントを送る。"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from device_ppg_demo_sender import _delete, _end_sessions, _post
from recorded_ppg_demo_sender import DEFAULT_SAMPLE, _chunk_payload, _load_rows

def _normal_ppg(start_index: int, count: int, sample_rate_hz: float) -> list[float]:
    values = []
    for offset in range(count):
        phase = ((start_index + offset) / sample_rate_hz * 1.05) % 1.0
        values.append(
            0.45
            + 0.22 * math.exp(-((phase - 0.16) / 0.075) ** 2)
            + 0.07 * math.exp(-((phase - 0.38) / 0.14) ** 2)
            - 0.025 * math.exp(-((phase - 0.49) / 0.055) ** 2)
        )
    return values

def _synthetic_irregular_rows(
    *, sample_rate_hz: float = 125.0, duration_seconds: float = 15.0
) -> list[dict]:
    """公開デモ用の合成不規則脈波。実患者データやAF正解例ではない。"""
    count = round(sample_rate_hz * duration_seconds)
    beat_times = []
    beat_time = 0.45
    intervals = (0.62, 1.08, 0.71, 1.24, 0.83, 0.66, 1.15, 0.74)
    interval_index = 0
    while beat_time < duration_seconds + 0.5:
        beat_times.append(beat_time)
        beat_time += intervals[interval_index % len(intervals)]
        interval_index += 1
    signal = []
    for sample_index in range(count):
        timestamp = sample_index / sample_rate_hz
        value = 0.44 + 0.008 * math.sin(2 * math.pi * 0.17 * timestamp)
        for pulse_time in beat_times:
            relative = timestamp - pulse_time
            if -0.08 <= relative <= 0.65:
                value += 0.23 * math.exp(-((relative - 0.08) / 0.07) ** 2)
                value += 0.065 * math.exp(-((relative - 0.29) / 0.13) ** 2)
                value -= 0.022 * math.exp(-((relative - 0.40) / 0.05) ** 2)
        signal.append(value)
    rows = []
    for sequence, start in enumerate(range(0, count, 31)):
        chunk = signal[start : start + 31]
        rows.append(
            {
                "schema_version": "1.1",
                "sequence_number": sequence,
                "sample_rate_hz": sample_rate_hz,
                "ppg": chunk,
                "ppg_valid": [True] * len(chunk),
            }
        )
    return rows

def _create_session(
    base_url: str,
    *,
    user_id: str,
    session_id: str,
    scenario_id: str,
    source_mode: str,
    sample_rate_hz: float,
    deployment_profile_id: str,
) -> dict:
    _delete(f"{base_url}/realtime/sessions/{session_id}")
    return _post(
        f"{base_url}/realtime/sessions",
        {
            "schema_version": "1.0",
            "user_id": user_id,
            "session_id": session_id,
            "sample_rate_hz": sample_rate_hz,
            "deployment_profile_id": deployment_profile_id,
            "source_mode": source_mode,
            "demo_scenario_id": scenario_id,
        },
    )

def _create_demo_downstream_inference(
    base_url: str,
    *,
    session_id: str,
    inference_sequence_number: int,
    input_sequence_id: int,
    window_start_seconds: float,
    window_end_seconds: float,
    stride_seconds: float,
    decision: str,
    episode_state: str,
    episode_start_seconds: float | None = None,
) -> dict:
    is_af_suspected = decision == "af_suspected"
    episode_id = (
        f"{session_id}:demo-episode-v1" if episode_state != "inactive" else None
    )
    return _post(
        f"{base_url}/downstream-inferences",
        {
            "schema_version": "downstream_v1",
            "session_id": session_id,
            "inference_sequence_number": inference_sequence_number,
            "model_version": "contest-demo-stub-v2",
            "inference_mode": "demo_stub",
            "frontend_schema_version": "1.1",
            "window": {
                "input_sequence_id": input_sequence_id,
                "start_seconds": window_start_seconds,
                "end_seconds": window_end_seconds,
                "window_seconds": window_end_seconds - window_start_seconds,
                "stride_seconds": stride_seconds,
            },
            "decidable": True,
            "abstention_reason": None,
            "af_probability": 0.91 if is_af_suspected else 0.08,
            "probability_is_calibrated": False,
            "decision": decision,
            "episode": {
                "state": episode_state,
                "episode_id": episode_id,
                "start_seconds": episode_start_seconds,
                "duration_seconds": (
                    window_end_seconds - episode_start_seconds
                    if episode_start_seconds is not None
                    else 0.0
                ),
            },
            "context": {
                "valid_ratio": 0.96,
                "n_valid_beats": max(
                    2, round((window_end_seconds - window_start_seconds) * 1.3)
                ),
                "sqi_window": 0.90,
                "gate_value": 0.82 if is_af_suspected else 0.18,
                "used_morphology": False,
                "frontend_model_version": "contest-front-demo-v1",
            },
            "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

def _apply_demo_action(base_url: str, event_id: str, action: str) -> dict:
    actor_role = "patient" if action.startswith("patient_") else "system"
    return _post(
        f"{base_url}/care-events/{event_id}/actions",
        {
            "idempotency_key": f"{event_id}:{action}:demo-v1",
            "action": action,
            "actor_role": actor_role,
        },
    )

def _send_demo_location(
    base_url: str,
    event_id: str,
    *,
    latitude: float,
    longitude: float,
) -> dict:
    return _post(
        f"{base_url}/care-events/{event_id}/actions",
        {
            "idempotency_key": f"{event_id}:demo-location-v1",
            "action": "update_location",
            "actor_role": "system",
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy_m": 15.0,
                "captured_at_seconds": time.time(),
                "provider": "demo_fixed_coordinate",
            },
        },
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--normal-user-id", default="test_user_01")
    parser.add_argument("--af-user-id", default="test_user_02")
    parser.add_argument("--af-patient-id", type=int, default=2)
    parser.add_argument("--trigger-seconds", type=float, default=5.0)
    parser.add_argument("--confirmation-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--patient-action",
        choices=["none", "patient_ok", "patient_unwell", "patient_help"],
        default="none",
    )
    parser.add_argument("--action-delay-seconds", type=float, default=3.0)
    parser.add_argument("--demo-latitude", type=float, default=33.938502)
    parser.add_argument("--demo-longitude", type=float, default=132.190863)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--deployment-profile-id", default="bidmc_train43_ppg_v1"
    )
    args = parser.parse_args()
    if (
        args.trigger_seconds < 0
        or args.confirmation_delay_seconds < 0
        or args.action_delay_seconds < 0
    ):
        raise ValueError("trigger and action delays must be non-negative")

    demo_run_id = uuid4().hex[:8]
    if args.input.is_file():
        rows = _load_rows(args.input)
        af_source_mode = "recorded_demo_replay"
        scenario_id = f"held-out-af-replay-v1-{demo_run_id}"
        af_source_label = "held-out_record"
    else:
        rows = _synthetic_irregular_rows()
        af_source_mode = "synthetic_demo"
        scenario_id = f"synthetic-irregular-pulse-v1-{demo_run_id}"
        af_source_label = "synthetic_irregular_not_reference_AF"
    sample_rate_hz = float(rows[0]["sample_rate_hz"])
    normal_session_id = f"contest-normal-{args.normal_user_id}-{demo_run_id}"
    af_session_id = f"contest-recorded-af-{args.af_user_id}-{demo_run_id}"
    sessions = [
        (0, args.normal_user_id, normal_session_id),
        (1, args.af_user_id, af_session_id),
    ]
    _create_session(
        args.base_url,
        user_id=args.normal_user_id,
        session_id=normal_session_id,
        scenario_id=f"synthetic-normal-v1-{demo_run_id}",
        source_mode="synthetic_demo",
        sample_rate_hz=sample_rate_hz,
        deployment_profile_id=args.deployment_profile_id,
    )
    _create_session(
        args.base_url,
        user_id=args.af_user_id,
        session_id=af_session_id,
        scenario_id=scenario_id,
        source_mode=af_source_mode,
        sample_rate_hz=sample_rate_hz,
        deployment_profile_id=args.deployment_profile_id,
    )
    print(
        f"demo_run_id={demo_run_id} demo_mode=true normal=synthetic "
        f"af={af_source_label} "
        "inference=demo_stub_not_model",
        flush=True,
    )

    started = time.monotonic()
    sample_offset = 0
    sequence = 0
    event: dict | None = None
    candidate_end_seconds: float | None = None
    action_sent = False
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                for row in rows:
                    if args.stop_file and args.stop_file.exists():
                        interrupted = True
                        break
                    frame_start_seconds = sample_offset / sample_rate_hz
                    count = len(row["ppg"])
                    normal_request = {
                        "schema_version": "1.0",
                        "sequence_number": sequence,
                        "input_timestamp_start_seconds": frame_start_seconds,
                        "ppg": _normal_ppg(sample_offset, count, sample_rate_hz),
                    }
                    af_request = _chunk_payload(
                        row,
                        sequence_number=sequence,
                        start_seconds=frame_start_seconds,
                    )
                    normal_future = executor.submit(
                        _post,
                        f"{args.base_url}/realtime/sessions/{normal_session_id}/chunks",
                        normal_request,
                    )
                    af_future = executor.submit(
                        _post,
                        f"{args.base_url}/realtime/sessions/{af_session_id}/chunks",
                        af_request,
                    )
                    normal_frame = normal_future.result()["frame"]
                    af_frame = af_future.result()["frame"]
                    sample_offset += count
                    elapsed_signal_seconds = sample_offset / sample_rate_hz
                    if (
                        candidate_end_seconds is None
                        and elapsed_signal_seconds >= args.trigger_seconds
                    ):
                        candidate_end_seconds = elapsed_signal_seconds
                        candidate_window_start = 0.0
                        candidate_window_seconds = (
                            candidate_end_seconds - candidate_window_start
                        )
                        _create_demo_downstream_inference(
                            args.base_url,
                            session_id=normal_session_id,
                            inference_sequence_number=0,
                            input_sequence_id=sequence,
                            window_start_seconds=candidate_window_start,
                            window_end_seconds=candidate_end_seconds,
                            stride_seconds=candidate_window_seconds,
                            decision="no_af_suspected",
                            episode_state="inactive",
                        )
                        _create_demo_downstream_inference(
                            args.base_url,
                            session_id=af_session_id,
                            inference_sequence_number=0,
                            input_sequence_id=sequence,
                            window_start_seconds=candidate_window_start,
                            window_end_seconds=candidate_end_seconds,
                            stride_seconds=candidate_window_seconds,
                            decision="af_suspected",
                            episode_state="candidate",
                            episode_start_seconds=candidate_window_start,
                        )
                        print(
                            "downstream_state=candidate inference=demo_stub",
                            flush=True,
                        )
                    if (
                        event is None
                        and candidate_end_seconds is not None
                        and elapsed_signal_seconds
                        >= candidate_end_seconds + args.confirmation_delay_seconds
                    ):
                        stride_seconds = elapsed_signal_seconds - candidate_end_seconds
                        confirmed = _create_demo_downstream_inference(
                            args.base_url,
                            session_id=af_session_id,
                            inference_sequence_number=1,
                            input_sequence_id=sequence,
                            window_start_seconds=(
                                elapsed_signal_seconds - candidate_end_seconds
                            ),
                            window_end_seconds=elapsed_signal_seconds,
                            stride_seconds=stride_seconds,
                            decision="af_suspected",
                            episode_state="confirmed",
                            episode_start_seconds=0.0,
                        )
                        event = confirmed["care_event"]["event"]
                        print(
                            f"care_event={event['event_id']} state={event['state']} "
                            "downstream_state=confirmed inference=demo_stub",
                            flush=True,
                        )
                        event = _send_demo_location(
                            args.base_url,
                            event["event_id"],
                            latitude=args.demo_latitude,
                            longitude=args.demo_longitude,
                        )["event"]
                    if (
                        event is not None
                        and not action_sent
                        and args.patient_action != "none"
                        and elapsed_signal_seconds
                        >= args.trigger_seconds + args.action_delay_seconds
                    ):
                        event = _apply_demo_action(
                            args.base_url, event["event_id"], args.patient_action
                        )["event"]
                        action_sent = True
                        print(
                            f"care_action={args.patient_action} state={event['state']}",
                            flush=True,
                        )
                    print(
                        f"sequence={sequence} signal_s={elapsed_signal_seconds:.2f} "
                        f"normal_display={normal_frame['generation_accepted']} "
                        f"af_display={af_frame['generation_accepted']}",
                        flush=True,
                    )
                    sequence += 1
                    if not args.no_wait:
                        deadline = started + elapsed_signal_seconds
                        time.sleep(max(0.0, deadline - time.monotonic()))
                if interrupted:
                    break
                if not args.loop:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nStop requested. Closing demo sessions.", flush=True)
    finally:
        _end_sessions(args.base_url, sessions)

    print(
        "Demo stopped cleanly." if interrupted else "Demo scenario completed.",
        flush=True,
    )

if __name__ == "__main__":
    main()
