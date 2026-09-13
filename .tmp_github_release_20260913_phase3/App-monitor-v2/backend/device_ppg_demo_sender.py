"""実機なしで135 Hz・4ch PPGから医師画面まで動かすデモ送信器。"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

def _post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8")) from error

def _delete(url: str) -> None:
    try:
        with urlopen(Request(url, method="DELETE"), timeout=5):
            pass
    except HTTPError as error:
        if error.code != 404:
            raise

def _end_sessions(base_url: str, sessions: list[tuple[int, str, str]]) -> None:
    """Best-effortでデモ用サーバーセッションを終了する。"""
    for _, user_id, session_id in sessions:
        try:
            _delete(f"{base_url}/realtime/sessions/{session_id}")
            print(f"session_ended={session_id} user={user_id}", flush=True)
        except (OSError, RuntimeError) as error:
            print(
                f"session_end_warning={session_id} error={error}",
                flush=True,
            )

def _packet(
    sequence: int,
    *,
    rate: int,
    seconds: float,
    origin_ns: int,
    stream_id: str,
    patient_index: int,
    scenario_id: str = "synthetic-normal-v1",
) -> dict:
    count = round(rate * seconds)
    start = sequence * count
    heart_rate_hz = 0.97 + 0.10 * patient_index
    base = [
        round(
            500_000
            + 58_000
            * math.exp(
                -(((((start + index) / rate * heart_rate_hz) % 1.0) - 0.16) / 0.075) ** 2
            )
            + 21_000
            * math.exp(
                -(((((start + index) / rate * heart_rate_hz) % 1.0) - 0.38) / 0.14) ** 2
            )
            - 8_000
            * math.exp(
                -(((((start + index) / rate * heart_rate_hz) % 1.0) - 0.49) / 0.055) ** 2
            )
            + 2_500 * math.sin(2 * math.pi * 0.18 * (start + index) / rate)
            + 700 * math.sin(2 * math.pi * 17 * (start + index) / rate + patient_index)
        )
        for index in range(count)
    ]
    end_index = start + count - 1
    return {
        "schema_version": "1.0",
        "stream_id": stream_id,
        "sequence_number": sequence,
        "configured_sample_rate_hz": rate,
        "estimated_sample_rate_hz": float(rate),
        "source_mode": "synthetic_demo",
        "demo_scenario_id": scenario_id,
        "device_timestamp_end_ns": origin_ns + round(end_index / rate * 1_000_000_000),
        "ppg0": base,
        "ppg1": [value + 250 for value in base],
        "ppg2": [value - 180 for value in base],
        "ambient0": [12_000] * count,
        "firmware_version": "demo",
        "sdk_version": "demo",
        "sensor_disconnected": False,
        "saturation": {},
        "device_sqi": {
            "skewness": 0.7,
            "perfusion_index": 0.02,
            "motion_level": 0.05,
            "algorithm_version": "demo-0.1",
            "window_seconds": float(seconds),
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--user-ids", nargs="+", default=["test_user_01", "test_user_02"]
    )
    parser.add_argument("--session-prefix", default="contest-device-demo")
    parser.add_argument("--packets", type=int, default=240)
    parser.add_argument("--packet-seconds", type=float, default=0.25)
    parser.add_argument("--source-rate", type=int, default=135)
    parser.add_argument(
        "--deployment-profile-id",
        default="bidmc_train43_ppg_v1",
    )
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()
    if args.packets <= 0 or args.packet_seconds <= 0 or args.source_rate <= 0:
        raise ValueError("packets, packet-seconds, and source-rate must be positive.")

    sessions = []
    for patient_index, user_id in enumerate(args.user_ids):
        session_id = f"{args.session_prefix}-{user_id}"
        _delete(f"{args.base_url}/realtime/sessions/{session_id}")
        session = _post(
            f"{args.base_url}/realtime/sessions",
            {
                "schema_version": "1.0",
                "user_id": user_id,
                "session_id": session_id,
                "sample_rate_hz": 125,
                "deployment_profile_id": args.deployment_profile_id,
                "source_mode": "synthetic_demo",
                "demo_scenario_id": "synthetic-normal-v1",
            },
        )
        sessions.append((patient_index, user_id, session["session_id"]))
        print(f"session={session['session_id']} user={session['user_id']}", flush=True)
    origin_ns = 700_000_000_000_000_000
    started = time.monotonic()

    def send_one(session, sequence):
        patient_index, user_id, session_id = session
        request_started = time.monotonic()
        response = _post(
            f"{args.base_url}/realtime/sessions/{session_id}/device-ppg-chunks",
            _packet(
                sequence,
                rate=args.source_rate,
                seconds=args.packet_seconds,
                origin_ns=origin_ns,
                stream_id=f"verity-demo-{user_id}",
                patient_index=patient_index,
            ),
        )
        visualization = response.get("visualization") or {}
        downstream = (response.get("downstream_ai") or {}).get("derived") or {}
        return (
            f"{user_id}:n={response['resampled_sample_count']},"
            f"display={visualization.get('generation_accepted')},"
            f"downstream={downstream.get('generation_accepted')},"
            f"http_ms={(time.monotonic() - request_started) * 1000:.0f}"
        )

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
            for sequence in range(args.packets):
                cycle_started = time.monotonic()
                futures = [
                    executor.submit(send_one, session, sequence) for session in sessions
                ]
                statuses = [future.result() for future in futures]
                cycle_ms = (time.monotonic() - cycle_started) * 1000
                print(
                    f"sequence={sequence} cycle_ms={cycle_ms:.0f} "
                    + " | ".join(statuses),
                    flush=True,
                )
                if not args.no_wait:
                    deadline = started + (sequence + 1) * args.packet_seconds
                    time.sleep(max(0.0, deadline - time.monotonic()))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStop requested. Closing demo sessions.", flush=True)
    finally:
        _end_sessions(args.base_url, sessions)

    if interrupted:
        print("Demo sender stopped cleanly.", flush=True)
    else:
        print("Demo sender completed.", flush=True)

if __name__ == "__main__":
    main()
