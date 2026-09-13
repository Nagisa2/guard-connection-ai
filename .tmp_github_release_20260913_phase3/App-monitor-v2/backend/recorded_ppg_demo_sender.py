"""記録済みPPGを通常の前段AI APIへ実時間速度で再生する。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from device_ppg_demo_sender import _delete, _post

DEFAULT_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "guard-connection-ai"
    / "artifacts"
    / "team_handoff"
    / "downstream_ai_v1_1"
    / "af_good.jsonl"
)

def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"recorded PPG sample was not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("recorded PPG sample is empty")
    rates = {float(row.get("sample_rate_hz", 0)) for row in rows}
    if len(rates) != 1 or next(iter(rates)) <= 0:
        raise ValueError("all recorded frames must use one positive sample rate")
    for expected_sequence, row in enumerate(rows):
        if row.get("schema_version") != "1.1":
            raise ValueError("recorded frame must use front AI schema 1.1")
        if row.get("sequence_number") != expected_sequence:
            raise ValueError("recorded frame sequence must be contiguous from zero")
        if not isinstance(row.get("ppg"), list) or not row["ppg"]:
            raise ValueError("recorded frame must contain PPG samples")
        if len(row.get("ppg_valid", [])) != len(row["ppg"]):
            raise ValueError("recorded PPG and valid mask lengths differ")
    return rows

def _chunk_payload(row: dict, *, sequence_number: int, start_seconds: float) -> dict:
    ppg = [
        value if valid else None
        for value, valid in zip(row["ppg"], row["ppg_valid"], strict=True)
    ]
    return {
        "schema_version": "1.0",
        "sequence_number": sequence_number,
        "input_timestamp_start_seconds": start_seconds,
        "ppg": ppg,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="test_user_02")
    parser.add_argument("--session-id", default="contest-recorded-af-test_user_02")
    parser.add_argument("--scenario-id", default="held-out-af-replay-v1")
    parser.add_argument("--deployment-profile-id", default="bidmc_train43_ppg_v1")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    sample_rate_hz = float(rows[0]["sample_rate_hz"])
    _delete(f"{args.base_url}/realtime/sessions/{args.session_id}")
    session = _post(
        f"{args.base_url}/realtime/sessions",
        {
            "schema_version": "1.0",
            "user_id": args.user_id,
            "session_id": args.session_id,
            "sample_rate_hz": sample_rate_hz,
            "deployment_profile_id": args.deployment_profile_id,
            "source_mode": "recorded_demo_replay",
            "demo_scenario_id": args.scenario_id,
        },
    )
    print(
        f"session={session['session_id']} user={session['user_id']} "
        f"source=recorded_demo_replay scenario={args.scenario_id}",
        flush=True,
    )
    print(
        "operator_note=AF reference selected; reference label is not sent to inference.",
        flush=True,
    )
    started = time.monotonic()
    sample_offset = 0
    interrupted = False
    try:
        sequence = 0
        while True:
            for row in rows:
                timestamp = sample_offset / sample_rate_hz
                request_started = time.monotonic()
                response = _post(
                    f"{args.base_url}/realtime/sessions/{args.session_id}/chunks",
                    _chunk_payload(
                        row,
                        sequence_number=sequence,
                        start_seconds=timestamp,
                    ),
                )
                frame = response["frame"]
                print(
                    f"sequence={sequence} samples={len(row['ppg'])} "
                    f"generation={frame['generation_accepted']} "
                    f"http_ms={(time.monotonic() - request_started) * 1000:.0f}",
                    flush=True,
                )
                sequence += 1
                sample_offset += len(row["ppg"])
                if not args.no_wait:
                    deadline = started + sample_offset / sample_rate_hz
                    time.sleep(max(0.0, deadline - time.monotonic()))
            if not args.loop:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nStop requested. Closing the replay session.", flush=True)
    finally:
        _delete(f"{args.base_url}/realtime/sessions/{args.session_id}")

    if interrupted:
        print("Recorded PPG replay stopped cleanly.", flush=True)
    else:
        print("Recorded PPG replay completed.", flush=True)

if __name__ == "__main__":
    main()
