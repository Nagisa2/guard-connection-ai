"""後段AI担当向け: 認証付きWebSocketで前段AI出力を受け取る最小例。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from websockets.asyncio.client import connect

def validate_front_ai_message(payload: dict) -> str:
    payload_type = payload.get("payload_type")
    if payload_type == "front_ai_device_handoff":
        derived = payload.get("derived")
        if not isinstance(derived, dict):
            raise ValueError("derived frame is missing")
        required = {
            "session_id",
            "sequence_number",
            "ppg",
            "ppg_valid",
            "beat_events",
            "technical_sqi",
            "signal_coverage",
            "generation_accepted",
        }
        missing = sorted(required - set(derived))
        if missing:
            raise ValueError(f"missing derived fields: {', '.join(missing)}")
        if len(derived["ppg"]) != len(derived["ppg_valid"]):
            raise ValueError("ppg and ppg_valid lengths differ")
        return "ppg"
    if payload.get("message_type") == "front_ai_auxiliary_stream":
        if payload.get("sensor_type") not in {"accelerometer", "gyroscope"}:
            raise ValueError("unsupported auxiliary sensor type")
        return "motion"
    raise ValueError("unsupported front AI message")

async def receive(base_url: str, session_id: str, api_key: str, once: bool) -> None:
    url = f"{base_url.rstrip('/')}/ws/front-ai/downstream/{session_id}"
    async with connect(url, additional_headers={"x-downstream-api-key": api_key}) as socket:
        async for raw in socket:
            payload = json.loads(raw)
            kind = validate_front_ai_message(payload)
            if kind == "ppg":
                derived = payload["derived"]
                print(
                    f"ppg seq={derived['sequence_number']} samples={len(derived['ppg'])} "
                    f"beats={len(derived['beat_events'])} sqi={derived['technical_sqi']:.3f}"
                )
            else:
                print(
                    f"motion stream={payload['stream_id']} seq={payload['sequence_number']} "
                    f"samples={payload['sample_count']}"
                )
            if once:
                return

def main() -> None:
    parser = argparse.ArgumentParser(description="前段AI出力WebSocketの受信例")
    parser.add_argument("session_id")
    parser.add_argument("--base-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="GUARD_DOWNSTREAM_API_KEY")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is required")
    asyncio.run(receive(args.base_url, args.session_id, api_key, args.once))

if __name__ == "__main__":
    main()
