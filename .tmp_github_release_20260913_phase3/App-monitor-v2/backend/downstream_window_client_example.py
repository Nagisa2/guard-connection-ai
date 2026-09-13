"""後段AI担当向け: 30秒窓を取得し、処理後にACKする最小例。"""

from __future__ import annotations

import argparse
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def _request_json(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    retries: int = 3,
) -> dict | None:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "x-downstream-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=5.0) as response:
                if response.status == 204:
                    return None
                return json.loads(response.read())
        except HTTPError as error:
            if error.code < 500 or attempt == retries - 1:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        except URLError as error:
            if attempt == retries - 1:
                raise RuntimeError(f"backend connection failed: {error}") from error
        time.sleep(0.25 * 2**attempt)
    raise RuntimeError("request retries exhausted")

def validate_window(window: dict) -> None:
    if window.get("schema_version") != "front_ai_window_v1":
        raise ValueError("unsupported front AI window schema")
    if window.get("primary_signal") != "ppg":
        raise ValueError("PPG must remain the primary signal")
    if "label" in window:
        raise ValueError("live inference input must not contain a training label")
    ppg = window.get("ppg", {})
    if len(ppg.get("values", [])) != len(ppg.get("valid_mask", [])):
        raise ValueError("PPG values and valid mask lengths differ")
    beats = window.get("beats", {})
    if beats.get("source") != "ppg":
        raise ValueError("unexpected beat source")
    pseudo = window.get("auxiliary_pseudo_ecg", {})
    if pseudo.get("diagnostic_ecg") is not False:
        raise ValueError("pseudo ECG must be explicitly non-diagnostic")

def acknowledge(base_url: str, session_id: str, api_key: str, sequence: int) -> dict:
    result = _request_json(
        f"{base_url.rstrip('/')}/front-ai/windows/{session_id}/ack",
        api_key,
        method="POST",
        payload={
            "schema_version": "front_ai_window_v1",
            "input_sequence_id": sequence,
        },
    )
    assert result is not None
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="後段AI向け未ラベル窓の取得確認")
    parser.add_argument("session_id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key-env", default="GUARD_DOWNSTREAM_API_KEY")
    parser.add_argument(
        "--ack",
        action="store_true",
        help="外部モデル処理が成功したものとして窓をACKする",
    )
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is required")
    window = _request_json(
        f"{args.base_url.rstrip('/')}/front-ai/windows/{args.session_id}/next",
        api_key,
    )
    if window is None:
        print("pending window: none")
        return
    validate_window(window)
    print(
        f"window={window['input_sequence_id']} "
        f"samples={len(window['ppg']['values'])} "
        f"beats={len(window['beats']['times_sec'])} "
        f"coverage={window['quality']['signal_coverage']:.3f}"
    )
    if args.ack:
        result = acknowledge(
            args.base_url,
            args.session_id,
            api_key,
            int(window["input_sequence_id"]),
        )
        print(f"ack changed={result['changed']}")
    else:
        print("ACKしていません。モデル処理成功後だけ --ack を使用してください。")

if __name__ == "__main__":
    main()
