"""村川さん側で生成したJSON/gzip JSONをサーバーと同じschemaで検証する。"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from pydantic import ValidationError

import schemas

def validate_payload(payload: dict) -> tuple[str, int]:
    if "sensor_type" in payload:
        parsed = schemas.DeviceMotionChunkRequest(**payload)
        return "motion", len(parsed.x)
    parsed = schemas.DevicePPGChunkRequest(**payload)
    return "ppg", len(parsed.ppg0)

def _read(path: Path) -> object:
    raw = path.read_bytes()
    if path.suffix.lower() == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))

def main() -> None:
    parser = argparse.ArgumentParser(description="端末送信payloadの互換性を検証します。")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        try:
            document = _read(path)
            payloads = document if isinstance(document, list) else [document]
            for index, payload in enumerate(payloads):
                if not isinstance(payload, dict):
                    raise ValueError("payload must be a JSON object")
                kind, count = validate_payload(payload)
                print(f"OK {path} item={index} kind={kind} samples={count}")
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
            failed = True
            print(f"ERROR {path}: {error}")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
