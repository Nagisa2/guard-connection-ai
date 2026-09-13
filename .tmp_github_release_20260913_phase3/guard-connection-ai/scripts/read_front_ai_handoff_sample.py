from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_frames(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("sample contains no frames.")
    for expected, row in enumerate(rows):
        if row["sequence_number"] != expected:
            raise ValueError(f"sequence gap: expected={expected}, actual={row['sequence_number']}")
        if not (
            len(row["ppg"])
            == len(row["ppg_valid"])
            == len(row["pseudo_ecg"])
            == len(row["pseudo_ecg_valid"])
        ):
            raise ValueError(f"array length mismatch at sequence={expected}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="前段AI引渡しJSONLを検査して読む。")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    frames = read_frames(args.path)
    accepted = sum(bool(frame["generation_accepted"]) for frame in frames)
    print(f"frames={len(frames)} accepted={accepted} session={frames[0]['session_id']}")
