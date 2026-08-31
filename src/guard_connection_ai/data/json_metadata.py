from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FILENAME_PATTERN = re.compile(r"(?P<subject_id>\d+)_(?P<signal_type>ECG|PPG)_(?P<recording_id>\d+)")


@dataclass(frozen=True)
class JsonSignalMetadata:
    subject_id: str
    signal_type: str
    recording_id: int
    path: Path
    num_data_records: float | None
    signal_shapes: dict[str, tuple[int, ...]]
    has_af_annotation: bool


def load_json_metadata(path: str | Path) -> JsonSignalMetadata:
    """Load one signal metadata JSON without pretending omitted arrays exist."""

    json_path = Path(path)
    match = _FILENAME_PATTERN.fullmatch(json_path.stem)
    if match is None:
        raise ValueError(f"Unsupported json_Data filename: {json_path.name}")
    payload: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict) or "shape" not in value:
            continue
        shape = value["shape"]
        if not isinstance(shape, list) or not all(isinstance(size, int) for size in shape):
            raise ValueError(f"Invalid shape metadata for {key}: {shape!r}")
        shapes[key] = tuple(shape)
    return JsonSignalMetadata(
        subject_id=match["subject_id"],
        signal_type=match["signal_type"],
        recording_id=int(match["recording_id"]),
        path=json_path,
        num_data_records=payload.get("num_data_records"),
        signal_shapes=shapes,
        has_af_annotation="AF_annotation" in payload,
    )


def find_json_metadata(data_root: str | Path) -> list[JsonSignalMetadata]:
    """Load all supported json_Data metadata files in stable order."""

    return [load_json_metadata(path) for path in sorted(Path(data_root).glob("*.json"))]
