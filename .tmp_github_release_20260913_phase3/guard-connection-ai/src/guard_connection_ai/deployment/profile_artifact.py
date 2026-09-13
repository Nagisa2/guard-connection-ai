from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from guard_connection_ai.data.causal_preprocessing import fit_fixed_robust_scaler
from guard_connection_ai.data.dataset import find_subject_files, load_subject_signals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_bidmc_ppg_profile_artifact(
    *,
    data_root: str | Path,
    split_path: str | Path,
    profile_id: str = "bidmc_train43_ppg_v1",
    sample_rate_hz: float = 125.0,
) -> dict[str, object]:
    data_root = Path(data_root)
    split_path = Path(split_path)
    split_document = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    split = split_document.get("split", {})
    if split.get("method") != "subject_wise":
        raise ValueError("deployment artifact requires a subject-wise split.")
    train_subjects = list(split.get("train_subjects", []))
    validation_subjects = list(split.get("validation_subjects", []))
    if not train_subjects or set(train_subjects) & set(validation_subjects):
        raise ValueError("train subjects must be non-empty and disjoint from validation.")

    available = dict(find_subject_files(data_root))
    missing = sorted(set(train_subjects) - set(available))
    if missing:
        raise FileNotFoundError(f"training subject files are missing: {missing}")

    training_signals = []
    sources = []
    finite_sample_count = 0
    for subject_id in train_subjects:
        path = available[subject_id]
        ppg, _ = load_subject_signals(path)
        training_signals.append(ppg)
        finite_count = int(np.isfinite(ppg).sum())
        finite_sample_count += finite_count
        sources.append({
            "subject_id": subject_id,
            "file_name": path.name,
            "sha256": _sha256(path),
            "sample_count": int(ppg.size),
            "finite_ppg_sample_count": finite_count,
        })
    scaler = fit_fixed_robust_scaler(training_signals)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "dataset": "BIDMC",
        "signal_column": "PLETH",
        "expected_sample_rate_hz": sample_rate_hz,
        "split": {
            "method": "subject_wise",
            "seed": split_document.get("seed"),
            "split_file_sha256": _sha256(split_path),
            "train_subjects": train_subjects,
            "excluded_validation_subjects": validation_subjects,
        },
        "sources": sources,
        "scaler": {
            "method": "median_mad_times_1.4826",
            "center": scaler.center,
            "scale": scaler.scale,
            "finite_training_sample_count": finite_sample_count,
        },
        "streaming": {
            "pulse_arrival_seconds": 0.2,
            "display_delay_seconds": 0.5,
            "quality_window_seconds": 10.0,
        },
        "clinically_validated": False,
        "af_model_connected": False,
        "intended_use": "research_preprocessing_only",
        "limitations": [
            "BIDMC is not the target AF deployment population.",
            "PAT and SQI thresholds are not clinically calibrated.",
            "This artifact does not contain or validate an AF classifier.",
        ],
    }
    digest = artifact_digest(payload)
    payload["artifact_sha256"] = digest
    payload["artifact_version"] = f"sha256:{digest[:16]}"
    return payload


def write_profile_artifact(payload: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
