from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from guard_connection_ai.deployment.profile_artifact import (
    artifact_digest,
    build_bidmc_ppg_profile_artifact,
    write_profile_artifact,
)


def _write_signal(path: Path, ppg: list[float]) -> None:
    pd.DataFrame({"PLETH": ppg, "II": [0.0] * len(ppg)}).to_csv(path, index=False)


def test_artifact_uses_train_subjects_only_and_has_verifiable_hash(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_signal(data_root / "bidmc_01_Signals.csv", [0.0, 1.0, 2.0])
    _write_signal(data_root / "bidmc_02_Signals.csv", [1000.0, 1001.0, 1002.0])
    split_path = tmp_path / "split.yaml"
    split_path.write_text(yaml.safe_dump({
        "seed": 42,
        "split": {
            "method": "subject_wise",
            "train_subjects": ["bidmc01"],
            "validation_subjects": ["bidmc02"],
        },
    }), encoding="utf-8")
    artifact = build_bidmc_ppg_profile_artifact(
        data_root=data_root, split_path=split_path, sample_rate_hz=125
    )
    assert artifact["scaler"]["center"] == 1.0
    assert [source["subject_id"] for source in artifact["sources"]] == ["bidmc01"]
    digest = artifact.pop("artifact_sha256")
    artifact.pop("artifact_version")
    assert digest == artifact_digest(artifact)


def test_artifact_writer_is_json_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    payload = {"schema_version": "1.0", "profile_id": "test"}
    write_profile_artifact(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
