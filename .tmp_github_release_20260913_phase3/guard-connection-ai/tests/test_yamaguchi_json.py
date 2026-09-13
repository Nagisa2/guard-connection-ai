from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from guard_connection_ai.data.yamaguchi_json import (
    YamaguchiSignalDataUnavailable,
    audit_yamaguchi_export,
    load_yamaguchi_ppg_recording,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metadata_only_export_fails_closed(tmp_path: Path) -> None:
    omitted = {
        "shape": [1, 200],
        "dtype": "float64",
        "note": "Large dataset omitted from full JSON export",
    }
    _write(
        tmp_path / "01_PPG_01.json",
        {
            "PPG_GREEN": omitted,
            "num_data_records": 2,
            "timestamp_sec_data_record": [[0, 1]],
        },
    )

    audit = audit_yamaguchi_export(tmp_path)
    assert audit.subject_count == 1
    assert audit.ppg_samples_per_record == (100.0,)
    assert audit.inferred_ppg_sample_rate_hz == pytest.approx(100.0)
    assert audit.can_run_partial_front_ai is False
    assert audit.can_run_full_cohort_front_ai is False
    assert audit.can_evaluate_r_timing is False
    with pytest.raises(YamaguchiSignalDataUnavailable, match="metadata-only"):
        load_yamaguchi_ppg_recording(tmp_path / "01_PPG_01.json")


def test_full_ppg_export_loads_without_preprocessing_values(tmp_path: Path) -> None:
    ppg = np.arange(200, dtype=float).reshape(1, -1).tolist()
    ambient = (1000 + np.arange(200)).reshape(1, -1).tolist()
    acceleration = np.arange(100, dtype=float).reshape(1, -1).tolist()
    _write(
        tmp_path / "08_PPG_07.json",
        {
            "PPG_GREEN": ppg,
            "PPG_AMBIENT": ambient,
            "Accelerometer_X": acceleration,
            "Accelerometer_Y": acceleration,
            "Accelerometer_Z": acceleration,
            "num_data_records": 2,
            "timestamp_sec_data_record": [[10.0, 11.0]],
        },
    )

    recording = load_yamaguchi_ppg_recording(tmp_path / "08_PPG_07.json")
    assert recording.subject_id == "08"
    assert recording.recording_id == 7
    assert recording.sample_rate_hz == pytest.approx(100.0)
    np.testing.assert_array_equal(recording.ppg_green, np.arange(200))
    assert recording.ppg_ambient is not None
    assert recording.accelerometer is not None
    assert recording.accelerometer.shape == (3, 100)
    audit = audit_yamaguchi_export(tmp_path)
    assert audit.can_run_partial_front_ai is True
    assert audit.can_run_full_cohort_front_ai is True
