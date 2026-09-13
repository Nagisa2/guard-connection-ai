from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.data.mimic_perform_af import (
    discover_mimic_perform_af,
    load_mimic_perform_signals,
)
from guard_connection_ai.schemas.front_ai_outputs import (
    to_doctor_visualization_frame,
    to_downstream_ai_frame,
)
from guard_connection_ai.streaming.realtime_session import RealtimePPGSession


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _schema(payload_type: str, doctor: bool) -> dict[str, object]:
    required = [
        "schema_version",
        "payload_type",
        "session_id",
        "sequence_number",
        "sample_rate_hz",
        "input_timestamp_start_seconds",
        "ppg",
        "ppg_valid",
        "pseudo_ecg",
        "pseudo_ecg_valid",
        "ppg_sqi",
        "technical_sqi",
        "signal_coverage",
        "pulse_interval_plausibility",
        "generation_accepted",
        "generation_abstention_reason",
        "generation_mode",
        "morphology_source",
        "timing_confidence",
        "morphology_confidence",
        "repolarization_duration_prior_ms",
        "p_wave_generated",
        "qt_measurement_supported",
        "pq_pr_measurement_supported",
    ]
    properties: dict[str, object] = {
        "schema_version": {"const": "1.1"},
        "payload_type": {"const": payload_type},
        "session_id": {"type": "string"},
        "sequence_number": {"type": "integer", "minimum": 0},
        "sample_rate_hz": {"type": "number", "exclusiveMinimum": 0},
        "input_timestamp_start_seconds": {"type": "number"},
        "ppg": {"type": "array", "items": {"type": "number"}},
        "ppg_valid": {"type": "array", "items": {"type": "boolean"}},
        "pseudo_ecg": {"type": "array", "items": {"type": "number"}},
        "pseudo_ecg_valid": {"type": "array", "items": {"type": "boolean"}},
        "ppg_sqi": {"type": "number", "minimum": 0, "maximum": 1},
        "technical_sqi": {"type": "number", "minimum": 0, "maximum": 1},
        "signal_coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "pulse_interval_plausibility": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "generation_accepted": {"type": "boolean"},
        "generation_abstention_reason": {"type": ["string", "null"]},
        "generation_mode": {
            "enum": [
                "interval_template",
                "population_prior",
                "learned_morphology",
                "blank",
            ]
        },
        "morphology_source": {"type": "string"},
        "timing_confidence": {"type": ["number", "null"]},
        "morphology_confidence": {"type": ["number", "null"]},
        "repolarization_duration_prior_ms": {"type": ["number", "null"]},
        "p_wave_generated": {"const": False},
        "qt_measurement_supported": {"const": False},
        "pq_pr_measurement_supported": {"const": False},
        "latest_pulse_interval_ms": {"type": ["number", "null"]},
        "heart_rate_bpm": {"type": ["number", "null"]},
        "interval_cv": {"type": ["number", "null"]},
        "rmssd_ms": {"type": ["number", "null"]},
        "interval_count": {
            "type": "integer",
            "minimum": 0,
            "deprecated": True,
            "description": "interval_window_countの後方互換alias。総拍数ではない。",
        },
        "estimated_pat_ms": {"type": ["number", "null"]},
        "display_delay_ms": {"type": "number", "minimum": 0},
    }
    if doctor:
        required.extend(
            [
                "display_timestamp_start_seconds",
                "waveform_type",
                "diagnostic_ecg",
                "display_label",
                "measurement_ui_enabled",
            ]
        )
        properties.update(
            {
                "display_timestamp_start_seconds": {"type": "number"},
                "waveform_type": {"const": "ppg_derived_pseudo_ecg"},
                "diagnostic_ecg": {"const": False},
                "display_label": {"type": "string"},
                "measurement_ui_enabled": {"const": False},
            }
        )
    else:
        required.extend(
            [
                "sample_count",
                "pseudo_ecg_timestamp_start_seconds",
                "beat_events",
                "interval_window_count",
                "pulse_event_count_total",
                "rejected_interval_count_total",
            ]
        )
        properties.update(
            {
                "sample_count": {"type": "integer", "minimum": 1},
                "pseudo_ecg_timestamp_start_seconds": {"type": "number"},
                "beat_events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "event_sample_index",
                            "detected_at_input_timestamp_seconds",
                            "estimated_r_timestamp_seconds",
                            "event_type",
                            "source",
                        ],
                        "properties": {
                            "event_sample_index": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "detected_at_input_timestamp_seconds": {
                                "type": "number"
                            },
                            "estimated_r_timestamp_seconds": {"type": "number"},
                            "event_type": {
                                "enum": [
                                    "estimated_ecg_r_event",
                                    "ppg_pulse_peak",
                                ]
                            },
                            "source": {
                                "enum": ["timing_head", "pulse_tracker_fallback"]
                            },
                            "confidence": {"type": ["number", "null"]},
                            "amplitude": {"type": ["number", "null"]},
                        },
                    },
                },
                "interval_window_count": {"type": "integer", "minimum": 0},
                "pulse_event_count_total": {"type": "integer", "minimum": 0},
                "rejected_interval_count_total": {
                    "type": "integer",
                    "minimum": 0,
                },
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": payload_type,
        "type": "object",
        "required": required,
        "properties": properties,
    }


def build_package(
    *,
    data_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    recording_id: str,
    duration_seconds: float,
    chunk_seconds: float,
    overwrite: bool,
) -> dict[str, object]:
    if duration_seconds <= 0 or chunk_seconds <= 0:
        raise ValueError("duration and chunk seconds must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = checkpoint["report"]
    matches = [
        record
        for record in discover_mimic_perform_af(data_root)
        if record.recording_id == recording_id
    ]
    if len(matches) != 1:
        raise ValueError(f"recording not found or ambiguous: {recording_id}")
    record = matches[0]
    if record.patient_id not in report["test_patients"]:
        raise ValueError("handoff sample must use a held-out fold-0 test patient.")
    _, source_ppg, _ = load_mimic_perform_signals(record)
    sample_rate = record.sample_rate_hz
    sample_count = round(duration_seconds * sample_rate)
    source = source_ppg[:sample_count].astype(np.float64)
    if source.size < sample_count:
        raise ValueError("recording is shorter than requested sample duration.")
    scenarios = {
        "good": source.copy(),
        "missing": source.copy(),
        "low_quality": np.full_like(source, float(np.nanmedian(source))),
    }
    missing_start = round(5.0 * sample_rate)
    missing_end = min(round(7.0 * sample_rate), sample_count)
    scenarios["missing"][missing_start:missing_end] = np.nan
    chunk_samples = max(round(chunk_seconds * sample_rate), 1)
    scaler = FixedRobustScaler(**report["ppg_scaler"])
    summary: dict[str, object] = {}
    for scenario, values in scenarios.items():
        session = RealtimePPGSession(
            session_id=f"handoff-{scenario}",
            sample_rate_hz=sample_rate,
            ppg_scaler=scaler,
        )
        downstream_rows = []
        doctor_rows = []
        for start in range(0, values.size, chunk_samples):
            frame = session.process(
                values[start : start + chunk_samples],
                input_timestamp_start_seconds=1_000.0 + start / sample_rate,
            )
            downstream_rows.append(to_downstream_ai_frame(frame).to_dict())
            doctor_rows.append(to_doctor_visualization_frame(frame).to_dict())
        _write_jsonl(output_dir / f"downstream_{scenario}.jsonl", downstream_rows)
        _write_jsonl(output_dir / f"doctor_{scenario}.jsonl", doctor_rows)
        summary[scenario] = {
            "frames": len(downstream_rows),
            "accepted_downstream_frames": sum(
                bool(row["generation_accepted"]) for row in downstream_rows
            ),
            "accepted_doctor_frames": sum(
                bool(row["generation_accepted"]) for row in doctor_rows
            ),
        }
    schemas = {
        "downstream_ai.schema.json": _schema("front_ai_downstream_input", False),
        "doctor_visualization.schema.json": _schema(
            "doctor_waveform_visualization", True
        ),
    }
    for name, schema in schemas.items():
        (output_dir / name).write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "front_ai_handoff_sample",
        "source_dataset": "MIMIC PERform AF",
        "recording_alias": "held_out_fold0_sample",
        "patient_identifier_included": False,
        "source_file_path_included": False,
        "reference_rhythm_label": record.rhythm_label.value,
        "sample_rate_hz": sample_rate,
        "duration_seconds": duration_seconds,
        "chunk_seconds": chunk_seconds,
        "contains_af_prediction": False,
        "pseudo_ecg_is_diagnostic": False,
        "model_checkpoint_used_for_display_pseudo_ecg": False,
        "pseudo_ecg_source": "pulse_interval_qrs_t_display_template",
        "scenarios": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="前段AIの匿名化引渡しsampleを作る。")
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/MIMIC_PERform_Datasets")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/causal_cpu/causal_generator_fold0.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/handoff/sample_v1")
    )
    parser.add_argument("--recording-id", default="mimic_perform_non_af_007")
    parser.add_argument("--duration-seconds", type=float, default=15.0)
    parser.add_argument("--chunk-seconds", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_package(
                data_root=args.data_root,
                checkpoint_path=args.checkpoint,
                output_dir=args.output_dir,
                recording_id=args.recording_id,
                duration_seconds=args.duration_seconds,
                chunk_seconds=args.chunk_seconds,
                overwrite=args.overwrite,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
