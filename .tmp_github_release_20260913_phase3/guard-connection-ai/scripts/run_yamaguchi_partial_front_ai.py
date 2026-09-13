"""山口大学JSONの実配列があるPPGだけで前段AIの接続確認を行う。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from guard_connection_ai.data.json_metadata import find_json_metadata
from guard_connection_ai.data.yamaguchi_json import (
    YamaguchiSignalDataUnavailable,
    audit_yamaguchi_export,
    load_yamaguchi_ppg_recording,
)
from guard_connection_ai.deployment.morphology_checkpoint import (
    load_morphology_checkpoint,
)
from guard_connection_ai.deployment.timing_checkpoint import (
    load_validated_timing_checkpoint,
)
from guard_connection_ai.schemas.device_ppg import DevicePPGPacket
from guard_connection_ai.streaming.device_ppg_ingress import (
    StatefulCausalADCNormalizer,
    StatefulDevicePPGIngress,
)
from guard_connection_ai.streaming.realtime_hybrid_session import (
    RealtimeHybridPPGSession,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_SAMPLE_RATE_HZ = 125.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packet(
    values: np.ndarray,
    ambient: np.ndarray | None,
    *,
    sequence_number: int,
    sample_offset: int,
    sample_rate_hz: float,
    stream_id: str,
) -> DevicePPGPacket:
    green = np.rint(values).astype(np.int64).tolist()
    ambient_values = (
        np.rint(ambient).astype(np.int64).tolist()
        if ambient is not None
        else [0] * len(green)
    )
    end_sample = sample_offset + len(green) - 1
    return DevicePPGPacket(
        schema_version="1.0",
        stream_id=stream_id,
        sequence_number=sequence_number,
        configured_sample_rate_hz=sample_rate_hz,
        estimated_sample_rate_hz=sample_rate_hz,
        device_timestamp_end_ns=round(end_sample / sample_rate_hz * 1_000_000_000),
        # このexportは単一green channelのため、3ch融合器へ同値を渡して
        # 信号を変えずに既存の欠損検証・因果resampling経路を再利用する。
        ppg0=green,
        ppg1=green,
        ppg2=green,
        ambient0=ambient_values,
        clock_domain="dataset_relative_time",
        device_model="Yamaguchi dataset device (unspecified)",
    )


def _run_recording(
    path: Path,
    *,
    alias: str,
    max_seconds: float,
    timing: object,
    morphology: object,
) -> dict[str, object]:
    recording = load_yamaguchi_ppg_recording(path)
    maximum_samples = min(
        recording.ppg_green.size,
        max(1, round(max_seconds * recording.sample_rate_hz)),
    )
    ppg = recording.ppg_green[:maximum_samples]
    ambient = (
        recording.ppg_ambient[:maximum_samples]
        if recording.ppg_ambient is not None
        else None
    )
    ingress = StatefulDevicePPGIngress(target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    normalizer = StatefulCausalADCNormalizer(sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    session = RealtimeHybridPPGSession(
        session_id=alias,
        sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        ppg_scaler=timing.ppg_scaler,
        timing_session=timing.new_session(),
        morphology_session=morphology.new_session(),
        morphology_source=morphology.morphology_source,
        quality_window_seconds=4.0,
    )
    source_chunk_samples = max(1, round(recording.sample_rate_hz))
    modes: Counter[str] = Counter()
    frame_samples = 0
    accepted_samples = 0
    technical_sqi = []
    interval_plausibility = []
    last_frame = None
    for sequence_number, start in enumerate(range(0, maximum_samples, source_chunk_samples)):
        end = min(start + source_chunk_samples, maximum_samples)
        adapted = ingress.process(
            _packet(
                ppg[start:end],
                ambient[start:end] if ambient is not None else None,
                sequence_number=sequence_number,
                sample_offset=start,
                sample_rate_hz=recording.sample_rate_hz,
                stream_id=alias,
            )
        )
        if adapted.ppg.size == 0:
            continue
        normalized = normalizer.process(adapted.ppg)
        frame = session.process(
            normalized,
            input_timestamp_start_seconds=adapted.input_timestamp_start_seconds,
        )
        last_frame = frame
        sample_count = len(frame.ppg)
        frame_samples += sample_count
        accepted_samples += sample_count * int(frame.visualization_accepted)
        modes[frame.generation_mode] += 1
        technical_sqi.append(frame.technical_sqi)
        interval_plausibility.append(frame.pulse_interval_plausibility)
    if last_frame is None:
        raise RuntimeError(f"no output frame was produced for {alias}.")
    return {
        "recording_alias": alias,
        "source_sha256": _sha256(path),
        "source_sample_rate_hz": recording.sample_rate_hz,
        "processed_source_samples": maximum_samples,
        "processed_duration_seconds": maximum_samples / recording.sample_rate_hz,
        "output_samples": frame_samples,
        "output_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "beat_event_count": last_frame.pulse_event_count_total,
        "visualization_accepted_sample_fraction": (
            accepted_samples / frame_samples if frame_samples else 0.0
        ),
        "mean_technical_sqi": float(np.mean(technical_sqi)),
        "mean_pulse_interval_plausibility": float(np.mean(interval_plausibility)),
        "generation_mode_frame_counts": dict(modes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "山口大学JSON内で実配列が存在するPPGを前段AIへ通す。"
            "精度評価やAF評価は行わない。"
        )
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/json_Data")
    parser.add_argument("--max-seconds-per-recording", type=float, default=60.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/validation/yamaguchi_partial_front_ai_smoke.json",
    )
    parser.add_argument(
        "--timing-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/timing_head_stage1_fold0_20260905/timing_head_fold0.pt",
    )
    parser.add_argument(
        "--morphology-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/cross_modal_morphology_stage3_release_gated_fold0_20260906"
        / "cross_modal_morphology_fold0.pt",
    )
    args = parser.parse_args()
    if args.max_seconds_per_recording <= 0:
        parser.error("--max-seconds-per-recording must be positive.")

    audit = audit_yamaguchi_export(args.data_root)
    timing = load_validated_timing_checkpoint(
        args.timing_checkpoint, sample_rate_hz=TARGET_SAMPLE_RATE_HZ
    )
    morphology = load_morphology_checkpoint(args.morphology_checkpoint)
    available_paths = []
    for metadata in find_json_metadata(args.data_root):
        if metadata.signal_type != "PPG":
            continue
        try:
            load_yamaguchi_ppg_recording(metadata.path)
        except YamaguchiSignalDataUnavailable:
            continue
        available_paths.append(metadata.path)

    recordings = [
        _run_recording(
            path,
            alias=f"yamaguchi_partial_{index:02d}",
            max_seconds=args.max_seconds_per_recording,
            timing=timing,
            morphology=morphology,
        )
        for index, path in enumerate(available_paths, start=1)
    ]
    report = {
        "schema_version": "1.0",
        "evaluation_kind": "unlabeled_partial_front_ai_smoke_test",
        "accuracy_claim": False,
        "af_performance_evaluated": False,
        "r_event_accuracy_evaluated": False,
        "data_audit": {
            "subjects_in_export": audit.subject_count,
            "ppg_files": audit.ppg_metadata_files,
            "ppg_files_with_signal_arrays": audit.ppg_signal_array_files,
            "ecg_files_with_signal_arrays": audit.ecg_signal_array_files,
            "qrs_files_with_arrays": audit.qrs_index_array_files,
            "af_annotation_files_with_arrays": audit.af_annotation_array_files,
        },
        "model_provenance": {
            "timing_checkpoint_sha256": _sha256(args.timing_checkpoint),
            "morphology_checkpoint_sha256": _sha256(args.morphology_checkpoint),
            "runtime_ecg_input": False,
            "diagnostic_ecg": False,
        },
        "recordings": recordings,
        "blockers": list(audit.blockers),
        "limitations": [
            "Only PPG files containing numeric arrays were processed.",
            "The single available green channel was triplicated only to reuse the 3-channel ingress path.",
            "No ECG, QRS, or AF annotation arrays were available for accuracy evaluation.",
            "This result is a data/loading/inference compatibility check, not external validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
