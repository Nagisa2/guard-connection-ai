from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from build_team_handoff_packages import _device_packet

from guard_connection_ai.deployment.morphology_checkpoint import (
    load_morphology_checkpoint,
)
from guard_connection_ai.deployment.timing_checkpoint import (
    load_validated_timing_checkpoint,
)
from guard_connection_ai.streaming.realtime_api import (
    RealtimeProtocolError,
    RealtimeSessionConfig,
    RealtimeSessionRegistry,
)
from guard_connection_ai.streaming.realtime_hybrid_session import (
    RealtimeHybridPPGSession,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="実機なしでStage 4の2 sessionを検証する。")
    parser.add_argument("--packets", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/validation/stage4_demo_synthetic.json",
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
    if args.packets < 36:
        parser.error("packets must be at least 36 to exercise low-quality recovery")
    timing = load_validated_timing_checkpoint(
        args.timing_checkpoint, sample_rate_hz=125
    )
    morphology = load_morphology_checkpoint(args.morphology_checkpoint)

    def factory(config, session_id):
        return RealtimeHybridPPGSession(
            session_id=session_id,
            sample_rate_hz=config.sample_rate_hz,
            ppg_scaler=timing.ppg_scaler,
            timing_session=timing.new_session(),
            morphology_session=morphology.new_session(),
            morphology_source=morphology.morphology_source,
            quality_window_seconds=4.0,
        )

    registry = RealtimeSessionRegistry(
        pipeline_factories={"timing_constrained_morphology_v1": factory}
    )
    session_ids = ("synthetic-user-01", "synthetic-user-02")
    config = {
        session_id: RealtimeSessionConfig(
            user_id=session_id,
            session_id=session_id,
            sample_rate_hz=125,
            scaler_center=timing.ppg_scaler.center,
            scaler_scale=timing.ppg_scaler.scale,
            generation_profile_id="timing_constrained_morphology_v1",
            normalize_device_adc=True,
            quality_window_seconds=4.0,
        )
        for session_id in session_ids
    }
    for value in config.values():
        registry.create(value)

    modes = {session_id: [] for session_id in session_ids}
    accepted = {session_id: 0 for session_id in session_ids}
    cycle_ms = []
    sequence_errors = []

    def process(session_id: str, sequence: int):
        packet = replace(
            _device_packet(sequence),
            stream_id=f"stream-{session_id}",
        )
        if session_id == session_ids[1] and 20 <= sequence < 32:
            missing = [None] * packet.sample_count
            packet = replace(
                packet,
                ppg0=missing,
                ppg1=missing,
                ppg2=missing,
                sensor_disconnected=True,
            )
        return registry.process_device_packet(session_id, packet)

    with ThreadPoolExecutor(max_workers=2) as executor:
        for sequence in range(args.packets):
            if sequence == 3:
                try:
                    process(session_ids[0], sequence + 1)
                except RealtimeProtocolError as error:
                    sequence_errors.append(error.code)
            started = time.perf_counter()
            futures = [
                executor.submit(process, session_id, sequence)
                for session_id in session_ids
            ]
            results = [future.result() for future in futures]
            cycle_ms.append(1000 * (time.perf_counter() - started))
            for session_id, result in zip(session_ids, results, strict=True):
                frame = result.waveform_frame
                if frame is None:
                    continue
                modes[session_id].append(frame.generation_mode)
                accepted[session_id] += int(frame.visualization_accepted)
            if sequence == 6:
                try:
                    process(session_ids[0], sequence)
                except RealtimeProtocolError as error:
                    sequence_errors.append(error.code)

    registry.end(session_ids[0])
    registry.create(config[session_ids[0]])
    reset = process(session_ids[0], 0).waveform_frame
    if reset is None:
        raise RuntimeError("reconnected session did not emit a waveform frame")
    percentiles = np.percentile(cycle_ms, [50, 95, 99])
    report = {
        "schema_version": "1.0",
        "test_kind": "synthetic_two_session_stage4",
        "real_device_data_used": False,
        "sessions": len(session_ids),
        "packets_per_session": args.packets,
        "sequence_errors_detected": sequence_errors,
        "display_accepted_frames": accepted,
        "generation_mode_counts": {
            session_id: {
                mode: values.count(mode) for mode in sorted(set(values))
            }
            for session_id, values in modes.items()
        },
        "missing_session_has_blank": "blank" in modes[session_ids[1]][20:32],
        "reconnect_sequence_reset": reset.sequence_number == 0,
        "reconnect_interval_state_reset": reset.interval_count == 0,
        "cycle_latency_ms": {
            "p50": float(percentiles[0]),
            "p95": float(percentiles[1]),
            "p99": float(percentiles[2]),
        },
        "pseudo_ecg_diagnostic": False,
        "p_wave_generated": False,
        "limitations": [
            "in-process benchmark; HTTP, WebSocket, and React rendering are excluded",
            "synthetic ADC packets are not evidence of real-device performance",
        ],
    }
    if sequence_errors != ["sequence_gap", "duplicate_packet"]:
        raise RuntimeError(f"unexpected sequence error results: {sequence_errors}")
    if not report["missing_session_has_blank"]:
        raise RuntimeError("missing device PPG did not stop pseudo ECG display")
    if not report["reconnect_sequence_reset"] or not report[
        "reconnect_interval_state_reset"
    ]:
        raise RuntimeError("session state was not reset on reconnect")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
