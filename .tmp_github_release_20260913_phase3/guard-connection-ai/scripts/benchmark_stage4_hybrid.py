from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from guard_connection_ai.deployment.morphology_checkpoint import (
    load_morphology_checkpoint,
)
from guard_connection_ai.deployment.timing_checkpoint import (
    load_validated_timing_checkpoint,
)
from guard_connection_ai.streaming.realtime_hybrid_session import (
    RealtimeHybridPPGSession,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 hybridのCPU chunk遅延を計測する。")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chunk-seconds", type=float, default=0.25)
    parser.add_argument(
        "--timing-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/timing_head_stage1_fold0_20260905/timing_head_fold0.pt"
        ),
    )
    parser.add_argument(
        "--morphology-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/cross_modal_morphology_stage3_release_gated_fold0_20260906"
            / "cross_modal_morphology_fold0.pt"
        ),
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.chunk_seconds <= 0:
        parser.error("iterations and chunk-seconds must be positive")
    timing = load_validated_timing_checkpoint(
        args.timing_checkpoint, sample_rate_hz=125
    )
    morphology = load_morphology_checkpoint(args.morphology_checkpoint)
    session = RealtimeHybridPPGSession(
        session_id="stage4-benchmark",
        sample_rate_hz=timing.sample_rate_hz,
        ppg_scaler=timing.ppg_scaler,
        timing_session=timing.new_session(),
        morphology_session=morphology.new_session(),
        morphology_source=morphology.morphology_source,
        quality_window_seconds=2.0,
    )
    count = round(timing.sample_rate_hz * args.chunk_seconds)
    elapsed_ms = []
    sample_index = 0
    for iteration in range(args.iterations + 10):
        indices = np.arange(sample_index, sample_index + count)
        ppg = 0.5 + 0.2 * np.sin(2 * np.pi * 1.15 * indices / timing.sample_rate_hz)
        started = time.perf_counter()
        session.process(
            ppg,
            input_timestamp_start_seconds=sample_index / timing.sample_rate_hz,
        )
        duration_ms = 1000 * (time.perf_counter() - started)
        if iteration >= 10:
            elapsed_ms.append(duration_ms)
        sample_index += count
    percentiles = np.percentile(elapsed_ms, [50, 95, 99])
    print(f"samples_per_chunk={count}")
    print(f"iterations={args.iterations}")
    print(f"p50_ms={percentiles[0]:.3f}")
    print(f"p95_ms={percentiles[1]:.3f}")
    print(f"p99_ms={percentiles[2]:.3f}")


if __name__ == "__main__":
    main()
