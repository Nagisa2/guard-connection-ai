from __future__ import annotations

import argparse
import time

import numpy as np

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.streaming.realtime_session import RealtimePPGSession


def main() -> None:
    parser = argparse.ArgumentParser(description="リアルタイムPPG処理のCPU遅延を計測する。")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sample-rate", type=float, default=125.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.sample_rate <= 0 or args.seconds <= 0:
        parser.error("iterations, sample-rate, seconds must be positive")

    sample_count = round(args.sample_rate * args.seconds)
    elapsed_ms = []
    for iteration in range(args.iterations + 1):
        time_axis = np.arange(sample_count) / args.sample_rate
        ppg = np.sin(2 * np.pi * 1.2 * time_axis)
        session = RealtimePPGSession(
            session_id=f"benchmark-{iteration}",
            sample_rate_hz=args.sample_rate,
            ppg_scaler=FixedRobustScaler(center=0.0, scale=1.0),
        )
        started = time.perf_counter()
        session.process(ppg)
        duration = 1000 * (time.perf_counter() - started)
        if iteration:
            elapsed_ms.append(duration)

    percentiles = np.percentile(elapsed_ms, [50, 95, 99])
    print(f"samples_per_chunk={sample_count}")
    print(f"iterations={args.iterations}")
    print(f"p50_ms={percentiles[0]:.3f}")
    print(f"p95_ms={percentiles[1]:.3f}")
    print(f"p99_ms={percentiles[2]:.3f}")


if __name__ == "__main__":
    main()
