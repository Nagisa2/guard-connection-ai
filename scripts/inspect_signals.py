from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
    / "bidmc_csv"
)

FS = 125
EXPECTED_SAMPLES = 60_000


def find_subject_files() -> list[tuple[int, Path]]:
    pattern = re.compile(r"bidmc_(\d+)_Signals\.csv")

    subject_files = []

    for path in DATA_ROOT.glob("*_Signals.csv"):
        match = pattern.fullmatch(path.name)

        if match is not None:
            subject_id = int(match.group(1))
            subject_files.append((subject_id, path))

    return sorted(subject_files)


def inspect_signal(
    signal: np.ndarray,
    prefix: str,
) -> dict:
    return {
        f"{prefix}_nan": int(np.isnan(signal).sum()),
        f"{prefix}_inf": int(np.isinf(signal).sum()),
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_min": float(np.min(signal)),
        f"{prefix}_max": float(np.max(signal)),
    }


def main() -> None:
    print("=" * 60)
    print("BIDMC Signal Quality Inspection")
    print("=" * 60)

    subject_files = find_subject_files()

    results = []

    for subject_id, path in subject_files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        ppg = df["PLETH"].to_numpy(dtype=np.float64)
        ecg = df["II"].to_numpy(dtype=np.float64)

        result = {
            "subject_id": subject_id,
            "original_samples": len(df),
            "used_samples": EXPECTED_SAMPLES,
            "ppg_length": len(ppg),
            "ecg_length": len(ecg),
        }

        result.update(
            inspect_signal(
                ppg,
                "ppg",
            )
        )

        result.update(
            inspect_signal(
                ecg,
                "ecg",
            )
        )

        results.append(result)

        print(
            f"Subject {subject_id:02d} | "
            f"PPG NaN={result['ppg_nan']} | "
            f"ECG NaN={result['ecg_nan']} | "
            f"PPG range=[{result['ppg_min']:.3f}, "
            f"{result['ppg_max']:.3f}] | "
            f"ECG range=[{result['ecg_min']:.3f}, "
            f"{result['ecg_max']:.3f}]"
        )

    summary = pd.DataFrame(results)

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "data_inspection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "signal_quality_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Overall Summary")
    print("=" * 60)

    print(
        "\nPPG NaN total:",
        summary["ppg_nan"].sum(),
    )

    print(
        "ECG NaN total:",
        summary["ecg_nan"].sum(),
    )

    print(
        "PPG Inf total:",
        summary["ppg_inf"].sum(),
    )

    print(
        "ECG Inf total:",
        summary["ecg_inf"].sum(),
    )

    print("\nPPG standard deviation:")

    print(
        summary["ppg_std"].describe()
    )

    print("\nECG standard deviation:")

    print(
        summary["ecg_std"].describe()
    )

    print(
        f"\nSaved:\n{output_path}"
    )


if __name__ == "__main__":
    main()