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


def find_subject_files() -> list[tuple[int, Path]]:
    """BIDMC Signals.csvファイルを被験者IDとともに取得する。"""

    pattern = re.compile(r"bidmc_(\d+)_Signals\.csv")

    subject_files = []

    for path in DATA_ROOT.glob("*_Signals.csv"):
        match = pattern.fullmatch(path.name)

        if match is None:
            continue

        subject_id = int(match.group(1))

        subject_files.append(
            (subject_id, path)
        )

    return sorted(subject_files)


def inspect_subject(
    subject_id: int,
    path: Path,
) -> dict:
    """1被験者分のCSV構造を検査する。"""

    df = pd.read_csv(path)

    # CSV内の余分な空白を除去
    df.columns = df.columns.str.strip()

    result = {
        "subject_id": subject_id,
        "file": path.name,
        "n_samples": len(df),
        "n_columns": len(df.columns),
        "columns": ", ".join(df.columns),
        "has_pleth": "PLETH" in df.columns,
        "has_ecg_ii": "II" in df.columns,
    }

    if "PLETH" in df.columns:
        ppg = df["PLETH"].to_numpy()

        result.update(
            {
                "ppg_nan": int(np.isnan(ppg).sum()),
                "ppg_inf": int(np.isinf(ppg).sum()),
                "ppg_min": float(np.nanmin(ppg)),
                "ppg_max": float(np.nanmax(ppg)),
            }
        )

    if "II" in df.columns:
        ecg = df["II"].to_numpy()

        result.update(
            {
                "ecg_nan": int(np.isnan(ecg).sum()),
                "ecg_inf": int(np.isinf(ecg).sum()),
                "ecg_min": float(np.nanmin(ecg)),
                "ecg_max": float(np.nanmax(ecg)),
            }
        )

    return result


def main() -> None:
    print("=" * 60)
    print("BIDMC Dataset Inspection")
    print("=" * 60)

    print(f"\nData root:")
    print(DATA_ROOT)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"\nData directory not found:\n{DATA_ROOT}"
        )

    subject_files = find_subject_files()

    print(f"\nSubjects found: {len(subject_files)}")

    print(
        "Subject IDs:",
        [subject_id for subject_id, _ in subject_files],
    )

    results = []

    for subject_id, path in subject_files:
        result = inspect_subject(
            subject_id,
            path,
        )

        results.append(result)

        print(
            f"\nSubject {subject_id:02d}"
            f" | samples={result['n_samples']}"
            f" | PLETH={result['has_pleth']}"
            f" | ECG II={result['has_ecg_ii']}"
        )

    summary = pd.DataFrame(results)

    output_dir = PROJECT_ROOT / "outputs" / "data_inspection"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "bidmc_summary.csv"

    summary.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\nSubjects: {len(summary)}")

    print(
        "\nSample count statistics:"
    )

    print(
        summary["n_samples"].describe()
    )

    print(f"\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()