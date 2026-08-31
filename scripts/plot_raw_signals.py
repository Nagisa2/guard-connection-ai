from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
    / "bidmc_csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "data_inspection"
    / "raw_signals"
)

FS = 125

# 確認する被験者
SUBJECT_IDS = [1, 10, 25, 40, 53]

# 10秒
DURATION_SECONDS = 10

# 開始位置
START_SECONDS = 60


def load_subject(subject_id: int) -> pd.DataFrame:
    path = (
        DATA_ROOT
        / f"bidmc_{subject_id:02d}_Signals.csv"
    )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = START_SECONDS * FS
    end = start + DURATION_SECONDS * FS

    time = [
        i / FS
        for i in range(end - start)
    ]

    for subject_id in SUBJECT_IDS:
        df = load_subject(subject_id)

        ppg = df["PLETH"].iloc[start:end]
        ecg = df["II"].iloc[start:end]

        # PPG
        plt.figure(figsize=(12, 4))

        plt.plot(
            time,
            ppg,
        )

        plt.xlabel("Time [s]")
        plt.ylabel("PPG")

        plt.title(
            f"Subject {subject_id:02d} - Raw PPG"
        )

        plt.tight_layout()

        ppg_path = (
            OUTPUT_DIR
            / f"subject_{subject_id:02d}_ppg.png"
        )

        plt.savefig(
            ppg_path,
            dpi=150,
        )

        plt.close()

        # ECG
        plt.figure(figsize=(12, 4))

        plt.plot(
            time,
            ecg,
        )

        plt.xlabel("Time [s]")
        plt.ylabel("ECG")

        plt.title(
            f"Subject {subject_id:02d} - Raw ECG"
        )

        plt.tight_layout()

        ecg_path = (
            OUTPUT_DIR
            / f"subject_{subject_id:02d}_ecg.png"
        )

        plt.savefig(
            ecg_path,
            dpi=150,
        )

        plt.close()

        print(
            f"Saved plots for Subject {subject_id:02d}"
        )


if __name__ == "__main__":
    main()