from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "stft"
OUTPUT_PATH = INPUT_DIR / "stft_magnitude_comparison.png"
MODES = ("none", "segment_zscore")


def main() -> None:
    archives = {
        mode: np.load(INPUT_DIR / f"stft_examples_{mode}.npz") for mode in MODES
    }
    reference = archives[MODES[0]]
    if not np.array_equal(reference["subject_id"], archives[MODES[1]]["subject_id"]):
        raise ValueError("Compared archives contain different subject ordering.")
    if not np.array_equal(reference["segment_id"], archives[MODES[1]]["segment_id"]):
        raise ValueError("Compared archives contain different segment ordering.")

    frequencies = reference["frequencies"]
    times = reference["times"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    titles = {
        ("none", "ppg_magnitude"): "PPG | none",
        ("none", "ecg_magnitude"): "ECG | none",
        ("segment_zscore", "ppg_magnitude"): "PPG | segment z-score",
        ("segment_zscore", "ecg_magnitude"): "ECG | segment z-score",
    }
    for row, mode in enumerate(MODES):
        for column, signal_name in enumerate(("ppg_magnitude", "ecg_magnitude")):
            axis = axes[row, column]
            image = np.log1p(archives[mode][signal_name][0])
            mesh = axis.pcolormesh(times, frequencies, image, shading="auto", cmap="magma")
            axis.set_title(titles[(mode, signal_name)])
            axis.set_ylabel("Frequency [Hz]")
            axis.set_xlabel("Time [s]")
            figure.colorbar(mesh, ax=axis, label="log1p magnitude")

    subject_id = str(reference["subject_id"][0])
    segment_id = int(reference["segment_id"][0])
    figure.suptitle(f"STFT magnitude comparison | {subject_id}, segment {segment_id}")
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
