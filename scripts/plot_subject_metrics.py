from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_PATH = EVALUATION_DIR / "subject_metrics_comparison.png"


def main() -> None:
    l1 = pd.read_csv(EVALUATION_DIR / "l1_subject_metrics.csv").set_index("subject_id")
    l1_ssim = pd.read_csv(EVALUATION_DIR / "l1_ssim_subject_metrics.csv").set_index("subject_id")
    if not l1.index.equals(l1_ssim.index):
        raise ValueError("L1 and L1+SSIM subject IDs do not match.")

    metrics = ("mae", "rmse", "prd", "correlation")
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, metric in zip(axes.flat, metrics):
        positions = range(len(l1.index))
        axis.bar(
            [position - 0.2 for position in positions],
            l1[metric],
            width=0.4,
            color="#1f77b4",
            label="L1",
        )
        axis.bar(
            [position + 0.2 for position in positions],
            l1_ssim[metric],
            width=0.4,
            color="#d95f02",
            label="L1 + SSIM",
        )
        axis.set_title(metric.upper())
        axis.set_xlabel("Validation subject")
        axis.set_xticks(list(positions), l1.index, rotation=90)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    figure.suptitle("Validation subject metrics: L1 vs L1 + SSIM")
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
