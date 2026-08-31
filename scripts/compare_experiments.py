from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_PATH = EVALUATION_DIR / "experiment_comparison.csv"


def main() -> None:
    experiments = {
        "l1": "L1",
        "l1_ssim": "L1 + SSIM",
        "l1_ssim_frequency": "L1 + SSIM + Frequency",
    }
    rows = []
    for file_stem, experiment_name in experiments.items():
        history = pd.read_csv(TRAINING_DIR / f"{file_stem}_history.csv")
        best_row = history.loc[history["val_loss"].idxmin()]
        subject_metrics = pd.read_csv(
            EVALUATION_DIR / f"{file_stem}_best_subject_metrics.csv"
        )
        rows.append(
            {
                "experiment": experiment_name,
                "best_epoch": int(best_row["epoch"]),
                "val_loss": float(best_row["val_loss"]),
                "val_mae": float(best_row["val_mae"]),
                "val_rmse": float(best_row["val_rmse"]),
                "val_prd": float(best_row["val_prd"]),
                "val_correlation": float(best_row["val_correlation"]),
                "subject_mean_mae": float(subject_metrics["mae"].mean()),
                "subject_mean_rmse": float(subject_metrics["rmse"].mean()),
                "subject_mean_prd": float(subject_metrics["prd"].mean()),
                "subject_mean_correlation": float(subject_metrics["correlation"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(frame.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
