from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_waveform_reconstruction import (
    evaluate_oracle_waveform_metrics,
    summarize_waveform_metrics,
)
from train import build_datasets, load_model_config

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import find_subject_files
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "resunet.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "waveform_comparison.csv"


def main(max_batches: int | None = None) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(DATA_CONFIG_PATH)
    _, validation_dataset = build_datasets(data_config)
    stft_config = stft_config_from_data_config(data_config)
    subject_files = dict(find_subject_files(DATA_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("*_best.pt"))
    if not checkpoint_files:
        checkpoint_files = sorted(CHECKPOINT_DIR.glob("*.pt"))

    if not checkpoint_files:
        print(f"No checkpoint files found in {CHECKPOINT_DIR}")
        return

    comparison_rows = []

    for checkpoint_path in checkpoint_files:
        experiment_name = checkpoint_path.stem.replace("resunet_attention_", "").replace(
            "_best", ""
        )
        print(f"\n--- Evaluating Waveform Reconstruction: {checkpoint_path.name} ({experiment_name}) ---")

        model_settings = model_config["model"]
        model = ResidualAttentionUNet(
            in_channels=int(model_settings["in_channels"]),
            out_channels=int(model_settings["out_channels"]),
            base_channels=int(model_settings["base_channels"]),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        frame = evaluate_oracle_waveform_metrics(
            model,
            validation_dataset,
            device=device,
            batch_size=int(model_config["training"]["batch_size"]),
            preprocessing_mode=data_config["preprocessing"]["mode"],
            stft_config=stft_config,
            subject_files=subject_files,
            max_batches=max_batches,
        )

        summary = summarize_waveform_metrics(frame)
        row = {
            "experiment": experiment_name,
            "checkpoint": checkpoint_path.name,
            "best_epoch": checkpoint.get("epoch", "N/A"),
            "mae_mean": summary["mae_mean"],
            "mae_std": summary["mae_std"],
            "mae_median": summary["mae_median"],
            "mae_iqr": summary["mae_iqr"],
            "rmse_mean": summary["rmse_mean"],
            "rmse_std": summary["rmse_std"],
            "rmse_median": summary["rmse_median"],
            "rmse_iqr": summary["rmse_iqr"],
            "prd_mean": summary["prd_mean"],
            "prd_median": summary["prd_median"],
            "prd_iqr": summary["prd_iqr"],
            "correlation_mean": summary["correlation_mean"],
            "correlation_median": summary["correlation_median"],
            "correlation_iqr": summary["correlation_iqr"],
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=======================================================")
    print("      WAVEFORM RECONSTRUCTION COMPARISON SUMMARY       ")
    print("=======================================================")
    display_cols = [
        "experiment",
        "best_epoch",
        "mae_mean",
        "mae_median",
        "rmse_mean",
        "rmse_median",
        "prd_mean%",
        "correlation_mean",
        "correlation_median",
    ]
    formatted_df = comparison_df.copy()
    formatted_df["prd_mean%"] = formatted_df["prd_mean"].map("{:.2f}%".format)
    print(formatted_df[[c for c in display_cols if c in formatted_df.columns]].to_string(index=False))
    print(f"\nSaved comparison report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(max_batches=args.max_batches)
