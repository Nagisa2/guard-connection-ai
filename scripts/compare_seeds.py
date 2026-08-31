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
    aggregate_subject_waveform_metrics,
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
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "seed_comparison.csv"


def main(max_batches: int | None = None) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(DATA_CONFIG_PATH)
    stft_config = stft_config_from_data_config(data_config)
    subject_files = dict(find_subject_files(DATA_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("*_best.pt"))

    if not checkpoint_files:
        print(f"No best checkpoint files found in {CHECKPOINT_DIR}")
        return

    results = []

    for ckpt_path in checkpoint_files:
        ckpt_name = ckpt_path.name
        # Extract seed if present, otherwise default to 42
        if "_seed" in ckpt_name:
            seed_str = ckpt_name.split("_seed")[1].split("_")[0]
            seed = int(seed_str)
        else:
            seed = 42

        # Extract loss tag
        exp_tag = ckpt_name.replace("resunet_attention_", "").replace("_best.pt", "")
        if f"_seed{seed}" in exp_tag:
            exp_tag = exp_tag.replace(f"_seed{seed}", "")

        print(f"\n--- Evaluating Checkpoint: {ckpt_name} (Loss: {exp_tag}, Seed: {seed}) ---")

        # Build validation dataset matching the checkpoint's seed
        current_data_config = data_config.copy()
        current_data_config["seed"] = seed
        _, validation_dataset = build_datasets(current_data_config)

        model_settings = model_config["model"]
        model = ResidualAttentionUNet(
            in_channels=int(model_settings["in_channels"]),
            out_channels=int(model_settings["out_channels"]),
            base_channels=int(model_settings["base_channels"]),
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        frame = evaluate_oracle_waveform_metrics(
            model,
            validation_dataset,
            device=device,
            batch_size=int(model_config["training"]["batch_size"]),
            preprocessing_mode=current_data_config["preprocessing"]["mode"],
            stft_config=stft_config,
            subject_files=subject_files,
            max_batches=max_batches,
        )

        overall_summary = summarize_waveform_metrics(frame)
        subject_summary = aggregate_subject_waveform_metrics(frame)

        row = {
            "checkpoint": ckpt_name,
            "loss": exp_tag,
            "seed": seed,
            "best_epoch": checkpoint.get("epoch", "N/A"),
            "mae_mean": overall_summary["mae_mean"],
            "mae_median": overall_summary["mae_median"],
            "mae_iqr": overall_summary["mae_iqr"],
            "rmse_mean": overall_summary["rmse_mean"],
            "rmse_median": overall_summary["rmse_median"],
            "rmse_iqr": overall_summary["rmse_iqr"],
            "prd_mean": overall_summary["prd_mean"],
            "prd_median": overall_summary["prd_median"],
            "prd_iqr": overall_summary["prd_iqr"],
            "correlation_mean": overall_summary["correlation_mean"],
            "correlation_median": overall_summary["correlation_median"],
            "correlation_iqr": overall_summary["correlation_iqr"],
            "subject_correlation_mean": float(subject_summary["correlation_mean"].mean()),
            "subject_correlation_median": float(subject_summary["correlation_median"].median()),
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_PATH, index=False)

    print("\n==========================================================================")
    print("           SEED & SUBJECT STATISTICAL EVALUATION SUMMARY                  ")
    print("==========================================================================")
    cols = [
        "checkpoint",
        "loss",
        "seed",
        "mae_median",
        "rmse_median",
        "correlation_median",
        "subject_correlation_median",
    ]
    print(df_results[[c for c in cols if c in df_results.columns]].to_string(index=False))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(max_batches=args.max_batches)
