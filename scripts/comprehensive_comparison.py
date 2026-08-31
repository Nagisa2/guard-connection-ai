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
from train import build_datasets, evaluate_epoch, load_model_config

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
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "comprehensive_model_comparison.csv"


def parse_checkpoint_info(filename: str) -> tuple[str, int]:
    """Extract loss type and seed from checkpoint filename."""
    stem = filename.replace("resunet_attention_", "").replace("_best.pt", "")
    seed = 42
    if "_seed" in stem:
        parts = stem.split("_seed")
        loss_type = parts[0]
        seed = int(parts[1].split("_")[0])
    else:
        loss_type = stem
    return loss_type, seed


def evaluate_checkpoint(
    ckpt_path: Path,
    data_config: dict,
    model_config: dict,
    subject_files: dict[str, Path],
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    loss_type, seed = parse_checkpoint_info(ckpt_path.name)
    current_data_config = data_config.copy()
    current_data_config["seed"] = seed
    stft_config = stft_config_from_data_config(current_data_config)

    _, validation_dataset = build_datasets(current_data_config, cache_stft=True)
    batch_size = int(model_config["training"]["batch_size"])
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_settings = model_config["model"]
    model = ResidualAttentionUNet(
        in_channels=int(model_settings["in_channels"]),
        out_channels=int(model_settings["out_channels"]),
        base_channels=int(model_settings["base_channels"]),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Determine loss weights for STFT evaluation
    ssim_weight = 0.1 if "ssim" in loss_type else 0.0
    frequency_weight = 0.001 if "frequency" in loss_type else 0.0
    l1_weight = 1.0

    val_loss, stft_metrics = evaluate_epoch(
        model,
        val_loader,
        device=device,
        max_batches=max_batches,
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        frequency_weight=frequency_weight,
    )

    waveform_frame = evaluate_oracle_waveform_metrics(
        model,
        validation_dataset,
        device=device,
        batch_size=batch_size,
        preprocessing_mode=current_data_config["preprocessing"]["mode"],
        stft_config=stft_config,
        subject_files=subject_files,
        max_batches=max_batches,
    )
    waveform_summary = summarize_waveform_metrics(waveform_frame)
    subject_summary = aggregate_subject_waveform_metrics(waveform_frame)

    return {
        "checkpoint": ckpt_path.name,
        "loss_type": loss_type,
        "seed": seed,
        "best_epoch": checkpoint.get("epoch", "N/A"),
        # STFT Spectrogram Metrics
        "stft_val_loss": val_loss,
        "stft_mae": stft_metrics["mae"],
        "stft_rmse": stft_metrics["rmse"],
        "stft_prd": stft_metrics["prd"],
        "stft_correlation": stft_metrics["correlation"],
        # Waveform Metrics (Overall)
        "waveform_mae_mean": waveform_summary["mae_mean"],
        "waveform_mae_median": waveform_summary["mae_median"],
        "waveform_mae_iqr": waveform_summary["mae_iqr"],
        "waveform_rmse_mean": waveform_summary["rmse_mean"],
        "waveform_rmse_median": waveform_summary["rmse_median"],
        "waveform_rmse_iqr": waveform_summary["rmse_iqr"],
        "waveform_prd_mean": waveform_summary["prd_mean"],
        "waveform_prd_median": waveform_summary["prd_median"],
        "waveform_correlation_mean": waveform_summary["correlation_mean"],
        "waveform_correlation_median": waveform_summary["correlation_median"],
        "waveform_correlation_iqr": waveform_summary["correlation_iqr"],
        # Waveform Metrics (Subject-level)
        "subject_waveform_corr_mean": float(subject_summary["correlation_mean"].mean()),
        "subject_waveform_corr_median": float(subject_summary["correlation_median"].median()),
    }


def main(max_batches: int | None = None) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(DATA_CONFIG_PATH)
    subject_files = dict(find_subject_files(DATA_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("resunet_attention_*_best.pt"))
    if not checkpoint_files:
        print(f"No *_best.pt checkpoints found in {CHECKPOINT_DIR}")
        return

    rows = []
    for ckpt_path in checkpoint_files:
        print(f"\n--- Evaluating: {ckpt_path.name} ---")
        row = evaluate_checkpoint(
            ckpt_path,
            data_config,
            model_config,
            subject_files,
            device,
            max_batches=max_batches,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("           COMPREHENSIVE MODEL COMPARISON SUMMARY           ")
    print("=" * 80)
    display_cols = [
        "loss_type",
        "seed",
        "best_epoch",
        "stft_mae",
        "stft_correlation",
        "waveform_mae_median",
        "waveform_correlation_median",
        "subject_waveform_corr_median",
    ]
    print(df[display_cols].to_string(index=False))
    print(f"\nSaved comprehensive comparison report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(max_batches=args.max_batches)
