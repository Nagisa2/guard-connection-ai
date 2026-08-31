from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import build_datasets, load_model_config

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import find_subject_files, load_subject_signals
from guard_connection_ai.data.preprocessing import preprocess_signal
from guard_connection_ai.data.stft import (
    compute_stft,
    griffin_lim_reconstruction,
    phase_transfer_reconstruction,
    waveform_reconstruction_metrics,
    zero_phase_reconstruction,
)
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "resunet.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "resunet_attention_l1_ssim_seed42_best.pt"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "evaluation" / "phase_reconstruction_comparison.csv"
)


def evaluate_phase_methods(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    preprocessing_mode: str,
    stft_config,
    subject_files: dict[str, Path],
    max_batches: int | None = None,
    gl_iters: int = 16,
) -> dict[str, pd.DataFrame]:
    methods = [
        "oracle_phase",
        "ppg_phase",
        "griffin_lim_ppg_init",
        "griffin_lim_zero_init",
        "zero_phase",
    ]
    method_rows: dict[str, list[dict]] = {m: [] for m in methods}
    signal_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            predictions = model(batch["input"].to(device)).detach().cpu().numpy()

            for item_index, subject_id in enumerate(batch["subject_id"]):
                sid_str = str(subject_id)
                if sid_str not in signal_cache:
                    sig_path = subject_files[sid_str]
                    ppg_raw, ecg_raw = load_subject_signals(sig_path)
                    signal_cache[sid_str] = (ppg_raw, ecg_raw)
                ppg_raw, ecg_raw = signal_cache[sid_str]

                start_sample = int(batch["start_sample"][item_index])
                end_sample = int(batch["end_sample"][item_index])
                segment_id = int(batch["segment_id"][item_index])

                ppg_segment = preprocess_signal(
                    ppg_raw[start_sample:end_sample], preprocessing_mode
                )
                ecg_segment = preprocess_signal(
                    ecg_raw[start_sample:end_sample], preprocessing_mode
                )

                ppg_stft = compute_stft(ppg_segment, stft_config)
                ecg_stft = compute_stft(ecg_segment, stft_config)
                pred_mag = predictions[item_index, 0]

                # 1. Oracle Phase
                rec_oracle = phase_transfer_reconstruction(
                    pred_mag, ecg_stft.phase, stft_config
                )
                m_oracle = waveform_reconstruction_metrics(ecg_segment, rec_oracle)

                # 2. PPG Phase Transfer
                rec_ppg = phase_transfer_reconstruction(
                    pred_mag, ppg_stft.phase, stft_config
                )
                m_ppg = waveform_reconstruction_metrics(ecg_segment, rec_ppg)

                # 3. Griffin-Lim with PPG phase init
                rec_gl_ppg = griffin_lim_reconstruction(
                    pred_mag, stft_config, n_iter=gl_iters, init_phase=ppg_stft.phase
                )
                m_gl_ppg = waveform_reconstruction_metrics(ecg_segment, rec_gl_ppg)

                # 4. Griffin-Lim with Zero phase init
                rec_gl_zero = griffin_lim_reconstruction(
                    pred_mag, stft_config, n_iter=gl_iters, init_phase=None
                )
                m_gl_zero = waveform_reconstruction_metrics(ecg_segment, rec_gl_zero)

                # 5. Zero Phase
                rec_zero = zero_phase_reconstruction(pred_mag, stft_config)
                m_zero = waveform_reconstruction_metrics(ecg_segment, rec_zero)

                metrics_map = {
                    "oracle_phase": m_oracle,
                    "ppg_phase": m_ppg,
                    "griffin_lim_ppg_init": m_gl_ppg,
                    "griffin_lim_zero_init": m_gl_zero,
                    "zero_phase": m_zero,
                }

                for m_name, m_dict in metrics_map.items():
                    method_rows[m_name].append(
                        {
                            "subject_id": sid_str,
                            "segment_id": segment_id,
                            "start_sample": start_sample,
                            "end_sample": end_sample,
                            **m_dict,
                        }
                    )

    return {m: pd.DataFrame(rows) for m, rows in method_rows.items()}


def summarize_method_metrics(df: pd.DataFrame) -> dict[str, float]:
    metrics = ["mae", "rmse", "prd", "correlation"]
    res = {}
    for m in metrics:
        res[f"{m}_mean"] = float(df[m].mean())
        res[f"{m}_std"] = float(df[m].std())
        res[f"{m}_median"] = float(df[m].median())
        q25 = float(df[m].quantile(0.25))
        q75 = float(df[m].quantile(0.75))
        res[f"{m}_iqr"] = float(q75 - q25)
    return res


def main(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    max_batches: int | None = None,
    gl_iters: int = 16,
) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(DATA_CONFIG_PATH)
    subject_files = dict(find_subject_files(DATA_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine seed from checkpoint
    ckpt_name = checkpoint_path.name
    seed = 42
    if "_seed" in ckpt_name:
        seed = int(ckpt_name.split("_seed")[1].split("_")[0])
    data_config["seed"] = seed
    stft_config = stft_config_from_data_config(data_config)

    _, validation_dataset = build_datasets(data_config, cache_stft=True)

    model_settings = model_config["model"]
    model = ResidualAttentionUNet(
        in_channels=int(model_settings["in_channels"]),
        out_channels=int(model_settings["out_channels"]),
        base_channels=int(model_settings["base_channels"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Evaluating Phase Reconstruction Methods on: {checkpoint_path.name}")
    print(f"Validation Segments: {len(validation_dataset)}")

    method_dfs = evaluate_phase_methods(
        model,
        validation_dataset,
        device=device,
        batch_size=int(model_config["training"]["batch_size"]),
        preprocessing_mode=data_config["preprocessing"]["mode"],
        stft_config=stft_config,
        subject_files=subject_files,
        max_batches=max_batches,
        gl_iters=gl_iters,
    )

    summary_rows = []
    for method_name, df in method_dfs.items():
        summary = summarize_method_metrics(df)
        subject_medians = df.groupby("subject_id")["correlation"].median()
        row = {
            "checkpoint": checkpoint_path.name,
            "method": method_name,
            "mae_mean": summary["mae_mean"],
            "mae_std": summary["mae_std"],
            "mae_median": summary["mae_median"],
            "mae_iqr": summary["mae_iqr"],
            "rmse_mean": summary["rmse_mean"],
            "rmse_median": summary["rmse_median"],
            "prd_mean": summary["prd_mean"],
            "prd_median": summary["prd_median"],
            "correlation_mean": summary["correlation_mean"],
            "correlation_median": summary["correlation_median"],
            "correlation_iqr": summary["correlation_iqr"],
            "subject_correlation_median": float(subject_medians.median()),
        }
        summary_rows.append(row)

    comparison_df = pd.DataFrame(summary_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("        PHASE RECONSTRUCTION METHOD COMPARISON SUMMARY        ")
    print("=" * 80)
    display_cols = [
        "method",
        "mae_median",
        "rmse_median",
        "prd_median",
        "correlation_median",
        "subject_correlation_median",
    ]
    print(comparison_df[display_cols].to_string(index=False))
    print(f"\nSaved report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--gl-iters", type=int, default=16)
    args = parser.parse_args()
    main(args.checkpoint, max_batches=args.max_batches, gl_iters=args.gl_iters)
