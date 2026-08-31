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
    oracle_phase_reconstruction_metrics,
)
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "resunet.yaml"
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"


def summarize_waveform_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Calculate overall mean, std, median, and IQR for waveform metrics."""
    metrics = ["mae", "rmse", "prd", "correlation"]
    summary = {}
    for m in metrics:
        summary[f"{m}_mean"] = float(df[m].mean())
        summary[f"{m}_std"] = float(df[m].std())
        summary[f"{m}_median"] = float(df[m].median())
        q25 = float(df[m].quantile(0.25))
        q75 = float(df[m].quantile(0.75))
        summary[f"{m}_iqr"] = float(q75 - q25)
    return summary


def aggregate_subject_waveform_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Group metrics by subject_id and compute mean, median, and IQR."""
    metrics = ["mae", "rmse", "prd", "correlation"]
    rows = []
    for subject_id, group in df.groupby("subject_id"):
        row: dict[str, float | str] = {"subject_id": str(subject_id)}
        for m in metrics:
            row[f"{m}_mean"] = float(group[m].mean())
            row[f"{m}_median"] = float(group[m].median())
            q25 = float(group[m].quantile(0.25))
            q75 = float(group[m].quantile(0.75))
            row[f"{m}_iqr"] = float(q75 - q25)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_oracle_waveform_metrics(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    preprocessing_mode: str,
    stft_config,
    subject_files: dict[str, Path],
    max_batches: int | None = None,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows: list[dict[str, float | str | int]] = []
    signal_cache: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            prediction = model(batch["input"].to(device)).detach().cpu()
            for item_index, subject_id in enumerate(batch["subject_id"]):
                sid_str = str(subject_id)
                if sid_str not in signal_cache:
                    signal_path = subject_files[sid_str]
                    _, ecg_full = load_subject_signals(signal_path)
                    signal_cache[sid_str] = ecg_full
                ecg = signal_cache[sid_str]
                start_sample = int(batch["start_sample"][item_index])
                end_sample = int(batch["end_sample"][item_index])
                segment_id = int(batch["segment_id"][item_index])
                ecg_segment = preprocess_signal(
                    ecg[start_sample:end_sample],
                    preprocessing_mode,
                )
                target_stft = compute_stft(ecg_segment, stft_config)
                metrics = oracle_phase_reconstruction_metrics(
                    ecg_segment,
                    prediction[item_index, 0].numpy(),
                    target_stft.phase,
                    stft_config,
                )
                rows.append(
                    {
                        "subject_id": sid_str,
                        "segment_id": segment_id,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        **metrics,
                    }
                )
    if not rows:
        raise ValueError("No waveform metrics were produced.")
    return pd.DataFrame(rows)


def main(
    checkpoint_path: Path,
    output_path: Path,
    *,
    max_batches: int | None = None,
    subject_summary_path: Path | None = None,
) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(DATA_CONFIG_PATH)
    _, validation_dataset = build_datasets(data_config)
    stft_config = stft_config_from_data_config(data_config)
    subject_files = dict(find_subject_files(DATA_ROOT))
    model_settings = model_config["model"]
    model = ResidualAttentionUNet(
        in_channels=int(model_settings["in_channels"]),
        out_channels=int(model_settings["out_channels"]),
        base_channels=int(model_settings["base_channels"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print("Segment-level Summary:")
    print(frame.describe().to_string())

    summary = summarize_waveform_metrics(frame)
    print("\nOverall Waveform Summary Metrics:")
    for key, val in summary.items():
        print(f"  {key}: {val:.6f}")

    if subject_summary_path is not None:
        subject_df = aggregate_subject_waveform_metrics(frame)
        subject_summary_path.parent.mkdir(parents=True, exist_ok=True)
        subject_df.to_csv(subject_summary_path, index=False)
        print(f"\nSaved Subject-level Summary: {subject_summary_path}")

    print(f"\nSaved Segment-level CSV: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-summary-output", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(
        args.checkpoint,
        args.output,
        max_batches=args.max_batches,
        subject_summary_path=args.subject_summary_output,
    )

