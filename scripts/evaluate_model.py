from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from train import build_datasets, load_model_config

from guard_connection_ai.data.config import load_data_config
from guard_connection_ai.data.dataset import BIDMCSTFTDataset
from guard_connection_ai.metrics.image_metrics import spectrogram_metrics
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "resunet.yaml"


def evaluate_subjects(
    model: torch.nn.Module,
    dataset: BIDMCSTFTDataset,
    device: torch.device,
    batch_size: int,
    max_batches: int | None = None,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: dict[str, list[torch.Tensor]] = {}
    targets: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            output = model(batch["input"].to(device)).cpu()
            target = batch["target"]
            for index, subject_id in enumerate(batch["subject_id"]):
                predictions.setdefault(subject_id, []).append(output[index : index + 1])
                targets.setdefault(subject_id, []).append(target[index : index + 1])
    if not predictions:
        raise ValueError("No batches were processed.")

    rows = []
    for subject_id in sorted(predictions):
        metrics = spectrogram_metrics(
            torch.cat(predictions[subject_id]),
            torch.cat(targets[subject_id]),
        )
        rows.append({"subject_id": subject_id, **metrics})
    return pd.DataFrame(rows)


def main(
    checkpoint_path: Path,
    output_path: Path,
    max_batches: int | None = None,
) -> None:
    model_config = load_model_config(MODEL_CONFIG_PATH)
    data_config = load_data_config(PROJECT_ROOT / "configs" / "data.yaml")
    _, validation_dataset = build_datasets(data_config)
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
    batch_size = int(model_config["training"]["batch_size"])
    frame = evaluate_subjects(model, validation_dataset, device, batch_size, max_batches)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame.describe().to_string())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(args.checkpoint, args.output, max_batches=args.max_batches)
