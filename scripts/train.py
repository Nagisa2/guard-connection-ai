from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import (
    BIDMCSTFTDataset,
    find_subject_files,
    load_subject_signals,
)
from guard_connection_ai.data.segmentation import build_segment_index
from guard_connection_ai.data.split import split_subjects_train_val
from guard_connection_ai.losses.reconstruction import combined_reconstruction_loss
from guard_connection_ai.metrics.image_metrics import spectrogram_metrics
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet
from guard_connection_ai.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "resunet.yaml"
SEGMENT_INDEX_PATH = PROJECT_ROOT / "outputs" / "segmentation" / "segment_index.csv"


def load_model_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_datasets(
    data_config: dict,
    cache_stft: bool = False,
) -> tuple[BIDMCSTFTDataset, BIDMCSTFTDataset]:
    data_root = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
    subject_files = dict(find_subject_files(data_root))
    subject_lengths = {
        subject_id: len(load_subject_signals(path)[0])
        for subject_id, path in subject_files.items()
    }
    train_subjects, validation_subjects = split_subjects_train_val(
        list(subject_files), train_size=43, random_state=int(data_config["seed"])
    )
    index = build_segment_index(
        subject_lengths,
        train_subjects,
        validation_subjects,
        window_samples=int(data_config["segmentation"]["window_samples"]),
        hop_samples=int(data_config["segmentation"]["hop_samples"]),
    )
    preprocessing = data_config["preprocessing"]
    return (
        BIDMCSTFTDataset(
            index,
            subject_files,
            stft_config=stft_config_from_data_config(data_config),
            preprocessing_mode=preprocessing["mode"],
            split="train",
            cache_stft=cache_stft,
        ),
        BIDMCSTFTDataset(
            index,
            subject_files,
            stft_config=stft_config_from_data_config(data_config),
            preprocessing_mode=preprocessing["mode"],
            split="validation",
            cache_stft=cache_stft,
        ),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None = None,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.0,
    frequency_weight: float = 0.0,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        with torch.set_grad_enabled(training):
            if scaler is not None and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    prediction = model(inputs)
                    loss = combined_reconstruction_loss(
                        prediction,
                        targets,
                        l1_weight=l1_weight,
                        ssim_weight=ssim_weight,
                        frequency_weight=frequency_weight,
                    )
            else:
                prediction = model(inputs)
                loss = combined_reconstruction_loss(
                    prediction,
                    targets,
                    l1_weight=l1_weight,
                    ssim_weight=ssim_weight,
                    frequency_weight=frequency_weight,
                )
        if training:
            optimizer.zero_grad()
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        batches += 1
    if batches == 0:
        raise ValueError("No batches were processed.")
    return total_loss / batches


def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.0,
    frequency_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(device)
            target = batch["target"].to(device)
            predictions.append(model(inputs))
            targets.append(target)
    if not predictions:
        raise ValueError("No batches were processed.")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return (
        combined_reconstruction_loss(
            prediction,
            target,
            l1_weight=l1_weight,
            ssim_weight=ssim_weight,
            frequency_weight=frequency_weight,
        ).item(),
        spectrogram_metrics(prediction, target),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
        },
        path,
    )


def main(
    epochs: int | None = None,
    max_batches: int | None = None,
    ssim_weight: float | None = None,
    frequency_weight: float | None = None,
    patience: int | None = None,
    seed: int | None = None,
    num_workers: int = 0,
    use_amp: bool = False,
    resume: Path | None = None,
    cache_stft: bool = False,
) -> None:
    data_config = load_data_config(DATA_CONFIG_PATH)
    model_config = load_model_config(MODEL_CONFIG_PATH)
    selected_seed = int(
        seed if seed is not None else model_config.get("experiment", {}).get("seed", 42)
    )
    set_seed(selected_seed)
    data_config["seed"] = selected_seed
    train_dataset, validation_dataset = build_datasets(data_config, cache_stft=cache_stft)
    training_config = model_config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
    )
    model_settings = model_config["model"]
    model = ResidualAttentionUNet(
        in_channels=int(model_settings["in_channels"]),
        out_channels=int(model_settings["out_channels"]),
        base_channels=int(model_settings["base_channels"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_config["learning_rate"]))

    start_epoch = 0
    if resume is not None:
        if not resume.exists():
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume}")
        print(f"Resuming training from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"Resumed from epoch {start_epoch}")

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    loss_config = model_config["loss"]
    selected_ssim_weight = float(
        loss_config["ssim_weight"] if ssim_weight is None else ssim_weight
    )
    selected_l1_weight = float(loss_config["l1_weight"])
    selected_frequency_weight = float(
        loss_config["frequency_weight"] if frequency_weight is None else frequency_weight
    )
    if selected_frequency_weight:
        experiment_suffix = "l1_ssim_frequency"
    elif selected_ssim_weight:
        experiment_suffix = "l1_ssim"
    else:
        experiment_suffix = "l1"

    seed_tag = f"_seed{selected_seed}" if seed is not None else ""
    checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / f"resunet_attention_{experiment_suffix}{seed_tag}_last.pt"
    )
    best_checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / f"resunet_attention_{experiment_suffix}{seed_tag}_best.pt"
    )
    history_path = (
        PROJECT_ROOT / "outputs" / "training" / f"{experiment_suffix}{seed_tag}_history.csv"
    )
    experiment_name = f"resunet-attention-{experiment_suffix}"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": model_settings["name"],
                "learning_rate": float(training_config["learning_rate"]),
                "batch_size": int(training_config["batch_size"]),
                "epochs": int(epochs or training_config["epochs"]),
                "optimizer": "Adam",
                "loss": (
                    "L1+SSIM+Frequency"
                    if selected_frequency_weight
                    else "L1+SSIM"
                    if selected_ssim_weight
                    else "L1"
                ),
                "l1_weight": float(loss_config["l1_weight"]),
                "ssim_weight": selected_ssim_weight,
                "frequency_weight": selected_frequency_weight,
                "preprocessing_mode": data_config["preprocessing"]["mode"],
                "seed": selected_seed,
                "train_subjects": 43,
                "validation_subjects": 10,
                "device": str(device),
                "num_workers": num_workers,
                "use_amp": use_amp,
                "cache_stft": cache_stft,
                "patience": int(patience or training_config.get("patience", 10)),
            }
        )
        history = []
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        selected_patience = patience or int(training_config.get("patience", 10))
        total_epochs = int(epochs or training_config["epochs"])
        for epoch in range(start_epoch, total_epochs):
            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches,
                selected_l1_weight,
                selected_ssim_weight,
                selected_frequency_weight,
                scaler=scaler,
            )
            validation_loss, validation_metrics = evaluate_epoch(
                model,
                validation_loader,
                device,
                max_batches,
                selected_l1_weight,
                selected_ssim_weight,
                selected_frequency_weight,
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "val_mae": validation_metrics["mae"],
                    "val_rmse": validation_metrics["rmse"],
                    "val_prd": validation_metrics["prd"],
                    "val_correlation": validation_metrics["correlation"],
                },
                step=epoch,
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": validation_loss,
                    "val_mae": validation_metrics["mae"],
                    "val_rmse": validation_metrics["rmse"],
                    "val_prd": validation_metrics["prd"],
                    "val_correlation": validation_metrics["correlation"],
                }
            )
            history_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(history_path, index=False)
            save_checkpoint(model, optimizer, epoch + 1, model_config, checkpoint_path)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, epoch + 1, model_config, best_checkpoint_path)
            else:
                epochs_without_improvement += 1
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"val_loss={validation_loss:.6f} "
                f"val_mae={validation_metrics['mae']:.6f} "
                f"val_rmse={validation_metrics['rmse']:.6f} "
                f"val_prd={validation_metrics['prd']:.6f} "
                f"val_correlation={validation_metrics['correlation']:.6f}"
            )
            if epochs_without_improvement >= selected_patience:
                print(f"early_stopping epoch={epoch + 1}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--ssim-weight", type=float, default=None)
    parser.add_argument("--frequency-weight", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--cache-stft", action="store_true")
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        max_batches=args.max_batches,
        ssim_weight=args.ssim_weight,
        frequency_weight=args.frequency_weight,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        resume=args.resume,
        cache_stft=args.cache_stft,
    )
