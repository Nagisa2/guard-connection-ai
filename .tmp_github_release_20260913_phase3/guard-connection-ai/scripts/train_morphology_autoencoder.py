from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from guard_connection_ai.data.af_split import (
    assign_stratified_patient_folds,
    partition_recordings,
)
from guard_connection_ai.data.causal_preprocessing import fit_fixed_robust_scaler
from guard_connection_ai.data.mimic_perform_af import (
    discover_mimic_perform_af,
    load_mimic_perform_signals,
)
from guard_connection_ai.data.morphology_dataset import ECGBeatMorphologyDataset
from guard_connection_ai.losses.morphology import MorphologyAutoencoderLoss
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "paired_waveform.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "morphology_autoencoder.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must contain a mapping: {path}")
    return value


def _epoch(
    model: MorphologyAutoencoder1D,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    batches = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            waveform = batch["waveform"].to(device)
            valid = batch["valid_mask"].to(device)
            prediction, latent = model(waveform)
            losses = criterion(prediction, waveform, valid, latent)
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    if batches == 0:
        raise ValueError("No morphology beats were available.")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def _reconstruction_metrics(
    model: MorphologyAutoencoder1D,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    references = []
    predictions = []
    for batch_index, batch in enumerate(loader):
        waveform = batch["waveform"].to(device)
        prediction, _ = model(waveform)
        valid = batch["valid_mask"].numpy().astype(bool)
        reference_values = waveform.cpu().numpy()
        prediction_values = prediction.cpu().numpy()
        references.append(reference_values[valid])
        predictions.append(prediction_values[valid])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    reference = np.concatenate(references)
    prediction = np.concatenate(predictions)
    error = prediction - reference
    correlation = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if np.std(reference) > 1e-8 and np.std(prediction) > 1e-8
        else float("nan")
    )
    return {
        "r_aligned_mae": float(np.mean(np.abs(error))),
        "r_aligned_rmse": float(np.sqrt(np.mean(error * error))),
        "r_aligned_pearson": correlation,
        "valid_samples": int(reference.size),
    }


def main(
    *,
    dataset_config_path: Path,
    model_config_path: Path,
    test_fold: int,
    output_dir: Path,
    epochs: int | None = None,
    max_batches: int | None = None,
    device_name: str = "auto",
) -> dict:
    dataset_config = _load_yaml(dataset_config_path)
    config = _load_yaml(model_config_path)
    seed = int(dataset_config["split"]["seed"])
    set_seed(seed)
    training = config["training"]
    torch.set_num_threads(int(training.get("torch_threads", torch.get_num_threads())))
    records = discover_mimic_perform_af(PROJECT_ROOT / config["dataset"]["root"])
    n_splits = int(dataset_config["split"]["n_splits"])
    assignments = assign_stratified_patient_folds(
        records, n_splits=n_splits, random_state=seed
    )
    validation_fold = (
        test_fold + int(dataset_config["split"]["validation_fold_offset"])
    ) % n_splits
    partitions = partition_recordings(
        records,
        assignments,
        test_fold=test_fold,
        validation_fold=validation_fold,
    )
    training_ecg = [
        load_mimic_perform_signals(recording)[2] for recording in partitions["train"]
    ]
    ecg_scaler = fit_fixed_robust_scaler(training_ecg)
    data_config = config["dataset"]
    bandpass = tuple(float(value) for value in data_config["ecg_bandpass_hz"])
    datasets = {
        split: ECGBeatMorphologyDataset(
            split_records,
            ecg_scaler=ecg_scaler,
            seconds_before_r=float(data_config["seconds_before_r"]),
            seconds_after_r=float(data_config["seconds_after_r"]),
            ecg_bandpass_hz=bandpass,
            minimum_valid_fraction=float(data_config["minimum_valid_fraction"]),
            neighboring_qrs_guard_seconds=float(
                data_config["neighboring_qrs_guard_seconds"]
            ),
        )
        for split, split_records in partitions.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    sample_rate = float(data_config["sample_rate_hz"])
    before_samples = round(float(data_config["seconds_before_r"]) * sample_rate)
    beat_samples = before_samples + round(
        float(data_config["seconds_after_r"]) * sample_rate
    )
    model_config = config["model"]
    model = MorphologyAutoencoder1D(
        beat_samples=beat_samples,
        latent_dim=int(model_config["latent_dim"]),
        base_channels=int(model_config["base_channels"]),
    )
    loss_config = config["loss"]
    criterion = MorphologyAutoencoderLoss(
        sample_rate_hz=sample_rate,
        r_index=before_samples,
        fft_sizes=loss_config["fft_sizes"],
        qrs_weight=float(loss_config["qrs_weight"]),
        t_weight=float(loss_config["t_weight"]),
        derivative_weight=float(loss_config["derivative_weight"]),
        spectral_weight=float(loss_config["spectral_weight"]),
        latent_weight=float(loss_config["latent_weight"]),
    )
    selected_device = (
        training.get("device", device_name) if device_name == "auto" else device_name
    )
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(selected_device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    selected_epochs = int(training["epochs"] if epochs is None else epochs)
    if selected_epochs <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("epochs and max_batches must be positive.")
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(selected_epochs):
        train_loss = _epoch(
            model, loaders["train"], criterion, device, optimizer, max_batches
        )
        validation_loss = _epoch(
            model, loaders["validation"], criterion, device, None, max_batches
        )
        history.append(
            {"epoch": epoch + 1, "train": train_loss, "validation": validation_loss}
        )
        print(
            f"epoch={epoch + 1} train={train_loss['total']:.6f} "
            f"validation={validation_loss['total']:.6f}"
        )
        if validation_loss["total"] < best_validation:
            best_validation = validation_loss["total"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_loss = _epoch(model, loaders["test"], criterion, device, None, max_batches)
    reconstruction = _reconstruction_metrics(
        model, loaders["test"], device, max_batches
    )
    report = {
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "patients": {
            split: sorted({recording.patient_id for recording in split_records})
            for split, split_records in partitions.items()
        },
        "beat_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "ecg_scaler": asdict(ecg_scaler),
        "ecg_bandpass_hz": list(bandpass),
        "beat_samples": beat_samples,
        "r_index": before_samples,
        "neighboring_qrs_guard_seconds": float(
            data_config["neighboring_qrs_guard_seconds"]
        ),
        "history": history,
        "test_loss": test_loss,
        "test_reconstruction": reconstruction,
        "device": str(device),
        "max_batches_per_epoch": max_batches,
        "training_duration_seconds": time.perf_counter() - started,
        "claims": (
            "engineering smoke run only"
            if max_batches is not None
            else "internal ECG morphology-prior training run"
        ),
        "runtime_ppg_to_ecg_model": False,
        "diagnostic_ecg": False,
        "patient_specific_morphology_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"morphology_autoencoder_fold{test_fold}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": config,
            "dataset_config": dataset_config,
            "report": report,
        },
        checkpoint,
    )
    checkpoint.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "morphology_autoencoder",
    )
    arguments = parser.parse_args()
    main(
        dataset_config_path=arguments.dataset_config,
        model_config_path=arguments.model_config,
        test_fold=arguments.test_fold,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        max_batches=arguments.max_batches,
        device_name=arguments.device,
    )
