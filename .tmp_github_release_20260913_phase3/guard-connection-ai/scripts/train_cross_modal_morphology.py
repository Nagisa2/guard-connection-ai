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
from guard_connection_ai.data.cross_modal_morphology_dataset import (
    PairedPPGMorphologyDataset,
)
from guard_connection_ai.data.mimic_perform_af import (
    discover_mimic_perform_af,
    load_mimic_perform_signals,
)
from guard_connection_ai.data.morphology_dataset import ECGBeatMorphologyDataset
from guard_connection_ai.losses.cross_modal_morphology import (
    CrossModalMorphologyLoss,
)
from guard_connection_ai.losses.morphology import MorphologyAutoencoderLoss
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.models.ppg_morphology_encoder import (
    PPGMorphologyEncoder,
    morphology_confidence,
)
from guard_connection_ai.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "paired_waveform.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "cross_modal_morphology.yaml"
DEFAULT_MORPHOLOGY_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "morphology_autoencoder_stage2_masked_fold0_20260905"
    / "morphology_autoencoder_fold0.pt"
)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must contain a mapping: {path}")
    return value


def _load_frozen_autoencoder(checkpoint: dict) -> MorphologyAutoencoder1D:
    report = checkpoint["report"]
    model_config = checkpoint["config"]["model"]
    model = MorphologyAutoencoder1D(
        beat_samples=int(report["beat_samples"]),
        latent_dim=int(model_config["latent_dim"]),
        base_channels=int(model_config["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _epoch(
    encoder: PPGMorphologyEncoder,
    autoencoder: MorphologyAutoencoder1D,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    encoder.train(optimizer is not None)
    totals: dict[str, float] = {}
    batches = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            ppg = batch["ppg_context"].to(device)
            ecg = batch["ecg_beat"].to(device)
            valid = batch["ecg_valid_mask"].to(device)
            with torch.no_grad():
                target_latent = autoencoder.encode(ecg)
            mean, log_variance = encoder(ppg)
            reconstruction = autoencoder.decode(mean)
            losses = criterion(
                mean,
                log_variance,
                target_latent,
                reconstruction,
                ecg,
                valid,
            )
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
        raise ValueError("No cross-modal morphology batches were available.")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def _population_latent(
    autoencoder: MorphologyAutoencoder1D,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> torch.Tensor:
    total = None
    count = 0
    for batch_index, batch in enumerate(loader):
        latent = autoencoder.encode(batch["ecg_beat"].to(device))
        batch_sum = torch.sum(latent, dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += latent.shape[0]
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if total is None or count == 0:
        raise ValueError("No training latent values were available.")
    return total / count


def _similarity(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    correlation = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if np.std(reference) > 1e-8 and np.std(prediction) > 1e-8
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson": correlation,
    }


@torch.no_grad()
def _evaluate(
    encoder: PPGMorphologyEncoder,
    autoencoder: MorphologyAutoencoder1D,
    loader: DataLoader,
    population_latent: torch.Tensor,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, object]:
    encoder.eval()
    references = []
    learned_predictions = []
    population_predictions = []
    latent_errors = []
    confidences = []
    for batch_index, batch in enumerate(loader):
        ppg = batch["ppg_context"].to(device)
        ecg = batch["ecg_beat"].to(device)
        valid = batch["ecg_valid_mask"].numpy().astype(bool)
        target_latent = autoencoder.encode(ecg)
        mean, log_variance = encoder(ppg)
        learned = autoencoder.decode(mean).cpu().numpy()
        population = (
            autoencoder.decode(population_latent[None, :].expand(ecg.shape[0], -1))
            .cpu()
            .numpy()
        )
        reference = ecg.cpu().numpy()
        references.append(reference[valid])
        learned_predictions.append(learned[valid])
        population_predictions.append(population[valid])
        latent_errors.append(
            torch.mean(torch.abs(mean - target_latent), dim=-1).cpu().numpy()
        )
        confidences.append(morphology_confidence(log_variance).cpu().numpy())
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    reference_values = np.concatenate(references)
    learned_values = np.concatenate(learned_predictions)
    population_values = np.concatenate(population_predictions)
    return {
        "learned_ppg_latent": _similarity(reference_values, learned_values),
        "population_mean_latent": _similarity(reference_values, population_values),
        "latent_mae_mean": float(np.mean(np.concatenate(latent_errors))),
        "morphology_confidence_mean": float(np.mean(np.concatenate(confidences))),
        "valid_samples": int(reference_values.size),
    }


def main(
    *,
    dataset_config_path: Path,
    model_config_path: Path,
    morphology_checkpoint_path: Path,
    test_fold: int,
    output_dir: Path,
    epochs: int | None = None,
    max_batches: int | None = None,
    device_name: str = "auto",
) -> dict:
    dataset_config = _load_yaml(dataset_config_path)
    config = _load_yaml(model_config_path)
    morphology_checkpoint = torch.load(
        morphology_checkpoint_path, map_location="cpu", weights_only=False
    )
    morphology_report = morphology_checkpoint["report"]
    if int(morphology_report["test_fold"]) != test_fold:
        raise ValueError("morphology checkpoint and cross-modal test fold must match.")
    if morphology_report.get("max_batches_per_epoch") is not None:
        raise ValueError("smoke morphology checkpoints cannot supervise Stage 3.")
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
        records, assignments, test_fold=test_fold, validation_fold=validation_fold
    )
    actual_patients = {
        split: sorted({recording.patient_id for recording in split_records})
        for split, split_records in partitions.items()
    }
    if actual_patients != morphology_report["patients"]:
        raise ValueError("patient split differs from the morphology checkpoint split.")
    train_ppg = [
        load_mimic_perform_signals(recording)[1] for recording in partitions["train"]
    ]
    ppg_scaler = fit_fixed_robust_scaler(train_ppg)
    morphology_config = morphology_checkpoint["config"]
    morphology_data = morphology_config["dataset"]
    ecg_scaler = morphology_report["ecg_scaler"]
    morphology_datasets = {
        split: ECGBeatMorphologyDataset(
            split_records,
            ecg_scaler=type(ppg_scaler)(**ecg_scaler),
            seconds_before_r=float(morphology_data["seconds_before_r"]),
            seconds_after_r=float(morphology_data["seconds_after_r"]),
            ecg_bandpass_hz=tuple(morphology_data["ecg_bandpass_hz"]),
            minimum_valid_fraction=float(morphology_data["minimum_valid_fraction"]),
            neighboring_qrs_guard_seconds=float(
                morphology_data["neighboring_qrs_guard_seconds"]
            ),
        )
        for split, split_records in partitions.items()
    }
    data_config = config["dataset"]
    datasets = {
        split: PairedPPGMorphologyDataset(
            dataset,
            ppg_scaler=ppg_scaler,
            output_delay_seconds=float(data_config["output_delay_seconds"]),
            context_seconds=float(data_config["ppg_context_seconds"]),
            minimum_ppg_valid_fraction=float(data_config["minimum_ppg_valid_fraction"]),
        )
        for split, dataset in morphology_datasets.items()
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
    selected_device = (
        training.get("device", device_name) if device_name == "auto" else device_name
    )
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(selected_device)
    autoencoder = _load_frozen_autoencoder(morphology_checkpoint).to(device)
    model_config = config["model"]
    encoder = PPGMorphologyEncoder(
        input_channels=int(model_config["input_channels"]),
        latent_dim=int(model_config["latent_dim"]),
        base_channels=int(model_config["base_channels"]),
    ).to(device)
    population_latent = _population_latent(
        autoencoder, loaders["train"], device, max_batches
    )
    encoder.set_population_latent(population_latent)
    morphology_loss_config = morphology_config["loss"]
    morphology_loss = MorphologyAutoencoderLoss(
        sample_rate_hz=float(morphology_data["sample_rate_hz"]),
        r_index=int(morphology_report["r_index"]),
        fft_sizes=morphology_loss_config["fft_sizes"],
        qrs_weight=float(morphology_loss_config["qrs_weight"]),
        t_weight=float(morphology_loss_config["t_weight"]),
        derivative_weight=float(morphology_loss_config["derivative_weight"]),
        spectral_weight=float(morphology_loss_config["spectral_weight"]),
        latent_weight=0.0,
    )
    loss_config = config["loss"]
    criterion = CrossModalMorphologyLoss(
        morphology_loss,
        latent_nll_weight=float(loss_config["latent_nll_weight"]),
        cosine_weight=float(loss_config["cosine_weight"]),
        reconstruction_weight=float(loss_config["reconstruction_weight"]),
    )
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=float(training["learning_rate"])
    )
    selected_epochs = int(training["epochs"] if epochs is None else epochs)
    if selected_epochs <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("epochs and max_batches must be positive.")
    initial_validation = _epoch(
        encoder,
        autoencoder,
        loaders["validation"],
        criterion,
        device,
        None,
        max_batches,
    )
    best_validation = initial_validation["reconstruction"]
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in encoder.state_dict().items()
    }
    stale_epochs = 0
    history = [{"epoch": 0, "validation": initial_validation}]
    started = time.perf_counter()
    for epoch in range(selected_epochs):
        train_loss = _epoch(
            encoder,
            autoencoder,
            loaders["train"],
            criterion,
            device,
            optimizer,
            max_batches,
        )
        validation_loss = _epoch(
            encoder,
            autoencoder,
            loaders["validation"],
            criterion,
            device,
            None,
            max_batches,
        )
        history.append(
            {"epoch": epoch + 1, "train": train_loss, "validation": validation_loss}
        )
        print(
            f"epoch={epoch + 1} train={train_loss['total']:.6f} "
            f"validation={validation_loss['total']:.6f}"
        )
        if validation_loss["reconstruction"] < best_validation:
            best_validation = validation_loss["reconstruction"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in encoder.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    encoder.load_state_dict(best_state)
    test_loss = _epoch(
        encoder, autoencoder, loaders["test"], criterion, device, None, max_batches
    )
    validation_evaluation = _evaluate(
        encoder,
        autoencoder,
        loaders["validation"],
        population_latent,
        device,
        max_batches,
    )
    test_evaluation = _evaluate(
        encoder, autoencoder, loaders["test"], population_latent, device, max_batches
    )
    learned_metrics = validation_evaluation["learned_ppg_latent"]
    population_metrics = validation_evaluation["population_mean_latent"]
    validation_candidate_selected = bool(
        learned_metrics["mae"] < population_metrics["mae"]
        and learned_metrics["pearson"] > population_metrics["pearson"]
    )
    test_learned = test_evaluation["learned_ppg_latent"]
    test_population = test_evaluation["population_mean_latent"]
    test_release_gate_passed = bool(
        not validation_candidate_selected
        or (
            test_learned["mae"] <= test_population["mae"]
            and test_learned["pearson"] >= test_population["pearson"]
        )
    )
    learned_selected = validation_candidate_selected and test_release_gate_passed
    report = {
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "patients": actual_patients,
        "beat_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "ppg_scaler": asdict(ppg_scaler),
        "morphology_checkpoint": morphology_checkpoint_path.name,
        "morphology_checkpoint_test_reconstruction": morphology_report[
            "test_reconstruction"
        ],
        "history": history,
        "test_loss": test_loss,
        "validation_evaluation": validation_evaluation,
        "test_evaluation": test_evaluation,
        "selection_split": "validation",
        "validation_candidate_selected": validation_candidate_selected,
        "test_release_gate_passed": test_release_gate_passed,
        "learned_morphology_selected": learned_selected,
        "deployment_morphology_source": (
            "learned_ppg_latent" if learned_selected else "population_mean_latent"
        ),
        "device": str(device),
        "max_batches_per_epoch": max_batches,
        "training_duration_seconds": time.perf_counter() - started,
        "runtime_ecg_input": False,
        "diagnostic_ecg": False,
        "generation_mode": (
            "learned_morphology" if learned_selected else "population_prior"
        ),
        "patient_specific_morphology_claim": False,
        "claims": (
            "engineering smoke run only"
            if max_batches is not None
            else "internal PPG-to-morphology-latent training run"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"cross_modal_morphology_fold{test_fold}.pt"
    torch.save(
        {
            "encoder_state_dict": best_state,
            "morphology_state_dict": morphology_checkpoint["model_state_dict"],
            "population_latent": population_latent.detach().cpu(),
            "config": config,
            "morphology_config": morphology_config,
            "dataset_config": dataset_config,
            "report": report,
        },
        checkpoint_path,
    )
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--morphology-checkpoint", type=Path, default=DEFAULT_MORPHOLOGY_CHECKPOINT
    )
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "cross_modal_morphology",
    )
    arguments = parser.parse_args()
    main(
        dataset_config_path=arguments.dataset_config,
        model_config_path=arguments.model_config,
        morphology_checkpoint_path=arguments.morphology_checkpoint,
        test_fold=arguments.test_fold,
        output_dir=arguments.output_dir,
        epochs=arguments.epochs,
        max_batches=arguments.max_batches,
        device_name=arguments.device,
    )
