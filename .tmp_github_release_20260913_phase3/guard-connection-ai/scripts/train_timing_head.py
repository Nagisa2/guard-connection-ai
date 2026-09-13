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
from guard_connection_ai.data.timing_dataset import PairedMIMICPerformTimingDataset
from guard_connection_ai.losses.timing import TimingHeadLoss
from guard_connection_ai.metrics.pulse_arrival import estimate_pulse_arrival_delay
from guard_connection_ai.metrics.timing import (
    calibrate_event_threshold,
    evaluate_probability_windows,
)
from guard_connection_ai.models.timing_head import CausalTimingHead
from guard_connection_ai.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "paired_waveform.yaml"
DEFAULT_TIMING_CONFIG = PROJECT_ROOT / "configs" / "timing_head.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must contain a mapping: {path}")
    return value


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            masks = batch["loss_mask"].to(device)
            losses = criterion(model(inputs), targets, masks)
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    if count == 0:
        raise ValueError("No timing batches were available.")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _collect_probability_windows(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader):
        output = torch.sigmoid(model(batch["input"].to(device))).cpu().numpy()
        target = batch["target"].numpy()
        mask = batch["loss_mask"].numpy().astype(bool)
        probabilities.extend(output[:, 0])
        targets.extend(target[:, 0])
        masks.extend(mask[:, 0])
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return probabilities, targets, masks


def main(
    *,
    dataset_config_path: Path,
    timing_config_path: Path,
    test_fold: int,
    output_dir: Path,
    epochs: int | None = None,
    max_batches: int | None = None,
    device_name: str = "auto",
) -> dict:
    dataset_config = _load_yaml(dataset_config_path)
    config = _load_yaml(timing_config_path)
    seed = int(dataset_config["split"]["seed"])
    set_seed(seed)
    training = config["training"]
    torch.set_num_threads(int(training.get("torch_threads", torch.get_num_threads())))
    recordings = discover_mimic_perform_af(PROJECT_ROOT / config["dataset"]["root"])
    n_splits = int(dataset_config["split"]["n_splits"])
    assignments = assign_stratified_patient_folds(
        recordings, n_splits=n_splits, random_state=seed
    )
    validation_fold = (
        test_fold + int(dataset_config["split"]["validation_fold_offset"])
    ) % n_splits
    partitions = partition_recordings(
        recordings, assignments, test_fold=test_fold, validation_fold=validation_fold
    )
    train_pairs = []
    train_ppg = []
    for recording in partitions["train"]:
        _, ppg, ecg = load_mimic_perform_signals(recording)
        train_ppg.append(ppg)
        train_pairs.append((ppg, ecg, recording.sample_rate_hz))
    ppg_scaler = fit_fixed_robust_scaler(train_ppg)
    alignment = config["alignment"]
    pat = estimate_pulse_arrival_delay(
        train_pairs,
        minimum_seconds=float(alignment["minimum_seconds"]),
        maximum_seconds=float(alignment["maximum_seconds"]),
        minimum_beats_per_recording=int(alignment["minimum_beats_per_recording"]),
        upper_quantile=float(alignment["upper_quantile"]),
        minimum_recording_coverage=float(alignment["minimum_recording_coverage"]),
    )
    output_delay_seconds = float(
        alignment.get("output_delay_seconds", pat.upper_quantile_seconds)
    )
    if output_delay_seconds < float(alignment["maximum_seconds"]):
        raise ValueError(
            "output_delay_seconds must be at least the configured maximum PAT "
            "to keep R-event supervision causal."
        )
    windowing = config["windowing"]
    datasets = {
        split: PairedMIMICPerformTimingDataset(
            values,
            ppg_scaler=ppg_scaler,
            output_delay_seconds=output_delay_seconds,
            window_seconds=float(windowing["window_seconds"]),
            hop_seconds=float(windowing["training_hop_seconds"]),
            loss_warmup_seconds=float(windowing["loss_warmup_seconds"]),
        )
        for split, values in partitions.items()
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
    model_config = config["model"]
    model = CausalTimingHead(
        input_channels=int(model_config["input_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        dilation_cycle=tuple(model_config["dilation_cycle"]),
        stacks=int(model_config["stacks"]),
        kernel_size=int(model_config["kernel_size"]),
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
    loss_config = config["loss"]
    criterion = TimingHeadLoss(
        focal_gamma=float(loss_config["focal_gamma"]),
        dice_weight=float(loss_config["dice_weight"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    selected_epochs = int(training["epochs"] if epochs is None else epochs)
    if selected_epochs <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("epochs and max_batches must be positive.")
    started = time.perf_counter()
    history = []
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(selected_epochs):
        train_losses = _epoch(
            model, loaders["train"], criterion, device, optimizer, max_batches
        )
        validation_losses = _epoch(
            model, loaders["validation"], criterion, device, None, max_batches
        )
        history.append(
            {"epoch": epoch + 1, "train": train_losses, "validation": validation_losses}
        )
        print(
            f"epoch={epoch + 1} train={train_losses['total']:.6f} validation={validation_losses['total']:.6f}"
        )
        if validation_losses["total"] < best_validation:
            best_validation = validation_losses["total"]
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
    test_losses = _epoch(model, loaders["test"], criterion, device, None, max_batches)
    deployment = config["deployment"]
    validation_windows = _collect_probability_windows(
        model, loaders["validation"], device, max_batches
    )
    calibration = calibrate_event_threshold(
        *validation_windows,
        sample_rate_hz=float(config.get("dataset", {}).get("sample_rate_hz", 125)),
        thresholds=tuple(
            float(value) for value in deployment["calibration_thresholds"]
        ),
        refractory_seconds=float(deployment["refractory_seconds"]),
        tolerance_seconds=float(deployment["event_tolerance_seconds"]),
    )
    test_windows = _collect_probability_windows(
        model, loaders["test"], device, max_batches
    )
    test_event_metrics = evaluate_probability_windows(
        *test_windows,
        sample_rate_hz=float(config.get("dataset", {}).get("sample_rate_hz", 125)),
        threshold=calibration.threshold,
        refractory_seconds=float(deployment["refractory_seconds"]),
        tolerance_seconds=float(deployment["event_tolerance_seconds"]),
    )
    report = {
        "test_fold": test_fold,
        "validation_fold": validation_fold,
        "patients": {
            split: sorted({recording.patient_id for recording in values})
            for split, values in partitions.items()
        },
        "ppg_scaler": asdict(ppg_scaler),
        "pulse_arrival_estimate": asdict(pat),
        "output_delay_seconds": output_delay_seconds,
        "history": history,
        "test_losses": test_losses,
        "calibrated_event_threshold": calibration.threshold,
        "validation_event_metrics": asdict(calibration.metrics),
        "test_event_metrics": asdict(test_event_metrics),
        "device": str(device),
        "max_batches_per_epoch": max_batches,
        "training_duration_seconds": time.perf_counter() - started,
        "claims": "engineering smoke run only"
        if max_batches is not None
        else "internal timing-head training run",
        "runtime_ecg_input": False,
        "diagnostic_ecg": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"timing_head_fold{test_fold}.pt"
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
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--timing-config", type=Path, default=DEFAULT_TIMING_CONFIG)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "timing_head"
    )
    args = parser.parse_args()
    main(
        dataset_config_path=args.dataset_config,
        timing_config_path=args.timing_config,
        test_fold=args.test_fold,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_batches=args.max_batches,
        device_name=args.device,
    )
