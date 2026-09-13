from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.models.timing_head import CausalTimingHead
from guard_connection_ai.streaming.learned_pseudo_ecg import (
    StatefulLearnedTimingPseudoECG,
)


@dataclass(frozen=True)
class TimingDeploymentProfile:
    model: CausalTimingHead
    ppg_scaler: FixedRobustScaler
    sample_rate_hz: float
    output_delay_seconds: float
    event_threshold: float
    test_f1: float
    test_recall: float

    def new_session(self) -> StatefulLearnedTimingPseudoECG:
        """Create isolated decoder and renderer state for one measurement session."""

        return StatefulLearnedTimingPseudoECG(
            self.model,
            sample_rate_hz=self.sample_rate_hz,
            output_delay_seconds=self.output_delay_seconds,
            event_threshold=self.event_threshold,
        )


def load_validated_timing_checkpoint(
    path: str | Path,
    *,
    minimum_test_f1: float = 0.8,
    minimum_test_recall: float = 0.8,
    require_full_training: bool = True,
    sample_rate_hz: float | None = None,
) -> TimingDeploymentProfile:
    """Load a Timing Head only after provenance and held-out metric checks."""

    if not 0 <= minimum_test_f1 <= 1 or not 0 <= minimum_test_recall <= 1:
        raise ValueError("minimum metrics must be between zero and one.")
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    report = checkpoint["report"]
    if report.get("runtime_ecg_input") is not False:
        raise ValueError("checkpoint does not explicitly exclude runtime ECG input.")
    if report.get("diagnostic_ecg") is not False:
        raise ValueError("checkpoint is not marked as non-diagnostic.")
    if require_full_training and report.get("max_batches_per_epoch") is not None:
        raise ValueError("engineering smoke checkpoints cannot be deployed.")
    metrics = report.get("test_event_metrics", {})
    test_f1 = float(metrics.get("f1", float("nan")))
    test_recall = float(metrics.get("recall", float("nan")))
    if test_f1 < minimum_test_f1 or test_recall < minimum_test_recall:
        raise ValueError(
            "held-out event metrics are below the deployment gate: "
            f"f1={test_f1:.3f}, recall={test_recall:.3f}."
        )
    config = checkpoint["config"]
    model_config = config["model"]
    model = CausalTimingHead(
        input_channels=int(model_config["input_channels"]),
        hidden_channels=int(model_config["hidden_channels"]),
        dilation_cycle=tuple(int(value) for value in model_config["dilation_cycle"]),
        stacks=int(model_config["stacks"]),
        kernel_size=int(model_config["kernel_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_delay = float(report["output_delay_seconds"])
    maximum_pat = float(config["alignment"]["maximum_seconds"])
    if output_delay < maximum_pat:
        raise ValueError(
            "checkpoint output delay is shorter than its maximum PAT assumption."
        )
    configured_rate = config["dataset"].get("sample_rate_hz", sample_rate_hz)
    if configured_rate is None or float(configured_rate) <= 0:
        raise ValueError(
            "checkpoint lacks a valid sample rate; provide sample_rate_hz explicitly."
        )
    return TimingDeploymentProfile(
        model=model,
        ppg_scaler=FixedRobustScaler(**report["ppg_scaler"]),
        sample_rate_hz=float(configured_rate),
        output_delay_seconds=output_delay,
        event_threshold=float(report["calibrated_event_threshold"]),
        test_f1=test_f1,
        test_recall=test_recall,
    )
