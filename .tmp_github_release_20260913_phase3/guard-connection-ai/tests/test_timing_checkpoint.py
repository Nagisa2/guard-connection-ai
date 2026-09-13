from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guard_connection_ai.deployment.timing_checkpoint import (
    load_validated_timing_checkpoint,
)
from guard_connection_ai.models.timing_head import CausalTimingHead


def _checkpoint(path: Path, *, f1: float = 0.9, smoke: bool = False) -> Path:
    model = CausalTimingHead(
        input_channels=2,
        hidden_channels=4,
        dilation_cycle=(1, 2),
        stacks=1,
        kernel_size=3,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "dataset": {"sample_rate_hz": 125},
                "alignment": {"maximum_seconds": 0.5},
                "model": {
                    "input_channels": 2,
                    "hidden_channels": 4,
                    "dilation_cycle": [1, 2],
                    "stacks": 1,
                    "kernel_size": 3,
                },
            },
            "report": {
                "runtime_ecg_input": False,
                "diagnostic_ecg": False,
                "max_batches_per_epoch": 1 if smoke else None,
                "test_event_metrics": {"f1": f1, "recall": 0.9},
                "ppg_scaler": {"center": 0.0, "scale": 1.0},
                "output_delay_seconds": 0.5,
                "calibrated_event_threshold": 0.6,
            },
        },
        path,
    )
    return path


def test_validated_checkpoint_creates_isolated_sessions(tmp_path: Path) -> None:
    profile = load_validated_timing_checkpoint(_checkpoint(tmp_path / "timing.pt"))
    first = profile.new_session()
    second = profile.new_session()
    assert first is not second
    assert first.decoder is not second.decoder
    assert profile.event_threshold == 0.6


def test_checkpoint_gate_rejects_smoke_and_low_f1(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke"):
        load_validated_timing_checkpoint(_checkpoint(tmp_path / "smoke.pt", smoke=True))
    with pytest.raises(ValueError, match="below"):
        load_validated_timing_checkpoint(_checkpoint(tmp_path / "weak.pt", f1=0.4))


def test_legacy_checkpoint_requires_explicit_sample_rate(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "legacy.pt")
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["config"]["dataset"].pop("sample_rate_hz")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="sample rate"):
        load_validated_timing_checkpoint(path)
    profile = load_validated_timing_checkpoint(path, sample_rate_hz=125)
    assert profile.sample_rate_hz == 125
