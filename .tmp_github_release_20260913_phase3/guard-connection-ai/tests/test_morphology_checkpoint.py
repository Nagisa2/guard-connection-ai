from __future__ import annotations

from pathlib import Path

import pytest
import torch

from guard_connection_ai.deployment.morphology_checkpoint import (
    load_morphology_checkpoint,
)
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.models.ppg_morphology_encoder import PPGMorphologyEncoder


def _checkpoint(path: Path, *, smoke: bool = False, selected: bool = False) -> Path:
    encoder = PPGMorphologyEncoder(latent_dim=4, base_channels=2)
    decoder = MorphologyAutoencoder1D(beat_samples=20, latent_dim=4, base_channels=2)
    learned_mae = 0.2 if selected else 0.4
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "morphology_state_dict": decoder.state_dict(),
            "population_latent": torch.zeros(4),
            "config": {
                "dataset": {
                    "output_delay_seconds": 0.5,
                    "ppg_context_seconds": 1.0,
                },
                "model": {"input_channels": 2, "latent_dim": 4, "base_channels": 2},
                "deployment": {"morphology_confidence_threshold": 0.5},
            },
            "morphology_config": {
                "dataset": {
                    "sample_rate_hz": 20,
                    "seconds_before_r": 0.2,
                    "seconds_after_r": 0.8,
                },
                "model": {"latent_dim": 4, "base_channels": 2},
            },
            "report": {
                "runtime_ecg_input": False,
                "diagnostic_ecg": False,
                "max_batches_per_epoch": 1 if smoke else None,
                "ppg_scaler": {"center": 0.0, "scale": 1.0},
                "learned_morphology_selected": selected,
                "deployment_morphology_source": (
                    "learned_ppg_latent" if selected else "population_mean_latent"
                ),
                "selection_split": "validation",
                "validation_candidate_selected": selected,
                "test_release_gate_passed": True,
                "validation_evaluation": {
                    "learned_ppg_latent": {"mae": learned_mae, "pearson": 0.5},
                    "population_mean_latent": {"mae": 0.3, "pearson": 0.4},
                },
                "test_evaluation": {
                    "learned_ppg_latent": {"mae": learned_mae, "pearson": 0.5},
                    "population_mean_latent": {"mae": 0.3, "pearson": 0.4},
                },
            },
        },
        path,
    )
    return path


def test_loader_preserves_population_fallback_and_session_isolation(
    tmp_path: Path,
) -> None:
    profile = load_morphology_checkpoint(_checkpoint(tmp_path / "fallback.pt"))
    assert not profile.learned_morphology_selected
    assert profile.morphology_source == "population_mean_latent"
    assert profile.new_session() is not profile.new_session()


def test_loader_accepts_learned_only_when_validation_metrics_improve(
    tmp_path: Path,
) -> None:
    profile = load_morphology_checkpoint(
        _checkpoint(tmp_path / "selected.pt", selected=True)
    )
    assert profile.learned_morphology_selected
    checkpoint = torch.load(tmp_path / "selected.pt", weights_only=False)
    checkpoint["report"]["learned_morphology_selected"] = False
    torch.save(checkpoint, tmp_path / "inconsistent.pt")
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        load_morphology_checkpoint(tmp_path / "inconsistent.pt")


def test_loader_preserves_population_fallback_when_test_release_fails(
    tmp_path: Path,
) -> None:
    path = _checkpoint(tmp_path / "release_failed.pt", selected=True)
    checkpoint = torch.load(path, weights_only=False)
    report = checkpoint["report"]
    report["test_evaluation"]["learned_ppg_latent"]["mae"] = 0.4
    report["test_release_gate_passed"] = False
    report["learned_morphology_selected"] = False
    report["deployment_morphology_source"] = "population_mean_latent"
    torch.save(checkpoint, path)
    profile = load_morphology_checkpoint(path)
    assert not profile.learned_morphology_selected


def test_loader_rejects_test_selected_checkpoint(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "legacy.pt")
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["report"].pop("selection_split")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="validation data"):
        load_morphology_checkpoint(path)


def test_loader_rejects_smoke_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke"):
        load_morphology_checkpoint(_checkpoint(tmp_path / "smoke.pt", smoke=True))
