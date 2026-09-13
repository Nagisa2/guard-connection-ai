from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from guard_connection_ai.data.causal_preprocessing import FixedRobustScaler
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.models.ppg_morphology_encoder import PPGMorphologyEncoder
from guard_connection_ai.streaming.constrained_morphology_renderer import (
    StatefulConstrainedMorphologyRenderer,
)


@dataclass(frozen=True)
class MorphologyDeploymentProfile:
    encoder: PPGMorphologyEncoder
    decoder: MorphologyAutoencoder1D
    population_latent: torch.Tensor
    ppg_scaler: FixedRobustScaler
    sample_rate_hz: float
    timing_output_delay_seconds: float
    seconds_before_r: float
    context_seconds: float
    confidence_threshold: float
    learned_morphology_selected: bool
    morphology_source: str

    def new_session(self) -> StatefulConstrainedMorphologyRenderer:
        return StatefulConstrainedMorphologyRenderer(
            morphology_decoder=self.decoder,
            ppg_encoder=self.encoder,
            population_latent=self.population_latent,
            sample_rate_hz=self.sample_rate_hz,
            timing_output_delay_seconds=self.timing_output_delay_seconds,
            seconds_before_r=self.seconds_before_r,
            context_seconds=self.context_seconds,
            confidence_threshold=self.confidence_threshold,
            allow_learned_morphology=self.learned_morphology_selected,
        )


def load_morphology_checkpoint(
    path: str | Path, *, require_full_training: bool = True
) -> MorphologyDeploymentProfile:
    """Load Stage 3 while preserving its learned-versus-population decision."""

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    report = checkpoint["report"]
    if report.get("runtime_ecg_input") is not False:
        raise ValueError("checkpoint does not explicitly exclude runtime ECG input.")
    if report.get("diagnostic_ecg") is not False:
        raise ValueError("checkpoint is not marked as non-diagnostic.")
    if require_full_training and report.get("max_batches_per_epoch") is not None:
        raise ValueError("engineering smoke checkpoints cannot be deployed.")
    config = checkpoint["config"]
    morphology_config = checkpoint["morphology_config"]
    model_config = config["model"]
    encoder = PPGMorphologyEncoder(
        input_channels=int(model_config["input_channels"]),
        latent_dim=int(model_config["latent_dim"]),
        base_channels=int(model_config["base_channels"]),
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()
    morphology_model = morphology_config["model"]
    morphology_data = morphology_config["dataset"]
    sample_rate = float(morphology_data["sample_rate_hz"])
    beat_samples = round(
        (
            float(morphology_data["seconds_before_r"])
            + float(morphology_data["seconds_after_r"])
        )
        * sample_rate
    )
    decoder = MorphologyAutoencoder1D(
        beat_samples=beat_samples,
        latent_dim=int(morphology_model["latent_dim"]),
        base_channels=int(morphology_model["base_channels"]),
    )
    decoder.load_state_dict(checkpoint["morphology_state_dict"])
    decoder.eval()
    selected = report.get("learned_morphology_selected") is True
    validation_candidate = report.get("validation_candidate_selected") is True
    release_passed = report.get("test_release_gate_passed") is True
    source = str(report.get("deployment_morphology_source", "population_mean_latent"))
    if report.get("selection_split") != "validation":
        raise ValueError("checkpoint morphology selection was not made on validation data.")
    evaluation = report.get("validation_evaluation", {})
    learned = evaluation.get("learned_ppg_latent", {})
    population = evaluation.get("population_mean_latent", {})
    validation_selects_learned = bool(
        float(learned.get("mae", float("inf")))
        < float(population.get("mae", float("inf")))
        and float(learned.get("pearson", float("-inf")))
        > float(population.get("pearson", float("-inf")))
    )
    if validation_candidate != validation_selects_learned:
        raise ValueError("checkpoint morphology candidate does not match validation metrics.")
    test_evaluation = report.get("test_evaluation", {})
    test_learned = test_evaluation.get("learned_ppg_latent", {})
    test_population = test_evaluation.get("population_mean_latent", {})
    expected_release = bool(
        not validation_candidate
        or (
            float(test_learned.get("mae", float("inf")))
            <= float(test_population.get("mae", float("-inf")))
            and float(test_learned.get("pearson", float("-inf")))
            >= float(test_population.get("pearson", float("inf")))
        )
    )
    if release_passed != expected_release or selected != (
        validation_candidate and release_passed
    ):
        raise ValueError("checkpoint morphology release gate metadata is inconsistent.")
    if source == "learned_ppg_latent" and not selected:
        raise ValueError("checkpoint morphology selection metadata is inconsistent.")
    return MorphologyDeploymentProfile(
        encoder=encoder,
        decoder=decoder,
        population_latent=checkpoint["population_latent"].detach().cpu(),
        ppg_scaler=FixedRobustScaler(**report["ppg_scaler"]),
        sample_rate_hz=sample_rate,
        timing_output_delay_seconds=float(config["dataset"]["output_delay_seconds"]),
        seconds_before_r=float(morphology_data["seconds_before_r"]),
        context_seconds=float(config["dataset"]["ppg_context_seconds"]),
        confidence_threshold=float(
            config["deployment"]["morphology_confidence_threshold"]
        ),
        learned_morphology_selected=selected,
        morphology_source=source,
    )
