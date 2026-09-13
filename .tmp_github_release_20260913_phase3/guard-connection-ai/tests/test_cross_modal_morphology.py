from __future__ import annotations

import torch

from guard_connection_ai.losses.cross_modal_morphology import (
    CrossModalMorphologyLoss,
)
from guard_connection_ai.losses.morphology import MorphologyAutoencoderLoss
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.models.ppg_morphology_encoder import (
    PPGMorphologyEncoder,
    morphology_confidence,
)


def test_ppg_encoder_outputs_latent_distribution_without_ecg_input() -> None:
    model = PPGMorphologyEncoder(latent_dim=8, base_channels=4)
    population = torch.linspace(-1, 1, 8)
    model.set_population_latent(population)
    mean, log_variance = model(torch.randn(3, 2, 188))
    assert mean.shape == (3, 8)
    assert log_variance.shape == (3, 8)
    confidence = morphology_confidence(log_variance)
    assert confidence.shape == (3,)
    assert torch.all((0 <= confidence) & (confidence <= 1))
    torch.testing.assert_close(mean, population[None, :].expand(3, -1))


def test_cross_modal_loss_updates_ppg_encoder_not_frozen_decoder() -> None:
    encoder = PPGMorphologyEncoder(latent_dim=8, base_channels=4)
    autoencoder = MorphologyAutoencoder1D(
        beat_samples=100, latent_dim=8, base_channels=4
    )
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    ppg = torch.randn(3, 2, 188)
    ecg = torch.randn(3, 1, 100)
    with torch.no_grad():
        target_latent = autoencoder.encode(ecg)
    mean, log_variance = encoder(ppg)
    reconstruction = autoencoder.decode(mean)
    criterion = CrossModalMorphologyLoss(
        MorphologyAutoencoderLoss(sample_rate_hz=125, r_index=20)
    )
    losses = criterion(
        mean,
        log_variance,
        target_latent,
        reconstruction,
        ecg,
        torch.ones_like(ecg),
    )
    losses["total"].backward()
    assert set(losses) == {
        "total",
        "latent_nll",
        "latent_cosine",
        "reconstruction",
    }
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in autoencoder.parameters())
