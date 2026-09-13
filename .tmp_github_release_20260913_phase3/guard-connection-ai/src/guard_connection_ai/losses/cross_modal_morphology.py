from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from guard_connection_ai.losses.morphology import MorphologyAutoencoderLoss


class CrossModalMorphologyLoss(nn.Module):
    """Latent uncertainty and frozen-decoder reconstruction loss."""

    def __init__(
        self,
        morphology_loss: MorphologyAutoencoderLoss,
        *,
        latent_nll_weight: float = 1.0,
        cosine_weight: float = 0.2,
        reconstruction_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if min(latent_nll_weight, cosine_weight, reconstruction_weight) < 0:
            raise ValueError("loss weights must be non-negative.")
        self.morphology_loss = morphology_loss
        self.latent_nll_weight = latent_nll_weight
        self.cosine_weight = cosine_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(
        self,
        predicted_mean: torch.Tensor,
        predicted_log_variance: torch.Tensor,
        target_latent: torch.Tensor,
        reconstructed_beat: torch.Tensor,
        target_beat: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not (
            predicted_mean.shape == predicted_log_variance.shape == target_latent.shape
        ):
            raise ValueError("latent tensors must share a shape.")
        squared_error = (predicted_mean - target_latent) ** 2
        latent_nll = 0.5 * torch.mean(
            torch.exp(-predicted_log_variance) * squared_error + predicted_log_variance
        )
        cosine = torch.mean(
            1.0 - functional.cosine_similarity(predicted_mean, target_latent, dim=-1)
        )
        morphology = self.morphology_loss(
            reconstructed_beat,
            target_beat,
            valid_mask,
            predicted_mean,
        )
        total = (
            self.latent_nll_weight * latent_nll
            + self.cosine_weight * cosine
            + self.reconstruction_weight * morphology["total"]
        )
        return {
            "total": total,
            "latent_nll": latent_nll,
            "latent_cosine": cosine,
            "reconstruction": morphology["total"],
        }
