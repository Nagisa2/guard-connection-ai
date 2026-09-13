from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from guard_connection_ai.losses.waveform_generation import WaveformGenerationLoss


class MorphologyAutoencoderLoss(nn.Module):
    """R-aligned ROI loss for a non-diagnostic ECG morphology prior."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        r_index: int,
        fft_sizes: Sequence[int] = (32, 64),
        qrs_weight: float = 4.0,
        t_weight: float = 2.0,
        derivative_weight: float = 0.5,
        spectral_weight: float = 0.25,
        latent_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if sample_rate_hz <= 0 or r_index < 0:
            raise ValueError("sample rate must be positive and r_index non-negative.")
        if (
            min(qrs_weight, t_weight, derivative_weight, spectral_weight, latent_weight)
            < 0
        ):
            raise ValueError("loss weights must be non-negative.")
        self.sample_rate_hz = sample_rate_hz
        self.r_index = r_index
        self.qrs_weight = qrs_weight
        self.t_weight = t_weight
        self.derivative_weight = derivative_weight
        self.spectral_weight = spectral_weight
        self.latent_weight = latent_weight
        self.spectral_loss = WaveformGenerationLoss(
            fft_sizes=fft_sizes,
            waveform_weight=0.0,
            derivative_weight=0.0,
            spectral_weight=1.0,
        )

    def _region_weights(self, samples: int, device: torch.device) -> torch.Tensor:
        weights = torch.ones(samples, device=device)
        qrs_start = max(self.r_index - round(0.08 * self.sample_rate_hz), 0)
        qrs_end = min(self.r_index + round(0.12 * self.sample_rate_hz), samples)
        t_start = qrs_end
        t_end = min(self.r_index + round(0.45 * self.sample_rate_hz), samples)
        weights[qrs_start:qrs_end] = self.qrs_weight
        weights[t_start:t_end] = self.t_weight
        return weights[None, None, :]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError("prediction, target, and valid_mask must share a shape.")
        if target.ndim != 3 or target.shape[1] != 1 or latent.ndim != 2:
            raise ValueError("invalid waveform or latent dimensions.")
        mask = valid_mask.to(prediction)
        region_weights = self._region_weights(target.shape[-1], prediction.device)
        pointwise = functional.smooth_l1_loss(prediction, target, reduction="none")
        weighted_mask = mask * region_weights
        roi = torch.sum(pointwise * weighted_mask) / torch.clamp_min(
            torch.sum(weighted_mask), 1.0
        )
        derivative_mask = mask[..., 1:] * mask[..., :-1]
        derivative_error = functional.smooth_l1_loss(
            torch.diff(prediction, dim=-1),
            torch.diff(target, dim=-1),
            reduction="none",
        )
        derivative = torch.sum(derivative_error * derivative_mask) / torch.clamp_min(
            torch.sum(derivative_mask), 1.0
        )
        spectral = self.spectral_loss(prediction, target, mask)["multi_resolution_stft"]
        latent_regularization = torch.mean(latent * latent)
        total = (
            roi
            + self.derivative_weight * derivative
            + self.spectral_weight * spectral
            + self.latent_weight * latent_regularization
        )
        return {
            "total": total,
            "roi": roi,
            "derivative": derivative,
            "multi_resolution_stft": spectral,
            "latent_regularization": latent_regularization,
        }
