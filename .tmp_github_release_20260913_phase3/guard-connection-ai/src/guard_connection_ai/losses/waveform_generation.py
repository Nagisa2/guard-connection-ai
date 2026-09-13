from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional


class WaveformGenerationLoss(nn.Module):
    """Oracle-phase-free waveform, derivative, and multi-resolution spectral loss."""

    def __init__(
        self,
        *,
        fft_sizes: Sequence[int] = (64, 128, 256),
        waveform_weight: float = 1.0,
        derivative_weight: float = 0.25,
        spectral_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not fft_sizes or any(size < 4 for size in fft_sizes):
            raise ValueError("fft_sizes must contain values of at least four.")
        if min(waveform_weight, derivative_weight, spectral_weight) < 0:
            raise ValueError("loss weights must be non-negative.")
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.waveform_weight = waveform_weight
        self.derivative_weight = derivative_weight
        self.spectral_weight = spectral_weight

    @staticmethod
    def _masked_smooth_l1(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pointwise = functional.smooth_l1_loss(prediction, target, reduction="none")
        return torch.sum(pointwise * mask) / torch.clamp_min(torch.sum(mask), 1.0)

    def _spectral_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        prediction_flat = prediction.squeeze(1)
        target_flat = target.squeeze(1)
        for fft_size in self.fft_sizes:
            if prediction.shape[-1] < fft_size:
                continue
            hop_length = fft_size // 4
            window = torch.hann_window(
                fft_size, device=prediction.device, dtype=prediction.dtype
            )
            prediction_stft = torch.stft(
                prediction_flat,
                n_fft=fft_size,
                hop_length=hop_length,
                window=window,
                center=False,
                return_complex=True,
            )
            target_stft = torch.stft(
                target_flat,
                n_fft=fft_size,
                hop_length=hop_length,
                window=window,
                center=False,
                return_complex=True,
            )
            prediction_magnitude = torch.abs(prediction_stft)
            target_magnitude = torch.abs(target_stft)
            denominator = torch.clamp_min(torch.linalg.vector_norm(target_magnitude), 1e-8)
            convergence = torch.linalg.vector_norm(
                prediction_magnitude - target_magnitude
            ) / denominator
            log_magnitude = torch.mean(
                torch.abs(
                    torch.log1p(prediction_magnitude) - torch.log1p(target_magnitude)
                )
            )
            losses.append(convergence + log_magnitude)
        if not losses:
            return prediction.new_zeros(())
        return torch.mean(torch.stack(losses))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1] != 1:
            raise ValueError("prediction and target must share shape [batch, 1, samples].")
        mask = torch.ones_like(target) if valid_mask is None else valid_mask.to(target)
        if mask.shape != target.shape:
            raise ValueError("valid_mask must have the same shape as target.")
        waveform = self._masked_smooth_l1(prediction, target, mask)
        derivative_mask = mask[..., 1:] * mask[..., :-1]
        derivative = self._masked_smooth_l1(
            torch.diff(prediction, dim=-1),
            torch.diff(target, dim=-1),
            derivative_mask,
        )
        spectral = self._spectral_loss(prediction * mask, target * mask)
        total = (
            self.waveform_weight * waveform
            + self.derivative_weight * derivative
            + self.spectral_weight * spectral
        )
        return {
            "total": total,
            "waveform": waveform,
            "derivative": derivative,
            "multi_resolution_stft": spectral,
        }
