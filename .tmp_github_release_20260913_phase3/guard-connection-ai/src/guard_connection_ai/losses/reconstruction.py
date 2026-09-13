from __future__ import annotations

import torch
from torch import nn


def l1_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate pixel-wise L1 loss for spectrogram regression."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    return nn.functional.l1_loss(prediction, target)


def ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Calculate differentiable local SSIM loss for spectrogram tensors."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [batch, channels, height, width].")
    if window_size <= 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer greater than one.")
    if data_range <= 0:
        raise ValueError("data_range must be positive.")

    padding = window_size // 2
    mean_prediction = nn.functional.avg_pool2d(
        prediction, window_size, stride=1, padding=padding, count_include_pad=False
    )
    mean_target = nn.functional.avg_pool2d(
        target, window_size, stride=1, padding=padding, count_include_pad=False
    )
    mean_prediction_squared = mean_prediction.square()
    mean_target_squared = mean_target.square()
    covariance = nn.functional.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding, count_include_pad=False
    ) - mean_prediction * mean_target
    variance_prediction = nn.functional.avg_pool2d(
        prediction.square(), window_size, stride=1, padding=padding, count_include_pad=False
    ) - mean_prediction_squared
    variance_target = nn.functional.avg_pool2d(
        target.square(), window_size, stride=1, padding=padding, count_include_pad=False
    ) - mean_target_squared

    constant_1 = (0.01 * data_range) ** 2
    constant_2 = (0.03 * data_range) ** 2
    ssim = (
        (2 * mean_prediction * mean_target + constant_1)
        * (2 * covariance + constant_2)
        / ((mean_prediction_squared + mean_target_squared + constant_1)
           * (variance_prediction + variance_target + constant_2))
    )
    return 1.0 - ssim.mean()


def combined_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.0,
    frequency_weight: float = 0.0,
) -> torch.Tensor:
    """Combine L1, SSIM, and 2D frequency losses."""

    if l1_weight < 0 or ssim_weight < 0 or frequency_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    loss = l1_weight * l1_reconstruction_loss(prediction, target)
    loss = loss + ssim_weight * ssim_loss(prediction, target)
    if frequency_weight:
        loss = loss + frequency_weight * frequency_domain_loss(prediction, target)
    return loss


def frequency_domain_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare 2D FFT magnitudes of spectrogram predictions and targets."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must have shape [batch, channels, height, width].")
    prediction_frequency = torch.fft.rfft2(prediction, dim=(-2, -1)).abs()
    target_frequency = torch.fft.rfft2(target, dim=(-2, -1)).abs()
    return nn.functional.l1_loss(prediction_frequency, target_frequency)
