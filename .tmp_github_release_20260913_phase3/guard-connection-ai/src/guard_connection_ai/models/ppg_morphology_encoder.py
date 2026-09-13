from __future__ import annotations

import torch
from torch import nn


class PPGMorphologyEncoder(nn.Module):
    """Estimate an ECG-prior latent distribution from past-only PPG context."""

    def __init__(
        self,
        *,
        input_channels: int = 2,
        latent_dim: int = 16,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or latent_dim <= 0 or base_channels <= 0:
            raise ValueError("channel and latent dimensions must be positive.")
        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels, base_channels, kernel_size=9, stride=2, padding=4
            ),
            nn.GELU(),
            nn.Conv1d(
                base_channels, base_channels * 2, kernel_size=7, stride=2, padding=3
            ),
            nn.GELU(),
            nn.Conv1d(
                base_channels * 2,
                base_channels * 2,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mean_projection = nn.Linear(base_channels * 2, latent_dim)
        nn.init.zeros_(self.mean_projection.weight)
        nn.init.zeros_(self.mean_projection.bias)
        self.residual_gate_projection = nn.Linear(base_channels * 2, 1)
        nn.init.zeros_(self.residual_gate_projection.weight)
        nn.init.constant_(self.residual_gate_projection.bias, -2.0)
        self.log_variance_projection = nn.Linear(base_channels * 2, latent_dim)
        self.register_buffer("population_latent", torch.zeros(latent_dim))

    def set_population_latent(self, latent: torch.Tensor) -> None:
        if latent.shape != self.population_latent.shape:
            raise ValueError("population latent has an unexpected shape.")
        self.population_latent.copy_(latent.detach().to(self.population_latent))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [batch, 2, context_samples].")
        hidden = self.features(inputs).squeeze(-1)
        residual = self.mean_projection(hidden)
        residual_gate = torch.sigmoid(self.residual_gate_projection(hidden))
        mean = self.population_latent[None, :] + residual_gate * residual
        log_variance = torch.clamp(self.log_variance_projection(hidden), -6.0, 3.0)
        return mean, log_variance


def morphology_confidence(log_variance: torch.Tensor) -> torch.Tensor:
    """Map predicted latent variance to a bounded non-clinical confidence score."""

    if log_variance.ndim != 2:
        raise ValueError("log_variance must have shape [batch, latent_dim].")
    return torch.exp(-0.5 * torch.mean(torch.exp(log_variance), dim=-1)).clamp(0.0, 1.0)
