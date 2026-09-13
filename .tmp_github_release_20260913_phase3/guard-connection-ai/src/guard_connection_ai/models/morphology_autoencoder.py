from __future__ import annotations

import torch
from torch import nn


class MorphologyAutoencoder1D(nn.Module):
    """Fixed-length ECG-beat autoencoder used only as a morphology prior."""

    def __init__(
        self,
        *,
        beat_samples: int,
        latent_dim: int = 16,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if beat_samples <= 0 or beat_samples % 4 != 0:
            raise ValueError("beat_samples must be positive and divisible by four.")
        if latent_dim <= 0 or base_channels <= 0:
            raise ValueError("latent_dim and base_channels must be positive.")
        self.beat_samples = beat_samples
        self.encoded_samples = beat_samples // 4
        hidden_channels = base_channels * 2
        self.encoder = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(
                base_channels, hidden_channels, kernel_size=5, stride=2, padding=2
            ),
            nn.GELU(),
        )
        self.to_latent = nn.Linear(hidden_channels * self.encoded_samples, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_channels * self.encoded_samples)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                hidden_channels, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose1d(base_channels, 1, kernel_size=4, stride=2, padding=1),
        )
        self.hidden_channels = hidden_channels

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1:] != (1, self.beat_samples):
            raise ValueError("waveform must have shape [batch, 1, beat_samples].")
        hidden = self.encoder(waveform)
        return self.to_latent(hidden.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must have shape [batch, latent_dim].")
        hidden = self.from_latent(latent).reshape(
            latent.shape[0], self.hidden_channels, self.encoded_samples
        )
        return self.decoder(hidden)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(waveform)
        return self.decode(latent), latent
