from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.main = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)
		self.skip = (
			nn.Identity()
			if in_channels == out_channels
			else nn.Conv2d(in_channels, out_channels, 1, bias=False)
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return F.relu(self.main(inputs) + self.skip(inputs), inplace=True)


class AttentionGate(nn.Module):
	def __init__(self, channels: int) -> None:
		super().__init__()
		self.gate = nn.Sequential(
			nn.Conv2d(channels, channels, 1),
			nn.BatchNorm2d(channels),
			nn.Sigmoid(),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return inputs * self.gate(inputs)


class ResidualAttentionUNet(nn.Module):
	"""Compact residual attention U-Net for spectrogram regression."""

	def __init__(
		self,
		in_channels: int = 1,
		out_channels: int = 1,
		base_channels: int = 32,
	) -> None:
		super().__init__()
		self.encoder1 = ResidualBlock(in_channels, base_channels)
		self.encoder2 = ResidualBlock(base_channels, base_channels * 2)
		self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 4)
		self.attention = AttentionGate(base_channels * 4)
		self.decoder2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
		self.decoder1 = ResidualBlock(base_channels * 2 + base_channels, base_channels)
		self.output = nn.Conv2d(base_channels, out_channels, 1)
		self.pool = nn.MaxPool2d(2)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		encoded1 = self.encoder1(inputs)
		encoded2 = self.encoder2(self.pool(encoded1))
		bottleneck = self.attention(self.bottleneck(self.pool(encoded2)))
		decoded2 = F.interpolate(bottleneck, size=encoded2.shape[-2:], mode="bilinear", align_corners=False)
		decoded2 = self.decoder2(torch.cat((decoded2, encoded2), dim=1))
		decoded1 = F.interpolate(decoded2, size=encoded1.shape[-2:], mode="bilinear", align_corners=False)
		decoded1 = self.decoder1(torch.cat((decoded1, encoded1), dim=1))
		return self.output(decoded1)
