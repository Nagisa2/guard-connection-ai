from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each time point without future access."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normalization(inputs.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    """A left-padded convolution whose output at t only depends on inputs <= t."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(functional.pad(inputs, (self.left_padding, 0)))


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.first = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation, bias=False
        )
        self.first_norm = ChannelLayerNorm(channels)
        self.second = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation, bias=False
        )
        self.second_norm = ChannelLayerNorm(channels)
        self.activation = nn.GELU()
        self.receptive_field_increment = 2 * dilation * (kernel_size - 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.first_norm(self.first(inputs)))
        hidden = self.second_norm(self.second(hidden))
        return self.activation(inputs + hidden)


class CausalTCNGenerator(nn.Module):
    """Deterministic PPG-to-standardized-ECG waveform baseline.

    The second input channel is an observed-sample mask. The architecture uses no
    target ECG phase, bidirectional filtering, symmetric padding, or time-axis
    normalization.
    """

    def __init__(
        self,
        *,
        input_channels: int = 2,
        hidden_channels: int = 48,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        stacks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0 or stacks <= 0:
            raise ValueError("channel counts and stacks must be positive.")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least two.")
        if not dilation_cycle or any(dilation <= 0 for dilation in dilation_cycle):
            raise ValueError("dilation_cycle must contain positive integers.")
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        dilations = dilation_cycle * stacks
        self.blocks = nn.ModuleList(
            CausalResidualBlock(
                hidden_channels,
                dilation=dilation,
                kernel_size=kernel_size,
            )
            for dilation in dilations
        )
        self.output_projection = nn.Conv1d(hidden_channels, 1, 1)
        self.receptive_field_samples = 1 + sum(
            block.receptive_field_increment for block in self.blocks
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, channels, samples].")
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.encode(inputs))


class StatefulCausalGenerator:
    """Chunk adapter with exact full-sequence semantics and bounded input history."""

    def __init__(self, model: CausalTCNGenerator) -> None:
        self.model = model
        self._history: torch.Tensor | None = None

    def reset(self) -> None:
        self._history = None

    @torch.no_grad()
    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3 or chunk.shape[-1] == 0:
            raise ValueError("chunk must have shape [batch, channels, positive samples].")
        history_samples = self.model.receptive_field_samples - 1
        if self._history is not None:
            if self._history.shape[:2] != chunk.shape[:2]:
                raise ValueError("batch and channel dimensions cannot change without reset().")
            model_input = torch.cat((self._history, chunk), dim=-1)
        else:
            model_input = chunk
        waveform = self.model(model_input)
        output = waveform[..., -chunk.shape[-1] :]
        self._history = model_input[..., -history_samples:].detach()
        return output
