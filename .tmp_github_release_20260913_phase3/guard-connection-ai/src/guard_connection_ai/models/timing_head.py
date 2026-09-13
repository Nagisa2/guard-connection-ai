from __future__ import annotations

import torch
from torch import nn

from guard_connection_ai.models.causal_generator import CausalResidualBlock


class CausalTimingHead(nn.Module):
    """Predict delayed ECG R-event evidence from PPG and its validity mask."""

    def __init__(
        self,
        *,
        input_channels: int = 2,
        hidden_channels: int = 32,
        dilation_cycle: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        stacks: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0 or stacks <= 0:
            raise ValueError("channel counts and stacks must be positive.")
        if (
            kernel_size < 2
            or not dilation_cycle
            or any(value <= 0 for value in dilation_cycle)
        ):
            raise ValueError("invalid kernel or dilation cycle.")
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(
                hidden_channels, dilation=dilation, kernel_size=kernel_size
            )
            for dilation in dilation_cycle * stacks
        )
        self.event_projection = nn.Conv1d(hidden_channels, 1, 1)
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
        return self.event_projection(self.encode(inputs))


class StatefulCausalTimingHead:
    """Chunk adapter whose outputs exactly match a full causal forward pass."""

    def __init__(self, model: CausalTimingHead) -> None:
        self.model = model
        self._history: torch.Tensor | None = None

    def reset(self) -> None:
        self._history = None

    @torch.no_grad()
    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3 or chunk.shape[-1] == 0:
            raise ValueError(
                "chunk must have shape [batch, channels, positive samples]."
            )
        history_samples = self.model.receptive_field_samples - 1
        model_input = (
            chunk if self._history is None else torch.cat((self._history, chunk), -1)
        )
        if self._history is not None and self._history.shape[:2] != chunk.shape[:2]:
            raise ValueError(
                "batch and channel dimensions cannot change without reset()."
            )
        output = self.model(model_input)[..., -chunk.shape[-1] :]
        self._history = model_input[..., -history_samples:].detach()
        return output
