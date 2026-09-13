from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class TimingHeadLoss(nn.Module):
    """Masked focal-BCE plus soft Dice for sparse R-event supervision."""

    def __init__(self, *, focal_gamma: float = 2.0, dice_weight: float = 0.5) -> None:
        super().__init__()
        if focal_gamma < 0 or dice_weight < 0:
            raise ValueError("loss parameters must be non-negative.")
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if logits.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError("logits, target, and valid_mask must share a shape.")
        if logits.ndim != 3 or logits.shape[1] != 1:
            raise ValueError("inputs must have shape [batch, 1, samples].")
        mask = valid_mask.to(logits)
        labels = target.to(logits)
        pointwise = functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        focal_factor = torch.abs(labels - probabilities).pow(self.focal_gamma)
        focal_bce = torch.sum(pointwise * focal_factor * mask) / torch.clamp_min(
            mask.sum(), 1.0
        )
        intersection = torch.sum(probabilities * labels * mask)
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            torch.sum(probabilities * mask) + torch.sum(labels * mask) + 1.0
        )
        total = focal_bce + self.dice_weight * dice
        return {"total": total, "focal_bce": focal_bce, "dice": dice}
