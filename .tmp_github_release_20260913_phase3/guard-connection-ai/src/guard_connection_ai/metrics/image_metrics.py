from __future__ import annotations

import numpy as np
import torch


def spectrogram_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Calculate reconstruction metrics over all spectrogram elements."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    prediction_array = prediction.detach().cpu().numpy().astype(np.float64).ravel()
    target_array = target.detach().cpu().numpy().astype(np.float64).ravel()
    difference = prediction_array - target_array
    denominator = np.linalg.norm(target_array)
    if np.std(prediction_array) == 0 or np.std(target_array) == 0:
        correlation = 1.0 if np.array_equal(prediction_array, target_array) else 0.0
    else:
        correlation = float(np.corrcoef(prediction_array, target_array)[0, 1])
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "prd": float(100.0 * np.linalg.norm(difference) / denominator) if denominator else 0.0,
        "correlation": correlation,
    }
