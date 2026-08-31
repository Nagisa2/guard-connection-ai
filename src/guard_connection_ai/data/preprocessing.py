from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

PreprocessingMode = Literal["none", "detrend", "bandpass", "segment_zscore"]


def preprocess_signal(
    signal: np.ndarray,
    mode: PreprocessingMode = "none",
    *,
    sampling_rate: float = 125.0,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 20.0,
    filter_order: int = 4,
) -> np.ndarray:
    """Apply one explicit preprocessing mode to a single signal segment."""

    signal_array = np.asarray(signal, dtype=np.float64)
    if signal_array.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if mode == "none":
        return signal_array.copy()
    if mode == "detrend":
        return detrend(signal_array)
    if mode == "segment_zscore":
        standard_deviation = np.std(signal_array)
        if standard_deviation == 0:
            return np.zeros_like(signal_array)
        return (signal_array - np.mean(signal_array)) / standard_deviation
    if mode == "bandpass":
        if not 0 < lowcut_hz < highcut_hz < sampling_rate / 2:
            raise ValueError("Bandpass frequencies must satisfy 0 < lowcut < highcut < Nyquist.")
        if filter_order <= 0:
            raise ValueError("filter_order must be positive.")
        filter_coefficients = butter(
            filter_order,
            [lowcut_hz, highcut_hz],
            btype="bandpass",
            fs=sampling_rate,
            output="sos",
        )
        return sosfiltfilt(filter_coefficients, signal_array)

    raise ValueError(f"Unsupported preprocessing mode: {mode}")
