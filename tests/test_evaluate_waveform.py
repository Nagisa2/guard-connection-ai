from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.data.stft import (
    STFTConfig,
    compute_stft,
    oracle_phase_reconstruction,
    oracle_phase_reconstruction_metrics,
    reconstruction_metrics,
)


def test_oracle_phase_reconstruction_exact_match() -> None:
    sampling_rate = 125.0
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Generate 1 Hz sine wave
    original = np.sin(2 * np.pi * 1.0 * t)

    config = STFTConfig(sampling_rate=sampling_rate, nperseg=64, noverlap=32, nfft=64)
    stft_result = compute_stft(original, config)

    # Reconstruction using exact magnitude and phase should yield near-perfect recovery
    reconstructed = oracle_phase_reconstruction(
        stft_result.magnitude,
        stft_result.phase,
        config,
    )
    assert reconstructed.shape[0] >= original.shape[0]

    metrics = oracle_phase_reconstruction_metrics(
        original,
        stft_result.magnitude,
        stft_result.phase,
        config,
    )

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "prd" in metrics
    assert "correlation" in metrics
    assert metrics["correlation"] > 0.99
    assert metrics["mae"] < 0.05
    assert metrics["rmse"] < 0.05
    assert metrics["prd"] < 10.0


def test_oracle_phase_reconstruction_with_noise() -> None:
    sampling_rate = 125.0
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    original = np.sin(2 * np.pi * 1.5 * t)

    config = STFTConfig(sampling_rate=sampling_rate, nperseg=64, noverlap=32, nfft=64)
    stft_result = compute_stft(original, config)

    # Add slight distortion to magnitude
    noisy_magnitude = stft_result.magnitude + 0.1 * np.random.RandomState(42).randn(
        *stft_result.magnitude.shape
    )
    noisy_magnitude = np.clip(noisy_magnitude, a_min=0, a_max=None)

    metrics = oracle_phase_reconstruction_metrics(
        original,
        noisy_magnitude,
        stft_result.phase,
        config,
    )

    assert 0.0 <= metrics["mae"]
    assert 0.0 <= metrics["rmse"]
    assert 0.0 <= metrics["prd"]
    assert -1.0 <= metrics["correlation"] <= 1.0


def test_reconstruction_metrics_shape_mismatch() -> None:
    signal1 = np.ones(100)
    signal2 = np.ones(50)
    with pytest.raises(ValueError, match="same shape"):
        reconstruction_metrics(signal1, signal2)


def test_oracle_phase_reconstruction_invalid_dimensions() -> None:
    mag = np.ones((33, 41))
    phase = np.ones((33, 40))  # Mismatched shape
    with pytest.raises(ValueError, match="same shape"):
        oracle_phase_reconstruction(mag, phase)
