from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.data.stft import (
    STFTConfig,
    compute_stft,
    griffin_lim_reconstruction,
    phase_transfer_reconstruction,
    waveform_reconstruction_metrics,
    zero_phase_reconstruction,
)


def test_oracle_phase_perfect_reconstruction() -> None:
    time = np.linspace(0, 10, 1250, endpoint=False)
    signal = np.sin(2 * np.pi * 1.5 * time) + 0.5 * np.cos(2 * np.pi * 3.0 * time)
    config = STFTConfig(sampling_rate=125.0, nperseg=64, noverlap=32, nfft=64)

    stft_res = compute_stft(signal, config)
    reconstructed = phase_transfer_reconstruction(
        stft_res.magnitude, stft_res.phase, config
    )

    metrics = waveform_reconstruction_metrics(signal, reconstructed)
    assert metrics["correlation"] > 0.99
    assert metrics["mae"] < 1e-10


def test_griffin_lim_reconstruction_output_shape_and_convergence() -> None:
    time = np.linspace(0, 10, 1250, endpoint=False)
    signal = np.sin(2 * np.pi * 1.5 * time) + 0.5 * np.cos(2 * np.pi * 3.0 * time)
    config = STFTConfig(sampling_rate=125.0, nperseg=64, noverlap=32, nfft=64)

    stft_res = compute_stft(signal, config)
    reconstructed = griffin_lim_reconstruction(stft_res.magnitude, config, n_iter=16)

    assert reconstructed.size >= signal.size
    metrics = waveform_reconstruction_metrics(signal, reconstructed)
    assert "correlation" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "prd" in metrics


def test_griffin_lim_with_init_phase() -> None:
    time = np.linspace(0, 10, 1250, endpoint=False)
    signal = np.sin(2 * np.pi * 2.0 * time)
    config = STFTConfig(sampling_rate=125.0, nperseg=64, noverlap=32, nfft=64)

    stft_res = compute_stft(signal, config)
    reconstructed = griffin_lim_reconstruction(
        stft_res.magnitude, config, n_iter=16, init_phase=stft_res.phase
    )

    metrics = waveform_reconstruction_metrics(signal, reconstructed)
    assert metrics["correlation"] > 0.95


def test_zero_phase_reconstruction() -> None:
    time = np.linspace(0, 10, 1250, endpoint=False)
    signal = np.sin(2 * np.pi * 2.0 * time)
    config = STFTConfig(sampling_rate=125.0, nperseg=64, noverlap=32, nfft=64)

    stft_res = compute_stft(signal, config)
    reconstructed = zero_phase_reconstruction(stft_res.magnitude, config)

    assert reconstructed.size >= signal.size
    metrics = waveform_reconstruction_metrics(signal, reconstructed)
    assert "correlation" in metrics
    assert "mae" in metrics


def test_phase_transfer_error_handling() -> None:
    config = STFTConfig()
    mag = np.ones((33, 41))
    wrong_phase = np.ones((33, 20))
    with pytest.raises(ValueError, match="same shape"):
        phase_transfer_reconstruction(mag, wrong_phase, config)


def test_griffin_lim_error_handling() -> None:
    config = STFTConfig()
    mag = np.ones((33, 41))
    with pytest.raises(ValueError, match="positive integer"):
        griffin_lim_reconstruction(mag, config, n_iter=0)

    wrong_init_phase = np.ones((33, 20))
    with pytest.raises(ValueError, match="same shape"):
        griffin_lim_reconstruction(mag, config, n_iter=5, init_phase=wrong_init_phase)
