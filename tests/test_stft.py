import numpy as np

from guard_connection_ai.data.config import (
	load_data_config,
	stft_config_from_data_config,
)
from guard_connection_ai.data.stft import (
	STFTConfig,
	compute_stft,
	inverse_stft,
	oracle_phase_reconstruction,
	reconstruction_metrics,
)


def test_data_yaml_produces_baseline_stft_config():
	config = load_data_config("configs/data.yaml")
	stft_config = stft_config_from_data_config(config)

	assert stft_config.sampling_rate == 125
	assert stft_config.nperseg == 64
	assert stft_config.noverlap == 32
	assert stft_config.nfft == 64
	assert stft_config.window == "hann"


def test_complex_stft_round_trip_reconstructs_signal():
	config = STFTConfig()
	time = np.arange(1250) / config.sampling_rate
	signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 8 * time)

	result = compute_stft(signal, config)
	reconstructed = inverse_stft(result, config)[: signal.size]
	metrics = reconstruction_metrics(signal, reconstructed)

	assert result.complex_stft.shape[0] == config.nfft // 2 + 1
	assert metrics["mae"] < 1e-10
	assert metrics["rmse"] < 1e-10
	assert metrics["prd"] < 1e-8
	assert metrics["correlation"] > 0.999999


def test_stft_exposes_magnitude_and_phase_for_future_representations():
	result = compute_stft(np.ones(1250))

	assert result.magnitude.shape == result.complex_stft.shape
	assert result.phase.shape == result.complex_stft.shape
	assert np.allclose(result.complex_stft, result.real + 1j * result.imaginary)


def test_oracle_phase_reconstruction_recovers_signal_with_target_phase():
	config = STFTConfig()
	time = np.arange(1250) / config.sampling_rate
	signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 8 * time)
	result = compute_stft(signal, config)

	reconstructed = oracle_phase_reconstruction(result.magnitude, result.phase, config)[: signal.size]
	metrics = reconstruction_metrics(signal, reconstructed)

	assert metrics["mae"] < 1e-10
	assert metrics["rmse"] < 1e-10
	assert metrics["prd"] < 1e-8
	assert metrics["correlation"] > 0.999999
