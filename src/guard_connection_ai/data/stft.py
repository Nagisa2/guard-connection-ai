from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft


@dataclass(frozen=True)
class STFTConfig:
	"""Parameters shared by PPG and ECG STFT transformations."""

	sampling_rate: float = 125.0
	nperseg: int = 64
	noverlap: int = 32
	nfft: int = 64
	window: str = "hann"


@dataclass(frozen=True)
class STFTResult:
	frequencies: np.ndarray
	times: np.ndarray
	complex_stft: np.ndarray

	@property
	def magnitude(self) -> np.ndarray:
		return np.abs(self.complex_stft)

	@property
	def phase(self) -> np.ndarray:
		return np.angle(self.complex_stft)

	@property
	def real(self) -> np.ndarray:
		return np.real(self.complex_stft)

	@property
	def imaginary(self) -> np.ndarray:
		return np.imag(self.complex_stft)


def compute_stft(
	signal: np.ndarray,
	config: STFTConfig | None = None,
) -> STFTResult:
	"""Compute a complex STFT with reconstruction-friendly edge handling."""

	if config is None:
		config = STFTConfig()
	signal_array = np.asarray(signal, dtype=np.float64)
	if signal_array.ndim != 1:
		raise ValueError("signal must be one-dimensional.")
	if config.noverlap >= config.nperseg:
		raise ValueError("noverlap must be smaller than nperseg.")
	if config.nfft < config.nperseg:
		raise ValueError("nfft must be at least nperseg.")

	frequencies, times, complex_stft = stft(
		signal_array,
		fs=config.sampling_rate,
		window=config.window,
		nperseg=config.nperseg,
		noverlap=config.noverlap,
		nfft=config.nfft,
		boundary="zeros",
		padded=True,
	)
	return STFTResult(frequencies, times, complex_stft)


def inverse_stft(
	result: STFTResult,
	config: STFTConfig | None = None,
) -> np.ndarray:
	"""Reconstruct a waveform from the complex STFT representation."""

	if config is None:
		config = STFTConfig()
	_, signal = istft(
		result.complex_stft,
		fs=config.sampling_rate,
		window=config.window,
		nperseg=config.nperseg,
		noverlap=config.noverlap,
		nfft=config.nfft,
		input_onesided=True,
		boundary=True,
	)
	return signal


def reconstruction_metrics(
	original: np.ndarray,
	reconstructed: np.ndarray,
) -> dict[str, float]:
	"""Return MAE, RMSE, PRD, and Pearson correlation for two waveforms."""

	original_array = np.asarray(original, dtype=np.float64)
	reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
	if original_array.shape != reconstructed_array.shape:
		raise ValueError("original and reconstructed must have the same shape.")

	difference = reconstructed_array - original_array
	denominator = np.linalg.norm(original_array)
	prd = float(100.0 * np.linalg.norm(difference) / denominator) if denominator else 0.0
	correlation = float(np.corrcoef(original_array, reconstructed_array)[0, 1])
	return {
		"mae": float(np.mean(np.abs(difference))),
		"rmse": float(np.sqrt(np.mean(difference**2))),
		"prd": prd,
		"correlation": correlation,
	}


def oracle_phase_reconstruction(
	magnitude: np.ndarray,
	phase: np.ndarray,
	config: STFTConfig | None = None,
) -> np.ndarray:
	"""Reconstruct a waveform using a predicted magnitude and an oracle phase."""
	return phase_transfer_reconstruction(magnitude, phase, config)


def phase_transfer_reconstruction(
	magnitude: np.ndarray,
	phase: np.ndarray,
	config: STFTConfig | None = None,
) -> np.ndarray:
	"""Reconstruct a waveform by combining a magnitude spectrogram with a given phase spectrogram."""
	if config is None:
		config = STFTConfig()
	magnitude_array = np.asarray(magnitude, dtype=np.float64)
	phase_array = np.asarray(phase, dtype=np.float64)
	if magnitude_array.shape != phase_array.shape:
		raise ValueError("magnitude and phase must have the same shape.")
	if magnitude_array.ndim != 2:
		raise ValueError("magnitude and phase must be 2D STFT arrays.")

	complex_stft = magnitude_array * np.exp(1j * phase_array)
	return inverse_stft(STFTResult(np.empty(0), np.empty(0), complex_stft), config)


def zero_phase_reconstruction(
	magnitude: np.ndarray,
	config: STFTConfig | None = None,
) -> np.ndarray:
	"""Reconstruct a waveform using magnitude and zero phase (phase = 0)."""
	magnitude_array = np.asarray(magnitude, dtype=np.float64)
	zero_phase = np.zeros_like(magnitude_array)
	return phase_transfer_reconstruction(magnitude_array, zero_phase, config)


def griffin_lim_reconstruction(
	magnitude: np.ndarray,
	config: STFTConfig | None = None,
	n_iter: int = 32,
	init_phase: np.ndarray | None = None,
) -> np.ndarray:
	"""Reconstruct a waveform from magnitude spectrogram using Griffin-Lim iterative phase estimation."""
	if config is None:
		config = STFTConfig()
	magnitude_array = np.asarray(magnitude, dtype=np.float64)
	if magnitude_array.ndim != 2:
		raise ValueError("magnitude must be a 2D STFT array.")
	if n_iter < 1:
		raise ValueError("n_iter must be a positive integer.")

	if init_phase is not None:
		phase_array = np.asarray(init_phase, dtype=np.float64)
		if phase_array.shape != magnitude_array.shape:
			raise ValueError("init_phase must have the same shape as magnitude.")
		complex_stft = magnitude_array * np.exp(1j * phase_array)
	else:
		complex_stft = magnitude_array.astype(np.complex128)

	for _ in range(n_iter):
		reconstructed_signal = inverse_stft(
			STFTResult(np.empty(0), np.empty(0), complex_stft), config
		)
		stft_res = compute_stft(reconstructed_signal, config)
		estimated_phase = stft_res.phase
		# Align dimensions if padding introduced extra columns
		if estimated_phase.shape != magnitude_array.shape:
			estimated_phase = estimated_phase[
				: magnitude_array.shape[0], : magnitude_array.shape[1]
			]
		complex_stft = magnitude_array * np.exp(1j * estimated_phase)

	final_signal = inverse_stft(STFTResult(np.empty(0), np.empty(0), complex_stft), config)
	return final_signal


def oracle_phase_reconstruction_metrics(
	original: np.ndarray,
	magnitude: np.ndarray,
	phase: np.ndarray,
	config: STFTConfig | None = None,
) -> dict[str, float]:
	"""Return waveform metrics for oracle-phase reconstruction against the original signal."""
	reconstructed = oracle_phase_reconstruction(magnitude, phase, config)
	original_array = np.asarray(original, dtype=np.float64)
	if reconstructed.size < original_array.size:
		raise ValueError("Reconstructed waveform is shorter than the original signal.")
	return reconstruction_metrics(original_array, reconstructed[: original_array.size])


def waveform_reconstruction_metrics(
	original: np.ndarray,
	reconstructed: np.ndarray,
) -> dict[str, float]:
	"""Return waveform metrics between original and reconstructed waveforms, truncating to original length."""
	original_array = np.asarray(original, dtype=np.float64)
	reconstructed_array = np.asarray(reconstructed, dtype=np.float64)
	if reconstructed_array.size < original_array.size:
		raise ValueError("Reconstructed waveform is shorter than the original signal.")
	return reconstruction_metrics(original_array, reconstructed_array[: original_array.size])

