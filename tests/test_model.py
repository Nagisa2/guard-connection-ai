import torch
import yaml

from guard_connection_ai.losses.reconstruction import (
	combined_reconstruction_loss,
	frequency_domain_loss,
	l1_reconstruction_loss,
	ssim_loss,
)
from guard_connection_ai.metrics.image_metrics import spectrogram_metrics
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet


def test_residual_attention_unet_preserves_stft_shape():
	model = ResidualAttentionUNet(base_channels=8)
	inputs = torch.randn(2, 1, 33, 41)

	outputs = model(inputs)

	assert outputs.shape == inputs.shape


def test_resunet_config_matches_baseline_stft_shape():
	with open("configs/resunet.yaml", encoding="utf-8") as stream:
		config = yaml.safe_load(stream)

	assert config["stft"] == {"n_fft": 64, "hop_length": 32, "win_length": 64}


def test_l1_reconstruction_loss_returns_scalar():
	prediction = torch.zeros(2, 1, 3, 4)
	target = torch.ones(2, 1, 3, 4)

	assert l1_reconstruction_loss(prediction, target).item() == 1.0


def test_spectrogram_metrics_report_perfect_reconstruction():
	target = torch.ones(2, 1, 3, 4)

	metrics = spectrogram_metrics(target, target)

	assert metrics == {"mae": 0.0, "rmse": 0.0, "prd": 0.0, "correlation": 1.0}


def test_ssim_and_combined_loss_are_zero_for_identical_inputs():
	target = torch.rand(2, 1, 12, 12)

	assert abs(ssim_loss(target, target).item()) < 1e-6
	assert abs(combined_reconstruction_loss(target, target, ssim_weight=1.0).item()) < 1e-6


def test_ssim_loss_supports_gradient():
	prediction = torch.rand(1, 1, 12, 12, requires_grad=True)
	target = torch.rand(1, 1, 12, 12)

	ssim_loss(prediction, target).backward()

	assert prediction.grad is not None


def test_frequency_domain_loss_is_zero_for_identical_inputs():
	target = torch.rand(1, 1, 12, 12)

	assert frequency_domain_loss(target, target).item() == 0.0
