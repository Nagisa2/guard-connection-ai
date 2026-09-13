from __future__ import annotations

import numpy as np
import torch

from guard_connection_ai.data.af_dataset import (
    delay_waveform_target,
    peaks_to_beat_envelope,
)
from guard_connection_ai.data.causal_preprocessing import (
    FixedRobustScaler,
    StatefulCausalPPGPreprocessor,
    StatefulCausalWaveformPreprocessor,
    fit_fixed_robust_scaler,
)
from guard_connection_ai.losses.waveform_generation import WaveformGenerationLoss
from guard_connection_ai.models.causal_generator import (
    CausalTCNGenerator,
    StatefulCausalGenerator,
)


def _model() -> CausalTCNGenerator:
    torch.manual_seed(7)
    model = CausalTCNGenerator(
        hidden_channels=8,
        dilation_cycle=(1, 2, 4),
        stacks=1,
    )
    model.eval()
    return model


def test_generator_preserves_length_and_is_strictly_causal() -> None:
    model = _model()
    inputs = torch.randn(2, 2, 200)
    changed_future = inputs.clone()
    changed_future[..., 120:] = torch.randn_like(changed_future[..., 120:])

    with torch.no_grad():
        output = model(inputs)
        changed_output = model(changed_future)

    assert output.shape == (2, 1, 200)
    torch.testing.assert_close(output[..., :120], changed_output[..., :120])


def test_stateful_chunks_equal_single_pass() -> None:
    model = _model()
    inputs = torch.randn(1, 2, 257)
    stateful = StatefulCausalGenerator(model)

    with torch.no_grad():
        expected = model(inputs)
        actual = torch.cat(
            (
                stateful.process(inputs[..., :31]),
                stateful.process(inputs[..., 31:96]),
                stateful.process(inputs[..., 96:193]),
                stateful.process(inputs[..., 193:]),
            ),
            dim=-1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_state_reset_starts_a_new_stream() -> None:
    model = _model()
    chunk = torch.randn(1, 2, 80)
    stateful = StatefulCausalGenerator(model)

    first = stateful.process(chunk)
    stateful.process(torch.randn(1, 2, 20))
    stateful.reset()
    repeated = stateful.process(chunk)

    torch.testing.assert_close(first, repeated)


def test_waveform_loss_is_finite_and_differentiable_without_phase_input() -> None:
    prediction = torch.randn(2, 1, 512, requires_grad=True)
    target = torch.randn(2, 1, 512)
    mask = torch.ones_like(target)
    mask[..., :25] = 0
    criterion = WaveformGenerationLoss(fft_sizes=(64, 128))

    losses = criterion(prediction, target, mask)
    losses["total"].backward()

    assert set(losses) == {"total", "waveform", "derivative", "multi_resolution_stft"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert prediction.grad is not None


def test_causal_preprocessor_chunks_equal_single_pass_and_marks_missing() -> None:
    sample_rate = 125
    time = np.arange(500) / sample_rate
    ppg = 2.0 + 0.4 * np.sin(2 * np.pi * 1.2 * time)
    ppg[173:180] = np.nan
    scaler = fit_fixed_robust_scaler([ppg[:150], ppg[200:]])
    single = StatefulCausalPPGPreprocessor(sample_rate_hz=sample_rate, scaler=scaler)
    chunked = StatefulCausalPPGPreprocessor(sample_rate_hz=sample_rate, scaler=scaler)

    expected = single.process(ppg)
    actual = np.concatenate(
        (
            chunked.process(ppg[:83]),
            chunked.process(ppg[83:211]),
            chunked.process(ppg[211:]),
        ),
        axis=-1,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert np.all(actual[1, 173:180] == 0)
    assert np.all(np.isfinite(actual[0]))


def test_fixed_scaler_rejects_constant_training_signal() -> None:
    with np.testing.assert_raises(ValueError):
        fit_fixed_robust_scaler([np.ones(100)])

    scaler = FixedRobustScaler(center=0.0, scale=1.0)
    assert scaler.scale == 1.0


def test_ecg_preprocessor_is_causal() -> None:
    sample_rate = 125
    values = np.random.default_rng(3).normal(size=500)
    changed = values.copy()
    changed[300:] += 10
    scaler = FixedRobustScaler(center=0.0, scale=1.0)
    first = StatefulCausalWaveformPreprocessor(
        sample_rate_hz=sample_rate, scaler=scaler, modality="ecg"
    ).process(values)
    second = StatefulCausalWaveformPreprocessor(
        sample_rate_hz=sample_rate, scaler=scaler, modality="ecg"
    ).process(changed)

    np.testing.assert_allclose(first[:, :300], second[:, :300])


def test_delayed_target_definition_moves_ecg_into_the_causal_future() -> None:
    waveform = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    valid = np.ones_like(waveform)
    delayed, delayed_valid = delay_waveform_target(waveform, valid, 2)

    assert delayed.tolist() == [[0.0, 0.0, 1.0, 2.0]]
    assert delayed_valid.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_beat_envelope_marks_reference_peaks_without_exceeding_unit_range() -> None:
    envelope = peaks_to_beat_envelope(
        np.asarray([5, 15]), sample_count=20, sample_rate_hz=100, half_width_seconds=0.02
    )

    assert envelope.shape == (20,)
    assert envelope[5] == 1.0
    assert envelope[15] == 1.0
    assert np.all((0 <= envelope) & (envelope <= 1))

