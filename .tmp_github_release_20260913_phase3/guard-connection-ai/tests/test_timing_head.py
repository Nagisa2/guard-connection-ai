from __future__ import annotations

import numpy as np
import torch

from guard_connection_ai.data.causal_preprocessing import (
    FixedRobustScaler,
    StatefulCausalWaveformPreprocessor,
)
from guard_connection_ai.data.timing_targets import (
    build_timing_targets,
    detect_ppg_feet,
)
from guard_connection_ai.losses.timing import TimingHeadLoss
from guard_connection_ai.metrics.timing import (
    calibrate_event_threshold,
    evaluate_event_timing,
    evaluate_probability_windows,
)
from guard_connection_ai.models.timing_head import (
    CausalTimingHead,
    StatefulCausalTimingHead,
)
from guard_connection_ai.streaming.learned_pseudo_ecg import (
    StatefulLearnedTimingPseudoECG,
)
from guard_connection_ai.streaming.timing_decoder import (
    StatefulTimingDecoder,
    TimingEvent,
)
from guard_connection_ai.visualization.timing_template_renderer import (
    StatefulTimingTemplateRenderer,
)


def _model() -> CausalTimingHead:
    torch.manual_seed(3)
    model = CausalTimingHead(hidden_channels=8, dilation_cycle=(1, 2, 4), stacks=1)
    model.eval()
    return model


def test_timing_head_is_causal_and_chunk_equivalent() -> None:
    model = _model()
    values = torch.randn(1, 2, 211)
    changed = values.clone()
    changed[..., 130:] += 5
    with torch.no_grad():
        expected = model(values)
        changed_output = model(changed)
    torch.testing.assert_close(expected[..., :130], changed_output[..., :130])

    streaming = StatefulCausalTimingHead(model)
    actual = torch.cat(
        [
            streaming.process(values[..., :37]),
            streaming.process(values[..., 37:101]),
            streaming.process(values[..., 101:]),
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_timing_targets_use_delayed_r_peaks_and_find_pulse_feet() -> None:
    ppg = np.asarray([0, 0, 1, 3, 2, 1, 0, 1, 4, 2, 0, 0], dtype=float)
    feet = detect_ppg_feet(
        ppg, np.asarray([3, 8]), sample_rate_hz=10, maximum_lookback_seconds=0.3
    )
    assert feet.tolist() == [0, 6]
    targets = build_timing_targets(
        ppg=np.tile(ppg, 10),
        ppg_peaks=np.asarray([30, 80]),
        ecg_r_peaks=np.asarray([27, 77]),
        sample_rate_hz=10,
        output_delay_seconds=0.5,
    )
    assert targets.delayed_ecg_r_peaks.tolist() == [32, 82]
    assert targets.beat_envelope[32] == 1
    assert not np.any(targets.valid_mask[:5])


def test_timing_loss_is_finite_and_differentiable() -> None:
    logits = torch.randn(2, 1, 100, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 20] = 1
    target[..., 70] = 1
    losses = TimingHeadLoss()(logits, target, torch.ones_like(target))
    losses["total"].backward()
    assert set(losses) == {"total", "focal_bce", "dice"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert logits.grad is not None


def test_decoder_and_renderer_do_not_duplicate_events_across_chunks() -> None:
    probabilities = np.zeros(20)
    probabilities[[4, 12]] = 0.9
    decoder = StatefulTimingDecoder(sample_rate_hz=10, refractory_seconds=0.3)
    events = decoder.process(probabilities[:5], np.ones(5, dtype=bool))
    events += decoder.process(probabilities[5:13], np.ones(8, dtype=bool))
    events += decoder.process(probabilities[13:], np.ones(7, dtype=bool))
    assert [event.sample_index for event in events] == [4, 12]

    renderer = StatefulTimingTemplateRenderer(
        sample_rate_hz=10, template=np.asarray([1, 0.5])
    )
    first, _ = renderer.process(
        5, [TimingEvent(4, 0.9)], valid_mask=np.ones(5, dtype=bool)
    )
    second, _ = renderer.process(
        8, [TimingEvent(12, 0.9)], valid_mask=np.ones(8, dtype=bool)
    )
    third, _ = renderer.process(7, [], valid_mask=np.ones(7, dtype=bool))
    combined = np.concatenate([first, second, third])
    assert np.flatnonzero(combined == 1).tolist() == [4, 12]
    assert combined[5] == 0.5
    assert combined[13] == 0.5


def test_renderer_blanks_invalid_samples_and_morphology_band_is_opt_in() -> None:
    renderer = StatefulTimingTemplateRenderer(sample_rate_hz=10, template=np.ones(3))
    output, valid = renderer.process(
        5,
        [TimingEvent(1, 0.8)],
        valid_mask=np.asarray([True, True, False, True, True]),
    )
    assert output.tolist() == [0, 1, 0, 1, 0]
    assert valid.tolist() == [True, True, False, True, True]

    processor = StatefulCausalWaveformPreprocessor(
        sample_rate_hz=125,
        scaler=FixedRobustScaler(0, 1),
        modality="ecg",
        bandpass_hz=(0.5, 35.0),
    )
    assert processor.bandpass_hz == (0.5, 35.0)


def test_event_metrics_prevent_duplicate_matches() -> None:
    metrics = evaluate_event_timing(
        np.asarray([100, 200, 300]),
        np.asarray([98, 102, 202]),
        sample_rate_hz=100,
        tolerance_seconds=0.05,
    )
    assert metrics.matched_events == 2
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3
    assert metrics.timing_mae_seconds == 0.02


def test_learned_pipeline_has_session_local_state_and_reset() -> None:
    model = _model()
    first = StatefulLearnedTimingPseudoECG(
        model, sample_rate_hz=100, output_delay_seconds=0.1, event_threshold=0.5
    )
    second = StatefulLearnedTimingPseudoECG(
        model, sample_rate_hz=100, output_delay_seconds=0.1, event_threshold=0.5
    )
    values = np.stack([np.sin(np.arange(40) / 5), np.ones(40)]).astype(np.float32)
    first_output = first.process(values, render_enabled=True)
    second_output = second.process(values, render_enabled=True)
    np.testing.assert_allclose(
        first_output.event_probability, second_output.event_probability
    )
    assert not np.any(first_output.pseudo_ecg_valid[:10])

    first.process(values, render_enabled=False)
    first.reset()
    repeated = first.process(values, render_enabled=True)
    np.testing.assert_allclose(
        first_output.event_probability, repeated.event_probability
    )
    np.testing.assert_array_equal(
        first_output.pseudo_ecg_valid, repeated.pseudo_ecg_valid
    )


def test_threshold_calibration_uses_runtime_decoder() -> None:
    target = np.zeros(40, dtype=np.float32)
    target[[10, 30]] = 1
    probability = np.zeros(40, dtype=np.float32)
    probability[[10, 30]] = 0.65
    probability[20] = 0.35
    valid = np.ones(40, dtype=bool)
    calibrated = calibrate_event_threshold(
        [probability],
        [target],
        [valid],
        sample_rate_hz=20,
        thresholds=(0.3, 0.6, 0.8),
    )
    assert calibrated.threshold == 0.6
    assert calibrated.metrics.f1 == 1
    evaluated = evaluate_probability_windows(
        [probability],
        [target],
        [valid],
        sample_rate_hz=20,
        threshold=calibrated.threshold,
    )
    assert evaluated.matched_events == 2
