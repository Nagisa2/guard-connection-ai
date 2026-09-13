from __future__ import annotations

import numpy as np
import torch

from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D
from guard_connection_ai.streaming.constrained_morphology_renderer import (
    StatefulConstrainedMorphologyRenderer,
    stabilize_display_morphology,
)
from guard_connection_ai.streaming.timing_decoder import TimingEvent


def _renderer() -> StatefulConstrainedMorphologyRenderer:
    decoder = MorphologyAutoencoder1D(beat_samples=20, latent_dim=4, base_channels=2)
    return StatefulConstrainedMorphologyRenderer(
        morphology_decoder=decoder,
        population_latent=torch.zeros(4),
        sample_rate_hz=20,
        timing_output_delay_seconds=0.5,
        seconds_before_r=0.2,
        context_seconds=0.5,
        allow_learned_morphology=False,
    )


def test_renderer_uses_population_fallback_and_is_session_local() -> None:
    first = _renderer()
    second = _renderer()
    ppg = np.stack([np.sin(np.arange(15)), np.ones(15)]).astype(np.float32)
    output = first.process(ppg, [TimingEvent(10, 0.9)], render_enabled=True)
    assert output.events[0].source == "population_mean_latent"
    assert output.events[0].confidence == 0
    assert 300 <= output.events[0].repolarization_duration_prior_ms <= 480
    assert first.display_delay_seconds == 0.7
    assert second._sample_index == 0


def test_new_event_truncates_previous_tail_and_invalid_input_blanks() -> None:
    renderer = _renderer()
    ppg = np.stack([np.ones(15), np.ones(15)]).astype(np.float32)
    renderer.process(ppg, [TimingEvent(10, 0.9)], render_enabled=True)
    invalid = np.stack([np.ones(5), np.zeros(5)]).astype(np.float32)
    output = renderer.process(invalid, [TimingEvent(17, 0.9)], render_enabled=True)
    assert not np.any(output.pseudo_ecg_valid)
    assert np.all(output.pseudo_ecg == 0)


def test_renderer_reset_repeats_the_same_output() -> None:
    renderer = _renderer()
    ppg = np.stack([np.ones(15), np.ones(15)]).astype(np.float32)
    first = renderer.process(ppg, [TimingEvent(10, 0.9)], render_enabled=True)
    renderer.reset()
    repeated = renderer.process(ppg, [TimingEvent(10, 0.9)], render_enabled=True)
    np.testing.assert_allclose(first.pseudo_ecg, repeated.pseudo_ecg)
    np.testing.assert_array_equal(first.pseudo_ecg_valid, repeated.pseudo_ecg_valid)


def test_display_prior_removes_p_content_and_preserves_qrs_position() -> None:
    beat = np.zeros(100, dtype=np.float32)
    beat[12] = 0.4  # P-like content that must not be displayed.
    beat[20:25] = [0.2, -0.4, -2.0, 0.5, 0.2]
    beat[45:60] = np.sin(np.linspace(0, np.pi, 15)) * 0.5
    stabilized, duration_ms = stabilize_display_morphology(
        beat, r_index=22, sample_rate_hz=125, previous_rr_seconds=0.8
    )
    assert np.all(stabilized[:17] == 0)
    assert int(np.argmax(np.abs(stabilized[17:33]))) + 17 == 22
    assert stabilized[22] > 0
    assert np.max(stabilized[42:70]) > 0.2
    assert 300 <= duration_ms <= 480
    assert stabilized[-1] == 0


def test_display_prior_shortens_only_repolarization_tail_for_faster_rate() -> None:
    beat = np.zeros(100, dtype=np.float32)
    beat[20] = 1
    beat[30:90] = 0.3
    fast, fast_duration = stabilize_display_morphology(
        beat, r_index=20, sample_rate_hz=125, previous_rr_seconds=0.5
    )
    slow, slow_duration = stabilize_display_morphology(
        beat, r_index=20, sample_rate_hz=125, previous_rr_seconds=1.2
    )
    assert fast_duration < slow_duration
    np.testing.assert_allclose(fast[15:34], slow[15:34])
    assert np.flatnonzero(fast)[-1] < np.flatnonzero(slow)[-1]
