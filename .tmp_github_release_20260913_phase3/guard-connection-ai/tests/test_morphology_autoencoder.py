from __future__ import annotations

import numpy as np
import torch

from guard_connection_ai.data.morphology_dataset import (
    ECGBeatMorphologyDataset,
    extract_centered_beat,
    refine_peak_locations,
)
from guard_connection_ai.losses.morphology import MorphologyAutoencoderLoss
from guard_connection_ai.models.morphology_autoencoder import MorphologyAutoencoder1D


def test_centered_beat_preserves_r_location_without_time_warping() -> None:
    waveform = np.arange(20, dtype=np.float32)
    valid = np.ones(20, dtype=bool)
    beat, beat_valid = extract_centered_beat(
        waveform,
        valid,
        peak_index=8,
        samples_before=3,
        samples_after=5,
    )
    assert beat.tolist() == list(range(5, 13))
    assert beat[3] == waveform[8]
    assert np.all(beat_valid)


def test_peak_refinement_selects_largest_local_absolute_deflection() -> None:
    waveform = np.zeros(30)
    waveform[11] = -2
    waveform[21] = 3
    refined = refine_peak_locations(waveform, np.asarray([10, 20]), radius_samples=2)
    assert refined.tolist() == [11, 21]


def test_autoencoder_round_trip_shape_and_latent_decode() -> None:
    model = MorphologyAutoencoder1D(beat_samples=100, latent_dim=8, base_channels=4)
    waveform = torch.randn(3, 1, 100)
    reconstruction, latent = model(waveform)
    assert reconstruction.shape == waveform.shape
    assert latent.shape == (3, 8)
    torch.testing.assert_close(model.decode(latent), reconstruction)


def test_morphology_loss_is_finite_and_differentiable() -> None:
    model = MorphologyAutoencoder1D(beat_samples=100, latent_dim=8, base_channels=4)
    waveform = torch.randn(3, 1, 100)
    prediction, latent = model(waveform)
    mask = torch.ones_like(waveform)
    mask[..., :2] = 0
    losses = MorphologyAutoencoderLoss(sample_rate_hz=125, r_index=20)(
        prediction, waveform, mask, latent
    )
    losses["total"].backward()
    assert set(losses) == {
        "total",
        "roi",
        "derivative",
        "multi_resolution_stft",
        "latent_regularization",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dataset_masks_neighboring_qrs_from_encoder_input(monkeypatch) -> None:
    class Record:
        recording_id = "sample"
        patient_id = "patient"
        sample_rate_hz = 100.0

    raw = np.zeros(200)
    raw[[50, 110]] = 5
    monkeypatch.setattr(
        "guard_connection_ai.data.morphology_dataset.load_mimic_perform_signals",
        lambda _record: (np.arange(200) / 100, raw, raw),
    )
    monkeypatch.setattr(
        "guard_connection_ai.data.morphology_dataset.detect_rhythm_peaks",
        lambda *_args, **_kwargs: np.asarray([50, 110]),
    )
    monkeypatch.setattr(
        "guard_connection_ai.data.morphology_dataset.refine_peak_locations",
        lambda _waveform, peaks, **_kwargs: peaks,
    )
    dataset = ECGBeatMorphologyDataset(
        [Record()],
        ecg_scaler=type("Scaler", (), {"center": 0.0, "scale": 1.0})(),
        seconds_before_r=0.2,
        seconds_after_r=0.8,
        ecg_bandpass_hz=(0.5, 35.0),
        neighboring_qrs_guard_seconds=0.12,
    )
    item = dataset[0]
    valid = item["valid_mask"].numpy()[0].astype(bool)
    waveform = item["waveform"].numpy()[0]
    assert not np.any(valid[68:])
    assert np.all(waveform[~valid] == 0)
