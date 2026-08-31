import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from guard_connection_ai.data.dataset import (
	BIDMCSTFTDataset,
	find_subject_files,
	load_subject_signals,
)
from guard_connection_ai.data.json_metadata import (
	find_json_metadata,
	load_json_metadata,
)
from guard_connection_ai.data.preprocessing import preprocess_signal
from guard_connection_ai.data.segmentation import (
	build_segment_index,
	segment_signal_pair,
	segment_starts,
)
from guard_connection_ai.data.stft import STFTConfig


def test_load_subject_signals_reads_paired_columns(tmp_path):
	path = tmp_path / "bidmc_01_Signals.csv"
	pd.DataFrame({" PLETH ": [1, 2], "II": [3, 4]}).to_csv(path, index=False)

	files = find_subject_files(tmp_path)
	ppg, ecg = load_subject_signals(files[0][1])

	assert files == [("bidmc01", path)]
	assert np.array_equal(ppg, [1.0, 2.0])
	assert np.array_equal(ecg, [3.0, 4.0])


def test_json_metadata_loader_parses_signal_identity_and_shapes(tmp_path):
	path = tmp_path / "07_ECG_02.json"
	path.write_text(
		'{"ECG": {"shape": [100, 1], "dtype": "float64"}, '
		'"AF_annotation": {"shape": [100, 1]}, "num_data_records": 100}',
		encoding="utf-8",
	)

	metadata = load_json_metadata(path)

	assert metadata.subject_id == "07"
	assert metadata.signal_type == "ECG"
	assert metadata.recording_id == 2
	assert metadata.signal_shapes["ECG"] == (100, 1)
	assert metadata.has_af_annotation
	assert find_json_metadata(tmp_path) == [metadata]


def test_stft_dataset_returns_paired_model_tensors(tmp_path):
	path = tmp_path / "bidmc_01_Signals.csv"
	time = np.arange(1250) / 125.0
	pd.DataFrame({"PLETH": np.sin(time), "II": np.cos(time)}).to_csv(path, index=False)
	index = pd.DataFrame(
		[
			{
				"subject_id": "bidmc01",
				"split": "train",
				"segment_id": 0,
				"start_sample": 0,
				"end_sample": 1250,
			}
		]
	)

	dataset = BIDMCSTFTDataset(
		index,
		{"bidmc01": path},
		stft_config=STFTConfig(),
		split="train",
	)
	item = dataset[0]

	assert len(dataset) == 1
	assert item["input"].shape == (1, 33, 41)
	assert item["target"].shape == (1, 33, 41)
	assert item["subject_id"] == "bidmc01"
	assert item["segment_id"] == 0


def test_stft_dataset_batches_with_dataloader(tmp_path):
	path = tmp_path / "bidmc_01_Signals.csv"
	time = np.arange(1875) / 125.0
	pd.DataFrame({"PLETH": np.sin(time), "II": np.cos(time)}).to_csv(path, index=False)
	index = pd.DataFrame(
		[
			{
				"subject_id": "bidmc01",
				"split": "train",
				"segment_id": segment_id,
				"start_sample": start_sample,
				"end_sample": start_sample + 1250,
			}
			for segment_id, start_sample in enumerate((0, 625))
		]
	)
	dataset = BIDMCSTFTDataset(index, {"bidmc01": path})

	batch = next(iter(DataLoader(dataset, batch_size=2)))

	assert batch["input"].shape == (2, 1, 33, 41)
	assert batch["target"].shape == (2, 1, 33, 41)
	assert list(batch["segment_id"].numpy()) == [0, 1]


def test_preprocessing_modes_preserve_raw_and_normalize_per_segment():
	signal = np.linspace(-2.0, 3.0, 1250)

	assert np.array_equal(preprocess_signal(signal, "none"), signal)

	normalized = preprocess_signal(signal, "segment_zscore")
	assert abs(normalized.mean()) < 1e-12
	assert abs(normalized.std() - 1.0) < 1e-12


def test_segment_zscore_handles_constant_signal():
	result = preprocess_signal(np.ones(1250), "segment_zscore")

	assert np.array_equal(result, np.zeros(1250))


def test_bandpass_returns_finite_signal():
	time = np.arange(1250) / 125.0
	signal = np.sin(2 * np.pi * 2 * time) + np.sin(2 * np.pi * 40 * time)

	filtered = preprocess_signal(signal, "bandpass")

	assert filtered.shape == signal.shape
	assert np.isfinite(filtered).all()


def test_segment_starts_for_bidmc_length():
	starts = segment_starts(60001, window_samples=1250, hop_samples=625)

	assert len(starts) == 95
	assert starts[:3] == [0, 625, 1250]
	assert starts[-1] == 58750


def test_segment_signal_pair_uses_identical_boundaries():
	ppg = np.arange(3000)
	ecg = ppg + 10000

	segments = segment_signal_pair(
		"subject_01",
		ppg,
		ecg,
		window_samples=1250,
		hop_samples=625,
	)

	assert len(segments) == 3
	assert [(segment.start_sample, segment.end_sample) for segment in segments] == [
		(0, 1250),
		(625, 1875),
		(1250, 2500),
	]
	assert np.array_equal(segments[1].ppg, ppg[625:1875])
	assert np.array_equal(segments[1].ecg, ecg[625:1875])


def test_segment_signal_pair_rejects_mismatched_lengths():
	with pytest.raises(ValueError, match="same number of samples"):
		segment_signal_pair("subject_01", np.zeros(10), np.zeros(9))


def test_build_segment_index_contains_split_metadata():
	index = build_segment_index(
		{"subject_01": 2500, "subject_02": 1250},
		["subject_01"],
		["subject_02"],
		window_samples=1250,
		hop_samples=625,
	)

	assert list(index.columns) == [
		"subject_id",
		"split",
		"segment_id",
		"start_sample",
		"end_sample",
	]
	assert len(index) == 4
	assert index[index["subject_id"] == "subject_01"]["split"].eq("train").all()
	assert index[index["subject_id"] == "subject_02"]["split"].eq("validation").all()
