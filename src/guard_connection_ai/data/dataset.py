from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from guard_connection_ai.data.preprocessing import PreprocessingMode, preprocess_signal
from guard_connection_ai.data.stft import STFTConfig, compute_stft


def find_subject_files(data_root: str | Path) -> list[tuple[str, Path]]:
	"""Find BIDMC signal CSVs and return stable subject IDs."""

	root = Path(data_root)
	files = []
	for path in root.glob("bidmc_*_Signals.csv"):
		subject_id = path.stem.removeprefix("bidmc_").removesuffix("_Signals")
		if subject_id.isdigit():
			files.append((f"bidmc{int(subject_id):02d}", path))
	return sorted(files)


def load_subject_signals(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
	"""Load paired PLETH and ECG II signals from one BIDMC CSV."""

	frame = pd.read_csv(path)
	frame.columns = frame.columns.str.strip()
	missing = {column for column in ("PLETH", "II") if column not in frame}
	if missing:
		raise ValueError(f"Missing required signal columns: {sorted(missing)}")

	ppg = frame["PLETH"].to_numpy(dtype=np.float64)
	ecg = frame["II"].to_numpy(dtype=np.float64)
	if len(ppg) != len(ecg):
		raise ValueError("PPG and ECG must have the same number of samples.")
	return ppg, ecg


class BIDMCSTFTDataset(Dataset):
	"""On-demand paired PPG/ECG magnitude STFT dataset."""

	def __init__(
		self,
		segment_index: pd.DataFrame,
		subject_files: dict[str, str | Path],
		*,
		stft_config: STFTConfig | None = None,
		preprocessing_mode: PreprocessingMode = "none",
		split: str | None = None,
		cache_stft: bool = False,
	) -> None:
		required_columns = {
			"subject_id",
			"segment_id",
			"start_sample",
			"end_sample",
		}
		if not required_columns.issubset(segment_index.columns):
			raise ValueError(f"Segment index requires columns: {sorted(required_columns)}")
		if split is not None:
			if "split" not in segment_index.columns:
				raise ValueError("A split column is required when split is specified.")
			segment_index = segment_index[segment_index["split"] == split]
		self.segment_index = segment_index.reset_index(drop=True)
		self.subject_files = {str(key): Path(value) for key, value in subject_files.items()}
		self.stft_config = stft_config or STFTConfig()
		self.preprocessing_mode = preprocessing_mode
		self.cache_stft = cache_stft
		self._signal_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
		self._stft_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

	def __len__(self) -> int:
		return len(self.segment_index)

	def __getitem__(self, index: int) -> dict[str, object]:
		row = self.segment_index.iloc[index]
		subject_id = str(row["subject_id"])
		start_sample = int(row["start_sample"])
		end_sample = int(row["end_sample"])
		segment_id = int(row["segment_id"])

		if self.cache_stft and index in self._stft_cache:
			input_tensor, target_tensor = self._stft_cache[index]
			return {
				"input": input_tensor.clone(),
				"target": target_tensor.clone(),
				"subject_id": subject_id,
				"segment_id": segment_id,
				"start_sample": start_sample,
				"end_sample": end_sample,
			}

		if subject_id not in self.subject_files:
			raise KeyError(f"No signal file found for subject: {subject_id}")
		if subject_id not in self._signal_cache:
			self._signal_cache[subject_id] = load_subject_signals(self.subject_files[subject_id])
		ppg, ecg = self._signal_cache[subject_id]

		ppg_segment = preprocess_signal(ppg[start_sample:end_sample], self.preprocessing_mode)
		ecg_segment = preprocess_signal(ecg[start_sample:end_sample], self.preprocessing_mode)
		ppg_stft = compute_stft(ppg_segment, self.stft_config)
		ecg_stft = compute_stft(ecg_segment, self.stft_config)

		input_tensor = torch.from_numpy(ppg_stft.magnitude.astype(np.float32)).unsqueeze(0)
		target_tensor = torch.from_numpy(ecg_stft.magnitude.astype(np.float32)).unsqueeze(0)

		if self.cache_stft:
			self._stft_cache[index] = (input_tensor, target_tensor)

		return {
			"input": input_tensor,
			"target": target_tensor,
			"subject_id": subject_id,
			"segment_id": segment_id,
			"start_sample": start_sample,
			"end_sample": end_sample,
		}
