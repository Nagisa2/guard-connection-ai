from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalSegment:
    """A paired PPG/ECG signal segment and its source coordinates."""

    subject_id: str
    segment_id: int
    start_sample: int
    end_sample: int
    ppg: np.ndarray
    ecg: np.ndarray


def segment_starts(
    n_samples: int,
    window_samples: int = 1250,
    hop_samples: int = 625,
) -> list[int]:
    """Return start offsets for complete, non-padded segments."""

    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive.")
    if n_samples < window_samples:
        return []

    return list(range(0, n_samples - window_samples + 1, hop_samples))


def segment_signal_pair(
    subject_id: str,
    ppg: Sequence[float] | np.ndarray,
    ecg: Sequence[float] | np.ndarray,
    *,
    window_samples: int = 1250,
    hop_samples: int = 625,
) -> list[SignalSegment]:
    """Cut paired PPG and ECG arrays using identical source boundaries."""

    ppg_array = np.asarray(ppg)
    ecg_array = np.asarray(ecg)
    if ppg_array.ndim != 1 or ecg_array.ndim != 1:
        raise ValueError("PPG and ECG must be one-dimensional arrays.")
    if len(ppg_array) != len(ecg_array):
        raise ValueError("PPG and ECG must have the same number of samples.")

    segments = []
    for segment_id, start_sample in enumerate(
        segment_starts(len(ppg_array), window_samples, hop_samples)
    ):
        end_sample = start_sample + window_samples
        segments.append(
            SignalSegment(
                subject_id=subject_id,
                segment_id=segment_id,
                start_sample=start_sample,
                end_sample=end_sample,
                ppg=ppg_array[start_sample:end_sample],
                ecg=ecg_array[start_sample:end_sample],
            )
        )
    return segments


def build_segment_index(
    subject_lengths: dict[str, int],
    train_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    *,
    window_samples: int = 1250,
    hop_samples: int = 625,
) -> pd.DataFrame:
    """Build segment metadata without loading signal samples."""

    train = set(train_subjects)
    validation = set(validation_subjects)
    if train & validation:
        raise ValueError("Train and validation subjects must not overlap.")
    if set(subject_lengths) != train | validation:
        raise ValueError("Subject lengths must cover train and validation subjects exactly.")

    rows = []
    for subject_id in sorted(subject_lengths):
        split = "train" if subject_id in train else "validation"
        for segment_id, start_sample in enumerate(
            segment_starts(subject_lengths[subject_id], window_samples, hop_samples)
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "segment_id": segment_id,
                    "start_sample": start_sample,
                    "end_sample": start_sample + window_samples,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "split",
            "segment_id",
            "start_sample",
            "end_sample",
        ],
    )
