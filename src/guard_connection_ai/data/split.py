from __future__ import annotations

from typing import Sequence

from sklearn.model_selection import train_test_split


def split_subjects(
    subject_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split subject IDs into train/validation/test sets."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    subjects = list(dict.fromkeys(subject_ids))

    train_subjects, temp_subjects = train_test_split(
        subjects,
        train_size=train_ratio,
        random_state=random_state,
    )

    relative_val_ratio = val_ratio / (1.0 - train_ratio)

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=relative_val_ratio,
        random_state=random_state,
    )

    return train_subjects, val_subjects, test_subjects