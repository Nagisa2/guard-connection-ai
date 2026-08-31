from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def split_subjects_train_val(
    subject_ids: Sequence[str],
    train_size: float = 0.8,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    """Split unique subject IDs into reproducible train and validation sets."""

    subjects = sorted(set(subject_ids))
    if len(subjects) < 2:
        raise ValueError("At least two unique subjects are required.")

    if isinstance(train_size, int) and not isinstance(train_size, bool):
        if not 0 < train_size < len(subjects):
            raise ValueError("Integer train_size must be between 1 and n_subjects - 1.")
    elif not 0 < train_size < 1:
        raise ValueError("Float train_size must be between 0 and 1.")

    train_subjects, val_subjects = train_test_split(
        subjects,
        train_size=train_size,
        random_state=random_state,
    )
    return sorted(train_subjects), sorted(val_subjects)


def save_subject_split(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    path: str | Path,
    random_state: int = 42,
) -> None:
    """Save an explicit subject split as YAML."""

    train = sorted(set(train_subjects))
    validation = sorted(set(val_subjects))
    if set(train) & set(validation):
        raise ValueError("Train and validation subjects must not overlap.")

    output = {
        "seed": random_state,
        "split": {
            "method": "subject_wise",
            "train_subjects": train,
            "validation_subjects": validation,
        },
    }
    Path(path).write_text(
        yaml.safe_dump(output, sort_keys=False),
        encoding="utf-8",
    )


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