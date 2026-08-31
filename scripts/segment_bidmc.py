from pathlib import Path

from guard_connection_ai.data.dataset import (
    find_subject_files,
    load_subject_signals,
)
from guard_connection_ai.data.segmentation import build_segment_index
from guard_connection_ai.data.split import split_subjects_train_val

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "segmentation" / "segment_index.csv"


def main() -> None:
    subject_files = find_subject_files(DATA_ROOT)
    subject_ids = [subject_id for subject_id, _ in subject_files]
    train_subjects, validation_subjects = split_subjects_train_val(
        subject_ids,
        train_size=43,
        random_state=42,
    )
    subject_lengths = {
        subject_id: len(load_subject_signals(path)[0])
        for subject_id, path in subject_files
    }
    index = build_segment_index(
        subject_lengths,
        train_subjects,
        validation_subjects,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(OUTPUT_PATH, index=False)

    print(f"Total subjects: {len(subject_ids)}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(validation_subjects)}")
    print(f"Total segments: {len(index)}")
    print(f"Train segments: {(index['split'] == 'train').sum()}")
    print(f"Validation segments: {(index['split'] == 'validation').sum()}")
    print("Segments per subject:")
    print(index.groupby("subject_id").size().to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()