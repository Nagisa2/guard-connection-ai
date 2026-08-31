import yaml

from guard_connection_ai.data.split import (
    save_subject_split,
    split_subjects,
    split_subjects_train_val,
)


def test_subjects_do_not_overlap():
    subjects = [f"subject_{i:02d}" for i in range(53)]

    train, val, test = split_subjects(subjects)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)

    assert set(train) | set(val) | set(test) == set(subjects)


def test_train_validation_split_is_subject_wise_and_reproducible():
    subjects = [f"subject_{i:02d}" for i in range(53)]

    first = split_subjects_train_val(subjects, train_size=43, random_state=42)
    second = split_subjects_train_val(subjects, train_size=43, random_state=42)

    train, val = first
    assert first == second
    assert len(train) == 43
    assert len(val) == 10
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(subjects)


def test_train_validation_split_deduplicates_subject_ids():
    train, val = split_subjects_train_val(["subject_01", "subject_01", "subject_02"])

    assert set(train) | set(val) == {"subject_01", "subject_02"}
    assert set(train).isdisjoint(val)


def test_subject_split_can_be_saved_with_seed(tmp_path):
    train, val = split_subjects_train_val(
        [f"subject_{i:02d}" for i in range(53)],
        train_size=43,
        random_state=123,
    )
    output_path = tmp_path / "data_split.yaml"

    save_subject_split(train, val, output_path, random_state=123)
    saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert saved["seed"] == 123
    assert saved["split"]["train_subjects"] == train
    assert saved["split"]["validation_subjects"] == val