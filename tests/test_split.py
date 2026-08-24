from guard_connection_ai.data.split import split_subjects


def test_subjects_do_not_overlap():
    subjects = [f"subject_{i:02d}" for i in range(53)]

    train, val, test = split_subjects(subjects)

    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)

    assert set(train) | set(val) | set(test) == set(subjects)