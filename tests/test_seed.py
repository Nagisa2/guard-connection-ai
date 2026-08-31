from __future__ import annotations

import torch

from guard_connection_ai.data.split import split_subjects_train_val
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet
from guard_connection_ai.utils.seed import set_seed


def test_split_subjects_reproducibility() -> None:
    subject_ids = [f"subject_{i:02d}" for i in range(50)]

    train1, val1 = split_subjects_train_val(subject_ids, train_size=40, random_state=42)
    train2, val2 = split_subjects_train_val(subject_ids, train_size=40, random_state=42)
    train3, _val3 = split_subjects_train_val(subject_ids, train_size=40, random_state=43)

    assert train1 == train2
    assert val1 == val2
    assert train1 != train3  # Different seed yields different split


def test_model_initialization_seed_reproducibility() -> None:
    set_seed(42)
    model1 = ResidualAttentionUNet(in_channels=1, out_channels=1, base_channels=16)

    set_seed(42)
    model2 = ResidualAttentionUNet(in_channels=1, out_channels=1, base_channels=16)

    set_seed(43)
    model3 = ResidualAttentionUNet(in_channels=1, out_channels=1, base_channels=16)

    # Check identical parameters for same seed
    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)

    # Check different parameters for different seed
    different = False
    for p1, p3 in zip(model1.parameters(), model3.parameters(), strict=True):
        if not torch.equal(p1, p3):
            different = True
            break
    assert different
