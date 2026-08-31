from __future__ import annotations

from pathlib import Path

import torch

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import (
    BIDMCSTFTDataset,
    find_subject_files,
    load_subject_signals,
)
from guard_connection_ai.data.segmentation import build_segment_index
from guard_connection_ai.data.split import split_subjects_train_val
from guard_connection_ai.models.resunet_attention import ResidualAttentionUNet
from scripts.train import save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"


def test_dataset_cache_stft_consistency() -> None:
    data_config = load_data_config(DATA_CONFIG_PATH)
    subject_files = dict(find_subject_files(DATA_ROOT))
    subject_lengths = {
        subject_id: len(load_subject_signals(path)[0])
        for subject_id, path in subject_files.items()
    }
    train_subjects, val_subjects = split_subjects_train_val(
        list(subject_files), train_size=43, random_state=42
    )
    index = build_segment_index(subject_lengths, train_subjects, val_subjects)

    dataset_uncached = BIDMCSTFTDataset(
        index,
        subject_files,
        stft_config=stft_config_from_data_config(data_config),
        split="validation",
        cache_stft=False,
    )
    dataset_cached = BIDMCSTFTDataset(
        index,
        subject_files,
        stft_config=stft_config_from_data_config(data_config),
        split="validation",
        cache_stft=True,
    )

    # First fetch populates cache
    item_uncached = dataset_uncached[0]
    item_cached_first = dataset_cached[0]
    # Second fetch reads from cache
    item_cached_second = dataset_cached[0]

    assert torch.equal(item_uncached["input"], item_cached_first["input"])
    assert torch.equal(item_cached_first["input"], item_cached_second["input"])
    assert torch.equal(item_uncached["target"], item_cached_first["target"])
    assert torch.equal(item_cached_first["target"], item_cached_second["target"])


def test_checkpoint_resume_weights_integrity(tmp_path: Path) -> None:
    model = ResidualAttentionUNet(in_channels=1, out_channels=1, base_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "test_resume.pt"

    save_checkpoint(model, optimizer, epoch=5, model_config={}, path=ckpt_path)

    new_model = ResidualAttentionUNet(in_channels=1, out_channels=1, base_channels=16)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert checkpoint["epoch"] == 5
    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)
