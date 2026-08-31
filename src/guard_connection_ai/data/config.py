from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from guard_connection_ai.data.stft import STFTConfig


def load_data_config(path: str | Path) -> dict[str, Any]:
    """Load the project data configuration from YAML."""

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Data configuration must contain a YAML mapping.")
    return config


def stft_config_from_data_config(config: dict[str, Any]) -> STFTConfig:
    """Build STFT parameters from the data configuration mapping."""

    section = config.get("stft")
    if not isinstance(section, dict):
        raise TypeError("Data configuration must contain an stft mapping.")
    return STFTConfig(
        sampling_rate=float(section["sampling_rate"]),
        nperseg=int(section["nperseg"]),
        noverlap=int(section["noverlap"]),
        nfft=int(section["nfft"]),
        window=str(section["window"]),
    )
