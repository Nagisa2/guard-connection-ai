from pathlib import Path

import pandas as pd

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import find_subject_files, load_subject_signals
from guard_connection_ai.data.segmentation import segment_signal_pair
from guard_connection_ai.data.stft import (
    compute_stft,
    inverse_stft,
    reconstruction_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "stft" / "reconstruction_results.csv"


def main() -> None:
    config = stft_config_from_data_config(load_data_config(CONFIG_PATH))
    results = []
    for subject_id, path in find_subject_files(DATA_ROOT):
        _, ecg = load_subject_signals(path)
        segment = segment_signal_pair(subject_id, ecg, ecg)[0]
        transformed = compute_stft(segment.ecg, config)
        reconstructed = inverse_stft(transformed, config)[: segment.ecg.size]
        results.append({"subject_id": subject_id, **reconstruction_metrics(segment.ecg, reconstructed)})

    frame = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(frame.describe().to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
