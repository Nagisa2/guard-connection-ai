import argparse
from pathlib import Path

import numpy as np

from guard_connection_ai.data.config import (
    load_data_config,
    stft_config_from_data_config,
)
from guard_connection_ai.data.dataset import find_subject_files, load_subject_signals
from guard_connection_ai.data.preprocessing import PreprocessingMode, preprocess_signal
from guard_connection_ai.data.segmentation import segment_signal_pair
from guard_connection_ai.data.stft import compute_stft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DATA_ROOT = PROJECT_ROOT / "data" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
SUBJECT_IDS = {"bidmc01", "bidmc10", "bidmc25", "bidmc40", "bidmc53"}
SEGMENT_IDS = (0, 1)


def main(mode: PreprocessingMode, config_path: Path = CONFIG_PATH) -> None:
    data_config = load_data_config(config_path)
    config = stft_config_from_data_config(data_config)
    preprocessing_config = data_config["preprocessing"]
    bandpass_config = preprocessing_config["bandpass"]
    output_path = PROJECT_ROOT / "outputs" / "stft" / f"stft_examples_{mode}.npz"
    records = []
    for subject_id, path in find_subject_files(DATA_ROOT):
        if subject_id not in SUBJECT_IDS:
            continue
        ppg, ecg = load_subject_signals(path)
        segments = segment_signal_pair(subject_id, ppg, ecg)
        for segment_id in SEGMENT_IDS:
            segment = segments[segment_id]
            preprocessing_kwargs = {
                "sampling_rate": config.sampling_rate,
                "lowcut_hz": bandpass_config["lowcut_hz"],
                "highcut_hz": bandpass_config["highcut_hz"],
                "filter_order": bandpass_config["filter_order"],
            }
            processed_ppg = preprocess_signal(segment.ppg, mode, **preprocessing_kwargs)
            processed_ecg = preprocess_signal(segment.ecg, mode, **preprocessing_kwargs)
            ppg_stft = compute_stft(processed_ppg, config)
            ecg_stft = compute_stft(processed_ecg, config)
            if not np.array_equal(ppg_stft.frequencies, ecg_stft.frequencies):
                raise ValueError("PPG and ECG frequency bins do not match.")
            if not np.array_equal(ppg_stft.times, ecg_stft.times):
                raise ValueError("PPG and ECG time bins do not match.")
            records.append(
                {
                    "subject_id": subject_id,
                    "segment_id": segment.segment_id,
                    "start_sample": segment.start_sample,
                    "end_sample": segment.end_sample,
                    "ppg_magnitude": ppg_stft.magnitude.astype(np.float32),
                    "ecg_magnitude": ecg_stft.magnitude.astype(np.float32),
                    "frequencies": ppg_stft.frequencies,
                    "times": ppg_stft.times,
                }
            )

    if not records:
        raise RuntimeError("No STFT examples were generated.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        preprocessing_mode=mode,
        subject_id=np.array([record["subject_id"] for record in records]),
        segment_id=np.array([record["segment_id"] for record in records]),
        start_sample=np.array([record["start_sample"] for record in records]),
        end_sample=np.array([record["end_sample"] for record in records]),
        ppg_magnitude=np.stack([record["ppg_magnitude"] for record in records]),
        ecg_magnitude=np.stack([record["ecg_magnitude"] for record in records]),
        frequencies=records[0]["frequencies"],
        times=records[0]["times"],
    )
    print(f"Generated {len(records)} paired examples")
    print(f"Magnitude shape per signal: {records[0]['ppg_magnitude'].shape}")
    print(f"Preprocessing mode: {mode}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["none", "detrend", "bandpass", "segment_zscore"],
        default=None,
    )
    args = parser.parse_args()
    configured_mode = load_data_config(CONFIG_PATH)["preprocessing"]["mode"]
    main(args.mode or configured_mode)
