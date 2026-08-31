from pathlib import Path

import pandas as pd

from guard_connection_ai.data.json_metadata import find_json_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "json_Data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "json_data" / "metadata_manifest.csv"


def main() -> None:
    metadata = find_json_metadata(DATA_ROOT)
    rows = [
        {
            "subject_id": item.subject_id,
            "signal_type": item.signal_type,
            "recording_id": item.recording_id,
            "file": item.path.name,
            "num_data_records": item.num_data_records,
            "has_af_annotation": item.has_af_annotation,
            "signal_keys": ",".join(sorted(item.signal_shapes)),
        }
        for item in metadata
    ]
    frame = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Metadata files: {len(frame)}")
    print(f"ECG files: {(frame['signal_type'] == 'ECG').sum()}")
    print(f"PPG files: {(frame['signal_type'] == 'PPG').sum()}")
    print(f"AF annotation files: {frame['has_af_annotation'].sum()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
