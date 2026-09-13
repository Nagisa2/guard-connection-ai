"""山口大学MATLAB 7.3データの配列・時刻・注釈整合性を非破壊で監査する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _size(dataset: h5py.Dataset) -> int:
    return int(np.prod(dataset.shape))


def _scalar(dataset: h5py.Dataset) -> float:
    values = np.asarray(dataset[()]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"expected scalar dataset: {dataset.name}")
    return float(values[0])


def _ascii(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset[()]).reshape(-1)
    return "".join(chr(int(value)) for value in values)


def _clock_seconds(day: str, time: str) -> float:
    hours, minutes, seconds = (int(value) for value in time.split(":"))
    return int(day) * 86400 + hours * 3600 + minutes * 60 + seconds


def _unique_values_chunked(dataset: h5py.Dataset) -> tuple[float, ...]:
    result: set[float] = set()
    axis_size = dataset.shape[0]
    for start in range(0, axis_size, 100_000):
        values = np.asarray(dataset[start : start + 100_000]).reshape(-1)
        result.update(float(value) for value in np.unique(values[np.isfinite(values)]))
    return tuple(sorted(result))


def _flat_slice(dataset: h5py.Dataset, start: int, end: int) -> np.ndarray:
    if dataset.shape[0] == 1:
        return np.asarray(dataset[:, start:end]).reshape(-1)
    return np.asarray(dataset[start:end, :]).reshape(-1)


def audit(root: Path) -> dict[str, object]:
    files = sorted(root.glob("*.mat"))
    ecg_files = [path for path in files if "_ECG_" in path.name]
    ppg_files = [path for path in files if "_PPG_" in path.name]
    array_issues: list[str] = []
    timestamp_issues: list[str] = []
    complete_ecg_files = 0
    complete_ppg_files = 0
    ecg_ranges: dict[str, tuple[float, float]] = {}
    af_values: dict[str, tuple[float, ...]] = {}
    af_positive_counts: dict[str, int] = {}
    samples_per_record: dict[str, set[float]] = defaultdict(set)
    subject_ppg_counts: Counter[str] = Counter()
    ppg_gap_counts: Counter[str] = Counter()
    ppg_duplicate_counts: Counter[str] = Counter()
    ppg_reverse_counts: Counter[str] = Counter()
    ppg_record_counts: Counter[str] = Counter()
    ppg_ecg_overlap_record_counts: Counter[str] = Counter()
    ecg_record_counts: dict[str, int] = {}
    timestamp_difference_counts: Counter[str] = Counter()
    timestamp_elapsed_excess_seconds = 0.0
    maximum_timestamp_gap_seconds = 0.0
    identical_duplicate_timestamp_records = 0
    ppg_ranges: list[tuple[str, str, float, float]] = []

    for path in ecg_files:
        subject = path.name.split("_", maxsplit=1)[0]
        with h5py.File(path, "r") as handle:
            required = {
                "ECG",
                "QRSindex",
                "rr",
                "AF_annotation",
                "num_data_records",
                "recording_startday",
                "recording_starttime",
            }
            missing = sorted(required - set(handle))
            if missing:
                array_issues.append(f"{path.name}: missing {', '.join(missing)}")
                continue
            records = _scalar(handle["num_data_records"])
            ecg_record_counts[subject] = round(records)
            samples_per_record["ecg"].add(_size(handle["ECG"]) / records)
            qrs_count = _size(handle["QRSindex"])
            rr_count = _size(handle["rr"])
            af_count = _size(handle["AF_annotation"])
            if not (qrs_count == rr_count + 1 == af_count + 1):
                array_issues.append(
                    f"{path.name}: QRS/RR/AF lengths differ "
                    f"({qrs_count}/{rr_count}/{af_count})"
                )
            qrs_last = float(np.asarray(handle["QRSindex"][-1]).reshape(-1)[0])
            if qrs_last > _size(handle["ECG"]):
                array_issues.append(f"{path.name}: last QRS index exceeds ECG length")
            values = _unique_values_chunked(handle["AF_annotation"])
            af_values[subject] = values
            af_dataset = handle["AF_annotation"]
            positive = 0
            for start in range(0, af_dataset.shape[0], 100_000):
                chunk = np.asarray(af_dataset[start : start + 100_000]).reshape(-1)
                positive += int(np.count_nonzero(chunk == 1))
            af_positive_counts[subject] = positive
            start = _clock_seconds(
                _ascii(handle["recording_startday"]),
                _ascii(handle["recording_starttime"]),
            )
            ecg_ranges[subject] = (start, start + records)
            for field in ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"):
                if field in handle:
                    samples_per_record["ecg_accelerometer"].add(
                        _size(handle[field]) / records
                    )
            complete_ecg_files += 1

    for path in ppg_files:
        subject = path.name.split("_", maxsplit=1)[0]
        subject_ppg_counts[subject] += 1
        with h5py.File(path, "r") as handle:
            required = {
                "PPG_GREEN",
                "PPG_AMBIENT",
                "Accelerometer_X",
                "Accelerometer_Y",
                "Accelerometer_Z",
                "timestamp_sec_data_record",
                "num_data_records",
                "recording_startday",
                "recording_starttime",
            }
            missing = sorted(required - set(handle))
            if missing:
                array_issues.append(f"{path.name}: missing {', '.join(missing)}")
                continue
            records = _scalar(handle["num_data_records"])
            ppg_record_counts[subject] += round(records)
            for field in ("PPG_GREEN", "PPG_AMBIENT"):
                samples_per_record[field.lower()].add(_size(handle[field]) / records)
            for field in ("Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"):
                samples_per_record["ppg_accelerometer"].add(
                    _size(handle[field]) / records
                )
            timestamp_dataset = handle["timestamp_sec_data_record"]
            if _size(timestamp_dataset) != round(records):
                array_issues.append(f"{path.name}: timestamp count differs from records")
            timestamps = np.asarray(timestamp_dataset[()]).reshape(-1)
            differences = np.diff(timestamps)
            timestamp_difference_counts["zero_seconds"] += int(
                np.count_nonzero(np.abs(differences) < 0.01)
            )
            timestamp_difference_counts["one_second"] += int(
                np.count_nonzero(np.abs(differences - 1) < 0.01)
            )
            timestamp_difference_counts["two_seconds"] += int(
                np.count_nonzero(np.abs(differences - 2) < 0.01)
            )
            timestamp_elapsed_excess_seconds += float(
                timestamps[-1] - timestamps[0] - (timestamps.size - 1)
            )
            maximum_timestamp_gap_seconds = max(
                maximum_timestamp_gap_seconds, float(np.max(differences))
            )
            nonfinite_count = int(np.count_nonzero(~np.isfinite(timestamps)))
            duplicate_count = int(np.count_nonzero(differences == 0))
            reverse_count = int(np.count_nonzero(differences < 0))
            duplicate_or_reverse_count = duplicate_count + reverse_count
            if nonfinite_count or duplicate_or_reverse_count:
                timestamp_issues.append(
                    f"{path.name}: nonfinite={nonfinite_count}, "
                    f"duplicate_or_reverse={duplicate_or_reverse_count}"
                )
            ppg_gap_counts[subject] += int(np.count_nonzero(differences > 1.5))
            ppg_duplicate_counts[subject] += duplicate_count
            ppg_reverse_counts[subject] += reverse_count
            duplicate_indices = np.flatnonzero(np.abs(differences) < 0.01)
            green = handle["PPG_GREEN"]
            samples_in_record = round(_size(green) / records)
            for index in duplicate_indices:
                first = _flat_slice(
                    green, int(index) * samples_in_record, (int(index) + 1) * samples_in_record
                )
                second = _flat_slice(
                    green,
                    (int(index) + 1) * samples_in_record,
                    (int(index) + 2) * samples_in_record,
                )
                identical_duplicate_timestamp_records += int(
                    np.array_equal(first, second)
                )
            start = _clock_seconds(
                _ascii(handle["recording_startday"]),
                _ascii(handle["recording_starttime"]),
            )
            ppg_ranges.append(
                (path.name, subject, start + timestamps[0], start + timestamps[-1])
            )
            ecg_start, ecg_end = ecg_ranges[subject]
            absolute_timestamps = start + timestamps
            ppg_ecg_overlap_record_counts[subject] += int(
                np.count_nonzero(
                    (absolute_timestamps >= ecg_start)
                    & (absolute_timestamps <= ecg_end)
                )
            )
            complete_ppg_files += 1

    non_overlapping = []
    for filename, subject, start, end in ppg_ranges:
        ecg_start, ecg_end = ecg_ranges[subject]
        if min(end, ecg_end) <= max(start, ecg_start):
            non_overlapping.append(filename)

    return {
        "source_root": str(root.resolve()),
        "mat_files": len(files),
        "subjects": sorted(ecg_ranges),
        "ecg_files": len(ecg_files),
        "ppg_files": len(ppg_files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "samples_per_record": {
            key: sorted(values) for key, values in sorted(samples_per_record.items())
        },
        "ppg_files_per_subject": dict(sorted(subject_ppg_counts.items())),
        "ecg_record_hours_per_subject": {
            subject: records / 3600
            for subject, records in sorted(ecg_record_counts.items())
        },
        "ppg_record_hours_per_subject": {
            subject: records / 3600
            for subject, records in sorted(ppg_record_counts.items())
        },
        "ppg_records_within_ecg_clock_range_per_subject": dict(
            sorted(ppg_ecg_overlap_record_counts.items())
        ),
        "af_annotation_values_per_subject": {
            key: list(values) for key, values in sorted(af_values.items())
        },
        "af_positive_intervals_per_subject": dict(sorted(af_positive_counts.items())),
        "subjects_with_positive_af_annotations": [
            subject for subject, count in sorted(af_positive_counts.items()) if count > 0
        ],
        "ppg_timestamp_gaps_over_1_5_seconds_per_subject": dict(
            sorted(ppg_gap_counts.items())
        ),
        "ppg_duplicate_timestamps_per_subject": dict(
            sorted(ppg_duplicate_counts.items())
        ),
        "ppg_reverse_timestamps_per_subject": dict(sorted(ppg_reverse_counts.items())),
        "ppg_timestamp_difference_counts": dict(
            sorted(timestamp_difference_counts.items())
        ),
        "ppg_timestamp_elapsed_excess_seconds": timestamp_elapsed_excess_seconds,
        "maximum_ppg_timestamp_gap_seconds": maximum_timestamp_gap_seconds,
        "identical_waveforms_among_duplicate_timestamp_pairs": (
            identical_duplicate_timestamp_records
        ),
        "ppg_segments_without_ecg_clock_overlap": non_overlapping,
        "array_structural_issues": array_issues,
        "ppg_timestamp_issues": timestamp_issues,
        "can_load_all_ppg": complete_ppg_files == len(ppg_files) == 86,
        "can_load_all_ecg_qrs_af": complete_ecg_files == len(ecg_files) == 8,
        "can_align_every_ppg_segment_without_gap_policy": not timestamp_issues
        and not non_overlapping,
        "clock_overlap_is_not_sample_alignment_proof": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data/Data/Data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/validation/yamaguchi_mat_audit.json",
    )
    args = parser.parse_args()
    report = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
