from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from guard_connection_ai.data.causal_preprocessing import (
    FixedRobustScaler,
    StatefulCausalWaveformPreprocessor,
)
from guard_connection_ai.data.mimic_perform_af import (
    discover_mimic_perform_af,
    load_mimic_perform_signals,
)
from guard_connection_ai.deployment.morphology_checkpoint import (
    load_morphology_checkpoint,
)
from guard_connection_ai.deployment.timing_checkpoint import (
    load_validated_timing_checkpoint,
)
from guard_connection_ai.streaming.timing_decoder import TimingEvent


def _limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, (1, 99))
    margin = max((high - low) * 0.18, 0.1)
    return float(low - margin), float(high + margin)


def create_video(
    *,
    timing_checkpoint_path: Path,
    cross_modal_checkpoint_path: Path,
    morphology_checkpoint_path: Path,
    data_root: Path,
    output_path: Path,
    recording_id: str,
    start_seconds: float,
    duration_seconds: float,
    window_seconds: float,
    fps: int,
    chunk_seconds: float,
) -> dict[str, object]:
    if min(duration_seconds, window_seconds, chunk_seconds) <= 0 or fps <= 0:
        raise ValueError("duration, window, chunk, and fps must be positive.")
    timing_profile = load_validated_timing_checkpoint(
        timing_checkpoint_path, sample_rate_hz=125
    )
    morphology_profile = load_morphology_checkpoint(cross_modal_checkpoint_path)
    cross_modal_checkpoint = torch.load(
        cross_modal_checkpoint_path, map_location="cpu", weights_only=False
    )
    records = discover_mimic_perform_af(data_root)
    matches = [record for record in records if record.recording_id == recording_id]
    if len(matches) != 1:
        raise ValueError(f"recording not found or ambiguous: {recording_id}")
    record = matches[0]
    if record.patient_id not in cross_modal_checkpoint["report"]["patients"]["test"]:
        raise ValueError("comparison video must use a held-out test patient.")
    if record.sample_rate_hz != timing_profile.sample_rate_hz:
        raise ValueError("recording and Timing Head sample rates differ.")
    time_values, raw_ppg, raw_ecg = load_mimic_perform_signals(record)
    sample_rate = record.sample_rate_hz
    display_delay_samples = round(
        (timing_profile.output_delay_seconds + morphology_profile.seconds_before_r)
        * sample_rate
    )
    requested_end = round((start_seconds + duration_seconds) * sample_rate)
    processing_end = min(requested_end + display_delay_samples, raw_ppg.size)
    ppg_processor = StatefulCausalWaveformPreprocessor(
        sample_rate_hz=sample_rate,
        scaler=timing_profile.ppg_scaler,
        modality="ppg",
    )
    timing_session = timing_profile.new_session()
    morphology_session = morphology_profile.new_session()
    processed_parts = []
    emitted_parts = []
    valid_parts = []
    chunk_samples = max(round(chunk_seconds * sample_rate), 1)
    for chunk_start in range(0, processing_end, chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, processing_end)
        processed = ppg_processor.process(raw_ppg[chunk_start:chunk_end])
        timing_output = timing_session.process(processed, render_enabled=True)
        events = [
            TimingEvent(
                int(sample_index),
                float(timing_output.event_probability[int(sample_index) - chunk_start]),
            )
            for sample_index in timing_output.event_sample_indices
        ]
        morphology_output = morphology_session.process(
            processed, events, render_enabled=True
        )
        processed_parts.append(processed[0])
        emitted_parts.append(morphology_output.pseudo_ecg)
        valid_parts.append(morphology_output.pseudo_ecg_valid)
    processed_ppg = np.concatenate(processed_parts)
    emitted = np.concatenate(emitted_parts)
    emitted_valid = np.concatenate(valid_parts)
    aligned_count = processing_end - display_delay_samples
    aligned_pseudo = emitted[
        display_delay_samples : display_delay_samples + aligned_count
    ]
    aligned_valid = emitted_valid[
        display_delay_samples : display_delay_samples + aligned_count
    ]
    aligned_pseudo = np.where(aligned_valid, aligned_pseudo, np.nan)
    morphology_checkpoint = torch.load(
        morphology_checkpoint_path, map_location="cpu", weights_only=False
    )
    ecg_processor = StatefulCausalWaveformPreprocessor(
        sample_rate_hz=sample_rate,
        scaler=FixedRobustScaler(**morphology_checkpoint["report"]["ecg_scaler"]),
        modality="ecg",
        bandpass_hz=tuple(
            morphology_checkpoint["config"]["dataset"]["ecg_bandpass_hz"]
        ),
    )
    processed_ecg = ecg_processor.process(raw_ecg[:aligned_count])[0]
    view_start = round(start_seconds * sample_rate)
    view_end = min(requested_end, aligned_count)
    if view_end <= view_start:
        raise ValueError("selected aligned interval is empty.")
    timeline = time_values[view_start:view_end] - time_values[view_start]
    signals = (
        processed_ppg[view_start:view_end],
        aligned_pseudo[view_start:view_end],
        processed_ecg[view_start:view_end],
    )
    labels = (
        "PPG Input — measured pulse waveform",
        "PPG-derived pseudo ECG — non-diagnostic",
        "Original ECG — measured reference only",
    )
    colors = ("#059669", "#7c3aed", "#2563eb")
    limits = [_limits(signal) for signal in signals]
    width, height = 1280, 720
    figure, axes = plt.subplots(3, 1, figsize=(width / 100, height / 100), dpi=100)
    figure.patch.set_facecolor("#f8fafc")
    figure.suptitle(
        "Timing-constrained PPG-derived pseudo ECG comparison",
        fontsize=16,
        fontweight="bold",
    )
    lines = []
    for axis, label, color, ylim in zip(axes, labels, colors, limits, strict=True):
        axis.set_facecolor("white")
        axis.grid(True, alpha=0.22)
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold", color=color)
        axis.set_ylim(*ylim)
        axis.set_ylabel("standardized")
        (line,) = axis.plot([], [], color=color, linewidth=1.6)
        lines.append(line)
    axes[-1].set_xlabel("Aligned physiological time (s)")
    source_label = morphology_profile.morphology_source.replace("_", " ")
    warning = figure.text(
        0.5,
        0.015,
        f"Pseudo ECG source: {source_label}. PPG-derived, not measured ECG, not for diagnosis.",
        ha="center",
        fontsize=10,
        color="#991b1b",
        fontweight="bold",
    )
    warning.set_bbox({"facecolor": "#fee2e2", "edgecolor": "#ef4444", "pad": 5})
    figure.tight_layout(rect=(0.04, 0.055, 0.98, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)
    total_frames = max(round(float(timeline[-1]) * fps), 1)
    preview_path = output_path.with_suffix(".png")
    try:
        for frame_index in range(total_frames):
            cursor = min(frame_index / fps, float(timeline[-1]))
            left = max(0.0, cursor - window_seconds)
            right = max(window_seconds, cursor)
            visible = (timeline >= left) & (timeline <= cursor)
            for axis, line, signal in zip(axes, lines, signals, strict=True):
                line.set_data(timeline[visible], signal[visible])
                axis.set_xlim(left, right)
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.send(np.ascontiguousarray(rgba[:, :, :3]).tobytes())
            if frame_index == min(total_frames - 1, fps * 5):
                figure.savefig(preview_path, dpi=100)
    finally:
        writer.close()
        plt.close(figure)
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "timing_constrained_waveform_comparison_video",
        "output_file": output_path.name,
        "preview_file": preview_path.name,
        "recording_identifier_included": False,
        "patient_identifier_included": False,
        "held_out_test_recording": True,
        "sample_rate_hz": sample_rate,
        "display_delay_seconds": morphology_session.display_delay_seconds,
        "morphology_source": morphology_profile.morphology_source,
        "display_morphology_prior": "rr_adaptive_qrs_st_t_population_prior_v1",
        "repolarization_duration_semantics": (
            "bounded visual prior from previous RR; not measured or reconstructed QT/QTc"
        ),
        "p_wave_generated": False,
        "pq_pr_measurement_supported": False,
        "learned_morphology_selected": morphology_profile.learned_morphology_selected,
        "pseudo_ecg_diagnostic": False,
        "original_ecg_role": "synchronized visual reference only",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-checkpoint", type=Path, required=True)
    parser.add_argument("--cross-modal-checkpoint", type=Path, required=True)
    parser.add_argument("--morphology-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/MIMIC_PERform_Datasets")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/demo/hybrid_waveform_comparison.mp4"),
    )
    parser.add_argument("--recording-id", default="mimic_perform_non_af_007")
    parser.add_argument("--start-seconds", type=float, default=5.0)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--chunk-seconds", type=float, default=0.25)
    arguments = parser.parse_args()
    print(
        json.dumps(
            create_video(
                timing_checkpoint_path=arguments.timing_checkpoint,
                cross_modal_checkpoint_path=arguments.cross_modal_checkpoint,
                morphology_checkpoint_path=arguments.morphology_checkpoint,
                data_root=arguments.data_root,
                output_path=arguments.output,
                recording_id=arguments.recording_id,
                start_seconds=arguments.start_seconds,
                duration_seconds=arguments.duration_seconds,
                window_seconds=arguments.window_seconds,
                fps=arguments.fps,
                chunk_seconds=arguments.chunk_seconds,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
