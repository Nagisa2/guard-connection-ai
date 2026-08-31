from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft


# ============================================================
# Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
    / "bidmc_csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "stft_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FS = 125

SUBJECT_IDS = [1, 10, 25, 40, 53]

WINDOW_SECONDS = 10
WINDOW_SAMPLES = FS * WINDOW_SECONDS

NPERSEG = 64
NOVERLAP = 32
NFFT = 64


# ============================================================
# Load signal
# ============================================================

def load_subject(subject_id: int):
    file_path = DATA_ROOT / f"bidmc_{subject_id:02d}_Signals.csv"

    df = pd.read_csv(file_path)

    ppg = df[" PLETH"].to_numpy(dtype=np.float32)
    ecg = df[" II"].to_numpy(dtype=np.float32)

    return ppg, ecg


# ============================================================
# Compute STFT
# ============================================================

def compute_stft(signal: np.ndarray):
    frequencies, times, zxx = stft(
        signal,
        fs=FS,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        boundary=None,
        padded=False,
    )

    return frequencies, times, zxx


# ============================================================
# Plot
# ============================================================

def plot_stft_variants(
    subject_id: int,
    signal_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    zxx: np.ndarray,
):
    magnitude = np.abs(zxx)

    log_magnitude = np.log1p(magnitude)

    phase = np.angle(zxx)

    real = np.real(zxx)

    imag = np.imag(zxx)

    variants = [
        ("Magnitude", magnitude),
        ("Log Magnitude", log_magnitude),
        ("Phase", phase),
        ("Real", real),
        ("Imaginary", imag),
    ]

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(25, 5),
    )

    for ax, (title, data) in zip(axes, variants):

        image = ax.pcolormesh(
            times,
            frequencies,
            data,
            shading="auto",
        )

        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        fig.colorbar(
            image,
            ax=ax,
        )

    fig.suptitle(
        f"Subject {subject_id:02d} - {signal_name} STFT",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"subject_{subject_id:02d}_{signal_name.lower()}_stft.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Save raw STFT arrays
# ============================================================

def save_stft(
    subject_id: int,
    signal_name: str,
    frequencies: np.ndarray,
    times: np.ndarray,
    zxx: np.ndarray,
):
    output_path = (
        OUTPUT_DIR
        / f"subject_{subject_id:02d}_{signal_name.lower()}_stft.npz"
    )

    np.savez(
        output_path,
        frequencies=frequencies,
        times=times,
        complex_stft=zxx,
        magnitude=np.abs(zxx),
        phase=np.angle(zxx),
        real=np.real(zxx),
        imag=np.imag(zxx),
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("STFT Inspection")
    print("=" * 60)

    print()
    print(f"Sampling rate: {FS} Hz")
    print(f"Window: {WINDOW_SECONDS} seconds")
    print(f"nperseg: {NPERSEG}")
    print(f"noverlap: {NOVERLAP}")
    print(f"nfft: {NFFT}")

    for subject_id in SUBJECT_IDS:

        print()
        print(f"Processing Subject {subject_id:02d}")

        ppg, ecg = load_subject(subject_id)

        ppg_segment = ppg[:WINDOW_SAMPLES]
        ecg_segment = ecg[:WINDOW_SAMPLES]

        for signal_name, signal in [
            ("PPG", ppg_segment),
            ("ECG", ecg_segment),
        ]:

            frequencies, times, zxx = compute_stft(signal)

            print(
                f"  {signal_name}: "
                f"input={signal.shape}, "
                f"STFT={zxx.shape}"
            )

            plot_stft_variants(
                subject_id,
                signal_name,
                frequencies,
                times,
                zxx,
            )

            save_stft(
                subject_id,
                signal_name,
                frequencies,
                times,
                zxx,
            )

    print()
    print("=" * 60)
    print("Completed")
    print("=" * 60)
    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()