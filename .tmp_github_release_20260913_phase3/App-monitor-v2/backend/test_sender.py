import argparse
import asyncio
import json
import math
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DATA_DIR = BASE_DIR / "json_output" / "Data"

def load_json_dataset(patient_id: int):
    """json_output/Data のメタデータを読み込み、信号のサンプル数と特徴量を参照する。"""
    ecg_path = JSON_DATA_DIR / f"{patient_id:02d}_ECG_01.json"
    ppg_path = JSON_DATA_DIR / f"{patient_id:02d}_PPG_01.json"

    ecg_meta = {}
    ppg_meta = {}

    for path, target in [(ecg_path, "ECG"), (ppg_path, "PPG")]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                if target == "ECG":
                    ecg_meta = data
                else:
                    ppg_meta = data

    return {
        "ECG": ecg_meta,
        "PPG": ppg_meta,
        "AF_annotation": ecg_meta.get("AF_annotation"),
    }

def get_sample_count(meta: dict, key: str, fallback: int = 120):
    try:
        shape = meta.get(key, {}).get("shape", [fallback, 1])
        if isinstance(shape, list) and len(shape) > 0:
            return max(20, int(shape[0]))
    except (AttributeError, TypeError, ValueError):
        return fallback
    return fallback

def generate_ecg_values(patient_id: int, sample_count: int, cycle_index: int):
    """1拍ごとにP波・QRS・T波の典型パターンを含むECGを生成する。"""
    heart_rate_bpm = 68 + patient_id * 2
    beat_samples = round(100 * 60 / heart_rate_bpm)
    values = []
    for i in range(sample_count):
        beat_phase = ((cycle_index + i) % beat_samples) / beat_samples
        p_wave = 0.12 * math.exp(-((beat_phase - 0.18) ** 2) / 0.006)
        q_wave = -0.22 * math.exp(-((beat_phase - 0.34) ** 2) / 0.002)
        r_wave = 1.10 * math.exp(-((beat_phase - 0.46) ** 2) / 0.0009)
        s_wave = -0.18 * math.exp(-((beat_phase - 0.54) ** 2) / 0.003)
        t_wave = 0.28 * math.exp(-((beat_phase - 0.76) ** 2) / 0.025)
        baseline = 0.05 * math.sin(2 * math.pi * (beat_phase + patient_id * 0.07))
        arrhythmia = 0.15 * math.sin(2 * math.pi * 2.5 * beat_phase + patient_id)
        value = p_wave + q_wave + r_wave + s_wave + t_wave + baseline + arrhythmia
        values.append(round(value, 3))
    return values

def generate_ppg_values(patient_id: int, sample_count: int, cycle_index: int):
    """1拍ごとのPPGパルス波形を生成し、ECGと同じく拍ごとの立ち上がりを明確にする。"""
    heart_rate_bpm = 68 + patient_id * 2
    beat_samples = round(100 * 60 / heart_rate_bpm)
    values = []
    for i in range(sample_count):
        beat_phase = ((cycle_index + i) % beat_samples) / beat_samples
        rise = 0.42 * math.exp(-((beat_phase - 0.18) ** 2) / 0.006)
        systolic = 0.55 * math.exp(-((beat_phase - 0.30) ** 2) / 0.003)
        decay = 0.18 * math.exp(-((beat_phase - 0.62) ** 2) / 0.02)
        baseline = 0.18 + 0.03 * math.sin(2 * math.pi * beat_phase + patient_id)
        value = baseline + rise + systolic + decay
        values.append(round(max(0.08, min(1.1, value)), 3))
    return values

def generate_hr_values(patient_id: int, sample_count: int, cycle_index: int):
    base_rate = 60 + patient_id * 3
    values = []
    for i in range(sample_count):
        wave = math.sin((cycle_index + i) / 50.0)
        rate = base_rate + 5 * wave + (patient_id % 2) * 2
        values.append(round(rate, 1))
    return values

def build_payloads(
    *,
    user_id: str,
    sequence_number: int,
    base_time: int,
    time_deltas: list[int],
    ecg_values: list[float],
    ppg_values: list[float],
    hr_value: float,
    signal_types: set[str],
) -> list[dict]:
    payloads = {
        "ECG": {
            "signal_type": "ECG",
            "waveform_type": "simulated_ecg",
            "diagnostic_ecg": False,
            "display_role": "supplemental_simulated_ecg",
            "schema_version": "legacy-simulator-1",
            "session_id": f"simulator_{user_id}",
            "sequence_number": sequence_number,
            "sample_rate_hz": 100,
            "user_id": user_id,
            "base_timestamp_ms": base_time,
            "time_deltas": time_deltas,
            "values": ecg_values,
            "source": "synthetic_display_template",
        },
        "PPG": {
            "signal_type": "PPG",
            "waveform_type": "simulated_ppg",
            "schema_version": "legacy-simulator-1",
            "session_id": f"simulator_{user_id}",
            "sequence_number": sequence_number,
            "sample_rate_hz": 100,
            "user_id": user_id,
            "base_timestamp_ms": base_time,
            "time_deltas": time_deltas,
            "values": ppg_values,
            "source": "synthetic_display_template",
        },
        "HR": {
            "signal_type": "HR",
            "schema_version": "legacy-simulator-1",
            "session_id": f"simulator_{user_id}",
            "sequence_number": sequence_number,
            "user_id": user_id,
            "base_timestamp_ms": base_time,
            "time_deltas": [time_deltas[-1]],
            "values": [hr_value],
            "source": "synthetic_display_template",
        },
    }
    return [payloads[signal_type] for signal_type in ("ECG", "PPG", "HR") if signal_type in signal_types]

async def send_to_user(
    uri: str,
    user_id: str,
    patient_id: int,
    signal_types: set[str],
):
    import websockets

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to server for {user_id} (patient_id={patient_id}).")
            dataset = load_json_dataset(patient_id)
            ecg_count = get_sample_count(dataset["ECG"], "ECG", 600)
            ppg_count = get_sample_count(dataset["PPG"], "PPG", 600)
            packet_size = min(25, ecg_count, ppg_count)
            stream_start_ms = int(time.time() * 1000)
            cycle_index = 0
            sequence_number = 0
            while True:
                ecg_values = generate_ecg_values(patient_id, packet_size, cycle_index)
                ppg_values = generate_ppg_values(patient_id, packet_size, cycle_index)
                hr_values = generate_hr_values(patient_id, packet_size, cycle_index)
                base_time = stream_start_ms + cycle_index * 10
                time_deltas = [i * 10 for i in range(packet_size)]

                payloads = build_payloads(
                    user_id=user_id,
                    sequence_number=sequence_number,
                    base_time=base_time,
                    time_deltas=time_deltas,
                    ecg_values=ecg_values,
                    ppg_values=ppg_values,
                    hr_value=hr_values[-1],
                    signal_types=signal_types,
                )

                for payload in payloads:
                    await websocket.send(json.dumps(payload))

                print(f"[{user_id}] Sent JSON-based batch data ({packet_size} samples) at {base_time}")
                cycle_index += packet_size
                sequence_number += 1
                await asyncio.sleep(0.25)
    except ConnectionRefusedError:
        print(f"Server is not running for {user_id}.")
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for {user_id}.")

JSON_PATIENT_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

async def main(user_ids: list[str], signal_types: set[str], ws_base_url: str):
    tasks = []
    for user_id in user_ids:
        try:
            patient_id = int(user_id.rsplit("_", maxsplit=1)[-1])
        except ValueError as error:
            raise ValueError(f"user_id must end with a numeric patient id: {user_id}") from error
        tasks.append(
            send_to_user(
                f"{ws_base_url}/ws/device/signal/{user_id}",
                user_id,
                patient_id,
                signal_types,
            )
        )
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=[f"test_user_{patient_id:02d}" for patient_id in JSON_PATIENT_IDS],
    )
    parser.add_argument(
        "--signal-types",
        nargs="+",
        choices=["ECG", "PPG", "HR"],
        default=["ECG", "PPG", "HR"],
    )
    parser.add_argument("--ws-base-url", default="ws://127.0.0.1:8000")
    arguments = parser.parse_args()
    try:
        asyncio.run(
            main(
                arguments.user_ids,
                set(arguments.signal_types),
                arguments.ws_base_url,
            )
        )
    except KeyboardInterrupt:
        print("\nSimulation sender stopped cleanly.", flush=True)
