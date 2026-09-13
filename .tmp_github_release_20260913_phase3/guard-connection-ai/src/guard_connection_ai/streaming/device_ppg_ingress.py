from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from guard_connection_ai.schemas.device_ppg import DevicePPGPacket


class DevicePPGIngressError(ValueError):
    def __init__(self, code: str, message: str, *, expected_sequence_number: int) -> None:
        super().__init__(message)
        self.code = code
        self.expected_sequence_number = expected_sequence_number


@dataclass(frozen=True)
class AdaptedPPGChunk:
    stream_id: str
    sequence_number: int
    source_sample_rate_hz: float
    target_sample_rate_hz: float
    input_timestamp_start_seconds: float
    ppg: np.ndarray
    ppg_valid: np.ndarray
    source_packet: DevicePPGPacket


class StatefulCausalADCNormalizer:
    """Map device ADC counts to a bounded PPG domain without future samples."""

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        adaptation_seconds: float = 2.0,
        output_center: float = 0.5,
        output_scale: float = 0.2,
    ) -> None:
        if min(sample_rate_hz, adaptation_seconds, output_scale) <= 0:
            raise ValueError("normalizer parameters must be positive.")
        self.alpha = 1.0 - np.exp(-1.0 / (sample_rate_hz * adaptation_seconds))
        self.output_center = output_center
        self.output_scale = output_scale
        self.reset()

    def reset(self) -> None:
        self._baseline: float | None = None
        self._deviation: float | None = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        values = np.asarray(chunk, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("chunk must be a non-empty one-dimensional array.")
        output = np.full(values.shape, np.nan, dtype=np.float64)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            if self._baseline is None:
                self._baseline = float(value)
                self._deviation = 1.0
            centered = float(value) - self._baseline
            scale = max(1.4826 * float(self._deviation), 1.0)
            standardized = float(np.clip(centered / scale, -3.0, 3.0))
            output[index] = self.output_center + self.output_scale * standardized
            self._baseline += self.alpha * centered
            self._deviation += self.alpha * (
                abs(centered) - float(self._deviation)
            )
        return output


class StatefulDevicePPGIngress:
    """検証、非破壊3ch融合、因果線形リサンプリングを行うstream adapter。"""

    def __init__(self, *, target_sample_rate_hz: float = 125.0) -> None:
        if target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive.")
        self.target_sample_rate_hz = target_sample_rate_hz
        self._stream_id: str | None = None
        self._expected_sequence_number = 0
        self._next_target_time: float | None = None
        self._previous_time: float | None = None
        self._previous_value: float | None = None
        self._previous_valid = False

    def reset(self) -> None:
        self._stream_id = None
        self._expected_sequence_number = 0
        self._next_target_time = None
        self._previous_time = None
        self._previous_value = None
        self._previous_valid = False

    @staticmethod
    def _fuse(packet: DevicePPGPacket) -> tuple[np.ndarray, np.ndarray]:
        signals = np.asarray(
            [
                [np.nan if value is None else value for value in packet.ppg0],
                [np.nan if value is None else value for value in packet.ppg1],
                [np.nan if value is None else value for value in packet.ppg2],
            ],
            dtype=np.float64,
        )
        for index, name in enumerate(("ppg0", "ppg1", "ppg2")):
            flags = packet.saturation.get(name)
            if flags is not None:
                signals[index, np.asarray(flags, dtype=bool)] = np.nan
        finite_count = np.sum(np.isfinite(signals), axis=0)
        valid = (finite_count >= 2) & (not packet.sensor_disconnected)
        fused = np.full(signals.shape[1], np.nan, dtype=np.float64)
        has_value = finite_count > 0
        fused[has_value] = np.nanmedian(signals[:, has_value], axis=0)
        fused[~valid] = np.nan
        return fused, valid

    def _reject(self, code: str, message: str) -> DevicePPGIngressError:
        return DevicePPGIngressError(
            code, message, expected_sequence_number=self._expected_sequence_number
        )

    def process(self, packet: DevicePPGPacket) -> AdaptedPPGChunk:
        if self._stream_id is None:
            self._stream_id = packet.stream_id
        elif packet.stream_id != self._stream_id:
            raise self._reject("stream_changed", "reset is required before changing stream_id.")
        if packet.sequence_number < self._expected_sequence_number:
            code = (
                "duplicate_packet"
                if packet.sequence_number == self._expected_sequence_number - 1
                else "out_of_order"
            )
            raise self._reject(code, "packet sequence is older than ingress state.")
        if packet.sequence_number > self._expected_sequence_number:
            raise self._reject("sequence_gap", "a device packet is missing.")

        source_rate = packet.configured_sample_rate_hz
        end_time = packet.device_timestamp_end_ns / 1_000_000_000
        source_times = end_time - np.arange(packet.sample_count - 1, -1, -1) / source_rate
        fused, valid = self._fuse(packet)
        if self._previous_time is not None:
            expected_start = self._previous_time + 1 / source_rate
            if abs(source_times[0] - expected_start) > 0.5 / source_rate:
                raise self._reject(
                    "timestamp_gap", "device timestamp is not contiguous with ingress state."
                )
            source_times = np.concatenate(([self._previous_time], source_times))
            fused = np.concatenate(([self._previous_value], fused))
            valid = np.concatenate(([self._previous_valid], valid))

        if self._next_target_time is None:
            self._next_target_time = float(source_times[0])
        output_start_time = self._next_target_time
        step = 1 / self.target_sample_rate_hz
        count = int(np.floor((source_times[-1] - self._next_target_time) / step + 1e-9)) + 1
        if count <= 0:
            target_times = np.empty(0, dtype=np.float64)
        else:
            target_times = self._next_target_time + np.arange(count) * step
        safe_fused = fused.copy()
        finite = np.isfinite(safe_fused)
        if np.any(finite):
            safe_fused[~finite] = np.interp(
                source_times[~finite], source_times[finite], safe_fused[finite]
            )
            output = np.interp(target_times, source_times, safe_fused)
            validity_score = np.interp(
                target_times, source_times, valid.astype(np.float64)
            )
            output_valid = validity_score >= 1.0 - 1e-9
            output[~output_valid] = np.nan
        else:
            output = np.full(target_times.size, np.nan)
            output_valid = np.zeros(target_times.size, dtype=bool)

        self._previous_time = float(source_times[-1])
        self._previous_value = float(fused[-1])
        self._previous_valid = bool(valid[-1])
        self._next_target_time += target_times.size * step
        self._expected_sequence_number += 1
        return AdaptedPPGChunk(
            stream_id=packet.stream_id,
            sequence_number=packet.sequence_number,
            source_sample_rate_hz=source_rate,
            target_sample_rate_hz=self.target_sample_rate_hz,
            input_timestamp_start_seconds=output_start_time,
            ppg=output,
            ppg_valid=output_valid,
            source_packet=packet,
        )
