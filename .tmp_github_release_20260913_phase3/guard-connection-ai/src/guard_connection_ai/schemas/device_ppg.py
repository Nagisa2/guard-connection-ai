from __future__ import annotations

from dataclasses import dataclass, field

PPG_CHANNEL_NAMES = ("ppg0", "ppg1", "ppg2", "ambient0")
SOURCE_MODES = ("live_device", "recorded_demo_replay", "synthetic_demo")


@dataclass(frozen=True)
class DeviceSQI:
    skewness: float | None = None
    perfusion_index: float | None = None
    motion_level: float | None = None
    algorithm_version: str | None = None
    window_seconds: float | None = None


@dataclass(frozen=True)
class DevicePPGPacket:
    """患者側から受け取る未正規化・未統合のPPG packet。"""

    schema_version: str
    stream_id: str
    sequence_number: int
    configured_sample_rate_hz: float
    device_timestamp_end_ns: int
    ppg0: list[int | None]
    ppg1: list[int | None]
    ppg2: list[int | None]
    ambient0: list[int | None]
    estimated_sample_rate_hz: float | None = None
    source_mode: str = "live_device"
    demo_scenario_id: str | None = None
    clock_domain: str = "polar_device_ns_since_2000_epoch"
    device_id_hash: str | None = None
    device_model: str = "Polar Verity Sense"
    firmware_version: str | None = None
    sdk_version: str | None = None
    sensor_disconnected: bool = False
    saturation: dict[str, list[bool]] = field(default_factory=dict)
    device_sqi: DeviceSQI | None = None
    phone_monotonic_timestamp_ns: int | None = None
    phone_utc_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported device PPG schema_version.")
        if not self.stream_id:
            raise ValueError("stream_id must not be empty.")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative.")
        if self.configured_sample_rate_hz <= 0:
            raise ValueError("configured_sample_rate_hz must be positive.")
        if self.device_timestamp_end_ns < 0:
            raise ValueError("device_timestamp_end_ns must be non-negative.")
        channels = (self.ppg0, self.ppg1, self.ppg2, self.ambient0)
        lengths = {len(values) for values in channels}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("all PPG channels must have the same positive length.")
        if self.estimated_sample_rate_hz is not None and self.estimated_sample_rate_hz <= 0:
            raise ValueError("estimated_sample_rate_hz must be positive when present.")
        if self.source_mode not in SOURCE_MODES:
            raise ValueError("unsupported source_mode.")
        if self.source_mode in {"recorded_demo_replay", "synthetic_demo"} and not self.demo_scenario_id:
            raise ValueError(
                "demo_scenario_id is required for recorded demo replay."
            )
        if self.source_mode == "live_device" and self.demo_scenario_id is not None:
            raise ValueError("live device packets must not declare a demo scenario.")
        unknown = set(self.saturation) - set(PPG_CHANNEL_NAMES)
        if unknown:
            raise ValueError(f"unknown saturation channel: {min(unknown)}.")
        sample_count = len(self.ppg0)
        if any(len(flags) != sample_count for flags in self.saturation.values()):
            raise ValueError("saturation flags must match the PPG sample count.")
        if (self.phone_monotonic_timestamp_ns is None) != (
            self.phone_utc_timestamp_ms is None
        ):
            raise ValueError("phone monotonic and UTC anchors must be provided together.")

    @property
    def sample_count(self) -> int:
        return len(self.ppg0)

    def channels(self) -> dict[str, list[int | None]]:
        return {name: getattr(self, name) for name in PPG_CHANNEL_NAMES}
