from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

class LoginRequest(BaseModel):
    login_id: str
    password: str

class LoginResponse(BaseModel):
    token: str
    doctor_id: int
    doctor_name: str
    department: str

class DoctorCreate(BaseModel):
    name: str
    login_id: str
    password: str
    department: Optional[str] = ""

class DoctorResponse(BaseModel):
    id: int
    name: str
    login_id: str
    department: str
    created_at: datetime
    class Config:
        from_attributes = True

class PatientBase(BaseModel):
    user_id: Optional[str] = None
    name: str
    birth_date: Optional[str] = ""
    gender: Optional[str] = ""
    room: Optional[str] = ""
    ward: Optional[str] = ""
    history: Optional[str] = ""
    note: Optional[str] = ""
    doctor_id: Optional[int] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class SensorRecordCreate(BaseModel):
    patient_id: int
    ecg: float
    ppg: float
    hr: float
    status: Optional[str] = "normal"
    is_anomaly: Optional[int] = 0

class SensorRecordResponse(SensorRecordCreate):
    id: int
    measured_at: datetime
    class Config:
        from_attributes = True

class HolterOverview(BaseModel):
    af_likelihood: float
    rr_variability_ms: int
    episode_count: int
    signal_quality: str
    heart_rate_mean_bpm: int
    rr_mean_ms: int

class HolterTrendPoint(BaseModel):
    time: str
    hr: int
    rr_ms: int
    af_risk: float
    rr_variability_ms: int

class HolterEpisode(BaseModel):
    start: str
    duration_min: int
    type: str
    confidence: float

class HolterReportResponse(BaseModel):
    patient_id: int
    session_id: str
    session_label: str
    recording_startday: list
    recording_starttime: list
    num_data_records: int
    ecg_points: int
    rr_points: int
    qrs_points: int
    available_fields: list[str]
    overview: HolterOverview
    trend: list[HolterTrendPoint]
    episodes: list[HolterEpisode]
    summary: str
    raw_metadata: dict

class AlertCreate(BaseModel):
    patient_id: int
    doctor_id: Optional[int] = None
    alert_type: Optional[str] = ""
    message: Optional[str] = ""

class AlertResponse(AlertCreate):
    id: int
    is_read: int
    created_at: datetime
    patient_name: Optional[str] = ""
    class Config:
        from_attributes = True

class RealtimeSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    user_id: str
    session_id: str | None = None
    sample_rate_hz: float
    deployment_profile_id: str = "bidmc_train43_ppg_v1"
    source_mode: Literal[
        "live_device", "recorded_demo_replay", "synthetic_demo"
    ] = "live_device"
    demo_scenario_id: str | None = None

    def model_post_init(self, __context, /) -> None:
        if (
            self.source_mode in {"recorded_demo_replay", "synthetic_demo"}
            and not self.demo_scenario_id
        ):
            raise ValueError("recorded demo replay requires demo_scenario_id")
        if self.source_mode == "live_device" and self.demo_scenario_id is not None:
            raise ValueError("live device input must not declare a demo scenario")

class RealtimeSessionResponse(BaseModel):
    schema_version: Literal["1.0"]
    session_id: str
    user_id: str
    expected_sequence_number: int
    generation_profile_id: str
    deployment_profile: dict
    source_mode: Literal["live_device", "recorded_demo_replay", "synthetic_demo"]
    demo_scenario_id: str | None

class RealtimeChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    sequence_number: int
    input_timestamp_start_seconds: float
    ppg: list[float | None]

class DeviceSQIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skewness: float | None = None
    perfusion_index: float | None = None
    motion_level: float | None = None
    algorithm_version: str | None = None
    window_seconds: float | None = None

class DevicePPGChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    stream_id: str
    sequence_number: int
    configured_sample_rate_hz: float
    device_timestamp_end_ns: int
    ppg0: list[int | None]
    ppg1: list[int | None]
    ppg2: list[int | None]
    ambient0: list[int | None]
    estimated_sample_rate_hz: float | None = None
    source_mode: Literal[
        "live_device", "recorded_demo_replay", "synthetic_demo"
    ] = "live_device"
    demo_scenario_id: str | None = None
    clock_domain: str = "polar_device_ns_since_2000_epoch"
    device_id_hash: str | None = None
    device_model: str = "Polar Verity Sense"
    firmware_version: str | None = None
    sdk_version: str | None = None
    sensor_disconnected: bool = False
    saturation: dict[str, list[bool]] = Field(default_factory=dict)
    device_sqi: DeviceSQIRequest | None = None
    phone_monotonic_timestamp_ns: int | None = None
    phone_utc_timestamp_ms: int | None = None

    def model_post_init(self, __context, /) -> None:
        if (
            self.source_mode in {"recorded_demo_replay", "synthetic_demo"}
            and not self.demo_scenario_id
        ):
            raise ValueError(
                "demo_scenario_id is required for recorded demo replay"
            )
        if self.source_mode == "live_device" and self.demo_scenario_id is not None:
            raise ValueError("live device packets must not declare a demo scenario")

class DeviceMotionChunkRequest(BaseModel):
    """PPGとは別時系列で送る加速度またはジャイロpacket。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["1.0"] = "1.0"
    stream_id: str
    sequence_number: int = Field(ge=0)
    sensor_type: Literal["accelerometer", "gyroscope"]
    units: Literal["mG", "deg/s"]
    configured_sample_rate_hz: float = Field(gt=0)
    estimated_sample_rate_hz: float | None = Field(default=None, gt=0)
    device_timestamp_end_ns: int = Field(ge=0)
    clock_domain: str
    x: list[float | None]
    y: list[float | None]
    z: list[float | None]
    sensor_disconnected: bool = False
    device_id_hash: str | None = None
    device_model: str | None = None
    firmware_version: str | None = None
    sdk_version: str | None = None

    def model_post_init(self, __context, /) -> None:
        if not self.stream_id or not self.clock_domain:
            raise ValueError("stream_id and clock_domain must not be empty")
        lengths = {len(self.x), len(self.y), len(self.z)}
        if len(lengths) != 1 or lengths == {0}:
            raise ValueError("x, y and z must have the same positive length")
        if self.sensor_type == "accelerometer" and self.units != "mG":
            raise ValueError("accelerometer units must be mG")
        if self.sensor_type == "gyroscope" and self.units != "deg/s":
            raise ValueError("gyroscope units must be deg/s")

class DownstreamWindow(BaseModel):
    """後段AIが判定に使用した入力窓。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    input_sequence_id: int = Field(ge=0)
    start_seconds: float
    end_seconds: float
    window_seconds: float = Field(default=30.0, gt=0)
    stride_seconds: float = Field(default=5.0, gt=0)

    def model_post_init(self, __context, /) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError("window end precedes its start")
        actual_duration = self.end_seconds - self.start_seconds
        if abs(actual_duration - self.window_seconds) > 1e-6:
            raise ValueError("window duration does not match window_seconds")

class DownstreamWindowAck(BaseModel):
    """後段AIが処理を完了した未ラベル入力窓のACK。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["front_ai_window_v1"] = "front_ai_window_v1"
    input_sequence_id: int = Field(ge=0)

class DownstreamEpisode(BaseModel):
    """後段AI内部の平滑化状態。診断確率として表示しない。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    state: str
    episode_id: str | None = None
    start_seconds: float | None = None
    duration_seconds: float = Field(default=0.0, ge=0)

    def model_post_init(self, __context, /) -> None:
        if not self.state:
            raise ValueError("episode state must not be empty")
        if self.state == "inactive":
            if self.episode_id is not None or self.start_seconds is not None:
                raise ValueError("inactive episode must not have an id or start time")
        elif not self.episode_id or self.start_seconds is None:
            raise ValueError("active episode requires episode_id and start_seconds")

class DownstreamInferenceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    valid_ratio: float = Field(ge=0, le=1)
    n_valid_beats: int = Field(ge=0)
    sqi_window: float = Field(ge=0, le=1)
    gate_value: float = Field(ge=0, le=1)
    used_morphology: bool
    frontend_model_version: str

class DownstreamInferenceCreate(BaseModel):
    """後段AIの窓単位出力。通知・確定診断そのものではない。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["downstream_v1"] = "downstream_v1"
    session_id: str
    inference_sequence_number: int = Field(ge=0)
    model_version: str
    inference_mode: Literal["model", "demo_stub"] = "model"
    frontend_schema_version: str
    window: DownstreamWindow
    decidable: bool
    abstention_reason: str | None = None
    af_probability: float | None = Field(default=None, ge=0, le=1)
    probability_is_calibrated: bool
    decision: Literal["af_suspected", "no_af_suspected", "undecidable"]
    episode: DownstreamEpisode
    context: DownstreamInferenceContext
    inference_timestamp_utc: datetime

    def model_post_init(self, __context, /) -> None:
        if (
            not self.session_id
            or not self.model_version
            or not self.frontend_schema_version
        ):
            raise ValueError("session and model version fields must not be empty")
        if (
            self.inference_timestamp_utc.tzinfo is None
            or self.inference_timestamp_utc.utcoffset() is None
        ):
            raise ValueError("inference_timestamp_utc must include a timezone")
        if self.decidable:
            if self.decision == "undecidable":
                raise ValueError("decidable inference cannot use undecidable decision")
            if self.af_probability is None:
                raise ValueError("decidable inference requires af_probability")
            if self.abstention_reason is not None:
                raise ValueError("decidable inference must not have abstention_reason")
        else:
            if self.decision != "undecidable":
                raise ValueError("non-decidable inference must use undecidable decision")
            if self.af_probability is not None:
                raise ValueError("non-decidable inference must not retain af_probability")
            if not self.abstention_reason:
                raise ValueError("non-decidable inference requires abstention_reason")

class AFSuspicionCreate(BaseModel):
    """後段AIのAF疑いを連携イベントへ変換する入力。"""

    model_config = ConfigDict(extra="forbid")

    idempotency_key: str
    user_id: str
    patient_id: int | None = None
    session_id: str
    source_mode: Literal["live_device", "recorded_demo_replay", "synthetic_demo"]
    demo_scenario_id: str | None = None
    af_probability: float = Field(ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    model_version: str
    inference_mode: Literal["model", "demo_stub"] = "model"
    analysis_window_start_seconds: float
    analysis_window_end_seconds: float
    occurred_at_seconds: float | None = None

    def model_post_init(self, __context, /) -> None:
        if (
            self.source_mode in {"recorded_demo_replay", "synthetic_demo"}
            and not self.demo_scenario_id
        ):
            raise ValueError("recorded demo replay requires demo_scenario_id")
        if self.source_mode == "live_device" and self.demo_scenario_id is not None:
            raise ValueError("live device input must not declare a demo scenario")
        if self.inference_mode == "demo_stub" and self.source_mode == "live_device":
            raise ValueError("demo stub inference cannot be used with live device input")
        if self.analysis_window_end_seconds < self.analysis_window_start_seconds:
            raise ValueError("analysis window end precedes its start")

class CareEventLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(ge=0)
    captured_at_seconds: float
    provider: str

class CareEventActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idempotency_key: str
    action: Literal[
        "patient_ok",
        "patient_unwell",
        "patient_help",
        "fall_detected",
        "patient_no_response",
        "family_acknowledged",
        "doctor_acknowledged",
        "update_location",
        "resolve",
    ]
    actor_role: Literal["patient", "family", "doctor", "system"]
    occurred_at_seconds: float | None = None
    location: CareEventLocation | None = None

class DemoControlStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    scenario: Literal["af_waiting", "af_unwell", "af_emergency"]
    loop: bool = True
