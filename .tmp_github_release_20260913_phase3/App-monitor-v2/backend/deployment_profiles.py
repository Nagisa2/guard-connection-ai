"""サーバー管理のリアルタイム処理profile。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class RealtimeDeploymentProfile:
    profile_id: str
    artifact_version: str
    artifact_sha256: str
    expected_sample_rate_hz: float
    scaler_center: float
    scaler_scale: float
    pulse_arrival_seconds: float
    display_delay_seconds: float
    quality_window_seconds: float
    clinically_validated: bool
    af_model_connected: bool
    generation_profile_id: str = "interval_template"
    input_domain: str = "dataset_scaled_ppg"
    normalize_device_adc: bool = False
    timing_checkpoint_path: str | None = None
    morphology_checkpoint_path: str | None = None

    def public_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata.pop("scaler_center")
        metadata.pop("scaler_scale")
        metadata.pop("timing_checkpoint_path")
        metadata.pop("morphology_checkpoint_path")
        return metadata

def _artifact_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _default_artifact_path() -> Path:
    configured = os.getenv("GUARD_REALTIME_PROFILE_PATH")
    if configured:
        return Path(configured)
    return (
        Path(__file__).resolve().parents[2]
        / "guard-connection-ai"
        / "artifacts"
        / "realtime"
        / "bidmc_train43_ppg_v1.json"
    )

def load_profile(path: str | Path) -> RealtimeDeploymentProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_digest = payload.get("artifact_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("artifact_sha256", None)
    digest_payload.pop("artifact_version", None)
    if expected_digest != _artifact_digest(digest_payload):
        raise ValueError("deployment profile artifact hash mismatch.")
    scaler = payload["scaler"]
    streaming = payload["streaming"]
    return RealtimeDeploymentProfile(
        profile_id=str(payload["profile_id"]),
        artifact_version=str(payload["artifact_version"]),
        artifact_sha256=str(expected_digest),
        expected_sample_rate_hz=float(payload["expected_sample_rate_hz"]),
        scaler_center=float(scaler["center"]),
        scaler_scale=float(scaler["scale"]),
        pulse_arrival_seconds=float(streaming["pulse_arrival_seconds"]),
        display_delay_seconds=float(streaming["display_delay_seconds"]),
        quality_window_seconds=float(streaming["quality_window_seconds"]),
        clinically_validated=bool(payload["clinically_validated"]),
        af_model_connected=bool(payload["af_model_connected"]),
        generation_profile_id=str(
            payload.get("generation_profile_id", "interval_template")
        ),
        input_domain=str(payload.get("input_domain", "dataset_scaled_ppg")),
        normalize_device_adc=bool(payload.get("normalize_device_adc", False)),
    )

_profile = load_profile(_default_artifact_path())
PROFILES = {_profile.profile_id: _profile}

def _add_stage4_profile_if_available() -> None:
    guard_root = Path(__file__).resolve().parents[2] / "guard-connection-ai"
    timing_path = (
        guard_root
        / "outputs"
        / "timing_head_stage1_fold0_20260905"
        / "timing_head_fold0.pt"
    )
    morphology_path = (
        guard_root
        / "outputs"
        / "cross_modal_morphology_stage3_release_gated_fold0_20260906"
        / "cross_modal_morphology_fold0.pt"
    )
    timing_report_path = timing_path.with_suffix(".json")
    if not all(
        path.is_file()
        for path in (timing_path, morphology_path, timing_report_path)
    ):
        return
    timing_report = json.loads(timing_report_path.read_text(encoding="utf-8"))
    scaler = timing_report["ppg_scaler"]
    digest = hashlib.sha256()
    for path in (timing_path, morphology_path):
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    artifact_sha = digest.hexdigest()
    profile = RealtimeDeploymentProfile(
        profile_id="mimic_fold0_stage4_st_prior_v1",
        artifact_version=f"sha256:{artifact_sha[:16]}",
        artifact_sha256=artifact_sha,
        expected_sample_rate_hz=125.0,
        scaler_center=float(scaler["center"]),
        scaler_scale=float(scaler["scale"]),
        pulse_arrival_seconds=0.0,
        display_delay_seconds=0.66,
        quality_window_seconds=10.0,
        clinically_validated=False,
        af_model_connected=False,
        generation_profile_id="timing_constrained_morphology_v1",
        input_domain="MIMIC_normalized_or_causally_normalized_device_ADC",
        normalize_device_adc=True,
        timing_checkpoint_path=str(timing_path),
        morphology_checkpoint_path=str(morphology_path),
    )
    PROFILES[profile.profile_id] = profile

_add_stage4_profile_if_available()

def get_profile(profile_id: str) -> RealtimeDeploymentProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unknown deployment_profile_id: {profile_id}") from error
