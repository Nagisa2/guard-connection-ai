"""guard-connection-ai のリアルタイム処理を医師側APIへ接続する薄い層。"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

def _load_guard_package() -> None:
    try:
        import guard_connection_ai  # noqa: F401
        return
    except ImportError:
        configured = os.getenv("GUARD_AI_SRC")
        source = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[2] / "guard-connection-ai" / "src"
        )
        if not source.is_dir():
            raise RuntimeError(
                "guard-connection-ai が見つかりません。pip install -e または GUARD_AI_SRC を設定してください。"
            )
        sys.path.insert(0, str(source))

_load_guard_package()
_realtime_api = import_module("guard_connection_ai.streaming.realtime_api")
_front_ai_outputs = import_module("guard_connection_ai.schemas.front_ai_outputs")
_device_ppg = import_module("guard_connection_ai.schemas.device_ppg")
RealtimeProtocolError = _realtime_api.RealtimeProtocolError
RealtimeSessionConfig = _realtime_api.RealtimeSessionConfig
RealtimeSessionRegistry = _realtime_api.RealtimeSessionRegistry
DevicePPGPacket = _device_ppg.DevicePPGPacket
DeviceSQI = _device_ppg.DeviceSQI

def _pipeline_factories():
    from deployment_profiles import PROFILES

    factories = {}
    try:
        timing_checkpoint = import_module(
            "guard_connection_ai.deployment.timing_checkpoint"
        )
        morphology_checkpoint = import_module(
            "guard_connection_ai.deployment.morphology_checkpoint"
        )
        hybrid_session = import_module(
            "guard_connection_ai.streaming.realtime_hybrid_session"
        )
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return factories
        raise
    for profile in PROFILES.values():
        if profile.generation_profile_id == "interval_template":
            continue
        if not profile.timing_checkpoint_path or not profile.morphology_checkpoint_path:
            continue
        timing = timing_checkpoint.load_validated_timing_checkpoint(
            profile.timing_checkpoint_path, sample_rate_hz=profile.expected_sample_rate_hz
        )
        morphology = morphology_checkpoint.load_morphology_checkpoint(
            profile.morphology_checkpoint_path
        )

        def factory(config, session_id, timing=timing, morphology=morphology):
            if abs(config.sample_rate_hz - timing.sample_rate_hz) > 1e-9:
                raise ValueError("Timing Head sample rate does not match the session.")
            if abs(config.sample_rate_hz - morphology.sample_rate_hz) > 1e-9:
                raise ValueError("morphology sample rate does not match the session.")
            return hybrid_session.RealtimeHybridPPGSession(
                session_id=session_id,
                sample_rate_hz=config.sample_rate_hz,
                ppg_scaler=timing.ppg_scaler,
                timing_session=timing.new_session(),
                morphology_session=morphology.new_session(),
                morphology_source=morphology.morphology_source,
                quality_window_seconds=config.quality_window_seconds,
            )

        factories[profile.generation_profile_id] = factory
    return factories

registry = RealtimeSessionRegistry(pipeline_factories=_pipeline_factories())

def waveform_envelope(frame, *, source_provenance: dict[str, object] | None = None) -> dict[str, object]:
    visualization = _front_ai_outputs.to_doctor_visualization_frame(frame)
    payload = {
        "message_type": "waveform_frame",
        "schema_version": registry.schema_version,
        "frame": visualization.to_dict(),
    }
    if source_provenance is not None:
        payload["source_provenance"] = source_provenance
    return payload

def device_result_envelope(result) -> dict[str, object]:
    visualization = (
        _front_ai_outputs.to_doctor_visualization_frame(result.waveform_frame)
        if result.waveform_frame is not None
        else None
    )
    return {
        "message_type": "device_ppg_result",
        "schema_version": "1.0",
        "device_sequence_number": result.packet.sequence_number,
        "resampled_sample_count": int(result.adapted.ppg.size),
        "source_provenance": {
            "source_mode": result.packet.source_mode,
            "demo_scenario_id": result.packet.demo_scenario_id,
        },
        "downstream_ai": result.downstream_payload(),
        "visualization": visualization.to_dict() if visualization else None,
    }

def error_envelope(error: RealtimeProtocolError) -> dict[str, object]:
    return {
        "message_type": "error",
        "schema_version": registry.schema_version,
        "error": error.to_dict(),
    }
