"""後段AIの3状態推論を受け取るHTTP API。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import care_event_routes
import schemas
from downstream_inferences import DownstreamInferenceError, registry
from realtime_bridge import RealtimeProtocolError
from realtime_bridge import registry as realtime_registry

router = APIRouter()
Broadcaster = Callable[[str, dict], Awaitable[None]]

async def _no_broadcast(_user_id: str, _payload: dict) -> None:
    return None

_broadcast: Broadcaster = _no_broadcast

def configure_broadcaster(broadcaster: Broadcaster) -> None:
    global _broadcast
    _broadcast = broadcaster

def _error_response(error: DownstreamInferenceError) -> JSONResponse:
    body: dict[str, object] = {
        "message_type": "error",
        "schema_version": "downstream_v1",
        "error": {"code": error.code, "message": error.message},
    }
    if error.expected_sequence_number is not None:
        body["error"]["expected_sequence_number"] = error.expected_sequence_number
    status_code = (
        404
        if error.code in {"inference_not_found", "session_not_found"}
        else 409
    )
    return JSONResponse(status_code=status_code, content=body)

def _session_provenance(session_id: str) -> tuple[str, dict[str, object]]:
    try:
        return (
            realtime_registry.user_id_for(session_id),
            realtime_registry.source_provenance_for(session_id),
        )
    except RealtimeProtocolError as error:
        raise DownstreamInferenceError(
            "session_not_found", "realtime session was not found"
        ) from error

def _envelope(payload: dict, *, changed: bool) -> dict:
    user_id, provenance = _session_provenance(str(payload["session_id"]))
    return {
        "message_type": "downstream_inference",
        "schema_version": "downstream_v1",
        "changed": changed,
        "user_id": user_id,
        "source_provenance": provenance,
        "inference": payload,
        "monitoring_summary": registry.summary(str(payload["session_id"])),
    }

@router.post("/downstream-inferences")
async def create_downstream_inference(request: schemas.DownstreamInferenceCreate):
    """推論結果を検証し、confirmed episodeだけを本人確認へ接続する。"""
    try:
        user_id, provenance = _session_provenance(request.session_id)
        if request.inference_mode == "demo_stub" and provenance["source_mode"] == "live_device":
            raise DownstreamInferenceError(
                "invalid_inference_mode",
                "demo stub inference cannot be attached to live device input",
            )
        payload, changed = registry.accept(request.model_dump(mode="json"))
        response = _envelope(payload, changed=changed)
        care_event = None
        if request.decision == "af_suspected" and request.episode.state == "confirmed":
            care_event = await care_event_routes.create_af_suspicion(
                schemas.AFSuspicionCreate(
                    idempotency_key=(
                        f"downstream:{request.session_id}:"
                        f"{request.episode.episode_id}:{request.model_version}"
                    ),
                    user_id=user_id,
                    patient_id=None,
                    session_id=request.session_id,
                    source_mode=provenance["source_mode"],
                    demo_scenario_id=provenance["demo_scenario_id"],
                    af_probability=request.af_probability,
                    uncertainty=None,
                    model_version=request.model_version,
                    inference_mode=request.inference_mode,
                    analysis_window_start_seconds=request.window.start_seconds,
                    analysis_window_end_seconds=request.window.end_seconds,
                    occurred_at_seconds=request.inference_timestamp_utc.timestamp(),
                )
            )
        response["care_event"] = care_event
        if changed:
            await _broadcast(user_id, response)
        return response
    except DownstreamInferenceError as error:
        return _error_response(error)

@router.get("/downstream-inferences/{session_id}/latest")
def get_latest_downstream_inference(session_id: str):
    try:
        _session_provenance(session_id)
        return _envelope(registry.latest(session_id), changed=False)
    except DownstreamInferenceError as error:
        return _error_response(error)
