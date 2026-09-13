"""PPGリアルタイム処理のHTTP/WebSocketルート。"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

import schemas
from deployment_profiles import get_profile
from downstream_windows import DownstreamWindowError
from downstream_windows import registry as downstream_window_registry
from motion_ingress import MotionIngressError
from motion_ingress import registry as motion_registry
from realtime_bridge import (
    DevicePPGPacket,
    DeviceSQI,
    RealtimeProtocolError,
    RealtimeSessionConfig,
    device_result_envelope,
    error_envelope,
    registry,
    waveform_envelope,
)

router = APIRouter()
Broadcaster = Callable[[str, dict], Awaitable[None]]

async def _no_broadcast(_user_id: str, _payload: dict) -> None:
    return None

_broadcast: Broadcaster = _no_broadcast
_downstream_connections: dict[str, list[WebSocket]] = {}

def configure_broadcaster(broadcaster: Broadcaster) -> None:
    global _broadcast
    _broadcast = broadcaster

def downstream_output_connection_count() -> int:
    return sum(len(items) for items in _downstream_connections.values())

def _downstream_key_is_valid(received_key: str | None) -> bool:
    expected_key = os.getenv("GUARD_DOWNSTREAM_API_KEY")
    return bool(
        expected_key
        and received_key
        and secrets.compare_digest(received_key, expected_key)
    )

def _window_error_response(error: DownstreamWindowError) -> JSONResponse:
    status_code = 404 if error.code == "window_session_not_found" else 409
    return JSONResponse(
        status_code=status_code,
        content={
            "message_type": "error",
            "schema_version": "front_ai_window_v1",
            "error": {"code": error.code, "message": error.message},
        },
    )

async def _broadcast_downstream(session_id: str, payload: dict) -> None:
    connections = list(_downstream_connections.get(session_id, []))

    async def send(connection: WebSocket):
        try:
            await asyncio.wait_for(connection.send_json(payload), timeout=0.2)
            return None
        except Exception:  # noqa: BLE001 - timeoutと切断を同じ接続不良として除去する
            return connection

    stale = [
        connection
        for connection in await asyncio.gather(*(send(item) for item in connections))
        if connection is not None
    ]
    for connection in stale:
        if connection in _downstream_connections.get(session_id, []):
            _downstream_connections[session_id].remove(connection)

@router.post("/realtime/sessions", response_model=schemas.RealtimeSessionResponse)
def create_realtime_session(request: schemas.RealtimeSessionCreate):
    try:
        profile = get_profile(request.deployment_profile_id)
        if abs(request.sample_rate_hz - profile.expected_sample_rate_hz) > 1e-9:
            raise ValueError(
                f"sample_rate_hz must equal profile rate {profile.expected_sample_rate_hz}."
            )
        result = registry.create(RealtimeSessionConfig(
            session_id=request.session_id,
            user_id=request.user_id,
            sample_rate_hz=request.sample_rate_hz,
            scaler_center=profile.scaler_center,
            scaler_scale=profile.scaler_scale,
            pulse_arrival_seconds=profile.pulse_arrival_seconds,
            display_delay_seconds=profile.display_delay_seconds,
            quality_window_seconds=profile.quality_window_seconds,
            generation_profile_id=profile.generation_profile_id,
            normalize_device_adc=profile.normalize_device_adc,
            source_mode=request.source_mode,
            demo_scenario_id=request.demo_scenario_id,
        ))
        result["deployment_profile"] = profile.public_metadata()
        return result
    except (RealtimeProtocolError, ValueError) as error:
        if isinstance(error, RealtimeProtocolError):
            payload = error_envelope(error)
        else:
            payload = {
                "message_type": "error",
                "schema_version": "1.0",
                "error": {"code": "invalid_profile", "message": str(error)},
            }
        return JSONResponse(status_code=400, content=payload)

@router.delete("/realtime/sessions/{session_id}")
async def end_realtime_session(session_id: str):
    try:
        registry.end(session_id)
    except RealtimeProtocolError as error:
        return JSONResponse(status_code=404, content=error_envelope(error))
    from downstream_inferences import registry as downstream_registry

    downstream_registry.end_session(session_id)
    downstream_window_registry.end_session(session_id)
    motion_registry.end_session(session_id)
    connections = _downstream_connections.pop(session_id, [])
    await asyncio.gather(
        *(connection.close(code=1000) for connection in connections),
        return_exceptions=True,
    )
    return {"schema_version": "1.0", "session_id": session_id, "ended": True}

def _process_realtime_chunk(session_id: str, request: schemas.RealtimeChunkRequest):
    return registry.process_chunk(
        session_id,
        sequence_number=request.sequence_number,
        input_timestamp_start_seconds=request.input_timestamp_start_seconds,
        ppg=request.ppg,
    )

@router.post("/realtime/sessions/{session_id}/chunks")
async def process_realtime_chunk(session_id: str, request: schemas.RealtimeChunkRequest):
    try:
        frame = _process_realtime_chunk(session_id, request)
        payload = waveform_envelope(
            frame,
            source_provenance=registry.source_provenance_for(session_id),
        )
        await _broadcast(registry.user_id_for(session_id), payload)
        return payload
    except RealtimeProtocolError as error:
        status_code = 404 if error.code == "session_not_found" else 409
        return JSONResponse(status_code=status_code, content=error_envelope(error))

@router.post("/realtime/sessions/{session_id}/device-ppg-chunks")
async def process_device_ppg_chunk(
    session_id: str, request: schemas.DevicePPGChunkRequest
):
    """Android想定の4ch PPG packetを前段AIと医師表示へ分岐する。"""
    try:
        values = request.model_dump()
        device_sqi = values.pop("device_sqi")
        packet = DevicePPGPacket(
            **values,
            device_sqi=DeviceSQI(**device_sqi) if device_sqi else None,
        )
        result = registry.process_device_packet(session_id, packet)
        payload = device_result_envelope(result)
        try:
            windows = downstream_window_registry.ingest_ppg(
                session_id, payload["downstream_ai"]
            )
            payload["downstream_window_status"] = {
                "emitted_count": len(windows),
                "latest_input_sequence_id": (
                    None if not windows else windows[-1]["input_sequence_id"]
                ),
            }
        except DownstreamWindowError as error:
            payload["downstream_window_status"] = {
                "emitted_count": 0,
                "error": {"code": error.code, "message": error.message},
            }
        await _broadcast_downstream(session_id, payload["downstream_ai"])
        if result.waveform_frame is not None:
            await _broadcast(
                registry.user_id_for(session_id),
                waveform_envelope(
                    result.waveform_frame,
                    source_provenance=registry.source_provenance_for(session_id),
                ),
            )
        return payload
    except (RealtimeProtocolError, ValueError) as error:
        if isinstance(error, RealtimeProtocolError):
            payload = error_envelope(error)
        else:
            payload = {
                "message_type": "error",
                "schema_version": "1.0",
                "error": {"code": "invalid_device_packet", "message": str(error)},
            }
        status_code = 404 if getattr(error, "code", None) == "session_not_found" else 409
        return JSONResponse(status_code=status_code, content=payload)

@router.post("/realtime/sessions/{session_id}/device-motion-chunks")
async def process_device_motion_chunk(
    session_id: str, request: schemas.DeviceMotionChunkRequest
):
    """異なる周波数・clock domainの慣性センサーを別ストリームで受信する。"""
    try:
        provenance = registry.source_provenance_for(session_id)
        payload = motion_registry.accept(session_id, request.model_dump())
        payload["source_provenance"] = provenance
        try:
            downstream_window_registry.ingest_motion(session_id, payload)
        except DownstreamWindowError as error:
            payload["downstream_window_error"] = {
                "code": error.code,
                "message": error.message,
            }
        await _broadcast_downstream(session_id, payload)
        return payload
    except RealtimeProtocolError as error:
        return JSONResponse(status_code=404, content=error_envelope(error))
    except MotionIngressError as error:
        return JSONResponse(
            status_code=409,
            content={
                "message_type": "error",
                "schema_version": "1.0",
                "error": {
                    "code": error.code,
                    "message": str(error),
                    "expected_sequence_number": error.expected_sequence_number,
                },
            },
        )

@router.get("/front-ai/windows/{session_id}/next")
def get_next_downstream_window(
    session_id: str,
    x_downstream_api_key: str | None = Header(default=None),
):
    """後段AIへ未ACKの最古窓を返す。再取得してもACKまでは同じ窓を返す。"""
    if not _downstream_key_is_valid(x_downstream_api_key):
        return JSONResponse(
            status_code=401,
            content={
                "message_type": "error",
                "schema_version": "front_ai_window_v1",
                "error": {
                    "code": "downstream_authentication_failed",
                    "message": "a valid downstream API key is required",
                },
            },
        )
    try:
        registry.user_id_for(session_id)
    except RealtimeProtocolError as error:
        return JSONResponse(status_code=404, content=error_envelope(error))
    window = downstream_window_registry.next_window(session_id)
    if window is None:
        return Response(status_code=204)
    return window

@router.post("/front-ai/windows/{session_id}/ack")
def acknowledge_downstream_window(
    session_id: str,
    request: schemas.DownstreamWindowAck,
    x_downstream_api_key: str | None = Header(default=None),
):
    """処理済み窓を順番にACKする。同じACKの再送は成功として扱う。"""
    if not _downstream_key_is_valid(x_downstream_api_key):
        return JSONResponse(
            status_code=401,
            content={
                "message_type": "error",
                "schema_version": "front_ai_window_v1",
                "error": {
                    "code": "downstream_authentication_failed",
                    "message": "a valid downstream API key is required",
                },
            },
        )
    try:
        registry.user_id_for(session_id)
        changed = downstream_window_registry.acknowledge(
            session_id, request.input_sequence_id
        )
        return {
            "message_type": "front_ai_window_ack",
            "schema_version": "front_ai_window_v1",
            "session_id": session_id,
            "input_sequence_id": request.input_sequence_id,
            "changed": changed,
        }
    except RealtimeProtocolError as error:
        return JSONResponse(status_code=404, content=error_envelope(error))
    except DownstreamWindowError as error:
        return _window_error_response(error)

@router.websocket("/ws/front-ai/downstream/{session_id}")
async def downstream_ai_output_ws(websocket: WebSocket, session_id: str):
    """APIキーで保護した後段AI向け前段出力ストリーム。"""
    received_key = websocket.headers.get("x-downstream-api-key")
    await websocket.accept()
    if not _downstream_key_is_valid(received_key):
        await websocket.send_json({
            "message_type": "error",
            "schema_version": "1.0",
            "error": {
                "code": "downstream_authentication_failed",
                "message": "a valid downstream API key is required",
            },
        })
        await websocket.close(code=4401)
        return
    try:
        registry.user_id_for(session_id)
    except RealtimeProtocolError as error:
        await websocket.send_json(error_envelope(error))
        await websocket.close(code=4404)
        return
    _downstream_connections.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return
    finally:
        if websocket in _downstream_connections.get(session_id, []):
            _downstream_connections[session_id].remove(websocket)

@router.websocket("/ws/realtime/{session_id}")
async def realtime_processing_ws(websocket: WebSocket, session_id: str):
    """順序検証付きPPG chunkを処理し、同じ接続と医師画面へ結果を返す。"""
    await websocket.accept()
    try:
        registry.user_id_for(session_id)
    except RealtimeProtocolError as error:
        await websocket.send_json(error_envelope(error))
        await websocket.close(code=4404)
        return
    while True:
        try:
            request = schemas.RealtimeChunkRequest(**await websocket.receive_json())
            frame = _process_realtime_chunk(session_id, request)
            payload = waveform_envelope(
                frame,
                source_provenance=registry.source_provenance_for(session_id),
            )
            await websocket.send_json(payload)
            await _broadcast(registry.user_id_for(session_id), payload)
        except WebSocketDisconnect:
            return
        except RealtimeProtocolError as error:
            await websocket.send_json(error_envelope(error))
        except (TypeError, ValueError, ValidationError) as error:
            await websocket.send_json({
                "message_type": "error",
                "schema_version": "1.0",
                "error": {"code": "invalid_request", "message": str(error)},
            })
