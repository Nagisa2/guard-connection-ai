"""後段AF判定から患者・家族・医師連携へ渡すAPI。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import schemas
from care_events import CareEventError, CareEventRegistry, registry
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()
Broadcaster = Callable[[str, dict], Awaitable[None]]

async def _no_broadcast(_user_id: str, _payload: dict) -> None:
    return None

_broadcast: Broadcaster = _no_broadcast

def configure_broadcaster(broadcaster: Broadcaster) -> None:
    global _broadcast
    _broadcast = broadcaster

def configure_registry(configured_registry: CareEventRegistry) -> None:
    global registry
    registry = configured_registry

def _event_envelope(event: dict, *, changed: bool) -> dict:
    return {
        "message_type": "care_event",
        "schema_version": "1.0",
        "changed": changed,
        "event": event,
    }

def _error_response(error: CareEventError) -> JSONResponse:
    status = 404 if error.code == "event_not_found" else 409
    if error.code.startswith("invalid_"):
        status = 400
    return JSONResponse(
        status_code=status,
        content={
            "message_type": "error",
            "schema_version": "1.0",
            "error": {"code": error.code, "message": error.message},
        },
    )

@router.post("/care-events/af-suspicions")
async def create_af_suspicion(request: schemas.AFSuspicionCreate):
    """後段AIが確定したAF疑いだけを連携イベントへ変換する。"""
    try:
        event, created = registry.create_af_suspicion(**request.model_dump())
    except CareEventError as error:
        return _error_response(error)
    payload = _event_envelope(event, changed=created)
    if created:
        await _broadcast(event["user_id"], payload)
    return payload

@router.post("/care-events/{event_id}/actions")
async def apply_care_event_action(
    event_id: str, request: schemas.CareEventActionRequest
):
    try:
        values = request.model_dump()
        location = values.pop("location")
        event, changed = registry.apply_action(
            event_id,
            **values,
            location=location,
        )
    except CareEventError as error:
        return _error_response(error)
    payload = _event_envelope(event, changed=changed)
    if changed:
        await _broadcast(event["user_id"], payload)
    return payload

@router.get("/care-events/{event_id}")
def get_care_event(event_id: str):
    try:
        event = registry.get(event_id)
    except CareEventError as error:
        return _error_response(error)
    return _event_envelope(event, changed=False)

@router.get("/care-events")
def list_care_events(user_id: str):
    return {
        "message_type": "care_event_list",
        "schema_version": "1.0",
        "events": registry.list_for_user(user_id),
    }
