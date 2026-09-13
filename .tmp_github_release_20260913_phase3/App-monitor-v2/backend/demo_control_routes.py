"""ローカルデモ操作画面向けAPI。"""

from __future__ import annotations

import schemas
from demo_control import DemoControlError, manager
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/demo-control", tags=["local-demo-control"])

def _error(error: DemoControlError) -> JSONResponse:
    return JSONResponse(
        status_code=409 if error.code == "already_running" else 400,
        content={
            "message_type": "error",
            "schema_version": "1.0",
            "error": {"code": error.code, "message": error.message},
        },
    )

@router.get("/status")
def demo_status():
    return {"message_type": "demo_control_status", **manager.status()}

@router.post("/start")
def start_demo(request: schemas.DemoControlStartRequest, http_request: Request):
    try:
        status = manager.start(
            base_url=str(http_request.base_url).rstrip("/"),
            scenario=request.scenario,
            loop=request.loop,
        )
    except DemoControlError as error:
        return _error(error)
    return {"message_type": "demo_control_status", **status}

@router.post("/stop")
def stop_demo():
    return {"message_type": "demo_control_status", **manager.stop()}
