import asyncio  # 非同期処理ライブラリ
import random
import math
import time
import os
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, Depends, HTTPException, status  # Webサーバーのフレームワーク
from fastapi.middleware.cors import CORSMiddleware  # CORS（クロスオリジンリソース共有）を設定するミドルウェア
from sqlalchemy import create_engine    # create_engine: SQLAlchemyでデータベース接続を作成する関数
from sqlalchemy.orm import sessionmaker, Session

import models, crud, schemas, auth
from realtime_routes import (
    configure_broadcaster,
    downstream_output_connection_count,
    router as realtime_router,
)
from care_event_routes import (
    configure_broadcaster as configure_care_event_broadcaster,
    configure_registry as configure_care_event_registry,
    router as care_event_router,
)
from care_events import CareEventRegistry
from demo_control import manager as demo_control_manager
from demo_control_routes import router as demo_control_router
from downstream_inference_routes import (
    configure_broadcaster as configure_downstream_inference_broadcaster,
    router as downstream_inference_router,
)
from downstream_inferences import registry as downstream_inference_registry
from downstream_windows import registry as downstream_window_registry
from realtime_bridge import registry as realtime_session_registry
from gzip_request import GzipRequestMiddleware
from motion_ingress import registry as motion_ingress_registry

active_connections = {}  # {user_id: [websocket1, ...]}

async def _broadcast(user_id: str, payload: dict) -> None:
    connections = list(active_connections.get(user_id, []))

    async def send(connection):
        try:
            await asyncio.wait_for(connection.send_json(payload), timeout=0.2)
            return None
        except Exception:  # noqa: BLE001 - 切断・timeoutを同じstale接続として除去する
            return connection

    stale = [
        connection
        for connection in await asyncio.gather(*(send(item) for item in connections))
        if connection is not None
    ]
    for connection in stale:
        if connection in active_connections.get(user_id, []):
            active_connections[user_id].remove(connection)

BACKEND_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = BACKEND_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "heart_monitor.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
models.Base.metadata.create_all(bind=engine)

JSON_DATA_DIR = Path(__file__).resolve().parent.parent / "json_output" / "Data"

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _load_holter_sample(patient_id: int):
    sample_no = ((patient_id - 1) % 8) + 1
    file_name = f"{sample_no:02d}_ECG_01.json"
    file_path = JSON_DATA_DIR / file_name
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def _build_holter_report(patient_id: int):
    raw = _load_holter_sample(patient_id)
    ecg_shape = raw.get("ECG", {}).get("shape", [0, 1])
    rr_shape = raw.get("rr", {}).get("shape", [0, 1])
    qrs_shape = raw.get("QRSindex", {}).get("shape", [0, 1])
    total_records = _safe_int(raw.get("num_data_records", 0), 0)

    ecg_points = _safe_int(ecg_shape[0], 0)
    rr_points = _safe_int(rr_shape[0], 0)
    qrs_points = _safe_int(qrs_shape[0], 0)
    baseline_rr = 800 + (patient_id % 6) * 35
    rr_variability = 55 + (patient_id % 5) * 12
    af_likelihood = round(0.12 + (patient_id % 5) * 0.14 + (0.08 if qrs_points > 500000 else 0.0), 3)
    episode_count = 1 + (patient_id % 3)
    signal_quality = "良好" if total_records > 500000 else "要確認"

    trend = []
    for hour in range(24):
        hour_value = hour / 24
        hr = 58 + (patient_id % 4) * 4 + int(math.sin(hour_value * 2 * math.pi + patient_id) * 8)
        rr_ms = baseline_rr + int(math.cos(hour_value * 3 * math.pi + patient_id) * rr_variability)
        risk = max(0.0, min(1.0, af_likelihood + (0.25 * math.sin(hour_value * math.pi + patient_id))))
        trend.append({
            "time": f"{hour:02d}:00",
            "hr": hr,
            "rr_ms": rr_ms,
            "af_risk": round(risk, 3),
            "rr_variability_ms": max(35, rr_variability + int(math.sin(hour_value * math.pi) * 12)),
        })

    episodes = []
    for i in range(episode_count):
        start_hour = (i * 7 + 2 + patient_id) % 24
        duration = 6 + (i % 3) * 4
        episodes.append({
            "start": f"{start_hour:02d}:00",
            "duration_min": duration,
            "type": "AF suspected",
            "confidence": round(min(0.99, af_likelihood + 0.25 + i * 0.1), 3),
        })

    summary = (
        f"対象患者は{signal_quality}の信号品質を維持し、"
        f"RR間隔のばらつきは{rr_variability} ms前後で推移し、"
        f"AF発症リスクは{af_likelihood:.2f}の水準です。"
        f"過去24時間では{episode_count}回のAF疑いイベントが確認され、"
        f"特に夜間〜早朝にリスクが高くなっています。"
    )

    available_fields = list(raw.keys()) if raw else []
    return {
        "patient_id": patient_id,
        "session_id": f"holter_session_{patient_id:02d}",
        "session_label": "長時間心電図セッション",
        "recording_startday": raw.get("recording_startday", []),
        "recording_starttime": raw.get("recording_starttime", []),
        "num_data_records": total_records,
        "ecg_points": ecg_points,
        "rr_points": rr_points,
        "qrs_points": qrs_points,
        "available_fields": available_fields,
        "overview": {
            "af_likelihood": af_likelihood,
            "rr_variability_ms": rr_variability,
            "episode_count": episode_count,
            "signal_quality": signal_quality,
            "heart_rate_mean_bpm": 68 + (patient_id % 5) * 3,
            "rr_mean_ms": baseline_rr,
        },
        "trend": trend,
        "episodes": episodes,
        "summary": summary,
        "raw_metadata": {
            "ECG_shape": ecg_shape,
            "rr_shape": rr_shape,
            "QRSindex_shape": qrs_shape,
            "signal_labels": raw.get("signalHeader", {}).get("signal_labels", [])
        },
    }

app = FastAPI(title="心拍モニタリング API v2")
app.add_middleware(GzipRequestMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(realtime_router)
configure_broadcaster(_broadcast)
app.include_router(care_event_router)
configure_care_event_broadcaster(_broadcast)
app.include_router(demo_control_router)
app.include_router(downstream_inference_router)
configure_downstream_inference_broadcaster(_broadcast)
CARE_EVENT_DATABASE_PATH = Path(
    os.getenv("CARE_EVENT_DB_PATH", str(DATABASE_DIR / "care_events.db"))
).resolve()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/demo-control/system-status")
def demo_system_status(db: Session = Depends(get_db)):
    """操作パネル用の非機微な稼働状況。診断状態は返さない。"""
    realtime = realtime_session_registry.status_snapshot()
    downstream = downstream_inference_registry.status_snapshot()
    motion = motion_ingress_registry.status_snapshot()
    windows = downstream_window_registry.status_snapshot()
    patient_count = db.query(models.Patient).count()
    doctor_view_connections = sum(len(items) for items in active_connections.values())
    stale_session_count = sum(
        1
        for item in realtime["sessions"]
        if item["last_input_age_seconds"] is not None
        and item["last_input_age_seconds"] > 2.0
    )
    inference_modes = sorted({
        item["inference_mode"]
        for item in downstream["sessions"]
        if item["inference_mode"]
    })
    if not inference_modes:
        downstream_operating_mode = "unconnected"
    elif len(inference_modes) == 1:
        downstream_operating_mode = inference_modes[0]
    else:
        downstream_operating_mode = "mixed"
    return {
        "message_type": "demo_system_status",
        "schema_version": "1.0",
        "ready": patient_count >= 2,
        "patient_count": patient_count,
        "doctor_view_connection_count": doctor_view_connections,
        "realtime": realtime,
        "downstream": downstream,
        "motion": motion,
        "downstream_windows": windows,
        "inference_modes": inference_modes,
        "downstream_operating_mode": downstream_operating_mode,
        "stale_session_count": stale_session_count,
        "interfaces": {
            "device_ppg_ingress": {
                "ready": True,
                "method": "POST",
                "path_template": "/realtime/sessions/{session_id}/device-ppg-chunks",
                "schema_version": "1.0",
            },
            "device_motion_ingress": {
                "ready": True,
                "method": "POST",
                "path_template": "/realtime/sessions/{session_id}/device-motion-chunks",
                "schema_version": "1.0",
            },
            "downstream_ai_result_ingress": {
                "ready": True,
                "method": "POST",
                "path": "/downstream-inferences",
                "schema_version": "downstream_v1",
            },
            "front_ai_downstream_output": {
                "ready": bool(os.getenv("GUARD_DOWNSTREAM_API_KEY")),
                "connected_consumer_count": downstream_output_connection_count(),
                "transport": "WebSocket",
                "path_template": "/ws/front-ai/downstream/{session_id}",
                "schema_version": "1.0",
                "payload_type": "front_ai_device_handoff",
                "authentication": "x-downstream-api-key",
            },
            "front_ai_window_output": {
                "ready": bool(os.getenv("GUARD_DOWNSTREAM_API_KEY")),
                "transport": "HTTP pull with ACK",
                "next_path_template": "/front-ai/windows/{session_id}/next",
                "ack_path_template": "/front-ai/windows/{session_id}/ack",
                "schema_version": "front_ai_window_v1",
                "window_seconds": 30.0,
                "stride_seconds": 5.0,
                "authentication": "x-downstream-api-key",
            },
            "doctor_waveform_stream": {
                "ready": True,
                "transport": "WebSocket",
                "path_template": "/ws/sensors/{user_id}",
                "schema_version": "1.0",
            },
        },
    }

PATIENT_SENSOR_CONFIG = {
    1: {"ecg_noise": 0.1, "ppg_noise": 0.05, "anomaly_prob": 0.00},  # 田中 太郎: 正常
    2: {"ecg_noise": 0.3, "ppg_noise": 0.1,  "anomaly_prob": 0.15},  # 佐藤 花子: 異常発生しやすい
    3: {"ecg_noise": 0.1, "ppg_noise": 0.2,  "anomaly_prob": 0.05},  # 鈴木 一郎: 少し異常あり
}

ECG_UPPER = 1.5
ECG_LOWER = -1.5
PPG_UPPER = 0.85
PPG_LOWER = 0.15

DEFAULT_PATIENT_SEEDS = [
    {
        "user_id": "test_user_01",
        "name": "田中 太郎",
        "birth_date": "1956-04-12",
        "gender": "男性",
        "room": "101",
        "ward": "A棟",
        "doctor_id": 1,
        "history": "高血圧、動悸の既往",
        "note": "安静時の心拍は安定。AF疑いの追跡対象。"
    },
    {
        "user_id": "test_user_02",
        "name": "佐藤 花子",
        "birth_date": "1950-08-23",
        "gender": "女性",
        "room": "203",
        "ward": "A棟",
        "doctor_id": 1,
        "history": "糖尿病、睡眠時無呼吸",
        "note": "夜間のRR変動が大きく、早朝のイベントが目立つ。"
    },
    {
        "user_id": "test_user_03",
        "name": "鈴木 一郎",
        "birth_date": "1943-01-05",
        "gender": "男性",
        "room": "315",
        "ward": "B棟",
        "doctor_id": 2,
        "history": "心房細動既往、肥満",
        "note": "定期的なホルター監視が必要。"
    },
    {
        "user_id": "test_user_04",
        "name": "山本 涼",
        "birth_date": "1968-11-14",
        "gender": "男性",
        "room": "104",
        "ward": "A棟",
        "doctor_id": 1,
        "history": "不整脈、喫煙",
        "note": "運動後の頻脈が見られる。"
    },
    {
        "user_id": "test_user_05",
        "name": "井上 美咲",
        "birth_date": "1979-02-07",
        "gender": "女性",
        "room": "212",
        "ward": "B棟",
        "doctor_id": 2,
        "history": "甲状腺機能亢進症",
        "note": "休息時の心拍数がやや高め。"
    },
    {
        "user_id": "test_user_06",
        "name": "中村 健太",
        "birth_date": "1986-07-19",
        "gender": "男性",
        "room": "406",
        "ward": "B棟",
        "doctor_id": 2,
        "history": "不眠、ストレス負荷",
        "note": "AFリスクは中等度。夜間に変動あり。"
    },
    {
        "user_id": "test_user_07",
        "name": "渡辺 直子",
        "birth_date": "1949-05-30",
        "gender": "女性",
        "room": "308",
        "ward": "A棟",
        "doctor_id": 1,
        "history": "高血圧、慢性閉塞性肺疾患",
        "note": "信号品質は良好だが、RR変動が大きい。"
    },
    {
        "user_id": "test_user_08",
        "name": "小林 勇太",
        "birth_date": "1972-09-03",
        "gender": "男性",
        "room": "502",
        "ward": "C棟",
        "doctor_id": 2,
        "history": "心筋症の疑い",
        "note": "長時間データに異常波形が含まれる。"
    },
]

@app.on_event("startup")
def startup_event():
    configure_care_event_registry(CareEventRegistry(CARE_EVENT_DATABASE_PATH))
    db = SessionLocal()
    try:
        if db.query(models.Doctor).count() == 0:
            crud.create_doctor(db, schemas.DoctorCreate(
                name="山田 太郎", login_id="yamada",
                password="password123", department="循環器内科"
            ))
            crud.create_doctor(db, schemas.DoctorCreate(
                name="鈴木 花子", login_id="suzuki",
                password="password123", department="内科"
            ))
            print("医師初期データを投入しました")

        existing_user_ids = {row[0] for row in db.query(models.Patient.user_id).all() if row[0]}
        inserted_count = 0
        for seed in DEFAULT_PATIENT_SEEDS:
            if seed["user_id"] in existing_user_ids:
                continue
            crud.create_patient(db, schemas.PatientCreate(**seed))
            existing_user_ids.add(seed["user_id"])
            inserted_count += 1

        if inserted_count > 0:
            print(f"患者初期データを {inserted_count} 人補完しました")
        else:
            print(f"患者初期データは既に {len(DEFAULT_PATIENT_SEEDS)} 人存在します")
    finally:
        db.close()

@app.on_event("shutdown")
def shutdown_event():
    demo_control_manager.stop()

TEST_ACCOUNTS = {
    "yamada": {"password": "password123", "name": "山田 太郎", "id": 1, "department": "循環器内科"},
    "suzuki": {"password": "password123", "name": "鈴木 花子", "id": 2, "department": "内科"},
}

@app.post("/auth/login", response_model=schemas.LoginResponse)
def login(req: schemas.LoginRequest, db: Session = Depends(get_db)):
    if req.login_id not in TEST_ACCOUNTS:
        print(f"ログイン失敗: login_id={req.login_id}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="IDまたはパスワードが正しくありません")

    account = TEST_ACCOUNTS[req.login_id]
    if req.password != account["password"]:
        print(f"ログイン失敗: login_id={req.login_id}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="IDまたはパスワードが正しくありません")

    print(f"ログイン成功: login_id={req.login_id}")
    token = auth.create_token(account["id"], req.login_id)
    return schemas.LoginResponse(
        token=token, doctor_id=account["id"],
        doctor_name=account["name"], department=account["department"],
    )

@app.get("/patients", response_model=list[schemas.PatientResponse])
def get_patients(
    name: str = "",        # 患者名
    birth_date: str = "",  # 生年月日
    room: str = "",        # 病室
    ward: str = "",        # 病棟
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    return crud.get_patients(db, name=name, birth_date=birth_date, room=room, ward=ward)

@app.post("/patients", response_model=schemas.PatientResponse)
def create_patient(
    patient: schemas.PatientCreate,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    return crud.create_patient(db, patient)

@app.get("/patients/{patient_id}", response_model=schemas.PatientResponse)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    p = crud.get_patient(db, patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="患者が見つかりません")
    return p

@app.put("/patients/{patient_id}", response_model=schemas.PatientResponse)
def update_patient(
    patient_id: int, patient: schemas.PatientUpdate,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    updated = crud.update_patient(db, patient_id, patient)
    if not updated:
        raise HTTPException(status_code=404, detail="患者が見つかりません")
    return updated

@app.delete("/patients/{patient_id}")
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    """患者を削除する（計測履歴・アラートも同時に削除）"""
    success = crud.delete_patient(db, patient_id)
    if not success:
        raise HTTPException(status_code=404, detail="患者が見つかりません")
    return {"status": "ok"}

@app.get("/sensors/{patient_id}", response_model=list[schemas.SensorRecordResponse])
def get_sensor_records(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    return crud.get_sensor_records(db, patient_id)

@app.get("/holter/{patient_id}", response_model=schemas.HolterReportResponse)
def get_holter_report(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="患者が見つかりません")

    report = _build_holter_report(patient_id)
    return schemas.HolterReportResponse(**report)

@app.get("/summary/{patient_id}", response_model=dict)
def get_patient_summary(
    patient_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    patient = crud.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="患者が見つかりません")

    report = _build_holter_report(patient_id)
    return {
        "patient_id": patient.id,
        "patient_name": patient.name,
        "summary": report["summary"],
        "af_likelihood": report["overview"]["af_likelihood"],
        "rr_variability_ms": report["overview"]["rr_variability_ms"],
        "episode_count": report["overview"]["episode_count"],
        "signal_quality": report["overview"]["signal_quality"],
        "key_findings": report["episodes"],
        "generated_at": datetime.utcnow().isoformat(),
    }

@app.get("/alerts", response_model=list[schemas.AlertResponse])
def get_alerts(
    unread_only: bool = False,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    alerts = crud.get_alerts(db, doctor_id=doctor_id, unread_only=unread_only)
    result = []
    for a in alerts:
        patient = crud.get_patient(db, a.patient_id)
        result.append(schemas.AlertResponse(
            id=a.id, patient_id=a.patient_id, doctor_id=a.doctor_id,
            alert_type=a.alert_type, message=a.message,
            is_read=a.is_read, created_at=a.created_at,
            patient_name=patient.name if patient else "",
        ))
    return result

@app.post("/alerts", response_model=schemas.AlertResponse)
def create_alert(
    alert: schemas.AlertCreate,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    return crud.create_alert(db, alert)

@app.put("/alerts/{alert_id}/read")
def mark_read(
    alert_id: int,
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    crud.mark_alert_read(db, alert_id)
    return {"status": "ok"}

@app.put("/alerts/read-all")
def mark_all_read(
    db: Session = Depends(get_db),
    doctor_id: int = Depends(auth.get_current_doctor_id),
):
    crud.mark_all_alerts_read(db, doctor_id)
    return {"status": "ok"}

@app.websocket("/ws/device/signal/{user_id}")
async def signal_receive_ws(websocket: WebSocket, user_id: str):
    """
    外部PC（センサーデバイス等）からの生体データを受信する。
    送信フォーマット例: バッチデータのJSON
    """
    await websocket.accept()
    print(f"[RECV] 外部PC 接続: user_id={user_id}")

    db = SessionLocal()
    try:
        while True:
            data = await websocket.receive_json()

            await _broadcast(user_id, data)

            values = data.get("values", [])
            signal_type = data.get("signal_type")
            if values and signal_type in ["ECG", "PPG", "HR"]:
                latest_val = values[-1]
                patient = crud.get_patient_by_user_id(db, user_id)
                if patient:
                    is_anomaly = 0
                    status = "normal"
                    if signal_type == "ECG" and (latest_val > ECG_UPPER or latest_val < ECG_LOWER):
                        is_anomaly = 1
                        status = "ecg_anomaly"
                    elif signal_type == "PPG" and (latest_val > PPG_UPPER or latest_val < PPG_LOWER):
                        is_anomaly = 1
                        status = "ppg_anomaly"

                    crud.create_sensor_record(db, schemas.SensorRecordCreate(
                        patient_id = patient.id,
                        ecg = latest_val if signal_type == "ECG" else 0.0,
                        ppg = latest_val if signal_type == "PPG" else 0.0,
                        hr  = latest_val if signal_type == "HR"  else 0.0,
                        status = status,
                        is_anomaly = is_anomaly
                    ))

    except Exception as e:
        print(f"[RECV] 外部PC 切断: user_id={user_id}, reason={e}")
    finally:
        db.close()

@app.websocket("/ws/sensors/{user_id}")
async def sensors_ws(websocket: WebSocket, user_id: str):
    """
    フロントエンドからの接続を受け付け、デバイスからのデータをブロードキャストする
    """
    await websocket.accept()
    print(f"WebSocket接続(フロント): user_id={user_id}")

    if user_id not in active_connections:
        active_connections[user_id] = []
    active_connections[user_id].append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(f"WebSocket切断(フロント): user_id={user_id}, reason={e}")
    finally:
        if user_id in active_connections and websocket in active_connections[user_id]:
            active_connections[user_id].remove(websocket)
