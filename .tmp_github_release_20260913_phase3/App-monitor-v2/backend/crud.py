from sqlalchemy.orm import Session
from sqlalchemy import or_
import models, schemas
from passlib.context import CryptContext

def hash_password(password: str) -> str:
    """テスト用: パスワードをそのまま返す（ハッシュ化なし）"""
    return password

def verify_password(plain: str, hashed: str) -> bool:
    """テスト用: 平文で直接比較"""
    return plain == hashed

def get_doctor_by_login_id(db: Session, login_id: str):
    return db.query(models.Doctor).filter(models.Doctor.login_id == login_id).first()

def create_doctor(db: Session, doctor: schemas.DoctorCreate):
    db_doctor = models.Doctor(
        name       = doctor.name,
        login_id   = doctor.login_id,
        password   = hash_password(doctor.password),
        department = doctor.department,
    )
    db.add(db_doctor)
    db.commit()
    db.refresh(db_doctor)
    return db_doctor

def get_patients(db: Session, name: str = "", birth_date: str = "", room: str = "", ward: str = ""):
    query = db.query(models.Patient)
    if name:
        query = query.filter(models.Patient.name.contains(name))
    if birth_date:
        query = query.filter(models.Patient.birth_date.contains(birth_date))
    if room:
        query = query.filter(models.Patient.room.contains(room))
    if ward:
        query = query.filter(models.Patient.ward.contains(ward))
    return query.order_by(models.Patient.id).all()

def get_patient(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def get_patient_by_user_id(db: Session, user_id: str):
    return db.query(models.Patient).filter(models.Patient.user_id == user_id).first()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(**patient.model_dump())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def update_patient(db: Session, patient_id: int, patient: schemas.PatientUpdate):
    db_patient = get_patient(db, patient_id)
    if not db_patient:
        return None
    for key, value in patient.model_dump().items():
        setattr(db_patient, key, value)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def delete_patient(db: Session, patient_id: int) -> bool:
    patient = get_patient(db, patient_id)
    if not patient:
        return False

    db.query(models.SensorRecord).filter(models.SensorRecord.patient_id == patient_id).delete()
    db.query(models.Alert).filter(models.Alert.patient_id == patient_id).delete()

    db.delete(patient)  # 患者レコード本体を削除
    db.commit()
    return True

def create_sensor_record(db: Session, record: schemas.SensorRecordCreate):
    db_record = models.SensorRecord(**record.model_dump())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

def get_sensor_records(db: Session, patient_id: int, limit: int = 100):
    return (
        db.query(models.SensorRecord)
        .filter(models.SensorRecord.patient_id == patient_id)
        .order_by(models.SensorRecord.measured_at.desc())
        .limit(limit)
        .all()
    )

def create_alert(db: Session, alert: schemas.AlertCreate):
    db_alert = models.Alert(**alert.model_dump())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_alerts(db: Session, doctor_id: int = None, unread_only: bool = False):
    query = db.query(models.Alert)
    if doctor_id:
        query = query.filter(models.Alert.doctor_id == doctor_id)
    if unread_only:
        query = query.filter(models.Alert.is_read == 0)
    return query.order_by(models.Alert.created_at.desc()).all()

def mark_alert_read(db: Session, alert_id: int):
    db_alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if db_alert:
        db_alert.is_read = 1
        db.commit()
        db.refresh(db_alert)
    return db_alert

def mark_all_alerts_read(db: Session, doctor_id: int):
    db.query(models.Alert).filter(
        models.Alert.doctor_id == doctor_id,
        models.Alert.is_read == 0
    ).update({"is_read": 1})
    db.commit()
