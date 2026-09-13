from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Doctor(Base):
    """医師テーブル。ログインアカウントの管理に使用
       パスワードはハッシュ化して保存
    """
    __tablename__ = "doctors"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    login_id   = Column(String, unique=True, nullable=False)
    password   = Column(String, nullable=False)
    department = Column(String, default="") # 診療科
    created_at = Column(DateTime, default=datetime.utcnow)

    patients = relationship("Patient", back_populates="doctor")
    alerts   = relationship("Alert",   back_populates="doctor")

class Patient(Base):
    """患者テーブル"""
    __tablename__ = "patients"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(String, unique=True, index=True, nullable=True) # 外部システム連携用ID (例: "test_user_01")
    name       = Column(String, nullable=False) # 患者名（必須）
    birth_date = Column(String, default="") # 生年月日
    gender     = Column(String, default="") # 性別（男性・女性・その他）
    room       = Column(String, default="") # 病室番号
    ward       = Column(String, default="") # 病棟
    doctor_id  = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    history    = Column(Text, default="") # 既往歴
    note       = Column(Text, default="") # 備考
    created_at = Column(DateTime, default=datetime.utcnow) # 登録日時

    doctor      = relationship("Doctor",    back_populates="patients")
    sensor_records = relationship("SensorRecord", back_populates="patient")
    alerts      = relationship("Alert",     back_populates="patient")

class SensorRecord(Base):
    """ECG/PPG計測記録テーブル
       1秒ごとにデータが送信されるたびに1レコード追加
       【長期運用でデータ量多くなるので、後で古いデータの削除機能を追加する】
    """
    __tablename__ = "sensor_records"
    id             = Column(Integer, primary_key=True, index=True)
    patient_id     = Column(Integer, ForeignKey("patients.id"), nullable=False)
    measured_at    = Column(DateTime, default=datetime.utcnow) # 計測日時
    ecg            = Column(Float, default=0.0) # 心電図
    ppg            = Column(Float, default=0.0) # 光電容積脈波
    hr             = Column(Float, default=0.0) # 心拍数
    status         = Column(String, default="normal")  # normal / ecg_anomaly / ppg_anomaly / both_anomaly
    is_anomaly     = Column(Integer, default=0)        # 0=正常, 1=異常

    patient = relationship("Patient", back_populates="sensor_records")

class Alert(Base):
    """アラート・通知テーブル"""
    __tablename__ = "alerts"
    id         = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id  = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    alert_type = Column(String, default="") # high_hr: 高心拍 / low_hr: 低心拍 / manual: 手動通知
    message    = Column(Text, default="")
    is_read    = Column(Integer, default=0)  # 0=未読, 1=既読
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="alerts")
    doctor  = relationship("Doctor",  back_populates="alerts")
