"""SQLAlchemy models for persistent storage."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from backend.app.core.config import get_settings

Base = declarative_base()


class InvestigationORM(Base):
    __tablename__ = "investigations"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    status = Column(String, default="initialized")
    textual_description = Column(Text, nullable=True)
    metadata_json = Column(JSON, default=dict)
    final_decision = Column(String, nullable=True)
    confidence_overall = Column(Float, nullable=True)
    uncertainty_reason = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    iteration = Column(Integer, default=0)
    latency_json = Column(JSON, default=dict)
    errors_json = Column(JSON, default=list)
    trace_json = Column(JSON, default=list)
    state_json = Column(JSON, default=dict)  # full serialized state for recovery
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    images = relationship("ImageORM", back_populates="investigation", cascade="all, delete-orphan")


class ImageORM(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    investigation_id = Column(String, ForeignKey("investigations.id"), index=True)
    path = Column(String, nullable=False)
    view_id = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    investigation = relationship("InvestigationORM", back_populates="images")


class EvaluationRunORM(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String)
    metrics_json = Column(JSON, default=dict)
    config_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine():
    settings = get_settings()
    connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
    return create_engine(settings.DATABASE_URL, connect_args=connect_args, echo=False)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return engine


SessionLocal = sessionmaker(autocommit=False, autoflush=False)
