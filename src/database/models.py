from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Boolean,
    DECIMAL,
    JSONB,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tasks = relationship("Task", back_populates="product")
    wiring_rules = relationship("WiringRule", back_populates="product")
    standard_images = relationship("StandardImage", back_populates="product")


class StandardImage(Base):
    __tablename__ = "standard_images"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    file_path = Column(String(255), nullable=False)
    file_name = Column(String(100), nullable=False)
    similarity_threshold = Column(DECIMAL(3, 2), default=0.85)
    description = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    product = relationship("Product", back_populates="standard_images")


class WiringRule(Base):
    __tablename__ = "wiring_rules"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    station_id = Column(Integer, nullable=False)
    layer = Column(Integer, nullable=True)
    hole_number = Column(String(20), nullable=False)
    wire_number = Column(String(50), nullable=True)
    wire_color = Column(String(20), nullable=True)
    has_connector = Column(Boolean, default=False)
    has_short_wire = Column(Boolean, default=False)
    has_jumper = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    product = relationship("Product", back_populates="wiring_rules")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    station_id = Column(Integer, nullable=False)
    layer = Column(Integer, default=1)
    status = Column(String(20), default="pending")
    image_path = Column(String(255), nullable=True)
    image_filename = Column(String(100), nullable=True)
    similarity_score = Column(DECIMAL(5, 4), nullable=True)
    overall_result = Column(String(20), nullable=True)
    error_details = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    product = relationship("Product", back_populates="tasks")
    results = relationship("DetectionResult", back_populates="task")


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False)
    module = Column(String(20), nullable=False)
    details = Column(JSONB, nullable=True)
    passed = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    task = relationship("Task", back_populates="results")
