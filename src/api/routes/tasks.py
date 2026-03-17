from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import shutil
import json

from src.database import get_db
from src.database.models import Task, DetectionResult
from src.utils.config import load_config

router = APIRouter()

config = load_config()
UPLOAD_DIR = config["storage"]["raw_images"]


class TaskCreate(BaseModel):
    product_id: int
    station_id: int
    layer: int = 1


class TaskResponse(BaseModel):
    id: int
    product_id: int
    station_id: int
    layer: int
    status: str
    similarity_score: Optional[float] = None
    overall_result: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class DetectionResultResponse(BaseModel):
    id: int
    task_id: int
    module: str
    details: Optional[dict] = None
    passed: Optional[bool] = None

    class Config:
        from_attributes = True


@router.get("", response_model=List[TaskResponse])
async def list_tasks(
    product_id: Optional[int] = None,
    station_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    query = db.query(Task)
    if product_id:
        query = query.filter(Task.product_id == product_id)
    if station_id:
        query = query.filter(Task.station_id == station_id)
    if status:
        query = query.filter(Task.status == status)

    tasks = query.order_by(Task.created_at.desc()).limit(limit).all()
    return [
        TaskResponse(
            id=t.id,
            product_id=t.product_id,
            station_id=t.station_id,
            layer=t.layer,
            status=t.status,
            similarity_score=float(t.similarity_score) if t.similarity_score else None,
            overall_result=t.overall_result,
            created_at=t.created_at.isoformat() if t.created_at else "",
            completed_at=t.completed_at.isoformat() if t.completed_at else None,
        )
        for t in tasks
    ]


@router.post("", response_model=TaskResponse)
async def create_task(task_data: TaskCreate, db: Session = Depends(get_db)):
    task = Task(
        product_id=task_data.product_id,
        station_id=task_data.station_id,
        layer=task_data.layer,
        status="pending",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    return TaskResponse(
        id=task.id,
        product_id=task.product_id,
        station_id=task.station_id,
        layer=task.layer,
        status=task.status,
        similarity_score=None,
        overall_result=None,
        created_at=task.created_at.isoformat() if task.created_at else "",
        completed_at=None,
    )


@router.post("/{task_id}/upload")
async def upload_image(
    task_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task.image_path = filepath
    task.image_filename = file.filename
    task.status = "uploaded"
    db.commit()

    return {"message": "上传成功", "filename": filename, "filepath": filepath}


@router.post("/{task_id}/detect")
async def run_detection(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not task.image_path or not os.path.exists(task.image_path):
        raise HTTPException(status_code=400, detail="图片未上传")

    from src.services.detection_service import DetectionService

    service = DetectionService()
    result = service.run_detection(task)

    task.similarity_score = result.get("similarity_score")
    task.overall_result = result.get("overall_result")
    task.error_details = result.get("errors")
    task.status = "completed" if result.get("overall_result") == "pass" else "failed"
    task.completed_at = datetime.utcnow()
    db.commit()

    module_results = result.get("module_results", {})
    for module_name, module_result in module_results.items():
        detection_result = DetectionResult(
            task_id=task.id,
            module=module_name,
            details=module_result,
            passed=module_result.get("passed", False),
        )
        db.add(detection_result)
    db.commit()

    return result


@router.get("/{task_id}/result")
async def get_task_result(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    results = db.query(DetectionResult).filter(DetectionResult.task_id == task_id).all()

    return {
        "task": TaskResponse(
            id=task.id,
            product_id=task.product_id,
            station_id=task.station_id,
            layer=task.layer,
            status=task.status,
            similarity_score=float(task.similarity_score)
            if task.similarity_score
            else None,
            overall_result=task.overall_result,
            created_at=task.created_at.isoformat() if task.created_at else "",
            completed_at=task.completed_at.isoformat() if task.completed_at else None,
        ),
        "image_path": task.image_path,
        "image_filename": task.image_filename,
        "results": [
            DetectionResultResponse(
                id=r.id,
                task_id=r.task_id,
                module=r.module,
                details=r.details,
                passed=r.passed,
            )
            for r in results
        ],
    }


@router.post("/{task_id}/layer/next")
async def next_layer(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.status != "completed" or task.overall_result != "pass":
        raise HTTPException(
            status_code=400, detail="当前层级检测未通过，无法进入下一层"
        )

    new_task = Task(
        product_id=task.product_id,
        station_id=task.station_id,
        layer=task.layer + 1,
        status="pending",
    )
    db.add(new_task)
    db.commit()
    db.refresh(new_task)

    return {
        "message": "已创建下一层级任务",
        "new_task_id": new_task.id,
        "new_layer": new_task.layer,
    }


@router.delete("/{task_id}")
async def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task.image_path and os.path.exists(task.image_path):
        os.remove(task.image_path)

    db.query(DetectionResult).filter(DetectionResult.task_id == task_id).delete()
    db.delete(task)
    db.commit()

    return {"message": "删除成功"}
