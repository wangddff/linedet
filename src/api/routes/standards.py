from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
import os
import shutil

from src.database import get_db
from src.database.models import StandardImage
from src.utils.config import load_config

router = APIRouter()

config = load_config()
STANDARD_DIR = config["storage"]["standard_images"]


class StandardImageResponse(BaseModel):
    id: int
    station_id: int
    product_id: int | None
    file_path: str
    file_name: str
    similarity_threshold: float
    description: str | None
    created_at: str

    class Config:
        from_attributes = True


@router.get("/{station_id}", response_model=List[StandardImageResponse])
async def list_standard_images(
    station_id: int, product_id: int | None = None, db: Session = Depends(get_db)
):
    query = db.query(StandardImage).filter(StandardImage.station_id == station_id)
    if product_id:
        query = query.filter(StandardImage.product_id == product_id)

    images = query.all()
    return [
        StandardImageResponse(
            id=img.id,
            station_id=img.station_id,
            product_id=img.product_id,
            file_path=img.file_path,
            file_name=img.file_name,
            similarity_threshold=float(img.similarity_threshold),
            description=img.description,
            created_at=img.created_at.isoformat() if img.created_at else "",
        )
        for img in images
    ]


@router.post("/{station_id}")
async def upload_standard_image(
    station_id: int,
    product_id: int | None = None,
    similarity_threshold: float = 0.85,
    description: str | None = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    station_dir = os.path.join(STANDARD_DIR, f"station_{station_id}")
    os.makedirs(station_dir, exist_ok=True)

    filename = file.filename or f"std_{station_id}.png"
    filepath = os.path.join(station_dir, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db_image = StandardImage(
        station_id=station_id,
        product_id=product_id,
        file_path=filepath,
        file_name=filename,
        similarity_threshold=similarity_threshold,
        description=description,
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return {"message": "标准图上传成功", "id": db_image.id, "file_path": filepath}


@router.delete("/{image_id}")
async def delete_standard_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(StandardImage).filter(StandardImage.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="标准图不存在")

    if os.path.exists(image.file_path):
        os.remove(image.file_path)

    db.delete(image)
    db.commit()

    return {"message": "删除成功"}
