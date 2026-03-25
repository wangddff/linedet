from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel
import os
import shutil

from src.utils.config import load_config

router = APIRouter()

config = load_config()
STANDARD_DIR = config["storage"]["standard_images"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class StandardImageResponse(BaseModel):
    id: int
    station_id: int
    product_id: int | None
    file_path: str
    file_name: str
    similarity_threshold: float
    description: str | None
    created_at: str


@router.get("/{station_id}", response_model=List[StandardImageResponse])
async def list_standard_images(station_id: int, product_id: int | None = None):
    """从文件系统扫描标准图，无需数据库。"""
    station_dir = os.path.join(STANDARD_DIR, f"station_{station_id}")
    if not os.path.isdir(station_dir):
        return []

    results = []
    for idx, fname in enumerate(sorted(os.listdir(station_dir)), start=1):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        results.append(
            StandardImageResponse(
                id=idx,
                station_id=station_id,
                product_id=product_id,
                file_path=os.path.join(station_dir, fname),
                file_name=fname,
                similarity_threshold=0.85,
                description=None,
                created_at="",
            )
        )
    return results


@router.post("/{station_id}")
async def upload_standard_image(
    station_id: int,
    product_id: int | None = None,
    similarity_threshold: float = 0.85,
    description: str | None = None,
    file: UploadFile = File(...),
):
    station_dir = os.path.join(STANDARD_DIR, f"station_{station_id}")
    os.makedirs(station_dir, exist_ok=True)

    filename = file.filename or f"std_{station_id}.png"
    filepath = os.path.join(station_dir, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": "标准图上传成功", "file_name": filename, "file_path": filepath}


@router.delete("/{station_id}/{file_name}")
async def delete_standard_image(station_id: int, file_name: str):
    filepath = os.path.join(STANDARD_DIR, f"station_{station_id}", file_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="标准图不存在")

    os.remove(filepath)
    return {"message": "删除成功"}
