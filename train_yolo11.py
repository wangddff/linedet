import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def prepare_dataset():
    """准备数据集：将图片复制到训练目录"""
    source_dir = Path(__file__).parent / "request" / "pic"
    train_img_dir = Path(__file__).parent / "datasets" / "images" / "train"
    val_img_dir = Path(__file__).parent / "datasets" / "images" / "val"
    train_label_dir = Path(__file__).parent / "datasets" / "labels" / "train"
    val_label_dir = Path(__file__).parent / "datasets" / "labels" / "val"

    for img_file in source_dir.glob("*.png"):
        shutil.copy(img_file, train_img_dir / img_file.name)
        label_file = train_label_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            label_file.touch()

    for img_file in source_dir.glob("*.jpg"):
        shutil.copy(img_file, train_img_dir / img_file.name)
        label_file = train_label_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            label_file.touch()

    print(f"数据集准备完成: {len(list(train_img_dir.glob('*')))} 张训练图片")


def train():
    """训练YOLO11s模型"""
    model = YOLO("yolo11s.pt")

    results = model.train(
        data="datasets/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    print("训练完成！")
    print(f"最佳模型路径: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    prepare_dataset()
    train()
