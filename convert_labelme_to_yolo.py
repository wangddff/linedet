import json
import os
from pathlib import Path


CLASS_NAMES = [
    "terminal_hole",
    "number_tube",
    "wire",
    "plc_module",
    "short_wire",
    "jumper",
    "connector",
]


def convert_labelme_to_yolo(
    json_file: str, output_dir: str, img_width: int, img_height: int
):
    """将labelme的JSON标注转换为YOLO格式"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    yolo_annotations = []

    for shape in shapes:
        label = shape.get("label", "")
        if label not in CLASS_NAMES:
            print(f"警告: 未知类别 '{label}', 跳过")
            continue

        class_id = CLASS_NAMES.index(label)
        points = shape.get("points", [])

        if len(points) != 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y2 + y1) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    output_file = Path(output_dir) / (Path(json_file).stem + ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_annotations))

    return len(yolo_annotations)


def batch_convert(json_dir: str, output_dir: str):
    """批量转换labelme标注文件"""
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    total_count = 0

    for json_file in json_files:
        img_width = 1920
        img_height = 1080
        count = convert_labelme_to_yolo(
            str(json_file), str(output_dir), img_width, img_height
        )
        total_count += count
        print(f"转换: {json_file.name} -> {count} 个标注")

    print(f"\n转换完成! 共处理 {len(json_files)} 个文件, {total_count} 个标注")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将labelme标注转换为YOLO格式")
    parser.add_argument(
        "--json-dir", default="datasets/images/train", help="labelme JSON文件目录"
    )
    parser.add_argument(
        "--output-dir", default="datasets/labels/train", help="YOLO标注输出目录"
    )
    args = parser.parse_args()

    batch_convert(args.json_dir, args.output_dir)
