#!/usr/bin/env python3
"""测试 ROI + OpenCV 轻量检测流程"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.core.roi import ROILoader, ROICropper


def test_single_image(image_path: str):
    """测试单张图片的完整流程"""
    print(f"\n{'=' * 50}")
    print(f"测试图片: {image_path}")
    print(f"{'=' * 50}")

    img = __import__("cv2").imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    print(f"\n[1] 图像预处理")
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess(image_path)
    scale = preprocessor.get_scale_factor()
    print(f"    原始尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"    预处理后: {processed.shape[1]}x{processed.shape[0]}")
    print(f"    缩放因子: {scale:.3f}")

    print(f"\n[2] 加载 ROI 标注")
    roi_loader = ROILoader()
    json_path = Path(image_path).with_suffix(".json")
    roi_data = roi_loader.load_from_labelme(str(json_path))
    detect_area = roi_data.get("detect_area")
    rois = roi_data.get("rois", [])

    print(f"    检测区: {detect_area.bbox if detect_area else 'None'}")
    print(f"    ROI 数量: {len(rois)}")

    label_counts = {}
    for roi in rois:
        label_counts[roi.label] = label_counts.get(roi.label, 0) + 1
    print(f"    标签分布: {label_counts}")

    if scale != 1.0 and detect_area:
        detect_area = detect_area.scale(scale)
        rois = [roi.scale(scale) for roi in rois]

    print(f"\n[3] ROI 裁剪")
    cropper = ROICropper()

    if detect_area:
        crop_result = cropper.crop_with_detect_area(processed, detect_area, rois)
        detect_area_img = crop_result["detect_area_img"]
        print(f"    检测区裁剪: {detect_area_img.shape}")

        number_tubes = [
            r for r in crop_result["rois"] if r.get("label") == "number_tube"
        ]
        terminals = [
            r for r in crop_result["rois"] if r.get("label") == "terminal_hole"
        ]
        wires = [
            r for r in crop_result["rois"] if r.get("label") in ["wire", "short_wire"]
        ]
        connectors = [r for r in crop_result["rois"] if r.get("label") == "connector"]

        print(f"    号码管: {len(number_tubes)}")
        print(f"    端子孔位: {len(terminals)}")
        print(f"    线材: {len(wires)}")
        print(f"    插头: {len(connectors)}")
    else:
        print("    无检测区，使用全图")

    print(f"\n[4] 测试完成")


def main():
    test_dir = Path("datasets/images/train")
    test_images = sorted(test_dir.glob("*.png"))

    if not test_images:
        print("未找到测试图片")
        return

    for img_path in test_images[:2]:
        test_single_image(str(img_path))
        print()


if __name__ == "__main__":
    main()
