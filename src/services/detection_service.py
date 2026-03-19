import numpy as np
import cv2
from pathlib import Path


class DetectionService:
    """检测服务主类 - 基于 ROI + OpenCV 的轻量检测方案"""

    def __init__(self):
        from src.utils.config import get_station_config

        self.station_config = get_station_config()
        self._preprocessed_image = None
        self._rois = None

    def _clean_for_json(self, obj):
        """清理对象中的numpy数组，用于JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        return obj

    def run_detection(self, task):
        """执行完整检测流程

        流程：
        1. 图像预处理 - 缩放至1080P + 降噪
        2. 加载ROI标注 + 精准裁剪
        3. 标准图对比
        4. OCR文字识别
        5. 颜色检测
        6. 规则校验
        """
        results = {}

        img = cv2.imread(task.image_path)
        if img is None:
            return {
                "passed": False,
                "error": f"无法读取图片: {task.image_path}",
            }

        from src.preprocessing.image_preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        preprocessed_img = preprocessor.preprocess(task.image_path)
        self._preprocessed_image = preprocessed_img

        scale_factor = preprocessor.get_scale_factor()
        print(f"\n========== 图像预处理 ==========")
        print(f"原始尺寸: {img.shape[1]}x{img.shape[0]}")
        print(f"缩放后: {preprocessed_img.shape[1]}x{preprocessed_img.shape[0]}")
        print(f"缩放因子: {scale_factor:.3f}")
        print(f"================================\n")

        from src.core.roi import ROILoader, ROICropper

        roi_loader = ROILoader(station_id=task.station_id)
        labelme_dir = "datasets/images/train"

        json_name = Path(task.image_path).stem
        json_path = Path(labelme_dir) / f"{json_name}.json"

        if not json_path.exists():
            json_path = Path(labelme_dir) / "train_1.json"

        roi_data = roi_loader.load_from_labelme(str(json_path))
        detect_area = roi_data.get("detect_area")
        rois = roi_data.get("rois", [])

        if scale_factor != 1.0:
            if detect_area:
                detect_area = detect_area.scale(scale_factor)
            rois = [roi.scale(scale_factor) for roi in rois]

        self._rois = rois
        roi_cropper = ROICropper()

        print(f"\n========== ROI 裁剪 ==========")
        if detect_area:
            print(f"检测区 (detect_area): bbox={detect_area.bbox}")

        print(f"加载的 ROI 数量: {len(rois)}")
        label_counts = {}
        for roi in rois:
            label_counts[roi.label] = label_counts.get(roi.label, 0) + 1
        print(f"各类 ROI: {label_counts}")

        if detect_area:
            crop_result = roi_cropper.crop_with_detect_area(
                preprocessed_img, detect_area, rois
            )
            detect_area_img = crop_result["detect_area_img"]

            number_tube_rois = roi_cropper.crop_by_label(
                detect_area_img,
                [r for r in rois if r.label == "number_tube"],
                "number_tube",
            )
            terminal_rois = roi_cropper.crop_by_label(
                detect_area_img,
                [r for r in rois if r.label == "terminal_hole"],
                "terminal_hole",
            )
            wire_rois = [
                r
                for r in crop_result["rois"]
                if r.get("label") in ["wire", "short_wire"]
            ]
            connector_rois = [
                r for r in crop_result["rois"] if r.get("label") == "connector"
            ]

            print(f"已在检测区内裁剪")
        else:
            number_tube_rois = roi_cropper.crop_by_label(
                preprocessed_img, rois, "number_tube"
            )
            terminal_rois = roi_cropper.crop_by_label(
                preprocessed_img, rois, "terminal_hole"
            )
            wire_rois = roi_cropper.extract_wire_regions(preprocessed_img, rois)
            connector_rois = roi_cropper.extract_connector_regions(
                preprocessed_img, rois
            )

        print(f"号码管 ROI: {len(number_tube_rois)}")
        print(f"端子 ROI: {len(terminal_rois)}")
        print(f"线材 ROI: {len(wire_rois)}")
        print(f"插头 ROI: {len(connector_rois)}")
        print(f"============================\n")

        from src.core.comparator.standard_comparator import StandardComparator

        comparator = StandardComparator(task.station_id, task.product_id)
        compare_result = comparator.compare(task.image_path)
        results["comparator"] = compare_result

        if not compare_result.get("passed", False):
            return {
                "similarity_score": compare_result.get("similarity_score", 0),
                "overall_result": "fail",
                "errors": [{"module": "comparator", "message": "标准图对比未通过"}],
                "module_results": results,
            }

        from src.core.ocr.text_recognizer import TextRecognizer

        ocr = TextRecognizer()

        all_rois_for_ocr = number_tube_rois + terminal_rois
        ocr_result = ocr.recognize(task.image_path, all_rois_for_ocr)
        results["ocr"] = ocr_result

        from src.core.color.color_detector import ColorDetector

        color_det = ColorDetector(station_id=task.station_id)
        color_result = color_det.detect(task.image_path, wire_rois)
        results["color"] = color_result

        from src.core.validator.rule_validator import RuleValidator

        validator = RuleValidator(task.product_id, task.station_id, task.layer)
        validate_result = validator.validate(ocr_result, color_result)
        results["validator"] = validate_result

        from src.utils.image_annotator import ImageAnnotator

        annotator = ImageAnnotator()

        detection_result = {
            "rois": {
                "number_tube": number_tube_rois,
                "terminal_hole": terminal_rois,
                "wire": wire_rois,
                "connector": connector_rois,
            }
        }

        try:
            annotated_path = annotator.annotate_image(
                task.image_path,
                detection_result,
                ocr_result,
                color_result,
            )
        except Exception as e:
            print(f"[DetectionService] 图像标注失败: {e}")
            annotated_path = task.image_path

        all_passed = all(r.get("passed", False) for r in results.values())

        return self._clean_for_json(
            {
                "similarity_score": compare_result.get("similarity_score", 0),
                "overall_result": "pass" if all_passed else "fail",
                "errors": self._collect_errors(results),
                "detections": detection_result,
                "ocr_results": ocr_result,
                "color_results": color_result,
                "validation_result": validate_result,
                "module_results": results,
                "annotated_image": annotated_path,
            }
        )

    def _collect_errors(self, results):
        """收集所有错误"""
        errors = []
        for module, result in results.items():
            if not result.get("passed", True):
                errors.append(
                    {
                        "module": module,
                        "message": result.get("error", "检测未通过"),
                        "details": result.get("details", {}),
                    }
                )
        return errors
