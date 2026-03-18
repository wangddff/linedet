import numpy as np


class DetectionService:
    """检测服务主类"""

    def __init__(self):
        from src.utils.config import get_station_config

        self.station_config = get_station_config()

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
        """执行完整检测流程"""
        results = {}

        # 1. 标准图对比
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

        # 2. 目标检测
        from src.core.detector.yolo_detector import YOLODetector

        detector = YOLODetector(station_id=task.station_id)
        detect_result = detector.detect(task.image_path)
        results["detector"] = detect_result

        # 3. OCR识别
        from src.core.ocr.text_recognizer import TextRecognizer

        ocr = TextRecognizer()
        ocr_result = ocr.recognize(task.image_path, detect_result.get("rois", []))
        results["ocr"] = ocr_result

        # 4. 颜色检测
        from src.core.color.color_detector import ColorDetector

        color_det = ColorDetector(station_id=task.station_id)
        color_result = color_det.detect(
            task.image_path, detect_result.get("wire_rois", [])
        )
        results["color"] = color_result

        # 5. 规则校验
        from src.core.validator.rule_validator import RuleValidator

        validator = RuleValidator(task.product_id, task.station_id, task.layer)
        validate_result = validator.validate(ocr_result, color_result)
        results["validator"] = validate_result

        # 综合判断
        all_passed = all(r.get("passed", False) for r in results.values())

        return self._clean_for_json(
            {
                "similarity_score": compare_result.get("similarity_score", 0),
                "overall_result": "pass" if all_passed else "fail",
                "errors": self._collect_errors(results),
                "detections": detect_result,
                "ocr_results": ocr_result,
                "color_results": color_result,
                "validation_result": validate_result,
                "module_results": results,
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
