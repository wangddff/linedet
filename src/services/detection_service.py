import numpy as np
from pathlib import Path


class DetectionService:
    """检测服务主类 - 基于 ROI + OpenCV 的轻量检测方案"""

    def __init__(self):
        from src.utils.config import get_station_config

        self.station_config = get_station_config()
        self._preprocessed_image = None
        self._rois = None

    def _clean_for_json(self, obj):
        """清理对象中的numpy类型，用于JSON序列化"""
        if obj is None:
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
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
        roi_compare_result = None
        roi_comparator = None

        import cv2

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

        std_json_path = Path(labelme_dir) / "train_1.json"
        roi_data = roi_loader.load_from_labelme(str(std_json_path))
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

        print(f"\n========== ROI 区域对比 ==========")
        from src.core.comparator.roi_comparator import ROIComparator
        from src.core.roi import ROILoader, ROICropper

        std_scale = 1.0
        std_json_path = Path(labelme_dir) / "train_1.json"
        if std_json_path.exists():
            std_roi_data = roi_loader.load_from_labelme(str(std_json_path))
            std_detect_area = std_roi_data.get("detect_area")
            std_rois = std_roi_data.get("rois", [])

            std_image_path = str(Path(labelme_dir) / "train_1.png")
            std_preprocessor = ImagePreprocessor()
            std_preprocessed_img = std_preprocessor.preprocess(std_image_path)
            std_scale = std_preprocessor.get_scale_factor()

            if std_scale != 1.0:
                if std_detect_area:
                    std_detect_area = std_detect_area.scale(std_scale)
                std_rois = [roi.scale(std_scale) for roi in std_rois]

            if detect_area:
                crop_result = roi_cropper.crop_with_detect_area(
                    preprocessed_img, detect_area, rois
                )
                test_detect_area_img = crop_result["detect_area_img"]
                test_roi_crops = crop_result["rois"]

                if std_detect_area:
                    offset_x = detect_area.bbox[0] - std_detect_area.bbox[0]
                    offset_y = detect_area.bbox[1] - std_detect_area.bbox[1]

                    adjusted_std_rois = []
                    for roi in std_rois:
                        new_points = []
                        for p in roi.points:
                            new_x = p[0] + offset_x
                            new_y = p[1] + offset_y
                            new_points.append([new_x, new_y])
                        from src.core.roi.roi_loader import ROI

                        adjusted_roi = ROI(roi.label, new_points, roi.group_id)
                        adjusted_std_rois.append(adjusted_roi)

                    std_crop_result = roi_cropper.crop_with_detect_area(
                        std_preprocessed_img, detect_area, adjusted_std_rois
                    )
                else:
                    std_crop_result = roi_cropper.crop_with_detect_area(
                        std_preprocessed_img, detect_area, std_rois
                    )

                std_detect_area_img = std_crop_result["detect_area_img"]
                std_roi_crops = std_crop_result["rois"]
            else:
                test_detect_area_img = preprocessed_img
                std_detect_area_img = std_preprocessed_img
                std_roi_crops = roi_cropper.crop_multiple(std_detect_area_img, std_rois)
                test_roi_crops = roi_cropper.crop_multiple(test_detect_area_img, rois)

            roi_comparator = ROIComparator(similarity_threshold=0.8)
            roi_compare_result = roi_comparator.compare_roi_regions(
                std_detect_area_img,
                test_detect_area_img,
                std_roi_crops,
                test_roi_crops,
            )
            results["roi_compare"] = roi_compare_result

            print(f"ROI 对比: {roi_compare_result.get('passed', False)}")
            print(f"  总 ROI 数: {roi_compare_result.get('total_rois', 0)}")
            print(f"  有效对比: {roi_compare_result.get('valid_rois', 0)}")
            print(f"  跳过(空ROI): {roi_compare_result.get('skipped_rois', 0)}")
            print(f"  通过: {roi_compare_result.get('passed_count', 0)}")
            print(f"  失败: {roi_compare_result.get('failed_count', 0)}")
            print(
                f"  平均相似度: {roi_compare_result.get('average_similarity', 0):.3f}"
            )
            print(f"==============================\n")

        from src.core.ocr.text_recognizer import TextRecognizer

        ocr = TextRecognizer()

        all_rois_for_ocr = number_tube_rois + terminal_rois
        ocr_result = ocr.recognize(task.image_path, all_rois_for_ocr)
        results["ocr"] = ocr_result

        from src.core.color.color_detector import ColorDetector

        std_wire_rois = None
        if "std_roi_crops" in dir() and std_roi_crops:
            std_wire_rois = [
                r for r in std_roi_crops if r.get("label") in ["wire", "short_wire"]
            ]

        color_det = ColorDetector(
            station_id=task.station_id,
            std_wire_rois=std_wire_rois,
            scale_factor=scale_factor,
            std_scale_factor=std_scale if "std_scale" in dir() else scale_factor,
            test_wire_rois=wire_rois,
        )
        color_result = color_det.detect(
            task.image_path,
            wire_rois,
            use_original_image=True,
            scale_factor=scale_factor,
        )
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

            if roi_compare_result and roi_compare_result.get("failed_rois"):
                import cv2

                annotated_img = cv2.imread(annotated_path)
                failed_rois = roi_compare_result.get("failed_rois", [])

                if scale_factor != 1.0 and detect_area:
                    offset_x = int(detect_area.bbox[0] / scale_factor)
                    offset_y = int(detect_area.bbox[1] / scale_factor)
                else:
                    offset_x = int(detect_area.bbox[0]) if detect_area else 0
                    offset_y = int(detect_area.bbox[1]) if detect_area else 0
                print(
                    f"[标注] scale_factor={scale_factor}, detect_area_offset=({offset_x}, {offset_y})"
                )

                annotated_img = roi_comparator.mark_diff_areas(
                    annotated_img,
                    failed_rois,
                    color=(0, 0, 255),
                    thickness=2,
                    detect_area_offset=(offset_x, offset_y),
                    use_original_coords=False,
                    scale_factor=scale_factor,
                )
                cv2.imwrite(annotated_path, annotated_img)
                print(f"[DetectionService] 已标注差异区域: {len(failed_rois)} 个")

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
