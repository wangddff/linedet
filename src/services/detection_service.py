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

        labelme_dir = "datasets/images/train"
        std_image_path = str(Path(labelme_dir) / f"train_{task.station_id}.png")
        std_preprocessor = ImagePreprocessor()
        std_preprocessed_img = std_preprocessor.preprocess(std_image_path)
        std_scale = std_preprocessor.get_scale_factor()

        from src.core.roi import ROILoader, ROICropper

        roi_loader = ROILoader(station_id=task.station_id)
        std_json_path = Path(labelme_dir) / f"train_{task.station_id}.json"
        roi_data = roi_loader.load_from_labelme(str(std_json_path))
        detect_area = roi_data.get("detect_area")
        rois = roi_data.get("rois", [])

        roi_cropper = ROICropper()

        crop_image = preprocessed_img
        use_alignment = False
        align_result = {"success": False}
        aligned_image = None

        SIFT_ALIGNMENT_ENABLED = True

        if SIFT_ALIGNMENT_ENABLED:
            from src.core.alignment.image_aligner import ImageAligner

            aligner = ImageAligner(task.station_id)
            align_result = aligner.align_from_array_to_size(
                preprocessed_img, std_preprocessed_img
            )

            if align_result.get("success"):
                aligned_image = align_result["aligned_image"]
                print(f"\n========== SIFT对齐(全图) ==========")
                print(f"匹配点数: {align_result.get('match_count')}")
                print(f"置信度: {align_result.get('confidence'):.2f}")
                print(f"耗时: {align_result.get('elapsed_ms'):.0f}ms")
                print(f"对齐后尺寸: {aligned_image.shape}")
                print(f"=========================\n")
                cv2.imwrite(f"data/temp/task_{task.id}_aligned.png", aligned_image)
                crop_image = aligned_image
                use_alignment = True
            else:
                print(f"\n========== SIFT对齐(全图) ==========")
                print(f"对齐失败: {align_result.get('error')}")
                print(f"使用原图继续检测")
                print(f"=========================\n")

        aligned_image_path = task.image_path
        if use_alignment and aligned_image is not None:
            aligned_image_path = f"data/temp/task_{task.id}_aligned.png"

        # 根据对齐结果选择正确的坐标缩放：
        # 对齐成功 → 输出在标准图空间，ROI 用 std_scale
        # 对齐失败 → 输出在测试图空间，ROI 用 scale_factor
        active_scale = std_scale if use_alignment else scale_factor

        print(f"\n========== [DEBUG] ROI 坐标信息 ==========")
        print(f"[DEBUG] 原始图像尺寸: {img.shape}")
        print(f"[DEBUG] 预处理后尺寸: {preprocessed_img.shape}")
        print(f"[DEBUG] scale_factor: {scale_factor}, std_scale: {std_scale}")
        print(f"[DEBUG] SIFT对齐是否成功: {align_result.get('success')}")
        if align_result.get("success"):
            print(f"[DEBUG] 对齐后图像尺寸: {align_result['aligned_image'].shape}")

        if detect_area:
            print(f"[DEBUG] detect_area原始: bbox={detect_area.bbox}")

        if active_scale != 1.0:
            if detect_area:
                detect_area = detect_area.scale(active_scale)
                print(f"[DEBUG] detect_area缩放后(scale={active_scale:.3f}): bbox={detect_area.bbox}")
            rois = [roi.scale(active_scale) for roi in rois]
            print(f"[DEBUG] ROI已应用scale={active_scale:.3f}")

        self._rois = rois

        print(f"\n========== ROI 裁剪 ==========")
        if detect_area:
            print(f"检测区 (detect_area): bbox={detect_area.bbox}")

        print(f"加载的 ROI 数量: {len(rois)}")
        label_counts = {}
        for roi in rois:
            label_counts[roi.label] = label_counts.get(roi.label, 0) + 1
        print(f"各类 ROI: {label_counts}")

        print(f"[DEBUG] 裁剪用图像: {crop_image.shape}")

        if detect_area:
            crop_detect_area = detect_area
            crop_rois = rois

            print(f"[DEBUG] crop_image尺寸: {crop_image.shape}")
            print(
                f"[DEBUG] detect_area: x={crop_detect_area.bbox[0]}, y={crop_detect_area.bbox[1]}, w={crop_detect_area.bbox[2]}, h={crop_detect_area.bbox[3]}"
            )

            crop_result = roi_cropper.crop_with_detect_area(
                crop_image, crop_detect_area, crop_rois
            )
            detect_area_img = crop_result["detect_area_img"]
            cv2.imwrite(
                f"data/debug_roi/test_detect_area_{task.station_id}.png",
                detect_area_img,
            )

            number_tube_rois = roi_cropper.crop_by_label(
                detect_area_img,
                [r for r in crop_rois if r.label == "number_tube"],
                "number_tube",
            )
            terminal_rois = roi_cropper.crop_by_label(
                detect_area_img,
                [r for r in crop_rois if r.label == "terminal_hole"],
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
                crop_image, rois, "number_tube"
            )
            terminal_rois = roi_cropper.crop_by_label(crop_image, rois, "terminal_hole")
            wire_rois = roi_cropper.extract_wire_regions(crop_image, rois)
            connector_rois = roi_cropper.extract_connector_regions(crop_image, rois)

        print(f"号码管 ROI: {len(number_tube_rois)}")
        print(f"端子 ROI: {len(terminal_rois)}")
        print(f"线材 ROI: {len(wire_rois)}")
        print(f"插头 ROI: {len(connector_rois)}")
        print(f"============================\n")

        from src.core.comparator.standard_comparator import StandardComparator

        # SIFT对齐成功时，用对齐置信度直接作为全局相似度，跳过基于轮廓的StandardComparator
        # （轮廓比较方案在upscale后图像上效果极差）
        if use_alignment:
            sift_confidence = align_result.get("confidence", 0)
            compare_result = {
                "passed": True,
                "similarity_score": float(sift_confidence),
                "method": "sift_alignment",
                "details": {
                    "match_count": align_result.get("match_count", 0),
                    "confidence": sift_confidence,
                },
            }
            print(f"\n========== 全局相似度（SIFT bypass） ==========")
            print(f"SIFT置信度: {sift_confidence:.2f}  → 自动通过")
            print(f"================================================\n")
        else:
            comparator = StandardComparator(task.station_id, task.product_id)
            compare_result = comparator.compare(aligned_image_path)
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

        if std_json_path.exists():
            # 加载标准图的 ROI 数据（独立于检测图的 ROI）
            std_roi_data = roi_loader.load_from_labelme(str(std_json_path))
            std_detect_area = std_roi_data.get("detect_area")
            std_rois = std_roi_data.get("rois", [])

            # 标准图 ROI 坐标总是用 std_scale
            if std_scale != 1.0:
                if std_detect_area:
                    std_detect_area = std_detect_area.scale(std_scale)
                std_rois = [roi.scale(std_scale) for roi in std_rois]

            if detect_area:
                crop_result = roi_cropper.crop_with_detect_area(
                    crop_image, detect_area, rois
                )
                test_detect_area_img = crop_result["detect_area_img"]
                test_roi_crops = crop_result["rois"]

                if std_detect_area and not use_alignment:
                    # 未对齐时，检测图与标准图的 detect_area 可能有位置偏差，需修正
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
                    # 对齐成功时两张图在同一坐标空间，直接用 std_detect_area 裁剪标准图
                    std_crop_da = std_detect_area if std_detect_area else detect_area
                    std_crop_result = roi_cropper.crop_with_detect_area(
                        std_preprocessed_img, std_crop_da, std_rois
                    )

                std_detect_area_img = std_crop_result["detect_area_img"]
                std_roi_crops = std_crop_result["rois"]
            else:
                test_detect_area_img = crop_image
                std_detect_area_img = std_preprocessed_img
                std_roi_crops = roi_cropper.crop_multiple(std_detect_area_img, std_rois)
                test_roi_crops = roi_cropper.crop_multiple(test_detect_area_img, rois)

            from src.utils.config import get_roi_similarity_threshold
            roi_comparator = ROIComparator(similarity_threshold=get_roi_similarity_threshold())
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
        ocr_result = ocr.recognize(aligned_image_path, all_rois_for_ocr)
        results["ocr"] = ocr_result

        from src.core.color.color_detector import ColorDetector

        std_wire_rois = None
        test_wire_rois = None
        if "std_roi_crops" in dir() and std_roi_crops:
            std_wire_rois = [
                r for r in std_roi_crops if r.get("label") in ["wire", "short_wire"]
            ]
        if "test_roi_crops" in dir() and test_roi_crops:
            test_wire_rois = [
                r for r in test_roi_crops if r.get("label") in ["wire", "short_wire"]
            ]

        color_det = ColorDetector(
            station_id=task.station_id,
            std_wire_rois=std_wire_rois,
            test_wire_rois=test_wire_rois,
            scale_factor=scale_factor,
        )
        color_result = color_det.detect()
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
            if use_alignment and aligned_image is not None:
                import cv2

                cv2.imwrite(f"data/temp/task_{task.id}_annotation.png", aligned_image)
                annotation_image_path = f"data/temp/task_{task.id}_annotation.png"
                annotation_scale = 1.0
                print(f"[DEBUG] 使用对齐后的图像进行标注")
            else:
                annotation_image_path = task.image_path
                annotation_scale = scale_factor
                print(f"[DEBUG] 使用原图进行标注, scale={annotation_scale}")

            output_path = f"data/exports/task_{task.id}_annotated.png"
            annotated_path = annotator.annotate_image(
                annotation_image_path,
                detection_result,
                ocr_result,
                color_result,
                output_path=output_path,
            )

            if roi_compare_result and roi_compare_result.get("failed_rois"):
                import cv2

                annotated_img = cv2.imread(annotated_path)
                failed_rois = roi_compare_result.get("failed_rois", [])

                if use_alignment:
                    # 对齐后 bbox 是相对于 detect_area 裁剪图的坐标，
                    # 需要加上 detect_area 在对齐图中的偏移才能画到正确位置
                    offset_x = detect_area.bbox[0] if detect_area else 0
                    offset_y = detect_area.bbox[1] if detect_area else 0
                    annot_scale = 1.0
                elif active_scale != 1.0 and detect_area:
                    offset_x = int(detect_area.bbox[0] / active_scale)
                    offset_y = int(detect_area.bbox[1] / active_scale)
                    annot_scale = active_scale
                else:
                    offset_x = int(detect_area.bbox[0]) if detect_area else 0
                    offset_y = int(detect_area.bbox[1]) if detect_area else 0
                    annot_scale = active_scale

                print(
                    f"[标注] offset=({offset_x}, {offset_y}), annot_scale={annot_scale}"
                )
                print(f"[DEBUG] 失败ROI数量: {len(failed_rois)}")

                annotated_img = roi_comparator.mark_diff_areas(
                    annotated_img,
                    failed_rois,
                    color=(0, 0, 255),
                    thickness=2,
                    detect_area_offset=(offset_x, offset_y),
                    use_original_coords=False,
                    scale_factor=annot_scale,
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
