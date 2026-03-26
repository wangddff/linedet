import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

DEBUG_SAVE_ROI_IMAGES = True


class ROIComparator:
    """ROI 区域对比模块 - 按标签对比标准图与待检图的各个 ROI 区域

    对比逻辑：
    1. 按标签（label）分组：terminal_hole, number_tube, wire, connector 等
    2. 同一标签下，按位置（center x 坐标）排序
    3. 同一标签的一一对比（按顺序配对）
    4. 计算每个 ROI 区域的相似度
    """

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold

    def compare_roi_regions(
        self,
        std_img: np.ndarray,
        test_img: np.ndarray,
        std_rois: List[Dict[str, Any]],
        test_rois: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """对比两个图片中的所有 ROI 区域（按标签配对）

        Args:
            std_img: 标准图
            test_img: 待检图
            std_rois: 标准图的 ROI 列表 [{"label": "wire", "roi": ..., "bbox": ..., "center": ...}, ...]
            test_rois: 待检图的 ROI 列表

        Returns:
            对比结果，包含每个 ROI 的相似度及差异区域
        """
        if not std_rois or not test_rois:
            return {
                "passed": True,
                "total_rois": len(std_rois),
                "compared_rois": [],
                "error": "无 ROI 数据",
            }

        compared_results = []
        total_similarity = 0
        passed_count = 0
        failed_rois = []

        test_rois_by_label = self._group_rois_by_label(test_rois)
        std_rois_by_label = self._group_rois_by_label(std_rois)

        debug_dir = None
        if DEBUG_SAVE_ROI_IMAGES:
            debug_dir = "data/debug_roi"
            os.makedirs(debug_dir, exist_ok=True)

        for label, std_label_rois in std_rois_by_label.items():
            test_label_rois = test_rois_by_label.get(label, [])

            for i, std_roi in enumerate(std_label_rois):
                std_crop = std_roi.get("roi")
                if std_crop is None or std_crop.size == 0:
                    continue

                if DEBUG_SAVE_ROI_IMAGES:
                    cv2.imwrite(f"{debug_dir}/std_{label}_{i}.png", std_crop)

                test_roi = test_label_rois[i] if i < len(test_label_rois) else None

                if DEBUG_SAVE_ROI_IMAGES and test_roi is not None:
                    test_crop = test_roi.get("roi")
                    if test_crop is not None and test_crop.size > 0:
                        cv2.imwrite(f"{debug_dir}/test_{label}_{i}.png", test_crop)

                if test_roi is None:
                    compared_results.append(
                        {
                            "label": label,
                            "index": i,
                            "passed": False,
                            "similarity": 0,
                            "error": "待检图中无对应 ROI",
                        }
                    )
                    failed_rois.append({"label": label, "index": i})
                    continue

                test_crop = test_roi.get("roi")
                if test_crop is None or test_crop.size == 0:
                    compared_results.append(
                        {
                            "label": label,
                            "index": i,
                            "passed": False,
                            "similarity": 0,
                            "error": "待检图 ROI 裁剪失败",
                        }
                    )
                    failed_rois.append({"label": label, "index": i})
                    continue

                similarity, diff_mask = self._compare_single_roi(std_crop, test_crop)

                std_center = std_roi.get("center", [0, 0])
                test_center = test_roi.get("center", [0, 0])

                compared_results.append(
                    {
                        "label": label,
                        "index": i,
                        "std_bbox": std_roi.get("bbox"),
                        "test_bbox": test_roi.get("bbox"),
                        "std_center": std_center,
                        "test_center": test_center,
                        "passed": similarity >= self.similarity_threshold,
                        "similarity": similarity,
                        "diff_mask": diff_mask,
                    }
                )

                total_similarity += similarity
                if similarity >= self.similarity_threshold:
                    passed_count += 1
                else:
                    failed_rois.append(
                        {
                            "label": label,
                            "index": i,
                            "similarity": similarity,
                            "std_bbox": std_roi.get("bbox"),
                            "test_bbox": test_roi.get("bbox"),
                        }
                    )

        avg_similarity = (
            total_similarity / len(compared_results) if compared_results else 0
        )
        overall_passed = (
            passed_count == len(compared_results) if compared_results else True
        )

        skipped_count = len(std_rois) - len(compared_results)

        return {
            "passed": overall_passed,
            "total_rois": len(std_rois),
            "valid_rois": len(compared_results),
            "skipped_rois": skipped_count,
            "compared_rois": compared_results,
            "passed_count": passed_count,
            "failed_count": len(compared_results) - passed_count,
            "average_similarity": avg_similarity,
            "failed_rois": failed_rois,
        }

    def _group_rois_by_label(
        self, rois: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """按标签分组 ROI"""
        groups = {}
        for roi in rois:
            label = roi.get("label", "unknown")
            if label not in groups:
                groups[label] = []
            groups[label].append(roi)

        for label in groups:
            groups[label].sort(
                key=lambda r: r.get("original_center", r.get("center", [0, 0]))[0]
            )

        return groups

    def _compare_single_roi(self, std_roi: np.ndarray, test_roi: np.ndarray) -> tuple:
        """对比单个 ROI 区域

        使用多方法融合计算相似度:
        1. SSIM 结构相似度
        2. 直方图相关性
        3. 像素级差异

        Returns:
            (similarity, diff_mask)
        """
        if std_roi.shape != test_roi.shape:
            test_roi = cv2.resize(
                test_roi,
                (std_roi.shape[1], std_roi.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        std_gray = (
            cv2.cvtColor(std_roi, cv2.COLOR_BGR2GRAY)
            if len(std_roi.shape) == 3
            else std_roi
        )
        test_gray = (
            cv2.cvtColor(test_roi, cv2.COLOR_BGR2GRAY)
            if len(test_roi.shape) == 3
            else test_roi
        )

        # 对每个 ROI 局部做 CLAHE 归一化，消除光照/曝光差异对比较的影响
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        std_gray = clahe.apply(std_gray)
        test_gray = clahe.apply(test_gray)

        ssim_score = self._calculate_ssim(std_gray, test_gray)

        hist_score = self._calculate_histogram_similarity(std_gray, test_gray)

        diff_mask, pixel_score = self._calculate_pixel_diff(std_gray, test_gray)

        similarity = ssim_score * 0.4 + hist_score * 0.4 + pixel_score * 0.2

        return similarity, diff_mask

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算 SSIM 结构相似度"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return float(np.mean(ssim_map))

    def _calculate_histogram_similarity(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """计算直方图相似度"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return max(0, correlation)

    def _calculate_pixel_diff(self, img1: np.ndarray, img2: np.ndarray) -> tuple:
        """计算像素级差异"""
        diff = cv2.absdiff(img1, img2)

        _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        diff_ratio = np.sum(binary_diff > 0) / (img1.shape[0] * img1.shape[1])
        pixel_score = 1.0 - diff_ratio

        return binary_diff, pixel_score

    def mark_diff_areas(
        self,
        image: np.ndarray,
        failed_rois: List[Dict[str, Any]],
        color: tuple = (0, 0, 255),
        thickness: int = 2,
        detect_area_offset: tuple = (0, 0),
        use_original_coords: bool = False,
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        """在图片上标注差异区域

        Args:
            image: 待标注的图片
            failed_rois: 失败的 ROI 列表
            color: 标注颜色 (BGR)
            thickness: 线条粗细
            detect_area_offset: 检测区偏移量 (offset_x, offset_y)，用于转换相对坐标到绝对坐标
            use_original_coords: 是否使用原始坐标（相对于原图的坐标），如果为True则不应用detect_area_offset
            scale_factor: 图像预处理缩放因子，用于将预处理后的坐标转换到原始图片坐标

        Returns:
            标注后的图片
        """
        result = image.copy()

        for roi_info in failed_rois:
            if use_original_coords:
                bbox = roi_info.get("original_bbox") or roi_info.get("std_bbox")
                if not bbox:
                    bbox = roi_info.get("test_bbox") or roi_info.get("std_bbox")
            else:
                bbox = roi_info.get("test_bbox") or roi_info.get("std_bbox")

            if bbox:
                offset_x, offset_y = detect_area_offset
                x, y, w, h = bbox
                print(
                    f"[mark_diff_areas] bbox={bbox}, scale_factor={scale_factor}, offset=({offset_x}, {offset_y})"
                )
                if scale_factor != 1.0:
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    w = int(w / scale_factor)
                    h = int(h / scale_factor)
                abs_x = x + offset_x
                abs_y = y + offset_y
                cv2.rectangle(
                    result, (abs_x, abs_y), (abs_x + w, abs_y + h), color, thickness
                )

                label = roi_info.get("label", "unknown")
                sim = roi_info.get("similarity", 0)
                text = f"{label}: {sim:.2f}"
                cv2.putText(
                    result,
                    text,
                    (abs_x, abs_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        return result


class ROIComparatorV2:
    """ROI 区域对比模块 V2 - 基于 detect_area 的精确对比"""

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        self.comparator = ROIComparator(similarity_threshold)

    def compare_with_detect_area(
        self,
        std_img: np.ndarray,
        test_img: np.ndarray,
        std_detect_area: Any,
        test_detect_area: Any,
        std_rois: List[Any],
        test_rois: List[Any],
    ) -> Dict[str, Any]:
        """在检测区内对比 ROI"""
        from src.core.roi import ROICropper

        cropper = ROICropper()

        if std_detect_area and test_detect_area:
            std_cropped = cropper.crop_detect_area(std_img, std_detect_area)
            test_cropped = cropper.crop_detect_area(test_img, test_detect_area)

            std_aligned = cropper.adjust_roi_coords_to_detect_area(
                std_rois, std_detect_area
            )
            test_aligned = cropper.adjust_roi_coords_to_detect_area(
                test_rois, test_detect_area
            )

            std_roi_results = cropper.crop_multiple(std_cropped, std_aligned)
            test_roi_results = cropper.crop_multiple(test_cropped, test_aligned)
        else:
            std_roi_results = cropper.crop_multiple(std_img, std_rois)
            test_roi_results = cropper.crop_multiple(test_img, test_rois)

        return self.comparator.compare_roi_regions(
            std_img, test_img, std_roi_results, test_roi_results
        )
