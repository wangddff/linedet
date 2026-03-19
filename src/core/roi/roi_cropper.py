import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from .roi_loader import ROI


class ROICropper:
    """根据 ROI 坐标精准裁剪图片区域

    支持 detect_area（检测区）裁剪：
    - 先裁剪检测区，生成子图
    - 再在子图上裁剪具体 ROI 区域
    - ROI 坐标会自动转换到子图坐标系
    """

    def __init__(self, target_size: Optional[tuple] = None):
        self.target_size = target_size

    def crop_detect_area(self, img: np.ndarray, detect_area: ROI) -> np.ndarray:
        """裁剪检测区（全局大 ROI）"""
        if detect_area is None:
            return img

        x, y, w, h = detect_area.bbox

        h_img, w_img = img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            return img

        cropped = img[y1:y2, x1:x2]

        if self.target_size and cropped.size > 0:
            cropped = self._resize_to_target(cropped)

        return cropped

    def adjust_roi_coords_to_detect_area(
        self, rois: List[ROI], detect_area: ROI
    ) -> List[ROI]:
        """将 ROI 坐标转换到检测区子图的坐标系"""
        if detect_area is None:
            return rois

        offset_x, offset_y = detect_area.bbox[0], detect_area.bbox[1]

        adjusted_rois = []
        for roi in rois:
            adjusted_points = [[p[0] - offset_x, p[1] - offset_y] for p in roi.points]
            adjusted_roi = ROI(roi.label, adjusted_points, roi.group_id)
            adjusted_rois.append(adjusted_roi)

        return adjusted_rois

    def crop_with_detect_area(
        self, img: np.ndarray, detect_area: Optional[ROI], rois: List[ROI]
    ) -> Dict[str, Any]:
        """裁剪检测区并在其中裁剪具体 ROI

        返回:
            {
                "detect_area_img": 裁剪后的检测区图像,
                "detect_area": 检测区 ROI,
                "rois": 在检测区内的 ROI 裁剪结果
            }
        """
        if detect_area is None:
            return {
                "detect_area_img": img,
                "detect_area": None,
                "rois": self.crop_multiple(img, rois),
            }

        detect_area_img = self.crop_detect_area(img, detect_area)

        adjusted_rois = self.adjust_roi_coords_to_detect_area(rois, detect_area)

        cropped_rois = self.crop_multiple(detect_area_img, adjusted_rois)

        for i, cropped in enumerate(cropped_rois):
            original_bbox = rois[i].bbox
            cropped["original_bbox"] = original_bbox
            cropped["original_center"] = rois[i].center

        return {
            "detect_area_img": detect_area_img,
            "detect_area": detect_area,
            "rois": cropped_rois,
        }

    def crop_single(self, img: np.ndarray, roi: ROI) -> np.ndarray:
        """裁剪单个 ROI 区域"""
        x, y, w, h = roi.bbox

        h_img, w_img = img.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            return np.array([])

        cropped = img[y1:y2, x1:x2]

        if self.target_size and cropped.size > 0:
            cropped = self._resize_to_target(cropped)

        return cropped

    def crop_multiple(self, img: np.ndarray, rois: List[ROI]) -> List[Dict[str, Any]]:
        """裁剪多个 ROI 区域"""
        results = []

        for roi in rois:
            cropped = self.crop_single(img, roi)

            results.append(
                {
                    "label": roi.label,
                    "bbox": roi.bbox,
                    "center": roi.center,
                    "group_id": roi.group_id,
                    "roi": cropped,
                    "roi_shape": cropped.shape if cropped.size > 0 else None,
                }
            )

        return results

    def crop_by_label(
        self, img: np.ndarray, rois: List[ROI], label: str
    ) -> List[Dict[str, Any]]:
        """按标签裁剪 ROI"""
        filtered = [r for r in rois if r.label == label]
        return self.crop_multiple(img, filtered)

    def crop_all_labels(
        self,
        img: np.ndarray,
        rois: List[ROI],
        label_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """按标签分组裁剪所有 ROI"""
        if label_map is None:
            label_map = {
                "terminal_hole": "terminal_hole",
                "number_tube": "number_tube",
                "wire": "wire",
                "connector": "connector",
            }

        results = {}

        for roi_label, target_label in label_map.items():
            if roi_label in [r.label for r in rois]:
                results[target_label] = self.crop_by_label(img, rois, roi_label)

        return results

    def _resize_to_target(self, img: np.ndarray) -> np.ndarray:
        """将裁剪的图片缩放到目标尺寸"""
        if self.target_size:
            return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        return img

    def extract_wire_regions(
        self, img: np.ndarray, rois: List[ROI]
    ) -> List[Dict[str, Any]]:
        """专门提取线材区域（带排序）"""
        wire_rois = [r for r in rois if r.label in ["wire", "short_wire"]]

        results = []
        for roi in wire_rois:
            cropped = self.crop_single(img, roi)
            if cropped.size > 0:
                results.append(
                    {
                        "label": roi.label,
                        "bbox": roi.bbox,
                        "center": roi.center,
                        "group_id": roi.group_id,
                        "roi": cropped,
                    }
                )

        results.sort(key=lambda x: x["center"][0])
        return results

    def extract_text_regions(
        self, img: np.ndarray, rois: List[ROI]
    ) -> List[Dict[str, Any]]:
        """专门提取文字区域（号码管等）"""
        text_rois = [r for r in rois if r.label in ["number_tube", "terminal_number"]]

        results = []
        for roi in text_rois:
            cropped = self.crop_single(img, roi)
            if cropped.size > 0:
                results.append(
                    {
                        "label": roi.label,
                        "bbox": roi.bbox,
                        "center": roi.center,
                        "group_id": roi.group_id,
                        "roi": cropped,
                    }
                )

        results.sort(key=lambda x: x["center"][0])
        return results

    def extract_connector_regions(
        self, img: np.ndarray, rois: List[ROI]
    ) -> List[Dict[str, Any]]:
        """专门提取插头区域"""
        connector_rois = [r for r in rois if r.label == "connector"]

        results = []
        for roi in connector_rois:
            cropped = self.crop_single(img, roi)
            if cropped.size > 0:
                results.append(
                    {
                        "label": roi.label,
                        "bbox": roi.bbox,
                        "center": roi.center,
                        "group_id": roi.group_id,
                        "roi": cropped,
                    }
                )

        results.sort(key=lambda x: x["center"][1])
        return results
