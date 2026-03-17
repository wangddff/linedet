import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class ColorDetector:
    """颜色检测模块 - 线材颜色检测、线径检测"""

    WIRE_COLORS = {
        "红色": {
            "h_min": 0,
            "h_max": 10,
            "s_min": 100,
            "s_max": 255,
            "v_min": 50,
            "v_max": 255,
        },
        "蓝色": {
            "h_min": 100,
            "h_max": 130,
            "s_min": 80,
            "s_max": 255,
            "v_min": 30,
            "v_max": 255,
        },
        "绿色": {
            "h_min": 40,
            "h_max": 80,
            "s_min": 60,
            "s_max": 255,
            "v_min": 30,
            "v_max": 255,
        },
        "黄色": {
            "h_min": 15,
            "h_max": 35,
            "s_min": 100,
            "s_max": 255,
            "v_min": 50,
            "v_max": 255,
        },
        "白色": {
            "h_min": 0,
            "h_max": 180,
            "s_min": 0,
            "s_max": 30,
            "v_min": 180,
            "v_max": 255,
        },
        "黑色": {
            "h_min": 0,
            "h_max": 180,
            "s_min": 0,
            "s_max": 50,
            "v_min": 0,
            "v_max": 100,
        },
        "橙色": {
            "h_min": 5,
            "h_max": 20,
            "s_min": 100,
            "s_max": 255,
            "v_min": 50,
            "v_max": 255,
        },
        "紫色": {
            "h_min": 130,
            "h_max": 160,
            "s_min": 60,
            "s_max": 255,
            "v_min": 30,
            "v_max": 255,
        },
        "棕色": {
            "h_min": 0,
            "h_max": 20,
            "s_min": 60,
            "s_max": 255,
            "v_min": 20,
            "v_max": 150,
        },
        "灰色": {
            "h_min": 0,
            "h_max": 180,
            "s_min": 0,
            "s_max": 30,
            "v_min": 80,
            "v_max": 180,
        },
    }

    PIXEL_TO_MM_RATIO = 0.05

    def __init__(self):
        self.hsv_image = None

    def detect(
        self, image_path: str, wire_rois: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """执行颜色检测"""
        if wire_rois is None:
            wire_rois = []

        img = cv2.imread(image_path)
        if img is None:
            return {
                "passed": False,
                "error": f"无法读取图片: {image_path}",
                "colors": [],
                "wire_diameters": [],
            }

        self.hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        results = []

        if wire_rois:
            for roi_info in wire_rois:
                roi_img = roi_info.get("roi")
                if roi_img is None or roi_img.size == 0:
                    continue

                color_result = self._detect_roi_color(roi_img)
                if color_result:
                    color_result["bbox"] = roi_info.get("bbox")
                    results.append(color_result)
        else:
            color_result = self._detect_roi_color(img)
            if color_result:
                results.append(color_result)

        wire_diameters = self._detect_wire_diameter(img, wire_rois)

        passed = len(results) > 0 and all(r.get("detected") for r in results)

        return {
            "passed": passed,
            "colors": results,
            "wire_diameters": wire_diameters,
            "total_wires": len(results),
        }

    def _detect_roi_color(self, roi: np.ndarray) -> Optional[Dict[str, Any]]:
        """检测ROI区域的线材颜色"""
        if roi.size == 0:
            return None

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_roi)

        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        detected_color = None
        max_similarity = 0

        for color_name, ranges in self.WIRE_COLORS.items():
            h_in_range = ranges["h_min"] <= h_mean <= ranges["h_max"]
            s_in_range = ranges["s_min"] <= s_mean <= ranges["s_max"]
            v_in_range = ranges["v_min"] <= v_mean <= ranges["v_max"]

            if h_in_range and s_in_range and v_in_range:
                similarity = 1.0
            else:
                h_dist = min(
                    abs(h_mean - ranges["h_min"]), abs(h_mean - ranges["h_max"]), 30
                )
                s_dist = (
                    min(abs(s_mean - ranges["s_min"]), abs(s_mean - ranges["s_max"]))
                    / 255
                )
                v_dist = (
                    min(abs(v_mean - ranges["v_min"]), abs(v_mean - ranges["v_max"]))
                    / 255
                )
                similarity = 1.0 - (h_dist / 30 + s_dist + v_dist) / 3

            if similarity > max_similarity:
                max_similarity = similarity
                detected_color = color_name

        if max_similarity < 0.5:
            detected_color = "未知"

        return {
            "detected": detected_color is not None,
            "color": detected_color or "未知",
            "confidence": max_similarity,
            "hsv": {"h": float(h_mean), "s": float(s_mean), "v": float(v_mean)},
        }

    def _detect_wire_diameter(
        self, img: np.ndarray, wire_rois: List[Dict]
    ) -> List[Dict[str, Any]]:
        """检测线径"""
        diameters = []

        if not wire_rois:
            wire_rois = [{"bbox": [0, 0, img.shape[1], img.shape[0]]}]

        for i, roi_info in enumerate(wire_rois):
            bbox = roi_info.get("bbox", [0, 0, img.shape[1], img.shape[0]])
            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                continue

            roi = img[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=20,
                maxLineGap=5,
            )

            if lines is not None and len(lines) > 0:
                wire_widths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    wire_widths.append(width)

                avg_pixel_width = np.median(wire_widths)
                diameter_mm = avg_pixel_width * self.PIXEL_TO_MM_RATIO

                diameters.append(
                    {
                        "wire_index": i,
                        "pixel_width": float(avg_pixel_width),
                        "diameter_mm": round(diameter_mm, 2),
                        "bbox": bbox,
                    }
                )

        return diameters

    def detect_with_color_card(
        self, roi: np.ndarray, color_card: Dict[str, Tuple[int, int, int]]
    ) -> str:
        """使用色卡比对检测颜色（余弦相似度）"""
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_mean, s_mean, v_mean = (
            np.mean(hsv_roi[:, :, 0]),
            np.mean(hsv_roi[:, :, 1]),
            np.mean(hsv_roi[:, :, 2]),
        )

        roi_vec = np.array([h_mean / 180, s_mean / 255, v_mean / 255])

        best_match = None
        best_similarity = -1

        for color_name, (h, s, v) in color_card.items():
            card_vec = np.array([h / 180, s / 255, v / 255])
            similarity = np.dot(roi_vec, card_vec) / (
                np.linalg.norm(roi_vec) * np.linalg.norm(card_vec) + 1e-6
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = color_name

        return best_match if best_similarity > 0.8 else "未知"

    def get_color_histogram(self, roi: np.ndarray) -> Dict[str, np.ndarray]:
        """获取颜色直方图"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        return {
            "h": h_hist.flatten(),
            "s": s_hist.flatten(),
            "v": v_hist.flatten(),
        }
