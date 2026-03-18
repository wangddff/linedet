import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


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

    def __init__(self, station_id: Optional[int] = None):
        self.hsv_image = None
        self.station_id = station_id
        self.standard_colors = {}
        if station_id:
            self.standard_colors = self._load_standard_colors(station_id)

    def _load_standard_colors(self, station_id: int) -> Dict[str, Any]:
        """从标准图提取标准颜色"""
        base_dir = Path("data/standard_images")
        station_dir = base_dir / f"station_{station_id}"

        if not station_dir.exists():
            print(f"[ColorDetector] 标准图目录不存在: {station_dir}")
            return {}

        standard_colors = {}
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for std_path in station_dir.glob(ext):
                std_img = cv2.imread(str(std_path))
                if std_img is None:
                    continue

                print(f"[ColorDetector] 正在提取标准图颜色: {std_path.name}")

                hsv_std = cv2.cvtColor(std_img, cv2.COLOR_BGR2HSV)

                wires = self._detect_wire_regions(hsv_std)

                for i, wire in enumerate(wires):
                    color_name = wire.get("color", "未知")
                    standard_colors[f"wire_{i}"] = {
                        "color": color_name,
                        "position": wire.get("position"),
                        "hsv": wire.get("hsv"),
                    }

                print(f"[ColorDetector] 标准图颜色提取完成: {standard_colors}")

        return standard_colors

    def _detect_wire_regions(self, hsv_img: np.ndarray) -> List[Dict[str, Any]]:
        """检测线材区域并识别颜色"""
        h, w = hsv_img.shape[:2]

        wire_regions = []

        h_channel = hsv_img[:, :, 0]
        s_channel = hsv_img[:, :, 1]

        mask = s_channel > 50

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            if ch < 5 or cw < 10:
                continue

            roi_h = hsv_img[y : y + ch, x : x + cw]
            if roi_h.size == 0:
                continue

            h_mean = float(np.mean(roi_h[:, :, 0]))
            s_mean = float(np.mean(roi_h[:, :, 1]))
            v_mean = float(np.mean(roi_h[:, :, 2]))

            color_name = self._get_color_name(h_mean, s_mean, v_mean)

            wire_regions.append(
                {
                    "color": color_name,
                    "position": (x + cw // 2, y + ch // 2),
                    "hsv": {"h": float(h_mean), "s": float(s_mean), "v": float(v_mean)},
                    "bbox": [x, y, cw, ch],
                }
            )

        wire_regions.sort(key=lambda x: x["position"][0])

        return wire_regions

    def _get_color_name(
        self, h: float, s: float, v: float, threshold: float = 0.5
    ) -> str:
        """根据 HSV 值获取颜色名称"""
        detected_color = None
        max_similarity = 0

        for color_name, ranges in self.WIRE_COLORS.items():
            h_min, h_max = ranges["h_min"], ranges["h_max"]
            s_min, s_max = ranges["s_min"], ranges["s_max"]
            v_min, v_max = ranges["v_min"], ranges["v_max"]

            if color_name == "红色":
                h_in_range = (0 <= h <= 10) or (170 <= h <= 180)
            else:
                h_in_range = h_min <= h <= h_max

            s_in_range = s_min <= s <= s_max
            v_in_range = v_min <= v <= v_max

            if h_in_range and s_in_range and v_in_range:
                similarity = 1.0
            else:
                if color_name == "红色":
                    if h <= 10:
                        h_dist = min(abs(h - h_min), abs(h - 0), 30)
                    else:
                        h_dist = min(abs(h - h_max), abs(h - 180), 30)
                else:
                    h_dist = min(abs(h - h_min), abs(h - h_max), 30)
                s_dist = min(abs(s - s_min), abs(s - s_max)) / 255
                v_dist = min(abs(v - v_min), abs(v - v_max)) / 255
                similarity = 1.0 - (h_dist / 30 + s_dist + v_dist) / 3

            if similarity > max_similarity:
                max_similarity = similarity
                detected_color = color_name

        if max_similarity < threshold:
            return "未知"
        return detected_color or "未知"

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

        compare_result = self._compare_with_standard(results)

        passed = (
            len(results) > 0
            and all(r.get("detected") for r in results)
            and compare_result.get("passed", False)
        )

        print(f"\n========== Color Detection Result ==========")
        print(f"检测到的颜色: {[r.get('color') for r in results]}")
        print(f"标准颜色: {[v.get('color') for v in self.standard_colors.values()]}")
        print(
            f"颜色比对结果: passed={compare_result.get('passed')}, message={compare_result.get('message')}"
        )
        if compare_result.get("errors"):
            print(f"颜色问题: {compare_result['errors']}")
        print(f"=============================================\n")

        return {
            "passed": passed,
            "colors": results,
            "wire_diameters": wire_diameters,
            "total_wires": len(results),
            "compare_result": compare_result,
        }

    def _compare_with_standard(self, current_colors: List[Dict]) -> Dict[str, Any]:
        """与标准图颜色比对"""
        if not self.standard_colors:
            return {"passed": True, "message": "无标准颜色，跳过比对", "errors": []}

        errors = []
        current_color_list = [c.get("color") for c in current_colors]
        standard_color_list = [v.get("color") for v in self.standard_colors.values()]

        if len(current_color_list) < len(standard_color_list):
            errors.append(
                {
                    "type": "missing_wire",
                    "message": f"线材数量不足: 期望 {len(standard_color_list)}, 实际 {len(current_color_list)}",
                }
            )

        for i, std_color in enumerate(standard_color_list):
            if i < len(current_color_list):
                curr_color = current_color_list[i]
                if (
                    curr_color != std_color
                    and curr_color != "未知"
                    and std_color != "未知"
                ):
                    errors.append(
                        {
                            "type": "color_mismatch",
                            "wire_index": i,
                            "expected": std_color,
                            "actual": curr_color,
                            "message": f"线{i + 1}颜色不匹配: 期望 {std_color}, 实际 {curr_color}",
                        }
                    )

        passed = len(errors) == 0
        return {
            "passed": passed,
            "message": "颜色比对通过" if passed else f"发现 {len(errors)} 个颜色问题",
            "errors": errors,
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
            h_min, h_max = ranges["h_min"], ranges["h_max"]

            if color_name == "红色":
                h_in_range = (0 <= h_mean <= 10) or (170 <= h_mean <= 180)
            else:
                h_in_range = h_min <= h_mean <= h_max

            s_in_range = ranges["s_min"] <= s_mean <= ranges["s_max"]
            v_in_range = ranges["v_min"] <= v_mean <= ranges["v_max"]

            if h_in_range and s_in_range and v_in_range:
                similarity = 1.0
            else:
                if color_name == "红色":
                    if h_mean <= 10:
                        h_dist = min(abs(h_mean - h_min), abs(h_mean - 0), 30)
                    else:
                        h_dist = min(abs(h_mean - h_max), abs(h_mean - 180), 30)
                else:
                    h_dist = min(abs(h_mean - h_min), abs(h_mean - h_max), 30)
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

        if best_similarity > 0.8 and best_match is not None:
            return str(best_match)
        return "未知"

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
