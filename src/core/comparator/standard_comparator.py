import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class StandardComparator:
    """标准图对比模块 - 核心新增模块"""

    def __init__(self, station_id: int, product_id: Optional[int] = None):
        self.station_id = station_id
        self.product_id = product_id
        self.standards = self._load_standard_images()
        self.threshold = 0.85

    def _load_standard_images(self) -> List[str]:
        """加载该工位的所有标准图"""
        base_dir = Path("data/standard_images")
        station_dir = base_dir / f"station_{self.station_id}"

        if not station_dir.exists():
            return []

        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(list(station_dir.glob(ext)))

        return [str(f) for f in image_files]

    def compare(self, image_path: str) -> Dict[str, Any]:
        """与所有标准图逐一比对，任一通过即可"""
        if not self.standards:
            return {
                "passed": True,
                "similarity_score": 1.0,
                "error": "未配置标准图，自动通过",
                "details": {},
            }

        img = cv2.imread(image_path)
        if img is None:
            return {
                "passed": False,
                "similarity_score": 0,
                "error": "无法读取检测图片",
                "details": {},
            }

        best_result = None
        best_score = 0

        for std_path in self.standards:
            result = self._compare_single(img, std_path)
            if result["similarity_score"] > best_score:
                best_score = result["similarity_score"]
                best_result = result

        if best_result is None:
            best_result = {
                "passed": False,
                "similarity_score": 0,
                "error": "标准图对比失败",
                "details": {},
            }

        best_result["passed"] = best_score >= self.threshold
        return best_result

    def _compare_single(self, img: np.ndarray, std_path: str) -> Dict[str, Any]:
        """与单张标准图对比"""
        std_img = cv2.imread(std_path)
        if std_img is None:
            return {
                "passed": False,
                "similarity_score": 0,
                "error": f"无法读取标准图: {std_path}",
                "details": {},
            }

        img = self._resize_to_match(img, std_img)

        targets1 = self._extract_features(img)
        targets2 = self._extract_features(std_img)

        match_result = self._match_targets(targets1, targets2)

        similarity = self._calc_similarity(
            match_result["matched_count"],
            match_result["position_errors"],
            len(targets1),
            len(targets2),
        )

        return {
            "passed": similarity >= self.threshold,
            "similarity_score": similarity,
            "standard_image": std_path,
            "detected_count": len(targets1),
            "standard_count": len(targets2),
            "matched_count": match_result["matched_count"],
            "position_errors": match_result["position_errors"],
            "details": match_result,
        }

    def _resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """调整图片尺寸一致"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 == h2 and w1 == w2:
            return img1

        scale = min(w2 / w1, h2 / h1)
        new_w, new_h = int(w1 * scale), int(h1 * scale)
        return cv2.resize(img1, (new_w, new_h))

    def _extract_features(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """提取图像特征（简化版：使用轮廓检测）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        features = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                features.append(
                    {
                        "bbox": (x, y, w, h),
                        "area": area,
                        "center": (x + w // 2, y + h // 2),
                    }
                )

        return features

    def _match_targets(self, targets1: List, targets2: List) -> Dict[str, Any]:
        """目标匹配"""
        if not targets1 or not targets2:
            return {
                "matched_count": 0,
                "position_errors": [],
                "unmatched_detected": len(targets1),
                "unmatched_standard": len(targets2),
            }

        matched = 0
        errors = []
        used_indices = set()

        for t1 in targets1:
            best_match_idx = -1
            best_distance = float("inf")

            for idx, t2 in enumerate(targets2):
                if idx in used_indices:
                    continue

                dist = np.sqrt(
                    (t1["center"][0] - t2["center"][0]) ** 2
                    + (t1["center"][1] - t2["center"][1]) ** 2
                )

                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = idx

            if best_match_idx >= 0 and best_distance < 100:
                matched += 1
                used_indices.add(best_match_idx)
                if best_distance > 20:
                    errors.append(
                        {
                            "detected": t1["center"],
                            "expected": targets2[best_match_idx]["center"],
                            "distance": best_distance,
                        }
                    )

        return {
            "matched_count": matched,
            "position_errors": errors,
            "unmatched_detected": len(targets1) - matched,
            "unmatched_standard": len(targets2) - matched,
        }

    def _calc_similarity(
        self,
        matched: int,
        position_errors: List,
        detected_count: int,
        standard_count: int,
    ) -> float:
        """计算相似度"""
        if standard_count == 0:
            return 1.0 if detected_count == 0 else 0.0

        count_ratio = matched / max(standard_count, 1)

        position_penalty = len(position_errors) * 0.1

        count_penalty = abs(detected_count - standard_count) * 0.15

        similarity = count_ratio - position_penalty - count_penalty

        return max(0, min(1, similarity))
