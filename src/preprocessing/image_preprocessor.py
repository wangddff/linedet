import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class ImagePreprocessor:
    """图像预处理模块"""

    STANDARD_WIDTH = 1920
    STANDARD_HEIGHT = 1080

    def __init__(self):
        self.CLAHE_CLIP_LIMIT = 2.0
        self.CLAHE_TILE_SIZE = (8, 8)
        self._original_image = None
        self._scale_factor = None

    def preprocess(self, image_path: str) -> np.ndarray:
        """完整预处理流程"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        self._original_image = img.copy()

        img = self.resize_to_standard(img)
        img = self.denoise(img)
        img = self._correct_distortion(img)
        img = self._enhance_contrast(img)
        img = self._auto_white_balance(img)

        return img

    def resize_to_standard(self, img: np.ndarray) -> np.ndarray:
        """手机图片缩放至标准分辨率 (1920x1080)，确保ROI坐标精准匹配"""
        h, w = img.shape[:2]

        if h == self.STANDARD_HEIGHT and w == self.STANDARD_WIDTH:
            self._scale_factor = 1.0
            return img

        scale = min(self.STANDARD_WIDTH / w, self.STANDARD_HEIGHT / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        result = np.zeros(
            (self.STANDARD_HEIGHT, self.STANDARD_WIDTH, 3), dtype=np.uint8
        )
        y_offset = (self.STANDARD_HEIGHT - new_h) // 2
        x_offset = (self.STANDARD_WIDTH - new_w) // 2
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        self._scale_factor = scale
        return result

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """轻量高斯模糊降噪，消除手机拍摄噪点"""
        return cv2.GaussianBlur(img, (5, 5), 0)

    def get_scale_factor(self) -> float:
        """获取缩放因子，用于坐标转换"""
        return self._scale_factor or 1.0

    def get_original_size(self) -> Tuple[int, int]:
        """获取原始图片尺寸"""
        if self._original_image is not None:
            return self._original_image.shape[1], self._original_image.shape[0]
        return 0, 0

    def transform_coords(self, coords: np.ndarray) -> np.ndarray:
        """将原始图片坐标转换到缩放后坐标"""
        if self._scale_factor is None or self._scale_factor == 1.0:
            return coords

        scale = self._scale_factor
        if len(coords) == 2:
            return [int(coords[0] * scale), int(coords[1] * scale)]
        return [[int(p[0] * scale), int(p[1] * scale)] for p in coords]

    def _correct_distortion(self, img: np.ndarray) -> np.ndarray:
        """相机畸变校正"""
        h, w = img.shape[:2]
        camera_matrix = np.array(
            [[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32
        )
        dist_coeffs = np.zeros(5)
        return cv2.undistort(img, camera_matrix, dist_coeffs)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """CLAHE直方图均衡化"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_SIZE
        )
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """自动白平衡"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])

        result[:, :, 1] = result[:, :, 1] - (
            (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
        )
        result[:, :, 2] = result[:, :, 2] - (
            (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
        )

        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def correct_skew(self, img: np.ndarray) -> np.ndarray:
        """倾斜矫正"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None or len(lines) == 0:
            return img

        angles = []
        for line in lines[:10]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)

        median_angle = np.median(angles)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

        return cv2.warpAffine(
            img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

    def crop_roi(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """裁剪ROI区域"""
        x, y, w, h = bbox
        return img[y : y + h, x : x + w]

    def resize_with_padding(
        self, img: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """带padding的缩放"""
        h, w = img.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return result
