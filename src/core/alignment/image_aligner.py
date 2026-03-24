import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class ImageAligner:
    """基于 SIFT 特征点的图像对齐模块"""

    def __init__(self, station_id: int, timeout_ms: int = 500):
        self.station_id = station_id
        self.timeout_ms = timeout_ms
        self.std_image_path = self._find_standard_image()
        self.std_img: Optional[np.ndarray] = None
        self.std_kp: Optional[List] = None
        self.std_des: Optional[np.ndarray] = None
        self.sift: Optional[cv2.SIFT] = None
        self.matcher: Optional[cv2.BFMatcher] = None

        if self.std_image_path:
            self._load_and_extract()

    def _find_standard_image(self) -> Optional[str]:
        """查找标准图路径"""
        base_dir = Path("data/standard_images")
        station_dir = base_dir / f"station_{self.station_id}"

        if not station_dir.exists():
            print(f"[ImageAligner] 标准图目录不存在: {station_dir}")
            return None

        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            files = list(station_dir.glob(ext))
            if files:
                return str(files[0])

        return None

    def _load_and_extract(self) -> bool:
        """加载标准图并提取特征"""
        if not self.std_image_path:
            return False

        try:
            self.std_img = cv2.imread(self.std_image_path)
            if self.std_img is None:
                print(f"[ImageAligner] 无法读取标准图: {self.std_image_path}")
                return False

            try:
                self.sift = cv2.SIFT_create()
            except AttributeError:
                self.sift = cv2.xfeatures2d.SIFT_create()

            self.std_kp, self.std_des = self.sift.detectAndCompute(self.std_img, None)

            if self.std_kp is None or len(self.std_kp) < 10:
                print(
                    f"[ImageAligner] 标准图特征点不足: {len(self.std_kp) if self.std_kp else 0}"
                )
                return False

            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            print(f"[ImageAligner] 已加载标准图: {self.std_image_path}")
            print(f"[ImageAligner] 特征点数量: {len(self.std_kp)}")
            return True

        except Exception as e:
            print(f"[ImageAligner] 加载标准图失败: {e}")
            return False

    def align(self, test_image_path: str) -> Dict[str, Any]:
        """执行图像对齐

        返回:
        {
            "aligned_image": 对齐后的图像,
            "homography": 3x3单应性矩阵,
            "corners_mapped": 4角点映射坐标,
            "match_count": 匹配点数量,
            "confidence": 置信度,
            "success": 是否成功,
            "error": 错误信息(如有)
        }
        """
        start_time = time.time()

        if self.std_img is None or self.sift is None:
            return self._fail_result("标准图未加载")

        test_img = cv2.imread(test_image_path)
        if test_img is None:
            return self._fail_result(f"无法读取测试图: {test_image_path}")

        test_img_resized = self._resize_to_match(test_img, self.std_img)

        test_kp, test_des = self.sift.detectAndCompute(test_img_resized, None)
        if test_kp is None or len(test_kp) < 10:
            return self._fail_result("测试图特征点不足")

        matches = self.matcher.knnMatch(self.std_des, test_des, k=2)

        good_matches = self._filter_matches(matches)
        if len(good_matches) < 10:
            return self._fail_result(f"匹配点不足: {len(good_matches)}")

        H, mask = self._compute_homography(good_matches, test_kp)
        if H is None:
            return self._fail_result("无法计算单应性矩阵")

        inliers = mask.ravel().sum() if mask is not None else 0
        confidence = inliers / len(good_matches) if good_matches else 0

        if confidence < 0.3:
            return self._fail_result(f"匹配置信度低: {confidence:.2f}")

        if test_img.shape[:2] != self.std_img.shape[:2]:
            H_adjusted = self._adjust_homography(H, test_img.shape, self.std_img.shape)
            aligned_img = self._warp_image(test_img, H_adjusted)
        else:
            aligned_img = self._warp_image(test_img, H)

        corners_mapped = self._map_corners(test_img, H)

        elapsed = (time.time() - start_time) * 1000
        print(
            f"[ImageAligner] 对齐完成: {len(good_matches)}匹配, 置信度:{confidence:.2f}, 耗时:{elapsed:.0f}ms"
        )

        return {
            "aligned_image": aligned_img,
            "homography": H,
            "corners_mapped": corners_mapped,
            "match_count": len(good_matches),
            "confidence": float(confidence),
            "success": True,
            "elapsed_ms": elapsed,
            "error": None,
        }

    def _resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """将img1缩放到与img2相同尺寸"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 == h2 and w1 == w2:
            return img1

        return cv2.resize(img1, (w2, h2))

    def _adjust_homography(
        self, H: np.ndarray, src_shape: tuple, dst_shape: tuple
    ) -> np.ndarray:
        """调整homography矩阵以适应原始尺寸"""
        scale_x = dst_shape[1] / src_shape[1]
        scale_y = dst_shape[0] / src_shape[0]

        scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        return scale_matrix @ H

    def _filter_matches(self, matches: List) -> List:
        """使用 Lowe's ratio test 过滤匹配点"""
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        return good_matches

    def _compute_homography(
        self, matches: List, test_kp: List
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """计算单应性矩阵"""
        src_pts = np.float32([self.std_kp[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([test_kp[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        if len(src_pts) < 4:
            return None, None

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask

    def _warp_image(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """对图像进行透视变换"""
        h, w = self.std_img.shape[:2]
        return cv2.warpPerspective(img, H, (w, h))

    def _map_corners(self, img: np.ndarray, H: np.ndarray) -> List[Tuple[float, float]]:
        """将测试图的4个角点映射到标准图坐标系"""
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(
            -1, 1, 2
        )
        mapped_corners = cv2.perspectiveTransform(corners, H)
        return [tuple(pt[0]) for pt in mapped_corners]

    def _fail_result(self, error: str) -> Dict[str, Any]:
        """返回失败结果"""
        print(f"[ImageAligner] 对齐失败: {error}")
        return {
            "aligned_image": None,
            "homography": None,
            "corners_mapped": None,
            "match_count": 0,
            "confidence": 0,
            "success": False,
            "elapsed_ms": 0,
            "error": error,
        }


def align_image(
    test_image_path: str, station_id: int, timeout_ms: int = 500
) -> Dict[str, Any]:
    """便捷函数：快速执行图像对齐"""
    aligner = ImageAligner(station_id, timeout_ms)
    return aligner.align(test_image_path)
