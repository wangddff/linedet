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

    def load_standard_with_preprocessing(self, preprocessed_test_img: np.ndarray):
        """使用与测试图相同的预处理方式加载标准图"""
        if not self.std_image_path:
            print(f"[ImageAligner] 标准图路径不存在")
            return False

        std_img_raw = cv2.imread(self.std_image_path)
        if std_img_raw is None:
            print(f"[ImageAligner] 无法读取标准图: {self.std_image_path}")
            return False

        target_h, target_w = preprocessed_test_img.shape[:2]
        self.std_img = cv2.resize(std_img_raw, (target_w, target_h))

        return self._extract_features()

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

    def _extract_features(self) -> bool:
        """提取标准图特征"""
        if self.std_img is None:
            return False

        try:
            try:
                self.sift = cv2.SIFT_create()
            except AttributeError:
                self.sift = cv2.xfeatures2d.SIFT_create()

            std_gray = self._preprocess_for_matching(self.std_img)
            self.std_kp, self.std_des = self.sift.detectAndCompute(std_gray, None)

            if self.std_kp is None or len(self.std_kp) < 10:
                print(
                    f"[ImageAligner] 标准图特征点不足: {len(self.std_kp) if self.std_kp else 0}"
                )
                return False

            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            print(f"[ImageAligner] 特征点数量: {len(self.std_kp)}")
            return True

        except Exception as e:
            print(f"[ImageAligner] 特征提取失败: {e}")
            return False

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

            std_gray = self._preprocess_for_matching(self.std_img)
            self.std_kp, self.std_des = self.sift.detectAndCompute(std_gray, None)

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

    def align_test_to_std(
        self, test_img: np.ndarray, std_img: np.ndarray
    ) -> Dict[str, Any]:
        """对齐测试图到标准图（输出标准图尺寸）

        Args:
            test_img: 测试图像
            std_img: 标准图像

        Returns:
            对齐结果
        """
        start_time = time.time()

        if test_img is None or test_img.size == 0:
            return self._fail_result("测试图像无效")
        if std_img is None or std_img.size == 0:
            return self._fail_result("标准图像无效")

        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            self.sift = cv2.xfeatures2d.SIFT_create()

        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        std_h, std_w = std_img.shape[:2]

        # 将测试图缩放到标准图尺寸，使两者在同一坐标空间中提取特征
        test_img_resized = cv2.resize(test_img, (std_w, std_h))

        std_gray = self._preprocess_for_matching(std_img)
        test_gray = self._preprocess_for_matching(test_img_resized)

        std_kp, std_des = self.sift.detectAndCompute(std_gray, None)
        test_kp, test_des = self.sift.detectAndCompute(test_gray, None)

        if std_kp is None or len(std_kp) < 10:
            return self._fail_result("标准图特征点不足")
        if test_kp is None or len(test_kp) < 10:
            return self._fail_result("测试图特征点不足")

        matches = self.matcher.knnMatch(std_des, test_des, k=2)
        good_matches = self._filter_matches(matches)
        if len(good_matches) < 10:
            return self._fail_result(f"匹配点不足: {len(good_matches)}")

        # 正确方向：src=测试图点，dst=标准图点 → H 映射 test→std
        src_pts = np.float32([test_kp[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([std_kp[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        if len(src_pts) < 4:
            return self._fail_result("匹配点不足，无法计算变换矩阵")

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return self._fail_result("无法计算单应性矩阵")

        inliers = mask.ravel().sum() if mask is not None else 0
        confidence = inliers / len(good_matches) if good_matches else 0

        if confidence < 0.3:
            return self._fail_result(f"匹配置信度低: {confidence:.2f}")

        # H 映射 test→std，warpPerspective 反向查找：dst(p') = src(H⁻¹·p') ✓
        aligned_img = cv2.warpPerspective(test_img_resized, H, (std_w, std_h))

        corners_mapped = self._map_corners(test_img_resized, H)

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

    def align_from_array_to_size(
        self, test_img: np.ndarray, std_img: np.ndarray
    ) -> Dict[str, Any]:
        """将测试图对齐到标准图尺寸（使用传入的标准图）

        Args:
            test_img: 测试图像数组
            std_img: 标准图像数组

        Returns:
            对齐结果字典
        """
        start_time = time.time()

        if test_img is None or test_img.size == 0:
            return self._fail_result("测试图像无效")

        if std_img is None or std_img.size == 0:
            return self._fail_result("标准图像无效")

        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            self.sift = cv2.xfeatures2d.SIFT_create()

        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        std_h, std_w = std_img.shape[:2]

        # 将测试图缩放到标准图尺寸，使两者在同一坐标空间中提取特征
        test_img_resized = (
            cv2.resize(test_img, (std_w, std_h))
            if test_img.shape[:2] != (std_h, std_w)
            else test_img
        )

        std_gray = self._preprocess_for_matching(std_img)
        test_gray = self._preprocess_for_matching(test_img_resized)

        std_kp, std_des = self.sift.detectAndCompute(std_gray, None)
        test_kp, test_des = self.sift.detectAndCompute(test_gray, None)

        if std_kp is None or len(std_kp) < 10:
            return self._fail_result("标准图特征点不足")

        if test_kp is None or len(test_kp) < 10:
            return self._fail_result("测试图特征点不足")

        matches = self.matcher.knnMatch(std_des, test_des, k=2)

        good_matches = self._filter_matches(matches)
        if len(good_matches) < 10:
            return self._fail_result(f"匹配点不足: {len(good_matches)}")

        # 正确方向：src=测试图点，dst=标准图点 → H 映射 test→std
        src_pts = np.float32([test_kp[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([std_kp[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        if len(src_pts) < 4:
            return self._fail_result("匹配点不足，无法计算变换矩阵")

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return self._fail_result("无法计算单应性矩阵")

        inliers = mask.ravel().sum() if mask is not None else 0
        confidence = inliers / len(good_matches) if good_matches else 0

        if confidence < 0.3:
            return self._fail_result(f"匹配置信度低: {confidence:.2f}")

        # H 映射 test→std，对缩放后的测试图做变换 → 输出标准图尺寸
        aligned_img = cv2.warpPerspective(test_img_resized, H, (std_w, std_h))

        corners_mapped = self._map_corners(test_img_resized, H)

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

    def align_from_array(self, test_img: np.ndarray) -> Dict[str, Any]:
        """直接使用 NumPy 数组进行对齐（推荐方式）

        Args:
            test_img: 已经预处理过的图像数组

        Returns:
            对齐结果字典
        """
        start_time = time.time()

        if self.std_img is None or self.sift is None:
            if not self.load_standard_with_preprocessing(test_img):
                return self._fail_result("标准图加载失败")

        if test_img is None or test_img.size == 0:
            return self._fail_result("测试图像无效")

        if self.std_img.shape[:2] != test_img.shape[:2]:
            self.std_img = cv2.resize(
                self.std_img, (test_img.shape[1], test_img.shape[0])
            )
            std_gray = self._preprocess_for_matching(self.std_img)
            self.std_kp, self.std_des = self.sift.detectAndCompute(std_gray, None)

        test_gray = self._preprocess_for_matching(test_img)
        test_kp, test_des = self.sift.detectAndCompute(test_gray, None)
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

    def align(self, test_image_path: str) -> Dict[str, Any]:
        """执行图像对齐（通过文件路径，已废弃，推荐使用 align_from_array）

        注意：此方法使用原始图像文件，可能与预处理后的图像尺寸不一致，
        导致对齐效果不佳。建议使用 align_from_array 代替。
        """
        start_time = time.time()

        if self.std_img is None or self.sift is None:
            return self._fail_result("标准图未加载")

        test_img = cv2.imread(test_image_path)
        if test_img is None:
            return self._fail_result(f"无法读取测试图: {test_image_path}")

        return self.align_from_array(test_img)

    def _resize_to_match(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """将img1缩放到与img2相同尺寸"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 == h2 and w1 == w2:
            return img1

        return cv2.resize(img1, (w2, h2))

    def _adjust_homography_for_size(
        self, H: np.ndarray, src_size: tuple, dst_size: tuple
    ) -> np.ndarray:
        """调整homography矩阵从缩放后图像坐标到原始图像坐标

        src_size: 缩放后图像的尺寸 (h, w)
        dst_size: 原始图像的尺寸 (h, w)
        """
        scale_x = dst_size[1] / src_size[1]
        scale_y = dst_size[0] / src_size[0]

        T = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

        return T @ H

    def _preprocess_for_matching(self, img: np.ndarray) -> np.ndarray:
        """转灰度并用 CLAHE 增强对比度，提升不同光照/曝光下的特征检测质量"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

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
        """计算单应性矩阵（H 映射 test→std）"""
        # 正确方向：src=测试图点，dst=标准图点 → H 映射 test→std
        src_pts = np.float32([test_kp[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([self.std_kp[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        if len(src_pts) < 4:
            return None, None

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H, mask

    def _warp_image(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """对图像进行透视变换（缩放到标准图尺寸）"""
        h, w = self.std_img.shape[:2]
        return cv2.warpPerspective(img, H, (w, h))

    def _warp_image_keep_size(self, img: np.ndarray, H: np.ndarray) -> np.ndarray:
        """对图像进行透视变换，保持原始尺寸"""
        h, w = img.shape[:2]
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
