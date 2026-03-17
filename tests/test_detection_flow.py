import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, patch
import numpy as np
import cv2


class TestDetectionFlow(unittest.TestCase):
    """串联测试：验证完整检测流程"""

    @classmethod
    def setUpClass(cls):
        cls.test_image_path = "request/pic/工位1_01.png"
        if not os.path.exists(cls.test_image_path):
            print(f"[警告] 测试图片不存在: {cls.test_image_path}")
            cls.test_image_path = None

    def test_01_yolo_detector(self):
        """测试YOLOv11目标检测模块"""
        print("\n=== 测试 1: YOLODetector ===")

        from src.core.detector.yolo_detector import YOLODetector

        detector = YOLODetector()

        if self.test_image_path and os.path.exists(self.test_image_path):
            result = detector.detect(self.test_image_path)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector._mock_detect(img)

        print(f"检测结果: passed={result.get('passed')}")
        print(f"目标数量: {result.get('total_count', 0)}")
        print(f"ROI数量: {len(result.get('rois', []))}")

        self.assertIn("passed", result)
        self.assertIn("detections", result)
        print("[✓] YOLODetector 测试通过")

    def test_02_text_recognizer(self):
        """测试OCR文字识别模块"""
        print("\n=== 测试 2: TextRecognizer ===")

        from src.core.ocr.text_recognizer import TextRecognizer

        recognizer = TextRecognizer()

        if self.test_image_path and os.path.exists(self.test_image_path):
            result = recognizer.recognize(self.test_image_path)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = recognizer._mock_recognize(img)

        print(f"识别结果: passed={result.get('passed')}")
        print(f"文本数量: {result.get('total_texts', 0)}")

        structure = result.get("structure", {})
        print(f"端子号: {structure.get('terminal_numbers', [])}")
        print(f"线号: {structure.get('wire_numbers', [])}")

        self.assertIn("passed", result)
        self.assertIn("text_results", result)
        print("[✓] TextRecognizer 测试通过")

    def test_03_color_detector(self):
        """测试颜色检测模块"""
        print("\n=== 测试 3: ColorDetector ===")

        from src.core.color.color_detector import ColorDetector

        detector = ColorDetector()

        if self.test_image_path and os.path.exists(self.test_image_path):
            result = detector.detect(self.test_image_path)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = detector.detect(None, [{"bbox": [0, 0, 640, 480], "roi": img}])

        print(f"检测结果: passed={result.get('passed')}")
        print(f"线材数量: {result.get('total_wires', 0)}")

        colors = result.get("colors", [])
        if colors:
            print(f"检测到的颜色: {[c.get('color') for c in colors]}")

        self.assertIn("passed", result)
        self.assertIn("colors", result)
        print("[✓] ColorDetector 测试通过")

    def test_04_rule_validator(self):
        """测试规则校验模块"""
        print("\n=== 测试 4: RuleValidator ===")

        from src.core.validator.rule_validator import RuleValidator

        validator = RuleValidator(product_id=1, station_id=1, layer=1)

        ocr_result = {
            "structure": {
                "terminal_numbers": [
                    {"text": "A1", "position": [100, 200], "confidence": 0.95},
                    {"text": "A2", "position": [200, 200], "confidence": 0.93},
                ],
                "wire_numbers": [
                    {"text": "24V", "position": [100, 250], "confidence": 0.91},
                    {"text": "12V", "position": [200, 250], "confidence": 0.89},
                ],
            }
        }

        color_result = {
            "colors": [
                {"color": "红色", "confidence": 0.85},
                {"color": "蓝色", "confidence": 0.80},
            ]
        }

        result = validator.validate(ocr_result, color_result)

        print(f"校验结果: passed={result.get('passed')}")
        print(f"错误数量: {len(result.get('errors', []))}")

        details = result.get("details", {})
        print(f"规则数量: {details.get('total_rules', 0)}")

        self.assertIn("passed", result)
        self.assertIn("errors", result)
        print("[✓] RuleValidator 测试通过")

    def test_05_detection_service(self):
        """测试检测服务完整流程"""
        print("\n=== 测试 5: DetectionService 完整流程 ===")

        from src.services.detection_service import DetectionService

        service = DetectionService()

        mock_task = Mock()
        mock_task.id = 1
        mock_task.product_id = 1
        mock_task.station_id = 1
        mock_task.layer = 1

        if self.test_image_path and os.path.exists(self.test_image_path):
            mock_task.image_path = self.test_image_path
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            temp_path = "/tmp/test_image.png"
            cv2.imwrite(temp_path, img)
            mock_task.image_path = temp_path

        result = service.run_detection(mock_task)

        print(f"整体结果: {result.get('overall_result')}")
        print(f"相似度: {result.get('similarity_score', 0):.4f}")

        module_results = result.get("module_results", {})
        print(f"模块数量: {len(module_results)}")
        for module, mod_result in module_results.items():
            print(f"  - {module}: passed={mod_result.get('passed')}")

        errors = result.get("errors", [])
        if errors:
            print(f"错误列表:")
            for err in errors:
                print(f"  - [{err.get('module')}] {err.get('message')}")

        self.assertIn("overall_result", result)
        self.assertIn("module_results", result)
        print("[✓] DetectionService 测试通过")

    def test_06_standard_comparator(self):
        """测试标准图对比模块"""
        print("\n=== 测试 6: StandardComparator ===")

        from src.core.comparator.standard_comparator import StandardComparator

        comparator = StandardComparator(station_id=1, product_id=1)

        if self.test_image_path and os.path.exists(self.test_image_path):
            result = comparator.compare(self.test_image_path)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            result = comparator._compare_single(img, None)

        print(f"对比结果: passed={result.get('passed')}")
        print(f"相似度: {result.get('similarity_score', 0):.4f}")

        self.assertIn("passed", result)
        self.assertIn("similarity_score", result)
        print("[✓] StandardComparator 测试通过")


def run_tests():
    """运行所有测试"""
    print("=" * 50)
    print("接线视觉检测系统 - 串联测试")
    print("=" * 50)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDetectionFlow)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ 所有测试通过!")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 失败, {len(result.errors)} 错误")
    print("=" * 50)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
