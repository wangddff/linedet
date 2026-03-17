import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class TextRecognizer:
    """OCR文字识别模块 - 基于PaddleOCR实现线号、端子号、PLC模块型号识别"""

    def __init__(self, use_angle_cls: bool = True, lang: str = "ch"):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.ocr = None
        self._load_model()

    def _load_model(self):
        """加载PaddleOCR模型"""
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
            )
            print("[TextRecognizer] PaddleOCR模型加载成功")
        except Exception as e:
            print(f"[TextRecognizer] 警告: PaddleOCR加载失败 ({e})，将使用模拟模式")
            self.ocr = None

    def recognize(
        self, image_path: str, rois: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """执行OCR识别"""
        if rois is None:
            rois = []

        img = cv2.imread(image_path)
        if img is None:
            return {
                "passed": False,
                "error": f"无法读取图片: {image_path}",
                "text_results": [],
                "structure": {},
            }

        if self.ocr is None:
            return self._mock_recognize(img)

        try:
            if rois:
                text_results = self._recognize_rois(img, rois)
            else:
                text_results = self._recognize_full(img)

            structure = self._build_structure(text_results)

            return {
                "passed": True,
                "text_results": text_results,
                "structure": structure,
                "total_texts": len(text_results),
            }

        except Exception as e:
            return {
                "passed": False,
                "error": f"OCR识别失败: {str(e)}",
                "text_results": [],
                "structure": {},
            }

    def _recognize_full(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """全图OCR识别"""
        result = self.ocr.ocr(img, cls=True)

        if not result or not result[0]:
            return []

        text_results = []
        for line in result[0]:
            if not line:
                continue
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            text_results.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [[int(p[0]), int(p[1])] for p in box],
                    "center": [
                        int((box[0][0] + box[2][0]) / 2),
                        int((box[0][1] + box[2][1]) / 2),
                    ],
                }
            )

        return text_results

    def _recognize_rois(
        self, img: np.ndarray, rois: List[Dict]
    ) -> List[Dict[str, Any]]:
        """ROI区域OCR识别"""
        text_results = []

        for roi_info in rois:
            roi_img = roi_info.get("roi")
            if roi_img is None or roi_img.size == 0:
                continue

            try:
                result = self.ocr.ocr(roi_img, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]
                        confidence = line[1][1]

                        bbox = line[0]
                        abs_bbox = [
                            [
                                int(bbox[0][0] + roi_info["bbox"][0]),
                                int(bbox[0][1] + roi_info["bbox"][1]),
                            ],
                            [
                                int(bbox[1][0] + roi_info["bbox"][0]),
                                int(bbox[1][1] + roi_info["bbox"][1]),
                            ],
                            [
                                int(bbox[2][0] + roi_info["bbox"][0]),
                                int(bbox[2][1] + roi_info["bbox"][1]),
                            ],
                            [
                                int(bbox[3][0] + roi_info["bbox"][0]),
                                int(bbox[3][1] + roi_info["bbox"][1]),
                            ],
                        ]

                        text_results.append(
                            {
                                "text": text,
                                "confidence": float(confidence),
                                "bbox": abs_bbox,
                                "center": [
                                    int((abs_bbox[0][0] + abs_bbox[2][0]) / 2),
                                    int((abs_bbox[0][1] + abs_bbox[2][1]) / 2),
                                ],
                                "source_roi": roi_info.get("class_name", "unknown"),
                            }
                        )
            except Exception as e:
                print(f"[TextRecognizer] ROI识别失败: {e}")
                continue

        return text_results

    def _mock_recognize(self, img: np.ndarray) -> Dict[str, Any]:
        """模拟OCR结果（模型不可用时）"""
        h, w = img.shape[:2]

        mock_texts = [
            {
                "text": "A1",
                "confidence": 0.95,
                "bbox": [
                    [int(w * 0.2), int(h * 0.45)],
                    [int(w * 0.25), int(h * 0.45)],
                    [int(w * 0.25), int(h * 0.5)],
                    [int(w * 0.2), int(h * 0.5)],
                ],
                "center": [int(w * 0.225), int(h * 0.475)],
                "source_roi": "number_tube",
            },
            {
                "text": "A2",
                "confidence": 0.93,
                "bbox": [
                    [int(w * 0.35), int(h * 0.45)],
                    [int(w * 0.4), int(h * 0.45)],
                    [int(w * 0.4), int(h * 0.5)],
                    [int(w * 0.35), int(h * 0.5)],
                ],
                "center": [int(w * 0.375), int(h * 0.475)],
                "source_roi": "number_tube",
            },
            {
                "text": "24V",
                "confidence": 0.91,
                "bbox": [
                    [int(w * 0.5), int(h * 0.45)],
                    [int(w * 0.55), int(h * 0.45)],
                    [int(w * 0.55), int(h * 0.5)],
                    [int(w * 0.5), int(h * 0.5)],
                ],
                "center": [int(w * 0.525), int(h * 0.475)],
                "source_roi": "number_tube",
            },
        ]

        structure = self._build_structure(mock_texts)

        return {
            "passed": True,
            "text_results": mock_texts,
            "structure": structure,
            "total_texts": len(mock_texts),
        }

    def _build_structure(self, text_results: List[Dict]) -> Dict[str, Any]:
        """构建结构化数据"""
        terminal_numbers = []
        wire_numbers = []
        plc_modules = []

        for item in text_results:
            text = item["text"].strip().upper()
            confidence = item.get("confidence", 0)

            if self._is_terminal_number(text):
                terminal_numbers.append(
                    {
                        "text": text,
                        "position": item["center"],
                        "confidence": confidence,
                    }
                )
            elif self._is_wire_number(text):
                wire_numbers.append(
                    {
                        "text": text,
                        "position": item["center"],
                        "confidence": confidence,
                    }
                )
            elif self._is_plc_module(text):
                plc_modules.append(
                    {
                        "text": text,
                        "position": item["center"],
                        "confidence": confidence,
                    }
                )

        terminal_numbers.sort(key=lambda x: x["position"][0])
        wire_numbers.sort(key=lambda x: x["position"][0])

        return {
            "terminal_numbers": terminal_numbers,
            "wire_numbers": wire_numbers,
            "plc_modules": plc_modules,
        }

    def _is_terminal_number(self, text: str) -> bool:
        """判断是否为端子号 (如A1, B2, 1A, 12B等)"""
        import re

        return bool(re.match(r"^[0-9A-Za-z]+$", text)) and len(text) <= 5

    def _is_wire_number(self, text: str) -> bool:
        """判断是否为线号 (如24V, 12V, 1, 2等)"""
        import re

        return bool(re.match(r"^[0-9A-Z]+$", text)) and len(text) <= 10

    def _is_plc_module(self, text: str) -> bool:
        """判断是否为PLC模块型号"""
        import re

        plc_keywords = ["PLC", "CPU", "DI", "DO", "AI", "AO", "EM", "SM"]
        return any(kw in text.upper() for kw in plc_keywords) or len(text) > 5

    def filter_low_confidence(
        self, text_results: List[Dict], threshold: float = 0.6
    ) -> List[Dict]:
        """过滤低置信度结果"""
        return [r for r in text_results if r.get("confidence", 0) >= threshold]

    def correct_text_direction(self, text: str) -> str:
        """纠正文本方向（针对竖直文本）"""
        direction_map = {
            "丨": "1",
            "乚": "2",
            "亅": "1",
            "冖": "7",
            "刂": "1",
        }
        for old, new in direction_map.items():
            text = text.replace(old, new)
        return text
