import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class ImageAnnotator:
    """图像标注工具 - 在检测结果图片上标注问题区域"""

    COLORS = {
        "error": (0, 0, 255),
        "warning": (0, 165, 255),
        "success": (0, 255, 0),
        "info": (255, 255, 0),
    }

    PIL_COLORS = {
        "error": (255, 0, 0),
        "warning": (0, 165, 255),
        "success": (0, 255, 0),
        "info": (255, 255, 0),
    }

    def __init__(self):
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        """加载中文字体"""
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, 24)
                except Exception:
                    continue

        return ImageFont.load_default()

    def _draw_text(
        self,
        img: np.ndarray,
        text: str,
        position: tuple,
        color_name: str = "error",
        font_size: int = 24,
    ) -> np.ndarray:
        """使用 PIL 绘制中文"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        color = self.PIL_COLORS.get(color_name, (255, 255, 255))

        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/PingFang.ttc",
                font_size,
            )
        except Exception:
            font = self.font

        draw.text(position, text, font=font, fill=color)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_rectangle(
        self,
        img: np.ndarray,
        pt1: tuple,
        pt2: tuple,
        color_name: str = "info",
        thickness: int = 2,
    ) -> np.ndarray:
        """绘制矩形框"""
        color = self.COLORS.get(color_name, (255, 255, 255))
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return img

    def annotate_image(
        self,
        image_path: str,
        detect_result: Dict,
        ocr_result: Dict,
        color_result: Dict,
        output_path: Optional[str] = None,
    ) -> str:
        """在图片上标注检测结果和问题区域"""
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        h, w = img.shape[:2]

        yolo_errors = (
            detect_result.get("compare_result", {}).get("details", {}).get("errors", [])
        )
        color_errors = color_result.get("compare_result", {}).get("errors", [])

        error_y = 30
        for error in yolo_errors:
            cls_name = error.get("class", "unknown")
            expected = error.get("expected", 0)
            actual = error.get("actual", 0)
            msg = f"缺{cls_name}: 需{expected}实{actual}"

            img = self._draw_text(img, msg, (10, error_y), "error", 24)
            error_y += 30

        for error in color_errors:
            error_type = error.get("type", "")
            if error_type == "color_mismatch":
                wire_index = error.get("wire_index", 0)
                expected = error.get("expected", "")
                actual = error.get("actual", "")
                msg = f"线{wire_index + 1}颜色错: {expected}!={actual}"

                img = self._draw_text(img, msg, (10, error_y), "error", 22)
                error_y += 28
            elif error_type == "missing_wire":
                msg = error.get("message", "线材数量不足")
                img = self._draw_text(img, msg, (10, error_y), "error", 24)
                error_y += 30

        ocr_structure = ocr_result.get("structure", {})
        if not ocr_structure.get("terminal_numbers") and not ocr_structure.get(
            "wire_numbers"
        ):
            img = self._draw_text(
                img, "未识别到文字", (10, min(h - 30, error_y + 30)), "warning", 24
            )

        # detections = detect_result.get("detections", [])
        # for det in detections:
        #     bbox = det.get("bbox", [])
        #     if len(bbox) >= 4:
        #         x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        #         cls_name = det.get("class_name", "")
        #         img = self._draw_rectangle(img, (x, y), (x + bw, y + bh), "info", 2)
        #         img = self._draw_text(img, cls_name, (x, y - 5), "info", 16)

        # color_list = color_result.get("colors", [])
        # for i, color_info in enumerate(color_list):
        #     bbox = color_info.get("bbox", [])
        #     if bbox and len(bbox) >= 4:
        #         x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        #         color_name = color_info.get("color", "未知")
        #         img = self._draw_rectangle(img, (x, y), (x + bw, y + bh), "success", 2)
        #         img = self._draw_text(img, color_name, (x, y + bh + 5), "success", 16)

        if output_path is None:
            input_name = Path(image_path).stem
            output_path = f"data/exports/{input_name}_annotated.png"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)

        print(f"[ImageAnnotator] 标注图片已保存: {output_path}")
        return output_path
