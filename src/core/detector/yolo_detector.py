import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path


class YOLODetector:
    """YOLOv11目标检测模块 - 端子孔位、号码管、线材、PLC模块、短接片检测"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self.model_name = "yolo11n"  # YOLOv11 nano (轻量版)
        self.class_names = [
            "terminal_hole",  # 端子孔位
            "number_tube",  # 号码管
            "wire",  # 线材
            "plc_module",  # PLC模块
            "short_wire",  # 短接线
            "jumper",  # 短接片
            "connector",  # 插头
        ]
        self._load_model()

    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        base_dir = Path(__file__).parent.parent.parent.parent / "models" / "yolo"
        default_model = base_dir / "best.pt"
        if default_model.exists():
            return str(default_model)
        base_dir = (
            Path(__file__).parent.parent.parent
            / "runs"
            / "detect"
            / "runs"
            / "detect"
            / "train"
            / "weights"
        )
        default_model = base_dir / "best.pt"
        return str(default_model) if default_model.exists() else ""

    def _load_model(self):
        """加载YOLOv11模型"""
        if self.model_path and Path(self.model_path).exists():
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.model_path)
                print(f"[YOLODetector] 自定义模型加载成功: {self.model_path}")
                return
            except Exception as e:
                print(f"[YOLODetector] 自定义模型加载失败: {e}")

        if not self.model_path:
            try:
                from ultralytics import YOLO

                self.model = YOLO(
                    self.model_path if self.model_path else f"{self.model_name}.pt"
                )
                print(f"[YOLODetector] YOLOv11预训练模型加载成功: {self.model_name}.pt")
            except Exception as e:
                print(f"[YOLODetector] 警告: 模型加载失败 ({e})，将使用模拟模式")
                self.model = None
        else:
            self.model = None

    def detect(self, image_path: str) -> Dict[str, Any]:
        """执行目标检测"""
        img = cv2.imread(image_path)
        if img is None:
            return {
                "passed": False,
                "error": f"无法读取图片: {image_path}",
                "detections": [],
                "rois": [],
                "wire_rois": [],
            }

        if self.model is None:
            return self._mock_detect(img)

        try:
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections = self._parse_results(results, img.shape)

            rois = self._extract_rois(detections, img, ["number_tube", "terminal_hole"])
            wire_rois = self._extract_rois(detections, img, ["wire", "short_wire"])

            return {
                "passed": True,
                "detections": detections,
                "rois": rois,
                "wire_rois": wire_rois,
                "total_count": len(detections),
            }

        except Exception as e:
            return {
                "passed": False,
                "error": f"检测失败: {str(e)}",
                "detections": [],
                "rois": [],
                "wire_rois": [],
            }

    def _mock_detect(self, img: np.ndarray) -> Dict[str, Any]:
        """模拟检测结果（模型不可用时）"""
        h, w = img.shape[:2]

        mock_detections = [
            {
                "class_id": 0,
                "class_name": "terminal_hole",
                "bbox": [int(w * 0.2), int(h * 0.3), int(w * 0.1), int(h * 0.08)],
                "confidence": 0.9,
            },
            {
                "class_id": 0,
                "class_name": "terminal_hole",
                "bbox": [int(w * 0.35), int(h * 0.3), int(w * 0.1), int(h * 0.08)],
                "confidence": 0.9,
            },
            {
                "class_id": 1,
                "class_name": "number_tube",
                "bbox": [int(w * 0.2), int(h * 0.45), int(w * 0.08), int(h * 0.05)],
                "confidence": 0.85,
            },
            {
                "class_id": 2,
                "class_name": "wire",
                "bbox": [int(w * 0.15), int(h * 0.5), int(w * 0.7), int(h * 0.1)],
                "confidence": 0.88,
            },
        ]

        rois = self._extract_rois(
            mock_detections, img, ["number_tube", "terminal_hole"]
        )
        wire_rois = self._extract_rois(mock_detections, img, ["wire", "short_wire"])

        return {
            "passed": True,
            "detections": mock_detections,
            "rois": rois,
            "wire_rois": wire_rois,
            "total_count": len(mock_detections),
        }

    def _parse_results(self, results, img_shape) -> List[Dict[str, Any]]:
        """解析YOLO检测结果"""
        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            if cls_id >= len(self.class_names):
                continue

            x1, y1, x2, y2 = box
            detections.append(
                {
                    "class_id": int(cls_id),
                    "class_name": self.class_names[int(cls_id)],
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(conf),
                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                }
            )

        return detections

    def _extract_rois(
        self, detections: List[Dict], img: np.ndarray, target_classes: List[str]
    ) -> List[Dict[str, Any]]:
        """提取指定类别的ROI区域"""
        rois = []
        h, w = img.shape[:2]

        for det in detections:
            if det["class_name"] not in target_classes:
                continue

            bbox = det["bbox"]
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(w, bbox[0] + bbox[2])
            y2 = min(h, bbox[1] + bbox[3])

            if x2 > x1 and y2 > y1:
                roi_img = img[y1:y2, x1:x2]
                rois.append(
                    {
                        "class_name": det["class_name"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "roi": roi_img,
                        "confidence": det["confidence"],
                    }
                )

        return rois

    def get_layer_info(
        self, detections: List[Dict], terminal_threshold: float = 50
    ) -> Dict[str, Any]:
        """根据端子位置信息判断层级"""
        terminal_holes = [d for d in detections if d["class_name"] == "terminal_hole"]

        if not terminal_holes:
            return {"layers": 1, "layer_bounds": []}

        centers_y = [d["center"][1] for d in terminal_holes]
        centers_y.sort()

        layers = []
        current_layer = [centers_y[0]]
        layer_threshold = terminal_threshold

        for y in centers_y[1:]:
            if y - current_layer[-1] < layer_threshold:
                current_layer.append(y)
            else:
                layers.append(current_layer)
                current_layer = [y]

        if current_layer:
            layers.append(current_layer)

        layer_bounds = []
        for layer in layers:
            layer_bounds.append(
                {
                    "min_y": min(layer),
                    "max_y": max(layer),
                    "count": len(layer),
                }
            )

        return {
            "layers": len(layers),
            "layer_bounds": layer_bounds,
        }
