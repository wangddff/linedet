import json
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class ROI:
    """ROI 数据类"""

    def __init__(self, label: str, points: List[List[float]], group_id: int = 1):
        self.label = label
        self.points = points
        self.group_id = group_id
        self._bbox = None
        self._center = None

    @property
    def bbox(self) -> List[int]:
        """获取边界框 [x, y, w, h]"""
        if self._bbox is None:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            self._bbox = [
                int(x_min),
                int(y_min),
                int(x_max - x_min),
                int(y_max - y_min),
            ]
        return self._bbox

    @property
    def center(self) -> List[int]:
        """获取中心点"""
        if self._center is None:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self._center = [int(sum(xs) / len(xs)), int(sum(ys) / len(ys))]
        return self._center

    def scale(self, factor: float) -> "ROI":
        """按比例缩放 ROI 坐标"""
        if factor == 1.0:
            return self

        scaled_points = [[p[0] * factor, p[1] * factor] for p in self.points]
        scaled_roi = ROI(self.label, scaled_points, self.group_id)
        if self._bbox:
            scaled_roi._bbox = [int(b * factor) for b in self._bbox]
        if self._center:
            scaled_roi._center = [int(c * factor) for c in self._center]
        return scaled_roi

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "label": self.label,
            "points": self.points,
            "bbox": self.bbox,
            "center": self.center,
            "group_id": self.group_id,
        }


class ROILoader:
    """从 LabelMe 标注文件加载 ROI

    标签说明:
    - detect_area: 检测区（全局大ROI，用于限定检测范围）
    - terminal_hole: 端子孔位
    - number_tube: 号码管
    - wire: 线材
    - connector: 插头
    - short_wire: 短接线
    - jumper: 短接片
    """

    def __init__(self, station_id: Optional[int] = None):
        self.station_id = station_id

    def load_from_labelme(self, json_path: str) -> Dict[str, Any]:
        """从 LabelMe JSON 文件加载 ROI

        返回:
            {
                "detect_area": ROI 或 None,
                "rois": List[ROI]  # 除 detect_area 外的其他 ROI
            }
        """
        json_file = Path(json_path)
        if not json_file.exists():
            print(f"[ROILoader] 标注文件不存在: {json_path}")
            return {"detect_area": None, "rois": []}

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ROILoader] 读取标注文件失败: {e}")
            return {"detect_area": None, "rois": []}

        detect_area = None
        rois = []

        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            points = shape.get("points", [])
            group_id = shape.get("group_id", 1)

            if not label or len(points) < 2:
                continue

            if label == "detect_area":
                detect_area = ROI(label, points, group_id)
            else:
                rois.append(ROI(label, points, group_id))

        print(f"[ROILoader] 从 {json_file.name} 加载了 {len(rois)} 个 ROI")
        if detect_area:
            print(f"[ROILoader] 检测区: bbox={detect_area.bbox}")

        return {"detect_area": detect_area, "rois": rois}

    def load_for_station(
        self, station_id: int, base_dir: str = "datasets/images"
    ) -> Dict[str, List[ROI]]:
        """加载指定工位的所有 ROI 标注"""
        station_dir = Path(base_dir) / f"station_{station_id}"

        if not station_dir.exists():
            station_dir = Path(base_dir)

        labelme_files = []
        for ext in ["*.json"]:
            labelme_files.extend(list(station_dir.glob(ext)))

        result = {}
        for json_file in labelme_files:
            rois = self.load_from_labelme(str(json_file))
            if rois:
                result[json_file.stem] = rois

        return result

    def get_rois_by_label(self, rois: List[ROI], label: str) -> List[ROI]:
        """按标签筛选 ROI"""
        return [r for r in rois if r.label == label]

    def get_roi_groups(self, rois: List[ROI]) -> Dict[int, List[ROI]]:
        """按 group_id 分组 ROI"""
        groups = {}
        for roi in rois:
            if roi.group_id not in groups:
                groups[roi.group_id] = []
            groups[roi.group_id].append(roi)

        for gid in groups:
            groups[gid].sort(key=lambda r: r.center[0])

        return groups
