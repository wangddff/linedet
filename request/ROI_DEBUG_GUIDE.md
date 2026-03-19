# ROI 对比调试方法

本文档记录了排查 ROI 对比问题的调试方法。

## 问题描述

ROI 对比时出现以下问题：
1. 标注位置偏移
2. 相似度异常低（应该接近 100% 的区域实际很低）
3. 配对错误

## 调试步骤

### 1. 添加 ROI 对比详细日志

在 `src/core/comparator/roi_comparator.py` 的 `compare_roi_regions` 方法中添加：

```python
# 在对比循环中添加详细输出
for label, std_label_rois in std_rois_by_label.items():
    test_label_rois = test_rois_by_label.get(label, [])
    
    print(f"[ROI对比] 标签={label}, 标准图数量={len(std_label_rois)}, 待检图数量={len(test_label_rois)}")
    
    for i, std_roi in enumerate(std_label_rois):
        # ... 现有代码 ...
        
        std_center = std_roi.get("center", [0, 0])
        test_center = test_roi.get("center", [0, 0])
        std_bbox = std_roi.get("bbox")
        test_bbox = test_roi.get("bbox")
        std_shape = std_roi.get("roi_shape")
        test_shape = test_roi.get("roi_shape")
        
        print(
            f"  [{label} #{i}] 相似度={similarity:.3f}, "
            f"std_center={std_center}, test_center={test_center}, "
            f"std_bbox={std_bbox}, test_bbox={test_bbox}, "
            f"std_shape={std_shape}, test_shape={test_shape}"
        )
```

### 2. 保存 ROI 裁剪图片用于视觉检查

在 `src/core/comparator/roi_comparator.py` 的 `_compare_single_roi` 调用处添加：

```python
import cv2
import os

debug_dir = "data/debug_roi"
os.makedirs(debug_dir, exist_ok=True)
cv2.imwrite(f"{debug_dir}/std_{label}_{i}.png", std_crop)
cv2.imwrite(f"{debug_dir}/test_{label}_{i}.png", test_crop)
```

### 3. 检查 detect_area 和 scale_factor

在 `src/services/detection_service.py` 的 ROI 对比部分添加：

```python
print(f"[ROI对比] 标准图缩放后: {std_preprocessed_img.shape[1]}x{std_preprocessed_img.shape[0]}, std_scale={std_scale:.3f}")
print(f"[ROI对比] 待检图缩放后: {preprocessed_img.shape[1]}x{preprocessed_img.shape[0]}, scale_factor={scale_factor:.3f}")

if std_detect_area:
    print(f"[ROI对比] detect_area={detect_area.bbox}, std_detect_area={std_detect_area.bbox}, offset=({offset_x}, {offset_y})")
```

### 4. 检查标注偏移

在 `src/core/comparator/roi_comparator.py` 的 `mark_diff_areas` 方法中添加：

```python
print(f"[mark_diff_areas] bbox={bbox}, scale_factor={scale_factor}, offset=({offset_x}, {offset_y})")
```

## 常见问题及解决方案

### 问题 1: 标注位置偏移

**原因**: 坐标转换时 scale_factor 处理错误

**解决**: 
- `test_bbox` 是相对于 detect_area 子图的坐标（预处理后）
- 需要除以 scale_factor 转换到原始图片坐标
- 再加上 detect_area 在原图上的 offset

### 问题 2: 相似度异常低

**原因**: 
1. 标准图和待检图使用不同的 detect_area
2. scale_factor 应用顺序错误导致重复缩放

**解决**:
- 确保待检图和标准图使用同一个 JSON 文件的 detect_area
- scale_factor 应该只应用一次（在各自的预处理阶段）

### 问题 3: ROI 配对错误

**原因**: 使用调整后的坐标排序，不在同一参考系

**解决**: 使用 `original_center`（原始坐标）进行排序

```python
groups[label].sort(key=lambda r: r.get("original_center", r.get("center", [0, 0]))[0])
```

### 问题 4: 标准图 ROI 裁剪为空

**原因**: 标准图的 ROI 坐标在被调整到待检图的 detect_area 坐标系后超出范围

**解决**: 
- 方案1: 使用同一个 detect_area（待检图的 detect_area）
- 方案2: 调整标准图的 ROI 坐标到待检图的 detect_area 坐标系

## Debug 开关配置

建议在 `src/utils/config.py` 中添加 debug 配置：

```python
DEBUG_CONFIG = {
    "roi_compare": {
        "enabled": False,
        "save_roi_images": False,
        "verbose_logging": False,
    }
}
```

然后在代码中根据配置决定是否输出调试信息：
```python
from src.utils.config import DEBUG_CONFIG

if DEBUG_CONFIG["roi_compare"]["verbose_logging"]:
    print(f"[ROI对比] ...")
```