# YOLO11 数据集标注指南

## 类别定义

| 类别ID | 类别名称 | 说明 |
|--------|----------|------|
| 0 | terminal_hole | 端子孔位 |
| 1 | number_tube | 号码管 |
| 2 | wire | 线材 |
| 3 | plc_module | PLC模块 |
| 4 | short_wire | 短接线 |
| 5 | jumper | 短接片 |
| 6 | connector | 插头 |

## 标注工具

推荐使用 **labelme** 进行标注：

```bash
pip install labelme
labelme
```

## 标注流程

1. 打开 labelme
2. 点击 "Open Dir" 打开图片目录：`linedect/datasets/images/train`
3. 使用矩形框工具标注目标
4. 选择对应的类别名称
5. 保存为 JSON 文件

## 转换为YOLO格式

运行转换脚本生成YOLO标注文件：

```bash
python convert_labelme_to_yolo.py
```

## 训练模型

```bash
cd linedect
python train_yolo11.py
```

## 数据集目录结构

```
linedect/
├── datasets/
│   ├── images/
│   │   ├── train/      # 训练图片
│   │   └── val/        # 验证图片
│   ├── labels/
│   │   ├── train/      # 训练标注 (YOLO格式)
│   │   └── val/        # 验证标注 (YOLO格式)
│   └── data.yaml       # 数据集配置
├── train_yolo11.py     # 训练脚本
└── convert_labelme_to_yolo.py  # 格式转换脚本
```

## YOLO标注格式

每个图片对应一个 `.txt` 文件，内容格式：
```
<class_id> <x_center> <y_center> <width> <height>
```
- 所有值都是归一化的 (0-1)
- x_center, y_center: 目标中心点
- width, height: 目标宽高