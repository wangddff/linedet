# 接线视觉检测系统

工业PLC柜/端子台接线视觉检测系统，基于Python技术栈实现。

## 功能特性

- **目标检测**: YOLOv11 端子孔位、号码管、线材、PLC模块检测
- **OCR识别**: PaddleOCR 线号、端子号、PLC模块型号识别
- **颜色检测**: HSV色彩空间线材颜色检测
- **规则校验**: 错接、漏接、多接、短接线、短接片校验
- **标准图对比**: 图像相似度比对
- **层级管控**: 多层端子逐层检测流程控制

## 技术栈

- **后端**: FastAPI + SQLAlchemy
- **AI模型**: YOLOv11, PaddleOCR
- **图像处理**: OpenCV, NumPy
- **数据库**: PostgreSQL

## 快速开始

### 1. 创建虚拟环境

```bash
cd linedect
uv venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 3. 配置数据库

编辑 `config/database.yaml` 配置数据库连接信息。

### 4. 启动服务

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 访问API

- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/health

## 项目结构

```
linedect/
├── src/
│   ├── main.py                 # FastAPI应用入口
│   ├── api/routes/             # API路由
│   │   ├── tasks.py            # 任务管理
│   │   ├── products.py         # 产品管理
│   │   ├── standards.py        # 标准图管理
│   │   └── wiring.py           # 接线规则管理
│   ├── core/
│   │   ├── comparator/         # 标准图对比
│   │   ├── detector/           # YOLO目标检测
│   │   ├── ocr/                # OCR文字识别
│   │   ├── color/              # 颜色检测
│   │   └── validator/          # 规则校验
│   ├── preprocessing/          # 图像预处理
│   ├── services/               # 检测服务
│   ├── database/               # 数据库模型
│   └── utils/                  # 工具函数
├── config/                     # 配置文件
├── web/                        # 前端页面
├── models/                     # 模型文件
├── data/                       # 数据目录
└── tests/                      # 测试用例
```

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/tasks` | GET | 获取任务列表 |
| `/api/tasks` | POST | 创建任务 |
| `/api/tasks/{id}/upload` | POST | 上传图片 |
| `/api/tasks/{id}/detect` | POST | 执行检测 |
| `/api/tasks/{id}/result` | GET | 获取检测结果 |
| `/api/products` | GET/POST | 产品管理 |
| `/api/standards/{station_id}` | GET/POST | 标准图管理 |
| `/api/wiring/{product_id}` | GET/POST | 接线规则管理 |
| `/api/stations` | GET | 获取工位列表 |

## 开发

### 运行测试

```bash
python tests/test_detection_flow.py
```

### Docker部署

```bash
docker build -t linedet .
docker run -p 8000:8000 linedet
```

## 许可证

MIT License