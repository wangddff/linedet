import yaml
from pathlib import Path
import os

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"

DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

DEBUG_CONFIG = {
    "roi_compare": {
        "enabled": DEBUG_MODE,
        "save_roi_images": DEBUG_MODE,
        "verbose_logging": DEBUG_MODE,
    }
}


def load_config(config_name: str = "settings.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_station_config() -> dict:
    with open(CONFIG_DIR / "station_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_detection_config() -> dict:
    """返回 settings.yaml 中 detection 节的配置，附带默认值兜底"""
    try:
        cfg = load_config("settings.yaml")
        return cfg.get("detection", {})
    except Exception:
        return {}


def get_roi_similarity_threshold() -> float:
    """ROI 相似度阈值：低于此值的区域标红报错"""
    return float(get_detection_config().get("roi_similarity_threshold", 0.65))


def get_color_hsv_thresholds() -> tuple[float, float]:
    """颜色 HSV 距离阈值 (彩色, 非彩色)"""
    cfg = get_detection_config()
    return (
        float(cfg.get("color_hsv_threshold_chroma", 0.25)),
        float(cfg.get("color_hsv_threshold_achroma", 0.18)),
    )


def is_debug_enabled(module: str = "roi_compare") -> bool:
    """检查指定模块的 debug 是否启用"""
    return DEBUG_CONFIG.get(module, {}).get("enabled", False)
