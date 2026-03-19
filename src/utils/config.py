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


def is_debug_enabled(module: str = "roi_compare") -> bool:
    """检查指定模块的 debug 是否启用"""
    return DEBUG_CONFIG.get(module, {}).get("enabled", False)
