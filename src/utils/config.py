import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_config(config_name: str = "settings.yaml") -> dict:
    config_path = CONFIG_DIR / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_station_config() -> dict:
    with open(CONFIG_DIR / "station_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
