import yaml
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("config/india_electronics.yaml")


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def get_port_coordinates(config: Dict[str, Any], port_name: str) -> Dict[str, float]:
    ports = config.get("ports", {})
    if port_name not in ports:
        raise KeyError(f"Port not defined in config: {port_name}")
    return {
        "latitude": float(ports[port_name]["latitude"]),
        "longitude": float(ports[port_name]["longitude"]),
    }


def get_route_map(config: Dict[str, Any], port_name: str) -> Dict[str, Any]:
    return config.get("route_maps", {}).get(port_name, {})
