"""Weather Risk agent (L3) — fetches live weather and scores severity."""

from typing import Any, Dict

from src.utils.yaml_utils import get_port_coordinates
from src.agents.state import GlobalState
from src.agents.weather_agent.client import compute_weather_severity, fetch_open_meteo


def weather_risk_monitoring_agent(state: GlobalState) -> Dict[str, Any]:
    metadata = state.event_metadata
    config = state.config
    if metadata is None or config is None:
        raise ValueError("Event metadata and config are required for weather monitoring.")
    if state.active_record and state.active_record.get("latitude") is not None:
        coords = {
            "latitude": float(state.active_record["latitude"]),
            "longitude": float(state.active_record["longitude"]),
        }
    else:
        coords = get_port_coordinates(config, metadata.affected_port)
    payload = fetch_open_meteo(coords["latitude"], coords["longitude"])
    severity = compute_weather_severity(payload)
    return {
        "live_weather_severity": severity,
        "agent_logs": state.agent_logs + ["L3: Weather risk assessment completed."],
    }
