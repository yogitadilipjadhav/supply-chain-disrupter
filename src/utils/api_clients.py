import requests
from typing import Any, Dict

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo(latitude: float, longitude: float, hourly: str = "windspeed_10m,precipitation,weathercode") -> Dict[str, Any]:
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": hourly,
                "timezone": "UTC",
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo API request failed: {exc}") from exc


def compute_weather_severity(weather_payload: Dict[str, Any]) -> float:
    try:
        hourly = weather_payload.get("hourly", {})
        wind = hourly.get("windspeed_10m", [])
        precip = hourly.get("precipitation", [])
        if not wind or not precip:
            return 0.0
        avg_wind = sum(wind) / len(wind)
        avg_precip = sum(precip) / len(precip)
        severity = min(1.0, (avg_wind / 30.0) + (avg_precip / 50.0))
        return round(severity, 3)
    except Exception as exc:
        raise RuntimeError(f"Failed to compute weather severity: {exc}") from exc
