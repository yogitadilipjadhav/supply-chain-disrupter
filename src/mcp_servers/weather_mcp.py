"""
MCP server — weather tool for the L1 Data Ingestion Agent.

Exposes a single tool: fetch_hub_weather(city, lat, lon)
The L1 agent calls this via MCP instead of calling Open-Meteo directly,
demonstrating the Model Context Protocol tool-calling pattern.

Run standalone:
    python -m src.mcp_servers.weather_mcp

Or import and use the client helper directly in tests:
    from src.mcp_servers.weather_mcp import fetch_hub_weather_via_mcp
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ── Severity computation (mirrors ref-spec formula) ──────────────────────────

def _severity_0_to_10(wind_kmh: float, precip_mm: float, weather_code: Optional[int]) -> float:
    score = 0.0
    if weather_code is not None and weather_code >= 95:
        score += 4
    if wind_kmh >= 60:
        score += 2
    elif wind_kmh >= 30:
        score += 1
    if precip_mm > 50:
        score += 2
    elif precip_mm > 10:
        score += 1
    return min(10.0, score)


# ── MCP tool implementation ───────────────────────────────────────────────────

def fetch_hub_weather(city: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    MCP tool: fetch current weather for a semiconductor hub city.

    Parameters
    ----------
    city : str
        Human-readable hub city name (e.g. "Hsinchu", "Rotterdam").
    lat : float
        Latitude of the hub city.
    lon : float
        Longitude of the hub city.

    Returns
    -------
    dict with keys: city, latitude, longitude, wind_speed_kmh, precipitation_mm,
                    weather_code, temperature_c, raw_severity_score
    """
    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
        "timezone": "UTC",
        "forecast_days": "1",
    }
    resp = requests.get(api_url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    codes = daily.get("weathercode", [None])
    code = codes[0] if codes else None
    wind = (daily.get("windspeed_10m_max") or [0.0])[0] or 0.0
    precip = (daily.get("precipitation_sum") or [0.0])[0] or 0.0
    temp_max = (daily.get("temperature_2m_max") or [None])[0]
    temp_min = (daily.get("temperature_2m_min") or [None])[0]
    temp_avg = (temp_max + temp_min) / 2 if (temp_max is not None and temp_min is not None) else None

    return {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "wind_speed_kmh": round(wind, 2),
        "precipitation_mm": round(precip, 2),
        "weather_code": code,
        "temperature_c": round(temp_avg, 2) if temp_avg is not None else None,
        "raw_severity_score": _severity_0_to_10(wind, precip, code),
    }


# ── MCP server definition (requires `mcp` package) ───────────────────────────

def build_mcp_server():
    """Build and return the FastMCP server object. Lazy import so the module
    can be imported even when `mcp` is not installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "The `mcp` package is required to run the MCP server. "
            "Install it with: pip install mcp"
        )

    mcp = FastMCP("supply-chain-weather")

    @mcp.tool()
    def get_hub_weather(city: str, lat: float, lon: float) -> str:
        """
        Fetch current weather conditions for a semiconductor hub city via Open-Meteo.
        Returns JSON with wind, precipitation, weather code, temperature,
        and a 0-10 raw_severity_score.
        """
        result = fetch_hub_weather(city, lat, lon)
        return json.dumps(result)

    return mcp


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        server = build_mcp_server()
        logger.info("Starting supply-chain-weather MCP server (stdio transport)...")
        server.run(transport="stdio")
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)
