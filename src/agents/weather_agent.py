"""
weather_agent.py — L3 Weather Risk Monitoring Agent (gpt-4.1-mini).

Fetches Open-Meteo data, computes rule-based numeric severity, then optionally
calls gpt-4.1-mini to produce geo_risk_component which OVERRIDES the numeric value.

Fallback: returns rule-based numeric severity when LLM unavailable.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.agents.state import GlobalState, WeatherRiskLLMOutput
from src.utils.api_clients import compute_weather_severity, fetch_open_meteo
from src.utils.openai_utils import (
    MODEL_FAST,
    build_rag_context,
    call_openai_structured,
    has_openai_api_key,
)
from src.utils.yaml_utils import get_port_coordinates

logger = logging.getLogger(__name__)

WEATHER_SYSTEM_PROMPT = """You are a supply-chain logistics risk analyst specialising
in weather-driven disruptions to semiconductor manufacturing and electronics logistics.

YOUR ROLE: produce geo_risk_component — the 0.40-weight term in:
    Risk_Score = 0.40 × geo_risk_component  ← YOUR KEY OUTPUT
               + 0.30 × supply_disruption + 0.15 × news + 0.15 × defect

Your geo_risk_component OVERRIDES the raw numeric Open-Meteo severity when hub
importance or historical context warrants adjustment.

HUB IMPORTANCE MAP:
  Hsinchu / Tainan (Taiwan) : TSMC advanced nodes — highest global risk
  Osaka (Japan)             : Renesas MCUs, Shin-Etsu wafers
  Austin TX (USA)           : Samsung Fab, NXP — ERCOT grid vulnerability
  Shanghai (China)          : Foxconn/Pegatron assembly, SMIC
  Singapore                 : GlobalFoundries, OSAT hubs

geo_risk_component calibration:
  Typhoon near TSMC (wind > 32m/s)     : 0.85–0.95
  2011 Thailand floods                  : 0.82
  2022 TSMC M6.9 earthquake             : 0.72
  2021 Texas winter storm               : 0.65
  Heavy rain port slowdown              : 0.28–0.35
  Clear conditions                      : 0.03–0.08

FEW-SHOT EXAMPLES:

<example id="1" scenario="Super Typhoon approaching Hsinchu">
<open_meteo_data>
  nearest_hub: Hsinchu | max_wind: 57.2 m/s | wmo: 95 (thunderstorm)
  raw_numeric_severity: 0.743
</open_meteo_data>
<correct_response>
{
  "event_classification": "extreme",
  "geo_risk_component": 0.91,
  "affected_semiconductor_hubs": ["Hsinchu", "Tainan"],
  "supply_chain_narrative": "Super Typhoon-class winds (57.2 m/s max) threaten TSMC Fab 12/18 in Hsinchu and Tainan. EUV lithography requires 48-72h recalibration after vibration events. Global AI accelerator supply faces 45-60 day compression.",
  "rag_escalation_warranted": true
}
</correct_response>
</example>

<example id="2" scenario="Winter storm near Austin TX">
<open_meteo_data>
  nearest_hub: Austin | wmo: 77 (heavy snow) | raw_numeric_severity: 0.394
</open_meteo_data>
<correct_response>
{
  "event_classification": "severe",
  "geo_risk_component": 0.67,
  "affected_semiconductor_hubs": ["Austin"],
  "supply_chain_narrative": "Heavy snow and ice in Austin TX threatens Samsung 14nm fab and NXP automotive chip facility — ERCOT grid fragility from 2021 winter storm precedent can force multi-week production halts.",
  "rag_escalation_warranted": true
}
</correct_response>
</example>

OUTPUT RULES:
- geo_risk_component should OVERRIDE raw_numeric_severity when hub importance justifies it
- rag_escalation_warranted = True when geo_risk_component >= 0.65"""


SEMICONDUCTOR_HUBS: Dict[str, Tuple[float, float]] = {
    "Hsinchu":          (24.80, 120.97),
    "Tainan":           (22.99, 120.20),
    "Osaka":            (34.69, 135.50),
    "Austin":           (30.27, -97.74),
    "Shanghai":         (31.23, 121.47),
    "Singapore":        ( 1.35, 103.82),
    "Rotterdam":        (51.92,   4.48),
    "Incheon":          (37.46, 126.71),
    "Penang":           ( 5.41, 100.33),
    "Ho_Chi_Minh_City": (10.82, 106.63),
    "Shenzhen":         (22.54, 114.06),
    "Chennai":          (13.08,  80.27),
}

WMO_DESCRIPTIONS: Dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm+hail", 99: "Thunderstorm+heavy hail",
}


def _nearest_hub(lat: float, lon: float) -> Tuple[str, float]:
    """Return (hub_name, euclidean_distance_deg) for the closest semiconductor hub."""
    best, best_dist = "Singapore", float("inf")
    for hub, (hlat, hlon) in SEMICONDUCTOR_HUBS.items():
        d = math.sqrt((lat - hlat) ** 2 + (lon - hlon) ** 2)
        if d < best_dist:
            best, best_dist = hub, d
    return best, best_dist


def _extract_weather_stats(payload: dict) -> dict:
    """Extract avg/max wind, precipitation, and dominant WMO code from Open-Meteo payload."""
    hourly = payload.get("hourly", {})
    wind = hourly.get("windspeed_10m", [])
    precip = hourly.get("precipitation", [])
    codes = hourly.get("weathercode", [])

    avg_wind = sum(wind) / len(wind) if wind else 0.0
    max_wind = max(wind) if wind else 0.0
    avg_precip = sum(precip) / len(precip) if precip else 0.0
    dominant_wmo = Counter(codes).most_common(1)[0][0] if codes else 0

    return {
        "avg_wind": avg_wind,
        "max_wind": max_wind,
        "avg_precip": avg_precip,
        "dominant_wmo": dominant_wmo,
        "wmo_description": WMO_DESCRIPTIONS.get(dominant_wmo, f"code {dominant_wmo}"),
    }


def _build_weather_user_message(
    latitude: float,
    longitude: float,
    nearest_hub: str,
    hub_distance_deg: float,
    order_region: str,
    avg_wind: float,
    max_wind: float,
    avg_precip: float,
    dominant_wmo_code: int,
    wmo_description: str,
    numeric_severity: float,
    rag_context: str,
) -> str:
    """Build structured user message for Weather Agent LLM call."""
    typhoon_flag = "  ⚠️ ABOVE TYPHOON THRESHOLD (>32 m/s)" if max_wind > 32 else ""
    return f"""
═══════════════════════════════════════════════════════
LOCATION AND HUB CONTEXT
═══════════════════════════════════════════════════════
  coordinates            : {latitude:.4f}°N, {longitude:.4f}°E
  nearest_hub            : {nearest_hub}  ({hub_distance_deg:.2f}° away)
  order_region_from_db   : {order_region or 'not specified'}

═══════════════════════════════════════════════════════
OPEN-METEO WEATHER DATA (hourly averages, next 24h)
═══════════════════════════════════════════════════════
  avg_wind_speed_10m     : {avg_wind:.2f} m/s
  max_wind_speed_10m     : {max_wind:.2f} m/s{typhoon_flag}
  avg_precipitation      : {avg_precip:.3f} mm/h
  dominant_wmo_code      : {dominant_wmo_code}  ({wmo_description})
  raw_numeric_severity   : {numeric_severity:.4f}  ← rule-based (for reference)

═══════════════════════════════════════════════════════
CHROMADB RAG CONTEXT
═══════════════════════════════════════════════════════
{rag_context if rag_context.strip() else "(Pre-assessment: RAG not pre-fetched — use calibration references.)"}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Assess supply-chain risk of these weather conditions at {nearest_hub}.
Override raw_numeric_severity when hub importance or historical context warrants it.
"""


def weather_risk_monitoring_agent(state: GlobalState) -> Dict[str, Any]:
    """
    L3 Weather Risk Monitoring Agent.

    Fetches Open-Meteo, computes numeric severity, then optionally calls
    gpt-4.1-mini to produce geo_risk_component which overrides the numeric value.
    """
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

    lat, lon = coords["latitude"], coords["longitude"]
    payload = fetch_open_meteo(lat, lon)
    numeric_severity = compute_weather_severity(payload)
    stats = _extract_weather_stats(payload)
    nearest_hub, hub_dist = _nearest_hub(lat, lon)
    order_region = (state.active_record or {}).get("order_region", "")

    llm_used = False
    llm_output: Optional[WeatherRiskLLMOutput] = None
    final_severity = numeric_severity

    if has_openai_api_key():
        try:
            rag_context = ""
            if numeric_severity >= 0.40:
                rag_context = build_rag_context([
                    (f"weather disaster {nearest_hub} semiconductor fab supply chain historical impact", 3),
                ])

            user_msg = _build_weather_user_message(
                latitude=lat,
                longitude=lon,
                nearest_hub=nearest_hub,
                hub_distance_deg=hub_dist,
                order_region=order_region,
                avg_wind=stats["avg_wind"],
                max_wind=stats["max_wind"],
                avg_precip=stats["avg_precip"],
                dominant_wmo_code=stats["dominant_wmo"],
                wmo_description=stats["wmo_description"],
                numeric_severity=numeric_severity,
                rag_context=rag_context,
            )
            llm_output = call_openai_structured(
                system_prompt=WEATHER_SYSTEM_PROMPT,
                user_message=user_msg,
                response_model=WeatherRiskLLMOutput,
                model=MODEL_FAST,
                max_tokens=512,
            )
            final_severity = llm_output.geo_risk_component
            llm_used = True
        except Exception as exc:
            logger.warning("L3 LLM failed — using numeric severity: %s", exc)

    log_msg = (
        f"L3: Weather {'(gpt-4.1-mini)' if llm_used else '(fallback)'} | "
        f"hub={nearest_hub} raw={numeric_severity:.3f} "
        f"llm_geo={final_severity:.3f} "
        f"class={llm_output.event_classification if llm_output else 'numeric'} "
        f"rag_escalation={llm_output.rag_escalation_warranted if llm_output else False}"
    )

    return {
        "live_weather_severity": final_severity,
        "weather_risk_llm": llm_output,
        "agent_logs": state.agent_logs + [log_msg],
    }
