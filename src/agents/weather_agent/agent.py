"""Weather Risk agent (L3) — SQLite-first classifier with Open-Meteo + LLM fallback.

Design principle (ref-spec v2): L1 is the ONLY external I/O boundary.
When L1 has already fetched and stored hub city weather in live_weather_ingest,
L3 reads severity from there (no live API call). This makes every pipeline run
fully replayable from SQLite alone.

Open-Meteo is called only when no live_weather_ingest rows exist for the current
ingestion_run_id — i.e. in demo / manual scenario mode where Run Now was not
executed before launching a scenario. In that fallback path, an optional LLM call
interprets the numeric severity for supply-chain context and overrides it when
geo_risk_component is available.
"""

import logging
import sys
from typing import Any, Dict, Optional

from src.utils.yaml_utils import get_port_coordinates
from src.agents.state import GlobalState, WeatherRiskLLMOutput

logger = logging.getLogger(__name__)


def _pkg():
    """Return the weather_agent package module (for patchable attribute lookups)."""
    return sys.modules.get("src.agents.weather_agent")


def weather_risk_monitoring_agent(state: GlobalState) -> Dict[str, Any]:
    metadata = state.event_metadata
    config = state.config
    if metadata is None or config is None:
        raise ValueError("Event metadata and config are required for weather monitoring.")

    severity: Optional[float] = None
    weather_risk_llm: Optional[WeatherRiskLLMOutput] = None

    # Primary path: read from live_weather_ingest populated by L1 batch run.
    if state.ingestion_run_id:
        try:
            from src.utils.db_utils import execute_query
            rows = execute_query(
                """
                SELECT raw_severity_score FROM live_weather_ingest
                WHERE run_id = ?
                ORDER BY raw_severity_score DESC LIMIT 1
                """,
                (state.ingestion_run_id,),
            )
            if rows:
                # ref-spec raw_severity is 0-10; convert to 0-1 for composite formula
                severity = round(float(rows[0][0]) / 10.0, 4)
                logger.info(
                    "L3: severity=%.4f from live_weather_ingest (run_id=%s)",
                    severity, state.ingestion_run_id,
                )
        except Exception as exc:
            logger.warning(
                "L3: live_weather_ingest read failed (%s) — falling back to live API.", exc
            )

    # Fallback: live Open-Meteo call (no prior batch run / demo mode).
    if severity is None:
        mod = _pkg()
        # Access through package module so unittest.mock.patch targets work.
        fetch_fn = getattr(mod, "fetch_open_meteo", None)
        severity_fn = getattr(mod, "compute_weather_severity", None)
        if fetch_fn is None or severity_fn is None:
            from src.agents.weather_agent.client import fetch_open_meteo, compute_weather_severity
            fetch_fn, severity_fn = fetch_open_meteo, compute_weather_severity

        if state.active_record and state.active_record.get("latitude") is not None:
            coords = {
                "latitude": float(state.active_record["latitude"]),
                "longitude": float(state.active_record["longitude"]),
            }
        else:
            coords = get_port_coordinates(config, metadata.affected_port)

        payload = fetch_fn(coords["latitude"], coords["longitude"])
        severity = severity_fn(payload)
        logger.info("L3: severity=%.4f from live Open-Meteo call (fallback).", severity)

        # Optional LLM enhancement: supply-chain interpretation of numeric weather.
        has_key_fn = getattr(mod, "has_openai_api_key", None)
        call_fn = getattr(mod, "call_openai_structured", None)
        if has_key_fn and call_fn and has_key_fn():
            try:
                from src.utils.openai_utils import MODEL_FAST
                system_prompt = (
                    "You are a semiconductor supply-chain risk analyst. "
                    "Classify the weather event's severity for chip manufacturing and logistics disruption. "
                    "Return a geo_risk_component between 0 and 1 representing the disruption risk."
                )
                user_message = (
                    f"Port/Hub: {metadata.affected_port}\n"
                    f"Disruption type: {metadata.disruption_type}\n"
                    f"Raw weather severity (0-1): {severity}\n"
                    f"Weather payload: {payload}"
                )
                weather_risk_llm = call_fn(
                    system_prompt,
                    user_message,
                    WeatherRiskLLMOutput,
                    model=MODEL_FAST,
                )
                severity = weather_risk_llm.geo_risk_component
                logger.info(
                    "L3: LLM override → severity=%.4f, classification=%s",
                    severity, weather_risk_llm.event_classification,
                )
            except Exception as exc:
                logger.warning("L3: LLM enhancement failed (%s) — keeping numeric severity.", exc)

    return {
        "live_weather_severity": severity,
        "weather_risk_llm": weather_risk_llm,
        "agent_logs": state.agent_logs + ["L3: Weather risk assessment completed."],
    }
