"""Weather Risk agent (L3) — SQLite-first severity reader with live API fallback."""

import logging
import sys
from typing import Any, Dict, Optional

from src.agents.state import GlobalState

logger = logging.getLogger(__name__)


def weather_risk_monitoring_agent(state: GlobalState) -> Dict[str, Any]:
    _pkg = sys.modules["src.agents.weather_agent"]
    fetch_open_meteo = _pkg.fetch_open_meteo
    compute_weather_severity = _pkg.compute_weather_severity
    has_openai_api_key = _pkg.has_openai_api_key
    call_openai_structured = _pkg.call_openai_structured

    metadata = state.event_metadata
    config = state.config
    if metadata is None or config is None:
        raise ValueError("Event metadata and config are required for weather monitoring.")

    # Primary path: read severity written by L1 batch run (no external call)
    severity: Optional[float] = None
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
                severity = round(float(rows[0][0]) / 10.0, 4)
                logger.info(
                    "L3: severity=%.2f from live_weather_ingest (run_id=%s)",
                    severity, state.ingestion_run_id,
                )
        except Exception as exc:
            logger.warning("L3: live_weather_ingest read failed (%s) — falling back to live API.", exc)

    weather_risk_llm = None

    if severity is None:
        # Fallback: live Open-Meteo call (demo / manual scenario mode)
        from src.utils.yaml_utils import get_port_coordinates
        if state.active_record and state.active_record.get("latitude") is not None:
            coords = {
                "latitude": float(state.active_record["latitude"]),
                "longitude": float(state.active_record["longitude"]),
            }
        else:
            coords = get_port_coordinates(config, metadata.affected_port)
        payload = fetch_open_meteo(coords["latitude"], coords["longitude"])
        severity = compute_weather_severity(payload)

        if has_openai_api_key():
            try:
                from src.agents.state import WeatherRiskLLMOutput
                system_prompt = (
                    "You are a supply-chain weather risk analyst. "
                    "Given raw weather data for a semiconductor manufacturing hub, "
                    "assess the supply-chain risk and provide a geo_risk_component (0.0–1.0)."
                )
                user_prompt = (
                    f"Port: {metadata.affected_port}\n"
                    f"Disruption type: {metadata.disruption_type}\n"
                    f"Raw weather payload: {payload}\n"
                    f"Computed numeric severity: {severity}"
                )
                weather_risk_llm = call_openai_structured(system_prompt, user_prompt, WeatherRiskLLMOutput)
                if weather_risk_llm and weather_risk_llm.geo_risk_component is not None:
                    severity = round(float(weather_risk_llm.geo_risk_component), 4)
                    logger.info("L3: LLM override severity=%.2f", severity)
            except Exception as exc:
                logger.warning("L3: LLM enrichment failed: %s", exc)

    delta: Dict[str, Any] = {
        "live_weather_severity": severity,
        "agent_logs": state.agent_logs + ["L3: Weather risk assessment completed."],
    }
    if weather_risk_llm is not None:
        delta["weather_risk_llm"] = weather_risk_llm
    return delta
