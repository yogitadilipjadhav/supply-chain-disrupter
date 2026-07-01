import importlib.util
import logging
from typing import Any, Dict

from src.utils.db_utils import fetch_time_series
from src.utils.yaml_utils import get_route_map
from src.agents.data_ingestion.agent import data_ingestion_agent
from src.agents.weather_agent.agent import weather_risk_monitoring_agent
from src.agents.news_agent.agent import news_event_analysis_agent
from src.agents.risk_classifier_agent import risk_classifier_agent
from src.agents.mitigation_agent import mitigation_recommendation_agent
from src.agents.state import ForecastResult, GlobalState, SimulationResult

logger = logging.getLogger(__name__)

# Optional heavy dependencies — agents that need these degrade gracefully when absent.
_PROPHET_AVAILABLE = importlib.util.find_spec("prophet") is not None
_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


# Bootstrap ingestion schema once per process (additive, never modifies lite_master)
try:
    from src.utils.ingestion_schema import ensure_ingestion_schema
    from src.agents.data_ingestion_agent import data_ingestion_agent_v2
    ensure_ingestion_schema()
    _INGESTION_V2_AVAILABLE = True
except Exception as _ingestion_bootstrap_exc:
    logger.warning("Ingestion schema bootstrap failed: %s", _ingestion_bootstrap_exc)
    _INGESTION_V2_AVAILABLE = False


def demand_forecasting_agent(state: GlobalState) -> Dict[str, Any]:
    """L5 — Prophet demand forecast (optional — skipped if prophet/pandas absent)."""
    if not _PROPHET_AVAILABLE or not _PANDAS_AVAILABLE:
        logger.warning("L5: prophet/pandas not installed — demand forecasting skipped.")
        return {
            "agent_logs": state.agent_logs + [
                "L5: SKIPPED — prophet or pandas not installed. Run: pip install prophet pandas"
            ],
        }

    if state.active_record is None:
        raise ValueError("Active record is required for demand forecasting.")

    ts = fetch_time_series(state.active_record["port"], state.active_record["sku"])
    if len(ts) < 10:
        return {
            "agent_logs": state.agent_logs + [
                f"L5: SKIPPED — only {len(ts)} history points available (need ≥ 10)."
            ],
        }

    import pandas as pd
    from prophet import Prophet

    df_records = [{"ds": row["event_date"], "y": row["demand"]} for row in ts]
    df = pd.DataFrame(df_records)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast_points = forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")
    demand_baseline = float(state.active_record.get("demand", 0.0))
    expected_drop = max(0.0, 1.0 - (forecast_points[-1]["yhat"] / (demand_baseline or 1.0)))
    return {
        "forecast_result": ForecastResult(
            prophet_forecast=forecast_points,
            expected_drop_pct=round(expected_drop * 100.0, 2),
        ),
        "agent_logs": state.agent_logs + ["L5: Demand forecasting completed."],
    }


def simulation_agent(state: GlobalState) -> Dict[str, Any]:
    """L6 — Monte Carlo stockout simulation (optional)."""
    if state.active_record is None or state.config is None:
        raise ValueError("Active record and config are required for simulation.")
    current_inventory = float(state.active_record.get("inventory_level", 0.0))
    incoming = float(state.active_record.get("incoming_supply", 0.0))
    lead_time = float(state.active_record.get("lead_time_days", 1.0))
    alt_route = get_route_map(state.config, state.active_record["port"]).get(
        "backup_route", "Cape of Good Hope"
    )
    stockout_probability = min(
        100.0,
        max(
            0.0,
            (state.risk_score_composite or 0.0) * 100.0
            + (1.0 - (current_inventory / (incoming + 1.0))) * 25.0
            + (lead_time / 30.0) * 25.0,
        ),
    )
    expected_gap = max(0.0, 100.0 - (current_inventory / (incoming + 1.0)) * 100.0)
    return {
        "simulation_result": SimulationResult(
            stockout_probability_pct=round(stockout_probability, 2),
            expected_inventory_gap_pct=round(expected_gap, 2),
            alternate_route=alt_route,
        ),
        "agent_logs": state.agent_logs + ["L6: Simulation completed."],
    }


def _run_optional(state: GlobalState, agent_fn, label: str) -> GlobalState:
    """Run an optional agent; on failure log SKIPPED and continue."""
    try:
        delta = agent_fn(state)
        return state.model_copy(update=delta)
    except Exception as exc:
        logger.warning("%s skipped: %s", label, exc)
        return state.model_copy(
            update={"agent_logs": state.agent_logs + [f"{label}: SKIPPED — {exc}"]}
        )


def run_agent_graph(payload: Dict[str, Any]) -> GlobalState:
    """
    Execute the full LangGraph agent pipeline.

    Critical path: L1 → L2 → L3 → L4
    Optional:      L5 (Prophet) → L6 (Simulation) → L7 (Mitigation)
    """
    state = GlobalState()

    # L1: try live-enriched v2 shim; fall back to legacy shim on any error
    if _INGESTION_V2_AVAILABLE:
        try:
            ingestion_delta = data_ingestion_agent_v2(state, payload)
        except Exception as _v2_exc:
            logger.warning("L1v2 failed, falling back to legacy: %s", _v2_exc)
            ingestion_delta = data_ingestion_agent(state, payload)
    else:
        ingestion_delta = data_ingestion_agent(state, payload)
    state = state.model_copy(update=ingestion_delta)

    news_delta = news_event_analysis_agent(state)
    state = state.model_copy(update=news_delta)

    weather_delta = weather_risk_monitoring_agent(state)
    state = state.model_copy(update=weather_delta)

    risk_delta = risk_classifier_agent(state)
    state = state.model_copy(update=risk_delta)

    # ── Optional agents — log and continue on failure ─────────────────────────
    state = _run_optional(state, demand_forecasting_agent, "L5")
    state = _run_optional(state, simulation_agent, "L6")
    state = _run_optional(state, mitigation_recommendation_agent, "L7")

    return state
