import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils.api_clients import compute_weather_severity, fetch_open_meteo
from src.utils.db_utils import (
    fetch_daily_record,
    fetch_inventory_snapshot,
    fetch_time_series,
    insert_mitigation_action,
    update_risk_label,
)
from src.utils.yaml_utils import get_port_coordinates, get_route_map, load_config
from src.agents.rag_agent import build_news_signals
from src.agents.state import (
    EventMetadata,
    ForecastResult,
    GlobalState,
    MitigationAction,
    NewsRiskSignal,
    SimulationResult,
)


class NewsAnalysisSchema(BaseModel):
    source_id: str = Field(..., description="Unique identifier for the news or report chunk")
    category: str = Field(..., description="Risk category extracted from the document")
    severity: float = Field(..., ge=0.0, le=1.0)
    summary: str = Field(..., description="Short summary of the relevant risk signal")
    signal_tags: List[str] = Field(..., description="Key topic tags extracted from the text")


class RiskClassificationSchema(BaseModel):
    composite_score: float = Field(..., ge=0.0, le=1.0)
    risk_label: str = Field(..., description="LOW, HIGH, or CRITICAL")
    rationale: str = Field(...)


class MitigationSchema(BaseModel):
    summary: str = Field(...)
    recommendations: List[str] = Field(..., min_items=3, max_items=3)
    cost_delta: str = Field(...)


def load_vector_signals(event_category: str) -> List[Dict[str, Any]]:
    # Placeholder for actual ChromaDB query path.
    return [
        {
            "source_id": "mock-001",
            "category": "weather",
            "severity": 0.8,
            "summary": "Historic monsoon port closures caused major delays.",
            "signal_tags": ["monsoon", "port closure", "inventory risk"],
        }
    ]


def data_ingestion_agent(state: GlobalState, payload: Dict[str, Any]) -> Dict[str, Any]:
    event_metadata = EventMetadata(**payload)
    state_updates: Dict[str, Any] = {
        "event_metadata": event_metadata,
        "config": load_config(),
        "agent_logs": state.agent_logs + ["L1: Data ingestion completed."],
    }
    record = fetch_daily_record(
        payload.get("event_date", ""),
        event_metadata.affected_port,
        payload.get("sku", "CHIP_AP"),
    )
    if record:
        state_updates["active_record"] = record
    return state_updates


def news_event_analysis_agent(state: GlobalState) -> Dict[str, Any]:
    metadata = state.event_metadata
    if metadata is None or state.config is None:
        raise ValueError("Event metadata and config are required for news analysis.")
    parsed_signals = build_news_signals(metadata.disruption_type)
    if not parsed_signals:
        parsed_signals = [
            NewsRiskSignal(
                source_id="fallback-001",
                category=metadata.disruption_type,
                severity=0.3,
                summary="Fallback risk signal for missing RAG data.",
                signal_tags=[metadata.disruption_type, "fallback"],
            )
        ]
    return {
        "news_signals": parsed_signals,
        "agent_logs": state.agent_logs + ["L2: News and event analysis completed."],
    }


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


def risk_classifier_agent(state: GlobalState) -> Dict[str, Any]:
    if state.event_metadata is None or state.active_record is None:
        raise ValueError("Data ingestion and record load are required for risk classification.")
    risk_inputs = {
        "disruption_type": state.event_metadata.disruption_type,
        "severity": state.event_metadata.severity,
        "weather_severity": state.live_weather_severity or 0.0,
        "news_signals": [signal.dict() for signal in state.news_signals],
        "chip_risk": float(state.active_record.get("chip_risk", 0.0)),
        "supplier_risk": float(state.active_record.get("supplier_risk", 0.0)),
    }

    composite = min(1.0, (risk_inputs["severity"] * 0.4) + (risk_inputs["weather_severity"] * 0.2) + (risk_inputs["chip_risk"] * 0.2) + (risk_inputs["supplier_risk"] * 0.2))
    label = "LOW"
    if composite >= 0.75:
        label = "CRITICAL"
    elif composite >= 0.4:
        label = "HIGH"
    update_risk_label(
        state.active_record["event_date"],
        state.active_record["port"],
        state.active_record["sku"],
        composite,
        label,
    )
    return {
        "risk_score_composite": round(composite, 3),
        "risk_label": label,
        "agent_logs": state.agent_logs + ["L4: Risk classification completed."],
    }


def demand_forecasting_agent(state: GlobalState) -> Dict[str, Any]:
    if state.active_record is None:
        raise ValueError("Active record is required for demand forecasting.")
    ts = fetch_time_series(state.active_record["port"], state.active_record["sku"])
    if len(ts) < 10:
        raise ValueError("Not enough historical data for forecasting.")

    df_records = [{"ds": row["event_date"], "y": row["demand"]} for row in ts]
    from prophet import Prophet
    import pandas as pd

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
    if state.active_record is None or state.config is None:
        raise ValueError("Active record and config are required for simulation.")
    current_inventory = float(state.active_record.get("inventory_level", 0.0))
    incoming = float(state.active_record.get("incoming_supply", 0.0))
    lead_time = float(state.active_record.get("lead_time_days", 1.0))
    alt_route = get_route_map(state.config, state.active_record["port"]).get("backup_route", "Cape of Good Hope")
    stockout_probability = min(100.0, max(0.0, (state.risk_score_composite or 0.0) * 100.0 + (1.0 - (current_inventory / (incoming + 1.0))) * 25.0 + (lead_time / 30.0) * 25.0))
    expected_gap = max(0.0, 100.0 - (current_inventory / (incoming + 1.0)) * 100.0)
    return {
        "simulation_result": SimulationResult(
            stockout_probability_pct=round(stockout_probability, 2),
            expected_inventory_gap_pct=round(expected_gap, 2),
            alternate_route=alt_route,
        ),
        "agent_logs": state.agent_logs + ["L6: Simulation completed."],
    }


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    if state.risk_label is None or state.simulation_result is None or state.forecast_result is None:
        raise ValueError("Risk label, simulation results, and forecast result are required for mitigation.")
    stockout = state.simulation_result.stockout_probability_pct
    forecast_drop = state.forecast_result.expected_drop_pct
    alt_route = state.simulation_result.alternate_route or "the configured backup route"
    recommendations = [
        f"Raise safety stock for the affected product using the {stockout:.1f}% stockout estimate.",
        f"Prepare diversion through {alt_route} and confirm carrier capacity.",
        f"Review alternate suppliers and align purchase orders to the {forecast_drop:.1f}% forecast variance.",
    ]
    cost_delta = (
        "High: expedite critical inventory and activate alternate sourcing."
        if state.risk_label == "CRITICAL"
        else "Moderate: reserve backup logistics and inventory capacity."
    )
    parsed = MitigationSchema(
        summary=(
            f"{state.risk_label} electronics supply-chain risk requires "
            "inventory, routing, and supplier actions."
        ),
        recommendations=recommendations,
        cost_delta=cost_delta,
    )
    insert_mitigation_action(
        state.active_record["event_date"],
        state.active_record["port"],
        state.active_record["sku"],
        state.risk_label,
        json.dumps(parsed.recommendations),
        parsed.cost_delta,
    )
    return {
        "mitigation_action": MitigationAction(**parsed.dict()),
        "agent_logs": state.agent_logs + ["L7: Mitigation recommendation generated and persisted."],
    }


def run_agent_graph(payload: Dict[str, Any]) -> GlobalState:
    state = GlobalState()
    ingestion_delta = data_ingestion_agent(state, payload)
    state = state.copy(update=ingestion_delta)

    news_delta = news_event_analysis_agent(state)
    state = state.copy(update=news_delta)

    weather_delta = weather_risk_monitoring_agent(state)
    state = state.copy(update=weather_delta)

    risk_delta = risk_classifier_agent(state)
    state = state.copy(update=risk_delta)

    forecast_delta = demand_forecasting_agent(state)
    state = state.copy(update=forecast_delta)

    simulation_delta = simulation_agent(state)
    state = state.copy(update=simulation_delta)

    mitigation_delta = mitigation_recommendation_agent(state)
    state = state.copy(update=mitigation_delta)
    return state
