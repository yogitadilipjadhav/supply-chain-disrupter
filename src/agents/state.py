from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class EventMetadata(BaseModel):
    disruption_type: str
    affected_port: str
    affected_route: str
    severity: float
    shock_duration_days: int
    recovery_window_days: int
    synthetic_ratio: float


class NewsRiskSignal(BaseModel):
    source_id: str
    category: str
    severity: float
    summary: str
    signal_tags: List[str]


class ForecastResult(BaseModel):
    prophet_forecast: List[Dict[str, Any]]
    expected_drop_pct: float


class SimulationResult(BaseModel):
    stockout_probability_pct: float
    expected_inventory_gap_pct: float
    alternate_route: Optional[str]


class MitigationAction(BaseModel):
    summary: str
    recommendations: List[str]
    cost_delta: str


class GlobalState(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    config: Optional[Dict[str, Any]] = None
    active_record: Optional[Dict[str, Any]] = None
    news_signals: List[NewsRiskSignal] = Field(default_factory=list)
    live_weather_severity: Optional[float] = None
    risk_label: Optional[str] = None
    risk_score_composite: Optional[float] = None
    forecast_result: Optional[ForecastResult] = None
    simulation_result: Optional[SimulationResult] = None
    mitigation_action: Optional[MitigationAction] = None
    agent_logs: List[str] = Field(default_factory=list)
