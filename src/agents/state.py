# src/agents/state.py
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
    expected_duration_days: Optional[float] = None  # NEW: extracted from RAG text by News Agent


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


class RiskClassificationResult(BaseModel):
    """Full audit trail for one risk classification run."""
    mode: str                          # "replay" or "live"
    composite_score: float = Field(..., ge=0.0, le=1.0)
    geo_component: float               # normalized 0-1
    supply_component: float            # normalized 0-1
    freight_component: float           # news_severity folded into freight; normalized 0-1
    defect_component: float            # normalized 0-1
    duration_days: Optional[float]     # None = no duration signal; drives escalation
    base_label: str                    # label BEFORE duration escalation
    final_label: str                   # label AFTER duration escalation — the authoritative output
    escalated: bool                    # True when duration pushed label up a tier
    rag_citations: List[str]           # deduplicated source filenames from ChromaDB hits
    rationale: str                     # human-readable explanation for the dashboard
    critical_flag: bool                # True iff final_label == "CRITICAL"; Mitigation Agent reads this for Slack


class GlobalState(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    config: Optional[Dict[str, Any]] = None
    active_record: Optional[Dict[str, Any]] = None
    news_signals: List[NewsRiskSignal] = Field(default_factory=list)
    live_weather_severity: Optional[float] = None
    risk_classification: Optional[RiskClassificationResult] = None
    forecast_result: Optional[ForecastResult] = None
    simulation_result: Optional[SimulationResult] = None
    mitigation_action: Optional[MitigationAction] = None
    agent_logs: List[str] = Field(default_factory=list)

    # ── backwards-compatible read-only properties ──────────────────────────────
    @property
    def risk_label(self) -> Optional[str]:
        """Deprecated shim — read risk_classification.final_label instead."""
        return self.risk_classification.final_label if self.risk_classification else None

    @property
    def risk_score_composite(self) -> Optional[float]:
        """Deprecated shim — read risk_classification.composite_score instead."""
        return self.risk_classification.composite_score if self.risk_classification else None
