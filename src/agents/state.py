# src/agents/state.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


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
    expected_duration_days: Optional[float] = Field(
        None,
        description="LLM duration estimate in days — drives Risk Classifier escalation matrix",
    )


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
    urgency: str = Field("HIGH")
    rag_citations: List[str] = Field(default_factory=list)
    india_sourcing_recommendations: List[str] = Field(default_factory=list)


# ── LLM output models (structured OpenAI responses) ──────────────────────────

class WeatherRiskLLMOutput(BaseModel):
    """L3 LLM output — supply-chain interpretation of Open-Meteo numeric data."""

    event_classification: Literal["extreme", "severe", "moderate", "minor", "clear"] = Field(
        ...,
        description=(
            "Supply-chain severity tier for current weather at this hub. "
            "extreme: typhoon/earthquake forcing fab closure (>72h). "
            "severe: major storm causing 24-72h logistics delays. "
            "moderate: contingency monitoring required. "
            "minor: marginal impact, normal ops with caution. "
            "clear: no weather risk."
        ),
    )
    geo_risk_component: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Enhanced geo risk for composite formula (weight 0.40). "
            "This value overrides the raw numeric Open-Meteo severity."
        ),
    )
    affected_semiconductor_hubs: List[str] = Field(
        ...,
        description=(
            "Impacted hubs from: Hsinchu, Tainan, Osaka, Austin, Shanghai, "
            "Singapore, Rotterdam, Incheon, Penang, Ho_Chi_Minh_City, Shenzhen, Chennai."
        ),
    )
    supply_chain_narrative: str = Field(
        ...,
        description=(
            "2-3 sentences naming the specific fab or logistics node at risk, "
            "estimated delay in days if severe+, and which product category is most exposed."
        ),
    )
    rag_escalation_warranted: bool = Field(
        ...,
        description=(
            "True when geo_risk_component >= 0.65 — signals Risk Classifier to query "
            "ChromaDB for historical weather precedents at this hub."
        ),
    )


class NewsAnalysisLLMOutput(BaseModel):
    """L2 LLM output — structured event classification from disruption type + RAG context."""

    category: Literal["weather", "geopolitical", "logistics", "raw_material", "demand_shock"] = Field(
        ...,
        description=(
            "geopolitical: export controls, sanctions, trade wars. "
            "logistics: port closures, shipping route disruptions. "
            "raw_material: rare earth, wafer supply constraints. "
            "demand_shock: AI surges, inventory gluts. "
            "weather: natural disasters."
        ),
    )
    severity: float = Field(..., ge=0.0, le=1.0, description="Event severity 0-1.")
    affected_regions: List[str] = Field(
        ...,
        description="DataCo region names e.g. Eastern Asia, Western Europe.",
    )
    affected_commodities: List[str] = Field(
        ...,
        description="Specific product classes e.g. advanced logic chips, DRAM memory.",
    )
    news_severity_component: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Freight/logistics proxy for composite formula (weight 0.15).",
    )
    expected_duration_days: float = Field(
        ...,
        gt=0,
        description="Days until primary disruption resolves. Drives escalation matrix.",
    )
    summary: str = Field(
        ...,
        description="2-3 sentences: disruption type, geography, recovery window, Flipkart risk.",
    )
    signal_tags: List[str] = Field(
        ...,
        description="3-6 lowercase hyphenated tags e.g. ['red-sea', 'logistics'].",
    )


class RiskClassifierLLMEnhancement(BaseModel):
    """L4 LLM output — narrative layer added ON TOP of rule-based classification."""

    primary_risk_driver: Literal["geo", "supply", "freight", "defect"] = Field(
        ...,
        description="Component with highest normalised contribution to composite score.",
    )
    enhanced_rationale: str = Field(
        ...,
        description="3-4 sentences for procurement manager citing RAG and component values.",
    )
    evaluator_one_liner: str = Field(
        ...,
        description="≤20 words for Streamlit dashboard risk card.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="high: composite > 0.80 OR delivery_status override applied.",
    )


class MitigationLLMOutput(BaseModel):
    """L7 LLM output — ranked mitigation plan with India sourcing and RAG citations."""

    summary: str = Field(
        ...,
        description="2-3 sentence executive summary for dashboard.",
    )
    ranked_actions: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 specific actions, most urgent first.",
    )
    cost_estimate: str = Field(
        ...,
        description="Format: '<LEVEL>: <reason>'. HIGH | MEDIUM | LOW.",
    )
    urgency: Literal["IMMEDIATE", "HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="IMMEDIATE=CRITICAL label, HIGH=HIGH label, etc.",
    )
    rag_citations: List[str] = Field(
        ...,
        min_length=1,
        description="At least 1 citation from provided RAG context.",
    )
    india_sourcing_recommendations: List[str] = Field(
        ...,
        min_length=1,
        description="At least 1 named India/ASEAN option.",
    )


# ── Ensemble signal models (L4 three-signal + judge) ─────────────────────────

class RuleBasedSignal(BaseModel):
    """Signal 1 — deterministic composite formula + overrides."""

    composite_score: float
    geo_component: float
    supply_component: float
    freight_component: float
    defect_component: float
    base_label: str
    escalated_label: str
    escalated: bool
    duration_days: Optional[float]
    delivery_status_override: Optional[str] = None


class DistilBERTSignal(BaseModel):
    """Signal 2 — Fine-tuned DistilBERT classifier (~20ms CPU)."""

    predicted_label: str
    confidence: float
    probability_distribution: Dict[str, float]
    model_source: str
    inference_ms: float


class LLMSignal(BaseModel):
    """Signal 3 — GPT-4o + two-stage RAG."""

    predicted_label: str
    rationale: str
    rag_citations: List[str]
    rag_chunks_used: int
    confidence_level: Literal["high", "medium", "low"]
    primary_driver: Literal["geo", "supply", "freight", "defect", "delivery_status"]


class JudgeVerdict(BaseModel):
    """LLM-as-Judge final decision after seeing all 3 signals."""

    final_label: str
    verdict_type: Literal[
        "unanimous",
        "majority_rule",
        "override_distilbert",
        "override_llm",
        "defer_to_rules",
    ]
    reasoning: str
    signals_agreed: bool
    disagreement_explanation: Optional[str] = None
    final_critical_flag: bool


class RiskClassificationResult(BaseModel):
    """Full audit trail for one risk classification run."""

    mode: str
    composite_score: float = Field(..., ge=0.0, le=1.0)
    geo_component: float
    supply_component: float
    freight_component: float
    defect_component: float
    duration_days: Optional[float]
    base_label: str
    final_label: str
    escalated: bool
    rag_citations: List[str] = Field(default_factory=list)
    rationale: str
    critical_flag: bool
    llm_enhanced_rationale: Optional[str] = None
    llm_evaluator_one_liner: Optional[str] = None
    llm_primary_driver: Optional[str] = None
    llm_confidence: Optional[str] = None
    rule_signal: Optional[RuleBasedSignal] = None
    distilbert_signal: Optional[DistilBERTSignal] = None
    llm_signal: Optional[LLMSignal] = None
    judge_verdict: Optional[JudgeVerdict] = None


class GlobalState(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    config: Optional[Dict[str, Any]] = None
    active_record: Optional[Dict[str, Any]] = None
    ingestion_run_id: Optional[str] = None  # UUID from L1; links state to live_news_ingest / live_weather_ingest rows
    news_signals: List[NewsRiskSignal] = Field(default_factory=list)
    live_weather_severity: Optional[float] = None
    risk_classification: Optional[RiskClassificationResult] = None
    forecast_result: Optional[ForecastResult] = None
    simulation_result: Optional[SimulationResult] = None
    mitigation_action: Optional[MitigationAction] = None
    agent_logs: List[str] = Field(default_factory=list)
    news_analysis_llm: Optional[NewsAnalysisLLMOutput] = None
    weather_risk_llm: Optional[WeatherRiskLLMOutput] = None
    risk_enhancement_llm: Optional[RiskClassifierLLMEnhancement] = None
    mitigation_llm: Optional[MitigationLLMOutput] = None
    judge_verdict: Optional[JudgeVerdict] = None

    @property
    def risk_label(self) -> Optional[str]:
        """Deprecated shim — read risk_classification.final_label instead."""
        return self.risk_classification.final_label if self.risk_classification else None

    @property
    def risk_score_composite(self) -> Optional[float]:
        """Deprecated shim — read risk_classification.composite_score instead."""
        return self.risk_classification.composite_score if self.risk_classification else None
