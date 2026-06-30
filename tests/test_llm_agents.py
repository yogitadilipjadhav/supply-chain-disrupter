"""
Test suite for OpenAI LLM agent integration.
Run: python -m pytest tests/test_llm_agents.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import (
    GlobalState,
    MitigationLLMOutput,
    NewsAnalysisLLMOutput,
    RiskClassificationResult,
    RuleBasedSignal,
    WeatherRiskLLMOutput,
)
from src.utils.openai_utils import build_rag_context


def test_missing_api_key():
    """_get_client raises RuntimeError when OPENAI_API_KEY is unset."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        from src.utils import openai_utils
        openai_utils._get_client.cache_clear()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            openai_utils._get_client()


def test_news_agent_fallback_on_llm_error():
    """News agent falls back when LLM call fails."""
    from src.agents.news_agent import news_event_analysis_agent
    from src.agents.state import EventMetadata

    state = GlobalState(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="Hsinchu to Singapore",
            severity=0.8,
            shock_duration_days=6,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        active_record={"order_region": "Eastern Asia", "year": 2022},
    )
    with patch("src.agents.news_agent.call_openai_structured", side_effect=RuntimeError("rate limit")):
        with patch("src.agents.news_agent.build_news_signals", return_value=[]):
            result = news_event_analysis_agent(state)
    assert len(result["news_signals"]) >= 1
    assert result["news_analysis_llm"] is None
    assert result["news_signals"][0].expected_duration_days is not None


def test_weather_llm_overrides_numeric_severity():
    """Weather agent uses LLM geo_risk_component when LLM succeeds."""
    from src.agents.weather_agent import weather_risk_monitoring_agent
    from src.agents.state import EventMetadata

    state = GlobalState(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="test",
            severity=0.9,
            shock_duration_days=0,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        config={"ports": {"Hsinchu": {"latitude": 24.8, "longitude": 120.97}}},
        active_record={"latitude": 24.8, "longitude": 120.97, "order_region": "Eastern Asia"},
    )
    mock_output = WeatherRiskLLMOutput(
        event_classification="extreme",
        geo_risk_component=0.91,
        affected_semiconductor_hubs=["Hsinchu"],
        supply_chain_narrative="Test narrative",
        rag_escalation_warranted=True,
    )
    with patch("src.agents.weather_agent.fetch_open_meteo", return_value={"hourly": {"windspeed_10m": [10], "precipitation": [1], "weathercode": [95]}}):
        with patch("src.agents.weather_agent.compute_weather_severity", return_value=0.5):
            with patch("src.agents.weather_agent.has_openai_api_key", return_value=True):
                with patch("src.agents.weather_agent.call_openai_structured", return_value=mock_output):
                    result = weather_risk_monitoring_agent(state)
    assert result["live_weather_severity"] == 0.91


def test_risk_classifier_llm_enhancement_non_blocking():
    """Risk classifier completes even when ensemble LLM calls fail."""
    from src.agents.risk_classifier_agent import risk_classifier_agent
    from src.agents.state import EventMetadata, NewsRiskSignal

    state = GlobalState(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="test",
            severity=0.8,
            shock_duration_days=0,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        active_record={
            "order_id": 99999,
            "delivery_status": "Late delivery",
            "supply_disruption_index": 7.5,
            "defect_rate_pct": 10.0,
            "natural_disaster_risk": 5.0,
            "export_control_level": 3.0,
            "order_region": "Eastern Asia",
            "year": 2024,
        },
        news_signals=[],
        live_weather_severity=0.5,
    )
    with patch("src.agents.risk_classifier_agent.run_llm_signal", return_value=None):
        with patch("src.agents.risk_classifier_agent.run_judge", return_value=None):
            with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
                with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
                    result = risk_classifier_agent(state)
    rc = result["risk_classification"]
    assert rc is not None
    assert isinstance(rc.critical_flag, bool)


def test_canceled_shipment_floor_when_llm_disagrees():
    """Shipping canceled must stay CRITICAL even if Signal 3 predicts lower."""
    from src.agents.risk_classifier_agent import risk_classifier_agent
    from src.agents.state import LLMSignal

    state = GlobalState(
        event_metadata=__import__("src.agents.state", fromlist=["EventMetadata"]).EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="test",
            severity=0.8,
            shock_duration_days=0,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        active_record={
            "order_id": 99999,
            "delivery_status": "Shipping canceled",
            "supply_disruption_index": 4.0,
            "defect_rate_pct": 5.0,
            "natural_disaster_risk": 3.0,
            "export_control_level": 2.0,
            "order_region": "Eastern Asia",
            "year": 2024,
        },
        news_signals=[],
        live_weather_severity=0.2,
    )
    llm_wrong = LLMSignal(
        predicted_label="MEDIUM",
        rationale="test",
        rag_citations=[],
        rag_chunks_used=0,
        confidence_level="medium",
        primary_driver="supply",
    )
    with patch("src.agents.risk_classifier_agent.run_llm_signal", return_value=llm_wrong):
        with patch("src.agents.risk_classifier_agent.run_judge", return_value=None):
            with patch("src.agents.risk_classifier_agent.run_distilbert_inference") as mock_db:
                from src.agents.state import DistilBERTSignal
                mock_db.return_value = DistilBERTSignal(
                    predicted_label="N/A", confidence=0.0,
                    probability_distribution={"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.0},
                    model_source="skipped", inference_ms=0.0,
                )
                with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
                    with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
                        with patch("src.agents.risk_classifier_agent._get_norm_bounds") as mock_bounds:
                            mock_bounds.return_value = {
                                "weather_severity_hub": (1.18, 10.0),
                                "natural_disaster_risk": (1.18, 10.0),
                                "supply_disruption_index": (4.09, 9.97),
                                "defect_rate_pct": (2.0, 19.82),
                                "disruption_news_count": (0.0, 17.0),
                            }
                            with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                                result = risk_classifier_agent(state)
    rc = result["risk_classification"]
    assert rc.final_label == "CRITICAL"
    assert rc.critical_flag is True


def test_mitigation_agent_llm_path():
    """Mitigation agent produces structured action when LLM succeeds."""
    from src.agents.mitigation_agent import mitigation_recommendation_agent

    risk = RiskClassificationResult(
        mode="live", composite_score=0.55,
        geo_component=0.4, supply_component=0.5, freight_component=0.6, defect_component=0.4,
        duration_days=None, base_label="HIGH", final_label="HIGH",
        escalated=False, rationale="test", critical_flag=False,
    )
    state = GlobalState(
        event_metadata=__import__("src.agents.state", fromlist=["EventMetadata"]).EventMetadata(
            disruption_type="port closure", affected_port="Rotterdam", affected_route="test",
            severity=0.6, shock_duration_days=0, recovery_window_days=60, synthetic_ratio=0.0,
        ),
        active_record={"event_date": "2024-01-01", "port": "Rotterdam", "sku": "CHIP_AP"},
        risk_classification=risk,
    )
    mock_output = MitigationLLMOutput(
        summary="HIGH risk test",
        ranked_actions=["[ROUTING] test", "[INVENTORY] test", "[SOURCING] test"],
        cost_estimate="MEDIUM: test",
        urgency="HIGH",
        rag_citations=["Source: test — relevance"],
        india_sourcing_recommendations=["Dixon Technologies — test"],
    )
    with patch("src.agents.mitigation_agent.has_openai_api_key", return_value=True):
        with patch("src.agents.mitigation_agent.call_openai_structured", return_value=mock_output):
            with patch("src.agents.mitigation_agent.build_mitigation_context", return_value=""):
                with patch("src.agents.mitigation_agent.insert_mitigation_action"):
                    result = mitigation_recommendation_agent(state)
    action = result["mitigation_action"]
    assert action.summary == "HIGH risk test"
    assert len(action.recommendations) == 3
    assert action.urgency == "HIGH"


def test_pydantic_schemas_are_flat():
    """All LLM output models must have flat JSON schemas (no $defs)."""
    from src.agents.state import RiskClassifierLLMEnhancement

    for Model in [NewsAnalysisLLMOutput, WeatherRiskLLMOutput, RiskClassifierLLMEnhancement, MitigationLLMOutput]:
        schema = Model.model_json_schema()
        assert schema["type"] == "object"
        assert "$defs" not in schema
        for field, prop in schema.get("properties", {}).items():
            assert prop.get("description"), f"{Model.__name__}.{field} missing description"


def test_build_rag_context_handles_none_queries():
    """build_rag_context skips None items without raising."""
    with patch("src.rag.utils.query_chroma_rag", return_value=[]):
        result = build_rag_context([("test query", 3), None, ("another query", 2)])
    assert isinstance(result, str)
