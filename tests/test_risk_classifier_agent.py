"""
tests/test_risk_classifier_agent.py

Unit tests for the Risk Classifier Agent (Agent 4).
Covers all 8 cases from the build spec §7 plus additional data-grounded assertions.

Run: pytest tests/test_risk_classifier_agent.py -v
"""
import json
import sqlite3
import sys
import os
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import (
    EventMetadata,
    GlobalState,
    NewsRiskSignal,
    RiskClassificationResult,
)
from src.agents.risk_classifier_agent import (
    _apply_delivery_floor,
    _base_label_from_delivery_status,
    _compute_components,
    _escalate_label,
    _max_duration_days,
    _norm,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_state(
    delivery_status: Optional[str] = None,
    risk_score_composite: Optional[float] = None,
    disruption_event_label: Optional[str] = None,
    supply_disruption_index: float = 7.0,
    defect_rate_pct: float = 10.0,
    natural_disaster_risk: float = 5.0,
    news_signals: Optional[list] = None,
    live_weather_severity: float = 0.5,
    shock_duration_days: int = 0,
    order_region: str = "Eastern Asia",
) -> GlobalState:
    """Helper to build a minimal GlobalState for testing."""
    return GlobalState(
        event_metadata=EventMetadata(
            disruption_type="earthquake",
            affected_port="Hsinchu",
            affected_route="Hsinchu to Singapore",
            severity=0.8,
            shock_duration_days=shock_duration_days,
            recovery_window_days=60,
            synthetic_ratio=0.0,
        ),
        active_record={
            "order_id": 99999,
            "order_date": "2024-01-15",
            "port": order_region,
            "sku": "CHIP_AP",
            "delivery_status": delivery_status,
            "risk_score_composite": risk_score_composite,
            "disruption_event_label": disruption_event_label,
            "supply_disruption_index": supply_disruption_index,
            "defect_rate_pct": defect_rate_pct,
            "natural_disaster_risk": natural_disaster_risk,
            "export_control_level": 3.0,
            "order_region": order_region,
            "year": 2024,
        },
        news_signals=news_signals or [],
        live_weather_severity=live_weather_severity,
    )


# ── §8 Test 1: Replay mode — stored composite returned unchanged ──────────────

class TestReplayMode:
    def test_replay_uses_stored_composite(self):
        """Replay mode must return the stored composite_score unchanged."""
        stored = 0.583
        state = make_state(
            risk_score_composite=stored,
            disruption_event_label="HIGH",
            delivery_status="Late delivery",
        )
        # Patch DB writes so test doesn't need a real DB
        with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
            with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
                with patch("src.agents.risk_classifier_agent.update_risk_label"):
                    with patch("src.agents.risk_classifier_agent._get_norm_bounds") as mock_bounds:
                        mock_bounds.return_value = {
                            "weather_severity_hub": (1.18, 10.0),
                            "natural_disaster_risk": (1.18, 10.0),
                            "supply_disruption_index": (4.09, 9.97),
                            "defect_rate_pct": (2.0, 19.82),
                            "disruption_news_count": (0.0, 17.0),
                        }
                        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                            from src.agents.risk_classifier_agent import risk_classifier_agent
                            result = risk_classifier_agent(state)

        rc = result["risk_classification"]
        assert rc.mode == "replay", f"Expected replay, got {rc.mode}"
        assert abs(rc.composite_score - stored) < 1e-6, \
            f"Replay mode must use stored composite {stored}, got {rc.composite_score}"

    def test_replay_never_writes_lite_master(self):
        """Replay mode must NOT call update_risk_label (would corrupt ground truth)."""
        state = make_state(
            risk_score_composite=0.55,
            disruption_event_label="HIGH",
            delivery_status="Late delivery",
        )
        with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
            with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
                with patch("src.agents.risk_classifier_agent.update_risk_label") as mock_update:
                    with patch("src.agents.risk_classifier_agent._get_norm_bounds") as mock_bounds:
                        mock_bounds.return_value = {
                            "weather_severity_hub": (1.18, 10.0),
                            "natural_disaster_risk": (1.18, 10.0),
                            "supply_disruption_index": (4.09, 9.97),
                            "defect_rate_pct": (2.0, 19.82),
                            "disruption_news_count": (0.0, 17.0),
                        }
                        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                            from src.agents.risk_classifier_agent import risk_classifier_agent
                            risk_classifier_agent(state)
        mock_update.assert_not_called()


# ── §8 Test 2: Live mode formula ──────────────────────────────────────────────

class TestLiveModeFormula:
    def test_formula_matches_spec_to_3_decimal_places(self):
        """
        Hand-constructed inputs, verify formula:
        composite = 0.4*geo + 0.3*supply + 0.15*freight + 0.15*defect
        """
        BOUNDS = {
            "weather_severity_hub": (1.18, 10.0),
            "natural_disaster_risk": (1.18, 10.0),
            "supply_disruption_index": (4.09, 9.97),
            "defect_rate_pct": (2.0, 19.82),
            "disruption_news_count": (0.0, 17.0),
        }

        # Inputs
        weather_sev = 0.7   # already 0-1 (live_weather_severity from Open-Meteo)
        nat_disaster = 5.0  # raw 0-10 scale
        sdi = 8.0           # raw 0-10 scale
        news_severity = 0.8 # already 0-1 from signal
        defect = 12.0       # raw %

        # Expected components
        geo = max(
            0.7,  # live_weather_severity is already 0-1 from api_clients
            _norm(nat_disaster, *BOUNDS["natural_disaster_risk"]),
        )
        supply = _norm(sdi, *BOUNDS["supply_disruption_index"])
        freight = 0.8
        defect_n = _norm(defect, *BOUNDS["defect_rate_pct"])

        expected = round(0.4 * geo + 0.3 * supply + 0.15 * freight + 0.15 * defect_n, 3)

        with patch("src.agents.risk_classifier_agent._get_norm_bounds", return_value=BOUNDS):
            components = _compute_components(
                live_weather_severity=weather_sev,
                natural_disaster_risk=nat_disaster,
                supply_disruption_index=sdi,
                news_signals=[
                    NewsRiskSignal(
                        source_id="t1", category="earthquake", severity=news_severity,
                        summary="test", signal_tags=[], expected_duration_days=None,
                    )
                ],
                defect_rate_pct=defect,
                order_region="Eastern Asia",
            )

        actual = round(
            0.4 * components["geo"] + 0.3 * components["supply"]
            + 0.15 * components["freight"] + 0.15 * components["defect"],
            3,
        )
        assert actual == expected, f"Formula mismatch: expected {expected}, got {actual}"


# ── §8 Test 3: All five duration escalation examples from the spec ─────────────

class TestDurationEscalation:
    def test_low_composite_5day_becomes_critical(self):
        """
        LOW base label + 5-day duration → CRITICAL (>= 4 day hard floor).
        Spec example: "minor weather blip ... port closure will last 5 days"
        """
        final, escalated = _escalate_label("LOW", 5.0)
        assert final == "CRITICAL", f"Expected CRITICAL, got {final}"
        assert escalated is True

    def test_high_1day_stays_high(self):
        """HIGH + 1 day → stays HIGH, no escalation."""
        final, escalated = _escalate_label("HIGH", 1.0)
        assert final == "HIGH"
        assert escalated is False

    def test_medium_2day_becomes_high(self):
        """MEDIUM + 2 days → escalates to HIGH."""
        final, escalated = _escalate_label("MEDIUM", 2.0)
        assert final == "HIGH"
        assert escalated is True

    def test_critical_already_1day_stays_critical(self):
        """CRITICAL + 1 day → stays CRITICAL (nothing above to escalate to)."""
        final, escalated = _escalate_label("CRITICAL", 1.0)
        assert final == "CRITICAL"
        assert escalated is False

    def test_no_duration_stays_unchanged(self):
        """None duration → no escalation regardless of base label."""
        for label in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            final, escalated = _escalate_label(label, None)
            assert final == label, f"With no duration {label} should not change"
            assert escalated is False

    def test_3day_escalates_one_tier(self):
        """3 days escalates exactly one tier."""
        final, escalated = _escalate_label("LOW", 3.0)
        assert final == "MEDIUM"
        assert escalated is True

    def test_4day_hard_floor_always_critical(self):
        """4 days → CRITICAL regardless of base label (hard floor)."""
        for label in ("LOW", "MEDIUM", "HIGH"):
            final, escalated = _escalate_label(label, 4.0)
            assert final == "CRITICAL", f"4-day floor failed for base={label}"

    def test_never_de_escalates(self):
        """
        Short duration (1 day) does NOT lower a CRITICAL label.
        The classifier never de-escalates — only raises.
        """
        final, escalated = _escalate_label("CRITICAL", 0.5)
        assert final == "CRITICAL"
        assert escalated is False


# ── §8 Test 4: delivery_status "Cancelled" → always CRITICAL ─────────────────

class TestDeliveryStatusMapping:
    @pytest.mark.parametrize("ds,expected", [
        ("Shipping canceled",  "CRITICAL"),
        ("Late delivery",      "HIGH"),
        ("Advance shipping",   "LOW"),
        ("Shipping on time",   "LOW"),
    ])
    def test_exact_delivery_status_strings(self, ds, expected):
        """Exact DataCo delivery_status strings must map to the correct base label."""
        label = _base_label_from_delivery_status(ds, composite_score=0.5)
        assert label == expected, f"'{ds}' → expected {expected}, got {label}"

    def test_cancelled_always_critical_regardless_of_score(self):
        """Shipping canceled → CRITICAL even when composite_score is near zero."""
        for score in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
            label = _base_label_from_delivery_status("Shipping canceled", score)
            assert label == "CRITICAL", f"Score {score}: expected CRITICAL, got {label}"

    def test_none_delivery_status_uses_score_thresholds(self):
        """When delivery_status is None, fall back to composite_score thresholds."""
        assert _base_label_from_delivery_status(None, 0.80) == "CRITICAL"
        assert _base_label_from_delivery_status(None, 0.60) == "HIGH"
        assert _base_label_from_delivery_status(None, 0.35) == "MEDIUM"
        assert _base_label_from_delivery_status(None, 0.10) == "LOW"

    def test_score_exactly_at_threshold_boundaries(self):
        """Boundary values for score-based thresholds."""
        assert _base_label_from_delivery_status(None, 0.75) == "CRITICAL"
        assert _base_label_from_delivery_status(None, 0.50) == "HIGH"
        assert _base_label_from_delivery_status(None, 0.25) == "MEDIUM"
        assert _base_label_from_delivery_status(None, 0.249) == "LOW"


class TestDeliveryFloor:
    def test_floor_raises_llm_label_below_canceled_override(self):
        assert _apply_delivery_floor("LOW", "CRITICAL") == "CRITICAL"
        assert _apply_delivery_floor("MEDIUM", "CRITICAL") == "CRITICAL"

    def test_floor_raises_llm_label_below_late_delivery_override(self):
        assert _apply_delivery_floor("LOW", "HIGH") == "HIGH"
        assert _apply_delivery_floor("MEDIUM", "HIGH") == "HIGH"

    def test_floor_keeps_higher_label(self):
        assert _apply_delivery_floor("CRITICAL", "HIGH") == "CRITICAL"

    def test_floor_no_override_passthrough(self):
        assert _apply_delivery_floor("MEDIUM", None) == "MEDIUM"


# ── §8 Test 5: RAG citation gating ────────────────────────────────────────────

class TestRAGGating:
    def test_no_rag_call_for_low_not_escalated(self):
        """LOW label + escalated=False → zero RAG calls."""
        with patch("src.agents.risk_classifier_agent.query_chroma_rag") as mock_rag:
            from src.agents.risk_classifier_agent import _gather_rag_citations
            citations, _ = _gather_rag_citations("LOW", escalated=False)
            assert mock_rag.call_count == 0
            assert citations == []

    def test_no_rag_call_for_medium_not_escalated(self):
        """MEDIUM label + escalated=False → zero RAG calls."""
        with patch("src.agents.risk_classifier_agent.query_chroma_rag") as mock_rag:
            from src.agents.risk_classifier_agent import _gather_rag_citations
            citations, _ = _gather_rag_citations("MEDIUM", escalated=False)
            assert mock_rag.call_count == 0

    def test_rag_fires_for_high(self):
        """HIGH label triggers RAG lookup."""
        mock_hit = {
            "text": "Red Sea disruption caused 40% cost increase",
            "metadata": {"source": "Disruptions_at_Red_Sea_route.docx", "type": "static_report"},
            "distance": 0.3,
        }
        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[mock_hit]):
            from src.agents.risk_classifier_agent import _gather_rag_citations
            citations, rationale = _gather_rag_citations("HIGH", escalated=False)
            assert len(citations) > 0

    def test_rag_fires_when_escalated_even_if_final_label_medium(self):
        """Escalated LOW→MEDIUM should still get RAG (most important moment for grounding)."""
        mock_hit = {
            "text": "Port strike lasted 3 days",
            "metadata": {"source": "playbook_port_strike.txt", "type": "mitigation_playbook"},
            "distance": 0.25,
        }
        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[mock_hit]):
            from src.agents.risk_classifier_agent import _gather_rag_citations
            citations, _ = _gather_rag_citations("MEDIUM", escalated=True)
            assert len(citations) > 0

    def test_rag_fires_for_critical(self):
        """CRITICAL label triggers RAG lookup."""
        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]) as mock_rag:
            from src.agents.risk_classifier_agent import _gather_rag_citations
            _gather_rag_citations("CRITICAL", escalated=False)
            assert mock_rag.call_count >= 1


# ── §8 Test 6: SQLite write behaviour ─────────────────────────────────────────

class TestSQLiteWrites:
    def test_live_mode_writes_risk_classification_row(self):
        """Live mode must write a row to risk_classifications."""
        state = make_state(
            risk_score_composite=None,  # None = triggers live mode
            delivery_status=None,
            live_weather_severity=0.9,
        )
        with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
            with patch("src.agents.risk_classifier_agent.insert_risk_classification") as mock_insert:
                with patch("src.agents.risk_classifier_agent.update_risk_label"):
                    with patch("src.agents.risk_classifier_agent._get_norm_bounds") as mock_bounds:
                        mock_bounds.return_value = {
                            "weather_severity_hub": (1.18, 10.0),
                            "natural_disaster_risk": (1.18, 10.0),
                            "supply_disruption_index": (4.09, 9.97),
                            "defect_rate_pct": (2.0, 19.82),
                            "disruption_news_count": (0.0, 17.0),
                        }
                        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                            from src.agents.risk_classifier_agent import risk_classifier_agent
                            risk_classifier_agent(state)
        mock_insert.assert_called_once()

    def test_replay_mode_does_not_update_lite_master(self):
        """Replay mode must call insert_risk_classification but NOT update_risk_label."""
        state = make_state(
            risk_score_composite=0.55,
            disruption_event_label="HIGH",
            delivery_status="Late delivery",
        )
        with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
            with patch("src.agents.risk_classifier_agent.insert_risk_classification") as mock_insert:
                with patch("src.agents.risk_classifier_agent.update_risk_label") as mock_update:
                    with patch("src.agents.risk_classifier_agent._get_norm_bounds") as mock_bounds:
                        mock_bounds.return_value = {
                            "weather_severity_hub": (1.18, 10.0),
                            "natural_disaster_risk": (1.18, 10.0),
                            "supply_disruption_index": (4.09, 9.97),
                            "defect_rate_pct": (2.0, 19.82),
                            "disruption_news_count": (0.0, 17.0),
                        }
                        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                            from src.agents.risk_classifier_agent import risk_classifier_agent
                            risk_classifier_agent(state)
        mock_insert.assert_called_once()    # audit row still written
        mock_update.assert_not_called()     # lite_master NOT overwritten


# ── Bonus: Demo scenario sanity checks ────────────────────────────────────────

class TestDemoScenarios:
    """
    Verify that the two demo scenarios (Taiwan earthquake, Red Sea) produce
    the expected direction of labels BEFORE any demo_injector integration.
    Uses only the pure helper functions.
    """

    def test_taiwan_earthquake_is_critical(self):
        """
        Taiwan earthquake: extreme weather + extreme natural disaster.
        Even without duration, the base label should be CRITICAL.
        """
        BOUNDS = {
            "weather_severity_hub": (1.18, 10.0),
            "natural_disaster_risk": (1.18, 10.0),
            "supply_disruption_index": (4.09, 9.97),
            "defect_rate_pct": (2.0, 19.82),
            "disruption_news_count": (0.0, 17.0),
        }
        with patch("src.agents.risk_classifier_agent._get_norm_bounds", return_value=BOUNDS):
            components = _compute_components(
                live_weather_severity=0.95,   # severe earthquake
                natural_disaster_risk=9.8,    # near max
                supply_disruption_index=9.5,  # chip supply severely hit
                news_signals=[
                    NewsRiskSignal(
                        source_id="s1", category="earthquake", severity=0.9,
                        summary="TSMC fabs offline for estimated 6 days",
                        signal_tags=["earthquake", "TSMC"],
                        expected_duration_days=6.0,
                    )
                ],
                defect_rate_pct=15.0,
                order_region="Eastern Asia",
            )
        composite = (
            0.4 * components["geo"]
            + 0.3 * components["supply"]
            + 0.15 * components["freight"]
            + 0.15 * components["defect"]
        )
        # Base label from score (delivery_status=None in demo)
        base = _base_label_from_delivery_status(None, composite)
        # Duration escalation (6-day estimated outage)
        final, _ = _escalate_label(base, 6.0)
        assert final == "CRITICAL", f"Taiwan earthquake should be CRITICAL, got {final}"

    def test_red_sea_crisis_escalates_with_duration(self):
        """
        Red Sea: news-only signal. Short-term composite may be MEDIUM,
        but multi-week duration (crisis lasted months) escalates to CRITICAL.
        """
        BOUNDS = {
            "weather_severity_hub": (1.18, 10.0),
            "natural_disaster_risk": (1.18, 10.0),
            "supply_disruption_index": (4.09, 9.97),
            "defect_rate_pct": (2.0, 19.82),
            "disruption_news_count": (0.0, 17.0),
        }
        with patch("src.agents.risk_classifier_agent._get_norm_bounds", return_value=BOUNDS):
            components = _compute_components(
                live_weather_severity=0.2,   # no severe weather
                natural_disaster_risk=3.0,   # low
                supply_disruption_index=7.5, # moderate
                news_signals=[
                    NewsRiskSignal(
                        source_id="s2", category="geopolitical", severity=0.85,
                        summary="Red Sea shipping attacks expected to persist for 30 days",
                        signal_tags=["red_sea", "freight"],
                        expected_duration_days=30.0,
                    )
                ],
                defect_rate_pct=10.0,
                order_region="Western Europe",
            )
        composite = (
            0.4 * components["geo"]
            + 0.3 * components["supply"]
            + 0.15 * components["freight"]
            + 0.15 * components["defect"]
        )
        base = _base_label_from_delivery_status(None, composite)
        final, escalated = _escalate_label(base, 30.0)
        assert final == "CRITICAL", f"Red Sea with 30-day duration should be CRITICAL, got {final}"
        assert escalated is True


# ── Normalization unit tests ───────────────────────────────────────────────────

class TestNorm:
    def test_min_value_returns_zero(self):
        assert _norm(1.18, 1.18, 10.0) == 0.0

    def test_max_value_returns_one(self):
        assert _norm(10.0, 1.18, 10.0) == pytest.approx(1.0)

    def test_midpoint(self):
        mid = (1.18 + 10.0) / 2
        assert abs(_norm(mid, 1.18, 10.0) - 0.5) < 0.01

    def test_clamped_below_zero(self):
        assert _norm(-5.0, 1.18, 10.0) == 0.0

    def test_clamped_above_one(self):
        assert _norm(100.0, 1.18, 10.0) == 1.0

    def test_equal_bounds_returns_zero(self):
        assert _norm(5.0, 5.0, 5.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
