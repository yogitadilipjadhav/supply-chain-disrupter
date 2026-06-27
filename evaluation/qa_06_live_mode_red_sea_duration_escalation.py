"""
QA-06 | Live mode — Red Sea geopolitical crisis with duration escalation
=========================================================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : risk_classifier_agent(state)
Data source  : Fully synthetic record (no DB row); real norm bounds from lite_master.

What this file verifies
-----------------------
This scenario tests the duration escalation path specifically:
  - The composite score alone produces a MEDIUM base_label (no weather event,
    moderate SDI — the label would stay MEDIUM without duration).
  - A prolonged crisis duration (45 days from EventMetadata, 30 days from
    news signal) forces the label up to CRITICAL via the hard floor (>= 4 days).
  - escalated=True confirms the duration drove the label change.

This is the critical distinction between this test and QA-05 (Taiwan):
  Taiwan → base is already CRITICAL from extreme composite score.
  Red Sea → base starts at MEDIUM but duration forces it to CRITICAL.

Scenario: 2024 Red Sea / Suez Canal Houthi attacks
  - Geopolitical disruption; no major weather or natural disaster component.
  - Low live_weather_severity (0.2), low natural_disaster_risk (3.0).
  - news signal severity 0.85 (freight proxy) with 30-day expected duration.
  - EventMetadata.shock_duration_days=45 (months-long crisis).
  - _max_duration_days picks max(news=30, shock=45) = 45 days.
  - 45 days >= 4-day hard floor → CRITICAL regardless of base label.

Expected outcome: all 3 assertions PASS.
  final_label=CRITICAL, escalated=True, duration_days=45.0
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

from unittest.mock import patch

from src.agents.risk_classifier_agent import risk_classifier_agent
from src.agents.state import EventMetadata, GlobalState, NewsRiskSignal


# ---------------------------------------------------------------------------
# Helper: print PASS/FAIL and track overall result
# ---------------------------------------------------------------------------
all_pass = True

def chk(condition: bool, msg: str) -> None:
    """Print PASS or FAIL for one assertion and update the global all_pass flag."""
    global all_pass
    if not condition:
        all_pass = False
    print("PASS |" if condition else "FAIL |", msg)


print("=== QA-06: Live Mode — Red Sea Crisis (Duration-Driven Escalation) ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — Build GlobalState with a synthetic Red Sea / Houthi scenario.
# Key design choices:
#   • natural_disaster_risk=3.0 (low — geopolitical, not a weather/seismic event)
#   • live_weather_severity=0.2 (no weather impact)
#   • supply_disruption_index=7.5 (moderate disruption, not extreme)
#   → These inputs produce a MEDIUM composite without duration.
#   • news_signal.expected_duration_days=30  (Houthi crisis duration from news)
#   • shock_duration_days=45                 (scenario-level estimate)
#   → max(30, 45)=45 days triggers the hard CRITICAL floor.
# ---------------------------------------------------------------------------
state = GlobalState(
    event_metadata=EventMetadata(
        disruption_type="geopolitical",
        affected_port="Rotterdam",
        affected_route="Suez Canal to Rotterdam",
        severity=0.75,
        shock_duration_days=45,        # months-long Red Sea crisis
        recovery_window_days=120,
        synthetic_ratio=1.0,
    ),
    active_record={
        "order_id": None,
        "order_date": "2024-01-15",
        "event_date": "2024-01-15",
        "port": "Western Europe",
        "sku": "ELECTRONICS_EU",
        "delivery_status": None,             # no delivery_status → score-based base_label
        "risk_score_composite": None,        # None → live mode
        "disruption_event_label": None,
        "supply_disruption_index": 7.5,      # moderate (range: 4.09–9.97)
        "defect_rate_pct": 10.0,             # near-average defect rate
        "natural_disaster_risk": 3.0,        # low — this is a geopolitical event
        "export_control_level": 3.0,
        "order_region": "Western Europe",
        "year": 2024,
        "inventory_level": 800,
        "incoming_supply": 200,
        "lead_time_days": 30,
        "chip_risk": 0.3,
        "supplier_risk": 0.4,
    },
    news_signals=[
        NewsRiskSignal(
            source_id="rs-001",
            category="geopolitical",
            severity=0.85,            # high freight disruption severity
            summary="Houthi attacks on Red Sea shipping expected to persist for 30 days",
            signal_tags=["red_sea", "freight", "shipping"],
            expected_duration_days=30.0,   # 30-day duration from this news signal
        ),
    ],
    live_weather_severity=0.2,             # no significant weather component
)

# ---------------------------------------------------------------------------
# Step 2 — Run the classifier with DB and RAG patched out.
# ---------------------------------------------------------------------------
with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
    with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
        with patch("src.agents.risk_classifier_agent.update_risk_label"):
            with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                result = risk_classifier_agent(state)

rc = result["risk_classification"]

# ---------------------------------------------------------------------------
# Step 3 — Print breakdown for the evaluation report.
# ---------------------------------------------------------------------------
print("--- Classification output ---")
print(f"  mode:              {rc.mode}")
print(f"  composite_score:   {rc.composite_score:.4f}  "
      f"[expect 0.2–0.7 — freight-heavy, low geo]")
print(f"  geo_component:     {rc.geo_component:.4f}  "
      f"[max(weather=0.2, norm(NDR=3.0)) — expect low]")
print(f"  freight_component: {rc.freight_component:.4f}  "
      f"[news severity 0.85 — expect ~0.85]")
print(f"  base_label:        {rc.base_label}  "
      f"[from composite_score thresholds — expect MEDIUM]")
print(f"  duration_days:     {rc.duration_days}  "
      f"[max(news=30.0, shock=45) = 45.0]")
print(f"  final_label:       {rc.final_label}  "
      f"[45d >= 4d hard floor -> CRITICAL]")
print(f"  escalated:         {rc.escalated}  "
      f"[True — duration pushed MEDIUM -> CRITICAL]")
print()

# ---------------------------------------------------------------------------
# Step 4 — Assertions
# ---------------------------------------------------------------------------
chk(rc.final_label == "CRITICAL",
    f"final_label='CRITICAL' — 45-day duration forces CRITICAL hard floor "
    f"(base was {rc.base_label!r}) (got {rc.final_label!r})")

chk(rc.escalated is True,
    f"escalated=True — label was changed by duration (base={rc.base_label!r}) "
    f"(got {rc.escalated})")

chk(rc.duration_days == 45.0,
    f"duration_days=45.0 — shock_duration_days=45 dominates over news=30 "
    f"(got {rc.duration_days})")

print()
print(f"All pass: {all_pass}")
