"""
QA-05 | Live mode — Taiwan earthquake scenario
===============================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : risk_classifier_agent(state)
Data source  : Fully synthetic record (no DB row); real norm bounds from lite_master.

What this file verifies
-----------------------
LIVE mode is triggered when the active_record has NO stored risk_score_composite
(i.e., it is None or absent).  This happens for:
  - Demo/scenario-injected orders submitted via the Scenario Analyzer UI
  - New orders that have not yet been classified

In live mode the classifier must:
  1. Recompute composite from scratch using the formula:
       composite = 0.4*geo + 0.3*supply + 0.15*freight + 0.15*defect
     where:
       geo    = max(live_weather_severity, norm(natural_disaster_risk))
       supply = norm(supply_disruption_index)
       freight = max(news_signal.severity)  [news folds into freight]
       defect = norm(defect_rate_pct)
  2. Derive base_label from composite_score thresholds (delivery_status=None).
  3. Apply duration escalation from news_signal.expected_duration_days and
     event_metadata.shock_duration_days (the maximum of the two is used).
  4. Write the result to risk_classifications (patched here to avoid DB side-effects).
  5. Call update_risk_label to persist the computed score to lite_master.

Scenario: 2024 Taiwan earthquake (modelled on the April 2024 Hualien M7.4 event)
  - TSMC fabs offline; semiconductor supply severely disrupted.
  - weather_severity=0.95, natural_disaster_risk=9.8, SDI=9.5.
  - Two news signals: severity 0.92 (with 6-day duration) and 0.85 (no duration).
  - shock_duration_days=6 (from EventMetadata).
  - Expected: composite >> 0.75 → CRITICAL even before duration escalation.

Expected outcome: all 4 assertions PASS.
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


print("=== QA-05: Live Mode — Taiwan Earthquake Scenario ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — Build GlobalState with a fully synthetic earthquake record.
# risk_score_composite=None signals live mode to the classifier.
# delivery_status=None means base_label will come from the composite score.
# ---------------------------------------------------------------------------
state = GlobalState(
    event_metadata=EventMetadata(
        disruption_type="earthquake",
        affected_port="Hsinchu",
        affected_route="Hsinchu to Singapore",
        severity=0.95,
        shock_duration_days=6,        # 6-day fab shutdown from scenario metadata
        recovery_window_days=90,
        synthetic_ratio=1.0,          # 100% synthetic — no real DataCo row backing it
    ),
    active_record={
        "order_id": None,
        "order_date": "2024-04-03",   # Hualien earthquake date
        "event_date": "2024-04-03",
        "port": "Eastern Asia",
        "sku": "CHIP_AP",
        "delivery_status": None,             # None → use composite score for base_label
        "risk_score_composite": None,        # None → triggers live mode
        "disruption_event_label": None,
        "supply_disruption_index": 9.5,      # near-maximum SDI (scale 4.09–9.97)
        "defect_rate_pct": 15.0,             # elevated defect rate during disruption
        "natural_disaster_risk": 9.8,        # near-maximum natural disaster risk (1.18–10.0)
        "export_control_level": 3.0,
        "order_region": "Eastern Asia",
        "year": 2024,
        "inventory_level": 500,
        "incoming_supply": 100,
        "lead_time_days": 45,
        "chip_risk": 0.8,
        "supplier_risk": 0.6,
    },
    news_signals=[
        NewsRiskSignal(
            source_id="eq-001",
            category="earthquake",
            severity=0.92,            # high-severity signal drives freight component
            summary="Magnitude 7.4 earthquake strikes Hualien, TSMC fabs offline for estimated 6 days",
            signal_tags=["earthquake", "TSMC", "semiconductor"],
            expected_duration_days=6.0,   # duration extracted from this news signal
        ),
        NewsRiskSignal(
            source_id="eq-002",
            category="earthquake",
            severity=0.85,            # second signal — freight = max(0.92, 0.85) = 0.92
            summary="Chipset supply expected to drop 40% for 4-6 weeks",
            signal_tags=["chip shortage"],
            expected_duration_days=None,  # no specific duration in this signal
        ),
    ],
    live_weather_severity=0.95,           # severe ground shaking = very high weather severity
)

# ---------------------------------------------------------------------------
# Step 2 — Run the classifier with all DB writes and RAG patched out.
# update_risk_label IS expected to be called in live mode (it persists the
# computed score), but we patch it to avoid test DB side-effects.
# ---------------------------------------------------------------------------
with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
    with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
        with patch("src.agents.risk_classifier_agent.update_risk_label"):
            with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                result = risk_classifier_agent(state)

rc = result["risk_classification"]

# ---------------------------------------------------------------------------
# Step 3 — Print the full breakdown for the evaluation report.
# ---------------------------------------------------------------------------
print("--- Classification output ---")
print(f"  mode:              {rc.mode}")
print(f"  composite_score:   {rc.composite_score:.4f}")
print(f"  geo_component:     {rc.geo_component:.4f}  "
      f"[max(weather=0.95, norm(NDR=9.8))]")
print(f"  supply_component:  {rc.supply_component:.4f}  "
      f"[norm(SDI=9.5)]")
print(f"  freight_component: {rc.freight_component:.4f}  "
      f"[max(news severity) = max(0.92, 0.85)]")
print(f"  defect_component:  {rc.defect_component:.4f}  "
      f"[norm(defect_rate=15.0)]")
print(f"  duration_days:     {rc.duration_days}  "
      f"[max(news_dur=6.0, shock_dur=6) = 6.0]")
print(f"  base_label:        {rc.base_label}  "
      f"[composite >= 0.75 -> CRITICAL, no delivery_status]")
print(f"  final_label:       {rc.final_label}")
print(f"  escalated:         {rc.escalated}  "
      f"[False — already CRITICAL before duration check]")
print(f"  critical_flag:     {rc.critical_flag}")
print(f"  rationale:         {rc.rationale[:100]}...")
print()

# ---------------------------------------------------------------------------
# Step 4 — Assertions
# ---------------------------------------------------------------------------
chk(rc.mode == "live",
    f"mode='live' because risk_score_composite was None (got {rc.mode!r})")

chk(rc.final_label == "CRITICAL",
    f"final_label='CRITICAL' — extreme geo+supply inputs drive composite >> 0.75 "
    f"(composite={rc.composite_score:.4f}) (got {rc.final_label!r})")

chk(rc.critical_flag is True,
    f"critical_flag=True when final_label is CRITICAL (got {rc.critical_flag})")

chk(rc.duration_days == 6.0,
    f"duration_days=6.0 — max(news eq-001=6.0, shock_duration=6) (got {rc.duration_days})")

print()
print(f"All pass: {all_pass}")
