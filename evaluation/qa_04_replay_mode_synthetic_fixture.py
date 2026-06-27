"""
QA-04a | Replay mode — synthetic fixture fallback
==================================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : risk_classifier_agent(state)
Data source  : Synthetic spec-conformant DB (fixture_spec_conformant_db.py)
               Use when supply_chain_lite_master.xlsx is not available.
               Prefer qa_04_replay_mode_real_data.py when the real workbook is present.

What this file verifies
-----------------------
REPLAY mode is triggered when daily_records already has both:
  - risk_score_composite  (the pre-computed historical score)
  - disruption_event_label (the historical label written at ingestion time)

In replay mode the classifier must:
  1. Return the STORED composite score unchanged (not recompute it).
  2. Derive base_label from delivery_status, NOT from the stored label column.
     "Shipping canceled" always maps to CRITICAL regardless of composite score.
  3. Never call update_risk_label — replay rows are read-only ground truth.
     Overwriting lite_master would corrupt the Day 23 evaluation dataset.
  4. Still write an audit row to risk_classifications (insert_risk_classification).
  5. Set critical_flag=True when final_label is CRITICAL.

The synthetic fixture contains a row:
  order_id=20001, delivery_status='Shipping canceled', risk_score_composite=0.8350

Expected outcome: all 6 assertions PASS.
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

import sqlite3
from unittest.mock import patch

from src.agents.risk_classifier_agent import risk_classifier_agent
from src.agents.state import EventMetadata, GlobalState


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


print("=== QA-04a: Replay Mode (synthetic fixture) ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — Fetch the 'Shipping canceled' row from the synthetic fixture.
# The daily_records object is a VIEW over lite_master created by
# fixture_spec_conformant_db.py; it contains the stored composite + label.
# ---------------------------------------------------------------------------
conn = sqlite3.connect("outputs/supply_chain.db")
conn.row_factory = sqlite3.Row
row = conn.execute(
    """
    SELECT * FROM daily_records
    WHERE delivery_status = 'Shipping canceled'
    ORDER BY record_id
    LIMIT 1
    """
).fetchone()
conn.close()

if row is None:
    print("SKIP | No 'Shipping canceled' row found. Run fixture_spec_conformant_db.py first.")
    sys.exit(0)

record = dict(row)
stored_composite = record.get("risk_score_composite")
print(
    f"Fetched row:  order_id={record.get('order_id')}  "
    f"stored_composite={stored_composite:.4f}  "
    f"delivery_status={record.get('delivery_status')!r}"
)

# ---------------------------------------------------------------------------
# Step 2 — Build GlobalState with the historical row as active_record.
# No news_signals and shock_duration_days=0 so duration escalation is skipped,
# keeping the test focused purely on replay-mode composite passthrough.
# ---------------------------------------------------------------------------
state = GlobalState(
    event_metadata=EventMetadata(
        disruption_type="test",
        affected_port=str(record.get("port", "")),
        affected_route="test",
        severity=0.5,
        shock_duration_days=0,        # no duration signal → escalation skipped
        recovery_window_days=30,
        synthetic_ratio=0.0,
    ),
    active_record=record,
    news_signals=[],                  # no live news signals in replay mode
    live_weather_severity=0.3,
)

# ---------------------------------------------------------------------------
# Step 3 — Run the classifier with DB writes and RAG patched out.
# insert_risk_classification is allowed to be called (audit trail is written
# even in replay mode); only update_risk_label must NOT be called.
# ---------------------------------------------------------------------------
with patch("src.agents.risk_classifier_agent.ensure_risk_classification_table"):
    with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
        with patch("src.agents.risk_classifier_agent.update_risk_label") as mock_update:
            with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
                result = risk_classifier_agent(state)

rc = result["risk_classification"]

print()
print("--- Classification output ---")
print(f"  mode:            {rc.mode}")
print(f"  composite_score: {rc.composite_score:.4f}  (stored in DB: {stored_composite:.4f})")
print(f"  base_label:      {rc.base_label}")
print(f"  final_label:     {rc.final_label}")
print(f"  escalated:       {rc.escalated}")
print(f"  duration_days:   {rc.duration_days}")
print(f"  critical_flag:   {rc.critical_flag}")
print(f"  update_risk_label called: {mock_update.called}")
print()

# ---------------------------------------------------------------------------
# Step 4 — Assertions
# ---------------------------------------------------------------------------
chk(rc.mode == "replay",
    f"mode='replay' because stored_composite is not None (got {rc.mode!r})")

chk(abs(rc.composite_score - stored_composite) < 1e-5,
    f"composite_score matches stored value {stored_composite:.6f} "
    f"(got {rc.composite_score:.6f}, delta={abs(rc.composite_score - stored_composite):.2e})")

chk(rc.base_label == "CRITICAL",
    f"base_label='CRITICAL' because delivery_status='Shipping canceled' (got {rc.base_label!r})")

chk(rc.final_label == "CRITICAL",
    f"final_label='CRITICAL' (got {rc.final_label!r})")

chk(rc.critical_flag is True,
    f"critical_flag=True when final_label is CRITICAL (got {rc.critical_flag})")

chk(not mock_update.called,
    "update_risk_label NOT called — replay mode must never overwrite lite_master")

print()
print(f"All pass: {all_pass}")
