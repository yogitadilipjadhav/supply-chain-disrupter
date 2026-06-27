"""
QA-04b | Replay mode — real workbook data (5,459 rows)
=======================================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : risk_classifier_agent(state)
Data source  : outputs/supply_chain.db built from supply_chain_lite_master.xlsx
               (ETL via scripts/build_databases.py or src/utils/etl_loader.py)

What this file verifies
-----------------------
Identical contract to qa_04_replay_mode_synthetic_fixture.py, but runs against
a genuine DataCo order row fetched from the 5,459-row production dataset.

REPLAY mode contract:
  1. Stored composite returned byte-for-byte (delta < 1e-5).
  2. base_label derived from delivery_status, not the stored disruption_event_label.
     "Shipping canceled" → CRITICAL regardless of the composite_score value.
  3. update_risk_label is never called (lite_master is read-only ground truth).
  4. critical_flag is True when final_label is CRITICAL.

The first 'Shipping canceled' row in the real DB is used (order_id=246,
stored_composite=0.4470).  Note: composite=0.4470 < 0.75, which would be HIGH
under the score-threshold path — this demonstrates that delivery_status takes
precedence and always pins 'Shipping canceled' to CRITICAL regardless of score.

Expected outcome: all 6 assertions PASS.
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

import sqlite3
from unittest.mock import patch

from src.agents.risk_classifier_agent import _get_norm_bounds, risk_classifier_agent
from src.agents.state import EventMetadata, GlobalState
from src.utils.db_utils import ensure_risk_classification_table

# Clear any stale LRU-cached bounds left over from a prior synthetic-fixture run.
# The cache is process-wide; if this script is run after the synthetic fixture the
# cached bounds (which match the spec exactly) would differ from the real DB values.
_get_norm_bounds.cache_clear()


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


print("=== QA-04b: Replay Mode (real 5,459-row dataset) ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — Ensure the risk_classifications audit table exists.
# load_excel_into_sqlite() replaces the whole DB file, so this table must be
# re-created each time after running the ETL build script.
# ---------------------------------------------------------------------------
ensure_risk_classification_table()

# ---------------------------------------------------------------------------
# Step 2 — Confirm DB objects are present (tables + daily_records VIEW).
# daily_records is a VIEW defined in etl_loader._create_schema(); its absence
# indicates the ETL has not been run.
# ---------------------------------------------------------------------------
conn = sqlite3.connect("outputs/supply_chain.db")
conn.row_factory = sqlite3.Row

db_objects = [
    r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
]
print(f"DB objects present: {db_objects}")

# ---------------------------------------------------------------------------
# Step 3 — Fetch the first 'Shipping canceled' row from the real dataset.
# ---------------------------------------------------------------------------
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
    print("FAIL | No 'Shipping canceled' row found — run the ETL build first.")
    sys.exit(1)

record = dict(row)
stored_composite = record.get("risk_score_composite")
print(
    f"\nFetched row:  order_id={record.get('order_id')}  "
    f"stored_composite={stored_composite:.4f}  "
    f"delivery_status={record.get('delivery_status')!r}"
)

# ---------------------------------------------------------------------------
# Step 4 — Build GlobalState with the real historical row.
# shock_duration_days=0 and no news_signals so duration escalation is skipped;
# the test focuses purely on replay-mode composite passthrough.
# ---------------------------------------------------------------------------
state = GlobalState(
    event_metadata=EventMetadata(
        disruption_type="test",
        affected_port=str(record.get("port", "")),
        affected_route="test",
        severity=0.5,
        shock_duration_days=0,        # no duration signal → escalation bypassed
        recovery_window_days=30,
        synthetic_ratio=0.0,
    ),
    active_record=record,
    news_signals=[],                  # real historical row has no live news
    live_weather_severity=0.3,
)

# ---------------------------------------------------------------------------
# Step 5 — Run the classifier.
# insert_risk_classification is patched to prevent writing to the test DB;
# update_risk_label is patched AND tracked — it must never be called in replay.
# query_chroma_rag is patched because ChromaDB may not be populated in CI.
# ---------------------------------------------------------------------------
with patch("src.agents.risk_classifier_agent.insert_risk_classification"):
    with patch("src.agents.risk_classifier_agent.update_risk_label") as mock_update:
        with patch("src.agents.risk_classifier_agent.query_chroma_rag", return_value=[]):
            result = risk_classifier_agent(state)

rc = result["risk_classification"]

print()
print("--- Classification output ---")
print(f"  mode:                      {rc.mode}")
print(f"  composite_score:           {rc.composite_score:.6f}  (stored: {stored_composite:.6f})")
print(f"  base_label:                {rc.base_label}")
print(f"  final_label:               {rc.final_label}")
print(f"  escalated:                 {rc.escalated}")
print(f"  duration_days:             {rc.duration_days}")
print(f"  critical_flag:             {rc.critical_flag}")
print(f"  update_risk_label called:  {mock_update.called}")
print()

# ---------------------------------------------------------------------------
# Step 6 — Assertions
# ---------------------------------------------------------------------------
chk(rc.mode == "replay",
    f"mode='replay' — stored_composite ({stored_composite}) is not None (got {rc.mode!r})")

chk(abs(rc.composite_score - stored_composite) < 1e-5,
    f"composite_score == stored {stored_composite:.6f} "
    f"(got {rc.composite_score:.6f}, delta={abs(rc.composite_score - stored_composite):.2e})")

chk(rc.base_label == "CRITICAL",
    f"base_label='CRITICAL' — delivery_status='Shipping canceled' overrides score "
    f"(composite={stored_composite:.4f} would be HIGH under score path) (got {rc.base_label!r})")

chk(rc.final_label == "CRITICAL",
    f"final_label='CRITICAL' (got {rc.final_label!r})")

chk(rc.critical_flag is True,
    f"critical_flag=True when final_label is CRITICAL (got {rc.critical_flag})")

chk(not mock_update.called,
    "update_risk_label NOT called — replay rows are read-only ground truth")

print()
print(f"All pass: {all_pass}")
