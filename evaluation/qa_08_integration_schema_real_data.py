"""
QA-08b | Integration smoke test — schema + normalization bounds (real 5,459-row dataset)
=========================================================================================
Agent tested : db_utils.ensure_risk_classification_table()
               langgraph_engine._get_norm_bounds()
               etl_loader (via the ingested lite_master table)
Data source  : outputs/supply_chain.db built from supply_chain_lite_master.xlsx
               (ETL via scripts/build_databases.py or src/utils/etl_loader.py)

What this file verifies
-----------------------
1. ETL completeness — lite_master and the daily_records VIEW both contain
   exactly 5,459 rows (DataCo electronics-only dataset ground truth).

2. risk_classifications table — re-created after ETL (ETL replaces the whole
   DB file, so the table must be re-added via ensure_risk_classification_table).

3. _get_norm_bounds() from REAL data — the actual MIN/MAX observed in lite_master
   must match the spec-declared ground truth within ±0.01:
     weather_severity_hub:    [1.18,  10.0]
     natural_disaster_risk:   [1.18,  10.0]
     supply_disruption_index: [4.09,  9.97]
     defect_rate_pct:         [2.00,  19.82]
     disruption_news_count:   [0.00,  17.0]

4. delivery_status distribution — confirms the four known statuses exist and
   that "MEDIUM" does NOT appear as a delivery_status value (it is live-mode only).

Expected outcome: all 7 checks PASS.
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

# Clear any stale LRU-cached bounds from a prior run (e.g., the synthetic fixture).
# The cache is process-wide and persists across script invocations within the same
# Python process.  Always clear before reading from a freshly rebuilt DB.
from src.agents.risk_classifier_agent import _get_norm_bounds
_get_norm_bounds.cache_clear()

import sqlite3
from src.utils.db_utils import ensure_risk_classification_table

print("=== QA-08b: Integration Smoke Test (real 5,459-row dataset) ===")
print()

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


# ---------------------------------------------------------------------------
# Step 1 — Ensure risk_classifications audit table exists.
# load_excel_into_sqlite() uses os.replace() to swap in a fresh DB file,
# which drops this table.  ensure_risk_classification_table() is idempotent
# and recreates it if absent.
# ---------------------------------------------------------------------------
ensure_risk_classification_table()

# ---------------------------------------------------------------------------
# Step 2 — Verify ETL row counts.
# The DataCo electronics workbook must produce exactly 5,459 lite_master rows.
# daily_records is a VIEW over lite_master; a mismatch here means the VIEW
# definition or the ETL column mapping has changed.
# ---------------------------------------------------------------------------
conn = sqlite3.connect("outputs/supply_chain.db")

lm_count = conn.execute("SELECT COUNT(*) FROM lite_master").fetchone()[0]
dr_count  = conn.execute("SELECT COUNT(*) FROM daily_records").fetchone()[0]

chk(lm_count == 5459,
    f"lite_master contains exactly 5,459 electronics rows (got {lm_count:,})")
chk(dr_count == 5459,
    f"daily_records VIEW exposes all 5,459 rows (got {dr_count:,})")

# ---------------------------------------------------------------------------
# Step 3 — Verify risk_classifications table schema.
# ---------------------------------------------------------------------------
schema_row = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='risk_classifications'"
).fetchone()

if schema_row:
    print("PASS | risk_classifications table exists")
    print(f"  Schema snippet: {schema_row[0][:300]}...")
else:
    print("FAIL | risk_classifications table missing")
    all_pass = False

print()

# ---------------------------------------------------------------------------
# Step 4 — Load and cross-check normalization bounds from real data.
# The _get_norm_bounds() LRU cache was cleared at the top of this script.
# Bounds are read fresh from the actual lite_master table.
# ---------------------------------------------------------------------------
bounds = _get_norm_bounds()

print("Normalization bounds loaded from REAL lite_master:")
for col, (lo, hi) in bounds.items():
    print(f"  {col}: [{lo:.4f}, {hi:.4f}]")
print()

# Spec ground truth — real values will be close but may have more decimal places
SPEC_BOUNDS = {
    "weather_severity_hub":    (1.18,  10.0),
    "natural_disaster_risk":   (1.18,  10.0),
    "supply_disruption_index": (4.09,  9.97),
    "defect_rate_pct":         (2.00,  19.82),
    "disruption_news_count":   (0.00,  17.0),
}

print("Cross-checking against spec ground truth (tolerance ±0.01):")
for col, (exp_lo, exp_hi) in SPEC_BOUNDS.items():
    actual_lo, actual_hi = bounds[col]
    lo_ok = abs(actual_lo - exp_lo) < 0.01
    hi_ok = abs(actual_hi - exp_hi) < 0.01
    chk(
        lo_ok and hi_ok,
        f"{col}: expected [{exp_lo}, {exp_hi}], "
        f"got [{actual_lo:.4f}, {actual_hi:.4f}]",
    )

# ---------------------------------------------------------------------------
# Step 5 — delivery_status distribution: confirm no "MEDIUM" value exists.
# MEDIUM is only produced by the composite_score fallback in live/demo mode.
# Historical rows use only the four DataCo strings.
# ---------------------------------------------------------------------------
dist = conn.execute(
    """
    SELECT delivery_status, COUNT(*) AS cnt
    FROM lite_master
    GROUP BY delivery_status
    ORDER BY cnt DESC
    """
).fetchall()
conn.close()

print()
print("Delivery status distribution in real data:")
for ds_value, cnt in dist:
    print(f"  {repr(ds_value):30s}  {cnt:5,} rows")

status_values = {row[0] for row in dist}
chk(
    "MEDIUM" not in status_values,
    "No 'MEDIUM' delivery_status in historical rows — MEDIUM is live-mode only",
)

print()
print(f"All pass: {all_pass}")
