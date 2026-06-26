"""
QA-08a | Integration smoke test — schema + normalization bounds (synthetic fixture)
====================================================================================
Agent tested : db_utils.ensure_risk_classification_table()
               langgraph_engine._get_norm_bounds()
Data source  : Synthetic spec-conformant DB (fixture_spec_conformant_db.py)
               Use when supply_chain_lite_master.xlsx is not available.
               Prefer qa_08_integration_schema_real_data.py when the real workbook is present.

What this file verifies
-----------------------
1. risk_classifications table — exists and has the correct schema.
   This table is used by insert_risk_classification() to write audit rows
   for every classification run (both replay and live modes).

2. _get_norm_bounds() — reads MIN/MAX from lite_master and caches them.
   All four formula components (geo, supply, freight, defect) are normalized
   using these bounds before entering the composite score formula.
   The bounds must match the spec ground truth within ±0.01:
     weather_severity_hub:    [1.18, 10.0]
     natural_disaster_risk:   [1.18, 10.0]
     supply_disruption_index: [4.09, 9.97]
     defect_rate_pct:         [2.00, 19.82]
     disruption_news_count:   [0.00, 17.0]

Expected outcome: all 6 checks PASS.
"""

import sys
import os
sys.path.insert(0, ".")  # allow imports from project root

# Clear any stale LRU-cached bounds from a prior run so we always read live from DB.
from src.agents.risk_classifier_agent import _get_norm_bounds
_get_norm_bounds.cache_clear()

import sqlite3
from src.utils.db_utils import ensure_risk_classification_table

print("=== QA-08a: Integration Smoke Test (synthetic fixture) ===")
print()

# ---------------------------------------------------------------------------
# Guard — skip gracefully if the DB doesn't exist at all
# ---------------------------------------------------------------------------
if not os.path.exists("outputs/supply_chain.db"):
    print("SKIP | outputs/supply_chain.db not found. "
          "Run fixture_spec_conformant_db.py first.")
    sys.exit(0)

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
# Step 1 — Ensure risk_classifications table exists.
# ensure_risk_classification_table() is idempotent (CREATE TABLE IF NOT EXISTS).
# ---------------------------------------------------------------------------
ensure_risk_classification_table()

conn = sqlite3.connect("outputs/supply_chain.db")
schema_row = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='risk_classifications'"
).fetchone()
conn.close()

if schema_row:
    print("PASS | risk_classifications table exists")
    print(f"  Schema snippet: {schema_row[0][:300]}...")
else:
    print("FAIL | risk_classifications table is missing")
    all_pass = False

print()

# ---------------------------------------------------------------------------
# Step 2 — Load normalization bounds from lite_master and cross-check vs spec.
# The spec ground truth values are derived from the actual 5,459-row DataCo
# workbook; the synthetic fixture seeds lite_master to exactly reproduce them.
# ---------------------------------------------------------------------------
bounds = _get_norm_bounds()

print("Normalization bounds loaded from lite_master:")
for col, (lo, hi) in bounds.items():
    print(f"  {col}: [{lo:.4f}, {hi:.4f}]")
print()

# Spec ground truth — tolerance ±0.01 to accommodate floating-point rounding
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

print()
print(f"All pass: {all_pass}")
