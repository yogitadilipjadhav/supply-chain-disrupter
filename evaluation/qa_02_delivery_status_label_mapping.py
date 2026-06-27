"""
QA-02 | delivery_status → risk label mapping
=============================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : _base_label_from_delivery_status(delivery_status, composite_score)
Data source  : Pure logic — no DB required

What this file verifies
-----------------------
The Risk Classifier derives a base risk label from the order's delivery_status
field before applying any duration escalation.  There are two code paths:

  Path A — Exact delivery_status string match (DataCo dataset ground truth):
    "Shipping canceled"  → CRITICAL   (order already in crisis state)
    "Late delivery"      → HIGH        (delay confirmed)
    "Advance shipping"   → LOW         (ahead of schedule, lower risk)
    "Shipping on time"   → LOW         (nominal state)
    NOTE: "MEDIUM" never appears as a delivery_status value; it only comes
    from Path B's composite_score thresholds in live/demo mode.

  Path B — Composite score threshold fallback (delivery_status is None):
    score >= 0.75  → CRITICAL
    score >= 0.50  → HIGH
    score >= 0.25  → MEDIUM
    score <  0.25  → LOW

Spec reference: Build Spec §4 "Base Label Derivation"

Expected outcome: all 12 cases PASS (4 exact strings + 8 score thresholds).
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

from src.agents.risk_classifier_agent import _base_label_from_delivery_status


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
# Path A — exact delivery_status strings from the DataCo dataset
# The composite_score (0.5) is intentionally mid-range so it does NOT
# interfere; these cases must be driven entirely by the status string.
# ---------------------------------------------------------------------------
print("=== QA-02: delivery_status -> label mapping ===")
print()
print("--- Path A: exact delivery_status strings ---")

DELIVERY_STATUS_CASES = [
    # (delivery_status_string,  expected_label)
    ("Shipping canceled", "CRITICAL"),  # cancelled order is always a supply crisis
    ("Late delivery",     "HIGH"),      # confirmed delay = elevated risk
    ("Advance shipping",  "LOW"),       # early shipment = lower risk signal
    ("Shipping on time",  "LOW"),       # nominal on-time = low risk
]

for status_str, expected_label in DELIVERY_STATUS_CASES:
    actual = _base_label_from_delivery_status(status_str, composite_score=0.5)
    chk(
        actual == expected_label,
        f"delivery_status={repr(status_str):25s} -> {actual} (expected {expected_label})",
    )


# ---------------------------------------------------------------------------
# Path B — score threshold fallback when delivery_status is None
# Covers both interior values and exact boundary conditions (>=, not >).
# ---------------------------------------------------------------------------
print()
print("--- Path B: composite_score threshold fallback (delivery_status=None) ---")

SCORE_THRESHOLD_CASES = [
    # (composite_score,  expected_label,  note)
    (0.80,  "CRITICAL", "above CRITICAL threshold"),
    (0.60,  "HIGH",     "above HIGH threshold, below CRITICAL"),
    (0.35,  "MEDIUM",   "above MEDIUM threshold, below HIGH"),
    (0.10,  "LOW",      "below MEDIUM threshold"),
    (0.75,  "CRITICAL", "exactly at CRITICAL boundary (>=0.75)"),
    (0.50,  "HIGH",     "exactly at HIGH boundary (>=0.50)"),
    (0.25,  "MEDIUM",   "exactly at MEDIUM boundary (>=0.25)"),
    (0.249, "LOW",      "just below MEDIUM boundary (<0.25)"),
]

for score, expected_label, note in SCORE_THRESHOLD_CASES:
    actual = _base_label_from_delivery_status(None, score)
    chk(
        actual == expected_label,
        f"score={score:<6}  -> {actual:<8}  expected {expected_label:<8}  [{note}]",
    )

print()
print(f"All pass: {all_pass}")
