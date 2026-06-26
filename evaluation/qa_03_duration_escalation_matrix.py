"""
QA-03 | Duration escalation matrix
====================================
Agent tested : Risk Classifier Agent (Agent 4) — langgraph_engine.py
Function     : _escalate_label(base_label, duration_days)
Data source  : Pure logic — no DB required

What this file verifies
-----------------------
After the base risk label is derived from delivery_status or composite score,
the classifier applies a duration escalation step driven by the longest
disruption duration signal (from news signals or shock_duration_days).

Escalation matrix (locked in spec §5):
  duration_days is None or <= 1  → no change, escalated=False
  duration_days in [2, 3]        → bump one tier up (LOW→MEDIUM, MEDIUM→HIGH,
                                    HIGH→CRITICAL); escalated=True
  duration_days >= 4             → force CRITICAL regardless of base label
                                    (hard floor); escalated=True if label changed

The classifier NEVER lowers a label — a short duration cannot de-escalate
a label that was already high from the composite score.

Tier order (ascending): LOW → MEDIUM → HIGH → CRITICAL

Expected outcome: all 9 cases PASS.
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

from src.agents.risk_classifier_agent import _escalate_label


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


print("=== QA-03: Duration Escalation Matrix ===")
print()

# ---------------------------------------------------------------------------
# Section 1 — Five spec-defined reference examples
# These are the exact scenarios described in the build spec §5.
# ---------------------------------------------------------------------------
print("--- Spec reference examples ---")

SPEC_EXAMPLES = [
    # (base_label, duration_days, expected_final_label, expected_escalated, description)
    ("LOW",      5.0,  "CRITICAL", True,  "5-day port closure: >= 4d hard floor -> CRITICAL"),
    ("HIGH",     1.0,  "HIGH",     False, "1-day delay: <= 1d -> no change, stays HIGH"),
    ("MEDIUM",   2.0,  "HIGH",     True,  "2-day disruption: one tier up MEDIUM -> HIGH"),
    ("CRITICAL", 1.0,  "CRITICAL", False, "CRITICAL + 1d: already max tier, stays CRITICAL"),
    ("LOW",      None, "LOW",      False, "No duration signal: unchanged, stays LOW"),
]

for base, dur, exp_final, exp_escalated, description in SPEC_EXAMPLES:
    final, escalated = _escalate_label(base, dur)
    ok = (final == exp_final) and (escalated == exp_escalated)
    chk(
        ok,
        f"base={base:<8} dur={str(dur)+'d':<6} -> {final:<8} escalated={escalated}"
        f"  (expected {exp_final}/{exp_escalated})  [{description}]",
    )

print()

# ---------------------------------------------------------------------------
# Section 2 — 3-day boundary: exactly one tier escalation
# 3 days falls in the [2,3] range, so it bumps exactly one tier.
# ---------------------------------------------------------------------------
print("--- 3-day single-tier escalation ---")

final, escalated = _escalate_label("LOW", 3.0)
chk(
    final == "MEDIUM" and escalated is True,
    f"LOW + 3d -> {final} escalated={escalated}  (expected MEDIUM/True - one tier up)",
)

print()

# ---------------------------------------------------------------------------
# Section 3 — 4-day hard floor: every base label becomes CRITICAL
# 4 days is the threshold where the hard floor kicks in regardless of
# how benign the base label was.
# ---------------------------------------------------------------------------
print("--- 4-day hard floor (all base labels -> CRITICAL) ---")

for base_label in ("LOW", "MEDIUM", "HIGH"):
    final, escalated = _escalate_label(base_label, 4.0)
    chk(
        final == "CRITICAL",
        f"{base_label:<8} + 4d -> {final}  (expected CRITICAL hard floor)",
    )

print()
print(f"All pass: {all_pass}")
