"""
QA-07 | Backwards-compatible GlobalState property shims
=========================================================
Agent tested : GlobalState (src/agents/state.py) — read-only property shims
Functions    : GlobalState.risk_label (property)
               GlobalState.risk_score_composite (property)
               RiskClassificationResult.critical_flag (field)
Data source  : In-memory only — no DB required

What this file verifies
-----------------------
Agents downstream of the Risk Classifier (Simulation Agent, Mitigation Agent)
were originally written to read `state.risk_label` and `state.risk_score_composite`
directly off the GlobalState object.

After the Risk Classifier was refactored to return a structured
RiskClassificationResult instead of bare fields, two read-only @property shims
were added to GlobalState for zero-friction backwards compatibility:

  GlobalState.risk_label
      → delegates to self.risk_classification.final_label
      → returns None if risk_classification is not set

  GlobalState.risk_score_composite
      → delegates to self.risk_classification.composite_score
      → returns None if risk_classification is not set

Additionally, RiskClassificationResult.critical_flag is verified to be
accessible from the GlobalState, because the Mitigation Agent reads it to
decide whether to fire the Slack webhook (hard business rule):

  if state.risk_classification and state.risk_classification.critical_flag:
      # fire Slack webhook

Expected outcome: all 3 assertions PASS.
"""

import sys
sys.path.insert(0, ".")  # allow imports from project root

from src.agents.state import GlobalState, RiskClassificationResult


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


print("=== QA-07: Backwards-Compatible GlobalState Property Shims ===")
print()

# ---------------------------------------------------------------------------
# Step 1 — Build a RiskClassificationResult that simulates a completed
# Red Sea classification: HIGH base escalated to CRITICAL over 45 days.
# All fields are set to plausible values to ensure the model validates.
# ---------------------------------------------------------------------------
rc = RiskClassificationResult(
    mode="live",
    composite_score=0.72,
    geo_component=0.80,
    supply_component=0.60,
    freight_component=0.85,       # news severity 0.85 drives freight
    defect_component=0.55,
    duration_days=45.0,           # 45-day Red Sea crisis
    base_label="HIGH",            # composite=0.72 → HIGH (>= 0.50, < 0.75)
    final_label="CRITICAL",       # 45d hard floor escalated HIGH → CRITICAL
    escalated=True,
    rag_citations=["Red_Sea_Report.pdf"],
    rationale="45-day duration escalated HIGH → CRITICAL",
    critical_flag=True,           # True because final_label == "CRITICAL"
)

# Wrap the classification result inside a GlobalState (simulating what
# risk_classifier_agent returns after it runs).
g = GlobalState(risk_classification=rc)

# ---------------------------------------------------------------------------
# Step 2 — Assertions
# ---------------------------------------------------------------------------

# Shim 1: risk_label should proxy to risk_classification.final_label
chk(
    g.risk_label == "CRITICAL",
    f"GlobalState.risk_label shim -> 'CRITICAL' via risk_classification.final_label "
    f"(got {g.risk_label!r})",
)

# Shim 2: risk_score_composite should proxy to risk_classification.composite_score
chk(
    abs(g.risk_score_composite - 0.72) < 1e-6,
    f"GlobalState.risk_score_composite shim -> 0.72 via risk_classification.composite_score "
    f"(got {g.risk_score_composite})",
)

# Mitigation Agent Slack rule: critical_flag must be reachable from GlobalState
chk(
    g.risk_classification.critical_flag is True,
    f"critical_flag accessible via state.risk_classification.critical_flag -> True "
    f"(Mitigation Agent Slack webhook trigger) (got {g.risk_classification.critical_flag})",
)

print()
print(f"All pass: {all_pass}")
