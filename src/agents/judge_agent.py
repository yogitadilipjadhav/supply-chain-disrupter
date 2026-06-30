"""
judge_agent.py — LLM-as-Judge: meta-reasoning arbiter for the three-signal ensemble.

Receives all three signal outputs and produces a final_label + verdict_type.
When signals DISAGREE, populates disagreement_explanation with root-cause analysis.
Returns JudgeVerdict or None on failure — never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.state import (
    DistilBERTSignal,
    JudgeVerdict,
    LLMSignal,
    RuleBasedSignal,
)
from src.utils.openai_utils import (
    MODEL_REASONING,
    call_openai_structured,
    format_semiconductor_signals,
    format_sqlite_record,
    has_openai_api_key,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are the LLM-as-Judge in a three-signal supply-chain risk ensemble for Flipkart.
You receive outputs from three independent signals and must produce a final_label and verdict_type.

SIGNAL 1 — Rule-based: deterministic formula (0.4×geo + 0.3×supply + 0.15×freight + 0.15×defect)
           + delivery_status overrides + duration escalation. Always auditable.
SIGNAL 2 — DistilBERT: fine-tuned 4-class classifier on 5,459 Flipkart electronics rows (~20ms CPU).
SIGNAL 3 — GPT-4o + two-stage RAG: semantic reasoning with historical precedent grounding.

Your job:
  1. Determine final_label (LOW | MEDIUM | HIGH | CRITICAL)
  2. Assign verdict_type explaining HOW you reached the decision
  3. When signals disagree, explain WHICH signal failed and WHY in disagreement_explanation

verdict_type options:
  unanimous           — all 3 agree
  majority_rule       — 2 of 3 agree, follow majority
  override_distilbert — rules + LLM agree vs DistilBERT; explain why DistilBERT was wrong
  override_llm        — rules + DistilBERT agree vs LLM; explain over-retrieval
  defer_to_rules      — 3-way split; locked formula is tiebreaker

HARD RULES:
  - "Shipping canceled" delivery_status MUST result in final_label=CRITICAL
  - final_critical_flag = (final_label == "CRITICAL")
  - Never lower a label below what delivery_status override requires

═══════════════════════════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════════════════════════════════════════════

<example id="1" scenario="Unanimous — all three agree CRITICAL">
<signals>
  rule_signal.escalated_label=CRITICAL (composite=0.842, delivery=Shipping canceled)
  distilbert_signal.predicted_label=CRITICAL (confidence=0.91)
  llm_signal.predicted_label=CRITICAL (primary_driver=geo)
</signals>
<correct_response>
{
  "final_label": "CRITICAL",
  "verdict_type": "unanimous",
  "reasoning": "All three signals agree on CRITICAL. Rule-based composite 0.842 with
    Shipping canceled override, DistilBERT 91% confidence CRITICAL, and LLM Signal 3
    confirming geo-driven CRITICAL from TSMC earthquake RAG precedent. High confidence
    unanimous verdict — no signal override needed.",
  "signals_agreed": true,
  "disagreement_explanation": null,
  "final_critical_flag": true
}
</correct_response>
</example>

<example id="2" scenario="override_distilbert — rules + LLM agree, DistilBERT wrong">
<signals>
  rule_signal.escalated_label=CRITICAL (delivery=Shipping canceled, composite=0.447)
  distilbert_signal.predicted_label=HIGH (confidence=0.72)
  llm_signal.predicted_label=CRITICAL (primary_driver=delivery_status)
</signals>
<correct_response>
{
  "final_label": "CRITICAL",
  "verdict_type": "override_distilbert",
  "reasoning": "CRITICAL is correct via delivery_status override rule. Rule-based Signal 1
    and LLM Signal 3 both identify CRITICAL. DistilBERT predicted HIGH (72% confidence)
    because its training distribution under-weighted Shipping-canceled rows with moderate
    composite scores — a known train/serve gap for delivery override cases.",
  "signals_agreed": false,
  "disagreement_explanation": "DistilBERT predicted HIGH because it learned the training
    distribution where Shipping-canceled rows in 2015-2018 data had moderate risk scores.
    The rule-based system correctly identifies this as CRITICAL via the delivery_status
    override rule. DistilBERT's training data lacked sufficient 2021-2024 disruption events.",
  "final_critical_flag": true
}
</correct_response>
</example>

<example id="3" scenario="defer_to_rules — 3-way split">
<signals>
  rule_signal.escalated_label=HIGH (composite=0.563)
  distilbert_signal.predicted_label=MEDIUM (confidence=0.68)
  llm_signal.predicted_label=CRITICAL (primary_driver=supply)
</signals>
<correct_response>
{
  "final_label": "HIGH",
  "verdict_type": "defer_to_rules",
  "reasoning": "Three-way split with no majority. Rule-based HIGH (composite 0.563,
    Late delivery override) is the most auditable signal and serves as tiebreaker.
    LLM CRITICAL appears driven by RAG over-retrieval of 2021 chip shortage precedent
    not matched by current SDI. DistilBERT MEDIUM reflects moderate training distribution.",
  "signals_agreed": false,
  "disagreement_explanation": "LLM Signal 3 over-retrieved 2021 chip shortage RAG context
    and escalated to CRITICAL despite composite 0.563. DistilBERT correctly identified
    MEDIUM from feature distribution. Rule-based HIGH with Late delivery override is
    the most defensible label — deferring to Signal 1 as tiebreaker.",
  "final_critical_flag": false
}
</correct_response>
</example>
"""


def _build_judge_user_message(
    rule_signal: RuleBasedSignal,
    distilbert_signal: DistilBERTSignal,
    llm_signal: Optional[LLMSignal],
    record: dict,
    semiconductor_rows: List[dict],
) -> str:
    """Format all three signals into a structured judge user message."""
    llm_block = "  (Signal 3 not available — LLM call failed or skipped)"
    if llm_signal is not None:
        llm_block = (
            f"  predicted_label     : {llm_signal.predicted_label}\n"
            f"  primary_driver      : {llm_signal.primary_driver}\n"
            f"  confidence_level    : {llm_signal.confidence_level}\n"
            f"  rag_chunks_used     : {llm_signal.rag_chunks_used}\n"
            f"  rationale           : {llm_signal.rationale[:400]}"
        )

    return f"""
═══════════════════════════════════════════════════════
SQLITE RECORD
═══════════════════════════════════════════════════════
{format_sqlite_record(record, "lite_master")}

═══════════════════════════════════════════════════════
SEMICONDUCTOR SIGNALS
═══════════════════════════════════════════════════════
{format_semiconductor_signals(semiconductor_rows)}

═══════════════════════════════════════════════════════
SIGNAL 1 — RULE-BASED
═══════════════════════════════════════════════════════
  composite_score     : {rule_signal.composite_score:.4f}
  base_label          : {rule_signal.base_label}
  escalated_label     : {rule_signal.escalated_label}
  escalated           : {rule_signal.escalated}
  duration_days       : {rule_signal.duration_days or 'N/A'}
  delivery_override   : {rule_signal.delivery_status_override or 'none'}
  geo_component       : {rule_signal.geo_component:.4f}
  supply_component    : {rule_signal.supply_component:.4f}
  freight_component   : {rule_signal.freight_component:.4f}
  defect_component    : {rule_signal.defect_component:.4f}

═══════════════════════════════════════════════════════
SIGNAL 2 — DISTILBERT
═══════════════════════════════════════════════════════
  predicted_label     : {distilbert_signal.predicted_label}
  confidence          : {distilbert_signal.confidence:.4f}
  model_source        : {distilbert_signal.model_source}
  prob_distribution   : {distilbert_signal.probability_distribution}
  inference_ms        : {distilbert_signal.inference_ms}

═══════════════════════════════════════════════════════
SIGNAL 3 — GPT-4o + TWO-STAGE RAG
═══════════════════════════════════════════════════════
{llm_block}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Produce a JudgeVerdict with final_label, verdict_type, reasoning, and
disagreement_explanation (required when signals_agreed=False).
"""


def run_judge(
    rule_signal: RuleBasedSignal,
    distilbert_signal: DistilBERTSignal,
    llm_signal: Optional[LLMSignal],
    record: dict,
    semiconductor_rows: List[dict],
) -> Optional[JudgeVerdict]:
    """
    Run LLM-as-Judge over all three signals. Returns None on failure — never raises.

    When signals disagree, disagreement_explanation identifies which signal failed and why.
    """
    if not has_openai_api_key():
        logger.info("Judge skipped — OPENAI_API_KEY not set.")
        return None

    try:
        user_msg = _build_judge_user_message(
            rule_signal=rule_signal,
            distilbert_signal=distilbert_signal,
            llm_signal=llm_signal,
            record=record,
            semiconductor_rows=semiconductor_rows,
        )

        verdict = call_openai_structured(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=user_msg,
            response_model=JudgeVerdict,
            model=MODEL_REASONING,
            max_tokens=768,
        )

        # Enforce hard rule: Shipping canceled → CRITICAL
        delivery = record.get("delivery_status", "")
        if delivery and str(delivery).strip() == "Shipping canceled":
            if verdict.final_label != "CRITICAL":
                logger.warning(
                    "Judge returned %s for Shipping canceled — forcing CRITICAL",
                    verdict.final_label,
                )
                verdict = verdict.model_copy(
                    update={
                        "final_label": "CRITICAL",
                        "final_critical_flag": True,
                        "verdict_type": "defer_to_rules",
                        "disagreement_explanation": (
                            "Judge label overridden: Shipping canceled delivery_status "
                            "hard rule requires CRITICAL regardless of judge output."
                        ),
                    }
                )

        logger.info(
            "Judge verdict: label=%s type=%s agreed=%s",
            verdict.final_label,
            verdict.verdict_type,
            verdict.signals_agreed,
        )
        return verdict

    except Exception as exc:
        logger.warning("Judge failed (non-blocking): %s", exc)
        return None
