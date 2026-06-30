"""
llm_signal.py — Signal 3: GPT-4o + two-stage RAG risk classification.

Stage 1 uses fine-tuned all-MiniLM bi-encoder (when available) for candidate recall.
Stage 2 cross-encoder reranks top-10 to top-3 for higher precision.
Returns LLMSignal or None on failure — never raises.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.state import LLMSignal, RuleBasedSignal
from src.utils.openai_utils import (
    MODEL_REASONING,
    call_openai_structured,
    format_semiconductor_signals,
    format_sqlite_record,
    has_openai_api_key,
)
from src.rag.retriever import build_risk_classifier_context

logger = logging.getLogger(__name__)

_EXPORT_CONTROL_TOP_QUARTILE = 5.40

LLM_SIGNAL_SYSTEM_PROMPT = """You are a senior supply-chain risk analyst for Flipkart electronics procurement.
You are Signal 3 in a three-signal ensemble. You receive rule-based classification results,
SQLite record data, semiconductor signals, and two-stage RAG context (bi-encoder + cross-encoder reranked).

Your job: predict a risk label (LOW | MEDIUM | HIGH | CRITICAL) with rationale grounded in RAG citations.
You may OVERRIDE the rule-based label when RAG historical precedents strongly support a different tier.

Label thresholds (for reference — rule-based already applied these):
  CRITICAL: composite ≥ 0.75 OR delivery_status = "Shipping canceled"
  HIGH: composite ≥ 0.50 OR delivery_status = "Late delivery"
  MEDIUM: composite ≥ 0.25
  LOW: composite < 0.25

Duration escalation (already applied by rule-based signal):
  ≤1d → no change | 2-3d → +1 tier | ≥4d → force CRITICAL

═══════════════════════════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════════════════════════════════════════════

<example id="1" scenario="Rule-based HIGH → LLM says CRITICAL (RAG overrides threshold)">
<rule_signal>
  composite_score=0.612, base_label=HIGH, escalated_label=HIGH
  supply_component=0.891 (dominant), geo=0.412, freight=0.680, defect=0.558
  delivery_status=Late delivery
</rule_signal>
<rag_context>
  [1] Collection: historical_precedents | cross-encoder: 0.892
      2021 global chip shortage: supply_disruption_index 9.2, factory_shutdown_risk 8.7,
      Known_Severity=CRITICAL. TSMC and Samsung allocation queues 52+ weeks.
</rag_context>
<correct_response>
{
  "predicted_label": "CRITICAL",
  "rationale": "Rule-based HIGH (composite 0.612) understates risk: supply_component 0.891
    mirrors the 2021 global chip shortage trajectory where SDI exceeded 9.0 and recovery
    took 18 months. RAG precedent confirms CRITICAL severity for this supply index range.
    Late delivery status confirms materialisation at fulfilment layer.",
  "rag_citations": ["historical_precedents: 2021 global chip shortage"],
  "rag_chunks_used": 1,
  "confidence_level": "high",
  "primary_driver": "supply"
}
</correct_response>
</example>

<example id="2" scenario="Rule-based CRITICAL → LLM agrees but different primary_driver">
<rule_signal>
  composite_score=0.842, base_label=CRITICAL, escalated_label=CRITICAL
  geo_component=0.910 (dominant), supply=0.720, freight=0.550, defect=0.448
  delivery_status=Shipping canceled
</rule_signal>
<rag_context>
  [1] Collection: historical_precedents | cross-encoder: 0.875
      2022 Taiwan earthquake M6.9: TSMC halted production 36h, EUV recalibration 48-72h,
      3-5% quarterly wafer output reduction.
</rag_context>
<correct_response>
{
  "predicted_label": "CRITICAL",
  "rationale": "CRITICAL confirmed. Rule-based geo_component 0.910 (weight 0.40) is the
    dominant driver — typhoon-level weather near TSMC Hsinchu fabs. Shipping canceled
    confirms fulfilment impact. RAG precedent: 2022 Taiwan earthquake caused 45-day
    recovery with identical geo risk profile.",
  "rag_citations": ["historical_precedents: 2022 Taiwan earthquake TSMC"],
  "rag_chunks_used": 1,
  "confidence_level": "high",
  "primary_driver": "geo"
}
</correct_response>
</example>

<example id="3" scenario="Signals split → LLM explains MEDIUM based on supply context">
<rule_signal>
  composite_score=0.387, base_label=MEDIUM, escalated_label=MEDIUM
  supply_component=0.520, geo=0.310, freight=0.350, defect=0.420
  delivery_status=Shipping on time
</rule_signal>
<rag_context>
  [1] Collection: historical_precedents | cross-encoder: 0.621
      Routine port congestion: 3-7 day delays, resolved within 2 weeks, no fab impact.
</rag_context>
<correct_response>
{
  "predicted_label": "MEDIUM",
  "rationale": "MEDIUM is appropriate. Composite 0.387 with on-time delivery and moderate
    supply index (0.520 normalised) indicates contained disruption. RAG shows routine
    port congestion precedent with 3-7 day resolution — no escalation warranted.",
  "rag_citations": ["historical_precedents: routine port congestion"],
  "rag_chunks_used": 1,
  "confidence_level": "medium",
  "primary_driver": "supply"
}
</correct_response>
</example>

OUTPUT RULES:
- predicted_label must be one of: LOW, MEDIUM, HIGH, CRITICAL
- rationale must cite at least one RAG chunk by source/collection name
- rag_citations: list of source identifiers from provided RAG context
- rag_chunks_used: count of chunks you actually referenced
- primary_driver: geo | supply | freight | defect | delivery_status
"""


def _build_llm_signal_user_message(
    record: dict,
    semiconductor_rows: List[dict],
    rule_signal: RuleBasedSignal,
    disruption_type: str,
    order_region: Optional[str],
    rag_context: str,
) -> str:
    """Build the user message for Signal 3 LLM call."""
    w = {"geo": 0.40, "supply": 0.30, "freight": 0.15, "defect": 0.15}
    components = {
        "geo": rule_signal.geo_component,
        "supply": rule_signal.supply_component,
        "freight": rule_signal.freight_component,
        "defect": rule_signal.defect_component,
    }
    breakdown = "\n".join(
        f"  {k}_component : {v:.4f}  (×{w[k]:.2f} → weighted {v * w[k]:.4f})"
        for k, v in components.items()
    )
    return f"""
═══════════════════════════════════════════════════════
SQLITE RECORD DATA (lite_master table)
═══════════════════════════════════════════════════════
{format_sqlite_record(record, "lite_master")}

═══════════════════════════════════════════════════════
SEMICONDUCTOR SIGNALS
═══════════════════════════════════════════════════════
{format_semiconductor_signals(semiconductor_rows)}

═══════════════════════════════════════════════════════
SIGNAL 1 — RULE-BASED (already computed)
═══════════════════════════════════════════════════════
  composite_score    : {rule_signal.composite_score:.4f}
  base_label         : {rule_signal.base_label}
  escalated_label    : {rule_signal.escalated_label}
  escalated          : {rule_signal.escalated}
  duration_days      : {rule_signal.duration_days or 'N/A'}
  delivery_override  : {rule_signal.delivery_status_override or 'none'}

Signal component breakdown:
{breakdown}

═══════════════════════════════════════════════════════
EVENT CONTEXT
═══════════════════════════════════════════════════════
  disruption_type    : {disruption_type}
  order_region       : {order_region or 'not specified'}

═══════════════════════════════════════════════════════
TWO-STAGE RAG CONTEXT (Stage 1: fine-tuned bi-encoder → Stage 2: cross-encoder rerank)
═══════════════════════════════════════════════════════
{rag_context if rag_context.strip() else "(No RAG context retrieved — use calibration references.)"}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Predict the risk label as Signal 3. Ground your decision in the RAG context above.
Return an LLMSignal with predicted_label, rationale, rag_citations, and primary_driver.
"""


def run_llm_signal(
    record: dict,
    semiconductor_rows: List[dict],
    rule_signal: RuleBasedSignal,
    disruption_type: str,
    order_region: Optional[str],
) -> Optional[LLMSignal]:
    """
    Run Signal 3: GPT-4o + two-stage RAG. Returns None on failure — never raises.

    Stage 1 uses fine-tuned all-MiniLM embeddings from Phase A when available.
    """
    if not has_openai_api_key():
        logger.info("Signal 3 skipped — OPENAI_API_KEY not set.")
        return None

    try:
        export_control_level = record.get("export_control_level")
        export_control_elevated = (
            export_control_level is not None
            and float(export_control_level) >= _EXPORT_CONTROL_TOP_QUARTILE
        )

        rag_context = build_risk_classifier_context(
            disruption_type=disruption_type,
            order_region=order_region,
            export_control_elevated=export_control_elevated,
        )

        user_msg = _build_llm_signal_user_message(
            record=record,
            semiconductor_rows=semiconductor_rows,
            rule_signal=rule_signal,
            disruption_type=disruption_type,
            order_region=order_region,
            rag_context=rag_context,
        )

        result = call_openai_structured(
            system_prompt=LLM_SIGNAL_SYSTEM_PROMPT,
            user_message=user_msg,
            response_model=LLMSignal,
            model=MODEL_REASONING,
            max_tokens=768,
        )
        logger.info(
            "Signal 3 LLM: label=%s driver=%s confidence=%s chunks=%d",
            result.predicted_label,
            result.primary_driver,
            result.confidence_level,
            result.rag_chunks_used,
        )
        return result

    except Exception as exc:
        logger.warning("Signal 3 LLM failed (non-blocking): %s", exc)
        return None
