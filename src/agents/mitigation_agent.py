"""
mitigation_agent.py — L7 Mitigation Recommendation Agent (gpt-4o).

Generates ranked mitigation actions with India sourcing recommendations and RAG citations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.agents.state import GlobalState, MitigationAction, MitigationLLMOutput
from src.utils.db_utils import execute_query, insert_mitigation_action
from src.utils.openai_utils import (
    MODEL_REASONING,
    call_openai_structured,
    format_semiconductor_signals,
    format_sqlite_record,
    has_openai_api_key,
)
from src.rag.retriever import build_mitigation_context

logger = logging.getLogger(__name__)

_EXPORT_CONTROL_TOP_QUARTILE = 5.40

MITIGATION_SYSTEM_PROMPT = """You are Flipkart's senior supply-chain mitigation strategist for electronics procurement.
Your recommendations are executed by the war-room in Bengaluru within hours of a CRITICAL alert.

INDIA SOURCING PRIORITY HIERARCHY:
TIER 1 — ISM / PLI eligible:
  Tata Electronics, Dixon Technologies, Foxconn India (Chennai), Wistron India (Bengaluru),
  Kaynes Technology (Mysuru), Vedanta-Foxconn (Dholera, Gujarat)
TIER 2 — ASEAN: Penang Malaysia, Vietnam (Samsung Display), Thailand
TIER 3 — Global OSAT: ASE Group Taiwan, Powertech Technology

HARD BUSINESS RULES (never violate):
  urgency: CRITICAL→IMMEDIATE, HIGH→HIGH, MEDIUM→MEDIUM, LOW→LOW
  ranked_actions: 3-5 items, format '[TYPE] <specific action>'
  rag_citations: at least 1 source from RAG context
  india_sourcing_recommendations: at least 1 named Tier-1 or Tier-2 entity

FEW-SHOT EXAMPLES:

<example id="1" scenario="CRITICAL earthquake near TSMC">
<correct_response>
{
  "summary": "CRITICAL earthquake near TSMC Hsinchu collapsed advanced logic chip availability with 73% stockout probability. Flipkart must activate emergency safety-stock cover within 24 hours.",
  "ranked_actions": [
    "[INVENTORY] Raise DRAM and advanced logic SKU safety stock to 45-day cover across Bengaluru, Mumbai, and Delhi FCs.",
    "[SOURCING] Place emergency allocation hold with Samsung Korea and ASE Group Taiwan OSAT packaging.",
    "[INDIA-SOURCING] Engage Tata Electronics (Hosur) for ISM-eligible OSAT capacity; Kaynes Technology (Mysuru) for PCB EMS buffer.",
    "[ROUTING] Divert in-transit shipments via Colombo hub to avoid Taiwan Strait corridor risk.",
    "[FINANCIAL] Pre-authorise air-freight budget for critical DRAM components — +20-25% per-unit premium justified."
  ],
  "cost_estimate": "HIGH: Air-freight adds ~$20-25/unit premium; emergency OSAT re-qualification ~$150-200K one-time.",
  "urgency": "IMMEDIATE",
  "rag_citations": ["Source: historical_precedents — 2022 TSMC earthquake: ASE Group OSAT reduced lead-time from 8 to 5 weeks."],
  "india_sourcing_recommendations": ["Tata Electronics (Hosur) — ISM semiconductor packaging within 72 hours.", "Kaynes Technology (Mysuru) — PLI EMS PCB assembly within 2-week lead time."]
}
</correct_response>
</example>

<example id="2" scenario="HIGH Red Sea routing disruption">
<correct_response>
{
  "summary": "HIGH Red Sea routing disruption extended Asia-Europe transit 10-14 days with 44.8% stockout probability. Flipkart Western Europe-sourced display panels require Cape of Good Hope rerouting.",
  "ranked_actions": [
    "[ROUTING] Activate Cape of Good Hope rerouting via Mauritius hub for Asia-Europe display panel shipments.",
    "[INVENTORY] Pre-position 30-day safety stock for Lenovo ThinkPad display panel SKUs at Bengaluru and Delhi FCs.",
    "[SOURCING] Engage alternate supplier for display panels; evaluate Vietnam Samsung Display within 4-week lead time.",
    "[INDIA-SOURCING] Contact Dixon Technologies (Tirupati PLI unit) for domestic flat-panel display sub-assembly."
  ],
  "cost_estimate": "MEDIUM: Cape rerouting adds ~8-12 days transit and ~$3-5/unit freight premium.",
  "urgency": "HIGH",
  "rag_citations": ["Source: historical_precedents — RED_SEA_CRISIS profile: avg lead-time variance 15.3 days across 234 orders."],
  "india_sourcing_recommendations": ["Dixon Technologies (Tirupati, AP — PLI Electronics) — display module substitution within 4-week ramp."]
}
</correct_response>
</example>"""


def _fetch_semiconductor_rows(year: Optional[Any]) -> List[dict]:
    """Fetch semiconductor_signals rows for the record year."""
    if year is None:
        return []
    try:
        rows = execute_query(
            "SELECT year, company, supply_disruption_index, export_control_level, "
            "known_disruption_event, known_severity FROM semiconductor_signals "
            "WHERE year = ? ORDER BY supply_disruption_index DESC LIMIT 5",
            (int(year),),
        )
        return [dict(r) for r in rows]
    except Exception:
        return []


def _build_mitigation_user_message(
    risk,
    event_metadata,
    record: dict,
    semiconductor_rows: List[dict],
    forecast_drop_pct: Optional[float],
    stockout_pct: Optional[float],
    alt_route: Optional[str],
    export_control_raw: Optional[float],
    rag_context: str,
) -> str:
    """Build structured user message for Mitigation Agent LLM call."""
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
RISK CLASSIFICATION (from Agent L4)
═══════════════════════════════════════════════════════
  risk_label                  : {risk.final_label}
  composite_score             : {risk.composite_score:.4f}
  geo_component               : {risk.geo_component:.4f}  (×0.40)
  supply_component            : {risk.supply_component:.4f}  (×0.30)
  freight_component           : {risk.freight_component:.4f}  (×0.15)
  defect_component            : {risk.defect_component:.4f}  (×0.15)
  escalated                   : {risk.escalated}  (duration={risk.duration_days}d)
  critical_flag               : {risk.critical_flag}
  llm_primary_driver          : {risk.llm_primary_driver or 'N/A'}
  llm_enhanced_rationale      : {risk.llm_enhanced_rationale or risk.rationale}
  rag_citations (L4)          : {'; '.join(risk.rag_citations) if risk.rag_citations else 'none'}

═══════════════════════════════════════════════════════
EVENT CONTEXT
═══════════════════════════════════════════════════════
  disruption_type             : {event_metadata.disruption_type}
  affected_port               : {event_metadata.affected_port}
  affected_route              : {event_metadata.affected_route}
  shock_duration_days         : {event_metadata.shock_duration_days}
  export_control_level (raw)  : {export_control_raw or 'N/A'}

═══════════════════════════════════════════════════════
DEMAND FORECAST (Prophet L5)
═══════════════════════════════════════════════════════
{'  forecast_demand_drop_pct: ' + f'{forecast_drop_pct:.1f}%' if forecast_drop_pct is not None else '  NOT AVAILABLE'}

═══════════════════════════════════════════════════════
SIMULATION OUTPUT (L6)
═══════════════════════════════════════════════════════
{'  stockout_probability_pct: ' + f'{stockout_pct:.1f}%' + chr(10) + '  alternate_route: ' + (alt_route or 'not specified') if stockout_pct is not None else '  NOT AVAILABLE'}

═══════════════════════════════════════════════════════
CHROMADB RAG CONTEXT (two-stage retrieve + rerank)
═══════════════════════════════════════════════════════
{rag_context if rag_context.strip() else "(No ChromaDB results — use domain knowledge and India sourcing hierarchy.)"}

═══════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════
Generate a MitigationLLMOutput for this {risk.final_label} disruption.
  urgency = {'IMMEDIATE' if risk.final_label == 'CRITICAL' else risk.final_label}
"""


def _rule_based_mitigation(risk_label: str, stockout_note: str, alt_route: str, forecast_note: str) -> MitigationAction:
    """Fallback mitigation when LLM is unavailable."""
    recommendations = [
        f"[INVENTORY] Raise safety stock — stockout estimate: {stockout_note}.",
        f"[ROUTING] Prepare diversion through {alt_route} and confirm carrier capacity.",
        f"[SOURCING] Review alternate suppliers aligned to forecast variance: {forecast_note}.",
    ]
    cost_delta = (
        "HIGH: expedite critical inventory and activate alternate sourcing."
        if risk_label == "CRITICAL"
        else "Moderate: reserve backup logistics and inventory capacity."
    )
    return MitigationAction(
        summary=f"{risk_label} electronics supply-chain risk requires inventory, routing, and supplier actions.",
        recommendations=recommendations,
        cost_delta=cost_delta,
        urgency="IMMEDIATE" if risk_label == "CRITICAL" else risk_label,
    )


def mitigation_recommendation_agent(state: GlobalState) -> Dict[str, Any]:
    """
    L7 Mitigation Recommendation Agent.

    LLM path: gpt-4o + three two-stage RAG queries + India sourcing hierarchy.
    """
    if state.risk_classification is None:
        raise ValueError("Risk classification is required for mitigation — run L4 first.")

    risk = state.risk_classification
    metadata = state.event_metadata
    record = state.active_record or {}

    stockout = state.simulation_result.stockout_probability_pct if state.simulation_result else None
    forecast_drop = state.forecast_result.expected_drop_pct if state.forecast_result else None
    alt_route = (
        state.simulation_result.alternate_route if state.simulation_result else "the configured backup route"
    ) or "the configured backup route"

    stockout_note = f"{stockout:.1f}%" if stockout is not None else "unknown"
    forecast_note = f"{forecast_drop:.1f}%" if forecast_drop is not None else "unknown"

    export_control_raw = record.get("export_control_level")
    export_control_elevated = (
        export_control_raw is not None and float(export_control_raw) >= _EXPORT_CONTROL_TOP_QUARTILE
    )

    semiconductor_rows = _fetch_semiconductor_rows(record.get("year"))
    rag_context = build_mitigation_context(
        disruption_type=metadata.disruption_type if metadata else "unknown",
        order_region=record.get("order_region"),
        risk_label=risk.final_label,
        export_control_elevated=export_control_elevated,
    )

    llm_output: Optional[MitigationLLMOutput] = None
    llm_used = False

    if has_openai_api_key() and metadata is not None:
        try:
            user_msg = _build_mitigation_user_message(
                risk=risk,
                event_metadata=metadata,
                record=record,
                semiconductor_rows=semiconductor_rows,
                forecast_drop_pct=forecast_drop,
                stockout_pct=stockout,
                alt_route=alt_route,
                export_control_raw=export_control_raw,
                rag_context=rag_context,
            )
            llm_output = call_openai_structured(
                system_prompt=MITIGATION_SYSTEM_PROMPT,
                user_message=user_msg,
                response_model=MitigationLLMOutput,
                model=MODEL_REASONING,
                max_tokens=2048,
            )
            llm_used = True
        except Exception as exc:
            logger.warning("L7 LLM failed — using rule-based fallback: %s", exc)

    if llm_output:
        final_action = MitigationAction(
            summary=llm_output.summary,
            recommendations=llm_output.ranked_actions,
            cost_delta=llm_output.cost_estimate,
            urgency=llm_output.urgency,
            rag_citations=llm_output.rag_citations,
            india_sourcing_recommendations=llm_output.india_sourcing_recommendations,
        )
    else:
        final_action = _rule_based_mitigation(risk.final_label, stockout_note, alt_route, forecast_note)

    insert_mitigation_action(
        record.get("event_date") or record.get("order_date", ""),
        record.get("port", ""),
        record.get("sku", ""),
        risk.final_label,
        json.dumps(final_action.recommendations),
        final_action.cost_delta,
    )

    log_msg = (
        f"L7: Mitigation {'(gpt-4o)' if llm_used else '(fallback)'} | "
        f"label={risk.final_label} urgency={final_action.urgency} "
        f"actions={len(final_action.recommendations)} "
        f"india={len(final_action.india_sourcing_recommendations)}"
    )

    return {
        "mitigation_llm": llm_output,
        "mitigation_action": final_action,
        "agent_logs": state.agent_logs + [log_msg],
    }
