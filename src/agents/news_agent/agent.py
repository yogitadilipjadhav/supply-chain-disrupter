"""News & Event Analysis agent (L2) — builds risk signals from the RAG corpus."""

import logging
import sys
from typing import Any, Dict, Optional

from src.agents.state import GlobalState, NewsAnalysisLLMOutput, NewsRiskSignal

logger = logging.getLogger(__name__)


def _pkg():
    """Return the news_agent package module (for patchable attribute lookups)."""
    return sys.modules.get("src.agents.news_agent")


def news_event_analysis_agent(state: GlobalState) -> Dict[str, Any]:
    metadata = state.event_metadata
    if metadata is None:
        raise ValueError("Event metadata is required for news analysis.")

    mod = _pkg()
    build_fn = getattr(mod, "build_news_signals", None)
    if build_fn is None:
        from src.agents.news_agent.rag import build_news_signals
        build_fn = build_news_signals

    parsed_signals = build_fn(metadata.disruption_type)
    if not parsed_signals:
        parsed_signals = [
            NewsRiskSignal(
                source_id="fallback-001",
                category=metadata.disruption_type,
                severity=0.3,
                summary="Fallback risk signal for missing RAG data.",
                signal_tags=[metadata.disruption_type, "fallback"],
                expected_duration_days=float(metadata.shock_duration_days),
            )
        ]

    # Optional LLM enrichment — structured event classification + duration estimate.
    news_analysis_llm: Optional[NewsAnalysisLLMOutput] = None
    call_fn = getattr(mod, "call_openai_structured", None)
    if call_fn is not None:
        try:
            from src.utils.openai_utils import MODEL_FAST, build_rag_context
            rag_context = build_rag_context([metadata.disruption_type])
            signal_summary = "\n".join(
                f"- [{s.category}] severity={s.severity:.2f}: {s.summary}"
                for s in parsed_signals
            )
            system_prompt = (
                "You are a semiconductor supply-chain risk analyst. "
                "Classify the disruption event and estimate its duration and impact."
            )
            user_message = (
                f"Disruption type: {metadata.disruption_type}\n"
                f"Port: {metadata.affected_port}\n"
                f"Route: {metadata.affected_route}\n"
                f"Severity: {metadata.severity}\n"
                f"Known duration: {metadata.shock_duration_days} days\n\n"
                f"RAG context:\n{rag_context}\n\n"
                f"Rule-based signals:\n{signal_summary}"
            )
            news_analysis_llm = call_fn(
                system_prompt,
                user_message,
                NewsAnalysisLLMOutput,
                model=MODEL_FAST,
            )
            if news_analysis_llm and news_analysis_llm.expected_duration_days:
                for sig in parsed_signals:
                    if sig.expected_duration_days is None:
                        object.__setattr__(sig, "expected_duration_days", news_analysis_llm.expected_duration_days)
        except Exception as exc:
            logger.warning("L2: LLM enrichment failed (%s) — rule-based signals only.", exc)
            news_analysis_llm = None

    return {
        "news_signals": parsed_signals,
        "news_analysis_llm": news_analysis_llm,
        "agent_logs": state.agent_logs + ["L2: News and event analysis completed."],
    }
