"""News & Event Analysis agent (L2) — RAG signals with optional LLM enrichment."""

import logging
import sys
from typing import Any, Dict

from src.agents.state import GlobalState, NewsRiskSignal

logger = logging.getLogger(__name__)


def news_event_analysis_agent(state: GlobalState) -> Dict[str, Any]:
    _pkg = sys.modules["src.agents.news_agent"]
    build_news_signals = _pkg.build_news_signals
    call_openai_structured = _pkg.call_openai_structured

    metadata = state.event_metadata
    if metadata is None:
        raise ValueError("Event metadata is required for news analysis.")

    parsed_signals = build_news_signals(metadata.disruption_type)
    if not parsed_signals:
        parsed_signals = [
            NewsRiskSignal(
                source_id="fallback-001",
                category=metadata.disruption_type,
                severity=0.3,
                summary="Fallback risk signal — no RAG data available.",
                signal_tags=[metadata.disruption_type, "fallback"],
                expected_duration_days=float(metadata.shock_duration_days),
            )
        ]

    news_analysis_llm = None
    try:
        from src.utils.openai_utils import has_openai_api_key
        if has_openai_api_key():
            from src.agents.state import NewsAnalysisLLMOutput
            system_prompt = (
                "You are a supply-chain risk analyst. "
                "Classify the disruption event and estimate its expected duration in days."
            )
            signals_summary = "\n".join(
                f"- [{s.category}] {s.summary} (severity={s.severity})"
                for s in parsed_signals
            )
            user_prompt = (
                f"Disruption type: {metadata.disruption_type}\n"
                f"Affected port: {metadata.affected_port}\n"
                f"RAG signals:\n{signals_summary}"
            )
            news_analysis_llm = call_openai_structured(system_prompt, user_prompt, NewsAnalysisLLMOutput)
    except Exception as exc:
        logger.warning("L2: LLM enrichment failed: %s", exc)

    return {
        "news_signals": parsed_signals,
        "news_analysis_llm": news_analysis_llm,
        "agent_logs": state.agent_logs + ["L2: News and event analysis completed."],
    }
