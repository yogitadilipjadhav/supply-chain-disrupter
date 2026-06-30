"""News & Event Analysis agent (L2) — builds risk signals from the RAG corpus."""

from typing import Any, Dict

from src.agents.news_agent.rag import build_news_signals
from src.agents.state import GlobalState, NewsRiskSignal


def news_event_analysis_agent(state: GlobalState) -> Dict[str, Any]:
    metadata = state.event_metadata
    if metadata is None or state.config is None:
        raise ValueError("Event metadata and config are required for news analysis.")
    parsed_signals = build_news_signals(metadata.disruption_type)
    if not parsed_signals:
        parsed_signals = [
            NewsRiskSignal(
                source_id="fallback-001",
                category=metadata.disruption_type,
                severity=0.3,
                summary="Fallback risk signal for missing RAG data.",
                signal_tags=[metadata.disruption_type, "fallback"],
            )
        ]
    return {
        "news_signals": parsed_signals,
        "agent_logs": state.agent_logs + ["L2: News and event analysis completed."],
    }
