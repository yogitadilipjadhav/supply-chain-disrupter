from typing import Any, Dict, List

from src.utils.rag_utils import build_rag_corpus_complete, query_chroma_rag
from src.agents.state import NewsRiskSignal


def query_news_signals(event_category: str) -> List[Dict[str, Any]]:
    query_text = f"Supply chain disruption signals for {event_category} in electronics imports"
    results = query_chroma_rag(query_text, n_results=8)
    if not results:
        try:
            build_rag_corpus_complete(flush_existing=True)
            results = query_chroma_rag(query_text, n_results=8)
        except FileNotFoundError:
            return []
    parsed: List[Dict[str, Any]] = []
    for idx, result in enumerate(results):
        parsed.append(
            {
                "source_id": f"rag-{idx}",
                "category": event_category,
                "severity": 0.5,
                "summary": result["text"],
                "signal_tags": [event_category, "supply chain", "electronics"],
            }
        )
    return parsed


def build_news_signals(event_category: str) -> List[NewsRiskSignal]:
    raw_signals = query_news_signals(event_category)
    return [NewsRiskSignal(**signal) for signal in raw_signals]
