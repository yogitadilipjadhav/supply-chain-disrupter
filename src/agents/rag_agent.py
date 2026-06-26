import re
from typing import Any, Dict, List, Optional

from src.utils.rag_utils import build_rag_corpus_complete, query_chroma_rag
from src.agents.state import NewsRiskSignal


def query_news_signals(event_category: str) -> List[Dict[str, Any]]:
    query_text = f"Supply chain disruption signals for {event_category} in electronics imports"
    results = query_chroma_rag(query_text, n_results=8)
    if not results:
        try:
            # Build without flushing — safe to call while the client is alive.
            # flush_existing=True (shutil.rmtree) would cause WinError 32 on Windows
            # when the singleton PersistentClient holds the file lock.
            build_rag_corpus_complete(flush_existing=False)
            results = query_chroma_rag(query_text, n_results=8)
        except (FileNotFoundError, Exception):
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


_DURATION_PATTERNS = [
    (r'\b(\d+)\s*-?\s*day\b',           lambda m: float(m.group(1))),
    (r'\b(\d+)\s*-?\s*week\b',          lambda m: float(m.group(1)) * 7),
    (r'\b(\d+)\s*-?\s*month\b',         lambda m: float(m.group(1)) * 30),
    (r'into its second week',            lambda m: 10.0),
    (r'month-long',                      lambda m: 30.0),
    (r'several weeks',                   lambda m: 21.0),
    (r'for \b(\d+)\b',                   lambda m: float(m.group(1))),
]


def _extract_duration_days(text: str) -> Optional[float]:
    """
    Regex pass over RAG-retrieved text to extract a disruption duration estimate.
    Returns None if nothing extractable. Never raises.
    """
    if not text:
        return None
    for pattern, extractor in _DURATION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return extractor(m)
            except Exception:
                continue
    return None


def build_news_signals(event_category: str) -> List[NewsRiskSignal]:
    raw_signals = query_news_signals(event_category)
    signals = []
    for sig_dict in raw_signals:
        duration = _extract_duration_days(sig_dict.get("summary", ""))
        signal = NewsRiskSignal(**sig_dict, expected_duration_days=duration)
        signals.append(signal)
    return signals
