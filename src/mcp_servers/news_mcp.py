"""
MCP server — news tool for the L1 Data Ingestion Agent.

Exposes a single tool: fetch_news_headlines(query, hub_city, hub_country, supplier_country)
The L1 agent calls this via MCP instead of calling Google News RSS directly,
demonstrating the Model Context Protocol tool-calling pattern.

Run standalone:
    python -m src.mcp_servers.news_mcp

Or import and use the helper directly in tests:
    from src.mcp_servers.news_mcp import fetch_news_headlines
"""

import importlib.util
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FEEDPARSER_AVAILABLE = importlib.util.find_spec("feedparser") is not None

_RELEVANCE_KEYWORDS = [
    "semiconductor", "chip", "supply chain", "disruption", "factory",
    "fab", "port", "export control", "shortage", "shutdown",
]


def _relevance_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text_lower)
    return round(hits / len(_RELEVANCE_KEYWORDS), 4)


# ── MCP tool implementation ───────────────────────────────────────────────────

def fetch_news_headlines(
    query: str,
    hub_city: Optional[str] = None,
    hub_country: Optional[str] = None,
    supplier_country: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    MCP tool: fetch news headlines for a hub city / country / supplier query.

    Parameters
    ----------
    query            : RSS search query string
    hub_city         : Semiconductor hub city name (or None)
    hub_country      : Hub country name (or None)
    supplier_country : Supplier nation name (or None)
    max_results      : Max headlines to return (default 10)

    Returns
    -------
    List of dicts with keys: headline, summary, published_at, url,
                             relevance_score, hub_city, hub_country, supplier_country
    """
    if not _FEEDPARSER_AVAILABLE:
        raise RuntimeError(
            "feedparser is required. Install it with: pip install feedparser"
        )
    import feedparser

    term = query.replace(" ", "+")
    feed_url = f"https://news.google.com/rss/search?q={term}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_results]

    # Reuters fallback when Google News returns < 3 results
    if len(entries) < 3:
        backup = feedparser.parse("https://feeds.reuters.com/reuters/technologyNews")
        query_words = query.lower().split()[:3]
        backup_matches = [
            e for e in backup.entries
            if any(w in (e.get("title", "") + e.get("summary", "")).lower() for w in query_words)
        ]
        entries = list(entries) + backup_matches[: max_results - len(entries)]

    results = []
    for entry in entries:
        headline = entry.get("title", "")
        summary = entry.get("summary", "")[:500]
        if not headline:
            continue
        results.append({
            "headline": headline,
            "summary": summary,
            "published_at": entry.get("published", ""),
            "url": entry.get("link", ""),
            "relevance_score": _relevance_score(headline + " " + summary),
            "hub_city": hub_city,
            "hub_country": hub_country,
            "supplier_country": supplier_country,
        })
    return results


# ── MCP server definition (requires `mcp` package) ───────────────────────────

def build_mcp_server():
    """Build and return the FastMCP server object."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "The `mcp` package is required to run the MCP server. "
            "Install it with: pip install mcp"
        )

    mcp = FastMCP("supply-chain-news")

    @mcp.tool()
    def get_news_headlines(
        query: str,
        hub_city: str = "",
        hub_country: str = "",
        supplier_country: str = "",
        max_results: int = 10,
    ) -> str:
        """
        Fetch semiconductor supply-chain news headlines from Google News RSS
        (with Reuters RSS fallback). Returns JSON array of articles with
        headline, summary, published_at, url, and relevance_score.
        """
        result = fetch_news_headlines(
            query=query,
            hub_city=hub_city or None,
            hub_country=hub_country or None,
            supplier_country=supplier_country or None,
            max_results=max_results,
        )
        return json.dumps(result)

    return mcp


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    try:
        server = build_mcp_server()
        logger.info("Starting supply-chain-news MCP server (stdio transport)...")
        server.run(transport="stdio")
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)
