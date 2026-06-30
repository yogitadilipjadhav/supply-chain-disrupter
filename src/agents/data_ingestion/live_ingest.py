"""
Developed by : Chiradeep Banerjee

Live data ingestion for the Data Ingestion agent (Agent 1).

Fetches live signals from external APIs and persists them to SQLite:

- Weather: Open-Meteo, one row per configured hub (idempotent per day).
- News: GDELT DOC 2.0 API, with an optional RSS fallback.

Design note: this layer only *transports and persists* raw signals (with
provenance: source_type + ingestion_ts, plus dedup). Risk scoring, NER and RAG
stay in the Weather/News agents. Coarse region/category tags here are only
ingest-time hints to make filtering cheap, not authoritative classifications.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


def _ssl_verify() -> bool:
    """TLS verification stays ON unless explicitly disabled via env.

    Some corporate networks intercept TLS with their own (or expired) certs,
    which breaks GDELT. Set INGEST_INSECURE_SSL=1 to opt out deliberately —
    never disable verification by default in committed code.
    """
    if os.getenv("INGEST_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        return False
    return True

from src.agents.weather_agent.client import fetch_open_meteo
from src.utils.db_utils import (
    ensure_ingestion_schema,
    get_table_count,
    insert_news_signals,
    upsert_weather_signal,
)
from src.utils.yaml_utils import load_config


# Weather ingestion (Open-Meteo)

# Open-Meteo weather codes that indicate disruptive conditions.
BAD_WEATHER_CODES = {
    51, 53, 55, 61, 63, 65, 66, 67,        # drizzle / rain / freezing rain
    71, 73, 75, 77, 80, 81, 82, 85, 86,    # snow / showers
    95, 96, 99,                            # thunderstorm
}


def _weather_factors(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Derive severity factor scores from an Open-Meteo hourly payload.

    Self-contained so ingestion does not depend on the (separately evolving)
    weather agent. Always returns a safe structure, never raises.
    """
    try:
        hourly = payload.get("hourly", {})

        wind = [w for w in hourly.get("windspeed_10m", []) if w is not None]
        if wind:
            wind_score = min(1.0, (sum(wind) / len(wind)) / 40.0)
            max_wind = max(wind)
        else:
            wind_score, max_wind = 0.0, 0.0

        precip = [p for p in hourly.get("precipitation", []) if p is not None]
        if precip:
            precipitation_score = min(1.0, (sum(precip) / len(precip)) / 25.0)
            max_precip = max(precip)
        else:
            precipitation_score, max_precip = 0.0, 0.0

        codes = [c for c in hourly.get("weathercode", []) if c is not None]
        if codes:
            bad_ratio = sum(1 for c in codes if int(c) in BAD_WEATHER_CODES) / len(codes)
            weather_code_score = min(0.3, bad_ratio * 0.3)
        else:
            weather_code_score = 0.0

        severity = round(
            min(1.0, wind_score + precipitation_score + weather_code_score), 3
        )
        summary = (
            f"Max wind {max_wind:.1f} km/h, max precip {max_precip:.1f} mm; "
            f"severity {severity}."
        )
        return {
            "severity": severity,
            "wind_score": round(wind_score, 3),
            "precipitation_score": round(precipitation_score, 3),
            "weather_code_score": round(weather_code_score, 3),
            "max_wind_speed": round(max_wind, 1),
            "max_precipitation": round(max_precip, 1),
            "weather_summary": summary,
        }
    except Exception:
        return {
            "severity": 0.0,
            "wind_score": 0.0,
            "precipitation_score": 0.0,
            "weather_code_score": 0.0,
            "max_wind_speed": 0.0,
            "max_precipitation": 0.0,
            "weather_summary": "Weather data unavailable.",
        }


def ingest_weather(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fetch and store live weather for every hub in the config."""
    ensure_ingestion_schema()
    config = config or load_config()
    hubs = config.get("ports", {})
    now_iso = datetime.now(timezone.utc).isoformat()
    observation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    stored, failures = 0, []
    for hub, coords in hubs.items():
        try:
            payload = fetch_open_meteo(
                float(coords["latitude"]), float(coords["longitude"])
            )
            factors = _weather_factors(payload)
            upsert_weather_signal(
                {
                    "hub": hub,
                    "latitude": float(coords["latitude"]),
                    "longitude": float(coords["longitude"]),
                    "observation_date": observation_date,
                    "source_type": "live_weather",
                    "ingestion_ts": now_iso,
                    **factors,
                }
            )
            stored += 1
        except Exception as exc:  # one hub failing must not stop the rest
            failures.append({"hub": hub, "error": str(exc)})

    return {
        "hubs": len(hubs),
        "stored": stored,
        "failures": failures,
        "weather_signals_total": get_table_count("weather_signals"),
    }


# ──────────────────────────────────────────────────────────────────────────
# News ingestion (GDELT DOC 2.0 API, optional RSS fallback)
# ──────────────────────────────────────────────────────────────────────────

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_NEWS_QUERIES = [
    "semiconductor shortage",
    "chip supply disruption",
    "port closure shipping delay",
    "supply chain disruption electronics",
    "Red Sea shipping crisis",
    "export controls semiconductors",
    "factory shutdown electronics",
]

# Default RSS feeds used only when --rss is requested or GDELT returns nothing.
DEFAULT_RSS_FEEDS = [
    "https://www.supplychaindive.com/feeds/news/",
    "https://feeds.reuters.com/reuters/businessNews",
]

_REGION_KEYWORDS = {
    "Taiwan": ["taiwan", "tsmc", "hsinchu"],
    "China": ["china", "chinese", "smic"],
    "Japan": ["japan", "japanese"],
    "Korea": ["korea", "samsung"],
    "India": ["india", "indian"],
    "Red Sea": ["red sea", "suez", "houthi"],
    "Europe": ["europe", "german", "netherlands", "asml"],
    "US": ["united states", "u.s.", "american", "chips act"],
}

_CATEGORY_KEYWORDS = {
    "chip shortage": ["chip", "semiconductor", "wafer", "foundry"],
    "port closure": ["port", "shipping", "vessel", "container"],
    "geopolitical": ["sanction", "export control", "tariff", "war", "trade"],
    "weather disruption": ["storm", "flood", "cyclone", "earthquake", "typhoon"],
    "supplier failure": ["shutdown", "strike", "lockdown", "factory"],
}


def _coarse_tag(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    lowered = text.lower()
    for label, keywords in mapping.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _content_hash(url: str, title: str, stamp: str) -> str:
    key = (url or (title + stamp)).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _parse_gdelt_date(seendate: str) -> Optional[str]:
    """GDELT seendate looks like '20240115T103000Z' -> ISO 8601."""
    if not seendate:
        return None
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        ).isoformat()
    except ValueError:
        return seendate


def fetch_gdelt(
    query: str, max_records: int = 50, timespan: str = "3d"
) -> List[Dict[str, Any]]:
    """Query the GDELT DOC 2.0 API. Returns a list of article dicts (may be empty)."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "timespan": timespan,
        "sort": "DateDesc",
    }
    response = requests.get(GDELT_URL, params=params, timeout=20, verify=_ssl_verify())
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        return []  # GDELT returns empty/non-JSON body when no matches
    return data.get("articles", []) or []


def _rows_from_gdelt(query: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for article in articles:
        url = article.get("url", "")
        title = article.get("title", "")
        seendate = article.get("seendate", "")
        rows.append(
            {
                "title": title,
                "summary": title,  # GDELT ArtList carries headline only, no body
                "url": url,
                "publisher": article.get("domain"),
                "published_at": _parse_gdelt_date(seendate),
                "detected_region": _coarse_tag(title, _REGION_KEYWORDS),
                "detected_category": _coarse_tag(title, _CATEGORY_KEYWORDS),
                "query_tag": query,
                "content_hash": _content_hash(url, title, seendate),
                "source_type": "live_news",
                "ingestion_ts": now_iso,
            }
        )
    return rows


def _fetch_rss_rows(feeds: List[str]) -> List[Dict[str, Any]]:
    """Fetch articles from RSS feeds. Returns [] if feedparser is unavailable."""
    try:
        import feedparser  # lazy: RSS is optional
    except ImportError:
        print("[news] feedparser not installed; skipping RSS feeds.")
        return []

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:
            print(f"[news] RSS feed failed {feed_url}: {exc}")
            continue
        for entry in parsed.entries:
            title = entry.get("title", "")
            url = entry.get("link", "")
            summary = entry.get("summary", title)
            published = entry.get("published", "")
            rows.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "publisher": parsed.feed.get("title") if parsed.feed else None,
                    "published_at": published or None,
                    "detected_region": _coarse_tag(f"{title} {summary}", _REGION_KEYWORDS),
                    "detected_category": _coarse_tag(
                        f"{title} {summary}", _CATEGORY_KEYWORDS
                    ),
                    "query_tag": "rss",
                    "content_hash": _content_hash(url, title, published),
                    "source_type": "live_news",
                    "ingestion_ts": now_iso,
                }
            )
    return rows


def ingest_news(
    queries: Optional[List[str]] = None,
    max_records: int = 50,
    timespan: str = "3d",
    use_rss: bool = False,
    rss_feeds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch and store live news from GDELT (and optionally RSS), deduped."""
    ensure_ingestion_schema()
    queries = queries or DEFAULT_NEWS_QUERIES

    rows: List[Dict[str, Any]] = []
    failures = []
    for query in queries:
        try:
            articles = fetch_gdelt(query, max_records=max_records, timespan=timespan)
            rows.extend(_rows_from_gdelt(query, articles))
        except Exception as exc:
            failures.append({"query": query, "error": str(exc)})

    if use_rss or not rows:
        rows.extend(_fetch_rss_rows(rss_feeds or DEFAULT_RSS_FEEDS))

    inserted = insert_news_signals(rows)
    return {
        "fetched": len(rows),
        "inserted": inserted,
        "duplicates_skipped": len(rows) - inserted,
        "failures": failures,
        "news_signals_total": get_table_count("news_signals"),
    }


def ingest_all(use_rss: bool = False) -> Dict[str, Any]:
    return {"weather": ingest_weather(), "news": ingest_news(use_rss=use_rss)}
