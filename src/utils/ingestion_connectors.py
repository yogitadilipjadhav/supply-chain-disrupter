"""
API connectors for the DataIngestionAgent.

All connectors follow the BaseConnector interface:
  fetch()     → List[Dict]  raw API response
  normalize() → List[Dict]  mapped to target table columns
  persist()   → (inserted, skipped)

Connectors never raise outside _run_connector() — all exceptions are caught there.
"""

import importlib.util
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.utils.db_utils import execute_many, execute_query, get_connection
from src.utils.ingestion_schema import get_last_fetched_key
from src.utils.ingestion_validator import DataValidator
from src.utils.yaml_utils import load_config

logger = logging.getLogger(__name__)

_FEEDPARSER_AVAILABLE = importlib.util.find_spec("feedparser") is not None
_YFINANCE_AVAILABLE = importlib.util.find_spec("yfinance") is not None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Global semiconductor hub cities (ref-spec: source-of-disruption monitoring) ──
HUB_CITIES: Dict[str, Tuple[float, float]] = {
    "Hsinchu":   (24.80, 120.97),
    "Osaka":     (34.69, 135.50),
    "Austin":    (30.27, -97.74),
    "Shanghai":  (31.23, 121.47),
    "Singapore": ( 1.35, 103.82),
    "Rotterdam": (51.92,   4.48),
}

# RSS query strings scoped to hub cities, hub countries, and key supplier nations
_HUB_CITY_QUERIES: Dict[str, str] = {
    "Hsinchu":   "Hsinchu semiconductor fab disruption supply chain",
    "Osaka":     "Osaka fab logistics delay chip supply",
    "Austin":    "Austin chip factory shutdown semiconductor",
    "Shanghai":  "Shanghai port logistics delay semiconductor export",
    "Singapore": "Singapore port congestion semiconductor supply chain",
    "Rotterdam": "Rotterdam port disruption electronics supply chain",
}

_HUB_COUNTRY_QUERIES: Dict[str, str] = {
    "Taiwan":      "Taiwan chip supply chain disruption TSMC export",
    "South Korea": "South Korea Samsung semiconductor factory supply",
    "Japan":       "Japan semiconductor shortage fab supply chain",
    "China":       "China semiconductor export restriction chip shortage",
}

_SUPPLIER_QUERIES: Dict[str, str] = {
    "Malaysia":    "Malaysia chip manufacturing supply chain shortage",
    "India":       "India semiconductor PLI manufacturing supply chain",
    "Germany":     "Germany ASML semiconductor equipment supply",
    "Netherlands": "Netherlands semiconductor equipment export control ASML",
}

# Keywords for relevance scoring (ref spec)
_RELEVANCE_KEYWORDS = [
    "semiconductor", "chip", "supply chain", "disruption", "factory",
    "fab", "port", "export control", "shortage", "shutdown",
]


def _compute_relevance_score(text: str) -> float:
    """Keyword-overlap relevance score in [0, 1]."""
    text_lower = text.lower()
    hits = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text_lower)
    return round(hits / len(_RELEVANCE_KEYWORDS), 4)


def _compute_hub_severity(wind_kmh: float, precip_mm: float, weather_code: Optional[int]) -> float:
    """Ref-spec severity formula: 0-10 scale for hub city weather."""
    score = 0.0
    if weather_code is not None and weather_code >= 95:
        score += 4
    if wind_kmh >= 60:
        score += 2
    elif wind_kmh >= 30:
        score += 1
    if precip_mm > 50:
        score += 2
    elif precip_mm > 10:
        score += 1
    return min(10.0, score)


# WMO weather codes
_FLOOD_CODES = {55, 65, 75, 82, 85, 95, 99}
_STORM_CODES = {95, 96, 99}

# Keyword category tagger (no LLM required)
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "chip_shortage":    ["semiconductor", "chip", "fab", "wafer", "tsmc", "nvidia", "intel", "shortage"],
    "port_closure":     ["port closure", "congestion", "container", "freight", "berth", "shipping lane"],
    "geopolitical":     ["sanction", "export control", "bis", "tariff", "trade war", "taiwan", "embargo"],
    "weather":          ["typhoon", "earthquake", "flood", "storm", "cyclone", "hurricane", "disaster"],
    "supplier_failure": ["factory shutdown", "bankrupt", "production halt", "supplier fail", "lockdown"],
    "export_control":   ["bis rule", "entity list", "chips act", "export restriction", "mofcom", "meti"],
}


def _tag_categories(text: str) -> List[str]:
    """Return list of matching risk categories based on keyword presence."""
    text_lower = text.lower()
    return [cat for cat, kws in _CATEGORY_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def _fetch_relevant_cities() -> List[str]:
    """Return distinct order_city values from lite_master for domain-scoped queries."""
    try:
        rows = execute_query(
            "SELECT DISTINCT order_city FROM lite_master "
            "WHERE order_city IS NOT NULL AND order_city != '' "
            "ORDER BY order_city LIMIT 20"
        )
        return [row[0] for row in rows if row[0]]
    except Exception as exc:
        logger.debug("Could not fetch relevant cities from lite_master: %s", exc)
        return []


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseConnector(ABC):
    SOURCE_NAME: str = "base"
    TARGET_TABLE: str = ""
    _run_id: Optional[str] = None  # set by DataIngestionAgent._run_connector() before persist()

    @abstractmethod
    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch raw data from external source."""

    @abstractmethod
    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform raw records into target table column layout."""

    @abstractmethod
    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Persist to SQLite. Returns (inserted, skipped)."""


# ── 1. Open-Meteo Enhanced ───────────────────────────────────────────────────

class OpenMeteoEnhancedConnector(BaseConnector):
    SOURCE_NAME = "open_meteo_enhanced"
    TARGET_TABLE = "weather_events"

    _API_URL = "https://api.open-meteo.com/v1/forecast"
    _PARAMS_DAILY = {
        "hourly": "windspeed_10m,precipitation,weathercode",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
        "timezone": "UTC",
        "forecast_days": "1",
    }

    def __init__(self) -> None:
        cfg = load_config()
        self._ports = cfg.get("ports", {})

    def fetch(self) -> List[Dict[str, Any]]:
        results = []
        # Indian delivery ports → weather_events (existing behaviour)
        for port_name, coords in self._ports.items():
            try:
                params = {
                    **self._PARAMS_DAILY,
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                }
                resp = requests.get(self._API_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                data["_port"] = port_name
                data["_lat"] = coords["latitude"]
                data["_lon"] = coords["longitude"]
                data["_target"] = "weather_events"
                results.append(data)
            except Exception as exc:
                logger.warning("Open-Meteo fetch failed for port %s: %s", port_name, exc)
        # Global semiconductor hub cities → live_weather_ingest (ref-spec v2)
        for city, (lat, lon) in HUB_CITIES.items():
            try:
                params = {
                    **self._PARAMS_DAILY,
                    "latitude": lat,
                    "longitude": lon,
                }
                resp = requests.get(self._API_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                data["_hub_city"] = city
                data["_lat"] = lat
                data["_lon"] = lon
                data["_target"] = "live_weather_ingest"
                results.append(data)
            except Exception as exc:
                logger.warning("Open-Meteo fetch failed for hub city %s: %s", city, exc)
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        ts = _now_utc()
        for data in raw:
            daily = data.get("daily", {})
            codes = daily.get("weathercode", [None])
            code = codes[0] if codes else None
            max_wind = (daily.get("windspeed_10m_max") or [0.0])[0] or 0.0
            precip = (daily.get("precipitation_sum") or [0.0])[0] or 0.0
            temp_max = (daily.get("temperature_2m_max") or [None])[0]
            temp_min = (daily.get("temperature_2m_min") or [None])[0]
            temp_avg = (temp_max + temp_min) / 2 if (temp_max is not None and temp_min is not None) else None

            target = data.get("_target", "weather_events")

            if target == "weather_events":
                storm_bonus = 0.3 if code in _STORM_CODES else 0.0
                flood_bonus = 0.2 if code in _FLOOD_CODES else 0.0
                severity = min(1.0, (max_wind / 60.0) + (precip / 100.0) + storm_bonus + flood_bonus)
                ndr = round(1.18 + severity * (10.0 - 1.18), 4)
                rows.append({
                    "_target": "weather_events",
                    "fetched_at_utc": ts,
                    "port": data["_port"],
                    "latitude": data["_lat"],
                    "longitude": data["_lon"],
                    "windspeed_10m_max_kmh": round(max_wind, 2),
                    "precipitation_mm_6h": round(precip, 2),
                    "temperature_2m_c": round(temp_avg, 2) if temp_avg is not None else None,
                    "weathercode": code,
                    "flood_risk_flag": int(code in _FLOOD_CODES) if code is not None else 0,
                    "storm_flag": int(code in _STORM_CODES) if code is not None else 0,
                    "derived_weather_severity": round(severity, 4),
                    "derived_natural_disaster_risk": ndr,
                    "source": "open-meteo",
                    "is_active": 1,
                })
            else:  # live_weather_ingest — ref-spec 0-10 severity formula
                raw_severity = _compute_hub_severity(max_wind, precip, code)
                rows.append({
                    "_target": "live_weather_ingest",
                    "fetched_at_utc": ts,
                    "hub_city": data["_hub_city"],
                    "latitude": data["_lat"],
                    "longitude": data["_lon"],
                    "wind_speed_kmh": round(max_wind, 2),
                    "precipitation_mm": round(precip, 2),
                    "weather_code": code,
                    "temperature_c": round(temp_avg, 2) if temp_avg is not None else None,
                    "raw_severity_score": raw_severity,
                    "is_trigger_hub": 0,  # set in persist after finding max
                })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        inserted = skipped = 0
        run_id = self._run_id or "unknown"

        weather_rows = [r for r in rows if r.get("_target") == "weather_events"]
        hub_rows = [r for r in rows if r.get("_target") == "live_weather_ingest"]

        # Mark the hub city with max severity as trigger hub
        if hub_rows:
            max_idx = max(range(len(hub_rows)), key=lambda i: hub_rows[i]["raw_severity_score"])
            if hub_rows[max_idx]["raw_severity_score"] >= 6.0:
                hub_rows[max_idx]["is_trigger_hub"] = 1

        for row in weather_rows:
            ok, errs = DataValidator.validate_weather_event(row)
            if not ok:
                logger.warning("Invalid weather_event row: %s", errs)
                skipped += 1
                continue
            try:
                with get_connection() as conn:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO weather_events
                        (fetched_at_utc, port, latitude, longitude,
                         windspeed_10m_max_kmh, precipitation_mm_6h, temperature_2m_c,
                         weathercode, flood_risk_flag, storm_flag,
                         derived_weather_severity, derived_natural_disaster_risk,
                         source, is_active)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row["fetched_at_utc"], row["port"], row["latitude"], row["longitude"],
                            row.get("windspeed_10m_max_kmh"), row.get("precipitation_mm_6h"),
                            row.get("temperature_2m_c"), row.get("weathercode"),
                            row["flood_risk_flag"], row["storm_flag"],
                            row["derived_weather_severity"], row["derived_natural_disaster_risk"],
                            row["source"], row["is_active"],
                        ),
                    )
                    conn.commit()
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
            except Exception as exc:
                logger.warning("weather_events insert failed: %s", exc)
                skipped += 1

        for row in hub_rows:
            try:
                with get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO live_weather_ingest
                        (run_id, fetched_at_utc, hub_city, latitude, longitude,
                         wind_speed_kmh, precipitation_mm, weather_code,
                         temperature_c, raw_severity_score, is_trigger_hub)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            run_id, row["fetched_at_utc"], row["hub_city"],
                            row["latitude"], row["longitude"],
                            row.get("wind_speed_kmh"), row.get("precipitation_mm"),
                            row.get("weather_code"), row.get("temperature_c"),
                            row["raw_severity_score"], row["is_trigger_hub"],
                        ),
                    )
                    conn.commit()
                inserted += 1
            except Exception as exc:
                logger.warning("live_weather_ingest insert failed for %s: %s", row.get("hub_city"), exc)
                skipped += 1

        return inserted, skipped


# ── 2. FRED (St. Louis Fed) ──────────────────────────────────────────────────

class FredConnector(BaseConnector):
    SOURCE_NAME = "fred"
    TARGET_TABLE = "freight_signals"

    _CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    # Only the semiconductor PPI — the three general macro series were dropped
    # because they don't map to the workbook's specific products.
    _SERIES: Dict[str, Tuple[str, str, str]] = {
        "WPSFD4131": ("PPI: Semiconductors & Electronic Components", "index", "semiconductor_ppi"),
    }

    def _fetch_series(self, series_id: str) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"id": series_id}
        last_key = get_last_fetched_key(f"fred_{series_id}")
        if last_key:
            params["vintage_date"] = last_key
        try:
            resp = requests.get(self._CSV_URL, params=params, timeout=20)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            results = []
            for line in lines[1:]:  # skip header
                parts = line.strip().split(",")
                if len(parts) < 2 or parts[1].strip() == ".":
                    continue
                try:
                    results.append({
                        "series_id": series_id,
                        "signal_date": parts[0].strip(),
                        "value": float(parts[1].strip()),
                    })
                except ValueError:
                    continue
            return results
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return []

    def fetch(self) -> List[Dict[str, Any]]:
        results = []
        for series_id in self._SERIES:
            rows = self._fetch_series(series_id)
            results.extend(rows)
            if rows:
                time.sleep(0.3)  # be polite to FRED
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Compute chip_price normalization bounds from lite_master once
        try:
            bounds_rows = execute_query("SELECT MIN(chip_price_index), MAX(chip_price_index) FROM lite_master")
            cpi_lo = float(bounds_rows[0][0] or 0.5)
            cpi_hi = float(bounds_rows[0][1] or 2.0)
        except Exception:
            cpi_lo, cpi_hi = 0.5, 2.0

        ts = _now_utc()
        rows = []
        for r in raw:
            series_id = r["series_id"]
            name, unit, _ = self._SERIES.get(series_id, (series_id, "index", "unknown"))
            value = r["value"]
            # Normalize semiconductor PPI to chip_price_index scale
            normalized_sdi = None
            if series_id == "WPSFD4131" and cpi_hi > cpi_lo:
                normalized_sdi = round(4.09 + (value / 300.0) * (9.97 - 4.09), 4)  # 300 = approx PPI range
            rows.append({
                "fetched_at_utc": ts,
                "signal_date": r["signal_date"],
                "source": "fred",
                "series_id": series_id,
                "series_name": name,
                "value": value,
                "unit": unit,
                "route": None,
                "normalized_sdi": normalized_sdi,
                "is_active": 1,
            })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        inserted = skipped = 0
        for row in rows:
            ok, errs = DataValidator.validate_freight_signal(row)
            if not ok:
                skipped += 1
                continue
            try:
                with get_connection() as conn:
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO freight_signals
                        (fetched_at_utc, signal_date, source, series_id, series_name,
                         value, unit, route, normalized_sdi, is_active)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row["fetched_at_utc"], row["signal_date"], row["source"],
                            row["series_id"], row["series_name"], row["value"],
                            row.get("unit"), row.get("route"), row.get("normalized_sdi"),
                            row["is_active"],
                        ),
                    )
                    conn.commit()
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
            except Exception as exc:
                logger.warning("freight_signals insert failed: %s", exc)
                skipped += 1
        return inserted, skipped


# ── 3. GDELT GKG v2 ──────────────────────────────────────────────────────────

class GdeltConnector(BaseConnector):
    SOURCE_NAME = "gdelt_gkg"
    TARGET_TABLE = "news_disruptions"

    _API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    _QUERIES = [
        "semiconductor shortage supply chain",
        "electronics factory shutdown chip",
        "chip export control sanctions",
        "port congestion freight disruption India",
        "Taiwan earthquake semiconductor production",
    ]

    def fetch(self) -> List[Dict[str, Any]]:
        # Extend base queries with up to 3 workbook-specific cities
        cities = _fetch_relevant_cities()
        city_queries = [
            f"{city} electronics supply chain disruption" for city in cities[:3]
        ]
        all_queries = self._QUERIES + city_queries

        results = []
        for query in all_queries:
            try:
                params = {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": "20",
                    "format": "json",
                    "timespan": "3d",
                    "sort": "DateDesc",
                }
                resp = requests.get(self._API_URL, params=params, timeout=20)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for article in data.get("articles", []):
                    results.append({
                        "url":       article.get("url", ""),
                        "headline":  article.get("title", ""),
                        "published_at": article.get("seendate", ""),
                        "geo_country": article.get("sourcecountry", ""),
                        "tone_raw":  str(article.get("tone", "")),
                        "_query":    query,
                    })
                time.sleep(0.5)  # GDELT rate limit
            except Exception as exc:
                logger.warning("GDELT fetch failed for query '%s': %s", query, exc)
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ts = _now_utc()
        rows = []
        for r in raw:
            headline = r.get("headline", "")
            url = r.get("url", "")
            if not headline:
                continue
            article_hash = DataValidator.compute_article_hash(url, headline)
            # Parse GDELT tone (first field of comma-separated string)
            tone: Optional[float] = None
            try:
                tone = float(str(r.get("tone_raw", "")).split(",")[0])
            except (ValueError, IndexError):
                pass
            severity = min(1.0, max(0.0, -tone / 20.0)) if tone is not None else 0.3
            categories = _tag_categories(headline)
            if severity <= 0.3 and not categories:
                continue
            port = DataValidator.normalize_port_name(r.get("geo_country", ""))
            rows.append({
                "fetched_at_utc": ts,
                "published_at": DataValidator.normalize_timestamp(r.get("published_at", "")),
                "source": "gdelt_gkg",
                "headline": headline[:1000],
                "url": url[:2048],
                "canonical_port": port,
                "risk_categories": json.dumps(categories),
                "gdelt_tone": round(tone, 4) if tone is not None else None,
                "severity_score": round(severity, 4),
                "article_hash": article_hash,
                "is_active": 1,
            })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        return _persist_news_disruptions(rows)


# ── 4. Google News RSS ───────────────────────────────────────────────────────

class GoogleNewsRSSConnector(BaseConnector):
    SOURCE_NAME = "google_news_rss"
    TARGET_TABLE = "news_disruptions"

    _SEARCH_TERMS = [
        "semiconductor+shortage+supply+chain",
        "chip+export+control+BIS+sanctions",
        "India+port+congestion+freight",
        "TSMC+production+disruption",
        "electronics+supply+chain+disruption",
    ]

    def fetch(self) -> List[Dict[str, Any]]:
        if not _FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed — GoogleNewsRSSConnector skipped")
            return []
        import feedparser

        results = []

        # ── Existing broad terms → news_disruptions ──────────────────────────
        cities = _fetch_relevant_cities()
        city_terms = [f"{c.replace(' ', '+')}+electronics+supply+chain" for c in cities[:3]]
        for term in self._SEARCH_TERMS + city_terms:
            try:
                url = f"https://news.google.com/rss/search?q={term}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    results.append({
                        "_target": "news_disruptions",
                        "headline": entry.get("title", ""),
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", ""),
                        "summary_snippet": entry.get("summary", "")[:500],
                        "source": "google_news_rss",
                    })
            except Exception as exc:
                logger.warning("Google News RSS failed for '%s': %s", term, exc)

        # ── Hub city / country / supplier queries → live_news_ingest (ref-spec) ──
        hub_query_map: List[Dict[str, Any]] = []
        for city, query in _HUB_CITY_QUERIES.items():
            hub_query_map.append({"query": query, "hub_city": city, "hub_country": None, "supplier_country": None})
        for country, query in _HUB_COUNTRY_QUERIES.items():
            hub_query_map.append({"query": query, "hub_city": None, "hub_country": country, "supplier_country": None})
        for supplier, query in _SUPPLIER_QUERIES.items():
            hub_query_map.append({"query": query, "hub_city": None, "hub_country": None, "supplier_country": supplier})

        for meta in hub_query_map:
            term = meta["query"].replace(" ", "+")
            try:
                url = f"https://news.google.com/rss/search?q={term}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)
                entries = feed.entries[:10]
                # Reuters RSS fallback when Google News returns < 3 results
                if len(entries) < 3:
                    backup = feedparser.parse("https://feeds.reuters.com/reuters/technologyNews")
                    entries = list(entries) + [
                        e for e in backup.entries
                        if any(kw in (e.get("title", "") + e.get("summary", "")).lower()
                               for kw in meta["query"].lower().split()[:3])
                    ][:10 - len(entries)]
                for entry in entries:
                    results.append({
                        "_target": "live_news_ingest",
                        "headline": entry.get("title", ""),
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:500],
                        "source": "google_news_rss",
                        "_query_term": meta["query"],
                        "_hub_city": meta["hub_city"],
                        "_hub_country": meta["hub_country"],
                        "_supplier_country": meta["supplier_country"],
                    })
            except Exception as exc:
                logger.warning("Google News RSS hub query failed for '%s': %s", meta["query"], exc)

        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ts = _now_utc()
        rows = []
        for r in raw:
            headline = r.get("headline", "")
            url = r.get("url", "")
            if not headline:
                continue
            target = r.get("_target", "news_disruptions")

            if target == "live_news_ingest":
                relevance = _compute_relevance_score(headline + " " + r.get("summary", ""))
                rows.append({
                    "_target": "live_news_ingest",
                    "fetched_at_utc": ts,
                    "source_feed": "google_news_rss",
                    "query_term": r.get("_query_term"),
                    "hub_city": r.get("_hub_city"),
                    "hub_country": r.get("_hub_country"),
                    "supplier_country": r.get("_supplier_country"),
                    "headline": headline[:1000],
                    "summary": r.get("summary", "")[:500],
                    "published_at": DataValidator.normalize_timestamp(r.get("published_at", "")),
                    "url": url[:2048],
                    "relevance_score": relevance,
                })
            else:
                article_hash = DataValidator.compute_article_hash(url, headline)
                categories = _tag_categories(headline + " " + r.get("summary_snippet", ""))
                severity = 0.4 if categories else 0.2
                if severity <= 0.3 and not categories:
                    continue
                rows.append({
                    "_target": "news_disruptions",
                    "fetched_at_utc": ts,
                    "published_at": DataValidator.normalize_timestamp(r.get("published_at", "")),
                    "source": "google_news_rss",
                    "headline": headline[:1000],
                    "url": url[:2048],
                    "canonical_port": None,
                    "risk_categories": json.dumps(categories),
                    "gdelt_tone": None,
                    "severity_score": round(severity, 4),
                    "article_hash": article_hash,
                    "is_active": 1,
                })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        run_id = self._run_id or "unknown"
        live_rows = [r for r in rows if r.get("_target") == "live_news_ingest"]
        disruption_rows = [r for r in rows if r.get("_target") != "live_news_ingest"]
        inserted = skipped = 0

        for row in live_rows:
            try:
                with get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO live_news_ingest
                        (run_id, fetched_at_utc, source_feed, query_term,
                         hub_city, hub_country, supplier_country,
                         headline, summary, published_at, url, relevance_score)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            run_id, row["fetched_at_utc"], row["source_feed"],
                            row.get("query_term"), row.get("hub_city"),
                            row.get("hub_country"), row.get("supplier_country"),
                            row["headline"], row.get("summary"),
                            row.get("published_at"), row.get("url"),
                            row["relevance_score"],
                        ),
                    )
                    conn.commit()
                inserted += 1
            except Exception as exc:
                logger.warning("live_news_ingest insert failed: %s", exc)
                skipped += 1

        ins2, sk2 = _persist_news_disruptions(disruption_rows)
        return inserted + ins2, skipped + sk2


# ── 5. Reuters RSS ───────────────────────────────────────────────────────────

class ReutersRSSConnector(BaseConnector):
    SOURCE_NAME = "reuters_rss"
    TARGET_TABLE = "supplier_risk_events"

    _RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/technologyNews",
    ]
    _FILTER_TERMS = [
        "semiconductor", "chip", "supply chain", "port", "freight",
        "taiwan", "sanctions", "export control", "shortage", "disruption",
        "factory", "foundry", "tsmc", "nvidia", "intel",
    ]

    def fetch(self) -> List[Dict[str, Any]]:
        if not _FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed — ReutersRSSConnector skipped")
            return []
        import feedparser
        results = []
        for feed_url in self._RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()
                    combined = title + " " + summary
                    if any(term in combined for term in self._FILTER_TERMS):
                        results.append({
                            "headline": entry.get("title", ""),
                            "url": entry.get("link", ""),
                            "published_at": entry.get("published", ""),
                            "summary_snippet": entry.get("summary", "")[:500],
                            "source": "reuters_rss",
                        })
            except Exception as exc:
                logger.warning("Reuters RSS failed for %s: %s", feed_url, exc)
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ts = _now_utc()
        rows = []
        for r in raw:
            headline = r.get("headline", "")
            url = r.get("url", "")
            if not headline:
                continue
            article_hash = DataValidator.compute_article_hash(url, headline)
            categories = _tag_categories(headline + " " + r.get("summary_snippet", ""))
            severity = min(1.0, 0.3 + len(categories) * 0.1)
            if severity <= 0.3 and not categories:
                continue
            rows.append({
                "fetched_at_utc": ts,
                "event_date": DataValidator.normalize_timestamp(r.get("published_at", ""))[:10],
                "source": "reuters_rss",
                "headline": headline[:1000],
                "summary_snippet": r.get("summary_snippet", ""),
                "canonical_port": None,
                "category": categories[0] if categories else "general",
                "severity_score": round(severity, 4),
                "disruption_news_count_contribution": 1,
                "article_hash": article_hash,
                "is_active": 1,
            })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        return _persist_supplier_risk_events(rows)


# ── 6. CISA / BIS Federal Register ──────────────────────────────────────────

class CisaBisRSSConnector(BaseConnector):
    SOURCE_NAME = "bis_federal_register"
    TARGET_TABLE = "news_disruptions"

    _BIS_URL = (
        "https://www.federalregister.gov/api/v1/documents.json"
        "?conditions[agencies][]=commerce-bureau-of-industry-and-security"
        "&per_page=20&order=newest"
    )

    def fetch(self) -> List[Dict[str, Any]]:
        results = []
        try:
            resp = requests.get(self._BIS_URL, timeout=15)
            if resp.ok:
                for doc in resp.json().get("results", []):
                    results.append({
                        "headline": doc.get("title", ""),
                        "url": doc.get("html_url", ""),
                        "published_at": doc.get("publication_date", ""),
                        "category": "export_control",
                        "source": "bis_federal_register",
                        "summary_snippet": doc.get("abstract", "")[:500],
                    })
        except Exception as exc:
            logger.warning("BIS Federal Register fetch failed: %s", exc)
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ts = _now_utc()
        rows = []
        for r in raw:
            headline = r.get("headline", "")
            url = r.get("url", "")
            if not headline:
                continue
            article_hash = DataValidator.compute_article_hash(url, headline)
            rows.append({
                "fetched_at_utc": ts,
                "published_at": DataValidator.normalize_timestamp(r.get("published_at", "")),
                "source": "bis_federal_register",
                "headline": headline[:1000],
                "url": url[:2048],
                "canonical_port": None,
                "risk_categories": json.dumps(["export_control"]),
                "gdelt_tone": None,
                "severity_score": 0.7,  # BIS rules are always high-severity signals
                "article_hash": article_hash,
                "is_active": 1,
            })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        return _persist_news_disruptions(rows)


# ── 7. yfinance (optional) ───────────────────────────────────────────────────

class YFinanceConnector(BaseConnector):
    SOURCE_NAME = "yfinance"
    TARGET_TABLE = "market_demand_signals"

    _TICKERS: Dict[str, Tuple[str, str]] = {
        "^SOX":  ("semiconductor_index", "Semiconductor Index (SOX)"),
        "BDRY":  ("shipping_etf",        "Breakwave Dry Bulk ETF"),
        "^GSPC": ("sp500",               "S&P 500"),
    }

    def fetch(self) -> List[Dict[str, Any]]:
        if not _YFINANCE_AVAILABLE:
            logger.info("yfinance not installed — YFinanceConnector skipped")
            return []
        import yfinance as yf
        results = []
        for ticker, (category, name) in self._TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                prev_close = None
                for date, row in hist.iterrows():
                    close = float(row["Close"])
                    pct_change = ((close - prev_close) / prev_close * 100) if prev_close else None
                    results.append({
                        "series_id": ticker,
                        "series_name": name,
                        "category": category,
                        "signal_date": date.strftime("%Y-%m-%d"),
                        "value": close,
                        "pct_change": round(pct_change, 4) if pct_change is not None else None,
                    })
                    prev_close = close
            except Exception as exc:
                logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return results

    def normalize(self, raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ts = _now_utc()
        # Get 52-week high for SOX to normalize SDI
        sox_52w_high = 600.0  # fallback; SOX peaked ~4000 in 2021 but typical range 200-600
        try:
            import yfinance as yf
            sox_info = yf.Ticker("^SOX").info
            sox_52w_high = float(sox_info.get("fiftyTwoWeekHigh", 600.0))
        except Exception:
            pass

        rows = []
        for r in raw:
            value = r["value"]
            # Chip price index: normalize SOX to chip_price_index scale
            normalized_cpi = None
            normalized_mgr = None
            if r["series_id"] == "^SOX" and sox_52w_high > 0:
                normalized_cpi = round(0.5 + (value / sox_52w_high) * 1.5, 4)
            if r.get("pct_change") is not None:
                normalized_mgr = round(max(-50.0, min(50.0, r["pct_change"])), 4)
            rows.append({
                "fetched_at_utc": ts,
                "signal_date": r["signal_date"],
                "source": "yfinance",
                "series_id": r["series_id"],
                "series_name": r["series_name"],
                "category": r["category"],
                "value": round(value, 4),
                "pct_change": r.get("pct_change"),
                "normalized_chip_price_index": normalized_cpi,
                "normalized_market_growth_rate": normalized_mgr,
                "is_active": 1,
            })
        return rows

    def persist(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        inserted = skipped = 0
        for row in rows:
            try:
                with get_connection() as conn:
                    cur = conn.execute(
                        """
                        INSERT OR REPLACE INTO market_demand_signals
                        (fetched_at_utc, signal_date, source, series_id, series_name,
                         category, value, pct_change, normalized_chip_price_index,
                         normalized_market_growth_rate, is_active)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row["fetched_at_utc"], row["signal_date"], row["source"],
                            row["series_id"], row["series_name"], row.get("category"),
                            row["value"], row.get("pct_change"),
                            row.get("normalized_chip_price_index"),
                            row.get("normalized_market_growth_rate"),
                            row["is_active"],
                        ),
                    )
                    conn.commit()
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
            except Exception as exc:
                logger.warning("market_demand_signals insert failed: %s", exc)
                skipped += 1
        return inserted, skipped


# ── Shared persist helpers ────────────────────────────────────────────────────

def _persist_news_disruptions(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    inserted = skipped = 0
    for row in rows:
        ok, errs = DataValidator.validate_news_row(row)
        if not ok:
            skipped += 1
            continue
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO news_disruptions
                    (fetched_at_utc, published_at, source, headline, url,
                     canonical_port, risk_categories, gdelt_tone, severity_score,
                     article_hash, is_active)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["fetched_at_utc"], row.get("published_at"), row["source"],
                        row["headline"], row.get("url"), row.get("canonical_port"),
                        row.get("risk_categories"), row.get("gdelt_tone"),
                        row.get("severity_score"), row.get("article_hash"), row.get("is_active", 1),
                    ),
                )
                conn.commit()
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
        except Exception as exc:
            logger.warning("news_disruptions insert failed: %s", exc)
            skipped += 1
    return inserted, skipped


def _persist_supplier_risk_events(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    inserted = skipped = 0
    for row in rows:
        ok, errs = DataValidator.validate_news_row(row)
        if not ok:
            skipped += 1
            continue
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO supplier_risk_events
                    (fetched_at_utc, event_date, source, headline, summary_snippet,
                     canonical_port, category, severity_score,
                     disruption_news_count_contribution, article_hash, is_active)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["fetched_at_utc"], row.get("event_date"), row["source"],
                        row["headline"], row.get("summary_snippet"),
                        row.get("canonical_port"), row.get("category"),
                        row.get("severity_score"), row.get("disruption_news_count_contribution", 1),
                        row.get("article_hash"), row.get("is_active", 1),
                    ),
                )
                conn.commit()
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
        except Exception as exc:
            logger.warning("supplier_risk_events insert failed: %s", exc)
            skipped += 1
    return inserted, skipped
