"""
DataValidator — validation and standardization for all ingestion tables.

Five pipeline stages per row:
  1. Schema    — required fields, type coercion; discard on failure
  2. Range     — clamp to normalization bounds with warning
  3. Dedup     — article_hash check for news; UNIQUE enforced at DB level
  4. Outlier   — Z-score advisory (log, never block)
  5. Conflict  — live_enrichment shift detection (log only)
"""

import email.utils
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.utils.db_utils import execute_query

logger = logging.getLogger(__name__)

# Normalization bounds mirroring risk_classifier_agent._get_norm_bounds()
BOUNDS: Dict[str, Tuple[float, float]] = {
    "weather_severity_hub":    (1.18, 10.0),
    "natural_disaster_risk":   (1.18, 10.0),
    "supply_disruption_index": (4.09, 9.97),
    "defect_rate_pct":         (2.0,  19.82),
    "disruption_news_count":   (0.0,  17.0),
    "derived_weather_severity": (0.0, 1.0),
    "severity_score":          (0.0,  1.0),
    "normalized_sdi":          (4.09, 9.97),
}

# Maps external place names → canonical YAML port names
PORT_NAME_MAP: Dict[str, Optional[str]] = {
    "JNPT": "JNPT", "Nhava Sheva": "JNPT", "Mumbai": "JNPT",
    "Mundra": "Mundra", "Kutch": "Mundra",
    "Chennai": "Chennai", "Madras": "Chennai",
    "Vizag": "Vizag", "Visakhapatnam": "Vizag",
    "Cochin": "Cochin", "Kochi": "Cochin", "Ernakulam": "Cochin",
    "Kolkata": "Kolkata", "Calcutta": "Kolkata", "Haldia": "Kolkata",
    "Pipavav": "Pipavav",
    # order_region values that appear in lite_master
    "South Asia": "JNPT",
    "Eastern Asia": None,
    "Southeast Asia": None,
    "West of USA": None,
    "Western Europe": None,
    "Eastern Europe": None,
}

_TIMESTAMP_FORMATS = [
    "%Y%m%dT%H%M%SZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


class DataValidator:

    # ── Timestamp ─────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_timestamp(ts_str: str) -> str:
        """Parse any reasonable date/time string → ISO-8601 UTC string."""
        if not ts_str:
            return datetime.now(timezone.utc).isoformat()
        for fmt in _TIMESTAMP_FORMATS:
            try:
                dt = datetime.strptime(ts_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue
        # Fallback: RFC-2822 (RSS feeds)
        try:
            parsed = email.utils.parsedate_to_datetime(ts_str)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()

    # ── Port name ─────────────────────────────────────────────────────────────

    @staticmethod
    def normalize_port_name(raw_name: str) -> Optional[str]:
        """Map external place names to canonical YAML port names."""
        if not raw_name:
            return None
        direct = PORT_NAME_MAP.get(raw_name)
        if direct is not None:
            return direct
        # Fuzzy match: check if any canonical name is a substring
        raw_lower = raw_name.lower()
        for key, canonical in PORT_NAME_MAP.items():
            if canonical and key.lower() in raw_lower:
                return canonical
        return None

    # ── Dedup ─────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_article_hash(url: str, headline: str) -> str:
        """sha256(url|headline) for dedup across news tables."""
        return hashlib.sha256(f"{url}|{headline}".encode()).hexdigest()

    # ── Range clamping ────────────────────────────────────────────────────────

    @staticmethod
    def clamp_to_bounds(value: float, field: str) -> float:
        """Clamp a value to the known normalization bounds, logging a warning if clamped."""
        if field not in BOUNDS:
            return value
        lo, hi = BOUNDS[field]
        clamped = max(lo, min(hi, value))
        if clamped != value:
            logger.warning("Clamped %s: %s → %s (bounds [%s, %s])", field, value, clamped, lo, hi)
        return clamped

    # ── Outlier detection ─────────────────────────────────────────────────────

    @staticmethod
    def check_outlier_zscore(
        value: float, field: str, table: str, window: int = 90
    ) -> bool:
        """Advisory Z-score check. Returns True if |z| > 3.0. Never blocks ingestion."""
        try:
            rows = execute_query(
                f"SELECT {field} FROM {table} WHERE {field} IS NOT NULL "
                f"ORDER BY rowid DESC LIMIT ?",
                (window,),
            )
            if len(rows) < 10:
                return False
            values = [float(r[0]) for r in rows if r[0] is not None]
            if not values:
                return False
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5
            if std == 0:
                return False
            z = abs((value - mean) / std)
            if z > 3.0:
                logger.warning(
                    "Outlier detected: %s.%s = %s (z=%.2f, mean=%.3f, std=%.3f)",
                    table, field, value, z, mean, std,
                )
                return True
        except Exception as exc:
            logger.debug("Outlier check failed for %s.%s: %s", table, field, exc)
        return False

    # ── Schema validation ─────────────────────────────────────────────────────

    @staticmethod
    def validate_weather_event(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for f in ("fetched_at_utc", "port", "latitude", "longitude"):
            if row.get(f) is None:
                errors.append(f"{f} is required")
        for f in ("latitude", "longitude"):
            if f in row and row[f] is not None:
                try:
                    float(row[f])
                except (TypeError, ValueError):
                    errors.append(f"{f} must be numeric")
        return len(errors) == 0, errors

    @staticmethod
    def validate_freight_signal(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for f in ("signal_date", "series_id", "series_name", "value", "source"):
            if row.get(f) is None:
                errors.append(f"{f} is required")
        try:
            if float(row.get("value", 0)) <= 0:
                errors.append("value must be positive")
        except (TypeError, ValueError):
            errors.append("value must be numeric")
        return len(errors) == 0, errors

    @staticmethod
    def validate_news_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for f in ("headline", "source"):
            if not row.get(f):
                errors.append(f"{f} is required")
        if len(row.get("headline", "")) > 1000:
            errors.append("headline exceeds 1000 chars")
        return len(errors) == 0, errors

    # ── Conflict detection (live_enrichment) ──────────────────────────────────

    @staticmethod
    def check_enrichment_conflict(port: str, new_row: Dict[str, Any]) -> None:
        """Log a warning if key signals shift dramatically vs. the previous row."""
        try:
            rows = execute_query(
                "SELECT weather_severity_live, supply_disruption_index_live "
                "FROM live_enrichment WHERE port = ? "
                "ORDER BY enrichment_ts_utc DESC LIMIT 1",
                (port,),
            )
            if not rows:
                return
            prev_weather = rows[0][0]
            prev_sdi = rows[0][1]
            new_weather = new_row.get("weather_severity_live")
            new_sdi = new_row.get("supply_disruption_index_live")
            if prev_weather and new_weather and abs(new_weather - prev_weather) > 0.5:
                logger.warning(
                    "live_enrichment conflict for %s: weather_severity %.3f → %.3f",
                    port, prev_weather, new_weather,
                )
            if prev_sdi and new_sdi and prev_sdi > 0:
                shift_pct = abs(new_sdi - prev_sdi) / prev_sdi * 100
                if shift_pct > 30:
                    logger.warning(
                        "live_enrichment conflict for %s: supply_disruption_index %.3f → %.3f (%.1f%% shift)",
                        port, prev_sdi, new_sdi, shift_pct,
                    )
        except Exception as exc:
            logger.debug("Conflict check failed for %s: %s", port, exc)
