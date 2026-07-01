"""
Ingestion schema — CREATE TABLE statements for all live-enrichment tables.

These tables are written by DataIngestionAgent and read by data_ingestion_agent_v2().
They never overlap with lite_master, which is immutable after ETL build.
"""

import logging
from typing import Optional

from src.utils.db_utils import execute_query, get_connection

logger = logging.getLogger(__name__)

_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS weather_events (
        id                            INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at_utc                TEXT NOT NULL,
        port                          TEXT NOT NULL,
        latitude                      REAL NOT NULL,
        longitude                     REAL NOT NULL,
        windspeed_10m_max_kmh         REAL,
        precipitation_mm_6h           REAL,
        temperature_2m_c              REAL,
        weathercode                   INTEGER,
        flood_risk_flag               INTEGER DEFAULT 0,
        storm_flag                    INTEGER DEFAULT 0,
        derived_weather_severity      REAL,
        derived_natural_disaster_risk REAL,
        source                        TEXT DEFAULT 'open-meteo',
        is_active                     INTEGER DEFAULT 1,
        UNIQUE(fetched_at_utc, port)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_weather_events_port_ts ON weather_events(port, fetched_at_utc DESC)",
    """
    CREATE TABLE IF NOT EXISTS freight_signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at_utc  TEXT NOT NULL,
        signal_date     TEXT NOT NULL,
        source          TEXT NOT NULL,
        series_id       TEXT NOT NULL,
        series_name     TEXT NOT NULL,
        value           REAL NOT NULL,
        unit            TEXT,
        route           TEXT,
        normalized_sdi  REAL,
        is_active       INTEGER DEFAULT 1,
        UNIQUE(signal_date, series_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_freight_signals_date ON freight_signals(signal_date, series_id)",
    """
    CREATE TABLE IF NOT EXISTS supplier_risk_events (
        id                                  INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at_utc                      TEXT NOT NULL,
        event_date                          TEXT,
        source                              TEXT NOT NULL,
        headline                            TEXT NOT NULL,
        summary_snippet                     TEXT,
        canonical_port                      TEXT,
        category                            TEXT,
        severity_score                      REAL,
        disruption_news_count_contribution  INTEGER DEFAULT 1,
        article_hash                        TEXT UNIQUE,
        is_active                           INTEGER DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_supplier_risk_events_date ON supplier_risk_events(event_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_supplier_risk_events_port ON supplier_risk_events(canonical_port, event_date DESC)",
    """
    CREATE TABLE IF NOT EXISTS market_demand_signals (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at_utc                  TEXT NOT NULL,
        signal_date                     TEXT NOT NULL,
        source                          TEXT NOT NULL,
        series_id                       TEXT NOT NULL,
        series_name                     TEXT NOT NULL,
        category                        TEXT,
        value                           REAL NOT NULL,
        pct_change                      REAL,
        normalized_chip_price_index     REAL,
        normalized_market_growth_rate   REAL,
        is_active                       INTEGER DEFAULT 1,
        UNIQUE(signal_date, series_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_market_demand_date ON market_demand_signals(signal_date, series_id)",
    """
    CREATE TABLE IF NOT EXISTS news_disruptions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        fetched_at_utc      TEXT NOT NULL,
        published_at        TEXT,
        source              TEXT NOT NULL,
        headline            TEXT NOT NULL,
        url                 TEXT,
        canonical_port      TEXT,
        risk_categories     TEXT,
        gdelt_tone          REAL,
        severity_score      REAL,
        article_hash        TEXT UNIQUE,
        is_active           INTEGER DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_news_disruptions_pub ON news_disruptions(published_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_news_disruptions_port ON news_disruptions(canonical_port, published_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_news_disruptions_sev ON news_disruptions(severity_score DESC)",
    """
    CREATE TABLE IF NOT EXISTS live_enrichment (
        id                              INTEGER PRIMARY KEY AUTOINCREMENT,
        enrichment_ts_utc               TEXT NOT NULL,
        port                            TEXT NOT NULL,
        sku                             TEXT,
        signal_date                     TEXT NOT NULL,
        weather_severity_live           REAL,
        natural_disaster_risk_live      REAL,
        supply_disruption_index_live    REAL,
        disruption_news_count_live      INTEGER,
        chip_price_index_live           REAL,
        market_growth_rate_live         REAL,
        export_control_flag             INTEGER DEFAULT 0,
        agent_run_id                    TEXT,
        is_consumed                     INTEGER DEFAULT 0,
        UNIQUE(port, signal_date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_live_enrichment_port ON live_enrichment(port, signal_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_live_enrichment_unconsumed ON live_enrichment(is_consumed, enrichment_ts_utc DESC)",
    """
    CREATE TABLE IF NOT EXISTS ingestion_run_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              TEXT NOT NULL,
        run_ts_utc          TEXT NOT NULL,
        source              TEXT NOT NULL,
        connector_class     TEXT NOT NULL,
        rows_fetched        INTEGER DEFAULT 0,
        rows_inserted       INTEGER DEFAULT 0,
        rows_skipped        INTEGER DEFAULT 0,
        duration_ms         INTEGER,
        status              TEXT NOT NULL,
        error_detail        TEXT,
        last_fetched_key    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ingestion_run_log_ts ON ingestion_run_log(run_ts_utc DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_run_log_source ON ingestion_run_log(source, run_ts_utc DESC)",
    # ── v2 tables: written by L1, read by L2/L3 via run_id (auditability / replay) ──
    """
    CREATE TABLE IF NOT EXISTS live_news_ingest (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              TEXT NOT NULL,
        fetched_at_utc      TEXT NOT NULL,
        source_feed         TEXT NOT NULL,
        query_term          TEXT,
        hub_city            TEXT,
        hub_country         TEXT,
        supplier_country    TEXT,
        headline            TEXT NOT NULL,
        summary             TEXT,
        published_at        TEXT,
        url                 TEXT,
        relevance_score     REAL NOT NULL DEFAULT 0.0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_live_news_run ON live_news_ingest(run_id, hub_country)",
    "CREATE INDEX IF NOT EXISTS idx_live_news_published ON live_news_ingest(published_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS live_weather_ingest (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id              TEXT NOT NULL,
        fetched_at_utc      TEXT NOT NULL,
        hub_city            TEXT NOT NULL,
        latitude            REAL NOT NULL,
        longitude           REAL NOT NULL,
        wind_speed_kmh      REAL,
        precipitation_mm    REAL,
        weather_code        INTEGER,
        temperature_c       REAL,
        raw_severity_score  REAL NOT NULL DEFAULT 0.0,
        is_trigger_hub      INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_live_weather_run ON live_weather_ingest(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_live_weather_city ON live_weather_ingest(hub_city, fetched_at_utc DESC)",
]


def ensure_ingestion_schema() -> None:
    """Idempotent: create all ingestion tables and indices if they don't exist."""
    with get_connection() as conn:
        for ddl in _DDL_STATEMENTS:
            conn.execute(ddl)
    logger.info("Ingestion schema verified.")


def get_last_fetched_key(source: str) -> Optional[str]:
    """Return the most recent last_fetched_key for a source from ingestion_run_log."""
    try:
        rows = execute_query(
            """
            SELECT last_fetched_key FROM ingestion_run_log
            WHERE source = ? AND status IN ('success', 'partial')
              AND last_fetched_key IS NOT NULL
            ORDER BY run_ts_utc DESC LIMIT 1
            """,
            (source,),
        )
        return rows[0][0] if rows else None
    except Exception:
        return None
