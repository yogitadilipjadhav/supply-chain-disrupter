import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path("outputs/supply_chain.db")


def get_connection(timeout: int = 30) -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(query: str, params: Tuple = ()) -> List[sqlite3.Row]:
    try:
        with get_connection() as conn:
            return conn.execute(query, params).fetchall()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite query failed: {exc}") from exc


def execute_non_query(query: str, params: Tuple = ()) -> None:
    try:
        with get_connection() as conn:
            conn.execute(query, params)
            conn.commit()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite write failed: {exc}") from exc


def ensure_schema() -> None:
    """Create only the writable agent-output table.

    The complete source schema is created by etl_loader.load_excel_into_sqlite().
    """
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mitigation_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT,
                port TEXT,
                sku TEXT,
                risk_label TEXT,
                recommendation TEXT,
                cost_delta TEXT,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def ensure_ingestion_schema() -> None:
    """Create the live-ingestion tables written by the Data Ingestion agent.

    These hold runtime signals fetched from external APIs (Open-Meteo, GDELT/RSS),
    kept separate from the historical workbook tables built by etl_loader. They are
    idempotent so the live poller can run repeatedly.
    """
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS weather_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hub TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                observation_date TEXT NOT NULL,
                severity REAL,
                wind_score REAL,
                precipitation_score REAL,
                weather_code_score REAL,
                max_wind_speed REAL,
                max_precipitation REAL,
                weather_summary TEXT,
                source_type TEXT DEFAULT 'live_weather',
                ingestion_ts TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (hub, observation_date)
            );

            CREATE TABLE IF NOT EXISTS news_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                summary TEXT,
                url TEXT,
                publisher TEXT,
                published_at TEXT,
                detected_region TEXT,
                detected_category TEXT,
                query_tag TEXT,
                content_hash TEXT UNIQUE,
                source_type TEXT DEFAULT 'live_news',
                ingestion_ts TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_weather_signals_hub
                ON weather_signals(hub, observation_date);
            CREATE INDEX IF NOT EXISTS idx_news_signals_region
                ON news_signals(detected_region);
            CREATE INDEX IF NOT EXISTS idx_news_signals_published
                ON news_signals(published_at);
            """
        )


def upsert_weather_signal(signal: Dict[str, Any]) -> None:
    """Insert or refresh one hub's weather signal for a given observation date."""
    execute_non_query(
        """
        INSERT INTO weather_signals (
            hub, latitude, longitude, observation_date, severity,
            wind_score, precipitation_score, weather_code_score,
            max_wind_speed, max_precipitation, weather_summary,
            source_type, ingestion_ts
        ) VALUES (
            :hub, :latitude, :longitude, :observation_date, :severity,
            :wind_score, :precipitation_score, :weather_code_score,
            :max_wind_speed, :max_precipitation, :weather_summary,
            :source_type, :ingestion_ts
        )
        ON CONFLICT(hub, observation_date) DO UPDATE SET
            severity = excluded.severity,
            wind_score = excluded.wind_score,
            precipitation_score = excluded.precipitation_score,
            weather_code_score = excluded.weather_code_score,
            max_wind_speed = excluded.max_wind_speed,
            max_precipitation = excluded.max_precipitation,
            weather_summary = excluded.weather_summary,
            ingestion_ts = excluded.ingestion_ts
        """,
        signal,
    )


def insert_news_signals(articles: List[Dict[str, Any]]) -> int:
    """Insert news rows, skipping duplicates by content_hash. Returns rows added."""
    if not articles:
        return 0
    inserted = 0
    with get_connection() as conn:
        for article in articles:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO news_signals (
                    title, summary, url, publisher, published_at,
                    detected_region, detected_category, query_tag,
                    content_hash, source_type, ingestion_ts
                ) VALUES (
                    :title, :summary, :url, :publisher, :published_at,
                    :detected_region, :detected_category, :query_tag,
                    :content_hash, :source_type, :ingestion_ts
                )
                """,
                article,
            )
            inserted += cursor.rowcount
        conn.commit()
    return inserted


def fetch_latest_weather_signal(hub: str) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT * FROM weather_signals
        WHERE hub = ?
        ORDER BY observation_date DESC, ingestion_ts DESC
        LIMIT 1
        """,
        (hub,),
    )
    return dict(rows[0]) if rows else None


def fetch_recent_news(
    region: Optional[str] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    if region:
        rows = execute_query(
            """
            SELECT * FROM news_signals
            WHERE detected_region = ?
            ORDER BY published_at DESC, ingestion_ts DESC
            LIMIT ?
            """,
            (region, limit),
        )
    else:
        rows = execute_query(
            """
            SELECT * FROM news_signals
            ORDER BY published_at DESC, ingestion_ts DESC
            LIMIT ?
            """,
            (limit,),
        )
    return [dict(row) for row in rows]


def get_table_count(table: str) -> int:
    rows = execute_query(f"SELECT COUNT(*) AS n FROM {table}")
    return int(rows[0]["n"]) if rows else 0


def fetch_daily_record(
    event_date: str, port: str, sku: str
) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT * FROM daily_records
        WHERE event_date = ? AND port = ? AND sku = ?
        ORDER BY record_id
        LIMIT 1
        """,
        (event_date, port, sku),
    )
    return dict(rows[0]) if rows else None


def fetch_time_series(port: str, sku: str) -> List[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT event_date, SUM(demand) AS demand,
               SUM(import_volume) AS import_volume,
               AVG(price_index) AS price_index
        FROM daily_records
        WHERE port = ? AND sku = ?
        GROUP BY event_date
        ORDER BY event_date
        """,
        (port, sku),
    )
    return [dict(row) for row in rows]


def fetch_inventory_snapshot(port: str, sku: str) -> Optional[Dict[str, Any]]:
    rows = execute_query(
        """
        SELECT inventory_level, incoming_supply, lead_time_days
        FROM daily_records
        WHERE port = ? AND sku = ?
        ORDER BY event_date DESC, record_id DESC
        LIMIT 1
        """,
        (port, sku),
    )
    return dict(rows[0]) if rows else None


def update_risk_label(
    event_date: str,
    port: str,
    sku: str,
    composite_score: float,
    label: str,
) -> None:
    execute_non_query(
        """
        UPDATE lite_master
        SET risk_score_composite = ?, disruption_event_label = ?
        WHERE record_id = (
            SELECT record_id FROM daily_records
            WHERE event_date = ? AND port = ? AND sku = ?
            ORDER BY record_id LIMIT 1
        )
        """,
        (composite_score, label, event_date, port, sku),
    )


def insert_mitigation_action(
    event_date: str,
    port: str,
    sku: str,
    risk_label: str,
    recommendation: str,
    cost_delta: str,
) -> None:
    execute_non_query(
        """
        INSERT INTO mitigation_actions
        (event_date, port, sku, risk_label, recommendation, cost_delta)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (event_date, port, sku, risk_label, recommendation, cost_delta),
    )


def fetch_scenario_options() -> List[Dict[str, Any]]:
    """Return valid region/product/date combinations with forecast history."""
    rows = execute_query(
        """
        SELECT
            port,
            sku,
            MAX(event_date) AS event_date,
            COUNT(DISTINCT event_date) AS history_points
        FROM daily_records
        WHERE port IS NOT NULL AND sku IS NOT NULL
        GROUP BY port, sku
        HAVING COUNT(DISTINCT event_date) >= 10
        ORDER BY port, sku
        """
    )
    return [dict(row) for row in rows]
