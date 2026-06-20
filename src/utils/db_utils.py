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
