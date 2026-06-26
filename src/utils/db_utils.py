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


def ensure_risk_classification_table() -> None:
    """Create risk_classifications table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_classifications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id        INTEGER,
                mode            TEXT NOT NULL,
                composite_score REAL NOT NULL,
                geo_component   REAL,
                supply_component REAL,
                freight_component REAL,
                defect_component REAL,
                duration_days   REAL,
                base_label      TEXT,
                final_label     TEXT,
                escalated       INTEGER,
                rag_citations   TEXT,
                rationale       TEXT,
                run_ts          TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def insert_risk_classification(
    order_id: Optional[int],
    mode: str,
    composite_score: float,
    geo_component: float,
    supply_component: float,
    freight_component: float,
    defect_component: float,
    duration_days: Optional[float],
    base_label: str,
    final_label: str,
    escalated: bool,
    rag_citations: List[str],
    rationale: str,
) -> None:
    import json as _json
    ensure_risk_classification_table()
    execute_non_query(
        """
        INSERT INTO risk_classifications
          (order_id, mode, composite_score, geo_component, supply_component,
           freight_component, defect_component, duration_days, base_label,
           final_label, escalated, rag_citations, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, mode, round(composite_score, 4),
            round(geo_component, 4), round(supply_component, 4),
            round(freight_component, 4), round(defect_component, 4),
            duration_days, base_label, final_label,
            int(escalated), _json.dumps(rag_citations), rationale,
        ),
    )


def fetch_scenario_options() -> List[Dict[str, Any]]:
    """Return valid region/product/date combinations with forecast history.

    Only genuine electronics categories are included. The DataCo source dataset
    labels sports/fashion products (golf balls, shoes) under 'Electronics';
    that category is excluded here. The clean categories are Cameras, Computers,
    Consumer Electronics, and Video Games.
    """
    rows = execute_query(
        """
        SELECT
            port,
            sku,
            MAX(event_date) AS event_date,
            COUNT(DISTINCT event_date) AS history_points
        FROM daily_records
        WHERE port IS NOT NULL
          AND sku IS NOT NULL
          AND category_name IN (
              'Cameras', 'Computers', 'Consumer Electronics', 'Video Games'
          )
        GROUP BY port, sku
        HAVING COUNT(DISTINCT event_date) >= 3
        ORDER BY history_points DESC, port, sku
        """
    )
    return [dict(row) for row in rows]
