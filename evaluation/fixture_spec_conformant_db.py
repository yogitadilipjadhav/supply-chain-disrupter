"""
fixture_spec_conformant_db.py
==============================
Purpose  : Build a minimal, spec-conformant SQLite test fixture for QA tasks
           that require database access (QA-04a and QA-08a) when the real
           supply_chain_lite_master.xlsx workbook is not available.

When to use
-----------
Run this script ONLY when the real workbook is unavailable.
When the workbook IS present, run scripts/build_databases.py (or
src/utils/etl_loader.load_excel_into_sqlite()) instead — those populate
the full 5,459-row lite_master and all supporting tables.

What this script creates
------------------------
outputs/supply_chain.db with:
  lite_master          — 4 synthetic rows seeded to exactly span the spec bounds
  daily_records        — VIEW over lite_master (identical DDL to etl_loader.py)
  semiconductor_signals — 2 minimal rows for SDI fallback queries
  risk_classifications — empty audit table (same schema as db_utils.py)

Spec-declared normalization bounds reproduced exactly:
  weather_severity_hub:    [1.18, 10.0]
  natural_disaster_risk:   [1.18, 10.0]
  supply_disruption_index: [4.09, 9.97]
  defect_rate_pct:         [2.00, 19.82]
  disruption_news_count:   [0,    17]

Synthetic rows include:
  • A spec-minimum-boundary row  (all low values)
  • A spec-maximum-boundary row  (all high values)
  • A 'Shipping canceled' CRITICAL row — used by QA-04a (order_id=20001,
    stored_composite=0.8350, disruption_event_label='CRITICAL')
  • A 'Late delivery' HIGH row   — general coverage

Usage
-----
  python evaluation/fixture_spec_conformant_db.py
"""

import os
import sqlite3
import sys

sys.path.insert(0, ".")  # allow imports from project root

DB_PATH = "outputs/supply_chain.db"


def drop_existing_db() -> None:
    """Remove any existing DB file so we always start from a clean state."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing DB: {DB_PATH}")


def create_schema(conn: sqlite3.Connection) -> None:
    """
    Create all tables and views required by the Risk Classifier agent.
    The daily_records VIEW definition is identical to etl_loader._create_schema()
    so that QA scripts work against either the synthetic or real DB.
    """
    conn.executescript("""
    PRAGMA journal_mode = WAL;

    CREATE TABLE lite_master (
        record_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        order_date                   TEXT NOT NULL,
        order_id                     INTEGER NOT NULL,
        order_region                 TEXT,
        category_name                TEXT NOT NULL,
        year                         INTEGER,
        order_city                   TEXT,
        latitude                     REAL,
        longitude                    REAL,
        weather_severity_hub         REAL,   -- range 1.18–10.0 (spec-declared)
        natural_disaster_risk        REAL,   -- range 1.18–10.0 (spec-declared)
        delivery_status              TEXT,   -- 'Shipping canceled'|'Late delivery'|'Advance shipping'|'Shipping on time'
        disruption_event_label       TEXT,   -- LOW|HIGH|CRITICAL (historical ground truth)
        risk_score_composite         REAL,   -- pre-computed composite score (0–1)
        export_control_level         REAL,
        supply_disruption_index      REAL,   -- range 4.09–9.97 (spec-declared)
        product_name                 TEXT,
        order_item_quantity          REAL,
        sales_usd                    REAL,
        unit_price_usd               REAL,
        chip_price_index             REAL,
        market_growth_rate           REAL,
        shipping_mode                TEXT,
        lead_time_variance_days      REAL,
        defect_rate_pct              REAL,   -- range 2.0–19.82 (spec-declared)
        safety_stock_units           REAL,
        stockout_probability_pct     REAL,
        simpy_revenue_impact_p50_usd REAL,
        disruption_news_count        INTEGER, -- range 0–17 (spec-declared)
        alternate_supplier_available INTEGER,
        mitigation_recommendation    TEXT
    );

    -- VIEW that exposes lite_master columns under the names expected by db_utils.py
    -- and risk_classifier_agent.  Column aliases match the real etl_loader VIEW.
    CREATE VIEW daily_records AS
    SELECT
        record_id,
        order_id,
        order_date              AS event_date,
        order_region            AS port,
        product_name            AS sku,
        order_item_quantity     AS demand,
        order_item_quantity     AS import_volume,
        chip_price_index        AS price_index,
        safety_stock_units      AS inventory_level,
        CASE
            WHEN alternate_supplier_available = 1 THEN order_item_quantity
            ELSE 0
        END                     AS incoming_supply,
        lead_time_variance_days AS lead_time_days,
        CASE
            WHEN natural_disaster_risk IS NULL THEN 0
            WHEN natural_disaster_risk > 1 THEN natural_disaster_risk / 10.0
            ELSE natural_disaster_risk
        END                     AS chip_risk,
        COALESCE(stockout_probability_pct, 0) / 100.0 AS supplier_risk,
        risk_score_composite,
        disruption_event_label,
        order_city,
        category_name,
        delivery_status,
        sales_usd,
        unit_price_usd,
        shipping_mode,
        defect_rate_pct,
        mitigation_recommendation,
        latitude,
        longitude,
        supply_disruption_index,
        natural_disaster_risk,
        export_control_level,
        order_region,
        year
    FROM lite_master;

    -- Minimal semiconductor_signals table for SDI fallback in live mode
    CREATE TABLE semiconductor_signals (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        year                    INTEGER,
        country                 TEXT,
        company                 TEXT,
        supply_disruption_index REAL,
        known_severity          TEXT
    );

    -- Audit table written by insert_risk_classification() on every classifier run
    CREATE TABLE risk_classifications (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id          INTEGER,
        mode              TEXT NOT NULL,
        composite_score   REAL NOT NULL,
        geo_component     REAL,
        supply_component  REAL,
        freight_component REAL,
        defect_component  REAL,
        duration_days     REAL,
        base_label        TEXT,
        final_label       TEXT,
        escalated         INTEGER,
        rag_citations     TEXT,
        rationale         TEXT,
        run_ts            TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)


def seed_lite_master(conn: sqlite3.Connection) -> None:
    """
    Insert four synthetic rows into lite_master.
    The rows are designed so that MIN/MAX of each normalised column exactly
    matches the spec-declared bounds.  This ensures _get_norm_bounds() returns
    the same values regardless of whether the real workbook is loaded.

    Column order for the INSERT:
      order_date, order_id, order_region, category_name, year,
      weather_severity_hub, natural_disaster_risk, delivery_status,
      disruption_event_label, risk_score_composite, export_control_level,
      supply_disruption_index, product_name, order_item_quantity,
      chip_price_index, lead_time_variance_days, defect_rate_pct,
      safety_stock_units, stockout_probability_pct, disruption_news_count,
      alternate_supplier_available
    """
    rows = [
        # Row 1: spec minimum-boundary values — sets the MIN for all bounds columns
        ("2020-01-10", 10001, "Eastern Asia",   "Electronics",         2020,
         1.18, 1.18, "Shipping on time",  "LOW",      0.10, 1.5,
         4.09, "CHIP_AP", 100.0, 1.0, 5.0,  2.0,   500.0,  5.0,  0, 1),

        # Row 2: spec maximum-boundary values — sets the MAX for all bounds columns
        ("2020-01-11", 10002, "Western Europe",  "Electronics",         2020,
         10.0, 10.0, "Advance shipping", "LOW",      0.15, 2.5,
         9.97, "CHIP_AP", 200.0, 2.0, 8.0, 19.82, 300.0, 10.0, 17, 0),

        # Row 3: CRITICAL replay row — 'Shipping canceled' with stored composite.
        #   QA-04a queries this row to verify that replay mode returns the stored
        #   composite unchanged and maps 'Shipping canceled' → CRITICAL.
        ("2021-03-15", 20001, "Eastern Asia",   "Electronics",         2021,
         8.50, 9.20, "Shipping canceled", "CRITICAL", 0.8350, 4.2,
         8.75, "CHIP_AP",  50.0, 1.5, 7.0, 15.0,  200.0, 65.0, 12, 0),

        # Row 4: HIGH row — 'Late delivery' for general coverage
        ("2021-04-20", 20002, "Southeast Asia", "Consumer Electronics", 2021,
         5.00, 6.50, "Late delivery",    "HIGH",     0.6100, 3.0,
         7.00, "CHIP_AP", 150.0, 1.2, 6.0, 10.0,  400.0, 40.0,  8, 1),
    ]

    conn.executemany(
        """
        INSERT INTO lite_master (
            order_date, order_id, order_region, category_name, year,
            weather_severity_hub, natural_disaster_risk, delivery_status,
            disruption_event_label, risk_score_composite, export_control_level,
            supply_disruption_index, product_name, order_item_quantity,
            chip_price_index, lead_time_variance_days, defect_rate_pct,
            safety_stock_units, stockout_probability_pct, disruption_news_count,
            alternate_supplier_available
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def seed_semiconductor_signals(conn: sqlite3.Connection) -> None:
    """
    Insert minimal semiconductor_signals rows so live-mode SDI fallback queries
    (used when active_record.supply_disruption_index is None) return a value.
    """
    conn.executemany(
        """
        INSERT INTO semiconductor_signals
            (year, country, company, supply_disruption_index, known_severity)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (2020, "Taiwan", "TSMC", 5.0, "LOW"),
            (2021, "Taiwan", "TSMC", 8.5, "HIGH"),
        ],
    )


def verify_fixture(conn: sqlite3.Connection) -> None:
    """
    Print a brief verification summary to confirm the fixture was built correctly.
    Checks row counts and the actual MIN/MAX bounds against the spec.
    """
    n = conn.execute("SELECT COUNT(*) FROM lite_master").fetchone()[0]
    canceled = conn.execute(
        "SELECT COUNT(*) FROM lite_master WHERE delivery_status = 'Shipping canceled'"
    ).fetchone()[0]
    bounds = conn.execute(
        """
        SELECT
            MIN(weather_severity_hub),  MAX(weather_severity_hub),
            MIN(natural_disaster_risk), MAX(natural_disaster_risk),
            MIN(supply_disruption_index), MAX(supply_disruption_index),
            MIN(defect_rate_pct),       MAX(defect_rate_pct),
            MIN(disruption_news_count), MAX(disruption_news_count)
        FROM lite_master
        """
    ).fetchone()

    print(f"lite_master rows inserted:   {n}")
    print(f"'Shipping canceled' rows:    {canceled}")
    print()
    print("Normalization bounds (actual vs spec):")
    print(f"  weather_severity_hub:    [{bounds[0]:.2f}, {bounds[1]:.2f}]   spec: [1.18, 10.0]")
    print(f"  natural_disaster_risk:   [{bounds[2]:.2f}, {bounds[3]:.2f}]   spec: [1.18, 10.0]")
    print(f"  supply_disruption_index: [{bounds[4]:.2f}, {bounds[5]:.2f}]   spec: [4.09, 9.97]")
    print(f"  defect_rate_pct:         [{bounds[6]:.2f}, {bounds[7]:.2f}]  spec: [2.00, 19.82]")
    print(f"  disruption_news_count:   [{bounds[8]}, {bounds[9]}]          spec: [0, 17]")
    print()
    print("Synthetic fixture ready — use for QA-04a and QA-08a.")


def main() -> None:
    """Entry point: drop → create schema → seed → verify."""
    os.makedirs("outputs", exist_ok=True)
    drop_existing_db()

    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        seed_lite_master(conn)
        seed_semiconductor_signals(conn)
        conn.commit()
        verify_fixture(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
