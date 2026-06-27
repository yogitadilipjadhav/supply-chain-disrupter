from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.db_utils import DB_PATH


EXCEL_SOURCE = Path("data/raw/supply_chain_lite_master.xlsx")
EXPECTED_SHEETS = {
    "Lite Master",
    "Column Guide (Lite)",
    "Legend",
    "Ops KPI (Filled)",
    "Semiconductor Signals",
}
ALLOWED_CATEGORIES = {
    "Electronics",
    "Consumer Electronics",
    "Computers",
    "Cameras",
    "Video Games",
}

# The DataCo source dataset mislabels sports/fashion products (golf balls, shoes,
# NFL merchandise) as "Electronics". Only these four categories contain genuine
# electronics products and are loaded into lite_master.
GENUINE_ELECTRONICS_CATEGORIES = {
    "Consumer Electronics",
    "Computers",
    "Cameras",
    "Video Games",
}
BEAUTY_TERMS = {
    "beauty",
    "cosmetic",
    "cosmetics",
    "skincare",
    "skin care",
    "makeup",
    "personal care",
    "haircare",
    "hair care",
}


def read_excel_sheets(excel_path: Path = EXCEL_SOURCE) -> Dict[str, pd.DataFrame]:
    """Read every Varun electronics sheet with its actual header row."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    workbook = pd.ExcelFile(excel_path)
    missing = EXPECTED_SHEETS.difference(workbook.sheet_names)
    if missing:
        raise ValueError(f"Missing expected workbook sheets: {sorted(missing)}")

    return {
        "Lite Master": pd.read_excel(excel_path, sheet_name="Lite Master", header=1),
        "Column Guide (Lite)": pd.read_excel(
            excel_path, sheet_name="Column Guide (Lite)", header=0
        ),
        "Legend": pd.read_excel(excel_path, sheet_name="Legend", header=None),
        "Ops KPI (Filled)": pd.read_excel(
            excel_path, sheet_name="Ops KPI (Filled)", header=2
        ),
        "Semiconductor Signals": pd.read_excel(
            excel_path, sheet_name="Semiconductor Signals", header=1
        ),
    }


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return [
        tuple(_clean_scalar(value) for value in row)
        for row in df[columns].itertuples(index=False, name=None)
    ]


def _validate_varun_dataset(sheets: Dict[str, pd.DataFrame]) -> None:
    master = sheets["Lite Master"]
    categories = {
        str(value).strip()
        for value in master["Category_Name"].dropna().unique().tolist()
    }
    disallowed = categories.difference(ALLOWED_CATEGORIES)
    if disallowed:
        raise ValueError(
            "Workbook contains non-electronics categories; refusing mixed-domain load: "
            f"{sorted(disallowed)}"
        )
    category_text = " ".join(categories).lower()
    found_beauty_terms = sorted(
        term for term in BEAUTY_TERMS if term in category_text
    )
    if found_beauty_terms:
        raise ValueError(
            "Workbook contains beauty/cosmetics categories; refusing load: "
            f"{found_beauty_terms}"
        )
    if master["Order_ID"].isna().any():
        raise ValueError("Lite Master contains null Order_ID values")


def _filter_genuine_electronics(master: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows whose Category_Name is not in GENUINE_ELECTRONICS_CATEGORIES.

    The DataCo source dataset categorises sports/fashion products (golf balls,
    Under Armour shoes, NFL merchandise) under 'Electronics'. Removing that
    category keeps only rows backed by real semiconductor supply-chain signals.
    """
    before = len(master)
    filtered = master[master["Category_Name"].isin(GENUINE_ELECTRONICS_CATEGORIES)].copy()
    removed = before - len(filtered)
    if removed:
        import logging
        logging.getLogger(__name__).info(
            "ETL: removed %d non-electronics rows (category 'Electronics' excluded). "
            "Remaining: %d",
            removed,
            len(filtered),
        )
    return filtered


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;

        CREATE TABLE ingestion_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE lite_master (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT NOT NULL,
            order_id INTEGER NOT NULL,
            order_region TEXT,
            category_name TEXT NOT NULL,
            year INTEGER,
            order_city TEXT,
            latitude REAL,
            longitude REAL,
            weather_severity_hub REAL,
            natural_disaster_risk REAL,
            delivery_status TEXT,
            disruption_event_label TEXT,
            risk_score_composite REAL,
            export_control_level REAL,
            supply_disruption_index REAL,
            product_name TEXT,
            order_item_quantity REAL,
            sales_usd REAL,
            unit_price_usd REAL,
            chip_price_index REAL,
            market_growth_rate REAL,
            shipping_mode TEXT,
            lead_time_variance_days REAL,
            defect_rate_pct REAL,
            safety_stock_units REAL,
            stockout_probability_pct REAL,
            simpy_revenue_impact_p50_usd REAL,
            disruption_news_count INTEGER,
            alternate_supplier_available INTEGER,
            mitigation_recommendation TEXT
        );

        CREATE TABLE ops_kpi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT,
            sku_id TEXT,
            region TEXT,
            supplier_region TEXT,
            price_usd REAL,
            promo_flag INTEGER,
            weather_index REAL,
            disruption_flag INTEGER,
            stockout_flag INTEGER,
            demand_actual REAL,
            forecast_baseline REAL,
            forecast_ai REAL,
            mape_baseline REAL,
            mape_ai REAL,
            mape_improvement_pct REAL,
            lead_time_baseline_days REAL,
            lead_time_after_days REAL,
            lead_time_saved_days REAL,
            otr_baseline REAL,
            otr_after REAL,
            cost_baseline_usd REAL,
            cost_after_usd REAL,
            cost_saving_usd REAL,
            disruption_event_label TEXT,
            agent_triggered TEXT,
            intervention_date TEXT
        );

        CREATE TABLE semiconductor_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER,
            country TEXT,
            company TEXT,
            production_capacity_wafers REAL,
            fab_count INTEGER,
            technology_node_nm TEXT,
            ai_chip_production REAL,
            foundry_revenue_usd REAL,
            global_market_share REAL,
            export_control_level REAL,
            sanctions_index REAL,
            trade_tension_level REAL,
            semiconductor_security_risk REAL,
            cn_export_control REAL,
            cn_security_risk REAL,
            natural_disaster_risk REAL,
            factory_shutdown_risk REAL,
            supply_disruption_index REAL,
            global_semiconductor_revenue REAL,
            ai_chip_revenue REAL,
            chip_price_index REAL,
            market_growth_rate REAL,
            known_disruption_event TEXT,
            known_severity TEXT
        );

        CREATE TABLE column_guide (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT,
            column_name TEXT,
            source_type TEXT,
            purpose TEXT
        );

        CREATE TABLE workbook_legend (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item TEXT,
            description TEXT
        );

        CREATE TABLE mitigation_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT,
            port TEXT,
            sku TEXT,
            risk_label TEXT,
            recommendation TEXT,
            cost_delta TEXT,
            inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIEW daily_records AS
        SELECT
            record_id,
            order_id,
            order_date AS event_date,
            order_region AS port,
            product_name AS sku,
            order_item_quantity AS demand,
            order_item_quantity AS import_volume,
            chip_price_index AS price_index,
            safety_stock_units AS inventory_level,
            CASE
                WHEN alternate_supplier_available = 1 THEN order_item_quantity
                ELSE 0
            END AS incoming_supply,
            lead_time_variance_days AS lead_time_days,
            CASE
                WHEN natural_disaster_risk IS NULL THEN 0
                WHEN natural_disaster_risk > 1 THEN natural_disaster_risk / 10.0
                ELSE natural_disaster_risk
            END AS chip_risk,
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
            year,
            supply_disruption_index,
            natural_disaster_risk,
            export_control_level,
            order_region
        FROM lite_master;

        CREATE INDEX idx_lite_master_date ON lite_master(order_date);
        CREATE INDEX idx_lite_master_order_id ON lite_master(order_id);
        CREATE INDEX idx_lite_master_region ON lite_master(order_region);
        CREATE INDEX idx_lite_master_product ON lite_master(product_name);
        CREATE INDEX idx_lite_master_label ON lite_master(disruption_event_label);
        CREATE INDEX idx_ops_kpi_week_sku ON ops_kpi(week_start, sku_id);
        CREATE INDEX idx_semiconductor_year_company
            ON semiconductor_signals(year, company);
        CREATE INDEX idx_semiconductor_event
            ON semiconductor_signals(known_disruption_event);
        """
    )


def _insert_frame(
    conn: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    source_columns: list[str],
    target_columns: list[str],
) -> None:
    placeholders = ",".join("?" for _ in target_columns)
    column_sql = ",".join(target_columns)
    conn.executemany(
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})",
        _rows(frame, source_columns),
    )


def load_excel_into_sqlite(
    flush_existing: bool = True,
    excel_path: Path = EXCEL_SOURCE,
    db_path: Path = DB_PATH,
) -> int:
    """Build a complete, lossless SQLite database from Varun's workbook."""
    sheets = read_excel_sheets(excel_path)
    _validate_varun_dataset(sheets)
    sheets["Lite Master"] = _filter_genuine_electronics(sheets["Lite Master"])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_suffix(".building.db")
    if temp_path.exists():
        temp_path.unlink()

    conn = sqlite3.connect(temp_path)
    try:
        _create_schema(conn)

        master_source = [
            "Order_Date", "Order_ID", "Order_Region", "Category_Name", "Year",
            "Order_City", "Latitude", "Longitude", "Weather_Severity_Hub",
            "Natural_Disaster_Risk", "Delivery_Status", "Disruption_Event_Label",
            "Risk_Score_Composite", "Export_Control_Level",
            "Supply_Disruption_Index", "Product_Name", "Order_Item_Quantity",
            "Sales_USD", "Unit_Price_USD", "Chip_Price_Index",
            "Market_Growth_Rate", "Shipping_Mode", "Lead_Time_Variance_Days",
            "Defect_Rate_Pct", "Safety_Stock_Units",
            "Stockout_Probability_Pct", "SimPy_Revenue_Impact_P50_USD",
            "Disruption_News_Count", "Alternate_Supplier_Available",
            "Mitigation_Recommendation",
        ]
        master_target = [
            "order_date", "order_id", "order_region", "category_name", "year",
            "order_city", "latitude", "longitude", "weather_severity_hub",
            "natural_disaster_risk", "delivery_status",
            "disruption_event_label", "risk_score_composite",
            "export_control_level", "supply_disruption_index", "product_name",
            "order_item_quantity", "sales_usd", "unit_price_usd",
            "chip_price_index", "market_growth_rate", "shipping_mode",
            "lead_time_variance_days", "defect_rate_pct", "safety_stock_units",
            "stockout_probability_pct", "simpy_revenue_impact_p50_usd",
            "disruption_news_count", "alternate_supplier_available",
            "mitigation_recommendation",
        ]
        _insert_frame(
            conn, "lite_master", sheets["Lite Master"], master_source, master_target
        )

        ops = sheets["Ops KPI (Filled)"].dropna(how="all")
        ops_columns = list(ops.columns)
        _insert_frame(
            conn,
            "ops_kpi",
            ops,
            ops_columns,
            [
                "week_start", "sku_id", "region", "supplier_region", "price_usd",
                "promo_flag", "weather_index", "disruption_flag", "stockout_flag",
                "demand_actual", "forecast_baseline", "forecast_ai",
                "mape_baseline", "mape_ai", "mape_improvement_pct",
                "lead_time_baseline_days", "lead_time_after_days",
                "lead_time_saved_days", "otr_baseline", "otr_after",
                "cost_baseline_usd", "cost_after_usd", "cost_saving_usd",
                "disruption_event_label", "agent_triggered", "intervention_date",
            ],
        )

        signals = sheets["Semiconductor Signals"].dropna(how="all")
        signal_columns = list(signals.columns)
        _insert_frame(
            conn,
            "semiconductor_signals",
            signals,
            signal_columns,
            [
                "year", "country", "company", "production_capacity_wafers",
                "fab_count", "technology_node_nm", "ai_chip_production",
                "foundry_revenue_usd", "global_market_share",
                "export_control_level", "sanctions_index", "trade_tension_level",
                "semiconductor_security_risk", "cn_export_control",
                "cn_security_risk", "natural_disaster_risk",
                "factory_shutdown_risk", "supply_disruption_index",
                "global_semiconductor_revenue", "ai_chip_revenue",
                "chip_price_index", "market_growth_rate",
                "known_disruption_event", "known_severity",
            ],
        )

        guide = sheets["Column Guide (Lite)"].dropna(how="all")
        _insert_frame(
            conn,
            "column_guide",
            guide,
            ["Agent", "Column", "Source Type", "Purpose"],
            ["agent", "column_name", "source_type", "purpose"],
        )

        legend = sheets["Legend"].dropna(how="all")
        legend.columns = ["item", "description"]
        _insert_frame(
            conn,
            "workbook_legend",
            legend,
            ["item", "description"],
            ["item", "description"],
        )

        metadata = {
            "source_file": str(excel_path.resolve()),
            "source_sha256": _sha256(excel_path),
            "domain": "electronics_semiconductors",
            "dataset_owner": "Varun",
            "beauty_products_included": "false",
            "electronics_only": "true",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "lite_master_rows": str(len(sheets["Lite Master"])),
            "ops_kpi_rows": str(len(ops)),
            "semiconductor_signal_rows": str(len(signals)),
        }
        conn.executemany(
            "INSERT INTO ingestion_metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        conn.commit()

        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")

        # Step 6 — update column_guide to reflect the correct 4-tier label set
        conn.execute(
            """
            UPDATE column_guide
            SET purpose = REPLACE(purpose, 'LOW / HIGH / CRITICAL', 'LOW / MEDIUM / HIGH / CRITICAL')
            WHERE purpose LIKE '%LOW / HIGH / CRITICAL%'
            """
        )
        conn.commit()
    except Exception:
        conn.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
    else:
        conn.close()

    if db_path.exists():
        db_path.unlink()
    os.replace(temp_path, db_path)
    return len(sheets["Lite Master"])


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_sqlite_stats(db_path: Path = DB_PATH) -> Dict[str, Any]:
    if not db_path.exists():
        return {"database_exists": False}

    with sqlite3.connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "lite_master",
                "ops_kpi",
                "semiconductor_signals",
                "column_guide",
                "workbook_legend",
                "daily_records",
            )
        }
        date_range = conn.execute(
            "SELECT MIN(order_date), MAX(order_date) FROM lite_master"
        ).fetchone()
        categories = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT category_name FROM lite_master ORDER BY category_name"
            )
        ]
        products = conn.execute(
            "SELECT COUNT(DISTINCT product_name) FROM lite_master"
        ).fetchone()[0]

    return {
        "database_exists": True,
        "tables": counts,
        "date_range": f"{date_range[0]} to {date_range[1]}",
        "categories": categories,
        "unique_products": products,
        "beauty_products_included": False,
        "size_mb": round(db_path.stat().st_size / 1_048_576, 2),
    }
