"""
Live Data Feed dashboard page.

Provides:
  - Manual "Run Now" button to trigger a full connector sweep
  - Auto-scheduler toggle (hourly background refresh)
  - Ingestion run log table
  - Per-port live enrichment status
  - External signal sparklines
"""

import streamlit as st

from src.utils.db_utils import execute_query
from src.utils.ingestion_schema import ensure_ingestion_schema


def _get_scheduler():
    """Return (and lazily start) the IngestionScheduler stored in session_state."""
    if "ingestion_scheduler" not in st.session_state:
        from src.agents.data_ingestion_agent import DataIngestionAgent
        from src.utils.ingestion_scheduler import IngestionScheduler
        scheduler = IngestionScheduler(
            agent_factory=DataIngestionAgent,
            interval_seconds=3600,
        )
        st.session_state["ingestion_scheduler"] = scheduler
    return st.session_state["ingestion_scheduler"]


def show_ingestion_dashboard() -> None:
    st.title("Live Data Feed")
    st.caption(
        "Fetches real-world signals from Open-Meteo, FRED, GDELT, Reuters, "
        "BIS Federal Register, and (optionally) yfinance. "
        "Signals are persisted to SQLite enrichment tables and overlaid on "
        "Scenario Analyzer runs without modifying historical order data."
    )

    # Ensure schema exists before any DB reads
    try:
        ensure_ingestion_schema()
    except Exception as exc:
        st.error(f"Failed to verify ingestion schema: {exc}")
        return

    scheduler = _get_scheduler()

    # ── Controls ──────────────────────────────────────────────────────────────
    col_run, col_auto, col_status = st.columns([1, 1, 2])

    with col_run:
        if st.button("Run Now", type="primary"):
            from src.agents.data_ingestion_agent import DataIngestionAgent
            agent = DataIngestionAgent()
            with st.spinner("Fetching external signals…"):
                result = agent.run_batch()
            if result.status == "success":
                st.success(f"Inserted {result.total_rows_inserted} rows across {len(result.connectors_run)} connectors.")
            elif result.status == "partial":
                st.warning(
                    f"Partial success: {result.total_rows_inserted} inserted, "
                    f"{len(result.errors)} connector(s) failed."
                )
            elif result.status == "skipped":
                st.info("Another ingestion run is already in progress.")
            else:
                st.error(f"Ingestion failed: {result.errors}")
            if result.errors:
                with st.expander("Connector errors"):
                    for err in result.errors:
                        st.text(err)

    with col_auto:
        auto_enabled = st.toggle(
            "Auto-refresh (hourly)",
            value=scheduler.is_running,
        )
        if auto_enabled and not scheduler.is_running:
            scheduler.start()
            st.success("Auto-refresh started.")
        elif not auto_enabled and scheduler.is_running:
            scheduler.stop()
            st.info("Auto-refresh stopped.")

    with col_status:
        last = scheduler.last_result
        if last:
            st.metric("Last auto-run", last.completed_at_utc[:19].replace("T", " ") + " UTC")
            st.caption(
                f"Status: {last.status} · "
                f"Inserted: {last.total_rows_inserted} · "
                f"Skipped: {last.total_rows_skipped}"
            )
        else:
            st.metric("Last auto-run", "Not yet run")

    st.divider()

    # ── Live Enrichment Status ────────────────────────────────────────────────
    st.subheader("Current Live Signals (per port)")
    try:
        enrichment_rows = execute_query(
            """
            SELECT port, signal_date, weather_severity_live,
                   natural_disaster_risk_live, supply_disruption_index_live,
                   disruption_news_count_live, chip_price_index_live,
                   export_control_flag, enrichment_ts_utc
            FROM live_enrichment
            ORDER BY port, enrichment_ts_utc DESC
            """,
        )
        if enrichment_rows:
            seen_ports = set()
            table_data = []
            for row in enrichment_rows:
                port = row[0]
                if port in seen_ports:
                    continue
                seen_ports.add(port)
                table_data.append({
                    "Port": port,
                    "Signal Date": row[1],
                    "Weather Severity": f"{row[2]:.3f}" if row[2] is not None else "—",
                    "Natural Disaster Risk": f"{row[3]:.2f}" if row[3] is not None else "—",
                    "Supply Disruption SDI": f"{row[4]:.3f}" if row[4] is not None else "—",
                    "News Count (7d)": row[5] if row[5] is not None else "—",
                    "Chip Price Index": f"{row[6]:.3f}" if row[6] is not None else "—",
                    "Export Control": "⚠ Yes" if row[7] else "No",
                    "Updated": row[8][:16].replace("T", " ") + " UTC" if row[8] else "—",
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No live enrichment data yet. Click 'Run Now' to fetch external signals.")
    except Exception as exc:
        st.warning(f"Could not load live enrichment data: {exc}")

    st.divider()

    # ── Hub City Weather Signals ──────────────────────────────────────────────
    st.subheader("Hub City Weather Signals (latest run)")
    try:
        hub_weather_rows = execute_query(
            """
            SELECT hub_city, wind_speed_kmh, precipitation_mm, weather_code,
                   temperature_c, raw_severity_score, is_trigger_hub, fetched_at_utc
            FROM live_weather_ingest
            WHERE run_id = (SELECT run_id FROM live_weather_ingest ORDER BY fetched_at_utc DESC LIMIT 1)
            ORDER BY raw_severity_score DESC
            """,
        )
        if hub_weather_rows:
            hub_weather_data = []
            for row in hub_weather_rows:
                hub_weather_data.append({
                    "Hub City": row[0],
                    "Wind (km/h)": f"{row[1]:.1f}" if row[1] is not None else "—",
                    "Precip (mm)": f"{row[2]:.1f}" if row[2] is not None else "—",
                    "WMO Code": row[3] if row[3] is not None else "—",
                    "Temp (°C)": f"{row[4]:.1f}" if row[4] is not None else "—",
                    "Severity (0-10)": f"{row[5]:.2f}" if row[5] is not None else "—",
                    "Trigger Hub": "🔴 YES" if row[6] else "No",
                    "Fetched (UTC)": row[7][:16].replace("T", " ") if row[7] else "—",
                })
            st.dataframe(hub_weather_data, use_container_width=True)
        else:
            st.info("No hub city weather data yet. Click 'Run Now' to fetch signals.")
    except Exception as exc:
        st.warning(f"Could not load hub city weather data: {exc}")

    st.divider()

    # ── Hub City News Signals ─────────────────────────────────────────────────
    st.subheader("Hub City News Signals (latest run)")
    try:
        hub_news_rows = execute_query(
            """
            SELECT hub_city, hub_country, supplier_country, headline,
                   published_at, relevance_score, query_term
            FROM live_news_ingest
            WHERE run_id = (SELECT run_id FROM live_news_ingest ORDER BY fetched_at_utc DESC LIMIT 1)
            ORDER BY relevance_score DESC
            LIMIT 30
            """,
        )
        if hub_news_rows:
            hub_news_data = []
            for row in hub_news_rows:
                hub_news_data.append({
                    "Hub City": row[0] or "—",
                    "Hub Country": row[1] or "—",
                    "Supplier Country": row[2] or "—",
                    "Headline": row[3],
                    "Published": (row[4] or "")[:16].replace("T", " "),
                    "Relevance": f"{row[5]:.2f}" if row[5] is not None else "—",
                    "Query Term": row[6] or "—",
                })
            st.dataframe(hub_news_data, use_container_width=True)
        else:
            st.info("No hub city news data yet. Click 'Run Now' to fetch signals.")
    except Exception as exc:
        st.warning(f"Could not load hub city news data: {exc}")

    st.divider()

    # ── Recent News Disruptions ───────────────────────────────────────────────
    st.subheader("Recent News & Disruption Signals")
    try:
        news_rows = execute_query(
            """
            SELECT published_at, source, headline, risk_categories, severity_score
            FROM news_disruptions
            WHERE is_active = 1
            ORDER BY published_at DESC LIMIT 25
            """,
        )
        if news_rows:
            import json as _json
            news_data = []
            for row in news_rows:
                cats_raw = row[3]
                try:
                    cats = ", ".join(_json.loads(cats_raw)) if cats_raw else ""
                except Exception:
                    cats = cats_raw or ""
                news_data.append({
                    "Published": (row[0] or "")[:16].replace("T", " "),
                    "Source": row[1],
                    "Headline": row[2],
                    "Categories": cats,
                    "Severity": f"{row[4]:.2f}" if row[4] is not None else "—",
                })
            st.dataframe(news_data, use_container_width=True)
        else:
            st.info("No news disruptions fetched yet.")
    except Exception as exc:
        st.warning(f"Could not load news data: {exc}")

    st.divider()

    # ── Ingestion Run Log ─────────────────────────────────────────────────────
    st.subheader("Ingestion Run Log")
    try:
        log_rows = execute_query(
            """
            SELECT run_ts_utc, source, status, rows_inserted, rows_skipped,
                   duration_ms, error_detail
            FROM ingestion_run_log
            ORDER BY run_ts_utc DESC LIMIT 50
            """,
        )
        if log_rows:
            log_data = []
            for row in log_rows:
                log_data.append({
                    "Timestamp (UTC)": (row[0] or "")[:19].replace("T", " "),
                    "Source": row[1],
                    "Status": row[2],
                    "Inserted": row[3],
                    "Skipped": row[4],
                    "Duration (ms)": row[5],
                    "Error": row[6] or "",
                })
            st.dataframe(log_data, use_container_width=True)
        else:
            st.info("No ingestion runs logged yet.")
    except Exception as exc:
        st.warning(f"Could not load run log: {exc}")
