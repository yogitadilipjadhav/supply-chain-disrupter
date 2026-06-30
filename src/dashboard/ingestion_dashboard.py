"""
Live Data Feed dashboard page.

Provides:
  - Manual "Run Now" button (background thread — never blocks the UI)
  - Auto-scheduler toggle (hourly background refresh)
  - Hub city weather signals (live_weather_ingest — global semiconductor hubs)
  - Hub city news signals (live_news_ingest — targeted RSS queries)
  - Per-port live enrichment status (Indian delivery ports)
  - Recent broad news disruptions (news_disruptions table)
  - Ingestion run log
"""

import threading
import time

import streamlit as st

from src.utils.db_utils import execute_query
from src.utils.ingestion_schema import ensure_ingestion_schema


# ── Background run helpers ────────────────────────────────────────────────────

def _is_manual_run_active() -> bool:
    thread = st.session_state.get("manual_run_thread")
    return thread is not None and thread.is_alive()


def _start_manual_run() -> None:
    """Launch run_batch() in a daemon thread so the UI stays responsive."""
    from src.agents.data_ingestion_agent import DataIngestionAgent

    def _run() -> None:
        try:
            agent = DataIngestionAgent()
            result = agent.run_batch()
            st.session_state["manual_run_result"] = result
            st.session_state["manual_run_error"] = None
        except Exception as exc:
            st.session_state["manual_run_result"] = None
            st.session_state["manual_run_error"] = str(exc)
        finally:
            st.session_state["manual_run_thread"] = None

    thread = threading.Thread(target=_run, daemon=True, name="manual-ingestion-run")
    st.session_state["manual_run_thread"] = thread
    st.session_state["manual_run_result"] = None
    st.session_state["manual_run_error"] = None
    thread.start()


def _get_scheduler():
    if "ingestion_scheduler" not in st.session_state:
        from src.agents.data_ingestion_agent import DataIngestionAgent
        from src.utils.ingestion_scheduler import IngestionScheduler
        st.session_state["ingestion_scheduler"] = IngestionScheduler(
            agent_factory=DataIngestionAgent,
            interval_seconds=3600,
        )
    return st.session_state["ingestion_scheduler"]


# ── Main page ─────────────────────────────────────────────────────────────────

def show_ingestion_dashboard() -> None:
    st.title("Live Data Feed")
    st.caption(
        "Fetches real-world signals from Open-Meteo (6 global hub cities + Indian ports), "
        "Google News RSS, Reuters RSS, GDELT, FRED, BIS Federal Register, and yfinance. "
        "Signals are persisted to SQLite and overlaid on Scenario Analyzer runs."
    )

    try:
        ensure_ingestion_schema()
    except Exception as exc:
        st.error(f"Failed to verify ingestion schema: {exc}")
        return

    scheduler = _get_scheduler()
    run_active = _is_manual_run_active()

    # ── Controls ──────────────────────────────────────────────────────────────
    col_run, col_auto, col_status = st.columns([1, 1, 2])

    with col_run:
        if st.button("Run Now", type="primary", disabled=run_active):
            _start_manual_run()
            st.rerun()

    with col_auto:
        auto_enabled = st.toggle("Auto-refresh (hourly)", value=scheduler.is_running)
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

    # ── In-progress banner (polls every 5 s until done) ───────────────────────
    if run_active:
        st.info(
            "Fetching signals from 7 connectors — this takes ~2 minutes. "
            "Page auto-refreshes every 5 seconds."
        )
        time.sleep(5)
        st.rerun()

    # Show result from the last completed manual run
    result = st.session_state.get("manual_run_result")
    error = st.session_state.get("manual_run_error")
    if result is not None:
        if result.status == "success":
            st.success(
                f"Done — inserted {result.total_rows_inserted} rows "
                f"across {len(result.connectors_run)} connectors."
            )
        elif result.status == "partial":
            st.warning(
                f"Partial: {result.total_rows_inserted} inserted, "
                f"{len(result.errors)} connector(s) failed."
            )
        elif result.status == "skipped":
            st.info("Another ingestion run was already in progress.")
        else:
            st.error(f"Ingestion failed: {result.errors}")
        if result.errors:
            with st.expander("Connector errors"):
                for err in result.errors:
                    st.text(err)
    if error:
        st.error(f"Run error: {error}")

    st.divider()

    # ── Hub City Weather Signals ──────────────────────────────────────────────
    st.subheader("Hub City Weather Signals")
    st.caption(
        "Latest weather reading per global semiconductor hub city "
        "(Hsinchu, Osaka, Austin, Shanghai, Singapore, Rotterdam). "
        "Severity 0-10: ≥6 triggers the pipeline."
    )
    try:
        hub_wx_rows = execute_query(
            """
            SELECT hub_city, latitude, longitude,
                   wind_speed_kmh, precipitation_mm, weather_code,
                   temperature_c, raw_severity_score, is_trigger_hub,
                   fetched_at_utc
            FROM live_weather_ingest
            WHERE id IN (
                SELECT MAX(id) FROM live_weather_ingest GROUP BY hub_city
            )
            ORDER BY raw_severity_score DESC
            """
        )
        if hub_wx_rows:
            wx_data = []
            for row in hub_wx_rows:
                severity = row[7] or 0.0
                wx_data.append({
                    "Hub City": row[0],
                    "Lat / Lon": f"{row[1]:.2f}, {row[2]:.2f}",
                    "Wind (km/h)": f"{row[3]:.1f}" if row[3] is not None else "—",
                    "Precip (mm)": f"{row[4]:.1f}" if row[4] is not None else "—",
                    "WMO Code": row[5] if row[5] is not None else "—",
                    "Temp (°C)": f"{row[6]:.1f}" if row[6] is not None else "—",
                    "Severity (0-10)": f"{severity:.1f}",
                    "Trigger": "🔴 YES" if row[8] else ("⚠ Near" if severity >= 4 else "✅ OK"),
                    "Fetched (UTC)": (row[9] or "")[:16].replace("T", " "),
                })
            st.dataframe(wx_data, use_container_width=True)
        else:
            st.info("No hub city weather data yet — click 'Run Now' to fetch.")
    except Exception as exc:
        st.warning(f"Could not load hub weather data: {exc}")

    st.divider()

    # ── Hub City News Signals ─────────────────────────────────────────────────
    st.subheader("Hub City & Country News Signals")
    st.caption(
        "Targeted RSS headlines for semiconductor hub cities, hub countries, "
        "and key supplier nations. Relevance score ≥0.4 = high signal."
    )
    try:
        hub_news_rows = execute_query(
            """
            SELECT hub_city, hub_country, supplier_country,
                   headline, relevance_score, published_at, source_feed, url
            FROM live_news_ingest
            ORDER BY relevance_score DESC, published_at DESC
            LIMIT 30
            """
        )
        if hub_news_rows:
            news_data = []
            for row in hub_news_rows:
                location = row[0] or row[1] or row[2] or "—"
                loc_type = "City" if row[0] else ("Country" if row[1] else "Supplier")
                news_data.append({
                    "Location": location,
                    "Type": loc_type,
                    "Headline": row[3],
                    "Relevance": f"{row[4]:.2f}" if row[4] is not None else "—",
                    "Published": (row[5] or "")[:16].replace("T", " "),
                    "Source": row[6] or "—",
                })
            st.dataframe(news_data, use_container_width=True)
        else:
            st.info("No hub city news data yet — click 'Run Now' to fetch.")
    except Exception as exc:
        st.warning(f"Could not load hub news data: {exc}")

    st.divider()

    # ── Indian Port Live Enrichment ───────────────────────────────────────────
    st.subheader("Indian Port Live Signals")
    st.caption(
        "Aggregated signals per delivery port (JNPT, Mundra, Chennai, etc.) "
        "overlaid on Scenario Analyzer runs."
    )
    try:
        enrichment_rows = execute_query(
            """
            SELECT port, signal_date, weather_severity_live,
                   natural_disaster_risk_live, supply_disruption_index_live,
                   disruption_news_count_live, chip_price_index_live,
                   export_control_flag, enrichment_ts_utc
            FROM live_enrichment
            ORDER BY port, enrichment_ts_utc DESC
            """
        )
        if enrichment_rows:
            seen = set()
            table_data = []
            for row in enrichment_rows:
                port = row[0]
                if port in seen:
                    continue
                seen.add(port)
                table_data.append({
                    "Port": port,
                    "Signal Date": row[1],
                    "Weather Severity": f"{row[2]:.3f}" if row[2] is not None else "—",
                    "Natural Disaster Risk": f"{row[3]:.2f}" if row[3] is not None else "—",
                    "Supply Disruption SDI": f"{row[4]:.3f}" if row[4] is not None else "—",
                    "News Count (7d)": row[5] if row[5] is not None else "—",
                    "Chip Price Index": f"{row[6]:.3f}" if row[6] is not None else "—",
                    "Export Control": "⚠ Yes" if row[7] else "No",
                    "Updated": (row[8] or "")[:16].replace("T", " ") + " UTC",
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No port enrichment data yet — click 'Run Now'.")
    except Exception as exc:
        st.warning(f"Could not load port enrichment data: {exc}")

    st.divider()

    # ── Recent Broad News Disruptions ─────────────────────────────────────────
    st.subheader("Recent News & Disruption Signals")
    try:
        news_rows = execute_query(
            """
            SELECT published_at, source, headline, risk_categories, severity_score
            FROM news_disruptions
            WHERE is_active = 1
            ORDER BY published_at DESC LIMIT 25
            """
        )
        if news_rows:
            import json as _json
            news_data = []
            for row in news_rows:
                try:
                    cats = ", ".join(_json.loads(row[3])) if row[3] else ""
                except Exception:
                    cats = row[3] or ""
                news_data.append({
                    "Published": (row[0] or "")[:16].replace("T", " "),
                    "Source": row[1],
                    "Headline": row[2],
                    "Categories": cats,
                    "Severity": f"{row[4]:.2f}" if row[4] is not None else "—",
                })
            st.dataframe(news_data, use_container_width=True)
        else:
            st.info("No broad news disruptions fetched yet.")
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
            """
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
