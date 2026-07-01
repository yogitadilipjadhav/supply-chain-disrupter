"""
DataIngestionAgent — enhanced L1 agent.

Responsibilities:
  - Run all external API connectors (weather, freight, news, commodity prices).
  - Validate, normalize, and deduplicate all records.
  - Persist to new enrichment tables without touching lite_master.
  - Aggregate per-port signals into live_enrichment rows.
  - Expose run_for_record() for use by run_agent_graph().
  - Log every connector run to ingestion_run_log.

Two execution modes:
  BATCH  — full connector sweep, run by scheduler or manual Streamlit button.
  RECORD — called from data_ingestion_agent_v2() for a single port/sku lookup.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.agents.state import EventMetadata, GlobalState
from src.utils.db_utils import execute_query, get_connection
from src.utils.ingestion_schema import ensure_ingestion_schema
from src.utils.ingestion_connectors import (
    BaseConnector,
    CisaBisRSSConnector,
    FredConnector,
    GdeltConnector,
    GoogleNewsRSSConnector,
    OpenMeteoEnhancedConnector,
    ReutersRSSConnector,
    YFinanceConnector,
)
from src.utils.ingestion_validator import DataValidator
from src.utils.yaml_utils import load_config, get_port_coordinates

logger = logging.getLogger(__name__)

_INGESTION_LOCK = threading.Lock()


class IngestionRunResult(BaseModel):
    run_id: str
    started_at_utc: str
    completed_at_utc: str
    connectors_run: List[str]
    total_rows_inserted: int
    total_rows_skipped: int
    errors: List[str]
    status: str  # 'success', 'partial', 'failed'


# ── Main agent ────────────────────────────────────────────────────────────────

class DataIngestionAgent:

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or load_config()
        ensure_ingestion_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_batch(self, ports: Optional[List[str]] = None) -> IngestionRunResult:
        """
        Full connector sweep. Acquires _INGESTION_LOCK to prevent concurrent Streamlit runs.
        After all connectors complete, calls _build_live_enrichment() to materialise
        live_enrichment rows for each port.
        """
        if not _INGESTION_LOCK.acquire(blocking=False):
            logger.warning("Ingestion already running — skipping concurrent run.")
            return IngestionRunResult(
                run_id="skipped",
                started_at_utc=_now_utc(),
                completed_at_utc=_now_utc(),
                connectors_run=[],
                total_rows_inserted=0,
                total_rows_skipped=0,
                errors=["Concurrent run skipped — another ingestion is in progress."],
                status="skipped",
            )
        try:
            return self._run_all_connectors(ports)
        finally:
            _INGESTION_LOCK.release()

    def run_for_record(
        self, port: str, sku: str, signal_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Called from data_ingestion_agent_v2() at scenario runtime.
        Returns the most recent live_enrichment row for (port, signal_date),
        or None if no fresh data exists. Marks the row is_consumed=1.
        """
        try:
            rows = execute_query(
                """
                SELECT weather_severity_live, natural_disaster_risk_live,
                       supply_disruption_index_live, disruption_news_count_live,
                       chip_price_index_live, market_growth_rate_live,
                       export_control_flag, agent_run_id
                FROM live_enrichment
                WHERE port = ?
                ORDER BY enrichment_ts_utc DESC LIMIT 1
                """,
                (port,),
            )
            if not rows:
                return None
            row = dict(rows[0])
            # Mark consumed (best-effort)
            try:
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE live_enrichment SET is_consumed = 1 "
                        "WHERE port = ? ORDER BY enrichment_ts_utc DESC LIMIT 1",
                        (port,),
                    )
                    conn.commit()
            except Exception:
                pass
            return row
        except Exception as exc:
            logger.warning("run_for_record failed for %s/%s: %s", port, sku, exc)
            return None

    # ── Internal orchestration ────────────────────────────────────────────────

    def _run_all_connectors(self, ports: Optional[List[str]]) -> IngestionRunResult:
        run_id = str(uuid.uuid4())
        started_at = _now_utc()
        connectors_run: List[str] = []
        total_inserted = 0
        total_skipped = 0
        errors: List[str] = []

        connector_classes = [
            OpenMeteoEnhancedConnector,
            FredConnector,
            GdeltConnector,
            GoogleNewsRSSConnector,
            ReutersRSSConnector,
            CisaBisRSSConnector,
            YFinanceConnector,
        ]

        for cls in connector_classes:
            connector = cls()
            result = self._run_connector(connector, run_id)
            connectors_run.append(connector.SOURCE_NAME)
            total_inserted += result["inserted"]
            total_skipped += result["skipped"]
            if result["error"]:
                errors.append(f"{connector.SOURCE_NAME}: {result['error']}")

        # Build live_enrichment aggregation (stamp run_id so L1 shim can recover it)
        all_ports = list(self._config.get("ports", {}).keys())
        target_ports = ports or all_ports
        enrichment_rows = self._build_live_enrichment(target_ports, run_id=run_id)
        logger.info("Built %d live_enrichment rows for %d ports.", enrichment_rows, len(target_ports))

        status = "success" if not errors else ("partial" if total_inserted > 0 else "failed")
        completed_at = _now_utc()

        return IngestionRunResult(
            run_id=run_id,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            connectors_run=connectors_run,
            total_rows_inserted=total_inserted,
            total_rows_skipped=total_skipped,
            errors=errors,
            status=status,
        )

    def _run_connector(
        self, connector: BaseConnector, run_id: str
    ) -> Dict[str, Any]:
        """Run one connector: fetch → normalize → persist → log. Never raises."""
        t0 = time.monotonic()
        inserted = skipped = 0
        error_detail: Optional[str] = None
        last_fetched_key: Optional[str] = None

        try:
            raw = connector.fetch()
            rows = connector.normalize(raw)
            connector._run_id = run_id  # passed into persist for live_news_ingest / live_weather_ingest
            inserted, skipped = connector.persist(rows)
            # Use the last signal_date or fetched_at as the incremental key
            if rows:
                last_fetched_key = rows[-1].get("signal_date") or rows[-1].get("fetched_at_utc", "")[:10]
            logger.info(
                "%s: fetched=%d normalized=%d inserted=%d skipped=%d",
                connector.SOURCE_NAME, len(raw), len(rows), inserted, skipped,
            )
        except Exception as exc:
            error_detail = str(exc)
            logger.error("%s connector error: %s", connector.SOURCE_NAME, exc)

        duration_ms = int((time.monotonic() - t0) * 1000)
        status = "success" if error_detail is None else "failed"
        self._log_run(
            run_id=run_id,
            source=connector.SOURCE_NAME,
            connector_class=type(connector).__name__,
            rows_fetched=inserted + skipped,
            rows_inserted=inserted,
            rows_skipped=skipped,
            duration_ms=duration_ms,
            status=status,
            error_detail=error_detail,
            last_fetched_key=last_fetched_key,
        )

        return {"inserted": inserted, "skipped": skipped, "error": error_detail}

    def _build_live_enrichment(self, ports: List[str], run_id: Optional[str] = None) -> int:
        """
        Aggregate latest signals per port into live_enrichment using INSERT OR REPLACE.
        Returns count of rows written.
        """
        ts = _now_utc()
        today = ts[:10]
        written = 0

        for port in ports:
            try:
                # Latest weather
                weather_rows = execute_query(
                    "SELECT derived_weather_severity, derived_natural_disaster_risk "
                    "FROM weather_events WHERE port = ? AND is_active = 1 "
                    "ORDER BY fetched_at_utc DESC LIMIT 1",
                    (port,),
                )
                weather_sev = weather_rows[0][0] if weather_rows else None
                nat_dis_risk = weather_rows[0][1] if weather_rows else None

                # Latest freight SDI
                freight_rows = execute_query(
                    "SELECT normalized_sdi FROM freight_signals "
                    "WHERE is_active = 1 AND normalized_sdi IS NOT NULL "
                    "ORDER BY signal_date DESC LIMIT 1",
                    (),
                )
                sdi_live = freight_rows[0][0] if freight_rows else None

                # Latest chip price index from market_demand_signals (SOX)
                cpi_rows = execute_query(
                    "SELECT normalized_chip_price_index FROM market_demand_signals "
                    "WHERE series_id = '^SOX' AND is_active = 1 "
                    "AND normalized_chip_price_index IS NOT NULL "
                    "ORDER BY signal_date DESC LIMIT 1",
                    (),
                )
                cpi_live = cpi_rows[0][0] if cpi_rows else None

                # Latest market growth rate from SOX pct_change
                mgr_rows = execute_query(
                    "SELECT normalized_market_growth_rate FROM market_demand_signals "
                    "WHERE series_id = '^SOX' AND is_active = 1 "
                    "AND normalized_market_growth_rate IS NOT NULL "
                    "ORDER BY signal_date DESC LIMIT 1",
                    (),
                )
                mgr_live = mgr_rows[0][0] if mgr_rows else None

                # News count (last 7 days) — supplier_risk_events + live_news_ingest
                supplier_rows = execute_query(
                    "SELECT COUNT(*) FROM supplier_risk_events "
                    "WHERE is_active = 1 AND event_date >= date('now', '-7 days')"
                    " AND (canonical_port = ? OR canonical_port IS NULL)",
                    (port,),
                )
                live_news_rows = execute_query(
                    "SELECT COUNT(*) FROM live_news_ingest "
                    "WHERE relevance_score >= 0.3 "
                    "AND fetched_at_utc >= date('now', '-7 days')",
                    (),
                )
                news_count = (
                    int(supplier_rows[0][0] if supplier_rows else 0)
                    + int(live_news_rows[0][0] if live_news_rows else 0)
                )
                # Cap at normalization bound of 17
                news_count = min(news_count, 17)

                # Export control flag (any BIS alert in last 30 days)
                export_rows = execute_query(
                    "SELECT COUNT(*) FROM news_disruptions "
                    "WHERE is_active = 1 "
                    "AND risk_categories LIKE '%export_control%' "
                    "AND published_at >= date('now', '-30 days')",
                    (),
                )
                export_flag = 1 if (export_rows and int(export_rows[0][0]) > 0) else 0

                new_row = {
                    "enrichment_ts_utc": ts,
                    "port": port,
                    "sku": None,
                    "signal_date": today,
                    "weather_severity_live": weather_sev,
                    "natural_disaster_risk_live": nat_dis_risk,
                    "supply_disruption_index_live": sdi_live,
                    "disruption_news_count_live": news_count,
                    "chip_price_index_live": cpi_live,
                    "market_growth_rate_live": mgr_live,
                    "export_control_flag": export_flag,
                    "agent_run_id": run_id,
                    "is_consumed": 0,
                }

                # Conflict check before writing
                DataValidator.check_enrichment_conflict(port, new_row)

                with get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO live_enrichment
                        (enrichment_ts_utc, port, sku, signal_date,
                         weather_severity_live, natural_disaster_risk_live,
                         supply_disruption_index_live, disruption_news_count_live,
                         chip_price_index_live, market_growth_rate_live,
                         export_control_flag, agent_run_id, is_consumed)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            ts, port, None, today,
                            weather_sev, nat_dis_risk, sdi_live, news_count,
                            cpi_live, mgr_live, export_flag, None, 0,
                        ),
                    )
                    conn.commit()
                written += 1
            except Exception as exc:
                logger.error("live_enrichment build failed for %s: %s", port, exc)

        return written

    def _log_run(
        self,
        run_id: str,
        source: str,
        connector_class: str,
        rows_fetched: int,
        rows_inserted: int,
        rows_skipped: int,
        duration_ms: int,
        status: str,
        error_detail: Optional[str],
        last_fetched_key: Optional[str],
    ) -> None:
        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO ingestion_run_log
                    (run_id, run_ts_utc, source, connector_class, rows_fetched,
                     rows_inserted, rows_skipped, duration_ms, status,
                     error_detail, last_fetched_key)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, _now_utc(), source, connector_class, rows_fetched,
                        rows_inserted, rows_skipped, duration_ms, status,
                        error_detail, last_fetched_key,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("ingestion_run_log insert failed: %s", exc)


# ── Integration shim for run_agent_graph() ────────────────────────────────────

def data_ingestion_agent_v2(
    state: GlobalState, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Drop-in replacement for data_ingestion_agent() in langgraph_engine.py.
    Loads the base record from lite_master (via daily_records view), then overlays
    the latest live_enrichment values for the signal fields that drive risk scoring.

    Only these fields are overlaid (immutable fields are never changed):
      weather_severity_hub, natural_disaster_risk, supply_disruption_index,
      disruption_news_count, chip_price_index

    delivery_status, risk_score_composite, order_id, record_id are never modified.
    """
    from src.utils.db_utils import fetch_daily_record
    from src.utils.yaml_utils import load_config

    event_metadata = EventMetadata(**payload)
    config = load_config()

    # Load base record from daily_records view (unchanged behaviour)
    record = fetch_daily_record(
        payload.get("event_date", ""),
        event_metadata.affected_port,
        payload.get("sku", "CHIP_AP"),
    )

    enrichment_source = "historical"
    ingestion_run_id: Optional[str] = None
    if record is not None:
        try:
            agent = DataIngestionAgent(config=config)
            enrichment = agent.run_for_record(
                port=event_metadata.affected_port,
                sku=payload.get("sku", ""),
                signal_date=payload.get("event_date", ""),
            )
            if enrichment:
                enriched = dict(record)
                # Overlay only signal fields — never touch audit/identity fields
                _safe_overlay(enriched, "weather_severity_hub",    enrichment.get("weather_severity_live"))
                _safe_overlay(enriched, "natural_disaster_risk",   enrichment.get("natural_disaster_risk_live"))
                _safe_overlay(enriched, "supply_disruption_index", enrichment.get("supply_disruption_index_live"))
                _safe_overlay(enriched, "disruption_news_count",   enrichment.get("disruption_news_count_live"))
                _safe_overlay(enriched, "chip_price_index",        enrichment.get("chip_price_index_live"))
                record = enriched
                enrichment_source = "live"
                ingestion_run_id = enrichment.get("agent_run_id")
        except Exception as exc:
            logger.warning("Live enrichment overlay failed: %s", exc)

    state_updates: Dict[str, Any] = {
        "event_metadata": event_metadata,
        "config": config,
        "ingestion_run_id": ingestion_run_id,
        "agent_logs": state.agent_logs + [
            f"L1: Data ingestion completed (enrichment={enrichment_source}, run_id={ingestion_run_id})."
        ],
    }
    if record is not None:
        state_updates["active_record"] = record

    return state_updates


def _safe_overlay(record: Dict[str, Any], field: str, value: Any) -> None:
    """Overlay a field only when the live value is non-None and numeric."""
    if value is not None:
        try:
            record[field] = float(value)
        except (TypeError, ValueError):
            pass


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
