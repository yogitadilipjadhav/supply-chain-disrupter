"""Data Ingestion agent (L1) — loads the scenario record and config into state.

Live external ingestion (weather/news → SQLite) lives in
``src.agents.data_ingestion.live_ingest`` and is run as a batch poller.
"""

from typing import Any, Dict

from src.utils.db_utils import fetch_daily_record
from src.utils.yaml_utils import load_config
from src.agents.state import EventMetadata, GlobalState


def data_ingestion_agent(state: GlobalState, payload: Dict[str, Any]) -> Dict[str, Any]:
    event_metadata = EventMetadata(**payload)
    state_updates: Dict[str, Any] = {
        "event_metadata": event_metadata,
        "config": load_config(),
        "agent_logs": state.agent_logs + ["L1: Data ingestion completed."],
    }
    record = fetch_daily_record(
        payload.get("event_date", ""),
        event_metadata.affected_port,
        payload.get("sku", "CHIP_AP"),
    )
    if record:
        state_updates["active_record"] = record
    return state_updates
