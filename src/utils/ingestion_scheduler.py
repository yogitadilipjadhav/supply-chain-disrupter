"""
IngestionScheduler — lightweight background scheduler for DataIngestionAgent.

Thread-based daemon (no APScheduler/Celery dependency).
Respects _INGESTION_LOCK in DataIngestionAgent to prevent overlapping runs.
Lifecycle is tied to the Streamlit process; the thread dies when the process exits.
"""

import logging
import threading
import time
from typing import Callable, Optional

from src.agents.data_ingestion_agent import DataIngestionAgent, IngestionRunResult

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 3600  # hourly


class IngestionScheduler:

    def __init__(
        self,
        agent_factory: Callable[[], DataIngestionAgent] = DataIngestionAgent,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._agent_factory = agent_factory
        self._interval = interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_result: Optional[IngestionRunResult] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background thread. Safe to call multiple times."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.debug("IngestionScheduler already running.")
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="ingestion-scheduler",
            )
            self._thread.start()
            logger.info(
                "IngestionScheduler started (interval=%ds).", self._interval
            )

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self) -> Optional[IngestionRunResult]:
        return self._last_result

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                agent = self._agent_factory()
                result = agent.run_batch()
                self._last_result = result
                logger.info(
                    "Ingestion run complete: inserted=%d skipped=%d status=%s errors=%d",
                    result.total_rows_inserted,
                    result.total_rows_skipped,
                    result.status,
                    len(result.errors),
                )
            except Exception as exc:
                logger.error("IngestionScheduler loop error: %s", exc)
            # Wait for next interval or stop signal
            self._stop_event.wait(self._interval)
