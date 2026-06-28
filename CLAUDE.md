# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An electronics/semiconductor supply-chain disruption predictor. It loads Varun's
electronics workbook into local SQLite + ChromaDB, then runs a multi-agent scenario
pipeline (risk classification, RAG grounding, forecasting, simulation, mitigation)
exposed through a Streamlit dashboard. Fully local — no LLM/OpenAI key required; the
only external calls are Open-Meteo (weather) and a first-run Hugging Face download of
the `all-MiniLM-L6-v2` embedding model.

## Commands

All commands run from the project root on Windows PowerShell, with the venv active.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt   # or: pip install -e ".[dev]"
```

- Build databases (SQLite + ChromaDB): `python -m src.build_databases`
  - Add `--no-rebuild` to upsert Chroma content instead of clearing the collection.
  - Rebuild only when the workbook, playbooks, or DB code changes.
- Run the app: `python -m streamlit run src/main.py` (opens http://localhost:8501)
- Run tests: `pytest tests/ -v`
- Run a single test: `pytest tests/test_risk_classifier_agent.py::<test_name> -v`

The `evaluation/` directory holds standalone QA scripts (run directly with
`python evaluation/qa_04_replay_mode_real_data.py`), not pytest tests. They document
and verify the Risk Classifier contract against synthetic fixtures and the real DB.

## Architecture

### Data flow (build time)
`scripts/build_databases.py` → `src/utils/etl_loader.py` (Excel → SQLite) and
`src/utils/rag_utils.py` (text/playbooks/PDF/DOCX → ChromaDB). `src/build_databases.py`
is a thin shim re-exporting `scripts.build_databases.main`.

- **Source workbook**: `data/raw/supply_chain_lite_master_v2.xlsx` (defined as
  `EXCEL_SOURCE` in `etl_loader.py`). Note: the README still references the older
  `supply_chain_lite_master.xlsx`; the code uses v2.
- **Category filtering**: the DataCo source mislabels sports/fashion items as
  "Electronics". `etl_loader.py` only loads `GENUINE_ELECTRONICS_CATEGORIES`
  (Consumer Electronics, Computers, Cameras, Video Games). Beauty/FMCG data is excluded.
- **Generated outputs** (gitignored): `outputs/supply_chain.db`, `outputs/chromadb/`.
- The build creates a `daily_records` SQLite view used by the scenario workflow as a
  compatibility layer over `lite_master`.

### Agent pipeline (run time)
`src/agents/langgraph_engine.py::run_agent_graph(payload)` orchestrates a fixed
sequence over a single shared `GlobalState` (Pydantic, `src/agents/state.py`). Despite
the filename there is no LangGraph dependency — it is a hand-rolled linear pipeline.
Each agent returns a dict delta that is merged via `state.copy(update=delta)`.

- **Critical agents (raise on failure)**: data ingestion (L1) → news/event analysis
  (L2, `rag_agent.py`) → weather risk (L3, Open-Meteo) → risk classifier (L4).
- **Optional agents (logged + skipped on failure via `_run_optional`)**: demand
  forecasting (L5, Prophet — degrades gracefully if prophet/pandas absent), simulation
  (L6), mitigation recommendation (L7, persists to `mitigation_actions`).

### Risk Classifier (`src/agents/risk_classifier_agent.py`) — the core logic
The most involved component. Read its module docstring before changing it.

- **Two modes**: REPLAY (order already has stored composite + label in `lite_master`;
  trust the stored composite, re-derive label from `delivery_status`, **never overwrite
  historical rows**) vs LIVE (new/injected order; recompute composite from the spec
  formula and write back via `update_risk_label`).
- **Composite**: weighted sum of four [0,1] components — geo 0.40, supply 0.30,
  freight 0.15, defect 0.15 (`_WEIGHTS`). Normalization bounds are read once from
  `lite_master` and cached (`_get_norm_bounds`, `@lru_cache`).
- **Label**: derived from exact `delivery_status` strings first (`Shipping canceled`
  → CRITICAL, `Late delivery` → HIGH, etc.), falling back to composite thresholds.
  `delivery_status` takes precedence over score.
- **Duration escalation**: never lowers a label. duration ≤1 day: no change; 2–3 days:
  +1 tier; ≥4 days: force CRITICAL. Duration comes from RAG-extracted
  `expected_duration_days` or `event_metadata.shock_duration_days`.
- **RAG grounding**: only queried for HIGH/CRITICAL or escalated outcomes (latency
  optimization). Export-control corpus queried only when `export_control_level` is in
  the top quartile.
- `critical_flag` is the hard business rule that gates the (stubbed) Slack webhook in
  the mitigation agent.

### Dashboard (`src/dashboard/`)
`main.py` → `dashboard.py::main()`. Three pages: Data Ingestion (`data_loader.py`),
RAG Search, Scenario Analyzer.

### Utilities (`src/utils/`)
- `db_utils.py`: SQLite access. `DB_PATH = outputs/supply_chain.db`. Connections use
  `check_same_thread=False` (Streamlit) and `Row` factory.
- `rag_utils.py`: ChromaDB. **Important Windows constraint** — `PersistentClient` holds
  an exclusive lock on `chroma.sqlite3`, so a module-level singleton (`_chroma_client`,
  lock-guarded) is used. Calling rebuild with `flush_existing=True` (shutil.rmtree)
  while the client is alive causes `WinError 32`; pass `flush_existing=False` in-process.
- `api_clients.py`: Open-Meteo fetch + `compute_weather_severity`.
- `yaml_utils.py`: loads `config/india_electronics.yaml` (port coordinates, backup routes).

## State model notes
`GlobalState.risk_label` and `risk_score_composite` are deprecated read-only shims —
prefer `risk_classification.final_label` / `.composite_score`. Backwards-compat is
tested in `evaluation/qa_07_backwards_compat_state_shims.py`; keep the shims working.
