# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An electronics/semiconductor supply-chain disruption predictor. It loads a v3.1 electronics
workbook into local SQLite + ChromaDB, then runs a multi-agent scenario pipeline (live signal
ingestion, news/event analysis, weather risk, risk classification, forecasting, simulation,
mitigation) exposed through a Streamlit dashboard.

External calls: Open-Meteo (weather), Google News RSS + Reuters RSS (news headlines), and
optionally OpenAI GPT-4o/4.1-mini (LLM enrichment — all agents degrade gracefully when
`OPENAI_API_KEY` is absent). First run downloads `all-MiniLM-L6-v2` from Hugging Face.

## Commands

All commands run from the project root on Windows PowerShell, with the venv active.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

- Build databases (SQLite + ChromaDB): `python -m src.build_databases`
  - Add `--no-rebuild` to upsert Chroma content instead of clearing the collection.
  - Rebuild when the workbook, playbooks, or DB schema changes.
- Run the app: `python -m streamlit run src/main.py` (opens http://localhost:8501)
- Run tests: `pytest tests/ -v`
- Run a single test: `pytest tests/test_risk_classifier_agent.py::<test_name> -v`
- Run MCP weather server: `python -m src.mcp_servers.weather_mcp`
- Run MCP news server: `python -m src.mcp_servers.news_mcp`

The `evaluation/` directory holds standalone QA scripts (run directly with
`python evaluation/qa_04_replay_mode_real_data.py`), not pytest tests.

## Architecture

### Data flow (build time)
`scripts/build_databases.py` → `src/utils/etl_loader.py` (Excel → SQLite) and
`src/utils/rag_utils.py` (text/playbooks/PDF/DOCX → ChromaDB). `src/build_databases.py`
is a thin shim re-exporting `scripts.build_databases.main`.

- **Source workbook**: `data/raw/supply_chain_lite_master.xlsx` (v3.1, defined as
  `EXCEL_SOURCE` in `etl_loader.py`). 11,559 rows, 32 columns, 2015–2025 date range.
  Adds `known_disruption_event` and `known_severity` columns (COVID-19, chip shortage, etc.)
  vs the older v2 file.
- **Category filtering**: `etl_loader.py` only loads `GENUINE_ELECTRONICS_CATEGORIES`
  (Consumer Electronics, Computers, Cameras, Video Games). Sports/fashion mislabelled as
  "Electronics" in the DataCo source are excluded.
- **Generated outputs** (gitignored): `outputs/supply_chain.db`, `outputs/chromadb/`.
- The build creates a `daily_records` SQLite view as a compatibility layer over `lite_master`.
- **Ingestion schema** (`src/utils/ingestion_schema.py`): `ensure_ingestion_schema()` creates
  two additional tables on first use — `live_news_ingest` and `live_weather_ingest` — and
  indices on `run_id`. Called once at process startup in `langgraph_engine.py`.

### Agent pipeline (run time)
`src/agents/langgraph_engine.py::run_agent_graph(payload)` orchestrates a fixed sequence over
a single shared `GlobalState` (Pydantic v2, `src/agents/state.py`). Hand-rolled linear
pipeline — no LangGraph dependency. Each agent returns a dict delta merged via
`state.model_copy(update=delta)`.

- **Critical agents (raise on failure)**:
  - L1: data ingestion (`data_ingestion_agent_v2` → falls back to legacy `data_ingestion_agent`)
  - L2: news/event analysis (`src/agents/news_agent/agent.py`)
  - L3: weather risk (`src/agents/weather_agent/agent.py`)
  - L4: risk classifier (`src/agents/risk_classifier_agent.py`)
- **Optional agents (logged + skipped via `_run_optional`)**:
  - L5: demand forecasting (Prophet — skipped if prophet/pandas absent)
  - L6: Monte Carlo simulation
  - L7: mitigation recommendation (`src/agents/mitigation_agent.py`)

**Core architectural principle**: L1 is the ONLY external I/O boundary. L2 reads from
ChromaDB RAG only. L3 reads weather severity from `live_weather_ingest` (populated by L1);
falls back to a live Open-Meteo call only when no batch run has occurred (demo/manual mode).

### Data Ingestion Agent v2 (`src/agents/data_ingestion_agent.py`)

The v2 enrichment framework. Runs as a batch job (hourly scheduler in
`src/utils/ingestion_scheduler.py`) and also on-demand from the dashboard.

**7 connectors** (`src/utils/ingestion_connectors.py`):
- `OpenMeteoEnhancedConnector` → fetches both Indian port weather (`weather_events`) AND
  6 global fab-hub cities (`live_weather_ingest`)
- `GoogleNewsRSSConnector` → hub city/country/supplier queries → `live_news_ingest`
  (Reuters RSS fallback when < 3 results)
- `FREDConnector` → freight signals → `freight_signals`
- `GDELTConnector` → news disruptions → `news_disruptions`
- `ReutersRSSConnector` → supplier risk → `supplier_risk_events`
- `CisaBisRSSConnector` → export control/regulatory → `news_disruptions`
- `YFinanceConnector` → semiconductor market data → `market_demand_signals`

**Hub cities** (source-of-disruption monitoring, distinct from Indian delivery ports):
```python
HUB_CITIES = {
    "Hsinchu": (24.80, 120.97),   # TSMC
    "Osaka":   (34.69, 135.50),   # Renesas
    "Austin":  (30.27, -97.74),   # Samsung/NXP
    "Shanghai": (31.23, 121.47),  # SMIC
    "Singapore": (1.35, 103.82),  # GlobalFoundries
    "Rotterdam": (51.92,   4.48), # European logistics hub
}
```

**`ingestion_run_id`**: UUID generated per batch run, stamped on `live_enrichment.agent_run_id`,
written into `GlobalState.ingestion_run_id`. Links every pipeline execution to the specific
`live_news_ingest` / `live_weather_ingest` rows that triggered it — full audit trail.

**Severity formula** (`live_weather_ingest.raw_severity_score`, 0–10 scale):
- WMO code ≥ 95 (thunderstorm/hail): +4
- Wind ≥ 60 km/h: +2; wind ≥ 30 km/h: +1
- Precipitation > 50 mm: +2; > 10 mm: +1
- Max achievable = 8 (not 10); `is_trigger_hub=1` set on city with highest severity if ≥ 6.0

**Relevance scoring** (`live_news_ingest.relevance_score`, 0–1):
10 keywords (semiconductor, chip, supply chain, disruption, factory, fab, port,
export control, shortage, shutdown) → count / 10.

**Trigger thresholds**: pipeline fires only when `COUNT(news WHERE relevance_score ≥ 0.4) ≥ 3`
OR `MAX(raw_severity_score) ≥ 6`.

### MCP Servers (`src/mcp_servers/`)

FastMCP servers demonstrating tool-calling layer between L1 agent and external APIs.

- `weather_mcp.py`: `get_hub_weather(city, lat, lon)` → Open-Meteo → dict with
  `raw_severity_score`. Run: `python -m src.mcp_servers.weather_mcp`
- `news_mcp.py`: `get_news_headlines(query, hub_city, hub_country, supplier_country,
  max_results)` → Google News RSS + Reuters fallback → list with relevance scores.
  Run: `python -m src.mcp_servers.news_mcp`

### L2 — News Agent (`src/agents/news_agent/`)

Package: `agent.py` (orchestrator), `rag.py` (ChromaDB signal builder), `__init__.py`
(re-exports for patching). `build_news_signals` and `call_openai_structured` are exposed at
package level so `unittest.mock.patch("src.agents.news_agent.X")` works in tests.

- Calls `build_news_signals(disruption_type)` → rule-based signals from ChromaDB
- Fallback signal always has `expected_duration_days` set (from `metadata.shock_duration_days`)
- Optional: calls `call_openai_structured(..., NewsAnalysisLLMOutput)` for structured
  event classification; stores result in `state.news_analysis_llm` (None if LLM fails/absent)

### L3 — Weather Agent (`src/agents/weather_agent/`)

Package: `agent.py` (orchestrator), `client.py` (Open-Meteo fetch + severity compute),
`__init__.py` (re-exports). `fetch_open_meteo`, `compute_weather_severity`, `has_openai_api_key`,
and `call_openai_structured` are at package level for test patching.

- **Primary path**: reads `raw_severity_score` from `live_weather_ingest WHERE run_id = ?`,
  converts 0–10 → 0–1 scale.
- **Fallback** (no batch run / demo mode): calls `fetch_open_meteo` + `compute_weather_severity`.
  If `has_openai_api_key()` → calls `call_openai_structured(..., WeatherRiskLLMOutput)` and
  uses `geo_risk_component` as final severity (overrides numeric).
- Result in `state.live_weather_severity` (float 0–1) and `state.weather_risk_llm` (optional).

### Risk Classifier (`src/agents/risk_classifier_agent.py`) — the core logic

Read its module docstring before changing it.

- **Two modes**: REPLAY (stored composite in `lite_master`; trust it, re-derive label from
  `delivery_status`, never overwrite historical rows) vs LIVE (new order; recompute composite,
  write back via `update_risk_label`).
- **Composite**: weighted sum — geo 0.40, supply 0.30, freight 0.15, defect 0.15 (`_WEIGHTS`).
  Normalization bounds cached via `_get_norm_bounds` (`@lru_cache`).
- **Label**: `delivery_status` strings take precedence over score (`Shipping canceled` →
  CRITICAL, `Late delivery` → HIGH, etc.), falling back to composite thresholds.
- **Duration escalation**: never lowers a label. ≤1 day: no change; 2–3 days: +1 tier;
  ≥4 days: force CRITICAL.
- **Ensemble** (when OpenAI key present): Rule signal + DistilBERT signal + LLM signal →
  Judge verdict. Without key, rule signal only.
- `critical_flag` gates the (stubbed) Slack webhook in the mitigation agent.

### LLM Enhancement layer (`src/utils/openai_utils.py`)

`call_openai_structured(system, user, ResponseModel)` — uses `client.beta.chat.completions.parse`
(structured output), retries on `RateLimitError` only, always `temperature=0`. Requires
`OPENAI_API_KEY` in `.env` (loaded via `python-dotenv`). `has_openai_api_key()` lets agents
gate LLM calls without raising.

Structured output models (all in `src/agents/state.py`):
- `NewsAnalysisLLMOutput` — L2 event classification + duration estimate
- `WeatherRiskLLMOutput` — L3 supply-chain weather interpretation + `geo_risk_component`
- `RiskClassifierLLMEnhancement` — L4 narrative rationale
- `MitigationLLMOutput` — L7 ranked actions + India sourcing recommendations
- `DistilBERTSignal`, `LLMSignal`, `JudgeVerdict` — L4 ensemble signals

### Fine-tuning (`fine_tuning/`)

Training pipeline for the three ML signals in L4 ensemble:
- `finetune_distilbert.py` → DistilBERT risk classifier (Signal 2, ~20ms CPU)
- `finetune_gpt4o_mini.py` → GPT-4o mini fine-tune (Signal 3 base)
- `finetune_embeddings.py` → domain-adapted embeddings for ChromaDB
- `generate_training_data.py` → builds all training splits from SQLite + ChromaDB
- `evaluate_all.py` → end-to-end evaluation across all three models

Run `python fine_tuning/generate_training_data.py` first in any new environment.

### Dashboard (`src/dashboard/`)

`main.py` → `dashboard.py::main()`. Four pages:

- **Live Data Feed** (`ingestion_dashboard.py`): Run Now button (background daemon thread,
  5-second polling — avoids Streamlit WebSocket timeout on long batch runs), hub city weather
  signals table (severity 0–10, trigger indicator), hub city/country news signals table
  (relevance score, location type), Indian port live signals, scheduler status and logs.
- **Data Loader** (`data_loader.py`): browse and filter `lite_master` rows.
- **RAG Search**: ChromaDB full-text search UI.
- **Scenario Analyzer** (`dashboard.py`): run the full L1→L7 pipeline on a selected order,
  displays risk card, forecast, simulation, and mitigation output.

### Utilities (`src/utils/`)

- `db_utils.py`: SQLite access. `DB_PATH = outputs/supply_chain.db`. `check_same_thread=False`.
- `rag_utils.py`: ChromaDB singleton (`_chroma_client`, lock-guarded). **Windows constraint**:
  `PersistentClient` holds an exclusive lock on `chroma.sqlite3`; `shutil.rmtree` while
  client is alive causes `WinError 32`. Pass `flush_existing=False` in-process.
- `openai_utils.py`: `call_openai_structured`, `build_rag_context`, `has_openai_api_key`.
  `build_rag_context` expects `List[Tuple[str, int]]` (query, n_results) — not bare strings.
- `ingestion_schema.py`: DDL for `live_news_ingest` and `live_weather_ingest`. Idempotent
  (`CREATE TABLE IF NOT EXISTS`). Call `ensure_ingestion_schema()` once at startup.
- `ingestion_connectors.py`: 7 connector classes + `HUB_CITIES`, `_compute_relevance_score`,
  `_compute_hub_severity`.
- `ingestion_scheduler.py`: APScheduler hourly job wrapping `DataIngestionAgent.run_batch()`.
- `ingestion_validator.py`: row validation before SQLite insert.
- `yaml_utils.py`: loads `config/india_electronics.yaml` (port coordinates, backup routes).
- `src/agents/weather_agent/client.py`: `fetch_open_meteo` + `compute_weather_severity`
  (formerly `src/utils/api_clients.py`).

## State model notes

`GlobalState` (Pydantic v2) key fields:

- `ingestion_run_id: Optional[str]` — UUID from L1 batch run; links state to
  `live_news_ingest` / `live_weather_ingest` rows. `None` in demo/manual scenario mode.
- `risk_label` and `risk_score_composite` — deprecated read-only shims; prefer
  `risk_classification.final_label` / `.composite_score`. Shims tested in
  `evaluation/qa_07_backwards_compat_state_shims.py` — keep them working.
- Use `state.model_copy(update=delta)` everywhere (Pydantic v2). `state.copy()` is deprecated.
- Optional LLM output fields: `news_analysis_llm`, `weather_risk_llm`, `risk_enhancement_llm`,
  `mitigation_llm`, `judge_verdict` — all `None` when OpenAI key is absent.
