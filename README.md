# Electronics Supply Chain Disruption Predictor

Local SQLite, ChromaDB, RAG search, forecasting, and disruption-scenario
dashboard built around Varun's electronics/semiconductor workbook.

Yogita's beauty/FMCG dataset is not loaded into either database.

## Current data

Source workbook:

```text
data/raw/supply_chain_lite_master.xlsx
```

The database build preserves:

- 5,459 Lite Master order records
- 200 operational KPI records
- 2,282 semiconductor signal records
- Workbook data dictionary and legend
- Duplicate business order IDs without dropping rows

Generated outputs:

```text
outputs/supply_chain.db
outputs/chromadb/
```

## Requirements

- Python 3.11 or 3.12
- Internet access during the first setup to download Python packages and the
  `all-MiniLM-L6-v2` embedding model
- Internet access when running scenarios because weather data comes from
  Open-Meteo

No OpenAI API key is required.

## Setup on Windows PowerShell

Run all commands from the project root:

```powershell
cd D:\supply-chain-disrupter
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build SQLite and ChromaDB

```powershell
python -m src.build_databases
```

Expected headline results:

```text
SQLite: loaded 5,459 Lite Master orders
ChromaDB: 306 chunks
```

The command safely rebuilds:

- SQLite tables for Lite Master, operational KPIs, semiconductor signals,
  workbook metadata, and mitigation outputs
- A `daily_records` compatibility view used by the scenario workflow
- An electronics-only ChromaDB collection containing semiconductor events,
  mitigation knowledge, playbooks, event profiles, and field definitions

## Run the application

```powershell
python -m streamlit run src/main.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

The application contains three pages:

1. **Data Ingestion** — rebuild and inspect SQLite and ChromaDB.
2. **RAG Search** — search semiconductor events, mitigations, and field
   definitions.
3. **Scenario Analyzer** — select an existing workbook region, product, and
   date; calculate risk, run a Prophet forecast, estimate stockout exposure,
   and persist mitigation guidance.

## Typical workflow

```powershell
cd D:\supply-chain-disrupter
.\.venv\Scripts\Activate.ps1
python -m src.build_databases
python -m streamlit run src/main.py
```

The databases only need to be rebuilt when the workbook, playbooks, or database
code changes.

## Project structure

```text
config/
  india_electronics.yaml       Port coordinates and fallback routes
  playbooks/                   Electronics disruption playbooks
data/
  raw/
    supply_chain_lite_master.xlsx
outputs/                       Generated SQLite and ChromaDB files
src/
  agents/                      Scenario workflow and state models
  dashboard/                   Streamlit pages
  utils/                       ETL, SQLite, RAG, weather, and YAML utilities
  build_databases.py           Database build command
  main.py                      Streamlit entry point
requirements.txt
README.md
```

## Troubleshooting

### `No module named streamlit`

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Database or collection is missing

```powershell
python -m src.build_databases
```

### Embedding model download warning

The first ChromaDB build downloads `all-MiniLM-L6-v2` from Hugging Face. A
Hugging Face token is optional; the model can be downloaded anonymously.

### Scenario weather request fails

The database and RAG pages still work offline after initial setup. Scenario
weather enrichment requires access to:

```text
https://api.open-meteo.com
```
