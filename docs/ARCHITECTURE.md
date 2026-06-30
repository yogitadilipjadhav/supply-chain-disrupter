# Supply Chain Disruption Predictor — Architecture

Capstone Project 8 · Varun Mathur · Zenteiq Aitech Innovations

## Pipeline Overview

```
L1 Data Ingestion → L2 News (gpt-4.1-mini) → L3 Weather (gpt-4.1-mini)
                  → L4 Risk Classifier (3-signal ensemble + Judge)
                  → L5 Prophet (optional) → L6 Simulation (optional)
                  → L7 Mitigation (gpt-4o)
```

## L2 — News Agent

**Model:** `gpt-4.1-mini` (or fine-tuned via `OPENAI_FT_NEWS_MODEL`)

**Flow:**
1. Fetch SQLite record + semiconductor_signals for the order year
2. Issue 3 ChromaDB RAG queries via `build_rag_context()`
3. Call OpenAI structured output → `NewsAnalysisLLMOutput`
4. Translate to `NewsRiskSignal` list (primary + regional signals)
5. `news_severity_component` feeds freight component (weight 0.15) in L4

**Fallback:** `FALLBACK_PARAMS` dict or `src.rag.agent.build_news_signals()` when no API key

## L3 — Weather Agent

**Model:** `gpt-4.1-mini`

**Flow:**
1. Fetch Open-Meteo hourly data for record coordinates
2. Compute rule-based `numeric_severity` via `compute_weather_severity()`
3. Find nearest semiconductor hub (12-hub map)
4. If `numeric_severity >= 0.40`, pre-fetch weather RAG context
5. LLM produces `geo_risk_component` which **overrides** numeric severity
6. `live_weather_severity` feeds geo component (weight 0.40) in L4

**Fallback:** Returns rule-based numeric severity unchanged

## L4 — Risk Classifier (Three-Signal Ensemble)

### Signal 1 — Rule-based (always runs)
- Formula: `0.4×geo + 0.3×supply + 0.15×freight + 0.15×defect`
- Delivery overrides: "Shipping canceled" → CRITICAL, "Late delivery" → HIGH
- Duration escalation: ≤1d no change, 2-3d +1 tier, ≥4d force CRITICAL

### Signal 2 — DistilBERT (fine-tuned, ~20ms CPU)
- Model: `fine_tuning/models/distilbert_risk_classifier/`
- 4-class softmax over LOW/MEDIUM/HIGH/CRITICAL
- Graceful skip when model not trained (`model_source="not-available-skipped"`)

### Signal 3 — GPT-4o + Two-Stage RAG
- **Stage 1:** Fine-tuned (or base) all-MiniLM bi-encoder → top-10 per collection
- **Stage 2:** Cross-encoder `ms-marco-MiniLM-L-6-v2` reranks → top-3
- Produces `LLMSignal` with label, rationale, RAG citations

### LLM-as-Judge (GPT-4o)
- Receives all 3 signals + SQLite record + semiconductor context
- Produces `JudgeVerdict` with `final_label`, `verdict_type`, `disagreement_explanation`
- Hard rule: "Shipping canceled" → CRITICAL regardless of judge output

### Final Label Fallback Chain
```
judge_verdict.final_label → llm_signal.predicted_label → rule_signal.escalated_label
critical_flag = (final_label == "CRITICAL")  # never from judge alone
```

## L7 — Mitigation Agent

**Model:** `gpt-4o`

**Flow:**
1. Receive L4 risk classification + L5 forecast + L6 simulation
2. Three two-stage RAG queries via `build_mitigation_context()`
3. LLM produces ranked actions + India sourcing + RAG citations

## RAG Package (`src/rag/`)

| Module | Role |
|--------|------|
| `utils.py` | ChromaDB client, embedding model, monolithic corpus build/query |
| `collections.py` | Named collection ingest (historical / export / India sourcing) |
| `retriever.py` | Two-stage retrieve (bi-encoder) + rerank (cross-encoder) |
| `agent.py` | News-signal fallback via RAG query |

CLI: `python scripts/build_rag_collections.py` (delegates to `src/rag/collections.py`)

## Fine-Tuning Integration

| Phase A Output | Used By |
|----------------|---------|
| `distilbert_risk_classifier/` | Signal 2 (distilbert_signal.py) |
| `supply_chain_embeddings/` | Stage 1 RAG (`src/rag/utils.get_embedding_model()`) |
| `gpt_ft_result.json` | L2 News Agent via OPENAI_FT_NEWS_MODEL |

After embedding fine-tuning, rebuild ChromaDB:
```bash
python scripts/build_rag_collections.py --flush
```

## Graceful Degradation

| Missing | Behavior |
|---------|----------|
| DistilBERT model | Signal 2 skipped, judge uses rules + LLM |
| OPENAI_API_KEY | Signals 3 + Judge skipped, L2/L3/L7 use fallbacks |
| Fine-tuned embedder | Base all-MiniLM-L6-v2 for Stage 1 |
| Cross-encoder | Bi-encoder distance sort for Stage 2 |

## QA Validation

```bash
python -m pytest tests/test_risk_classifier_agent.py -v
python -m pytest tests/test_llm_agents.py -v
python -m pytest tests/test_ensemble_signals.py -v
python evaluation/qa_04_replay_mode_real_data.py
python evaluation/qa_05_live_mode_taiwan_earthquake.py
```
