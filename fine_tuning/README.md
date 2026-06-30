# Fine-Tuning + Ensemble Architecture — Capstone Project 8

## Architecture Overview

This project combines three fine-tuning techniques with a three-signal ensemble
and LLM-as-Judge to classify supply chain disruption risk.

### Fine-Tuning Workflows

| # | Model | Task | Data | Target | GPU? |
|---|-------|------|------|--------|------|
| 1 | DistilBERT-base-uncased | 4-class risk classifier | 5,459 rows (lite_master) | F1 > 0.80 | Yes (~15 min) |
| 2 | all-MiniLM-L6-v2 | Domain-adapted RAG bi-encoder | ~600 QA pairs | top-3 acc > 85% | Yes (~10 min) |
| 3 | GPT-4o-mini | Structured News Agent JSON | ~50-100 JSONL rows | JSON acc > base | No (serverside) |

### How to Run

**Phase A: Fine-tune models (Colab T4 GPU for Workflows 1 and 2)**
```bash
python fine_tuning/generate_training_data.py
python fine_tuning/finetune_distilbert.py
python fine_tuning/finetune_embeddings.py
python scripts/build_rag_collections.py   # rebuild ChromaDB with fine-tuned embedder
python fine_tuning/finetune_gpt4o_mini.py  # optional: News Agent L2
```

**Phase B: Start the system**
```bash
streamlit run src/dashboard/dashboard.py
```

**Phase D: Day 23 evaluation**
```bash
python fine_tuning/evaluate_all.py
```

### Running on Google Colab (T4 GPU)

1. Upload project or mount Google Drive
2. `pip install transformers torch sentence-transformers datasets accelerate scikit-learn`
3. Copy `outputs/supply_chain.db` and `outputs/chromadb/` to Colab
4. Run generate_training_data → finetune_distilbert → finetune_embeddings
5. Download `fine_tuning/models/` back to local project
6. Rebuild ChromaDB: `python scripts/build_rag_collections.py`

### Evaluation Targets (Day 23)

| Metric | Target | Evidence |
|--------|--------|----------|
| DistilBERT F1 macro | > 0.80 | distilbert_val_metrics.json + confusion matrix |
| Bi-encoder retrieval top-3 | > 85% | retrieval_metrics.json |
| Ensemble disagreement explanation | Present when signals differ | Judge verdict panel |
| LLM-as-Judge verdict_type | All 5 types across demos | Dashboard judge panel |

## Signal 2 Input Design — Feature engineering decisions

### Decision 1: Delivery_Status excluded

`Disruption_Event_Label` is derived from `Delivery_Status`:

| Delivery_Status     | Label         |
|---------------------|---------------|
| Shipping canceled   | CRITICAL      |
| Late delivery       | HIGH / MEDIUM |
| Advance shipping    | LOW           |
| Shipping on time    | LOW           |

Including `Delivery_Status` in the DistilBERT input creates a tautological
shortcut — the model learns trivial string mapping (F1 > 0.90 trivially) not
supply chain signal patterns. Signal 1 (Rule-based formula) already owns
Delivery_Status explicitly. The LLM-as-Judge enforces the locked override
(`Shipping canceled → CRITICAL`) at inference time regardless of Signal 2.

### Decision 2: Known_Disruption_Event included

v3 Lite Master carries a macro event name joined from Semiconductor Signals:
`COVID-19 Pandemic`, `Global Chip Shortage + Texas Freeze`, etc.

This resolves 222 confusing training rows in 2020–2021 where:
- `Delivery_Status = "Shipping on time"` → label = LOW
- `Supply_Disruption_Index = 8.4` (mean), `Risk_Score_Composite = 0.63`

Without macro event context, DistilBERT sees high numeric signals mapping to
LOW — a contradictory training signal. With `Known_Disruption_Event =
"COVID-19 Pandemic"`, the model learns: "LOW order outcome + CRITICAL macro
event = order survived a disruption period" — a legitimate and informative
pattern, not an error.

### Input features (9 total)

| Feature | Source column | Role |
|---|---|---|
| Region | `order_region` | Geographic risk zone |
| Product | `product_name` | SKU type / category |
| Known disruption event | `known_disruption_event` | Macro event name (2020-2024) or — |
| News coverage | `disruption_news_count` | Media signal intensity |
| Supply disruption index | `supply_disruption_index` | Macro supply-side stress 0–10 |
| Defect rate | `defect_rate_pct` | Supplier quality signal |
| Export control level | `export_control_level` | Geopolitical/trade signal |
| Risk composite | `risk_score_composite` | Pre-computed ensemble score |
| Lead time variance | `lead_time_variance_days` | Logistics strain signal |

### Expected F1 impact
- Before (with delivery_status, without event): trivial F1 ~0.93 (string shortcut)
- After (signals + macro event): genuine F1 ~0.80–0.87 (learned patterns)

Target remains F1 > 0.80 on held-out test set. The v3 dataset's improved
class balance (CRITICAL: 1,246 rows vs V2's 416) plus the macro event feature
will help the model learn minority classes without delivery_status as a crutch.
