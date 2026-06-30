"""
generate_training_data.py — Training data for all three fine-tuning workflows.

Reads from SQLite (supply_chain.db) and ChromaDB only — no external fetches.
Outputs to fine_tuning/data/ for use by the three finetune_*.py scripts.

Run this FIRST on any new environment before any training scripts.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.db_utils import execute_query
from src.agents.distilbert_signal import build_distilbert_text
from src.agents.risk_classifier_agent import _escalate_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Path("fine_tuning/data").mkdir(parents=True, exist_ok=True)

LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Train-only duration augmentation — teaches the >=4d CRITICAL hard floor.
DURATION_AUGMENT_DAYS = (0.0, 2.0, 5.0, 30.0)


def _augment_train_rows(rows: list[dict]) -> tuple[list[str], list[int]]:
    """
    Expand each training row into four examples with different disruption durations.

    For every source row, generates texts at 0/2/5/30 days and applies the duration
    escalation matrix so labels reflect the >=4-day CRITICAL hard floor. Used on
    the train split only — val/test are not augmented.
    """
    texts, labels = [], []
    for row in rows:
        # Start from the ground-truth label in SQLite (not delivery_status).
        base = str(row.get("disruption_event_label") or "LOW").strip().upper()
        for dur in DURATION_AUGMENT_DAYS:
            dur_input = None if dur <= 0 else dur
            # Apply duration escalation matrix: >=4 days forces CRITICAL.
            final, _ = _escalate_label(base, dur_input)
            texts.append(build_distilbert_text(row, duration_days=dur))
            labels.append(LABEL2ID.get(final, 0))
    return texts, labels


def load_distilbert_data() -> tuple:
    """
    Build DistilBERT training/validation splits from SQLite lite_master.

    Queries Signal 2 features (the status field is excluded to prevent label leakage), performs a stratified
    80/10/10 split, applies duration augmentation on train, and saves the held-out
    test set to fine_tuning/data/distilbert_test_split.json.

    Returns:
        (X_train, y_train, X_val, y_val) — lists of text strings and integer labels.
    """
    from sklearn.model_selection import train_test_split

    # Step 1 — Query SQLite for Signal 2 features + ground-truth label.
    # delivery_status is deliberately omitted from SELECT (see build_distilbert_text):
    # including it would let DistilBERT trivially map status strings to labels.
    # known_disruption_event gives macro context (COVID, chip shortage, etc.).
    rows = execute_query(
        """SELECT order_region, product_name,
                  known_disruption_event,
                  disruption_news_count, supply_disruption_index,
                  defect_rate_pct, export_control_level,
                  risk_score_composite, lead_time_variance_days,
                  disruption_event_label
           FROM lite_master
           WHERE disruption_event_label IS NOT NULL
             AND disruption_event_label NOT IN ('', 'nan', 'None')"""
    )
    rows = [dict(r) for r in rows]

    # Step 2 — Map string labels to integers 0–3 for the classification head.
    raw_labels = [
        LABEL2ID.get(str(r["disruption_event_label"]).strip().upper(), 0) for r in rows
    ]

    # Step 3 — Log class balance and macro-event coverage for training audit.
    dist = Counter(str(r["disruption_event_label"]) for r in rows)
    logger.info("DistilBERT label distribution (raw): %s", dict(dist))
    logger.info("Total source rows: %d", len(rows))
    print(f"Label distribution: {dict(dist)}")
    print(f"Total samples: {len(rows)}")
    print("NOTE: shipping status excluded from DistilBERT input (see build_distilbert_text docstring).")
    print("      Signal 1 owns shipping status; DistilBERT learns numeric+macro event signals.")
    event_dist = {str(r.get("known_disruption_event", "—")) for r in rows}
    print(f"      Macro events present in training data: {sorted(event_dist)}")

    # Step 4 — Stratified 80/10/10 split (train / val / test).
    # Stratify preserves CRITICAL/HIGH/MEDIUM/LOW proportions in each fold.
    train_rows, temp_rows, _, temp_labels = train_test_split(
        rows, raw_labels, test_size=0.20, random_state=42, stratify=raw_labels
    )
    val_rows, test_rows, val_raw_labels, test_raw_labels = train_test_split(
        temp_rows, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels
    )

    # Step 5 — Build model inputs as natural-language strings via build_distilbert_text().
    # Train set only: 4× duration augmentation (0/2/5/30 days) with escalated labels.
    # Val/test: single text per row at duration=0 (no augmentation, no label change).
    X_train, y_train = _augment_train_rows(train_rows)
    X_val = [build_distilbert_text(r, duration_days=0.0) for r in val_rows]
    y_val = val_raw_labels
    X_test = [build_distilbert_text(r, duration_days=0.0) for r in test_rows]
    y_test = test_raw_labels

    # Step 6 — Persist held-out test split for offline evaluation (evaluate_all.py).
    with open("fine_tuning/data/distilbert_test_split.json", "w") as f:
        json.dump({"texts": X_test, "labels": y_test}, f)
    logger.info(
        "Train: %d (augmented from %d) | Val: %d | Test: %d",
        len(X_train), len(train_rows), len(X_val), len(X_test),
    )
    return X_train, y_train, X_val, y_val


def generate_qa_pairs_from_semiconductor_signals() -> list:
    """
    Build query→passage pairs from semiconductor_signals table rows.

    Each known disruption event (COVID, chip shortage, export controls, etc.)
    becomes a dense passage plus four paraphrased queries for RAG bi-encoder
    fine-tuning. Returns list of (query, passage) tuples.
    """
    rows = execute_query(
        """SELECT year, country, company, known_disruption_event, known_severity,
                  supply_disruption_index, export_control_level, chip_price_index,
                  natural_disaster_risk, factory_shutdown_risk
           FROM semiconductor_signals
           WHERE known_disruption_event IS NOT NULL
             AND known_disruption_event NOT IN ('—', '-', 'None', '')"""
    )
    pairs = []
    for r in rows:
        r = dict(r)
        event = r["known_disruption_event"]
        country, year, company, sev = r["country"], r["year"], r["company"], r["known_severity"]
        passage = (
            f"Semiconductor disruption: {event} ({sev}). "
            f"Year: {year}. Country: {country}. Company: {company}. "
            f"Supply disruption index: {r['supply_disruption_index']:.2f}. "
            f"Export control level: {r['export_control_level']:.2f}. "
            f"Chip price index: {r['chip_price_index']:.2f}. "
            f"Natural disaster risk: {r['natural_disaster_risk']:.2f}. "
            f"Factory shutdown risk: {r['factory_shutdown_risk']:.2f}."
        )
        for q in [
            f"What was the supply chain impact of {event} in {country}?",
            f"How did {event} affect semiconductor supply in {year}?",
            f"What is the risk level for {company} during {event}?",
            f"Electronics supply disruption {country} {year} {event}",
        ]:
            pairs.append((q, passage))
    return pairs


def generate_mitigation_qa_pairs() -> list:
    """
    Build query→passage pairs from lite_master mitigation recommendations.

    Groups unique (risk_label, recommendation) pairs and generates queries asking
    how to respond to each disruption severity level. Returns list of (query, passage) tuples.
    """
    rows = execute_query(
        """SELECT disruption_event_label, mitigation_recommendation,
                  order_region, risk_score_composite
           FROM lite_master
           WHERE mitigation_recommendation IS NOT NULL
             AND disruption_event_label IS NOT NULL
           GROUP BY disruption_event_label, mitigation_recommendation"""
    )
    pairs = []
    for r in rows:
        r = dict(r)
        label, rec = r["disruption_event_label"], r["mitigation_recommendation"]
        region = r["order_region"] or "global"
        passage = (
            f"Mitigation for {label} risk (composite: {r['risk_score_composite']:.3f}): "
            f"{rec}. Region: {region}."
        )
        for q in [
            f"What mitigation action for {label} supply chain risk in {region}?",
            f"How to respond to {label} electronics disruption?",
            f"Supply chain mitigation {label} risk semiconductor",
        ]:
            pairs.append((q, passage))
    return pairs


def generate_chromadb_qa_pairs() -> list:
    """
    Build query→passage pairs by querying existing ChromaDB collections.

    Runs fixed supply-chain topics against ChromaDB and pairs each retrieved chunk
    with synthetic queries. Skipped gracefully if ChromaDB is empty or unavailable.
    Returns list of (query, passage) tuples.
    """
    try:
        from src.rag.utils import query_chroma_rag
    except ImportError:
        logger.warning("rag_utils not importable — skipping ChromaDB QA pairs")
        return []

    topics = [
        "semiconductor supply chain disruption historical",
        "export control electronics restriction BIS entity list",
        "India semiconductor mission PLI sourcing Dixon Tata",
        "weather disaster fab production halt Taiwan earthquake",
        "shipping route disruption freight rate Red Sea",
    ]
    pairs = []
    for topic in topics:
        try:
            hits = query_chroma_rag(topic, n_results=10)
            for hit in hits:
                text = hit.get("text", "").strip()
                if len(text) < 50:
                    continue
                for q in [
                    f"Tell me about: {topic}",
                    f"What happened with {topic.split()[0]} supply chains?",
                    f"Electronics risk: {' '.join(topic.split()[:4])}",
                ]:
                    pairs.append((q, text[:500]))
        except Exception as e:
            logger.warning("ChromaDB query failed for '%s': %s", topic, e)
    return pairs


def save_all_qa_pairs() -> list:
    """
    Merge all QA sources, deduplicate, shuffle, and save to qa_pairs.json.

    Combines semiconductor signals, mitigation text, and ChromaDB chunks into one
    file consumed by finetune_embeddings.py. Returns the list of unique dicts
    with keys {query, positive}.
    """
    sc_pairs = generate_qa_pairs_from_semiconductor_signals()
    mit_pairs = generate_mitigation_qa_pairs()
    chroma_pairs = generate_chromadb_qa_pairs()

    all_pairs = sc_pairs + mit_pairs + chroma_pairs
    seen, unique = set(), []
    for q, p in all_pairs:
        key = q[:50] + p[:30]
        if key not in seen:
            seen.add(key)
            unique.append({"query": q, "positive": p})

    random.seed(42)
    random.shuffle(unique)

    with open("fine_tuning/data/qa_pairs.json", "w") as f:
        json.dump(unique, f, indent=2)
    logger.info(
        "Saved %d unique QA pairs (sc=%d, mitigation=%d, chromadb=%d)",
        len(unique), len(sc_pairs), len(mit_pairs), len(chroma_pairs),
    )
    return unique


def generate_gpt_finetune_jsonl() -> str:
    """
    Generate OpenAI fine-tuning JSONL for the News Agent (GPT-4o-mini L2).

    Maps semiconductor_signals disruption events to ground-truth JSON responses
    (category, severity, regions, commodities, duration) using keyword matching.
    Each line is a {messages: [system, user, assistant]} conversation saved to
    fine_tuning/data/gpt_finetune_train.jsonl. Returns the output file path.
    """
    try:
        from src.agents.news_agent import NEWS_SYSTEM_PROMPT
    except ImportError:
        logger.error("Cannot import NEWS_SYSTEM_PROMPT")
        return ""

    GROUND_TRUTH = {
        "COVID-19": {
            "category": "logistics", "severity": 0.95,
            "affected_regions": ["Eastern Asia", "Southeast Asia", "Western Europe"],
            "affected_commodities": ["advanced logic chips", "DRAM memory", "display panels"],
            "news_severity_component": 0.90, "expected_duration_days": 365.0,
            "summary": "COVID-19 caused simultaneous factory shutdowns across global electronics hubs.",
            "signal_tags": ["covid-19", "global-shutdown", "pandemic", "critical"],
        },
        "Chip Shortage": {
            "category": "raw_material", "severity": 0.85,
            "affected_regions": ["Eastern Asia", "West of USA", "Western Europe"],
            "affected_commodities": ["advanced logic chips (≤7nm)", "automotive MCUs", "DRAM"],
            "news_severity_component": 0.72, "expected_duration_days": 540.0,
            "summary": "2021 global chip shortage from COVID demand surges exhausting foundry capacity.",
            "signal_tags": ["chip-shortage", "tsmc", "foundry", "allocation-queue"],
        },
        "Export Controls": {
            "category": "geopolitical", "severity": 0.62,
            "affected_regions": ["Eastern Asia", "Southeast Asia"],
            "affected_commodities": ["advanced logic chips (≤7nm)", "AI accelerators", "EDA"],
            "news_severity_component": 0.52, "expected_duration_days": 730.0,
            "summary": "US BIS export control rules restricted advanced semiconductor exports to China.",
            "signal_tags": ["export-controls", "bis", "china", "geopolitical"],
        },
        "AI Demand Surge": {
            "category": "demand_shock", "severity": 0.68,
            "affected_regions": ["Eastern Asia", "West of USA"],
            "affected_commodities": ["AI accelerators (H100/A100)", "HBM memory"],
            "news_severity_component": 0.45, "expected_duration_days": 365.0,
            "summary": "2023-24 AI demand surge compressed advanced node foundry capacity.",
            "signal_tags": ["ai-demand", "gpu", "hbm-memory", "demand-shock"],
        },
    }

    rows = execute_query(
        """SELECT DISTINCT year, country, company, known_disruption_event, known_severity,
                  supply_disruption_index, export_control_level, chip_price_index,
                  disruption_news_count
           FROM semiconductor_signals
           WHERE known_disruption_event IS NOT NULL
             AND known_disruption_event NOT IN ('—', '-', 'None', '')
           ORDER BY year, country"""
    )

    examples, skipped = [], 0
    for r in rows:
        r = dict(r)
        event_key = r["known_disruption_event"].strip()
        gt = None
        for k, v in GROUND_TRUTH.items():
            if any(word in event_key for word in k.split()):
                gt = v
                break
        if gt is None:
            skipped += 1
            continue

        user_msg = (
            f"disruption_type: {gt['category']}\n"
            f"affected_port: {r['country']} semiconductor hub\n"
            f"Known_Disruption_Event: {event_key}\n"
            f"Supply_Disruption_Index: {r['supply_disruption_index']:.2f}"
        )
        examples.append({
            "messages": [
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": json.dumps(gt, ensure_ascii=False)},
            ]
        })

    jsonl_path = "fine_tuning/data/gpt_finetune_train.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d JSONL examples (%d skipped). File: %s", len(examples), skipped, jsonl_path)
    return jsonl_path


if __name__ == "__main__":
    print("=== Generating all fine-tuning training data ===\n")
    print("--- Workflow 1: DistilBERT ---")
    load_distilbert_data()
    print("\n--- Workflow 2: Sentence-Transformers QA pairs ---")
    save_all_qa_pairs()
    print("\n--- Workflow 3: GPT-4o-mini JSONL ---")
    generate_gpt_finetune_jsonl()
    print("\n=== Done. Check fine_tuning/data/ ===")
