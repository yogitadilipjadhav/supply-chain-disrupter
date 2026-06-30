"""
evaluate_all.py — Day 23 unified evaluation for fine-tuning + ensemble.

Produces:
  1. DistilBERT F1 on held-out test split + confusion matrix PNG
  2. Embedding retrieval baseline vs fine-tuned
  3. Cross-encoder reranking demo
  4. GPT-4o-mini fine-tuning result
  5. Consolidated evaluation_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate_distilbert() -> dict:
    """
    Evaluate fine-tuned DistilBERT on the held-out test split.

    Loads distilbert_test_split.json, runs predict_label() on all test texts,
    computes macro F1 and per-class report, and saves a confusion matrix PNG.
    Returns metrics dict with target_achieved flag (F1 >= 0.80).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    test_path = Path("fine_tuning/data/distilbert_test_split.json")
    if not test_path.exists():
        logger.error("Test split not found — run generate_training_data.py first")
        return {}

    with open(test_path) as f:
        split = json.load(f)
    X_test, y_true = split["texts"], split["labels"]

    model_path = Path("fine_tuning/models/distilbert_risk_classifier")
    if not model_path.exists():
        logger.error("Fine-tuned model not found — run finetune_distilbert.py first")
        return {}

    from fine_tuning.finetune_distilbert import predict_label
    from fine_tuning.generate_training_data import ID2LABEL, LABEL2ID

    y_pred_labels = predict_label(X_test)
    y_pred = [LABEL2ID[l] for l in y_pred_labels]

    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=[ID2LABEL[i] for i in range(4)], digits=4
    )
    logger.info("\n=== DISTILBERT TEST SET ===\n%s", report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(4)],
        yticklabels=[ID2LABEL[i] for i in range(4)], ax=ax,
    )
    ax.set_title(f"DistilBERT Confusion Matrix (F1={f1_macro:.4f})")
    Path("fine_tuning/data").mkdir(exist_ok=True)
    cm_path = "fine_tuning/data/distilbert_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return {
        "f1_macro": f1_macro,
        "report": report,
        "target_achieved": f1_macro >= 0.80,
        "confusion_matrix_path": cm_path,
    }


def evaluate_embeddings() -> dict:
    """
    Load embedding fine-tuning retrieval metrics from disk.

    Reads retrieval_metrics.json written by finetune_embeddings.py (baseline vs
    fine-tuned Accuracy@3). Returns empty dict if the file does not exist.
    """
    metrics_path = Path("fine_tuning/models/supply_chain_embeddings/retrieval_metrics.json")
    if not metrics_path.exists():
        logger.warning("Embedding metrics not found")
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def evaluate_cross_encoder_reranking() -> dict:
    """
    Demonstrate two-stage RAG: bi-encoder retrieval + cross-encoder reranking.

    Runs a sample Taiwan earthquake query through raw collection search and
    retrieve_and_rerank(), returning hit counts for each stage. Skipped if RAG
    collections or cross-encoder are unavailable.
    """
    try:
        from src.rag.retriever import retrieve_and_rerank
        from src.rag.collections import query_collection

        query = "Taiwan earthquake semiconductor supply chain production halt 2022"
        collection = "historical_precedents"
        raw = query_collection(collection, query, n_results=3)
        reranked = retrieve_and_rerank(query, [collection], bi_encoder_top_n=10, rerank_top_k=3)
        return {"query": query, "raw_count": len(raw), "reranked_count": len(reranked)}
    except Exception as e:
        logger.warning("Cross-encoder eval skipped: %s", e)
        return {}


def evaluate_gpt_finetuned() -> dict:
    """
    Load GPT-4o-mini fine-tuning job metadata from disk.

    Reads gpt_ft_result.json (file_id, job_id, model_id) written by
    finetune_gpt4o_mini.py. Returns empty dict if fine-tuning was not run.
    """
    result_path = Path("fine_tuning/data/gpt_ft_result.json")
    if not result_path.exists():
        return {}
    with open(result_path) as f:
        return json.load(f)


def run_all_evaluations() -> dict:
    """
    Run all Day 23 capstone evaluations and write a consolidated report.

    Executes DistilBERT test eval, loads embedding/GPT metrics, runs a cross-encoder
    demo, and saves everything to fine_tuning/data/evaluation_report.json.
    """
    logger.info("=== DAY 23 — CAPSTONE EVALUATION ===\n")
    results = {
        "distilbert": evaluate_distilbert(),
        "embeddings": evaluate_embeddings(),
        "cross_encoder": evaluate_cross_encoder_reranking(),
        "gpt_finetuned": evaluate_gpt_finetuned(),
    }
    Path("fine_tuning/data").mkdir(exist_ok=True)
    with open("fine_tuning/data/evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Report saved: fine_tuning/data/evaluation_report.json")
    return results


if __name__ == "__main__":
    run_all_evaluations()
