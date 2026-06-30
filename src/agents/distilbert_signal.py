"""
distilbert_signal.py — Signal 2: Fine-tuned DistilBERT classifier.

Loads from DISTILBERT_MODEL_ID (Hugging Face or local path), else
fine_tuning/models/distilbert_risk_classifier/ when available.
Falls back gracefully if model not present — never raises.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from src.agents.state import DistilBERTSignal

logger = logging.getLogger(__name__)

FINETUNED_MODEL_PATH = Path("fine_tuning/models/distilbert_risk_classifier")
LABEL2ID = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 256


def _resolve_model_path() -> Optional[str]:
    """Return Hugging Face repo id or local path, or None if no model configured."""
    env_id = os.getenv("DISTILBERT_MODEL_ID", "").strip()
    if env_id:
        return env_id
    if FINETUNED_MODEL_PATH.exists() and (FINETUNED_MODEL_PATH / "config.json").exists():
        return str(FINETUNED_MODEL_PATH)
    return None


def _model_available() -> bool:
    return _resolve_model_path() is not None


@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    """Load tokenizer + model once and cache for the process lifetime."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = _resolve_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def build_distilbert_text(record: dict, duration_days: Optional[float] = None) -> str:
    """
    Build a natural-language input sentence for DistilBERT (Signal 2).

    MUST match fine_tuning/generate_training_data.py usage exactly — any format
    divergence causes train/serve distribution shift.

    ARCHITECTURAL DECISIONS:

    1. Delivery_Status is intentionally EXCLUDED.
       Disruption_Event_Label derives from Delivery_Status (canceled→CRITICAL,
       late→HIGH, else→LOW). Including it creates a tautological shortcut where
       DistilBERT trivially maps the status string to its label without learning
       supply chain signal patterns. Signal 1 (Rule-based formula) already owns
       Delivery_Status explicitly.

    2. Known_Disruption_Event IS included.
       v3 Lite Master carries a macro event name (e.g. "COVID-19 Pandemic")
       joined from Semiconductor Signals for years 2020-2024. This resolves
       confusing training rows where Delivery_Status="Shipping on time" (→ LOW)
       but Supply_Disruption_Index > 7 during a CRITICAL macro disruption period.

    duration_days: shock / news duration from the live scenario (0 when unknown).
    """
    region = record.get("order_region") or record.get("port") or "unknown"
    product = (record.get("product_name") or record.get("sku") or "unknown")[:60]
    event = record.get("known_disruption_event") or "—"
    news_ct = int(record.get("disruption_news_count") or 0)
    sdi = float(record.get("supply_disruption_index") or 0.0)
    defect = float(record.get("defect_rate_pct") or 0.0)
    ecl = float(record.get("export_control_level") or 0.0)
    risk_sc = float(record.get("risk_score_composite") or 0.0)
    ldv = float(record.get("lead_time_variance_days") or 0.0)
    dur = 0.0 if duration_days is None else float(duration_days)

    return (
        f"Region: {region}. Product: {product}. "
        f"Known disruption event: {event}. "
        f"News coverage: {news_ct} articles. "
        f"Supply disruption index: {sdi:.2f}. "
        f"Defect rate: {defect:.1f}%. "
        f"Export control level: {ecl:.2f}. "
        f"Risk composite score: {risk_sc:.3f}. "
        f"Lead time variance: {ldv:.1f} days. "
        f"Disruption duration: {dur:.0f} days."
    )


# Backwards-compatible alias used at inference call sites.
build_inference_text = build_distilbert_text


def run_distilbert_inference(
    record: dict,
    duration_days: Optional[float] = None,
) -> DistilBERTSignal:
    """
    Run Signal 2 inference on a single SQLite record. Never raises.

    Returns model_source="not-available-skipped" when model is absent.
    """
    if not _model_available():
        logger.info("DistilBERT model not found — Signal 2 will be skipped.")
        return DistilBERTSignal(
            predicted_label="N/A",
            confidence=0.0,
            probability_distribution={"LOW": 0.25, "MEDIUM": 0.25, "HIGH": 0.25, "CRITICAL": 0.25},
            model_source="not-available-skipped",
            inference_ms=0.0,
        )

    try:
        import torch
        import torch.nn.functional as F

        t0 = time.monotonic()
        tokenizer, model = _load_model_and_tokenizer()
        text = build_distilbert_text(record, duration_days=duration_days)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])
        prob_dist: Dict[str, float] = {ID2LABEL[i]: float(probs[i]) for i in range(4)}
        elapsed_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "DistilBERT Signal 2: label=%s confidence=%.3f in %.0fms",
            ID2LABEL[pred_idx],
            confidence,
            elapsed_ms,
        )

        return DistilBERTSignal(
            predicted_label=ID2LABEL[pred_idx],
            confidence=round(confidence, 4),
            probability_distribution={k: round(v, 4) for k, v in prob_dist.items()},
            model_source="fine-tuned",
            inference_ms=round(elapsed_ms, 1),
        )

    except Exception as exc:
        logger.warning("DistilBERT inference failed (non-blocking): %s", exc)
        return DistilBERTSignal(
            predicted_label="N/A",
            confidence=0.0,
            probability_distribution={"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.0},
            model_source=f"error:{str(exc)[:60]}",
            inference_ms=0.0,
        )
