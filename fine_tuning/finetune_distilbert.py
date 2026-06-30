"""
finetune_distilbert.py — Fine-tune DistilBERT for 4-class risk classification.

Classes: LOW(0) | MEDIUM(1) | HIGH(2) | CRITICAL(3)
Target: F1 macro > 0.80
Produces: fine_tuning/models/distilbert_risk_classifier/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_tuning.generate_training_data import ID2LABEL, LABEL2ID, load_distilbert_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
MODEL_OUTPUT = Path("fine_tuning/models/distilbert_risk_classifier")
DATA_DIR = Path("fine_tuning/data")
MAX_LENGTH = 256
NUM_LABELS = 4

TRAINING_ARGS = TrainingArguments(
    output_dir=str(MODEL_OUTPUT / "checkpoints"),
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.10,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    report_to="none",
    seed=42,
)


def tokenize_batch(batch: dict, tokenizer) -> dict:
    """
    Convert a batch of raw text strings to DistilBERT token IDs.

    Truncates to MAX_LENGTH and leaves padding to DataCollatorWithPadding so each
    batch is padded only to its longest sequence.
    """
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False)


def compute_metrics(eval_pred) -> dict:
    """
    Hugging Face Trainer callback: compute classification metrics on validation.

    Converts model logits to predicted class indices and returns macro/weighted F1
    plus macro precision and recall. Used for early stopping on f1_macro.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
    }


def _latest_checkpoint(output_dir: Path) -> str | None:
    """Return the path to the most recent Trainer checkpoint, or None if none exist."""
    ckpt_dir = output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(
        (d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def run_finetuning() -> str:
    """
    End-to-end DistilBERT fine-tuning for Signal 2 (4-class risk classifier).

    Loads data from SQLite, trains distilbert-base-uncased with a new classification
    head, saves the best model to fine_tuning/models/distilbert_risk_classifier/,
    and writes validation metrics. Target: macro F1 > 0.80.

    Returns:
        Path to the saved model directory as a string.
    """
    # Step 1 — Load labelled training data from SQLite (lite_master).
    # generate_training_data.load_distilbert_data() builds natural-language
    # sentences via build_distilbert_text() (region, product, macro event,
    # numeric signals — delivery_status intentionally excluded) and splits
    # 80/10/10 with duration augmentation on the train set only.
    logger.info("Loading training data from SQLite...")
    X_train, y_train, X_val, y_val = load_distilbert_data()

    # Step 2 — Wrap texts + integer labels in Hugging Face Dataset objects
    # so the Trainer API can batch, shuffle, and evaluate cleanly.
    train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "label": y_val})
    dataset = DatasetDict({"train": train_ds, "validation": val_ds})

    # Step 3 — Tokenize: convert each sentence to DistilBERT subword IDs.
    # Truncation at MAX_LENGTH=256; padding is deferred to the collator so
    # each batch is padded only to its longest sequence (more efficient).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenised = dataset.map(
        lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=["text"]
    )

    # Step 4 — Load pre-trained DistilBERT and replace the classification head
    # with a fresh 4-class head (LOW/MEDIUM/HIGH/CRITICAL). Head weights are
    # randomly initialized; encoder weights start from distilbert-base-uncased.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    logger.info("Model: %s  Parameters: %.1fM", BASE_MODEL, sum(p.numel() for p in model.parameters()) / 1e6)

    # Step 5 — Configure the Hugging Face Trainer: cosine LR schedule, epoch-level
    # eval, early stopping on f1_macro, fp16 on GPU. DataCollatorWithPadding
    # right-pads variable-length batches to a uniform tensor shape.
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Step 6 — Train (or resume from the latest checkpoint if one exists).
    # Up to 5 epochs; stops early if val f1_macro does not improve for 2 epochs.
    resume_from = _latest_checkpoint(MODEL_OUTPUT)
    if resume_from:
        logger.info("Resuming fine-tuning from %s ...", resume_from)
    else:
        logger.info("Starting fine-tuning (5 epochs, early stopping patience=2)...")
    trainer.train(resume_from_checkpoint=resume_from)

    # Step 7 — Persist the best checkpoint (weights + tokenizer vocab) to
    # fine_tuning/models/distilbert_risk_classifier/ for Signal 2 inference.
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUTPUT))
    tokenizer.save_pretrained(str(MODEL_OUTPUT))

    # Step 8 — Final validation pass: per-class precision/recall/F1 report.
    # Target: macro F1 > 0.80. Metrics saved to distilbert_val_metrics.json.
    preds_output = trainer.predict(tokenised["validation"])
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    report = classification_report(
        y_val, y_pred, target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=4
    )
    logger.info("\n=== VALIDATION CLASSIFICATION REPORT ===\n%s", report)

    metrics = {
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
        "per_class_report": report,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "distilbert_val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if metrics["f1_macro"] < 0.80:
        logger.warning("F1 %.4f below target 0.80", metrics["f1_macro"])
    return str(MODEL_OUTPUT)


def predict_label(texts: list, model_path: str | None = None) -> list:
    """
    Run batch inference on fine-tuned DistilBERT and return label strings.

    Loads the saved model/tokenizer from model_path (defaults to MODEL_OUTPUT),
    tokenizes inputs, and maps argmax logits to LOW/MEDIUM/HIGH/CRITICAL.
    Used by evaluate_all.py and available for manual testing.
    """
    path = model_path or str(MODEL_OUTPUT)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    inputs = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1).tolist()
    return [ID2LABEL[p] for p in predictions]


if __name__ == "__main__":
    saved_path = run_finetuning()
    print(f"\nFine-tuned DistilBERT model at: {saved_path}")
